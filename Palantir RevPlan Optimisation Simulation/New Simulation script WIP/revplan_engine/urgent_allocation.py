"""
urgent_allocation.py — Tier-1 urgent-WIP-lot pre-pass.

⚠️ MLWB ADDITION — this module is NOT part of the Palantir source. It exists so that
`allocation_engine.py` stays a near-verbatim Palantir port: that file only gains one call
to `run_urgent_prepass()` plus a small demand-credit seam; all the Tier-1 logic lives here.

The customer's allocation priority is TWO-TIER:
  * Tier 1 — every lot flagged `is_urgent` (membership in PK1_URGENCY_WIP_LOT; there is NO
             priority column) is top priority, allocated to COMPLETION first, ordered by
             `urgency_creation_date` ascending (oldest = highest, first-come-first-serve).
  * Tier 2 — the existing weighted model-priority loop ranks the remaining demand.

Urgent lots are a subset of WIP, so their routing/steps are already in `lots_by_id`; this is
pure allocation SEQUENCING. Because capacity lives in one shared `AllocationState`, allocating
urgent steps first consumes the earliest capacity, and the weighted loop then (a) sees reduced
capacity and (b) skips the now-complete urgent lots automatically (0 remaining steps).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from .config import AllocationConfig
from .models import AllocationState, LotStepAllocation

# model_priority sentinel stamped on urgent-lot allocation records (lower = higher priority;
# below any real rank, so these rows are identifiable in the output). It also opts urgent lots
# into the fast-track path (more steps/day) — aligned with urgency, and it can only raise the
# per-day step cap, never cause an allocation to fail.
URGENT_PRIORITY = -1


def _urgent_lot_ids_ordered(lots_by_id: Dict[str, List[dict]]) -> List[str]:
    """Lot ids flagged urgent, ordered by `urgency_creation_date` ASC (oldest first).

    The flag rides on every WIP row of the lot, so it is read off the lot's first step.
    A flagged lot with no parseable date sorts last (shouldn't happen in practice).
    """
    urgent = [lid for lid, steps in lots_by_id.items() if steps and steps[0].get("is_urgent")]
    urgent.sort(key=lambda lid: lots_by_id[lid][0].get("urgency_creation_date") or datetime.max)
    return urgent


def run_urgent_prepass(
    *,
    lots_by_id: Dict[str, List[dict]],
    lot_max_sequence: Dict[str, int],
    state: AllocationState,
    demand_by_model_month: Dict[Any, int],
    equipment_capacity: Dict[str, int],
    process_to_equipment: Dict[str, List[str]],
    negative_constraints: Dict[Any, set],
    allocation_results: List[LotStepAllocation],
    failed_allocation_results: List[LotStepAllocation],
    revenue_plan_id: str,
    simulation_id: str,
    run_id: str,
    run_timestamp: datetime,
    start_date,
    config: AllocationConfig,
    model_metadata_lookup: Dict[str, Dict[str, Any]] = None,
) -> Dict[str, int]:
    """PHASE 0 — allocate every urgent WIP lot to completion, oldest-created first, before
    the weighted month/model loop runs.

    Mutates the shared `state` (capacity/allocated-steps) and appends to
    `allocation_results` / `failed_allocation_results`, exactly like the normal loop.

    Returns `pre_allocated_by_model`: units produced per model (counted only when a lot
    completes). The caller credits these against demand so Phase 2 doesn't re-produce them.
    """
    # Deferred import breaks the import cycle (allocation_engine imports this module at top).
    from .allocation_engine import (
        get_next_allocatable_step,
        _try_allocate_lot_step,
        _build_allocation_record,
    )

    pre_allocated_by_model: Dict[str, int] = {}
    ordered = _urgent_lot_ids_ordered(lots_by_id)
    if not ordered:
        return pre_allocated_by_model

    # Earliest demand month per model — used only as `target_month` for the
    # hold-until-target-month final-step path and delay stats (urgent WIP lots carry no
    # intrinsic target month). Falls back to the sim start month.
    earliest_demand_month: Dict[str, str] = {}
    for (m, mon) in demand_by_model_month.keys():
        if m not in earliest_demand_month or mon < earliest_demand_month[m]:
            earliest_demand_month[m] = mon

    print(f"\n=== Phase 0 (Tier 1): {len(ordered)} urgent WIP lots, oldest-created first ===")
    for lot_id in ordered:
        steps = lots_by_id.get(lot_id) or []
        model_id = steps[0].get("model_id") if steps else None
        if not model_id:
            continue
        target_month = earliest_demand_month.get(model_id, config.constraints.start_month)

        produced = 0
        while True:
            next_step = get_next_allocatable_step(lot_id, lots_by_id, state)
            if next_step is None:
                break                                   # lot fully allocated
            is_new_lot = next_step.get("is_new_lot", False) or lot_id.startswith("VL-")

            result = _try_allocate_lot_step(
                lot_step=next_step,
                model_id=model_id,
                model_priority=URGENT_PRIORITY,
                target_month=target_month,
                is_new_lot=is_new_lot,
                revenue_plan_id=revenue_plan_id,
                state=state,
                equipment_capacity=equipment_capacity,
                process_to_equipment=process_to_equipment,
                negative_constraints=negative_constraints,
                lots_by_id=lots_by_id,
                lot_max_sequence=lot_max_sequence,
                start_date=start_date,
                config=config,
                model_metadata_lookup=model_metadata_lookup,
            )
            alloc = _build_allocation_record(
                lot_step=next_step,
                result=result,
                model_id=model_id,
                model_priority=URGENT_PRIORITY,
                target_month=target_month,
                is_new_lot=is_new_lot,
                revenue_plan_id=revenue_plan_id,
                simulation_id=simulation_id,
                run_id=run_id,
                run_timestamp=run_timestamp,
                lots_by_id=lots_by_id,
                lot_max_sequence=lot_max_sequence,
                state=state,
                config=config,
            )
            if result.success:
                allocation_results.append(alloc)
                produced += alloc.units_produced        # nonzero only on the completing step
            else:
                failed_allocation_results.append(alloc)
                print(f"   ✗ urgent lot {lot_id} blocked at seq {next_step.get('sequence')}: "
                      f"{result.blocking_reason}")
                break

        if produced:
            pre_allocated_by_model[model_id] = pre_allocated_by_model.get(model_id, 0) + produced

    print(f"=== Phase 0 complete: {sum(pre_allocated_by_model.values()):,} urgent units produced "
          f"across {len(pre_allocated_by_model)} model(s) ===")
    return pre_allocated_by_model
