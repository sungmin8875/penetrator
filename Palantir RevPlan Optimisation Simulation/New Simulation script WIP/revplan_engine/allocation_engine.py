"""
[MLWB PORT] This module is a VERBATIM copy of the Palantir Foundry source
file "revenue_driven_allocation_refactored.py" from ../Python Script Allocation Engine/. The ONLY change is import
rewrites (transforms.api -> ._foundry_shim shim; myproject.datasets.allocation_engine
-> relative package imports). The pure logic is byte-identical to the source so it
can be re-synced if the Palantir engine changes. Foundry decorators are no-ops here.
"""
"""
Revenue-Driven Capacity Allocation with Virtual Lot Creation

Main allocation transform that:
1. Allocates existing WIP lots month-by-month by priority
2. Creates virtual lots for shortfalls
3. Allocates virtual lots
4. Tracks models without routing definitions

INCREMENTAL PROCESSING:
- Tracks which simulations have been processed with what configuration
- Only processes new/changed simulations or configuration changes
- Aborts if no new work is needed

CONSTRAINTS APPLIED (see config.py for values):
1. Sequential step processing (step N must complete before step N+1)
2. Equipment capacity limits (sheets/day per equipment)
3. Max steps per lot per day
4. Allocation horizon (cannot schedule past max year)
5. ET JIG daily capacity (lots per day per model from model metadata)

DEMAND CARRYOVER LOGIC:
- Carryover is based on units ALLOCATED (lots scheduled), not on-time completion
- Once lots are created/allocated for a month's demand, that demand is considered covered
- Lots may complete late, but they still count toward the original demand
- This prevents over-production from creating duplicate lots for already-covered demand
"""

import heapq
import polars as pl
from ._foundry_shim import (
    transform,
    Input,
    Output,
    incremental,
    IncrementalLightweightInput,
    IncrementalLightweightOutput,
    LightweightContext,
)
from datetime import datetime, timedelta, date
from typing import Any, Dict, Tuple, List, Set

from .config import (
    DEFAULT_CONFIG,
    AllocationConfig,
    ConstraintType,
    load_config_for_simulation,
    build_config_lookup,
)
from .models import (
    AllocationState,
    AllocationStatus,
    DelayRecord,
    AllocationAttemptResult,
    LotStepAllocation,
    UnroutedModelDemand,
)
from .allocation_helpers import (
    generate_allocation_run_id,
    build_model_metadata_lookup,
    build_model_process_steps_lookup,
    build_model_priority_lookup,
    build_equipment_capacity_lookup,
    build_process_to_equipment_lookup,
    build_negative_constraints_lookup,
    find_least_loaded_equipment_for_process,
    find_models_with_blocked_equipment_paths,
    parse_month_to_eom_date,
)
from .virtual_lot_creator import create_virtual_lots
from .urgent_allocation import run_urgent_prepass  # ⚠️ MLWB ADDITION — Tier-1 urgent pre-pass
from .outsourced_allocation import (  # ⚠️ MLWB ADDITION — unmapped op = 외주 (customer 2026-07-15 §4)
    install_plan_lt_lookup,
    print_outsourced_summary,
    try_allocate_outsourced_step,
)
import os as _os  # ⚠️ MLWB ADDITION — REVPLAN_SCAN_MEMO switch (see _scan_memo_enabled)
from .allocation_run_tracker import (
    create_config_hash,
    create_run_record,
    RUN_TRACKER_SCHEMA,
)

# =============================================================================
# Output Schema Definitions
# =============================================================================

ALLOCATION_OUTPUT_SCHEMA = {
    "allocation_id": pl.Utf8,
    "lot_id": pl.Utf8,
    "model_id": pl.Utf8,
    "process_id": pl.Utf8,
    "sequence": pl.Int32,
    "equipment_group": pl.Utf8,
    "equipment_id": pl.Utf8,
    "allocated_date": pl.Date,
    "target_month": pl.Utf8,
    "actual_completion_month": pl.Utf8,
    "is_delayed_completion": pl.Boolean,
    "simulation_id": pl.Utf8,
    "revenue_plan_id": pl.Utf8,
    "model_priority": pl.Int32,
    "remaining_steps": pl.Int32,
    "capacity_used_sheets": pl.Int32,
    "units_produced": pl.Int32,
    "delay_reasons": pl.Utf8,
    "is_new_lot": pl.Boolean,
    "allocation_run_id": pl.Utf8,
    "allocation_run_ts": pl.Datetime,
    "model_customer_name": pl.Utf8,
    "model_end_customer": pl.Utf8,
    "model_sales_team": pl.Utf8,
    "grouping_model": pl.Utf8,
    "unit_process_name": pl.Utf8,
    "simulation_name": pl.Utf8,
    "simulation_revenue_id": pl.Utf8,
    "final_production_units": pl.Int32,
    "total_revenue": pl.Float64,
    "total_margin": pl.Float64,
}

FAILED_ALLOCATION_SCHEMA = {
    "failed_allocation_id": pl.Utf8,
    "lot_id": pl.Utf8,
    "model_id": pl.Utf8,
    "process_id": pl.Utf8,
    "sequence": pl.Int32,
    "equipment_group": pl.Utf8,
    "target_month": pl.Utf8,
    "simulation_id": pl.Utf8,
    "simulation_name": pl.Utf8,
    "revenue_plan_id": pl.Utf8,
    "model_priority": pl.Int32,
    "remaining_steps": pl.Int32,
    "capacity_needed_sheets": pl.Int32,
    "units_in_lot": pl.Int32,
    "days_searched": pl.Int32,
    "failure_reason": pl.Utf8,
    "failure_status": pl.Utf8,
    "delay_details": pl.Utf8,
    "is_new_lot": pl.Boolean,
    "allocation_run_id": pl.Utf8,
    "allocation_run_ts": pl.Datetime,
}

NEW_LOTS_OUTPUT_SCHEMA = {
    "lot_id": pl.Utf8,
    "model_id": pl.Utf8,
    "target_month": pl.Utf8,
    "target_panels": pl.Int64,
    "target_units": pl.Int64,
    "target_lot_start_date": pl.Date,
    "lead_time_days": pl.Int32,
    "simulation_id": pl.Utf8,
    "revenue_plan_id": pl.Utf8,
    "allocation_run_id": pl.Utf8,
    "allocation_run_ts": pl.Datetime,
}

UNROUTED_MODEL_DEMAND_SCHEMA = {
    "title": pl.Utf8,
    "new_model_risk_description": pl.Utf8,
    "new_model_risk_id": pl.Utf8,
    "model_id": pl.Utf8,
    "demand_qty_ea": pl.Int64,
    "customer_name": pl.Utf8,
    "end_customer": pl.Utf8,
    "model_sales_team": pl.Utf8,
    "simulation_name": pl.Utf8,
    "simulation_id": pl.Utf8,
    "remediation_suggestion": pl.Utf8,
    "shortfall_month": pl.Utf8,
    "allocation_run_id": pl.Utf8,
    "allocation_run_ts": pl.Datetime,
}
# =============================================================================
# Lot Processing Helpers
# =============================================================================


def get_model_lookup_key(
    model_id: str, row: dict = None, model_metadata_lookup: Dict[str, Dict[str, Any]] = None
) -> str:
    """
    Get the key for lot lookup - uses grouping_model if available, else model_id.

    This ensures lots are grouped by their grouping_model so that demand for a
    grouping model can be fulfilled by any model_id within that family.

    Args:
        model_id: The model ID
        row: Optional row dict that may contain grouping_model
        model_metadata_lookup: Optional metadata lookup to find grouping_model

    Returns:
        grouping_model if available, otherwise model_id
    """
    grouping = row.get("grouping_model") if row else None
    if not grouping and model_metadata_lookup and model_id in model_metadata_lookup:
        grouping = model_metadata_lookup[model_id].get("grouping_model")
    return grouping or model_id


def group_wip_by_lot(wip_rows: List[dict]) -> Dict[str, List[dict]]:
    """
    Group WIP rows by lot_id and sort each lot's steps by sequence.
    Marks the final step of each lot with _is_final_step flag for
    hold-until-target-month optimization.

    Returns:
        Dictionary: lot_id -> [step1, step2, step3, ...] (sorted by sequence)
    """
    lots_by_id: Dict[str, List[dict]] = {}

    for row in wip_rows:
        lot_id = row.get("lot_id")
        if lot_id:
            if lot_id not in lots_by_id:
                lots_by_id[lot_id] = []
            lots_by_id[lot_id].append(row)

    for lot_id in lots_by_id:
        lots_by_id[lot_id].sort(key=lambda x: x.get("sequence", 0))
        # Mark the final step for hold-until-target-month optimization
        if lots_by_id[lot_id]:
            lots_by_id[lot_id][-1]["_is_final_step"] = True

    return lots_by_id


def get_next_allocatable_step(
    lot_id: str,
    lots_by_id: Dict[str, List[dict]],
    state: AllocationState,
) -> dict | None:
    """
    Get the next step for a lot that can be allocated (respecting sequence order).
    Returns:
        Next unallocated step dict, or None if all steps are allocated.
    """
    lot_steps = lots_by_id.get(lot_id, [])

    for step in lot_steps:
        process_id = step.get("process_id")
        sequence = step.get("sequence")
        if not state.is_step_allocated(lot_id, process_id, sequence):
            return step

    return None


def count_remaining_steps(
    lot_id: str,
    lots_by_id: Dict[str, List[dict]],
    state: AllocationState,
) -> int:
    """Count how many steps remain unallocated for a lot."""
    lot_steps = lots_by_id.get(lot_id, [])
    return sum(
        1 for step in lot_steps if not state.is_step_allocated(lot_id, step.get("process_id"), step.get("sequence"))
    )


def calculate_dynamic_wip_earliest_start(
    lot_id: str,
    lot_step: dict,
    target_month: str,
    lots_by_id: Dict[str, List[dict]],
    state: AllocationState,
    start_date: date,
    config: AllocationConfig,
) -> date:
    """
    Calculate earliest start date for WIP lots based on target month, lead time, and completion progress.

    This dynamic calculation allows lots closer to completion to start earlier while
    preventing lots far from completion from starting too early and clogging capacity.

    Args:
        lot_id: The lot identifier
        lot_step: Current step data dictionary
        target_month: Target month for completion (YYYYMM format)
        lots_by_id: Lookup of lot_id -> list of steps
        state: Current allocation state
        start_date: Simulation start date (floor for earliest start)
        config: Allocation configuration

    Returns:
        Calculated earliest start date for the WIP lot
    """
    constraints = config.constraints

    target_eom = parse_month_to_eom_date(target_month)

    remaining_steps_input = lot_step.get("remaining_steps")
    final_work_sequence = lot_step.get("final_work_sequence")

    total_steps = final_work_sequence if final_work_sequence and final_work_sequence > 0 else 1

    steps_allocated_in_sim = sum(
        1 for s in lots_by_id.get(lot_id, []) if state.is_step_allocated(lot_id, s.get("process_id"), s.get("sequence"))
    )
    if remaining_steps_input is not None:
        remaining_steps = max(0, remaining_steps_input - steps_allocated_in_sim)
    else:
        remaining_steps = count_remaining_steps(lot_id, lots_by_id, state)

    lead_time_days = remaining_steps / 2.0

    remaining_fraction = remaining_steps / total_steps if total_steps > 0 else 1.0

    completion_boost = 1.0 + (1.0 - remaining_fraction)
    dynamic_buffer = constraints.wip_lead_time_buffer_factor * completion_boost

    base_adjusted_lead_time = lead_time_days * dynamic_buffer

    total_lead_time = total_steps / 2.0
    min_buffer_days = total_lead_time * (1.0 - remaining_fraction) * 0.15

    adjusted_lead_time_days = max(min_buffer_days, base_adjusted_lead_time)

    max_lead_time_start = target_eom - timedelta(days=adjusted_lead_time_days)

    return max(max_lead_time_start, start_date)


# =============================================================================
# Main Allocation Algorithm
# =============================================================================


def allocate_month_by_month(
    wip_with_equipment: pl.DataFrame,
    demand_by_model_month: Dict[Tuple[str, str], int],
    model_priority_lookup: Dict[Tuple[str, str], int],
    model_metadata_lookup: Dict[str, Dict[str, Any]],
    model_process_steps_lookup: Dict[str, List[Dict[str, Any]]],
    equipment_capacity: Dict[str, int],
    process_to_equipment: Dict[str, List[str]],
    negative_constraints: Dict[Tuple[str, str], Set[str]],
    revenue_plan_id: str,
    simulation_id: str,
    simulation_name: str,
    run_id: str,
    run_timestamp: datetime,
    blocked_demand: Dict[Tuple[str, str], Tuple[int, str, str]] = None,
    config: AllocationConfig = DEFAULT_CONFIG,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Allocate WIP lots + create/allocate virtual lots month-by-month.

    DEMAND CARRYOVER LOGIC:
    - `units_allocated`: Tracks lots we've scheduled (includes late completions)
    - Carryover = total_demand - units_allocated
    - Once lots are allocated for a demand, that demand is covered (even if lots complete late)

    This ensures that:
    1. We don't create duplicate lots for demand that's already covered by allocated lots
    2. Carryover only includes demand we truly couldn't allocate
    3. Late completions are tracked via actual_completion_month for reporting

    Args:
        process_to_equipment: Dict mapping process_id -> list of equipment_ids
        negative_constraints: Dict mapping (grouping_model, process_id) -> set of blocked equipment_ids
        blocked_demand: Dict of (model_id, month) -> (qty, process_id, reason) for models with all equipment blocked

    Returns:
        Tuple of (allocation_df, new_lots_df, unrouted_demand_df, failed_allocations_df)
    """
    blocked_demand = blocked_demand or {}

    state = AllocationState()
    # ⚠️ MLWB ADDITION — (model, op) -> Run/Wait LT for the outsourced pass-through
    # (lot-step dicts don't carry routing LTs; see outsourced_allocation.py).
    install_plan_lt_lookup(state, model_process_steps_lookup)
    start_date = config.effective_start_date

    allocation_results: List[LotStepAllocation] = []
    failed_allocation_results: List[LotStepAllocation] = []
    unrouted_demand_results: List[UnroutedModelDemand] = []
    all_new_lots_dfs: List[pl.DataFrame] = []

    # Track lot starts per (grouping_model, date) to respect ET JIG daily capacity
    # This persists across all months to prevent exceeding daily limits
    lot_starts_by_model_date: Dict[Tuple[str, date], int] = {}

    for (model_id, target_month), (
        qty,
        blocking_process_id,
        reason,
    ) in blocked_demand.items():
        model_priority = model_priority_lookup.get((model_id, target_month), config.defaults.model_priority)

        process_steps = model_process_steps_lookup.get(model_id, [])
        equipment_group = None
        for step in process_steps:
            if step.get("process_id") == blocking_process_id:
                equipment_group = step.get("equipment_group_id")
                break

        failed_allocation_results.append(
            LotStepAllocation(
                allocation_id=f"{simulation_id}_{model_id}_{target_month}_BLOCKED",
                lot_id=f"BLOCKED_{model_id}_{target_month}",
                model_id=model_id,
                process_id=blocking_process_id,
                sequence=0,
                equipment_group=equipment_group,
                equipment_id=None,
                allocated_date=None,
                target_month=target_month,
                actual_completion_month=None,
                is_delayed_completion=True,
                simulation_id=simulation_id,
                revenue_plan_id=revenue_plan_id,
                model_priority=model_priority,
                remaining_steps=len(process_steps),
                capacity_used_sheets=0,
                units_produced=qty,
                days_delayed=0,
                delay_reasons=reason,
                is_new_lot=False,
                allocation_status=AllocationStatus.FAILED_NO_EQUIPMENT,
                allocation_run_id=run_id,
                allocation_run_ts=run_timestamp,
            )
        )

    wip_rows = wip_with_equipment.to_dicts()
    lots_by_id = group_wip_by_lot(wip_rows)

    lot_max_sequence: Dict[str, int] = {}
    for lot_id, steps in lots_by_id.items():
        lot_max_sequence[lot_id] = max(s.get("sequence", 0) for s in steps) if steps else 0

    # Build lot lookup keyed by grouping_model (or model_id if no grouping)
    # This allows demand for a grouping_model to find ALL lots from any model in that family
    lot_ids_by_model: Dict[str, Set[str]] = {}
    for row in wip_rows:
        model_id = row.get("model_id")
        lot_id = row.get("lot_id")
        if model_id and lot_id:
            lookup_key = get_model_lookup_key(model_id, row, model_metadata_lookup)
            if lookup_key not in lot_ids_by_model:
                lot_ids_by_model[lookup_key] = set()
            lot_ids_by_model[lookup_key].add(lot_id)

    # ⚠️ MLWB ADDITION — Tier-1 urgent WIP lots (see urgent_allocation.py). Allocate every
    # urgent lot to completion (oldest urgency_creation_date first) BEFORE the weighted
    # month/model loop, consuming shared capacity in `state`. `pre_allocated_by_model`
    # (units produced per model) is credited against demand in the loop below so Phase 2
    # does not re-produce it.
    pre_allocated_by_model = run_urgent_prepass(
        lots_by_id=lots_by_id,
        lot_max_sequence=lot_max_sequence,
        state=state,
        demand_by_model_month=demand_by_model_month,
        equipment_capacity=equipment_capacity,
        process_to_equipment=process_to_equipment,
        negative_constraints=negative_constraints,
        allocation_results=allocation_results,
        failed_allocation_results=failed_allocation_results,
        revenue_plan_id=revenue_plan_id,
        simulation_id=simulation_id,
        run_id=run_id,
        run_timestamp=run_timestamp,
        start_date=start_date,
        config=config,
        model_metadata_lookup=model_metadata_lookup,
    )

    all_months = sorted(set(month for (_, month) in demand_by_model_month.keys()))
    all_months = [m for m in all_months if m >= config.constraints.start_month]
    print(f"\n=== Processing months in order: {all_months} ===")

    # Track carryover by model - this accumulates unfulfilled demand
    carryover_by_model: Dict[str, int] = {}

    for target_month in all_months:
        print(f"\n--- Processing Month: {target_month} ---")

        # Get all models that have demand this month OR have carryover from previous months
        models_with_demand = set(model for (model, month) in demand_by_model_month.keys() if month == target_month)
        models_with_carryover = set(model for model, carry in carryover_by_model.items() if carry > 0)
        all_models_to_process = models_with_demand | models_with_carryover

        models_this_month = [
            (
                model,
                model_priority_lookup.get((model, target_month), config.defaults.model_priority),
            )
            for model in all_models_to_process
        ]
        models_this_month.sort(key=lambda x: x[1])
        print(
            f"Models to fulfill (by priority): "
            f"{[(m, p) for m, p in models_this_month[:10]]}"
            f"{'...' if len(models_this_month) > 10 else ''}"
        )

        for model_id, model_priority in models_this_month:
            # Calculate total demand: base demand for this month + carryover from previous months
            base_demand = demand_by_model_month.get((model_id, target_month), 0)
            carryover = carryover_by_model.get(model_id, 0)
            total_demand = base_demand + carryover

            if total_demand <= 0:
                carryover_by_model[model_id] = 0
                continue

            print(
                f"\n  Model {model_id} (priority={model_priority}): "
                f"demand={base_demand:,}"
                + (f" + carryover={carryover:,}" if carryover > 0 else "")
                + f" = {total_demand:,} units needed"
            )

            # Phase 1: Allocate existing lots (WIP + previously created virtual lots)
            # units_allocated tracks lots we've scheduled (for allocation decisions)
            units_allocated = _allocate_existing_lots_for_model(
                model_id=model_id,
                model_priority=model_priority,
                target_month=target_month,
                total_demand=total_demand,
                revenue_plan_id=revenue_plan_id,
                lot_ids_by_model=lot_ids_by_model,
                lots_by_id=lots_by_id,
                lot_max_sequence=lot_max_sequence,
                state=state,
                equipment_capacity=equipment_capacity,
                process_to_equipment=process_to_equipment,
                negative_constraints=negative_constraints,
                allocation_results=allocation_results,
                failed_allocation_results=failed_allocation_results,
                run_id=run_id,
                run_timestamp=run_timestamp,
                simulation_id=simulation_id,
                start_date=start_date,
                config=config,
                model_metadata_lookup=model_metadata_lookup,
            )

            print(f"    Phase 1 - Existing lots allocated: {units_allocated:,} units")

            # ⚠️ MLWB ADDITION — credit Tier-1 urgent output (produced in Phase 0) against
            # this model's demand so Phase 2 does not re-produce it. Credit only up to the
            # remaining demand; any surplus rolls forward to later months (avoids the
            # clamp-to-zero loss a negative carryover seed would cause at total_demand<=0).
            urgent_credit = min(pre_allocated_by_model.get(model_id, 0),
                                max(0, total_demand - units_allocated))
            if urgent_credit:
                units_allocated += urgent_credit
                pre_allocated_by_model[model_id] -= urgent_credit
                print(f"    Phase 0 credit - Urgent lots pre-produced: {urgent_credit:,} units")

            # Phase 2: Create virtual lots for any shortfall in allocation
            shortfall_for_allocation = total_demand - units_allocated

            if shortfall_for_allocation > 0:
                print(f"    Phase 2 - Creating virtual lots for: {shortfall_for_allocation:,} units")

                model_metadata = model_metadata_lookup.get(model_id, {})
                model_process_steps = model_process_steps_lookup.get(model_id, [])

                if not model_process_steps:
                    print(f"      ⚠️ No routing for model {model_id} - cannot create virtual lots")

                    customer_name = model_metadata.get("customer_name")
                    end_customer = model_metadata.get("end_customer")
                    model_sales_team = model_metadata.get("sales_team")

                    risk_id = f"{simulation_id}_{model_id}_{target_month}"
                    title = f"Missing Routing: {model_id}"
                    description = (
                        f"Model {model_id} has demand of {shortfall_for_allocation:,} units for {target_month} "
                        f"but no process routing is defined. Virtual lots cannot be created."
                    )
                    remediation = "이 모델의 제조 공정을 정의하려면 공정 엔지니어링에 문의하십시오."

                    unrouted_demand_results.append(
                        UnroutedModelDemand(
                            new_model_risk_id=risk_id,
                            model_id=model_id,
                            demand_qty_ea=shortfall_for_allocation,
                            shortfall_month=target_month,
                            customer_name=customer_name,
                            end_customer=end_customer,
                            model_sales_team=model_sales_team,
                            simulation_id=simulation_id,
                            simulation_name=simulation_name,
                            title=title,
                            new_model_risk_description=description,
                            remediation_suggestion=remediation,
                            allocation_run_id=run_id,
                            allocation_run_ts=run_timestamp,
                        )
                    )
                    # Mark as allocated for tracking purposes (unrouted demand is tracked separately)
                    units_allocated = total_demand
                else:
                    virtual_lots, new_lots_df = create_virtual_lots(
                        model_id=model_id,
                        target_month=target_month,
                        shortfall_units=shortfall_for_allocation,
                        model_metadata=model_metadata,
                        model_process_steps=model_process_steps,
                        run_timestamp=run_timestamp,
                        simulation_id=simulation_id,
                        revenue_plan_id=revenue_plan_id,
                        run_id=run_id,
                        config=config,
                        lot_starts_by_model_date=lot_starts_by_model_date,
                    )
                    all_new_lots_dfs.append(new_lots_df)

                    print(f"      Created {len(new_lots_df)} virtual lots with {len(virtual_lots)} total steps")

                    virtual_lot_groups = group_wip_by_lot(virtual_lots)
                    for vl_lot_id, vl_steps in virtual_lot_groups.items():
                        lots_by_id[vl_lot_id] = vl_steps
                        lot_max_sequence[vl_lot_id] = max(s.get("sequence", 0) for s in vl_steps) if vl_steps else 0
                        # Add virtual lot to lookup using grouping_model key
                        lookup_key = get_model_lookup_key(model_id, None, model_metadata_lookup)
                        if lookup_key not in lot_ids_by_model:
                            lot_ids_by_model[lookup_key] = set()
                        lot_ids_by_model[lookup_key].add(vl_lot_id)

                    vl_units = _allocate_virtual_lots(
                        virtual_lot_ids=list(virtual_lot_groups.keys()),
                        model_id=model_id,
                        model_priority=model_priority,
                        target_month=target_month,
                        total_demand=total_demand,
                        units_fulfilled_so_far=units_allocated,
                        revenue_plan_id=revenue_plan_id,
                        lots_by_id=lots_by_id,
                        lot_max_sequence=lot_max_sequence,
                        state=state,
                        equipment_capacity=equipment_capacity,
                        process_to_equipment=process_to_equipment,
                        negative_constraints=negative_constraints,
                        allocation_results=allocation_results,
                        failed_allocation_results=failed_allocation_results,
                        run_id=run_id,
                        run_timestamp=run_timestamp,
                        simulation_id=simulation_id,
                        start_date=start_date,
                        config=config,
                        model_metadata_lookup=model_metadata_lookup,
                    )
                    units_allocated += vl_units
                    print(f"      Virtual lots allocated: {vl_units:,} units")

            # =================================================================
            # CARRYOVER CALCULATION: Based on ALLOCATION, not on-time completion
            # =================================================================
            # Carryover = demand we couldn't allocate at all
            # Once lots are allocated (even if they complete late), the demand is covered
            # This prevents over-production from creating duplicate lots
            final_shortfall = max(0, total_demand - units_allocated)

            print(f"    ✓ Total allocated: {units_allocated:,} units")

            if final_shortfall > 0:
                print(f"    → Carrying forward to next month: {final_shortfall:,} units (couldn't allocate)")
                carryover_by_model[model_id] = final_shortfall
            else:
                carryover_by_model[model_id] = 0

    # ⚠️ MLWB ADDITION — audit trail: op codes assumed outsourced this run.
    print_outsourced_summary(state)

    allocation_df = _create_allocation_dataframe(allocation_results)
    failed_allocations_df = _create_failed_allocations_dataframe(failed_allocation_results, simulation_name)
    new_lots_df = _create_new_lots_dataframe(all_new_lots_dfs)
    unrouted_demand_df = _create_unrouted_demand_dataframe(unrouted_demand_results)

    print(f"\n=== Simulation {simulation_id} Summary ===")
    print(f"Total allocation rows: {allocation_df.height:,}")
    print(f"Total failed allocations: {failed_allocations_df.height:,}")
    print(f"Total virtual lots created: {new_lots_df.height:,}")
    print(f"Total unrouted model demands: {unrouted_demand_df.height:,}")

    return allocation_df, new_lots_df, unrouted_demand_df, failed_allocations_df


# =============================================================================
# Allocation Functions
# =============================================================================


def _select_lots_for_demand(
    lot_ids: Set[str],
    lots_by_id: Dict[str, List[dict]],
    state: AllocationState,
    target_units: int,
    config: AllocationConfig,
) -> List[str]:
    """
    Select only enough lots to fulfill demand * buffer.

    Uses a MIN heap to prioritize lots CLOSEST to completion first.
    This ensures we work on lots that can finish soonest and avoid
    clogging capacity with lots that won't be needed.

    Args:
        lot_ids: Set of all lot IDs for this model
        lots_by_id: Lookup of lot_id -> list of steps
        state: Current allocation state
        target_units: Demand units to fulfill
        config: Allocation configuration

    Returns:
        List of lot_ids to work on, ordered by remaining steps (ascending)
    """
    columns = config.columns
    buffer = config.constraints.demand_fulfillment_buffer

    target_with_buffer = int(target_units * buffer)

    lot_candidates = []
    for lot_id in lot_ids:
        remaining = count_remaining_steps(lot_id, lots_by_id, state)
        if remaining > 0:
            lot_steps = lots_by_id.get(lot_id, [])
            unit_qty = 0
            if lot_steps:
                unit_qty = lot_steps[0].get(columns.unit_quantity) or 0
            lot_candidates.append((remaining, unit_qty, lot_id))

    lot_candidates.sort(key=lambda x: x[0])

    selected_lots = []
    potential_units = 0

    for remaining, unit_qty, lot_id in lot_candidates:
        selected_lots.append(lot_id)
        potential_units += unit_qty

        if potential_units >= target_with_buffer:
            break

    return selected_lots


def _allocate_existing_lots_for_model(
    model_id: str,
    model_priority: int,
    target_month: str,
    total_demand: int,
    revenue_plan_id: str,
    lot_ids_by_model: Dict[str, Set[str]],
    lots_by_id: Dict[str, List[dict]],
    lot_max_sequence: Dict[str, int],
    state: AllocationState,
    equipment_capacity: Dict[str, int],
    process_to_equipment: Dict[str, List[str]],
    negative_constraints: Dict[Tuple[str, str], Set[str]],
    allocation_results: List[LotStepAllocation],
    failed_allocation_results: List[LotStepAllocation],
    run_id: str,
    run_timestamp: datetime,
    simulation_id: str,
    start_date: date,
    config: AllocationConfig,
    model_metadata_lookup: Dict[str, Dict[str, Any]] = None,
) -> int:
    """
    Allocate existing lots (WIP + previously-created virtual lots) for a model.

    Selection: Picks lots CLOSEST to completion to fulfill demand * buffer.
    Allocation: Uses a MAX heap to advance lots FURTHEST from completion first.

    This ensures we:
    1. Only work on enough lots to meet demand (avoid capacity clogging)
    2. Within selected lots, advance the ones furthest back first

    Returns:
        Units fulfilled from existing lots.
    """
    units_fulfilled = 0

    # Look up lots using grouping_model as key (or model_id if no grouping)
    # This ensures we find ALL lots that can fulfill this demand
    lookup_key = get_model_lookup_key(model_id, None, model_metadata_lookup)
    lot_ids = lot_ids_by_model.get(lookup_key, set())

    if not lot_ids:
        return 0

    selected_lot_ids = _select_lots_for_demand(
        lot_ids=lot_ids,
        lots_by_id=lots_by_id,
        state=state,
        target_units=total_demand,
        config=config,
    )

    if not selected_lot_ids:
        return 0

    print(
        f"      Selected {len(selected_lot_ids)} of {len(lot_ids)} lots "
        f"(buffer={config.constraints.demand_fulfillment_buffer}x)"
    )

    lot_heap = []
    for lot_id in selected_lot_ids:
        remaining = count_remaining_steps(lot_id, lots_by_id, state)
        if remaining > 0:
            heapq.heappush(lot_heap, (-remaining, lot_id))

    while lot_heap and units_fulfilled < total_demand:
        neg_remaining, lot_id = heapq.heappop(lot_heap)

        next_step = get_next_allocatable_step(lot_id, lots_by_id, state)
        if next_step is None:
            continue

        is_new_lot = next_step.get("is_new_lot", False) or lot_id.startswith("VL-")

        result = _try_allocate_lot_step(
            lot_step=next_step,
            model_id=model_id,
            model_priority=model_priority,
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

        if result.success:
            alloc = _build_allocation_record(
                lot_step=next_step,
                result=result,
                model_id=model_id,
                model_priority=model_priority,
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
            allocation_results.append(alloc)
            units_fulfilled += alloc.units_produced

            new_remaining = count_remaining_steps(lot_id, lots_by_id, state)
            if new_remaining > 0:
                heapq.heappush(lot_heap, (-new_remaining, lot_id))
        else:
            failed_alloc = _build_allocation_record(
                lot_step=next_step,
                result=result,
                model_id=model_id,
                model_priority=model_priority,
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
            failed_allocation_results.append(failed_alloc)

            process_id = next_step.get("process_id", "UNKNOWN")
            print(
                f"        ✗ Blocked at step {next_step.get('sequence')} for lot {lot_id} : "
                f"{result.blocking_reason} - process_id: {process_id} "
                f"[{failed_alloc.allocation_status.value}]"
            )
            break
    return units_fulfilled


def _allocate_virtual_lots(
    virtual_lot_ids: List[str],
    model_id: str,
    model_priority: int,
    target_month: str,
    total_demand: int,
    units_fulfilled_so_far: int,
    revenue_plan_id: str,
    lots_by_id: Dict[str, List[dict]],
    lot_max_sequence: Dict[str, int],
    state: AllocationState,
    equipment_capacity: Dict[str, int],
    process_to_equipment: Dict[str, List[str]],
    negative_constraints: Dict[Tuple[str, str], Set[str]],
    allocation_results: List[LotStepAllocation],
    failed_allocation_results: List[LotStepAllocation],
    run_id: str,
    run_timestamp: datetime,
    simulation_id: str,
    start_date: date,
    config: AllocationConfig,
    model_metadata_lookup: Dict[str, Dict[str, Any]] = None,
) -> int:
    """
    Allocate virtual lots one at a time, completing each lot before starting next.

    Returns:
        Additional units fulfilled from virtual lots.
    """
    vl_units_fulfilled = 0

    for lot_idx, lot_id in enumerate(virtual_lot_ids):
        if (units_fulfilled_so_far + vl_units_fulfilled) >= total_demand:
            print(f"      Demand satisfied after {lot_idx} virtual lots")
            break

        lot_steps = lots_by_id.get(lot_id, [])
        if not lot_steps:
            continue

        while True:
            next_step = get_next_allocatable_step(lot_id, lots_by_id, state)
            if next_step is None:
                break

            process_id = next_step.get("process_id")
            if not process_id:
                print(f"        ⚠️ Step {next_step.get('sequence')} missing process_id")
                break

            result = _try_allocate_lot_step(
                lot_step=next_step,
                model_id=model_id,
                model_priority=model_priority,
                target_month=target_month,
                is_new_lot=True,
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

            if result.success:
                alloc = _build_allocation_record(
                    lot_step=next_step,
                    result=result,
                    model_id=model_id,
                    model_priority=model_priority,
                    target_month=target_month,
                    is_new_lot=True,
                    revenue_plan_id=revenue_plan_id,
                    simulation_id=simulation_id,
                    run_id=run_id,
                    run_timestamp=run_timestamp,
                    lots_by_id=lots_by_id,
                    lot_max_sequence=lot_max_sequence,
                    state=state,
                    config=config,
                )
                allocation_results.append(alloc)

                if alloc.units_produced > 0:
                    vl_units_fulfilled += alloc.units_produced
                    break
            else:
                failed_alloc = _build_allocation_record(
                    lot_step=next_step,
                    result=result,
                    model_id=model_id,
                    model_priority=model_priority,
                    target_month=target_month,
                    is_new_lot=True,
                    revenue_plan_id=revenue_plan_id,
                    simulation_id=simulation_id,
                    run_id=run_id,
                    run_timestamp=run_timestamp,
                    lots_by_id=lots_by_id,
                    lot_max_sequence=lot_max_sequence,
                    state=state,
                    config=config,
                )
                failed_allocation_results.append(failed_alloc)

                print(
                    f"        ✗ Blocked at step {next_step.get('sequence')} for lot {lot_id} : "
                    f"{result.blocking_reason} - process_id: {process_id} "
                    f"[{failed_alloc.allocation_status.value}]"
                )
                break

    return vl_units_fulfilled


# =============================================================================
# ⚠️ MLWB DIVERGENCE — saturation scan memo (opt-in: REVPLAN_SCAN_MEMO=1)
# =============================================================================
# WHY: within one allocation run, capacity_usage only ever GROWS — a (machine,
# day) that could not fit q sheets stays unable to fit >= q sheets forever.
# The stock engine has no memory of that: when demand outruns capacity it
# re-scans the same provably-full days for every lot in the batch (100 virtual
# lots x 90 days x N machines of identical, pre-decided failures = the
# overnight runs of 2026-07). Two layers, both sound under the monotonic-fill
# invariant, both OFF unless REVPLAN_SCAN_MEMO=1:
#
#   L1 oversize precheck — if sheet_qty exceeds the largest TOTAL daily
#      capacity among the step's unblocked candidate machines, no day can ever
#      fit it (loaded or empty). Fail immediately with the same terminal
#      status/reason the 90-day scan would have produced (incl. the horizon-
#      exceeded case), replicating the loop's one state mutation (day-1
#      fast-track designation) so other lots see identical state.
#      Measured 2026-07-30: VLs are cut at ~30 sheets vs median DailyCapa 5 —
#      most VL steps are oversize-for-everything, each burning a full scan.
#
#   L2 full-day memo — (process, blocked-set, day) -> smallest sheet_qty
#      PROVEN not to fit that day. A later scan of the same key with qty >=
#      proven skips the per-machine sweep (one summary DelayRecord instead of
#      one per machine) and walks on to the next day. Allocation decisions,
#      dates and failure statuses are unchanged; only delay_reasons /
#      capacity_shortage diagnostics get thinner on deduped days.
#
# Kill switch: unset REVPLAN_SCAN_MEMO (or =0) restores byte-identical
# Palantir behaviour. Validate by diffing SIM tables of one run with/without.
# =============================================================================
def _scan_memo_enabled() -> bool:
    """Read the switch lazily so the notebook can set it any time before run."""
    return _os.environ.get("REVPLAN_SCAN_MEMO", "0") == "1"


def _get_scan_memo(state: AllocationState) -> dict:
    """(process_id, frozenset(blocked), date) -> min sheet_qty proven unfittable.

    Attached onto AllocationState (same pattern as install_plan_lt_lookup) so its
    lifetime — and the monotonic-fill invariant it relies on — matches the run's.
    """
    memo = getattr(state, "_mlwb_scan_memo", None)
    if memo is None:
        memo = {}
        state._mlwb_scan_memo = memo
    return memo


def _try_allocate_lot_step(
    lot_step: dict,
    model_id: str,
    model_priority: int,
    target_month: str,
    is_new_lot: bool,
    revenue_plan_id: str,
    state: AllocationState,
    equipment_capacity: Dict[str, int],
    process_to_equipment: Dict[str, List[str]],
    negative_constraints: Dict[Tuple[str, str], Set[str]],
    lots_by_id: Dict[str, List[dict]],
    lot_max_sequence: Dict[str, int],
    start_date: date,
    config: AllocationConfig,
    model_metadata_lookup: Dict[str, Dict[str, Any]] = None,
) -> AllocationAttemptResult:
    """
    Try to allocate a single lot step.

    Uses process_id directly to find available equipment instead of equipment_groups.

    FAST PATH: Final steps with hold-until-target-month equipment groups (e.g., 입고대기)
    are allocated directly to the first day of target month without capacity search,
    since they have infinite capacity configured.

    Returns:
        AllocationAttemptResult with success status, allocated date, equipment,
        and detailed delay reasons.
    """
    lot_id = lot_step.get("lot_id")
    process_id = lot_step.get("process_id")
    sequence = lot_step.get("sequence")
    equipment_group = lot_step.get("equipment_group_id")
    is_final_step = lot_step.get("_is_final_step", False)

    if not lot_id or not process_id or sequence is None:
        return AllocationAttemptResult(
            success=False,
            blocking_reason="Missing required fields (lot_id, process_id, or sequence)",
        )

    # =========================================================================
    # FAST PATH: Final step with hold-until-target-month equipment group
    # These have infinite capacity, so allocate directly without searching
    # =========================================================================
    if (
        is_final_step
        and config.constraints.enable_hold_until_target_month
        and equipment_group in config.constraints.hold_until_target_month_equipment_groups
    ):
        # Parse target month first day
        target_year = int(target_month[:4])
        target_month_num = int(target_month[4:6])
        target_month_first_day = date(target_year, target_month_num, 1)

        # Get previous step completion date
        prev_step_date = state.get_lot_last_step_date(lot_id)

        # Determine allocation date: max of (prev_step_date, target_month_first_day)
        if prev_step_date and prev_step_date >= target_month_first_day:
            allocation_date = prev_step_date
        else:
            allocation_date = target_month_first_day

        # Pick any equipment for this process (infinite capacity, doesn't matter)
        available_equipment = process_to_equipment.get(process_id, [])
        selected_equipment = available_equipment[0] if available_equipment else None

        # Update state (skip capacity tracking since it's infinite)
        state.mark_step_allocated(lot_id, process_id, sequence)
        state.set_lot_last_step_date(lot_id, allocation_date)
        if selected_equipment:
            state.add_equipment_to_path(lot_id, selected_equipment)

        return AllocationAttemptResult(
            success=True,
            allocated_date=allocation_date,
            equipment_id=selected_equipment,
            equipment_group=equipment_group,
            days_delayed=0,
            delay_records=[],  # No delays - just waiting for target month
        )

    available_equipment = process_to_equipment.get(process_id, [])
    if not available_equipment:
        # =====================================================================
        # LOGICAL NO-OP PASS-THROUGH (Gumi workshop 2026-07, meeting_summary §10)
        # Only ~149 of ~397 routing ops have equipment constraints; the rest are
        # outsourced OR logical no-ops (e.g. M000N "LOT START"/receiving). A step
        # whose op-code is configured logical passes through here — allocated, no
        # capacity consumed — instead of failing. Real machine ops missing from
        # the constraints table STILL block below (bottlenecks not silently erased).
        # =====================================================================
        if process_id in config.constraints.logical_passthrough_operation_codes:
            prev_step_date = state.get_lot_last_step_date(lot_id)
            if prev_step_date:
                allocation_date = prev_step_date
            elif is_new_lot:
                allocation_date = max(lot_step.get("target_lot_start_date", start_date), start_date)
            else:
                allocation_date = start_date

            state.mark_step_allocated(lot_id, process_id, sequence)
            state.set_lot_last_step_date(lot_id, allocation_date)

            return AllocationAttemptResult(
                success=True,
                allocated_date=allocation_date,
                equipment_id=None,
                equipment_group=equipment_group,
                days_delayed=0,
                delay_records=[],
            )

        # =====================================================================
        # ⚠️ MLWB ADDITION (customer meeting 2026-07-15 §4): unmapped op = 외주.
        # Planned-complete after its Plan LT instead of failing — the lot flows
        # on. Returns None only when treat_unmapped_as_outsourced is off, which
        # falls through to the original FAILED_NO_EQUIPMENT behaviour.
        # See outsourced_allocation.py for semantics + audit trail.
        # =====================================================================
        outsourced_result = try_allocate_outsourced_step(
            lot_step=lot_step,
            model_id=model_id,
            state=state,
            start_date=start_date,
            is_new_lot=is_new_lot,
            config=config,
        )
        if outsourced_result is not None:
            return outsourced_result

        return AllocationAttemptResult(
            success=False,
            blocking_reason=f"No equipment mapped to process {process_id}",
        )

    grouping_model = lot_step.get("grouping_model")
    if not grouping_model and model_metadata_lookup and model_id in model_metadata_lookup:
        grouping_model = model_metadata_lookup[model_id].get("grouping_model") or model_id
    grouping_model = grouping_model or model_id

    blocked_equipment = negative_constraints.get((grouping_model, process_id), set()) if negative_constraints else set()
    unblocked_equipment = [eq for eq in available_equipment if eq not in blocked_equipment]
    if not unblocked_equipment:
        return AllocationAttemptResult(
            success=False,
            blocking_reason=f"All equipment for process {process_id} is blocked for model (grouping_model={grouping_model})",
        )

    columns = config.columns
    constraints = config.constraints

    sheet_qty = lot_step.get(columns.sheet_quantity) or 0
    target_start = lot_step.get("target_lot_start_date", start_date)

    prev_step_date = state.get_lot_last_step_date(lot_id)
    if prev_step_date:
        earliest_start = prev_step_date
    elif is_new_lot:
        earliest_start: date = max(target_start, start_date)
    else:
        if constraints.use_dynamic_wip_earliest_start:
            earliest_start = calculate_dynamic_wip_earliest_start(
                lot_id=lot_id,
                lot_step=lot_step,
                target_month=target_month,
                lots_by_id=lots_by_id,
                state=state,
                start_date=start_date,
                config=config,
            )
        else:
            earliest_start = start_date

    target_eom = parse_month_to_eom_date(target_month)
    has_logged_insufficient_capacity = False

    # =========================================================================
    # ⚠️ MLWB DIVERGENCE L1 (opt-in REVPLAN_SCAN_MEMO=1) — oversize precheck.
    # A lot larger than every candidate machine's TOTAL daily capacity can
    # never fit on any day; skip the scan and emit the identical terminal
    # verdict the full loop would have reached (see block comment above
    # _scan_memo_enabled). Outcome parity notes:
    #   * day-0 horizon exit happens BEFORE fast-track designation (loop order);
    #   * otherwise the doomed lot still claims its day-1 fast-track slot
    #     (a real state mutation other lots must see identically);
    #   * horizon vs INSUFFICIENT_CAPACITY chosen by whichever the walk would
    #     have hit first; days_delayed matches the walked count.
    # =========================================================================
    _memo_on = _scan_memo_enabled()
    if _memo_on and sheet_qty > 0:
        _max_total_cap = max(
            equipment_capacity.get(eq, config.defaults.daily_capacity_sheets)
            for eq in unblocked_equipment
        )
        if sheet_qty > _max_total_cap:
            _jan1_horizon = date(constraints.max_allocation_year, 1, 1)
            _days_to_horizon = max(0, (_jan1_horizon - earliest_start).days)

            if _days_to_horizon == 0:
                # loop iteration 1 would return on its horizon check, before
                # designating fast-track or logging any delay record
                return AllocationAttemptResult(
                    success=False,
                    days_delayed=0,
                    delay_records=[],
                    blocking_reason=f"Scheduling horizon exceeded (reached {earliest_start.year})",
                )

            _hit_horizon = _days_to_horizon < constraints.max_delay_days
            _days_final = _days_to_horizon if _hit_horizon else constraints.max_delay_days
            _end_date = earliest_start + timedelta(days=_days_final)

            # replicate the loop's ONE state mutation: the fast-track claim.
            # Baseline retries the claim each scanned day (the slot counter is
            # per-date), so the doomed lot grabs a slot on the FIRST walked day
            # with one free — other lots must see the identical claim.
            if not state.is_lot_fast_track(lot_id) and model_priority <= constraints.fast_track_priority_threshold:
                for _k in range(_days_final):
                    _d = earliest_start + timedelta(days=_k)
                    if state.get_fast_track_count(_d) < constraints.fast_track_lots_per_day:
                        state.designate_lot_fast_track(lot_id, _d)
                        break

            _records: List[DelayRecord] = [
                # one summary record in place of days x machines identical ones
                DelayRecord(
                    constraint_type=ConstraintType.EQUIPMENT_CAPACITY_FULL,
                    delay_date=earliest_start,
                    equipment_id="ALL_CANDIDATES_OVERSIZE",
                    capacity_used=0,
                    capacity_total=_max_total_cap,
                    capacity_needed=sheet_qty,
                )
            ]
            _insufficient_log_date = max(earliest_start, target_eom + timedelta(days=1))
            if _insufficient_log_date <= _end_date:
                _records.append(
                    DelayRecord(
                        constraint_type=ConstraintType.INSUFFICIENT_CAPACITY,
                        delay_date=_insufficient_log_date,
                        equipment_group=lot_step.get("equipment_group_id"),
                    )
                )

            if _hit_horizon:
                _reason = f"Scheduling horizon exceeded (reached {_end_date.year})"
            else:
                _reason = (
                    f"INSUFFICIENT_CAPACITY: Could not allocate within {constraints.max_delay_days} days"
                    f" [oversize: needs {sheet_qty} sheets > max candidate machine capacity {_max_total_cap}/day]"
                )
            return AllocationAttemptResult(
                success=False,
                days_delayed=_days_final,
                delay_records=_records,
                blocking_reason=_reason,
            )

    # ⚠️ MLWB DIVERGENCE L2 lookup key (only consulted when _memo_on)
    _memo = _get_scan_memo(state) if _memo_on else None
    _blocked_key = frozenset(blocked_equipment) if _memo_on else None

    current_date = earliest_start
    days_delayed = 0
    delay_records: List[DelayRecord] = []

    while days_delayed < constraints.max_delay_days:
        if current_date.year >= constraints.max_allocation_year:
            return AllocationAttemptResult(
                success=False,
                days_delayed=days_delayed,
                delay_records=delay_records,
                blocking_reason=f"Scheduling horizon exceeded (reached {current_date.year})",
            )

        is_fast_track = state.is_lot_fast_track(lot_id)

        if not is_fast_track:
            if model_priority <= constraints.fast_track_priority_threshold:
                fast_track_used = state.get_fast_track_count(current_date)
                if fast_track_used < constraints.fast_track_lots_per_day:
                    state.designate_lot_fast_track(lot_id, current_date)
                    is_fast_track = True

        max_steps_today = (
            constraints.max_steps_per_lot_per_day_fast_track if is_fast_track else constraints.max_steps_per_lot_per_day
        )

        steps_today = state.get_lot_steps_today(lot_id, current_date)
        if steps_today >= max_steps_today:
            delay_records.append(
                DelayRecord(
                    constraint_type=ConstraintType.MAX_STEPS_PER_LOT_PER_DAY,
                    delay_date=current_date,
                )
            )
            current_date += timedelta(days=1)
            days_delayed += 1
            continue

        # =====================================================================
        # ⚠️ MLWB DIVERGENCE L2 (opt-in REVPLAN_SCAN_MEMO=1): a day already
        # PROVEN unable to fit <= sheet_qty for this (process, blocked-set)
        # cannot fit it now — capacity only ever fills within a run. Skip the
        # per-machine sweep; keep the day walk (dates/days_delayed identical),
        # emit ONE summary record instead of one per machine, and mirror the
        # loop's target-EOM INSUFFICIENT_CAPACITY logging before advancing.
        # =====================================================================
        if _memo_on:
            _proven = _memo.get((process_id, _blocked_key, current_date))
            if _proven is not None and sheet_qty >= _proven:
                delay_records.append(
                    DelayRecord(
                        constraint_type=ConstraintType.EQUIPMENT_CAPACITY_FULL,
                        delay_date=current_date,
                        equipment_id="ALL_CANDIDATES_MEMO",
                        capacity_needed=sheet_qty,
                    )
                )
                if not has_logged_insufficient_capacity and current_date > target_eom:
                    delay_records.append(
                        DelayRecord(
                            constraint_type=ConstraintType.INSUFFICIENT_CAPACITY,
                            delay_date=current_date,
                            equipment_group=lot_step.get("equipment_group_id"),
                        )
                    )
                    has_logged_insufficient_capacity = True
                current_date += timedelta(days=1)
                days_delayed += 1
                continue

        selected_equipment, remaining_cap, blocked_with_capacity = find_least_loaded_equipment_for_process(
            process_id=process_id,
            target_date=current_date,
            process_to_equipment=process_to_equipment,
            equipment_capacity=equipment_capacity,
            capacity_usage=state.capacity_usage,
            required_sheets=sheet_qty,
            config=config,
            negative_constraints=negative_constraints,
            model_id=model_id,
            grouping_model=grouping_model,
        )

        if selected_equipment:
            state.add_capacity_usage(selected_equipment, current_date, sheet_qty)
            state.increment_lot_steps(lot_id, current_date)
            state.mark_step_allocated(lot_id, process_id, sequence)
            state.set_lot_last_step_date(lot_id, current_date)
            state.add_equipment_to_path(lot_id, selected_equipment)

            return AllocationAttemptResult(
                success=True,
                allocated_date=current_date,
                equipment_id=selected_equipment,
                equipment_group=None,
                days_delayed=days_delayed,
                delay_records=delay_records,
            )
        else:
            for equip_id, used_cap, total_cap in blocked_with_capacity:
                delay_records.append(
                    DelayRecord(
                        constraint_type=ConstraintType.EQUIPMENT_BLOCKED_FOR_MODEL,
                        delay_date=current_date,
                        equipment_id=equip_id,
                        capacity_used=used_cap,
                        capacity_total=total_cap,
                        capacity_needed=sheet_qty,
                    )
                )

            for equip_id in available_equipment:
                if equip_id in blocked_equipment:
                    continue
                total_cap = equipment_capacity.get(equip_id, config.defaults.daily_capacity_sheets)
                used_cap = state.get_capacity_used(equip_id, current_date)
                if used_cap + sheet_qty > total_cap:
                    delay_records.append(
                        DelayRecord(
                            constraint_type=ConstraintType.EQUIPMENT_CAPACITY_FULL,
                            delay_date=current_date,
                            equipment_id=equip_id,
                            capacity_used=used_cap,
                            capacity_total=total_cap,
                            capacity_needed=sheet_qty,
                        )
                    )

            # ⚠️ MLWB DIVERGENCE L2: record the proof — sheet_qty did not fit
            # this (process, blocked-set, day). Monotonic fill makes it final;
            # keep the SMALLEST proven quantity so the guard stays sound.
            if _memo_on:
                _mk = (process_id, _blocked_key, current_date)
                _prev = _memo.get(_mk)
                if _prev is None or sheet_qty < _prev:
                    _memo[_mk] = sheet_qty

        if not has_logged_insufficient_capacity and current_date > target_eom:
            equipment_group = lot_step.get("equipment_group_id")
            delay_records.append(
                DelayRecord(
                    constraint_type=ConstraintType.INSUFFICIENT_CAPACITY,
                    delay_date=current_date,
                    equipment_group=equipment_group,
                )
            )
            has_logged_insufficient_capacity = True

        current_date += timedelta(days=1)
        days_delayed += 1

    if not has_logged_insufficient_capacity:
        equipment_group = lot_step.get("equipment_group_id")
        delay_records.append(
            DelayRecord(
                constraint_type=ConstraintType.INSUFFICIENT_CAPACITY,
                delay_date=current_date,
                equipment_group=equipment_group,
            )
        )

    return AllocationAttemptResult(
        success=False,
        days_delayed=days_delayed,
        delay_records=delay_records,
        blocking_reason=f"INSUFFICIENT_CAPACITY: Could not allocate within {constraints.max_delay_days} days",
    )


def _determine_failure_status(result: AllocationAttemptResult) -> AllocationStatus:
    """Determine the appropriate failure status from an allocation result."""
    if result.success:
        return AllocationStatus.ALLOCATED

    blocking_reason = result.blocking_reason or ""
    reason_lower = blocking_reason.lower()

    if "horizon" in reason_lower:
        return AllocationStatus.FAILED_HORIZON_EXCEEDED
    elif "no equipment" in reason_lower or "all equipment" in reason_lower:
        return AllocationStatus.FAILED_NO_EQUIPMENT
    else:
        return AllocationStatus.FAILED_INSUFFICIENT_CAPACITY


def _build_allocation_record(
    lot_step: dict,
    result: AllocationAttemptResult,
    model_id: str,
    model_priority: int,
    target_month: str,
    is_new_lot: bool,
    revenue_plan_id: str,
    simulation_id: str,
    run_id: str,
    run_timestamp: datetime,
    lots_by_id: Dict[str, List[dict]],
    lot_max_sequence: Dict[str, int],
    state: AllocationState,
    config: AllocationConfig,
) -> LotStepAllocation:
    """Build a LotStepAllocation record from allocation result (success or failure)."""
    lot_id = lot_step.get("lot_id")
    process_id = lot_step.get("process_id")
    sequence = lot_step.get("sequence")

    equipment_group = lot_step.get("equipment_group_id")

    columns = config.columns

    sheet_qty = lot_step.get(columns.sheet_quantity) or 0
    unit_qty = lot_step.get(columns.unit_quantity) or 0

    lot_steps = lots_by_id.get(lot_id, [])
    total_steps = len(lot_steps)
    steps_allocated = sum(
        1 for s in lot_steps if state.is_step_allocated(lot_id, s.get("process_id"), s.get("sequence"))
    )
    remaining_steps = total_steps - steps_allocated

    allocation_status = _determine_failure_status(result)

    if result.success:
        max_seq = lot_max_sequence.get(lot_id, 0)
        is_complete = remaining_steps == 0 and sequence == max_seq

        actual_month = result.allocated_date.strftime("%Y%m")
        is_delayed_completion = actual_month > target_month

        return LotStepAllocation(
            allocation_id=f"{simulation_id}_{lot_id}_{sequence}",
            lot_id=lot_id,
            model_id=model_id,
            process_id=process_id,
            sequence=sequence,
            equipment_group=equipment_group,
            equipment_id=result.equipment_id,
            allocated_date=result.allocated_date,
            target_month=target_month,
            actual_completion_month=actual_month,
            is_delayed_completion=is_delayed_completion,
            simulation_id=simulation_id,
            revenue_plan_id=revenue_plan_id,
            model_priority=model_priority,
            remaining_steps=remaining_steps,
            capacity_used_sheets=sheet_qty,
            units_produced=unit_qty if is_complete else 0,
            days_delayed=result.days_delayed,
            delay_reasons=result.format_delay_reasons(),
            is_new_lot=is_new_lot,
            allocation_status=allocation_status,
            allocation_run_id=run_id,
            allocation_run_ts=run_timestamp,
        )
    else:
        delay_reasons = result.format_delay_reasons()
        if result.blocking_reason:
            if delay_reasons:
                delay_reasons = f"{delay_reasons}; BLOCKED: {result.blocking_reason}"
            else:
                delay_reasons = f"BLOCKED: {result.blocking_reason}"

        return LotStepAllocation(
            allocation_id=f"{simulation_id}_{lot_id}_{sequence}_FAILED",
            lot_id=lot_id,
            model_id=model_id,
            process_id=process_id,
            sequence=sequence,
            equipment_group=equipment_group,
            equipment_id=None,
            allocated_date=None,
            target_month=target_month,
            actual_completion_month=None,
            is_delayed_completion=True,
            simulation_id=simulation_id,
            revenue_plan_id=revenue_plan_id,
            model_priority=model_priority,
            remaining_steps=remaining_steps + 1,
            capacity_used_sheets=sheet_qty,
            units_produced=unit_qty,
            days_delayed=result.days_delayed,
            delay_reasons=delay_reasons,
            is_new_lot=is_new_lot,
            allocation_status=allocation_status,
            allocation_run_id=run_id,
            allocation_run_ts=run_timestamp,
        )


# =============================================================================
# DataFrame Builders
# =============================================================================


def _create_allocation_dataframe(
    allocations: List[LotStepAllocation],
) -> pl.DataFrame:
    """Create allocation DataFrame from successful allocations only."""
    if not allocations:
        return pl.DataFrame(schema=ALLOCATION_OUTPUT_SCHEMA)

    records = [a.to_dict() for a in allocations]

    # Ensure delay_reasons is always a string (never None)
    for record in records:
        if record.get("delay_reasons") is None:
            record["delay_reasons"] = ""

    df = pl.DataFrame(records)

    # Add missing enrichment columns as null strings
    for col_name in [
        "model_customer_name",
        "model_end_customer",
        "model_sales_team",
        "grouping_model",
        "unit_process_name",
        "simulation_name",
        "simulation_revenue_id",
    ]:
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col_name))

    # Add missing numeric enrichment columns
    for col_name in ["final_production_units"]:
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Int32).alias(col_name))

    for col_name in ["total_revenue", "total_margin"]:
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

    return df.select(list(ALLOCATION_OUTPUT_SCHEMA.keys()))


def _create_failed_allocations_dataframe(
    failed_allocations: List[LotStepAllocation],
    simulation_name: str = None,
) -> pl.DataFrame:
    """Create failed allocations DataFrame."""
    if not failed_allocations:
        return pl.DataFrame(schema=FAILED_ALLOCATION_SCHEMA)

    records = []
    for alloc in failed_allocations:
        # Ensure all string fields are strings, not None
        records.append(
            {
                "failed_allocation_id": alloc.allocation_id,
                "lot_id": alloc.lot_id,
                "model_id": alloc.model_id,
                "process_id": alloc.process_id,
                "sequence": alloc.sequence,
                "equipment_group": alloc.equipment_group or "",
                "target_month": alloc.target_month,
                "simulation_id": alloc.simulation_id,
                "simulation_name": simulation_name or "",
                "revenue_plan_id": alloc.revenue_plan_id or "",
                "model_priority": alloc.model_priority,
                "remaining_steps": alloc.remaining_steps,
                "capacity_needed_sheets": alloc.capacity_used_sheets,
                "units_in_lot": alloc.units_produced,
                "days_searched": alloc.days_delayed,
                "failure_reason": alloc.delay_reasons or "",
                "failure_status": alloc.allocation_status.value,
                "delay_details": alloc.delay_reasons or "",
                "is_new_lot": alloc.is_new_lot,
                "allocation_run_id": alloc.allocation_run_id,
                "allocation_run_ts": alloc.allocation_run_ts,
            }
        )

    return pl.DataFrame(records, schema=FAILED_ALLOCATION_SCHEMA)


def _create_new_lots_dataframe(all_dfs: List[pl.DataFrame]) -> pl.DataFrame:
    """Create new lots DataFrame from list of DataFrames."""
    if not all_dfs:
        return pl.DataFrame(schema=NEW_LOTS_OUTPUT_SCHEMA)
    return pl.concat(all_dfs)


def _create_unrouted_demand_dataframe(
    demands: List[UnroutedModelDemand],
) -> pl.DataFrame:
    """Create unrouted demand DataFrame from results."""
    if not demands:
        return pl.DataFrame(schema=UNROUTED_MODEL_DEMAND_SCHEMA)

    records = [d.to_dict() for d in demands]
    return pl.DataFrame(records, schema=UNROUTED_MODEL_DEMAND_SCHEMA)


# =============================================================================
# Transform Definition
# =============================================================================


@incremental(
    snapshot_inputs=[
        "wip_lots",
        "net_demand",
        "model_priorities",
        "equipment_capacity",
        "equipment_constraints",
        "equipment_to_process",
        "model_dataset",
        "model_unit_conversion",
        "planned_process_steps",
        "simulation_config",
    ]
)
@transform.using(
    allocation_output=Output("ri.foundry.main.dataset.b84b2bb5-1cbf-4970-8f73-0e1eb9d4148f"),
    new_lots_created_output=Output("ri.foundry.main.dataset.b2420a83-e21e-45e4-a412-98ffb3451381"),
    unrouted_model_demand_output=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/unrouted_model_demand"
    ),
    failed_allocations_output=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/failed_allocations"
    ),
    run_tracker_output=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/allocation_run_tracker"
    ),
    wip_lots=Input("ri.foundry.main.dataset.66a29010-1e03-4080-a5ed-a52c109de1ee"),
    net_demand=Input("ri.foundry.main.dataset.122a25b5-165e-4eba-adfa-e7865add43d1"),
    model_priorities=Input("ri.foundry.main.dataset.19b16719-7cad-410d-b77d-cb03409dad14"),
    equipment_capacity=Input("ri.foundry.main.dataset.e41de987-f1f7-4576-bcbb-7d662d529a6b"),
    equipment_constraints=Input("ri.foundry.main.dataset.49eb30f7-fc94-4f66-b39c-010bb6cf52b8"),
    equipment_to_process=Input("ri.foundry.main.dataset.ebcd8101-59c4-4630-8fc3-0ab916cd3619"),
    model_dataset=Input("ri.foundry.main.dataset.2573f6cb-7e22-499e-b7e1-be7d895b1d1b"),
    model_unit_conversion=Input("ri.foundry.main.dataset.b806e6e0-24d7-4241-9568-4d92700bc7ef"),
    planned_process_steps=Input("ri.foundry.main.dataset.ea5a565d-e3c1-4985-a5c2-2ad112ddac96"),
    simulation_config=Input("ri.foundry.main.dataset.7c220b40-47f1-4701-87e1-29d3e2938a85"),
).with_resources(cpu_cores=8, memory_gb=64)
def compute(
    ctx: LightweightContext,
    wip_lots: IncrementalLightweightInput,
    net_demand: IncrementalLightweightInput,
    model_priorities: IncrementalLightweightInput,
    equipment_capacity: IncrementalLightweightInput,
    equipment_constraints: IncrementalLightweightInput,
    equipment_to_process: IncrementalLightweightInput,
    model_dataset: IncrementalLightweightInput,
    model_unit_conversion: IncrementalLightweightInput,
    planned_process_steps: IncrementalLightweightInput,
    simulation_config: IncrementalLightweightInput,
    allocation_output: IncrementalLightweightOutput,
    new_lots_created_output: IncrementalLightweightOutput,
    unrouted_model_demand_output: IncrementalLightweightOutput,
    failed_allocations_output: IncrementalLightweightOutput,
    run_tracker_output: IncrementalLightweightOutput,
) -> None:
    """
    Main transform: Revenue-driven capacity allocation with virtual lot creation.

    INCREMENTAL LOGIC:
    - Tracks which simulations have been processed with what configuration
    - Only processes new/changed simulations or configuration changes
    - Aborts if no new work is needed

    CONFIGURATION:
    - Per-simulation configuration is loaded from simulation_config dataset
    - Each simulation can have its own constraints, defaults, and equipment capacity overrides
    - Falls back to DEFAULT_CONFIG if no simulation-specific config is found

    Outputs:
    - allocation_output: Successfully allocated lot steps with timing and delay details
    - new_lots_created_output: Virtual lots created to fill shortfalls
    - unrouted_model_demand_output: Models with demand but no routing defined
    - failed_allocations_output: Lot steps that could not be allocated (insufficient capacity)
    - run_tracker_output: Tracks which simulations have been processed with what config
    """
    run_timestamp = datetime.utcnow()

    # Load simulation configuration dataset
    print("\n=== Loading simulation configuration ===")
    simulation_config_df = simulation_config.polars()
    config_lookup = build_config_lookup(simulation_config_df)
    print(f"Loaded configurations for {len(config_lookup)} simulations")

    # Use DEFAULT_CONFIG for initial setup logging
    default_config = DEFAULT_CONFIG
    print(f"Default Start Date: {default_config.effective_start_date}")
    print(f"Default Config Hash: {create_config_hash(default_config)}")
    # =========================================================================
    # OPTIMIZATION: Check for new simulations BEFORE loading heavy datasets
    # =========================================================================

    # Step 1: Load only the minimal data needed to check for new simulations
    print("\n=== Checking for new simulations to process ===")

    # Check if any config has force_snapshot enabled
    force_snapshot_enabled = True
    for sim_id, sim_config in config_lookup.items():
        if sim_config.constraints.force_snapshot:
            force_snapshot_enabled = True
            print(f"⚠️ FORCE SNAPSHOT enabled for simulation {sim_id} - will reprocess all simulations")
            break

    # Check if DEFAULT_CONFIG has force_snapshot enabled
    if not force_snapshot_enabled and default_config.constraints.force_snapshot:
        force_snapshot_enabled = True
        print(f"⚠️ FORCE SNAPSHOT enabled in DEFAULT_CONFIG - will reprocess all simulations")

    try:
        previous_runs_df = run_tracker_output.polars("previous")
    except:
        previous_runs_df = pl.DataFrame(schema=RUN_TRACKER_SCHEMA)
        print("No previous run tracker data - first execution")

    # Load only priorities to check active simulations (lightweight)
    priorities_df = model_priorities.polars()

    # Define simulations to exclude
    EXCLUDED_SIMULATIONS = [
        "445a0272-d087-415e-bdf9-4c0463246086",
        "31386cc1-7421-4846-bea8-de1a0ea601dc",
        "b74dc7f5-b790-4d53-a6bc-46ab31267d69",
        "9840e8a1-d14c-4eaa-a5e7-73c5400e6d33",
    ]

    # Filter out deleted records and excluded simulations
    priorities_df = priorities_df.filter(
        (pl.col("__is_deleted") == False) & (~pl.col("simulation_id").is_in(EXCLUDED_SIMULATIONS))
    )

    # Get unique revenue plan IDs from priorities
    revenue_plan_ids = priorities_df.select("revenue_plan_id").unique().to_series().to_list()
    print(f"Active revenue plan IDs: {revenue_plan_ids}")

    # Identify active simulations
    active_simulations = [sid for sid in priorities_df["simulation_id"].unique().to_list() if sid]

    # Determine which simulations need processing
    # If force_snapshot is enabled, process ALL simulations regardless of history
    # Otherwise, check if (simulation_id, config_hash) pair exists in run_tracker with SUCCESS status
    simulations_to_process = []

    if force_snapshot_enabled:
        # Force snapshot: process ALL active simulations
        simulations_to_process = list(active_simulations)
        print(f"\n🔄 FORCE SNAPSHOT MODE: Will reprocess ALL {len(simulations_to_process)} active simulations")
        for sim_id in simulations_to_process[:5]:  # Show first 5
            sim_config = config_lookup.get(sim_id, DEFAULT_CONFIG)
            sim_config_hash = create_config_hash(sim_config)
            print(f"  - {sim_id}: FORCED reprocess (config_hash={sim_config_hash[:8]}...)")
        if len(simulations_to_process) > 5:
            print(f"  ... and {len(simulations_to_process) - 5} more simulations")
    else:
        # Normal incremental mode: only process new or changed simulations
        for sim_id in active_simulations:
            # Get the config for this simulation (or default)
            sim_config = config_lookup.get(sim_id, DEFAULT_CONFIG)
            sim_config_hash = create_config_hash(sim_config)

            # Check if this specific (simulation_id, config_hash) pair exists with SUCCESS status
            if previous_runs_df.height > 0:
                already_processed = (
                    previous_runs_df.filter(
                        (pl.col("simulation_id") == sim_id)
                        & (pl.col("config_hash") == sim_config_hash)
                        & (pl.col("status") == "SUCCESS")
                    ).height
                    > 0
                )
            else:
                already_processed = False

            if not already_processed:
                simulations_to_process.append(sim_id)
                print(f"  - {sim_id}: NEW simulation to process (config_hash={sim_config_hash[:8]}...)")
            else:
                print(f"  - {sim_id}: Already processed with current config (skipping)")

    # EARLY EXIT if no new work needed
    if not simulations_to_process:
        print("\n" + "=" * 60)
        print("NO NEW SIMULATIONS TO PROCESS - ABORTING")
        print("All active simulations have been processed with current configuration")
        print("=" * 60)
        ctx.abort_job()
        return

    print(f"\n{len(simulations_to_process)} new simulations to process - proceeding with data load")

    # =========================================================================
    # Step 2: Load heavy datasets ONLY if there's work to do
    # =========================================================================
    print("\n=== Loading required datasets ===")

    wip_df = wip_lots.polars()
    capacity_df = equipment_capacity.polars()
    constraints_df = equipment_constraints.polars()
    equipment_to_process_df = equipment_to_process.polars()
    model_df = model_dataset.polars()
    model_unit_conversion_df = model_unit_conversion.polars()
    planned_steps_df = planned_process_steps.polars()
    demand_df = net_demand.polars()

    # Filter demand to only include revenue plans from active simulations
    active_revenue_plan_ids = priorities_df.select("revenue_plan_id").unique().to_series().to_list()
    demand_df = demand_df.filter(pl.col("revenue_plan_id").is_in(active_revenue_plan_ids))

    print(f"Total lot steps in input: {wip_df.height:,}")
    print(f"Equipment constraints loaded: {constraints_df.height:,} rows")
    print(f"Equipment to process links: {equipment_to_process_df.height:,} rows")

    # =========================================================================
    # Step 3: Build lookups and process data
    # =========================================================================
    wip_with_equipment = wip_df.filter(pl.col("process_id").is_not_null())
    print(f"Steps with valid process_id: {wip_with_equipment.height:,}")

    equipment_capacity_lookup = build_equipment_capacity_lookup(capacity_df, DEFAULT_CONFIG)
    process_to_equipment_lookup = build_process_to_equipment_lookup(equipment_to_process_df)
    model_metadata_lookup = build_model_metadata_lookup(model_df, model_unit_conversion_df, DEFAULT_CONFIG)
    model_process_steps_lookup = build_model_process_steps_lookup(planned_steps_df, DEFAULT_CONFIG)
    negative_constraints_lookup = build_negative_constraints_lookup(constraints_df, model_metadata_lookup)

    print(f"Equipment with capacity: {len(equipment_capacity_lookup)}")
    print(f"Processes with equipment mapping: {len(process_to_equipment_lookup)}")
    print(f"Models with metadata: {len(model_metadata_lookup)}")
    print(f"Models with process steps: {len(model_process_steps_lookup)}")
    print(f"Negative constraints: {len(negative_constraints_lookup)} model-process combinations")

    # Build demand lookup - DO NOT filter here, we need all demand for processing
    demand_by_model_month_all: Dict[Tuple[str, str], int] = {}
    for row in demand_df.iter_rows(named=True):
        model = row.get("model_id")
        month = row.get("plan_month")
        qty = row.get("net_production_demand_ea") or 0
        if model and month and qty > 0:
            demand_by_model_month_all[(model, month)] = qty

    print(f"Total net production demand: {sum(demand_by_model_month_all.values()):,} units")

    # Identify blocked models - but don't remove them from demand yet
    demand_model_ids = set(model for model, _ in demand_by_model_month_all.keys())
    blocked_models = find_models_with_blocked_equipment_paths(
        demand_model_ids=demand_model_ids,
        model_process_steps_lookup=model_process_steps_lookup,
        process_to_equipment=process_to_equipment_lookup,
        negative_constraints=negative_constraints_lookup,
    )

    print(f"Models with all equipment blocked: {len(blocked_models)}")
    print(f"Total models with demand: {len(demand_model_ids)}")

    # Prepare blocked demand tracking but keep original demand intact
    blocked_demand: Dict[Tuple[str, str], Tuple[int, str, str]] = {}
    for (model, month), qty in demand_by_model_month_all.items():
        if model in blocked_models:
            process_id, reason = blocked_models[model]
            blocked_demand[(model, month)] = (qty, process_id, reason)

    # Create working copy with blocked models removed for allocation
    demand_by_model_month = {
        (model, month): qty for (model, month), qty in demand_by_model_month_all.items() if model not in blocked_models
    }

    print(f"Models available for allocation (non-blocked): {len(set(m for m, _ in demand_by_model_month.keys()))}")
    print(f"Demand for non-blocked models: {sum(demand_by_model_month.values()):,} units")

    # =========================================================================
    # Step 4: Process new simulations
    # =========================================================================
    all_allocations: List[pl.DataFrame] = []
    all_new_lots: List[pl.DataFrame] = []
    all_unrouted: List[pl.DataFrame] = []
    all_failed: List[pl.DataFrame] = []
    new_run_records: List[Dict[str, Any]] = []

    for simulation_id in simulations_to_process:
        print(f"\n{'=' * 60}\nProcessing Simulation: {simulation_id}\n{'=' * 60}")

        # Load simulation-specific config (or use default)
        sim_config = config_lookup.get(simulation_id, DEFAULT_CONFIG)
        sim_config_hash = create_config_hash(sim_config)

        print(f"  Config source: {sim_config.config_source}")
        print(f"  Config hash: {sim_config_hash[:16]}...")
        print(f"  Start date: {sim_config.effective_start_date}")
        print(f"  Start month: {sim_config.constraints.start_month}")
        print(f"  Max delay days: {sim_config.constraints.max_delay_days}")
        print(f"  Demand buffer: {sim_config.constraints.demand_fulfillment_buffer}")
        # ⚠️ MLWB ADDITION — make the A/B state of every run auditable from its log
        print(f"  Scan memo (REVPLAN_SCAN_MEMO): {'ON — saturated-day dedup + oversize precheck' if _scan_memo_enabled() else 'off (stock Palantir scan behaviour)'}")
        if sim_config.equipment_capacity_overrides:
            print(f"  Equipment capacity overrides: {len(sim_config.equipment_capacity_overrides)} groups")

        run_id = generate_allocation_run_id(run_timestamp, simulation_id)
        sim_priorities_df = priorities_df.filter(pl.col("simulation_id") == simulation_id)
        model_priority_lookup = build_model_priority_lookup(sim_priorities_df)

        sim_name_rows = sim_priorities_df.select("simulation_name").unique()
        simulation_name = sim_name_rows.item(0, 0) if sim_name_rows.height > 0 else None

        # Get revenue_plan_id for this simulation from priorities
        sim_revenue_plan_rows = sim_priorities_df.select("revenue_plan_id").unique()
        simulation_revenue_plan_id = sim_revenue_plan_rows.item(0, 0) if sim_revenue_plan_rows.height > 0 else None

        # Build equipment capacity lookup with simulation-specific overrides
        sim_equipment_capacity_lookup = build_equipment_capacity_lookup(capacity_df, sim_config)

        # Apply any equipment capacity overrides from the simulation config
        if sim_config.equipment_capacity_overrides:
            for equip_group, capacity in sim_config.equipment_capacity_overrides.items():
                # Find equipment IDs that belong to this group and override their capacity
                # Note: This is a simplified approach - in production you might want to
                # maintain a mapping of equipment_group -> equipment_ids
                for equip_id in sim_equipment_capacity_lookup:
                    if equip_group.lower() in equip_id.lower():
                        sim_equipment_capacity_lookup[equip_id] = capacity

        try:
            allocation_df, new_lots_df, unrouted_df, failed_df = allocate_month_by_month(
                wip_with_equipment=wip_with_equipment,
                demand_by_model_month=demand_by_model_month,
                model_priority_lookup=model_priority_lookup,
                model_metadata_lookup=model_metadata_lookup,
                model_process_steps_lookup=model_process_steps_lookup,
                equipment_capacity=sim_equipment_capacity_lookup,
                process_to_equipment=process_to_equipment_lookup,
                negative_constraints=negative_constraints_lookup,
                revenue_plan_id=simulation_revenue_plan_id,
                simulation_id=simulation_id,
                simulation_name=simulation_name,
                run_id=run_id,
                run_timestamp=run_timestamp,
                blocked_demand=blocked_demand,
                config=sim_config,
            )

            all_allocations.append(allocation_df)
            all_new_lots.append(new_lots_df)
            all_unrouted.append(unrouted_df)
            all_failed.append(failed_df)

            new_run_records.append(
                create_run_record(
                    simulation_id=simulation_id,
                    revenue_plan_id=simulation_revenue_plan_id,
                    simulation_name=simulation_name,
                    config=sim_config,
                    config_hash=sim_config_hash,
                    run_id=run_id,
                    run_timestamp=run_timestamp,
                    status="SUCCESS",
                    row_counts={
                        "allocation": allocation_df.height,
                        "new_lots": new_lots_df.height,
                        "unrouted": unrouted_df.height,
                        "failed": failed_df.height,
                    },
                )
            )

        except Exception as e:
            print(f"ERROR processing simulation {simulation_id}: {str(e)}")
            new_run_records.append(
                create_run_record(
                    simulation_id=simulation_id,
                    revenue_plan_id=simulation_revenue_plan_id,
                    simulation_name=simulation_name,
                    config=sim_config,
                    config_hash=sim_config_hash,
                    run_id=run_id,
                    run_timestamp=run_timestamp,
                    status="FAILED",
                    error_message=str(e),
                )
            )
            continue

    # =========================================================================
    # Step 5: Determine write mode and load previous results if needed
    # =========================================================================
    # We need to determine if we can use APPEND mode (just add new data) or
    # need SNAPSHOT mode (replace all data). We need SNAPSHOT if:
    # 1. Any simulations need to be removed (reprocessed or excluded)
    # 2. First run (no previous data)
    #
    # We can use APPEND if we're only adding NEW simulations

    # Get all previously processed simulations (across all config hashes)
    all_processed_simulations = set()
    if previous_runs_df.height > 0:
        all_processed_simulations = set(
            previous_runs_df.filter(pl.col("status") == "SUCCESS")["simulation_id"].unique().to_list()
        )

    # Check if any previously processed simulations need to be removed
    simulations_to_remove = all_processed_simulations.intersection(
        set(simulations_to_process) | set(EXCLUDED_SIMULATIONS)
    )

    # Determine write mode
    is_first_run = True  # previous_runs_df.height == 0
    needs_snapshot = True  # is_first_run

    if needs_snapshot:
        print(f"\nUsing SNAPSHOT mode (replace all data)")
        if is_first_run:
            print("  - Reason: First run, no previous data")
        else:
            print(f"  - Reason: {len(simulations_to_remove)} simulations need to be removed/reprocessed")
    else:
        print(f"\nUsing APPEND mode (adding new data only)")
        print(f"  - {len(simulations_to_process)} new simulations will be appended")

    simulations_to_preserve = []  # all_processed_simulations - set(simulations_to_process) - set(EXCLUDED_SIMULATIONS)

    # Only load previous data if we need SNAPSHOT mode and have data to preserve
    if needs_snapshot and simulations_to_preserve:
        print(f"\nPreserving previous results for {len(simulations_to_preserve)} simulations")
        print(f"  - Active but unchanged: {len(simulations_to_preserve.intersection(set(active_simulations)))}")
        print(f"  - Inactive but preserved: {len(simulations_to_preserve - set(active_simulations))}")

        try:
            prev_allocations_pl = allocation_output.polars("previous")
            prev_allocations_filtered = prev_allocations_pl.filter(
                pl.col("simulation_id").is_in(list(simulations_to_preserve))
            )
            if prev_allocations_filtered.height > 0:
                all_allocations.append(prev_allocations_filtered)
                print(f"  - Preserved {prev_allocations_filtered.height:,} allocation rows")

            prev_new_lots_pl = new_lots_created_output.polars("previous")
            prev_new_lots_filtered = prev_new_lots_pl.filter(
                pl.col("simulation_id").is_in(list(simulations_to_preserve))
            )
            if prev_new_lots_filtered.height > 0:
                all_new_lots.append(prev_new_lots_filtered)
                print(f"  - Preserved {prev_new_lots_filtered.height:,} new lots rows")

            prev_unrouted_pl = unrouted_model_demand_output.polars("previous")
            prev_unrouted_filtered = prev_unrouted_pl.filter(
                pl.col("simulation_id").is_in(list(simulations_to_preserve))
            )
            if prev_unrouted_filtered.height > 0:
                all_unrouted.append(prev_unrouted_filtered)
                print(f"  - Preserved {prev_unrouted_filtered.height:,} unrouted demand rows")

            prev_failed_pl = failed_allocations_output.polars("previous")
            prev_failed_filtered = prev_failed_pl.filter(pl.col("simulation_id").is_in(list(simulations_to_preserve)))
            if prev_failed_filtered.height > 0:
                all_failed.append(prev_failed_filtered)
                print(f"  - Preserved {prev_failed_filtered.height:,} failed allocation rows")

        except Exception as e:
            print(f"  Warning: Could not load some previous results: {e}")

    # =========================================================================
    # Step 6: Combine results and create final outputs
    # =========================================================================
    # Cast all DataFrames to ensure schema consistency before concatenating
    def ensure_schema(df: pl.DataFrame, schema: Dict[str, pl.DataType]) -> pl.DataFrame:
        """Ensure DataFrame matches target schema by casting columns."""
        if df.height == 0:
            return df

        for col_name, col_type in schema.items():
            if col_name in df.columns:
                if df[col_name].dtype != col_type:
                    df = df.with_columns(pl.col(col_name).cast(col_type))
        return df

    # Apply schema consistency
    all_allocations = [ensure_schema(df, ALLOCATION_OUTPUT_SCHEMA) for df in all_allocations]
    all_new_lots = [ensure_schema(df, NEW_LOTS_OUTPUT_SCHEMA) for df in all_new_lots]
    all_unrouted = [ensure_schema(df, UNROUTED_MODEL_DEMAND_SCHEMA) for df in all_unrouted]
    all_failed = [ensure_schema(df, FAILED_ALLOCATION_SCHEMA) for df in all_failed]

    combined_allocations = (
        pl.concat(all_allocations, how="diagonal_relaxed")
        if all_allocations
        else pl.DataFrame(schema=ALLOCATION_OUTPUT_SCHEMA)
    )
    combined_new_lots = (
        pl.concat(all_new_lots, how="diagonal_relaxed") if all_new_lots else pl.DataFrame(schema=NEW_LOTS_OUTPUT_SCHEMA)
    )
    combined_unrouted = (
        pl.concat(all_unrouted, how="diagonal_relaxed")
        if all_unrouted
        else pl.DataFrame(schema=UNROUTED_MODEL_DEMAND_SCHEMA)
    )
    combined_failed = (
        pl.concat(all_failed, how="diagonal_relaxed") if all_failed else pl.DataFrame(schema=FAILED_ALLOCATION_SCHEMA)
    )

    print(f"\n{'=' * 60}\nFINAL SUMMARY\n{'=' * 60}")
    print(f"Total allocation rows: {combined_allocations.height:,}")
    print(f"Total failed allocations: {combined_failed.height:,}")
    print(f"Total virtual lots created: {combined_new_lots.height:,}")
    print(f"Total unrouted model demands: {combined_unrouted.height:,}")
    print(f"New simulations processed: {len(simulations_to_process)}")
    print(f"Previously processed simulations preserved: {len(simulations_to_preserve)}")
    print(f"Simulations excluded: {len(set(EXCLUDED_SIMULATIONS))}")

    if combined_failed.height > 0:
        status_counts = combined_failed.group_by("failure_status").agg(pl.count().alias("count")).sort("failure_status")
        print("\nFailed Allocation Breakdown:")
        for row in status_counts.iter_rows(named=True):
            print(f"  {row['failure_status']}: {row['count']:,}")

    # =========================================================================
    # Step 7: Enrich allocation data with additional columns
    # =========================================================================
    if combined_allocations.height > 0:
        # Drop existing placeholder columns before enrichment to avoid duplicates
        # These columns were added as null placeholders in _create_allocation_dataframe
        columns_to_drop = [
            "model_customer_name",
            "model_end_customer",
            "model_sales_team",
            "grouping_model",
            "unit_process_name",
            "simulation_name",
            "simulation_revenue_id",
            "final_production_units",
            "total_revenue",
            "total_margin",
        ]
        existing_cols_to_drop = [c for c in columns_to_drop if c in combined_allocations.columns]
        if existing_cols_to_drop:
            combined_allocations = combined_allocations.drop(existing_cols_to_drop)

        model_enrichment = model_df.select(
            [
                "model_id",
                pl.col("customer_name").alias("model_customer_name"),
                pl.col("end_customer").alias("model_end_customer"),
                pl.col("sales_team").alias("model_sales_team"),
            ]
        ).unique(subset=["model_id"], maintain_order=True)

        process_enrichment = planned_steps_df.select(
            [
                "process_id",
                "model_id",
                "grouping_model",
                pl.col("process_name").alias("unit_process_name"),
            ]
        ).unique(subset=["process_id", "model_id"], maintain_order=True)

        simulation_enrichment = priorities_df.select(
            [
                "simulation_id",
                pl.col("simulation_name"),
                pl.col("revenue_plan_id").alias("simulation_revenue_id"),
            ]
        ).unique(subset=["simulation_id"], maintain_order=True)

        financial_lookup = priorities_df.select(
            [
                "simulation_id",
                "model_id",
                "month",
                pl.col("margin_amount_per_unit").alias("margin_per_unit"),
                pl.col("amount_per_unit").alias("revenue_per_unit"),
            ]
        ).unique(subset=["simulation_id", "model_id", "month"], maintain_order=True)

        completed_lots = combined_allocations.select(
            "lot_id", "simulation_id", "model_id", "target_month", "units_produced"
        ).filter(pl.col("units_produced") > 0)

        financial_info = (
            completed_lots.join(
                financial_lookup,
                left_on=["simulation_id", "model_id", "target_month"],
                right_on=["simulation_id", "model_id", "month"],
                how="left",
            )
            .with_columns(
                [
                    (pl.col("units_produced") * pl.col("revenue_per_unit")).alias("total_revenue"),
                    (pl.col("units_produced") * pl.col("margin_per_unit")).alias("total_margin"),
                ]
            )
            .rename({"units_produced": "final_production_units"})
            .drop("revenue_per_unit", "margin_per_unit", "model_id", "target_month")
        )

        combined_allocations = (
            combined_allocations.join(model_enrichment, on="model_id", how="left")
            .join(process_enrichment, on=["process_id", "model_id"], how="left")
            .join(simulation_enrichment, on="simulation_id", how="left")
            .join(financial_info, on=["simulation_id", "lot_id"], how="left")
        )

        print("✓ Enriched allocation output with additional columns and financials")

    # =========================================================================
    # Step 8: Write outputs and update tracker
    # =========================================================================
    # Set write mode based on whether we need snapshot or can append
    if needs_snapshot:
        # SNAPSHOT mode: write all data (new + preserved previous)
        print("\nWriting outputs in SNAPSHOT mode (replacing all data)...")
        allocation_output.set_mode("replace")
        new_lots_created_output.set_mode("replace")
        unrouted_model_demand_output.set_mode("replace")
        failed_allocations_output.set_mode("replace")
        run_tracker_output.set_mode("replace")

        allocation_output.write_table(combined_allocations)
        new_lots_created_output.write_table(combined_new_lots)
        unrouted_model_demand_output.write_table(combined_unrouted)
        failed_allocations_output.write_table(combined_failed)

        # For tracker, combine previous (filtered) with new records
        if new_run_records:
            new_runs_df = pl.DataFrame(new_run_records, schema=RUN_TRACKER_SCHEMA)
            previous_to_keep = previous_runs_df.filter(
                (~pl.col("simulation_id").is_in(list(simulations_to_process)))
                & (~pl.col("simulation_id").is_in(EXCLUDED_SIMULATIONS))
            )
            updated_tracker = pl.concat([previous_to_keep, new_runs_df])
        else:
            updated_tracker = previous_runs_df.filter(~pl.col("simulation_id").is_in(EXCLUDED_SIMULATIONS))
        run_tracker_output.write_table(updated_tracker)
    else:
        # MODIFY mode: append new data to existing dataset
        print("\nWriting outputs in MODIFY mode (appending new data only)...")
        allocation_output.set_mode("modify")
        new_lots_created_output.set_mode("modify")
        unrouted_model_demand_output.set_mode("modify")
        failed_allocations_output.set_mode("modify")
        run_tracker_output.set_mode("modify")

        # Write only the newly processed data (appended to existing)
        allocation_output.write_table(combined_allocations)
        new_lots_created_output.write_table(combined_new_lots)
        unrouted_model_demand_output.write_table(combined_unrouted)
        failed_allocations_output.write_table(combined_failed)

        # For tracker, just append new records
        if new_run_records:
            new_runs_df = pl.DataFrame(new_run_records, schema=RUN_TRACKER_SCHEMA)
            run_tracker_output.write_table(new_runs_df)

    print(f"\n✓ Outputs written successfully")
    print(f"✓ Run tracker updated with {len(new_run_records)} new records")
