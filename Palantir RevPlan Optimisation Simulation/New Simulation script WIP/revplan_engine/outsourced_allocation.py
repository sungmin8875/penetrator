"""
outsourced_allocation.py — unmapped-operation outsourced pass-through.

⚠️ MLWB ADDITION — this module is NOT part of the Palantir source. Like
urgent_allocation.py it exists so `allocation_engine.py` stays a near-verbatim
Palantir port: that file only gains one call to `try_allocate_outsourced_step()`
in its no-equipment branch plus two one-line seams (lookup install + summary).

CUSTOMER DECISION (meeting 2026-07-15, §4 배정 실패 처리):
  * Most assignment failures are steps whose Operation Code has NO equipment
    mapping (only ~149 of ~397 routing ops are constrained). Surfacing these as
    errors has no practical value — the agreed rule is: NO MAPPING = OUTSOURCED
    (외주). Outsourced ops are not machine-planned in RTS, so no mapping is the
    expected state for them.
  * Instead of failing, the step is PLANNED-COMPLETE: completion date = step
    start + Plan LT, where Plan LT is joined from the model's routing by
    Operation Code — the customer-validated formula RunLt × sheets + WaitLt
    (hours), the same one used for virtual-lot lead times. When the routing
    carries no Run/Wait for the step, a config default duration is assigned
    (the "계획완료일 임의 부여" case).
  * The lot then CONTINUES to its next step ("흐름이 끊기지 않고 끝까지 진행").

Semantics:
  * Vendor work consumes NO in-house equipment capacity and does NOT count
    against the lot's daily in-house step cap (max_steps_per_lot_per_day) —
    the vendor runs in parallel to line availability.
  * The completion date is written to set_lot_last_step_date, so the NEXT
    step's earliest start is after the vendor returns the lot — outsourced
    lead time is real elapsed time, not cosmetic.
  * equipment_id is stamped `OUTSOURCED` on the allocation record. The rule is
    a HEURISTIC: a genuinely in-house op that is merely missing its mapping
    passes through here too (in-house sites start F1/F2/… per the customer —
    an automated site check is possible later). The per-run op-code summary
    printed at the end of allocation is the audit trail for spotting such
    gaps; keep it visible in the run log.

Ordering inside the engine's no-equipment branch:
  logical pass-through (M000N LOT START etc., zero duration)  →  outsourced
  (this module, Plan-LT duration)  →  original FAILED_NO_EQUIPMENT (only when
  `treat_unmapped_as_outsourced` is off — the A/B / parity kill switch).
"""
from __future__ import annotations

from datetime import date, timedelta
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

from .config import AllocationConfig
from .models import AllocationAttemptResult, AllocationState

# equipment_id sentinel stamped on outsourced allocation records — filter
# `equipment_id == 'OUTSOURCED'` in the frontend to see every assumed-outsourced step.
OUTSOURCED_EQUIPMENT_ID = "OUTSOURCED"

# Hours in a planning day for the Run/Wait → days estimate (matches
# virtual_lot_creator._HOURS_PER_DAY: substrate fabs run 24/7).
_HOURS_PER_DAY = 24.0


def install_plan_lt_lookup(
    state: AllocationState,
    model_process_steps_lookup: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Attach a (model_id, process_id) -> (run_lt, wait_lt) lookup to the shared state.

    Lot-step dicts (WIP rows, virtual-lot steps) do not carry run_lt/wait_lt —
    those live on the routing steps — so the meeting's "Operation Code 기준으로
    모델 시퀀스와 조인" is resolved through this lookup. Riding on the
    AllocationState (a plain dataclass) avoids threading a new parameter
    through the ported call chain.
    """
    lut: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for model_id, steps in (model_process_steps_lookup or {}).items():
        for s in steps:
            pid = s.get("process_id")
            if pid:
                lut[(model_id, pid)] = (s.get("run_lt"), s.get("wait_lt"), s.get("plan_lt"))
    state.outsourced_plan_lt_lookup = lut
    state.outsourced_assumed_ops = {}


# ⚠️ MLWB ADDITION (2026-08-04, customer 2026-07-30: actuals over standards) —
# measured step durations from o_custom_WipHistory. Installed module-level by
# run_simulation (so the ported engine call chain needs no new parameter) and
# consulted by _plan_lt_days as the TOP duration preference, but ONLY under
# REVPLAN_MEASURED_LT=1 — default OFF keeps runs byte-identical while the
# ingestion is validated dark. Keys: (model_id, op) exact, (None, op) fallback.
import os as _os

_MEASURED_LT: Dict[Tuple[Optional[str], str], float] = {}


def _measured_lt_enabled() -> bool:
    return _os.environ.get("REVPLAN_MEASURED_LT", "0") == "1"


def install_measured_lt(df) -> None:
    """Load the measured_step_durations frame into the module lookup (each run)."""
    global _MEASURED_LT
    _MEASURED_LT = {}
    if df is None or getattr(df, "height", 0) == 0:
        print("   ▶ measured LT: no data — outsourced timing on PlanLt chain"
              + ("" if not _measured_lt_enabled() else " (REVPLAN_MEASURED_LT=1 has nothing to use)"))
        return
    for r in df.iter_rows(named=True):
        h = r.get("measured_step_hours")
        if h is not None and h > 0 and r.get("process_id"):
            _MEASURED_LT[(r.get("model_id"), r["process_id"])] = float(h)
    n_mo = sum(1 for k in _MEASURED_LT if k[0] is not None)
    print(f"   ▶ measured LT installed: {n_mo:,} (model,op) + {len(_MEASURED_LT) - n_mo:,} op-level medians — "
          + ("ACTIVE (REVPLAN_MEASURED_LT=1): preferred over PlanLt for outsourced steps"
             if _measured_lt_enabled() else "dark (set REVPLAN_MEASURED_LT=1 to use)"))


def _plan_lt_days(
    lot_step: dict,
    model_id: str,
    state: AllocationState,
    config: AllocationConfig,
) -> int:
    """Planned outsourced duration in whole days for one step (minimum 1).

    Duration source, best first: MEASURED median (WipHistory 실측, only under
    REVPLAN_MEASURED_LT=1 — (model,op) then op-level), then (2026-07-16) the
    routing's own per-step PlanLt column (99.1% filled — the planner's number),
    then the customer-validated computation RunLt × sheets + WaitLt (hours; Run
    per-sheet, Wait per-lot), then `outsourced_step_days_default`.
    """
    total_hours = None
    if _MEASURED_LT and _measured_lt_enabled():
        _pid = lot_step.get("process_id")
        _m = _MEASURED_LT.get((model_id, _pid))
        if _m is None:
            _m = _MEASURED_LT.get((None, _pid))
        if _m is not None and _m > 0:
            total_hours = float(_m)

    if total_hours is None:
        plan_lt = lot_step.get("plan_lt")
        run_lt = lot_step.get("run_lt")
        wait_lt = lot_step.get("wait_lt")
        if plan_lt is None and run_lt is None and wait_lt is None:
            lut = getattr(state, "outsourced_plan_lt_lookup", None) or {}
            entry = lut.get((model_id, lot_step.get("process_id")), (None, None, None))
            run_lt, wait_lt = entry[0], entry[1]
            plan_lt = entry[2] if len(entry) > 2 else None
        if plan_lt is not None and float(plan_lt) > 0:
            total_hours = float(plan_lt)
        else:
            sheets = lot_step.get(config.columns.sheet_quantity) or 0
            total_hours = float(run_lt or 0.0) * float(sheets) + float(wait_lt or 0.0)
    if total_hours <= 0:
        return max(1, int(config.constraints.outsourced_step_days_default))
    days = max(1, ceil(total_hours / _HOURS_PER_DAY))

    # Outlier cap: routing PlanLt carries garbage rows (observed: step durations
    # placing lots in 2082). One bad row must not march a lot past the horizon —
    # cap the step and record it for the run audit (data fix belongs at the source).
    max_days = int(config.constraints.outsourced_step_max_days)
    if max_days > 0 and days > max_days:
        capped = getattr(state, "outsourced_capped_ops", None)
        if capped is None:
            capped = state.outsourced_capped_ops = {}
        pid = lot_step.get("process_id")
        cnt, worst = capped.get(pid, (0, 0))
        capped[pid] = (cnt + 1, max(worst, days))
        return max_days
    return days


def try_allocate_outsourced_step(
    lot_step: dict,
    model_id: str,
    state: AllocationState,
    start_date: date,
    is_new_lot: bool,
    config: AllocationConfig,
) -> Optional[AllocationAttemptResult]:
    """Allocate an unmapped step as outsourced, or return None to fall through.

    None means the toggle is off — the caller keeps the original
    FAILED_NO_EQUIPMENT behaviour. Otherwise the step is marked allocated with
    completion = step start + Plan LT days, no capacity usage, no daily-step-cap
    consumption, and the lot's last-step date advances to the completion so the
    next step waits for the vendor.
    """
    if not config.constraints.treat_unmapped_as_outsourced:
        return None

    lot_id = lot_step.get("lot_id")
    process_id = lot_step.get("process_id")
    sequence = lot_step.get("sequence")

    # Step start mirrors the logical-passthrough date logic in the engine.
    prev_step_date = state.get_lot_last_step_date(lot_id)
    if prev_step_date:
        step_start = prev_step_date
    elif is_new_lot:
        step_start = max(lot_step.get("target_lot_start_date", start_date), start_date)
    else:
        step_start = start_date

    completion = step_start + timedelta(days=_plan_lt_days(lot_step, model_id, state, config))

    state.mark_step_allocated(lot_id, process_id, sequence)
    state.set_lot_last_step_date(lot_id, completion)
    state.add_equipment_to_path(lot_id, OUTSOURCED_EQUIPMENT_ID)

    ops = getattr(state, "outsourced_assumed_ops", None)
    if ops is None:
        ops = state.outsourced_assumed_ops = {}
    ops[process_id] = ops.get(process_id, 0) + 1

    return AllocationAttemptResult(
        success=True,
        allocated_date=completion,
        equipment_id=OUTSOURCED_EQUIPMENT_ID,
        equipment_group=lot_step.get("equipment_group_id"),
        days_delayed=0,
        delay_records=[],
    )


def print_outsourced_summary(state: AllocationState) -> None:
    """Audit trail: which op codes were ASSUMED outsourced this run, how often.

    Review this list periodically — an op code that should be in-house (site
    F1/F2/…) appearing here means its equipment mapping is missing, not that
    it is really outsourced.
    """
    ops = getattr(state, "outsourced_assumed_ops", None)
    if not ops:
        return
    total = sum(ops.values())
    top = sorted(ops.items(), key=lambda kv: -kv[1])
    listed = ", ".join(f"{op}×{n}" for op, n in top[:20])
    more = f" (+{len(top) - 20} more ops)" if len(top) > 20 else ""
    print(
        f"\n  ⚠ OUTSOURCED-ASSUMED: {total} steps across {len(top)} unmapped operation codes "
        f"planned-complete via Plan LT (equipment_id='{OUTSOURCED_EQUIPMENT_ID}'):\n"
        f"    {listed}{more}\n"
        "    → audit: an op that should be IN-HOUSE in this list = missing equipment mapping."
    )
    capped = getattr(state, "outsourced_capped_ops", None)
    if capped:
        worst = sorted(capped.items(), key=lambda kv: -kv[1][1])
        listed = ", ".join(f"{op}×{cnt} (worst {days}d)" for op, (cnt, days) in worst[:15])
        more = f" (+{len(worst) - 15} more ops)" if len(worst) > 15 else ""
        print(
            f"  ⚠ PLAN-LT OUTLIERS CAPPED: {sum(c for c, _ in capped.values())} outsourced steps "
            f"exceeded outsourced_step_max_days and were capped:\n"
            f"    {listed}{more}\n"
            "    → audit: these ops' routing PlanLt values are garbage — fix at the source (RTS)."
        )
