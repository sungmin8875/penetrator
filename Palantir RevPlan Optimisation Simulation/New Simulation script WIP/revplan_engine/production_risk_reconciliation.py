"""
[MLWB PORT] This module is a VERBATIM copy of the Palantir Foundry source
file "production_risk_reconciliation.py" from ../Python Script Allocation Engine/. The ONLY change is
the import rewrite (transforms.api -> ._foundry_shim shim). The pure logic is byte-identical to the
source so it can be re-synced if the Palantir engine changes. Foundry decorators are no-ops here. The
engine drives compute() in-memory via the wrapper in run_simulation.py
(compute_production_risk_reconciliation) — all four inputs are the engine's own outputs from the same
run (allocation, failed_allocations, fulfillment_wide, et_jig_risk), so no celonis_io wiring is needed.
NOTE the arg order: inputs FIRST, the single output LAST (like et_jig_risk). Ported 2026-07-16 — the
last stub; every Palantir analysis now runs in MLWB.
"""
"""
Production Risk Reconciliation Dataset

Captures ALL root causes of production shortfall and delays, reconciling with
`simulation_monthly_fulfillment_wide` as the source of truth.

The shortfall formula from fulfillment is:
  shortfall = production_and_inventory + impossible_to_simulate - planned_demand

When shortfall < 0, it means unmet demand (we produced less than planned).
This dataset breaks down that unmet demand into categories.

Risk Categories:
1. FAILED_ALLOCATION - Lots that could NOT be scheduled at all because required equipment
   was completely blocked for the model. These should match `impossible_to_simulate_ea`.
   NOTE: Failed allocations due to EQUIPMENT BLOCKED are ALWAYS surfaced (even when
   shortfall >= 0) because they represent production that was impossible. These are
   tracked separately from shortfall contribution - they contribute to impossible_to_simulate.
   EXCLUDES: "Scheduling horizon exceeded" failures (these are simulation boundary issues,
   not actual equipment constraints).

2. FAILED_NO_EQUIPMENT - Lots that could NOT be scheduled because no valid equipment was
   available for the model/process combination. These are tracked as "impossible to simulate"
   and do NOT contribute to the shortfall column directly. Identified by failure_status =
   'FAILED_NO_EQUIPMENT' in the failed_allocations dataset.
   EXCLUDES: FAILED_HORIZON_EXCEEDED (scheduling horizon issues).
3. ET_JIG_CAPACITY_ISSUE - Lots delayed due to insufficient ET JIG testing capacity.
   Identified from the et_jig_capacity_risk_analysis dataset. The bottleneck is the
   ET JIG testing process (burn-in testing), NOT the E/T equipment group.
   NOTE: units_at_risk represents ALL lots delayed by ET JIG capacity, but many still
   complete within the target month. Only the portion contributing to actual shortfall
   is counted here.

4. EQUIPMENT_CAPACITY_DELAY - Lots that WERE scheduled but completed AFTER target month
   due to specific equipment being at capacity.

5. LEAD_TIME_SHORTFALL - Remaining shortfall not explained by the above categories.
   This is the gap between planned_demand and (production_and_inventory + impossible_to_simulate).

Key Business Rules:
- EXCLUDE January 2026 data (target_month = '202601') - simulation boundary
- Shortfall numbers MUST reconcile with simulation_monthly_fulfillment_wide
- For each simulation+month: SUM(shortfall_contribution) should equal ABS(shortfall_ea) from fulfillment
- Failed allocations due to EQUIPMENT BLOCKED are ALWAYS shown (units_affected column)
  even when they don't contribute to shortfall (shortfall_contribution may be 0)
- FAILED_NO_EQUIPMENT records are always surfaced but contribute to impossible_to_simulate,
  NOT to shortfall_contribution

Output: One row per risk event (lot-level for allocations/delays, model-month for shortfall reconciliation)
"""

import polars as pl
from ._foundry_shim import transform, Input, Output, lightweight
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import re


# =============================================================================
# Risk Category Descriptions
# =============================================================================

RISK_CATEGORY_DESCRIPTIONS = {
    "FAILED_ALLOCATION": (
        "Lots that could not be scheduled at all because the required equipment was completely "
        "blocked for this model. This typically occurs when all equipment in a process group is "
        "either at capacity or explicitly blocked for certain models. These units are counted as "
        "'impossible to simulate' in the fulfillment calculation."
    ),
    "FAILED_NO_EQUIPMENT": (
        "Lots that could not be scheduled because no valid equipment was available for the "
        "model/process combination. This occurs when the equipment configuration does not support "
        "the required process for this model. These units are counted as 'impossible to simulate' "
        "in the fulfillment calculation and do NOT contribute to the shortfall column directly. "
        "This is distinct from equipment being at capacity - here, the equipment simply cannot "
        "process this model type."
    ),
    "ET_JIG_CAPACITY_ISSUE": (
        "Lots delayed due to insufficient ET JIG (Electrical Test Jig) testing capacity. "
        "ET JIG testing is a burn-in process that validates product quality before shipment. "
        "Each JIG can only test a limited number of lots per day, and when demand exceeds capacity, "
        "lots must wait for available JIG slots, causing delivery delays beyond the target month."
    ),
    "EQUIPMENT_CAPACITY_DELAY": (
        "Lots that were successfully scheduled but completed after the target month due to "
        "equipment capacity constraints. The lot was allocated but had to wait for equipment "
        "availability, pushing completion beyond the planned delivery date."
    ),
    "LEAD_TIME_SHORTFALL": (
        "Remaining shortfall not explained by failed allocations, ET JIG capacity issues, or "
        "equipment delays. This residual gap typically reflects inherent production lead time "
        "constraints, insufficient raw material availability, or other systemic capacity limitations "
        "that prevent meeting demand even when equipment is available."
    ),
}


PRODUCTION_RISK_RECONCILIATION_SCHEMA = {
    "risk_record_id": pl.Utf8,
    "simulation_id": pl.Utf8,
    "simulation_name": pl.Utf8,
    "model_id": pl.Utf8,
    "target_month": pl.Utf8,
    "risk_category": pl.Utf8,
    "risk_category_description": pl.Utf8,
    "lot_id": pl.Utf8,
    "affected_lot_ids": pl.List(pl.Utf8),
    "source_dataset": pl.Utf8,
    "source_record_id": pl.Utf8,
    "source_fulfillment_id": pl.Utf8,
    "units_affected": pl.Int64,
    "total_units_at_risk": pl.Int64,
    "lots_affected": pl.Int64,
    "actual_completion_month": pl.Utf8,
    "months_delayed": pl.Int64,
    "days_delayed": pl.Int64,
    "max_days_delayed": pl.Int64,
    "bottleneck_process": pl.Utf8,
    "bottleneck_equipment_group": pl.Utf8,
    "bottleneck_equipment_ids": pl.List(pl.Utf8),
    "delay_dates": pl.List(pl.Utf8),
    "failure_reason": pl.Utf8,
    "failure_reason_summary": pl.Utf8,
    "is_model_blocked": pl.Boolean,
    "current_capacity": pl.Int64,
    "required_capacity": pl.Int64,
    "capacity_gap": pl.Int64,
    "capacity_utilization_pct": pl.Float64,
    "planned_demand_ea": pl.Int64,
    "planned_demand_krw": pl.Float64,
    "production_and_inventory_ea": pl.Int64,
    "production_and_inventory_krw": pl.Float64,
    "impossible_to_simulate_ea": pl.Int64,
    "impossible_to_simulate_krw": pl.Float64,
    "fulfillment_shortfall_ea": pl.Int64,
    "fulfillment_shortfall_krw": pl.Float64,
    "shortfall_contribution_ea": pl.Int64,
    "shortfall_contribution_krw": pl.Float64,
    "shortfall_contribution_pct": pl.Float64,
    "revenue_at_risk": pl.Float64,
    "margin_at_risk": pl.Float64,
    "unit_price_krw": pl.Float64,
    "grouping_model": pl.Utf8,
    "model_customer_name": pl.Utf8,
    "model_end_customer": pl.Utf8,
    "model_sales_team": pl.Utf8,
    "model_priority": pl.Int64,
    "revenue_plan_id": pl.Utf8,
    "mitigation_strategy": pl.Utf8,
    "action_required": pl.Utf8,
    "allocation_run_id": pl.Utf8,
    "allocation_run_ts": pl.Datetime,
}


def is_equipment_blocked_failure(failure_reason: str) -> bool:
    """
    Check if failure reason indicates equipment was blocked (not scheduling horizon exceeded).

    Returns True for failures like:
      - "All 17 equipment blocked for process M218N"
      - "All 5 equipment blocked for process MA21N"

    Returns False for failures like:
      - "BLOCKED: Scheduling horizon exceeded (reached 2027)"
      - "2026-12-31: max steps/lot/day limit; BLOCKED: Scheduling horizon exceeded (reached 2027)"
    """
    if not failure_reason:
        return False

    if "Scheduling horizon exceeded" in failure_reason:
        return False

    if "equipment blocked for process" in failure_reason:
        return True

    return False


def is_no_equipment_failure(failure_status: str) -> bool:
    """
    Check if failure status indicates no valid equipment was available.

    Returns True for failure_status = 'FAILED_NO_EQUIPMENT'
    Returns False for failure_status = 'FAILED_HORIZON_EXCEEDED' or other statuses
    """
    if not failure_status:
        return False

    return failure_status == "FAILED_NO_EQUIPMENT"


def parse_equipment_group_from_failure(failure_reason: str) -> Tuple[Optional[str], int]:
    """Extract equipment group and count from failure reason."""
    if not failure_reason:
        return None, 0

    match = re.search(r"All (\d+) equipment blocked for process ([A-Z0-9]+)", failure_reason)
    if match:
        count = int(match.group(1))
        process = match.group(2)
        return process, count

    return None, 0


def summarize_failure_reason(failure_reason: str) -> str:
    """Create a concise summary of failure reason."""
    if not failure_reason:
        return ""

    if "Scheduling horizon exceeded" in failure_reason:
        return "Scheduling horizon exceeded"

    if "equipment blocked for process" in failure_reason:
        match = re.search(r"All (\d+) equipment blocked for process ([A-Z0-9]+)", failure_reason)
        if match:
            return f"{match.group(1)} equipment blocked for {match.group(2)}"

    if len(failure_reason) > 100:
        return failure_reason[:97] + "..."

    return failure_reason


def has_equipment_capacity_issue(delay_reasons: str) -> bool:
    """Check if delay reasons indicate equipment capacity issue."""
    if not delay_reasons:
        return False

    lower = delay_reasons.lower()
    return "full" in lower or "BLOCKED" in delay_reasons or "insufficient" in lower


def extract_equipment_ids_from_delay_reasons(delay_reasons: str) -> Tuple[List[str], List[str], bool]:
    """Extract equipment IDs from delay reasons."""
    if not delay_reasons:
        return [], [], False

    equipment_ids = []
    blocked_ids = []
    is_blocked = "BLOCKED" in delay_reasons

    for match in re.finditer(r"([A-Z0-9_-]+(?:_\d+)?)\s*(?:FULL|full|BLOCKED)", delay_reasons):
        equip_id = match.group(1)
        if "BLOCKED" in delay_reasons[match.start() : match.end() + 20]:
            blocked_ids.append(equip_id)
        equipment_ids.append(equip_id)

    return list(set(equipment_ids)), list(set(blocked_ids)), is_blocked


def extract_delay_dates(delay_reasons: str) -> List[str]:
    """Extract delay dates from delay reasons."""
    if not delay_reasons:
        return []

    dates = []
    for match in re.finditer(r"\d{4}-\d{2}-\d{2}", delay_reasons):
        dates.append(match.group(0))

    return list(set(dates))


def calculate_months_delayed(target_month: str, actual_month: str) -> int:
    """Calculate number of months delayed."""
    if not target_month or not actual_month:
        return 0

    try:
        target = datetime.strptime(target_month, "%Y%m")
        actual = datetime.strptime(actual_month, "%Y%m")
        delta = (actual.year - target.year) * 12 + (actual.month - target.month)
        return max(0, delta)
    except:
        return 0


def get_action_for_risk(risk_category: str, context: dict) -> str:
    """Generate action required based on risk category."""
    if risk_category == "FAILED_ALLOCATION":
        equipment_group = context.get("equipment_group", "")
        return (
            f"Investigate equipment blocking for {equipment_group}"
            if equipment_group
            else "Investigate failed allocations"
        )

    elif risk_category == "FAILED_NO_EQUIPMENT":
        equipment_group = context.get("equipment_group", "")
        process_id = context.get("process_id", "")
        if equipment_group and process_id:
            return f"Add equipment support for {equipment_group} to process {process_id}"
        return "Configure equipment for this model/process combination"

    elif risk_category == "ET_JIG_CAPACITY_ISSUE":
        additional_jigs = context.get("additional_jigs", 0)
        return (
            f"Add {additional_jigs} ET JIGs to testing capacity" if additional_jigs > 0 else "Increase ET JIG capacity"
        )

    elif risk_category == "EQUIPMENT_CAPACITY_DELAY":
        equipment_group = context.get("equipment_group", "")
        months_delayed = context.get("months_delayed", 0)
        if equipment_group:
            return f"Increase {equipment_group} capacity (delays: {months_delayed} months)"
        return "Increase equipment capacity"

    elif risk_category == "LEAD_TIME_SHORTFALL":
        return "Review production lead times and capacity planning"

    return "Review and address capacity constraint"


@lightweight(cpu_cores=2, memory_gb=16)
@transform(
    risk_reconciliation_output=Output("ri.foundry.main.dataset.46972f1a-e7cd-4b12-a30a-ac05ba0fc14e"),
    allocation_output=Input("ri.foundry.main.dataset.b84b2bb5-1cbf-4970-8f73-0e1eb9d4148f"),
    failed_allocations=Input("ri.foundry.main.dataset.96248695-b74a-4d42-b5e9-30a48bc8ff5d"),
    fulfillment_wide=Input("ri.foundry.main.dataset.f9dd33b1-6764-4bd9-8be3-883cc279a614"),
    et_jig_risk=Input("ri.foundry.main.dataset.46d6125b-62ba-48ba-b9d2-575abadd6eb7"),
)
def compute(
    allocation_output,
    failed_allocations,
    fulfillment_wide,
    et_jig_risk,
    risk_reconciliation_output,
) -> None:
    """
    Build comprehensive production risk reconciliation dataset.

    The shortfall from fulfillment is:
      shortfall = production_and_inventory + impossible_to_simulate - planned_demand

    When shortfall < 0, we have unmet demand. This dataset attributes that shortfall to:
    1. FAILED_ALLOCATION - impossible_to_simulate (lots that couldn't be scheduled due to blocked equipment)
    2. FAILED_NO_EQUIPMENT - impossible_to_simulate (lots that couldn't be scheduled due to no valid equipment)
       NOTE: These do NOT contribute to shortfall_contribution - they are tracked in impossible_to_simulate
    3. ET_JIG_CAPACITY_ISSUE - lots delayed due to ET JIG testing capacity constraints
    4. EQUIPMENT_CAPACITY_DELAY - lots delayed due to equipment capacity
    5. LEAD_TIME_SHORTFALL - remaining gap = planned_demand - production_and_inventory - impossible_to_simulate

    Additionally, FAILED_ALLOCATION and FAILED_NO_EQUIPMENT records are created for ALL
    equipment-blocked/no-equipment failures (excluding scheduling horizon exceeded),
    even when there's no negative shortfall. These appear with units_affected > 0 but
    may have shortfall_contribution_ea = 0.
    """
    print("=" * 70)
    print("PRODUCTION RISK RECONCILIATION")
    print("=" * 70)

    print("\n--- Loading Input Data ---")
    allocation_df = allocation_output.polars()
    failed_df = failed_allocations.polars()
    fulfillment_df = fulfillment_wide.polars()
    et_jig_df = et_jig_risk.polars()

    print(f"  Allocation records: {allocation_df.height:,}")
    print(f"  Failed allocations: {failed_df.height:,}")
    print(f"  Fulfillment records: {fulfillment_df.height:,}")
    print(f"  ET JIG risk records: {et_jig_df.height:,}")

    fulfillment_with_shortfall = fulfillment_df.filter((pl.col("month_str") != "202601") & (pl.col("shortfall_ea") < 0))

    print(f"  Fulfillment records with shortfall (excl 202601): {fulfillment_with_shortfall.height:,}")

    fulfillment_lookup: Dict[Tuple[str, str, str], dict] = {}
    for row in fulfillment_df.filter(pl.col("month_str") != "202601").iter_rows(named=True):
        key = (row.get("simulation_id"), row.get("model_id"), row.get("month_str"))
        planned_demand_ea = row.get("planned_demand_ea", 0) or 0
        planned_demand_krw = row.get("planned_demand_krw", 0) or 0
        unit_price = planned_demand_krw / planned_demand_ea if planned_demand_ea > 0 else 0

        fulfillment_lookup[key] = {
            "fulfillment_stat_id": row.get("fulfillment_stat_id"),
            "planned_demand_ea": planned_demand_ea,
            "planned_demand_krw": planned_demand_krw,
            "production_and_inventory_ea": row.get("production_and_inventory_ea", 0) or 0,
            "production_and_inventory_krw": row.get("production_and_inventory_krw", 0) or 0,
            "impossible_to_simulate_ea": row.get("impossible_to_simulate_ea", 0) or 0,
            "impossible_to_simulate_krw": row.get("impossible_to_simulate_krw", 0) or 0,
            "shortfall_ea": row.get("shortfall_ea", 0) or 0,
            "shortfall_krw": row.get("shortfall_krw", 0) or 0,
            "simulation_name": row.get("simulation_name"),
            "grouping_model": row.get("grouping_model"),
            "model_customer_name": row.get("model_customer_name"),
            "model_end_customer": row.get("model_end_customer"),
            "model_sales_team": row.get("model_sales_team"),
            "revenue_plan_id": row.get("revenue_plan_id"),
            "unit_price_krw": unit_price,
        }

    print(f"  Fulfillment lookup entries: {len(fulfillment_lookup):,}")

    et_jig_by_key: Dict[Tuple[str, str, str], dict] = {}
    for row in et_jig_df.filter(pl.col("target_month") != "202601").iter_rows(named=True):
        key = (row.get("simulation_id"), row.get("model_id"), row.get("target_month"))
        et_jig_by_key[key] = {
            "et_jig_risk_id": row.get("et_jig_risk_id"),
            "title": row.get("title"),
            "units_at_risk": row.get("units_at_risk", 0) or 0,
            "lots_at_risk": row.get("lots_at_risk", 0) or 0,
            "average_days_late": row.get("average_days_late", 0) or 0,
            "max_days_late": row.get("max_days_late", 0) or 0,
            "current_et_jig_count": row.get("current_et_jig_count", 0) or 0,
            "current_daily_lot_capacity": row.get("current_daily_lot_capacity", 0) or 0,
            "current_monthly_lot_capacity": row.get("current_monthly_lot_capacity", 0) or 0,
            "additional_et_jigs_needed": row.get("additional_et_jigs_needed", 0) or 0,
            "total_et_jigs_needed": row.get("total_et_jigs_needed", 0) or 0,
            "capacity_increase_pct": row.get("capacity_increase_pct", 0) or 0,
            "risk_level": row.get("risk_level"),
            "risk_reason": row.get("risk_reason"),
            "mitigation_strategy": row.get("mitigation_strategy"),
            "revenue_at_risk": row.get("revenue_at_risk", 0) or 0,
            "margin_at_risk": row.get("margin_at_risk", 0) or 0,
            "at_risk_lot_ids": row.get("at_risk_lot_ids") or [],
            "allocation_run_id": row.get("allocation_run_id"),
            "allocation_run_ts": row.get("allocation_run_ts"),
            "model_priority": row.get("model_priority"),
        }

    print(f"  ET JIG risk lookup entries: {len(et_jig_by_key):,}")

    results = []

    print("\n--- Processing FAILED_NO_EQUIPMENT Failed Allocations ---")

    processed_failed_lot_ids: Set[Tuple[str, str, str, str]] = set()
    no_equipment_count = 0

    for row in failed_df.filter(pl.col("target_month") != "202601").iter_rows(named=True):
        failure_status = row.get("failure_status", "")
        failure_reason = row.get("failure_reason", "")

        if not is_no_equipment_failure(failure_status):
            continue

        sim_id = row.get("simulation_id")
        model_id = row.get("model_id")
        target_month = row.get("target_month")
        lot_id = row.get("lot_id")

        if not sim_id or not model_id or not target_month or not lot_id:
            continue

        lot_key = (sim_id, model_id, target_month, lot_id)
        if lot_key in processed_failed_lot_ids:
            continue
        processed_failed_lot_ids.add(lot_key)
        no_equipment_count += 1

        fulfillment_key = (sim_id, model_id, target_month)
        fulfillment_data = fulfillment_lookup.get(fulfillment_key, {})

        units = row.get("units_in_lot", 0) or 0

        shortfall_ea = fulfillment_data.get("shortfall_ea", 0) or 0

        planned_demand_ea = fulfillment_data.get("planned_demand_ea", 0)
        planned_demand_krw = fulfillment_data.get("planned_demand_krw", 0)
        unit_price_krw = planned_demand_krw / planned_demand_ea if planned_demand_ea > 0 else 0

        equipment_group, equip_count = parse_equipment_group_from_failure(failure_reason)
        if not equipment_group:
            equipment_group = row.get("equipment_group")

        process_id = row.get("process_id")

        results.append(
            {
                "risk_record_id": f"FNE_{sim_id}_{model_id}_{target_month}_{lot_id}",
                "simulation_id": sim_id,
                "simulation_name": fulfillment_data.get("simulation_name") or row.get("simulation_name"),
                "model_id": model_id,
                "target_month": target_month,
                "risk_category": "FAILED_NO_EQUIPMENT",
                "risk_category_description": RISK_CATEGORY_DESCRIPTIONS["FAILED_NO_EQUIPMENT"],
                "lot_id": lot_id,
                "affected_lot_ids": [lot_id],
                "source_dataset": "failed_allocations",
                "source_record_id": row.get("failed_allocation_id"),
                "source_fulfillment_id": fulfillment_data.get("fulfillment_stat_id"),
                "units_affected": units,
                "total_units_at_risk": units,
                "lots_affected": 1,
                "actual_completion_month": None,
                "months_delayed": 0,
                "days_delayed": None,
                "max_days_delayed": None,
                "bottleneck_process": process_id,
                "bottleneck_equipment_group": equipment_group,
                "bottleneck_equipment_ids": [equipment_group] if equipment_group else None,
                "delay_dates": None,
                "failure_reason": failure_reason,
                "failure_reason_summary": summarize_failure_reason(failure_reason),
                "is_model_blocked": True,
                "current_capacity": equip_count if equip_count > 0 else None,
                "required_capacity": None,
                "capacity_gap": None,
                "capacity_utilization_pct": 100.0,
                "planned_demand_ea": fulfillment_data.get("planned_demand_ea"),
                "planned_demand_krw": fulfillment_data.get("planned_demand_krw"),
                "production_and_inventory_ea": fulfillment_data.get("production_and_inventory_ea"),
                "production_and_inventory_krw": fulfillment_data.get("production_and_inventory_krw"),
                "impossible_to_simulate_ea": fulfillment_data.get("impossible_to_simulate_ea"),
                "impossible_to_simulate_krw": fulfillment_data.get("impossible_to_simulate_krw"),
                "fulfillment_shortfall_ea": shortfall_ea,
                "fulfillment_shortfall_krw": fulfillment_data.get("shortfall_krw"),
                "shortfall_contribution_ea": 0,
                "shortfall_contribution_krw": 0,
                "shortfall_contribution_pct": 0,
                "revenue_at_risk": units * unit_price_krw,
                "margin_at_risk": None,
                "unit_price_krw": unit_price_krw,
                "grouping_model": fulfillment_data.get("grouping_model") or row.get("grouping_model"),
                "model_customer_name": fulfillment_data.get("model_customer_name") or row.get("model_customer_name"),
                "model_end_customer": fulfillment_data.get("model_end_customer") or row.get("model_end_customer"),
                "model_sales_team": fulfillment_data.get("model_sales_team") or row.get("model_sales_team"),
                "model_priority": row.get("model_priority"),
                "revenue_plan_id": fulfillment_data.get("revenue_plan_id") or row.get("revenue_plan_id"),
                "mitigation_strategy": None,
                "action_required": get_action_for_risk(
                    "FAILED_NO_EQUIPMENT", {"equipment_group": equipment_group, "process_id": process_id}
                ),
                "allocation_run_id": row.get("allocation_run_id"),
                "allocation_run_ts": row.get("allocation_run_ts"),
            }
        )

    print(f"  FAILED_NO_EQUIPMENT allocations: {no_equipment_count:,}")

    print("\n--- Processing Shortfall Records ---")

    failed_by_key: Dict[Tuple[str, str, str], List[dict]] = {}
    for row in failed_df.filter(pl.col("target_month") != "202601").iter_rows(named=True):
        key = (row.get("simulation_id"), row.get("model_id"), row.get("target_month"))
        if key not in failed_by_key:
            failed_by_key[key] = []
        failed_by_key[key].append(row)

    delayed_lots = (
        allocation_df.filter(pl.col("is_delayed_completion") == True)
        .filter(pl.col("target_month") != "202601")
        .filter(pl.col("delay_reasons").is_not_null())
        .filter(pl.col("delay_reasons") != "")
        .filter(
            pl.col("delay_reasons").str.contains("(?i)full")
            | pl.col("delay_reasons").str.contains("BLOCKED")
            | pl.col("delay_reasons").str.contains("(?i)INSUFFICIENT")
        )
    )

    delayed_by_key: Dict[Tuple[str, str, str], List[dict]] = {}
    if delayed_lots.height > 0:
        lot_summary = delayed_lots.group_by(
            ["simulation_id", "lot_id", "model_id", "target_month", "actual_completion_month", "revenue_plan_id"]
        ).agg(
            [
                pl.col("simulation_name").first(),
                pl.col("units_produced").max().alias("units_affected"),
                pl.col("equipment_group").drop_nulls().first().alias("equipment_group"),
                pl.col("process_id").drop_nulls().first().alias("process_id"),
                pl.col("delay_reasons").drop_nulls().first().alias("delay_reasons"),
                pl.col("model_customer_name").first(),
                pl.col("model_end_customer").first(),
                pl.col("model_sales_team").first(),
                pl.col("model_priority").first(),
                pl.col("grouping_model").first(),
                pl.col("allocation_run_id").first(),
                pl.col("allocation_run_ts").first(),
                pl.col("allocation_id").first().alias("source_allocation_id"),
                pl.col("total_margin").max().alias("total_margin"),
                pl.col("total_revenue").max().alias("total_revenue"),
                pl.col("final_production_units").max().alias("final_units"),
            ]
        )

        for row in lot_summary.iter_rows(named=True):
            key = (row.get("simulation_id"), row.get("model_id"), row.get("target_month"))
            if key not in delayed_by_key:
                delayed_by_key[key] = []
            delayed_by_key[key].append(row)

    processed_lot_ids: Set[str] = set()

    for lot_key in processed_failed_lot_ids:
        processed_lot_ids.add(lot_key[3])

    for row in fulfillment_with_shortfall.iter_rows(named=True):
        sim_id = row.get("simulation_id")
        model_id = row.get("model_id")
        target_month = row.get("month_str")

        if not sim_id or not model_id or not target_month:
            continue

        key = (sim_id, model_id, target_month)

        fulfillment_stat_id = row.get("fulfillment_stat_id")
        planned_demand_ea = row.get("planned_demand_ea", 0) or 0
        planned_demand_krw = row.get("planned_demand_krw", 0) or 0
        prod_inv_ea = row.get("production_and_inventory_ea", 0) or 0
        prod_inv_krw = row.get("production_and_inventory_krw", 0) or 0
        impossible_ea = row.get("impossible_to_simulate_ea", 0) or 0
        impossible_krw = row.get("impossible_to_simulate_krw", 0) or 0
        shortfall_ea = row.get("shortfall_ea", 0) or 0
        shortfall_krw = row.get("shortfall_krw", 0) or 0

        unit_price_krw = planned_demand_krw / planned_demand_ea if planned_demand_ea > 0 else 0

        abs_shortfall_ea = abs(shortfall_ea)
        abs_shortfall_krw = abs(shortfall_krw)

        krw_per_unit = abs_shortfall_krw / abs_shortfall_ea if abs_shortfall_ea > 0 else 0

        remaining_shortfall_ea = abs_shortfall_ea
        remaining_shortfall_krw = abs_shortfall_krw

        if impossible_ea > 0:
            failed_lots = failed_by_key.get(key, [])

            no_equipment_units = sum(
                fa.get("units_in_lot", 0) or 0
                for fa in failed_lots
                if is_no_equipment_failure(fa.get("failure_status", ""))
            )

            remaining_impossible = impossible_ea - no_equipment_units

            if remaining_impossible > 0 and remaining_shortfall_ea > 0:
                contribution_ea = min(remaining_impossible, remaining_shortfall_ea)
                contribution_krw = contribution_ea * krw_per_unit
                contribution_pct = (contribution_ea / abs_shortfall_ea * 100) if abs_shortfall_ea > 0 else 0

                results.append(
                    {
                        "risk_record_id": f"FA_{sim_id}_{model_id}_{target_month}_AGG",
                        "simulation_id": sim_id,
                        "simulation_name": row.get("simulation_name"),
                        "model_id": model_id,
                        "target_month": target_month,
                        "risk_category": "FAILED_ALLOCATION",
                        "risk_category_description": RISK_CATEGORY_DESCRIPTIONS["FAILED_ALLOCATION"],
                        "lot_id": None,
                        "affected_lot_ids": None,
                        "source_dataset": "simulation_monthly_fulfillment_wide",
                        "source_record_id": fulfillment_stat_id,
                        "source_fulfillment_id": fulfillment_stat_id,
                        "units_affected": remaining_impossible,
                        "total_units_at_risk": remaining_impossible,
                        "lots_affected": None,
                        "actual_completion_month": None,
                        "months_delayed": 0,
                        "days_delayed": None,
                        "max_days_delayed": None,
                        "bottleneck_process": None,
                        "bottleneck_equipment_group": None,
                        "bottleneck_equipment_ids": None,
                        "delay_dates": None,
                        "failure_reason": f"Aggregated impossible_to_simulate: {remaining_impossible:,} units could not be scheduled (excludes FAILED_NO_EQUIPMENT which is tracked separately)",
                        "failure_reason_summary": f"{remaining_impossible:,} units could not be scheduled (no lot-level detail available)",
                        "is_model_blocked": False,
                        "current_capacity": None,
                        "required_capacity": None,
                        "capacity_gap": None,
                        "capacity_utilization_pct": None,
                        "planned_demand_ea": planned_demand_ea,
                        "planned_demand_krw": planned_demand_krw,
                        "production_and_inventory_ea": prod_inv_ea,
                        "production_and_inventory_krw": prod_inv_krw,
                        "impossible_to_simulate_ea": impossible_ea,
                        "impossible_to_simulate_krw": impossible_krw,
                        "fulfillment_shortfall_ea": shortfall_ea,
                        "fulfillment_shortfall_krw": shortfall_krw,
                        "shortfall_contribution_ea": contribution_ea,
                        "shortfall_contribution_krw": contribution_krw,
                        "shortfall_contribution_pct": contribution_pct,
                        "revenue_at_risk": contribution_ea * unit_price_krw,
                        "margin_at_risk": None,
                        "unit_price_krw": unit_price_krw,
                        "grouping_model": row.get("grouping_model"),
                        "model_customer_name": row.get("model_customer_name"),
                        "model_end_customer": row.get("model_end_customer"),
                        "model_sales_team": row.get("model_sales_team"),
                        "model_priority": None,
                        "revenue_plan_id": row.get("revenue_plan_id"),
                        "mitigation_strategy": None,
                        "action_required": "Investigate failed allocations - may include other simulation boundary issues.",
                        "allocation_run_id": None,
                        "allocation_run_ts": None,
                    }
                )

                remaining_shortfall_ea -= contribution_ea
                remaining_shortfall_krw -= contribution_krw

        if remaining_shortfall_ea > 0:
            et_jig_data = et_jig_by_key.get(key)

            if et_jig_data and et_jig_data.get("units_at_risk", 0) > 0:
                total_units_at_risk = et_jig_data.get("units_at_risk", 0)
                at_risk_lot_ids = et_jig_data.get("at_risk_lot_ids", []) or []
                lots_at_risk = et_jig_data.get("lots_at_risk", 0)

                contribution_ea = min(total_units_at_risk, remaining_shortfall_ea)
                contribution_krw = contribution_ea * krw_per_unit
                contribution_pct = (contribution_ea / abs_shortfall_ea * 100) if abs_shortfall_ea > 0 else 0

                additional_jigs = et_jig_data.get("additional_et_jigs_needed", 0)
                current_jig_count = et_jig_data.get("current_et_jig_count", 0)
                total_jigs_needed = et_jig_data.get("total_et_jigs_needed", 0)
                risk_reason = et_jig_data.get("risk_reason", "")
                mitigation = et_jig_data.get("mitigation_strategy", "")
                avg_days_late = et_jig_data.get("average_days_late", 0)
                max_days_late = et_jig_data.get("max_days_late", 0)

                capacity_util = None
                if current_jig_count > 0 and total_jigs_needed > 0:
                    capacity_util = min(100.0, (total_jigs_needed / current_jig_count) * 100)

                lots_contributing = lots_at_risk
                if total_units_at_risk > 0 and contribution_ea < total_units_at_risk:
                    lots_contributing = max(1, int(lots_at_risk * contribution_ea / total_units_at_risk))

                failure_reason_detail = (
                    f"ET JIG capacity constraint: {lots_at_risk} lots ({total_units_at_risk:,} total units at risk) "
                    f"delayed avg {avg_days_late:.1f} days (max {max_days_late} days). "
                    f"Of these, {contribution_ea:,} units contribute to shortfall. "
                    f"Current JIGs: {current_jig_count}, Need: {total_jigs_needed} (+{additional_jigs}). "
                    f"{risk_reason}"
                )

                results.append(
                    {
                        "risk_record_id": f"ETJIG_{sim_id}_{model_id}_{target_month}",
                        "simulation_id": sim_id,
                        "simulation_name": row.get("simulation_name"),
                        "model_id": model_id,
                        "target_month": target_month,
                        "risk_category": "ET_JIG_CAPACITY_ISSUE",
                        "risk_category_description": RISK_CATEGORY_DESCRIPTIONS["ET_JIG_CAPACITY_ISSUE"],
                        "lot_id": None,
                        "affected_lot_ids": at_risk_lot_ids if at_risk_lot_ids else None,
                        "source_dataset": "et_jig_capacity_risk_analysis",
                        "source_record_id": et_jig_data.get("et_jig_risk_id"),
                        "source_fulfillment_id": fulfillment_stat_id,
                        "units_affected": contribution_ea,
                        "total_units_at_risk": total_units_at_risk,
                        "lots_affected": lots_contributing,
                        "actual_completion_month": None,
                        "months_delayed": int(avg_days_late / 30) if avg_days_late else 0,
                        "days_delayed": int(avg_days_late) if avg_days_late else None,
                        "max_days_delayed": max_days_late if max_days_late else None,
                        "bottleneck_process": "ET_JIG_START",
                        "bottleneck_equipment_group": "ET_JIG",
                        "bottleneck_equipment_ids": ["ET_JIG"],
                        "delay_dates": None,
                        "failure_reason": failure_reason_detail,
                        "failure_reason_summary": f"ET JIG capacity: {lots_at_risk} lots at risk, need {additional_jigs} more JIGs, avg delay {avg_days_late:.0f} days",
                        "is_model_blocked": False,
                        "current_capacity": current_jig_count,
                        "required_capacity": total_jigs_needed,
                        "capacity_gap": additional_jigs,
                        "capacity_utilization_pct": capacity_util,
                        "planned_demand_ea": planned_demand_ea,
                        "planned_demand_krw": planned_demand_krw,
                        "production_and_inventory_ea": prod_inv_ea,
                        "production_and_inventory_krw": prod_inv_krw,
                        "impossible_to_simulate_ea": impossible_ea,
                        "impossible_to_simulate_krw": impossible_krw,
                        "fulfillment_shortfall_ea": shortfall_ea,
                        "fulfillment_shortfall_krw": shortfall_krw,
                        "shortfall_contribution_ea": contribution_ea,
                        "shortfall_contribution_krw": contribution_krw,
                        "shortfall_contribution_pct": contribution_pct,
                        "revenue_at_risk": et_jig_data.get("revenue_at_risk", 0) or (contribution_ea * unit_price_krw),
                        "margin_at_risk": et_jig_data.get("margin_at_risk"),
                        "unit_price_krw": unit_price_krw,
                        "grouping_model": row.get("grouping_model"),
                        "model_customer_name": row.get("model_customer_name"),
                        "model_end_customer": row.get("model_end_customer"),
                        "model_sales_team": row.get("model_sales_team"),
                        "model_priority": et_jig_data.get("model_priority"),
                        "revenue_plan_id": row.get("revenue_plan_id"),
                        "mitigation_strategy": mitigation,
                        "action_required": get_action_for_risk(
                            "ET_JIG_CAPACITY_ISSUE", {"additional_jigs": additional_jigs}
                        ),
                        "allocation_run_id": et_jig_data.get("allocation_run_id"),
                        "allocation_run_ts": et_jig_data.get("allocation_run_ts"),
                    }
                )

                remaining_shortfall_ea -= contribution_ea
                remaining_shortfall_krw -= contribution_krw

                for lot_id in at_risk_lot_ids:
                    processed_lot_ids.add(lot_id)

        if remaining_shortfall_ea > 0:
            delayed_lots_for_key = delayed_by_key.get(key, [])

            for dl in delayed_lots_for_key:
                lot_id = dl.get("lot_id")
                if lot_id in processed_lot_ids:
                    continue
                processed_lot_ids.add(lot_id)

                units = dl.get("final_units") or dl.get("units_affected") or 0
                delay_reasons = dl.get("delay_reasons", "")

                if not has_equipment_capacity_issue(delay_reasons):
                    continue

                equipment_ids, blocked_ids, is_blocked = extract_equipment_ids_from_delay_reasons(delay_reasons)
                delay_dates = extract_delay_dates(delay_reasons)

                if not delay_dates and not equipment_ids:
                    continue

                contribution_ea = min(units, remaining_shortfall_ea)
                contribution_krw = contribution_ea * krw_per_unit
                contribution_pct = (contribution_ea / abs_shortfall_ea * 100) if abs_shortfall_ea > 0 else 0

                months_delayed = calculate_months_delayed(target_month, dl.get("actual_completion_month"))

                equipment_group = dl.get("equipment_group")
                process_id = dl.get("process_id")

                results.append(
                    {
                        "risk_record_id": f"ECD_{sim_id}_{lot_id}_{target_month}",
                        "simulation_id": sim_id,
                        "simulation_name": row.get("simulation_name"),
                        "model_id": model_id,
                        "target_month": target_month,
                        "risk_category": "EQUIPMENT_CAPACITY_DELAY",
                        "risk_category_description": RISK_CATEGORY_DESCRIPTIONS["EQUIPMENT_CAPACITY_DELAY"],
                        "lot_id": lot_id,
                        "affected_lot_ids": [lot_id],
                        "source_dataset": "revenue_plan_optimization_events",
                        "source_record_id": dl.get("source_allocation_id"),
                        "source_fulfillment_id": fulfillment_stat_id,
                        "units_affected": units,
                        "total_units_at_risk": units,
                        "lots_affected": 1,
                        "actual_completion_month": dl.get("actual_completion_month"),
                        "months_delayed": months_delayed,
                        "days_delayed": None,
                        "max_days_delayed": None,
                        "bottleneck_process": process_id,
                        "bottleneck_equipment_group": equipment_group,
                        "bottleneck_equipment_ids": equipment_ids if equipment_ids else None,
                        "delay_dates": delay_dates if delay_dates else None,
                        "failure_reason": delay_reasons[:500] if delay_reasons else None,
                        "failure_reason_summary": summarize_failure_reason(delay_reasons),
                        "is_model_blocked": is_blocked,
                        "current_capacity": None,
                        "required_capacity": None,
                        "capacity_gap": None,
                        "capacity_utilization_pct": 100.0,
                        "planned_demand_ea": planned_demand_ea,
                        "planned_demand_krw": planned_demand_krw,
                        "production_and_inventory_ea": prod_inv_ea,
                        "production_and_inventory_krw": prod_inv_krw,
                        "impossible_to_simulate_ea": impossible_ea,
                        "impossible_to_simulate_krw": impossible_krw,
                        "fulfillment_shortfall_ea": shortfall_ea,
                        "fulfillment_shortfall_krw": shortfall_krw,
                        "shortfall_contribution_ea": contribution_ea,
                        "shortfall_contribution_krw": contribution_krw,
                        "shortfall_contribution_pct": contribution_pct,
                        "revenue_at_risk": dl.get("total_revenue") or (contribution_ea * unit_price_krw),
                        "margin_at_risk": dl.get("total_margin"),
                        "unit_price_krw": unit_price_krw,
                        "grouping_model": row.get("grouping_model"),
                        "model_customer_name": row.get("model_customer_name"),
                        "model_end_customer": row.get("model_end_customer"),
                        "model_sales_team": row.get("model_sales_team"),
                        "model_priority": dl.get("model_priority"),
                        "revenue_plan_id": row.get("revenue_plan_id"),
                        "mitigation_strategy": None,
                        "action_required": get_action_for_risk(
                            "EQUIPMENT_CAPACITY_DELAY",
                            {"equipment_group": equipment_group, "months_delayed": months_delayed},
                        ),
                        "allocation_run_id": dl.get("allocation_run_id"),
                        "allocation_run_ts": dl.get("allocation_run_ts"),
                    }
                )

                remaining_shortfall_ea -= contribution_ea
                remaining_shortfall_krw -= contribution_krw

                if remaining_shortfall_ea <= 0:
                    break

        if remaining_shortfall_ea > 0:
            contribution_pct = (remaining_shortfall_ea / abs_shortfall_ea * 100) if abs_shortfall_ea > 0 else 0

            results.append(
                {
                    "risk_record_id": f"LT_{sim_id}_{model_id}_{target_month}",
                    "simulation_id": sim_id,
                    "simulation_name": row.get("simulation_name"),
                    "model_id": model_id,
                    "target_month": target_month,
                    "risk_category": "LEAD_TIME_SHORTFALL",
                    "risk_category_description": RISK_CATEGORY_DESCRIPTIONS["LEAD_TIME_SHORTFALL"],
                    "lot_id": None,
                    "affected_lot_ids": None,
                    "source_dataset": "simulation_monthly_fulfillment_wide",
                    "source_record_id": fulfillment_stat_id,
                    "source_fulfillment_id": fulfillment_stat_id,
                    "units_affected": remaining_shortfall_ea,
                    "total_units_at_risk": remaining_shortfall_ea,
                    "lots_affected": None,
                    "actual_completion_month": None,
                    "months_delayed": None,
                    "days_delayed": None,
                    "max_days_delayed": None,
                    "bottleneck_process": None,
                    "bottleneck_equipment_group": None,
                    "bottleneck_equipment_ids": None,
                    "delay_dates": None,
                    "failure_reason": (
                        f"Remaining shortfall of {remaining_shortfall_ea:,} units ({remaining_shortfall_krw:,.0f} KRW) "
                        f"not explained by failed allocations, ET JIG capacity, or equipment delays. "
                        f"Likely due to lead time constraints or production capacity limitations."
                    ),
                    "failure_reason_summary": f"Residual shortfall: {remaining_shortfall_ea:,} units unexplained by specific bottlenecks",
                    "is_model_blocked": False,
                    "current_capacity": None,
                    "required_capacity": None,
                    "capacity_gap": None,
                    "capacity_utilization_pct": None,
                    "planned_demand_ea": planned_demand_ea,
                    "planned_demand_krw": planned_demand_krw,
                    "production_and_inventory_ea": prod_inv_ea,
                    "production_and_inventory_krw": prod_inv_krw,
                    "impossible_to_simulate_ea": impossible_ea,
                    "impossible_to_simulate_krw": impossible_krw,
                    "fulfillment_shortfall_ea": shortfall_ea,
                    "fulfillment_shortfall_krw": shortfall_krw,
                    "shortfall_contribution_ea": remaining_shortfall_ea,
                    "shortfall_contribution_krw": remaining_shortfall_krw,
                    "shortfall_contribution_pct": contribution_pct,
                    "revenue_at_risk": remaining_shortfall_ea * unit_price_krw,
                    "margin_at_risk": None,
                    "unit_price_krw": unit_price_krw,
                    "grouping_model": row.get("grouping_model"),
                    "model_customer_name": row.get("model_customer_name"),
                    "model_end_customer": row.get("model_end_customer"),
                    "model_sales_team": row.get("model_sales_team"),
                    "model_priority": None,
                    "revenue_plan_id": row.get("revenue_plan_id"),
                    "mitigation_strategy": None,
                    "action_required": get_action_for_risk("LEAD_TIME_SHORTFALL", {}),
                    "allocation_run_id": None,
                    "allocation_run_ts": None,
                }
            )

    print(f"\n  Generated {len(results):,} risk records")

    if not results:
        print("  ⚠️ No risk records generated")
        result_df = pl.DataFrame(schema=PRODUCTION_RISK_RECONCILIATION_SCHEMA)
    else:
        result_df = pl.DataFrame(results, schema=PRODUCTION_RISK_RECONCILIATION_SCHEMA)

    print("\n" + "=" * 70)
    print("RECONCILIATION CHECK")
    print("=" * 70)

    fulfillment_totals = (
        fulfillment_with_shortfall.group_by(["simulation_id", "simulation_name"])
        .agg(
            [
                pl.col("shortfall_ea").abs().sum().alias("fulfillment_shortfall_ea"),
                pl.col("shortfall_krw").abs().sum().alias("fulfillment_shortfall_krw"),
            ]
        )
        .sort("simulation_name")
    )

    if result_df.height > 0:
        risk_totals = result_df.group_by(["simulation_id", "simulation_name"]).agg(
            [
                pl.col("shortfall_contribution_ea").sum().alias("risk_contribution_ea"),
                pl.col("shortfall_contribution_krw").sum().alias("risk_contribution_krw"),
                pl.col("units_affected").sum().alias("total_units_affected"),
            ]
        )

        comparison = fulfillment_totals.join(risk_totals, on=["simulation_id", "simulation_name"], how="left")

        print(f"\n{'Simulation':<25} {'Fulfillment EA':<18} {'Risk EA':<18} {'Units Affected':<18}")
        print("-" * 79)
        for row in comparison.iter_rows(named=True):
            sim_name = (row.get("simulation_name") or "Unknown")[:24]
            f_ea = row.get("fulfillment_shortfall_ea", 0) or 0
            r_ea = row.get("risk_contribution_ea", 0) or 0
            u_aff = row.get("total_units_affected", 0) or 0
            print(f"{sim_name:<25} {f_ea:>17,} {r_ea:>17,} {u_aff:>17,}")

        print("\n--- By Risk Category ---")
        category_summary = (
            result_df.group_by("risk_category")
            .agg(
                [
                    pl.count().alias("record_count"),
                    pl.col("shortfall_contribution_ea").sum().alias("shortfall_ea"),
                    pl.col("units_affected").sum().alias("units_affected"),
                    pl.col("shortfall_contribution_krw").sum().alias("total_krw"),
                ]
            )
            .sort("total_krw", descending=True)
        )

        print(f"\n{'Risk Category':<30} {'Records':<12} {'Shortfall EA':<18} {'Units Affected':<18} {'KRW':<20}")
        print("-" * 98)
        for row in category_summary.iter_rows(named=True):
            print(
                f"{row['risk_category']:<30} "
                f"{row['record_count']:<12,} "
                f"{row['shortfall_ea']:<18,} "
                f"{row['units_affected']:<18,} "
                f"{row['total_krw']:>19,.0f}"
            )

    print("\n--- Writing Output ---")
    risk_reconciliation_output.write_table(result_df.unique(subset=["risk_record_id"]))
    print(f"✓ Wrote {result_df.height:,} production risk reconciliation records")

    print("\n" + "=" * 70)
    print("✓ PRODUCTION RISK RECONCILIATION COMPLETE")
    print("=" * 70)
