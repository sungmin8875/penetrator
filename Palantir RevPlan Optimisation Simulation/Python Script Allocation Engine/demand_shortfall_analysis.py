"""
Demand Shortfall Analysis

Analyzes why demand for a particular model in a particular month for a specific
simulation could not be met. Produces one row per model-month-simulation with shortfall.

Only creates entries when we can attribute the shortfall to a capacity issue
(equipment at capacity, equipment blocked, no valid equipment, or lead time constraints).

Shortfall Reasons:
1. INSUFFICIENT_CAPACITY - Equipment groups were genuinely at capacity
2. EQUIPMENT_BLOCKED - Equipment had capacity but model was blocked (negative constraints)
3. NO_VALID_EQUIPMENT - Process step has no available equipment (all blocked or none mapped)
4. LEAD_TIME_AND_ET_JIG_CAPACITY - Lead time too long to complete within target month
"""

import polars as pl
from transforms.api import transform, Input, Output, lightweight
from datetime import datetime, date, timedelta
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass
import re


# =============================================================================
# Output Schema Definition
# =============================================================================

DEMAND_SHORTFALL_SCHEMA = {
    # === Shortfall Context (model-month level) ===
    "demand_shortfall_id": pl.Utf8,  # {simulation_id}_{model_id}_{target_month}
    "title": pl.Utf8,  # Human-readable title for display
    "simulation_id": pl.Utf8,
    "simulation_name": pl.Utf8,
    "revenue_plan_id": pl.Utf8,
    "model_id": pl.Utf8,
    "grouping_model": pl.Utf8,
    "model_sales_team": pl.Utf8,
    "model_customer_name": pl.Utf8,
    "model_end_customer": pl.Utf8,
    "model_priority": pl.Int32,
    "target_month": pl.Utf8,
    # === Shortfall Metrics (at model-month level) ===
    "demand_qty_units": pl.Int64,  # Total demand for this model-month
    "produced_on_time_units": pl.Int64,  # Units that completed for this target_month (including early)
    "shortfall_qty_units": pl.Int64,  # demand - produced_on_time
    "shortfall_pct": pl.Float64,  # Percentage of demand that is shortfall
    # === Financial Impact ===
    "shortfall_revenue_impact": pl.Float64,  # Revenue impact from shortfall
    "shortfall_margin_impact": pl.Float64,  # Margin impact from shortfall
    # === Lot Statistics ===
    "delayed_lot_count": pl.Int32,  # Number of lots delayed
    "failed_lot_count": pl.Int32,  # Number of lots that failed allocation
    "delayed_units": pl.Int64,  # Units from delayed lots
    "failed_units": pl.Int64,  # Units from failed lots
    "lead_time_constrained_lot_count": pl.Int32,  # Lots constrained by lead time
    "lead_time_constrained_units": pl.Int64,  # Units from lead time constrained lots
    # === Reason Categorization ===
    "primary_reason": pl.Utf8,  # INSUFFICIENT_CAPACITY, EQUIPMENT_BLOCKED, NO_VALID_EQUIPMENT, LEAD_TIME_AND_ET_JIG_CAPACITY
    "reason_detail": pl.Utf8,  # Human-readable explanation with lot-level detail
    # === Blocking Process Step Details (for NO_VALID_EQUIPMENT) ===
    "blocking_process_id": pl.Utf8,  # Process step ID where failure occurred
    "blocking_process_equipment_group": pl.Utf8,  # Equipment group for blocking process
    # === Capacity Issue Details (when applicable) ===
    "bottleneck_equipment_ids": pl.List(pl.Utf8),  # Equipment IDs at capacity
    "bottleneck_equipment_group_ids": pl.List(pl.Utf8),  # Equipment group IDs at capacity
    "blocked_equipment_ids": pl.List(pl.Utf8),  # Equipment that HAD capacity but was blocked
    "blocked_equipment_count": pl.Int32,  # Total count of blocked equipment
    # === Timing Details ===
    "earliest_delay_date": pl.Date,
    "latest_delay_date": pl.Date,
    "target_month_end_date": pl.Date,  # End of target month for reference
    "earliest_possible_completion": pl.Date,  # When lots could complete at earliest
    # === Links to Related Records ===
    "shortage_ids": pl.List(pl.Utf8),  # Array of shortage_ids from equipment_capacity_shortages
    # === Run Metadata ===
    "allocation_run_id": pl.Utf8,
    "allocation_run_ts": pl.Datetime,
}


# =============================================================================
# Data Classes for Rich Detail Collection
# =============================================================================


@dataclass
class EquipmentCapacityInfo:
    """Aggregated capacity info for an equipment."""

    equipment_id: str
    event_type: str  # "FULL" or "BLOCKED"
    capacity_used: Optional[float] = None
    capacity_total: Optional[float] = None
    capacity_needed: Optional[int] = None
    capacity_available: Optional[float] = None
    event_count: int = 1
    earliest_date: Optional[date] = None
    latest_date: Optional[date] = None


@dataclass
class WaitingPeriodInfo:
    """Information from lot_waiting_periods dataset."""

    lot_id: str
    model_id: str
    target_month: Optional[str]
    process_id: Optional[str]
    equipment_group: Optional[str]
    waiting_start_date: Optional[date]
    waiting_end_date: Optional[date]
    waiting_days: int
    is_failed_allocation: bool
    failure_status: Optional[str]
    failure_detail: Optional[str]
    blocked_equipment_list: Optional[str]
    blocked_equipment_count: int
    delay_reasons_summary: Optional[str]
    units_affected: int
    margin_at_risk: float


@dataclass
class FailedAllocationInfo:
    """Information from failed_allocations dataset."""

    lot_id: str
    model_id: str
    target_month: str
    process_id: Optional[str]
    equipment_group: Optional[str]
    units_in_lot: int
    failure_reason: Optional[str]
    failure_status: Optional[str]
    delay_details: Optional[str]


@dataclass
class LeadTimeConstraintInfo:
    """Detailed information about a lead-time constrained lot."""

    lot_id: str
    units: int
    start_date: date
    expected_completion_date: date
    lead_time_days: int
    target_month_end: date
    days_over: int  # How many days past month end


# =============================================================================
# Helper Functions
# =============================================================================


def parse_month_to_eom_date(month_str: str) -> date:
    """Parse month string like '202512' to end-of-month date."""
    year = int(month_str[:4])
    month = int(month_str[4:6])

    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)

    return next_month_first - timedelta(days=1)


def format_month_display(month_str: str) -> str:
    """Format month string like '202512' to 'Dec 2025'."""
    if not month_str or len(month_str) < 6:
        return month_str
    year = month_str[:4]
    month_num = int(month_str[4:6])
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{month_names[month_num - 1]} {year}"


def format_date_display(d: Optional[date]) -> str:
    """Format date for display."""
    if d is None:
        return "N/A"
    return d.strftime("%Y-%m-%d")


def parse_equipment_capacity_from_summary(delay_reasons_summary: str) -> List[EquipmentCapacityInfo]:
    """
    Parse the delay_reasons_summary from lot_waiting_periods to extract equipment capacity details.

    Example formats:
    - "GPSOB-02 full (15/16.0 sheets, needed 5)"
    - "MCB-00009 BLOCKED for model (had ? sheets available)"
    - "PLT-00107 full (30/35.0 sheets, needed 10)"
    """
    results = []
    if not delay_reasons_summary:
        return results

    # Pattern for capacity full: "EQUIP_ID full (X/Y sheets, needed Z)"
    capacity_pattern = r"([A-Za-z0-9\-]+)\s+full\s+\((\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s+sheets,\s+needed\s+(\d+)\)"

    # Pattern for blocked: "EQUIP_ID BLOCKED for model (had X sheets available)" or "(had ? sheets available)"
    blocked_pattern = r"([A-Za-z0-9\-]+)\s+BLOCKED\s+for\s+model\s+\(had\s+(\d+(?:\.\d+)?|\?)\s+sheets\s+available\)"

    # Find all capacity issues
    for match in re.finditer(capacity_pattern, delay_reasons_summary):
        equipment_id, used, total, needed = match.groups()
        results.append(
            EquipmentCapacityInfo(
                equipment_id=equipment_id,
                event_type="FULL",
                capacity_used=float(used),
                capacity_total=float(total),
                capacity_needed=int(needed),
            )
        )

    # Find all blocked issues
    for match in re.finditer(blocked_pattern, delay_reasons_summary):
        equipment_id, available = match.groups()
        avail_val = None if available == "?" else float(available)
        results.append(
            EquipmentCapacityInfo(
                equipment_id=equipment_id,
                event_type="BLOCKED",
                capacity_available=avail_val,
            )
        )

    return results


def convert_epoch_to_date(val) -> Optional[date]:
    """Convert epoch milliseconds or date to date object."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, (int, float)):
        # Assume milliseconds
        return datetime.fromtimestamp(val / 1000).date()
    return None


# =============================================================================
# Rich Description Builders
# =============================================================================


def build_title(
    model_id: str,
    target_month: str,
    shortfall_units: int,
    primary_reason: str,
    customer_name: Optional[str] = None,
) -> str:
    """Build a human-readable title for the shortfall record."""
    month_display = format_month_display(target_month)

    reason_labels = {
        "INSUFFICIENT_CAPACITY": "Capacity Shortage",
        "EQUIPMENT_BLOCKED": "Blocked Equipment",
        "NO_VALID_EQUIPMENT": "No Available Equipment",
        "LEAD_TIME_AND_ET_JIG_CAPACITY": "Lead Time Constraint",
    }
    reason_label = reason_labels.get(primary_reason, primary_reason)

    if customer_name:
        return f"{reason_label}: {model_id} ({customer_name}) - {shortfall_units:,} units in {month_display}"
    return f"{reason_label}: {model_id} - {shortfall_units:,} units in {month_display}"


def build_rich_reason_detail(
    primary_reason: str,
    waiting_periods: List[WaitingPeriodInfo],
    failed_allocations: List[FailedAllocationInfo],
    lead_time_lots: List[LeadTimeConstraintInfo],
    target_month: str,
    target_month_end: date,
) -> str:
    """
    Build a rich, detailed description of the shortfall reason.

    This provides actionable information including:
    - Specific equipment IDs and their capacity status (with utilization %)
    - Exact dates when delays occurred
    - Lot-level details for tracking
    - Process step information
    """
    month_display = format_month_display(target_month)
    detail_parts = []

    # === LEAD TIME CONSTRAINTS ===
    if lead_time_lots:
        total_lt_units = sum(lt.units for lt in lead_time_lots)

        lt_summary = f"**Lead Time Constraints ({len(lead_time_lots)} lots, {total_lt_units:,} units):**\n"
        lt_summary += f"These lots cannot complete within {month_display} because their start date + lead time exceeds the month end ({format_date_display(target_month_end)}).\n\n"

        # Show detailed examples (up to 5 lots)
        for lt in sorted(lead_time_lots, key=lambda x: -x.units)[:5]:
            lt_summary += (
                f"  • Lot {lt.lot_id}: {lt.units:,} units\n"
                f"    - Planned start: {format_date_display(lt.start_date)}\n"
                f"    - Lead time required: {lt.lead_time_days} days\n"
                f"    - Expected completion: {format_date_display(lt.expected_completion_date)}\n"
                f"    - Overruns month end by: {lt.days_over} days\n"
            )

        if len(lead_time_lots) > 5:
            remaining = len(lead_time_lots) - 5
            remaining_units = sum(lt.units for lt in sorted(lead_time_lots, key=lambda x: -x.units)[5:])
            lt_summary += f"  • ... and {remaining} more lots ({remaining_units:,} units)\n"

        detail_parts.append(lt_summary)

    # === FAILED ALLOCATIONS - NO VALID EQUIPMENT ===
    failed_no_equipment = [
        fa for fa in failed_allocations if fa.failure_status and "NO_EQUIPMENT" in fa.failure_status.upper()
    ]
    failed_capacity = [
        fa for fa in failed_allocations if fa.failure_status and "INSUFFICIENT" in fa.failure_status.upper()
    ]

    if primary_reason == "NO_VALID_EQUIPMENT" and failed_no_equipment:
        total_units = sum(fa.units_in_lot for fa in failed_no_equipment)

        # Group by process step
        by_process: Dict[str, List[FailedAllocationInfo]] = {}
        for fa in failed_no_equipment:
            proc = fa.process_id or "Unknown"
            by_process.setdefault(proc, []).append(fa)

        ne_summary = f"**No Valid Equipment ({len(failed_no_equipment)} lots, {total_units:,} units):**\n"
        ne_summary += "All equipment for these process steps is blocked for this model.\n\n"

        for proc_id, fas in by_process.items():
            proc_units = sum(fa.units_in_lot for fa in fas)
            eq_group = next((fa.equipment_group for fa in fas if fa.equipment_group), None)

            ne_summary += f"  • Process: '{proc_id}'"
            if eq_group:
                ne_summary += f" (Equipment Group: {eq_group})"
            ne_summary += f"\n"
            ne_summary += f"    - Affected: {len(fas)} lots, {proc_units:,} units\n"

            # Show failure details
            for fa in fas[:2]:
                if fa.failure_reason:
                    ne_summary += f"    - Detail: {fa.failure_reason}\n"

            # Show lot IDs
            lot_ids = [fa.lot_id for fa in fas[:5]]
            ne_summary += f"    - Lot IDs: {', '.join(lot_ids)}"
            if len(fas) > 5:
                ne_summary += f" (+{len(fas) - 5} more)"
            ne_summary += "\n"

        detail_parts.append(ne_summary)

    # === WAITING PERIODS ANALYSIS ===
    # Separate waiting periods by type
    delayed_blocked = [wp for wp in waiting_periods if not wp.is_failed_allocation and wp.blocked_equipment_count > 0]
    delayed_capacity = [wp for wp in waiting_periods if not wp.is_failed_allocation and wp.blocked_equipment_count == 0]

    # === EQUIPMENT BLOCKED (had capacity but blocked) ===
    if primary_reason == "EQUIPMENT_BLOCKED" and delayed_blocked:
        total_units = sum(wp.units_affected for wp in delayed_blocked)
        total_margin = sum(wp.margin_at_risk for wp in delayed_blocked)

        # Collect all blocked equipment across all waiting periods
        all_blocked_equipment: Set[str] = set()
        for wp in delayed_blocked:
            if wp.blocked_equipment_list:
                for eq in wp.blocked_equipment_list.split(", "):
                    if eq.strip():
                        all_blocked_equipment.add(eq.strip())

        # Parse equipment capacity details
        all_capacity_info: List[EquipmentCapacityInfo] = []
        for wp in delayed_blocked:
            if wp.delay_reasons_summary:
                all_capacity_info.extend(parse_equipment_capacity_from_summary(wp.delay_reasons_summary))

        eb_summary = f"**Equipment Blocked ({len(delayed_blocked)} lots, {total_units:,} units, ${total_margin:,.0f} margin at risk):**\n"
        eb_summary += "Equipment had available capacity but is blocked for this model (negative constraint).\n\n"

        # Show blocked equipment with any capacity info we have
        blocked_with_info = [ci for ci in all_capacity_info if ci.event_type == "BLOCKED"]
        if blocked_with_info:
            eb_summary += "  • Blocked Equipment:\n"
            for ci in blocked_with_info[:8]:
                if ci.capacity_available is not None:
                    eb_summary += (
                        f"    - {ci.equipment_id}: had {ci.capacity_available:.0f} sheets available but BLOCKED\n"
                    )
                else:
                    eb_summary += f"    - {ci.equipment_id}: BLOCKED for this model\n"
            if len(blocked_with_info) > 8:
                eb_summary += f"    - ... and {len(blocked_with_info) - 8} more blocked equipment\n"
        elif all_blocked_equipment:
            eb_summary += f"  • Blocked Equipment: {', '.join(sorted(all_blocked_equipment)[:10])}"
            if len(all_blocked_equipment) > 10:
                eb_summary += f" (+{len(all_blocked_equipment) - 10} more)"
            eb_summary += "\n"

        # Show date range
        all_dates = []
        for wp in delayed_blocked:
            if wp.waiting_start_date:
                all_dates.append(wp.waiting_start_date)
            if wp.waiting_end_date:
                all_dates.append(wp.waiting_end_date)
        if all_dates:
            eb_summary += f"\n  • Blocking Period: {format_date_display(min(all_dates))} to {format_date_display(max(all_dates))}\n"

        # Show example lots
        eb_summary += "\n  • Affected Lots:\n"
        for wp in sorted(delayed_blocked, key=lambda x: -x.units_affected)[:3]:
            eb_summary += f"    - {wp.lot_id}: {wp.units_affected:,} units, waited {wp.waiting_days} days"
            if wp.equipment_group:
                eb_summary += f" at {wp.equipment_group}"
            eb_summary += "\n"
        if len(delayed_blocked) > 3:
            eb_summary += f"    - ... and {len(delayed_blocked) - 3} more lots\n"

        detail_parts.append(eb_summary)

    # === INSUFFICIENT CAPACITY ===
    if primary_reason == "INSUFFICIENT_CAPACITY":
        capacity_periods = delayed_capacity + [
            WaitingPeriodInfo(
                lot_id=fa.lot_id,
                model_id=fa.model_id,
                target_month=fa.target_month,
                process_id=fa.process_id,
                equipment_group=fa.equipment_group,
                waiting_start_date=None,
                waiting_end_date=None,
                waiting_days=0,
                is_failed_allocation=True,
                failure_status=fa.failure_status,
                failure_detail=fa.failure_reason,
                blocked_equipment_list=None,
                blocked_equipment_count=0,
                delay_reasons_summary=fa.delay_details,
                units_affected=fa.units_in_lot,
                margin_at_risk=0.0,
            )
            for fa in failed_capacity
        ]

        if capacity_periods:
            total_units = sum(wp.units_affected for wp in capacity_periods)
            total_margin = sum(wp.margin_at_risk for wp in capacity_periods)

            # Parse all equipment capacity details
            equip_capacity: Dict[str, List[EquipmentCapacityInfo]] = {}
            for wp in capacity_periods:
                if wp.delay_reasons_summary:
                    for ci in parse_equipment_capacity_from_summary(wp.delay_reasons_summary):
                        if ci.event_type == "FULL":
                            equip_capacity.setdefault(ci.equipment_id, []).append(ci)

            ic_summary = f"**Insufficient Capacity ({len(capacity_periods)} lots, {total_units:,} units, ${total_margin:,.0f} margin at risk):**\n"
            ic_summary += "Equipment was fully utilized and could not accommodate additional lots.\n\n"

            # Show equipment utilization details
            if equip_capacity:
                ic_summary += "  • Equipment Utilization:\n"
                for equip_id, infos in sorted(equip_capacity.items(), key=lambda x: -len(x[1]))[:6]:
                    ci = infos[0]
                    utilization = (ci.capacity_used / ci.capacity_total * 100) if ci.capacity_total else 100
                    ic_summary += f"    - {equip_id}: {ci.capacity_used:.0f}/{ci.capacity_total:.0f} sheets ({utilization:.0f}% utilized)"
                    ic_summary += f", needed {ci.capacity_needed} sheets"
                    if len(infos) > 1:
                        ic_summary += f" [{len(infos)} occurrences]"
                    ic_summary += "\n"
                if len(equip_capacity) > 6:
                    ic_summary += f"    - ... and {len(equip_capacity) - 6} more equipment at capacity\n"

            # Show date range
            all_dates = []
            for wp in capacity_periods:
                if wp.waiting_start_date:
                    all_dates.append(wp.waiting_start_date)
                if wp.waiting_end_date:
                    all_dates.append(wp.waiting_end_date)
            if all_dates:
                ic_summary += f"\n  • Shortage Period: {format_date_display(min(all_dates))} to {format_date_display(max(all_dates))}\n"

            # Group by equipment group
            by_eq_group: Dict[str, List[WaitingPeriodInfo]] = {}
            for wp in capacity_periods:
                eg = wp.equipment_group or "Unknown"
                by_eq_group.setdefault(eg, []).append(wp)

            if len(by_eq_group) > 1:
                ic_summary += "\n  • By Equipment Group:\n"
                for eg, wps in sorted(by_eq_group.items(), key=lambda x: -sum(w.units_affected for w in x[1]))[:5]:
                    eg_units = sum(w.units_affected for w in wps)
                    ic_summary += f"    - {eg}: {len(wps)} lots, {eg_units:,} units\n"

            # Show example lots
            ic_summary += "\n  • Most Affected Lots:\n"
            for wp in sorted(capacity_periods, key=lambda x: -x.units_affected)[:3]:
                ic_summary += f"    - {wp.lot_id}: {wp.units_affected:,} units, waited {wp.waiting_days} days"
                if wp.process_id:
                    ic_summary += f" at process {wp.process_id}"
                ic_summary += "\n"
            if len(capacity_periods) > 3:
                ic_summary += f"    - ... and {len(capacity_periods) - 3} more lots\n"

            detail_parts.append(ic_summary)

    return "\n".join(detail_parts) if detail_parts else f"Shortfall of demand in {month_display}"


# =============================================================================
# Pre-processing Functions
# =============================================================================


def preprocess_lead_time_constrained_lots(new_lots_df: pl.DataFrame) -> pl.DataFrame:
    """
    Pre-process new lots to identify lead time constrained lots.
    """
    # Check the actual dtype of target_lot_start_date column
    target_col_dtype = new_lots_df.schema.get("target_lot_start_date")

    # Convert target_lot_start_date to date based on actual column type
    if target_col_dtype == pl.Date:
        lots_with_dates = new_lots_df.with_columns([pl.col("target_lot_start_date").alias("start_date")])
    elif target_col_dtype in (pl.Datetime, pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")):
        lots_with_dates = new_lots_df.with_columns([pl.col("target_lot_start_date").dt.date().alias("start_date")])
    else:
        # Assume it's an epoch timestamp in milliseconds
        lots_with_dates = new_lots_df.with_columns(
            [
                pl.when(pl.col("target_lot_start_date").is_not_null())
                .then(pl.from_epoch(pl.col("target_lot_start_date"), time_unit="ms").dt.date())
                .otherwise(None)
                .alias("start_date"),
            ]
        )

    # Calculate expected completion date
    lots_with_dates = lots_with_dates.with_columns(
        [
            (pl.col("start_date") + pl.duration(days=pl.col("lead_time_days")))
            .dt.date()
            .alias("expected_completion_date")
        ]
    )

    # Parse YYYYMM format to get last day of month
    lots_with_dates = lots_with_dates.with_columns(
        [
            pl.col("target_month").str.slice(0, 4).cast(pl.Int32).alias("year"),
            pl.col("target_month").str.slice(4, 2).cast(pl.Int32).alias("month"),
        ]
    )

    # Calculate end of month date
    lots_with_dates = lots_with_dates.with_columns(
        [pl.date(pl.col("year"), pl.col("month"), 1).dt.month_end().alias("target_month_eom")]
    )

    # Mark lots as lead time constrained
    lots_with_dates = lots_with_dates.with_columns(
        [
            (pl.col("expected_completion_date") > pl.col("target_month_eom")).alias("is_lead_time_constrained"),
            (pl.col("expected_completion_date") - pl.col("target_month_eom"))
            .dt.total_days()
            .alias("days_over_month_end"),
        ]
    )

    return lots_with_dates.select(
        [
            "simulation_id",
            "model_id",
            "target_month",
            "lot_id",
            "target_units",
            "lead_time_days",
            "start_date",
            "expected_completion_date",
            "target_month_eom",
            "is_lead_time_constrained",
            "days_over_month_end",
            "revenue_plan_id",
            "allocation_run_id",
            "allocation_run_ts",
        ]
    )


def calculate_model_month_shortfalls(
    allocation_df: pl.DataFrame,
    failed_df: pl.DataFrame,
    demand_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate shortfall at model-month-simulation level.

    Key logic:
    1. Join demand to simulations based on matching revenue_plan_id
    2. Count production by target_month (not by actual_completion_month)
       - Units produced early still count toward the target_month they were planned for
    3. Include failed allocations in the shortfall calculation
    """
    # Get the revenue_plan_id for each simulation from allocation data
    simulation_revenue_plan = allocation_df.select(["simulation_id", "revenue_plan_id"]).unique()

    # Also get simulation info from failed allocations (in case some simulations only have failures)
    failed_simulation_revenue_plan = failed_df.select(["simulation_id", "revenue_plan_id"]).unique()

    # Combine both sources
    all_simulation_revenue_plan = pl.concat([simulation_revenue_plan, failed_simulation_revenue_plan]).unique()

    print(f"    Found {all_simulation_revenue_plan.height} simulation-revenue_plan combinations")

    # Production aggregated by target_month (regardless of when it actually completed)
    # This correctly attributes early production to the demand month it was meant to fulfill
    production_by_target = (
        allocation_df.filter(pl.col("units_produced") > 0)
        .group_by(["simulation_id", "model_id", "target_month", "revenue_plan_id"])
        .agg(pl.col("units_produced").sum().alias("produced_units"))
    )

    # Prepare demand with revenue_plan_id preserved
    demand_renamed = demand_df.select(
        [
            pl.col("revenue_plan_id"),
            pl.col("model_id"),
            pl.col("plan_month").alias("target_month"),
            pl.col("net_production_demand_ea").alias("demand_qty_units"),
            pl.col("sales_team").alias("demand_sales_team"),
            pl.col("grouping_model").alias("demand_grouping_model"),
        ]
    )

    # Join demand with simulations ON revenue_plan_id (not cross-join!)
    # This ensures we only create shortfall records for demand that actually exists
    # for the simulation's revenue plan
    demand_with_sim = demand_renamed.join(all_simulation_revenue_plan, on="revenue_plan_id", how="inner")

    print(f"    Demand records after joining with simulations: {demand_with_sim.height}")

    # Calculate shortfall
    shortfall_df = (
        demand_with_sim.join(
            production_by_target,
            on=["simulation_id", "model_id", "target_month", "revenue_plan_id"],
            how="left",
        )
        .with_columns([pl.col("produced_units").fill_null(0).alias("produced_on_time_units")])
        .with_columns([(pl.col("demand_qty_units") - pl.col("produced_on_time_units")).alias("shortfall_qty_units")])
        .filter(pl.col("shortfall_qty_units") > 0)
    )

    return shortfall_df


def determine_primary_reason(
    has_capacity_issue: bool,
    has_blocked_issue: bool,
    has_no_equipment: bool,
    lead_time_lot_count: int,
) -> Optional[str]:
    """Determine the primary reason for shortfall based on flags."""

    # Priority 1: No valid equipment (most severe - nothing can be done)
    if has_no_equipment:
        return "NO_VALID_EQUIPMENT"

    # Priority 2: Equipment blocked (equipment exists but blocked for this model)
    if has_blocked_issue and not has_capacity_issue:
        return "EQUIPMENT_BLOCKED"

    # Priority 3: Insufficient capacity
    if has_capacity_issue:
        return "INSUFFICIENT_CAPACITY"

    # Priority 4: Lead time constraints only
    if lead_time_lot_count > 0:
        return "LEAD_TIME_AND_ET_JIG_CAPACITY"

    return None


# =============================================================================
# Main Analysis Function
# =============================================================================


def analyze_shortfalls(
    allocation_df: pl.DataFrame,
    failed_df: pl.DataFrame,
    shortfall_df: pl.DataFrame,
    new_lots_df: pl.DataFrame,
    priorities_df: pl.DataFrame,
    waiting_periods_df: pl.DataFrame,
) -> List[dict]:
    """
    Analyze shortfalls using failed_allocations, waiting_periods, and lead time data.
    """
    print("  Pre-processing lead time constrained lots...")
    lead_time_preprocessed = preprocess_lead_time_constrained_lots(new_lots_df)

    # Build lead time lookup by (simulation_id, model_id, target_month)
    lead_time_lookup: Dict[Tuple[str, str, str], List[LeadTimeConstraintInfo]] = {}
    lt_constrained = lead_time_preprocessed.filter(pl.col("is_lead_time_constrained") == True)
    for row in lt_constrained.iter_rows(named=True):
        key = (row["simulation_id"], row["model_id"], row["target_month"])
        if key not in lead_time_lookup:
            lead_time_lookup[key] = []
        lead_time_lookup[key].append(
            LeadTimeConstraintInfo(
                lot_id=row["lot_id"],
                units=row["target_units"] or 0,
                start_date=row["start_date"],
                expected_completion_date=row["expected_completion_date"],
                lead_time_days=row["lead_time_days"] or 0,
                target_month_end=row["target_month_eom"],
                days_over=row["days_over_month_end"] or 0,
            )
        )

    # Build failed allocations lookup by (simulation_id, model_id, target_month)
    print("  Building failed allocations lookup...")
    failed_lookup: Dict[Tuple[str, str, str], List[FailedAllocationInfo]] = {}
    for row in failed_df.iter_rows(named=True):
        sim_id = row.get("simulation_id")
        model_id = row.get("model_id")
        target_month = row.get("target_month")
        if not sim_id or not model_id or not target_month:
            continue

        key = (sim_id, model_id, target_month)
        if key not in failed_lookup:
            failed_lookup[key] = []

        failed_lookup[key].append(
            FailedAllocationInfo(
                lot_id=row.get("lot_id"),
                model_id=model_id,
                target_month=target_month,
                process_id=row.get("process_id"),
                equipment_group=row.get("equipment_group"),
                units_in_lot=row.get("units_in_lot") or 0,
                failure_reason=row.get("failure_reason"),
                failure_status=row.get("failure_status"),
                delay_details=row.get("delay_details"),
            )
        )

    # Build waiting periods lookup by (simulation_id, model_id, target_month)
    print("  Building waiting periods lookup...")
    waiting_lookup: Dict[Tuple[str, str, str], List[WaitingPeriodInfo]] = {}

    for row in waiting_periods_df.iter_rows(named=True):
        sim_id = row.get("simulation_id")
        model_id = row.get("model_id")
        target_month = row.get("target_month")
        if not sim_id or not model_id or not target_month:
            continue

        key = (sim_id, model_id, target_month)
        if key not in waiting_lookup:
            waiting_lookup[key] = []

        # Convert dates
        waiting_start = convert_epoch_to_date(row.get("waiting_start_date"))
        waiting_end = convert_epoch_to_date(row.get("waiting_end_date"))

        waiting_lookup[key].append(
            WaitingPeriodInfo(
                lot_id=row.get("lot_id"),
                model_id=model_id,
                target_month=target_month,
                process_id=row.get("process_id"),
                equipment_group=row.get("equipment_group"),
                waiting_start_date=waiting_start,
                waiting_end_date=waiting_end,
                waiting_days=row.get("waiting_days") or 0,
                is_failed_allocation=row.get("is_failed_allocation") or False,
                failure_status=row.get("failure_status"),
                failure_detail=row.get("failure_detail"),
                blocked_equipment_list=row.get("blocked_equipment_list"),
                blocked_equipment_count=row.get("blocked_equipment_count") or 0,
                delay_reasons_summary=row.get("delay_reasons_summary"),
                units_affected=row.get("units_affected") or 0,
                margin_at_risk=row.get("margin_at_risk") or 0.0,
            )
        )

    # Build metadata lookups
    print("  Building metadata lookups...")

    # Get unique revenue_plan_ids from shortfall to filter priorities
    shortfall_revenue_plans = shortfall_df.select("revenue_plan_id").unique().to_series().to_list()

    # Priorities for financial data - filter to relevant revenue plans
    priorities_for_join = (
        priorities_df.filter(pl.col("revenue_plan_id").is_in(shortfall_revenue_plans))
        .group_by(["simulation_id", "model_id", "month", "revenue_plan_id"])
        .agg(
            [
                pl.col("amount_per_unit").first().alias("amount_per_unit"),
                pl.col("margin_amount_per_unit").first().alias("margin_amount_per_unit"),
                pl.col("simulation_name").first().alias("simulation_name"),
            ]
        )
        .rename({"month": "target_month"})
    )

    # Model metadata from allocation
    allocation_meta = (
        allocation_df.select(
            [
                "model_id",
                "simulation_id",
                "revenue_plan_id",
                "model_customer_name",
                "model_end_customer",
                "model_sales_team",
                "grouping_model",
                "allocation_run_id",
                "allocation_run_ts",
            ]
        )
        .filter(pl.col("model_customer_name").is_not_null())
        .unique(subset=["model_id", "simulation_id", "revenue_plan_id"])
    )

    # Join shortfall with metadata
    shortfall_enriched = (
        shortfall_df.join(
            priorities_for_join, on=["simulation_id", "model_id", "target_month", "revenue_plan_id"], how="left"
        )
        .join(allocation_meta, on=["model_id", "simulation_id", "revenue_plan_id"], how="left")
        .with_columns(
            [
                pl.col("amount_per_unit").fill_null(0.0),
                pl.col("margin_amount_per_unit").fill_null(0.0),
                (pl.col("shortfall_qty_units") * pl.col("amount_per_unit")).alias("shortfall_revenue_impact"),
                (pl.col("shortfall_qty_units") * pl.col("margin_amount_per_unit")).alias("shortfall_margin_impact"),
                (pl.col("shortfall_qty_units") / pl.col("demand_qty_units") * 100).alias("shortfall_pct"),
                pl.coalesce(["model_sales_team", "demand_sales_team"]).alias("final_sales_team"),
                pl.coalesce(["grouping_model", "demand_grouping_model"]).alias("final_grouping_model"),
            ]
        )
    )

    print(f"  Processing {shortfall_enriched.height:,} model-month shortfall records...")

    results = []

    for row in shortfall_enriched.iter_rows(named=True):
        simulation_id = row["simulation_id"]
        model_id = row["model_id"]
        target_month = row["target_month"]
        revenue_plan_id = row["revenue_plan_id"]

        key = (simulation_id, model_id, target_month)

        # Get data for this model-month
        lead_time_lots = lead_time_lookup.get(key, [])
        failed_allocations = failed_lookup.get(key, [])
        waiting_periods = waiting_lookup.get(key, [])

        target_month_end = parse_month_to_eom_date(target_month)

        # Determine flags from failed allocations and waiting periods
        has_no_equipment = any(
            fa.failure_status and "NO_EQUIPMENT" in fa.failure_status.upper() for fa in failed_allocations
        )
        has_blocked_issue = any(wp.blocked_equipment_count > 0 for wp in waiting_periods) or any(
            fa.failure_status and "BLOCKED" in fa.failure_status.upper() for fa in failed_allocations
        )
        has_capacity_issue = any(
            (wp.delay_reasons_summary and "full" in wp.delay_reasons_summary.lower())
            or (wp.failure_status and "INSUFFICIENT" in wp.failure_status.upper())
            for wp in waiting_periods
        ) or any(fa.failure_status and "INSUFFICIENT" in fa.failure_status.upper() for fa in failed_allocations)

        # Determine primary reason
        primary_reason = determine_primary_reason(
            has_capacity_issue=has_capacity_issue,
            has_blocked_issue=has_blocked_issue,
            has_no_equipment=has_no_equipment,
            lead_time_lot_count=len(lead_time_lots),
        )

        # Skip if no attributable reason
        if primary_reason is None:
            continue

        # Build rich reason detail
        reason_detail = build_rich_reason_detail(
            primary_reason=primary_reason,
            waiting_periods=waiting_periods,
            failed_allocations=failed_allocations,
            lead_time_lots=lead_time_lots,
            target_month=target_month,
            target_month_end=target_month_end,
        )

        # Build title
        title = build_title(
            model_id=model_id,
            target_month=target_month,
            shortfall_units=row["shortfall_qty_units"],
            primary_reason=primary_reason,
            customer_name=row.get("model_customer_name"),
        )

        # Collect equipment IDs from waiting periods
        bottleneck_ids = set()
        blocked_ids = set()
        equipment_groups = set()

        for wp in waiting_periods:
            if wp.delay_reasons_summary:
                for ci in parse_equipment_capacity_from_summary(wp.delay_reasons_summary):
                    if ci.event_type == "FULL":
                        bottleneck_ids.add(ci.equipment_id)
                    elif ci.event_type == "BLOCKED":
                        blocked_ids.add(ci.equipment_id)
            if wp.equipment_group:
                equipment_groups.add(wp.equipment_group)

        # Also get equipment groups from failed allocations
        for fa in failed_allocations:
            if fa.equipment_group:
                equipment_groups.add(fa.equipment_group)

        # Get delay dates
        all_dates = []
        for wp in waiting_periods:
            if wp.waiting_start_date:
                all_dates.append(wp.waiting_start_date)
            if wp.waiting_end_date:
                all_dates.append(wp.waiting_end_date)

        earliest_date = min(all_dates) if all_dates else None
        latest_date = max(all_dates) if all_dates else None

        # Add lead time dates
        if lead_time_lots:
            lt_dates = [lt.expected_completion_date for lt in lead_time_lots]
            if earliest_date is None:
                earliest_date = min(lt_dates)
            if latest_date is None:
                latest_date = max(lt_dates)
            else:
                latest_date = max(latest_date, max(lt_dates))

        # Get blocking process info from failed allocations
        blocking_process_id = None
        blocking_equipment_group = None
        for fa in failed_allocations:
            if fa.failure_status and "NO_EQUIPMENT" in fa.failure_status.upper():
                blocking_process_id = fa.process_id
                blocking_equipment_group = fa.equipment_group
                break

        # Calculate statistics
        delayed_lots = [wp for wp in waiting_periods if not wp.is_failed_allocation]
        failed_lot_count = len(failed_allocations)
        failed_units = sum(fa.units_in_lot for fa in failed_allocations)

        # Earliest possible completion for lead time lots
        earliest_possible_completion = None
        if lead_time_lots:
            earliest_possible_completion = min(lt.expected_completion_date for lt in lead_time_lots)

        results.append(
            {
                "demand_shortfall_id": f"{simulation_id}_{model_id}_{target_month}",
                "title": title,
                "simulation_id": simulation_id,
                "simulation_name": row.get("simulation_name"),
                "revenue_plan_id": revenue_plan_id,
                "model_id": model_id,
                "grouping_model": row.get("final_grouping_model"),
                "model_sales_team": row.get("final_sales_team"),
                "model_customer_name": row.get("model_customer_name"),
                "model_end_customer": row.get("model_end_customer"),
                "model_priority": None,
                "target_month": target_month,
                "demand_qty_units": row["demand_qty_units"],
                "produced_on_time_units": row["produced_on_time_units"],
                "shortfall_qty_units": row["shortfall_qty_units"],
                "shortfall_pct": row.get("shortfall_pct"),
                "shortfall_revenue_impact": row.get("shortfall_revenue_impact"),
                "shortfall_margin_impact": row.get("shortfall_margin_impact"),
                "delayed_lot_count": len(delayed_lots),
                "failed_lot_count": failed_lot_count,
                "delayed_units": sum(wp.units_affected for wp in delayed_lots),
                "failed_units": failed_units,
                "lead_time_constrained_lot_count": len(lead_time_lots),
                "lead_time_constrained_units": sum(lt.units for lt in lead_time_lots),
                "primary_reason": primary_reason,
                "reason_detail": reason_detail,
                "blocking_process_id": blocking_process_id,
                "blocking_process_equipment_group": blocking_equipment_group,
                "bottleneck_equipment_ids": sorted(bottleneck_ids) if bottleneck_ids else None,
                "bottleneck_equipment_group_ids": sorted(equipment_groups) if equipment_groups else None,
                "blocked_equipment_ids": sorted(blocked_ids) if blocked_ids else None,
                "blocked_equipment_count": len(blocked_ids),
                "earliest_delay_date": earliest_date,
                "latest_delay_date": latest_date,
                "target_month_end_date": target_month_end,
                "earliest_possible_completion": earliest_possible_completion,
                "shortage_ids": None,
                "allocation_run_id": row.get("allocation_run_id"),
                "allocation_run_ts": row.get("allocation_run_ts"),
            }
        )

    return results


# =============================================================================
# Transform Definition
# =============================================================================


@lightweight(cpu_cores=2, memory_gb=16)
@transform(
    demand_shortfall_output=Output("ri.foundry.main.dataset.a715dabf-cf86-40be-83b0-4dbbf44aa169"),
    allocation_output=Input("ri.foundry.main.dataset.b84b2bb5-1cbf-4970-8f73-0e1eb9d4148f"),
    failed_allocations=Input("ri.foundry.main.dataset.96248695-b74a-4d42-b5e9-30a48bc8ff5d"),
    new_lots_created=Input("ri.foundry.main.dataset.b2420a83-e21e-45e4-a412-98ffb3451381"),
    net_demand=Input("ri.foundry.main.dataset.122a25b5-165e-4eba-adfa-e7865add43d1"),
    model_priorities=Input("ri.foundry.main.dataset.19b16719-7cad-410d-b77d-cb03409dad14"),
    equipment_shortages=Input("ri.foundry.main.dataset.6f0ce94e-0072-4ec1-8e3e-3188844b9b52"),
    lot_waiting_periods=Input("ri.foundry.main.dataset.332070f1-98dc-442a-a48f-0a617d61be5b"),
)
def compute(
    allocation_output,
    failed_allocations,
    new_lots_created,
    net_demand,
    model_priorities,
    equipment_shortages,
    lot_waiting_periods,
    demand_shortfall_output,
) -> None:
    """
    Analyze demand shortfalls across all simulations.

    Key logic:
    1. Joins demand to simulations based on matching revenue_plan_id (not cross-join)
    2. Counts production by target_month - early production counts toward demand
    3. Uses failed_allocations and lot_waiting_periods for rich delay information

    Produces one row per model-month-simulation with shortfall,
    ONLY when we can attribute the shortfall to a capacity issue.

    Shortfall Reasons:
    - INSUFFICIENT_CAPACITY: Equipment at capacity
    - EQUIPMENT_BLOCKED: Equipment had capacity but was blocked for model
    - NO_VALID_EQUIPMENT: No equipment available for process step
    - LEAD_TIME_AND_ET_JIG_CAPACITY: Lead time too long to complete within target month
    """
    print("=" * 60)
    print("DEMAND SHORTFALL ANALYSIS (Model-Month Level)")
    print("=" * 60)

    # Load data
    allocation_df = allocation_output.polars()
    failed_df = failed_allocations.polars()
    new_lots_df = new_lots_created.polars()
    demand_df = net_demand.polars()
    priorities_df = model_priorities.polars().filter(pl.col("__is_deleted") == False)
    waiting_periods_df = lot_waiting_periods.polars()

    print(f"\nInput data loaded:")
    print(f"  Allocation records: {allocation_df.height:,}")
    print(f"  Failed allocations: {failed_df.height:,}")
    print(f"  New lots created: {new_lots_df.height:,}")
    print(f"  Demand records: {demand_df.height:,}")
    print(f"  Model priorities: {priorities_df.height:,}")
    print(f"  Lot waiting periods: {waiting_periods_df.height:,}")

    # Show revenue plan distribution
    print("\n--- Revenue Plans in Data ---")
    alloc_plans = allocation_df.select("revenue_plan_id").unique().to_series().to_list()
    demand_plans = demand_df.select("revenue_plan_id").unique().to_series().to_list()
    print(f"  Allocation revenue plans: {alloc_plans}")
    print(f"  Demand revenue plans (sample): {demand_plans[:5]}...")

    # Calculate model-month shortfalls
    print("\n--- Calculating Model-Month Shortfalls ---")
    shortfall_df = calculate_model_month_shortfalls(allocation_df, failed_df, demand_df)
    print(f"  Model-months with shortfalls: {shortfall_df.height:,}")

    if shortfall_df.height > 0:
        total_shortfall = shortfall_df.select(pl.col("shortfall_qty_units").sum()).item()
        print(f"  Total shortfall units: {total_shortfall:,}")

        # Show sample shortfall records for debugging
        print("\n  Sample shortfall records:")
        sample = shortfall_df.head(5)
        for row in sample.iter_rows(named=True):
            print(
                f"    {row['model_id'][:20]:<20} {row['target_month']} "
                f"demand={row['demand_qty_units']:>10,} "
                f"produced={row['produced_on_time_units']:>10,} "
                f"shortfall={row['shortfall_qty_units']:>10,}"
            )

    # Analyze shortfalls
    print("\n--- Analyzing Shortfalls ---")
    results = analyze_shortfalls(
        allocation_df,
        failed_df,
        shortfall_df,
        new_lots_df,
        priorities_df,
        waiting_periods_df,
    )
    print(f"  Model-month shortfall records with attributable capacity issues: {len(results):,}")

    if not results:
        print("\n⚠️ No shortfall records with attributable capacity issues")
        result_df = pl.DataFrame(schema=DEMAND_SHORTFALL_SCHEMA)
    else:
        result_df = pl.DataFrame(results, schema=DEMAND_SHORTFALL_SCHEMA)

    # Print summary by reason
    if result_df.height > 0:
        print("\n--- Shortfall Summary by Reason ---")
        reason_summary = (
            result_df.group_by("primary_reason")
            .agg(
                [
                    pl.count().alias("model_month_count"),
                    pl.col("shortfall_qty_units").sum().alias("total_units"),
                    pl.col("shortfall_margin_impact").sum().alias("total_margin_impact"),
                ]
            )
            .sort("total_margin_impact", descending=True)
        )

        print(f"  {'Reason':<30} {'Model-Months':<15} {'Units':<15} {'Margin Impact':<20}")
        print("  " + "-" * 80)
        for row in reason_summary.iter_rows(named=True):
            margin = row["total_margin_impact"] or 0
            print(
                f"  {row['primary_reason']:<30} "
                f"{row['model_month_count']:<15,} "
                f"{row['total_units']:<15,} "
                f"${margin:>18,.0f}"
            )

        # Print top shortfalls
        print("\n--- Top 10 Model-Month Shortfalls (by margin impact) ---")
        top_shortfalls = result_df.sort("shortfall_margin_impact", descending=True).head(10)

        print(f"  {'Model':<20} {'Month':<10} {'Units':<12} {'Reason':<30} {'Margin Impact':<15}")
        print("  " + "-" * 87)
        for row in top_shortfalls.iter_rows(named=True):
            margin = row["shortfall_margin_impact"] or 0
            print(
                f"  {row['model_id'][:20]:<20} "
                f"{row['target_month']:<10} "
                f"{row['shortfall_qty_units']:<12,} "
                f"{row['primary_reason'][:30]:<30} "
                f"${margin:>13,.0f}"
            )

        # Print sample of rich detail
        print("\n--- Sample Reason Details ---")
        for row in top_shortfalls.head(2).iter_rows(named=True):
            print(f"\n{'=' * 60}")
            print(f"Title: {row['title']}")
            print(f"{'=' * 60}")
            print(row["reason_detail"])

    # Write output
    print(f"\n--- Writing Output ---")
    demand_shortfall_output.write_table(result_df.unique(subset=["demand_shortfall_id"]))
    print(f"✓ Wrote {result_df.height:,} demand shortfall records")

    print("\n" + "=" * 60)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 60)
