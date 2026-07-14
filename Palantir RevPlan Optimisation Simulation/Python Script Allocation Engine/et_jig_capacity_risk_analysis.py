"""
ET JIG Capacity Risk Analysis

Analyzes the risk of not meeting demand due to ET JIG capacity constraints.
For each model-month, calculates:
1. How many lots are at risk of missing their target month due to lead time
2. How many additional ET JIGs would be needed to start lots earlier and avoid delays
3. Financial impact of the capacity shortfall

The analysis works by:
- Identifying lots that cannot complete within their target month given their start date + lead time
- Calculating how much earlier lots would need to start to complete on time
- Determining the ET JIG capacity needed to enable those earlier starts (minimum JIGs, as early as possible)
- Comparing against current capacity to identify the gap
- Ensuring JIG procurement dates are not in the past
"""

import polars as pl
from transforms.api import transform, Input, Output, lightweight
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


# =============================================================================
# Output Schema Definition
# =============================================================================

ET_JIG_CAPACITY_RISK_SCHEMA = {
    # === Risk Context (model-month level) ===
    "et_jig_risk_id": pl.Utf8,  # {simulation_id}_{model_id}_{target_month}
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
    "target_month_end_date": pl.Date,
    # === Demand and Production ===
    "demand_qty_units": pl.Int64,  # Total demand for this model-month
    "total_lots_created": pl.Int32,  # Total number of lots created
    "total_units_in_lots": pl.Int64,  # Total units across all lots
    # === Demand and Production ===
    "demand_qty_units": pl.Int64,  # Total demand for this model-month
    "total_lots_created": pl.Int32,  # Total number of lots created
    "total_units_in_lots": pl.Int64,  # Total units across all lots
    # === Risk Metrics ===
    "lots_at_risk": pl.Int32,  # Number of lots that will miss target month
    "units_at_risk": pl.Int64,  # Units in lots that will miss target month
    "risk_percentage": pl.Float64,  # Percentage of demand at risk
    "average_days_late": pl.Float64,  # Average days past target month end
    "max_days_late": pl.Int32,  # Maximum days past target month end
    # === Current ET JIG Capacity ===
    "current_et_jig_count": pl.Int32,  # Current number of ET JIG units for this model
    "current_daily_lot_capacity": pl.Float64,  # Current daily lot production capacity
    "current_monthly_lot_capacity": pl.Float64,  # Current monthly lot capacity (30 days)
    # === Required Capacity Analysis ===
    "required_start_date": pl.Date,  # When lots need to start to complete on time
    "actual_earliest_start": pl.Date,  # When lots are actually starting
    "start_date_gap_days": pl.Int32,  # Days between required and actual start
    "additional_et_jigs_needed": pl.Int32,  # Additional ET JIGs needed to meet demand
    "total_et_jigs_needed": pl.Int32,  # Total ET JIGs needed (current + additional)
    "capacity_increase_pct": pl.Float64,  # Percentage increase in capacity needed
    "earliest_jig_needed_date": pl.Date,  # Earliest date by which additional JIGs are needed
    "jig_procurement_warning": pl.Utf8,  # Warning if JIG needed in the past
    # === Financial Impact ===
    "revenue_at_risk": pl.Float64,  # Revenue impact from units at risk
    "margin_at_risk": pl.Float64,  # Margin impact from units at risk
    "amount_per_unit": pl.Float64,  # Revenue per unit
    "margin_per_unit": pl.Float64,  # Margin per unit
    # === Lot Details ===
    "earliest_lot_start": pl.Date,  # Earliest lot start date
    "latest_lot_start": pl.Date,  # Latest lot start date
    # === Risk Categorization ===
    "risk_level": pl.Utf8,  # HIGH, MEDIUM, LOW, NO_RISK
    "risk_reason": pl.Utf8,  # Detailed explanation of the risk
    "mitigation_strategy": pl.Utf8,  # Suggested mitigation approach
    # === Staggering Analysis ===
    "optimal_stagger_days": pl.Int32,  # Optimal days to stagger lot starts
    "lots_per_stagger_group": pl.Int32,  # Number of lots per stagger group
    "stagger_groups_needed": pl.Int32,  # Number of stagger groups needed
    # === Links to Related Data ===
    "at_risk_lot_ids": pl.List(pl.Utf8),  # List of lot IDs at risk
    "related_shortage_ids": pl.List(pl.Utf8),  # Related shortage IDs if any
    # === Run Metadata ===
    "allocation_run_id": pl.Utf8,
    "allocation_run_ts": pl.Datetime,
    "analysis_run_date": pl.Date,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ETJigCapacityInfo:
    """Information about ET JIG capacity for a model."""

    model_id: str
    total_jig_units: int  # Total JIG대수 (number of physical JIG units)
    daily_lot_capacity: float  # Total Capa_Lot per day
    daily_sheet_capacity: float  # Total Capa_Sheet per day
    lots_per_jig: float  # Average Capa_Lot per JIG


@dataclass
class JigRequirement:
    """Calculated JIG requirement to meet demand."""

    additional_jigs_needed: int
    earliest_date_needed: Optional[date]
    procurement_warning: str
    total_jigs_needed: int
    capacity_increase_pct: float


# =============================================================================
# Helper Functions
# =============================================================================


def calculate_jig_requirements(
    total_lots: int,
    current_daily_capacity: float,
    lots_per_jig: float,
    days_available: int,
    target_month_start: date,
    analysis_run_date: date,
) -> JigRequirement:
    """
    Calculate minimum JIGs needed and earliest day needed.
    Strategy: Minimize JIGs by getting them as early as possible.

    Args:
        total_lots: Number of lots that need to be started (typically lots_at_risk, not total)
        current_daily_capacity: Current daily lot production capacity (lots/day)
        lots_per_jig: Lot capacity per JIG (lots/day per JIG)
        days_available: Days available to start lots (days until target - lead time)
        target_month_start: Start date of the target month
        analysis_run_date: Current date (cannot order JIGs in the past)

    Returns:
        JigRequirement with additional JIGs needed, earliest date, and warnings
    """
    # Ensure minimum values to avoid division by zero or negative days
    days_available = max(1, days_available)
    lots_per_jig = max(1, lots_per_jig)
    current_daily_capacity = max(1, current_daily_capacity)

    # How many lots can current capacity produce?
    lots_with_current = current_daily_capacity * days_available

    if lots_with_current >= total_lots:
        # No additional JIGs needed
        return JigRequirement(
            additional_jigs_needed=0,
            earliest_date_needed=None,
            procurement_warning="",
            total_jigs_needed=int(current_daily_capacity / lots_per_jig) if lots_per_jig > 0 else 1,
            capacity_increase_pct=0.0,
        )

    lots_deficit = total_lots - lots_with_current

    # Minimum additional JIGs needed (assuming we get them on day 1 of the target month)
    # This gives us maximum time to use them
    additional_jigs = math.ceil(lots_deficit / (lots_per_jig * days_available))

    # Calculate earliest day needed for this many JIGs
    # Work backwards: how many days do we need the extra capacity?
    extra_daily_capacity = additional_jigs * lots_per_jig
    days_needed_with_extra = math.ceil(lots_deficit / extra_daily_capacity)

    # Calculate the day number (1-indexed from target_month_start)
    earliest_day_number = days_available - days_needed_with_extra + 1

    # Convert to actual date
    earliest_date_needed = target_month_start + timedelta(days=max(0, earliest_day_number - 1))

    # Check if this date is in the past
    procurement_warning = ""
    if earliest_date_needed < analysis_run_date:
        days_overdue = (analysis_run_date - earliest_date_needed).days
        procurement_warning = f"⚠️ JIG needed {days_overdue} days ago (on {earliest_date_needed.strftime('%Y-%m-%d')}). Risk already materialized."
        # Set to today since we can't order in the past
        earliest_date_needed = analysis_run_date

    # Calculate total JIGs and increase percentage
    current_jigs = int(current_daily_capacity / lots_per_jig) if lots_per_jig > 0 else 1
    total_jigs = current_jigs + additional_jigs
    capacity_increase_pct = (additional_jigs / current_jigs * 100) if current_jigs > 0 else 100.0

    return JigRequirement(
        additional_jigs_needed=additional_jigs,
        earliest_date_needed=earliest_date_needed,
        procurement_warning=procurement_warning,
        total_jigs_needed=total_jigs,
        capacity_increase_pct=capacity_increase_pct,
    )


def parse_month_to_dates(month_str: str) -> Tuple[date, date]:
    """Parse month string to start and end of month dates."""
    year = int(month_str[:4])
    month = int(month_str[4:6])

    start_date = date(year, month, 1)

    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    return start_date, end_date


def format_month_display(month_str: str) -> str:
    """Format month string for display."""
    if not month_str or len(month_str) < 6:
        return month_str
    year = month_str[:4]
    month_num = int(month_str[4:6])
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{month_names[month_num - 1]} {year}"


def determine_risk_level(risk_pct: float, days_late: float, additional_jigs: int) -> str:
    """Determine risk level based on metrics."""
    if risk_pct == 0:
        return "NO_RISK"
    elif risk_pct > 50 or days_late > 15 or additional_jigs > 5:
        return "HIGH"
    elif risk_pct > 20 or days_late > 7 or additional_jigs > 2:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_stagger_plan(
    total_lots: int, lead_time_days: int, target_month_end: date, daily_capacity: float
) -> Tuple[int, int, int]:
    """
    Calculate optimal staggering plan for lot starts.

    Returns:
        - optimal_stagger_days: Days to stagger between groups
        - lots_per_group: Number of lots per stagger group
        - groups_needed: Number of stagger groups
    """
    if total_lots <= 1 or daily_capacity <= 0:
        return 0, total_lots, 1

    # Calculate how many days we have to work with
    days_available = 30  # Assume a month

    # Calculate how many lots can be processed per day
    lots_per_day = max(1, int(daily_capacity))

    # Calculate groups needed
    groups_needed = math.ceil(total_lots / lots_per_day)
    lots_per_group = math.ceil(total_lots / groups_needed)

    # Calculate stagger days
    if groups_needed > 1:
        optimal_stagger_days = max(1, days_available // groups_needed)
    else:
        optimal_stagger_days = 0

    return optimal_stagger_days, lots_per_group, groups_needed


def build_risk_reason(
    lots_at_risk: int,
    units_at_risk: int,
    avg_days_late: float,
    jig_req: JigRequirement,
) -> str:
    """Build detailed risk reason explanation."""
    if lots_at_risk == 0:
        return "No risk - all lots can complete within target month with current capacity"

    parts = []

    # Add procurement warning first if it exists
    if jig_req.procurement_warning:
        parts.append(jig_req.procurement_warning)

    parts.append(f"{lots_at_risk:,} lots ({units_at_risk:,} units) at risk of missing target month")

    if avg_days_late > 0:
        parts.append(f"Average delay: {avg_days_late:.1f} days past month end")

    if jig_req.additional_jigs_needed > 0:
        if jig_req.earliest_date_needed:
            parts.append(
                f"Need {jig_req.additional_jigs_needed} additional ET JIGs "
                f"({jig_req.capacity_increase_pct:.0f}% increase) "
                f"by {jig_req.earliest_date_needed.strftime('%Y-%m-%d')}"
            )
        else:
            parts.append(
                f"Need {jig_req.additional_jigs_needed} additional ET JIGs "
                f"({jig_req.capacity_increase_pct:.0f}% increase) to enable earlier starts"
            )

    return " • ".join(parts)


def build_mitigation_strategy(
    risk_level: str,
    jig_req: JigRequirement,
    stagger_groups: int,
    avg_days_late: float,
) -> str:
    """Build mitigation strategy recommendation."""
    if risk_level == "NO_RISK":
        return "No mitigation needed - current capacity sufficient"

    strategies = []

    if jig_req.additional_jigs_needed > 0:
        if jig_req.earliest_date_needed:
            strategies.append(
                f"Order {jig_req.additional_jigs_needed} ET JIGs immediately "
                f"(needed by {jig_req.earliest_date_needed.strftime('%Y-%m-%d')})"
            )
        else:
            strategies.append(f"Add {jig_req.additional_jigs_needed} ET JIGs to increase capacity")

    if stagger_groups > 1:
        strategies.append(f"Stagger lot starts across {stagger_groups} groups")

    if avg_days_late > 10:
        strategies.append("Consider expediting critical lots or adjusting target dates")
    elif avg_days_late > 5:
        strategies.append("Optimize scheduling to minimize delays")

    if not strategies:
        strategies.append("Monitor closely and adjust scheduling as needed")

    return " | ".join(strategies)


# =============================================================================
# Analysis Functions
# =============================================================================


def analyze_et_jig_capacity_risk(
    new_lots_df: pl.DataFrame,
    et_jig_df: pl.DataFrame,
    demand_df: pl.DataFrame,
    priorities_df: pl.DataFrame,
    allocation_df: pl.DataFrame,
    analysis_run_date: date,
) -> pl.DataFrame:
    """
    Analyze ET JIG capacity risk for each model-month.
    """

    # Prepare ET JIG capacity from master data
    jig_capacity = (
        et_jig_df.filter(pl.col("JIG_상태") == "정상")
        .rename({"대상_모델": "model_id"})
        .group_by("model_id")
        .agg(
            [
                pl.col("JIG대수").sum().alias("total_jig_units"),
                (pl.col("Capa_Lot") * pl.col("JIG대수")).sum().alias("daily_lot_capacity"),
                (pl.col("Capa_Sheet") * pl.col("JIG대수")).sum().alias("daily_sheet_capacity"),
                pl.col("Capa_Lot").mean().alias("lots_per_jig"),
            ]
        )
    )

    et_jig_lookup = {}
    for row in jig_capacity.iter_rows(named=True):
        model_id = row["model_id"]
        if model_id:
            et_jig_lookup[model_id] = ETJigCapacityInfo(
                model_id=model_id,
                total_jig_units=int(row.get("total_jig_units", 1)),
                daily_lot_capacity=row.get("daily_lot_capacity", 2.0),
                daily_sheet_capacity=row.get("daily_sheet_capacity", 10.0),
                lots_per_jig=row.get("lots_per_jig", 2.0),
            )

    # Process new lots to identify risks
    lots_processed = (
        new_lots_df.with_columns(
            [
                # Parse dates
                pl.col("target_month").str.slice(0, 4).cast(pl.Int32).alias("year"),
                pl.col("target_month").str.slice(4, 2).cast(pl.Int32).alias("month"),
            ]
        )
        .with_columns(
            [
                # Calculate month start and end dates
                pl.date(pl.col("year"), pl.col("month"), 1).alias("target_month_start"),
                pl.date(pl.col("year"), pl.col("month"), 1).dt.month_end().alias("target_month_end"),
                # Convert start date to date type if needed
                pl.when(pl.col("target_lot_start_date").is_not_null())
                .then(
                    # Try to cast to date, handling both Date and Datetime/timestamp inputs
                    pl.col("target_lot_start_date").cast(pl.Date, strict=False)
                )
                .otherwise(None)
                .alias("start_date"),
            ]
        )
        .with_columns(
            [
                # Calculate expected completion
                (pl.col("start_date") + pl.duration(days=pl.col("lead_time_days")))
                .dt.date()
                .alias("expected_completion"),
            ]
        )
        .with_columns(
            [
                # Calculate if at risk and by how much
                (pl.col("expected_completion") > pl.col("target_month_end")).alias("is_at_risk"),
                pl.when(pl.col("expected_completion") > pl.col("target_month_end"))
                .then((pl.col("expected_completion") - pl.col("target_month_end")).dt.total_days())
                .otherwise(0)
                .alias("days_late"),
            ]
        )
    )

    # Aggregate by model-month-simulation
    model_month_summary = (
        lots_processed.group_by(
            ["simulation_id", "model_id", "target_month", "revenue_plan_id", "allocation_run_id", "allocation_run_ts"]
        )
        .agg(
            [
                # Total lots and units
                pl.count().alias("total_lots_created"),
                pl.col("target_units").sum().alias("total_units_in_lots"),
                # At-risk metrics
                pl.col("is_at_risk").sum().alias("lots_at_risk"),
                pl.when(pl.col("is_at_risk")).then(pl.col("target_units")).otherwise(0).sum().alias("units_at_risk"),
                # Timing metrics
                pl.col("days_late").filter(pl.col("days_late") > 0).mean().alias("average_days_late"),
                pl.col("days_late").max().alias("max_days_late"),
                # Lead time for calculation
                pl.col("lead_time_days").mean().alias("average_lead_time_days"),
                # Date ranges
                pl.col("start_date").min().alias("earliest_lot_start"),
                pl.col("start_date").max().alias("latest_lot_start"),
                pl.col("target_month_start").first().alias("target_month_start_date"),
                pl.col("target_month_end").first().alias("target_month_end_date"),
                # Lot IDs at risk
                pl.col("lot_id").filter(pl.col("is_at_risk")).drop_nulls().implode().alias("at_risk_lot_ids"),
            ]
        )
        .with_columns(
            [
                # Calculate risk percentage
                (pl.col("units_at_risk") / pl.col("total_units_in_lots") * 100).fill_null(0).alias("risk_percentage"),
            ]
        )
    )

    # Get demand data for context
    demand_summary = (
        demand_df.rename({"plan_month": "target_month"})
        .group_by(["revenue_plan_id", "model_id", "target_month"])
        .agg(
            [
                pl.col("net_production_demand_ea").sum().alias("demand_qty_units"),
                pl.col("sales_team").first().alias("model_sales_team"),
                pl.col("grouping_model").first().alias("grouping_model"),
            ]
        )
    )

    # Get financial data from priorities
    priorities_summary = (
        priorities_df.filter(pl.col("__is_deleted") == False)
        .rename({"month": "target_month"})
        .group_by(["simulation_id", "model_id", "target_month", "revenue_plan_id"])
        .agg(
            [
                pl.col("amount_per_unit").mean().alias("amount_per_unit"),
                pl.col("margin_amount_per_unit").mean().alias("margin_per_unit"),
                pl.col("simulation_name").first().alias("simulation_name"),
                pl.col("priority").first().alias("model_priority"),
            ]
        )
    )

    # Get model metadata from allocation
    model_metadata = allocation_df.select(
        ["model_id", "simulation_id", "revenue_plan_id", "model_customer_name", "model_end_customer"]
    ).unique()

    # Join all data together
    analysis_df = (
        model_month_summary.join(demand_summary, on=["revenue_plan_id", "model_id", "target_month"], how="left")
        .join(priorities_summary, on=["simulation_id", "model_id", "target_month", "revenue_plan_id"], how="left")
        .join(model_metadata, on=["model_id", "simulation_id", "revenue_plan_id"], how="left")
        .with_columns(
            [
                # Fill nulls with defaults
                pl.col("demand_qty_units").fill_null(0),
                pl.col("amount_per_unit").fill_null(0),
                pl.col("margin_per_unit").fill_null(0),
                pl.col("average_days_late").fill_null(0),
                pl.col("max_days_late").fill_null(0),
            ]
        )
    )

    # Calculate ET JIG requirements
    results = []

    for row in analysis_df.iter_rows(named=True):
        model_id = row["model_id"]

        # Skip if no lots at risk - no actual shortfall
        if row["lots_at_risk"] == 0:
            continue

        # Get ET JIG capacity for this model
        et_jig_info = et_jig_lookup.get(
            model_id,
            ETJigCapacityInfo(
                model_id=model_id,
                total_jig_units=1,
                daily_lot_capacity=2.0,
                daily_sheet_capacity=10.0,
                lots_per_jig=2.0,
            ),
        )

        # Calculate days available to start lots
        avg_lead_time = row["average_lead_time_days"] or 30
        target_month_end = row["target_month_end_date"]
        target_month_start = row["target_month_start_date"]

        # Days available = days from month start to (month end - lead time)
        required_start_date = target_month_end - timedelta(days=int(avg_lead_time))

        if target_month_start and required_start_date >= target_month_start:
            days_available = (required_start_date - target_month_start).days + 1
        else:
            # If required start is before month start, we have issues - use full month
            days_available = 30

        # Calculate JIG requirements based on LOTS AT RISK, not total lots
        # The question is: how many additional JIGs do we need to process
        # the at-risk lots earlier so they complete on time?
        jig_req = calculate_jig_requirements(
            total_lots=row["lots_at_risk"],  # Only at-risk lots need earlier starts
            current_daily_capacity=et_jig_info.daily_lot_capacity,
            lots_per_jig=et_jig_info.lots_per_jig,
            days_available=days_available,
            target_month_start=target_month_start,
            analysis_run_date=analysis_run_date,
        )

        # Skip if no additional JIGs needed - no capacity shortfall
        if jig_req.additional_jigs_needed == 0:
            continue

        # Calculate start date gap
        actual_earliest = row["earliest_lot_start"]
        if actual_earliest and required_start_date:
            start_gap_days = (actual_earliest - required_start_date).days
        else:
            start_gap_days = 0

        # Calculate financial impact
        revenue_at_risk = row["units_at_risk"] * row["amount_per_unit"]
        margin_at_risk = row["units_at_risk"] * row["margin_per_unit"]

        # Determine risk level
        risk_level = determine_risk_level(
            row["risk_percentage"], row["average_days_late"], jig_req.additional_jigs_needed
        )

        # Calculate stagger plan
        optimal_stagger, lots_per_group, stagger_groups = calculate_stagger_plan(
            row["total_lots_created"],
            avg_lead_time,
            target_month_end,
            et_jig_info.daily_lot_capacity,
        )

        # Build risk reason
        risk_reason = build_risk_reason(
            row["lots_at_risk"],
            row["units_at_risk"],
            row["average_days_late"],
            jig_req,
        )

        # Build mitigation strategy
        mitigation = build_mitigation_strategy(risk_level, jig_req, stagger_groups, row["average_days_late"])

        # Build title
        month_display = format_month_display(row["target_month"])
        title = (
            f"ET JIG Risk: {model_id} - {row['units_at_risk']:,} units at risk in {month_display} "
            f"(need {jig_req.additional_jigs_needed} more JIGs)"
        )

        results.append(
            {
                "et_jig_risk_id": f"{row['simulation_id']}_{model_id}_{row['target_month']}",
                "title": title,
                "simulation_id": row["simulation_id"],
                "simulation_name": row.get("simulation_name"),
                "revenue_plan_id": row["revenue_plan_id"],
                "model_id": model_id,
                "grouping_model": row.get("grouping_model"),
                "model_sales_team": row.get("model_sales_team"),
                "model_customer_name": row.get("model_customer_name"),
                "model_end_customer": row.get("model_end_customer"),
                "model_priority": row.get("model_priority"),
                "target_month": row["target_month"],
                "target_month_end_date": target_month_end,
                "demand_qty_units": row["demand_qty_units"],
                "total_lots_created": row["total_lots_created"],
                "total_units_in_lots": row["total_units_in_lots"],
                "lots_at_risk": row["lots_at_risk"],
                "units_at_risk": row["units_at_risk"],
                "risk_percentage": row["risk_percentage"],
                "average_days_late": row["average_days_late"],
                "max_days_late": row["max_days_late"],
                "current_et_jig_count": et_jig_info.total_jig_units,
                "current_daily_lot_capacity": et_jig_info.daily_lot_capacity,
                "current_monthly_lot_capacity": et_jig_info.daily_lot_capacity * 30,
                "required_start_date": required_start_date,
                "actual_earliest_start": actual_earliest,
                "start_date_gap_days": start_gap_days,
                "additional_et_jigs_needed": jig_req.additional_jigs_needed,
                "total_et_jigs_needed": jig_req.total_jigs_needed,
                "capacity_increase_pct": jig_req.capacity_increase_pct,
                "earliest_jig_needed_date": jig_req.earliest_date_needed,
                "jig_procurement_warning": jig_req.procurement_warning,
                "revenue_at_risk": revenue_at_risk,
                "margin_at_risk": margin_at_risk,
                "amount_per_unit": row["amount_per_unit"],
                "margin_per_unit": row["margin_per_unit"],
                "earliest_lot_start": actual_earliest,
                "latest_lot_start": row["latest_lot_start"],
                "risk_level": risk_level,
                "risk_reason": risk_reason,
                "mitigation_strategy": mitigation,
                "optimal_stagger_days": optimal_stagger,
                "lots_per_stagger_group": lots_per_group,
                "stagger_groups_needed": stagger_groups,
                "at_risk_lot_ids": row["at_risk_lot_ids"],
                "related_shortage_ids": None,
                "allocation_run_id": row["allocation_run_id"],
                "allocation_run_ts": row["allocation_run_ts"],
                "analysis_run_date": analysis_run_date,
            }
        )

    if results:
        return pl.DataFrame(results, schema=ET_JIG_CAPACITY_RISK_SCHEMA)
    else:
        return pl.DataFrame(schema=ET_JIG_CAPACITY_RISK_SCHEMA)


# =============================================================================
# Transform Definition
# =============================================================================


@lightweight(cpu_cores=2, memory_gb=16)
@transform(
    et_jig_risk_output=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/datasets/allocation_engine/et_jig_capacity_risk_analysis"
    ),
    new_lots_created=Input("ri.foundry.main.dataset.b2420a83-e21e-45e4-a412-98ffb3451381"),
    et_jig_master=Input("ri.foundry.main.dataset.81475837-c270-4f18-abbe-bfc8d8e5f94d"),
    net_demand=Input("ri.foundry.main.dataset.122a25b5-165e-4eba-adfa-e7865add43d1"),
    model_priorities=Input("ri.foundry.main.dataset.19b16719-7cad-410d-b77d-cb03409dad14"),
    allocation_output=Input("ri.foundry.main.dataset.b84b2bb5-1cbf-4970-8f73-0e1eb9d4148f"),
)
def compute(
    new_lots_created,
    et_jig_master,
    net_demand,
    model_priorities,
    allocation_output,
    et_jig_risk_output,
) -> None:
    """
    Analyze ET JIG capacity risk for models by month.

    This transform:
    1. Identifies lots that will miss their target month due to lead time constraints
    2. Calculates minimum additional ET JIGs needed (with earliest procurement date)
    3. Provides financial impact and mitigation strategies
    4. Creates easily discoverable records per model per month
    """
    print("=" * 60)
    print("ET JIG CAPACITY RISK ANALYSIS")
    print("=" * 60)

    # Load data
    new_lots_df = new_lots_created.polars()
    et_jig_df = et_jig_master.polars()
    demand_df = net_demand.polars()
    priorities_df = model_priorities.polars()
    allocation_df = allocation_output.polars()

    print(f"\nInput data loaded:")
    print(f"  New lots created: {new_lots_df.height:,} records")
    print(f"  ET JIG capacity: {et_jig_df.height:,} models")
    print(f"  Demand records: {demand_df.height:,}")
    print(f"  Model priorities: {priorities_df.height:,}")
    print(f"  Allocation output: {allocation_df.height:,}")

    # Perform analysis
    print("\n--- Analyzing ET JIG Capacity Risk ---")
    analysis_date = date.today()
    print(f"  Analysis run date: {analysis_date}")

    result_df = analyze_et_jig_capacity_risk(
        new_lots_df,
        et_jig_df,
        demand_df,
        priorities_df,
        allocation_df,
        analysis_date,
    )

    print(f"  Generated {result_df.height:,} risk analysis records")

    # Print summary statistics
    if result_df.height > 0:
        # Risk level summary
        print("\n--- Risk Level Summary ---")
        risk_summary = (
            result_df.group_by("risk_level")
            .agg(
                [
                    pl.count().alias("model_months"),
                    pl.col("units_at_risk").sum().alias("total_units_at_risk"),
                    pl.col("margin_at_risk").sum().alias("total_margin_at_risk"),
                    pl.col("additional_et_jigs_needed").sum().alias("total_jigs_needed"),
                ]
            )
            .sort("risk_level")
        )

        print(
            f"  {'Risk Level':<12} {'Model-Months':<15} {'Units at Risk':<15} {'Margin at Risk':<20} {'JIGs Needed':<12}"
        )
        print("  " + "-" * 94)
        for row in risk_summary.iter_rows(named=True):
            margin = row["total_margin_at_risk"] or 0
            print(
                f"  {row['risk_level']:<12} "
                f"{row['model_months']:<15,} "
                f"{row['total_units_at_risk']:<15,} "
                f"${margin:<19,.0f} "
                f"{row['total_jigs_needed']:<12,}"
            )

        # Top risks by margin impact
        print("\n--- Top 10 ET JIG Risks (by margin impact) ---")
        top_risks = result_df.filter(pl.col("risk_level") != "NO_RISK").sort("margin_at_risk", descending=True).head(10)

        if top_risks.height > 0:
            print(
                f"  {'Model':<20} {'Month':<10} {'Units at Risk':<15} {'JIGs Needed':<12} {'Needed By':<12} {'Margin Impact':<15}"
            )
            print("  " + "-" * 94)
            for row in top_risks.iter_rows(named=True):
                margin = row["margin_at_risk"] or 0
                needed_date = (
                    row["earliest_jig_needed_date"].strftime("%Y-%m-%d") if row["earliest_jig_needed_date"] else "N/A"
                )
                print(
                    f"  {row['model_id'][:20]:<20} "
                    f"{row['target_month']:<10} "
                    f"{row['units_at_risk']:<15,} "
                    f"{row['additional_et_jigs_needed']:<12} "
                    f"{needed_date:<12} "
                    f"${margin:>13,.0f}"
                )

        # Sample detailed records
        print("\n--- Sample Risk Details ---")
        for row in top_risks.head(2).iter_rows(named=True):
            print(f"\n{'=' * 60}")
            print(f"Model: {row['model_id']}")
            print(f"Month: {format_month_display(row['target_month'])}")
            print(f"Customer: {row.get('model_customer_name', 'N/A')}")
            print(f"{'=' * 60}")
            print(f"Risk Level: {row['risk_level']}")
            print(f"Risk: {row['risk_reason']}")
            print(f"Mitigation: {row['mitigation_strategy']}")
            print(f"Current ET JIGs: {row['current_et_jig_count']}")
            print(f"Additional Needed: {row['additional_et_jigs_needed']}")
            if row["earliest_jig_needed_date"]:
                print(f"Needed By: {row['earliest_jig_needed_date'].strftime('%Y-%m-%d')}")
            print(f"Capacity Increase: {row['capacity_increase_pct']:.0f}%")
            if row["jig_procurement_warning"]:
                print(f"⚠️  {row['jig_procurement_warning']}")

    # Write output
    print(f"\n--- Writing Output ---")
    et_jig_risk_output.write_table(result_df)
    print(f"✓ Wrote {result_df.height:,} ET JIG capacity risk records")

    print("\n" + "=" * 60)
    print("✓ ET JIG CAPACITY RISK ANALYSIS COMPLETE")
    print("=" * 60)

