"""
Capacity Shortage Analysis

Analyzes allocation output to identify equipment capacity shortages:
1. Total duration of the shortage (contiguous periods)
2. Number of lots deferred during each shortage period
3. Financial impact (revenue and margin at risk)
4. Sum of delay days across all lots

Includes both delayed allocations (successfully allocated but late) and
failed allocations (could not be allocated at all due to insufficient capacity).

Note: NO_EQUIPMENT failures are excluded from capacity shortage analysis as they
represent configuration issues (no equipment available for model) rather than
capacity constraints.
"""

import polars as pl
from transforms.api import transform, Input, Output, lightweight
from datetime import timedelta
import re


# =============================================================================
# Output Schema Definition
# =============================================================================

EQUIPMENT_SHORTAGE_SCHEMA = {
    "shortage_id": pl.Utf8,
    "simulation_id": pl.Utf8,
    "simulation_name": pl.Utf8,
    "equipment_group": pl.Utf8,
    "shortage_start_date": pl.Date,
    "shortage_end_date": pl.Date,
    "shortage_duration_days": pl.Int32,
    "total_delay_days": pl.Int32,
    "deferred_lots_count": pl.Int32,
    "failed_lots_count": pl.Int32,
    "total_units_deferred": pl.Int64,
    "total_revenue_at_risk": pl.Float64,
    "total_margin_at_risk": pl.Float64,
    "allocation_run_id": pl.Utf8,
    "allocation_run_ts": pl.Datetime,
}

LOT_WAITING_PERIOD_SCHEMA = {
    "waiting_period_id": pl.Utf8,
    "simulation_id": pl.Utf8,
    "simulation_name": pl.Utf8,
    "lot_id": pl.Utf8,
    "model_id": pl.Utf8,
    "process_id": pl.Utf8,
    "equipment_group": pl.Utf8,
    "sales_team": pl.Utf8,
    "revenue_plan_id": pl.Utf8,
    "model_priority": pl.Float64,
    "waiting_start_date": pl.Date,
    "waiting_end_date": pl.Date,
    "waiting_days": pl.Int32,
    "is_failed_allocation": pl.Boolean,
    "failure_status": pl.Utf8,
    "failure_detail": pl.Utf8,
    "blocked_equipment_list": pl.Utf8,
    "blocked_equipment_count": pl.Int32,
    "delay_reasons_summary": pl.Utf8,
    "units_affected": pl.Int64,
    "revenue_at_risk": pl.Float64,
    "margin_at_risk": pl.Float64,
    "allocation_run_id": pl.Utf8,
    "allocation_run_ts": pl.Datetime,
}


# =============================================================================
# Helper Functions
# =============================================================================


def is_no_equipment_failure(failure_status: str, delay_details: str = None) -> bool:
    """
    Check if a failure is due to NO_EQUIPMENT (no equipment available for model).

    These are configuration issues, not capacity shortages.
    """
    if failure_status and "NO_EQUIPMENT" in str(failure_status).upper():
        return True
    if delay_details and "NO_EQUIPMENT" in str(delay_details).upper():
        return True
    return False


def parse_blocked_equipment_info(failure_reason: str) -> tuple[int, str, str]:
    """
    Parse blocked equipment information from failure_reason.

    Example inputs:
    - "All 2 equipment blocked for process ME31N (grouping_model=MGS792C2)"
    - "All 5 equipment blocked for process MA21N (grouping_model=SPCCP3007G.KMA4)"

    Returns:
        Tuple of (equipment_count, process_id, detail_message)
    """
    if not failure_reason:
        return (0, None, None)

    # Pattern: "All X equipment blocked for process PROCESS_ID (grouping_model=MODEL)"
    match = re.search(r"All (\d+) equipment blocked for process (\S+)\s*\(grouping_model=([^)]+)\)", failure_reason)

    if match:
        count = int(match.group(1))
        process_id = match.group(2)
        grouping_model = match.group(3)
        detail = f"All {count} equipment(s) are blocked for process step '{process_id}' (grouping_model: {grouping_model}). No valid equipment available to run this process."
        return (count, process_id, detail)

    return (0, None, failure_reason)


def extract_equipment_from_delay_reasons(delay_reasons: str) -> tuple[list[str], list[str]]:
    """
    Extract blocked equipment IDs and full equipment IDs from delay reasons.

    Parses patterns like:
    - "2026-01-15: GPLIJ-03 BLOCKED for model (had 396.0 sheets available)"
    - "2026-01-15: GPLIJ-01 full (536/540.0 sheets, needed 9)"

    Returns:
        Tuple of (blocked_equipment_list, full_equipment_list)
    """
    if not delay_reasons:
        return ([], [])

    reasons = [r.strip() for r in str(delay_reasons).split(";") if r.strip()]

    blocked_equipment = set()
    full_equipment = set()

    for reason in reasons:
        # Pattern for blocked equipment: "EQUIPMENT_ID BLOCKED for model"
        blocked_match = re.search(r":\s*(\S+)\s+BLOCKED for model", reason)
        if blocked_match:
            blocked_equipment.add(blocked_match.group(1))

        # Pattern for full equipment: "EQUIPMENT_ID full (X/Y sheets, needed Z)"
        full_match = re.search(r":\s*(\S+)\s+full\s+\(", reason)
        if full_match:
            full_equipment.add(full_match.group(1))

    return (sorted(list(blocked_equipment)), sorted(list(full_equipment)))


def parse_delay_reason(delay_reason: str) -> tuple[str, str]:
    """
    Parse a single delay reason line to extract date and reason text.

    Example input:
    "2024-01-15: EQ001 full (100/100 sheets, needed 50)"

    Returns: ("2024-01-15", "EQ001 full (100/100 sheets, needed 50)")
    """
    parts = delay_reason.split(":", 1)
    if len(parts) >= 2:
        date_str = parts[0].strip()
        reason_text = parts[1].strip()
        # Validate it looks like a date (YYYY-MM-DD)
        if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
            return (date_str, reason_text)
    return (None, None)


def extract_delay_dates(delay_reasons: str) -> list[str]:
    """
    Extract all unique dates from delay_reasons string.

    Filters out "max steps/lot/day limit" delays as those are not capacity shortages.

    Example input:
    "2024-01-15: EQ001 full; 2024-01-16: max steps/lot/day limit"

    Returns: ["2024-01-15"] (excludes the max steps/lot/day date)
    """
    if not delay_reasons:
        return []

    reasons = [r.strip() for r in str(delay_reasons).split(";") if r.strip()]

    dates = []
    for reason in reasons:
        # Skip "max steps/lot/day limit" reasons
        if "max steps/lot/day" in reason.lower():
            continue

        parts = reason.split(":", 1)
        if len(parts) >= 2:
            date_str = parts[0].strip()
            # Validate it looks like a date (YYYY-MM-DD)
            if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
                dates.append(date_str)

    return sorted(list(set(dates)))


def identify_contiguous_shortage_periods(df: pl.DataFrame, failed_df: pl.DataFrame = None) -> list[dict]:
    """
    Identify contiguous shortage periods for each equipment group.

    A shortage period is a sequence of consecutive days where at least one lot
    experienced a delay on that equipment group.

    Args:
        df: DataFrame with columns: simulation_id, simulation_name, equipment_group,
            lot_id, allocated_date, delay_dates (list of date strings), units_produced,
            total_revenue, total_margin, allocation_run_id, allocation_run_ts
        failed_df: Optional DataFrame of failed allocations to include in the analysis

    Returns:
        List of shortage period dictionaries
    """
    shortage_periods = []

    # Group by simulation and equipment group
    for (sim_id, sim_name, eq_group), group_df in df.group_by(["simulation_id", "simulation_name", "equipment_group"]):
        # Explode delay_dates to get one row per delay date per lot
        exploded = group_df.explode("delay_dates").filter(pl.col("delay_dates").is_not_null())

        if exploded.height == 0:
            continue

        # Convert delay_dates string to actual dates and get unique delay dates
        daily_delays = (
            exploded.with_columns([pl.col("delay_dates").str.to_date("%Y-%m-%d").alias("delay_date")])
            .group_by("delay_date")
            .agg(
                [
                    pl.col("lot_id").unique().alias("lots_on_date"),
                ]
            )
            .sort("delay_date")
        )

        # Get allocation run info (same for all rows in this group)
        run_id = group_df.select("allocation_run_id").item(0, 0)
        run_ts = group_df.select("allocation_run_ts").item(0, 0)

        # Create a lookup for lot financial data (each lot counted only once)
        lot_financials = {}
        for row in group_df.select(["lot_id", "final_production_units", "total_revenue", "total_margin"]).iter_rows(
            named=True
        ):
            lot_id = row["lot_id"]
            if lot_id not in lot_financials:
                lot_financials[lot_id] = {
                    "units": row["final_production_units"] or 0,
                    "revenue": row["total_revenue"] or 0.0,
                    "margin": row["total_margin"] or 0.0,
                }

        # Get failed lots for this simulation and equipment group
        failed_lots_for_group = set()
        if failed_df is not None and failed_df.height > 0:
            failed_group = failed_df.filter(
                (pl.col("simulation_id") == sim_id) & (pl.col("equipment_group") == eq_group)
            )
            if failed_group.height > 0:
                failed_lots_for_group = set(failed_group.select("lot_id").unique().to_series().to_list())

        # Identify contiguous periods (days with no more than 1 day gap)
        current_period_start = None
        current_period_end = None
        current_period_lots = set()

        for i, row in enumerate(daily_delays.iter_rows(named=True)):
            delay_date = row["delay_date"]
            lots = set(row["lots_on_date"])

            if current_period_start is None:
                # Start new period
                current_period_start = delay_date
                current_period_end = delay_date
                current_period_lots = lots
            else:
                # Check if this date is contiguous with current period
                days_gap = (delay_date - current_period_end).days

                if days_gap <= 1:  # Same day or next day = contiguous
                    # Extend current period
                    current_period_end = delay_date
                    current_period_lots.update(lots)
                else:
                    # Gap too large - save current period and start new one
                    # Calculate financials for this period (each lot counted once)
                    period_units = sum(
                        lot_financials[lot]["units"] for lot in current_period_lots if lot in lot_financials
                    )
                    period_revenue = sum(
                        lot_financials[lot]["revenue"] for lot in current_period_lots if lot in lot_financials
                    )
                    period_margin = sum(
                        lot_financials[lot]["margin"] for lot in current_period_lots if lot in lot_financials
                    )

                    # Count failed lots in this period
                    failed_in_period = len(current_period_lots.intersection(failed_lots_for_group))

                    shortage_periods.append(
                        {
                            "simulation_id": sim_id,
                            "simulation_name": sim_name,
                            "equipment_group": eq_group,
                            "shortage_start_date": current_period_start,
                            "shortage_end_date": current_period_end,
                            "shortage_duration_days": ((current_period_end - current_period_start).days + 1),
                            "total_delay_days": None,  # Will calculate after
                            "deferred_lots_count": len(current_period_lots),
                            "failed_lots_count": failed_in_period,
                            "total_units_deferred": period_units,
                            "total_revenue_at_risk": period_revenue,
                            "total_margin_at_risk": period_margin,
                            "allocation_run_id": run_id,
                            "allocation_run_ts": run_ts,
                        }
                    )

                    # Start new period
                    current_period_start = delay_date
                    current_period_end = delay_date
                    current_period_lots = lots

        # Don't forget the last period
        if current_period_start is not None:
            # Calculate financials for this period (each lot counted once)
            period_units = sum(lot_financials[lot]["units"] for lot in current_period_lots if lot in lot_financials)
            period_revenue = sum(lot_financials[lot]["revenue"] for lot in current_period_lots if lot in lot_financials)
            period_margin = sum(lot_financials[lot]["margin"] for lot in current_period_lots if lot in lot_financials)

            # Count failed lots in this period
            failed_in_period = len(current_period_lots.intersection(failed_lots_for_group))

            shortage_periods.append(
                {
                    "simulation_id": sim_id,
                    "simulation_name": sim_name,
                    "equipment_group": eq_group,
                    "shortage_start_date": current_period_start,
                    "shortage_end_date": current_period_end,
                    "shortage_duration_days": ((current_period_end - current_period_start).days + 1),
                    "total_delay_days": None,  # Will calculate after
                    "deferred_lots_count": len(current_period_lots),
                    "failed_lots_count": failed_in_period,
                    "total_units_deferred": period_units,
                    "total_revenue_at_risk": period_revenue,
                    "total_margin_at_risk": period_margin,
                    "allocation_run_id": run_id,
                    "allocation_run_ts": run_ts,
                }
            )

    return shortage_periods


def calculate_total_delay_days_for_period(df: pl.DataFrame, eq_group: str, start_date, end_date) -> int:
    """
    Calculate total delay days for all lots in a specific shortage period.

    This sums up the number of delay days each lot experienced within
    the shortage period.
    """
    period_lots = df.filter((pl.col("equipment_group") == eq_group))

    total_days = 0
    for row in period_lots.iter_rows(named=True):
        delay_dates = row.get("delay_dates", [])
        if delay_dates:
            # Count how many delay dates fall within this period
            for date_str in delay_dates:
                try:
                    from datetime import datetime

                    delay_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    if start_date <= delay_date <= end_date:
                        total_days += 1
                except:
                    pass

    return total_days


# =============================================================================
# Main Analysis Functions
# =============================================================================


def analyze_lot_waiting_periods(
    allocation_df: pl.DataFrame, failed_allocations_df: pl.DataFrame = None
) -> pl.DataFrame:
    """
    Create a row for each lot's waiting period on a specific equipment group.

    A waiting period represents a lot waiting for free capacity on an equipment group.
    One row per lot per equipment group where delays occurred (filtered for capacity issues only).

    Filters out "max steps/lot/day limit" delays as those are not capacity shortages.

    Now includes failed allocations as waiting periods with is_failed_allocation=True.

    Note: NO_EQUIPMENT failures are included but marked distinctly - they represent
    configuration issues (no equipment available for model) rather than capacity constraints.
    """
    waiting_periods = []

    # Process successful allocations with delays
    delayed_lots = allocation_df.filter(pl.col("delay_reasons").is_not_null())

    if delayed_lots.height > 0:
        for row in delayed_lots.iter_rows(named=True):
            delay_reasons = row.get("delay_reasons", "")
            if not delay_reasons:
                continue

            # Extract delay dates (already filters out max steps/lot/day)
            delay_dates_list = extract_delay_dates(delay_reasons)

            if not delay_dates_list:
                continue

            # Parse dates
            from datetime import datetime

            date_objects = []
            for date_str in delay_dates_list:
                try:
                    date_objects.append(datetime.strptime(date_str, "%Y-%m-%d").date())
                except:
                    pass

            if not date_objects:
                continue

            # Determine waiting period (first to last delay date)
            waiting_start = min(date_objects)
            waiting_end = max(date_objects)
            waiting_days = len(date_objects)  # Count of unique delay days

            # Extract equipment information from delay reasons
            blocked_equipment, full_equipment = extract_equipment_from_delay_reasons(delay_reasons)
            all_equipment = sorted(list(set(blocked_equipment + full_equipment)))
            equipment_list_str = ", ".join(all_equipment) if all_equipment else None

            # Summarize delay reasons (remove dates, keep unique reasons)
            delay_events = [r.strip() for r in str(delay_reasons).split(";") if r.strip()]
            unique_reasons = set()
            for event in delay_events:
                _, reason_text = parse_delay_reason(event)
                if reason_text and "max steps/lot/day" not in reason_text.lower():
                    unique_reasons.add(reason_text)

            delay_reasons_summary = "; ".join(sorted(unique_reasons))

            # Get financial impact
            units = row.get("final_production_units", 0) or 0
            total_revenue = row.get("total_revenue", 0) or 0
            total_margin = row.get("total_margin", 0) or 0

            # Get model priority
            model_priority = row.get("model_priority")

            waiting_periods.append(
                {
                    "simulation_id": row.get("simulation_id"),
                    "simulation_name": row.get("simulation_name"),
                    "lot_id": row.get("lot_id"),
                    "model_id": row.get("model_id"),
                    "process_id": row.get("process_id"),
                    "equipment_group": row.get("equipment_group"),
                    "sales_team": row.get("model_sales_team"),
                    "revenue_plan_id": row.get("revenue_plan_id"),
                    "model_priority": model_priority,
                    "waiting_start_date": waiting_start,
                    "waiting_end_date": waiting_end,
                    "waiting_days": waiting_days,
                    "is_failed_allocation": False,
                    "failure_status": None,
                    "failure_detail": None,
                    "blocked_equipment_list": equipment_list_str,
                    "blocked_equipment_count": len(blocked_equipment),
                    "delay_reasons_summary": delay_reasons_summary,
                    "units_affected": units,
                    "revenue_at_risk": total_revenue,
                    "margin_at_risk": total_margin,
                    "allocation_run_id": row.get("allocation_run_id"),
                    "allocation_run_ts": row.get("allocation_run_ts"),
                }
            )

    # Process failed allocations
    if failed_allocations_df is not None and failed_allocations_df.height > 0:
        for row in failed_allocations_df.iter_rows(named=True):
            delay_details = row.get("delay_details", "") or row.get("failure_reason", "")
            failure_reason = row.get("failure_reason", "")
            failure_status = row.get("failure_status", "UNKNOWN")
            process_id = row.get("process_id")

            # Check if this is a NO_EQUIPMENT failure
            is_no_equip = is_no_equipment_failure(failure_status, delay_details)

            # Parse blocked equipment information from failure_reason
            blocked_count, parsed_process_id, failure_detail = parse_blocked_equipment_info(failure_reason)

            # Use parsed process_id if available, otherwise use from row
            if parsed_process_id:
                process_id = parsed_process_id

            # Extract delay dates from failed allocation
            delay_dates_list = extract_delay_dates(delay_details)

            from datetime import datetime

            if delay_dates_list:
                date_objects = []
                for date_str in delay_dates_list:
                    try:
                        date_objects.append(datetime.strptime(date_str, "%Y-%m-%d").date())
                    except:
                        pass

                if date_objects:
                    waiting_start = min(date_objects)
                    waiting_end = max(date_objects)
                    waiting_days = row.get("days_searched", len(date_objects))
                else:
                    # No parseable dates - use days_searched
                    waiting_start = None
                    waiting_end = None
                    waiting_days = row.get("days_searched", 0)
            else:
                waiting_start = None
                waiting_end = None
                waiting_days = row.get("days_searched", 0)

            # Extract equipment information from delay details
            blocked_equipment, full_equipment = extract_equipment_from_delay_reasons(delay_details)
            all_equipment = sorted(list(set(blocked_equipment + full_equipment)))
            equipment_list_str = ", ".join(all_equipment) if all_equipment else None

            # Summarize delay reasons
            delay_events = [r.strip() for r in str(delay_details).split(";") if r.strip()]
            unique_reasons = set()
            for event in delay_events:
                _, reason_text = parse_delay_reason(event)
                if reason_text and "max steps/lot/day" not in reason_text.lower():
                    unique_reasons.add(reason_text)

            delay_reasons_summary = "; ".join(sorted(unique_reasons))

            # Build comprehensive failure detail if not already parsed
            if not failure_detail:
                if is_no_equip:
                    failure_detail = f"NO_EQUIPMENT: All equipment blocked for process step '{process_id}'. No valid equipment available to run this process for this model."
                else:
                    failure_detail = f"FAILED: {failure_status}"
                    if delay_reasons_summary:
                        failure_detail = delay_reasons_summary

            if not delay_reasons_summary:
                delay_reasons_summary = failure_detail

            # Get units from the lot (units that would have been produced if allocated)
            units_in_lot = row.get("units_in_lot", 0) or 0

            waiting_periods.append(
                {
                    "simulation_id": row.get("simulation_id"),
                    "simulation_name": row.get("simulation_name"),
                    "lot_id": row.get("lot_id"),
                    "model_id": row.get("model_id"),
                    "process_id": process_id,
                    "equipment_group": row.get("equipment_group"),
                    "sales_team": row.get("model_sales_team"),
                    "revenue_plan_id": row.get("revenue_plan_id"),
                    "model_priority": row.get("model_priority"),
                    "waiting_start_date": waiting_start,
                    "waiting_end_date": waiting_end,
                    "waiting_days": waiting_days,
                    "is_failed_allocation": True,
                    "failure_status": failure_status,
                    "failure_detail": failure_detail,
                    "blocked_equipment_list": equipment_list_str,
                    "blocked_equipment_count": blocked_count if blocked_count > 0 else len(blocked_equipment),
                    "delay_reasons_summary": delay_reasons_summary,
                    "units_affected": units_in_lot,  # Units that would have been produced
                    "revenue_at_risk": 0.0,  # Could be enriched with expected revenue
                    "margin_at_risk": 0.0,  # Could be enriched with expected margin
                    "allocation_run_id": row.get("allocation_run_id"),
                    "allocation_run_ts": row.get("allocation_run_ts"),
                }
            )

    if not waiting_periods:
        return pl.DataFrame(schema=LOT_WAITING_PERIOD_SCHEMA)

    # Convert to DataFrame with explicit schema to handle mixed None/str values
    result_df = pl.DataFrame(waiting_periods, schema=LOT_WAITING_PERIOD_SCHEMA, infer_schema_length=None)

    # Create waiting period ID
    result_df = result_df.with_columns(
        [
            (
                pl.col("simulation_id").cast(pl.Utf8)
                + "_"
                + pl.col("lot_id").cast(pl.Utf8)
                + "_"
                + pl.col("model_priority").cast(pl.Utf8)
                + "_"
                + pl.col("equipment_group").cast(pl.Utf8)
                + "_"
                + pl.when(pl.col("waiting_start_date").is_not_null())
                .then(pl.col("waiting_start_date").cast(pl.Utf8))
                .otherwise(pl.lit("NO_DATE"))
                + "_"
                + pl.when(pl.col("is_failed_allocation")).then(pl.lit("FAILED")).otherwise(pl.lit("DELAYED"))
            ).alias("waiting_period_id")
        ]
    )

    return result_df.select(list(LOT_WAITING_PERIOD_SCHEMA.keys()))


def analyze_equipment_shortages(
    allocation_df: pl.DataFrame, failed_allocations_df: pl.DataFrame = None
) -> pl.DataFrame:
    """
    Analyze equipment capacity shortages from allocation output.

    Identifies contiguous shortage periods where lots were delayed on each
    equipment group. Includes both successful-but-delayed allocations and
    failed allocations (excluding NO_EQUIPMENT failures which are configuration issues).
    """
    # Filter to lots with delays
    delayed_lots = allocation_df.filter(
        pl.col("delay_reasons").is_not_null() & pl.col("delay_reasons").str.contains("(?i)full|insuf")
    )

    # Filter out NO_EQUIPMENT failures from capacity shortage analysis
    # These are configuration issues, not capacity shortages
    if failed_allocations_df is not None and failed_allocations_df.height > 0:
        # Check which column contains the failure status
        if "failure_status" in failed_allocations_df.columns:
            failed_allocations_df = failed_allocations_df.filter(
                ~pl.col("failure_status").str.to_uppercase().str.contains("NO_EQUIPMENT")
            )
        # Also check delay_details for NO_EQUIPMENT
        if "delay_details" in failed_allocations_df.columns:
            failed_allocations_df = failed_allocations_df.filter(
                ~pl.col("delay_details").fill_null("").str.to_uppercase().str.contains("NO_EQUIPMENT")
            )

    if delayed_lots.height == 0 and (failed_allocations_df is None or failed_allocations_df.height == 0):
        return pl.DataFrame(schema=EQUIPMENT_SHORTAGE_SCHEMA)

    # Extract delay dates from delay_reasons for successful allocations
    if delayed_lots.height > 0:
        delay_dates_list = [
            extract_delay_dates(row.get("delay_reasons"))
            for row in delayed_lots.select("delay_reasons").iter_rows(named=True)
        ]

        delayed_lots = delayed_lots.with_columns([pl.Series("delay_dates", delay_dates_list)])

        # Filter to only lots that actually have delay dates
        delayed_lots = delayed_lots.filter(pl.col("delay_dates").list.len() > 0)

    # Process failed allocations and add their delay dates
    if failed_allocations_df is not None and failed_allocations_df.height > 0:
        # Extract delay dates from failed allocations
        failed_delay_dates_list = [
            extract_delay_dates(row.get("delay_details", "") or row.get("failure_reason", ""))
            for row in failed_allocations_df.select(
                ["delay_details", "failure_reason"]
                if "delay_details" in failed_allocations_df.columns
                else ["failure_reason"]
            ).iter_rows(named=True)
        ]

        failed_with_dates = failed_allocations_df.with_columns(
            [
                pl.Series("delay_dates", failed_delay_dates_list),
                pl.lit(0).cast(pl.Int64).alias("final_production_units"),
                pl.lit(0.0).cast(pl.Float64).alias("total_revenue"),
                pl.lit(0.0).cast(pl.Float64).alias("total_margin"),
            ]
        ).filter(pl.col("delay_dates").list.len() > 0)

        # Combine delayed lots with failed allocations if both exist
        if delayed_lots.height > 0 and failed_with_dates.height > 0:
            # Select common columns for union
            common_cols = [
                "simulation_id",
                "simulation_name",
                "equipment_group",
                "lot_id",
                "delay_dates",
                "final_production_units",
                "total_revenue",
                "total_margin",
                "allocation_run_id",
                "allocation_run_ts",
            ]

            # Ensure both DataFrames have the required columns
            for col in common_cols:
                if col not in delayed_lots.columns:
                    delayed_lots = delayed_lots.with_columns(pl.lit(None).alias(col))
                if col not in failed_with_dates.columns:
                    failed_with_dates = failed_with_dates.with_columns(pl.lit(None).alias(col))

            # Cast columns to consistent types before concat
            delayed_lots_typed = delayed_lots.select(common_cols).with_columns(
                [
                    pl.col("final_production_units").cast(pl.Int64),
                    pl.col("total_revenue").cast(pl.Float64),
                    pl.col("total_margin").cast(pl.Float64),
                ]
            )
            failed_with_dates_typed = failed_with_dates.select(common_cols).with_columns(
                [
                    pl.col("final_production_units").cast(pl.Int64),
                    pl.col("total_revenue").cast(pl.Float64),
                    pl.col("total_margin").cast(pl.Float64),
                ]
            )

            combined_lots = pl.concat(
                [
                    delayed_lots_typed,
                    failed_with_dates_typed,
                ]
            )
        elif delayed_lots.height > 0:
            combined_lots = delayed_lots
        else:
            # Only failed allocations with dates
            common_cols = [
                "simulation_id",
                "simulation_name",
                "equipment_group",
                "lot_id",
                "delay_dates",
                "final_production_units",
                "total_revenue",
                "total_margin",
                "allocation_run_id",
                "allocation_run_ts",
            ]
            for col in common_cols:
                if col not in failed_with_dates.columns:
                    failed_with_dates = failed_with_dates.with_columns(pl.lit(None).alias(col))
            combined_lots = failed_with_dates.select(common_cols)
    else:
        combined_lots = delayed_lots

    if combined_lots.height == 0:
        return pl.DataFrame(schema=EQUIPMENT_SHORTAGE_SCHEMA)

    # Identify contiguous shortage periods
    shortage_periods = identify_contiguous_shortage_periods(combined_lots, failed_allocations_df)

    if not shortage_periods:
        return pl.DataFrame(schema=EQUIPMENT_SHORTAGE_SCHEMA)

    # Calculate total_delay_days for each period
    for period in shortage_periods:
        period["total_delay_days"] = calculate_total_delay_days_for_period(
            combined_lots,
            period["equipment_group"],
            period["shortage_start_date"],
            period["shortage_end_date"],
        )

    # Convert to DataFrame
    result_df = pl.DataFrame(shortage_periods)

    # Create shortage ID
    result_df = result_df.with_columns(
        [
            (
                pl.col("simulation_id").cast(pl.Utf8)
                + "_"
                + pl.col("equipment_group").cast(pl.Utf8)
                + "_"
                + pl.col("shortage_start_date").cast(pl.Utf8)
            ).alias("shortage_id")
        ]
    )

    return result_df.select(list(EQUIPMENT_SHORTAGE_SCHEMA.keys()))


# =============================================================================
# Transform Definition
# =============================================================================


@lightweight(cpu_cores=2, memory_gb=16)
@transform(
    equipment_shortages_output=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/equipment_capacity_shortages"
    ),
    lot_waiting_periods_output=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/lot_waiting_periods"
    ),
    allocation_output=Input("ri.foundry.main.dataset.b84b2bb5-1cbf-4970-8f73-0e1eb9d4148f"),
    failed_allocations_input=Input("ri.foundry.main.dataset.96248695-b74a-4d42-b5e9-30a48bc8ff5d"),
)
def compute(
    allocation_output,
    failed_allocations_input,
    equipment_shortages_output,
    lot_waiting_periods_output,
) -> None:
    """
    Analyze equipment capacity shortages from allocation output.

    Identifies CONTIGUOUS shortage periods (sequences of consecutive days where
    lots were delayed) rather than just min/max dates.

    Includes both delayed allocations (successfully allocated but late) and
    failed allocations (could not be allocated at all due to insufficient capacity).

    Note: NO_EQUIPMENT failures are excluded from capacity shortage analysis but
    are included in lot waiting periods for visibility. They represent configuration
    issues (no equipment available for model) rather than capacity constraints.

    Outputs:
    - equipment_shortages_output: Equipment group-level capacity shortages with:
        * shortage_duration_days: Calendar time span of contiguous shortage period
        * total_delay_days: Sum of all lot delay days within the period
        * deferred_lots_count: Number of unique lots affected in the period
        * failed_lots_count: Number of lots that completely failed to allocate
        * total_revenue_at_risk: Financial revenue impact
        * total_margin_at_risk: Financial margin impact

    - lot_waiting_periods_output: Lot waiting periods with:
        * One row per lot per equipment group where the lot waited for capacity
        * waiting_start_date and waiting_end_date: First and last day of waiting
        * waiting_days: Count of unique days the lot was delayed
        * is_failed_allocation: True if lot could not be allocated at all
        * failure_status: Status code for failed allocations (including NO_EQUIPMENT)
        * Lot, model, sales team, and revenue plan details
        * Summary of delay reasons (capacity issues only, not processing limits)
    """
    print("=" * 60)
    print("EQUIPMENT CAPACITY SHORTAGE ANALYSIS")
    print("=" * 60)

    # Load allocation data
    allocation_df = allocation_output.polars()
    failed_allocations_df = failed_allocations_input.polars()
    print(f"\nTotal allocation records: {allocation_df.height:,}")
    print(f"Total failed allocation records: {failed_allocations_df.height:,}")

    # Count NO_EQUIPMENT failures for reporting
    no_equipment_count = 0
    if failed_allocations_df.height > 0 and "failure_status" in failed_allocations_df.columns:
        no_equipment_count = failed_allocations_df.filter(
            pl.col("failure_status").str.to_uppercase().str.contains("NO_EQUIPMENT")
        ).height
        if no_equipment_count > 0:
            print(
                f"  (Note: {no_equipment_count:,} NO_EQUIPMENT failures will be excluded from capacity shortage analysis)"
            )

    # Check required columns for allocation_output
    required_cols = [
        "simulation_id",
        "simulation_name",
        "lot_id",
        "equipment_group",
        "allocated_date",
        "units_produced",
        "allocation_run_id",
        "allocation_run_ts",
        "delay_reasons",
        "total_revenue",
        "total_margin",
    ]

    missing_cols = [col for col in required_cols if col not in allocation_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in allocation_output: {missing_cols}")

    # Analyze capacity shortages (contiguous periods) - includes failed allocations (excluding NO_EQUIPMENT)
    print("\n--- Analyzing Equipment Capacity Shortages (Contiguous Periods) ---")
    shortage_df = analyze_equipment_shortages(allocation_df, failed_allocations_df)
    print(f"✓ Identified {shortage_df.height:,} contiguous shortage periods")

    if shortage_df.height > 0:
        # Show summary statistics
        print("\n  Summary Statistics:")
        print(f"  Total shortage periods: {shortage_df.height:,}")
        print(
            f"  Total deferred lots (across all periods): {shortage_df.select(pl.col('deferred_lots_count').sum()).item():,}"
        )
        print(
            f"  Total failed lots (across all periods): {shortage_df.select(pl.col('failed_lots_count').sum()).item():,}"
        )
        print(f"  Total delay days: {shortage_df.select(pl.col('total_delay_days').sum()).item():,}")
        print(f"  Total revenue at risk: ${shortage_df.select(pl.col('total_revenue_at_risk').sum()).item():,.2f}")
        print(f"  Total margin at risk: ${shortage_df.select(pl.col('total_margin_at_risk').sum()).item():,.2f}")

        # Show top shortages
        top_shortages = shortage_df.sort("total_margin_at_risk", descending=True).head(10)

        print("\n  Top Shortage Periods (by margin at risk):")
        print(
            f"  {'Equipment Group':<30} {'Start Date':<12} {'Duration':<10} {'Lots':<8} {'Failed':<8} {'Delay Days':<12} {'Margin Risk':<15}"
        )
        print("  " + "-" * 103)

        for row in top_shortages.iter_rows(named=True):
            print(
                f"  {row['equipment_group']:<30} "
                f"{str(row['shortage_start_date']):<12} "
                f"{row['shortage_duration_days']:<10} "
                f"{row['deferred_lots_count']:<8,} "
                f"{row['failed_lots_count']:<8,} "
                f"{row['total_delay_days']:<12,} "
                f"${row['total_margin_at_risk']:>13,.0f}"
            )

    # Analyze lot waiting periods - includes ALL failed allocations (including NO_EQUIPMENT for visibility)
    print("\n--- Analyzing Lot Waiting Periods ---")
    waiting_df = analyze_lot_waiting_periods(allocation_df, failed_allocations_input.polars())
    print(f"✓ Identified {waiting_df.height:,} lot waiting periods")

    if waiting_df.height > 0:
        # Show summary statistics
        delayed_count = waiting_df.filter(~pl.col("is_failed_allocation")).height
        failed_count = waiting_df.filter(pl.col("is_failed_allocation")).height

        print("\n  Lot Waiting Period Summary:")
        print(f"  Total waiting periods: {waiting_df.height:,}")
        print(f"    - Delayed (eventually allocated): {delayed_count:,}")
        print(f"    - Failed (could not allocate): {failed_count:,}")
        print(f"  Unique lots: {waiting_df.select(pl.col('lot_id').n_unique()).item():,}")
        print(f"  Average waiting days: {waiting_df.select(pl.col('waiting_days').mean()).item():.1f}")
        print(f"  Max waiting days: {waiting_df.select(pl.col('waiting_days').max()).item()}")

        # Show breakdown by equipment group
        eq_breakdown = (
            waiting_df.group_by("equipment_group")
            .agg(
                [
                    pl.count().alias("waiting_periods"),
                    pl.col("is_failed_allocation").sum().alias("failed_count"),
                    pl.col("waiting_days").sum().alias("total_waiting_days"),
                    pl.col("margin_at_risk").sum().alias("total_margin"),
                ]
            )
            .sort("total_margin", descending=True)
            .head(10)
        )

        print("\n  Top Equipment Groups (by total margin at risk):")
        print(
            f"  {'Equipment Group':<40} {'Periods':<10} {'Failed':<10} {'Total Wait Days':<18} {'Total Margin Risk':<20}"
        )
        print("  " + "-" * 98)

        for row in eq_breakdown.iter_rows(named=True):
            print(
                f"  {row['equipment_group']:<40} "
                f"{row['waiting_periods']:<10,} "
                f"{row['failed_count']:<10,} "
                f"{row['total_waiting_days']:<18,} "
                f"${row['total_margin']:>18,.0f}"
            )

        # Show failure status breakdown if there are failed allocations
        if failed_count > 0:
            failure_breakdown = (
                waiting_df.filter(pl.col("is_failed_allocation"))
                .group_by("failure_status")
                .agg(
                    [
                        pl.count().alias("count"),
                        pl.col("waiting_days").sum().alias("total_waiting_days"),
                    ]
                )
                .sort("count", descending=True)
            )

            print("\n  Failed Allocation Breakdown by Status:")
            print(f"  {'Failure Status':<35} {'Count':<10} {'Total Wait Days':<18}")
            print("  " + "-" * 63)

            for row in failure_breakdown.iter_rows(named=True):
                status = row["failure_status"] or "UNKNOWN"
                note = " (config issue)" if "NO_EQUIPMENT" in status.upper() else ""
                print(f"  {status:<35} {row['count']:<10,} {row['total_waiting_days']:<18,}{note}")

    # Write outputs
    print("\n--- Writing Outputs ---")
    equipment_shortages_output.write_table(shortage_df)
    print(f"✓ Wrote {shortage_df.height:,} equipment shortage period records")

    lot_waiting_periods_output.write_table(waiting_df.unique(subset=["waiting_period_id"]))
    print(f"✓ Wrote {waiting_df.height:,} lot waiting period records")

    print("\n" + "=" * 60)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 60)
