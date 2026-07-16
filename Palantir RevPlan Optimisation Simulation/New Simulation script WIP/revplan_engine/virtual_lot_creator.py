"""
[MLWB PORT] This module was a VERBATIM copy of the Palantir Foundry source
file "virtual_lot_creator.py" from ../Python Script Allocation Engine/. Import rewrites
apart (transforms.api -> ._foundry_shim shim; myproject.datasets.allocation_engine
-> relative package imports), the logic was byte-identical to the source. Foundry
decorators are no-ops here.

⚠️ INTENTIONAL DIVERGENCE (2026-07, Gumi workshop, meeting_summary §5): the virtual-lot
lead-time estimate no longer uses the `(num_steps + 1) / 2` proxy. When routing carries
per-step Run/Wait LT (RunLt/WaitLt), it computes Plan LT ≈ Σ(Run × sheets + conditional
Wait); it falls back to the legacy proxy when those columns are unavailable. Do NOT blindly
re-sync this one computation from the Palantir source — it is a deliberate customer change.
"""
"""
Virtual Lot Creation Logic

Handles creation of virtual lots when WIP is insufficient to meet demand.
"""

import math
import polars as pl
from datetime import datetime, timedelta, date
import uuid
from typing import Any, Dict, Tuple, List

from .config import DEFAULT_CONFIG, AllocationConfig

# Hours in a planning day for the Run/Wait → days lead-time estimate. Substrate fabs run
# 24/7, so a calendar-day divisor is the reasonable default; make it a config knob if a
# working-hours convention is later confirmed.
_HOURS_PER_DAY = 24.0
from .allocation_helpers import (
    parse_month_to_eom_date,
)


def _find_available_start_dates(
    num_lots_needed: int,
    daily_capacity_lots: int,
    base_start_date: date,
    grouping_model: str,
    lot_starts_by_model_date: Dict[Tuple[str, date], int],
    max_search_days: int = 200,
) -> List[date]:
    """
    Find available start dates for virtual lots, respecting ET JIG daily capacity.

    Checks lot_starts_by_model_date to see how many lots have already been
    scheduled for each day, and finds days with available capacity.

    Args:
        num_lots_needed: Number of virtual lots to create
        daily_capacity_lots: Max lots that can start per day (ET JIG capacity)
        base_start_date: Earliest possible start date
        grouping_model: Grouping model for capacity lookup
        lot_starts_by_model_date: Dict tracking (grouping_model, date) -> lots started
        max_search_days: Maximum days to search forward

    Returns:
        List of dates, one per lot, when each lot should start
    """
    lot_start_dates: List[date] = []
    current_date = base_start_date
    days_searched = 0

    while len(lot_start_dates) < num_lots_needed and days_searched < max_search_days:
        # Check how many lots are already scheduled for this day
        key = (grouping_model, current_date)
        lots_on_day = lot_starts_by_model_date.get(key, 0)

        # Calculate remaining capacity for this day
        remaining_capacity = daily_capacity_lots - lots_on_day

        if remaining_capacity > 0:
            # Add as many lots as we can to this day
            lots_to_add = min(remaining_capacity, num_lots_needed - len(lot_start_dates))
            for _ in range(lots_to_add):
                lot_start_dates.append(current_date)

        current_date += timedelta(days=1)
        days_searched += 1

    # If we still need more dates (shouldn't happen with reasonable max_search_days)
    while len(lot_start_dates) < num_lots_needed:
        lot_start_dates.append(current_date)
        current_date += timedelta(days=1)

    return lot_start_dates


def create_virtual_lots(
    model_id: str,
    target_month: str,
    shortfall_units: int,
    model_metadata: Dict[str, Any],
    model_process_steps: List[Dict[str, Any]],
    run_timestamp: datetime,
    simulation_id: str,
    revenue_plan_id: str,
    run_id: str,
    config: AllocationConfig = DEFAULT_CONFIG,
    # New parameter: track lot starts per day across all months
    lot_starts_by_model_date: Dict[Tuple[str, date], int] = None,
) -> Tuple[List[Dict[str, Any]], pl.DataFrame]:
    """
    Create virtual lots to fill a shortfall.

    Algorithm:
    1. Calculate how many lots needed based on panels per lot
    2. Find available start dates respecting ET JIG daily capacity
       (checking lot_starts_by_model_date to avoid exceeding daily limits)
    3. Create lot steps for all planned process steps

    NOTE: Virtual lots only need process_id for each step. Equipment lookup
    is handled at allocation time via process_id -> equipment_id mapping.

    Args:
        model_id: Model ID to create lots for
        target_month: Target month (e.g., '202512')
        shortfall_units: Units of shortfall to fill
        model_metadata: Model metadata dict
        model_process_steps: List of process steps for this model
        run_timestamp: Current run timestamp
        simulation_id: Simulation ID
        revenue_plan_id: Revenue plan ID
        run_id: Allocation run ID
        config: Allocation configuration
        lot_starts_by_model_date: Dict tracking (grouping_model, date) -> lots started
            Used to respect ET JIG daily capacity across all months

    Returns:
        Tuple of (virtual_lot_steps list, new_lots_df)
    """
    defaults = config.defaults
    columns = config.columns

    if not model_process_steps:
        print(f"      ⚠️ No process steps found for model {model_id}, cannot create virtual lots")
        return [], pl.DataFrame(
            schema={
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
        )

    # Initialize lot_starts tracker if not provided
    if lot_starts_by_model_date is None:
        lot_starts_by_model_date = {}

    # Get metadata with defaults from config
    units_per_panel = model_metadata.get("units_per_panel", defaults.units_per_panel)
    panels_per_sheet = model_metadata.get("panels_per_sheet", defaults.panels_per_sheet)
    daily_capacity_lots = model_metadata.get("daily_capacity_lots", defaults.daily_capacity_lots)
    # Ensure daily_capacity_lots is at least 1 to avoid division by zero
    if daily_capacity_lots <= 0:
        daily_capacity_lots = defaults.daily_capacity_lots

    lead_time_days = model_metadata.get("lead_time_days", defaults.lead_time_days)
    # ⚠️ MLWB DIVERGENCE (2026-07-16): null-safe fallback (`or model_id`), NOT
    # `.get(key, model_id)`. The metadata dict ALWAYS contains these keys — with
    # value None when the model master's GroupingModel is NULL (the known DQ gap)
    # — so `.get`'s default never fired and EVERY model got grouping_model=None.
    # The lot-start throttle is keyed (grouping_model, date), so all virtual lots
    # of ALL models shared ONE (None, date) daily budget, each model chaining
    # after the max date of every other; with ~37k lots at a few per day, start
    # dates marched ~50 years out ("reached 2082" horizon failures, TargetLot-
    # StartDate 2041 in earlier runs). get_model_lookup_key in the engine already
    # used the null-safe idiom — this aligns the VL creator with it.
    base_model = model_metadata.get("base_model") or model_id
    grouping_model = model_metadata.get("grouping_model") or model_id

    # Get model-specific panels_per_lot (from maximum_lot_size_sht * 6) or use default
    panels_per_lot = model_metadata.get("panels_per_lot", defaults.panels_per_lot)

    # Calculate number of lots needed
    units_per_lot = panels_per_lot * units_per_panel
    num_lots_needed = int((shortfall_units + units_per_lot - 1) / units_per_lot)  # Ceiling division

    # Calculate base panels per lot and remainder for distribution
    total_panels_needed = int((shortfall_units + units_per_panel - 1) / units_per_panel)
    base_panels_per_lot = total_panels_needed // num_lots_needed
    remainder_panels = total_panels_needed % num_lots_needed

    print(f"      → Creating {num_lots_needed} virtual lots ({total_panels_needed} panels total)")

    # Find the max date already used for this model (grouping_model)
    # This ensures we don't exceed ET JIG daily capacity from previous months' allocations
    max_existing_date = None
    for (gm, d), count in lot_starts_by_model_date.items():
        if gm == grouping_model and count > 0:
            if max_existing_date is None or d > max_existing_date:
                max_existing_date = d

    # Determine base start date
    if config.constraints.use_dynamic_wip_earliest_start:
        # Dynamic calculation: work backwards from EOM using lead time
        eom_date = parse_month_to_eom_date(target_month)
        latest_start_date = eom_date - timedelta(days=lead_time_days)
        days_to_stagger = (num_lots_needed - 1) // daily_capacity_lots
        earliest_start_date = latest_start_date - timedelta(days=days_to_stagger)
        base_start_date = max(earliest_start_date, config.effective_start_date)
    else:
        # Simple approach: start from simulation start date
        base_start_date = config.effective_start_date

    # If we have existing lots, we may need to start from a later date
    # to avoid exceeding daily capacity
    if max_existing_date is not None:
        # Check capacity on max_existing_date
        existing_count = lot_starts_by_model_date.get((grouping_model, max_existing_date), 0)
        if existing_count >= daily_capacity_lots:
            # That day is full, start from next day
            base_start_date = max(base_start_date, max_existing_date + timedelta(days=1))
        else:
            # Can still add lots to that day
            base_start_date = max(base_start_date, max_existing_date)

    # Lead-time estimate setup (Gumi workshop 2026-07, meeting_summary §5). When routing
    # carries per-step Run/Wait LT, Plan LT = Σ_steps (Run LT × sheets + Wait LT) — the
    # customer's worked example was 0.3 h/sheet × 5 sheets + 10 h = 11.4 h. Computed PER LOT
    # below (it scales with that lot's sheet count). Falls back to the legacy (num_steps+1)/2
    # proxy when RunLt/WaitLt are unavailable (plan prerequisite P1 unmet).
    # NOTE: the conditional-Wait rule (treat a Wait beyond wait_stall_threshold_hours as
    # immediately loadable) is a SCHEDULER concern for already-stalled WIP, NOT this fresh
    # virtual-lot planned estimate — it is deferred with the time-based scheduler follow-up.
    num_steps = len(model_process_steps)
    _have_lt = bool(model_process_steps) and all(s.get("run_lt") is not None for s in model_process_steps)

    # Find available start dates respecting ET JIG capacity
    lot_start_dates = _find_available_start_dates(
        num_lots_needed=num_lots_needed,
        daily_capacity_lots=daily_capacity_lots,
        base_start_date=base_start_date,
        grouping_model=grouping_model,
        lot_starts_by_model_date=lot_starts_by_model_date,
        max_search_days=config.constraints.max_delay_days,
    )

    # Create virtual lots
    virtual_lot_steps = []
    new_lots_records = []

    for lot_idx in range(num_lots_needed):
        # Generate unique lot ID
        lot_uuid = str(uuid.uuid4())[:8]
        lot_id = f"VL-{model_id}-{target_month}-{lot_uuid}"

        # Determine panels for this lot (first 'remainder_panels' lots get +1)
        panels_this_lot = base_panels_per_lot + (1 if lot_idx < remainder_panels else 0)

        # Get the start date for this lot
        lot_start_date = lot_start_dates[lot_idx] if lot_idx < len(lot_start_dates) else base_start_date

        # Calculate quantities
        units_this_lot = panels_this_lot * units_per_panel
        sheets_this_lot = int((panels_this_lot + panels_per_sheet - 1) / panels_per_sheet)  # Ceiling

        # Lead time for THIS lot (see the setup note above): Σ(Run × sheets + Wait), raw.
        if _have_lt:
            total_hours = sum(
                float(s.get("run_lt") or 0.0) * sheets_this_lot + float(s.get("wait_lt") or 0.0)
                for s in model_process_steps
            )
            calculated_lead_time_days = max(1, math.ceil(total_hours / _HOURS_PER_DAY))
        else:
            calculated_lead_time_days = int((num_steps + 1) / 2)  # legacy proxy (RunLt/WaitLt absent)

        # Add to new lots records
        new_lots_records.append(
            {
                "lot_id": lot_id,
                "model_id": model_id,
                "target_month": target_month,
                "target_panels": panels_this_lot,
                "target_units": units_this_lot,
                "target_lot_start_date": lot_start_date,
                "lead_time_days": calculated_lead_time_days,  # Based on number of steps
                "simulation_id": simulation_id,
                "revenue_plan_id": revenue_plan_id,
                "allocation_run_id": run_id,
                "allocation_run_ts": run_timestamp,
            }
        )

        # Update the lot_starts tracker (mutates the passed dict)
        key = (grouping_model, lot_start_date)
        lot_starts_by_model_date[key] = lot_starts_by_model_date.get(key, 0) + 1

        # Create a lot step for each process step
        num_steps = len(model_process_steps)
        for step_idx, step_data in enumerate(model_process_steps):
            is_last_step = step_idx == num_steps - 1
            virtual_lot_steps.append(
                {
                    "lot_id": lot_id,
                    "model_id": model_id,
                    "base_model": base_model,
                    "grouping_model": grouping_model,
                    "process_id": step_data["process_id"],
                    "sequence": step_data["sequence"],
                    "equipment_group_id": step_data.get("equipment_group_id"),  # Captured for reporting
                    # No longer need equipment_groups - lookup via process_id at allocation time
                    "remaining_steps": num_steps - step_data["sequence"],
                    columns.sheet_quantity: sheets_this_lot,
                    columns.unit_quantity: units_this_lot,
                    "is_new_lot": True,
                    "target_lot_start_date": lot_start_date,
                    "target_panels": panels_this_lot,
                    "_is_final_step": is_last_step,  # For hold-until-target-month optimization
                }
            )

    new_lots_df = pl.DataFrame(
        new_lots_records,
        schema={
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
        },
    )

    return virtual_lot_steps, new_lots_df
