"""
Step 2: Capacity-Constrained Allocation Engine

This transform allocates prioritized process steps to equipment groups based on
daily capacity constraints. When the predicted median date would exceed daily
capacity, steps are moved to the next available day based on priority.

Algorithm:
1. Join prioritized steps with equipment group mapping (process_id -> equipment_group)
2. Join with capacity constraints (equipment_group -> daily_capa_in_sheet)
3. For each day and equipment group:
   - Allocate steps in priority order (lowest priority_rank first = highest priority)
   - When daily capacity is exceeded, move remaining steps to next day
4. Output includes original and allocated dates for transparency

Inputs:
- Prioritized process steps (from Step 1)
- Equipment group mapping (process_id -> equipment_group)
- Capacity constraints (equipment_group -> daily_capa_in_sheet)

Output:
- All input columns preserved
- allocated_date: The date the step is allocated to (may differ from predicted)
- allocation_run_id: Unique identifier for this allocation run
- allocation_run_ts: Timestamp of allocation
- capacity_overflow: Boolean indicating if step was moved due to capacity
- days_delayed: Number of days the step was delayed from original predicted date
"""

import polars as pl
from transforms.api import (
    transform,
    Input,
    Output,
    incremental,
    lightweight,
)
from datetime import datetime, timezone, timedelta, date
import hashlib
from typing import Dict, Tuple


# =============================================================================
# Allocation Configuration
# =============================================================================

# Column name in process steps that contains the sheet quantity to allocate
SHEET_QUANTITY_COLUMN = "latest_sheet_quantity"

# Column name for the predicted date to use for initial allocation
PREDICTED_DATE_COLUMN = "predicted_median_start_time"

# Default capacity when equipment group has no defined capacity
DEFAULT_DAILY_CAPACITY = 10  # Conservative default


# =============================================================================
# Allocation Run Identification
# =============================================================================


def generate_allocation_run_id(run_timestamp: datetime) -> str:
    """
    Generate a unique identifier for this allocation run.

    Format: allocation_{timestamp}_{short_hash}

    Args:
        run_timestamp: Timestamp of the run

    Returns:
        Unique run identifier string
    """
    ts_str = run_timestamp.strftime("%Y%m%d_%H%M%S")
    hash_input = f"allocation_{run_timestamp.isoformat()}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    return f"allocation_{ts_str}_{short_hash}"


# =============================================================================
# Optimized Allocation Algorithm
# =============================================================================


def allocate_with_capacity_constraints_optimized(
    df: pl.LazyFrame,
    capacity_lookup: Dict[str, int],
    run_id: str,
    run_timestamp: datetime,
) -> pl.LazyFrame:
    """
    Allocate process steps to dates respecting daily capacity constraints.

    This optimized version uses Polars operations where possible and only
    falls back to row-by-row processing for the capacity tracking logic.

    Args:
        df: LazyFrame with prioritized steps including equipment_group and predicted_date
        capacity_lookup: Dictionary mapping equipment_group to daily capacity
        run_id: Unique identifier for this allocation run
        run_timestamp: Timestamp for this allocation run

    Returns:
        LazyFrame with allocated_date and allocation metadata
    """
    # Collect only the columns we need for allocation logic
    # This reduces memory footprint significantly
    allocation_cols = [
        "predicted_event_lot_process_id",  # Unique ID for joining back
        "equipment_group",
        "predicted_date",
        SHEET_QUANTITY_COLUMN,
        "priority_rank",
    ]

    # Collect minimal data for allocation
    allocation_df = df.select(
        [col for col in allocation_cols if col in df.collect_schema().names()]
    ).collect()

    # Sort by priority_rank (ascending = highest priority first)
    allocation_df = allocation_df.sort("priority_rank")

    # Track capacity usage: {(equipment_group, date): sheets_used}
    capacity_usage: Dict[Tuple[str, date], int] = {}

    # Process each row and determine allocated date
    allocated_dates = []
    capacity_overflows = []
    days_delayed_list = []

    for row in allocation_df.iter_rows(named=True):
        equipment_group = row.get("equipment_group")
        predicted_date = row.get("predicted_date")
        sheet_qty = row.get(SHEET_QUANTITY_COLUMN) or 0

        # Ensure sheet_qty is numeric
        if sheet_qty is None:
            sheet_qty = 0
        try:
            sheet_qty = int(sheet_qty)
        except (ValueError, TypeError):
            sheet_qty = 0

        # Handle missing data
        if equipment_group is None or predicted_date is None:
            allocated_dates.append(predicted_date)
            capacity_overflows.append(False)
            days_delayed_list.append(0)
            continue

        # Get daily capacity for this equipment group
        daily_capacity = capacity_lookup.get(equipment_group, DEFAULT_DAILY_CAPACITY)
        if daily_capacity is None:
            daily_capacity = DEFAULT_DAILY_CAPACITY

        # Find the first available date with capacity
        current_date = predicted_date
        days_delayed = 0
        max_delay_days = 365  # Safety limit

        while days_delayed < max_delay_days:
            key = (equipment_group, current_date)
            current_usage = capacity_usage.get(key, 0)

            if current_usage + sheet_qty <= daily_capacity:
                capacity_usage[key] = current_usage + sheet_qty
                break
            else:
                current_date = current_date + timedelta(days=1)
                days_delayed += 1

        allocated_dates.append(current_date)
        capacity_overflows.append(days_delayed > 0)
        days_delayed_list.append(days_delayed)

    # Create allocation results DataFrame
    allocation_results = pl.DataFrame(
        {
            "predicted_event_lot_process_id": allocation_df[
                "predicted_event_lot_process_id"
            ],
            "allocated_date": allocated_dates,
            "capacity_overflow": capacity_overflows,
            "days_delayed": days_delayed_list,
            "allocation_run_id": [run_id] * len(allocated_dates),
            "allocation_run_ts": [run_timestamp] * len(allocated_dates),
        }
    )

    # Join allocation results back to original data
    result = df.collect().join(
        allocation_results,
        on="predicted_event_lot_process_id",
        how="left",
    )

    return result.lazy()


# =============================================================================
# Transform Definition
# =============================================================================


@lightweight(cpu_cores=4, memory_gb=32)
@transform(
    output=Output(
        "/LG Innotek-0e7800/[WF]SCM Planning Intelligence/logic/datasets/capacity_allocation"
    ),
    # Use the raw input for now - in production, this would be the Step 1 output
    # Once Step 1 output exists, change this to:
    prioritized_steps=Input("/LG Innotek-0e7800/[WF]SCM Planning Intelligence/logic/datasets/step_priority_assignment"),
    equipment_mapping=Input(
        "ri.foundry.main.dataset.f7273b7f-3cac-40b2-b8ef-8aaaaa269432"
    ),
    capacity_constraints=Input(
        "ri.foundry.main.dataset.9cd32886-97da-4659-a3b9-cdef3d909e17"
    ),
)
def compute(prioritized_steps, equipment_mapping, capacity_constraints, output) -> None:
    """
    Allocate prioritized process steps to dates respecting capacity constraints.

    This transform:
    1. Joins prioritized steps with equipment group mapping
    2. Applies capacity constraints to allocate steps to available dates
    3. Moves steps to future dates when daily capacity is exceeded

    Output columns added:
    - equipment_group: The equipment group for the process
    - allocated_date: The date allocated (may differ from predicted_median_start_time)
    - capacity_overflow: True if step was moved due to capacity constraints
    - days_delayed: Number of days delayed from original predicted date
    - allocation_run_id: Unique identifier for this allocation run
    - allocation_run_ts: Timestamp of allocation run

    Incremental Processing:
    - Each run appends new allocations to the output dataset
    - Use allocation_run_id and allocation_run_ts to distinguish between runs
    """
    # Capture run timestamp
    run_timestamp = datetime.now(timezone.utc)
    run_id = generate_allocation_run_id(run_timestamp)

    # Read inputs as LazyFrames for memory efficiency
    steps_df = prioritized_steps.polars(lazy=True)
    mapping_df = equipment_mapping.polars(lazy=True)
    capacity_df = capacity_constraints.polars()  # Small dataset, collect immediately

    # DIAGNOSTIC: Count input rows
    input_count = steps_df.select(pl.len()).collect().item()
    print(f"Input rows from step_priority_assignment: {input_count}")

    # =========================================================================
    # Build capacity lookup dictionary
    # =========================================================================
    capacity_lookup: Dict[str, int] = {}
    for row in capacity_df.iter_rows(named=True):
        equip_group = row.get("equipment_process_id")
        capacity = row.get("daily_capa_in_sheet")
        if equip_group and capacity is not None:
            capacity_lookup[equip_group] = capacity

    # =========================================================================
    # If input doesn't have priority columns, compute them here
    # This allows the transform to work standalone for testing
    # In production with Step 1 output, this block can be removed
    # =========================================================================
    schema_names = steps_df.collect_schema().names()
    if "priority_rank" not in schema_names:
        # Add computed priority columns (same logic as Step 1)
        steps_df = steps_df.with_columns(
            [
                pl.col("lot_latest_process_completion_date").alias(
                    "order_delivery_date"
                ),
                (pl.col("final_work_sequence") - pl.col("planned_work_sequence")).alias(
                    "remaining_steps"
                ),
            ]
        )

        # Sort and add priority rank
        steps_df = steps_df.sort(
            by=["order_delivery_date", "remaining_steps"],
            descending=[False, True],
            nulls_last=True,
        ).with_row_index(name="priority_rank", offset=1)

        # Add priority metadata
        priority_run_id = f"inline_priority_{run_timestamp.strftime('%Y%m%d_%H%M%S')}"
        steps_df = steps_df.with_columns(
            [
                pl.lit(priority_run_id).alias("priority_run_id"),
                pl.lit(run_timestamp).alias("priority_run_ts"),
                pl.lit("fifo_remaining_steps").alias("priority_strategy"),
            ]
        )

    # =========================================================================
    # Join steps with equipment mapping
    # process_id in steps -> process_id in mapping -> equipment_group
    # =========================================================================

    # DIAGNOSTIC: Check for process_id nulls in steps
    null_process_count = steps_df.filter(pl.col("process_id").is_null()).select(pl.len()).collect().item()
    print(f"Rows with null process_id in steps: {null_process_count}")

    # DIAGNOSTIC: Check mapping coverage
    unique_process_ids_steps = steps_df.select(pl.col("process_id").unique()).collect()
    unique_process_ids_mapping = mapping_df.select(pl.col("process_id").unique()).collect()
    print(f"Unique process_ids in steps: {unique_process_ids_steps.height}")
    print(f"Unique process_ids in mapping: {unique_process_ids_mapping.height}")

    steps_with_equipment = steps_df.join(
        mapping_df.select(["process_id", "equipment_group"]),
        on="process_id",
        how="left",
    )

    # DIAGNOSTIC: Count after join
    after_join_count = steps_with_equipment.select(pl.len()).collect().item()
    print(f"Rows after left join with equipment_mapping: {after_join_count}")

    # DIAGNOSTIC: Check for null equipment_group after join
    null_equipment_count = steps_with_equipment.filter(pl.col("equipment_group").is_null()).select(pl.len()).collect().item()
    print(f"Rows with null equipment_group after join: {null_equipment_count}")

    # =========================================================================
    # Add predicted_date column from predicted_median_start_time
    # =========================================================================
    steps_with_equipment = steps_with_equipment.with_columns(
        [
            pl.col(PREDICTED_DATE_COLUMN)
            .cast(pl.Datetime)
            .dt.date()
            .alias("predicted_date"),
        ]
    )

    # DIAGNOSTIC: Check for null predicted_date
    null_predicted_date_count = steps_with_equipment.filter(pl.col("predicted_date").is_null()).select(pl.len()).collect().item()
    print(f"Rows with null predicted_date: {null_predicted_date_count}")

    # =========================================================================
    # Perform capacity-constrained allocation
    # =========================================================================
    allocated_df = allocate_with_capacity_constraints_optimized(
        df=steps_with_equipment,
        capacity_lookup=capacity_lookup,
        run_id=run_id,
        run_timestamp=run_timestamp,
    )

    # DIAGNOSTIC: Count output rows
    output_count = allocated_df.select(pl.len()).collect().item()
    print(f"Output rows after allocation: {output_count}")
    print(f"Row loss: {input_count - output_count} ({100 * (input_count - output_count) / input_count:.1f}%)")

    # Write output
    output.write_table(allocated_df)

