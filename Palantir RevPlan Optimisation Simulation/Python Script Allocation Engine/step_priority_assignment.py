"""
Step 1: Priority Assignment for Order Allocation Engine

This transform assigns priority rankings to process steps based on configurable
ordering criteria. The priority is used in subsequent allocation steps to
transparently allocate orders to limited machine capacity.

Ordering Strategies:
- Highest: emergency_lot_flag - emergency lots get absolute priority
- Primary: incoming_target_date (order delivery date) - earliest first
- Secondary: remaining_steps - most remaining steps first (when delivery date is missing/tied)
- Tertiary: input_date - earliest/oldest input date first (lots waiting longest)

The output adds:
- `priority_rank`: Integer ranking where lower number = higher priority (1 is highest)
- `priority_run_id`: Unique identifier for each priority calculation run
- `priority_run_ts`: Timestamp when the priority was calculated

This is designed to be flexible, extensible, and supports incremental runs
to distinguish between multiple priority calculation executions.

Incremental Processing:
- Each run appends new priority assignments to the output dataset
- Use priority_run_id and priority_run_ts to distinguish between runs
- Historical priority assignments are preserved for audit and comparison
"""

import polars as pl
from transforms.api import (
    transform,
    Input,
    Output,
    incremental,
    lightweight,
)
from datetime import datetime, timezone
import hashlib


# =============================================================================
# Priority Strategy Configuration
# =============================================================================
# This section defines the priority ordering logic.
# You can extend this by adding more columns or changing sort orders.
#
# To create a new ordering strategy:
# 1. Define a new PRIORITY_COLUMNS_* constant
# 2. Update ACTIVE_PRIORITY_STRATEGY to point to it
# 3. Update PRIORITY_STRATEGY_NAME to describe the strategy
#
# Column Mapping:
# - emergency_lot_flag: Boolean flag for emergency lots (True = highest priority)
# - incoming_target_date: Order delivery date (nullable)
# - remaining_steps: Calculated as final_work_sequence - planned_work_sequence
# - input_date: Date when lot was input into the system

# Multi-level priority with emergency flag, delivery date, remaining steps, and input date
PRIORITY_COLUMNS_EMERGENCY_DELIVERY_STEPS_INPUT = [
    # (column_name, ascending: True=earliest/smallest first, False=largest first)
    (
        "emergency_lot_flag",
        False,
    ),  # Highest: emergency lots first (True > False, descending)
    (
        "incoming_target_date",
        True,
    ),  # Primary: earliest delivery date first (nulls last)
    ("remaining_steps", False),  # Secondary: most remaining steps first
    ("input_date", True),  # Tertiary: oldest input date first (longest waiting)
]

# Example: Alternative strategy - Most work remaining first
PRIORITY_COLUMNS_MOST_WORK_FIRST = [
    ("emergency_lot_flag", False),  # Still prioritize emergencies
    ("remaining_steps", False),  # Most remaining steps first
    ("incoming_target_date", True),  # Tiebreaker: earliest delivery date
    ("input_date", True),  # Tiebreaker: oldest input date
]

# =============================================================================
# Active Strategy Selection
# =============================================================================
# Change these to switch between different priority strategies

ACTIVE_PRIORITY_STRATEGY = PRIORITY_COLUMNS_EMERGENCY_DELIVERY_STEPS_INPUT
PRIORITY_STRATEGY_NAME = (
    "emergency_delivery_steps_input"  # Human-readable name for the strategy
)


# =============================================================================
# Priority Run Identification
# =============================================================================


def generate_priority_run_id(strategy_name: str, run_timestamp: datetime) -> str:
    """
    Generate a unique identifier for this priority calculation run.

    Format: {strategy_name}_{timestamp_iso}_{short_hash}

    The hash is derived from the strategy name and exact timestamp to ensure
    uniqueness even for runs with the same strategy.

    Args:
        strategy_name: Name of the priority strategy used
        run_timestamp: Timestamp of the run

    Returns:
        Unique run identifier string
    """
    ts_str = run_timestamp.strftime("%Y%m%d_%H%M%S")

    # Create a short hash for additional uniqueness
    hash_input = f"{strategy_name}_{run_timestamp.isoformat()}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    return f"{strategy_name}_{ts_str}_{short_hash}"


# =============================================================================
# Computed Columns for Priority
# =============================================================================


def add_computed_priority_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add computed columns needed for priority calculation.

    This function creates derived columns that may not exist in the source data.
    Customize this function to compute the columns needed for your priority strategy.

    Currently computes:
    - remaining_steps: Calculated as final_work_sequence - planned_work_sequence
      (Higher value = more steps remaining)

    Note: incoming_target_date and input_date already exist in the source dataset

    Args:
        df: Input LazyFrame

    Returns:
        LazyFrame with added computed columns
    """
    df_with_computed = df.with_columns(
        [
            # Remaining steps calculation (if not already in dataset)
            # final_work_sequence is the total number of steps
            # planned_work_sequence is the current step number
            # remaining = total - current
            (pl.col("final_work_sequence") - pl.col("planned_work_sequence")).alias(
                "remaining_steps"
            ),
        ]
    )

    return df_with_computed


# =============================================================================
# Priority Ranking Logic
# =============================================================================


def apply_priority_ranking(
    df: pl.LazyFrame,
    priority_columns: list,
    strategy_name: str,
    run_timestamp: datetime,
) -> pl.LazyFrame:
    """
    Apply priority ranking based on configured columns.

    Args:
        df: Input LazyFrame with lot/process step data
        priority_columns: List of (column_name, ascending) tuples defining sort order
        strategy_name: Name of the priority strategy for identification
        run_timestamp: Timestamp of this priority run

    Returns:
        LazyFrame with added priority columns:
        - priority_rank: Integer ranking (1 = highest priority)
        - priority_run_id: Unique identifier for this run
        - priority_run_ts: Timestamp of when priority was calculated
        - priority_strategy: Name of the strategy used
    """
    # Build sort expressions from configuration
    sort_columns = []
    sort_descending = []

    for col_name, ascending in priority_columns:
        sort_columns.append(col_name)
        sort_descending.append(not ascending)  # polars uses descending param

    # Sort by priority columns
    df_sorted = df.sort(
        by=sort_columns,
        descending=sort_descending,
        nulls_last=True,  # Push null values to end (lower priority)
    )

    # Generate run identification
    run_id = generate_priority_run_id(strategy_name, run_timestamp)

    # Add priority rank (1-based ranking) and run metadata
    df_ranked = df_sorted.with_row_index(name="priority_rank", offset=1).with_columns(
        [
            pl.lit(run_id).alias("priority_run_id"),
            pl.lit(run_timestamp).alias("priority_run_ts"),
            pl.lit(strategy_name).alias("priority_strategy"),
        ]
    )

    # Cast priority_rank to proper integer type
    df_ranked = df_ranked.with_columns(
        [
            pl.col("priority_rank").cast(pl.Int64),
        ]
    )

    return df_ranked


@lightweight(cpu_cores=4, memory_gb=16)
@transform(
    output=Output(
        "/LG Innotek-0e7800/[WF]SCM Planning Intelligence/logic/datasets/step_priority_assignment"
    ),
    process_steps=Input("ri.foundry.main.dataset.0e2db33b-d5bd-4722-9604-0c4c627f28e9"),
)
def compute(process_steps, output) -> None:
    """
    Compute priority rankings for process steps (Incremental).

    This transform keeps ALL columns from the input and adds:
    - priority_rank: Integer ranking (1 = highest priority)
    - priority_run_id: Unique identifier for this priority calculation run
    - priority_run_ts: Timestamp when priority was calculated
    - priority_strategy: Name of the strategy used for ranking

    Priority Logic (Multi-level sorting):
    0. emergency_lot_flag: Emergency lots (True) always get highest priority
       - Sorted descending so True comes before False
    1. incoming_target_date (order delivery date): Earliest dates have highest priority
       - Null values are sorted to the end (lowest priority)
    2. remaining_steps: When delivery dates are missing or tied, lots with more
       remaining steps get higher priority
    3. input_date: Final tiebreaker - older input dates (lots waiting longer) get
       higher priority

    Example ranking:
    - Rank 1-100: Emergency lots (emergency_lot_flag=True), sorted by delivery date, steps, input date
    - Rank 101+: Non-emergency lots (emergency_lot_flag=False), sorted by delivery date, steps, input date

    Computed columns for priority:
    - remaining_steps: Calculated as final_work_sequence - planned_work_sequence

    All other input columns are passed through unchanged.

    Incremental Processing:
    - Each run appends new priority assignments to the output dataset
    - Use priority_run_id and priority_run_ts to distinguish between runs
    - Historical priority assignments are preserved for audit and comparison
    - When input data changes, only new/changed records are processed
    """
    # Capture run timestamp at the start for consistency
    run_timestamp = datetime.now(timezone.utc)

    # Read input as Polars LazyFrame for efficient processing
    # For incremental, this will only contain new/changed records
    # Keep ALL columns from input
    df = process_steps.polars(lazy=True)

    # Add computed columns needed for priority calculation
    # df = add_computed_priority_columns(df)

    # Apply priority ranking with run identification
    df_ranked = apply_priority_ranking(
        df=df,
        priority_columns=ACTIVE_PRIORITY_STRATEGY,
        strategy_name=PRIORITY_STRATEGY_NAME,
        run_timestamp=run_timestamp,
    )

    # Write output (will be appended in incremental mode)
    output.write_table(df_ranked)
