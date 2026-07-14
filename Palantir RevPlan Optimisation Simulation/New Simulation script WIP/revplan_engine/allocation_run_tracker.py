"""
[MLWB PORT] This module is a VERBATIM copy of the Palantir Foundry source
file "allocation_run_tracker (1).py" from ../Python Script Allocation Engine/. The ONLY change is import
rewrites (transforms.api -> ._foundry_shim shim; myproject.datasets.allocation_engine
-> relative package imports). The pure logic is byte-identical to the source so it
can be re-synced if the Palantir engine changes. Foundry decorators are no-ops here.
"""
"""
Allocation Run Tracker

Tracks which simulations have been processed with what configuration
to enable incremental processing and avoid reprocessing.
"""

import polars as pl
from datetime import datetime
from typing import Dict, Any, Set, Optional
import hashlib
import json

from .config import AllocationConfig


def create_config_hash(config: AllocationConfig) -> str:
    """
    Create a deterministic hash of the configuration for comparison.

    Args:
        config: AllocationConfig object

    Returns:
        SHA256 hash of the configuration
    """
    # Convert config to a sorted dictionary for consistent hashing
    config_dict = {
        "effective_start_date": config.effective_start_date.isoformat(),
        "constraints": {
            "start_month": config.constraints.start_month,
            "max_steps_per_lot_per_day": config.constraints.max_steps_per_lot_per_day,
            "max_steps_per_lot_per_day_fast_track": config.constraints.max_steps_per_lot_per_day_fast_track,
            "fast_track_lots_per_day": config.constraints.fast_track_lots_per_day,
            "fast_track_priority_threshold": config.constraints.fast_track_priority_threshold,
            "max_delay_days": config.constraints.max_delay_days,
            "max_allocation_year": config.constraints.max_allocation_year,
            "demand_fulfillment_buffer": config.constraints.demand_fulfillment_buffer,
            "wip_lead_time_buffer_factor": config.constraints.wip_lead_time_buffer_factor,
            "use_dynamic_wip_earliest_start": config.constraints.use_dynamic_wip_earliest_start,
        },
        "defaults": {
            "model_priority": config.defaults.model_priority,
            "daily_capacity_sheets": config.defaults.daily_capacity_sheets,
            "units_per_panel": config.defaults.units_per_panel,
        },
        "columns": {
            "sheet_quantity": config.columns.sheet_quantity,
            "unit_quantity": config.columns.unit_quantity,
        },
    }

    # Convert to JSON with sorted keys for consistent hashing
    config_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_json.encode()).hexdigest()


def get_processed_simulations(
    run_tracker_df: pl.DataFrame, revenue_plan_id: Optional[str], config_hash: str
) -> Set[str]:
    """
    Get set of simulation IDs that have already been processed with the current config.

    Args:
        run_tracker_df: DataFrame with previous run tracking data
        revenue_plan_id: Revenue plan ID to filter on (None to get all)
        config_hash: Hash of current configuration

    Returns:
        Set of simulation IDs that have been processed
    """
    if run_tracker_df.height == 0:
        return set()

    # Filter to matching config and status
    if revenue_plan_id is not None:
        matching_runs = run_tracker_df.filter(
            (pl.col("revenue_plan_id") == revenue_plan_id)
            & (pl.col("config_hash") == config_hash)
            & (pl.col("status") == "SUCCESS")
        )
    else:
        matching_runs = run_tracker_df.filter((pl.col("config_hash") == config_hash) & (pl.col("status") == "SUCCESS"))

    if matching_runs.height == 0:
        return set()

    return set(matching_runs["simulation_id"].to_list())


def create_run_record(
    simulation_id: str,
    revenue_plan_id: str,
    simulation_name: str,
    config: AllocationConfig,
    config_hash: str,
    run_id: str,
    run_timestamp: datetime,
    status: str = "SUCCESS",
    error_message: Optional[str] = None,
    row_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Create a run tracking record for a simulation.

    Args:
        simulation_id: Simulation ID
        revenue_plan_id: Revenue plan ID
        simulation_name: Optional simulation name
        config: AllocationConfig used
        config_hash: Hash of configuration
        run_id: Unique run ID
        run_timestamp: Timestamp of run
        status: SUCCESS, FAILED, or SKIPPED
        error_message: Optional error message if failed
        row_counts: Optional dict of output row counts

    Returns:
        Dictionary representing the run record
    """
    return {
        "simulation_id": simulation_id,
        "revenue_plan_id": revenue_plan_id,
        "simulation_name": simulation_name,
        "config_hash": config_hash,
        "config_json": json.dumps(
            {
                "effective_start_date": config.effective_start_date.isoformat(),
                "constraints": {
                    "start_month": config.constraints.start_month,
                    "max_steps_per_lot_per_day": config.constraints.max_steps_per_lot_per_day,
                    "max_steps_per_lot_per_day_fast_track": config.constraints.max_steps_per_lot_per_day_fast_track,
                    "fast_track_lots_per_day": config.constraints.fast_track_lots_per_day,
                    "fast_track_priority_threshold": config.constraints.fast_track_priority_threshold,
                    "max_delay_days": config.constraints.max_delay_days,
                    "max_allocation_year": config.constraints.max_allocation_year,
                    "demand_fulfillment_buffer": config.constraints.demand_fulfillment_buffer,
                    "wip_lead_time_buffer_factor": config.constraints.wip_lead_time_buffer_factor,
                    "use_dynamic_wip_earliest_start": config.constraints.use_dynamic_wip_earliest_start,
                },
                "defaults": {
                    "model_priority": config.defaults.model_priority,
                    "daily_capacity_sheets": config.defaults.daily_capacity_sheets,
                    "units_per_panel": config.defaults.units_per_panel,
                },
            }
        ),
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "status": status,
        "error_message": error_message,
        "allocation_rows": row_counts.get("allocation", 0) if row_counts else 0,
        "new_lots_rows": row_counts.get("new_lots", 0) if row_counts else 0,
        "unrouted_rows": row_counts.get("unrouted", 0) if row_counts else 0,
        "failed_rows": row_counts.get("failed", 0) if row_counts else 0,
    }


# Schema for run tracker dataset
RUN_TRACKER_SCHEMA = {
    "simulation_id": pl.Utf8,
    "revenue_plan_id": pl.Utf8,
    "simulation_name": pl.Utf8,
    "config_hash": pl.Utf8,
    "config_json": pl.Utf8,
    "run_id": pl.Utf8,
    "run_timestamp": pl.Datetime,
    "status": pl.Utf8,  # SUCCESS, FAILED, SKIPPED
    "error_message": pl.Utf8,
    "allocation_rows": pl.Int64,
    "new_lots_rows": pl.Int64,
    "unrouted_rows": pl.Int64,
    "failed_rows": pl.Int64,
}
