"""
[MLWB PORT] This module is a VERBATIM copy of the Palantir Foundry source
file "allocation_helpers (1).py" from ../Python Script Allocation Engine/. The ONLY change is import
rewrites (transforms.api -> ._foundry_shim shim; myproject.datasets.allocation_engine
-> relative package imports). The pure logic is byte-identical to the source so it
can be re-synced if the Palantir engine changes. Foundry decorators are no-ops here.
"""
"""
Allocation Engine Helper Functions

Contains utility functions for building lookups from input data.
"""

import polars as pl
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
import hashlib
from typing import Any, Dict, Tuple, List, Set

from .config import DEFAULT_CONFIG, AllocationConfig


# =============================================================================
# ID Generation
# =============================================================================


def generate_allocation_run_id(run_timestamp: datetime, simulation_id: str) -> str:
    """Generate unique run ID."""
    ts_str = run_timestamp.strftime("%Y%m%d_%H%M%S")
    hash_input = f"revenue_allocation_{run_timestamp.isoformat()}_{simulation_id}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    return f"rev_alloc_{ts_str}_{short_hash}"


# =============================================================================
# Date Utilities
# =============================================================================


def parse_month_to_eom_date(month_str: str) -> date:
    """
    Parse month string like '202512' to end-of-month date.

    Args:
        month_str: Month in format YYYYMM (e.g., '202512')

    Returns:
        Last day of the month as date object
    """
    year = int(month_str[:4])
    month = int(month_str[4:6])

    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)

    return next_month_first - timedelta(days=1)


def get_month_display_name(month_str: str) -> str:
    """Convert month string like '202511' to 'Nov 2025'."""
    month_names = {
        "01": "Jan",
        "02": "Feb",
        "03": "Mar",
        "04": "Apr",
        "05": "May",
        "06": "Jun",
        "07": "Jul",
        "08": "Aug",
        "09": "Sep",
        "10": "Oct",
        "11": "Nov",
        "12": "Dec",
    }
    year = month_str[:4]
    month = month_str[4:6]
    return f"{month_names.get(month, month)} {year}"


# =============================================================================
# Model Metadata Builders
# =============================================================================


def build_model_metadata_lookup(
    model_df: pl.DataFrame,
    model_unit_conversion_df: pl.DataFrame,
    config: AllocationConfig = DEFAULT_CONFIG,
) -> Dict[str, Dict[str, Any]]:
    """
    Build comprehensive model metadata lookup.

    Combines model data with unit conversion data to provide:
    - Lead time information
    - Daily capacity limits
    - Unit conversion factors (panels -> sheets, units, etc.)

    Args:
        model_df: Model dataset with lead_time, daily_capacity, etc.
        model_unit_conversion_df: Unit conversion factors
        config: Allocation configuration

    Returns:
        Dict mapping model_id -> {metadata dict}
    """
    model_lookup: Dict[str, Dict[str, Any]] = {}
    defaults = config.defaults

    # Build conversion lookup first
    conversion_lookup = {}
    for row in model_unit_conversion_df.iter_rows(named=True):
        model_id = row.get("model_id")
        if model_id:
            units_per_panel = row.get("units_per_panel")
            conversion_lookup[model_id] = {
                "panels_per_sheet": row.get("panels_per_sheet") or defaults.panels_per_sheet,
                "units_per_sheet": row.get("units_per_sheet") or defaults.units_per_sheet,
                "units_per_panel": units_per_panel or defaults.units_per_panel,
            }

    # Build model metadata
    for row in model_df.iter_rows(named=True):
        model_id = row.get("model_id")
        if not model_id:
            continue

        # Get lead time in seconds, convert to days
        lead_time_seconds = row.get("total_mprod_default_lot_size_median_lt_in_s")
        lead_time_days = int(lead_time_seconds / 86400) if lead_time_seconds else defaults.lead_time_days
        # Get daily capacity for staggering
        daily_capacity_lots = row.get("total_daily_capacity_lot")
        daily_capacity_lots = min(int(daily_capacity_lots), 1) if daily_capacity_lots else defaults.daily_capacity_lots
        # Get conversion factors
        conversion = conversion_lookup.get(model_id, {})

        # Get maximum lot size in sheets from model data
        # If available, convert to panels (panels = sheets * panels_per_sheet, always 6)
        maximum_lot_size_sht = row.get("maximum_lot_size_sht")
        panels_per_sheet_constant = 6  # Always 6 panels per sheet
        if maximum_lot_size_sht is not None and maximum_lot_size_sht > 0:
            panels_per_lot = int(maximum_lot_size_sht) * panels_per_sheet_constant
        else:
            panels_per_lot = defaults.panels_per_lot

        model_lookup[model_id] = {
            "base_model": row.get("base_model"),
            "grouping_model": row.get("grouping_model"),
            "customer_name": row.get("customer_name"),
            "end_customer": row.get("end_customer"),
            "sales_team": row.get("sales_team"),
            "lead_time_days": lead_time_days,
            "daily_capacity_lots": daily_capacity_lots,
            "panels_per_sheet": conversion.get("panels_per_sheet", defaults.panels_per_sheet),
            "units_per_panel": conversion.get("units_per_panel", defaults.units_per_panel),
            "units_per_sheet": conversion.get("units_per_sheet", defaults.units_per_sheet),
            "panels_per_lot": panels_per_lot,  # Model-specific or default
        }

    return model_lookup


def build_model_process_steps_lookup(
    planned_steps_df: pl.DataFrame,
    config: AllocationConfig = DEFAULT_CONFIG,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build lookup of planned process steps per model with processing time information.

    Filters to latest production plan only.

    NOTE: This function no longer handles equipment_group mapping for allocation logic.
    Equipment lookup is now done via process_id -> equipment_id mapping
    from the equipment_to_process_link dataset.

    However, equipment_group_id is still captured for downstream reporting/analysis.

    Args:
        planned_steps_df: Planned process steps dataset
        config: Allocation configuration

    Returns:
        Dict mapping model_id -> list of process step dicts (sorted by sequence)
        Each step dict includes:
        - process_id, sequence
        - equipment_group_id: For downstream reporting (not used in allocation logic)
        - grouping_model: For constraint lookup
        - q1_panel_in_seconds: Processing time per panel for this step
        - mpi_Q1_wait_time_in_seconds: Wait time after this step
    """
    # Filter to latest plan per model
    latest_steps_df = planned_steps_df.filter(pl.col("is_from_latest_production_plan"))
    model_steps: Dict[str, List[Dict[str, Any]]] = {}

    for row in latest_steps_df.iter_rows(named=True):
        model_id = row.get("model_id")
        if not model_id:
            continue

        if model_id not in model_steps:
            model_steps[model_id] = []

        # Get processing time fields (default to 0 if not available)
        q1_panel_in_seconds = row.get("q1_panel_in_seconds") or 0.0
        mpi_Q1_wait_time_in_seconds = row.get("mpi_Q1_wait_time_in_seconds") or 0

        model_steps[model_id].append(
            {
                "process_id": row.get("process_id"),
                "sequence": row.get("sequence") or 0,
                "equipment_group_id": row.get("equipment_group_id"),  # Captured for reporting
                "grouping_model": row.get("grouping_model"),  # For constraint lookup
                "q1_panel_in_seconds": float(q1_panel_in_seconds),
                "mpi_Q1_wait_time_in_seconds": float(mpi_Q1_wait_time_in_seconds),
                # Per-step lead times (hours) from routing (Gumi §5). None when unsourced (P1);
                # used by virtual_lot_creator for a real lead-time estimate.
                "run_lt": row.get("run_lt"),
                "wait_lt": row.get("wait_lt"),
            }
        )

    # Sort each model's steps by sequence
    for model_id in model_steps:
        model_steps[model_id].sort(key=lambda x: x["sequence"])

    return model_steps


def build_model_priority_lookup(
    priorities_df: pl.DataFrame,
) -> Dict[Tuple[str, str], int]:
    """
    Build model priority lookup from priorities dataset.

    Priority is per model per month - a model's priority can change month-to-month.
    Lower priority number = higher importance.

    Args:
        priorities_df: Priorities dataset with model_id, month, priority

    Returns:
        Dictionary mapping (model_id, month) -> priority_value
    """
    priority_lookup = {}

    for row in priorities_df.iter_rows(named=True):
        model_id = row.get("model_id")
        month: Any | None = row.get("month")
        priority = row.get("priority")

        if model_id and month and priority is not None:
            priority_lookup[(model_id, month)] = priority

    return priority_lookup


# =============================================================================
# Equipment Capacity Builders
# =============================================================================


def build_equipment_capacity_lookup(
    capacity_df: pl.DataFrame,
    config: AllocationConfig = DEFAULT_CONFIG,
) -> Dict[str, int]:
    """
    Build equipment capacity lookup from equipment-level capacity dataset.

    IMPORTANT: Equipment with null/missing capacity is treated as having
    effectively unlimited capacity (uses config default).

    Equipment groups designated as infinite capacity will be assigned
    the infinite_capacity_sheets value from config, overriding any existing capacity.

    Args:
        capacity_df: Equipment capacity dataset with equipment_id,
                     equipment_group_id, daily_capacity_in_sht
        config: Allocation configuration

    Returns:
        Dict mapping equipment_id -> daily_capacity_in_sheets
    """
    equipment_capacity: Dict[str, int] = {}
    defaults = config.defaults

    for row in capacity_df.iter_rows(named=True):
        equipment_id = row.get("equipment_id")
        equipment_group_id = row.get("equipment_group_id")
        capacity = row.get("daily_capacity_in_sht")

        if not equipment_id:
            continue

        # Check if this equipment group has infinite capacity (from config)
        if equipment_group_id in config.infinite_capacity_groups:
            equipment_capacity[equipment_id] = defaults.infinite_capacity_sheets
        else:
            ### REMOVE IN THE FUTURE or confirm - temporary capacity adjustments
            if equipment_group_id == "2D/3D AFVI":
                capacity = capacity * 2 if capacity is not None else None
            elif equipment_group_id == "Hole VRS":
                capacity = capacity * 10 if capacity is not None else None
            elif equipment_group_id == "Hole AOI":
                capacity = capacity * 10 if capacity is not None else None
            elif equipment_group_id == "투입":
                capacity = 1000

            # Use adjusted/provided capacity or default
            equipment_capacity[equipment_id] = capacity if capacity is not None else defaults.daily_capacity_sheets

    return equipment_capacity


def build_process_to_equipment_lookup(
    equipment_to_process_df: pl.DataFrame,
) -> Dict[str, List[str]]:
    """
    Build lookup mapping process_id to list of equipment_ids that can handle it.

    This replaces the equipment_group -> equipment indirection with a direct
    process -> equipment mapping.

    Args:
        equipment_to_process_df: Dataset with equipment_id and process_id columns

    Returns:
        Dict mapping process_id -> List[equipment_id]
    """
    process_to_equipment: Dict[str, List[str]] = {}

    for row in equipment_to_process_df.iter_rows(named=True):
        equipment_id = row.get("equipment_id")
        process_id = row.get("process_id")

        if equipment_id and process_id:
            if process_id not in process_to_equipment:
                process_to_equipment[process_id] = []
            if equipment_id not in process_to_equipment[process_id]:
                process_to_equipment[process_id].append(equipment_id)

    return process_to_equipment


def build_negative_constraints_lookup(
    constraints_df: pl.DataFrame,
    model_metadata_lookup: Dict[str, Dict[str, Any]] = None,
) -> Dict[Tuple[str, str], Set[str]]:
    """
    Build negative equipment constraints lookup.

    Negative constraints specify which equipment_ids are PROHIBITED for
    a given (grouping_model, process_id) combination.

    Uses grouping_model for constraint matching since models with the same
    grouping_model share equipment constraints.
    Args:
        constraints_df: Equipment constraints dataset with columns:
                        - grouping_model: The grouping model this constraint applies to
                        - process_id: The process step
                        - equipment_id: The equipment to block
                        - constraint_type: Should be 'NEGATIVE'
        model_metadata_lookup: Optional metadata lookup (not used, kept for compatibility)

    Returns:
        Dict mapping (grouping_model, process_id) -> Set[blocked_equipment_ids]
    """
    constraints_lookup: Dict[Tuple[str, str], Set[str]] = {}

    # Filter to negative constraints only
    negative_constraints = constraints_df.filter(pl.col("constraint_type") == "NEGATIVE")

    # Build the lookup keyed by grouping_model
    for row in negative_constraints.iter_rows(named=True):
        grouping_model = row.get("grouping_model")
        process_id = row.get("process_id")
        equipment_id = row.get("equipment_id")
        if not grouping_model or not process_id or not equipment_id:
            continue

        key = (grouping_model, process_id)
        if key not in constraints_lookup:
            constraints_lookup[key] = set()
        constraints_lookup[key].add(equipment_id)

    return constraints_lookup


def find_models_with_blocked_equipment_paths(
    demand_model_ids: Set[str],
    model_process_steps_lookup: Dict[str, List[Dict[str, Any]]],
    process_to_equipment: Dict[str, List[str]],
    negative_constraints: Dict[Tuple[str, str], Set[str]],
) -> Dict[str, Tuple[str, str]]:
    """
    Find models where all equipment is blocked for at least one process step.

    Uses grouping_model from process steps to check constraints (models with
    same grouping_model share the same constraints).
    Args:
        demand_model_ids: Set of model_ids with demand
        model_process_steps_lookup: model_id -> list of process steps (includes grouping_model)
        process_to_equipment: process_id -> list of equipment_ids
        negative_constraints: (grouping_model, process_id) -> set of blocked equipment_ids
    Returns:
        Dict of model_id -> (blocking_process_id, reason)
    """
    blocked_models: Dict[str, Tuple[str, str]] = {}

    for model_id in demand_model_ids:
        process_steps = model_process_steps_lookup.get(model_id, [])
        if not process_steps:
            # No routing - handled separately as unrouted demand
            continue

        # Check each process step
        for step in process_steps:
            process_id = step.get("process_id")
            if not process_id:
                continue

            # Get grouping_model from the process step itself
            grouping_model = step.get("grouping_model") or model_id

            # Get available equipment for this process
            available_equipment = process_to_equipment.get(process_id, [])
            if not available_equipment:
                # No equipment mapped - skip (handled elsewhere)
                continue

            # Get blocked equipment for this grouping_model + process
            blocked_equipment = negative_constraints.get((grouping_model, process_id), set())

            # Check if ALL equipment is blocked
            unblocked = [eq for eq in available_equipment if eq not in blocked_equipment]
            if not unblocked:
                reason = (
                    f"All {len(available_equipment)} equipment blocked for process {process_id} "
                    f"(grouping_model={grouping_model})"
                )
                blocked_models[model_id] = (process_id, reason)
                break  # One blocked step is enough to block the model

    return blocked_models


def summarize_process_coverage(
    planned_process_steps: pl.DataFrame,
    equipment_to_process: pl.DataFrame,
    logical_ops: Set[str],
) -> Dict[str, Any]:
    """Diagnostic (no behaviour change): how many routing operations have equipment.

    The Gumi workshop (meeting_summary §10) confirmed equipment constraints cover only a
    SUBSET of routing ops (~149 of ~397); the rest are outsourced or logical no-ops. This
    surfaces that split so the two are distinguishable in the run log:
      * covered              — op has ≥1 equipment in the constraints map.
      * uncovered / logical  — no equipment but in `logical_ops` → passes through the gate.
      * uncovered / blocked  — no equipment and NOT logical → a real machine op that will
                               still FAIL_NO_EQUIPMENT (actionable: extend the RTS allow-list
                               or the logical-op list).

    Returns the counts + the blocked op list (for tests / follow-up); also prints a summary.
    """
    def _ops(df: pl.DataFrame) -> Set[str]:
        if df is None or df.is_empty() or "process_id" not in df.columns:
            return set()
        return set(df.select(pl.col("process_id")).drop_nulls().to_series().to_list())

    routing_ops = _ops(planned_process_steps)
    mapped_ops = _ops(equipment_to_process)
    logical_ops = set(logical_ops or set())

    covered = routing_ops & mapped_ops
    uncovered = routing_ops - mapped_ops
    logical_uncovered = uncovered & logical_ops
    blocked_uncovered = uncovered - logical_ops

    total = len(routing_ops)
    pct = (len(covered) / total * 100.0) if total else 0.0
    print(
        f"   ▶ process coverage: {len(covered)}/{total} routing ops have equipment "
        f"({pct:.1f}%); {len(logical_uncovered)} uncovered-but-logical (pass through), "
        f"{len(blocked_uncovered)} uncovered real-machine ops STILL BLOCK."
    )
    if blocked_uncovered:
        sample = sorted(blocked_uncovered)[:20]
        print(f"     blocked ops (first {len(sample)} of {len(blocked_uncovered)}): {sample}")

    return {
        "total_routing_ops": total,
        "covered": covered,
        "uncovered_logical": logical_uncovered,
        "uncovered_blocked": blocked_uncovered,
    }


def build_equipment_constraints_lookup(
    constraints_df: pl.DataFrame,
) -> Dict[Tuple[str, str], List[str]]:
    """
    Build equipment constraints lookup for POSITIVE constraints.

    POSITIVE constraints specify that for a given (model_id, process_id) combination,
    ONLY certain equipment_ids are allowed to be used.
    Args:
        constraints_df: Equipment constraints dataset (filtered to POSITIVE constraints)
    Returns:
        Dict mapping (model_id, process_id) -> List[allowed_equipment_ids]
    """
    constraints_lookup: Dict[Tuple[str, str], List[str]] = {}

    for row in constraints_df.iter_rows(named=True):
        model_id = row.get("model_id")
        process_id = row.get("process_id")
        equipment_id = row.get("equipment_id")

        if model_id and process_id and equipment_id:
            key = (model_id, process_id)
            if key not in constraints_lookup:
                constraints_lookup[key] = []
            constraints_lookup[key].append(equipment_id)

    return constraints_lookup


@dataclass
class EquipmentSearchResult:
    """Result of searching for available equipment."""

    # Selected equipment (if found)
    equipment_id: str | None = None
    remaining_capacity: int | None = None

    # Blocked equipment that had sufficient capacity (for delay tracking)
    blocked_with_capacity: List[Tuple[str, int, int]] = field(default_factory=list)
    # List of (equipment_id, capacity_used, capacity_total)

    @property
    def success(self) -> bool:
        return self.equipment_id is not None


def find_least_loaded_equipment_for_process(
    process_id: str,
    target_date: date,
    process_to_equipment: Dict[str, List[str]],
    equipment_capacity: Dict[str, int],
    capacity_usage: Dict[Tuple[str, date], int],
    required_sheets: int,
    config: AllocationConfig = DEFAULT_CONFIG,
    negative_constraints: Dict[Tuple[str, str], Set[str]] = None,
    model_id: str = None,
    grouping_model: str = None,
) -> Tuple[str | None, int | None, List[Tuple[str, int, int]]]:
    """
    Find the least loaded equipment for a process that can handle the required capacity.

    This uses a direct process_id -> equipment_id lookup instead of equipment_group indirection.
    Args:
        process_id: Process ID to find equipment for
        target_date: Date to check capacity for
        process_to_equipment: Mapping of process_id -> list of equipment_ids
        equipment_capacity: Mapping of equipment_id -> daily capacity
        capacity_usage: Current capacity usage tracking
        required_sheets: Number of sheets needed
        config: Allocation configuration
        negative_constraints: Optional dict mapping (grouping_model, process_id) -> set of blocked equipment_ids
        model_id: Model ID (kept for compatibility, grouping_model preferred)
        grouping_model: Grouping model for constraint lookup (preferred over model_id)

    Returns:
        Tuple of (equipment_id, remaining_capacity, blocked_with_capacity)
        - equipment_id, remaining_capacity: Selected equipment or (None, None) if not found
        - blocked_with_capacity: List of (equipment_id, capacity_used, capacity_total)
          for blocked equipment that had sufficient capacity
    """
    candidates = []
    blocked_with_capacity: List[Tuple[str, int, int]] = []

    # Get equipment that can handle this process
    equipment_ids = process_to_equipment.get(process_id, [])

    if not equipment_ids:
        return None, None, blocked_with_capacity

    # Get blocked equipment for this grouping_model-process combination
    # Prefer grouping_model, fall back to model_id
    constraint_key_model = grouping_model or model_id
    blocked_equipment: Set[str] = set()
    if negative_constraints and constraint_key_model:
        blocked_equipment = negative_constraints.get((constraint_key_model, process_id), set())

    for equipment_id in equipment_ids:
        capacity_key = (equipment_id, target_date)
        current_usage = capacity_usage.get(capacity_key, 0)
        max_capacity = equipment_capacity.get(equipment_id, config.defaults.daily_capacity_sheets)
        remaining = max_capacity - current_usage

        # Check if this equipment is blocked
        is_blocked = equipment_id in blocked_equipment

        if is_blocked:
            if remaining >= required_sheets:
                blocked_with_capacity.append((equipment_id, current_usage, max_capacity))
        else:
            if remaining >= required_sheets:
                candidates.append((equipment_id, current_usage, remaining))

    if not candidates:
        return None, None, blocked_with_capacity

    # Sort by current usage (ascending) to pick the least loaded machine
    candidates.sort(key=lambda x: x[1])

    best_equipment_id, _, remaining = candidates[0]
    return best_equipment_id, remaining, blocked_with_capacity


# =============================================================================
# Risk Builders
# =============================================================================


def build_new_model_risks(
    shortfalls_df: pl.DataFrame,
    demand_by_model_month: Dict[Tuple[str, str], int],
) -> pl.DataFrame:
    """
    Build new model risk dataset for models without routing definitions.

    These are models that have demand but no planned process steps defined,
    making it impossible to create virtual lots or allocate capacity.

    Args:
        shortfalls_df: Raw shortfall data from allocation (filtered to NO_ROUTING_DEFINED)
        demand_by_model_month: Total demand lookup (model, month) -> units

    Returns:
        DataFrame with new model risk information
    """
    # Filter to only NO_ROUTING_DEFINED shortfalls
    no_routing_df = shortfalls_df.filter(pl.col("shortfall_reason") == "NO_ROUTING_DEFINED")

    if no_routing_df.height == 0:
        return pl.DataFrame(
            schema={
                "title": pl.Utf8,
                "new_model_risk_description": pl.Utf8,
                "new_model_risk_id": pl.Utf8,
                "affected_model_month_id": pl.Utf8,
                "affected_model": pl.Utf8,
                "affected_model_id": pl.Utf8,
                "demand_qty_ea": pl.Int64,
                "revenue_plan_id": pl.Utf8,
                "simulation_id": pl.Utf8,
                "remediation_suggestion": pl.Utf8,
                "remediation_id": pl.Utf8,
                "severity": pl.Utf8,
                "shortfall_month": pl.Utf8,
                "allocation_run_id": pl.Utf8,
                "allocation_run_ts": pl.Datetime,
            }
        )

    risk_records = []

    for row in no_routing_df.iter_rows(named=True):
        model_id = row.get("model_id")
        shortfall_month = row.get("shortfall_month")
        demand_qty = row.get("shortfall_qty") or 0
        revenue_plan_id = row.get("revenue_plan_id")
        simulation_id = row.get("simulation_id")
        allocation_run_id = row.get("allocation_run_id")
        allocation_run_ts = row.get("allocation_run_ts")
        # Build display month name
        month_display = get_month_display_name(shortfall_month)

        # Build IDs
        affected_model_month_id = f"{shortfall_month}-{model_id}"
        new_model_risk_id = f"{simulation_id}-{revenue_plan_id}-{model_id}-{shortfall_month}-NEW_MODEL"
        remediation_id = f"define-routing-{model_id}"

        # Build title
        title = f"New Model Without Routing: {model_id} ({month_display})"

        # Build description
        new_model_risk_description = (
            f"Model {model_id} has demand of {demand_qty:,} units in {month_display} "
            f"but no routing/process steps are defined. Cannot create virtual lots or allocate capacity "
            f"until the production routing is configured in the planned process steps."
        )

        # Build remediation suggestion
        remediation_suggestion = f"Define production routing for model {model_id} in planned process steps"

        risk_records.append(
            {
                "title": title,
                "new_model_risk_description": new_model_risk_description,
                "new_model_risk_id": new_model_risk_id,
                "affected_model_month_id": affected_model_month_id,
                "affected_model": model_id,
                "affected_model_id": model_id,
                "demand_qty_ea": demand_qty,
                "revenue_plan_id": revenue_plan_id,
                "simulation_id": simulation_id,
                "remediation_suggestion": remediation_suggestion,
                "remediation_id": remediation_id,
                "severity": "severity",
                "shortfall_month": shortfall_month,
                "allocation_run_id": allocation_run_id,
                "allocation_run_ts": allocation_run_ts,
            }
        )

    return pl.DataFrame(
        risk_records,
        schema={
            "title": pl.Utf8,
            "new_model_risk_description": pl.Utf8,
            "new_model_risk_id": pl.Utf8,
            "affected_model_month_id": pl.Utf8,
            "affected_model": pl.Utf8,
            "affected_model_id": pl.Utf8,
            "demand_qty_ea": pl.Int64,
            "revenue_plan_id": pl.Utf8,
            "simulation_id": pl.Utf8,
            "remediation_suggestion": pl.Utf8,
            "remediation_id": pl.Utf8,
            "severity": pl.Utf8,
            "shortfall_month": pl.Utf8,
            "allocation_run_id": pl.Utf8,
            "allocation_run_ts": pl.Datetime,
        },
    )


def build_wip_shortfall_risks(
    shortfalls_df: pl.DataFrame,
    demand_by_model_month: Dict[Tuple[str, str], int],
    config: AllocationConfig = DEFAULT_CONFIG,
    model_metadata_lookup: Dict[str, Dict[str, Any]] = None,
) -> pl.DataFrame:
    """
    Build enriched WIP shortfall risks dataset with remediation suggestions.

    Excludes NO_ROUTING_DEFINED shortfalls (these are tracked in new_model_risks).

    Handles different shortfall reasons:
    - PARTIAL_VIRTUAL_LOT_ALLOCATION: Some virtual lots allocated but not enough
    - INSUFFICIENT_CAPACITY: Not enough equipment capacity
    - NOT_ENOUGH_ET_JIG_CAPACITY_TO_START_LOTS: Not enough ET_JIG capacity, production will complete in a later month

    Args:
        shortfalls_df: Raw shortfall data from allocation
        demand_by_model_month: Total demand lookup (model, month) -> units
        config: Allocation configuration
        model_metadata_lookup: Optional model metadata lookup for model-specific panels_per_lot

    Returns:
        DataFrame with enriched shortfall risk information
    """
    # Filter OUT NO_ROUTING_DEFINED shortfalls (they go to new_model_risks instead)
    shortfalls_df = shortfalls_df.filter(pl.col("shortfall_reason") != "NO_ROUTING_DEFINED")

    if shortfalls_df.height == 0:
        return pl.DataFrame(
            schema={
                "title": pl.Utf8,
                "wip_shortfall_description": pl.Utf8,
                "wip_shortfall_risk_id": pl.Utf8,
                "affected_model_month_id": pl.Utf8,
                "affected_model": pl.Utf8,
                "shortfall_qty_ea": pl.Int64,
                "affected_model_id": pl.Utf8,
                "revenue_plan_id": pl.Utf8,
                "simulation_id": pl.Utf8,
                "remediation_suggestion": pl.Utf8,
                "remediation_id": pl.Utf8,
                "total_target_qty_ea": pl.Int64,
                "vl_needed_qty": pl.Int64,
                "vl_creation_start_date": pl.Int64,
                "severity": pl.Utf8,
                "shortfall_reason": pl.Utf8,
            }
        )

    risk_records = []

    for row in shortfalls_df.iter_rows(named=True):
        model_id = row.get("model_id")
        shortfall_month = row.get("shortfall_month")
        shortfall_qty = row.get("shortfall_qty") or 0
        shortfall_reason = row.get("shortfall_reason") or "UNKNOWN"
        revenue_plan_id = row.get("revenue_plan_id")
        simulation_id = row.get("simulation_id")
        priority = row.get("priority_of_model_for_month")

        # Get total target quantity from demand
        total_target_qty = demand_by_model_month.get((model_id, shortfall_month), 0)

        # Build display month name
        month_display = get_month_display_name(shortfall_month)

        # Build IDs - include reason to make unique for overflow vs other shortfalls
        affected_model_month_id = f"{shortfall_month}-{model_id}"
        wip_shortfall_risk_id = f"{simulation_id}-{revenue_plan_id}-{model_id}-{shortfall_month}-{shortfall_reason}"

        # Calculate vl_needed_qty using model-specific panels_per_lot if available
        panels_per_lot = config.defaults.panels_per_lot
        if model_metadata_lookup and model_id in model_metadata_lookup:
            panels_per_lot = model_metadata_lookup[model_id].get("panels_per_lot", panels_per_lot)
        units_per_lot = panels_per_lot
        vl_needed_qty = max(0, (shortfall_qty + units_per_lot - 1) // units_per_lot)

        # Calculate vl_creation_start_date (placeholder)
        vl_creation_start_date = 0

        # Build title, description, and remediation based on shortfall reason
        if shortfall_reason == "NOT_ENOUGH_ET_JIG_CAPACITY_TO_START_LOTS":
            title = f"{model_id} ET_JIG Capacity Constraint for {month_display}"
            remediation_id = f"et-jig-capacity-{shortfall_month}-{model_id}"

            wip_shortfall_description = (
                f"In {month_display}, {shortfall_qty:,} units will complete AFTER the target month "
                f"due to insufficient ET_JIG capacity to start lots on time. "
                f"These units are scheduled for production but will overflow into the following month(s). "
                f"Total demand for this month: {total_target_qty:,} units."
            )

            remediation_suggestion = (
                f"Consider increasing ET_JIG capacity or starting lots earlier to complete "
                f"{shortfall_qty:,} units within {month_display}"
            )

            # ET_JIG capacity constraint is typically MEDIUM severity - production will happen, just late
            severity = "MEDIUM"

        else:
            # Standard shortfall handling
            title = f"{model_id} WIP Shortfall for {month_display}"
            remediation_id = f"vl-creation-{shortfall_month}-{model_id}"

            wip_shortfall_description = (
                f"In {month_display} we will be short {shortfall_qty:,} units out of {total_target_qty:,}. "
                f"Need {vl_needed_qty} more lots to overcome this shortfall."
            )

            remediation_suggestion = f"Create {vl_needed_qty} Virtual Lots"

            # Calculate severity based on percentage
            shortfall_pct = (shortfall_qty / total_target_qty * 100) if total_target_qty > 0 else 0
            if shortfall_pct >= 50:
                severity = "HIGH"
            elif shortfall_pct >= 20:
                severity = "MEDIUM"
            else:
                severity = "LOW"

        risk_records.append(
            {
                "title": title,
                "wip_shortfall_description": wip_shortfall_description,
                "wip_shortfall_risk_id": wip_shortfall_risk_id,
                "affected_model_month_id": affected_model_month_id,
                "affected_model": model_id,
                "shortfall_qty_ea": shortfall_qty,
                "affected_model_id": model_id,
                "revenue_plan_id": revenue_plan_id,
                "simulation_id": simulation_id,
                "remediation_suggestion": remediation_suggestion,
                "remediation_id": remediation_id,
                "total_target_qty_ea": total_target_qty,
                "vl_needed_qty": vl_needed_qty,
                "vl_creation_start_date": vl_creation_start_date,
                "severity": severity,
                "shortfall_reason": shortfall_reason,
            }
        )

    return pl.DataFrame(
        risk_records,
        schema={
            "title": pl.Utf8,
            "wip_shortfall_description": pl.Utf8,
            "wip_shortfall_risk_id": pl.Utf8,
            "affected_model_month_id": pl.Utf8,
            "affected_model": pl.Utf8,
            "shortfall_qty_ea": pl.Int64,
            "affected_model_id": pl.Utf8,
            "revenue_plan_id": pl.Utf8,
            "simulation_id": pl.Utf8,
            "remediation_suggestion": pl.Utf8,
            "remediation_id": pl.Utf8,
            "total_target_qty_ea": pl.Int64,
            "vl_needed_qty": pl.Int64,
            "vl_creation_start_date": pl.Int64,
            "severity": pl.Utf8,
            "shortfall_reason": pl.Utf8,
        },
    )
