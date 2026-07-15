"""
[MLWB PORT] This module is a VERBATIM copy of the Palantir Foundry source
file "config.py" from ../Python Script Allocation Engine/. The ONLY change is import
rewrites (transforms.api -> ._foundry_shim shim; myproject.datasets.allocation_engine
-> relative package imports). The pure logic is byte-identical to the source so it
can be re-synced if the Palantir engine changes. Foundry decorators are no-ops here.
"""
"""
Allocation Engine Configuration

All configurable constants and defaults in one place.
No magic numbers should exist outside this file.

Configuration can be loaded from a Foundry dataset (Materialized Revenue Plan Scenarios)
or fall back to hardcoded defaults when dataset values are not available.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta, datetime
from enum import Enum
from typing import FrozenSet, Dict, Any, Optional
import polars as pl


# =============================================================================
# Constraint Types - What can block/delay allocation
# =============================================================================


class ConstraintType(str, Enum):
    """Types of constraints that can delay or block lot step allocation."""

    # Capacity constraints - equipment is full
    EQUIPMENT_CAPACITY_FULL = "EQUIPMENT_CAPACITY_FULL"

    # Insufficient capacity - allocation pushed past target month
    INSUFFICIENT_CAPACITY = "INSUFFICIENT_CAPACITY"

    # Negative constraints - equipment blocked for this model/process
    EQUIPMENT_BLOCKED_FOR_MODEL = "EQUIPMENT_BLOCKED_FOR_MODEL"

    # Throughput constraints - too many operations
    MAX_STEPS_PER_LOT_PER_DAY = "MAX_STEPS_PER_LOT_PER_DAY"

    # Configuration issues - missing setup
    NO_EQUIPMENT_IN_GROUP = "NO_EQUIPMENT_IN_GROUP"
    MISSING_EQUIPMENT_GROUP = "MISSING_EQUIPMENT_GROUP"
    MISSING_REQUIRED_FIELDS = "MISSING_REQUIRED_FIELDS"

    # Hard limits - cannot proceed
    HORIZON_EXCEEDED = "HORIZON_EXCEEDED"


# =============================================================================
# Allocation Constraints Configuration
# =============================================================================


@dataclass(frozen=True)
class AllocationConstraints:
    """
    Immutable configuration for allocation constraints.

    All constraint thresholds documented in one place.
    Frozen to prevent accidental modification during runtime.
    """

    # Hold-until-target-month configuration
    # Final steps using these equipment groups cannot complete before the target month
    # They will be automatically allocated to the first day of the target month
    # These equipment groups should have infinite capacity configured
    hold_until_target_month_equipment_groups: FrozenSet[str] = frozenset({"입고대기"})
    enable_hold_until_target_month: bool = True

    # Logical no-op operation codes with no equipment (Gumi workshop 2026-07,
    # meeting_summary §10): only ~149 of ~397 routing ops have equipment constraints;
    # the rest are outsourced OR logical no-ops (e.g. M000N "LOT START"/receiving).
    # A step whose process_id is in this set is allowed to PASS THROUGH the equipment
    # gate (allocated, no capacity consumed) instead of failing FAILED_NO_EQUIPMENT.
    # Real machine ops missing from the constraints table STILL block, by design, so
    # bottlenecks (E/T, drill, laser) are not silently erased. Seeded with M000N; the
    # customer must confirm the full logical-op list (plan prerequisite P4).
    logical_passthrough_operation_codes: FrozenSet[str] = frozenset({"M000N"})

    # Unmapped-operation handling (customer meeting 2026-07-15 §4): an operation
    # with NO equipment mapping is OUTSOURCED (외주) — the step is planned-complete
    # after its Plan LT (RunLt×sheets+WaitLt via outsourced_allocation.py) instead
    # of failing FAILED_NO_EQUIPMENT and killing the lot's remaining steps.
    # False restores the pre-2026-07-15 fail behaviour (A/B, parity runs).
    treat_unmapped_as_outsourced: bool = True

    # Planned duration (days) for an outsourced step whose routing carries no
    # Run/Wait LT — the meeting's "계획완료일 임의 부여" fallback.
    outsourced_step_days_default: int = 1

    # Conditional-Wait threshold in hours (meeting_summary §5): a step stalled beyond
    # this with no queued WIP should be treated as immediately loadable, so an excessive
    # Wait LT does not push the whole plan backward. RESERVED for the (deferred) time-based
    # scheduler — the virtual-lot planned lead-time estimate uses the RAW Run×sheets+Wait,
    # so this knob is not consumed yet. Placeholder default pending the customer's exact
    # figure (they cited "~5 hours" as an open action item).
    wait_stall_threshold_hours: float = 5.0

    # Maximum steps that can be processed for a single lot in one day
    # Rationale: Physical limit on how fast a lot can move through the line
    max_steps_per_lot_per_day: int = 2

    # Fast-track configuration: High-priority lots can process more steps per day
    # Maximum steps for fast-tracked lots
    # Rationale: Allow high-priority lots to move faster through the line
    max_steps_per_lot_per_day_fast_track: int = 5

    # Maximum number of lots that can be fast-tracked per day
    # Rationale: Limit fast-track to ~20% of daily throughput to prevent overwhelming the line
    fast_track_lots_per_day: int = 200

    # Priority threshold for fast-track eligibility (lower = higher priority)
    # Lots with priority <= this value are eligible for fast-track
    # Rationale: Only the highest priority models should get fast-track treatment
    fast_track_priority_threshold: int = 10000

    # Maximum days to search for available capacity before giving up on a step
    # Rationale: Beyond this, scheduling is unreliable
    max_delay_days: int = 200

    # Year after which we stop trying to allocate (short-circuit)
    # Rationale: Planning horizon limit
    max_allocation_year: int = 2027

    # Starting month for allocation (format: YYYYMM)
    # Rationale: Don't process historical months
    start_month: str = "202602"

    # WIP lot lead time buffer multiplier
    # Rationale: Buffer factor for calculating earliest start date for WIP lots
    # Applied as: adjusted_lead_time = lead_time * remaining_fraction * buffer
    wip_lead_time_buffer_factor: float = 1.2

    # Enable dynamic WIP earliest start date calculation
    # When True: Calculate earliest start based on target month, lead time, and completion progress
    # When False: Use simulation start_date (Nov 1st) for all WIP lots
    # Rationale: Dynamic calculation can optimize scheduling but may be too aggressive
    use_dynamic_wip_earliest_start: bool = False

    # Demand fulfillment buffer multiplier
    # Rationale: Only work on enough lots to fulfill demand * this multiplier
    # This prevents lots that won't be needed from consuming capacity
    # Example: 1.2 means we'll work on enough lots to produce 120% of demand
    demand_fulfillment_buffer: float = 1.1

    # Force snapshot mode - when True, reprocess everything regardless of run history
    # Rationale: Useful for debugging, fixing data issues, or forcing a complete refresh
    # This will override incremental processing and reprocess all simulations
    force_snapshot: bool = False


# =============================================================================
# Default Values - Used when data is missing
# =============================================================================


@dataclass(frozen=True)
class DefaultValues:
    """
    Default values used when input data is missing or null.

    Each default has a rationale documented.
    """

    # Equipment capacity when not specified (sheets/day)
    daily_capacity_sheets: int = 10

    # Model priority when not specified
    # Rationale: Lowest priority (higher number = lower priority)
    model_priority: int = 99999999

    # Lead time when not specified (days)
    # Rationale: Conservative estimate for planning
    lead_time_days: int = 30

    # Daily capacity for lot starts per model when not specified
    # Rationale: Typical staggering limit
    daily_capacity_lots: int = 2

    # Panels per virtual lot when not specified
    # Rationale: Standard lot size
    panels_per_lot: int = 30

    # Panels per sheet when not specified
    # Rationale: Common configuration
    panels_per_sheet: int = 6

    # Units per panel when not specified
    # Rationale: Conservative estimate
    units_per_panel: int = 2400

    # Units per sheet when not specified
    # Rationale: Common configuration
    units_per_sheet: int = 14400

    # Infinite capacity value for special equipment groups
    # Rationale: Effectively unlimited capacity for certain process groups
    infinite_capacity_sheets: int = 10_000_000


# =============================================================================
# Column Names - Input data schema
# =============================================================================


@dataclass(frozen=True)
class ColumnNames:
    """
    Column names used when reading input datasets.

    Centralized to avoid string typos and make schema changes easier.
    """

    # Quantity columns in WIP data
    sheet_quantity: str = "latest_sheet_quantity"
    unit_quantity: str = "latest_unit_quantity"


# =============================================================================
# Blocked Equipment Groups - Cannot create virtual lots
# =============================================================================

# Equipment groups that block virtual lot creation
# Models with steps using these groups are excluded from virtual lot creation
BLOCKED_EQUIPMENT_GROUPS: FrozenSet[str] = frozenset(
    {
        "D/F 박리",
        "AFVI전 세정",
    }
)


# =============================================================================
# Infinite Capacity Equipment Groups
# =============================================================================


# This list will be used to override equipment capacity during allocation.
# In the future, this should be moved to a Foundry dataset for easier management.

INFINITE_CAPACITY_EQUIPMENT_GROUPS: FrozenSet[str] = frozenset(
    {
        "PET PEELING",
        "적층전처리(클리닝)",
        "AOI(VRS)",
        "회로 정면",
        "휨검사",
        "BAKING(UNIT)",
        "박스분리",
        "AOI(Scan)",  # Future - adjusted
        "입고대기",
        "V/M(휨검사)",
        "임피던스 측정",
        "(SOP)DEFLUX",
        "진공포장",
        "적층전처리(CZ)",
        "TP",
        "PNL 분리",
        "TNR PACKING",
        "Q/A",
    }
)


# =============================================================================
# Equipment Group Name Mapping (Dataset column -> Equipment group name)
# =============================================================================

# Maps the column names in the config dataset to the actual equipment group names
# Column format: equip_capacity_{normalized_name}
# Equipment group names may contain special characters that are normalized in column names
EQUIPMENT_COLUMN_TO_GROUP_NAME: Dict[str, str] = {
    "equip_capacity3d측정기": "3D측정기",
    "equip_capacity_aoi_scan": "AOI(Scan)",
    "equip_capacity_aoi_vrs": "AOI(VRS)",
    "equip_capacity_baking_unit": "BAKING(UNIT)",
    "equip_capacity_pet_peeling": "PET PEELING",
    "equip_capacity_pnl_분리": "PNL 분리",
    "equip_capacity_qa": "Q/A",
    "equip_capacity_smt_v_m": "SMT(V/M)",
    "equip_capacity_sop_deflux": "(SOP)DEFLUX",
    "equip_capacity_tnr_packing": "TNR PACKING",
    "equip_capacity_tp": "TP",
    "equip_capacity_vm_휨검사": "V/M(휨검사)",
    "equip_capacity_박스분리": "박스분리",
    "equip_capacity_임피던스_측정": "임피던스 측정",
    "equip_capacity_입고대기": "입고대기",
    "equip_capacity_적층전처리_cz": "적층전처리(CZ)",
    "equip_capacity_적층전처리_클리닝": "적층전처리(클리닝)",
    "equip_capacity_진공포장": "진공포장",
    "equip_capacity_회로_정면": "회로 정면",
    "equip_capacity_휨검사": "휨검사",
    "equip_capacity_ldi_노광": "LDI 노광",
}


# =============================================================================
# Runtime Configuration - Combines all settings
# =============================================================================


@dataclass(frozen=True)
class AllocationConfig:
    """
    Complete allocation configuration.

    Single source of truth for all settings.
    """

    constraints: AllocationConstraints = AllocationConstraints()
    defaults: DefaultValues = DefaultValues()
    columns: ColumnNames = ColumnNames()
    blocked_equipment_groups: FrozenSet[str] = BLOCKED_EQUIPMENT_GROUPS
    infinite_capacity_groups: FrozenSet[str] = INFINITE_CAPACITY_EQUIPMENT_GROUPS

    # Equipment capacity overrides from config dataset
    # Maps equipment group name -> capacity (sheets/day)
    # When set, these override the infinite_capacity_groups for specific equipment
    equipment_capacity_overrides: Dict[str, int] = field(default_factory=dict)

    # Simulation start date
    start_date: date = field(default_factory=lambda: datetime.now().date())
    start_date_offset_days: int = 0

    # Source tracking
    config_source: str = "hardcoded_defaults"
    simulation_id: Optional[str] = None
    revenue_plan_id: Optional[str] = None

    @property
    def effective_start_date(self) -> date:
        """Calculate actual start date with offset."""
        return self.start_date + timedelta(days=self.start_date_offset_days)


# =============================================================================
# Configuration Loading from Dataset
# =============================================================================


def _safe_get(row: Dict[str, Any], key: str, default: Any, cast_type: type = None) -> Any:
    """Safely get a value from a row dict, returning default if null/missing."""
    value = row.get(key)
    if value is None:
        return default
    if cast_type is not None:
        try:
            return cast_type(value)
        except (ValueError, TypeError):
            return default
    return value


def _parse_op_code_set(value: Any, default: FrozenSet[str]) -> FrozenSet[str]:
    """Parse an OE-row operation-code set. Accepts a comma/space-separated string or an
    iterable; None/empty falls back to `default`. Codes are upper-cased and stripped."""
    if value is None:
        return default
    if isinstance(value, str):
        parts = [p.strip().upper() for p in value.replace(",", " ").split()]
    else:
        try:
            parts = [str(p).strip().upper() for p in value]
        except TypeError:
            return default
    parts = [p for p in parts if p]
    return frozenset(parts) if parts else default


def load_config_from_row(row: Dict[str, Any], source_description: str = "dataset") -> AllocationConfig:
    """
    Build an AllocationConfig from a configuration dataset row.

    Args:
        row: Dictionary containing configuration values from the dataset
        source_description: Description of where this config came from

    Returns:
        AllocationConfig populated from the row, with defaults for missing values
    """
    # Get default instances for fallback values
    default_constraints = AllocationConstraints()
    default_values = DefaultValues()

    # Parse effective_start_date
    effective_start = row.get("effective_start_date")
    if effective_start is None:
        start_date = datetime.now().date()
    elif isinstance(effective_start, date):
        start_date = effective_start
    elif isinstance(effective_start, datetime):
        start_date = effective_start.date()
    else:
        start_date = datetime.now().date()

    # Parse start_month - convert integer to string format YYYYMM
    start_month_raw = row.get("start_month")
    if start_month_raw is None:
        start_month = default_constraints.start_month
    elif isinstance(start_month_raw, int):
        start_month = str(start_month_raw)
    else:
        start_month = str(start_month_raw)

    # ⚠️ MLWB DIVERGENCE (2026-07-15): the allocation horizon defaults RELATIVE to the
    # plan start (start year + 2) instead of the hardcoded 2027. The engine short-
    # circuits at Jan 1 of max_allocation_year, so the fixed value gave late-2026
    # target months almost no runway — any slip past Dec 31 failed
    # FAILED_HORIZON_EXCEEDED structurally, masking the real (capacity) cause.
    # An explicit max_allocation_year on the config row still wins.
    try:
        _horizon_default = int(str(start_month)[:4]) + 2
    except (ValueError, TypeError):
        _horizon_default = default_constraints.max_allocation_year

    # Build constraints from row
    constraints = AllocationConstraints(
        hold_until_target_month_equipment_groups=default_constraints.hold_until_target_month_equipment_groups,
        enable_hold_until_target_month=default_constraints.enable_hold_until_target_month,
        logical_passthrough_operation_codes=_parse_op_code_set(
            row.get("logical_passthrough_operation_codes"),
            default_constraints.logical_passthrough_operation_codes,
        ),
        wait_stall_threshold_hours=_safe_get(
            row, "wait_stall_threshold_hours", default_constraints.wait_stall_threshold_hours, float
        ),
        treat_unmapped_as_outsourced=_safe_get(
            row, "treat_unmapped_as_outsourced", default_constraints.treat_unmapped_as_outsourced, bool
        ),
        outsourced_step_days_default=_safe_get(
            row, "outsourced_step_days_default", default_constraints.outsourced_step_days_default, int
        ),
        max_steps_per_lot_per_day=_safe_get(
            row, "max_steps_per_lot_per_day", default_constraints.max_steps_per_lot_per_day, int
        ),
        max_steps_per_lot_per_day_fast_track=_safe_get(
            row, "max_steps_per_lot_per_day_fast_track", default_constraints.max_steps_per_lot_per_day_fast_track, int
        ),
        fast_track_lots_per_day=_safe_get(
            row, "fast_track_lots_per_day", default_constraints.fast_track_lots_per_day, int
        ),
        fast_track_priority_threshold=_safe_get(
            row, "fast_track_priority_threshold", default_constraints.fast_track_priority_threshold, int
        ),
        max_delay_days=_safe_get(row, "max_delay_days", default_constraints.max_delay_days, int),
        max_allocation_year=_safe_get(row, "max_allocation_year", _horizon_default, int),
        start_month=start_month,
        wip_lead_time_buffer_factor=_safe_get(
            row, "wip_lead_time_buffer_factor", default_constraints.wip_lead_time_buffer_factor, float
        ),
        use_dynamic_wip_earliest_start=_safe_get(
            row, "use_dynamic_wip_earliest_start", default_constraints.use_dynamic_wip_earliest_start, bool
        ),
        demand_fulfillment_buffer=_safe_get(
            row, "demand_fulfillment_buffer", default_constraints.demand_fulfillment_buffer, float
        ),
        force_snapshot=_safe_get(row, "force_snapshot", default_constraints.force_snapshot, bool),
    )

    # Build defaults from row
    defaults = DefaultValues(
        daily_capacity_sheets=_safe_get(
            row, "default_daily_capacity_sheets", default_values.daily_capacity_sheets, int
        ),
        model_priority=_safe_get(row, "default_model_priority", default_values.model_priority, int),
        lead_time_days=_safe_get(row, "default_lead_time_days", default_values.lead_time_days, int),
        daily_capacity_lots=_safe_get(row, "default_daily_capacity_lots", default_values.daily_capacity_lots, int),
        panels_per_lot=_safe_get(row, "default_panels_per_lot", default_values.panels_per_lot, int),
        panels_per_sheet=_safe_get(row, "default_panels_per_sheet", default_values.panels_per_sheet, int),
        units_per_panel=_safe_get(row, "default_units_per_panel", default_values.units_per_panel, int),
        units_per_sheet=_safe_get(row, "default_units_per_sheet", default_values.units_per_sheet, int),
        infinite_capacity_sheets=default_values.infinite_capacity_sheets,
    )

    # Build equipment capacity overrides from equip_capacity_* columns
    equipment_capacity_overrides: Dict[str, int] = {}
    for col_name, equip_group in EQUIPMENT_COLUMN_TO_GROUP_NAME.items():
        capacity = row.get(col_name)
        if capacity is not None:
            try:
                cap_int = int(capacity)
                # Only override if it's a meaningful capacity (not the default infinite)
                if cap_int > 0:
                    equipment_capacity_overrides[equip_group] = cap_int
            except (ValueError, TypeError):
                pass

    # Determine infinite capacity groups based on overrides
    # Equipment with high capacity (>= 1M) is considered infinite
    INFINITE_THRESHOLD = 1_000_000
    infinite_groups = set()
    for equip_group, capacity in equipment_capacity_overrides.items():
        if capacity >= INFINITE_THRESHOLD:
            infinite_groups.add(equip_group)

    # Merge with default infinite groups (those not explicitly set remain infinite)
    final_infinite_groups = INFINITE_CAPACITY_EQUIPMENT_GROUPS.union(frozenset(infinite_groups))

    return AllocationConfig(
        constraints=constraints,
        defaults=defaults,
        columns=ColumnNames(),
        blocked_equipment_groups=BLOCKED_EQUIPMENT_GROUPS,
        infinite_capacity_groups=final_infinite_groups,
        equipment_capacity_overrides=equipment_capacity_overrides,
        start_date=start_date,
        start_date_offset_days=0,
        config_source=source_description,
        simulation_id=row.get("primary_key_"),  # primary_key_ is the simulation_id in this dataset
        revenue_plan_id=row.get("revenue_plan_id"),
    )


def load_config_for_simulation(
    config_df: pl.DataFrame,
    simulation_id: str,
) -> AllocationConfig:
    """
    Load configuration for a specific simulation from the config dataset.

    The config dataset uses primary_key_ as the simulation identifier.

    Args:
        config_df: Polars DataFrame containing configuration data
        simulation_id: The simulation ID to load config for (matches primary_key_ in dataset)

    Returns:
        AllocationConfig for the simulation, or DEFAULT_CONFIG if not found
    """
    # Filter to the specific simulation, excluding deleted records
    filtered = config_df.filter((pl.col("primary_key_") == simulation_id) & (pl.col("__is_deleted") == False))

    if filtered.height == 0:
        print(f"  ⚠️ No config found for simulation_id={simulation_id}, using defaults")
        return DEFAULT_CONFIG

    # If multiple configs (shouldn't happen with primary_key), take the most recently created one
    if filtered.height > 1:
        filtered = filtered.sort("created_at", descending=True).head(1)
        print(f"  ℹ️ Multiple configs found for simulation_id={simulation_id}, using most recent")

    row = filtered.to_dicts()[0]
    config = load_config_from_row(row, f"dataset:simulation_id={simulation_id}")

    print(f"  ✓ Loaded config for simulation_id={simulation_id}")
    print(f"    - config_name: {row.get('config_name', 'N/A')}")
    print(f"    - revenue_plan_id: {config.revenue_plan_id}")
    print(f"    - start_month: {config.constraints.start_month}")
    print(f"    - max_delay_days: {config.constraints.max_delay_days}")
    print(f"    - demand_fulfillment_buffer: {config.constraints.demand_fulfillment_buffer}")
    print(f"    - equipment_capacity_overrides: {len(config.equipment_capacity_overrides)} groups")

    return config


def build_config_lookup(config_df: pl.DataFrame) -> Dict[str, AllocationConfig]:
    """
    Build a lookup dictionary of simulation_id -> AllocationConfig.

    Args:
        config_df: Polars DataFrame containing configuration data

    Returns:
        Dictionary mapping simulation_id (primary_key_) to AllocationConfig
    """
    # Filter out deleted records
    active_configs = config_df.filter(pl.col("__is_deleted") == False)

    configs: Dict[str, AllocationConfig] = {}

    for row in active_configs.iter_rows(named=True):
        simulation_id = row.get("primary_key_")
        if simulation_id and simulation_id not in configs:
            config = load_config_from_row(row, f"dataset:simulation_id={simulation_id}")
            configs[simulation_id] = config

    print(f"Built config lookup with {len(configs)} simulations")
    return configs


# =============================================================================
# Default Config Instance
# =============================================================================

# Use this instance throughout the codebase when no dataset config is available
DEFAULT_CONFIG = AllocationConfig()
