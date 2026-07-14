"""
[MLWB PORT] This module is a VERBATIM copy of the Palantir Foundry source
file "models.py" from ../Python Script Allocation Engine/. The ONLY change is import
rewrites (transforms.api -> ._foundry_shim shim; myproject.datasets.allocation_engine
-> relative package imports). The pure logic is byte-identical to the source so it
can be re-synced if the Palantir engine changes. Foundry decorators are no-ops here.
"""
"""
Allocation Engine Data Models

Core data structures used throughout the allocation process.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Dict, Tuple, Set

from .config import ConstraintType


# =============================================================================
# Allocation Status
# =============================================================================


class AllocationStatus(str, Enum):
    """Status of an allocation attempt."""

    ALLOCATED = "ALLOCATED"
    FAILED_INSUFFICIENT_CAPACITY = "FAILED_INSUFFICIENT_CAPACITY"
    FAILED_HORIZON_EXCEEDED = "FAILED_HORIZON_EXCEEDED"
    FAILED_NO_EQUIPMENT = "FAILED_NO_EQUIPMENT"


# =============================================================================
# Delay Tracking
# =============================================================================


@dataclass
class DelayRecord:
    """
    Records why a lot step was delayed on a specific date.

    Answers: "Why couldn't this step run on this date?"
    """

    constraint_type: ConstraintType
    delay_date: date
    equipment_group: Optional[str] = None
    equipment_id: Optional[str] = None

    # Capacity details (for EQUIPMENT_CAPACITY_FULL)
    capacity_used: Optional[int] = None
    capacity_total: Optional[int] = None
    capacity_needed: Optional[int] = None

    def format(self) -> str:
        """Format as concise string for output."""
        date_str = self.delay_date.strftime("%Y-%m-%d")

        if self.constraint_type == ConstraintType.EQUIPMENT_CAPACITY_FULL:
            return (
                f"{date_str}: {self.equipment_id} full "
                f"({self.capacity_used}/{self.capacity_total} sheets, needed {self.capacity_needed})"
            )
        elif self.constraint_type == ConstraintType.EQUIPMENT_BLOCKED_FOR_MODEL:
            return (
                f"{date_str}: {self.equipment_id} BLOCKED for model "
                f"(had {self.capacity_total - self.capacity_used if self.capacity_total and self.capacity_used else '?'} sheets available)"
            )
        elif self.constraint_type == ConstraintType.MAX_STEPS_PER_LOT_PER_DAY:
            return f"{date_str}: max steps/lot/day limit"
        elif self.constraint_type == ConstraintType.NO_EQUIPMENT_IN_GROUP:
            return f"{date_str}: no equipment in group '{self.equipment_group}'"
        elif self.constraint_type == ConstraintType.MISSING_EQUIPMENT_GROUP:
            return f"{date_str}: equipment group not configured"
        elif self.constraint_type == ConstraintType.INSUFFICIENT_CAPACITY:
            if self.equipment_group:
                return f"{date_str}: INSUFFICIENT_CAPACITY for equipment group '{self.equipment_group}' - allocation pushed past target month"
            return f"{date_str}: INSUFFICIENT_CAPACITY - allocation pushed past target month"
        elif self.constraint_type == ConstraintType.HORIZON_EXCEEDED:
            return f"horizon exceeded (reached {self.delay_date.year})"
        else:
            return f"{date_str}: {self.constraint_type.value}"


@dataclass
class AllocationAttemptResult:
    """
    Result of attempting to allocate a single lot step.
    """

    success: bool
    allocated_date: Optional[date] = None
    equipment_id: Optional[str] = None
    equipment_group: Optional[str] = None  # Which equipment group was actually used
    days_delayed: int = 0
    delay_records: List[DelayRecord] = field(default_factory=list)
    blocking_reason: Optional[str] = None  # If success=False, ultimate reason

    def format_delay_reasons(self) -> Optional[str]:
        """Format all delay records as semicolon-separated string."""
        if not self.delay_records:
            return None

        # Deduplicate by (type, date, equipment)
        seen: Set[Tuple] = set()
        unique_reasons: List[str] = []

        for record in self.delay_records:
            key = (record.constraint_type, record.delay_date, record.equipment_id)
            if key not in seen:
                seen.add(key)
                unique_reasons.append(record.format())

        return "; ".join(unique_reasons)


# =============================================================================
# Allocation State
# =============================================================================
@dataclass
class AllocationState:
    """
    Mutable state tracked during allocation.

    Encapsulates all tracking dictionaries in one place.
    """

    # Capacity tracking: (equipment_id, date) -> sheets used
    capacity_usage: Dict[Tuple[str, date], int] = field(default_factory=dict)

    # Throughput tracking: (lot_id, date) -> steps processed
    lot_steps_per_day: Dict[Tuple[str, date], int] = field(default_factory=dict)

    # Completion tracking: set of (lot_id, process_id, sequence)
    allocated_lot_steps: Set[Tuple[str, str, int]] = field(default_factory=set)

    # Lot routing: lot_id -> list of equipment used
    lot_equipment_path: Dict[str, List[str]] = field(default_factory=dict)

    # Lot timing: lot_id -> date of last completed step
    lot_last_step_date: Dict[str, date] = field(default_factory=dict)

    # Fast-track tracking: date -> count of lots fast-tracked that day
    fast_track_lots_per_day: Dict[date, int] = field(default_factory=dict)

    # Lot fast-track designation: lot_id -> is_fast_track
    # Once a lot is designated fast-track, it stays fast-track for all its steps
    lot_fast_track_status: Dict[str, bool] = field(default_factory=dict)

    def get_capacity_used(self, equipment_id: str, on_date: date) -> int:
        return self.capacity_usage.get((equipment_id, on_date), 0)

    def add_capacity_usage(self, equipment_id: str, on_date: date, sheets: int) -> None:
        key = (equipment_id, on_date)
        self.capacity_usage[key] = self.capacity_usage.get(key, 0) + sheets

    def get_lot_steps_today(self, lot_id: str, on_date: date) -> int:
        return self.lot_steps_per_day.get((lot_id, on_date), 0)

    def increment_lot_steps(self, lot_id: str, on_date: date) -> None:
        key = (lot_id, on_date)
        self.lot_steps_per_day[key] = self.lot_steps_per_day.get(key, 0) + 1

    def is_step_allocated(self, lot_id: str, process_id: str, sequence: int) -> bool:
        return (lot_id, process_id, sequence) in self.allocated_lot_steps

    def mark_step_allocated(self, lot_id: str, process_id: str, sequence: int) -> None:
        self.allocated_lot_steps.add((lot_id, process_id, sequence))

    def set_lot_last_step_date(self, lot_id: str, step_date: date) -> None:
        self.lot_last_step_date[lot_id] = step_date

    def get_lot_last_step_date(self, lot_id: str) -> Optional[date]:
        return self.lot_last_step_date.get(lot_id)

    def add_equipment_to_path(self, lot_id: str, equipment_id: str) -> None:
        if lot_id not in self.lot_equipment_path:
            self.lot_equipment_path[lot_id] = []
        self.lot_equipment_path[lot_id].append(equipment_id)

    def get_fast_track_count(self, on_date: date) -> int:
        """Get number of lots fast-tracked on a given date."""
        return self.fast_track_lots_per_day.get(on_date, 0)

    def designate_lot_fast_track(self, lot_id: str, on_date: date) -> None:
        """Designate a lot as fast-tracked and increment daily counter."""
        self.lot_fast_track_status[lot_id] = True
        self.fast_track_lots_per_day[on_date] = self.fast_track_lots_per_day.get(on_date, 0) + 1

    def is_lot_fast_track(self, lot_id: str) -> bool:
        """Check if a lot is designated as fast-track."""
        return self.lot_fast_track_status.get(lot_id, False)


# =============================================================================
# Output Records
# =============================================================================


@dataclass
class LotStepAllocation:
    """
    A single allocated lot step - maps to one row in allocation_output.

    Includes both successful allocations and failed allocation attempts.
    """

    # Identity
    allocation_id: str
    lot_id: str
    model_id: str
    process_id: str
    sequence: int

    # Assignment (None for failed allocations)
    equipment_group: Optional[str]
    equipment_id: Optional[str]
    allocated_date: Optional[date]

    # Timing context
    target_month: str
    actual_completion_month: Optional[str]
    is_delayed_completion: bool

    # Simulation context
    simulation_id: str
    revenue_plan_id: str
    model_priority: int

    # Progress tracking
    remaining_steps: int

    # Resource usage
    capacity_used_sheets: int
    units_produced: int  # Only >0 on final step

    # Delay tracking
    days_delayed: int
    delay_reasons: Optional[str]

    # Lot metadata
    is_new_lot: bool

    # Allocation status - indicates success or type of failure
    allocation_status: AllocationStatus

    # Run metadata
    allocation_run_id: str
    allocation_run_ts: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "allocation_id": self.allocation_id,
            "lot_id": self.lot_id,
            "model_id": self.model_id,
            "process_id": self.process_id,
            "sequence": self.sequence,
            "equipment_group": self.equipment_group,
            "equipment_id": self.equipment_id,
            "allocated_date": self.allocated_date,
            "target_month": self.target_month,
            "actual_completion_month": self.actual_completion_month,
            "is_delayed_completion": self.is_delayed_completion,
            "simulation_id": self.simulation_id,
            "revenue_plan_id": self.revenue_plan_id,
            "model_priority": self.model_priority,
            "remaining_steps": self.remaining_steps,
            "capacity_used_sheets": self.capacity_used_sheets,
            "units_produced": self.units_produced,
            "delay_reasons": self.delay_reasons,
            "is_new_lot": self.is_new_lot,
            "allocation_status": self.allocation_status.value,
            "allocation_run_id": self.allocation_run_id,
            "allocation_run_ts": self.allocation_run_ts,
        }


@dataclass
class FailedAllocation:
    """
    A failed allocation attempt - maps to one row in failed_allocations_output.

    Captures lot steps that could not be allocated within the search horizon.
    """

    # Identity
    failed_allocation_id: str
    lot_id: str
    model_id: str
    process_id: str
    sequence: int

    # Context
    equipment_group: Optional[str]
    target_month: str

    # Simulation context
    simulation_id: str
    revenue_plan_id: str
    model_priority: int

    # Progress tracking
    remaining_steps: int

    # Resource that would have been needed
    capacity_needed_sheets: int
    potential_units: int  # Units that would have been produced if lot completed

    # Failure details
    days_searched: int
    failure_reason: str
    failure_status: AllocationStatus
    delay_details: Optional[str]

    # Lot metadata
    is_new_lot: bool

    # Run metadata
    allocation_run_id: str
    allocation_run_ts: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "failed_allocation_id": self.failed_allocation_id,
            "lot_id": self.lot_id,
            "model_id": self.model_id,
            "process_id": self.process_id,
            "sequence": self.sequence,
            "equipment_group": self.equipment_group,
            "target_month": self.target_month,
            "simulation_id": self.simulation_id,
            "revenue_plan_id": self.revenue_plan_id,
            "model_priority": self.model_priority,
            "remaining_steps": self.remaining_steps,
            "capacity_needed_sheets": self.capacity_needed_sheets,
            "potential_units": self.potential_units,
            "days_searched": self.days_searched,
            "failure_reason": self.failure_reason,
            "failure_status": self.failure_status.value,
            "delay_details": self.delay_details,
            "is_new_lot": self.is_new_lot,
            "allocation_run_id": self.allocation_run_id,
            "allocation_run_ts": self.allocation_run_ts,
        }


@dataclass
class UnroutedModelDemand:
    """
    A model with demand but no routing - cannot create virtual lots.
    Maps to one row in unrouted_model_demand_output.

    Designed as a risk record to surface to end users for remediation.
    """

    # Identity
    new_model_risk_id: str  # Unique ID for this risk record
    model_id: str

    # Demand details
    demand_qty_ea: int
    shortfall_month: str  # Month we wanted to produce the model

    # Model metadata (for user context)
    customer_name: Optional[str]
    end_customer: Optional[str]
    model_sales_team: Optional[str]

    # Simulation context
    simulation_id: str
    simulation_name: Optional[str]

    # User-facing fields
    title: str  # Human-readable title
    new_model_risk_description: str  # Detailed explanation
    remediation_suggestion: str  # Action to take (in Korean)

    # Run metadata
    allocation_run_id: str
    allocation_run_ts: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "title": self.title,
            "new_model_risk_description": self.new_model_risk_description,
            "new_model_risk_id": self.new_model_risk_id,
            "model_id": self.model_id,
            "demand_qty_ea": self.demand_qty_ea,
            "customer_name": self.customer_name,
            "end_customer": self.end_customer,
            "model_sales_team": self.model_sales_team,
            "simulation_name": self.simulation_name,
            "simulation_id": self.simulation_id,
            "remediation_suggestion": self.remediation_suggestion,
            "shortfall_month": self.shortfall_month,
            "allocation_run_id": self.allocation_run_id,
            "allocation_run_ts": self.allocation_run_ts,
        }
