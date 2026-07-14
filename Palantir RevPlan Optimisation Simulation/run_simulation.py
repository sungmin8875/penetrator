"""
================================================================================
 RevPlan Allocation Simulation  —  MLWB orchestrator  (LAYER 1: the compute)
================================================================================

WHAT THIS FILE IS
-----------------
This is the single "driver" script that replaces the whole Palantir Foundry
pipeline. In Foundry, each step was a separate transform wired together by
datasets. Here, every step is just a Python FUNCTION, and the data flows
between them IN MEMORY (as DataFrames) — no datasets in between.

You run ONE thing:  run_simulation(params)   ->   a dict of result tables.

There are TWO kinds of placeholders in this file. They are clearly marked:

   🔌 CONNECTION placeholder   = YOU fill this in (read from / write to Celonis).
                                 These are the parts you said you'd do manually.

   🧠 LOGIC placeholder        = the actual calculation, ported from your
                                 existing Palantir file. Marked NotImplementedError
                                 for now — this is the next step to fill in
                                 (one file at a time, so each can be checked).

NOTES
-----
* This is a SINGLE-SCENARIO run (one button click = one scenario). All of the
  Foundry multi-simulation / incremental machinery (run-tracker skipping,
  EXCLUDED_SIMULATIONS, force_snapshot) is intentionally dropped — not needed.
* The logic uses the `polars` DataFrame library (same as the Palantir code), so
  the ported logic can stay almost unchanged. pycelonis usually gives you
  `pandas`, so convert at the edges only:
        read : polars_df = pl.from_pandas(pandas_df)
        write: pandas_df = polars_df.to_pandas()
  That keeps the conversion inside the connection layer you're filling in.
================================================================================
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, FrozenSet

import polars as pl


# ==============================================================================
# 0.  HARDCODED EQUIPMENT RULES  (same values as Palantir config.py)
#     These were never user-tunable, so they stay as fixed constants.
# ==============================================================================

# Models routed through these groups do NOT get virtual lots created.
BLOCKED_EQUIPMENT_GROUPS: FrozenSet[str] = frozenset({
    "D/F 박리",
    "AFVI전 세정",
})

# Treated as having unlimited capacity (no scheduling needed).
INFINITE_CAPACITY_EQUIPMENT_GROUPS: FrozenSet[str] = frozenset({
    "PET PEELING", "적층전처리(클리닝)", "AOI(VRS)", "회로 정면", "휨검사",
    "BAKING(UNIT)", "박스분리", "AOI(Scan)", "입고대기", "V/M(휨검사)",
    "임피던스 측정", "(SOP)DEFLUX", "진공포장", "적층전처리(CZ)", "TP",
    "PNL 분리", "TNR PACKING", "Q/A",
})

# Final steps on these groups are held until the 1st of the target month.
HOLD_UNTIL_TARGET_MONTH_GROUPS: FrozenSet[str] = frozenset({"입고대기"})

# A capacity override >= this value is treated as "infinite".
INFINITE_CAPACITY_THRESHOLD: int = 1_000_000


# ==============================================================================
# 1.  PARAMETERS  —  the tuning knobs the user sets on the button/screen
#     (These are the values that used to live in the simulation_config row.)
# ==============================================================================

@dataclass
class SimulationParams:
    # ---- scenario identity -------------------------------------------------
    simulation_id: Optional[str] = None          # an id for this run
    simulation_name: Optional[str] = None
    revenue_plan_id: Optional[str] = None         # which revenue plan to run against

    # ---- timing & horizon --------------------------------------------------
    start_date: date = field(default_factory=date.today)  # simulation clock start
    start_month: str = "202602"                   # earliest month to allocate (YYYYMM)
    max_allocation_year: int = 2027               # stop scheduling beyond this year
    max_delay_days: int = 200                     # days to search for capacity before failing

    # ---- demand & throughput ----------------------------------------------
    demand_fulfillment_buffer: float = 1.1        # build enough lots for 110% of demand
    max_steps_per_lot_per_day: int = 2            # normal throughput per lot
    max_steps_per_lot_per_day_fast_track: int = 5 # throughput for fast-tracked lots
    fast_track_lots_per_day: int = 200            # daily fast-track quota
    fast_track_priority_threshold: int = 10000    # priority <= this => fast-track eligible

    # ---- WIP handling ------------------------------------------------------
    wip_lead_time_buffer_factor: float = 1.2
    use_dynamic_wip_earliest_start: bool = False

    # ---- per-equipment-group capacity overrides (sheets/day) ---------------
    # e.g. {"LDI 노광": 5000, "AOI(Scan)": 12000}.  >= 1,000,000 => treated as infinite.
    equipment_capacity_overrides: Dict[str, int] = field(default_factory=dict)

    # ---- fallback defaults (used only when source data is missing) ---------
    default_daily_capacity_sheets: int = 10
    default_model_priority: int = 99_999_999
    default_lead_time_days: int = 30
    default_daily_capacity_lots: int = 2
    default_panels_per_lot: int = 30
    default_panels_per_sheet: int = 6
    default_units_per_panel: int = 2400
    default_units_per_sheet: int = 14400
    infinite_capacity_sheets: int = 10_000_000


# ==============================================================================
# 2.  INPUT READERS   🔌 CONNECTION PLACEHOLDER  (you fill these in)
#     Replace each placeholder with a pycelonis read from your Data Pool.
#     Each must return a polars DataFrame. The expected columns are listed so
#     you know what each table needs to contain.
# ==============================================================================

def _todo_table(name: str, columns: List[str]) -> pl.DataFrame:
    """Returns an EMPTY table with the right column names, so the structure is
    valid before you wire up the real Celonis reads. Replace calls to this."""
    print(f"   🔌 TODO: connect input '{name}'  (expected columns: {columns})")
    return pl.DataFrame({c: [] for c in columns})


def read_inputs(params: SimulationParams) -> Dict[str, pl.DataFrame]:
    """
    🔌 CONNECTION PLACEHOLDER
    Pull every source table from Celonis here. For each one, replace
    `_todo_table(...)` with something like:

        pandas_df = data_pool.get_table("YOUR_TABLE").to_pandas()   # pycelonis
        return pl.from_pandas(pandas_df)

    (Exact pycelonis call depends on your setup — that's the part you'll fill in.)
    """
    return {
        # --- core demand / plan ---
        "revenue_plan":          _todo_table("revenue_plan", ["model_id", "plan_month", "revenue_plan_id", "demand_qty_ea", "amount_krw", "margin_krw"]),
        "available_inventory":   _todo_table("available_inventory", ["model_id", "available_qty_ea"]),

        # --- WIP & routing ---
        "wip_lots":              _todo_table("wip_lots", ["lot_id", "model_id", "process_id", "sequence", "equipment_group_id", "latest_sheet_quantity", "latest_unit_quantity", "remaining_steps", "final_work_sequence"]),
        "planned_process_steps": _todo_table("planned_process_steps", ["model_id", "process_sequence", "process_id", "std_time_per_panel"]),

        # --- model master / conversions / priorities ---
        "model_master":          _todo_table("model_master", ["model_id", "grouping_model", "customer_name", "end_customer", "sales_team", "lead_time_days"]),
        "model_unit_conversion": _todo_table("model_unit_conversion", ["model_id", "panels_per_sheet", "units_per_panel", "units_per_sheet", "panels_per_lot"]),
        "model_priorities":      _todo_table("model_priorities", ["simulation_id", "simulation_name", "revenue_plan_id", "model_id", "plan_month", "priority_rank", "__is_deleted"]),

        # --- equipment ---
        "equipment_capacity":    _todo_table("equipment_capacity", ["equipment_id", "equipment_group_id", "process_id", "daily_capacity_sht", "site_id"]),
        "equipment_constraints": _todo_table("equipment_constraints", ["model_id", "process_id", "equipment_id", "is_blocked"]),
        "equipment_to_process":  _todo_table("equipment_to_process", ["process_id", "equipment_id", "equipment_group_id"]),

        # --- materials (for material depletion step) ---
        "material_inventories":  _todo_table("material_inventories", ["material_id", "on_hand_qty"]),
        "model_boms":            _todo_table("model_boms", ["model_id", "material_id", "qty_per_unit"]),
        "planned_material_arrivals": _todo_table("planned_material_arrivals", ["material_id", "arrival_date", "arrival_qty"]),

        # --- ET jig (for ET jig risk step) ---
        "et_jig_master":         _todo_table("et_jig_master", ["model_id", "lots_per_jig", "jig_units"]),

        # --- external risk input to the fulfillment step ---
        # NOTE: in Palantir this was RID 704c37b3 ("new_model_demand_risk"); it is
        # NOT produced by the engine. Identify its Celonis source, or if it turns
        # out to be the engine's own `unrouted` output, pass that instead.
        "new_model_demand_risk": _todo_table("new_model_demand_risk", ["model_id", "demand_qty_ea", "shortfall_month"]),
    }


# ==============================================================================
# 3.  PIPELINE STEPS  🧠 LOGIC PLACEHOLDER  (port from your existing files)
#     Each function below = one former Foundry transform. The signature (inputs
#     and outputs) is already correct and matches the data flow. The BODY is what
#     gets ported next, one file at a time, from the file named in each docstring.
# ==============================================================================

@dataclass
class AllocationResult:
    """The 5 tables the core engine produces."""
    allocation:  pl.DataFrame    # allocation_output
    new_lots:    pl.DataFrame    # new_lots_created
    unrouted:    pl.DataFrame    # unrouted_model_demand
    failed:      pl.DataFrame    # failed_allocations
    run_tracker: pl.DataFrame    # run_tracker


def compute_net_production_demand(revenue_plan: pl.DataFrame,
                                  available_inventory: pl.DataFrame,
                                  params: SimulationParams) -> pl.DataFrame:
    """🧠 PORT FROM: net_production_demand.py
    In : revenue_plan, available_inventory
    Out: net_demand  (model_id, plan_month, net_production_demand_ea, revenue_plan_id, ...)
    Demand minus on-hand inventory = what actually has to be produced."""
    raise NotImplementedError("Port logic from net_production_demand.py")


def run_allocation_engine(wip_lots: pl.DataFrame,
                          net_demand: pl.DataFrame,
                          model_priorities: pl.DataFrame,
                          equipment_capacity: pl.DataFrame,
                          equipment_constraints: pl.DataFrame,
                          equipment_to_process: pl.DataFrame,
                          model_master: pl.DataFrame,
                          model_unit_conversion: pl.DataFrame,
                          planned_process_steps: pl.DataFrame,
                          params: SimulationParams) -> AllocationResult:
    """🧠 PORT FROM: revenue_driven_allocation_refactored.py  (+ its helper files:
       config.py, models.py, allocation_helpers.py, virtual_lot_creator.py,
       allocation_run_tracker.py)
    This is the heart of the simulation: month-by-month, priority-driven capacity
    allocation that also creates virtual lots to cover shortfalls.
    Out: AllocationResult(allocation, new_lots, unrouted, failed, run_tracker)."""
    raise NotImplementedError("Port logic from revenue_driven_allocation_refactored.py")


def compute_capacity_shortage(allocation: pl.DataFrame,
                              failed: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """🧠 PORT FROM: capacity_shortage_analysis.py
    Out: (equipment_shortages, lot_waiting_periods)"""
    raise NotImplementedError("Port logic from capacity_shortage_analysis.py")


def compute_material_depletion(allocation: pl.DataFrame,
                               material_inventories: pl.DataFrame,
                               model_boms: pl.DataFrame,
                               planned_material_arrivals: pl.DataFrame
                               ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """🧠 PORT FROM: material_depletion.py
    Out: (material_consumption_events, material_depletion_events,
          constrained_production_lots, material_data_quality_issues)"""
    raise NotImplementedError("Port logic from material_depletion.py")


def compute_et_jig_risk(new_lots: pl.DataFrame,
                        et_jig_master: pl.DataFrame,
                        net_demand: pl.DataFrame,
                        model_priorities: pl.DataFrame,
                        allocation: pl.DataFrame) -> pl.DataFrame:
    """🧠 PORT FROM: et_jig_capacity_risk_analysis.py
    Out: et_jig_risk"""
    raise NotImplementedError("Port logic from et_jig_capacity_risk_analysis.py")


def compute_demand_shortfall(allocation: pl.DataFrame,
                             failed: pl.DataFrame,
                             new_lots: pl.DataFrame,
                             net_demand: pl.DataFrame,
                             model_priorities: pl.DataFrame,
                             equipment_shortages: pl.DataFrame,
                             lot_waiting_periods: pl.DataFrame) -> pl.DataFrame:
    """🧠 PORT FROM: demand_shortfall_analysis.py
    Out: demand_shortfall"""
    raise NotImplementedError("Port logic from demand_shortfall_analysis.py")


def compute_monthly_fulfillment(net_demand: pl.DataFrame,
                                allocation: pl.DataFrame,
                                model_master: pl.DataFrame,
                                model_unit_conversion: pl.DataFrame,
                                model_priorities: pl.DataFrame,
                                demand_shortfall: pl.DataFrame,
                                new_model_demand_risk: pl.DataFrame
                                ) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """🧠 PORT FROM: simulation_monthly_fulfillment.py   (the headline scorecard)
    Out: (fulfillment_wide, fulfillment_long)"""
    raise NotImplementedError("Port logic from simulation_monthly_fulfillment.py")


def compute_production_risk_reconciliation(allocation: pl.DataFrame,
                                           failed: pl.DataFrame,
                                           fulfillment_wide: pl.DataFrame,
                                           et_jig_risk: pl.DataFrame) -> pl.DataFrame:
    """🧠 PORT FROM: production_risk_reconciliation.py
    Out: production_risk_reconciliation"""
    raise NotImplementedError("Port logic from production_risk_reconciliation.py")


# ==============================================================================
# 4.  ORCHESTRATOR  —  runs every step in the correct order, in memory.
#     (This is the bit that used to be Foundry's build engine.)
# ==============================================================================

def run_simulation(params: SimulationParams,
                   inputs: Optional[Dict[str, pl.DataFrame]] = None) -> Dict[str, pl.DataFrame]:
    """Run the full simulation for ONE scenario and return all result tables."""
    print("=== RevPlan simulation: start ===")
    if inputs is None:
        inputs = read_inputs(params)

    # --- Tier 1: net demand ---------------------------------------------------
    net_demand = compute_net_production_demand(
        inputs["revenue_plan"], inputs["available_inventory"], params)

    # --- Tier 2: the core allocation engine -----------------------------------
    eng = run_allocation_engine(
        wip_lots=inputs["wip_lots"],
        net_demand=net_demand,
        model_priorities=inputs["model_priorities"],
        equipment_capacity=inputs["equipment_capacity"],
        equipment_constraints=inputs["equipment_constraints"],
        equipment_to_process=inputs["equipment_to_process"],
        model_master=inputs["model_master"],
        model_unit_conversion=inputs["model_unit_conversion"],
        planned_process_steps=inputs["planned_process_steps"],
        params=params,
    )

    # --- Tier 3: analyses that depend only on the engine output ---------------
    equipment_shortages, lot_waiting_periods = compute_capacity_shortage(
        eng.allocation, eng.failed)

    mat_consumption, mat_depletion, constrained_lots, mat_dq = compute_material_depletion(
        eng.allocation, inputs["material_inventories"], inputs["model_boms"],
        inputs["planned_material_arrivals"])

    et_jig_risk = compute_et_jig_risk(
        eng.new_lots, inputs["et_jig_master"], net_demand,
        inputs["model_priorities"], eng.allocation)

    # --- Tier 4: demand shortfall (needs capacity-shortage results) -----------
    demand_shortfall = compute_demand_shortfall(
        eng.allocation, eng.failed, eng.new_lots, net_demand,
        inputs["model_priorities"], equipment_shortages, lot_waiting_periods)

    # --- Tier 5: monthly fulfillment scorecard (needs demand shortfall) -------
    fulfillment_wide, fulfillment_long = compute_monthly_fulfillment(
        net_demand, eng.allocation, inputs["model_master"],
        inputs["model_unit_conversion"], inputs["model_priorities"],
        demand_shortfall, inputs["new_model_demand_risk"])

    # --- Tier 6: production risk reconciliation (needs fulfillment + et jig) --
    risk_reconciliation = compute_production_risk_reconciliation(
        eng.allocation, eng.failed, fulfillment_wide, et_jig_risk)

    print("=== RevPlan simulation: done ===")

    # Every result table the dashboards/app will read:
    return {
        "allocation":                    eng.allocation,
        "new_lots_created":              eng.new_lots,
        "unrouted_model_demand":         eng.unrouted,
        "failed_allocations":            eng.failed,
        "run_tracker":                   eng.run_tracker,
        "equipment_capacity_shortages":  equipment_shortages,
        "lot_waiting_periods":           lot_waiting_periods,
        "material_consumption_events":   mat_consumption,
        "material_depletion_events":     mat_depletion,
        "constrained_production_lots":   constrained_lots,
        "material_data_quality_issues":  mat_dq,
        "et_jig_capacity_risk":          et_jig_risk,
        "demand_shortfall":              demand_shortfall,
        "monthly_fulfillment_wide":      fulfillment_wide,
        "monthly_fulfillment_long":      fulfillment_long,
        "production_risk_reconciliation": risk_reconciliation,
    }


# ==============================================================================
# 5.  OUTPUT WRITER   🔌 CONNECTION PLACEHOLDER  (you fill this in)
#     Push each result table back to Celonis. Convert polars -> pandas first.
# ==============================================================================

def write_outputs(results: Dict[str, pl.DataFrame], params: SimulationParams) -> None:
    """
    🔌 CONNECTION PLACEHOLDER
    For each result table, push it to your Data Pool. Something like:

        for name, df in results.items():
            pandas_df = df.to_pandas()
            # pycelonis: create/replace the pool table, e.g. f"sim_{name}"
            # data_pool.create_table(pandas_df, table_name=f"sim_{name}", ...)
    """
    for name, df in results.items():
        print(f"   🔌 TODO: write result '{name}'  ({df.height} rows) back to Celonis")


# ==============================================================================
# 6.  ENTRY POINT  —  this is what the button ultimately triggers.
# ==============================================================================

def main(params: Optional[SimulationParams] = None) -> Dict[str, pl.DataFrame]:
    # 🔌 In the real flow, `params` is built from the scenario row the button wrote.
    if params is None:
        params = SimulationParams(
            simulation_name="manual test run",
            # override any tuning knob here, e.g.:
            # max_delay_days=120,
            # equipment_capacity_overrides={"LDI 노광": 5000},
        )

    inputs = read_inputs(params)          # 🔌 connection
    results = run_simulation(params, inputs)
    write_outputs(results, params)        # 🔌 connection
    return results


if __name__ == "__main__":
    # In a Jupyter/MLWB cell you can instead just call:  results = main()
    main()
