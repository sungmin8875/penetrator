"""
================================================================================
 RevPlan Allocation Simulation — MLWB single-run orchestrator
================================================================================
Replaces the whole Palantir Foundry pipeline with ONE in-memory run:

        main(params)  ->  dict of result tables

WHAT IS PORTED FAITHFULLY
-------------------------
The heavy logic lives in byte-identical copies of the Palantir source (see each
module header); Foundry coupling is isolated in `_foundry_shim`. This file only:
  * builds the run's config/params (incl. the OE-table lookup),
  * drives each ported module, and
  * drops the Foundry-only multi-simulation / incremental / run-tracker-skipping
    machinery (a button click = exactly one scenario).

TWO KINDS OF PLACEHOLDER remain, clearly marked:
  🔌 CONNECTION placeholder = YOU wire this to Celonis (pycelonis read/write).
  🧩 STUB (port next)       = a heavy analysis returned EMPTY so the core runs
                              end-to-end; port it from the named Palantir file
                              when you need it.

SCOPE (agreed): "Runnable core + hooks", business-logic changes OFF by default.
  Core that actually computes: net demand -> priorities -> allocation + virtual
  lots -> monthly-fulfillment scorecard (+ the engine's `unrouted` output).
  Stubbed: capacity-shortage, material-depletion, ET-jig risk, demand-shortfall,
  production-risk-reconciliation.

DATA LIBRARY: polars throughout (same as Palantir). pycelonis usually hands you
pandas — convert at the connection edges only:
        read : pl.from_pandas(pandas_df)
        write: polars_df.to_pandas()
================================================================================
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import polars as pl

# --- ported engine (byte-identical to Palantir source) ------------------------
from . import net_production_demand as _npd_mod
from . import monthly_fulfillment as _mf_mod
from . import et_jig_risk as _ejr_mod
from . import capacity_shortage as _cs_mod
from .allocation_engine import (
    allocate_month_by_month,
    ALLOCATION_OUTPUT_SCHEMA,
    NEW_LOTS_OUTPUT_SCHEMA,
    UNROUTED_MODEL_DEMAND_SCHEMA,
    FAILED_ALLOCATION_SCHEMA,
)
from .allocation_helpers import (
    generate_allocation_run_id,
    build_model_metadata_lookup,
    build_model_process_steps_lookup,
    build_model_priority_lookup,
    build_equipment_capacity_lookup,
    build_process_to_equipment_lookup,
    build_negative_constraints_lookup,
    find_models_with_blocked_equipment_paths,
    summarize_process_coverage,
)
from .allocation_run_tracker import create_config_hash, create_run_record, RUN_TRACKER_SCHEMA
from .config import AllocationConfig, DEFAULT_CONFIG, load_config_from_row
from . import priority_generation
from ._foundry_shim import InMemoryInput, InMemoryOutput

# --- Celonis I/O (all platform coupling lives in celonis_io.py) ---------------
# read_oe_table_row / read_inputs / write_outputs are now REAL pycelonis reads
# (available tables wired, missing/partial ones returned as loud placeholders).
from .celonis_io import read_oe_table_row, read_inputs, write_outputs


# ==============================================================================
# 1.  PARAMETERS — the tuning knobs the button/OE-table row carries.
# ==============================================================================

@dataclass
class SimulationParams:
    # ---- scenario identity ----------------------------------------------------
    simulation_id: Optional[str] = None
    simulation_name: Optional[str] = None
    revenue_plan_id: Optional[str] = None

    # ---- timing & horizon -----------------------------------------------------
    start_date: date = field(default_factory=date.today)
    # Earliest month to allocate (YYYYMM). Defaults to the CURRENT month — and
    # build_config clamps any earlier value up to start_date's month anyway.
    start_month: str = field(default_factory=lambda: date.today().strftime("%Y%m"))
    # None -> config derives the horizon RELATIVE to start_month (start year + 2,
    # see load_config_from_row). Set a year here only to pin an explicit wall —
    # the old fixed 2027 default silently overrode the relative logic.
    max_allocation_year: Optional[int] = None
    max_delay_days: int = 200

    # ---- demand & throughput --------------------------------------------------
    demand_fulfillment_buffer: float = 1.1
    max_steps_per_lot_per_day: int = 2
    max_steps_per_lot_per_day_fast_track: int = 5
    fast_track_lots_per_day: int = 200
    fast_track_priority_threshold: int = 10000

    # ---- WIP handling ---------------------------------------------------------
    wip_lead_time_buffer_factor: float = 1.2
    use_dynamic_wip_earliest_start: bool = False

    # ---- per-equipment-group capacity overrides (sheets/day) ------------------
    equipment_capacity_overrides: Dict[str, int] = field(default_factory=dict)

    # ---- fallback defaults (used only when source data is missing) ------------
    default_daily_capacity_sheets: int = 10
    default_model_priority: int = 99_999_999
    default_lead_time_days: int = 30
    default_daily_capacity_lots: int = 2
    default_panels_per_lot: int = 30
    default_panels_per_sheet: int = 6
    default_units_per_panel: int = 2400
    default_units_per_sheet: int = 14400

    # ==========================================================================
    # HTML business-logic changes — ALL OFF BY DEFAULT (Palantir-faithful).
    # Flip a flag on only once its input data is ready & validated.
    # ==========================================================================
    # Priority rule (customer's default is target-step; weighted is opt-in). See
    # priority_generation.build_model_priorities. "target_step" | "weighted" | "passthrough".
    # use_weighted_priority=True forces "weighted" regardless of this value.
    priority_mode: str = "target_step"
    use_weighted_priority: bool = False     # generate model_priorities from weights
    weight_revenue: float = 0.0             # OE-table: weight_revenue
    weight_margin: float = 0.0              # OE-table: weight_margin
    weight_delivery: float = 0.0            # OE-table: weight_delivery
    prototype_pct: float = 0.0              # OE-table: prototype_pct (시제율)
    prototype_models: frozenset = frozenset()

    def __post_init__(self):
        # Clamp start_month up to start_date's month AT PARAMS CREATION so every
        # consumer agrees — read_inputs pins available_inventory/shipped to
        # params.start_month BEFORE build_config runs, so a clamp that lived only
        # in build_config left inventory pinned to a past month while the engine
        # planned from the current one (observed 2026-07-15: inventory at 202602,
        # engine at 202607). Backdate start_date for a backtest; a future
        # start_month is respected.
        sm = str(self.start_month or "").strip()
        sd_month = self.start_date.strftime("%Y%m")
        if len(sm) == 6 and sm.isdigit() and sm < sd_month:
            print(f"  ⚠ start_month {sm} predates effective_start_date "
                  f"({self.start_date.isoformat()}) — raising to {sd_month} "
                  "(backdate start_date instead for a backtest)")
            self.start_month = sd_month

    # placeholders for the remaining HTML changes (not yet wired):
    enable_lot_de_leveling: bool = False
    enable_positive_constraints: bool = False
    enable_yield_loss: bool = False
    enable_panel_separation: bool = False
    enable_mix_capa: bool = False


def build_config(params: SimulationParams) -> AllocationConfig:
    """Build the engine's AllocationConfig from params, using the SAME construction
    as Palantir (`config.load_config_from_row`) so behavior matches the baseline."""
    # ── Align start_month with the scheduling start (2026-07-15) ────────────────
    # effective_start_date (default: today) is when allocation actually begins
    # placing work, so plan months BEFORE it can only ever complete "late" — they
    # poison the delayed-share KPI by construction (a 202602 start_month run in
    # July marks Feb–Jun demand delayed before a single sheet is scheduled).
    # Clamp start_month up to the month of start_date. A deliberately backdated
    # start_date keeps a past start_month valid (historical backtests); an
    # explicit later start_month still wins.
    start_month = str(params.start_month or "").strip()
    _sd_month = params.start_date.strftime("%Y%m")
    if len(start_month) == 6 and start_month.isdigit() and start_month < _sd_month:
        print(f"  ⚠ start_month {start_month} predates effective_start_date "
              f"({params.start_date.isoformat()}) — raising to {_sd_month}. Months before the "
              "scheduling start can only complete late; backdate start_date instead for a backtest.")
        start_month = _sd_month
    row = {
        "effective_start_date": params.start_date,
        "start_month": start_month,
        "max_allocation_year": params.max_allocation_year,
        "max_delay_days": params.max_delay_days,
        "demand_fulfillment_buffer": params.demand_fulfillment_buffer,
        "max_steps_per_lot_per_day": params.max_steps_per_lot_per_day,
        "max_steps_per_lot_per_day_fast_track": params.max_steps_per_lot_per_day_fast_track,
        "fast_track_lots_per_day": params.fast_track_lots_per_day,
        "fast_track_priority_threshold": params.fast_track_priority_threshold,
        "wip_lead_time_buffer_factor": params.wip_lead_time_buffer_factor,
        "use_dynamic_wip_earliest_start": params.use_dynamic_wip_earliest_start,
        "force_snapshot": False,
        "default_daily_capacity_sheets": params.default_daily_capacity_sheets,
        "default_model_priority": params.default_model_priority,
        "default_lead_time_days": params.default_lead_time_days,
        "default_daily_capacity_lots": params.default_daily_capacity_lots,
        "default_panels_per_lot": params.default_panels_per_lot,
        "default_panels_per_sheet": params.default_panels_per_sheet,
        "default_units_per_panel": params.default_units_per_panel,
        "default_units_per_sheet": params.default_units_per_sheet,
        "primary_key_": params.simulation_id,
        "revenue_plan_id": params.revenue_plan_id,
    }
    config = load_config_from_row(row, source_description="mlwb_oe_table")
    # Inject the dict-form capacity overrides (the OE table may pass groups that
    # have no dedicated equip_capacity_* column). Frozen dataclass -> replace().
    if params.equipment_capacity_overrides:
        merged = {**config.equipment_capacity_overrides, **params.equipment_capacity_overrides}
        config = dataclasses.replace(config, equipment_capacity_overrides=merged)
    return config


# ==============================================================================
# 2.  PARAMS FROM THE OE-TABLE ROW
#     read_oe_table_row() now lives in celonis_io.py (imported above) and does a
#     real pycelonis read of SIMULATION_OE_Table.
# ==============================================================================

def _as_bool(v) -> bool:
    """Coerce an OE-table cell (bool / number / string) to a boolean — tolerant of the
    string forms a Celonis Knowledge-Model or Action-Flow field tends to carry."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def params_from_oe_row(simulation_id: str, row: Dict) -> SimulationParams:
    """Build SimulationParams from an OE-table row.

    Weighted priority turns ON only via an EXPLICIT `use_weighted_priority` flag on
    the OE row (2026-07-15 meeting: weights judged not practically useful — keep
    disabled; 이봉준 책임 "가중치 추가는 의미 없음"). It was previously INFERRED from
    any non-zero weight, but the Action Flow/frontend always sends default weights
    (e.g. 50/30/20), which silently forced weighted mode on every frontend run.
    The weights are still parsed and stored so an explicit opt-in uses them, and
    the engine's data-availability guard at run time still applies
    (priority_generation.build_model_priorities).
    """
    def g(key, default):
        v = row.get(key)
        return default if v is None else v

    weights = (float(g("weight_revenue", 0)), float(g("weight_margin", 0)), float(g("weight_delivery", 0)))
    explicit = row.get("use_weighted_priority")     # absent unless added to the OE table + read_oe_table_row
    use_weighted = _as_bool(explicit) if explicit is not None else False
    if not use_weighted and sum(weights) > 0:
        print(f"  ℹ weights {weights} received but weighted priority stays OFF "
              "(2026-07-15 decision; opt in with an explicit use_weighted_priority flag)")
    return SimulationParams(
        simulation_id=simulation_id,
        simulation_name=g("scenario_name", None),
        revenue_plan_id=g("revenue_plan_id", None),
        start_month=str(g("start_month", date.today().strftime("%Y%m"))),
        max_delay_days=int(g("max_delay_days", 200)),
        weight_revenue=weights[0],
        weight_margin=weights[1],
        weight_delivery=weights[2],
        prototype_pct=float(g("prototype_pct", 0)),
        use_weighted_priority=use_weighted,
        priority_mode=str(g("priority_mode", "target_step")),
    )


# NOTE: read_inputs() now lives in celonis_io.py (imported above). It pulls each
# source table from the Celonis Data Pool/Model, mapping PPS/RTS column names to the
# engine's names — with available tables WIRED and missing/partial ones returned as
# loud, schema-correct placeholders. See celonis_io.read_inputs for per-table status.


# ==============================================================================
# 3.  PIPELINE STEPS
# ==============================================================================

@dataclass
class AllocationResult:
    allocation: pl.DataFrame      # allocation_output
    new_lots: pl.DataFrame        # new_lots_created
    unrouted: pl.DataFrame        # unrouted_model_demand (== new_model_demand_risk)
    failed: pl.DataFrame          # failed_allocations
    run_tracker: pl.DataFrame     # run_tracker (single-row, single run)


# ---- Tier 1: net demand — byte-identical logic, driven via the shim ----------
def compute_net_production_demand(revenue_plan: pl.DataFrame,
                                  available_inventory: pl.DataFrame) -> pl.DataFrame:
    """PORTED FROM net_production_demand.py (run verbatim via the shim)."""
    out = InMemoryOutput()
    _npd_mod.compute(InMemoryInput(revenue_plan), InMemoryInput(available_inventory), out)
    return out.result if out.result is not None else pl.DataFrame()


# ---- Tier 2: the core allocation engine — single-run faithful distillation ----
def run_allocation_engine(*,
                          wip_lots: pl.DataFrame,
                          net_demand: pl.DataFrame,
                          model_priorities: pl.DataFrame,
                          equipment_capacity: pl.DataFrame,
                          equipment_constraints: pl.DataFrame,
                          equipment_to_process: pl.DataFrame,
                          model_master: pl.DataFrame,
                          model_unit_conversion: pl.DataFrame,
                          planned_process_steps: pl.DataFrame,
                          params: SimulationParams,
                          config: AllocationConfig) -> AllocationResult:
    """Single-scenario distillation of revenue_driven_allocation_refactored.compute().

    Keeps the engine's data prep, the `allocate_month_by_month` call, and the
    post-allocation financial/customer enrichment. DROPS the Foundry-only parts:
    config_lookup over many sims, EXCLUDED_SIMULATIONS, force_snapshot, run-tracker
    skip/early-exit, previous-snapshot preservation, and append/replace write modes.
    """
    run_timestamp = datetime.utcnow()

    priorities_df = model_priorities
    if "__is_deleted" in priorities_df.columns:
        priorities_df = priorities_df.filter(pl.col("__is_deleted") == False)  # noqa: E712

    # --- build lookups (engine compute() lines ~1665-1672) --------------------
    wip_with_equipment = wip_lots.filter(pl.col("process_id").is_not_null())
    equipment_capacity_lookup = build_equipment_capacity_lookup(equipment_capacity, config)
    process_to_equipment_lookup = build_process_to_equipment_lookup(equipment_to_process)
    model_metadata_lookup = build_model_metadata_lookup(model_master, model_unit_conversion, config)
    model_process_steps_lookup = build_model_process_steps_lookup(planned_process_steps, config)
    negative_constraints_lookup = build_negative_constraints_lookup(equipment_constraints, model_metadata_lookup)

    # --- demand dict + blocked-model handling (lines ~1681-1716) --------------
    demand_by_model_month_all: Dict[Tuple[str, str], int] = {}
    for row in net_demand.iter_rows(named=True):
        model = row.get("model_id")
        month = row.get("plan_month")
        qty = row.get("net_production_demand_ea") or 0
        if model and month and qty > 0:
            demand_by_model_month_all[(model, month)] = qty

    # ── Guard: does the selected plan cover any month the engine will process? ──
    # allocate_month_by_month only processes months >= start_month. A stale plan
    # revision (e.g. a January H1 plan run in July) leaves NOTHING to plan — the
    # run then quietly produces urgent-prepass rows only, 0 virtual lots
    # (observed 2026-07-15: "Processing months in order: []"). Say it loudly.
    _plan_months = sorted({m for (_, m) in demand_by_model_month_all.keys()})
    _live_months = [m for m in _plan_months if m >= config.constraints.start_month]
    if _plan_months and not _live_months:
        print(f"\n  ⛔ SELECTED PLAN HAS NO MONTHS AT/AFTER start_month="
              f"{config.constraints.start_month}: plan covers {_plan_months[0]}–{_plan_months[-1]} "
              f"({len(_plan_months)} months, ALL in the past).\n"
              "     Tier-2 allocation will process NOTHING (urgent pre-pass only, 0 virtual lots).\n"
              "     → Pick a CURRENT plan revision in the frontend, or backdate start_date "
              "to simulate this plan as of its own period (backtest).\n")

    demand_model_ids = set(model for model, _ in demand_by_model_month_all.keys())
    blocked_models = find_models_with_blocked_equipment_paths(
        demand_model_ids=demand_model_ids,
        model_process_steps_lookup=model_process_steps_lookup,
        process_to_equipment=process_to_equipment_lookup,
        negative_constraints=negative_constraints_lookup,
    )
    blocked_demand: Dict[Tuple[str, str], Tuple[int, str, str]] = {}
    for (model, month), qty in demand_by_model_month_all.items():
        if model in blocked_models:
            process_id, reason = blocked_models[model]
            blocked_demand[(model, month)] = (qty, process_id, reason)
    demand_by_model_month = {
        (model, month): qty
        for (model, month), qty in demand_by_model_month_all.items()
        if model not in blocked_models
    }

    # --- single simulation (lines ~1727-1808) ---------------------------------
    simulation_id = params.simulation_id
    simulation_name = params.simulation_name
    revenue_plan_id = params.revenue_plan_id
    run_id = generate_allocation_run_id(run_timestamp, simulation_id or "mlwb_run")
    model_priority_lookup = build_model_priority_lookup(priorities_df)

    # apply equipment capacity overrides by substring match (lines ~1758-1765)
    if config.equipment_capacity_overrides:
        for equip_group, capacity in config.equipment_capacity_overrides.items():
            for equip_id in list(equipment_capacity_lookup.keys()):
                if equip_group.lower() in equip_id.lower():
                    equipment_capacity_lookup[equip_id] = capacity

    allocation_df, new_lots_df, unrouted_df, failed_df = allocate_month_by_month(
        wip_with_equipment=wip_with_equipment,
        demand_by_model_month=demand_by_model_month,
        model_priority_lookup=model_priority_lookup,
        model_metadata_lookup=model_metadata_lookup,
        model_process_steps_lookup=model_process_steps_lookup,
        equipment_capacity=equipment_capacity_lookup,
        process_to_equipment=process_to_equipment_lookup,
        negative_constraints=negative_constraints_lookup,
        revenue_plan_id=revenue_plan_id,
        simulation_id=simulation_id,
        simulation_name=simulation_name,
        run_id=run_id,
        run_timestamp=run_timestamp,
        blocked_demand=blocked_demand,
        config=config,
    )

    # --- schema consistency (lines ~1909-1924, single frame) ------------------
    def ensure_schema(df: pl.DataFrame, schema: Dict) -> pl.DataFrame:
        if df.height == 0:
            return df
        for col_name, col_type in schema.items():
            if col_name in df.columns and df[col_name].dtype != col_type:
                df = df.with_columns(pl.col(col_name).cast(col_type))
        return df

    allocation_df = ensure_schema(allocation_df, ALLOCATION_OUTPUT_SCHEMA)
    new_lots_df = ensure_schema(new_lots_df, NEW_LOTS_OUTPUT_SCHEMA)
    unrouted_df = ensure_schema(unrouted_df, UNROUTED_MODEL_DEMAND_SCHEMA)
    failed_df = ensure_schema(failed_df, FAILED_ALLOCATION_SCHEMA)

    print(f"\n=== ALLOCATION SUMMARY ===")
    print(f"  allocation rows : {allocation_df.height:,}")
    print(f"  failed          : {failed_df.height:,}")
    print(f"  virtual lots    : {new_lots_df.height:,}")
    print(f"  unrouted demand : {unrouted_df.height:,}")

    # --- enrichment (lines ~1961-2044) ----------------------------------------
    allocation_df = _enrich_allocation(allocation_df, model_master, planned_process_steps, priorities_df)

    # --- single-row run tracker (faithful record, no skip logic) --------------
    config_hash = create_config_hash(config)
    run_record = create_run_record(
        simulation_id=simulation_id,
        revenue_plan_id=revenue_plan_id,
        simulation_name=simulation_name,
        config=config,
        config_hash=config_hash,
        run_id=run_id,
        run_timestamp=run_timestamp,
        status="SUCCESS",
        row_counts={
            "allocation": allocation_df.height,
            "new_lots": new_lots_df.height,
            "unrouted": unrouted_df.height,
            "failed": failed_df.height,
        },
    )
    run_tracker_df = pl.DataFrame([run_record], schema=RUN_TRACKER_SCHEMA)

    return AllocationResult(allocation_df, new_lots_df, unrouted_df, failed_df, run_tracker_df)


def _enrich_allocation(combined_allocations: pl.DataFrame,
                       model_df: pl.DataFrame,
                       planned_steps_df: pl.DataFrame,
                       priorities_df: pl.DataFrame) -> pl.DataFrame:
    """Faithful copy of the engine's Step-7 enrichment. Guards the financial join
    so a baseline priorities table without per-unit economics doesn't crash the
    runnable core (those columns then come through null, as in Foundry)."""
    if combined_allocations.height == 0:
        return combined_allocations

    columns_to_drop = [
        "model_customer_name", "model_end_customer", "model_sales_team", "grouping_model",
        "unit_process_name", "simulation_name", "simulation_revenue_id",
        "final_production_units", "total_revenue", "total_margin",
    ]
    existing = [c for c in columns_to_drop if c in combined_allocations.columns]
    if existing:
        combined_allocations = combined_allocations.drop(existing)

    model_enrichment = model_df.select([
        "model_id",
        pl.col("customer_name").alias("model_customer_name"),
        pl.col("end_customer").alias("model_end_customer"),
        pl.col("sales_team").alias("model_sales_team"),
    ]).unique(subset=["model_id"], maintain_order=True)

    process_enrichment = planned_steps_df.select([
        "process_id", "model_id", "grouping_model",
        pl.col("process_name").alias("unit_process_name"),
    ]).unique(subset=["process_id", "model_id"], maintain_order=True)

    simulation_enrichment = priorities_df.select([
        "simulation_id",
        pl.col("simulation_name"),
        pl.col("revenue_plan_id").alias("simulation_revenue_id"),
    ]).unique(subset=["simulation_id"], maintain_order=True)

    combined_allocations = (
        combined_allocations.join(model_enrichment, on="model_id", how="left")
        .join(process_enrichment, on=["process_id", "model_id"], how="left")
        .join(simulation_enrichment, on="simulation_id", how="left")
    )

    # Financial enrichment — only if the priorities table carries per-unit economics.
    if {"amount_per_unit", "margin_amount_per_unit", "month"}.issubset(set(priorities_df.columns)):
        financial_lookup = priorities_df.select([
            "simulation_id", "model_id", "month",
            pl.col("margin_amount_per_unit").alias("margin_per_unit"),
            pl.col("amount_per_unit").alias("revenue_per_unit"),
        ]).unique(subset=["simulation_id", "model_id", "month"], maintain_order=True)

        completed_lots = combined_allocations.select(
            "lot_id", "simulation_id", "model_id", "target_month", "units_produced"
        ).filter(pl.col("units_produced") > 0)

        financial_info = (
            completed_lots.join(
                financial_lookup,
                left_on=["simulation_id", "model_id", "target_month"],
                right_on=["simulation_id", "model_id", "month"],
                how="left",
            )
            .with_columns([
                (pl.col("units_produced") * pl.col("revenue_per_unit")).alias("total_revenue"),
                (pl.col("units_produced") * pl.col("margin_per_unit")).alias("total_margin"),
            ])
            .rename({"units_produced": "final_production_units"})
            .drop("revenue_per_unit", "margin_per_unit", "model_id", "target_month")
        )
        combined_allocations = combined_allocations.join(financial_info, on=["simulation_id", "lot_id"], how="left")
        print("  ✓ enriched allocation with customer + financial columns")
    else:
        print("  ⚠️ priorities table lacks per-unit economics — financial columns left null "
              "(set use_weighted_priority=True or supply amount_per_unit/margin_amount_per_unit)")

    return combined_allocations


# ---- Tier 5: monthly fulfillment scorecard — byte-identical via the shim -----
def compute_monthly_fulfillment(*,
                                net_demand: pl.DataFrame,
                                allocation: pl.DataFrame,
                                model_master: pl.DataFrame,
                                model_unit_conversion: pl.DataFrame,
                                model_priorities: pl.DataFrame,
                                demand_shortfall: pl.DataFrame,
                                new_model_demand_risk: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """PORTED FROM simulation_monthly_fulfillment.py (run verbatim via the shim).

    Its `revenue_optimization_events` input is the engine's `allocation` output
    (same Foundry RID b84b2bb5), and `new_model_demand_risk` is the engine's own
    `unrouted` output (confirmed last session)."""
    out_wide, out_long = InMemoryOutput(), InMemoryOutput()
    _mf_mod.compute(
        out_wide, out_long,
        InMemoryInput(net_demand),
        InMemoryInput(allocation),
        InMemoryInput(model_master),
        InMemoryInput(model_unit_conversion),
        InMemoryInput(model_priorities),
        InMemoryInput(demand_shortfall),
        InMemoryInput(new_model_demand_risk),
    )
    return (out_wide.result if out_wide.result is not None else pl.DataFrame(),
            out_long.result if out_long.result is not None else pl.DataFrame())


# ==============================================================================
# 4.  🧩 STUBS — heavy analyses returned EMPTY so the core runs end-to-end.
#     Port each from the named Palantir file when you need it.
# ==============================================================================

def compute_capacity_shortage(allocation, failed):
    """PORTED FROM capacity_shortage_analysis.py (run verbatim via the shim, 2026-07-16).

    Inputs are the engine's own allocation + failed_allocations outputs (the same
    Foundry datasets the source read). Produces:
      * equipment_shortages  — contiguous per-equipment-group shortage periods with
        lot counts, delay-day totals and revenue/margin at risk (the "which op is
        the bottleneck, when, how bad" table — e.g. the M710N story).
      * lot_waiting_periods  — one row per lot × equipment group wait, delayed and
        failed lots both, with parsed blocked/full equipment lists.
    NO_EQUIPMENT failures are excluded from shortages by the source itself (config
    issues, not capacity) — outsourced-assumed steps never fail, so post-2026-07-15
    the failed input is capacity-only anyway.
    """
    out_shortages, out_waiting = InMemoryOutput(), InMemoryOutput()
    _cs_mod.compute(
        InMemoryInput(allocation),
        InMemoryInput(failed),
        out_shortages,
        out_waiting,
    )
    return (out_shortages.result if out_shortages.result is not None else pl.DataFrame(),
            out_waiting.result if out_waiting.result is not None else pl.DataFrame())


def compute_material_depletion(allocation, material_inventories, model_boms, planned_material_arrivals):
    """🧩 PORT NEXT from material_depletion.py -> 4 tables."""
    print("   🧩 STUB: material_depletion (returning empty)")
    return pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), pl.DataFrame()


def compute_et_jig_risk(new_lots, et_jig_master, net_demand, model_priorities, allocation):
    """PORTED FROM et_jig_capacity_risk_analysis.py (run verbatim via the shim).

    NOTE the arg order: et_jig's compute() takes INPUTS first and its single OUTPUT LAST
    (unlike monthly_fulfillment, whose outputs come first). Empty et_jig_master → empty result
    (the analysis returns a schema-correct empty frame when no lots are at risk)."""
    out = InMemoryOutput()
    _ejr_mod.compute(
        InMemoryInput(new_lots),
        InMemoryInput(et_jig_master),
        InMemoryInput(net_demand),
        InMemoryInput(model_priorities),
        InMemoryInput(allocation),
        out,
    )
    return out.result if out.result is not None else pl.DataFrame()


def compute_demand_shortfall(allocation, failed, new_lots, net_demand, model_priorities,
                             equipment_shortages, lot_waiting_periods):
    """🧩 PORT NEXT from demand_shortfall_analysis.py -> demand_shortfall.

    NOTE: monthly_fulfillment consumes this. Empty is fine for the core run, but it
    hard-references these columns, so the empty frame MUST carry their schema."""
    print("   🧩 STUB: demand_shortfall_analysis (returning empty)")
    return pl.DataFrame(schema={
        "simulation_id": pl.Utf8, "revenue_plan_id": pl.Utf8, "model_id": pl.Utf8,
        "primary_reason": pl.Utf8, "target_month": pl.Utf8,
        "failed_units": pl.Int64, "shortfall_margin_impact": pl.Float64,
    })


def compute_production_risk_reconciliation(allocation, failed, fulfillment_wide, et_jig_risk):
    """🧩 PORT NEXT from production_risk_reconciliation.py -> production_risk_reconciliation."""
    print("   🧩 STUB: production_risk_reconciliation (returning empty)")
    return pl.DataFrame()


# ==============================================================================
# 5.  ORCHESTRATOR — run every step in order, in memory.
# ==============================================================================

def run_simulation(params: SimulationParams,
                   inputs: Optional[Dict[str, pl.DataFrame]] = None) -> Dict[str, pl.DataFrame]:
    print("=== RevPlan simulation: start ===")
    if inputs is None:
        inputs = read_inputs(params)

    config = build_config(params)
    print(f"  config_source={config.config_source}  start_month={config.constraints.start_month}  "
          f"max_delay_days={config.constraints.max_delay_days}  "
          f"priority_mode={'weighted' if params.use_weighted_priority else params.priority_mode}")

    # Diagnostic (no behaviour change): routing-op equipment coverage + logical/blocked split.
    summarize_process_coverage(
        inputs["planned_process_steps"],
        inputs["equipment_to_process"],
        config.constraints.logical_passthrough_operation_codes,
    )

    # --- Tier 1: net demand ---------------------------------------------------
    net_demand = compute_net_production_demand(inputs["revenue_plan"], inputs["available_inventory"])

    # --- Tier 1b: priorities (default target_step; weighted is opt-in via the toggle) -
    model_priorities = priority_generation.build_model_priorities(
        use_weighted_priority=params.use_weighted_priority,
        priority_mode=params.priority_mode,
        supplied_priorities=inputs.get("model_priorities"),
        net_demand=net_demand,
        revenue_plan=inputs["revenue_plan"],
        planned_process_steps=inputs["planned_process_steps"],
        effective_start_date=config.effective_start_date,
        weight_revenue=params.weight_revenue,
        weight_margin=params.weight_margin,
        weight_delivery=params.weight_delivery,
        prototype_pct=params.prototype_pct,
        prototype_models=set(params.prototype_models),
        simulation_id=params.simulation_id,
        simulation_name=params.simulation_name,
        revenue_plan_id=params.revenue_plan_id,
    )

    # --- Tier 2: core allocation engine + virtual lots ------------------------
    eng = run_allocation_engine(
        wip_lots=inputs["wip_lots"],
        net_demand=net_demand,
        model_priorities=model_priorities,
        equipment_capacity=inputs["equipment_capacity"],
        equipment_constraints=inputs["equipment_constraints"],
        equipment_to_process=inputs["equipment_to_process"],
        model_master=inputs["model_master"],
        model_unit_conversion=inputs["model_unit_conversion"],
        planned_process_steps=inputs["planned_process_steps"],
        params=params,
        config=config,
    )

    # --- Tier 3: analyses (capacity_shortage stubbed; et_jig_risk wired) ------
    equipment_shortages, lot_waiting_periods = compute_capacity_shortage(eng.allocation, eng.failed)
    et_jig_risk = compute_et_jig_risk(
        eng.new_lots, inputs["et_jig_master"], net_demand, model_priorities, eng.allocation)

    # --- Tier 4: demand shortfall (stubbed) -----------------------------------
    demand_shortfall = compute_demand_shortfall(
        eng.allocation, eng.failed, eng.new_lots, net_demand, model_priorities,
        equipment_shortages, lot_waiting_periods)

    # --- Tier 5: monthly fulfillment scorecard (CORE) -------------------------
    #   new_model_demand_risk == the engine's own `unrouted` output.
    fulfillment_wide, fulfillment_long = compute_monthly_fulfillment(
        net_demand=net_demand,
        allocation=eng.allocation,
        model_master=inputs["model_master"],
        model_unit_conversion=inputs["model_unit_conversion"],
        model_priorities=model_priorities,
        demand_shortfall=demand_shortfall,
        new_model_demand_risk=eng.unrouted,
    )

    # --- Tier 6: production risk reconciliation (stubbed) ---------------------
    risk_reconciliation = compute_production_risk_reconciliation(
        eng.allocation, eng.failed, fulfillment_wide, et_jig_risk)

    print("=== RevPlan simulation: done ===")
    results = {
        "allocation": eng.allocation,
        "new_lots_created": eng.new_lots,
        "unrouted_model_demand": eng.unrouted,
        "failed_allocations": eng.failed,
        "run_tracker": eng.run_tracker,
        "monthly_fulfillment_wide": fulfillment_wide,
        "monthly_fulfillment_long": fulfillment_long,
        # stubbed analyses (empty until ported):
        "equipment_capacity_shortages": equipment_shortages,
        "lot_waiting_periods": lot_waiting_periods,
        "et_jig_capacity_risk": et_jig_risk,
        "demand_shortfall": demand_shortfall,
        "production_risk_reconciliation": risk_reconciliation,
    }

    # --- Stamp organization_code onto every output keyed by model_id --------------
    # The OCDM Model object's ID = 'Model_' || ORGANIZATION_CODE || MODEL_NO, but the engine
    # keys models on model_id (= MODEL_NO) only. Attach the org from model_master so each SIM_
    # table can build the Model FK ('Model_' || "organization_code" || "model_id") WITHOUT a
    # join in the object transformation. run_tracker (no model_id) and empty frames are untouched.
    _mm = inputs.get("model_master")
    if _mm is not None and "organization_code" in _mm.columns:
        _org_lookup = _mm.select(["model_id", "organization_code"]).unique(
            subset=["model_id"], maintain_order=True)

        def _with_org(df):
            if df is None or df.height == 0 or "model_id" not in df.columns:
                return df
            if "organization_code" in df.columns:
                df = df.drop("organization_code")
            return df.join(_org_lookup, on="model_id", how="left")

        results = {name: _with_org(df) for name, df in results.items()}
    return results


# NOTE: write_outputs() now lives in celonis_io.py (imported above). It pushes each
# result table to the Data Pool as {OUTPUT_PREFIX}<name>, APPEND-ONLY: tables are
# created on the first run and appended thereafter, so SIM_* keeps every run's rows
# (select runs via allocation_run_id; SIM_run_tracker is the run registry).


# ==============================================================================
# 7.  ENTRY POINT — what the button ultimately triggers.
# ==============================================================================

def main(params: Optional[SimulationParams] = None,
         simulation_id: Optional[str] = None) -> Dict[str, pl.DataFrame]:
    """Run one scenario.

    Real flow: the notebook is triggered with {dpInstanceId}. Pass it as
    `simulation_id`; this reads the OE-table row and builds params. For manual
    testing, pass a `params` object directly.
    """
    if params is None:
        if simulation_id is not None:
            row = read_oe_table_row(simulation_id)          # 🔌 connection
            params = params_from_oe_row(simulation_id, row)
        else:
            params = SimulationParams(simulation_name="manual test run")

    inputs = read_inputs(params)                            # 🔌 connection
    results = run_simulation(params, inputs)
    write_outputs(results, params)                          # 🔌 connection
    return results


if __name__ == "__main__":
    # In an MLWB cell you can instead just call:  results = main(simulation_id="...")
    main()
