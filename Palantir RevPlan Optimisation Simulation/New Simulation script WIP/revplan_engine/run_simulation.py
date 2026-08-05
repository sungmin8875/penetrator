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

ALL ANALYSES ARE PORTED (2026-07-16) — no 🧩 stubs remain. The only markers left:
  🔌 CONNECTION placeholder = YOU wire this to Celonis (pycelonis read/write).

SCOPE (agreed): "Runnable core + hooks", business-logic changes OFF by default.
  Full chain: net demand -> priorities -> allocation + virtual lots ->
  capacity-shortage -> ET-jig risk -> material-depletion -> demand-shortfall ->
  monthly-fulfillment scorecard -> production-risk-reconciliation
  (+ the engine's `unrouted` output).

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
from . import demand_shortfall as _dsf_mod
from . import material_depletion as _md_mod
from . import production_risk_reconciliation as _prr_mod
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


def build_wip_master(allocation: pl.DataFrame, lot_summary) -> pl.DataFrame:
    """SIM_WIPMaster: allocation grain + per-lot ACTUAL history columns (left join).

    Row count is provably preserved (lot_summary is 1 row/lot by construction);
    an empty/missing summary degrades to all-null actual columns, same schema."""
    if allocation is None or allocation.height == 0:
        return pl.DataFrame()
    _actual_cols = ["actual_first_start", "actual_last_event", "actual_steps_done",
                    "actual_last_process", "actual_last_equipment", "actual_sheet_qty",
                    "aps_prodcategory"]
    if lot_summary is None or lot_summary.height == 0:
        return allocation.with_columns([
            pl.lit(None, dtype=pl.Datetime).alias("actual_first_start"),
            pl.lit(None, dtype=pl.Datetime).alias("actual_last_event"),
            pl.lit(None, dtype=pl.Int64).alias("actual_steps_done"),
            pl.lit(None, dtype=pl.Utf8).alias("actual_last_process"),
            pl.lit(None, dtype=pl.Utf8).alias("actual_last_equipment"),
            pl.lit(None, dtype=pl.Float64).alias("actual_sheet_qty"),
            pl.lit(None, dtype=pl.Utf8).alias("aps_prodcategory"),
        ])
    out = allocation.join(lot_summary.unique(subset=["lot_id"]), on="lot_id", how="left")
    _matched = out.filter(pl.col("actual_steps_done").is_not_null())["lot_id"].n_unique()
    print(f"   ✓ SimWIPMaster: {out.height:,} rows; {_matched:,} lots carry actual history")
    return out


def build_stock_master(cons_events, inventories, arrivals, allocation) -> pl.DataFrame:
    """SIM_StockMaster: material x date event ledger (OPENING/CONSUMPTION/ARRIVAL).

    CONSUMPTION rows come from material_consumption_events (running balances +
    chasu already computed there); OPENING = one row per material from OnHand;
    ARRIVAL = PoArrivePlan rows (⚠ purchaser's ESTIMATE, customer 2026-07-30).
    Stamped with the run's allocation_run_id/simulation_id for latest-run views."""
    schema = {
        "material_id": pl.Utf8, "event_date": pl.Date, "event_type": pl.Utf8,
        "quantity": pl.Float64, "running_inventory_after": pl.Float64,
        "running_inventory_after_with_arrivals": pl.Float64, "chasu": pl.Float64,
        "lot_id": pl.Utf8, "model_id": pl.Utf8,
        "simulation_id": pl.Utf8, "allocation_run_id": pl.Utf8,
    }
    _run_id, _sim_id = None, None
    if allocation is not None and allocation.height:
        if "allocation_run_id" in allocation.columns:
            _run_id = allocation["allocation_run_id"][0]
        if "simulation_id" in allocation.columns:
            _sim_id = allocation["simulation_id"][0]

    def _conform(df: pl.DataFrame) -> pl.DataFrame:
        for c, dt in schema.items():
            if c not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=dt).alias(c))
            else:
                df = df.with_columns(pl.col(c).cast(dt, strict=False))
        return df.select(list(schema.keys()))

    parts = []
    if cons_events is not None and getattr(cons_events, "height", 0):
        _m = {k: v for k, v in {"allocated_date": "event_date",
                                "total_consumption": "quantity"}.items()
              if k in cons_events.columns}
        c = cons_events.rename(_m)
        c = c.with_columns(pl.lit("CONSUMPTION").alias("event_type"))
        parts.append(_conform(c))
    if inventories is not None and getattr(inventories, "height", 0):
        o = inventories.group_by("material_id").agg(
            pl.col("onhand_quantity").sum().alias("quantity"))
        o = o.with_columns([pl.lit("OPENING").alias("event_type"),
                            pl.lit(_sim_id, dtype=pl.Utf8).alias("simulation_id"),
                            pl.lit(_run_id, dtype=pl.Utf8).alias("allocation_run_id")])
        parts.append(_conform(o))
    if arrivals is not None and getattr(arrivals, "height", 0):
        a = arrivals.rename({"plan_date": "event_date"} if "plan_date" in arrivals.columns else {})
        a = a.with_columns([pl.lit("ARRIVAL").alias("event_type"),
                            pl.lit(_sim_id, dtype=pl.Utf8).alias("simulation_id"),
                            pl.lit(_run_id, dtype=pl.Utf8).alias("allocation_run_id")])
        parts.append(_conform(a))
    if not parts:
        return pl.DataFrame(schema=schema)
    out = pl.concat(parts, how="vertical").sort(["material_id", "event_date"])
    _n = {t: out.filter(pl.col("event_type") == t).height for t in ("OPENING", "CONSUMPTION", "ARRIVAL")}
    print(f"   ✓ SimStockMaster: {out.height:,} rows "
          f"(OPENING {_n['OPENING']:,} / CONSUMPTION {_n['CONSUMPTION']:,} / ARRIVAL {_n['ARRIVAL']:,})")
    return out


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
        "modified_group",
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

    # ⚠️ MLWB ADDITION (2026-08-04, frontend ask): ModifiedGroup from the routing
    # (o_custom_ModelRoute.ModifiedGroup — read as planned_steps' equipment_group_id)
    # as an EXPLICIT column. The allocation's own equipment_group column is a known
    # MIXTURE (ModifiedGroup for VL/exploded-WIP rows, raw machine codes on fallback
    # paths); modified_group is the clean per-(model, op) routing group.
    process_enrichment = planned_steps_df.select([
        "process_id", "model_id", "grouping_model",
        pl.col("process_name").alias("unit_process_name"),
        pl.col("equipment_group_id").alias("modified_group"),
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
# 4.  ANALYSES — all Palantir analyses now ported (verbatim via the shim);
#     each wrapper below documents its inputs and any MLWB divergences.
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

    ⚠️ MLWB DIVERGENCE (wrapper only): the module's console summary formats
    equipment_group / total_revenue / total_margin with f-string specs that crash
    on None ("unsupported format string passed to NoneType"). In Foundry these were
    never null (weighted economics always ran); here a run without per-unit
    economics leaves them null — so they are filled ("" / 0.0) BEFORE the call,
    which also matches how the outputs looked on runs where economics were present.
    """
    def _print_safe(df):
        fills = {"equipment_group": pl.lit(""), "total_revenue": pl.lit(0.0),
                 "total_margin": pl.lit(0.0)}
        exprs = [pl.col(c).fill_null(v) if c in df.columns else v.alias(c)
                 for c, v in fills.items()
                 if c in df.columns or c != "equipment_group"]
        return df.with_columns(exprs) if exprs else df

    out_shortages, out_waiting = InMemoryOutput(), InMemoryOutput()
    _cs_mod.compute(
        InMemoryInput(_print_safe(allocation)),
        InMemoryInput(_print_safe(failed)),
        out_shortages,
        out_waiting,
    )
    return (out_shortages.result if out_shortages.result is not None else pl.DataFrame(),
            out_waiting.result if out_waiting.result is not None else pl.DataFrame())


def compute_material_depletion(allocation, material_inventories, model_boms, planned_material_arrivals):
    """PORTED FROM material_depletion.py (run verbatim via the shim, 2026-07-16).

    Runs the ENRICHED allocation (needs final_production_units) against the BOM to
    emit 4 tables — the 자재 쇼티지 screen's data (M510N 프리프레그/카파포일 story):
      * material_consumption_events — one row per allocation-step × material, with a
        running inventory before/after and is_shortage(with/without planned arrivals).
      * material_depletion_events   — date-level ≥0→<0 inventory transitions per
        material (WITH arrivals), incl. shortfall qty and the lots depleting it.
      * constrained_production_lots — lots whose steps hit a material shortage, with
        the furthest completable sequence.
      * data_quality_issues         — BOM materials with NO inventory record
        (MISSING_INVENTORY), scoped to models actually in the revenue plan.

    ⚠️ MLWB DIVERGENCES (wrapper only — the ported module is untouched):
      * final_production_units is only populated by the financial enrichment branch;
        when absent/null it falls back to units_produced (Palantir's financial join
        set final_production_units = the lot's units_produced anyway).
      * List columns (depleting_lot_ids, constraining_material_ids, …) are joined to
        comma-separated strings — the Data Pool push can't take polars List columns.
      * chasu (차수) is post-joined onto consumption events (customer meeting
        2026-07-30, 57:11–58:33): 소요 is split per 차수 for the 자재 화면, while
        보유 차감 stays a per-material TOTAL ("총 보유 수량으로 해야지 혼선이
        없다") — which the ported module already does via its per-material running
        balance, so the label adds NO change to any inventory number.
    """
    if allocation.height == 0 or model_boms.height == 0:
        print("   ⏭  material_depletion skipped: "
              f"allocation={allocation.height} rows, model_boms={model_boms.height} rows")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    events = allocation
    if "final_production_units" not in events.columns:
        events = events.with_columns(pl.lit(None, dtype=pl.Int32).alias("final_production_units"))
    if "units_produced" in events.columns:
        events = events.with_columns(
            pl.col("final_production_units").fill_null(pl.col("units_produced")))

    out_cons, out_depl, out_constr, out_dq = (
        InMemoryOutput(), InMemoryOutput(), InMemoryOutput(), InMemoryOutput())
    _md_mod.compute(
        out_cons, out_depl, out_constr, out_dq,
        InMemoryInput(events),
        InMemoryInput(material_inventories),
        InMemoryInput(model_boms),
        InMemoryInput(planned_material_arrivals),
    )

    def _stringify_lists(df):
        if df is None:
            return pl.DataFrame()
        list_cols = [c for c, dt in df.schema.items() if isinstance(dt, pl.List)]
        if list_cols:
            df = df.with_columns([
                pl.col(c).cast(pl.List(pl.Utf8)).list.join(",").alias(c) for c in list_cols])
        return df

    res = list(_stringify_lists(o.result) for o in (out_cons, out_depl, out_constr, out_dq))

    # ── 차수 enrichment (2026-07-30 meeting) ──────────────────────────────────
    # The ported join drops the BOM's chasu column; re-attach it here so each
    # consumption event states WHICH round (1차/2차/…) consumed the material.
    # Purely a label: the running balance above is per-material total, so no
    # inventory number moves (이중 차감 없음 — the meeting's rule holds already).
    cons = res[0]
    if cons.height and "chasu" in model_boms.columns and "material_id" in cons.columns:
        try:
            _lut = (model_boms
                    .select(["model_id", "process_id", "work_sequence", "material_id", "chasu"])
                    .rename({"work_sequence": "sequence"})
                    .unique(subset=["model_id", "process_id", "sequence", "material_id"])
                    .with_columns(pl.col("sequence").cast(cons.schema["sequence"], strict=False)))
            cons = cons.join(_lut, on=["model_id", "process_id", "sequence", "material_id"], how="left")
            _multi = (cons.filter(pl.col("chasu").is_not_null())
                          .unique(subset=["material_id", "chasu"])
                          .group_by("material_id")
                          .agg(pl.col("chasu").n_unique().alias("_n"))
                          .filter(pl.col("_n") > 1).height)
            res[0] = cons
            print(f"   ✓ 차수 enrichment: consumption events carry chasu; "
                  f"{_multi} material(s) consumed across MULTIPLE 차수 — "
                  "inventory still deducted once per material (총 보유 기준)")
        except Exception as ex:  # noqa: BLE001 — a label must never sink the analysis
            print(f"   ⚠ 차수 enrichment skipped ({type(ex).__name__}: {ex}) — "
                  "consumption events emitted without chasu")

    res = tuple(res)
    print(f"   ✓ material_depletion: {res[0].height} consumption events, "
          f"{res[1].height} depletion events, {res[2].height} constrained lots, "
          f"{res[3].height} data-quality issues")
    return res


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
    """PORTED FROM demand_shortfall_analysis.py (run verbatim via the shim, 2026-07-16).

    One row per (simulation, model, target_month) with unmet demand, attributed to
    a capacity cause: INSUFFICIENT_CAPACITY / EQUIPMENT_BLOCKED / NO_VALID_EQUIPMENT /
    LEAD_TIME_AND_ET_JIG_CAPACITY. Its equipment_shortages / lot_waiting_periods
    inputs come from the capacity_shortage port (same run, one tier up), and
    monthly_fulfillment consumes the output for the impossible_to_simulate split —
    which the old empty stub silently pinned to 0.
    Empty result falls back to the stub's minimal schema so fulfillment's hard
    column references keep working either way.
    """
    out = InMemoryOutput()
    _dsf_mod.compute(
        InMemoryInput(allocation),
        InMemoryInput(failed),
        InMemoryInput(new_lots),
        InMemoryInput(net_demand),
        InMemoryInput(model_priorities),
        InMemoryInput(equipment_shortages),
        InMemoryInput(lot_waiting_periods),
        out,
    )
    if out.result is not None and out.result.height >= 0 and out.result.width > 0:
        return out.result
    return pl.DataFrame(schema={
        "simulation_id": pl.Utf8, "revenue_plan_id": pl.Utf8, "model_id": pl.Utf8,
        "primary_reason": pl.Utf8, "target_month": pl.Utf8,
        "failed_units": pl.Int64, "shortfall_margin_impact": pl.Float64,
    })


def compute_production_risk_reconciliation(allocation, failed, fulfillment_wide, et_jig_risk):
    """PORTED FROM production_risk_reconciliation.py (run verbatim via the shim, 2026-07-16).

    The "why did we miss the plan" rollup — one row per risk event, reconciling with
    fulfillment_wide's shortfall (shortfall = production_and_inventory +
    impossible_to_simulate − planned_demand) across 5 categories: FAILED_ALLOCATION,
    FAILED_NO_EQUIPMENT, ET_JIG_CAPACITY_ISSUE, EQUIPMENT_CAPACITY_DELAY,
    LEAD_TIME_SHORTFALL. All four inputs are this run's own outputs — no new sources.
    NOTE the source hard-excludes target_month/month_str == '202601' as its simulation
    boundary; on the 202601 backtest/demo the first month is therefore absent HERE by
    design (fulfillment still covers it).

    ⚠️ MLWB DIVERGENCES (wrapper only — the ported module is untouched):
      * final_production_units / total_revenue / total_margin exist on allocation only
        when the financial-enrichment branch ran — null-filled here otherwise (the
        source read them as nulls from Foundry in the same situation).
      * fulfillment_wide / et_jig_risk frames missing their key columns (a truly empty
        upstream) short-circuit to the module's own empty output schema.
      * List columns (affected_lot_ids, bottleneck_equipment_ids, delay_dates) are
        joined to comma-separated strings for the Data Pool push.
    """
    def _empty_result():
        return pl.DataFrame(schema=_prr_mod.PRODUCTION_RISK_RECONCILIATION_SCHEMA)

    if fulfillment_wide.height == 0 or "month_str" not in fulfillment_wide.columns:
        print("   ⏭  production_risk_reconciliation skipped: fulfillment_wide is empty "
              "(the shortfall reconciliation has no source of truth to reconcile against)")
        return _empty_result()
    if et_jig_risk.height > 0 and "target_month" not in et_jig_risk.columns:
        et_jig_risk = pl.DataFrame()
    if et_jig_risk.height == 0:
        # module filters on these columns before any row access
        et_jig_risk = pl.DataFrame(schema={
            "simulation_id": pl.Utf8, "model_id": pl.Utf8, "target_month": pl.Utf8})

    for col, dtype in (("final_production_units", pl.Int32),
                       ("total_revenue", pl.Float64), ("total_margin", pl.Float64)):
        if col not in allocation.columns:
            allocation = allocation.with_columns(pl.lit(None, dtype=dtype).alias(col))

    out = InMemoryOutput()
    _prr_mod.compute(
        InMemoryInput(allocation),
        InMemoryInput(failed),
        InMemoryInput(fulfillment_wide),
        InMemoryInput(et_jig_risk),
        out,
    )
    res = out.result if out.result is not None else _empty_result()
    list_cols = [c for c, dt in res.schema.items() if isinstance(dt, pl.List)]
    if list_cols:
        res = res.with_columns([
            pl.col(c).cast(pl.List(pl.Utf8)).list.join(",").alias(c) for c in list_cols])
    return res


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

    # ⚠️ MLWB ADDITION (2026-08-04): measured step durations (o_custom_WipHistory)
    # into the outsourced pass-through's duration chain. DARK unless
    # REVPLAN_MEASURED_LT=1 — the install line logs coverage either way.
    from .outsourced_allocation import install_measured_lt
    install_measured_lt(inputs.get("measured_step_durations"))

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

    # --- Tier 3.5: material depletion (BOM × allocation) -----------------------
    #   eng.allocation is already enriched (final_production_units when economics
    #   are present; the wrapper falls back to units_produced otherwise).
    (material_consumption_events, material_depletion_events,
     constrained_production_lots, material_data_quality) = compute_material_depletion(
        eng.allocation,
        inputs.get("material_inventories", pl.DataFrame()),
        inputs.get("model_boms", pl.DataFrame()),
        inputs.get("planned_material_arrivals", pl.DataFrame()),
    )

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

    # --- Tier 6: production risk reconciliation (ported 2026-07-16) -----------
    risk_reconciliation = compute_production_risk_reconciliation(
        eng.allocation, eng.failed, fulfillment_wide, et_jig_risk)

    print("=== RevPlan simulation: done ===")
    # --- ⚠️ MLWB ADDITION (2026-08-05): two frontend master tables ---------------
    # SimWIPMaster  = allocation steps + 1-row-per-lot ACTUAL history (WipHistory).
    #   Join proven in the backend beforehand (SQL check 2026-08-05): lot-level
    #   aggregate -> 1:1, row conservation exact, 87.5% real-lot coverage in the
    #   window. VL lots carry null actual columns by construction (no past).
    # SimStockMaster = material x date ledger: OPENING (OnHand) + CONSUMPTION
    #   (allocation x BOM, from material_consumption_events, chasu-labeled) +
    #   ARRIVAL (PoArrivePlan) rows — the 자재 쇼티지 그래프 backing table.
    wip_master = build_wip_master(eng.allocation, inputs.get("wip_history_lot_summary"))
    stock_master = build_stock_master(
        material_consumption_events, inputs.get("material_inventories"),
        inputs.get("planned_material_arrivals"), eng.allocation)

    results = {
        "allocation": eng.allocation,
        "WIPMaster": wip_master,
        "StockMaster": stock_master,
        "new_lots_created": eng.new_lots,
        "unrouted_model_demand": eng.unrouted,
        "failed_allocations": eng.failed,
        "run_tracker": eng.run_tracker,
        "monthly_fulfillment_wide": fulfillment_wide,
        "monthly_fulfillment_long": fulfillment_long,
        "equipment_capacity_shortages": equipment_shortages,
        "lot_waiting_periods": lot_waiting_periods,
        "et_jig_capacity_risk": et_jig_risk,
        "demand_shortfall": demand_shortfall,
        # material depletion (ported 2026-07-16) — the 자재 쇼티지 screen tables:
        "material_consumption_events": material_consumption_events,
        "material_depletion_events": material_depletion_events,
        "constrained_production_lots": constrained_production_lots,
        "material_data_quality_issues": material_data_quality,
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
