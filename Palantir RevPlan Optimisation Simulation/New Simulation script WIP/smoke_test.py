"""
smoke_test.py — run this FIRST in an environment that has polars (e.g. MLWB).
================================================================================
This sandbox blocks PyPI, so the package was validated statically (syntax,
import graph, shim coverage, engine call-site signatures) but NOT executed.
Run this file where polars is installed to confirm it runs end-to-end:

    cd "New Simulation script WIP"
    python smoke_test.py

It runs four graduated tiers and prints a PASS/FAIL line for each:
  T1 imports + Foundry shim         (confident)
  T2 build_config from params       (confident)
  T3 priority toggle (off + on)     (confident)
  T4 full pipeline on synthetic data (the real integration test)

If T4 fails, the traceback points at the exact ported step to look at; the
synthetic fixtures below are deliberately tiny and may need a column tweak for
your real schema — that's expected on first contact with live data.
================================================================================
"""
import sys
import traceback
from datetime import date

import polars as pl

from revplan_engine.run_simulation import (
    SimulationParams, main, run_simulation, build_config,
)
from revplan_engine import priority_generation

# Live-mode guard raises past start_months to the current month, so pinned
# fixture months went stale — use next month (차월), like production does.
_T = date.today()
SMOKE_MONTH = f"{_T.year + (1 if _T.month == 12 else 0)}{(_T.month % 12) + 1:02d}"


def _ok(label):  print(f"  PASS  {label}")
def _bad(label, e):
    print(f"  FAIL  {label}: {type(e).__name__}: {e}")
    traceback.print_exc()


# ------------------------------------------------------------------ T1
def t1_imports():
    from revplan_engine import _foundry_shim as sh
    out = sh.InMemoryOutput()
    out.write_table(pl.DataFrame({"a": [1, 2]}))
    assert out.result.height == 2
    # `.polars("previous")` must return an empty, same-schema frame (first run)
    prev = sh.InMemoryInput(pl.DataFrame({"a": [1]})).polars("previous")
    assert prev.height == 0 and prev.columns == ["a"]
    _ok("T1 imports + shim")


# ------------------------------------------------------------------ T2
def t2_config():
    cfg = build_config(SimulationParams(simulation_id="s1", start_month=SMOKE_MONTH,
                                        max_delay_days=120,
                                        equipment_capacity_overrides={"LDI 노광": 5000}))
    assert cfg.constraints.start_month == SMOKE_MONTH
    assert cfg.constraints.max_delay_days == 120
    assert cfg.equipment_capacity_overrides.get("LDI 노광") == 5000
    _ok("T2 build_config")


# ------------------------------------------------------------------ T3
def t3_priority_toggle():
    # OFF (baseline) — passthrough of supplied MES priorities
    supplied = pl.DataFrame({"model_id": ["M1", "M2"], "month": [SMOKE_MONTH, SMOKE_MONTH],
                             "priority": [1, 2]})
    base = priority_generation.build_model_priorities(
        use_weighted_priority=False, supplied_priorities=supplied)
    assert base.height == 2 and "priority" in base.columns

    # ON (weighted) — derive from a tiny revenue plan
    rp = pl.DataFrame({
        "model_id": ["M1", "M2"], "plan_month": [SMOKE_MONTH, SMOKE_MONTH],
        "revenue_plan_id": ["RP1", "RP1"], "quantity_ea": [100, 100],
        "amount_krw": [1000, 5000], "margin_krw": [100, 400],
    })
    w = priority_generation.build_model_priorities(
        use_weighted_priority=True, supplied_priorities=None, revenue_plan=rp,
        weight_revenue=1.0, weight_margin=0.0, weight_delivery=0.0,
        simulation_id="s1", revenue_plan_id="RP1")
    # M2 has higher revenue/unit -> should rank priority 1
    top = w.filter(pl.col("priority") == 1)["model_id"].to_list()
    assert top == ["M2"], f"expected M2 top, got {top}"
    _ok("T3 priority toggle (off + on)")


# ------------------------------------------------------------------ T4
def _synthetic_inputs():
    """Tiny but schema-complete fixtures matching read_inputs() column names."""
    revenue_plan = pl.DataFrame({
        "model_id": ["M1"], "plan_month": [SMOKE_MONTH], "revenue_plan_id": ["RP1"],
        "quantity_ea": [1000], "amount_krw": [1_000_000], "margin_krw": [200_000],
        "grouping_model": ["M1"], "sales_team": ["ST"], "revenue_type": ["제품"],
    })
    available_inventory = pl.DataFrame({
        "model_id": ["M1"], "grouping_model": ["M1"], "plan_month": [SMOKE_MONTH],
        "shipped_quantity_ea": [0], "onhand_quantity_ea": [0],
        "total_inventory_ea": [0], "total_inventory_sht": [0],
    })
    model_priorities = pl.DataFrame({
        "simulation_id": ["s1"], "simulation_name": ["smoke"], "revenue_plan_id": ["RP1"],
        "model_id": ["M1"], "month": [SMOKE_MONTH], "priority": [1],
        "amount_per_unit": [1000.0], "margin_amount_per_unit": [200.0], "__is_deleted": [False],
    })
    wip_lots = pl.DataFrame({
        "lot_id": ["L1"], "model_id": ["M1"], "process_id": ["P1"], "sequence": [1],
        "equipment_group_id": ["G1"], "latest_sheet_quantity": [10], "latest_unit_quantity": [1440],
        "remaining_steps": [1], "final_work_sequence": [1],
    })
    planned_process_steps = pl.DataFrame({
        "model_id": ["M1"], "grouping_model": ["M1"], "sequence": [1], "process_id": ["P1"],
        "process_name": ["proc1"], "equipment_group_id": ["G1"],
        "is_from_latest_production_plan": [True],
        "q1_panel_in_seconds": [1.0], "mpi_Q1_wait_time_in_seconds": [0.0],
    })
    model_master = pl.DataFrame({
        "model_id": ["M1"], "grouping_model": ["M1"], "customer_name": ["C"], "end_customer": ["EC"],
        "sales_team": ["ST"], "lead_time_days": [30],
    })
    model_unit_conversion = pl.DataFrame({
        "model_id": ["M1"], "panels_per_sheet": [6], "units_per_panel": [2400],
        "units_per_sheet": [14400], "panels_per_lot": [30],
    })
    equipment_capacity = pl.DataFrame({
        "equipment_id": ["E1"], "equipment_group_id": ["G1"], "process_id": ["P1"],
        "daily_capacity_in_sht": [10000], "site_id": ["S1"],
    })
    equipment_constraints = pl.DataFrame(
        schema={"grouping_model": pl.Utf8, "process_id": pl.Utf8,
                "equipment_id": pl.Utf8, "constraint_type": pl.Utf8},
    )
    equipment_to_process = pl.DataFrame({
        "process_id": ["P1"], "equipment_id": ["E1"], "equipment_group_id": ["G1"],
    })
    # et_jig_risk port reads the Palantir source's Korean column names verbatim
    et_jig_master = pl.DataFrame(
        schema={"대상_모델": pl.Utf8, "JIG대수": pl.Int64, "Capa_Lot": pl.Float64,
                "Capa_Sheet": pl.Float64, "JIG_상태": pl.Utf8},
    )
    return {
        "revenue_plan": revenue_plan, "available_inventory": available_inventory,
        "model_priorities": model_priorities, "wip_lots": wip_lots,
        "planned_process_steps": planned_process_steps, "model_master": model_master,
        "model_unit_conversion": model_unit_conversion, "equipment_capacity": equipment_capacity,
        "equipment_constraints": equipment_constraints, "equipment_to_process": equipment_to_process,
        "et_jig_master": et_jig_master,
    }


def t4_full_pipeline():
    params = SimulationParams(simulation_id="s1", simulation_name="smoke",
                              revenue_plan_id="RP1", start_month=SMOKE_MONTH)
    results = run_simulation(params, inputs=_synthetic_inputs())
    expected = {"allocation", "new_lots_created", "unrouted_model_demand", "failed_allocations",
                "run_tracker", "monthly_fulfillment_wide", "monthly_fulfillment_long"}
    missing = expected - set(results)
    assert not missing, f"missing result tables: {missing}"
    print("        result tables:")
    for name, df in results.items():
        print(f"          {name:32s} {df.height:>6} rows")
    _ok("T4 full pipeline on synthetic data")


if __name__ == "__main__":
    fails = 0
    for fn in (t1_imports, t2_config, t3_priority_toggle, t4_full_pipeline):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            _bad(fn.__name__, e); fails += 1
    print("\n" + ("ALL SMOKE TESTS PASSED" if fails == 0 else f"{fails} TIER(S) FAILED"))
    sys.exit(1 if fails else 0)
