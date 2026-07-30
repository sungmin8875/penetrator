# RevPlan Allocation Simulation — MLWB port

Single-scenario, in-memory port of the Palantir Foundry RevPlan "Scenario-Based
Simulation" (UC2-2) capacity-allocation pipeline, for running inside a Celonis
MLWB notebook. One button click = one `main()` call = one scenario.

## How it's structured

| File | What it is |
|---|---|
| `run_simulation.py` | **Entry point + orchestrator.** OE-table lookup, param/config build, runs every step in memory, stubs. Celonis I/O is imported from `celonis_io.py`. |
| `celonis_io.py` | **New.** All Celonis (pycelonis 2.x) coupling: connect, `read_oe_table_row`, `read_inputs` (PPS/RTS→engine column mapping; available tables wired, missing/partial as placeholders), `write_outputs`. Same isolation role for Celonis that `_foundry_shim.py` plays for Foundry. |
| `priority_generation.py` | **New.** Weighted-priority generation (revenue/margin/delivery + prototype). **OFF by default.** |
| `outsourced_allocation.py` | **New.** Unmapped-op = 외주 pass-through (customer 2026-07-15 §4): planned-complete after Plan LT, `equipment_id='OUTSOURCED'`, run-log audit of assumed op codes. **ON by default** (`treat_unmapped_as_outsourced=False` restores the old FAILED_NO_EQUIPMENT). |
| `_foundry_shim.py` | **New.** No-op stand-ins for `transforms.api` + in-memory adapters so the ported `compute()`s run outside Foundry. |
| `allocation_engine.py` | Copy of `revenue_driven_allocation_refactored.py`; every departure from the source is a marked `⚠️ MLWB ADDITION/DIVERGENCE` block (outsourced hook, urgent pre-pass, opt-in scan memo — see the Runtime section). |
| `allocation_helpers.py`, `virtual_lot_creator.py`, `allocation_run_tracker.py`, `config.py`, `models.py` | Byte-identical copies of the engine's pure helpers. |
| `net_production_demand.py`, `monthly_fulfillment.py` | Byte-identical copies; driven via the shim. |

The heavy logic modules are **byte-for-byte the Palantir source** — only their
import lines were rewritten (`transforms.api` → `_foundry_shim`, `myproject.…` →
relative). So they can be **re-synced** if the Palantir engine changes, and the
allocation math is guaranteed identical to the baseline.

## What runs vs. what's stubbed (agreed scope: "runnable core + hooks")

**Core (actually computes):** net production demand → model priorities → allocation
engine + virtual lots → monthly-fulfillment scorecard (+ the engine's `unrouted`
output, which feeds the scorecard as `new_model_demand_risk`).

**All five analyses are now ported and live** (2026-07-16): `capacity_shortage_analysis`,
`material_depletion`, `et_jig_capacity_risk_analysis`, `demand_shortfall_analysis`,
`production_risk_reconciliation`. Zero stubs remain.

## ⚡ Runtime: saturation scan memo (`REVPLAN_SCAN_MEMO=1`, OFF by default)

**The problem (diagnosed 2026-07-30).** When demand outruns capacity, the stock
engine re-discovers the same "no room" verdict tens of thousands of times: every
lot that can't be placed walks `max_delay_days` days × every candidate machine,
and every virtual lot in a shortfall batch (a 600k-unit shortfall ≈ 100 VLs)
repeats the identical doomed walk. Measured: a step that allocates costs ~23 µs;
a doomed 90-day scan costs ~1,000–7,000 µs. After the NotuseFlag equipment filter
removed ~500 phantom machines (2026-07-16), the simulated factory genuinely
saturates and runs went from ~20 min to overnight. On top of that, VLs are cut at
~30 sheets while the median machine `DailyCapa` is 5 — most VL steps physically
cannot fit **any** machine on **any** day, yet each one burned a full scan.

**The fix — two layers, both sound because `capacity_usage` only ever GROWS
within a run** (a day proven unable to fit q sheets stays unable to fit ≥ q):

1. **Oversize precheck** — if `sheet_qty` exceeds the largest *total* daily
   capacity among the step's unblocked candidate machines, fail instantly with
   the same terminal status/reason/`days_delayed` the full scan would have
   produced (horizon-exceeded case included; the day-1 fast-track claim other
   lots would have observed is replicated).
2. **Full-day memo** — `(process, blocked-set, day) → smallest qty proven not to
   fit`. Later scans of the same key with qty ≥ proven skip the per-machine
   sweep. The day walk itself is preserved, so allocation dates, statuses and
   `days_delayed` are unchanged.

**What changes / what doesn't.** Allocations, dates, equipment choices, failure
statuses, fast-track state: **identical** (verified: 7-scenario A/B suite +
6,000-step mixed-saturation run, outcomes and all state dicts equal; 6.1×
end-to-end there, 40×+ on oversize-dominated shapes). What differs: deduped days
log ONE summary `DelayRecord` (`ALL_CANDIDATES_MEMO` / `ALL_CANDIDATES_OVERSIZE`)
instead of one per machine, so `delay_reasons` strings and the
`capacity_shortage` per-machine detail get thinner on those days.

**How to run.** Production runs are triggered by the **Action Flow**, so the
switch is a papermill parameter: `scan_memo` in the notebook's parameters cell
(`run_revplan_simulation.ipynb` cell 1), which the params cell writes into
`REVPLAN_SCAN_MEMO` before the engine runs.

- **Notebook default: `scan_memo = '1'` (ON)** — a triggered run picks the fix
  up with no flow change.
- **Baseline / kill switch:** pass `"scan_memo": "0"` in the flow's
  `/executions` params (or edit cell 1) → byte-identical stock Palantir scan.
- Other callers (smoke tests, ad-hoc scripts): the **engine-level env default
  is OFF**; set `REVPLAN_SCAN_MEMO=1` yourself.

Every run log states the switch ("Scan memo: ON/off") next to the config
header, so any log can be attributed to its mode. Code:
`allocation_engine.py`, blocks marked `⚠️ MLWB DIVERGENCE L1/L2` (all gated on
`_scan_memo_enabled()`).

## Business-logic changes — ALL OFF by default

Out of the box the engine reproduces Palantir behavior exactly. Each HTML-checklist
change is a flag on `SimulationParams`, flip on once its data is ready & validated:
`use_weighted_priority` (+ `weight_*`, `prototype_pct`), `enable_lot_de_leveling`,
`enable_positive_constraints`, `enable_yield_loss`, `enable_panel_separation`,
`enable_mix_capa`. Only `use_weighted_priority` is implemented so far; the rest are
placeholders.

## 🔌 Celonis wiring (now in `celonis_io.py`)

Connection + reads/writes are implemented (pycelonis 2.x). Status per input is
marked in `celonis_io.read_inputs`:

- ✅ **Wired** (PPS/RTS source, columns mapped/derived): `revenue_plan` (PK1_MPLAN),
  `wip_lots` (RTSP_WIP_N), `planned_process_steps` (RTSP_MODEL_ROUTING_M),
  `model_master` + `model_unit_conversion` (PK1_MODEL), `equipment_capacity` (PK1_EQUIPMENT).
- ⚠️ **Partial placeholder**: `available_inventory` (no shipped-qty feed / no monthly grain),
  `equipment_to_process` (derive from routing).
- ❌ **Missing placeholder**: `model_priorities` (MES 우선도 + margin), `equipment_constraints`
  (maintained block-list).
- ✅ **Model key (resolved)**: RTS `PRODID` **is** the PPS `MODEL_NO` (validated 2026-07-06:
  94.97% of WIP `PRODID` match `PK1_MODEL.MODEL_NO` exactly on the full dotted string). RTS-sourced
  tables (`wip_lots`, `planned_process_steps`) use `PRODID` directly as `model_id` — no crosswalk.
  The ~5% that miss the master (new/obsolete/synthetic models) LEFT-join to null and are ignored.

**Prerequisite (in the Celonis platform):** the PPS/RTS tables must already be extracted into a
Data Pool (Data Connection + extraction / Data Job, or upload) and added to a Data Model. Config
via env vars: `CELONIS_DATA_POOL` (default `2. Simulation`), `CELONIS_DATA_MODEL`.
Inside MLWB, `get_celonis()` needs no credentials.

Self-test the connection:  `python -m revplan_engine.celonis_io`  (lists pool tables + model).

## Run it

```python
# MLWB notebook (triggered with {dpInstanceId}):
from revplan_engine.run_simulation import main
results = main(simulation_id=dpInstanceId)

# manual test with explicit params:
from revplan_engine.run_simulation import main, SimulationParams
results = main(SimulationParams(simulation_name="test", start_month="202603"))
```

## Validate first

```bash
python smoke_test.py        # T1 imports/shim · T2 config · T3 priority toggle · T4 full synthetic run
```

> **Note:** this package was validated *statically* (syntax, import graph, shim
> coverage, engine call-site signatures all pass) but **not executed**, because the
> build sandbox blocks installing `polars`. `smoke_test.py` is the first runtime
> check — run it where polars exists (MLWB). T4's fixtures are tiny and may need a
> column tweak against your live schema.
