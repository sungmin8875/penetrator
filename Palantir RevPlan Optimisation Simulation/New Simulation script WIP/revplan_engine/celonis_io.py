"""
================================================================================
 celonis_io.py — the ONE place all Celonis (pycelonis 2.x) coupling lives.
================================================================================
Same idea as `_foundry_shim.py` isolates Palantir: every read from / write to the
Celonis platform is here, so the ported engine stays platform-agnostic.

WHAT THIS DOES
--------------
  * connect()           — auth (MLWB zero-config, or env creds for local testing)
  * read_oe_table_row() — pull the scenario row the button wrote
  * read_inputs()       — pull every engine input table, mapping PPS/RTS column
                          names to the names the engine expects
  * write_outputs()     — push the result tables back to the Data Pool

STATUS LEGEND for each input (see read_inputs):
  ✅ WIRED    — sourced from a real PPS/RTS table, columns mapped/derived here.
  ⚠️ PARTIAL  — structure exists but a field or grain is missing; returned as a
                placeholder until that gap is filled (don't silently fake it).
  ❌ MISSING  — no PPS/RTS source at all; placeholder until a feed is supplied.
  (NOTE: RTS PRODID == PPS MODEL_NO — validated 94.97% exact match 2026-07-06 —
   so RTS-sourced rows join to PPS directly; there is NO crosswalk step.)

HOW THE DATA GETS HERE (read the connection guide at the bottom of this file):
  The PPS/RTS tables must already be EXTRACTED into a Celonis Data Pool (via a
  Celonis Data Connection + extraction/Data Job, or a manual upload) and added to
  a Data Model. This module reads them from that Data Model via PQL.

CONFIG — via environment variables (so no secrets live in the engine package):
  CELONIS_BASE_URL    e.g. https://lg-innotek.eu-1.celonis.cloud   (local only)
  CELONIS_API_KEY     an APP_KEY / USER_KEY                         (local only)
  CELONIS_KEY_TYPE    "USER_KEY" (default) or "APP_KEY"
  CELONIS_DATA_POOL   default "2. Simulation"
  CELONIS_DATA_MODEL  default "RevPlan Simulation"  (the model over the PPS/RTS tables)
Inside an MLWB notebook you usually need NONE of these — get_celonis() with no
args picks up the workbench's own credentials, and the pool/model names default.
================================================================================
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import polars as pl

# ------------------------------------------------------------------------------
# Config (overridable by env; sensible defaults from check_connection.py)
# ------------------------------------------------------------------------------
# Defaults are the CONFIRMED ids on lg-innotek.eu-1 (from the connect log), so a fresh kernel
# resolves with NO env vars. _resolve() accepts an id OR a name, so either works as an override.
POOL_NAME       = os.environ.get("CELONIS_DATA_POOL",  "41c041fa-1a49-4afa-8799-7779fa61e86c")  # "2. Simulation"
DATA_MODEL_NAME = os.environ.get("CELONIS_DATA_MODEL", "73bc8779-7ddb-45a0-81dc-5e7f87c6ac92")  # perspective_custom_Simulation (non-test; test: twin is 6570e74c)
OE_TABLE_NAME   = os.environ.get("CELONIS_OE_TABLE", "SIMULATION_OE_Table")
OUTPUT_PREFIX   = os.environ.get("CELONIS_OUTPUT_PREFIX", "SIM_")
# RTS PRODID == PPS MODEL_NO (validated 2026-07-06: 94.97% of distinct RTSP_WIP_N.PRODID
# match PK1_MODEL.MODEL_NO exactly on the full dotted string, e.g. SFANP40005.KMA0). So
# RTS-sourced rows use PRODID DIRECTLY as model_id — no crosswalk table, no lot-id prefix.

# simulation_id values that mean "no real Digital Process Instance was injected" — the
# manual/dev sentinels the notebook falls back to when the Action Flow didn't pass
# {{1.instanceId}} as dpInstanceId. These must NEVER reach Celonis: run_simulation stamps
# simulation_id onto every allocation row + the run_tracker (and run_id embeds it), so a
# placeholder run pollutes the SIM_* tables with 'local-test' rows the OE can never
# correlate. write_outputs refuses to write on one of these. Kept in sync with the
# notebook's PLACEHOLDER_INSTANCE_IDS (compared case-insensitively on the stripped value).
PLACEHOLDER_SIMULATION_IDS = frozenset({"", "local-test", "none", "null", "mlwb_run"})


def _is_placeholder_simulation_id(sim_id) -> bool:
    """True when sim_id is empty or one of the manual/dev sentinels (case-insensitive)."""
    return str(sim_id or "").strip().lower() in PLACEHOLDER_SIMULATION_IDS


# Release-status filter for EquipmentConstraints (Gumi workshop 2026-07, meeting_summary §8):
# the sim should use the last RELEASED simulation version, not a "Created (unreleased)" one.
# It is UNCONFIRMED whether the OCPM object exposes a release-status column (plan prerequisite
# P2), and _pull errors on a non-existent column, so this is OPT-IN: set the env var to the
# real source column name once known. Until then read_equipment_to_process keeps its existing
# max(SimulVersion) behaviour unchanged (degrade safely, never guess a column into existence).
RELEASE_STATUS_COL = os.environ.get("REVPLAN_RELEASE_STATUS_COL")  # e.g. "ReleaseStatus"
RELEASED_VALUES = frozenset(
    v.strip().upper()
    for v in os.environ.get("REVPLAN_RELEASED_VALUES", "RELEASED,RELEASE,릴리즈,Y,CONFIRMED").split(",")
    if v.strip()
)


# available_inventory month grain (see read_available_inventory). OnHandLot is a
# point-in-time snapshot with no plan-month of its own, so we must assign one:
#   "start_month" (DEFAULT) — pin the current on-hand as OPENING inventory for the run's
#                             start_month (params.start_month). net_production_demand then
#                             consumes it from start_month and rolls the excess forward.
#                             ROBUST: the snapshot always lands inside the plan horizon.
#   "batch_month"           — bucket each model into the month of its CREATION_DATE. Faithful
#                             to a dated snapshot, but a whole load lands in ONE month, and any
#                             model whose month isn't in the plan is silently dropped downstream.
ONHAND_MONTH_MODE = os.environ.get("REVPLAN_ONHAND_MONTH_MODE", "start_month").strip().lower()

# Which SubinventoryCode sub-buckets of OnHandLot count as available FG (read_available_inventory).
# This is a business rule, so both sets are env-overridable. Palantir-exact default (260611 - FP
# Java Logic.txt): on-hand = 'FGI', transit = 'FGI-TRN'; FGI-RCV/STG/NON/RWK/RMA are EXCLUDED
# (received/staged/non-conforming/rework/returns are not available to fulfil demand). To include
# received or staged stock later, set e.g. REVPLAN_ONHAND_FGI_CODES="FGI,FGI-RCV,FGI-STG".
ONHAND_FGI_CODES = frozenset(
    v.strip().upper()
    for v in os.environ.get("REVPLAN_ONHAND_FGI_CODES", "FGI").split(",")
    if v.strip()
)
# Shipped bucket of available_inventory (2026-07-15, o_custom_Shipping wired):
# month-to-date shipments for the run's start month count as inventory in the
# engine contract (netting priority transit -> shipped -> onhand). "off" restores
# the previous shipped=0 behaviour for A/B and parity runs.
SHIPPED_MODE = os.environ.get("REVPLAN_SHIPPED_MODE", "start_month").strip().lower()
# VersionName sets: RESULT rows add, RTN rows (returns) subtract.
SHIPPING_RESULT_VERSIONS = frozenset(
    v.strip().upper()
    for v in os.environ.get("REVPLAN_SHIPPING_RESULT_VERSIONS", "SALES_RESULT").split(",")
    if v.strip())
SHIPPING_RTN_VERSIONS = frozenset(
    v.strip().upper()
    for v in os.environ.get("REVPLAN_SHIPPING_RTN_VERSIONS", "SALES_RESULT_RTN").split(",")
    if v.strip())

ONHAND_TRANSIT_CODES = frozenset(
    v.strip().upper()
    for v in os.environ.get("REVPLAN_ONHAND_TRANSIT_CODES", "FGI-TRN").split(",")
    if v.strip()
)


# ------------------------------------------------------------------------------
# Source-name mapping — engine's logical PPS/RTS names -> actual names in THIS
# Celonis data model (test:perspective_custom_Simulation, an OCPM perspective).
# The input readers below keep referring to the logical PPS names; _pull()
# translates through these maps, so re-mapping is a ONE-PLACE edit here.
#   TABLE_MAP : logical table name -> data-model table name
#   COLUMN_MAP: logical table name -> { logical SOURCE_COL -> data-model column }
# Anything not listed passes through unchanged, so partially-mapped tables still
# work and an un-mapped table fails loudly (SaolaPy "Table not found") until added.
# ------------------------------------------------------------------------------
TABLE_MAP: Dict[str, str] = {
    "PK1_MPLAN":            "o_custom_MovePlan",   # revenue_plan
    "PK1_MODEL":            "o_custom_Model",      # model master + unit conversion
    "RTSP_WIP_N":           "o_custom_WipDaily",   # wip lots
    "RTSP_MODEL_ROUTING_M": "o_custom_ModelRoute", # planned process steps
    "PK1_EQUIPMENT":        "o_custom_Equipment",  # equipment capacity
    "PK1_URGENCY_WIP_LOT":  "o_custom_UrgencyWipLot",  # ✅ confirmed 2026-07-06 — Tier-1 urgent WIP lots (folded into read_wip_lots)
    "PK1_EQUIPMENT_CONSTRAINTS": "o_custom_EquipmentConstraints",  # ✅ 2026-07-06 — POSITIVE model×operation→equipment assignment (feeds equipment_to_process)
    "PK1_EQUIPMENT_GROUP":       "o_custom_EquipmentGroup",       # ✅ 2026-07-06 — EquipmentCode↔OpMachineNm group taxonomy + InfiniteCapaYn (enriches equipment_capacity)
    "PK1_ONHAND_TRN":            "o_custom_OnHand",               # ⚠️ coarse item/material snapshot (ItemNo-keyed, RAW-MTL/FGI); NOT used for available_inventory anymore — candidate source for material_inventories
    "PK1_ERP_ONHAND_LOT":        "o_custom_OnHandLot",            # ✅ 2026-07-10 — lot-level FG on-hand (MODEL_CODE-keyed; detailed SubinventoryCode FGI/FGI-TRN/…) → feeds available_inventory (on-hand + transit)
    "PK1_JIG_MASTER":            "o_custom_JIGMaster",            # ✅ 2026-07-10 — ET-JIG capacity/qty (JigQty, JigCapa); NO model col — model comes from JIGDetail (see read_et_jig_master)
    "PK1_JIG_DETAIL":            "o_custom_JIGDetail",            # ✅ 2026-07-10 — per-jig target model (ModelNo); joined to JIGMaster via the OCPM relationship in read_et_jig_master
    "PK1_SHIPPING":              "o_custom_Shipping",             # ✅ 2026-07-15 — shipment lines (ItemNo=MODEL_NO, ShippedQuantity EA, ShippingDate) → shipped bucket of available_inventory
}
COLUMN_MAP: Dict[str, Dict[str, str]] = {
    "PK1_MPLAN": {
        "MODEL_NO":    "ModelNo",
        "REVISION_NO": "RevisionNo",
        "YYYYMM":      "YYYYMM",
        "PLAN_QTY":    "PlanQty",
        "PLAN_AMT":    "PlanAmt",
    },
    "PK1_MODEL": {
        "MODEL_NO":          "ModelNo",
        "ORGANIZATION_CODE": "OrganizationCode",   # ⚠️ requires an 'OrganizationCode' attribute on the Model object type
        "CUSTOMER_NAME":     "CustomerName",
        "END_CUSTOMER":      "EndCustomer",
        "SALES_TEAM":        "SalesTeam",
        "ADJUST_LEADTIME":   "AdjustLeadtime",
        "LOT_SIZE":          "LotSize",
        "EA_IN_PANNEL":      "EaInPannel",
        "EA_IN_SHEET":       "EaInSheet",
    },
    "RTSP_WIP_N": {
        "LOTID":           "LOTID",
        "PRODID":          "PRODID",          # ✅ PRODID == MODEL_NO -> used directly as model_id (no crosswalk)
        "PROCID":          "PROCID",
        "SEQ":             "SEQ",
        "EQPTID":          "EQPTID",
        "REAL_WIPSHTQTY":  "RealWipshtqty",
        "REAL_WIPUNITQTY": "RealWipunitqty",
    },
    "RTSP_MODEL_ROUTING_M": {
        "PRODID":          "ModelNo",         # ✅ routing carries the PPS MODEL_NO directly -> NO crosswalk
        "SEQ":             "WorkSeq",
        "PROCID":          "OperationCode",   # ⚠️ ASSUMPTION — confirm process-id source (OperationCode vs OpCode)
        "PROCNAME":        "OperationName",
        "OP_MACHINE_CODE": "ModifiedGroup",   # ✅ the real equipment group (better than the raw machine code)
        "RUN_LT":          "RunLt",           # ✅ per-sheet run time (hours) — Gumi §5 (P1: confirm column exists)
        "WAIT_LT":         "WaitLt",          # ✅ per-lot wait time (hours)  — Gumi §5 (P1: confirm column exists)
    },
    "PK1_EQUIPMENT": {
        "EQUIPMENT_CODE": "EquipmentCode",
        "MAPPING_NAME":   "MappingName",
        "DAILY_CAPA":     "DailyCapa",
        "SITE":           "SITE",
    },
    "PK1_URGENCY_WIP_LOT": {
        "LOT_ID":        "LotId",         # ✅ confirmed 2026-07-06 from o_custom_UrgencyWipLot cols
        "CREATION_DATE": "CreationDate",  # ✅ standard 24h timestamp (NOT the Korean 12h string) — see _add_urgency_flag
    },
    "PK1_ONHAND_TRN": {
        "ITEM_NO":           "ItemNo",            # item/material code (NOT a MODEL_NO) — coarse object, no longer wired
        "ONHAND_QTY":        "OnhandQty",
        "SUBINVENTORY_CODE": "SubinventoryCode",  # coarse: 'RAW-MTL' / 'FGI'
        "BATCH_DATE":        "BatchDate",
        "BATCH_HOUR":        "BatchHour",
    },
    "PK1_SHIPPING": {
        "MODEL_NO":      "ItemNo",           # ✅ MODEL_NO form (MGF42040) → model_id, direct join (Model FK exists too)
        "SHIPPED_QTY":   "ShippedQuantity",  # ✅ EA (26,130 EA @ $0.144/EA — substrate-unit scale)
        "SHIPPING_DATE": "ShippingDate",     # ✅ physical ship event; precedes record CreationDate (posting lag) — the bucketing key
        "VERSION_NAME":  "VersionName",      # ✅ SALES_RESULT (shipments) / SALES_RESULT_RTN (returns — subtract)
        "ORGANIZATION_CODE": "OrganizationCode",  # PK1 (unused for now; single site)
    },
    "PK1_ERP_ONHAND_LOT": {
        "MODEL_CODE":        "ModelCode",          # ✅ MODEL_NO form (MGS832G2 / SPCCP30021.KMC2) → model_id, direct join
        "ITEM_TYPE":         "ItemType",           # 'FGI' classifier (not used as a filter — SubinventoryCode is authoritative, per Palantir)
        "SUBINVENTORY_CODE": "SubinventoryCode",   # ✅ detailed: FGI / FGI-TRN / FGI-RCV / FGI-STG / FGI-NON / FGI-RWK / FGI-RMA
        "ONHAND_QTY":        "OnhandQty",           # ✅ on-hand AND transit both come from this column (transit = FGI-TRN rows; IntransitStock ignored)
        "INTRANSIT_STOCK":   "IntransitStock",      # intentionally UNUSED — Palantir took transit from FGI-TRN rows' ONHAND_QTY, not this column
        "LOT_NUMBER":        "LotNumber",
        "CREATION_DATE":     "CreationDate",        # only for ONHAND_MONTH_MODE=batch_month
    },
}


# ==============================================================================
# 1.  CONNECTION  — auth + pool/data-model handles (lazy, cached per process)
# ==============================================================================
_CELONIS = None
_POOL = None
_DM = None

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def _resolve(collection, ident):
    """Look up a pool / data model by id when `ident` looks like a UUID, else by name.

    pycelonis `.find()` defaults to search_attribute='name', so passing a UUID makes
    it search for a pool literally NAMED that UUID and raises PyCelonisNotFoundError.
    This mirrors the working Excel-upload notebook, which matches on `.id`.
    """
    ident = ident.strip()
    try:
        if _UUID_RE.fullmatch(ident):
            return collection.find_by_id(ident)
        return collection.find(ident)  # default: search_attribute='name'
    except Exception:
        # Self-diagnosing: on a miss, show what IS available so a wrong id/name is obvious.
        try:
            print(f"   ⛔ could not resolve {ident!r} — available in this collection:")
            for o in collection:
                print(f"       - {getattr(o, 'name', '?')} (id={getattr(o, 'id', '?')})")
        except Exception:
            pass
        raise


def _with_scheme(base: str) -> str:
    """The MLWB-injected CELONIS_URL has no scheme; pycelonis wants one."""
    base = base.strip()
    return base if base.startswith("http") else "https://" + base


def _user_key_override() -> Optional[str]:
    """A USER_KEY supplied out-of-band, so it never lives in the notebook and does not
    depend on any cell running first. Checked in order:
      1. REVPLAN_USER_KEY env var
      2. ~/.revplan_user_key file (reliable across MLWB terminal + papermill kernels)
    Returns the key string, or None if neither is present.
    """
    key = os.environ.get("REVPLAN_USER_KEY")
    if key:
        return key.strip()
    kf = os.path.expanduser("~/.revplan_user_key")
    if os.path.isfile(kf):
        with open(kf) as fh:
            return fh.read().strip()
    return None


def connect():
    """Authenticate. Preference order (first that resolves wins):
      1. Explicit env creds:  CELONIS_BASE_URL + CELONIS_API_KEY  (+ CELONIS_KEY_TYPE).
      2. Out-of-band USER_KEY: REVPLAN_USER_KEY env var or ~/.revplan_user_key file.
         This overrides the permission-less MLWB app key WITHOUT editing the notebook or
         worrying about cell order — the key is loaded here, at connect time.
      3. MLWB zero-config get_celonis()  — the workbench-injected app key.
    """
    from pycelonis import get_celonis  # imported lazily so the engine imports w/o pycelonis
    url = os.environ.get("CELONIS_BASE_URL")
    key = os.environ.get("CELONIS_API_KEY")
    key_type = os.environ.get("CELONIS_KEY_TYPE", "USER_KEY")

    # (2) fall back to an out-of-band USER_KEY when explicit creds weren't given.
    if not (url and key):
        override = _user_key_override()
        if override:
            key = override
            url = url or os.environ.get("CELONIS_URL", "lg-innotek.eu-1.celonis.cloud")

    if url and key:
        url = _with_scheme(url)
        print(f"   ✓ auth: explicit key -> {url} (key_type={key_type})")
        return get_celonis(url, key, key_type=key_type)

    print("   ⚠ auth: MLWB zero-config (workbench app key) — reads will 403 if it lacks "
          "data permissions; set REVPLAN_USER_KEY or ~/.revplan_user_key to override.")
    return get_celonis()  # MLWB zero-config


def pool():
    global _CELONIS, _POOL
    if _POOL is None:
        _CELONIS = connect()
        _POOL = _resolve(_CELONIS.data_integration.get_data_pools(), POOL_NAME)
        print(f"   ✓ Celonis pool: {_POOL.name} (id={_POOL.id})")
    return _POOL


def data_model():
    """The Data Model that exposes the PPS/RTS source tables for PQL reads.
    Data models belong to a pool in pycelonis 2.x: pool().get_data_models()."""
    global _DM
    if _DM is None:
        _DM = _resolve(pool().get_data_models(), DATA_MODEL_NAME)
        print(f"   ✓ Celonis data model: {_DM.name} (id={_DM.id})")
    return _DM


# ==============================================================================
# 2.  READ PRIMITIVE  — pull selected columns of one source table into polars.
#     This is THE single line to adjust if your pycelonis read differs.
# ==============================================================================
def _pull(table: str, colmap: Dict[str, str], distinct: bool = False) -> pl.DataFrame:
    """PQL-read `table`, aliasing each source column to the engine column name.

        colmap = {engine_col: SOURCE_COL}   ->  SELECT "TABLE"."SOURCE_COL" AS engine_col

    Returns a polars frame whose columns are already the engine's names.
    `table`/`src` are the engine's LOGICAL PPS names; they are translated to the
    actual data-model names via TABLE_MAP / COLUMN_MAP before the query is built.
    """
    from pycelonis.pql import PQL, PQLColumn
    import pycelonis.pql as pql
    dm = data_model()
    actual_table = TABLE_MAP.get(table, table)
    colrename = COLUMN_MAP.get(table, {})
    q = PQL(distinct=distinct)
    for engine_col, src in colmap.items():
        actual_src = colrename.get(src, src)
        q += PQLColumn(name=engine_col, query=f'"{actual_table}"."{actual_src}"')
    # Prefer SaolaPy's interactive-compute path. `export_data_frame` hits the bulk
    # data-export endpoint (/compute/{dm}/export/query), which is DISABLED BY DEFAULT
    # per team and 403s as PyCelonisDataExportNotEnabledError. SaolaPy uses the same
    # compute backend Studio/Views use and typically works without that team flag.
    # Fall back to the deprecated exporter only if this build predates SaolaPy.
    # NOTE: SaolaPy, the v1 exporter, AND the v2 exporter all POST to the data-model
    # export endpoint, which is gated by the TEAM-level "Data Export" flag (off by
    # default). Confirmed 2026-07-02 that v1 AND v2 both 403 even for a fully-permissioned
    # USER_KEY, so there is no API workaround — Data Export must be enabled for the team.
    try:
        pdf = pql.DataFrame.from_pql(q, data_model=dm).to_pandas()   # pandas DataFrame
    except AttributeError:
        pdf = dm.export_data_frame(q)                                # SaolaPy absent -> legacy exporter
    return pl.from_pandas(pdf)


def _empty(name: str, columns, why: str) -> pl.DataFrame:
    """Schema-correct empty frame for a MISSING/PARTIAL input — never silently faked.

    `columns` is either a list of names (all typed Utf8) or a {name: polars_dtype}
    mapping. Real dtypes MATTER even at zero rows: `pl.DataFrame({c: []})` infers every
    column as the `Null` dtype, and polars then raises
    SchemaError("join keys don't match … null on right") the instant the engine joins
    this placeholder against real data — e.g. available_inventory ⋈ revenue_plan on
    grouping_model/plan_month in net_production_demand. Declaring the schema fixes it.
    """
    print(f"   ⛔ placeholder '{name}': {why}")
    schema = columns if isinstance(columns, dict) else {c: pl.Utf8 for c in columns}
    return pl.DataFrame(schema=schema)


def _add_missing(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    """Add any engine-expected columns the source lacks, as nulls, so the
    downstream schema is complete (matches how these come through null in Foundry)."""
    for c in cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    return df


# ==============================================================================
# 3.  MODEL KEY  — RTS PRODID == PPS MODEL_NO (NO crosswalk).
#     Validated 2026-07-06: 94.97% of distinct RTSP_WIP_N.PRODID match
#     PK1_MODEL.MODEL_NO exactly on the full dotted string (e.g. SFANP40005.KMA0);
#     PK1_MODEL stores the dotted form, so RTS rows use PRODID directly as model_id.
#     The ~5% that miss the master (new/obsolete/synthetic models) LEFT-join to
#     null and are ignored for now. No crosswalk table, no lot-id-prefix derivation.
# ==============================================================================


# ==============================================================================
# 4.  OE-TABLE ROW  — the scenario the Celonis button wrote.
# ==============================================================================
def read_oe_table_row(simulation_id: str) -> Dict:
    """Pull the single SIMULATION_OE_Table row for this simulation_id."""
    # NOTE: params_from_oe_row infers weighted-priority ON/OFF from the weights. If you
    # later add an explicit "use_weighted_priority" column to SIMULATION_OE_Table, add it
    # to this list too and it will override the inference (see params_from_oe_row). Don't
    # add a column that doesn't exist in the table — the whole PQL read fails if you do.
    cols = {c: c for c in ["simulation_id", "scenario_name", "start_month",
                           "weight_revenue", "weight_margin", "weight_delivery",
                           "prototype_pct", "max_delay_days", "created_by",
                           "revenue_plan_id", "status"]}
    try:
        df = _pull(OE_TABLE_NAME, cols).filter(pl.col("simulation_id") == simulation_id)
        if df.height == 0:
            print(f"   ⚠️ no OE row for simulation_id={simulation_id}; using defaults")
            return {}
        return df.row(0, named=True)
    except Exception as ex:  # noqa: BLE001
        print(f"   ⚠️ read_oe_table_row failed ({ex}); using defaults")
        return {}


# ==============================================================================
# 5.  INPUT READERS  — one per engine input.  ✅ wired / ⚠️ partial / ❌ missing
# ==============================================================================

# ---- ✅ revenue_plan  <-  PKG_MGR.PK1_MPLAN -----------------------------------
def read_revenue_plan(params) -> pl.DataFrame:
    df = _pull("PK1_MPLAN", {
        "model_id":        "MODEL_NO",
        "revenue_plan_id": "REVISION_NO",
        "plan_month":      "YYYYMM",
        "quantity_ea":     "PLAN_QTY",
        "amount_krw":      "PLAN_AMT",     # ⚠️ PK1_MPLAN has NO currency col — assumed KRW
    })
    if df.height:
        df = df.with_columns([
            pl.col("plan_month").cast(pl.Int64, strict=False).cast(pl.Utf8),  # 201505.0 -> "201505"
            pl.col("quantity_ea").cast(pl.Float64, strict=False),
            pl.col("amount_krw").cast(pl.Float64, strict=False),
        ])
        # ── Filter to the SELECTED revision (fix 2026-07-15) ──────────────────
        # PK1_MPLAN carries EVERY revision back to 2015 (~636 of them). Unfiltered,
        # net_production_demand processes them ALL and the allocation demand dict
        # mixes rows across revisions per (model, month) — inflating demand by
        # orders of magnitude (the 43M-unit carryovers). The frontend selects ONE
        # plan version; simulate exactly that one.
        _rp = str(getattr(params, "revenue_plan_id", None) or "").strip()
        _revs = df["revenue_plan_id"].unique().to_list()
        if _rp and _rp in _revs:
            df = df.filter(pl.col("revenue_plan_id") == _rp)
            print(f"   ✓ revenue_plan: revision {_rp} selected — {df.height:,} rows "
                  f"(of {len(_revs)} revisions in PK1_MPLAN)")
        else:
            # No/unknown revision param -> newest revision key (MPyyyymm-woW-nnn sorts
            # chronologically). ⚠️ approximates the customer's "last RELEASED" rule —
            # PK1_MPLAN exposes no release-status column yet; revisit when it does.
            _latest = max(_revs)
            df = df.filter(pl.col("revenue_plan_id") == _latest)
            print(f"   ⚠ revenue_plan: revenue_plan_id={_rp!r} not found "
                  f"(revisions: {len(_revs)}) — falling back to NEWEST revision {_latest} "
                  f"({df.height:,} rows). NOTE: newest-by-name approximates the "
                  "'last released' rule (no release-status column on PK1_MPLAN).")
        # sales_team is NOT on PK1_MPLAN -> bring it from the model master (PK1_MODEL).
        try:
            mm = _pull("PK1_MODEL", {"model_id": "MODEL_NO", "sales_team": "SALES_TEAM"}).unique("model_id")
            df = df.join(mm, on="model_id", how="left")
        except Exception:  # noqa: BLE001
            df = df.with_columns(pl.lit(None).alias("sales_team"))
    # ❌ margin_krw: no cost/margin column anywhere in PPS/RTS -> null.
    # ⚠️ grouping_model / revenue_type: not in PK1_MPLAN -> null (engine tolerates).
    return _add_missing(df, ["margin_krw", "grouping_model", "sales_team", "revenue_type"])


def _shipped_quantities(start_month: Optional[str]) -> Optional[pl.DataFrame]:
    """Net EA shipped per model within `start_month` (YYYYMM), from o_custom_Shipping.

    Month-to-date shipments against the current plan month are inventory in the
    engine contract — goods that already left and fulfilled demand. Bucketing key
    is ShippingDate (the physical event; record CreationDate lags it). VersionName
    SALES_RESULT rows ADD; SALES_RESULT_RTN rows (returns) SUBTRACT as absolute
    values (sign-convention-proof); net is floored at 0 per model.

    Returns (model_id, shipped_quantity_ea), or None to skip (mode off, no
    start_month, object unreadable, or no rows in the month) — the caller then
    keeps shipped=0, the pre-2026-07-15 behaviour.
    """
    if SHIPPED_MODE == "off":
        print("   ⚠ REVPLAN_SHIPPED_MODE=off — shipped=0 (previous behaviour)")
        return None
    if not start_month:
        print("   ⚠ shipped: params has no start_month; shipped=0")
        return None
    try:
        sh = _pull("PK1_SHIPPING", {
            "model_id":      "MODEL_NO",
            "shipped_qty":   "SHIPPED_QTY",
            "shipping_date": "SHIPPING_DATE",
            "version_name":  "VERSION_NAME",
        })
    except Exception as ex:  # noqa: BLE001
        print(f"   ⚠ shipped: Shipping object not readable ({type(ex).__name__}); shipped=0 "
              "(TABLE_MAP 'PK1_SHIPPING' -> 'o_custom_Shipping')")
        return None
    if sh.height == 0:
        return None
    sh = sh.filter(pl.col("model_id").is_not_null()).with_columns([
        pl.col("shipped_qty").cast(pl.Float64, strict=False),
        pl.col("shipping_date").cast(pl.Datetime, strict=False).dt.strftime("%Y%m").alias("_ship_month"),
        pl.col("version_name").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("_version"),
    ]).filter(pl.col("_ship_month") == str(start_month))
    result_mask = pl.col("_version").is_in(list(SHIPPING_RESULT_VERSIONS))
    rtn_mask    = pl.col("_version").is_in(list(SHIPPING_RTN_VERSIONS))
    sh = sh.filter(result_mask | rtn_mask)
    if sh.height == 0:
        print(f"   ✓ shipped: no {start_month} rows in Shipping object "
              f"({sorted(SHIPPING_RESULT_VERSIONS | SHIPPING_RTN_VERSIONS)}); shipped=0")
        return None
    agg = (
        sh.group_by("model_id")
          .agg([
              pl.col("shipped_qty").filter(result_mask).sum().fill_null(0.0).alias("_shipped"),
              pl.col("shipped_qty").filter(rtn_mask).abs().sum().fill_null(0.0).alias("_returned"),
          ])
          .with_columns(
              pl.max_horizontal(pl.col("_shipped") - pl.col("_returned"), pl.lit(0.0))
                .alias("shipped_quantity_ea"))
          .select(["model_id", "shipped_quantity_ea"])
    )
    tot_s = float(agg["shipped_quantity_ea"].sum() or 0)
    print(f"   ✓ shipped ({start_month}): {agg.height} models, net {tot_s:,.0f} EA "
          f"(SALES_RESULT − |SALES_RESULT_RTN|, floored at 0)")
    return agg


# ---- ✅ available_inventory  <-  OnHandLot object (FGI on-hand + transit) ----
_AVAIL_INV_SCHEMA = {
    "model_id": pl.Utf8, "grouping_model": pl.Utf8, "plan_month": pl.Utf8,
    "shipped_quantity_ea": pl.Float64, "onhand_quantity_ea": pl.Float64,
    "total_inventory_ea": pl.Float64, "total_inventory_sht": pl.Float64,
}


def read_available_inventory(params) -> pl.DataFrame:
    """Finished-goods ON-HAND + TRANSIT from the OnHandLot object (PPS PK1_ERP_ONHAND_LOT).

    Repointed 2026-07-10 from the coarse OnHand object (PK1_ONHAND_TRN) to OnHandLot — the
    correct finished-goods source: keyed by MODEL_CODE (=MODEL_NO, direct join), with a
    DETAILED SubinventoryCode that reproduces Palantir's TWO branches. (The OnHand object was
    ItemNo/material-keyed with only a coarse RAW-MTL/FGI code and could not represent transit.)

    Palantir-exact recipe (260611 - FP Java Logic.txt — on-hand branch filters
    SUBINVENTORY_CODE=='FGI' @238, transit branch filters =='FGI-TRN' @1109-1114, both taking
    the ONHAND_QTY column), reproduced here and made config-driven:

      • on-hand = SUM(OnhandQty) where SubinventoryCode ∈ ONHAND_FGI_CODES     (default {'FGI'})
      • transit = SUM(OnhandQty) where SubinventoryCode ∈ ONHAND_TRANSIT_CODES (default {'FGI-TRN'})
      • FGI-RCV / STG / NON / RWK / RMA are EXCLUDED (not available to fulfil demand).
      • SubinventoryCode is the authoritative selector — we do NOT filter on ITEM_TYPE (Palantir
        didn't either; transit rows can carry a non-'FGI' ItemType).
      • Transit uses the FGI-TRN rows' OnhandQty, NOT the IntransitStock column (using both
        would double-count) — IntransitStock is deliberately left unread.
      • shipped = net EA shipped in the run's START MONTH from o_custom_Shipping
        (2026-07-15; see _shipped_quantities — SALES_RESULT − returns, floored at 0).
        Applied in start_month mode only; total_inventory_ea = on-hand + transit + shipped
        so net_production_demand's re-derivation transit = total − shipped − onhand
        round-trips exactly. REVPLAN_SHIPPED_MODE=off restores shipped=0.
      • total_inventory_sht = total_inventory_ea / units_per_sheet (EA_IN_SHEET on PK1_MODEL).
      • plan_month: the snapshot has no month, so ONHAND_MONTH_MODE pins it (start_month default).

    Degrades to a schema-correct EMPTY frame — the old placeholder's behaviour — if the object /
    columns aren't readable or no on-hand/transit rows survive the sub-bucket filter.
    """
    try:
        df = _pull("PK1_ERP_ONHAND_LOT", {
            "model_id":           "MODEL_CODE",
            "onhand_quantity_ea": "ONHAND_QTY",
            "subinventory_code":  "SUBINVENTORY_CODE",
            "creation_date":      "CREATION_DATE",
        })
    except Exception as ex:  # noqa: BLE001
        return _empty("available_inventory", _AVAIL_INV_SCHEMA,
                      f"FALLBACK — OnHandLot object not readable ({ex}); confirm the object name "
                      "(TABLE_MAP 'PK1_ERP_ONHAND_LOT' -> 'o_custom_OnHandLot') + columns via introspection")

    # Normalize the sub-bucket code and keep only the on-hand / transit buckets we count.
    df = df.filter(pl.col("model_id").is_not_null()).with_columns([
        pl.col("onhand_quantity_ea").cast(pl.Float64, strict=False),
        pl.col("subinventory_code").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("_subinv"),
    ])
    onhand_mask  = pl.col("_subinv").is_in(list(ONHAND_FGI_CODES))
    transit_mask = pl.col("_subinv").is_in(list(ONHAND_TRANSIT_CODES))
    df = df.filter(onhand_mask | transit_mask)
    if df.height == 0:
        return _empty("available_inventory", _AVAIL_INV_SCHEMA,
                      f"no OnHandLot rows in on-hand {sorted(ONHAND_FGI_CODES)} or transit "
                      f"{sorted(ONHAND_TRANSIT_CODES)} sub-buckets (SubinventoryCode filter)")

    # Sum on-hand and transit separately per model (Palantir's two branches, per-model).
    agg = (
        df.group_by("model_id")
          .agg([
              pl.col("onhand_quantity_ea").filter(onhand_mask).sum().fill_null(0.0).alias("onhand_quantity_ea"),
              pl.col("onhand_quantity_ea").filter(transit_mask).sum().fill_null(0.0).alias("_transit_ea"),
              pl.col("creation_date").max().alias("_creation"),
          ])
    )

    # units_per_sheet -> total_inventory_sht (EA_IN_SHEET on the model master).
    try:
        conv = _pull("PK1_MODEL", {"model_id": "MODEL_NO", "units_per_sheet": "EA_IN_SHEET"}).unique("model_id")
        conv = conv.with_columns(pl.col("units_per_sheet").cast(pl.Float64, strict=False))
        agg = agg.join(conv, on="model_id", how="left")
    except Exception:  # noqa: BLE001
        agg = agg.with_columns(pl.lit(None, dtype=pl.Float64).alias("units_per_sheet"))

    # plan_month: snapshot has no month of its own — assign per ONHAND_MONTH_MODE.
    if ONHAND_MONTH_MODE == "batch_month":
        agg = agg.with_columns(
            pl.col("_creation").cast(pl.Datetime, strict=False).dt.strftime("%Y%m").alias("plan_month"))
        month_note = "batch_month (per model's CreationDate)"
        shipped = None  # shipped is keyed to the start month; incompatible with batch_month rows
        if SHIPPED_MODE != "off":
            print("   ⚠ shipped: skipped under ONHAND_MONTH_MODE=batch_month "
                  "(shipped quantities are start-month-keyed)")
    else:
        start_month = str(getattr(params, "start_month", None) or "").strip() or None
        # shipped: month-to-date shipments count as inventory. Full-join so a model
        # with shipments but no FGI/FGI-TRN rows still gets its shipped credit; the
        # start_month literal below covers those joined-in rows too.
        shipped = _shipped_quantities(start_month)
        if shipped is not None:
            agg = agg.join(shipped, on="model_id", how="full", coalesce=True).with_columns([
                pl.col("onhand_quantity_ea").fill_null(0.0),
                pl.col("_transit_ea").fill_null(0.0),
                pl.col("shipped_quantity_ea").fill_null(0.0),
            ])
        agg = agg.with_columns(pl.lit(start_month).cast(pl.Utf8).alias("plan_month"))
        month_note = f"pinned to start_month={start_month}"
    if shipped is None and "shipped_quantity_ea" not in agg.columns:
        agg = agg.with_columns(pl.lit(0.0).alias("shipped_quantity_ea"))

    # total = on-hand + transit + shipped: net_production_demand re-derives
    # transit = total − shipped − onhand, so shipped MUST be inside the total or
    # the derived transit goes negative by exactly the shipped amount.
    _total = (pl.col("onhand_quantity_ea") + pl.col("_transit_ea") + pl.col("shipped_quantity_ea"))
    onhand = agg.with_columns([
        pl.lit(None, dtype=pl.Utf8).alias("grouping_model"),   # engine coalesces -> model_id
        _total.alias("total_inventory_ea"),
        pl.when((pl.col("units_per_sheet").is_not_null()) & (pl.col("units_per_sheet") > 0))
          .then(_total / pl.col("units_per_sheet"))
          .otherwise(None).alias("total_inventory_sht"),
    ])

    tot_oh = float(onhand["onhand_quantity_ea"].sum() or 0)
    tot_tr = float(onhand["_transit_ea"].sum() or 0)
    tot_sh = float(onhand["shipped_quantity_ea"].sum() or 0)
    print(f"   ✓ available_inventory: {onhand.height} models from OnHandLot — on-hand "
          f"{tot_oh:,.0f} EA {sorted(ONHAND_FGI_CODES)} + transit {tot_tr:,.0f} EA "
          f"{sorted(ONHAND_TRANSIT_CODES)} + shipped {tot_sh:,.0f} EA; {month_note}")
    return onhand.select(list(_AVAIL_INV_SCHEMA.keys()))


# ---- ❌ model_priorities  (MES 우선도, external) ------------------------------
def read_model_priorities(params) -> pl.DataFrame:
    # MISSING: the priority RANK is the MES 우선도 grade (not in PPS/RTS), and
    # margin_amount_per_unit has no source (no cost column anywhere). Revenue/unit
    # (amount_per_unit) COULD be rebuilt from PK1_MODEL.SALES_PRICE + PK1_EX_RATE.
    # Until the MES feed is wired, return empty and let priority_generation handle it.
    return _empty("model_priorities",
                  ["simulation_id", "simulation_name", "revenue_plan_id", "model_id",
                   "month", "priority", "amount_per_unit", "margin_amount_per_unit", "__is_deleted"],
                  "MISSING — priority rank = MES 우선도 (external); margin has no source")


# ---- ✅ wip_lots  <-  RTSP_MGR.RTSP_WIP_N  (grain fixed 2026-07-16) ------------
# WipDaily is a DAILY SNAPSHOT table (data-verified 2026-07-16: 67 distinct
# SnapshotDates, every long-lived lot has exactly one row per date). The old read
# handed ALL snapshot rows to the engine, which treated a lot's 67-day POSITION
# HISTORY as steps to allocate — re-planning the lot's past, never its remaining
# route. The read now: (1) filters to the run's site scope, (2) keeps only the
# LATEST snapshot (one row per lot = its current position), and (3) explodes each
# lot into its REMAINING routing steps (route WorkSeq >= current SEQ) — the same
# shape virtual lots use, so the engine plans the actual road ahead.
WIP_SITES = frozenset(
    s.strip().upper()
    for s in os.environ.get("REVPLAN_WIP_SITES", "PK1").split(",")
    if s.strip() and s.strip().lower() != "all")
# "remaining_route" (default) | "current_only" (filters, no explode) | "legacy" (old raw read)
WIP_MODE = os.environ.get("REVPLAN_WIP_MODE", "remaining_route").strip().lower()


def read_wip_lots(params) -> pl.DataFrame:
    cols = {
        "lot_id":               "LOTID",
        "model_id":             "PRODID",          # ✅ PRODID == MODEL_NO — used directly (no crosswalk)
        # 🔴 process_id MUST be the un-prefixed OperationCode (e.g. 'ML30N'), NOT PROCID.
        # WipDaily.PROCID is site-prefixed ('FK1ML30N' / 'PK1ML30N') while routing and the
        # capability map key on the SHORT code.
        "process_id":           "OperationCode",
        "sequence":             "SEQ",
        "equipment_group_id":   "EQPTID",          # ⚠️ specific machine, NOT the group
        "latest_sheet_quantity": "REAL_WIPSHTQTY",
        "latest_unit_quantity":  "REAL_WIPUNITQTY",
    }
    df = None
    if WIP_MODE != "legacy":
        # receipt_target_day: the lot's real due date (WipDaily.ReceiptTargetDay,
        # ~58% filled — G4 2026-07-16). Urgent lots use it as their target month
        # instead of the synthetic start-month stamp; null falls back downstream.
        try:
            df = _pull("RTSP_WIP_N", {**cols,
                                      "receipt_target_day": "ReceiptTargetDay",
                                      "_snapshot_date":     "SnapshotDate",
                                      "_site":              "PlanningSiteCode"})
        except Exception:  # noqa: BLE001 — retry without the due-date column
            try:
                df = _pull("RTSP_WIP_N", {**cols,
                                          "_snapshot_date": "SnapshotDate",
                                          "_site":          "PlanningSiteCode"})
            except Exception as ex:  # noqa: BLE001
                print(f"   ⚠ wip_lots: SnapshotDate/PlanningSiteCode not readable ({type(ex).__name__}); "
                      "falling back to the legacy raw read (NO snapshot/site filter)")
    if df is None:
        df = _pull("RTSP_WIP_N", cols)
    if df.height:
        # SEQ orders a lot's steps in group_wip_by_lot; real WIP carries null SEQ, and a
        # null mixed with floats breaks that sort. Coalesce to 0 under a single Int64
        # dtype; cast quantities to Float64 for consistent capacity math.
        df = df.with_columns([
            pl.col("sequence").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("latest_sheet_quantity").cast(pl.Float64, strict=False),
            pl.col("latest_unit_quantity").cast(pl.Float64, strict=False),
        ])

    # (1) site scope — the simulated equipment/capacity is PK1-side; other sites'
    # lots (FK1 ≈ 9%) would run "outsourced" through processes that are really the
    # other fab's in-house ops. REVPLAN_WIP_SITES=all disables.
    if "_site" in df.columns and WIP_SITES:
        pre = df.height
        df = df.filter(
            pl.col("_site").cast(pl.Utf8).str.strip_chars().str.to_uppercase().is_in(list(WIP_SITES)))
        print(f"   ✓ wip_lots: site filter {sorted(WIP_SITES)} — {df.height:,} of {pre:,} rows kept")

    # (2) latest snapshot only -> one row per lot (its CURRENT position).
    if "_snapshot_date" in df.columns and df.height:
        max_snap = df["_snapshot_date"].max()
        pre_lots = df["lot_id"].n_unique()
        df = df.filter(pl.col("_snapshot_date") == max_snap)
        # safety: if a lot still has >1 row in one snapshot, keep its furthest position
        df = df.sort("sequence", descending=True).unique(subset=["lot_id"], keep="first")
        print(f"   ✓ wip_lots: latest snapshot {max_snap} — {df.height:,} current lots "
              f"(of {pre_lots:,} lots seen across all snapshots)")
    df = df.drop([c for c in ("_snapshot_date", "_site") if c in df.columns])

    # (3) remaining-route explosion — plan the road AHEAD of each lot.
    if WIP_MODE == "remaining_route" and df.height:
        df = _explode_wip_remaining_route(df)

    # ⚠️ equipment_group_id: EQPTID is a machine; needs machine->group rollup. Left raw.
    # ❌ remaining_steps / final_work_sequence: engine has a fallback, so null is safe.

    # ✅ Tier-1 urgency flag  <-  PKG_MGR.PK1_URGENCY_WIP_LOT (after the explosion so
    # every step row of an urgent lot carries the flag; urgent pre-pass reads steps[0]).
    df = _add_urgency_flag(df)

    return _add_missing(df, ["remaining_steps", "final_work_sequence",
                             "is_urgent", "urgency_creation_date"])


def _explode_wip_remaining_route(current: pl.DataFrame) -> pl.DataFrame:
    """One row per lot × REMAINING routing step, from each lot's current position.

    The latest snapshot gives each lot's current (process, SEQ); the road ahead is
    the model's routing steps with WorkSeq >= current SEQ (>=: the current step is
    still WAIT/in-progress and must complete). WipDaily.SEQ and ModelRoute.WorkSeq
    share the same scale (RoutingSeq == SEQ, data-verified). The lot's current
    sheet/unit quantities ride on every remaining step — the same shape
    virtual_lot_creator emits, so the engine consumes it unchanged. Routing Run/Wait
    LTs are carried per step (the outsourced pass-through uses them directly).

    Lots whose model has no routing keep their single current-position row (the old
    behaviour) so they are not silently dropped.
    """
    rt_base = {
        "model_id":       "PRODID",
        "_route_seq":     "SEQ",
        "_route_process": "PROCID",
        "_route_group":   "OP_MACHINE_CODE",
    }
    rt = None
    for extra in ({"run_lt": "RUN_LT", "wait_lt": "WAIT_LT", "plan_lt": "PlanLt"},
                  {"run_lt": "RUN_LT", "wait_lt": "WAIT_LT"},
                  {}):
        try:
            rt = _pull("RTSP_MODEL_ROUTING_M", {**rt_base, **extra})
            break
        except Exception as ex:  # noqa: BLE001
            last_ex = ex
    if rt is None:
        print(f"   ⚠ wip_lots: routing not readable for the remaining-route explosion "
              f"({type(last_ex).__name__}); keeping current-position rows only")
        return current
    if rt.height == 0:
        print("   ⚠ wip_lots: routing empty; keeping current-position rows only")
        return current

    rt = rt.with_columns(pl.col("_route_seq").cast(pl.Int64, strict=False).fill_null(0))
    for c in ("run_lt", "wait_lt", "plan_lt"):
        if c in rt.columns:
            rt = rt.with_columns(pl.col(c).cast(pl.Float64, strict=False))

    lot_cols = ["lot_id", "model_id", "sequence",
                "latest_sheet_quantity", "latest_unit_quantity", "equipment_group_id"]
    if "receipt_target_day" in current.columns:
        lot_cols.append("receipt_target_day")   # due date rides every exploded step
    lots = current.select(lot_cols).rename({
        "sequence": "_current_seq",
        "equipment_group_id": "_current_eqpt",
    })
    exploded = (
        lots.join(rt, on="model_id", how="inner")
            .filter(pl.col("_route_seq") >= pl.col("_current_seq"))
            .with_columns([
                pl.col("_route_seq").alias("sequence"),
                pl.col("_route_process").alias("process_id"),
                pl.col("_route_group").alias("equipment_group_id"),
            ])
            .drop(["_route_seq", "_route_process", "_route_group", "_current_eqpt"])
    )
    covered = set(exploded["lot_id"].unique().to_list())
    orphans = current.filter(~pl.col("lot_id").is_in(list(covered)))
    if orphans.height:
        print(f"   ⚠ wip_lots: {orphans.height:,} lots have no routing (or none at/after their "
              "current step) — kept as single current-position rows")
        exploded = pl.concat([exploded, orphans.select(exploded.columns)
                              if set(orphans.columns) >= set(exploded.columns)
                              else orphans], how="diagonal_relaxed")
    n_lots = exploded["lot_id"].n_unique()
    print(f"   ✓ wip_lots: remaining-route explosion — {n_lots:,} lots × "
          f"avg {exploded.height / max(n_lots, 1):.0f} remaining steps = {exploded.height:,} step rows")
    return exploded.drop([c for c in ("_current_seq",) if c in exploded.columns])


def _add_urgency_flag(df: pl.DataFrame) -> pl.DataFrame:
    """Left-join PK1_URGENCY_WIP_LOT onto WIP to add `is_urgent` + `urgency_creation_date`.

    CreationDate on the OCPM object o_custom_UrgencyWipLot is a normalized 24-hour
    timestamp ('2026-06-30 08:48:13' — confirmed 2026-07-06), NOT the Korean 12-hour
    string ('2026-05-19 오후 2:19:38') the raw PPS table carried. It may arrive already
    typed as Datetime (OCPM timestamp attribute) or as a string; handle both so Phase 0
    can order urgent lots oldest-first. Any leftover Korean AM/PM markers are still
    stripped defensively. Degrades gracefully: if the table is absent/unreadable, every
    WIP row gets is_urgent=False so Tier 1 no-ops.
    """
    try:
        urg = _pull("PK1_URGENCY_WIP_LOT", {
            "lot_id":                    "LOT_ID",
            "urgency_creation_date_raw": "CREATION_DATE",
        })
        raw_dtype = urg.schema["urgency_creation_date_raw"]
        if raw_dtype in (pl.Datetime, pl.Date):
            # Already a real timestamp (OCPM date attribute) — just normalize the type.
            date_expr = pl.col("urgency_creation_date_raw").cast(pl.Datetime, strict=False)
        else:
            # String form: strip any Korean AM/PM markers, then infer ('2026-06-30 08:48:13').
            date_expr = (
                pl.col("urgency_creation_date_raw").cast(pl.Utf8)
                  .str.replace("오전", "AM").str.replace("오후", "PM")
                  .str.to_datetime(strict=False)
            )
        urg = urg.with_columns(
            date_expr.alias("urgency_creation_date")
        ).drop("urgency_creation_date_raw").unique("lot_id")

        matched = df.join(urg, on="lot_id", how="inner").height
        stale = max(urg.height - matched, 0)          # urgency rows with no live WIP lot
        print(f"   ✓ urgency: {matched} WIP lots flagged urgent; "
              f"{stale} urgency rows not in WIP (stale — ignored)")

        df = df.join(urg, on="lot_id", how="left")
        return df.with_columns(pl.col("urgency_creation_date").is_not_null().alias("is_urgent"))
    except Exception as ex:  # noqa: BLE001
        print(f"   ⛔ urgency join skipped ({ex}); is_urgent=False for all WIP (Tier 1 off)")
        return df.with_columns([
            pl.lit(None, dtype=pl.Datetime).alias("urgency_creation_date"),
            pl.lit(False).alias("is_urgent"),
        ])


# ---- ✅ planned_process_steps  <-  RTSP_MODEL_ROUTING_M (+ PK1_MODEL_ROUTE) ----
def read_planned_process_steps(params) -> pl.DataFrame:
    base_cols = {
        "model_id":           "PRODID",            # ✅ routing carries MODEL_NO directly (COLUMN_MAP: PRODID->ModelNo)
        "sequence":           "SEQ",
        "process_id":         "PROCID",
        "process_name":       "PROCNAME",
        "equipment_group_id": "OP_MACHINE_CODE",   # ⚠️ machine code; group via PK1_OPERATION.MODIFIED_GROUP
    }
    # Per-step lead times (Gumi workshop 2026-07, meeting_summary §5): Run LT (per-sheet,
    # ~fixed) + Wait LT (per-lot), both in hours. They live on o_custom_ModelRoute
    # (RunLt/WaitLt) but their presence is UNCONFIRMED (plan prerequisite P1); _pull errors
    # on a missing column, so try WITH them and fall back WITHOUT (lead-time proxy stays).
    df = None
    for extra in ({"run_lt": "RUN_LT", "wait_lt": "WAIT_LT", "plan_lt": "PlanLt"},
                  {"run_lt": "RUN_LT", "wait_lt": "WAIT_LT"},
                  {}):
        try:
            df = _pull("RTSP_MODEL_ROUTING_M", {**base_cols, **extra})
            break
        except Exception as ex:  # noqa: BLE001
            last_ex = ex
    if df is None:
        raise last_ex
    if "plan_lt" not in df.columns and "run_lt" not in df.columns:
        print("   ⚠ planned_process_steps: per-step lead-time columns (PlanLt/RunLt/WaitLt) not "
              "readable; proceeding without them (lead-time proxy stays)")
    if df.height:
        # ⚠️ is_from_latest_production_plan: not a source column. The engine HARD-FILTERS
        #    on it (==True), so default True to pass rows through (revisit when you have a
        #    'latest plan' marker, else you allocate against stale routings).
        df = df.with_columns(pl.lit(True).alias("is_from_latest_production_plan"))
        for c in ("run_lt", "wait_lt", "plan_lt"):
            if c in df.columns:
                df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    # ❌ q1_panel_in_seconds / mpi_Q1_wait_time_in_seconds: DERIVED event-time analytics.
    #    The scheduler still treats these as 0 throughput-time; the real Run/Wait now feed
    #    the virtual-lot lead-time estimate (virtual_lot_creator). Time-based scheduling is
    #    a separate follow-up. run_lt/wait_lt default null when the columns are absent.
    return _add_missing(df, ["grouping_model", "is_from_latest_production_plan",
                             "q1_panel_in_seconds", "mpi_Q1_wait_time_in_seconds",
                             "run_lt", "wait_lt", "plan_lt"])


def _route_plan_lt_seconds(models: pl.DataFrame) -> Optional[pl.DataFrame]:
    """Per-model Plan LT in SECONDS from the routing's Run/Wait columns.

    Customer-confirmed replacement (2026-07-15) for the ADJUST_LEADTIME proxy:
    Plan LT = Σ over routing steps of (RunLt × lot sheets + WaitLt), the planner
    formula validated live in the 2026-07 workshop (0.3 × 5 + 10 = 11.4 h for a
    5-sheet lot; Run is per-sheet, Wait per-lot, both in HOURS). Lot sheets =
    LotSize (maximum_lot_size_sht, sheets/lot) from the model master.

    Returns a (model_id, _lt_plan_s) frame, or None when the source degrades —
    the caller then falls back to ADJUST_LEADTIME for every model. Models whose
    routing carries no Run/Wait values sum to 0 h and come back NULL (per-model
    fallback). Set REVPLAN_LEADTIME_SOURCE=adjust to force the old proxy for
    A/B and parity runs.
    """
    if os.environ.get("REVPLAN_LEADTIME_SOURCE", "planlt").strip().lower() == "adjust":
        print("   ⚠ REVPLAN_LEADTIME_SOURCE=adjust — ADJUST_LEADTIME proxy forced for all models")
        return None
    # Prefer the routing's own per-step PlanLt column (99.1% filled, data-verified
    # 2026-07-16) — the planner's number, no formula to defend. Fall back to the
    # customer-validated computation RunLt×sheets+WaitLt when PlanLt isn't readable.
    rt, has_planlt = None, False
    try:
        rt = _pull("RTSP_MODEL_ROUTING_M", {
            "model_id": "PRODID",
            "plan_lt":  "PlanLt",
        })
        has_planlt = True
    except Exception:  # noqa: BLE001 — PlanLt column absent -> computed fallback
        try:
            rt = _pull("RTSP_MODEL_ROUTING_M", {
                "model_id": "PRODID",
                "run_lt":   "RUN_LT",
                "wait_lt":  "WAIT_LT",
            })
        except Exception as ex:  # noqa: BLE001
            print(f"   ⚠ Plan LT: routing PlanLt/Run/Wait not readable ({type(ex).__name__}); "
                  "ADJUST_LEADTIME fallback for all models")
            return None
    if rt.height == 0:
        print("   ⚠ Plan LT: routing returned no rows; ADJUST_LEADTIME fallback for all models")
        return None
    if has_planlt:
        rt = rt.with_columns(
            pl.col("plan_lt").cast(pl.Float64, strict=False).fill_null(0.0).alias("_step_lt_h"))
    else:
        default_sheets = 5.0  # config defaults: 30 panels/lot ÷ 6 panels/sheet
        rt = rt.with_columns([
            pl.col("run_lt").cast(pl.Float64, strict=False),
            pl.col("wait_lt").cast(pl.Float64, strict=False),
        ]).join(models.select(["model_id", "maximum_lot_size_sht"]), on="model_id", how="left")
        rt = rt.with_columns(
            (pl.col("run_lt").fill_null(0.0) * pl.col("maximum_lot_size_sht").fill_null(default_sheets)
             + pl.col("wait_lt").fill_null(0.0)).alias("_step_lt_h")
        )
    agg = (
        rt.group_by("model_id")
          .agg(pl.col("_step_lt_h").sum().alias("_plan_lt_h"))
          .with_columns(
              pl.when(pl.col("_plan_lt_h") > 0)
                .then(pl.col("_plan_lt_h") * 3600.0)   # hours -> engine-contract seconds
                .otherwise(None)
                .alias("_lt_plan_s"))
          .select(["model_id", "_lt_plan_s"])
    )
    n = agg.filter(pl.col("_lt_plan_s").is_not_null()).height
    src = "Σ PlanLt column" if has_planlt else "Σ RunLt×sheets + WaitLt (computed fallback)"
    print(f"   ✓ Plan LT: route-derived lead time for {n} models ({src}, h→s)")
    return agg


# ---- ✅ model_master  <-  PKG_MGR.PK1_MODEL -----------------------------------
def read_model_master(params) -> pl.DataFrame:
    df = _pull("PK1_MODEL", {
        "model_id":       "MODEL_NO",
        "customer_name":  "CUSTOMER_NAME",
        "end_customer":   "END_CUSTOMER",
        "sales_team":     "SALES_TEAM",
        "_adjust_leadtime_days": "ADJUST_LEADTIME",  # ✅ DAYS (customer-confirmed 2026-07-15) — FALLBACK only, see below
        "_lot_size":      "LOT_SIZE",
        "_ea_in_sheet":   "EA_IN_SHEET",
    }).unique("model_id")
    if df.height:
        df = df.with_columns([
            # Fallback lead time: ADJUST_LEADTIME is in DAYS (customer-confirmed
            # 2026-07-15), converted to the engine's seconds contract.
            (pl.col("_adjust_leadtime_days").cast(pl.Float64, strict=False) * 86400.0)
                .alias("_lt_adjust_s"),
            # ✅ LotSize is in EA (data-verified 2026-07-15: LotSize ÷ EaInSheet = 5
            # sheets exactly across families — the standard 30-panel lot; real WIP
            # lots run 0.1-10 sheets). maximum_lot_size_sht = LotSize ÷ EaInSheet.
            # Reading LotSize as sheets inflated lots ~10,000× (11,415-panel VLs,
            # never-fitting steps, weeks-long outsourced Plan LTs → 2042 horizons).
            # Null/0 EaInSheet -> null -> engine default (30 panels = 5 sheets).
            # REVPLAN_LOTSIZE_UNIT=sheets restores the old direct read (A/B).
            pl.when(
                (pl.lit(os.environ.get("REVPLAN_LOTSIZE_UNIT", "ea").strip().lower()) == "sheets")
            ).then(pl.col("_lot_size").cast(pl.Float64, strict=False))
             .otherwise(
                pl.when(pl.col("_ea_in_sheet").cast(pl.Float64, strict=False) > 0)
                  .then(pl.col("_lot_size").cast(pl.Float64, strict=False)
                        / pl.col("_ea_in_sheet").cast(pl.Float64, strict=False))
                  .otherwise(None)
             ).alias("maximum_lot_size_sht"),
        ]).drop(["_adjust_leadtime_days", "_lot_size", "_ea_in_sheet"])
        # Lead time: route-derived Plan LT (customer-confirmed 2026-07-15 to REPLACE
        # the ADJUST_LEADTIME proxy), per-model fallback to ADJUST_LEADTIME where the
        # routing has no Run/Wait. Feeds lead_time_days -> virtual-lot start dates
        # (allocation_helpers.build_model_metadata_lookup divides by 86400).
        _plan = _route_plan_lt_seconds(df)
        if _plan is not None:
            df = df.join(_plan, on="model_id", how="left")
        else:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("_lt_plan_s"))
        df = df.with_columns(
            pl.coalesce([pl.col("_lt_plan_s"), pl.col("_lt_adjust_s")])
              .alias("total_mprod_default_lot_size_median_lt_in_s")
        ).drop(["_lt_plan_s", "_lt_adjust_s"])
    # organization_code: the OCDM Model object's ID is ('Model_' || ORGANIZATION_CODE || MODEL_NO),
    # so SIM_ outputs need the org to build the Model FK. Pulled SEPARATELY + guarded so a Model
    # object that doesn't yet expose an "OrganizationCode" attribute doesn't break the whole
    # model_master read — org comes through null then (and the FK won't match until that attribute
    # is added to o_custom_Model). Assumes MODEL_NO -> exactly one org (engine already .unique's model_id).
    try:
        _org = _pull("PK1_MODEL", {
            "model_id":          "MODEL_NO",
            "organization_code": "ORGANIZATION_CODE",
        }).unique("model_id")
        df = df.join(_org, on="model_id", how="left")
    except Exception as _ex:  # noqa: BLE001
        print(f"   ⚠️ organization_code not read from o_custom_Model ({_ex}); add an "
              "'OrganizationCode' attribute to the Model object type. Leaving org null.")
    # ✅ total_daily_capacity_lot (2026-07-15): Σ JigCapa per model from JIGDetail⋈JIGMaster
    # (JigCapa = final lots/day per jig row) — fills the pModel rollup PPS has no column for.
    # Models without jigs stay null -> engine default. REVPLAN_JIG_LOT_CAPACITY=off to skip.
    _jig_capa = _jig_daily_lot_capacity()
    if _jig_capa is not None:
        df = df.join(_jig_capa, on="model_id", how="left")
    # ⚠️ base_model / grouping_model: no source column -> null (derive a grouping rule later).
    return _add_missing(df, ["base_model", "grouping_model", "total_daily_capacity_lot", "organization_code"])


# ---- ✅ model_unit_conversion  <-  PKG_MGR.PK1_MODEL --------------------------
def read_model_unit_conversion(params) -> pl.DataFrame:
    df = _pull("PK1_MODEL", {
        "model_id":        "MODEL_NO",
        "units_per_panel": "EA_IN_PANNEL",
        "units_per_sheet": "EA_IN_SHEET",
        "_lot_size":       "LOT_SIZE",
    }).unique("model_id")
    if df.height:
        df = df.with_columns([
            pl.col("units_per_panel").cast(pl.Float64, strict=False),
            pl.col("units_per_sheet").cast(pl.Float64, strict=False),
        ])
        # panels_per_sheet = units_per_sheet / units_per_panel (guard /0)
        df = df.with_columns(
            pl.when(pl.col("units_per_panel") > 0)
              .then(pl.col("units_per_sheet") / pl.col("units_per_panel"))
              .otherwise(None).alias("panels_per_sheet")
        )
        # ✅ panels_per_lot = (LotSize ÷ EaInSheet) × panels_per_sheet — LotSize is EA
        # (data-verified 2026-07-15; ÷EaInSheet = 5 sheets, ×6 = the standard 30 panels).
        # Consistent with model_master's maximum_lot_size_sht (sheets) × 6 path.
        _sheets_per_lot = (
            pl.when(pl.col("units_per_sheet") > 0)
              .then(pl.col("_lot_size").cast(pl.Float64, strict=False) / pl.col("units_per_sheet"))
              .otherwise(None)
        )
        if os.environ.get("REVPLAN_LOTSIZE_UNIT", "ea").strip().lower() == "sheets":
            _sheets_per_lot = pl.col("_lot_size").cast(pl.Float64, strict=False)
        df = df.with_columns(
            (_sheets_per_lot * pl.col("panels_per_sheet")).alias("panels_per_lot")
        ).drop("_lot_size")
    return _add_missing(df, ["panels_per_sheet", "panels_per_lot"])


# Honor EquipmentGroup.InfiniteCapaYn by handing the engine a very large capacity
# (build_equipment_capacity_lookup uses any non-null capacity verbatim). Same intent as
# config.infinite_capacity_sheets, kept local so celonis_io stays config-free.
_INFINITE_CAPA_SHT = 10_000_000


# ---- ✅ equipment_capacity  <-  o_custom_Equipment (+ o_custom_EquipmentGroup) -
def read_equipment_capacity(params) -> pl.DataFrame:
    df = _pull("PK1_EQUIPMENT", {
        "equipment_id":          "EQUIPMENT_CODE",   # -> EquipmentCode
        "_mapping_name":         "MAPPING_NAME",      # -> MappingName (mostly null; real group lives in EquipmentGroup)
        "daily_capacity_in_sht": "DAILY_CAPA",        # -> DailyCapa (null for outsourced -> engine defaults)
        "site_id":               "SITE",
        "_notuse_flag":          "NotuseFlag",        # decommissioned marker: 'Y' = retired
    })
    # Drop decommissioned equipment (NotuseFlag='Y') so allocation can't assign to it;
    # null / 'N' are kept as active. Safe on an empty frame (filter/drop no-op).
    _before = df.height
    df = df.filter(pl.col("_notuse_flag").fill_null("N") != "Y").drop("_notuse_flag")
    if _before != df.height:
        print(f"   ✓ equipment_capacity: dropped {_before - df.height} decommissioned (NotuseFlag=Y); {df.height} active")
    if df.height:
        df = df.with_columns(pl.col("daily_capacity_in_sht").cast(pl.Float64, strict=False))

    # equipment_group_id + infinite flag come from the dedicated EquipmentGroup object:
    # EquipmentCode -> OpMachineNm (the group NAME namespace that config.infinite_capacity_
    # groups and the capacity overrides key on). Equipment.MappingName is largely null, so
    # this is where the real taxonomy is. One equipment can sit in several groups -> collapse
    # to one row: first group name, and infinite if ANY of its groups is InfiniteCapaYn='Y'.
    try:
        grp = _pull("PK1_EQUIPMENT_GROUP", {
            "equipment_id": "EquipmentCode",
            "_group_name":  "OpMachineNm",
            "_infinite_yn": "InfiniteCapaYn",
        })
        grp = grp.group_by("equipment_id").agg([
            pl.col("_group_name").drop_nulls().first().alias("_grp_name"),
            (pl.col("_infinite_yn") == "Y").any().alias("_is_infinite"),
        ])
        df = df.join(grp, on="equipment_id", how="left").with_columns(
            pl.coalesce([pl.col("_grp_name"), pl.col("_mapping_name")]).alias("equipment_group_id")
        )
        # InfiniteCapaYn='Y' -> effectively-unbounded capacity so the group is never a
        # false bottleneck; otherwise keep DailyCapa (null -> engine default downstream).
        df = df.with_columns(
            pl.when(pl.col("_is_infinite").fill_null(False))
              .then(pl.lit(float(_INFINITE_CAPA_SHT)))
              .otherwise(pl.col("daily_capacity_in_sht"))
              .alias("daily_capacity_in_sht")
        )
        if df.height:
            n_grp = int(df.select(pl.col("_grp_name").is_not_null().sum()).item())
            n_inf = int(df.select(pl.col("_is_infinite").fill_null(False).sum()).item())
            print(f"   ✓ equipment groups: {n_grp}/{df.height} equipment mapped to a group "
                  f"({n_inf} flagged infinite-capacity)")
        df = df.drop(["_grp_name", "_is_infinite", "_mapping_name"])
    except Exception as ex:  # noqa: BLE001
        print(f"   ⛔ EquipmentGroup enrich skipped ({ex}); equipment_group_id <- MappingName")
        if "_mapping_name" in df.columns:
            df = df.rename({"_mapping_name": "equipment_group_id"})

    # ❌ process_id: no process link on the equipment master — derive from routing if needed.
    return _add_missing(df, ["process_id", "equipment_group_id"])


# ---- ❌ equipment_constraints  (NEGATIVE block-list — no source) ---------------
def read_equipment_constraints(params) -> pl.DataFrame:
    # MISSING: build_negative_constraints_lookup wants a (grouping_model, process_id) ->
    # blocked-equipment list, filtered on constraint_type == 'NEGATIVE'. The
    # o_custom_EquipmentConstraints object is POSITIVE (allowed assignments), so it feeds
    # equipment_to_process instead (see below), NOT this negative list. No negative feed
    # exists, so nothing is blocked — return empty (correct: the positive map alone governs
    # candidates; wire a real block-list here only if the business supplies one).
    return _empty("equipment_constraints",
                  ["grouping_model", "process_id", "equipment_id", "constraint_type"],
                  "MISSING — NEGATIVE block-list has no source; the EquipmentConstraints "
                  "object is positive and feeds equipment_to_process instead")


# ---- ✅ equipment_to_process  <-  o_custom_EquipmentConstraints ----------------
def read_equipment_to_process(params) -> pl.DataFrame:
    # The EquipmentConstraints object is a POSITIVE assignment list: one row per
    # (model, operation, equipment) meaning that equipment RUNS that operation (with a
    # preference rank EqptSeqInList + per-type daily capacities). That IS the engine's
    # process_id -> [equipment] capability map (build_process_to_equipment_lookup), which
    # allocation REQUIRES — an empty map fails every step with "No equipment mapped to
    # process". process_id keys on OperationCode because planned_process_steps.process_id
    # is OperationCode (COLUMN_MAP RTSP_MODEL_ROUTING_M.PROCID -> OperationCode) and the
    # allocation walks a model's routing steps: process_to_equipment.get(step.process_id).
    #
    # ⚠️ Model-agnostic: the positive map is keyed by process only, so equipment is pooled
    #    across every model sharing an OperationCode. Per-model precision needs the NEGATIVE
    #    block-list (read_equipment_constraints), which has no source yet.
    # SimulVersion: the object accumulates a stamped batch per regeneration (e.g.
    #    '20260703_011' = date_seq). We keep ONLY the latest version — the YYYYMMDD_seq
    #    format sorts chronologically as a string, so max() is newest. Older versions would
    #    over-permit equipment the current plan no longer assigns. (If the run should instead
    #    target a specific PlanId, filter on that here.) Pulled WITHOUT distinct so the
    #    version filter happens on raw rows, then deduped in polars.
    # Wrapped like _add_urgency_flag: a wrong object name / column degrades to the empty
    # placeholder (allocation stays sparse) instead of killing the whole read_inputs.
    try:
        base_cols = {
            "process_id":         "OperationCode",
            "equipment_id":       "EquipmentCode",
            "equipment_group_id": "EquipmentType",   # coarse type code; unused by the lookup, kept for reporting
            "_simul_version":     "SimulVersion",
        }
        # Optionally also pull a release-status column (§8, opt-in via REVPLAN_RELEASE_STATUS_COL).
        # If the object has no such column the PQL read errors, so retry without it.
        df = None
        if RELEASE_STATUS_COL:
            try:
                df = _pull("PK1_EQUIPMENT_CONSTRAINTS", {**base_cols, "_release_status": RELEASE_STATUS_COL})
            except Exception as ex:  # noqa: BLE001
                print(f"   ⚠ equipment_to_process: release-status column '{RELEASE_STATUS_COL}' not "
                      f"readable ({type(ex).__name__}); falling back to latest SimulVersion only")
                df = None
        if df is None:
            df = _pull("PK1_EQUIPMENT_CONSTRAINTS", base_cols)

        # Prefer RELEASED rows when a release-status column is present (§8: use the last
        # released version, never a 'Created/unreleased' one). Absent column → unchanged.
        if "_release_status" in df.columns and df.height:
            keep = df.filter(
                pl.col("_release_status").cast(pl.Utf8).str.strip_chars().str.to_uppercase().is_in(list(RELEASED_VALUES))
            )
            if keep.height:
                print(f"   ✓ equipment_to_process: kept {keep.height} Released rows "
                      f"(dropped {df.height - keep.height} non-released)")
                df = keep
            else:
                print(f"   ⚠ equipment_to_process: release-status present but no rows matched "
                      f"{sorted(RELEASED_VALUES)}; keeping all versions")
            df = df.drop("_release_status")

        if df.height and df["_simul_version"].drop_nulls().len():
            latest = df["_simul_version"].drop_nulls().max()
            df = df.filter(pl.col("_simul_version") == latest)
            print(f"   ✓ equipment_to_process: latest SimulVersion={latest}")
        df = df.drop("_simul_version").unique()
        df = df.filter(pl.col("process_id").is_not_null() & pl.col("equipment_id").is_not_null())
        print(f"   ✓ equipment_to_process: {df.height} distinct process↔equipment links "
              f"({df['process_id'].n_unique()} processes, {df['equipment_id'].n_unique()} equipment)")
        return _add_missing(df, ["process_id", "equipment_id", "equipment_group_id"])
    except Exception as ex:  # noqa: BLE001
        return _empty("equipment_to_process",
                      ["process_id", "equipment_id", "equipment_group_id"],
                      f"FALLBACK — read from o_custom_EquipmentConstraints failed ({ex}); "
                      "confirm object name + columns via the data-model introspection")


def _pull_jig_rows() -> pl.DataFrame:
    """Raw jig×model rows (model_id, jig_qty, jig_capa) from JIGDetail⋈JIGMaster.

    Shared by read_et_jig_master (risk-analysis feed) and _jig_daily_lot_capacity
    (per-model lot-start rate rollup for the model master). Raises with a
    diagnostic message when the objects/columns/relationship aren't readable —
    each caller degrades in its own way.
    """
    master = TABLE_MAP.get("PK1_JIG_MASTER", "o_custom_JIGMaster")
    detail = TABLE_MAP.get("PK1_JIG_DETAIL", "o_custom_JIGDetail")
    model_col = os.environ.get("REVPLAN_JIG_MODEL_COLUMN", "ModelNo")   # on JIGDetail
    qty_col   = os.environ.get("REVPLAN_JIG_QTY_COLUMN",   "JigQty")    # on JIGMaster
    capa_col  = os.environ.get("REVPLAN_JIG_CAPA_COLUMN",  "JigCapa")   # on JIGMaster
    try:
        from pycelonis.pql import PQL, PQLColumn
        import pycelonis.pql as pql
        dm = data_model()
        q = PQL()
        q += PQLColumn(name="model_id", query=f'"{detail}"."{model_col}"')
        q += PQLColumn(name="jig_qty",  query=f'"{master}"."{qty_col}"')
        q += PQLColumn(name="jig_capa", query=f'"{master}"."{capa_col}"')
        try:
            pdf = pql.DataFrame.from_pql(q, data_model=dm).to_pandas()
        except AttributeError:
            pdf = dm.export_data_frame(q)                               # SaolaPy absent -> legacy exporter
        df = pl.from_pandas(pdf)
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(
            f"read from {detail}⋈{master} failed ({ex}); confirm object names + columns "
            f"({detail}.{model_col}, {master}.{qty_col}/{capa_col}) and that the two objects "
            "are related in the data model — via data_model().get_tables()") from ex
    return df.filter(pl.col("model_id").is_not_null())


def _jig_daily_lot_capacity() -> Optional[pl.DataFrame]:
    """Per-model daily lot-start capacity: Σ JigCapa over the model's jigs.

    JigCapa is the FINAL daily lots/day figure per jig row (customer 2026-07-15:
    "이미 계산되어 결과(box)로 제공 → 그대로 사용"), so the model-level rate is a
    plain sum — this fills total_daily_capacity_lot, which Palantir's pModel
    carried but PPS has no direct column for (replacement summary §4 rollup gap).
    Feeds build_model_metadata_lookup -> daily_capacity_lots -> the virtual-lot
    start stagger. A jig shared by several models is counted fully for each
    (same open point as the risk analysis). Returns None to skip (env off or
    source unreadable) — models then keep the engine default.
    """
    if os.environ.get("REVPLAN_JIG_LOT_CAPACITY", "on").strip().lower() == "off":
        print("   ⚠ REVPLAN_JIG_LOT_CAPACITY=off — total_daily_capacity_lot stays null (engine default)")
        return None
    try:
        rows = _pull_jig_rows()
    except Exception as ex:  # noqa: BLE001
        print(f"   ⚠ jig lot-capacity rollup skipped ({ex}); total_daily_capacity_lot stays null")
        return None
    if rows.height == 0:
        return None
    agg = (
        rows.with_columns(pl.col("jig_capa").cast(pl.Float64, strict=False))
            .group_by("model_id")
            .agg(pl.col("jig_capa").sum().alias("total_daily_capacity_lot"))
            .filter(pl.col("total_daily_capacity_lot") > 0)
    )
    print(f"   ✓ total_daily_capacity_lot: Σ JigCapa for {agg.height} models (lots/day, JIG rollup)")
    return agg


def read_et_jig_master(params) -> pl.DataFrame:
    """ET-JIG capacity per model — feeds compute_et_jig_risk (the et_jig_capacity_risk_analysis port).

    The ported analysis (revplan_engine/et_jig_risk.py) is byte-identical to the Palantir source, so it
    still reads the source's Korean column names 대상_모델 / JIG대수 / Capa_Lot / Capa_Sheet / JIG_상태 —
    we emit exactly those.

    SOURCE IS TWO OCPM OBJECTS (confirmed 2026-07-10):
      * o_custom_JIGMaster — jig quantity + capacity (JigQty, JigCapa). It has NO model column.
      * o_custom_JIGDetail — the per-jig TARGET MODEL (ModelNo).
    The two objects are already connected in the data model, so a single PQL query that selects
    ModelNo from JIGDetail alongside JigQty/JigCapa from JIGMaster resolves the join automatically
    (no explicit FK needed). Result grain = one row per JIGDetail (jig×model) row carrying the
    parent jig's qty/capa — which is exactly the per-(jig, model) grain the analysis then groups by
    model. Object/column names are env-overridable so a rename never needs a code edit.

    ✅ CONFIRMED (customer meeting 2026-07-15): JigCapa is the FINAL capacity number
    ("이미 계산되어 결과(box)로 제공 → 재계산 없이 그대로 사용") — NOT per jig unit. The
    byte-identical analysis computes daily capacity as Σ(Capa_Lot × JIG대수)
    (et_jig_risk.py:384-385, correct for Palantir's per-jig source), so feeding JigCapa
    straight into Capa_Lot would inflate every JigQty>1 jig by ×JigQty. We therefore emit
    Capa_Lot = JigCapa ÷ JigQty (per-jig), so the analysis's ×JIG대수 reconstructs the
    confirmed total exactly, total_jig_units stays truthful, and lots_per_jig is a real
    per-jig average. Null/0 JigQty counts as 1.

    ✅ CONFIRMED (same meeting): no jig-status column exists anywhere — "everything in the
    jig table is Active" is the agreed business rule, so JIG_상태='정상' for all rows is no
    longer a proxy but the spec.

    Still open: a jig serving several models (e.g. JIGMaster_1008 → 2 JIGDetail rows) has
    its capacity counted fully for EACH model by the analysis's per-model group-by —
    shared-jig capacity is not split/competed. Capa_Sheet reuses JigCapa (the analysis
    computes but never uses it; JIGMaster.ATTRIBUTE3 ≈ JigCapa×5 is the likely real
    Capa_Sheet if ever needed).
    """
    _KOREAN = {"대상_모델": pl.Utf8, "JIG대수": pl.Int64, "Capa_Lot": pl.Float64,
               "Capa_Sheet": pl.Float64, "JIG_상태": pl.Utf8}
    try:
        df = _pull_jig_rows()
    except Exception as ex:  # noqa: BLE001
        return _empty("et_jig_master", _KOREAN, str(ex))
    if df.height == 0:
        return _empty("et_jig_master", _KOREAN, "JIGDetail⋈JIGMaster returned no rows")
    df = df.filter(pl.col("model_id").is_not_null())
    # JigCapa is the FINAL total (2026-07-15) but the verbatim analysis multiplies
    # Capa_Lot × JIG대수 — emit per-jig values so that product equals JigCapa again.
    _qty      = pl.col("jig_qty").cast(pl.Int64, strict=False).fill_null(1)
    _qty_safe = pl.when(_qty > 0).then(_qty).otherwise(1)
    _per_jig  = pl.col("jig_capa").cast(pl.Float64, strict=False) / _qty_safe
    out = df.select([
        pl.col("model_id").cast(pl.Utf8).alias("대상_모델"),
        _qty_safe.alias("JIG대수"),
        _per_jig.alias("Capa_Lot"),
        _per_jig.alias("Capa_Sheet"),
        pl.lit("정상").alias("JIG_상태"),
    ])
    print(f"   ✓ et_jig_master: {out.height} jig×model rows across {out['대상_모델'].n_unique()} models "
          "(Capa_Lot = JigCapa ÷ JigQty; analysis reconstructs the confirmed total via ×JIG대수)")
    return out


# ==============================================================================
# 6.  read_inputs  — assemble the dict the engine consumes.
# ==============================================================================
def read_inputs(params) -> Dict[str, pl.DataFrame]:
    print("=== Celonis read_inputs ===")
    return {
        "revenue_plan":          read_revenue_plan(params),           # ✅
        "available_inventory":   read_available_inventory(params),    # ✅ OnHand FGI on-hand (shipped/transit=0)
        "model_priorities":      read_model_priorities(params),       # ❌ placeholder
        "wip_lots":              read_wip_lots(params),               # ✅ (🔴 crosswalk)
        "planned_process_steps": read_planned_process_steps(params),  # ✅ (🔴 crosswalk)
        "model_master":          read_model_master(params),           # ✅
        "model_unit_conversion": read_model_unit_conversion(params),  # ✅
        "equipment_capacity":    read_equipment_capacity(params),     # ✅ + EquipmentGroup (group/infinite)
        "equipment_constraints": read_equipment_constraints(params),  # ❌ placeholder (no NEGATIVE source)
        "equipment_to_process":  read_equipment_to_process(params),   # ✅ o_custom_EquipmentConstraints (positive map)
        "et_jig_master":         read_et_jig_master(params),          # ✅ 2026-07-10 — o_custom_JIGMaster (feeds compute_et_jig_risk)
    }
    # NOTE — stub-analysis inputs (port alongside the stubs in run_simulation.py):
    #   model_boms          ✅  <- PK1_BOM   (MODEL_NO, MATERIAL_NO, WORK_SEQ, OPERATION_CODE, CHASU, REQ_QTY)
    #   materials           ✅  <- PK1_MATERIAL (MATERIAL_NO, MATERIAL_DESC, UOM, VENDOR_LEAD_TIME, THICKNESS)
    #   material_inventories ⚠️ <- PK1_DAILY_INVENTORY / PK1_ERP_ONHAND_LOT
    #   planned_material_arrivals ❌  (no PO/receipt feed)
    #   et_jig_master       ✅  <- o_custom_JIGMaster (wired 2026-07-10; see read_et_jig_master)


# ==============================================================================
# 7.  write_outputs  — push result tables back to the Data Pool.
# ==============================================================================
# APPEND-ONLY RUN HISTORY (2026-07-15): each SIM_* table is created once and
# APPENDED on every later run — never dropped — so the tables keep EVERY run's
# rows and the frontend can select runs via allocation_run_id (SIM_run_tracker
# is the run registry: one appended row per run). The previous behaviour
# (drop_if_exists on every run) kept only the latest run, clobbered OTHER
# simulations' results, and could even leave tables on DIFFERENT runs (0-row
# frames were skipped, so a stale table survived an otherwise-overwriting run).
# Read-side rule that goes with this: every view/transformation filters on
# allocation_run_id, defaulting to the latest SUCCESS row in SIM_run_tracker
# per simulation_id.
# Set CELONIS_OUTPUT_RESET=1 to intentionally drop + recreate all SIM_* tables
# (schema change, dev cleanup) — this DISCARDS previous runs' rows.
def write_outputs(results: Dict[str, pl.DataFrame], params) -> None:
    print("=== Celonis write_outputs (append-only run history) ===")

    # Refuse to write placeholder-stamped results. run_simulation stamps params.simulation_id
    # onto every allocation row + the run_tracker (run_id embeds it too), so writing a run
    # whose id is still 'local-test' (the Action Flow didn't inject a real dpInstanceId, or
    # this is a manual/dev run) would pollute the SIM_* tables with rows the OE can never
    # correlate. Skip loudly instead — symmetric with the notebook SKIPPING the OE resume
    # webhook on the same placeholder. Set CELONIS_ALLOW_PLACEHOLDER_WRITE=1 to force a dev write.
    sim_id = getattr(params, "simulation_id", None)
    if _is_placeholder_simulation_id(sim_id) and not os.environ.get("CELONIS_ALLOW_PLACEHOLDER_WRITE"):
        print(f"   ⛔ SKIP write_outputs: simulation_id is a placeholder ({str(sim_id)!r}). "
              "No real dpInstanceId was injected, so these rows would be stamped as a "
              "placeholder and pollute the SIM_* tables. Nothing written.\n"
              "      → Fix: have the Action Flow pass \"dpInstanceId\": \"{{1.instanceId}}\" in the "
              "/executions params (blueprint_trigger_revplan_mlwb.json already maps it), or set "
              "CELONIS_ALLOW_PLACEHOLDER_WRITE=1 for an intentional dev write.")
        return

    p = pool()
    reset = bool(os.environ.get("CELONIS_OUTPUT_RESET"))
    if reset:
        print("   ⚠ CELONIS_OUTPUT_RESET=1 — dropping + recreating SIM_* tables "
              "(previous runs' rows are DISCARDED)")

    def _string_column_config(frame):
        """Size each STRING column for the table CREATE (min 256, cap 4000, 2× the
        longest current value) so later runs' appends fit without truncation — the
        widths are fixed at creation, and appends must live inside them. Returns a
        list of {columnName, columnType, fieldLength} dicts, or None when there
        are no string columns."""
        str_cols = [c for c, dt in frame.schema.items() if dt == pl.Utf8]
        if not str_cols:
            return None
        cfg = []
        for c in str_cols:
            try:
                m = frame.get_column(c).drop_nulls().str.len_chars().max()
            except Exception:  # noqa: BLE001 — str method name / dtype varies across polars
                m = None
            cfg.append({"columnName": c, "columnType": "STRING",
                        "fieldLength": max(256, min(int(m) * 2, 4000)) if m else 256})
        return cfg

    def _find_table(name):
        """Existing Data Pool table by name, or None. Tolerates both pycelonis
        collection APIs (.find raising vs returning None) and falls back to a
        plain scan so an API quirk can't misreport 'missing' as 'exists'."""
        try:
            tables = p.get_tables()
        except Exception as ex:  # noqa: BLE001
            print(f"   ⚠ get_tables() failed ({ex}); treating '{name}' as not yet created")
            return None
        try:
            found = tables.find(name)
            if found is not None:
                return found
        except Exception:  # noqa: BLE001 — .find raises PyCelonisNotFoundError when absent
            pass
        for t in tables:
            if getattr(t, "name", None) == name:
                return t
        return None

    for name, df in results.items():
        table_name = f"{OUTPUT_PREFIX}{name}"
        if df.height == 0:
            # pycelonis rejects a 0-row create/append ("Can't add empty data frame").
            # Under append-only history this is harmless: nothing is dropped, and a run
            # that produced 0 rows simply contributes no rows for its allocation_run_id.
            print(f"   ⏭  skip {table_name} (0 rows this run — consumers filter by allocation_run_id)")
            continue
        pdf = df.to_pandas()

        # ---- append path: table exists and no reset requested -> add this run's rows
        existing = None if reset else _find_table(table_name)
        if existing is not None:
            try:
                existing.append(pdf)
                print(f"   ✓ appended {table_name}  (+{df.height} rows)")
            except Exception as ex:  # noqa: BLE001
                # Do NOT fall back to drop/recreate — that would silently erase the run
                # history this function exists to keep. Typical cause is schema drift
                # (new/renamed column, string longer than the column width chosen at
                # creation); resolve it deliberately.
                print(f"   ✗ append to '{table_name}' failed: {ex}\n"
                      "      → table NOT dropped (run history preserved). If the schema "
                      "changed intentionally, re-run with CELONIS_OUTPUT_RESET=1 to "
                      "recreate the SIM_* tables (discards previous runs).")
            continue

        # ---- create path: first ever run, or explicit CELONIS_OUTPUT_RESET=1
        cfg = _string_column_config(df)
        # column_config: preserve real string lengths (with append headroom). Retry
        # without it if this pycelonis build rejects its shape — still lands the data
        # (falling back to the VARCHAR(80) default).
        attempts = ([{"drop_if_exists": True, "force": True, "column_config": cfg}] if cfg else [])
        attempts.append({"drop_if_exists": True, "force": True})
        ok, last = False, None
        for kw in attempts:
            try:
                p.create_table(pdf, table_name, **kw)
                note = "" if "column_config" in kw else "  (⚠ default VARCHAR(80))"
                print(f"   ✓ created {table_name}  ({df.height} rows){note}")
                ok = True
                break
            except Exception as ex:  # noqa: BLE001
                last = ex
        if not ok:
            print(f"   ✗ write '{table_name}' failed: {last}")
    # NOTE: no SIMULATION_OE_Table status flip here — pool tables aren't row-updatable via
    #       pycelonis, and the OE now resumes via the notebook's finish-simulation event
    #       (orchestration.py), not a status data-trigger, so the flip isn't needed.


# ==============================================================================
# 8.  self-test — `python -m revplan_engine.celonis_io` lists the pool & model.
# ==============================================================================
if __name__ == "__main__":
    print("Connecting…")
    pl_pool = pool()
    print("Tables in pool:")
    for t in pl_pool.get_tables():
        print("  •", t.name)
    try:
        print("Data model:", data_model().name)
    except Exception as ex:  # noqa: BLE001
        print("Data model not found:", ex)
