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
from typing import Dict, List, Optional, Tuple

import polars as pl

# ------------------------------------------------------------------------------
# Config (overridable by env; sensible defaults from check_connection.py)
# ------------------------------------------------------------------------------
# Defaults are the CONFIRMED ids on lg-innotek.eu-1 (from the connect log), so a fresh kernel
# resolves with NO env vars. _resolve() accepts an id OR a name, so either works as an override.
POOL_NAME       = os.environ.get("CELONIS_DATA_POOL",  "41c041fa-1a49-4afa-8799-7779fa61e86c")  # "2. Simulation"
DATA_MODEL_NAME = os.environ.get("CELONIS_DATA_MODEL", "6570e74c-ff78-464e-bd16-142101117d26")  # test:perspective_custom_Simulation — the FRONTEND's perspective (switched 2026-08-04 so notebook+frontend read ONE model; o_custom_WipHistory lives here). Non-test twin (old default): 73bc8779-7ddb-45a0-81dc-5e7f87c6ac92 — restore via env var if the twins turn out to have drifted.
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

# Which SubinventoryCode rows of the coarse OnHand snapshot count as RAW MATERIAL stock
# (read_material_inventories → material_depletion). The object mixes material rows
# ('RAW-MTL', ItemNo = material code) with finished-goods rows ('FGI', ItemNo = MODEL_NO);
# only the material rows belong in material_inventories.
MATERIAL_SUBINV_CODES = frozenset(
    v.strip().upper()
    for v in os.environ.get("REVPLAN_MATERIAL_SUBINV_CODES", "RAW-MTL").split(",")
    if v.strip()
)

# Drop unusable equipment everywhere (2026-07-16): flagged machines must contribute
# no capacity and no candidacy. TWO flags, TWO switches:
#   * NotuseFlag='Y'  — trusted decommission marker, filtered BY DEFAULT.
#   * ActiveFlag='N'  — filtered BY DEFAULT since 2026-07-30: the customer confirmed
#     비활성 설비 제외 (only ActiveFlag='Y' machines are planned on; null counts as
#     active). ⚠️ Know what this costs: ~2,000 of ~3,500 machines carry 'N'
#     (measured 2026-07-16), so coverage shrinks hard, more processes lose ALL
#     machines (audited in read_equipment_to_process — those become outsourced
#     pass-throughs), and saturation deepens (runtime is protected by the scan
#     memo). Kill switch: REVPLAN_EQUIPMENT_ACTIVEFLAG=0 -> NotuseFlag only.
# REVPLAN_EQUIPMENT_ACTIVE_ONLY=0 disables BOTH (include-everything parity mode).
EQUIPMENT_ACTIVE_ONLY = os.environ.get("REVPLAN_EQUIPMENT_ACTIVE_ONLY", "1") != "0"
EQUIPMENT_USE_ACTIVEFLAG = os.environ.get("REVPLAN_EQUIPMENT_ACTIVEFLAG", "1") == "1"

# BOM CHASU (차수 / revision round) handling for read_model_boms:
#   "per_key"          (default) — per (model, op, seq, material) keep the row with the
#                       highest CHASU. Never double-counts a material, never drops one
#                       that only appears in an older round.
#   "latest_per_model" — keep only rows whose CHASU == the model's max CHASU. Correct if
#                       every round re-lists the FULL bom; drops materials otherwise.
BOM_CHASU_MODE = os.environ.get("REVPLAN_BOM_CHASU_MODE", "per_key").strip().lower()

# ACTIVE-MATERIAL WHITELIST (2026-07-27) — the fix for the 자재 커버리지 problem.
# o_custom_BOM is the ENGINEERING bom: every material ever specified for a model
# across all CHASU rounds, including obsolete revisions and alternates. Measured
# 2026-07-16: only ~11% of its materials have a RAW-MTL on-hand row, so
# material_depletion treated the rest as zero stock and flagged thousands of
# phantom shortages (31,688 constrained lots, most of them artifacts).
# WipDaily.BomMaterialList is the OPERATIONAL list — the materials the MES actually
# expects to be issued for a lot — and its materials match on-hand at ~74%. Using it
# as a WHITELIST over the BOM keeps everything the simulation needs from the BOM
# (ReqQty for volume, OperationCode/WorkSeq for timing) and drops the dead materials.
#   DEFAULT ON again (2026-07-30): the 2026-07-29 failure hunt exonerated the
#   whitelist — the week's triggered-run deaths were the YYYYMM-in-start_date
#   flow param (fixed in the notebook), and the 07-30 full run completed with
#   read_inputs healthy. Kill switch: REVPLAN_BOM_ACTIVE_ONLY=0 (set the env var
#   in any notebook cell before celonis_io is first imported).
BOM_ACTIVE_ONLY = os.environ.get("REVPLAN_BOM_ACTIVE_ONLY", "1") == "1"
# A model with NO WipDaily list — a new model whose demand is served entirely by
# virtual lots — has no per-model whitelist. How its BOM rows are treated:
#   "global" (default) — filter against the union of ALL models' active materials
#                        (kills obsolete materials without over-filtering the model)
#   "keep"             — leave that model's BOM unfiltered (old behaviour, per model)
#   "drop"             — drop the model's BOM rows entirely (aggressive; not advised)
BOM_WHITELIST_FALLBACK = os.environ.get("REVPLAN_BOM_WHITELIST_FALLBACK", "global").strip().lower()


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
    "PK1_BOM":                   "o_custom_BOM",                  # ✅ 2026-07-16 — model×op×material requirements (ReqQty per EA) → model_boms for material_depletion
    "PK1_MATERIAL":              "o_custom_Material",             # ✅ 2026-07-16 — material master; only UOM is joined onto model_boms (BOM has no UOM column)
    "PK1_PO_ARRIVE_PLAN":        "o_custom_PoArrivePlan",         # ✅ 2026-07-16 — PO arrival plan snapshot (BatchDate/BatchHour batched) → planned_material_arrivals
    "RTSP_MGR_WIP_HISTORY_N":    "o_custom_WipHistory",           # ⚠ LEGACY since 2026-08-19 (kill switch REVPLAN_ACTUALS_SOURCE=wiphistory) — replaced by o_custom_WIP
    "RTSP_WIP_HOURLY":           "o_custom_WIP",                  # ✅ 2026-08-19 — hourly WIP snapshot (RTSP_WIP_N projection, YYYYMMDDHH) → actuals + prev-step durations (business decision: WIP replaces WipHistory)
    "PK1_SO_LINE":               "o_custom_SalesOrderLine",       # ✅ 2026-08-26 — sales order lines → committed/uncommitted demand (uncommitted_demand.py port)
}
COLUMN_MAP: Dict[str, Dict[str, str]] = {
    "PK1_MPLAN": {
        "MODEL_NO":    "ModelNo",
        "REVISION_NO": "RevisionNo",
        "YYYYMM":      "YYYYMM",
        "PLAN_QTY":    "PlanQty",
        "PLAN_AMT":    "PlanAmt",
        # ✅ 2026-09-09 — LG added to the object: PK1_MODEL.ATTRIBUTE4 joined on
        # (org, model). Read ONLY by read_grouping_model_map (stage-1 analysis
        # axis); read_revenue_plan deliberately does NOT pull it — see the
        # warning there.
        "GROUPING_MODEL": "GroupingModel",
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
        # ✅ 2026-07-27 — the MES's OPERATIONAL material list for the lot, e.g.
        # '5PPP00118A(투입예정/충분), SMFICF00001-Z(기투입/해당없음)'. Verified: identical at
        # every operation and every snapshot of a lot (route-level, not per-step), PK1 only
        # (FK1 is 100% null), max length 876 chars = not truncated. Feeds _active_materials().
        "BOM_MATERIAL_LIST":     "BomMaterialList",
        "BOM_MATERIAL_SHORTAGE": "BomMaterialShortageCnt",
        "BOM_MATERIAL_TOTAL":    "BomMaterialTotalCnt",
        # ✅ 2026-08-10 — lot hold status (object definition confirmed: both verbatim).
        # STATE ∈ {HOLD, WAIT, PROC}; WIPHOLD ∈ {Y, N} (Y ⟺ STATE=HOLD). Holds are
        # released over time, so the latest-snapshot read gives "currently held".
        "STATE":   "STATE",
        "WIPHOLD": "WIPHOLD",
    },
    "RTSP_MODEL_ROUTING_M": {
        "PRODID":          "ModelNo",         # ✅ routing carries the PPS MODEL_NO directly -> NO crosswalk
        "SEQ":             "WorkSeq",
        "PROCID":          "OperationCode",   # ⚠️ ASSUMPTION — confirm process-id source (OperationCode vs OpCode)
        "PROCNAME":        "OperationName",
        "OP_MACHINE_CODE": "ModifiedGroup",   # ✅ the real equipment group (better than the raw machine code)
        "CHASU":           "CHASU",           # ✅ 2026-08-06 — routing 차수 (confirmed on the object) -> SimWIPMaster
        "GUBUN":           "GubunColumn",     # ✅ 2026-08-06 — process-group label (투입/외층/SR …) -> SimWIPMaster
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
    "PK1_BOM": {
        "MODEL_NO":       "ModelNo",        # MODEL_NO form (SPCCE3000C.K000) → joins allocation model_id directly
        "MATERIAL_NO":    "MaterialNo",     # material code (SMFICF00001-Z / 5PPP01253A style)
        "WORK_SEQ":       "WorkSeq",        # float in source (61.0) — cast to the allocation's Int32 sequence for the join
        "OPERATION_CODE": "OperationCode",  # joins allocation process_id
        "CHASU":          "CHASU",          # 차수 (BOM revision round) — deduped in read_model_boms
        "REQ_QTY":        "ReqQty",         # required qty PER UNIT (EA) — × final_production_units = consumption
    },
    "PK1_MATERIAL": {
        "MATERIAL_NO": "MaterialNo",
        "UOM":         "UOM",
        "COP_CLASS":   "CopClass",   # 자재 등급 (SimStockMaster; possibly the ABC/D classification)
    },
    "PK1_PO_ARRIVE_PLAN": {
        "PART_NO":    "PartNo",     # material code — same key space as BOM MaterialNo
        "PLAN_DATE":  "PlanDate",   # arrival date (datetime at midnight) → cast to Date
        "QTY":        "QTY",
        "BATCH_DATE": "BatchDate",  # snapshot batch — read_planned_material_arrivals keeps the latest batch only
        "BATCH_HOUR": "BatchHour",
    },
    # o_custom_WIP (2026-08-19): hourly WIP snapshot — aliases confirmed from the
    # object SQL (SELECT ... FROM RTSP_WIP_N). OP_CODE is the UN-prefixed
    # operation code (unlike site-prefixed PROCID) — always join on OpCode.
    "RTSP_WIP_HOURLY": {
        "YYYYMMDDHH":             "YYYYMMDDHH",
        "LOTID":                  "LOTID",
        "PRODID":                 "PRODID",
        "OP_CODE":                "OpCode",
        "SEQ":                    "SEQ",
        "EQPTID":                 "EQPTID",
        "PLANNING_SITE_CODE":     "PlanningSiteCode",
        "APS_PRODCATEGORY":       "ApsProdcategory",
        "WIPDTTM_ST":             "WipdttmSt",
        "PREV_ACTUAL_EQPTID":     "PrevActualEqptid",
        "PREV_ACTUAL_START_DATE": "PrevActualStartDate",
        "PREV_ACTUAL_END_DATE":   "PrevActualEndDate",
        "WIPSHTQTY":              "WIPSHTQTY",
        "WIPPNLQTY":              "WIPPNLQTY",
        "WIPUNITQTY":             "WIPUNITQTY",
        "G_LOT_CREATE_DTTM":      "GLotCreateDttm",
    },
    # o_custom_SalesOrderLine (2026-08-26): aliases confirmed from the object extract.
    "PK1_SO_LINE": {
        "MODEL_ID":         "ModelId",          # full MODEL_NO with suffix (SPSCPB000M.K030) — same key space as allocation model_id
        "ORDER_QUANTITY":   "OrderQuantity",    # EA
        "SHIPPED_QUANTITY": "ShippedQuantity",  # EA
        "LINE_STATUS":      "LineStatus",       # Awaiting Shipping / Booked / Cancelled / Closed / Entered / Picked (Partial)
    },
    # o_custom_WipHistory (2026-08-04): object aliases from the user's SQL — some
    # raw names kept verbatim (LOTID/PRODID/PROCID/YYYYMMDD), timestamps + site
    # camelized. Only the columns read_measured_step_durations needs.
    "RTSP_MGR_WIP_HISTORY_N": {
        "LOTID":              "LOTID",
        "PRODID":             "PRODID",
        "PROCID":             "PROCID",             # site-prefixed (PK1MF18N) — stripped in the aggregator
        "PLANNING_SITE_CODE": "PlanningSiteCode",
        "WIPDTTM_ST":         "WipdttmSt",
        "WIPDTTM_ED":         "WipdttmEd",
        "YYYYMMDD":           "YYYYMMDD",
        # SimWIPMaster lot summary (2026-08-05):
        "EQPTID":             "EQPTID",
        "APS_PRODCATEGORY":   "ApsProdcategory",
        "WIPSHEETQTY_ED":     "WipsheetqtyEd",
        "WIPPNLQTY_ED":       "WippnlqtyEd",
        "WIPQTY_ED":          "WipqtyEd",
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
def _pull(table: str, colmap: Dict[str, str], distinct: bool = False,
          filters: Optional[List[str]] = None) -> pl.DataFrame:
    """PQL-read `table`, aliasing each source column to the engine column name.

        colmap = {engine_col: SOURCE_COL}   ->  SELECT "TABLE"."SOURCE_COL" AS engine_col

    Returns a polars frame whose columns are already the engine's names.
    `table`/`src` are the engine's LOGICAL PPS names; they are translated to the
    actual data-model names via TABLE_MAP / COLUMN_MAP before the query is built.
    `filters` are raw PQL FILTER statements (already in DATA-MODEL column names —
    build them via _pql_col) applied SERVER-SIDE, so a huge source (e.g. a decade
    of hourly snapshot batches) never has to travel to the notebook kernel.
    """
    from pycelonis.pql import PQL, PQLColumn, PQLFilter
    import pycelonis.pql as pql
    dm = data_model()
    actual_table = TABLE_MAP.get(table, table)
    colrename = COLUMN_MAP.get(table, {})
    q = PQL(distinct=distinct)
    for f in (filters or []):
        q += PQLFilter(query=f)
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


def _pql_col(table: str, src: str) -> str:
    """Fully-qualified data-model column reference for raw PQL (FILTER strings etc.)."""
    actual_table = TABLE_MAP.get(table, table)
    actual_src = COLUMN_MAP.get(table, {}).get(src, src)
    return f'"{actual_table}"."{actual_src}"'


def _latest_batch_filter(table: str, what: str, date_col: str = "BATCH_DATE") -> Optional[List[str]]:
    """Server-side FILTER limiting a date-stamped snapshot table to its newest day.

    o_custom_OnHand / o_custom_PoArrivePlan carry EVERY historical batch (multiple
    per day since 2015/2018) and o_custom_WipDaily carries ~67 daily snapshots —
    pulling any of them whole OOMs the notebook kernel. This probes MAX(<date_col>)
    with a one-row aggregate query, then returns a `FILTER col >= {d'<that day>'}`
    so only the newest day travels; _latest_batch() still refines to the exact
    batch+hour in polars where that applies.
    Returns None when the probe fails (caller decides whether to risk a full pull).
    """
    from pycelonis.pql import PQL, PQLColumn
    import pycelonis.pql as pql
    col = _pql_col(table, date_col)
    try:
        q = PQL()
        q += PQLColumn(name="max_bd", query=f"MAX({col})")
        try:
            pdf = pql.DataFrame.from_pql(q, data_model=data_model()).to_pandas()
        except AttributeError:
            pdf = data_model().export_data_frame(q)
        max_bd = pdf["max_bd"].iloc[0] if len(pdf) else None
        if max_bd is None:
            print(f"   ⚠ {what}: MAX(BatchDate) probe returned nothing")
            return None
        day = str(max_bd)[:10]  # 'YYYY-MM-DD' from Timestamp/str alike
        print(f"   ✓ {what}: newest snapshot day {day} — filtering server-side")
        return [f"FILTER {col} >= {{d'{day}'}}"]
    except Exception as ex:  # noqa: BLE001
        print(f"   ⚠ {what}: MAX(BatchDate) probe failed ({ex})")
        return None


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
        "amount_krw":      "PLAN_AMT",     # ✅ customer confirmed 2026-07-30: value used AS IS (plain KRW — no 백만원 scaling; PK1_MPLAN has no currency col)
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
        _rp_raw = str(getattr(params, "revenue_plan_id", None) or "").strip()
        # 2026-07-30 run: the Action Flow sent "'MP202607-27W-001'" WITH literal
        # apostrophes -> matched no revision -> silent newest fallback, and the
        # quoted string was stamped on every allocation row, so demand_shortfall
        # and production_risk_reconciliation joined 0 rows. Strip wrapping
        # quote characters before matching.
        _rp = _rp_raw.strip("'\"").strip()
        if _rp != _rp_raw:
            print(f"   ⚠ revenue_plan: stripped quote characters from revenue_plan_id "
                  f"{_rp_raw!r} -> {_rp!r} (fix the Action Flow mapping with "
                  "replace(...; \"'\"; \"\") to stop sending them)")
        _revs = df["revenue_plan_id"].unique().to_list()
        if _rp and _rp in _revs:
            df = df.filter(pl.col("revenue_plan_id") == _rp)
            _chosen = _rp
            print(f"   ✓ revenue_plan: revision {_rp} selected — {df.height:,} rows "
                  f"(of {len(_revs)} revisions in PK1_MPLAN)")
        else:
            # No/unknown revision param -> newest revision key (MPyyyymm-woW-nnn sorts
            # chronologically). ⚠️ approximates the customer's "last RELEASED" rule —
            # PK1_MPLAN exposes no release-status column yet; revisit when it does.
            _latest = max(_revs)
            df = df.filter(pl.col("revenue_plan_id") == _latest)
            _chosen = _latest
            print(f"   ⚠ revenue_plan: revenue_plan_id={_rp!r} not found "
                  f"(revisions: {len(_revs)}) — falling back to NEWEST revision {_latest} "
                  f"({df.height:,} rows). NOTE: newest-by-name approximates the "
                  "'last released' rule (no release-status column on PK1_MPLAN).")
        # Self-healing stamp: write the revision ACTUALLY simulated back onto the
        # params, so run_simulation stamps allocation rows with the id the demand
        # data really came from — the downstream analyses join allocations to
        # demand on this id, and a raw/stale parameter empties them silently.
        if getattr(params, "revenue_plan_id", None) != _chosen:
            try:
                params.revenue_plan_id = _chosen
                print(f"   ✓ revenue_plan: params.revenue_plan_id set to the revision "
                      f"actually used ({_chosen}) so allocation stamps match the demand data")
            except Exception:  # noqa: BLE001 — read-only params object: stamps stay raw
                print("   ⚠ revenue_plan: could not update params.revenue_plan_id "
                      f"(read-only?) — allocation rows will carry {_rp_raw!r}")
        # sales_team is NOT on PK1_MPLAN -> bring it from the model master (PK1_MODEL).
        try:
            mm = _pull("PK1_MODEL", {"model_id": "MODEL_NO", "sales_team": "SALES_TEAM"}).unique("model_id")
            df = df.join(mm, on="model_id", how="left")
        except Exception:  # noqa: BLE001
            df = df.with_columns(pl.lit(None).alias("sales_team"))
    # ❌ margin_krw: no cost/margin column anywhere in PPS/RTS -> null.
    # ⚠️ grouping_model: o_custom_MovePlan DOES carry GroupingModel since 2026-09-09,
    # but this read keeps it NULL ON PURPOSE (stage 1). Populating it here would
    # flip the netting join and the demand keys to group grain while
    # available_inventory stays model-keyed — the keys stop matching (net demand
    # inflates) and demand lands on group ids without routings (unrouted). That
    # grain switch is stage 2 and needs LG's 3 business rules (대표 모델 선정 /
    # 공정 기준 / 신규 lot 귀속) first. The analysis-axis mapping is read
    # separately by read_grouping_model_map and stamped onto OUTPUTS only.
    return _add_missing(df, ["margin_krw", "grouping_model", "sales_team", "revenue_type"])


# ---- ✅ grouping_model_map  <-  PK1_MPLAN.GroupingModel (stage 1, 2026-09-09) --
def read_grouping_model_map(params) -> pl.DataFrame:
    """model_id -> grouping_model, from the MovePlan object's GroupingModel column
    (= PK1_MODEL.ATTRIBUTE4, joined into the object by LG on 2026-09-09).

    ⚠️ MLWB ADDITION — stage-1 analysis axis ONLY: run_simulation stamps this onto
    output tables that already carry a grouping_model column, so 재원/매출/부족
    aggregations can group by it in the frontend. It never enters the netting or
    allocation engine (numbers unchanged) — that is stage 2, gated on business rules.
    """
    _schema = {"model_id": pl.Utf8, "grouping_model": pl.Utf8}
    try:
        df = _pull("PK1_MPLAN", {"model_id": "MODEL_NO",
                                 "grouping_model": "GROUPING_MODEL"}, distinct=True)
    except Exception as ex:  # noqa: BLE001 — older data model without the column
        print(f"   ⚠ grouping_model_map: GroupingModel not readable from o_custom_MovePlan "
              f"({type(ex).__name__}) — grouping analysis axis stays null this run "
              "(has the data model been reloaded since the column was added?)")
        return pl.DataFrame(schema=_schema)

    df = (df.with_columns([
            pl.col("model_id").cast(pl.Utf8).str.strip_chars(),
            pl.col("grouping_model").cast(pl.Utf8).str.strip_chars(),
          ])
          .filter(pl.col("model_id").is_not_null() & (pl.col("model_id") != "")
                  & pl.col("grouping_model").is_not_null() & (pl.col("grouping_model") != ""))
          .unique())
    # One grouping per model expected (ATTRIBUTE4 is on the model master). If the
    # join ever fans out (e.g. org-code collisions), keep one deterministically and
    # say so — a silent random pick would make the analysis axis unstable across runs.
    _conflicts = (df.group_by("model_id").len().filter(pl.col("len") > 1))
    if _conflicts.height:
        print(f"   ⚠ grouping_model_map: {_conflicts.height} models carry >1 GroupingModel "
              "value — keeping the lexicographic first per model (ask LG which is right)")
    out = df.sort(["model_id", "grouping_model"]).unique(subset=["model_id"], keep="first")
    print(f"   ✓ grouping_model_map: {out.height:,} models mapped to "
          f"{out['grouping_model'].n_unique():,} grouping models (MovePlan.GroupingModel)")
    return out.select(list(_schema.keys()))


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

# ⚠️ MLWB ADDITION (2026-08-10) — HOLD-lot exclusion. Baseline (= Palantir) schedules
# lots that are currently ON HOLD (STATE=HOLD / WIPHOLD=Y at the latest snapshot) as if
# they were free to run — an optimistic bias, since sampled lots sit held for months.
# With REVPLAN_EXCLUDE_HOLD_LOTS=1 those lots are dropped from wip_lots BEFORE the
# remaining-route explosion (their demand then surfaces as shortfall/virtual lots
# instead of phantom fulfilled supply). Engine default OFF; the notebook param
# `exclude_hold` turns it ON for triggered runs (same pattern as measured_lt).
def _exclude_hold_enabled() -> bool:
    return os.environ.get("REVPLAN_EXCLUDE_HOLD_LOTS", "0") == "1"


# held lot ids of the current read — consumed by _add_urgency_flag to warn when an
# URGENT lot was dropped for being on hold (HOLD wins; the conflict must be visible).
_HELD_LOT_IDS: set = set()


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
        # ⚠️ BOUNDED READ (2026-07-29): WipDaily accumulates one full snapshot per day
        # (~50k lots × ~80 days ≈ 4M rows and growing). This read used to pull ALL of
        # it and keep one day in polars — the pull grew ~50k rows/day until it OOM'd
        # the kernel (~8-min silent deaths since 07-27; last success 07-16 at ~67
        # days). The engine only ever uses the LATEST snapshot, so filter to it (and
        # the site scope) SERVER-SIDE; the polars latest-snapshot/dedupe logic below
        # still runs and yields an identical frame. Probe failure falls back to the
        # unbounded pull (wip_lots is an essential input) with a loud warning.
        wip_flt = _latest_batch_filter("RTSP_WIP_N", "wip_lots (WipDaily)",
                                       date_col="SnapshotDate")
        if wip_flt is None:
            print("   ⚠ wip_lots: MAX(SnapshotDate) probe failed — UNBOUNDED WipDaily pull "
                  "(OOM risk grows daily; investigate before it recurs)")
            wip_flt = []
        elif WIP_SITES:
            sites = ", ".join(f"'{s}'" for s in sorted(WIP_SITES))
            wip_flt = wip_flt + [f"FILTER {_pql_col('RTSP_WIP_N', 'PlanningSiteCode')} IN ({sites})"]
        # receipt_target_day: the lot's real due date (WipDaily.ReceiptTargetDay,
        # ~58% filled — G4 2026-07-16). Urgent lots use it as their target month
        # instead of the synthetic start-month stamp; null falls back downstream.
        # Ladder degrades column-by-column: hold-status cols first (older objects
        # without STATE/WIPHOLD just lose the HOLD exclusion), then the due date.
        _hold_cols = {"_state": "STATE", "_wiphold": "WIPHOLD"}
        _snap_cols = {"_snapshot_date": "SnapshotDate", "_site": "PlanningSiteCode"}
        _attempts = [
            {**cols, **_hold_cols, "receipt_target_day": "ReceiptTargetDay", **_snap_cols},
            {**cols, "receipt_target_day": "ReceiptTargetDay", **_snap_cols},
            {**cols, **_snap_cols},
        ]
        for _i, _cm in enumerate(_attempts):
            try:
                df = _pull("RTSP_WIP_N", _cm, filters=wip_flt)
                if "_state" not in df.columns and _exclude_hold_enabled():
                    print("   ⚠ wip_lots: STATE/WIPHOLD not readable — HOLD exclusion "
                          "requested (REVPLAN_EXCLUDE_HOLD_LOTS=1) but INACTIVE this run")
                break
            except Exception as ex:  # noqa: BLE001 — degrade to the next column set
                if _i == len(_attempts) - 1:
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

    # (2b) HOLD-lot exclusion (⚠️ MLWB ADDITION 2026-08-10, gated). After the
    # latest-snapshot dedupe "held" means CURRENTLY held — released holds show their
    # released state in the newest snapshot. Nulls are kept (unknown ≠ held).
    global _HELD_LOT_IDS
    _HELD_LOT_IDS = set()
    if df.height and ("_state" in df.columns or "_wiphold" in df.columns):
        _hold_expr = pl.lit(False)
        if "_wiphold" in df.columns:
            _hold_expr = _hold_expr | (
                pl.col("_wiphold").cast(pl.Utf8).str.strip_chars()
                  .str.to_uppercase().eq("Y").fill_null(False))
        if "_state" in df.columns:
            _hold_expr = _hold_expr | (
                pl.col("_state").cast(pl.Utf8).str.strip_chars()
                  .str.to_uppercase().eq("HOLD").fill_null(False))
        held = df.filter(_hold_expr)
        if held.height:
            _HELD_LOT_IDS = set(held["lot_id"].to_list())
            _sht = float(held["latest_sheet_quantity"].cast(pl.Float64, strict=False).sum() or 0)
            _ea = float(held["latest_unit_quantity"].cast(pl.Float64, strict=False).sum() or 0)
            if _exclude_hold_enabled():
                df = df.filter(~_hold_expr)
                print(f"   ✓ HOLD 제외: {held.height:,} lots ({_sht:,.0f} SHT / {_ea:,.0f} EA) "
                      f"excluded — STATE=HOLD/WIPHOLD=Y at latest snapshot; "
                      f"{df.height:,} schedulable lots remain")
            else:
                print(f"   ▶ HOLD lots: {held.height:,} currently held ({_sht:,.0f} SHT / "
                      f"{_ea:,.0f} EA) — KEPT and scheduled as free "
                      "(set REVPLAN_EXCLUDE_HOLD_LOTS=1 to exclude)")
    df = df.drop([c for c in ("_snapshot_date", "_site", "_state", "_wiphold") if c in df.columns])

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
    for extra in ({"run_lt": "RUN_LT", "wait_lt": "WAIT_LT", "plan_lt": "PlanLt",
                   "chasu": "CHASU", "gubun": "GUBUN"},
                  {"run_lt": "RUN_LT", "wait_lt": "WAIT_LT", "plan_lt": "PlanLt"},
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

        # HOLD beats urgent: a lot dropped by the HOLD exclusion cannot be urgent-
        # scheduled, but the conflict must be visible — MES says 급 while MES says hold.
        if _HELD_LOT_IDS and _exclude_hold_enabled():
            conflict = _HELD_LOT_IDS & set(urg["lot_id"].to_list())
            if conflict:
                sample = ", ".join(sorted(conflict)[:10])
                print(f"   ⚠ URGENT∩HOLD conflict: {len(conflict)} lot(s) are flagged urgent "
                      f"but currently ON HOLD — excluded from scheduling (HOLD wins): {sample}"
                      + (" …" if len(conflict) > 10 else ""))

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
    for extra in ({"run_lt": "RUN_LT", "wait_lt": "WAIT_LT", "plan_lt": "PlanLt",
                   "chasu": "CHASU", "gubun": "GUBUN"},
                  {"run_lt": "RUN_LT", "wait_lt": "WAIT_LT", "plan_lt": "PlanLt"},
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
_INACTIVE_EQUIPMENT: Optional[frozenset] = None


def _inactive_equipment() -> frozenset:
    """Equipment codes that must not be planned on.

    DEFAULT criterion: NotuseFlag='Y' only. ActiveFlag='N' joins the criterion only
    under REVPLAN_EQUIPMENT_ACTIVEFLAG=1 — measured 2026-07-16, ~2,000 of ~3,500
    machines carry ActiveFlag='N', so it cannot mean "retired"; filtering on it
    amputated 57% of the fleet and blew a run up into hours of 200-day failure
    scans. Semantics are a pending customer question.

    One master-data set, applied in BOTH places an unusable machine could leak in:
      * read_equipment_capacity — else it contributes daily sheets to its group;
      * read_equipment_to_process — else it stays a CANDIDATE and, being absent
        from the capacity lookup, would run on the engine's DEFAULT capacity
        (worse than counting its real capacity).
    Cached per process; empty set (+ warning) when the flags aren't readable, and
    always empty when REVPLAN_EQUIPMENT_ACTIVE_ONLY=0 (parity kill switch).
    """
    global _INACTIVE_EQUIPMENT
    if not EQUIPMENT_ACTIVE_ONLY:
        return frozenset()
    if _INACTIVE_EQUIPMENT is not None:
        return _INACTIVE_EQUIPMENT
    cols = {"equipment_id": "EQUIPMENT_CODE", "_active": "ActiveFlag", "_notuse": "NotuseFlag"}
    if not EQUIPMENT_USE_ACTIVEFLAG:
        cols.pop("_active")
    try:
        df = _pull("PK1_EQUIPMENT", cols)
    except Exception:  # noqa: BLE001 — ActiveFlag column may not exist on the object
        try:
            df = _pull("PK1_EQUIPMENT", {k: v for k, v in cols.items() if k != "_active"})
            print("   ⚠ equipment master has no readable ActiveFlag — filtering on NotuseFlag only")
        except Exception as ex:  # noqa: BLE001
            print(f"   ⚠ inactive-equipment probe failed ({ex}) — NO equipment filtered")
            _INACTIVE_EQUIPMENT = frozenset()
            return _INACTIVE_EQUIPMENT
    norm = lambda c, default: pl.col(c).cast(pl.Utf8).str.strip_chars().str.to_uppercase().fill_null(default)  # noqa: E731
    crit = norm("_notuse", "N") == "Y"
    if "_active" in df.columns:
        crit = crit | (norm("_active", "Y") == "N")
    bad = df.filter(crit)
    _INACTIVE_EQUIPMENT = frozenset(bad["equipment_id"].drop_nulls().to_list())
    if _INACTIVE_EQUIPMENT:
        which = "ActiveFlag=N or NotuseFlag=Y" if "_active" in df.columns else "NotuseFlag=Y"
        sample = ", ".join(sorted(_INACTIVE_EQUIPMENT)[:8])
        more = f" (+{len(_INACTIVE_EQUIPMENT) - 8} more)" if len(_INACTIVE_EQUIPMENT) > 8 else ""
        print(f"   ✓ unusable equipment excluded from planning: {len(_INACTIVE_EQUIPMENT)} "
              f"({which}): {sample}{more}")
    return _INACTIVE_EQUIPMENT


def read_equipment_capacity(params) -> pl.DataFrame:
    df = _pull("PK1_EQUIPMENT", {
        "equipment_id":          "EQUIPMENT_CODE",   # -> EquipmentCode
        "_mapping_name":         "MAPPING_NAME",      # -> MappingName (mostly null; real group lives in EquipmentGroup)
        "daily_capacity_in_sht": "DAILY_CAPA",        # -> DailyCapa (null for outsourced -> engine defaults)
        "site_id":               "SITE",
    })
    # Drop inactive machines (ActiveFlag='N' or NotuseFlag='Y') so they contribute no
    # capacity; null flags are kept as active. Safe on an empty frame (filter no-op).
    inactive = _inactive_equipment()
    if inactive:
        _before = df.height
        df = df.filter(~pl.col("equipment_id").is_in(list(inactive)))
        if _before != df.height:
            print(f"   ✓ equipment_capacity: dropped {_before - df.height} inactive; {df.height} active")
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
        # Inactive machines must not stay CANDIDATES: absent from the capacity lookup
        # they'd run on the engine's default capacity instead of disappearing.
        inactive = _inactive_equipment()
        if inactive:
            _before = df.height
            _procs_before = set(df["process_id"].to_list())
            df = df.filter(~pl.col("equipment_id").is_in(list(inactive)))
            if _before != df.height:
                print(f"   ✓ equipment_to_process: dropped {_before - df.height} links to inactive equipment")
                # AUDIT (2026-07-30): a process whose machines are ALL filtered out
                # vanishes from the candidate map and falls into the outsourced
                # pass-through (planned-complete via Plan LT, NO capacity cost) —
                # an inactive-equipment filter silently converting a capacity
                # constraint into a free pass is the optimistic-bias hole; make it
                # loud so every run states which operations it affected.
                _lost = sorted(_procs_before - set(df["process_id"].to_list()))
                if _lost:
                    _shown = ", ".join(_lost[:15]) + (f" (+{len(_lost) - 15} more)" if len(_lost) > 15 else "")
                    print(f"   ⚠ equipment_to_process: {len(_lost)} process(es) lost ALL machines to the "
                          f"inactive-equipment filter and will be treated as OUTSOURCED pass-through "
                          f"(no capacity cost — optimistic): {_shown}")
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


# ------------------------------------------------------------------------------
# 5c.  MATERIAL INPUTS  — the three frames material_depletion.py consumes
#      (wired 2026-07-16; unlocks the 자재 쇼티지 screen — M510N 프리프레그/카파포일).
#      Contracts come from the verbatim Palantir module:
#        model_boms(model_id, process_id, work_sequence, material_id, required_quantity, uom)
#        material_inventories(material_id, onhand_quantity)
#        planned_material_arrivals(material_id, plan_date, quantity)
# ------------------------------------------------------------------------------
_BOM_SCHEMA = {"model_id": pl.Utf8, "process_id": pl.Utf8, "work_sequence": pl.Int32,
               "material_id": pl.Utf8, "required_quantity": pl.Float64, "uom": pl.Utf8,
               "chasu": pl.Float64}
_MATINV_SCHEMA = {"material_id": pl.Utf8, "onhand_quantity": pl.Float64}
_ARRIVALS_SCHEMA = {"material_id": pl.Utf8, "plan_date": pl.Date, "quantity": pl.Float64}


def _latest_batch(df: pl.DataFrame, what: str) -> pl.DataFrame:
    """Keep only the newest BatchDate+BatchHour snapshot of a batch-stamped object.

    o_custom_OnHand and o_custom_PoArrivePlan are FULL periodic snapshots (like
    WipDaily): every batch re-states the whole picture, so mixing batches would
    multiply quantities. BatchHour is a zero-padded string ('07') — lexicographic
    max is the numeric max."""
    if df.height == 0 or "batch_date" not in df.columns:
        return df
    latest_date = df["batch_date"].max()
    df = df.filter(pl.col("batch_date") == latest_date)
    if "batch_hour" in df.columns and df["batch_hour"].null_count() < df.height:
        df = df.filter(pl.col("batch_hour") == df["batch_hour"].max())
    print(f"   ✓ {what}: latest snapshot batch {latest_date} "
          f"h={df['batch_hour'][0] if 'batch_hour' in df.columns and df.height else '?'} "
          f"({df.height} rows)")
    return df


_ACTIVE_MATERIALS: Optional[Tuple[Dict[str, frozenset], frozenset]] = None


def _parse_bom_material_list(raw: str) -> List[str]:
    """'5PPP00118A(투입예정/충분), SMFICF00001-Z(기투입/해당없음)' → ['5PPP00118A', 'SMFICF00001-Z'].

    Entries REPEAT when a material is consumed at several points of the route (a
    prepreg across 3 lamination layers appears 3×) — the caller dedupes. Material
    codes carry hyphens but never parentheses (verified 2026-07-27 across a full
    lot history), so splitting on '(' is unambiguous. Status tokens seen so far:
    투입예정/충분, 기투입/해당없음.
    """
    out: List[str] = []
    for token in (raw or "").split(","):
        code = token.split("(", 1)[0].strip()
        if code:
            out.append(code)
    return out


def _active_materials() -> Tuple[Dict[str, frozenset], frozenset]:
    """(model_id -> active materials, union of all active materials) from WipDaily.

    ⚠️ BOUNDED READ (2026-07-27, after an OOM): the list is per-LOT, so a plain
    DISTINCT barely dedupes — across ~67 daily snapshots that is ~2.3M rows of
    ~440-char strings (~1 GB) and it killed the kernel. Since the list is IDENTICAL
    at every snapshot and every operation of a lot (verified), the LATEST SNAPSHOT
    alone carries the whole picture: the read is filtered server-side to that day
    (and to the run's site scope), leaving ~48k rows before DISTINCT.
    Cached per process. Empty result (+ warning) disables the filter rather than
    silently emptying the BOM — a whitelist we could not read must never look like a
    whitelist that matched nothing.
    """
    global _ACTIVE_MATERIALS
    if _ACTIVE_MATERIALS is not None:
        return _ACTIVE_MATERIALS

    flt = _latest_batch_filter("RTSP_WIP_N", "active-material whitelist (WipDaily)",
                               date_col="SnapshotDate")
    if flt is None:
        # Refuse the unbounded pull — that is what OOM'd the kernel. No filter is
        # strictly better than no run.
        print("   ⚠ active-material whitelist: MAX(SnapshotDate) probe failed — refusing the "
              "unbounded WipDaily pull; BOM stays UNFILTERED")
        _ACTIVE_MATERIALS = ({}, frozenset())
        return _ACTIVE_MATERIALS
    if WIP_SITES:
        sites = ", ".join(f"'{s}'" for s in sorted(WIP_SITES))
        flt = flt + [f"FILTER {_pql_col('RTSP_WIP_N', 'PlanningSiteCode')} IN ({sites})"]

    try:
        df = _pull("RTSP_WIP_N", {"model_id": "PRODID", "_bom_list": "BOM_MATERIAL_LIST"},
                   distinct=True, filters=flt)
    except Exception as ex:  # noqa: BLE001 — column absent on this data model
        print(f"   ⚠ active-material whitelist: WipDaily.BomMaterialList not readable ({ex}) "
              "— BOM stays UNFILTERED (engineering BOM, phantom-shortage risk)")
        _ACTIVE_MATERIALS = ({}, frozenset())
        return _ACTIVE_MATERIALS

    df = df.filter(pl.col("model_id").is_not_null() & pl.col("_bom_list").is_not_null())
    print(f"   · active-material whitelist: {df.height:,} distinct (model, list) rows pulled")

    per_model: Dict[str, set] = {}
    global_set: set = set()
    for row in df.iter_rows(named=True):
        mats = _parse_bom_material_list(row["_bom_list"])
        if not mats:
            continue
        per_model.setdefault(row["model_id"], set()).update(mats)
        global_set.update(mats)

    if not global_set:
        print("   ⚠ active-material whitelist: no BomMaterialList values found "
              "— BOM stays UNFILTERED")
    else:
        print(f"   ✓ active-material whitelist: {len(global_set):,} distinct materials "
              f"across {len(per_model):,} models (from WipDaily.BomMaterialList)")
    _ACTIVE_MATERIALS = ({m: frozenset(s) for m, s in per_model.items()}, frozenset(global_set))
    return _ACTIVE_MATERIALS


def _apply_active_material_whitelist(df: pl.DataFrame) -> pl.DataFrame:
    """Drop BOM rows whose material the MES no longer issues (see BOM_ACTIVE_ONLY).

    Models WITH a WipDaily list are filtered against their own list; models without
    one (new models served only by virtual lots) follow BOM_WHITELIST_FALLBACK.
    """
    if not BOM_ACTIVE_ONLY or df.height == 0:
        return df
    per_model, global_set = _active_materials()
    if not global_set:
        return df  # unreadable whitelist -> no filtering (warned in _active_materials)

    known_models = list(per_model.keys())
    allow = pl.DataFrame(
        {"model_id": [m for m, mats in per_model.items() for _ in mats],
         "material_id": [x for mats in per_model.values() for x in mats]},
        schema={"model_id": pl.Utf8, "material_id": pl.Utf8},
    )
    is_known = pl.col("model_id").is_in(known_models)
    known_rows, unknown_rows = df.filter(is_known), df.filter(~is_known)

    kept_known = known_rows.join(allow, on=["model_id", "material_id"], how="semi")
    if BOM_WHITELIST_FALLBACK == "keep":
        kept_unknown = unknown_rows
    elif BOM_WHITELIST_FALLBACK == "drop":
        kept_unknown = unknown_rows.clear()
    else:  # "global"
        kept_unknown = unknown_rows.filter(pl.col("material_id").is_in(list(global_set)))

    out = pl.concat([kept_known, kept_unknown], how="vertical")
    print(f"   ✓ BOM active-material filter: {out.height:,} of {df.height:,} rows kept "
          f"({kept_known.height:,} from {known_rows['model_id'].n_unique() if known_rows.height else 0} "
          f"models with a WipDaily list; {kept_unknown.height:,} from "
          f"{unknown_rows['model_id'].n_unique() if unknown_rows.height else 0} models without one, "
          f"fallback={BOM_WHITELIST_FALLBACK})")
    return out


def read_model_boms(params) -> pl.DataFrame:
    """Model×operation×material requirements from PPS PK1_BOM (o_custom_BOM).

    ReqQty is the per-EA usage (observed 1.5e-06 scale — per-unit fractions of a
    sheet/roll); material_depletion multiplies it by final_production_units, so no
    unit conversion happens here. BOM carries no UOM — joined from PK1_MATERIAL
    (display-only downstream; null-safe). CHASU (revision round) is deduped per
    BOM_CHASU_MODE so one step never counts a material twice."""
    try:
        df = _pull("PK1_BOM", {
            "model_id":          "MODEL_NO",
            "process_id":        "OPERATION_CODE",
            "work_sequence":     "WORK_SEQ",
            "material_id":       "MATERIAL_NO",
            "required_quantity": "REQ_QTY",
            "chasu":             "CHASU",
        })
    except Exception as ex:  # noqa: BLE001
        return _empty("model_boms", _BOM_SCHEMA, f"PK1_BOM pull failed: {ex}")
    if df.height == 0:
        return _empty("model_boms", _BOM_SCHEMA, "o_custom_BOM returned no rows")

    df = df.filter(pl.col("model_id").is_not_null() & pl.col("material_id").is_not_null())
    df = df.with_columns([
        pl.col("work_sequence").cast(pl.Float64, strict=False).cast(pl.Int32),  # 61.0 → 61, matches allocation.sequence dtype
        pl.col("required_quantity").cast(pl.Float64, strict=False),
        pl.col("chasu").cast(pl.Float64, strict=False).fill_null(0.0),
    ])
    # ── CHASU semantics audit (2026-07-30 meeting, 57:11–58:33) ────────────────
    # Two readings of CHASU coexist and the dedup is only correct under (a):
    #   (a) REVISION round — same (model, op, seq, material) re-listed per round;
    #       dedup to the newest is right, else a step counts a material twice.
    #   (b) PRODUCTION round (1차/2차 적층 …) — the same material recurring at
    #       DIFFERENT work_seqs; those rows must all survive (they do — the key
    #       includes work_sequence) because 1차 consumes early and 4차 late.
    # Print both signals so the run log shows which reading the data supports.
    _rev_like = (df.group_by(["model_id", "process_id", "work_sequence", "material_id"])
                   .agg(pl.col("chasu").n_unique().alias("_n"))
                   .filter(pl.col("_n") > 1).height)
    _round_like = (df.unique(subset=["model_id", "material_id", "chasu"])
                     .group_by(["model_id", "material_id"])
                     .agg(pl.col("chasu").n_unique().alias("_n"))
                     .filter(pl.col("_n") > 1).height)
    print(f"   ✓ model_boms CHASU audit: {_rev_like:,} step-material keys carry MULTIPLE "
          f"chasu (revision-like → deduped to newest); {_round_like:,} (model, material) "
          f"pairs span multiple chasu (round-like 차수 usage → all kept, timed by work_seq)")

    if BOM_CHASU_MODE == "latest_per_model":
        df = df.filter(pl.col("chasu") == pl.col("chasu").max().over("model_id"))
    else:  # per_key (default)
        df = df.sort("chasu", descending=True).unique(
            subset=["model_id", "process_id", "work_sequence", "material_id"], keep="first")

    # Keep only materials the MES actually issues (WipDaily.BomMaterialList) — the
    # engineering BOM's obsolete/alternate materials have no on-hand row and would
    # otherwise be read as zero stock and flagged short. See _active_materials().
    df = _apply_active_material_whitelist(df)
    if df.height == 0:
        return _empty("model_boms", _BOM_SCHEMA,
                      "active-material whitelist removed every BOM row — check "
                      "REVPLAN_BOM_ACTIVE_ONLY / BOM_WHITELIST_FALLBACK")

    # UOM enrichment from the material master (guarded — nulls are fine downstream).
    try:
        uom = _pull("PK1_MATERIAL", {"material_id": "MATERIAL_NO", "uom": "UOM"}, distinct=True)
        uom = uom.unique(subset=["material_id"], maintain_order=True)
        df = df.join(uom, on="material_id", how="left")
    except Exception as ex:  # noqa: BLE001
        print(f"   ⚠ model_boms: PK1_MATERIAL UOM join skipped ({ex}) — uom left null")
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("uom"))

    # chasu kept in the output (2026-07-30 meeting): consumption events carry the
    # 차수 so the 자재 화면 can split 소요 per round — 보유 stays a per-material
    # total downstream (총 보유 수량, deducted once — see compute_material_depletion).
    out = df.select(["model_id", "process_id", "work_sequence", "material_id",
                     "required_quantity", "uom", "chasu"])
    print(f"   ✓ model_boms: {out.height} rows, {out['model_id'].n_unique()} models, "
          f"{out['material_id'].n_unique()} materials (CHASU mode={BOM_CHASU_MODE})")
    return out


def read_material_inventories(params) -> pl.DataFrame:
    """Raw-material on-hand from the coarse OnHand snapshot (PK1_ONHAND_TRN / o_custom_OnHand).

    The object mixes RAW-MTL rows (ItemNo = material code — what we want) with FGI
    rows (ItemNo = MODEL_NO — finished goods, already covered by available_inventory),
    filtered via MATERIAL_SUBINV_CODES. Snapshot table → latest BatchDate/BatchHour
    only, then summed per material (the analysis treats it as one global opening
    inventory, exactly like Palantir's material_inventories input)."""
    flt = _latest_batch_filter("PK1_ONHAND_TRN", "material_inventories (OnHand)")
    if flt is None:
        # A full-history pull (decade of multi-daily batches) OOMs the kernel — refuse.
        return _empty("material_inventories", _MATINV_SCHEMA,
                      "MAX(BatchDate) probe failed — refusing the unbounded OnHand pull")
    try:
        df = _pull("PK1_ONHAND_TRN", {
            "material_id":     "ITEM_NO",
            "onhand_quantity": "ONHAND_QTY",
            "_subinv":         "SUBINVENTORY_CODE",
            "batch_date":      "BATCH_DATE",
            "batch_hour":      "BATCH_HOUR",
        }, filters=flt)
    except Exception as ex:  # noqa: BLE001
        return _empty("material_inventories", _MATINV_SCHEMA, f"PK1_ONHAND_TRN pull failed: {ex}")
    df = _latest_batch(df, "material_inventories (OnHand)")
    df = df.filter(pl.col("_subinv").cast(pl.Utf8).str.to_uppercase().is_in(list(MATERIAL_SUBINV_CODES)))
    if df.height == 0:
        return _empty("material_inventories", _MATINV_SCHEMA,
                      f"no OnHand rows in material sub-buckets {sorted(MATERIAL_SUBINV_CODES)}")
    out = (df.with_columns(pl.col("onhand_quantity").cast(pl.Float64, strict=False))
             .group_by("material_id")
             .agg(pl.col("onhand_quantity").sum()))
    print(f"   ✓ material_inventories: {out.height} materials, "
          f"{out['onhand_quantity'].sum():,.0f} total qty {sorted(MATERIAL_SUBINV_CODES)}")
    return out


def read_planned_material_arrivals(params) -> pl.DataFrame:
    """Planned PO arrivals from PK1_PO_ARRIVE_PLAN (o_custom_PoArrivePlan).

    Snapshot table → latest BatchDate/BatchHour batch only. Arrivals dated ON OR
    BEFORE the snapshot date are dropped: those receipts are already inside the
    OnHand snapshot read_material_inventories starts from, and material_depletion
    adds arrivals ON TOP of initial inventory — keeping them would double-count.
    plan_date is cast to Date so the module's `plan_date <= allocated_date`
    comparison matches the engine's pl.Date allocated_date."""
    flt = _latest_batch_filter("PK1_PO_ARRIVE_PLAN", "planned_material_arrivals (PoArrivePlan)")
    if flt is None:
        return _empty("planned_material_arrivals", _ARRIVALS_SCHEMA,
                      "MAX(BatchDate) probe failed — refusing the unbounded PoArrivePlan pull")
    try:
        df = _pull("PK1_PO_ARRIVE_PLAN", {
            "material_id": "PART_NO",
            "plan_date":   "PLAN_DATE",
            "quantity":    "QTY",
            "batch_date":  "BATCH_DATE",
            "batch_hour":  "BATCH_HOUR",
        }, filters=flt)
    except Exception as ex:  # noqa: BLE001
        return _empty("planned_material_arrivals", _ARRIVALS_SCHEMA,
                      f"PK1_PO_ARRIVE_PLAN pull failed: {ex}")
    df = _latest_batch(df, "planned_material_arrivals (PoArrivePlan)")
    if df.height == 0:
        return _empty("planned_material_arrivals", _ARRIVALS_SCHEMA,
                      "o_custom_PoArrivePlan returned no rows")
    snapshot_date = df.select(pl.col("batch_date").cast(pl.Date, strict=False).max()).item()
    out = (df.with_columns([
                pl.col("plan_date").cast(pl.Date, strict=False),
                pl.col("quantity").cast(pl.Float64, strict=False),
           ])
           .filter(pl.col("plan_date").is_not_null() & (pl.col("plan_date") > snapshot_date))
           .select(["material_id", "plan_date", "quantity"]))
    if out.height == 0:
        return _empty("planned_material_arrivals", _ARRIVALS_SCHEMA,
                      f"no arrivals dated after snapshot {snapshot_date} — "
                      "past-dated plans are already inside the OnHand snapshot")
    print(f"   ✓ planned_material_arrivals: {out.height} rows, "
          f"{out['material_id'].n_unique()} materials, "
          f"{out['plan_date'].min()} → {out['plan_date'].max()} (after snapshot {snapshot_date})")
    return out


# ---- ✅ measured_step_durations  <-  o_custom_WipHistory (2026-08-04) ---------
# The per-step WIP event history (실측 착공/완공) → median measured durations per
# (model, op) + per op. Feeds the outsourced pass-through's duration chain as the
# top preference UNDER REVPLAN_MEASURED_LT=1 (default OFF — ingestion is dark
# until validated; see outsourced_allocation.install_measured_lt). Customer
# direction 2026-07-30: actuals over PlanLt standards.
MEASURED_LT_WINDOW_DAYS = int(os.environ.get("REVPLAN_MEASURED_LT_WINDOW_DAYS", "90") or 90)
MEASURED_LT_MIN_SAMPLES = int(os.environ.get("REVPLAN_MEASURED_LT_MIN_SAMPLES", "5") or 5)

_MEASURED_LT_SCHEMA = {
    "model_id": pl.Utf8,               # null on op-level fallback rows
    "process_id": pl.Utf8,             # site prefix STRIPPED (PK1MF18N -> MF18N)
    "measured_step_hours": pl.Float64, # median 착공→다음 스텝 착공 (run + queue) — the pass-through duration
    "measured_proc_hours": pl.Float64, # median 착공→완공 (run only) — reference/diagnostics
    "sample_n": pl.Int64,
}


def _aggregate_measured_durations(df: pl.DataFrame) -> pl.DataFrame:
    """Pure transform (unit-testable offline): raw WipHistory events -> duration medians.

    * PROCID arrives site-prefixed (PK1MF18N); strip each row's own site code.
    * step duration = this row's 착공(ST) -> the LOT's next row's 착공 (run+queue,
      i.e. "how long until the lot moves on" — what the pass-through must model).
      A lot's LAST row falls back to its own ST->ED processing time.
    * MEDIAN per key (robust: a lot that slept 3 months on hold must not drag the
      mean), kept only at sample_n >= MEASURED_LT_MIN_SAMPLES.
    * op-level fallback rows carry model_id = null.
    """
    df = df.with_columns([
        pl.col("_st").cast(pl.Datetime, strict=False),
        pl.col("_ed").cast(pl.Datetime, strict=False),
    ]).filter(
        pl.col("_st").is_not_null() & pl.col("model_id").is_not_null()
        & pl.col("_procid_raw").is_not_null()
    )
    if df.height == 0:
        return pl.DataFrame(schema=_MEASURED_LT_SCHEMA)

    # strip each row's own site prefix (few distinct sites -> literal loop is cheap)
    for s in df["_site"].drop_nulls().unique().to_list():
        df = df.with_columns(
            pl.when((pl.col("_site") == s) & pl.col("_procid_raw").str.starts_with(s))
              .then(pl.col("_procid_raw").str.slice(len(s)))
              .otherwise(pl.col("_procid_raw"))
              .alias("_procid_raw"))
    df = df.rename({"_procid_raw": "process_id"})

    df = df.sort(["lot_id", "_st"]).with_columns(
        pl.col("_st").shift(-1).over("lot_id").alias("_next_st"))
    df = df.with_columns([
        ((pl.col("_next_st") - pl.col("_st")).dt.total_seconds() / 3600.0).alias("step_h"),
        ((pl.col("_ed") - pl.col("_st")).dt.total_seconds() / 3600.0).alias("proc_h"),
    ]).with_columns(
        pl.coalesce([pl.col("step_h"), pl.col("proc_h")]).alias("step_h")
    ).filter(pl.col("step_h") > 0)

    def _agg(frame: pl.DataFrame, keys: List[str]) -> pl.DataFrame:
        return (frame.group_by(keys)
                .agg([pl.col("step_h").median().alias("measured_step_hours"),
                      pl.col("proc_h").median().alias("measured_proc_hours"),
                      pl.len().cast(pl.Int64).alias("sample_n")])
                .filter(pl.col("sample_n") >= MEASURED_LT_MIN_SAMPLES))

    per_mo = _agg(df, ["model_id", "process_id"])
    per_op = _agg(df, ["process_id"]).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("model_id"))
    cols = list(_MEASURED_LT_SCHEMA.keys())
    return pl.concat([per_mo.select(cols), per_op.select(cols)], how="vertical")


# One WipHistory pull serves TWO consumers (durations + the SimWIPMaster lot
# summary) — cached per process so read_inputs hits the backend once.
_WIPHIST_EVENTS: Optional[pl.DataFrame] = None
_WIPHIST_ERR: Optional[str] = None


def _wip_history_events() -> Tuple[Optional[pl.DataFrame], Optional[str]]:
    global _WIPHIST_EVENTS, _WIPHIST_ERR
    if _WIPHIST_EVENTS is not None or _WIPHIST_ERR is not None:
        return _WIPHIST_EVENTS, _WIPHIST_ERR
    from datetime import date as _d, timedelta as _td
    cutoff = int((_d.today() - _td(days=MEASURED_LT_WINDOW_DAYS)).strftime("%Y%m%d"))
    _cols = {
        "lot_id":      "LOTID",
        "model_id":    "PRODID",
        "_procid_raw": "PROCID",
        "_site":       "PLANNING_SITE_CODE",
        "_st":         "WIPDTTM_ST",
        "_ed":         "WIPDTTM_ED",
        # SimWIPMaster lot-summary columns (2026-08-05):
        "_eqptid":     "EQPTID",
        "_category":   "APS_PRODCATEGORY",
        "_sheet_qty":  "WIPSHEETQTY_ED",
        "_pnl_qty":    "WIPPNLQTY_ED",
        "_unit_qty":   "WIPQTY_ED",
    }
    _site_flt = []
    if WIP_SITES:
        sites = ", ".join(f"'{s}'" for s in sorted(WIP_SITES))
        _site_flt = [f"FILTER {_pql_col('RTSP_MGR_WIP_HISTORY_N', 'PLANNING_SITE_CODE')} IN ({sites})"]
    _ymd = _pql_col("RTSP_MGR_WIP_HISTORY_N", "YYYYMMDD")
    # YYYYMMDD's storage type in the object is unknown (INT vs STRING) — try the
    # numeric comparison first, retry quoted; the window bound is NOT optional
    # (the raw history reaches back to 2019), so both failing = empty + warning.
    df, _last = None, None
    for _f in (f"FILTER {_ymd} >= {cutoff}", f"FILTER {_ymd} >= '{cutoff}'"):
        try:
            df = _pull("RTSP_MGR_WIP_HISTORY_N", _cols, filters=[_f] + _site_flt)
            break
        except Exception as ex:  # noqa: BLE001 — a calibration input must never sink read_inputs
            _last = ex
    if df is None:
        _WIPHIST_ERR = str(_last)
        return None, _WIPHIST_ERR
    _WIPHIST_EVENTS = df
    return df, None


# ⚠️ MLWB CHANGE (2026-08-19, business decision): o_custom_WIP (hourly snapshot of
# RTSP_WIP_N) REPLACES o_custom_WipHistory as the actuals source, and only the
# NEWEST YYYYMMDDHH hour is read — filtered in the BACKEND (server-side PQL), never
# in the frontend. Kill switch: REVPLAN_ACTUALS_SOURCE=wiphistory restores the
# legacy readers unchanged.
# Known, accepted costs of the swap (assessed vs A/B runs: ~2-3% outcome shift):
#   * measured LT shrinks to ONE completed-step duration per lot (PrevActual*),
#     attributed to the lot's previous ROUTING step — coverage drops vs the 90d
#     event history, most outsourced steps fall back to the PlanLt chain.
#   * actual_first_start comes from GLotCreateDttm (nullable), actual_steps_done
#     becomes the lot's current SEQ (a position, not an event count).
def _actuals_source() -> str:
    return os.environ.get("REVPLAN_ACTUALS_SOURCE", "wip").strip().lower()


_WIP_SNAPSHOT: Optional[pl.DataFrame] = None
_WIP_SNAPSHOT_ERR: Optional[str] = None


def _wip_scalar(expr: str, filters: Optional[List[str]] = None):
    """One-value PQL probe against the data model (MAX/COUNT style)."""
    from pycelonis.pql import PQL, PQLColumn
    import pycelonis.pql as pql
    q = PQL()
    for f in (filters or []):
        q += f
    q += PQLColumn(name="v", query=expr)
    try:
        pdf = pql.DataFrame.from_pql(q, data_model=data_model()).to_pandas()
    except AttributeError:
        pdf = data_model().export_data_frame(q)
    return pdf["v"].iloc[0] if len(pdf) else None


def _wip_hourly_snapshot() -> Tuple[Optional[pl.DataFrame], Optional[str]]:
    """Newest-hour rows of o_custom_WIP, one row per lot — cached per process.

    * MAX(YYYYMMDDHH) probe + server-side equality filter: the raw table holds
      24 snapshots/day of the whole fleet, an unbounded pull is an OOM (the
      daily-grain WipDaily already proved that at 1/24th of the volume).
    * HALF-WRITTEN-HOUR GUARD: an hourly extraction can be mid-load when we
      read. If the newest hour holds < 50% of the previous hour's lots, fall
      back one hour with a loud warning instead of simulating a shrunken fleet.
    * Site scope follows REVPLAN_WIP_SITES (default PK1), like every WIP read.
    """
    global _WIP_SNAPSHOT, _WIP_SNAPSHOT_ERR
    if _WIP_SNAPSHOT is not None or _WIP_SNAPSHOT_ERR is not None:
        return _WIP_SNAPSHOT, _WIP_SNAPSHOT_ERR
    _T = "RTSP_WIP_HOURLY"
    _hcol = _pql_col(_T, "YYYYMMDDHH")
    _lcol = _pql_col(_T, "LOTID")
    _site_flt = []
    if WIP_SITES:
        sites = ", ".join(f"'{s}'" for s in sorted(WIP_SITES))
        _site_flt = [f"FILTER {_pql_col(_T, 'PLANNING_SITE_CODE')} IN ({sites})"]

    def _eq(v):  # YYYYMMDDHH storage type unknown (INT vs STRING) — build both forms
        return [f"FILTER {_hcol} = {v}", f"FILTER {_hcol} = '{v}'"]

    try:
        max_h = _wip_scalar(f"MAX({_hcol})")
    except Exception as ex:  # noqa: BLE001
        _WIP_SNAPSHOT_ERR = f"MAX(YYYYMMDDHH) probe failed ({ex})"
        return None, _WIP_SNAPSHOT_ERR
    if max_h is None:
        _WIP_SNAPSHOT_ERR = "o_custom_WIP is empty (MAX(YYYYMMDDHH) returned nothing)"
        return None, _WIP_SNAPSHOT_ERR
    hour = str(max_h).split(".")[0].strip()

    # guard probes (best-effort — a probe failure only skips the guard, not the read)
    n_latest = n_prev = prev_h = None
    for _f in _eq(hour):
        try:
            n_latest = _wip_scalar(f"COUNT(DISTINCT {_lcol})", _site_flt + [_f])
            break
        except Exception:  # noqa: BLE001
            continue
    for _f in (f"FILTER {_hcol} < {hour}", f"FILTER {_hcol} < '{hour}'"):
        try:
            prev_h = _wip_scalar(f"MAX({_hcol})", [_f])
            break
        except Exception:  # noqa: BLE001
            continue
    if prev_h is not None:
        prev_h = str(prev_h).split(".")[0].strip()
        for _f in _eq(prev_h):
            try:
                n_prev = _wip_scalar(f"COUNT(DISTINCT {_lcol})", _site_flt + [_f])
                break
            except Exception:  # noqa: BLE001
                continue
    if (n_latest is not None and n_prev is not None and float(n_prev) > 0
            and float(n_latest) < 0.5 * float(n_prev)):
        print(f"   ⚠ o_custom_WIP: newest hour {hour} holds only {int(n_latest):,} lots vs "
              f"{int(n_prev):,} in hour {prev_h} — snapshot looks HALF-WRITTEN, "
              f"falling back to hour {prev_h}")
        hour = prev_h

    _cols = {
        "lot_id":       "LOTID",
        "model_id":     "PRODID",
        "op_code":      "OP_CODE",
        "seq":          "SEQ",
        "_site":        "PLANNING_SITE_CODE",
        "_eqptid":      "EQPTID",
        "_category":    "APS_PRODCATEGORY",
        "_cur_st":      "WIPDTTM_ST",
        "_prev_eqptid": "PREV_ACTUAL_EQPTID",
        "_prev_st":     "PREV_ACTUAL_START_DATE",
        "_prev_ed":     "PREV_ACTUAL_END_DATE",
        "_sheet_qty":   "WIPSHTQTY",
        "_pnl_qty":     "WIPPNLQTY",
        "_unit_qty":    "WIPUNITQTY",
        "_lot_create":  "G_LOT_CREATE_DTTM",
    }
    df, _last = None, None
    for _f in _eq(hour):
        try:
            df = _pull(_T, _cols, filters=[_f] + _site_flt)
            break
        except Exception as ex:  # noqa: BLE001
            _last = ex
    if df is None:
        _WIP_SNAPSHOT_ERR = f"newest-hour pull failed ({_last})"
        return None, _WIP_SNAPSHOT_ERR
    pre = df.height
    df = (df.filter(pl.col("lot_id").is_not_null())
            .sort("seq", descending=True, nulls_last=True)
            .unique(subset=["lot_id"], keep="first"))
    print(f"   ✓ o_custom_WIP snapshot: hour {hour} — {df.height:,} lots "
          f"({pre:,} rows; site {sorted(WIP_SITES) if WIP_SITES else 'all'})"
          + (f"; guard OK ({int(n_latest):,} vs prev {int(n_prev):,})"
             if n_latest is not None and n_prev is not None else "; guard probes unavailable"))
    _WIP_SNAPSHOT = df
    return df, None


def _aggregate_wip_prev_durations(df: pl.DataFrame,
                                  routing: Optional[pl.DataFrame]) -> pl.DataFrame:
    """Pure transform: newest-hour WIP rows -> duration medians (_MEASURED_LT_SCHEMA).

    The snapshot's only complete history is the lot's PREVIOUS step
    (PrevActualStartDate/EndDate) — but that step's OPERATION is not on the row,
    so it is looked up from the routing: the step with the greatest WorkSeq
    strictly below the lot's current SEQ. Durations keep the measured-LT
    semantic: 착공→다음 착공 (prev start -> current start = run + queue);
    prev start -> prev end is the run-only reference.
    """
    if routing is None or getattr(routing, "height", 0) == 0:
        return pl.DataFrame(schema=_MEASURED_LT_SCHEMA)
    need = {"model_id", "sequence", "process_id"}
    if not need.issubset(set(routing.columns)):
        return pl.DataFrame(schema=_MEASURED_LT_SCHEMA)

    df = df.with_columns([
        pl.col("_prev_st").cast(pl.Datetime, strict=False),
        pl.col("_prev_ed").cast(pl.Datetime, strict=False),
        pl.col("_cur_st").cast(pl.Datetime, strict=False),
        pl.col("seq").cast(pl.Int64, strict=False),
    ]).filter(pl.col("_prev_st").is_not_null() & pl.col("model_id").is_not_null()
              & pl.col("seq").is_not_null())
    if df.height == 0:
        return pl.DataFrame(schema=_MEASURED_LT_SCHEMA)

    rt = (routing.select([
              pl.col("model_id"),
              pl.col("sequence").cast(pl.Int64, strict=False),
              pl.col("process_id").alias("_prev_op")])
          .filter(pl.col("sequence").is_not_null() & pl.col("_prev_op").is_not_null())
          .unique(subset=["model_id", "sequence"], keep="first")
          .sort(["model_id", "sequence"]))
    # strictly-previous routing step: asof-backward on (current SEQ - 1)
    df = (df.with_columns((pl.col("seq") - 1).alias("_k"))
            .sort(["model_id", "_k"])
            .join_asof(rt, left_on="_k", right_on="sequence",
                       by="model_id", strategy="backward")
            .filter(pl.col("_prev_op").is_not_null()))
    if df.height == 0:
        return pl.DataFrame(schema=_MEASURED_LT_SCHEMA)

    df = df.rename({"_prev_op": "process_id"}).with_columns([
        ((pl.coalesce([pl.col("_cur_st"), pl.col("_prev_ed")]) - pl.col("_prev_st"))
         .dt.total_seconds() / 3600.0).alias("step_h"),
        ((pl.col("_prev_ed") - pl.col("_prev_st"))
         .dt.total_seconds() / 3600.0).alias("proc_h"),
    ]).filter(pl.col("step_h") > 0)

    def _agg(frame: pl.DataFrame, keys: List[str]) -> pl.DataFrame:
        return (frame.group_by(keys)
                .agg([pl.col("step_h").median().alias("measured_step_hours"),
                      pl.col("proc_h").median().alias("measured_proc_hours"),
                      pl.len().cast(pl.Int64).alias("sample_n")])
                .filter(pl.col("sample_n") >= MEASURED_LT_MIN_SAMPLES))

    per_mo = _agg(df, ["model_id", "process_id"])
    per_op = _agg(df, ["process_id"]).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("model_id"))
    cols = list(_MEASURED_LT_SCHEMA.keys())
    return pl.concat([per_mo.select(cols), per_op.select(cols)], how="vertical")


def read_measured_step_durations(params, routing: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    if _actuals_source() != "wiphistory":
        snap, _err = _wip_hourly_snapshot()
        if snap is None:
            return _empty("measured_step_durations", _MEASURED_LT_SCHEMA,
                          f"o_custom_WIP pull failed ({_err}) — outsourced timing stays on PlanLt")
        if routing is None or getattr(routing, "height", 0) == 0:
            return _empty("measured_step_durations", _MEASURED_LT_SCHEMA,
                          "no routing frame for prev-step op attribution — "
                          "outsourced timing stays on PlanLt")
        out = _aggregate_wip_prev_durations(
            snap.select(["lot_id", "model_id", "seq", "_prev_st", "_prev_ed", "_cur_st"]),
            routing)
        n_mo = int(out.filter(pl.col("model_id").is_not_null()).height)
        n_op = out.height - n_mo
        print(f"   ✓ measured_step_durations (o_custom_WIP prev-step): {snap.height:,} lots → "
              f"{n_mo:,} (model,op) + {n_op:,} op-level medians (min n={MEASURED_LT_MIN_SAMPLES}) — "
              "coverage is ONE completed step per lot; uncovered ops use the PlanLt chain")
        return out

    df, _err = _wip_history_events()
    if df is None:
        return _empty("measured_step_durations", _MEASURED_LT_SCHEMA,
                      f"o_custom_WipHistory pull failed ({_err}) — outsourced timing stays on PlanLt")
    df = df.select(["lot_id", "model_id", "_procid_raw", "_site", "_st", "_ed"])
    if df.height == 0:
        return _empty("measured_step_durations", _MEASURED_LT_SCHEMA,
                      f"o_custom_WipHistory has no events in the last {MEASURED_LT_WINDOW_DAYS}d — "
                      "outsourced timing stays on PlanLt")
    out = _aggregate_measured_durations(df)
    n_mo = int(out.filter(pl.col("model_id").is_not_null()).height)
    n_op = out.height - n_mo
    print(f"   ✓ measured_step_durations (LEGACY WipHistory): {df.height:,} events (last {MEASURED_LT_WINDOW_DAYS}d) → "
          f"{n_mo:,} (model,op) + {n_op:,} op-level medians (min n={MEASURED_LT_MIN_SAMPLES})")
    return out


# ---- ✅ wip_history_lot_summary  (SimWIPMaster's ACTUAL side, 2026-08-05) ------
# One row per lot from the WipHistory event log: verified past to annotate the
# simulated future. Join-safety proven in the backend (2026-08-05 SQL check):
# GROUP BY lot -> 1:1, row conservation 794,442 = 794,442; 87.5% of real sim
# lots matched (the rest had no movement inside the window — itself a signal).
_WIP_LOT_SUMMARY_SCHEMA = {
    "lot_id": pl.Utf8,
    "actual_first_start": pl.Datetime,
    "actual_last_event": pl.Datetime,
    "actual_steps_done": pl.Int64,
    "actual_last_process": pl.Utf8,    # site prefix stripped (PK1ML20N -> ML20N)
    "actual_last_equipment": pl.Utf8,
    "actual_sheet_qty": pl.Float64,
    "actual_pnl_qty": pl.Float64,
    "actual_unit_qty": pl.Float64,
    "aps_prodcategory": pl.Utf8,       # 양산/시재 classification
}


def _aggregate_wip_lot_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Pure transform (unit-testable offline): raw WipHistory events -> 1 row/lot."""
    df = df.with_columns(pl.col("_st").cast(pl.Datetime, strict=False)).filter(
        pl.col("_st").is_not_null() & pl.col("lot_id").is_not_null())
    if df.height == 0:
        return pl.DataFrame(schema=_WIP_LOT_SUMMARY_SCHEMA)
    for s in df["_site"].drop_nulls().unique().to_list():
        df = df.with_columns(
            pl.when((pl.col("_site") == s) & pl.col("_procid_raw").str.starts_with(s))
              .then(pl.col("_procid_raw").str.slice(len(s)))
              .otherwise(pl.col("_procid_raw"))
              .alias("_procid_raw"))
    df = df.sort(["lot_id", "_st"])
    out = df.group_by("lot_id").agg([
        pl.col("_st").min().alias("actual_first_start"),
        pl.col("_st").max().alias("actual_last_event"),
        pl.len().cast(pl.Int64).alias("actual_steps_done"),
        pl.col("_procid_raw").last().alias("actual_last_process"),
        pl.col("_eqptid").last().alias("actual_last_equipment"),
        pl.col("_sheet_qty").cast(pl.Float64, strict=False).last().alias("actual_sheet_qty"),
        pl.col("_pnl_qty").cast(pl.Float64, strict=False).last().alias("actual_pnl_qty"),
        pl.col("_unit_qty").cast(pl.Float64, strict=False).last().alias("actual_unit_qty"),
        pl.col("_category").drop_nulls().first().alias("aps_prodcategory"),
    ])
    return out.select(list(_WIP_LOT_SUMMARY_SCHEMA.keys()))


# ---- ✅ material_classes  <-  o_custom_Material (2026-08-05) -------------------
# CopClass per material for SimStockMaster (frontend spec: Date / MaterialNo /
# 보유·사용·보충량 / CopClass). Column-name resilient: the object's alias is
# expected as 'CopClass' (mapped), but a raw 'COP_CLASS' build is retried too.
_MATERIAL_CLASS_SCHEMA = {"material_id": pl.Utf8, "cop_class": pl.Utf8}


def read_material_classes(params) -> pl.DataFrame:
    _last = None
    for colmap in ({"material_id": "MATERIAL_NO", "cop_class": "COP_CLASS"},   # mapped -> "CopClass"
                   {"material_id": "MaterialNo",  "cop_class": "COP_CLASS"},   # raw alias variants
                   {"material_id": "MaterialNo",  "cop_class": "CopClass"}):
        try:
            df = _pull("PK1_MATERIAL", colmap, distinct=True)
            df = (df.filter(pl.col("material_id").is_not_null())
                    .unique(subset=["material_id"], maintain_order=True))
            n_cls = int(df.select(pl.col("cop_class").is_not_null().sum()).item()) if df.height else 0
            print(f"   ✓ material_classes: {df.height:,} materials, {n_cls:,} with CopClass")
            return df
        except Exception as ex:  # noqa: BLE001
            _last = ex
    return _empty("material_classes", _MATERIAL_CLASS_SCHEMA,
                  f"CopClass not readable from o_custom_Material ({_last}) — "
                  "SimStockMaster.cop_class will be null")


def _aggregate_wip_snapshot_lot_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Pure transform: newest-hour WIP rows (1/lot) -> _WIP_LOT_SUMMARY_SCHEMA.

    Semantics vs the WipHistory original (announced degradations, 2026-08-19):
      * actual_first_start  <- GLotCreateDttm (nullable in the source)
      * actual_last_event   <- current step 착공 (WipdttmSt), else prev step end
      * actual_steps_done   <- current SEQ (a POSITION, not an event count)
      * actual_last_process <- OpCode (already un-prefixed — no site strip needed)
    """
    if df.height == 0:
        return pl.DataFrame(schema=_WIP_LOT_SUMMARY_SCHEMA)
    out = df.select([
        pl.col("lot_id"),
        pl.col("_lot_create").cast(pl.Datetime, strict=False).alias("actual_first_start"),
        pl.coalesce([pl.col("_cur_st").cast(pl.Datetime, strict=False),
                     pl.col("_prev_ed").cast(pl.Datetime, strict=False)]).alias("actual_last_event"),
        pl.col("seq").cast(pl.Int64, strict=False).alias("actual_steps_done"),
        pl.col("op_code").cast(pl.Utf8).alias("actual_last_process"),
        pl.coalesce([pl.col("_eqptid").cast(pl.Utf8),
                     pl.col("_prev_eqptid").cast(pl.Utf8)]).alias("actual_last_equipment"),
        pl.col("_sheet_qty").cast(pl.Float64, strict=False).alias("actual_sheet_qty"),
        pl.col("_pnl_qty").cast(pl.Float64, strict=False).alias("actual_pnl_qty"),
        pl.col("_unit_qty").cast(pl.Float64, strict=False).alias("actual_unit_qty"),
        pl.col("_category").cast(pl.Utf8).alias("aps_prodcategory"),
    ])
    return out.select(list(_WIP_LOT_SUMMARY_SCHEMA.keys()))


# ---- ✅ sales_order_lines  <-  o_custom_SalesOrderLine (2026-08-26) ------------
# Feeds the uncommitted_demand.py port: open (committed) quantity per model =
# Σ(OrderQuantity − ShippedQuantity) over lines with shipped < order. The module
# consumes columns named item_no / order_quantity / shipped_quantity (Palantir's
# dataset names) — mapped here so the port stays verbatim.
_SO_LINE_SCHEMA = {"item_no": pl.Utf8, "order_quantity": pl.Float64, "shipped_quantity": pl.Float64}


def read_sales_order_lines(params) -> pl.DataFrame:
    try:
        df = _pull("PK1_SO_LINE", {
            "item_no":          "MODEL_ID",
            "order_quantity":   "ORDER_QUANTITY",
            "shipped_quantity": "SHIPPED_QUANTITY",
            "_status":          "LINE_STATUS",
        })
    except Exception as ex:  # noqa: BLE001 — a risk-analysis input must never sink read_inputs
        return _empty("sales_order_lines", _SO_LINE_SCHEMA,
                      f"o_custom_SalesOrderLine pull failed ({ex}) — "
                      "uncommitted-demand tables will be empty this run")
    pre = df.height
    # ⚠️ MLWB DIVERGENCE (2026-08-26, deliberate): drop CANCELLED lines. A cancelled
    # order was typically never shipped, so the source's shipped<order rule would
    # count it as committed demand — inflating the committed pool with dead orders
    # (296 of ~13.4k lines at wiring time). Everything else passes through; the
    # shipped<order filter itself stays inside the ported module (parity).
    df = (df.filter(pl.col("item_no").is_not_null())
            .filter(pl.col("_status").cast(pl.Utf8).str.strip_chars()
                    .str.to_uppercase().fill_null("") != "CANCELLED")
            .drop("_status")
            .with_columns([
                pl.col("order_quantity").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("shipped_quantity").cast(pl.Float64, strict=False).fill_null(0.0),
            ]))
    n_open = int(df.filter(pl.col("shipped_quantity") < pl.col("order_quantity")).height)
    print(f"   ✓ sales_order_lines: {df.height:,} lines ({pre - df.height:,} cancelled excluded) — "
          f"{n_open:,} OPEN (shipped < order) across "
          f"{df.filter(pl.col('shipped_quantity') < pl.col('order_quantity'))['item_no'].n_unique():,} models")
    return df


def read_wip_history_lot_summary(params) -> pl.DataFrame:
    if _actuals_source() != "wiphistory":
        snap, _err = _wip_hourly_snapshot()
        if snap is None:
            return _empty("wip_history_lot_summary", _WIP_LOT_SUMMARY_SCHEMA,
                          f"o_custom_WIP pull failed ({_err}) — SimWIPMaster gets null actual columns")
        try:
            out = _aggregate_wip_snapshot_lot_summary(snap)
        except Exception as ex:  # noqa: BLE001 — an annotation input must never sink read_inputs
            return _empty("wip_history_lot_summary", _WIP_LOT_SUMMARY_SCHEMA,
                          f"snapshot lot-summary build failed ({ex}) — SimWIPMaster gets null actual columns")
        _n_fs = int(out.select(pl.col("actual_first_start").is_not_null().sum()).item()) if out.height else 0
        print(f"   ✓ wip_lot_summary (o_custom_WIP newest hour): {out.height:,} lots — "
              f"actual_first_start filled for {_n_fs:,} "
              "(GLotCreateDttm; actual_steps_done = current SEQ position)")
        return out

    df, _err = _wip_history_events()
    if df is None:
        return _empty("wip_history_lot_summary", _WIP_LOT_SUMMARY_SCHEMA,
                      f"o_custom_WipHistory pull failed ({_err}) — SimWIPMaster gets null actual columns")
    try:
        out = _aggregate_wip_lot_summary(
            df.select(["lot_id", "_procid_raw", "_site", "_st", "_eqptid", "_category",
                       "_sheet_qty", "_pnl_qty", "_unit_qty"]))
    except Exception as ex:  # noqa: BLE001 — an annotation input must never sink read_inputs
        return _empty("wip_history_lot_summary", _WIP_LOT_SUMMARY_SCHEMA,
                      f"lot-summary aggregation failed ({ex}) — SimWIPMaster gets null actual columns")
    print(f"   ✓ wip_history_lot_summary: {out.height:,} lots with actual history "
          f"(last {MEASURED_LT_WINDOW_DAYS}d)")
    return out


# ==============================================================================
# 6.  read_inputs  — assemble the dict the engine consumes.
# ==============================================================================
def read_inputs(params) -> Dict[str, pl.DataFrame]:
    print("=== Celonis read_inputs ===")
    # routing is read once and reused: the WIP-snapshot duration builder needs it
    # to attribute each lot's PrevActual* duration to its previous ROUTING step.
    planned_steps = read_planned_process_steps(params)               # ✅ (🔴 crosswalk)
    return {
        "revenue_plan":          read_revenue_plan(params),           # ✅
        "available_inventory":   read_available_inventory(params),    # ✅ OnHand FGI on-hand (shipped/transit=0)
        "model_priorities":      read_model_priorities(params),       # ❌ placeholder
        "wip_lots":              read_wip_lots(params),               # ✅ (🔴 crosswalk)
        "planned_process_steps": planned_steps,
        "model_master":          read_model_master(params),           # ✅
        "model_unit_conversion": read_model_unit_conversion(params),  # ✅
        "equipment_capacity":    read_equipment_capacity(params),     # ✅ + EquipmentGroup (group/infinite)
        "equipment_constraints": read_equipment_constraints(params),  # ❌ placeholder (no NEGATIVE source)
        "equipment_to_process":  read_equipment_to_process(params),   # ✅ o_custom_EquipmentConstraints (positive map)
        "et_jig_master":         read_et_jig_master(params),          # ✅ 2026-07-10 — o_custom_JIGMaster (feeds compute_et_jig_risk)
        # material trio (2026-07-16) — consumed by compute_material_depletion:
        "model_boms":                read_model_boms(params),                # ✅ o_custom_BOM (+ UOM from o_custom_Material)
        "material_inventories":      read_material_inventories(params),      # ✅ o_custom_OnHand RAW-MTL rows, latest batch
        "planned_material_arrivals": read_planned_material_arrivals(params), # ✅ o_custom_PoArrivePlan, latest batch, future-dated only
        "measured_step_durations":   read_measured_step_durations(params, routing=planned_steps),  # ✅ 2026-08-19 — o_custom_WIP prev-step medians (legacy: REVPLAN_ACTUALS_SOURCE=wiphistory)
        "wip_history_lot_summary":   read_wip_history_lot_summary(params),   # ✅ 2026-08-19 — o_custom_WIP newest-hour lot actuals (same cached pull)
        "material_classes":          read_material_classes(params),          # ✅ 2026-08-05 — CopClass per material (SimStockMaster)
        "sales_order_lines":         read_sales_order_lines(params),         # ✅ 2026-08-26 — o_custom_SalesOrderLine (uncommitted demand)
        "grouping_model_map":        read_grouping_model_map(params),        # ✅ 2026-09-09 — MovePlan.GroupingModel (stage-1 analysis axis, outputs only)
    }
    # NOTE — remaining stub-analysis inputs: production_risk_reconciliation needs no
    # new sources (it reads other analyses' outputs) — port it alongside its stub.


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
#
# TARGETED SCHEMA MIGRATION (2026-07-30): when a release adds a column to ONE
# table, nuking every SIM_* table is disproportionate and clicking the Data Pool
# UI is manual toil. Tables named in REVPLAN_SCHEMA_MIGRATE (comma list, logical
# name or full SIM_ name; '0'/'none' disables) are allowed to fall through to
# drop+recreate WHEN — and only when — their append fails. Scoped so only the
# named tables can ever lose history, and only on an actual append failure.
#   Name tables here only while a release actually migrates them (the 차수/chasu
#   column shipped this way on 2026-07-30) — a permanent allowlist would let a
#   transient push error silently reset a table's history.
#   Past one-shots (all landed): 'allocation' (2026-08-04 modified_group column),
#   'StockMaster' (2026-08-07 opening_qty -> initial_onhand_qty rename; recreate
#   confirmed in the 2026-08-10 run log).
#   ACTIVE one-shot (2026-09-07): 'allocation,WIPMaster' — moveplan_yymm column
#   added for the 이동계획-기준 매출 axis. SIM_allocation + SIM_WIPMaster are
#   recreated on their first append failure; CLEAR the default back to "" once a
#   run log confirms both tables recreated with the new column.
#   'offplan_inventory' (2026-09-09): grouping_model column added before most
#   deployments ever created the table — covers the edge where a run already
#   created it without the column.
_SCHEMA_MIGRATE_RAW = os.environ.get("REVPLAN_SCHEMA_MIGRATE",
                                     "allocation,WIPMaster,offplan_inventory")
SCHEMA_MIGRATE_TABLES = frozenset(
    t.strip() for t in _SCHEMA_MIGRATE_RAW.split(",")
    if t.strip() and t.strip().lower() not in ("0", "none", "off")
)


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
        # The data-push Parquet validator rejects NESTED types, so polars List
        # columns 400 every attempt (2026-08-10 root cause: SIM_et_jig_capacity_risk
        # / SIM_demand_shortfall carry *_ids list columns and had failed on every
        # run — deterministic, not transient). Flatten to comma-joined strings at
        # this boundary so every current and future result table is covered (the
        # reconciliation/material wrappers already flatten; this is the safety net).
        _list_cols = [c for c, dt in df.schema.items() if isinstance(dt, pl.List)]
        if _list_cols:
            df = df.with_columns([
                pl.col(c).cast(pl.List(pl.Utf8)).list.join(",").alias(c) for c in _list_cols])
            print(f"   ✓ {table_name}: flattened list column(s) {_list_cols} to "
                  "comma-joined strings (Data Pool push rejects nested Parquet types)")
        pdf = df.to_pandas()

        # ---- append path: table exists and no reset requested -> add this run's rows
        # Migration-listed tables skip the append ENTIRELY and recreate: waiting
        # for an append failure proved unreliable (2026-08-05 — the platform
        # silently ADDED the new columns and appended, leaving SIM_StockMaster a
        # union of old ledger + new pivot schemas instead of failing).
        _migrate_this = name in SCHEMA_MIGRATE_TABLES or table_name in SCHEMA_MIGRATE_TABLES
        if _migrate_this and not reset and _find_table(table_name) is not None:
            print(f"   ⚠ {table_name} is in REVPLAN_SCHEMA_MIGRATE — recreating with the "
                  "current schema (THIS table's previous runs are discarded; all others keep history)")
        existing = None if (reset or _migrate_this) else _find_table(table_name)
        if existing is not None:
            _append_err = None
            for _try in (1, 2):  # 2026-08-07: platform 400 'Parquet validation'
                try:              # errors self-describe as transient — retry once
                    existing.append(pdf)
                    print(f"   ✓ appended {table_name}  (+{df.height} rows)"
                          + ("  (retry succeeded)" if _try == 2 else ""))
                    _append_err = None
                    break
                except Exception as _ex:  # noqa: BLE001
                    _append_err = _ex
                    if _try == 1:
                        import time as _t; _t.sleep(5)
            if _append_err is None:
                continue
            ex = _append_err
            if True:  # preserved structure of the original error handling below
                if name in SCHEMA_MIGRATE_TABLES or table_name in SCHEMA_MIGRATE_TABLES:
                    # Explicitly allowlisted migration: this release changed the
                    # table's schema, so the failed append is expected — recreate
                    # WITH the new schema (drop_if_exists on the create path) and
                    # accept losing this one table's history.
                    print(f"   ⚠ append to '{table_name}' failed ({ex}) — table is in "
                          "REVPLAN_SCHEMA_MIGRATE, recreating it with the new schema "
                          "(THIS table's previous runs are discarded; all others keep history)")
                else:
                    # Do NOT fall back to drop/recreate — that would silently erase the run
                    # history this function exists to keep. Typical cause is schema drift
                    # (new/renamed column, string longer than the column width chosen at
                    # creation); resolve it deliberately.
                    print(f"   ✗ append to '{table_name}' failed: {ex}\n"
                          "      → table NOT dropped (run history preserved). If the schema "
                          "changed intentionally, re-run with CELONIS_OUTPUT_RESET=1 to "
                          "recreate the SIM_* tables (discards previous runs), or name the "
                          "table in REVPLAN_SCHEMA_MIGRATE for a targeted recreate.")
                    continue

        # ---- create path: first ever run, or explicit CELONIS_OUTPUT_RESET=1
        cfg = _string_column_config(df)
        # column_config: preserve real string lengths (with append headroom). Retry
        # without it if this pycelonis build rejects its shape — still lands the data
        # (falling back to the VARCHAR(80) default).
        attempts = ([{"drop_if_exists": True, "force": True, "column_config": cfg}] if cfg else [])
        attempts.append({"drop_if_exists": True, "force": True})
        ok, last = False, None
        for _round in (1, 2):  # 2026-08-07: retry the whole attempt list once (transient 400s)
            for kw in attempts:
                try:
                    p.create_table(pdf, table_name, **kw)
                    note = "" if "column_config" in kw else "  (⚠ default VARCHAR(80))"
                    print(f"   ✓ created {table_name}  ({df.height} rows){note}"
                          + ("  (retry succeeded)" if _round == 2 else ""))
                    ok = True
                    break
                except Exception as ex:  # noqa: BLE001
                    last = ex
            if ok:
                break
            if _round == 1:
                import time as _t; _t.sleep(5)
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
