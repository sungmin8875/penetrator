"""
priority_generation.py — model-level priority table for the allocation engine.
================================================================================
This module is NEW (it has no Palantir counterpart). It exists because the new
Celonis UI lets a user set scenario weights (weight_revenue / weight_margin /
weight_delivery) and a prototype reservation (prototype_pct) on the
SIMULATION_OE_Table, whereas the Palantir engine simply CONSUMED an externally
supplied `model_priorities` table whose `priority` traces back to the MES 우선도
shop-floor grade (see memory note "revplan-simulation-priority").

The allocation engine only ever reads three columns from `model_priorities`:
    model_id, month, priority           (lower number = higher importance)
plus, for downstream financial enrichment / the fulfillment scorecard:
    simulation_id, simulation_name, revenue_plan_id,
    amount_per_unit, margin_amount_per_unit, __is_deleted

────────────────────────────────────────────────────────────────────────────
DEFAULT IS TARGET-STEP (customer's real practitioner rule — Gumi workshop 2026-07,
meeting_summary §2–3): priority = the "daily target step" critical ratio
    critical_ratio = remaining process steps ÷ remaining days
ranked within each demand month (higher ratio = more urgent). The revenue/margin/
delivery WEIGHTED score the client called "low value / dead system" is now an
OPT-IN mode (`use_weighted_priority=True`), not the default. Emergency-first is
handled upstream by the Tier-1 urgent pre-pass (urgent_allocation.py), so this
module only ranks the non-urgent remainder.

`priority_mode` selects the rule:
    "target_step" (DEFAULT) — critical ratio, delivery basis (demand-month EOM).
    "weighted"             — the OE-weighted revenue/margin/delivery score.
    "passthrough"          — return the supplied MES-derived table unchanged
                             (Palantir-faithful baseline).
`use_weighted_priority=True` forces "weighted"; if the weighted signal is
unsatisfiable it falls back to target_step (not to an empty MES placeholder).

WEIGHTED MODE is a documented DEFAULT formula, not a locked business rule — refine
`generate_weighted_priorities` once the business confirms the scoring. Sales-basis
target step (Rule 3, customer deadline) is deferred: no deadline field is sourced.
================================================================================
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import polars as pl

from .allocation_helpers import parse_month_to_eom_date


# Columns the allocation engine + enrichment expect on the priorities table.
REQUIRED_PRIORITY_COLUMNS = [
    "simulation_id",
    "simulation_name",
    "revenue_plan_id",
    "model_id",
    "month",
    "priority",
    "amount_per_unit",
    "margin_amount_per_unit",
    "__is_deleted",
]


def build_model_priorities(
    *,
    use_weighted_priority: bool,
    supplied_priorities: Optional[pl.DataFrame],
    priority_mode: str = "target_step",
    net_demand: Optional[pl.DataFrame] = None,
    revenue_plan: Optional[pl.DataFrame] = None,
    planned_process_steps: Optional[pl.DataFrame] = None,
    effective_start_date: Optional[date] = None,
    weight_revenue: float = 0.0,
    weight_margin: float = 0.0,
    weight_delivery: float = 0.0,
    prototype_pct: float = 0.0,
    prototype_models: Optional[set] = None,
    simulation_id: Optional[str] = None,
    simulation_name: Optional[str] = None,
    revenue_plan_id: Optional[str] = None,
) -> pl.DataFrame:
    """Dispatcher: return the `model_priorities` table the engine will consume.

    Mode resolution: `use_weighted_priority=True` forces "weighted"; otherwise
    `priority_mode` ("target_step" default | "weighted" | "passthrough") applies.

      * TARGET-STEP (default) -> critical ratio (remaining steps ÷ remaining days),
                                 the customer's real practitioner rule.
      * WEIGHTED              -> OE-weighted revenue/margin/delivery score, subject to
                                 a DATA-availability guard (margin needs a real
                                 margin_krw feed, null today → its weight is dropped);
                                 if nothing weightable survives, falls back to
                                 target_step.
      * PASS-THROUGH          -> return the supplied MES-derived table unchanged.
    """
    mode = "weighted" if use_weighted_priority else priority_mode

    if mode == "weighted":
        eff_revenue, eff_margin, eff_delivery, reason = _resolve_effective_weights(
            True, weight_revenue, weight_margin, weight_delivery, revenue_plan
        )
        if (eff_revenue + eff_margin + eff_delivery) > 0:
            print(f"   ▶ model_priorities: WEIGHTED — revenue={eff_revenue:g} margin={eff_margin:g} "
                  f"delivery={eff_delivery:g} ({reason})")
            return generate_weighted_priorities(
                net_demand=net_demand,
                revenue_plan=revenue_plan,
                weight_revenue=eff_revenue,
                weight_margin=eff_margin,
                weight_delivery=eff_delivery,
                prototype_pct=prototype_pct,
                prototype_models=prototype_models or set(),
                simulation_id=simulation_id,
                simulation_name=simulation_name,
                revenue_plan_id=revenue_plan_id,
            )
        # Weighted requested but unsatisfiable → fall back to the practitioner default.
        print(f"   ▶ model_priorities: WEIGHTED unsatisfiable ({reason}) → falling back to target_step")
        mode = "target_step"

    if mode == "target_step":
        result = generate_target_step_priorities(
            planned_process_steps=planned_process_steps,
            net_demand=net_demand,
            revenue_plan=revenue_plan,
            effective_start_date=effective_start_date,
            simulation_id=simulation_id,
            simulation_name=simulation_name,
            revenue_plan_id=revenue_plan_id,
        )
        if result is not None and not result.is_empty():
            print(f"   ▶ model_priorities: TARGET-STEP (critical ratio = remaining steps ÷ "
                  f"remaining days, delivery basis) — {result.height} (model,month) rows")
            return result
        # Not enough routing/demand to compute the critical ratio → baseline pass-through.
        print("   ▶ model_priorities: target_step unavailable (no routing/demand/start_date) "
              "→ PASS-THROUGH baseline")

    # passthrough (explicit, or a fallback from the two modes above)
    if supplied_priorities is None:
        raise ValueError(
            "priority_mode resolved to pass-through and no supplied `model_priorities` "
            "table was provided — wire one in read_inputs() (it may be empty, but must exist)."
        )
    print("   ▶ model_priorities: PASS-THROUGH baseline (supplied MES priorities)")
    return _ensure_required_columns(supplied_priorities)


def generate_weighted_priorities(
    *,
    net_demand: Optional[pl.DataFrame],
    revenue_plan: Optional[pl.DataFrame],
    weight_revenue: float,
    weight_margin: float,
    weight_delivery: float,
    prototype_pct: float,
    prototype_models: set,
    simulation_id: Optional[str],
    simulation_name: Optional[str],
    revenue_plan_id: Optional[str],
) -> pl.DataFrame:
    """Generate a per-(model, month) priority table from scenario weights.

    Scoring (DEFAULT formula — confirm with business before relying on it):
      * Derive per-unit economics from the revenue plan:
            amount_per_unit  = amount_krw / quantity_ea
            margin_per_unit  = margin_krw / quantity_ea
      * Per plan_month, min-max normalise three signals to [0, 1]:
            revenue_signal = amount_per_unit * net_demand        (total revenue at stake)
            margin_signal  = margin_per_unit * net_demand        (total margin at stake)
            delivery_signal= earliness of the month               (sooner = more urgent)
      * score = wR*revenue_signal + wM*margin_signal + wD*delivery_signal
      * priority = dense rank of score DESC within each month (1 = most important).
      * Prototype reservation: `prototype_models` get their score boosted so the top
        `prototype_pct` of each month's capacity is effectively reserved for them.
    """
    if revenue_plan is None or revenue_plan.is_empty():
        # Nothing to rank — return an empty, correctly-typed table.
        return _empty_priorities()

    rp = revenue_plan
    # Per-unit economics (guard divide-by-zero).
    rp = rp.with_columns(
        [
            pl.when(pl.col("quantity_ea") > 0)
            .then(pl.col("amount_krw") / pl.col("quantity_ea"))
            .otherwise(0.0)
            .alias("amount_per_unit"),
            pl.when(pl.col("quantity_ea") > 0)
            .then(pl.col("margin_krw") / pl.col("quantity_ea"))
            .otherwise(0.0)
            .alias("margin_amount_per_unit"),
        ]
    )

    # Quantity at stake: prefer net production demand if provided, else plan demand.
    if net_demand is not None and not net_demand.is_empty():
        nd = net_demand.select(
            pl.col("model_id"),
            pl.col("plan_month").alias("month"),
            pl.col("net_production_demand_ea").alias("qty"),
        )
        rp = rp.rename({"plan_month": "month"}).join(nd, on=["model_id", "month"], how="left")
        rp = rp.with_columns(pl.col("qty").fill_null(pl.col("quantity_ea")))
    else:
        rp = rp.rename({"plan_month": "month"}).with_columns(pl.col("quantity_ea").alias("qty"))

    rp = rp.with_columns(
        [
            (pl.col("amount_per_unit") * pl.col("qty")).alias("revenue_signal"),
            (pl.col("margin_amount_per_unit") * pl.col("qty")).alias("margin_signal"),
        ]
    )

    # Delivery urgency: earlier month => higher signal. Built in Python (no
    # with_row_index/with_row_count) so it works across polars versions.
    months = sorted(m for m in rp.select("month").unique().to_series().to_list() if m is not None)
    denom = max(len(months) - 1, 1)
    month_rank = pl.DataFrame({
        "month": months,
        "delivery_signal": [1.0 - i / denom for i in range(len(months))],
    })
    rp = rp.join(month_rank, on="month", how="left")

    # Min-max normalise the revenue/margin signals WITHIN each month.
    def _norm(col: str) -> pl.Expr:
        lo = pl.col(col).min().over("month")
        hi = pl.col(col).max().over("month")
        return (
            pl.when(hi > lo)
            .then((pl.col(col) - lo) / (hi - lo))
            .otherwise(0.0)
            .alias(col + "_n")
        )

    rp = rp.with_columns([_norm("revenue_signal"), _norm("margin_signal")])

    rp = rp.with_columns(
        (
            weight_revenue * pl.col("revenue_signal_n")
            + weight_margin * pl.col("margin_signal_n")
            + weight_delivery * pl.col("delivery_signal")
        ).alias("score")
    )

    # Prototype reservation: boost prototype models above all others (+1.0 > max
    # possible weighted score) so they occupy the top of each month's ranking. The
    # `prototype_pct` knob is carried through for the engine's capacity reservation;
    # here it documents intent — refine once the prototype source is confirmed.
    if prototype_models:
        rp = rp.with_columns(
            pl.when(pl.col("model_id").is_in(list(prototype_models)))
            .then(pl.col("score") + 1.0)
            .otherwise(pl.col("score"))
            .alias("score")
        )

    # priority = rank of score DESC within month (1 = highest importance).
    rp = rp.with_columns(
        pl.col("score").rank(method="ordinal", descending=True).over("month").cast(pl.Int64).alias("priority")
    )

    out = rp.select(
        pl.lit(simulation_id).alias("simulation_id"),
        pl.lit(simulation_name).alias("simulation_name"),
        pl.lit(revenue_plan_id).alias("revenue_plan_id"),
        pl.col("model_id"),
        pl.col("month"),
        pl.col("priority"),
        pl.col("amount_per_unit"),
        pl.col("margin_amount_per_unit"),
        pl.lit(False).alias("__is_deleted"),
    )
    # One row per (model, month): collapse any duplicate plan rows.
    out = out.unique(subset=["model_id", "month"], keep="first", maintain_order=True)
    return out


def generate_target_step_priorities(
    *,
    planned_process_steps: Optional[pl.DataFrame],
    net_demand: Optional[pl.DataFrame],
    revenue_plan: Optional[pl.DataFrame],
    effective_start_date: Optional[date],
    simulation_id: Optional[str],
    simulation_name: Optional[str],
    revenue_plan_id: Optional[str],
) -> pl.DataFrame:
    """Rank each (model, month) by the customer's "daily target step" critical ratio.

    critical_ratio = remaining_steps / remaining_days
      * remaining_steps = the model's routing step count (from planned_process_steps).
      * remaining_days  = days from the sim start to the demand month's end
                          (DELIVERY basis — meeting_summary §3 Rule 2), clamped ≥ 1.
    Higher ratio = more urgent = lower priority number (1 = most urgent) within month.

    Returns an empty (correctly-typed) table when routing/demand/start-date are missing,
    which the dispatcher treats as "fall back to pass-through baseline".
    """
    if planned_process_steps is None or planned_process_steps.is_empty():
        return _empty_priorities()
    if effective_start_date is None:
        return _empty_priorities()

    universe = _demand_universe(net_demand, revenue_plan)
    if universe is None or universe.is_empty():
        return _empty_priorities()

    # remaining_steps: routing step count per model (mirrors virtual_lot_creator's
    # `num_steps = len(model_process_steps)`).
    steps = (
        planned_process_steps
        .filter(pl.col("model_id").is_not_null())
        .group_by("model_id")
        .agg(pl.len().alias("num_steps"))
    )

    df = universe.join(steps, on="model_id", how="left")
    df = df.filter(pl.col("num_steps").is_not_null() & (pl.col("num_steps") > 0))
    if df.is_empty():
        return _empty_priorities()

    # remaining_days per demand month = (month EOM − sim start), clamped ≥ 1 day.
    months = [m for m in df.select("month").unique().to_series().to_list() if m is not None]
    rows = []
    for m in months:
        try:
            days = (parse_month_to_eom_date(str(m)) - effective_start_date).days
        except Exception:
            days = None
        rows.append({"month": m, "remaining_days": max(days, 1) if days is not None else 1})
    remaining_days = pl.DataFrame(rows, schema={"month": pl.Utf8, "remaining_days": pl.Int64})
    df = df.join(remaining_days, on="month", how="left")

    df = df.with_columns(
        (pl.col("num_steps") / pl.col("remaining_days")).alias("critical_ratio")
    )
    # priority = rank of critical_ratio DESC within month (1 = most urgent).
    df = df.with_columns(
        pl.col("critical_ratio").rank(method="ordinal", descending=True).over("month").cast(pl.Int64).alias("priority")
    )

    df = _attach_unit_economics(df, revenue_plan)

    out = df.select(
        pl.lit(simulation_id).alias("simulation_id"),
        pl.lit(simulation_name).alias("simulation_name"),
        pl.lit(revenue_plan_id).alias("revenue_plan_id"),
        pl.col("model_id"),
        pl.col("month"),
        pl.col("priority"),
        pl.col("amount_per_unit"),
        pl.col("margin_amount_per_unit"),
        pl.lit(False).alias("__is_deleted"),
    )
    return out.unique(subset=["model_id", "month"], keep="first", maintain_order=True)


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------

def _demand_universe(
    net_demand: Optional[pl.DataFrame], revenue_plan: Optional[pl.DataFrame]
) -> Optional[pl.DataFrame]:
    """Distinct (model_id, month) with demand — prefer net production demand, else the plan."""
    for src in (net_demand, revenue_plan):
        if src is not None and not src.is_empty() and {"model_id", "plan_month"} <= set(src.columns):
            return (
                src.select(
                    pl.col("model_id").cast(pl.Utf8),
                    pl.col("plan_month").cast(pl.Utf8).alias("month"),  # Utf8 so it joins remaining_days
                )
                .filter(pl.col("model_id").is_not_null() & pl.col("month").is_not_null())
                .unique()
            )
    return None


def _attach_unit_economics(df: pl.DataFrame, revenue_plan: Optional[pl.DataFrame]) -> pl.DataFrame:
    """Left-join per-unit amount/margin from the revenue plan (downstream financial
    enrichment reads these). Missing source/columns → null columns, matching the
    weighted path's output schema."""
    cols = set(revenue_plan.columns) if (revenue_plan is not None and not revenue_plan.is_empty()) else set()
    if not {"model_id", "plan_month", "quantity_ea", "amount_krw"} <= cols:
        return df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("amount_per_unit"),
            pl.lit(None, dtype=pl.Float64).alias("margin_amount_per_unit"),
        ])
    margin_expr = (
        pl.when((pl.col("quantity_ea") > 0) & pl.col("margin_krw").is_not_null())
        .then(pl.col("margin_krw") / pl.col("quantity_ea"))
        .otherwise(None)
        .alias("margin_amount_per_unit")
        if "margin_krw" in cols
        else pl.lit(None, dtype=pl.Float64).alias("margin_amount_per_unit")
    )
    econ = (
        revenue_plan.rename({"plan_month": "month"})
        .with_columns([
            pl.when(pl.col("quantity_ea") > 0)
            .then(pl.col("amount_krw") / pl.col("quantity_ea"))
            .otherwise(0.0)
            .alias("amount_per_unit"),
            margin_expr,
        ])
        .select(["model_id", "month", "amount_per_unit", "margin_amount_per_unit"])
        .unique(subset=["model_id", "month"], keep="first")
    )
    return df.join(econ, on=["model_id", "month"], how="left")


def _empty_priorities() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "simulation_id": pl.Utf8,
            "simulation_name": pl.Utf8,
            "revenue_plan_id": pl.Utf8,
            "model_id": pl.Utf8,
            "month": pl.Utf8,
            "priority": pl.Int64,
            "amount_per_unit": pl.Float64,
            "margin_amount_per_unit": pl.Float64,
            "__is_deleted": pl.Boolean,
        }
    )


def _margin_data_available(revenue_plan: Optional[pl.DataFrame]) -> bool:
    """True only if the revenue plan actually carries margin figures. `margin_krw` is
    a placeholder null column today (no cost source in PPS/RTS), so a margin weight
    would silently contribute nothing — detect that here rather than pretend it works."""
    if revenue_plan is None or revenue_plan.is_empty():
        return False
    if "margin_krw" not in revenue_plan.columns:
        return False
    non_null = revenue_plan.select(pl.col("margin_krw").is_not_null().sum()).item()
    return bool(non_null)


def _resolve_effective_weights(
    use_weighted_priority: bool,
    weight_revenue: float,
    weight_margin: float,
    weight_delivery: float,
    revenue_plan: Optional[pl.DataFrame],
) -> tuple:
    """Turn the REQUESTED weights into the weights we can actually honor, given the data.

    Returns (revenue, margin, delivery, reason). A zero total tells the dispatcher to
    fall back to the MES/baseline pass-through. Survivors are intentionally NOT rescaled:
    priority is a rank of a weighted sum, and rank is invariant under positive scaling,
    so dropping a signal already yields the correct relative ordering.
    """
    if not use_weighted_priority:
        return 0.0, 0.0, 0.0, "weighted-priority toggle off"

    wr = max(float(weight_revenue), 0.0)
    wm = max(float(weight_margin), 0.0)
    wd = max(float(weight_delivery), 0.0)

    if (wr + wm + wd) <= 0:
        return 0.0, 0.0, 0.0, "all weights zero"

    notes = []
    if wm > 0 and not _margin_data_available(revenue_plan):
        notes.append(f"margin weight {wm:g} dropped (margin_krw unsourced/all-null)")
        wm = 0.0

    if (wr + wm + wd) <= 0:
        return 0.0, 0.0, 0.0, ("; ".join(notes) + " — no weighted signal left")

    return wr, wm, wd, ("; ".join(notes) if notes else "weights supplied")


def _ensure_required_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Pass-through baseline: make sure the supplied table has the columns the
    engine reads, filling sensible defaults for any optional ones that are absent."""
    missing = [c for c in ("model_id", "month", "priority") if c not in df.columns]
    if missing:
        raise ValueError(
            f"supplied model_priorities is missing required column(s) {missing}. "
            "The engine reads model_id, month, priority."
        )
    fill_defaults = {
        "simulation_id": pl.lit(None, dtype=pl.Utf8),
        "simulation_name": pl.lit(None, dtype=pl.Utf8),
        "revenue_plan_id": pl.lit(None, dtype=pl.Utf8),
        "amount_per_unit": pl.lit(None, dtype=pl.Float64),
        "margin_amount_per_unit": pl.lit(None, dtype=pl.Float64),
        "__is_deleted": pl.lit(False, dtype=pl.Boolean),
    }
    add = [expr.alias(name) for name, expr in fill_defaults.items() if name not in df.columns]
    return df.with_columns(add) if add else df
