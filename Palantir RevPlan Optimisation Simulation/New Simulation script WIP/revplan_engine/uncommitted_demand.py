"""
[MLWB PORT] This module is a VERBATIM copy of the Palantir Foundry source
file "uncommitted_demand.py" from ../Python Script Allocation Engine/ (ported 2026-08-26,
unblocked by the o_custom_SalesOrderLine extraction). The ONLY change is the import
rewrite (transforms.api -> ._foundry_shim shim); the pure logic is byte-identical so it
can be re-synced if the Palantir engine changes. Foundry decorators are no-ops here; the
engine drives compute() in-memory via the wrapper in run_simulation.py
(compute_uncommitted_demand). Its sales_order_lines input comes from
celonis_io.read_sales_order_lines (o_custom_SalesOrderLine; Cancelled lines are excluded
AT READ TIME as a marked divergence — the source never saw cancelled rows), and its
revenue_plan_optimization_events input is this run's ENRICHED allocation frame.
"""
import polars as pl
from ._foundry_shim import Input, Output, LightweightInput, LightweightOutput, transform


@transform.using(
    uncommitted_demand_by_model=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/uncommitted_demand_by_model"
    ),
    uncommitted_demand_by_lot=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/uncommitted_demand_by_lot"
    ),
    sales_order_lines=Input("ri.foundry.main.dataset.2b70a0bb-50d2-48d3-b91f-a8110ef1788c"),
    revenue_plan_optimization_events=Input("ri.foundry.main.dataset.b84b2bb5-1cbf-4970-8f73-0e1eb9d4148f"),
)
def compute(
    uncommitted_demand_by_model: LightweightOutput,
    uncommitted_demand_by_lot: LightweightOutput,
    sales_order_lines: LightweightInput,
    revenue_plan_optimization_events: LightweightInput,
) -> None:
    # Load data
    sales_orders = sales_order_lines.polars()
    optimization_events = revenue_plan_optimization_events.polars()

    # Step 1: Calculate total committed (open) demand per model from sales orders
    # Open = order_quantity - shipped_quantity where shipped_quantity < order_quantity
    # item_no is the model_id in this dataset
    committed_demand = (
        sales_orders.filter(pl.col("shipped_quantity") < pl.col("order_quantity"))
        .with_columns([(pl.col("order_quantity") - pl.col("shipped_quantity")).alias("open_quantity")])
        .group_by("item_no")
        .agg(
            [
                pl.col("open_quantity").sum().alias("total_committed_quantity"),
                pl.count().alias("sales_order_line_count"),
            ]
        )
        .rename({"item_no": "model_id"})
    )

    # Step 2: Get unique lots with their size (final_production_units is lot size, same across all steps)
    # Filter to new lots and deduplicate by lot_id since is_new_lot=True appears for ALL steps in a new lot
    lot_starts = (
        optimization_events.filter(pl.col("is_new_lot") == True)
        .unique(subset=["simulation_id", "lot_id"], keep="first")
        .select(
            [
                "simulation_id",
                "simulation_name",
                "lot_id",
                "model_id",
                "allocated_date",
                "target_month",
                "final_production_units",  # This is the lot size
                "total_revenue",
                "total_margin",
                "revenue_plan_id",
                "model_priority",
                "model_customer_name",
                "model_end_customer",
                "model_sales_team",
                "grouping_model",
            ]
        )
    )

    # Step 3: Calculate cumulative production by simulation and model
    # Sort by allocated_date to track chronological production
    lot_starts_with_cumulative = lot_starts.sort(
        ["simulation_id", "model_id", "allocated_date", "lot_id"]
    ).with_columns(
        [
            pl.col("final_production_units")
            .cum_sum()
            .over(["simulation_id", "model_id"])
            .alias("cumulative_production_units")
        ]
    )

    # Step 4: Join with committed demand and determine if lot is committed or uncommitted
    # A lot is uncommitted if cumulative production BEFORE starting it already exceeds committed demand
    lot_level_analysis = (
        lot_starts_with_cumulative.join(committed_demand, on="model_id", how="left")
        .with_columns(
            [
                # Fill nulls with 0 for models with no committed demand
                pl.col("total_committed_quantity").fill_null(0),
                pl.col("sales_order_line_count").fill_null(0),
            ]
        )
        .with_columns(
            [
                # Calculate cumulative production BEFORE this lot started
                (pl.col("cumulative_production_units") - pl.col("final_production_units")).alias(
                    "cumulative_before_lot"
                ),
            ]
        )
        .with_columns(
            [
                # A lot is UNCOMMITTED if we already had enough production to cover committed demand
                # before starting this lot
                (pl.col("cumulative_before_lot") >= pl.col("total_committed_quantity")).alias("is_uncommitted_lot"),
                # Flag the FIRST uncommitted lot (cumulative before < committed, but this lot pushes us over)
                (
                    (pl.col("cumulative_before_lot") < pl.col("total_committed_quantity"))
                    & (pl.col("cumulative_production_units") >= pl.col("total_committed_quantity"))
                ).alias("is_first_uncommitted_lot"),
            ]
        )
    )
    # Step 5: Output lot-level uncommitted demand (filter to only uncommitted lots)
    uncommitted_lots = (
        lot_level_analysis.filter(pl.col("is_uncommitted_lot") == True)
        .with_columns(
            [
                # Create stable unique ID by concatenating simulation_id with lot_id
                (pl.col("simulation_id") + "_" + pl.col("lot_id")).alias("uncommitted_lot_id")
            ]
        )
        .select(
            [
                "uncommitted_lot_id",
                "simulation_id",
                "simulation_name",
                "lot_id",
                "model_id",
                "allocated_date",
                "target_month",
                "final_production_units",
                "cumulative_production_units",
                "cumulative_before_lot",
                "total_committed_quantity",
                "sales_order_line_count",
                "is_first_uncommitted_lot",
                "total_revenue",
                "total_margin",
                "revenue_plan_id",
                "model_priority",
                "model_customer_name",
                "model_end_customer",
                "model_sales_team",
                "grouping_model",
            ]
        )
    )

    # Step 6: Create model-level aggregation
    model_level_summary = (
        lot_level_analysis.group_by(["simulation_id", "simulation_name", "model_id"])
        .agg(
            [
                # Committed demand info
                pl.col("total_committed_quantity").first(),
                pl.col("sales_order_line_count").first(),
                # Total planned production (sum lot sizes, not duplicated step data)
                pl.col("final_production_units").sum().alias("total_planned_production_units"),
                pl.col("total_revenue").sum().alias("total_planned_revenue"),
                pl.col("total_margin").sum().alias("total_planned_margin"),
                # Uncommitted lot metrics
                pl.col("final_production_units")
                .filter(pl.col("is_uncommitted_lot") == True)
                .sum()
                .fill_null(0)
                .alias("total_uncommitted_units"),
                pl.col("is_uncommitted_lot").sum().alias("uncommitted_lot_count"),
                pl.col("lot_id").n_unique().alias("total_lot_count"),
                # First uncommitted lot details
                pl.col("allocated_date")
                .filter(pl.col("is_first_uncommitted_lot") == True)
                .min()
                .alias("first_uncommitted_lot_date"),
                pl.col("lot_id")
                .filter(pl.col("is_first_uncommitted_lot") == True)
                .first()
                .alias("first_uncommitted_lot_id"),
                # Model metadata
                pl.col("model_customer_name").first(),
                pl.col("model_end_customer").first(),
                pl.col("model_sales_team").first(),
                pl.col("grouping_model").first(),
                pl.col("model_priority").first(),
            ]
        )
        .with_columns(
            [
                # Create stable unique ID by concatenating simulation_id with model_id
                (pl.col("simulation_id") + "_" + pl.col("model_id")).alias("uncommitted_model_id"),
                # Calculate overage
                (pl.col("total_planned_production_units") - pl.col("total_committed_quantity")).alias(
                    "production_overage_units"
                ),
                # Flag models with any uncommitted demand
                (pl.col("total_uncommitted_units") > 0).alias("has_uncommitted_demand"),
            ]
        )
        .select(
            [
                "uncommitted_model_id",
                "simulation_id",
                "simulation_name",
                "model_id",
                "total_committed_quantity",
                "sales_order_line_count",
                "total_planned_production_units",
                "total_planned_revenue",
                "total_planned_margin",
                "total_uncommitted_units",
                "uncommitted_lot_count",
                "total_lot_count",
                "first_uncommitted_lot_date",
                "first_uncommitted_lot_id",
                "model_customer_name",
                "model_end_customer",
                "model_sales_team",
                "grouping_model",
                "model_priority",
                "production_overage_units",
                "has_uncommitted_demand",
            ]
        )
        .sort(["simulation_id", "model_id"])
    )

    # Write outputs
    uncommitted_demand_by_lot.write_table(uncommitted_lots)
    uncommitted_demand_by_model.write_table(model_level_summary)
