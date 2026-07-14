"""
Revenue Plan Uncommitted Demand Calculator

This transform calculates committed vs uncommitted demand for each Revenue Plan
at the model-month level, BEFORE any simulation runs. This helps identify whether
the plan itself has issues from the outset.

Key Logic:
1. Get total committed (open) quantity per model from Sales Order Lines
   - Open = order_quantity - shipped_quantity where shipped_quantity < order_quantity
2. For each revenue_plan_id + model_id, process months chronologically
3. Deplete committed quantity month-by-month against planned demand
4. Track how much of each month's demand is committed vs uncommitted

This differs from uncommitted_demand.py which looks at simulation engine output (lots).
This transform looks at the Revenue Plan itself before any simulation.

Inputs:
- Revenue Plan Monthly Model: Planned demand by revenue_plan_id, model_id, plan_month
- Sales Order Lines: Open orders to determine committed quantity per model
  - item_no = model_id
  - committed = order_quantity - shipped_quantity (where shipped < ordered)

Output:
- One row per revenue_plan_id, model_id, plan_month showing:
  - Planned demand (quantity and KRW)
  - Committed demand (backed by sales orders)
  - Uncommitted demand (not backed by sales orders)
  - Cumulative tracking of commitment status
"""

import polars as pl
from transforms.api import transform, Input, Output, LightweightInput, LightweightOutput


@transform.using(
    revenue_plan_uncommitted_demand=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/revenue_plan_uncommitted_demand"
    ),
    revenue_plan_monthly_model=Input("ri.foundry.main.dataset.2e72556a-1f0b-4ecc-9bdb-3e172781a406"),
    sales_order_lines=Input("ri.foundry.main.dataset.2b70a0bb-50d2-48d3-b91f-a8110ef1788c"),
)
def compute(
    revenue_plan_uncommitted_demand: LightweightOutput,
    revenue_plan_monthly_model: LightweightInput,
    sales_order_lines: LightweightInput,
) -> None:
    # Load data
    revenue_plan = revenue_plan_monthly_model.polars()
    sales_orders = sales_order_lines.polars()

    print(f"Revenue Plan Monthly Model rows: {revenue_plan.height:,}")
    print(f"Sales Order Lines rows: {sales_orders.height:,}")

    # =========================================================================
    # Step 1: Calculate total committed (open) quantity per model from sales orders
    # Open = order_quantity - shipped_quantity where shipped_quantity < order_quantity
    # item_no is the model_id in this dataset
    # =========================================================================
    committed_by_model = (
        sales_orders.filter(pl.col("shipped_quantity") < pl.col("order_quantity"))
        .with_columns([(pl.col("order_quantity") - pl.col("shipped_quantity")).alias("open_quantity")])
        .group_by("item_no")
        .agg(
            [
                pl.col("open_quantity").sum().alias("total_committed_quantity_ea"),
                pl.count().alias("sales_order_line_count"),
            ]
        )
        .rename({"item_no": "model_id"})
    )

    print(f"Models with committed demand: {committed_by_model.height:,}")

    # =========================================================================
    # Step 2: Prepare Revenue Plan data
    # =========================================================================
    revenue_plan_clean = (
        revenue_plan.filter(
            pl.col("revenue_plan_id").is_not_null()
            & pl.col("model_id").is_not_null()
            & pl.col("plan_month").is_not_null()
        )
        .with_columns(
            [
                pl.col("quantity_ea").fill_null(0).alias("planned_quantity_ea"),
                pl.col("amount_krw").fill_null(0).alias("planned_amount_krw"),
            ]
        )
        .select(
            [
                "revenue_plan_id",
                "model_id",
                "grouping_model",
                "plan_month",
                "planned_quantity_ea",
                "planned_amount_krw",
                "sales_team",
            ]
        )
    )

    print(f"Revenue Plan rows after cleaning: {revenue_plan_clean.height:,}")
    print(f"Unique revenue plans: {revenue_plan_clean['revenue_plan_id'].n_unique()}")
    print(f"Unique models: {revenue_plan_clean['model_id'].n_unique()}")

    # =========================================================================
    # Step 3: Join revenue plan with committed demand
    # =========================================================================
    plan_with_committed = revenue_plan_clean.join(
        committed_by_model,
        on="model_id",
        how="left",
    ).with_columns(
        [
            pl.col("total_committed_quantity_ea").fill_null(0),
            pl.col("sales_order_line_count").fill_null(0),
        ]
    )

    # Sort by revenue_plan_id, model_id, plan_month for chronological processing
    plan_with_committed = plan_with_committed.sort(["revenue_plan_id", "model_id", "plan_month"])

    # =========================================================================
    # Step 4: Calculate cumulative demand and deplete committed quantity
    # =========================================================================
    # Calculate cumulative planned quantity within each revenue_plan + model
    plan_with_cumulative = plan_with_committed.with_columns(
        [
            pl.col("planned_quantity_ea")
            .cum_sum()
            .over(["revenue_plan_id", "model_id"])
            .alias("cumulative_planned_quantity_ea"),
        ]
    ).with_columns(
        [
            # Cumulative demand BEFORE this month
            (pl.col("cumulative_planned_quantity_ea") - pl.col("planned_quantity_ea")).alias(
                "cumulative_before_month_ea"
            ),
        ]
    )

    # Calculate committed vs uncommitted for each month
    result = (
        plan_with_cumulative.with_columns(
            [
                # How much committed quantity is still available before this month?
                # remaining_committed = total_committed - cumulative_before_month (but not less than 0)
                pl.max_horizontal(
                    pl.col("total_committed_quantity_ea") - pl.col("cumulative_before_month_ea"),
                    pl.lit(0),
                ).alias("committed_remaining_before_month_ea"),
            ]
        )
        .with_columns(
            [
                # Committed quantity for THIS month = min(remaining_committed, planned_this_month)
                pl.min_horizontal(
                    pl.col("committed_remaining_before_month_ea"),
                    pl.col("planned_quantity_ea"),
                ).alias("committed_quantity_ea"),
            ]
        )
        .with_columns(
            [
                # Uncommitted quantity = planned - committed
                (pl.col("planned_quantity_ea") - pl.col("committed_quantity_ea")).alias("uncommitted_quantity_ea"),
                # Calculate unit price from revenue plan
                pl.when(pl.col("planned_quantity_ea") > 0)
                .then(pl.col("planned_amount_krw") / pl.col("planned_quantity_ea"))
                .otherwise(0)
                .alias("unit_price_krw"),
            ]
        )
        .with_columns(
            [
                # Calculate KRW amounts based on unit price
                (pl.col("committed_quantity_ea") * pl.col("unit_price_krw")).alias("committed_amount_krw"),
                (pl.col("uncommitted_quantity_ea") * pl.col("unit_price_krw")).alias("uncommitted_amount_krw"),
                # Calculate percentages
                pl.when(pl.col("planned_quantity_ea") > 0)
                .then(pl.col("committed_quantity_ea") / pl.col("planned_quantity_ea") * 100.0)
                .otherwise(0.0)
                .alias("committed_pct"),
                pl.when(pl.col("planned_quantity_ea") > 0)
                .then(pl.col("uncommitted_quantity_ea") / pl.col("planned_quantity_ea") * 100.0)
                .otherwise(0.0)
                .alias("uncommitted_pct"),
                # Flag if this month has any uncommitted demand
                (pl.col("cumulative_before_month_ea") >= pl.col("total_committed_quantity_ea")).alias(
                    "is_fully_uncommitted_month"
                ),
                # Flag if this is the first month where uncommitted demand appears
                (
                    (pl.col("cumulative_before_month_ea") < pl.col("total_committed_quantity_ea"))
                    & (pl.col("cumulative_planned_quantity_ea") >= pl.col("total_committed_quantity_ea"))
                ).alias("is_first_uncommitted_month"),
            ]
        )
    )

    # Create primary key
    result = result.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("revenue_plan_id"),
                    pl.col("model_id"),
                    pl.col("plan_month"),
                ],
                separator="_",
            ).alias("pk"),
        ]
    )

    # Select final columns
    output_df = result.select(
        [
            "pk",
            "revenue_plan_id",
            "model_id",
            "grouping_model",
            "plan_month",
            "sales_team",
            # Planned demand
            "planned_quantity_ea",
            "planned_amount_krw",
            # Committed demand (backed by sales orders)
            "committed_quantity_ea",
            "committed_amount_krw",
            "committed_pct",
            # Uncommitted demand (not backed by sales orders)
            "uncommitted_quantity_ea",
            "uncommitted_amount_krw",
            "uncommitted_pct",
            # Cumulative tracking
            "cumulative_planned_quantity_ea",
            "cumulative_before_month_ea",
            "total_committed_quantity_ea",
            "committed_remaining_before_month_ea",
            "sales_order_line_count",
            # Flags
            "is_fully_uncommitted_month",
            "is_first_uncommitted_month",
        ]
    ).sort(["revenue_plan_id", "model_id", "plan_month"])

    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    total_planned_qty = output_df["planned_quantity_ea"].sum()
    total_committed_qty = output_df["committed_quantity_ea"].sum()
    total_uncommitted_qty = output_df["uncommitted_quantity_ea"].sum()
    total_planned_krw = output_df["planned_amount_krw"].sum()
    total_committed_krw = output_df["committed_amount_krw"].sum()
    total_uncommitted_krw = output_df["uncommitted_amount_krw"].sum()

    print(f"\n=== Revenue Plan Uncommitted Demand Summary ===")
    print(f"Total rows: {output_df.height:,}")
    print(f"Unique revenue plans: {output_df['revenue_plan_id'].n_unique()}")
    print(f"Unique models: {output_df['model_id'].n_unique()}")
    print(f"\nQuantity (EA):")
    print(f"  Total planned:     {total_planned_qty:,}")
    print(f"  Total committed:   {total_committed_qty:,}")
    print(f"  Total uncommitted: {total_uncommitted_qty:,}")
    if total_planned_qty > 0:
        print(f"  Committed %:       {total_committed_qty / total_planned_qty * 100:.1f}%")
    print(f"\nAmount (KRW):")
    print(f"  Total planned:     {total_planned_krw:,.0f}")
    print(f"  Total committed:   {total_committed_krw:,.0f}")
    print(f"  Total uncommitted: {total_uncommitted_krw:,.0f}")

    # Summary by revenue plan
    print(f"\n=== Summary by Revenue Plan ===")
    plan_summary = (
        output_df.group_by("revenue_plan_id")
        .agg(
            [
                pl.col("planned_quantity_ea").sum().alias("total_planned"),
                pl.col("committed_quantity_ea").sum().alias("total_committed"),
                pl.col("uncommitted_quantity_ea").sum().alias("total_uncommitted"),
                pl.col("model_id").n_unique().alias("model_count"),
                pl.count().alias("row_count"),
            ]
        )
        .sort("revenue_plan_id")
    )

    for row in plan_summary.iter_rows(named=True):
        plan_id = row["revenue_plan_id"]
        planned = row["total_planned"]
        committed = row["total_committed"]
        uncommitted = row["total_uncommitted"]
        models = row["model_count"]
        committed_pct = (committed / planned * 100) if planned > 0 else 0
        print(
            f"  {plan_id}: {models:,} models | "
            f"Planned: {planned:,} | Committed: {committed:,} ({committed_pct:.1f}%) | "
            f"Uncommitted: {uncommitted:,}"
        )

    # =========================================================================
    # Write Output
    # =========================================================================
    revenue_plan_uncommitted_demand.write_table(output_df)

    print(f"\n✓ Output written: {output_df.height:,} rows")
