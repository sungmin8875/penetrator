"""
Net Production Demand Calculator

This transform calculates the net production demand by model and month after
accounting for available inventory with forward-rolling logic.

Key Logic:
1. Each revenue_plan_id represents a different planning scenario
2. For each revenue plan, for each model grouping, process months in chronological order:
   - Use grouping_model when available, otherwise fall back to model_id
   - Start with inventory available in current month (shipped, onhand, transit)
   - Add any excess inventory rolled forward from previous month
   - Subtract from planned demand using priority: transit first, then shipped, then onhand
   - If inventory exceeds demand, roll excess forward to next month
   - If demand exceeds inventory, calculate net production needed
3. Output ONE ROW per revenue_plan_id per grouping_model per month with clear accounting
4. Net production demand excludes "기타매출" (other revenue) as it comes from non-production sources

Inputs:
- Revenue Plan: Monthly planned production targets (quantity_ea per model per month per plan)
  - Has grouping_model column (populated in ~96% of rows)
- Available Inventory: Inventory already available per model per month
  - shipped_quantity_ea: Already shipped inventory
  - onhand_quantity_ea: On-hand inventory
  - transit calculated as: total_inventory_ea - shipped_quantity_ea - onhand_quantity_ea
  - Has grouping_model column (populated in ~29% of rows)

Output:
- One row per revenue_plan_id per grouping_model per month showing:
  - revenue_plan_id: The specific revenue plan scenario
  - grouping_model: The grouping used (grouping_model if available, else model_id)
  - model_id: The original model_id value
  - revenue_category: "기타매출" for other revenue, "제품" for products
  - planned_demand_ea: What the revenue plan requires
  - revplan_amount_in_krw: Revenue amount in KRW from the revenue plan
  - shipped_inventory_this_month_ea: Shipped inventory available this month
  - shipped_inventory_this_month_krw: KRW value of shipped inventory this month
  - onhand_inventory_this_month_ea: On-hand inventory available this month
  - transit_inventory_this_month_ea: Transit inventory available this month
  - total_shipped_all_models_ea: Total shipped inventory across ALL models for that month
  - inventory_from_previous_month_ea: Excess rolled forward from previous month
  - total_inventory_available_ea: Total inventory available to use
  - shipped_consumed_ea: How much shipped inventory was used
  - onhand_consumed_ea: How much on-hand inventory was used
  - transit_consumed_ea: How much transit inventory was used
  - inventory_consumed_ea: Total inventory consumed
  - inventory_consumed_amount_krw: KRW value of inventory consumed
  - inventory_rolled_to_next_month_ea: Excess inventory rolled forward
  - net_production_demand_ea: What still needs to be produced by WIP (0 for 기타매출)
  - net_production_amount_krw: KRW value of net production demand (0 for 기타매출)
  - fulfillment_pct: % of demand fulfilled by inventory

Note: Rolling is FORWARD ONLY - excess inventory flows to future months, never backward.
Note: Inventory is shared across all revenue plans - each plan scenario uses the same inventory pool.
Note: grouping_model contains grouping_model when available (non-null and non-empty), otherwise model_id.
Note: Net production demand for "기타매출" is always 0 as it comes from other sources (not production).
Note: Consumption priority: transit first, then shipped, then onhand.
"""

import polars as pl
from transforms.api import transform, Input, Output, lightweight
from datetime import datetime


@lightweight(cpu_cores=2, memory_gb=8)
@transform(
    output=Output("ri.foundry.main.dataset.122a25b5-165e-4eba-adfa-e7865add43d1"),
    revenue_plan=Input("ri.foundry.main.dataset.2e72556a-1f0b-4ecc-9bdb-3e172781a406"),
    available_inventory=Input("ri.foundry.main.dataset.a8f8675b-a7b2-47f5-b83f-97f7028cc1b5"),
)
def compute(revenue_plan, available_inventory, output) -> None:
    """
    Calculate net production demand with forward-rolling inventory logic.

    This transform:
    1. Processes each revenue_plan_id separately
    2. For each plan, applies available inventory with forward rolling between months
    3. Tracks shipped, onhand, and transit inventory separately
    4. Uses grouping_model when available, otherwise falls back to model_id
    5. Categorizes revenue as "기타매출" (other) or "제품" (product)
    6. Sets net production demand to 0 for "기타매출" as it comes from other sources
    7. Outputs net production requirements (one row per plan-grouping_model-month)

    Rolling Logic:
    - Process each plan's grouping_model-months chronologically
    - Excess inventory in month N is added to inventory in month N+1
    - Inventory never rolls backward
    - Consumption priority: transit first, then shipped, then onhand
    - Each revenue plan scenario gets its own calculation (inventory is independent per plan)

    Output columns:
    - revenue_plan_id: The specific revenue plan scenario
    - grouping_model: The effective grouping (grouping_model if available, else model_id)
    - model_id: Original model_id column value
    - revenue_category: "기타매출" for other revenue, "제품" for products
    - plan_month: The planning month (YYYYMM format)
    - planned_demand_ea: Total demand from revenue plan
    - revplan_amount_in_krw: Revenue amount in KRW from the revenue plan
    - shipped_inventory_this_month_ea: Shipped inventory available this month
    - onhand_inventory_this_month_ea: On-hand inventory available this month
    - transit_inventory_this_month_ea: Transit inventory available this month
    - total_shipped_all_models_ea: Total shipped inventory across ALL models for that month
    - inventory_from_previous_month_ea: Rolled forward from previous month
    - total_inventory_available_ea: Sum of all inventory types + rolled forward
    - shipped_consumed_ea: How much shipped inventory was used
    - onhand_consumed_ea: How much on-hand inventory was used
    - transit_consumed_ea: How much transit inventory was used
    - inventory_consumed_ea: Total inventory consumed
    - inventory_consumed_amount_krw: KRW value of inventory consumed
    - inventory_rolled_to_next_month_ea: Excess rolled to next month
    - net_production_demand_ea: Remaining demand to be fulfilled by WIP (0 for 기타매출)
    - net_production_amount_krw: KRW value of net production demand (0 for 기타매출)
    - fulfillment_pct: Percentage of demand fulfilled by inventory
    - calculation_timestamp: When this calculation was performed
    """

    # =========================================================================
    # Load and prepare inputs
    # =========================================================================
    revenue_df = revenue_plan.polars()
    inventory_df = available_inventory.polars()

    print(f"Revenue plan rows: {revenue_df.height:,}")
    print(f"Available inventory rows: {inventory_df.height:,}")

    # =========================================================================
    # Prepare Revenue Plan - use grouping_model when available, else model_id
    # =========================================================================
    revenue_clean = (
        revenue_df.filter(
            pl.col("revenue_plan_id").is_not_null()
            & pl.col("model_id").is_not_null()
            & pl.col("plan_month").is_not_null()
            & pl.col("quantity_ea").is_not_null()
        )
        .with_columns(
            [
                # Update grouping_model: use existing value if available, else model_id
                pl.when(pl.col("grouping_model").is_not_null() & (pl.col("grouping_model") != ""))
                .then(pl.col("grouping_model"))
                .otherwise(pl.col("model_id"))
                .alias("grouping_model")
            ]
        )
        .group_by(["revenue_plan_id", "grouping_model", "plan_month"])
        .agg(
            [
                pl.col("quantity_ea").sum().alias("planned_demand_ea"),
                pl.col("amount_krw").sum().alias("revplan_amount_in_krw"),
                # Keep track of original model_id for reference
                pl.col("model_id").first().alias("model_id"),
                # Keep track of sales_team
                pl.col("sales_team").first().alias("sales_team"),
            ]
        )
    )

    print(f"Revenue plan: {revenue_clean.height:,} rows across all plans")
    unique_plans = revenue_clean["revenue_plan_id"].n_unique()
    unique_groupings = revenue_clean["grouping_model"].n_unique()
    print(f"Unique revenue plans: {unique_plans}")
    print(f"Unique model groupings: {unique_groupings}")

    # =========================================================================
    # Prepare Available Inventory - use grouping_model when available, else model_id
    # Calculate transit as: total - shipped - onhand
    # =========================================================================
    inventory_clean = (
        inventory_df.filter(pl.col("model_id").is_not_null() & pl.col("plan_month").is_not_null())
        .with_columns(
            [
                # Update grouping_model: use existing value if available, else model_id
                pl.when(pl.col("grouping_model").is_not_null() & (pl.col("grouping_model") != ""))
                .then(pl.col("grouping_model"))
                .otherwise(pl.col("model_id"))
                .alias("grouping_model"),
                # Fill nulls before calculation
                pl.col("shipped_quantity_ea").fill_null(0).alias("shipped_quantity_ea"),
                pl.col("onhand_quantity_ea").fill_null(0).alias("onhand_quantity_ea"),
                pl.col("total_inventory_ea").fill_null(0).alias("total_inventory_ea"),
                pl.col("total_inventory_sht").fill_null(0).alias("total_inventory_sht"),
            ]
        )
        .with_columns(
            [
                # Calculate transit inventory
                (pl.col("total_inventory_ea") - pl.col("shipped_quantity_ea") - pl.col("onhand_quantity_ea")).alias(
                    "transit_quantity_ea"
                )
            ]
        )
        .group_by(["grouping_model", "plan_month"])
        .agg(
            [
                pl.col("shipped_quantity_ea").sum().alias("shipped_inventory_this_month_ea"),
                pl.col("onhand_quantity_ea").sum().alias("onhand_inventory_this_month_ea"),
                pl.col("transit_quantity_ea").sum().alias("transit_inventory_this_month_ea"),
                pl.col("total_inventory_sht").sum().alias("total_inventory_this_month_sht"),
            ]
        )
    )

    print(f"Available inventory: {inventory_clean.height:,} unique grouping_model-month combinations")

    # =========================================================================
    # Calculate total shipped inventory across ALL models per month
    # This aggregates shipped_quantity_ea by plan_month only (not by grouping_model)
    # =========================================================================
    total_shipped_by_month = (
        inventory_df.filter(pl.col("plan_month").is_not_null())
        .with_columns([pl.col("shipped_quantity_ea").fill_null(0).alias("shipped_quantity_ea")])
        .group_by("plan_month")
        .agg([pl.col("shipped_quantity_ea").sum().alias("total_shipped_all_models_ea")])
    )

    print(f"Total shipped by month: {total_shipped_by_month.height:,} months")

    # =========================================================================
    # Process each revenue plan separately
    # =========================================================================
    all_results = []

    for plan_id in revenue_clean["revenue_plan_id"].unique().sort():
        print(f"\nProcessing revenue plan: {plan_id}")

        # Filter to this plan
        plan_data = revenue_clean.filter(pl.col("revenue_plan_id") == plan_id)

        # Join with inventory on grouping_model (left join - only planned items)
        plan_with_inventory = plan_data.join(
            inventory_clean,
            on=["grouping_model", "plan_month"],
            how="left",
        )

        # Join with total shipped by month (left join on plan_month)
        plan_with_inventory = plan_with_inventory.join(
            total_shipped_by_month,
            on="plan_month",
            how="left",
        )

        # Fill nulls with 0
        plan_with_inventory = plan_with_inventory.with_columns(
            [
                pl.col("shipped_inventory_this_month_ea").fill_null(0),
                pl.col("onhand_inventory_this_month_ea").fill_null(0),
                pl.col("transit_inventory_this_month_ea").fill_null(0),
                pl.col("total_inventory_this_month_sht").fill_null(0),
                pl.col("total_shipped_all_models_ea").fill_null(0),
            ]
        )

        # Sort by grouping_model and month for chronological processing
        plan_with_inventory = plan_with_inventory.sort(["grouping_model", "plan_month"])

        print(f"  Grouping_model-month combinations: {plan_with_inventory.height:,}")

        # =====================================================================
        # Apply Forward-Rolling Inventory Logic for this plan
        # =====================================================================
        rows = plan_with_inventory.to_dicts()

        # Track rolled forward inventory per grouping_model (within this plan)
        rolled_forward_by_grouping = {}

        for row in rows:
            revenue_plan_id = plan_id
            grouping_model = row["grouping_model"]
            model_id = row["model_id"]
            sales_team = row["sales_team"]
            plan_month = row["plan_month"]
            planned_demand = row["planned_demand_ea"]
            revplan_amount_krw = row["revplan_amount_in_krw"]
            shipped_inventory = row["shipped_inventory_this_month_ea"]
            onhand_inventory = row["onhand_inventory_this_month_ea"]
            transit_inventory = row["transit_inventory_this_month_ea"]
            total_inventory_sht = row["total_inventory_this_month_sht"]
            total_shipped_all_models = row["total_shipped_all_models_ea"]

            # Determine revenue category
            revenue_category = "기타매출" if model_id == "기타매출" else "제품"

            # Get inventory rolled forward from previous month
            inventory_from_previous = rolled_forward_by_grouping.get(grouping_model, 0)

            # Total inventory available = shipped + onhand + transit + rolled forward
            total_inventory_available = (
                shipped_inventory + onhand_inventory + transit_inventory + inventory_from_previous
            )

            # =====================================================================
            # Calculate consumption with priority: transit, shipped, onhand, rolled
            # =====================================================================
            remaining_demand = planned_demand

            # Consume transit first
            transit_consumed = min(transit_inventory, remaining_demand)
            remaining_demand -= transit_consumed

            # Then consume shipped
            shipped_consumed = min(shipped_inventory, remaining_demand)
            remaining_demand -= shipped_consumed

            # Then consume onhand
            onhand_consumed = min(onhand_inventory, remaining_demand)
            remaining_demand -= onhand_consumed

            # Then consume rolled forward from previous month
            rolled_consumed = min(inventory_from_previous, remaining_demand)
            remaining_demand -= rolled_consumed

            # Total consumed
            inventory_consumed = shipped_consumed + onhand_consumed + transit_consumed + rolled_consumed

            # Net production = remaining demand after all inventory consumed
            net_production = remaining_demand

            # Calculate inventory to roll forward
            # Remaining inventory = total available - total consumed
            inventory_rolled_to_next = total_inventory_available - inventory_consumed

            # Override net production for "기타매출" - it comes from other sources
            if model_id == "기타매출":
                net_production = 0

            # Update rolled forward for next month of this grouping_model
            rolled_forward_by_grouping[grouping_model] = inventory_rolled_to_next

            # Calculate fulfillment percentage
            fulfillment_pct = (inventory_consumed / planned_demand * 100.0) if planned_demand > 0 else 0.0

            # Calculate KRW values based on unit price
            unit_price_krw = (revplan_amount_krw / planned_demand) if planned_demand > 0 else 0
            shipped_inventory_this_month_krw = shipped_inventory * unit_price_krw
            total_inventory_this_month_krw = (shipped_inventory + onhand_inventory + transit_inventory) * unit_price_krw
            inventory_consumed_amount_krw = inventory_consumed * unit_price_krw
            shipped_consumed_amount_krw = shipped_consumed * unit_price_krw
            onhand_consumed_amount_krw = onhand_consumed * unit_price_krw
            transit_consumed_amount_krw = transit_consumed * unit_price_krw
            net_production_amount_krw = net_production * unit_price_krw

            # Calculate SHT values for consumed inventory
            # Proportionally allocate total_inventory_sht based on consumption
            total_inventory_available_for_sht_calc = shipped_inventory + onhand_inventory + transit_inventory
            if total_inventory_available_for_sht_calc > 0:
                shipped_consumed_sht = (shipped_consumed / total_inventory_available_for_sht_calc) * total_inventory_sht
                onhand_consumed_sht = (onhand_consumed / total_inventory_available_for_sht_calc) * total_inventory_sht
                transit_consumed_sht = (transit_consumed / total_inventory_available_for_sht_calc) * total_inventory_sht
                inventory_consumed_sht = shipped_consumed_sht + onhand_consumed_sht + transit_consumed_sht
            else:
                shipped_consumed_sht = 0.0
                onhand_consumed_sht = 0.0
                transit_consumed_sht = 0.0
                inventory_consumed_sht = 0.0

            # Append result
            all_results.append(
                {
                    "revenue_plan_id": revenue_plan_id,
                    "grouping_model": grouping_model,
                    "model_id": model_id,
                    "revenue_category": revenue_category,
                    "sales_team": sales_team,
                    "plan_month": plan_month,
                    "planned_demand_ea": planned_demand,
                    "revplan_amount_in_krw": revplan_amount_krw,
                    "shipped_inventory_this_month_ea": shipped_inventory,
                    "shipped_inventory_this_month_krw": shipped_inventory_this_month_krw,
                    "onhand_inventory_this_month_ea": onhand_inventory,
                    "transit_inventory_this_month_ea": transit_inventory,
                    "total_inventory_this_month_sht": total_inventory_sht,
                    "total_inventory_this_month_krw": total_inventory_this_month_krw,
                    "total_shipped_all_models_ea": total_shipped_all_models,
                    "inventory_from_previous_month_ea": inventory_from_previous,
                    "total_inventory_available_ea": total_inventory_available,
                    "shipped_consumed_ea": shipped_consumed,
                    "shipped_consumed_sht": shipped_consumed_sht,
                    "shipped_consumed_amount_krw": shipped_consumed_amount_krw,
                    "onhand_consumed_ea": onhand_consumed,
                    "onhand_consumed_sht": onhand_consumed_sht,
                    "onhand_consumed_amount_krw": onhand_consumed_amount_krw,
                    "transit_consumed_ea": transit_consumed,
                    "transit_consumed_sht": transit_consumed_sht,
                    "transit_consumed_amount_krw": transit_consumed_amount_krw,
                    "inventory_consumed_ea": inventory_consumed,
                    "inventory_consumed_sht": inventory_consumed_sht,
                    "inventory_consumed_amount_krw": inventory_consumed_amount_krw,
                    "inventory_rolled_to_next_month_ea": inventory_rolled_to_next,
                    "net_production_demand_ea": net_production,
                    "net_production_amount_krw": net_production_amount_krw,
                    "fulfillment_pct": fulfillment_pct,
                }
            )

        # Print summary for this plan
        plan_total_demand = sum(r["planned_demand_ea"] for r in all_results if r["revenue_plan_id"] == plan_id)
        plan_total_consumed = sum(r["inventory_consumed_ea"] for r in all_results if r["revenue_plan_id"] == plan_id)
        plan_total_shipped = sum(r["shipped_consumed_ea"] for r in all_results if r["revenue_plan_id"] == plan_id)
        plan_total_onhand = sum(r["onhand_consumed_ea"] for r in all_results if r["revenue_plan_id"] == plan_id)
        plan_total_transit = sum(r["transit_consumed_ea"] for r in all_results if r["revenue_plan_id"] == plan_id)
        plan_total_net = sum(r["net_production_demand_ea"] for r in all_results if r["revenue_plan_id"] == plan_id)
        plan_total_other_revenue = sum(
            r["planned_demand_ea"]
            for r in all_results
            if r["revenue_plan_id"] == plan_id and r["revenue_category"] == "기타매출"
        )
        print(f"  Total demand: {plan_total_demand:,}")
        print(f"  Total demand (기타매출): {plan_total_other_revenue:,}")
        print(f"  Inventory consumed: {plan_total_consumed:,}")
        print(f"    - Shipped: {plan_total_shipped:,}")
        print(f"    - Onhand: {plan_total_onhand:,}")
        print(f"    - Transit: {plan_total_transit:,}")
        print(f"  Net production needed: {plan_total_net:,}")

    # =========================================================================
    # Convert to DataFrame and finalize
    # =========================================================================
    # Explicitly define schema to avoid type inference issues
    result_df = pl.DataFrame(
        all_results,
        schema={
            "revenue_plan_id": pl.Utf8,
            "grouping_model": pl.Utf8,
            "model_id": pl.Utf8,
            "revenue_category": pl.Utf8,
            "sales_team": pl.Utf8,
            "plan_month": pl.Utf8,
            "planned_demand_ea": pl.Int64,
            "revplan_amount_in_krw": pl.Int64,
            "shipped_inventory_this_month_ea": pl.Int64,
            "shipped_inventory_this_month_krw": pl.Float64,
            "onhand_inventory_this_month_ea": pl.Int64,
            "transit_inventory_this_month_ea": pl.Int64,
            "total_inventory_this_month_sht": pl.Float64,
            "total_inventory_this_month_krw": pl.Float64,
            "total_shipped_all_models_ea": pl.Int64,
            "inventory_from_previous_month_ea": pl.Int64,
            "total_inventory_available_ea": pl.Int64,
            "shipped_consumed_ea": pl.Int64,
            "shipped_consumed_sht": pl.Float64,
            "shipped_consumed_amount_krw": pl.Float64,
            "onhand_consumed_ea": pl.Int64,
            "onhand_consumed_sht": pl.Float64,
            "onhand_consumed_amount_krw": pl.Float64,
            "transit_consumed_ea": pl.Int64,
            "transit_consumed_sht": pl.Float64,
            "transit_consumed_amount_krw": pl.Float64,
            "inventory_consumed_ea": pl.Int64,
            "inventory_consumed_sht": pl.Float64,
            "inventory_consumed_amount_krw": pl.Float64,
            "inventory_rolled_to_next_month_ea": pl.Int64,
            "net_production_demand_ea": pl.Int64,
            "net_production_amount_krw": pl.Float64,
            "fulfillment_pct": pl.Float64,
        },
    )

    # Add calculation timestamp and net_production_plan_id
    result_df = result_df.with_columns(
        [
            pl.lit(datetime.utcnow()).alias("calculation_timestamp"),
            pl.concat_str(
                [
                    pl.col("plan_month"),
                    pl.col("revenue_plan_id"),
                    pl.col("grouping_model"),
                ],
                separator="_",
            ).alias("net_production_plan_id"),
        ]
    )

    # =========================================================================
    # Print Overall Summary Statistics
    # =========================================================================
    total_planned = result_df["planned_demand_ea"].sum()
    total_shipped_fresh = result_df["shipped_inventory_this_month_ea"].sum()
    total_onhand_fresh = result_df["onhand_inventory_this_month_ea"].sum()
    total_transit_fresh = result_df["transit_inventory_this_month_ea"].sum()
    total_shipped_consumed = result_df["shipped_consumed_ea"].sum()
    total_onhand_consumed = result_df["onhand_consumed_ea"].sum()
    total_transit_consumed = result_df["transit_consumed_ea"].sum()
    total_consumed = result_df["inventory_consumed_ea"].sum()
    total_net_production = result_df["net_production_demand_ea"].sum()
    total_other_revenue = result_df.filter(pl.col("revenue_category") == "기타매출")["planned_demand_ea"].sum()

    print(f"\n=== Overall Net Production Demand Summary (with Forward Rolling) ===")
    print(f"Total rows (all plans): {result_df.height:,}")
    print(f"Unique revenue plans: {result_df['revenue_plan_id'].n_unique()}")
    print(f"Unique model groupings: {result_df['grouping_model'].n_unique()}")
    print(f"Total planned demand (all plans): {total_planned:,} units")
    print(f"Total planned demand (기타매출): {total_other_revenue:,} units")
    print(f"Total fresh inventory:")
    print(f"  - Shipped: {total_shipped_fresh:,} units")
    print(f"  - Onhand: {total_onhand_fresh:,} units")
    print(f"  - Transit: {total_transit_fresh:,} units")
    print(f"Total inventory consumed: {total_consumed:,} units")
    print(f"  - Shipped: {total_shipped_consumed:,} units")
    print(f"  - Onhand: {total_onhand_consumed:,} units")
    print(f"  - Transit: {total_transit_consumed:,} units")
    print(f"Total net production demand: {total_net_production:,} units")

    # Show summary by plan
    print(f"\n=== Summary by Revenue Plan ===")
    plan_summary = (
        result_df.group_by("revenue_plan_id")
        .agg(
            [
                pl.col("planned_demand_ea").sum().alias("total_planned"),
                pl.col("inventory_consumed_ea").sum().alias("total_consumed"),
                pl.col("shipped_consumed_ea").sum().alias("total_shipped"),
                pl.col("onhand_consumed_ea").sum().alias("total_onhand"),
                pl.col("transit_consumed_ea").sum().alias("total_transit"),
                pl.col("net_production_demand_ea").sum().alias("total_net_production"),
                pl.count().alias("row_count"),
            ]
        )
        .sort("revenue_plan_id")
    )

    for row in plan_summary.iter_rows(named=True):
        plan_id = row["revenue_plan_id"]
        planned = row["total_planned"]
        consumed = row["total_consumed"]
        shipped = row["total_shipped"]
        onhand = row["total_onhand"]
        transit = row["total_transit"]
        net_prod = row["total_net_production"]
        count = row["row_count"]
        fulfillment = (consumed / planned * 100) if planned > 0 else 0
        print(
            f"  {plan_id}: {count:,} rows | Demand: {planned:,} | "
            f"Inventory: {consumed:,} ({fulfillment:.1f}%) [Shipped: {shipped:,}, Onhand: {onhand:,}, Transit: {transit:,}] | "
            f"Net Prod: {net_prod:,}"
        )

    # =========================================================================
    # Write Output
    # =========================================================================
    output.write_table(result_df)

    print(f"\n✓ Output written: {result_df.height:,} rows")
