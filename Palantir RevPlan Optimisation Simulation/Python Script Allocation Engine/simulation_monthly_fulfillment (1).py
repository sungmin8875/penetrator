"""
Simulation Monthly Fulfillment Datasets

Creates two datasets for Workshop visualization:
1. simulation_monthly_fulfillment_wide - Wide format with all metrics as columns
2. simulation_monthly_fulfillment_long - Long/pivoted format with 3-level category hierarchy

Grain: simulation_id + revenue_plan_id + model_id + month

Category Hierarchy (Long Format):
- Level 1: demand | fulfilled | shortfall | adjustment
- Level 2: null | inventory | produced | null | simulation_failure
- Level 3: null | null | wip | new_lots | null

Includes:
- Planned demand from net revenue plan
- Inventory consumed
- WIP produced (existing lots)
- New lots produced
- Impossible to simulate (NO_VALID_EQUIPMENT failures + missing routing models)
- True production shortfall (after accounting for impossible_to_simulate)
"""

import polars as pl
from transforms.api import transform, Input, Output, LightweightInput, LightweightOutput


# Input dataset RIDs
NET_REVENUE_PLAN_RID = "ri.foundry.main.dataset.122a25b5-165e-4eba-adfa-e7865add43d1"
REVENUE_OPTIMIZATION_EVENTS_RID = "ri.foundry.main.dataset.b84b2bb5-1cbf-4970-8f73-0e1eb9d4148f"
MODEL_MASTER_RID = "ri.foundry.main.dataset.2573f6cb-7e22-499e-b7e1-be7d895b1d1b"
MODEL_UNIT_CONVERSION_RID = "ri.foundry.main.dataset.b806e6e0-24d7-4241-9568-4d92700bc7ef"
MODEL_PRIORITIES_RID = "ri.foundry.main.dataset.19b16719-7cad-410d-b77d-cb03409dad14"
DEMAND_SHORTFALL_ANALYSIS_RID = "ri.foundry.main.dataset.a715dabf-cf86-40be-83b0-4dbbf44aa169"
NEW_MODEL_DEMAND_RISK_RID = "ri.foundry.main.dataset.704c37b3-e21b-42be-8cf6-0f0842ba8d54"

# Output paths
OUTPUT_PATH_WIDE = (
    "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/simulation_monthly_fulfillment_wide"
)
OUTPUT_PATH_LONG = (
    "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/simulation_monthly_fulfillment_long"
)


@transform.using(
    output_wide=Output(OUTPUT_PATH_WIDE),
    output_long=Output(OUTPUT_PATH_LONG),
    net_revenue_plan=Input(NET_REVENUE_PLAN_RID),
    revenue_optimization_events=Input(REVENUE_OPTIMIZATION_EVENTS_RID),
    model_master=Input(MODEL_MASTER_RID),
    model_unit_conversion=Input(MODEL_UNIT_CONVERSION_RID),
    model_priorities=Input(MODEL_PRIORITIES_RID),
    demand_shortfall_analysis=Input(DEMAND_SHORTFALL_ANALYSIS_RID),
    new_model_demand_risk=Input(NEW_MODEL_DEMAND_RISK_RID),
)
def compute(
    output_wide: LightweightOutput,
    output_long: LightweightOutput,
    net_revenue_plan: LightweightInput,
    revenue_optimization_events: LightweightInput,
    model_master: LightweightInput,
    model_unit_conversion: LightweightInput,
    model_priorities: LightweightInput,
    demand_shortfall_analysis: LightweightInput,
    new_model_demand_risk: LightweightInput,
):
    # Load all inputs as Polars DataFrames
    df_net_rev_plan = net_revenue_plan.polars()
    df_events = revenue_optimization_events.polars()
    df_model = model_master.polars()
    df_unit_conv = model_unit_conversion.polars()
    df_model_priorities = model_priorities.polars()
    df_demand_shortfall = demand_shortfall_analysis.polars()
    df_new_model_risk = new_model_demand_risk.polars()

    # =========================================================================
    # Step 1: Get distinct simulations from events (to filter to current rev plan)
    # =========================================================================
    # Fix: Ensure unique simulation_id + revenue_plan_id combinations
    df_simulations = (
        df_events.select(["simulation_id", "simulation_name", "revenue_plan_id"])
        .group_by(["simulation_id", "revenue_plan_id"])
        .agg(pl.col("simulation_name").first())  # Take first simulation_name if duplicates exist
    )

    active_revenue_plan_ids = df_simulations.select("revenue_plan_id").unique()
    # =========================================================================
    # Step 2: Filter net revenue plan to only active revenue plans
    # =========================================================================
    df_net_rev_plan_filtered = df_net_rev_plan.join(active_revenue_plan_ids, on="revenue_plan_id", how="inner")
    # =========================================================================
    # Step 3: Prepare model master for customer dimension enrichment
    # Use model_id as the only join key (grouping_model is inconsistent across sources)
    # =========================================================================
    df_model_dims_by_model = df_model.select(
        [
            pl.col("model_id"),
            pl.col("grouping_model").alias("grouping_model_master"),
            pl.col("end_customer").alias("model_end_customer_master"),
            pl.col("customer_name").alias("model_customer_name_master"),
            pl.col("sales_team").alias("model_sales_team_master"),
        ]
    ).unique(subset=["model_id"])
    # =========================================================================
    # Step 4: Prepare unit conversion for sheets calculation
    # =========================================================================
    df_unit_conv_clean = df_unit_conv.select(["model_id", "units_per_sheet"]).unique(subset=["model_id"])
    # =========================================================================
    # Step 5: Prepare model priorities for margin per unit
    # =========================================================================
    # Fix: Specify unique subset to prevent duplicates
    df_margin_per_unit = (
        df_model_priorities.filter(pl.col("__is_deleted") == False)
        .select(
            [
                "model_id",
                "month",
                "revenue_plan_id",
                "simulation_id",
                "margin_amount_per_unit",
                "amount_per_unit",
            ]
        )
        .group_by(["model_id", "month", "revenue_plan_id", "simulation_id"])
        .agg(
            [
                pl.col("margin_amount_per_unit").first(),
                pl.col("amount_per_unit").first(),
            ]
        )
    )

    # =========================================================================
    # Step 6: Prepare impossible_to_simulate data
    # =========================================================================
    # Source 1: NO_VALID_EQUIPMENT failures
    df_no_equipment = (
        df_demand_shortfall.filter(pl.col("primary_reason") == "NO_VALID_EQUIPMENT")
        .select(
            [
                pl.col("simulation_id"),
                pl.col("revenue_plan_id"),
                pl.col("model_id"),
                pl.col("target_month").alias("month_str"),
                pl.col("failed_units").alias("impossible_to_simulate_ea"),
                pl.col("shortfall_margin_impact").alias("impossible_to_simulate_margin_krw"),
            ]
        )
        .filter(pl.col("impossible_to_simulate_ea") > 0)
    )

    # Source 2: new_model_demand_risk (missing routing)
    df_missing_routing = df_new_model_risk.select(
        [
            pl.col("simulation_id"),
            pl.col("model_id"),
            pl.col("shortfall_month").alias("month_str"),
            pl.col("demand_qty_ea").alias("impossible_to_simulate_ea"),
        ]
    )

    df_missing_routing = (
        df_missing_routing.join(
            df_margin_per_unit.rename({"month": "month_str"}),
            on=["simulation_id", "model_id", "month_str"],
            how="left",
        )
        .with_columns(
            [
                (pl.col("impossible_to_simulate_ea") * pl.col("margin_amount_per_unit").fill_null(0)).alias(
                    "impossible_to_simulate_margin_krw"
                ),
            ]
        )
        .select(
            [
                "simulation_id",
                "revenue_plan_id",
                "model_id",
                "month_str",
                "impossible_to_simulate_ea",
                "impossible_to_simulate_margin_krw",
            ]
        )
    )

    df_impossible = pl.concat([df_no_equipment, df_missing_routing], how="vertical_relaxed")

    # Aggregate impossible_to_simulate by simulation/model/month (no grouping_model)
    df_impossible_agg = df_impossible.group_by(
        [
            "simulation_id",
            "revenue_plan_id",
            "model_id",
            "month_str",
        ]
    ).agg(
        [
            pl.col("impossible_to_simulate_ea").sum(),
            pl.col("impossible_to_simulate_margin_krw").sum(),
        ]
    )

    # =========================================================================
    # Step 7: Aggregate production events by simulation/model/month
    # NOTE: We aggregate WITHOUT grouping_model to avoid mismatches between sources
    # =========================================================================
    df_events_completed = df_events.filter(pl.col("remaining_steps") == 0)

    df_wip_produced = (
        df_events_completed.filter(pl.col("is_new_lot") == False)
        .group_by(
            [
                "simulation_id",
                "simulation_name",
                "revenue_plan_id",
                "model_id",
                "actual_completion_month",
            ]
        )
        .agg(
            [
                pl.col("units_produced").sum().alias("wip_produced_ea"),
                pl.col("total_margin").sum().alias("wip_produced_margin_krw"),
            ]
        )
    )

    df_new_lots_produced = (
        df_events_completed.filter(pl.col("is_new_lot") == True)
        .group_by(
            [
                "simulation_id",
                "simulation_name",
                "revenue_plan_id",
                "model_id",
                "actual_completion_month",
            ]
        )
        .agg(
            [
                pl.col("units_produced").sum().alias("new_lots_produced_ea"),
                pl.col("total_margin").sum().alias("new_lots_produced_margin_krw"),
            ]
        )
    )

    df_production = df_wip_produced.join(
        df_new_lots_produced,
        on=[
            "simulation_id",
            "simulation_name",
            "revenue_plan_id",
            "model_id",
            "actual_completion_month",
        ],
        how="outer_coalesce",
    ).with_columns(
        [
            pl.col("wip_produced_ea").fill_null(0),
            pl.col("wip_produced_margin_krw").fill_null(0),
            pl.col("new_lots_produced_ea").fill_null(0),
            pl.col("new_lots_produced_margin_krw").fill_null(0),
        ]
    )

    # =========================================================================
    # Step 8: Prepare net revenue plan data
    # =========================================================================
    df_plan_data = df_net_rev_plan_filtered.select(
        [
            pl.col("revenue_plan_id"),
            pl.col("grouping_model"),  # Keep for output, but not for joins
            pl.col("model_id"),
            pl.col("plan_month").alias("month_str"),
            pl.col("planned_demand_ea"),
            pl.col("inventory_consumed_ea"),
            pl.col("inventory_consumed_amount_krw"),
            pl.col("net_production_demand_ea"),
            pl.col("net_production_amount_krw").alias("net_production_demand_krw"),
        ]
    )

    # =========================================================================
    # Step 9: Cross join plan data with simulations
    # =========================================================================
    # Fix: Ensure no duplicate combinations before join
    df_plan_data_unique = df_plan_data.unique(subset=["revenue_plan_id", "model_id", "month_str"])
    df_simulations_unique = df_simulations.unique(subset=["simulation_id", "revenue_plan_id"])

    df_plan_with_simulations = df_plan_data_unique.join(df_simulations_unique, on="revenue_plan_id", how="inner")

    # =========================================================================
    # Step 10: Join production data with plan data (using OUTER join to capture unplanned production)
    # NOTE: Join WITHOUT grouping_model to avoid mismatches
    # =========================================================================
    df_combined = df_plan_with_simulations.join(
        df_production.rename({"actual_completion_month": "month_str"}),
        on=[
            "simulation_id",
            "simulation_name",
            "revenue_plan_id",
            "model_id",
            "month_str",
        ],
        how="outer_coalesce",
    ).with_columns(
        [
            pl.col("planned_demand_ea").fill_null(0),
            pl.col("inventory_consumed_ea").fill_null(0),
            pl.col("inventory_consumed_amount_krw").fill_null(0.0),
            pl.col("net_production_demand_ea").fill_null(0),
            pl.col("net_production_demand_krw").fill_null(0),
            pl.col("wip_produced_ea").fill_null(0),
            pl.col("wip_produced_margin_krw").fill_null(0),
            pl.col("new_lots_produced_ea").fill_null(0),
            pl.col("new_lots_produced_margin_krw").fill_null(0),
        ]
    )

    # =========================================================================
    # Step 11: Join with impossible_to_simulate data
    # =========================================================================
    df_combined = df_combined.join(
        df_impossible_agg,
        on=["simulation_id", "revenue_plan_id", "model_id", "month_str"],
        how="left",
    ).with_columns(
        [
            pl.col("impossible_to_simulate_ea").fill_null(0),
            pl.col("impossible_to_simulate_margin_krw").fill_null(0),
        ]
    )

    # =========================================================================
    # Step 12: Enrich with customer dimensions from model master
    # =========================================================================
    df_combined = (
        df_combined.join(
            df_model_dims_by_model,
            on=["model_id"],
            how="left",
        )
        .with_columns(
            [
                # Use grouping_model from plan if available, otherwise from master
                pl.coalesce([pl.col("grouping_model"), pl.col("grouping_model_master")]).alias("grouping_model"),
                pl.col("model_end_customer_master").alias("model_end_customer"),
                pl.col("model_customer_name_master").alias("model_customer_name"),
                pl.col("model_sales_team_master").alias("model_sales_team"),
            ]
        )
        .drop(
            [
                "grouping_model_master",
                "model_end_customer_master",
                "model_customer_name_master",
                "model_sales_team_master",
            ]
        )
    )

    # =========================================================================
    # Step 13: Join with margin per unit for planned margin calculations
    # =========================================================================
    df_combined = df_combined.join(
        df_margin_per_unit.rename({"month": "month_str"}),
        on=["model_id", "month_str", "revenue_plan_id", "simulation_id"],
        how="left",
    )

    # =========================================================================
    # Step 14: Join with unit conversion for sheets calculation
    # =========================================================================
    df_combined = df_combined.join(df_unit_conv_clean, on="model_id", how="left")

    # =========================================================================
    # Step 15: Calculate all metrics for WIDE format
    # =========================================================================
    df_wide = (
        df_combined.with_columns(
            [
                # Create month_date from month_str (YYYYMM -> date)
                pl.concat_str(
                    [
                        pl.col("month_str").str.slice(0, 4),
                        pl.lit("-"),
                        pl.col("month_str").str.slice(4, 2),
                        pl.lit("-01"),
                    ]
                )
                .str.to_date("%Y-%m-%d")
                .alias("month_date"),
                # Planned demand margin
                (pl.col("planned_demand_ea") * pl.col("margin_amount_per_unit")).alias("planned_demand_margin_krw"),
                # Planned demand KRW
                (pl.col("planned_demand_ea") * pl.col("amount_per_unit")).alias("planned_demand_krw"),
                # Net production demand margin
                (pl.col("net_production_demand_ea") * pl.col("margin_amount_per_unit")).alias(
                    "net_production_demand_margin_krw"
                ),
                # Inventory consumed margin (calculate from ea * margin_amount_per_unit)
                (pl.col("inventory_consumed_ea") * pl.col("margin_amount_per_unit")).alias(
                    "inventory_consumed_margin_krw"
                ),
                # Inventory consumed KRW - use source value from net_revenue_plan for accuracy
                pl.col("inventory_consumed_amount_krw").alias("inventory_consumed_krw"),
                # Total produced
                (pl.col("wip_produced_ea") + pl.col("new_lots_produced_ea")).alias("total_produced_ea"),
                (pl.col("wip_produced_margin_krw") + pl.col("new_lots_produced_margin_krw")).alias(
                    "total_produced_margin_krw"
                ),
            ]
        )
        .with_columns(
            [
                # WIP produced KRW
                (pl.col("wip_produced_ea") * pl.col("amount_per_unit")).alias("wip_produced_krw"),
                # New lots produced KRW
                (pl.col("new_lots_produced_ea") * pl.col("amount_per_unit")).alias("new_lots_produced_krw"),
                # Total produced KRW
                (pl.col("total_produced_ea") * pl.col("amount_per_unit")).alias("total_produced_krw"),
                # Impossible to simulate KRW
                (pl.col("impossible_to_simulate_ea") * pl.col("amount_per_unit")).alias("impossible_to_simulate_krw"),
                # Production and inventory combined
                (pl.col("total_produced_ea") + pl.col("inventory_consumed_ea")).alias("production_and_inventory_ea"),
                (
                    pl.col("total_produced_margin_krw").fill_null(0.0)
                    + pl.col("inventory_consumed_margin_krw").fill_null(0.0)
                ).alias("production_and_inventory_margin_krw"),
            ]
        )
        .with_columns(
            [
                # Production and inventory KRW (sum of total_produced_krw and inventory_consumed_krw)
                (pl.col("total_produced_krw").fill_null(0.0) + pl.col("inventory_consumed_krw").fill_null(0.0)).alias(
                    "production_and_inventory_krw"
                ),
            ]
        )
        .with_columns(
            [
                # Shortfall = (production_and_inventory + impossible_to_simulate) - planned
                # Negative shortfall means we are missing that amount (e.g., -5000 = shortfall of 5000)
                (
                    pl.col("production_and_inventory_ea")
                    + pl.col("impossible_to_simulate_ea")
                    - pl.col("planned_demand_ea")
                ).alias("shortfall_ea"),
                (
                    pl.col("production_and_inventory_krw")
                    + pl.col("impossible_to_simulate_krw")
                    - pl.col("planned_demand_krw")
                ).alias("shortfall_krw"),
                (
                    pl.col("production_and_inventory_margin_krw")
                    + pl.col("impossible_to_simulate_margin_krw")
                    - pl.col("planned_demand_margin_krw")
                ).alias("shortfall_margin_krw"),
                # Sheets calculations
                (pl.col("total_produced_ea").cast(pl.Float64) / pl.col("units_per_sheet")).alias("sheets_produced"),
                (pl.col("production_and_inventory_ea").cast(pl.Float64) / pl.col("units_per_sheet")).alias(
                    "sheets_production_and_inventory"
                ),
                (pl.col("impossible_to_simulate_ea").cast(pl.Float64) / pl.col("units_per_sheet")).alias(
                    "sheets_impossible_to_simulate"
                ),
            ]
        )
    )

    # Create primary key
    df_wide = df_wide.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("simulation_id"),
                    pl.lit("_"),
                    pl.col("revenue_plan_id"),
                    pl.lit("_"),
                    pl.col("model_id"),
                    pl.lit("_"),
                    pl.col("month_str"),
                ]
            ).alias("fulfillment_stat_id")
        ]
    )

    # Add deduplication check and aggregation for safety
    df_wide = df_wide.group_by("fulfillment_stat_id").agg(
        [
            # Take first value for all dimension columns
            pl.col("simulation_id").first(),
            pl.col("simulation_name").first(),
            pl.col("revenue_plan_id").first(),
            pl.col("month_date").first(),
            pl.col("month_str").first(),
            pl.col("model_id").first(),
            pl.col("grouping_model").first(),
            pl.col("model_end_customer").first(),
            pl.col("model_customer_name").first(),
            pl.col("model_sales_team").first(),
            # Sum metrics in case of duplicates
            pl.col("planned_demand_ea").sum(),
            pl.col("planned_demand_krw").sum(),
            pl.col("planned_demand_margin_krw").sum(),
            pl.col("net_production_demand_ea").sum(),
            pl.col("net_production_demand_krw").sum(),
            pl.col("net_production_demand_margin_krw").sum(),
            pl.col("inventory_consumed_ea").sum(),
            pl.col("inventory_consumed_krw").sum(),
            pl.col("inventory_consumed_margin_krw").sum(),
            pl.col("wip_produced_ea").sum(),
            pl.col("wip_produced_krw").sum(),
            pl.col("wip_produced_margin_krw").sum(),
            pl.col("new_lots_produced_ea").sum(),
            pl.col("new_lots_produced_krw").sum(),
            pl.col("new_lots_produced_margin_krw").sum(),
            pl.col("total_produced_ea").sum(),
            pl.col("total_produced_krw").sum(),
            pl.col("total_produced_margin_krw").sum(),
            pl.col("production_and_inventory_ea").sum(),
            pl.col("production_and_inventory_krw").sum(),
            pl.col("production_and_inventory_margin_krw").sum(),
            pl.col("impossible_to_simulate_ea").sum(),
            pl.col("impossible_to_simulate_krw").sum(),
            pl.col("impossible_to_simulate_margin_krw").sum(),
            pl.col("shortfall_ea").sum(),
            pl.col("shortfall_krw").sum(),
            pl.col("shortfall_margin_krw").sum(),
            pl.col("sheets_produced").sum(),
            pl.col("sheets_production_and_inventory").sum(),
            pl.col("sheets_impossible_to_simulate").sum(),
        ]
    )

    # Select and order final columns for wide output
    df_wide_final = df_wide.select(
        [
            # Primary key
            "fulfillment_stat_id",
            # Dimensions
            "simulation_id",
            "simulation_name",
            "revenue_plan_id",
            "month_date",
            "month_str",
            "model_id",
            "grouping_model",
            "model_end_customer",
            "model_customer_name",
            "model_sales_team",
            # Planned demand
            "planned_demand_ea",
            "planned_demand_krw",
            "planned_demand_margin_krw",
            # Net production demand
            "net_production_demand_ea",
            "net_production_demand_krw",
            "net_production_demand_margin_krw",
            # Inventory consumed
            "inventory_consumed_ea",
            "inventory_consumed_krw",
            "inventory_consumed_margin_krw",
            # WIP produced
            "wip_produced_ea",
            "wip_produced_krw",
            "wip_produced_margin_krw",
            # New lots produced
            "new_lots_produced_ea",
            "new_lots_produced_krw",
            "new_lots_produced_margin_krw",
            # Total produced
            "total_produced_ea",
            "total_produced_krw",
            "total_produced_margin_krw",
            # Production and inventory
            "production_and_inventory_ea",
            "production_and_inventory_krw",
            "production_and_inventory_margin_krw",
            # Impossible to simulate
            "impossible_to_simulate_ea",
            "impossible_to_simulate_krw",
            "impossible_to_simulate_margin_krw",
            # Shortfall (true production shortfall)
            "shortfall_ea",
            "shortfall_krw",
            "shortfall_margin_krw",
            # Sheets
            "sheets_produced",
            "sheets_production_and_inventory",
            "sheets_impossible_to_simulate",
        ]
    )

    # =========================================================================
    # Step 16: Create LONG format with 3-level category hierarchy
    # =========================================================================
    # Category definitions: (category_l1, category_l2, category_l3, ea_col, margin_col)
    categories = [
        # Demand
        ("demand", None, None, "planned_demand_ea", "planned_demand_margin_krw"),
        # Fulfilled > Inventory
        ("fulfilled", "inventory", None, "inventory_consumed_ea", "inventory_consumed_margin_krw"),
        # Fulfilled > Produced > WIP
        ("fulfilled", "produced", "wip", "wip_produced_ea", "wip_produced_margin_krw"),
        # Fulfilled > Produced > New Lots
        ("fulfilled", "produced", "new_lots", "new_lots_produced_ea", "new_lots_produced_margin_krw"),
        # Fulfilled > Produced > Impossible to Simulate (adjustment)
        (
            "fulfilled",
            "produced",
            "impossible_to_simulate",
            "impossible_to_simulate_ea",
            "impossible_to_simulate_margin_krw",
        ),
        # Shortfall (top-level metric)
        ("shortfall", None, None, "shortfall_ea", "shortfall_margin_krw"),
    ]

    dimension_cols = [
        "fulfillment_stat_id",
        "simulation_id",
        "simulation_name",
        "revenue_plan_id",
        "month_date",
        "month_str",
        "model_id",
        "grouping_model",
        "model_end_customer",
        "model_customer_name",
        "model_sales_team",
    ]

    # Re-join with units_per_sheet for long format
    df_wide_with_units = df_wide_final.join(df_unit_conv_clean, on="model_id", how="left")

    long_dfs = []
    for cat_l1, cat_l2, cat_l3, ea_col, margin_col in categories:
        df_cat = (
            df_wide_with_units.select(dimension_cols + ["units_per_sheet", ea_col, margin_col])
            .with_columns(
                [
                    pl.lit(cat_l1).alias("category_l1"),
                    pl.lit(cat_l2).alias("category_l2"),
                    pl.lit(cat_l3).alias("category_l3"),
                    pl.col(ea_col).cast(pl.Float64).alias("amount_ea"),
                    pl.col(margin_col).cast(pl.Float64).alias("amount_margin_krw"),
                ]
            )
            .with_columns(
                [
                    (pl.col("amount_ea") / pl.col("units_per_sheet").cast(pl.Float64)).alias("amount_sheets"),
                ]
            )
            .drop([ea_col, margin_col])
        )
        long_dfs.append(df_cat)

    df_long = pl.concat(long_dfs, how="vertical_relaxed")

    # Filter out rows where amount_ea is 0
    df_long = df_long.filter(pl.col("amount_ea") != 0)

    # Create primary key for long format
    df_long_final = df_long.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("fulfillment_stat_id"),
                    pl.lit("_"),
                    pl.coalesce([pl.col("category_l1"), pl.lit("null")]),
                    pl.lit("_"),
                    pl.coalesce([pl.col("category_l2"), pl.lit("null")]),
                    pl.lit("_"),
                    pl.coalesce([pl.col("category_l3"), pl.lit("null")]),
                ]
            ).alias("fulfillment_stat_category_id")
        ]
    ).select(
        [
            "fulfillment_stat_category_id",
            "simulation_id",
            "simulation_name",
            "revenue_plan_id",
            "month_date",
            "month_str",
            "model_id",
            "grouping_model",
            "model_end_customer",
            "model_customer_name",
            "model_sales_team",
            "category_l1",
            "category_l2",
            "category_l3",
            "amount_ea",
            "amount_margin_krw",
            "amount_sheets",
        ]
    )

    # Write outputs
    output_wide.write_table(df_wide_final)
    output_long.write_table(df_long_final)
