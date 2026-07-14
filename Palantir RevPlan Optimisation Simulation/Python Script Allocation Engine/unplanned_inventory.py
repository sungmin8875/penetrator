"""
Unplanned Inventory Calculator

This transform identifies inventory that exists but is NOT part of the revenue plan.
It captures inventory (shipped, onhand, transit) for models that have inventory
available but are not included in the revenue plan for a specific plan scenario.

Key Logic:
1. For a specific revenue plan (MP202601-05W-004), identify all inventory that exists
2. Compare against what is in the revenue plan
3. Output rows for inventory that has NO corresponding entry in the revenue plan
4. Calculate the KRW value using model master's sales_unit_price_krw

This is useful for:
- Identifying excess/surplus inventory that isn't allocated to any plan
- Understanding what inventory exists outside of planned demand
- Tracking shipped, onhand, and transit inventory for models not in the plan

Output:
- One row per model per month for inventory NOT in the revenue plan showing:
  - unplanned_inventory_id: Unique identifier (revenue_plan_id + grouping_model + plan_month)
  - revenue_plan_id: The reference plan (MP202601-05W-004)
  - grouping_model: The grouping used (grouping_model if available, else model_id)
  - model_id: The original model_id value
  - plan_month: The month
  - shipped_inventory_ea: Shipped inventory not in plan
  - onhand_inventory_ea: On-hand inventory not in plan
  - transit_inventory_ea: Transit inventory not in plan
  - total_inventory_ea: Total unplanned inventory
  - total_inventory_sht: Total unplanned inventory in sheets
  - unit_price_krw: Price per unit from model master
  - shipped_amount_krw: KRW value of shipped inventory
  - onhand_amount_krw: KRW value of on-hand inventory
  - transit_amount_krw: KRW value of transit inventory
  - total_amount_krw: Total KRW value of unplanned inventory
"""

import polars as pl
from transforms.api import transform, Input, Output, lightweight
from datetime import datetime


# Reference plan for which we track unplanned inventory
REFERENCE_PLAN_ID = "MP202601-05W-004"


@lightweight(cpu_cores=2, memory_gb=8)
@transform(
    output=Output("/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/unplanned_inventory"),
    revenue_plan=Input("ri.foundry.main.dataset.2e72556a-1f0b-4ecc-9bdb-3e172781a406"),
    available_inventory=Input("ri.foundry.main.dataset.a8f8675b-a7b2-47f5-b83f-97f7028cc1b5"),
    model_master=Input("ri.foundry.main.dataset.2573f6cb-7e22-499e-b7e1-be7d895b1d1b"),
)
def compute(revenue_plan, available_inventory, model_master, output) -> None:
    """
    Calculate unplanned inventory - inventory that exists but is not in the revenue plan.

    This transform:
    1. Gets the revenue plan for the reference plan (MP202601-05W-004)
    2. Gets all available inventory
    3. Identifies inventory NOT in the revenue plan
    4. Calculates KRW values using model master prices
    5. Outputs unplanned inventory details

    Output columns:
    - unplanned_inventory_id: Unique identifier (revenue_plan_id + grouping_model + plan_month)
    - revenue_plan_id: The reference plan ID
    - grouping_model: The effective grouping (grouping_model if available, else model_id)
    - model_id: Original model_id value
    - plan_month: The planning month
    - shipped_inventory_ea: Shipped inventory not in plan
    - onhand_inventory_ea: On-hand inventory not in plan
    - transit_inventory_ea: Transit inventory not in plan
    - total_inventory_ea: Total unplanned inventory (EA)
    - total_inventory_sht: Total unplanned inventory (sheets)
    - unit_price_krw: Price per unit from model master
    - shipped_amount_krw: KRW value of shipped inventory
    - onhand_amount_krw: KRW value of on-hand inventory
    - transit_amount_krw: KRW value of transit inventory
    - total_amount_krw: Total KRW value of unplanned inventory
    - calculation_timestamp: When this calculation was performed
    """

    # =========================================================================
    # Load and prepare inputs
    # =========================================================================
    revenue_df = revenue_plan.polars()
    inventory_df = available_inventory.polars()
    model_df = model_master.polars()

    print(f"Revenue plan rows: {revenue_df.height:,}")
    print(f"Available inventory rows: {inventory_df.height:,}")
    print(f"Model master rows: {model_df.height:,}")

    # =========================================================================
    # Get unit prices from model master
    # =========================================================================
    model_unit_prices = (
        model_df.filter(pl.col("model_id").is_not_null() & pl.col("sales_unit_price_krw").is_not_null())
        .select(["model_id", pl.col("sales_unit_price_krw").alias("unit_price_krw")])
        .unique(subset=["model_id"], keep="first")
    )

    print(f"Model master unit prices: {model_unit_prices.height:,} models with price data")

    # =========================================================================
    # Prepare Revenue Plan - filter to reference plan and get planned grouping_models
    # =========================================================================
    revenue_clean = (
        revenue_df.filter(
            (pl.col("revenue_plan_id") == REFERENCE_PLAN_ID)
            & pl.col("model_id").is_not_null()
            & pl.col("plan_month").is_not_null()
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
        .select(["grouping_model", "plan_month"])
        .unique()
    )

    print(f"Revenue plan ({REFERENCE_PLAN_ID}): {revenue_clean.height:,} unique grouping_model-month combinations")

    # Get the plan months from the revenue plan
    plan_months = revenue_clean["plan_month"].unique().to_list()
    if not plan_months:
        print("No plan months found in revenue plan, outputting empty dataset")
        output.write_table(
            pl.DataFrame(
                schema={
                    "unplanned_inventory_id": pl.Utf8,
                    "revenue_plan_id": pl.Utf8,
                    "grouping_model": pl.Utf8,
                    "model_id": pl.Utf8,
                    "plan_month": pl.Utf8,
                    "shipped_inventory_ea": pl.Int64,
                    "onhand_inventory_ea": pl.Int64,
                    "transit_inventory_ea": pl.Int64,
                    "total_inventory_ea": pl.Int64,
                    "total_inventory_sht": pl.Float64,
                    "unit_price_krw": pl.Float64,
                    "shipped_amount_krw": pl.Float64,
                    "onhand_amount_krw": pl.Float64,
                    "transit_amount_krw": pl.Float64,
                    "total_amount_krw": pl.Float64,
                    "calculation_timestamp": pl.Datetime,
                }
            )
        )
        return

    print(f"Plan months: {len(plan_months)} months from {min(plan_months)} to {max(plan_months)}")

    # =========================================================================
    # Prepare Available Inventory - use grouping_model when available, else model_id
    # Filter to only months relevant to the plan
    # =========================================================================
    inventory_clean = (
        inventory_df.filter(
            pl.col("model_id").is_not_null()
            & pl.col("plan_month").is_not_null()
            & pl.col("plan_month").is_in(plan_months)
        )
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
                pl.col("shipped_quantity_ea").sum().alias("shipped_inventory_ea"),
                pl.col("onhand_quantity_ea").sum().alias("onhand_inventory_ea"),
                pl.col("transit_quantity_ea").sum().alias("transit_inventory_ea"),
                pl.col("total_inventory_ea").sum().alias("total_inventory_ea"),
                pl.col("total_inventory_sht").sum().alias("total_inventory_sht"),
                # Keep track of original model_id from inventory for reference
                pl.col("model_id").first().alias("model_id"),
            ]
        )
    )

    print(
        f"Available inventory (filtered to plan months): {inventory_clean.height:,} unique grouping_model-month combinations"
    )

    # =========================================================================
    # Find inventory NOT in the revenue plan using anti-join
    # =========================================================================
    unplanned_inventory = inventory_clean.join(
        revenue_clean,
        on=["grouping_model", "plan_month"],
        how="anti",  # Only keep rows that DON'T match the revenue plan
    )

    print(f"Unplanned inventory: {unplanned_inventory.height:,} unique grouping_model-month combinations")

    if unplanned_inventory.height == 0:
        print("No unplanned inventory found, outputting empty dataset")
        output.write_table(
            pl.DataFrame(
                schema={
                    "unplanned_inventory_id": pl.Utf8,
                    "revenue_plan_id": pl.Utf8,
                    "grouping_model": pl.Utf8,
                    "model_id": pl.Utf8,
                    "plan_month": pl.Utf8,
                    "shipped_inventory_ea": pl.Int64,
                    "onhand_inventory_ea": pl.Int64,
                    "transit_inventory_ea": pl.Int64,
                    "total_inventory_ea": pl.Int64,
                    "total_inventory_sht": pl.Float64,
                    "unit_price_krw": pl.Float64,
                    "shipped_amount_krw": pl.Float64,
                    "onhand_amount_krw": pl.Float64,
                    "transit_amount_krw": pl.Float64,
                    "total_amount_krw": pl.Float64,
                    "calculation_timestamp": pl.Datetime,
                }
            )
        )
        return

    # =========================================================================
    # Add reference plan ID
    # =========================================================================
    unplanned_inventory = unplanned_inventory.with_columns(pl.lit(REFERENCE_PLAN_ID).alias("revenue_plan_id"))

    # =========================================================================
    # Join with model master to get unit prices
    # =========================================================================
    unplanned_inventory = unplanned_inventory.join(
        model_unit_prices,
        on="model_id",
        how="left",
    )

    # Fill null prices with 0
    unplanned_inventory = unplanned_inventory.with_columns(pl.col("unit_price_krw").fill_null(0.0))

    # =========================================================================
    # Calculate KRW amounts
    # =========================================================================
    unplanned_inventory = unplanned_inventory.with_columns(
        [
            (pl.col("shipped_inventory_ea") * pl.col("unit_price_krw")).alias("shipped_amount_krw"),
            (pl.col("onhand_inventory_ea") * pl.col("unit_price_krw")).alias("onhand_amount_krw"),
            (pl.col("transit_inventory_ea") * pl.col("unit_price_krw")).alias("transit_amount_krw"),
            (pl.col("total_inventory_ea") * pl.col("unit_price_krw")).alias("total_amount_krw"),
        ]
    )

    # =========================================================================
    # Add unique identifier and calculation timestamp
    # =========================================================================
    unplanned_inventory = unplanned_inventory.with_columns(
        [
            # Create unique ID by concatenating revenue_plan_id, grouping_model, and plan_month
            pl.concat_str(
                [
                    pl.col("revenue_plan_id"),
                    pl.lit("_"),
                    pl.col("grouping_model"),
                    pl.lit("_"),
                    pl.col("plan_month"),
                ]
            ).alias("unplanned_inventory_id"),
            pl.lit(datetime.utcnow()).alias("calculation_timestamp"),
        ]
    )

    # =========================================================================
    # Select and order final columns
    # =========================================================================
    result_df = unplanned_inventory.select(
        [
            "unplanned_inventory_id",
            "revenue_plan_id",
            "grouping_model",
            "model_id",
            "plan_month",
            "shipped_inventory_ea",
            "onhand_inventory_ea",
            "transit_inventory_ea",
            "total_inventory_ea",
            "total_inventory_sht",
            "unit_price_krw",
            "shipped_amount_krw",
            "onhand_amount_krw",
            "transit_amount_krw",
            "total_amount_krw",
            "calculation_timestamp",
        ]
    )

    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    total_shipped = result_df["shipped_inventory_ea"].sum()
    total_onhand = result_df["onhand_inventory_ea"].sum()
    total_transit = result_df["transit_inventory_ea"].sum()
    total_inventory = result_df["total_inventory_ea"].sum()
    total_amount = result_df["total_amount_krw"].sum()

    print(f"\n=== Unplanned Inventory Summary (for {REFERENCE_PLAN_ID}) ===")
    print(f"Total rows: {result_df.height:,}")
    print(f"Unique model groupings: {result_df['grouping_model'].n_unique()}")
    print(f"Total unplanned inventory:")
    print(f"  - Shipped: {total_shipped:,} units")
    print(f"  - On-hand: {total_onhand:,} units")
    print(f"  - Transit: {total_transit:,} units")
    print(f"  - Total: {total_inventory:,} units")
    print(f"  - Total value: {total_amount:,.0f} KRW")

    # Show summary by month
    print(f"\n=== Summary by Month ===")
    month_summary = (
        result_df.group_by("plan_month")
        .agg(
            [
                pl.col("shipped_inventory_ea").sum().alias("shipped"),
                pl.col("onhand_inventory_ea").sum().alias("onhand"),
                pl.col("transit_inventory_ea").sum().alias("transit"),
                pl.col("total_inventory_ea").sum().alias("total_ea"),
                pl.col("total_amount_krw").sum().alias("total_krw"),
                pl.count().alias("model_count"),
            ]
        )
        .sort("plan_month")
    )

    for row in month_summary.iter_rows(named=True):
        month = row["plan_month"]
        shipped = row["shipped"]
        onhand = row["onhand"]
        transit = row["transit"]
        total_ea = row["total_ea"]
        total_krw = row["total_krw"]
        count = row["model_count"]
        print(
            f"  {month}: {count:,} models | "
            f"Shipped: {shipped:,} | Onhand: {onhand:,} | Transit: {transit:,} | "
            f"Total: {total_ea:,} EA ({total_krw:,.0f} KRW)"
        )

    # =========================================================================
    # Write Output
    # =========================================================================
    output.write_table(result_df)

    print(f"\n✓ Output written: {result_df.height:,} rows")


"""
Unplanned Inventory Calculator

This transform identifies inventory that exists but is NOT part of the revenue plan.
It captures inventory (shipped, onhand, transit) for models that have inventory
available but are not included in the revenue plan for a specific plan scenario.

Key Logic:
1. For a specific revenue plan (MP202601-05W-004), identify all inventory that exists
2. Compare against what is in the revenue plan
3. Output rows for inventory that has NO corresponding entry in the revenue plan
4. Calculate the KRW value using model master's sales_unit_price_krw

This is useful for:
- Identifying excess/surplus inventory that isn't allocated to any plan
- Understanding what inventory exists outside of planned demand
- Tracking shipped, onhand, and transit inventory for models not in the plan

Output:
- One row per model per month for inventory NOT in the revenue plan showing:
  - unplanned_inventory_id: Unique identifier (revenue_plan_id + grouping_model + plan_month)
  - revenue_plan_id: The reference plan (MP202601-05W-004)
  - grouping_model: The grouping used (grouping_model if available, else model_id)
  - model_id: The original model_id value
  - plan_month: The month
  - shipped_inventory_ea: Shipped inventory not in plan
  - onhand_inventory_ea: On-hand inventory not in plan
  - transit_inventory_ea: Transit inventory not in plan
  - total_inventory_ea: Total unplanned inventory
  - total_inventory_sht: Total unplanned inventory in sheets
  - unit_price_krw: Price per unit from model master
  - shipped_amount_krw: KRW value of shipped inventory
  - onhand_amount_krw: KRW value of on-hand inventory
  - transit_amount_krw: KRW value of transit inventory
  - total_amount_krw: Total KRW value of unplanned inventory
"""

import polars as pl
from transforms.api import transform, Input, Output, lightweight
from datetime import datetime


# Reference plan for which we track unplanned inventory
REFERENCE_PLAN_ID = "MP202601-05W-004"


@lightweight(cpu_cores=2, memory_gb=8)
@transform(
    output=Output("/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/unplanned_inventory"),
    revenue_plan=Input("ri.foundry.main.dataset.2e72556a-1f0b-4ecc-9bdb-3e172781a406"),
    available_inventory=Input("ri.foundry.main.dataset.a8f8675b-a7b2-47f5-b83f-97f7028cc1b5"),
    model_master=Input("ri.foundry.main.dataset.2573f6cb-7e22-499e-b7e1-be7d895b1d1b"),
)
def compute(revenue_plan, available_inventory, model_master, output) -> None:
    """
    Calculate unplanned inventory - inventory that exists but is not in the revenue plan.

    This transform:
    1. Gets the revenue plan for the reference plan (MP202601-05W-004)
    2. Gets all available inventory
    3. Identifies inventory NOT in the revenue plan
    4. Calculates KRW values using model master prices
    5. Outputs unplanned inventory details

    Output columns:
    - unplanned_inventory_id: Unique identifier (revenue_plan_id + grouping_model + plan_month)
    - revenue_plan_id: The reference plan ID
    - grouping_model: The effective grouping (grouping_model if available, else model_id)
    - model_id: Original model_id value
    - plan_month: The planning month
    - shipped_inventory_ea: Shipped inventory not in plan
    - onhand_inventory_ea: On-hand inventory not in plan
    - transit_inventory_ea: Transit inventory not in plan
    - total_inventory_ea: Total unplanned inventory (EA)
    - total_inventory_sht: Total unplanned inventory (sheets)
    - unit_price_krw: Price per unit from model master
    - shipped_amount_krw: KRW value of shipped inventory
    - onhand_amount_krw: KRW value of on-hand inventory
    - transit_amount_krw: KRW value of transit inventory
    - total_amount_krw: Total KRW value of unplanned inventory
    - calculation_timestamp: When this calculation was performed
    """

    # =========================================================================
    # Load and prepare inputs
    # =========================================================================
    revenue_df = revenue_plan.polars()
    inventory_df = available_inventory.polars()
    model_df = model_master.polars()

    print(f"Revenue plan rows: {revenue_df.height:,}")
    print(f"Available inventory rows: {inventory_df.height:,}")
    print(f"Model master rows: {model_df.height:,}")

    # =========================================================================
    # Get unit prices from model master
    # =========================================================================
    model_unit_prices = (
        model_df.filter(pl.col("model_id").is_not_null() & pl.col("sales_unit_price_krw").is_not_null())
        .select(["model_id", pl.col("sales_unit_price_krw").alias("unit_price_krw")])
        .unique(subset=["model_id"], keep="first")
    )

    print(f"Model master unit prices: {model_unit_prices.height:,} models with price data")

    # =========================================================================
    # Prepare Revenue Plan - filter to reference plan and get planned grouping_models
    # =========================================================================
    revenue_clean = (
        revenue_df.filter(
            (pl.col("revenue_plan_id") == REFERENCE_PLAN_ID)
            & pl.col("model_id").is_not_null()
            & pl.col("plan_month").is_not_null()
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
        .select(["grouping_model", "plan_month"])
        .unique()
    )

    print(f"Revenue plan ({REFERENCE_PLAN_ID}): {revenue_clean.height:,} unique grouping_model-month combinations")

    # Get the plan months from the revenue plan
    plan_months = revenue_clean["plan_month"].unique().to_list()
    if not plan_months:
        print("No plan months found in revenue plan, outputting empty dataset")
        output.write_table(
            pl.DataFrame(
                schema={
                    "unplanned_inventory_id": pl.Utf8,
                    "revenue_plan_id": pl.Utf8,
                    "grouping_model": pl.Utf8,
                    "model_id": pl.Utf8,
                    "plan_month": pl.Utf8,
                    "shipped_inventory_ea": pl.Int64,
                    "onhand_inventory_ea": pl.Int64,
                    "transit_inventory_ea": pl.Int64,
                    "total_inventory_ea": pl.Int64,
                    "total_inventory_sht": pl.Float64,
                    "unit_price_krw": pl.Float64,
                    "shipped_amount_krw": pl.Float64,
                    "onhand_amount_krw": pl.Float64,
                    "transit_amount_krw": pl.Float64,
                    "total_amount_krw": pl.Float64,
                    "calculation_timestamp": pl.Datetime,
                }
            )
        )
        return

    print(f"Plan months: {len(plan_months)} months from {min(plan_months)} to {max(plan_months)}")

    # =========================================================================
    # Prepare Available Inventory - use grouping_model when available, else model_id
    # Filter to only months relevant to the plan
    # =========================================================================
    inventory_clean = (
        inventory_df.filter(
            pl.col("model_id").is_not_null()
            & pl.col("plan_month").is_not_null()
            & pl.col("plan_month").is_in(plan_months)
        )
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
                pl.col("shipped_quantity_ea").sum().alias("shipped_inventory_ea"),
                pl.col("onhand_quantity_ea").sum().alias("onhand_inventory_ea"),
                pl.col("transit_quantity_ea").sum().alias("transit_inventory_ea"),
                pl.col("total_inventory_ea").sum().alias("total_inventory_ea"),
                pl.col("total_inventory_sht").sum().alias("total_inventory_sht"),
                # Keep track of original model_id from inventory for reference
                pl.col("model_id").first().alias("model_id"),
            ]
        )
    )

    print(
        f"Available inventory (filtered to plan months): {inventory_clean.height:,} unique grouping_model-month combinations"
    )

    # =========================================================================
    # Find inventory NOT in the revenue plan using anti-join
    # =========================================================================
    unplanned_inventory = inventory_clean.join(
        revenue_clean,
        on=["grouping_model", "plan_month"],
        how="anti",  # Only keep rows that DON'T match the revenue plan
    )

    print(f"Unplanned inventory: {unplanned_inventory.height:,} unique grouping_model-month combinations")

    if unplanned_inventory.height == 0:
        print("No unplanned inventory found, outputting empty dataset")
        output.write_table(
            pl.DataFrame(
                schema={
                    "unplanned_inventory_id": pl.Utf8,
                    "revenue_plan_id": pl.Utf8,
                    "grouping_model": pl.Utf8,
                    "model_id": pl.Utf8,
                    "plan_month": pl.Utf8,
                    "shipped_inventory_ea": pl.Int64,
                    "onhand_inventory_ea": pl.Int64,
                    "transit_inventory_ea": pl.Int64,
                    "total_inventory_ea": pl.Int64,
                    "total_inventory_sht": pl.Float64,
                    "unit_price_krw": pl.Float64,
                    "shipped_amount_krw": pl.Float64,
                    "onhand_amount_krw": pl.Float64,
                    "transit_amount_krw": pl.Float64,
                    "total_amount_krw": pl.Float64,
                    "calculation_timestamp": pl.Datetime,
                }
            )
        )
        return

    # =========================================================================
    # Add reference plan ID
    # =========================================================================
    unplanned_inventory = unplanned_inventory.with_columns(pl.lit(REFERENCE_PLAN_ID).alias("revenue_plan_id"))

    # =========================================================================
    # Join with model master to get unit prices
    # =========================================================================
    unplanned_inventory = unplanned_inventory.join(
        model_unit_prices,
        on="model_id",
        how="left",
    )

    # Fill null prices with 0
    unplanned_inventory = unplanned_inventory.with_columns(pl.col("unit_price_krw").fill_null(0.0))

    # =========================================================================
    # Calculate KRW amounts
    # =========================================================================
    unplanned_inventory = unplanned_inventory.with_columns(
        [
            (pl.col("shipped_inventory_ea") * pl.col("unit_price_krw")).alias("shipped_amount_krw"),
            (pl.col("onhand_inventory_ea") * pl.col("unit_price_krw")).alias("onhand_amount_krw"),
            (pl.col("transit_inventory_ea") * pl.col("unit_price_krw")).alias("transit_amount_krw"),
            (pl.col("total_inventory_ea") * pl.col("unit_price_krw")).alias("total_amount_krw"),
        ]
    )

    # =========================================================================
    # Add unique identifier and calculation timestamp
    # =========================================================================
    unplanned_inventory = unplanned_inventory.with_columns(
        [
            # Create unique ID by concatenating revenue_plan_id, grouping_model, and plan_month
            pl.concat_str(
                [
                    pl.col("revenue_plan_id"),
                    pl.lit("_"),
                    pl.col("grouping_model"),
                    pl.lit("_"),
                    pl.col("plan_month"),
                ]
            ).alias("unplanned_inventory_id"),
            pl.lit(datetime.utcnow()).alias("calculation_timestamp"),
        ]
    )

    # =========================================================================
    # Select and order final columns
    # =========================================================================
    result_df = unplanned_inventory.select(
        [
            "unplanned_inventory_id",
            "revenue_plan_id",
            "grouping_model",
            "model_id",
            "plan_month",
            "shipped_inventory_ea",
            "onhand_inventory_ea",
            "transit_inventory_ea",
            "total_inventory_ea",
            "total_inventory_sht",
            "unit_price_krw",
            "shipped_amount_krw",
            "onhand_amount_krw",
            "transit_amount_krw",
            "total_amount_krw",
            "calculation_timestamp",
        ]
    )

    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    total_shipped = result_df["shipped_inventory_ea"].sum()
    total_onhand = result_df["onhand_inventory_ea"].sum()
    total_transit = result_df["transit_inventory_ea"].sum()
    total_inventory = result_df["total_inventory_ea"].sum()
    total_amount = result_df["total_amount_krw"].sum()

    print(f"\n=== Unplanned Inventory Summary (for {REFERENCE_PLAN_ID}) ===")
    print(f"Total rows: {result_df.height:,}")
    print(f"Unique model groupings: {result_df['grouping_model'].n_unique()}")
    print(f"Total unplanned inventory:")
    print(f"  - Shipped: {total_shipped:,} units")
    print(f"  - On-hand: {total_onhand:,} units")
    print(f"  - Transit: {total_transit:,} units")
    print(f"  - Total: {total_inventory:,} units")
    print(f"  - Total value: {total_amount:,.0f} KRW")

    # Show summary by month
    print(f"\n=== Summary by Month ===")
    month_summary = (
        result_df.group_by("plan_month")
        .agg(
            [
                pl.col("shipped_inventory_ea").sum().alias("shipped"),
                pl.col("onhand_inventory_ea").sum().alias("onhand"),
                pl.col("transit_inventory_ea").sum().alias("transit"),
                pl.col("total_inventory_ea").sum().alias("total_ea"),
                pl.col("total_amount_krw").sum().alias("total_krw"),
                pl.count().alias("model_count"),
            ]
        )
        .sort("plan_month")
    )

    for row in month_summary.iter_rows(named=True):
        month = row["plan_month"]
        shipped = row["shipped"]
        onhand = row["onhand"]
        transit = row["transit"]
        total_ea = row["total_ea"]
        total_krw = row["total_krw"]
        count = row["model_count"]
        print(
            f"  {month}: {count:,} models | "
            f"Shipped: {shipped:,} | Onhand: {onhand:,} | Transit: {transit:,} | "
            f"Total: {total_ea:,} EA ({total_krw:,.0f} KRW)"
        )

    # =========================================================================
    # Write Output
    # =========================================================================
    output.write_table(result_df)

    print(f"\n✓ Output written: {result_df.height:,} rows")
