import polars as pl
from transforms.api import Input, Output, LightweightInput, LightweightOutput, transform


@transform.using(
    material_consumption_events=Output("ri.foundry.main.dataset.34e31856-9529-4232-bede-87058a04bfab"),
    material_depletion_events=Output("ri.foundry.main.dataset.590fbc89-3520-41f8-a624-a2e7d8813a94"),
    constrained_production_lots=Output("ri.foundry.main.dataset.809fb8bf-1565-4831-bf9f-81003aaed640"),
    data_quality_issues=Output("ri.foundry.main.dataset.3967652b-82ea-4eff-b35e-255908d7c5f7"),
    revenue_plan_optimization_events=Input("ri.foundry.main.dataset.b84b2bb5-1cbf-4970-8f73-0e1eb9d4148f"),
    material_inventories=Input("ri.foundry.main.dataset.401b1405-fc50-49a6-8421-339ab4b65f42"),
    model_boms=Input("ri.foundry.main.dataset.030d06f4-6442-4bc5-a065-740f73bbf537"),
    planned_material_arrivals=Input("ri.foundry.main.dataset.ce618b18-6d61-4486-8ac9-85c09ce700fc"),
)
def compute(
    material_consumption_events: LightweightOutput,
    material_depletion_events: LightweightOutput,
    constrained_production_lots: LightweightOutput,
    data_quality_issues: LightweightOutput,
    revenue_plan_optimization_events: LightweightInput,
    material_inventories: LightweightInput,
    model_boms: LightweightInput,
    planned_material_arrivals: LightweightInput,
) -> None:
    # Load input datasets
    events_df = revenue_plan_optimization_events.polars()
    boms_df = model_boms.polars()
    inventory_df = material_inventories.polars()
    arrivals_df = planned_material_arrivals.polars()

    # Get the set of models that are actually in the revenue plan
    # This ensures we only check data quality for models being produced
    models_in_revenue_plan = events_df.select("model_id").unique()

    # Filter BOM to only include models in the revenue plan
    boms_df_filtered = boms_df.join(models_in_revenue_plan, on="model_id", how="inner")

    # Extract material UOM lookup (one UOM per material_id)
    # Take the first UOM for each material since it should be consistent
    material_uom_lookup = boms_df.select(["material_id", "uom"]).unique(subset=["material_id"], maintain_order=True)

    # Step 1: Join production events with BOM to get material requirements
    # Match on model_id, process_id, and sequence (events) = work_sequence (boms)
    material_requirements = events_df.join(
        boms_df_filtered,
        left_on=["model_id", "process_id", "sequence"],
        right_on=["model_id", "process_id", "work_sequence"],
        how="inner",
    ).select(
        [
            "allocation_id",
            "lot_id",
            "model_id",
            "process_id",
            "sequence",
            "equipment_id",
            "allocated_date",
            "material_id",
            "required_quantity",
            "uom",
            "final_production_units",
            "simulation_id",
            "simulation_name",
        ]
    )

    # DATA QUALITY CHECK: Identify materials in BOM without inventory
    # ONLY for models in the revenue plan (filtered BOM)
    materials_in_bom = boms_df_filtered.select("material_id").unique()
    materials_with_inventory = inventory_df.select("material_id").unique()

    materials_missing_inventory = materials_in_bom.join(
        materials_with_inventory, on="material_id", how="anti"
    ).with_columns(
        [
            pl.lit("MISSING_INVENTORY").alias("issue_type"),
            pl.concat_str(
                [
                    pl.lit("Material "),
                    pl.col("material_id"),
                    pl.lit(" exists in BOM but has no inventory record"),
                ]
            ).alias("issue_description"),
            pl.concat_str([pl.lit("MISSING_INV_"), pl.col("material_id")]).alias("issue_id"),
        ]
    )

    # Find which model/process combinations use these materials
    # Use filtered BOM to only show models in revenue plan
    materials_missing_inventory_detail = (
        materials_missing_inventory.join(
            boms_df_filtered.select(["material_id", "model_id", "process_id", "work_sequence"]).unique(),
            on="material_id",
            how="left",
        )
        .group_by(["issue_id", "issue_type", "issue_description", "material_id"])
        .agg(
            [
                pl.col("model_id").unique().alias("affected_models"),
                pl.col("process_id").unique().alias("affected_processes"),
                pl.col("model_id").n_unique().alias("affected_model_count"),
                pl.col("process_id").n_unique().alias("affected_process_count"),
            ]
        )
    )

    # Step 2: Calculate actual material consumption (required_quantity is per unit)
    # Multiply by final_production_units to get total consumption
    material_requirements = material_requirements.with_columns(
        [(pl.col("required_quantity") * pl.col("final_production_units")).alias("total_consumption")]
    )

    # Step 3: Get initial inventory per material
    # NOTE: This is global inventory, not per-simulation
    initial_inventory = inventory_df.group_by("material_id").agg(
        pl.col("onhand_quantity").sum().alias("initial_inventory")
    )

    # Step 3b: Prepare planned arrivals - aggregate by material and date
    # Arrivals are available at the START of the plan_date
    planned_arrivals_agg = arrivals_df.group_by(["material_id", "plan_date"]).agg(
        pl.col("quantity").sum().alias("arrival_quantity")
    )

    # Step 4: Sort consumption events chronologically for sequential processing
    # Group by simulation_id and sort within each simulation
    # Use allocation_id as tiebreaker for deterministic ordering within same date
    consumption_events = material_requirements.sort(["simulation_id", "allocated_date", "allocation_id"])

    # Step 5: Process events chronologically and track running inventory PER SIMULATION
    # Calculate cumulative consumption per material per simulation over time
    consumption_with_cumsum = consumption_events.with_columns(
        [pl.col("total_consumption").cum_sum().over(["simulation_id", "material_id"]).alias("cumulative_consumption")]
    )

    # Step 5b: Join with arrivals to calculate cumulative arrivals up to each date
    # For each consumption event, sum all arrivals that happened on or before that date
    # Create a cross join context to calculate cumulative arrivals
    consumption_with_arrivals = (
        consumption_with_cumsum.join(planned_arrivals_agg, on="material_id", how="left")
        .with_columns(
            [
                # Only count arrivals that happen on or before the consumption date
                pl.when(pl.col("plan_date") <= pl.col("allocated_date"))
                .then(pl.col("arrival_quantity"))
                .otherwise(0)
                .alias("applicable_arrival")
            ]
        )
        .group_by(
            [
                "simulation_id",
                "allocation_id",
                "lot_id",
                "model_id",
                "process_id",
                "sequence",
                "equipment_id",
                "allocated_date",
                "material_id",
                "required_quantity",
                "uom",
                "final_production_units",
                "simulation_name",
                "total_consumption",
                "cumulative_consumption",
            ]
        )
        .agg(pl.col("applicable_arrival").sum().alias("cumulative_arrivals"))
        .sort(["simulation_id", "allocated_date", "allocation_id"])
    )

    # Join with initial inventory to calculate running inventory (WITHOUT arrivals)
    consumption_with_inventory = consumption_with_arrivals.join(
        initial_inventory, on="material_id", how="left"
    ).with_columns(
        [
            # WITHOUT arrivals (original logic)
            (pl.col("initial_inventory").fill_null(0) - pl.col("cumulative_consumption")).alias(
                "running_inventory_after"
            ),
            (pl.col("initial_inventory").fill_null(0) - pl.col("cumulative_consumption") < 0).alias("is_shortage"),
            # WITH arrivals (new logic)
            (
                pl.col("initial_inventory").fill_null(0)
                + pl.col("cumulative_arrivals").fill_null(0)
                - pl.col("cumulative_consumption")
            ).alias("running_inventory_after_with_arrivals"),
            (
                pl.col("initial_inventory").fill_null(0)
                + pl.col("cumulative_arrivals").fill_null(0)
                - pl.col("cumulative_consumption")
                < 0
            ).alias("is_shortage_with_arrivals"),
        ]
    )

    # Calculate inventory before this event (both with and without arrivals)
    consumption_with_inventory = consumption_with_inventory.with_columns(
        [
            (pl.col("running_inventory_after") + pl.col("total_consumption")).alias("running_inventory_before"),
            (pl.col("running_inventory_after_with_arrivals") + pl.col("total_consumption")).alias(
                "running_inventory_before_with_arrivals"
            ),
        ]
    )

    # Create unique key for material consumption events derived from first principles
    # Use row_number within each simulation to ensure uniqueness
    consumption_with_inventory = consumption_with_inventory.with_columns(
        [pl.int_range(pl.len()).over("simulation_id").alias("row_num")]
    ).with_columns(
        [
            pl.concat_str(
                [
                    pl.col("simulation_id"),
                    pl.lit("_"),
                    pl.col("material_id"),
                    pl.lit("_"),
                    pl.col("allocated_date").cast(pl.Utf8),
                    pl.lit("_"),
                    pl.col("lot_id"),
                    pl.lit("_"),
                    pl.col("sequence").cast(pl.Utf8),
                    pl.lit("_"),
                    pl.col("row_num").cast(pl.Utf8),
                ]
            ).alias("consumption_event_id")
        ]
    )

    # Output 1: Material Consumption Events
    material_consumption_output = consumption_with_inventory.select(
        [
            "consumption_event_id",
            "simulation_id",
            "simulation_name",
            "allocation_id",
            "lot_id",
            "model_id",
            "process_id",
            "sequence",
            "equipment_id",
            "allocated_date",
            "material_id",
            "uom",
            "required_quantity",
            "final_production_units",
            "total_consumption",
            "cumulative_arrivals",
            "running_inventory_before",
            "running_inventory_after",
            "is_shortage",
            "running_inventory_before_with_arrivals",
            "running_inventory_after_with_arrivals",
            "is_shortage_with_arrivals",
        ]
    )

    # Step 6: Identify material depletion events PER SIMULATION (WITH ARRIVALS)
    # A material can have MULTIPLE depletion events if it depletes, gets topped up, then depletes again

    # First, aggregate to daily level to get minimum inventory per date
    # This prevents false positives from row-level transitions within the same day
    daily_min_inventory = (
        consumption_with_inventory.group_by(["simulation_id", "material_id", "allocated_date"])
        .agg(
            [
                pl.col("running_inventory_after_with_arrivals").min().alias("min_inventory_on_date"),
                pl.col("simulation_name").first().alias("simulation_name"),
                pl.col("initial_inventory").first().alias("initial_inventory"),
            ]
        )
        .sort(["simulation_id", "material_id", "allocated_date"])
    )

    # Now detect transitions at the date level (not row level)
    depletion_dates = (
        daily_min_inventory.with_columns(
            [
                # Get previous date's minimum inventory
                pl.col("min_inventory_on_date")
                .shift(1)
                .over(["simulation_id", "material_id"])
                .alias("prev_date_min_inventory"),
            ]
        )
        .with_columns(
            [
                # A depletion event occurs when we transition from >= 0 to < 0 at the DATE level
                (
                    (pl.col("min_inventory_on_date") < 0)
                    & ((pl.col("prev_date_min_inventory") >= 0) | pl.col("prev_date_min_inventory").is_null())
                ).alias("is_depletion_event")
            ]
        )
        .filter(pl.col("is_depletion_event"))
        .select(
            [
                "simulation_id",
                "material_id",
                "allocated_date",
                "simulation_name",
                "initial_inventory",
            ]
        )
        .rename({"allocated_date": "depletion_date"})
    )

    # Get the actual shortage amount on each depletion date
    shortage_on_depletion_date = (
        consumption_with_inventory.join(
            depletion_dates.select(["simulation_id", "material_id", "depletion_date"]),
            left_on=["simulation_id", "material_id", "allocated_date"],
            right_on=["simulation_id", "material_id", "depletion_date"],
            how="inner",
        )
        .filter(pl.col("is_shortage_with_arrivals"))
        .group_by(["simulation_id", "material_id", "allocated_date"])
        .agg([pl.col("running_inventory_after_with_arrivals").min().abs().alias("shortfall_quantity")])
        .rename({"allocated_date": "depletion_date"})
    )

    # Get all lots consuming material on each depletion date
    depleting_lots = (
        consumption_with_inventory.join(
            depletion_dates.select(["simulation_id", "material_id", "depletion_date"]),
            left_on=["simulation_id", "material_id", "allocated_date"],
            right_on=["simulation_id", "material_id", "depletion_date"],
            how="inner",
        )
        .group_by(["simulation_id", "material_id", "allocated_date"])
        .agg(
            [
                pl.col("lot_id").unique().alias("depleting_lot_ids"),
                pl.col("allocation_id").unique().alias("depleting_allocation_ids"),
                pl.col("total_consumption").sum().alias("total_demand_on_depletion_date"),
                pl.col("lot_id").n_unique().alias("depleting_lots_count"),
            ]
        )
        .rename({"allocated_date": "depletion_date"})
    )

    # Count total constrained lots per material per simulation (with arrivals)
    # For multiple depletion events, we need to count across all shortage periods
    total_constrained_lots_per_material = (
        consumption_with_inventory.filter(pl.col("is_shortage_with_arrivals"))
        .group_by(["simulation_id", "material_id"])
        .agg([pl.col("lot_id").n_unique().alias("total_lots_constrained")])
    )

    # Combine all depletion info
    depletion_events = (
        depletion_dates.join(
            shortage_on_depletion_date,
            on=["simulation_id", "material_id", "depletion_date"],
            how="left",
        )
        .join(
            depleting_lots,
            on=["simulation_id", "material_id", "depletion_date"],
            how="left",
        )
        .join(
            total_constrained_lots_per_material,
            on=["simulation_id", "material_id"],
            how="left",
        )
        .with_columns(
            [
                # Create sequence number for multiple depletions of same material
                pl.col("depletion_date").cum_count().over(["simulation_id", "material_id"]).alias("depletion_sequence"),
            ]
        )
        .with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("simulation_id"),
                        pl.lit("_"),
                        pl.col("material_id"),
                        pl.lit("_"),
                        pl.col("depletion_date").cast(pl.Utf8),
                        pl.lit("_"),
                        pl.col("depletion_sequence").cast(pl.Utf8),
                    ]
                ).alias("depletion_event_id"),
                pl.concat_str(
                    [
                        pl.lit("Material "),
                        pl.col("material_id"),
                        pl.lit(" depleted on "),
                        pl.col("depletion_date").cast(pl.Utf8),
                        pl.lit(" (occurrence #"),
                        pl.col("depletion_sequence").cast(pl.Utf8),
                        pl.lit("): demand of "),
                        pl.col("total_demand_on_depletion_date").cast(pl.Utf8),
                        pl.lit(" units exceeded available inventory, affecting "),
                        pl.col("depleting_lots_count").cast(pl.Utf8),
                        pl.lit(" lots on that day ("),
                        pl.col("total_lots_constrained").cast(pl.Utf8),
                        pl.lit(" total lots constrained)"),
                    ]
                ).alias("depletion_description"),
            ]
        )
        .join(material_uom_lookup, on="material_id", how="left")
    )

    # Output 2: Material Depletion Events
    material_depletion_output = depletion_events.select(
        [
            "depletion_event_id",
            "simulation_id",
            "simulation_name",
            "material_id",
            "uom",
            "depletion_date",
            "depletion_sequence",
            "depleting_lot_ids",
            "depleting_allocation_ids",
            "depleting_lots_count",
            "total_demand_on_depletion_date",
            "initial_inventory",
            "shortfall_quantity",
            "total_lots_constrained",
            "depletion_description",
        ]
    )

    # Step 7: Identify constrained production lots PER SIMULATION (WITH ARRIVALS)
    # A lot is constrained if any of its events have a shortage - they cannot complete
    constrained_lots_with_materials = (
        consumption_with_inventory.filter(pl.col("is_shortage_with_arrivals"))
        .group_by(["simulation_id", "lot_id"])
        .agg(
            [
                pl.col("model_id").first().alias("model_id"),
                pl.col("allocated_date").min().alias("constraint_date"),
                pl.col("material_id").unique().alias("constraining_material_ids"),
                pl.col("sequence").min().alias("first_constrained_sequence"),
                pl.col("sequence").max().alias("last_sequence_attempted"),
                pl.col("simulation_name").first().alias("simulation_name"),
            ]
        )
    )

    # Get total sequences per lot from the original events
    lot_sequences = events_df.group_by(["simulation_id", "lot_id"]).agg(
        [
            pl.col("sequence").max().alias("total_sequences"),
            pl.col("sequence").min().alias("first_sequence"),
        ]
    )

    constrained_lots = (
        constrained_lots_with_materials.join(lot_sequences, on=["simulation_id", "lot_id"], how="left")
        .with_columns(
            [
                (pl.col("first_constrained_sequence") - 1).alias("furthest_completable_sequence"),
                pl.col("constraining_material_ids").list.len().alias("constraining_materials_count"),
                pl.concat_str([pl.col("simulation_id"), pl.lit("_"), pl.col("lot_id")]).alias("constrained_lot_id"),
            ]
        )
        .with_columns(
            [
                # After calculating furthest_completable_sequence, determine if partial
                (pl.col("furthest_completable_sequence") < pl.col("total_sequences")).alias("is_partial_completion"),
                pl.concat_str(
                    [
                        pl.lit("Lot "),
                        pl.col("lot_id"),
                        pl.lit(" ("),
                        pl.col("model_id"),
                        pl.lit(") blocked by "),
                        pl.col("constraining_materials_count").cast(pl.Utf8),
                        pl.lit(" material(s) on "),
                        pl.col("constraint_date").cast(pl.Utf8),
                        pl.lit(" - can complete "),
                        pl.col("furthest_completable_sequence").cast(pl.Utf8),
                        pl.lit(" of "),
                        pl.col("total_sequences").cast(pl.Utf8),
                        pl.lit(" sequences"),
                    ]
                ).alias("constraint_description"),
            ]
        )
    )

    # Output 3: Constrained Production Lots
    constrained_lots_output = constrained_lots.select(
        [
            "constrained_lot_id",
            "simulation_id",
            "simulation_name",
            "lot_id",
            "model_id",
            "constraint_date",
            "constraining_material_ids",
            "constraining_materials_count",
            "furthest_completable_sequence",
            "first_constrained_sequence",
            "total_sequences",
            "is_partial_completion",
            "constraint_description",
        ]
    )

    # Output 4: Data Quality Issues
    data_quality_output = materials_missing_inventory_detail.select(
        [
            "issue_id",
            "issue_type",
            "material_id",
            "issue_description",
            "affected_models",
            "affected_processes",
            "affected_model_count",
            "affected_process_count",
        ]
    )

    # Write all outputs
    material_consumption_events.write_table(material_consumption_output)
    material_depletion_events.write_table(material_depletion_output)
    constrained_production_lots.write_table(constrained_lots_output)
    data_quality_issues.write_table(data_quality_output)
