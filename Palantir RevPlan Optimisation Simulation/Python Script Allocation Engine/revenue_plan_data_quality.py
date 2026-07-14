"""
Revenue Plan Data Quality Issues

Identifies data quality issues for models, BOMs, and materials
that are part of an active revenue plan.
"""

import polars as pl
from transforms.api import Input, Output, LightweightInput, LightweightOutput, transform


@transform.using(
    revenue_plan_data_quality_issues=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/revenue_plan_data_quality_issues"
    ),
    revenue_plan_monthly_models=Input("ri.foundry.main.dataset.2e72556a-1f0b-4ecc-9bdb-3e172781a406"),
    model_master=Input("ri.foundry.main.dataset.2573f6cb-7e22-499e-b7e1-be7d895b1d1b"),
    model_boms=Input("ri.foundry.main.dataset.030d06f4-6442-4bc5-a065-740f73bbf537"),
    material_master=Input("ri.foundry.main.dataset.837f757a-1747-4fde-a6b2-d2fd777c3d2c"),
    model_routings=Input("ri.foundry.main.dataset.ea5a565d-e3c1-4985-a5c2-2ad112ddac96"),
    model_unit_conversions=Input("ri.foundry.main.dataset.b806e6e0-24d7-4241-9568-4d92700bc7ef"),
).with_resources(cpu_cores=2, memory_gb=16)
def compute(
    revenue_plan_data_quality_issues: LightweightOutput,
    revenue_plan_monthly_models: LightweightInput,
    model_master: LightweightInput,
    model_boms: LightweightInput,
    material_master: LightweightInput,
    model_routings: LightweightInput,
    model_unit_conversions: LightweightInput,
) -> None:
    # Load input datasets - only select columns we need to minimize memory
    revenue_plan_monthly_models_df = revenue_plan_monthly_models.polars(lazy=True).select(
        ["revenue_plan_id", "model_id", "amount_krw"]
    )

    model_master_df = model_master.polars(lazy=True).select(["model_id", "grouping_model", "maximum_lot_size_ea"])
    boms_df = model_boms.polars(lazy=True).select(["model_id", "material_id"])
    materials_df = material_master.polars(lazy=True).select(["material_id"])
    routings_df = model_routings.polars(lazy=True).select(["model_id"])
    unit_conversions_df = model_unit_conversions.polars(lazy=True).select(
        ["model_id", "panels_per_sheet", "units_per_panel", "units_per_sheet"]
    )

    # Get unique models per revenue plan from the monthly models dataset
    # This is our source of truth for which models are in each revenue plan
    # Exclude "기타매출" which is a stand-in for non-model revenue
    models_in_revenue_plans = (
        revenue_plan_monthly_models_df.filter(pl.col("model_id") != "기타매출")
        .select(["revenue_plan_id", "model_id"])
        .unique()
    )

    # Get unique models with routing
    models_with_routing = routings_df.select("model_id").unique()

    # Get unique models with BOM
    models_with_bom = boms_df.select("model_id").unique()

    # Get unique materials in material master
    known_materials = materials_df.select("material_id").unique()

    # Prepare model master lookup (relevant columns only)
    model_master_lookup = model_master_df.unique(subset=["model_id"])

    all_issues = []

    # =========================================================================
    # Issue Type 1: MISSING_ROUTING
    # Models in revenue plan with no routing defined
    # =========================================================================
    missing_routing = models_in_revenue_plans.join(models_with_routing, on="model_id", how="anti").with_columns(
        [
            pl.lit("MISSING_ROUTING").alias("issue_type"),
            pl.lit(None).cast(pl.Utf8).alias("affected_entity_id"),
            pl.concat_str(
                [
                    pl.lit("Model "),
                    pl.col("model_id"),
                    pl.lit(" has no routing defined"),
                ]
            ).alias("description_en"),
            pl.concat_str(
                [
                    pl.lit("모델 "),
                    pl.col("model_id"),
                    pl.lit("에 라우팅이 정의되어 있지 않습니다"),
                ]
            ).alias("description_ko"),
        ]
    )
    all_issues.append(missing_routing)

    # =========================================================================
    # Issue Type 2: MISSING_BOM
    # Models in revenue plan with no BOM defined
    # =========================================================================
    missing_bom = models_in_revenue_plans.join(models_with_bom, on="model_id", how="anti").with_columns(
        [
            pl.lit("MISSING_BOM").alias("issue_type"),
            pl.lit(None).cast(pl.Utf8).alias("affected_entity_id"),
            pl.concat_str(
                [
                    pl.lit("Model "),
                    pl.col("model_id"),
                    pl.lit(" has no BOM defined"),
                ]
            ).alias("description_en"),
            pl.concat_str(
                [
                    pl.lit("모델 "),
                    pl.col("model_id"),
                    pl.lit("에 BOM이 정의되어 있지 않습니다"),
                ]
            ).alias("description_ko"),
        ]
    )
    all_issues.append(missing_bom)

    # =========================================================================
    # Issue Type 3: INVALID_MATERIAL_REFERENCE
    # BOM entries for models in revenue plan referencing unknown materials
    # =========================================================================
    # First, get BOMs for models in revenue plans
    boms_for_revenue_plan_models = boms_df.unique().join(models_in_revenue_plans, on="model_id", how="inner")

    # Find materials referenced in BOMs that don't exist in material master
    invalid_material_refs = boms_for_revenue_plan_models.join(
        known_materials, on="material_id", how="anti"
    ).with_columns(
        [
            pl.lit("INVALID_MATERIAL_REFERENCE").alias("issue_type"),
            pl.col("material_id").alias("affected_entity_id"),
            pl.concat_str(
                [
                    pl.lit("BOM for model "),
                    pl.col("model_id"),
                    pl.lit(" references unknown material "),
                    pl.col("material_id"),
                ]
            ).alias("description_en"),
            pl.concat_str(
                [
                    pl.lit("모델 "),
                    pl.col("model_id"),
                    pl.lit("의 BOM이 알 수 없는 자재 "),
                    pl.col("material_id"),
                    pl.lit("를 참조합니다"),
                ]
            ).alias("description_ko"),
        ]
    )
    all_issues.append(invalid_material_refs)

    # =========================================================================
    # Issue Type 4: MISSING_GROUPING_MODEL
    # Models in revenue plan with no grouping model assigned
    # =========================================================================
    models_with_grouping_info = models_in_revenue_plans.join(
        model_master_lookup.select(["model_id", "grouping_model"]), on="model_id", how="left"
    )

    missing_grouping_model = (
        models_with_grouping_info.filter(
            pl.col("grouping_model").is_null() | (pl.col("grouping_model").str.strip_chars() == "")
        )
        .with_columns(
            [
                pl.lit("MISSING_GROUPING_MODEL").alias("issue_type"),
                pl.lit(None).cast(pl.Utf8).alias("affected_entity_id"),
                pl.concat_str(
                    [
                        pl.lit("Model "),
                        pl.col("model_id"),
                        pl.lit(" has no grouping model assigned"),
                    ]
                ).alias("description_en"),
                pl.concat_str(
                    [
                        pl.lit("모델 "),
                        pl.col("model_id"),
                        pl.lit("에 그룹핑 모델이 지정되지 않았습니다"),
                    ]
                ).alias("description_ko"),
            ]
        )
        .select(["revenue_plan_id", "model_id", "issue_type", "affected_entity_id", "description_en", "description_ko"])
    )
    all_issues.append(missing_grouping_model)

    # =========================================================================
    # Issue Type 5: INVALID_LOT_SIZE
    # Models in revenue plan where maximum_lot_size_ea = 1
    # =========================================================================
    models_with_lot_size = models_in_revenue_plans.join(
        model_master_lookup.select(["model_id", "maximum_lot_size_ea"]), on="model_id", how="left"
    )

    invalid_lot_size = (
        models_with_lot_size.filter(pl.col("maximum_lot_size_ea") == 1)
        .with_columns(
            [
                pl.lit("INVALID_LOT_SIZE").alias("issue_type"),
                pl.lit(None).cast(pl.Utf8).alias("affected_entity_id"),
                pl.concat_str(
                    [
                        pl.lit("Model "),
                        pl.col("model_id"),
                        pl.lit(" has lot size of 1 (invalid)"),
                    ]
                ).alias("description_en"),
                pl.concat_str(
                    [
                        pl.lit("모델 "),
                        pl.col("model_id"),
                        pl.lit("의 로트 크기가 1입니다 (유효하지 않음)"),
                    ]
                ).alias("description_ko"),
            ]
        )
        .select(["revenue_plan_id", "model_id", "issue_type", "affected_entity_id", "description_en", "description_ko"])
    )
    all_issues.append(invalid_lot_size)

    # =========================================================================
    # Issue Type 6: ZERO_REVENUE
    # Models in revenue plan where amount_krw = 0 or null
    # =========================================================================
    # Aggregate revenue by revenue_plan_id and model_id
    models_with_revenue = revenue_plan_monthly_models_df.group_by(["revenue_plan_id", "model_id"]).agg(
        [pl.col("amount_krw").sum().alias("total_revenue")]
    )

    zero_revenue = (
        models_with_revenue.filter(pl.col("total_revenue").is_null() | (pl.col("total_revenue") == 0))
        .with_columns(
            [
                pl.lit("ZERO_REVENUE").alias("issue_type"),
                pl.lit(None).cast(pl.Utf8).alias("affected_entity_id"),
                pl.concat_str(
                    [
                        pl.lit("Model "),
                        pl.col("model_id"),
                        pl.lit(" has zero or no revenue defined"),
                    ]
                ).alias("description_en"),
                pl.concat_str(
                    [
                        pl.lit("모델 "),
                        pl.col("model_id"),
                        pl.lit("의 매출이 0이거나 정의되지 않았습니다"),
                    ]
                ).alias("description_ko"),
            ]
        )
        .select(["revenue_plan_id", "model_id", "issue_type", "affected_entity_id", "description_en", "description_ko"])
    )
    all_issues.append(zero_revenue)

    # =========================================================================
    # Issue Type 7: INVALID_UNIT_CONVERSION
    # Models in revenue plan with null or zero unit conversion values
    # =========================================================================
    models_with_unit_conversions = models_in_revenue_plans.join(
        unit_conversions_df.unique(subset=["model_id"]), on="model_id", how="left"
    )

    invalid_unit_conversion = (
        models_with_unit_conversions.filter(
            pl.col("panels_per_sheet").is_null()
            | (pl.col("panels_per_sheet") == 0)
            | pl.col("units_per_panel").is_null()
            | (pl.col("units_per_panel") == 0)
            | pl.col("units_per_sheet").is_null()
            | (pl.col("units_per_sheet") == 0)
        )
        .with_columns(
            [
                pl.lit("INVALID_UNIT_CONVERSION").alias("issue_type"),
                pl.lit(None).cast(pl.Utf8).alias("affected_entity_id"),
                pl.concat_str(
                    [
                        pl.lit("Model "),
                        pl.col("model_id"),
                        pl.lit(
                            " has invalid unit conversion (panels_per_sheet, units_per_panel, or units_per_sheet is null or zero)"
                        ),
                    ]
                ).alias("description_en"),
                pl.concat_str(
                    [
                        pl.lit("모델 "),
                        pl.col("model_id"),
                        pl.lit(
                            "의 단위 변환이 유효하지 않습니다 (panels_per_sheet, units_per_panel 또는 units_per_sheet가 null이거나 0입니다)"
                        ),
                    ]
                ).alias("description_ko"),
            ]
        )
        .select(["revenue_plan_id", "model_id", "issue_type", "affected_entity_id", "description_en", "description_ko"])
    )
    all_issues.append(invalid_unit_conversion)

    # =========================================================================
    # Combine all issues and create unique issue IDs
    # =========================================================================
    # Ensure all dataframes have the same schema before concatenating
    schema_columns = [
        "revenue_plan_id",
        "model_id",
        "issue_type",
        "affected_entity_id",
        "description_en",
        "description_ko",
    ]

    standardized_issues = []
    for issue_df in all_issues:
        standardized_issues.append(issue_df.select(schema_columns))

    combined_issues = pl.concat(standardized_issues)

    # Create unique issue_id
    combined_issues = combined_issues.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("revenue_plan_id"),
                    pl.lit("_"),
                    pl.col("model_id"),
                    pl.lit("_"),
                    pl.col("issue_type"),
                    pl.lit("_"),
                    pl.col("affected_entity_id").fill_null(""),
                ]
            ).alias("issue_id")
        ]
    ).select(
        [
            "issue_id",
            "revenue_plan_id",
            "model_id",
            "issue_type",
            "affected_entity_id",
            "description_en",
            "description_ko",
        ]
    )

    # Write output - collect the lazy frame
    revenue_plan_data_quality_issues.write_table(combined_issues.collect())
