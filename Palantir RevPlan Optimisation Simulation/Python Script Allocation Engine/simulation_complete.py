import polars as pl
from datetime import datetime, timezone
from transforms.api import Input, Output, LightweightInput, LightweightOutput, transform


@transform.using(
    simulation_complete_output=Output(
        "/LG Innotek-0e7800/[WF] SCM Planning Intelligence/logic/datasets/simulation_complete"
    ),
    revenue_plan_simulations=Input(
        "ri.foundry.main.dataset.7c220b40-47f1-4701-87e1-29d3e2938a85"
    ),
    revenue_plan_optimization_events=Input(
        "ri.foundry.main.dataset.b84b2bb5-1cbf-4970-8f73-0e1eb9d4148f"
    ),
    revenue_plan_optimization_equipment_load=Input(
        "ri.foundry.main.dataset.0b7e1043-86cb-4c9f-a4f3-fe71d13700e2"
    ),
    revenue_plan_optimization_model_stats_long=Input(
        "ri.foundry.main.dataset.360e9503-0163-4b39-9eb0-7ca882c6cdde"
    ),
    revenue_plan_optimization_model_stats_wide=Input(
        "ri.foundry.main.dataset.f9dd33b1-6764-4bd9-8be3-883cc279a614"
    ),
    equipment_group_capacity_shortages=Input(
        "ri.foundry.main.dataset.6f0ce94e-0072-4ec1-8e3e-3188844b9b52"
    ),
    lot_waiting_periods=Input(
        "ri.foundry.main.dataset.332070f1-98dc-442a-a48f-0a617d61be5b"
    ),
    material_depletion_events=Input(
        "ri.foundry.main.dataset.590fbc89-3520-41f8-a624-a2e7d8813a94"
    ),
    demand_analysis=Input("ri.foundry.main.dataset.a715dabf-cf86-40be-83b0-4dbbf44aa169")
)
def compute(
    simulation_complete_output: LightweightOutput,
    revenue_plan_simulations: LightweightInput,
    revenue_plan_optimization_events: LightweightInput,
    revenue_plan_optimization_equipment_load: LightweightInput,
    revenue_plan_optimization_model_stats_long: LightweightInput,
    revenue_plan_optimization_model_stats_wide: LightweightInput,
    equipment_group_capacity_shortages: LightweightInput,
    lot_waiting_periods: LightweightInput,
    material_depletion_events: LightweightInput,
    demand_analysis: LightweightInput,
) -> None:
    simulations_df = revenue_plan_simulations.polars()

    # Add completed_at timestamp (current time in UTC to match created_at timezone)
    completed_at = datetime.now(timezone.utc)

    # Add completed_at and calculate duration_seconds
    result_df = simulations_df.with_columns(
        [
            pl.lit(completed_at).alias("completed_at"),
            ((pl.lit(completed_at) - pl.col("created_at")).dt.total_seconds()).alias(
                "duration_seconds"
            ),
        ]
    ).select(
        "primary_key_",
        "created_at",
        "completed_at",
        "duration_seconds"
    )

    simulation_complete_output.write_table(result_df)
