"""
Transform to calculate missing predicted start and finish times (median, Q1, Q3, std)
for Predicted Event Lot going Through Process dataset.
Also propagates quantity information through the predicted events.

Lots on hold: If calculated start times are before October 31st, 2025, they are
adjusted to October 31st as this represents when lots come off hold.

Quantity columns: All quantity columns (units, sheets, panels, strips) are converted
from doubles to integers using ceiling function to round up fractional quantities.

Time types:
- median, Q1, Q3: Based on historical data
- std: Based on standard/planned data
"""

from transforms.api import transform, Input, Output, lightweight
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define the hold release date (October 31, 2025) - treat this as "today"
# Using pd.Timestamp with Korea timezone (Asia/Seoul)
HOLD_RELEASE_DATE = pd.Timestamp("2025-10-31", tz="Asia/Seoul")


def process_single_lot(lot_data_tuple):
    """
    Process a single lot's events to calculate start and finish times.
    This function is designed to be run in parallel for different lots.

    Args:
        lot_data_tuple: Tuple of (lot_id, lot_data_dict) where lot_data_dict
                        contains the DataFrame as a dictionary
    """
    lot_id, lot_data_dict = lot_data_tuple

    # Reconstruct DataFrame from dict
    lot_data = pd.DataFrame(lot_data_dict)

    # Sort by planned_work_sequence to ensure correct order
    lot_data = lot_data.sort_values("planned_work_sequence").reset_index(drop=True)

    # Create local event_id mapping
    local_event_id_to_idx = {
        event_id: idx
        for idx, event_id in enumerate(lot_data["predicted_event_lot_process_id"])
        if pd.notna(event_id)
    }

    # Process events sequentially within this lot
    for idx in range(len(lot_data)):
        row = lot_data.iloc[idx]

        # Get the process durations (already in seconds) and wait time (in seconds)
        median_process_duration = (
            float(row.get("predicted_total_median_panel_in_seconds", 0))
            if pd.notna(row.get("predicted_total_median_panel_in_seconds"))
            else 0.0
        )
        q1_process_duration = (
            float(row.get("predicted_total_Q1_panel_in_seconds", 0))
            if pd.notna(row.get("predicted_total_Q1_panel_in_seconds"))
            else median_process_duration
        )
        q3_process_duration = (
            float(row.get("predicted_total_Q3_panel_in_seconds", 0))
            if pd.notna(row.get("predicted_total_Q3_panel_in_seconds"))
            else median_process_duration
        )
        std_process_duration = (
            float(row.get("predicted_total_std_time_per_panel_in_s", 0))
            if pd.notna(row.get("predicted_total_std_time_per_panel_in_s"))
            else 0.0
        )
        wait_time = (
            float(row.get("median_prev_process_wait_in_seconds", 0))
            if pd.notna(row.get("median_prev_process_wait_in_seconds"))
            else 0.0
        )
        std_wait_time = (
            float(row.get("std_wait_time_in_s", 0))
            if pd.notna(row.get("std_wait_time_in_s"))
            else 0.0
        )

        # Calculate start times
        if pd.isna(row["predicted_median_start_time"]):
            # Look for previous event's finish time
            prev_event_id = row.get("prev_predicted_event_lot_process_id")

            if pd.notna(prev_event_id) and prev_event_id in local_event_id_to_idx:
                # Find the previous event
                prev_idx = local_event_id_to_idx[prev_event_id]

                # Use previous event's finish time + wait time (in seconds) as start time
                if pd.notna(lot_data.at[prev_idx, "predicted_median_finish_time"]):
                    lot_data.at[idx, "predicted_median_start_time"] = lot_data.at[
                        prev_idx, "predicted_median_finish_time"
                    ] + timedelta(seconds=wait_time)

                if pd.notna(lot_data.at[prev_idx, "predicted_q1_finish_time"]):
                    lot_data.at[idx, "predicted_q1_start_time"] = lot_data.at[
                        prev_idx, "predicted_q1_finish_time"
                    ] + timedelta(seconds=wait_time)
                elif pd.notna(lot_data.at[idx, "predicted_median_start_time"]):
                    lot_data.at[idx, "predicted_q1_start_time"] = lot_data.at[
                        idx, "predicted_median_start_time"
                    ]

                if pd.notna(lot_data.at[prev_idx, "predicted_q3_finish_time"]):
                    lot_data.at[idx, "predicted_q3_start_time"] = lot_data.at[
                        prev_idx, "predicted_q3_finish_time"
                    ] + timedelta(seconds=wait_time)
                elif pd.notna(lot_data.at[idx, "predicted_median_start_time"]):
                    lot_data.at[idx, "predicted_q3_start_time"] = lot_data.at[
                        idx, "predicted_median_start_time"
                    ]

                # Calculate std start time using previous std finish time + std wait time
                if pd.notna(lot_data.at[prev_idx, "predicted_std_finish_time"]):
                    lot_data.at[idx, "predicted_std_start_time"] = lot_data.at[
                        prev_idx, "predicted_std_finish_time"
                    ] + timedelta(seconds=std_wait_time)
                elif pd.notna(lot_data.at[idx, "predicted_median_start_time"]):
                    lot_data.at[idx, "predicted_std_start_time"] = lot_data.at[
                        idx, "predicted_median_start_time"
                    ]
            else:
                # If median start time exists, use it for Q1, Q3, and std as well
                lot_data.at[idx, "predicted_q1_start_time"] = lot_data.at[
                    idx, "predicted_median_start_time"
                ]
                lot_data.at[idx, "predicted_q3_start_time"] = lot_data.at[
                    idx, "predicted_median_start_time"
                ]
                lot_data.at[idx, "predicted_std_start_time"] = lot_data.at[
                    idx, "predicted_median_start_time"
                ]

        # Apply hold release date constraint - no start times before October 31st
        # Use pd.notna() and ensure timezone-aware comparison
        median_start = lot_data.at[idx, "predicted_median_start_time"]
        if pd.notna(median_start):
            median_start_ts = pd.Timestamp(median_start)
            if median_start_ts < HOLD_RELEASE_DATE:
                lot_data.at[idx, "predicted_median_start_time"] = HOLD_RELEASE_DATE
                # Recalculate finish time based on adjusted start time
                lot_data.at[idx, "predicted_median_finish_time"] = (
                    HOLD_RELEASE_DATE + timedelta(seconds=median_process_duration)
                )

        q1_start = lot_data.at[idx, "predicted_q1_start_time"]
        if pd.notna(q1_start):
            q1_start_ts = pd.Timestamp(q1_start)
            if q1_start_ts < HOLD_RELEASE_DATE:
                lot_data.at[idx, "predicted_q1_start_time"] = HOLD_RELEASE_DATE
                # Recalculate finish time based on adjusted start time
                lot_data.at[idx, "predicted_q1_finish_time"] = (
                    HOLD_RELEASE_DATE + timedelta(seconds=q1_process_duration)
                )

        q3_start = lot_data.at[idx, "predicted_q3_start_time"]
        if pd.notna(q3_start):
            q3_start_ts = pd.Timestamp(q3_start)
            if q3_start_ts < HOLD_RELEASE_DATE:
                lot_data.at[idx, "predicted_q3_start_time"] = HOLD_RELEASE_DATE
                # Recalculate finish time based on adjusted start time
                lot_data.at[idx, "predicted_q3_finish_time"] = (
                    HOLD_RELEASE_DATE + timedelta(seconds=q3_process_duration)
                )

        std_start = lot_data.at[idx, "predicted_std_start_time"]
        if pd.notna(std_start):
            std_start_ts = pd.Timestamp(std_start)
            if std_start_ts < HOLD_RELEASE_DATE:
                lot_data.at[idx, "predicted_std_start_time"] = HOLD_RELEASE_DATE
                # Recalculate finish time based on adjusted start time
                lot_data.at[idx, "predicted_std_finish_time"] = (
                    HOLD_RELEASE_DATE + timedelta(seconds=std_process_duration)
                )

        # Calculate finish times based on start times + process durations (in seconds)
        # Only calculate if not already set by hold date adjustment above
        if pd.notna(lot_data.at[idx, "predicted_median_start_time"]) and pd.isna(
            lot_data.at[idx, "predicted_median_finish_time"]
        ):
            lot_data.at[idx, "predicted_median_finish_time"] = lot_data.at[
                idx, "predicted_median_start_time"
            ] + timedelta(seconds=median_process_duration)

        if pd.notna(lot_data.at[idx, "predicted_q1_start_time"]) and pd.isna(
            lot_data.at[idx, "predicted_q1_finish_time"]
        ):
            lot_data.at[idx, "predicted_q1_finish_time"] = lot_data.at[
                idx, "predicted_q1_start_time"
            ] + timedelta(seconds=q1_process_duration)

        if pd.notna(lot_data.at[idx, "predicted_q3_start_time"]) and pd.isna(
            lot_data.at[idx, "predicted_q3_finish_time"]
        ):
            lot_data.at[idx, "predicted_q3_finish_time"] = lot_data.at[
                idx, "predicted_q3_start_time"
            ] + timedelta(seconds=q3_process_duration)

        if pd.notna(lot_data.at[idx, "predicted_std_start_time"]) and pd.isna(
            lot_data.at[idx, "predicted_std_finish_time"]
        ):
            lot_data.at[idx, "predicted_std_finish_time"] = lot_data.at[
                idx, "predicted_std_start_time"
            ] + timedelta(seconds=std_process_duration)

    return lot_id, lot_data


@lightweight(memory_gb=32, cpu_cores=8)
@transform(
    input_data=Input("ri.foundry.main.dataset.02c1ac2d-5d15-4f18-95ae-2cdcdebaed4a"),
    output_data=Output(
        "/LG Innotek-0e7800/[WF]SCM Planning Intelligence/data/objects/Predicted Event Lot Through Process Enhanced"
    ),
)
def compute_predicted_times(input_data, output_data):
    """
    Calculate missing predicted start and finish times (median, Q1, Q3, std) by chaining events together.
    Each event's start time = previous event's finish time + wait time.
    Each event's finish time = start time + process duration.

    Time types:
    - median/Q1/Q3: Use historical durations (predicted_total_median/Q1/Q3_panel_in_seconds)
                    and historical wait time (median_prev_process_wait_in_seconds)
    - std: Use standard durations (predicted_total_std_time_per_panel_in_s)
           and standard wait time (std_wait_time_in_s)

    All durations and wait times are in seconds.

    Also propagates quantity information (units, sheets, panels, strips) from the input dataset.
    These quantities represent the latest known quantities and will decrease through the process
    due to inefficiencies/scrap.

    Quantity columns are converted from doubles to integers using ceiling function to round up
    fractional quantities.

    Optimized with parallel processing per lot using ProcessPoolExecutor.

    Hold logic: Any start times calculated before October 31st, 2025 (Korea timezone) are adjusted to
    October 31st, representing when lots come off hold. This applies to all time types (median, Q1, Q3, std).
    """

    # Read the input data as pandas DataFrame
    df = input_data.pandas()

    print(f"Processing {len(df)} rows")

    # Convert timestamp columns to datetime if they're in milliseconds
    if "predicted_median_start_time" in df.columns:
        df["predicted_median_start_time"] = pd.to_datetime(
            df["predicted_median_start_time"], unit="ms", errors="coerce"
        )
    if "predicted_median_finish_time" in df.columns:
        df["predicted_median_finish_time"] = pd.to_datetime(
            df["predicted_median_finish_time"], unit="ms", errors="coerce"
        )

    # Add new columns for Q1, Q3, and std times (as timestamps)
    df["predicted_q1_start_time"] = pd.NaT
    df["predicted_q3_start_time"] = pd.NaT
    df["predicted_q1_finish_time"] = pd.NaT
    df["predicted_q3_finish_time"] = pd.NaT
    df["predicted_std_start_time"] = pd.NaT
    df["predicted_std_finish_time"] = pd.NaT

    # Add source and timestamp columns
    df["prediction_source"] = "wip simulation"
    df["simulation_run_ts"] = datetime.now()

    # Group by lot_id for parallel processing
    lot_groups = df.groupby("lot_id", sort=False)

    print(f"Processing {len(lot_groups)} lots in parallel")

    # Prepare data for multiprocessing - convert each lot's DataFrame to dict
    lot_data_list = [
        (lot_id, lot_data.to_dict("list")) for lot_id, lot_data in lot_groups
    ]

    # Process lots in parallel using ProcessPoolExecutor for true parallelism
    processed_lots_dict = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit all lot processing tasks
        futures = {
            executor.submit(process_single_lot, lot_tuple): lot_tuple[0]
            for lot_tuple in lot_data_list
        }

        # Collect results as they complete
        for future in as_completed(futures):
            lot_id = futures[future]
            try:
                returned_lot_id, processed_lot = future.result()
                processed_lots_dict[returned_lot_id] = processed_lot
                print(f"Completed processing lot {returned_lot_id}")
            except Exception as exc:
                print(f"Lot {lot_id} generated an exception: {exc}")
                raise

    # Reconstruct DataFrame in the original lot order
    processed_lots = [processed_lots_dict[lot_id] for lot_id, _ in lot_data_list]
    df = pd.concat(processed_lots, ignore_index=True)

    # Sort back to original order (by lot_id and planned_work_sequence)
    df = df.sort_values(["lot_id", "planned_work_sequence"]).reset_index(drop=True)

    # Convert quantity columns from doubles to integers using ceiling
    quantity_columns = [
        "latest_unit_quantity",
        "latest_sheet_quantity",
        "latest_panel_quantity",
        "latest_strip_quantity",
    ]

    for col in quantity_columns:
        if col in df.columns:
            # Apply ceiling then convert to integer type, handling NaN values
            df[col] = df[col].apply(
                lambda x: int(np.ceil(x)) if pd.notna(x) else np.nan
            )
            # Explicitly convert to Int64 (nullable integer type)
            df[col] = df[col].astype("Int64")

    print(f"Finished processing, writing {len(df)} rows")

    # Write the enhanced dataset
    # Quantity columns are converted from doubles to integers using ceiling and propagated
    # through all predicted events for each lot
    output_data.write_pandas(df)
