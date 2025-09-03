import pandas as pd
from datetime import timedelta
import logging

def read_local_csv(csv_path):
    logging.info(f"Reading local base station observation data")
    return pd.read_csv(csv_path)

def preprocess_and_align(
    df_local_raw,
    df_dalat,
    local_time_col="Time",          # Original time column in local CSV
    dalat_time_col="Time_Malaysia"
):
    logging.info("Preprocessing local base station observation data")
    df_local = df_local_raw.copy()

    # --- Safe parsing of local time ---
    if pd.api.types.is_datetime64_any_dtype(df_local["Date"]):
        date_str = df_local["Date"].dt.strftime("%Y-%m-%d")
    else:
        date_str = df_local["Date"].astype(str).str.strip()

    time_str = df_local[local_time_col].astype(str).str.strip()
    combined = date_str + " " + time_str

    # Let pandas auto-detect format (handles both HH:MM and HH:MM:SS)
    df_local["Time"] = pd.to_datetime(combined, errors="coerce")

    # --- Get per-day start and end from local ---
    local_day_ranges = (
        df_local.groupby(df_local["Time"].dt.normalize())["Time"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"Time": "Date", "min": "Start", "max": "End"})
    )
    logging.info(f"Local day ranges:\n{local_day_ranges}")

    start_time_global = local_day_ranges["Start"].min()
    df_local["Seconds"] = (df_local["Time"] - start_time_global).dt.total_seconds()

    # --- Filter Dalat according to each day's window ---
    df_dalat = df_dalat.copy()
    df_dalat[dalat_time_col] = pd.to_datetime(df_dalat[dalat_time_col], errors="coerce")

    mask = pd.Series(False, index=df_dalat.index)
    for _, row in local_day_ranges.iterrows():
        day_mask = (df_dalat[dalat_time_col] >= row["Start"]) & (df_dalat[dalat_time_col] <= row["End"])
        mask |= day_mask

    df_dalat_filtered = df_dalat[mask].copy()
    df_dalat_filtered["Seconds"] = (
        df_dalat_filtered[dalat_time_col] - start_time_global
    ).dt.total_seconds()

    return df_local, df_dalat_filtered

# def preprocess_and_align(
#     df_local_raw,
#     df_dalat,
#     local_time_col="Time",          # Original time column in local CSV
#     dalat_time_col="Time_Malaysia"
# ):
#     logging.info("Preprocessing local base station observation data")
#     df_local = df_local_raw.copy()

#     # Parse local time
#     df_local["Time"] = pd.to_datetime(
#         df_local["Date"] + " " + df_local[local_time_col].astype(str),
#         format="%Y-%m-%d %H:%M", errors="coerce"
#     )

#     # --- Get per-day start and end from local ---
#     local_day_ranges = (
#         df_local.groupby(df_local["Time"].dt.normalize())["Time"]
#         .agg(["min", "max"])
#         .reset_index()
#         .rename(columns={"Time": "Date", "min": "Start", "max": "End"})
#     )
#     logging.info(f"Local day ranges:\n{local_day_ranges}")

#     start_time_global = local_day_ranges["Start"].min()
#     df_local["Seconds"] = (df_local["Time"] - start_time_global).dt.total_seconds()

#     # --- Filter Dalat according to each day's window ---
#     df_dalat = df_dalat.copy()
#     df_dalat[dalat_time_col] = pd.to_datetime(df_dalat[dalat_time_col], errors="coerce")

#     mask = pd.Series(False, index=df_dalat.index)
#     for _, row in local_day_ranges.iterrows():
#         day_mask = (df_dalat[dalat_time_col] >= row["Start"]) & (df_dalat[dalat_time_col] <= row["End"])
#         mask |= day_mask

#     df_dalat_filtered = df_dalat[mask].copy()
#     df_dalat_filtered["Seconds"] = (
#         df_dalat_filtered[dalat_time_col] - start_time_global
#     ).dt.total_seconds()

#     return df_local, df_dalat_filtered

# def preprocess_and_align(
#     df_local_raw,
#     df_dalat,
#     local_time_col="Time",          # Original time column in local CSV
#     dalat_time_col="Time_Malaysia"
# ):
#     logging.info(f"Preprocessing local base station observation data")
#     df_local = df_local_raw.copy()

#     # Parse local time with date
#     df_local["Time"] = pd.to_datetime(
#         df_local['Date'] + " " + df_local[local_time_col].astype(str),
#         format="%Y-%m-%d %H:%M", errors="coerce"
#     )

#     minutes = 0
#     logging.info(f"Extend {minutes} minutes for interpolation model to have more surrounding trend context")

#     start_time = df_local["Time"].min() - timedelta(minutes=minutes)
#     end_time   = df_local["Time"].max() + timedelta(minutes=minutes)
#     logging.info(f"Start time: {start_time}, End time: {end_time}")

#     df_local["Seconds"] = (df_local["Time"] - start_time).dt.total_seconds()

#     # Filter Dalat data
#     df_dalat = df_dalat.copy()
#     df_dalat[dalat_time_col] = pd.to_datetime(df_dalat[dalat_time_col], errors="coerce")
#     df_dalat_filtered = df_dalat[
#         (df_dalat[dalat_time_col] >= start_time) &
#         (df_dalat[dalat_time_col] <= end_time)
#     ].copy()
#     df_dalat_filtered["Seconds"] = (df_dalat_filtered[dalat_time_col] - start_time).dt.total_seconds()

#     return df_local, df_dalat_filtered
