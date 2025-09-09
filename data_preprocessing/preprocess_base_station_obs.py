import pandas as pd
from datetime import timedelta
import logging
import matplotlib.pyplot as plt

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

def interpolate_or_trend_transfer(df_local, temp_dir):
    """
    Interpolate local base station data to 1s resolution.
    - If local data has 1-min resolution → direct linear interpolation.
    - Else → apply GRU trend transfer with Dalat reference.
    """
    linear_interpolate_flag = 2 #initilize
    local_time_diff = (
        df_local["Time"]
        .sort_values()          
        .diff()                   
        .dropna()                  
        .dt.total_seconds()      
        .mode()[0]             
    )

    if local_time_diff <= 60:  # ~1-min resolution
        logging.info("Local base station has <= 60s spacing. Using linear interpolation to 1-sec.")
        logging.info("Skipping model trend transfer.")
        
        df_local = df_local.set_index("Time")
        full_index = pd.date_range(df_local.index.min(), df_local.index.max(), freq="1S")
        df_local = df_local.reindex(full_index)
        df_local["interpolated_magnetic_reading"] = df_local["Magnetic Reading"].interpolate(method="linear")
        df_local = df_local.reset_index().rename(columns={"index": "Time"})
  
        df_local.to_csv(f'{temp_dir}/linear_interpolated_base_station.csv', index=False)

        linear_interpolate_flag = 1

        plt.figure(figsize=(10,5))
        plt.plot(df_local["Time"], df_local["interpolated_magnetic_reading"], label="Linear Interpolated", color="tab:blue")
        plt.scatter(df_local["Time"], df_local["Magnetic Reading"], s=10, color="tab:red", alpha=0.6, label="Data Points")
        plt.title("Local Base Station Magnetic Reading (Linear Interpolation)")
        plt.xlabel("Time")
        plt.ylabel("Magnetic Reading (nT)")
        plt.legend()
        plt.tight_layout()
        plot_path = f"{temp_dir}/linear_interpolation_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info(f"Saved interpolation plot into {plot_path}")

        df_local = df_local[['Time','interpolated_magnetic_reading']].copy()

    else:
      logging.info("Local base station has > 60s spacing. Using Trend Trasnfer interpolation to 1-sec.")
      linear_interpolate_flag = 0

    return linear_interpolate_flag, df_local
