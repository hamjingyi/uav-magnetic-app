import pandas as pd
import numpy as np
import logging
import os
import re

def smart_parse_date(x):
    if pd.isna(x):
        return pd.NaT
    x = str(x).strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", x):  # matches 2023-05-10
        return x
    else:  # assume day/month/year
        return pd.to_datetime(x,format='%d/%m/%Y', errors='coerce')


def resample_magnetic_data(input_dir: str, output_dir: str, mean_column='Total_field_anomaly__nT_', exclu_column = ['Date', 'Time_hh:mm:ss']):

    df = pd.read_csv(input_dir, low_memory=False)
    logging.info(f"Original magnetic signalling shape: {df.shape}")
    logging.info(f"Original columns name: {df.columns.tolist()}")

    # cols = ['Date', 'Time_hh:mm:ss', 'Line', 'Timestamp__ms_', 'Total_field_anomaly__nT_',
    #         'Longitude__°_', 'Latitude__°_', 'Easting', 'Northing', 'Altitude__m_']

    # df = df[cols].copy()

    base_cols = [
        'Date',
        'Timestamp__ms_',
        'Time_hh:mm:ss',
        'Line',
        'Total_field_anomaly__nT_',
        'Longitude__°_',
        'Latitude__°_',
        'Easting',
        'Northing',
        'Altitude__m_'
    ]

    def normalize_columns(df, base_cols):
        rename_map = {}
        for col in df.columns:
            col_clean = col.replace("_", "").lower()
            for base in base_cols:
                if base.replace("_", "").lower() in col_clean or col_clean in base.replace("_", "").lower():
                    rename_map[col] = base
                    break
        df = df.rename(columns=rename_map)
        return df.reindex(columns=[c for c in base_cols if c in df.columns], fill_value=pd.NA)

    df = normalize_columns(df, base_cols)
    logging.info(f"After normalizing columns: {df.columns.tolist()}")

    df = df.dropna(how='all')
    logging.info(f"After dropping empty rows: {df.shape}")

    agg_dict = {
        col: (np.mean if col == mean_column else 'first')
        for col in df.columns
        if col not in exclu_column
    }

    # df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%Y', errors='coerce')
    df['Date'] = df['Date'].apply(smart_parse_date)

    df_resampled = df.groupby(exclu_column).agg(agg_dict).reset_index()

    if "Date" in df.columns:
        # Combine Date + Time
        df_resampled["DateTime"] = pd.to_datetime(
            df_resampled["Date"].astype(str) + " " + df_resampled["Time_hh:mm:ss"].astype(str),
            errors="coerce"
        )
    else:
        # Only Time
        df_resampled["DateTime"] = pd.to_datetime(df_resampled['Time_hh:mm:ss'], format='%H:%M:%S', errors='coerce' ).dt.time

    # drop old Time_hh:mm:ss and replace
    df_resampled['Date'] = pd.to_datetime(df_resampled['Date'], errors='coerce')
    df_resampled.drop(columns=["Time_hh:mm:ss"], inplace=True)
    df_resampled.rename(columns={"DateTime": "Time_hh:mm:ss"}, inplace=True)

    cols = ["Time_hh:mm:ss"] + [c for c in df_resampled.columns if c != "Time_hh:mm:ss"]
    df_resampled = df_resampled[cols]

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, '1_sec_resampled_magnetic_data.csv')
    df_resampled.to_csv(output_path, index=False)
    logging.info(f"Resampled file saved to: {output_path}")

    return df_resampled

