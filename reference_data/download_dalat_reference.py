import requests
import pandas as pd
import numpy as np
import os
import logging

def download_dalat_data_fast(
    start_date: str,
    duration_days: int = 1,
    output_dir: str = None,
    filename: str = 'dalat_base_station.csv'
) -> pd.DataFrame:
    """
    Quickly downloads and parses Dalat INTERMAGNET base station HTML table using pandas.read_html() directly.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        duration_days (int): Number of days to fetch (default = 1).
        output_dir (str): Folder to save CSV file (optional).
        filename (str): Output CSV filename (default = 'dalat_base_station.csv').

    Returns:
        pd.DataFrame: Retrieved magnetic data table.
    """
    url = (
        "https://imag-data.bgs.ac.uk/GIN_V1/GINServices?"
        f"Request=GetData&format=HTML&testObsys=0&observatoryIagaCode=DLT"
        f"&samplesPerDay=second&publicationState=Best%20available"
        f"&dataStartDate={start_date}&dataDuration={duration_days}&orientation=native"
    )

    try:
        logging.info(f"Retrieving Dalat data for {start_date} ({duration_days} days)")
        response = requests.get(url)
        response.raise_for_status()

        tables = pd.read_html(response.text)
        if not tables:
            logging.warning("No tables found in the HTML response.")
            return pd.DataFrame()

        df = tables[0]
        logging.info(f"Dalat data retrieved for {start_date} ({duration_days} days)")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
            df.to_csv(save_path, index=False)
            logging.info(f"Saved to: {save_path}")

        return df

    except Exception as e:
        logging.error(f"Failed to fetch Dalat data: {e}")
        raise


def process_dalat_data(
    input_csv: str = None,
    df: pd.DataFrame = None,
    *,
    output_csv: str = None,
    time_col: str = "Time",
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    target_tz: str = "Asia/Ho_Chi_Minh",   
    extra_offset_hours: int = 0,          
    drop_rows_missing_xyz: bool = True
) -> pd.DataFrame:
    """
    Clean and process Dalat base station data:
    - Normalize HTML non-breaking spaces to NaN
    - Coerce X/Y/Z to numeric and compute N = sqrt(X^2 + Y^2 + Z^2)
    - Parse Time as UTC, add optional manual offset, convert to target timezone (naive)
    - Optionally save to CSV

    Provide either input_csv OR df.

    Returns:
        pd.DataFrame: Processed DataFrame with columns:
            [time_col, x_col, y_col, z_col, 'N', 'Time_local']
    """
    if (input_csv is None) and (df is None):
        raise ValueError("Provide either input_csv or df.")

    if df is None:
        logging.info(f"Loading Dalat CSV: {input_csv}")
        df = pd.read_csv(input_csv, low_memory=False)

    df = df.replace("\xa0", pd.NA)

    for c in (x_col, y_col, z_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in DataFrame.")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if drop_rows_missing_xyz:
        before = len(df)
        df = df.dropna(subset=[x_col, y_col, z_col])
        logging.info(f"Dropped {before - len(df)} rows with missing {x_col}/{y_col}/{z_col}")

    df["N"] = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)

    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in DataFrame.")

    t_utc = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    if extra_offset_hours != 0:
        t_utc = t_utc + pd.Timedelta(hours=extra_offset_hours)

    df["Time_Malaysia"] = t_utc.dt.tz_convert(target_tz).dt.tz_localize(None)

    logging.info(f"Processed Dalat data → Time zone: {target_tz}, offset applied: {extra_offset_hours}h")

    if output_csv:
        outdir = os.path.dirname(output_csv)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        # df.to_csv(output_csv, index=False)
        logging.info(f"Saved processed CSV to: {output_csv}")

    return df

