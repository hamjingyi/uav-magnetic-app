import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import logging

def savgol_filter_microleveling(df_all: pd.DataFrame,
                              base_col: str = 'direction_corrected_huber',
                              clamp_val: float = 5.0) -> pd.DataFrame:
    """
    Perform safe Savitzky-Golay microleveling (across-track, zero-mean, clamped).

    Parameters
    ----------
    df_all : pd.DataFrame
        Input DataFrame with columns:
            ['Line','Easting','Northing','date', base_col]
    base_col : str
        The magnetic field column to be microleveled (default='TMI_direction_corrected_huber').
    clamp_val : float
        Maximum absolute bias correction in nT (default=5.0).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
            'micro_bias_sg' : per-line correction bias
            'TMI_micro_sg'  : baseline-preserved corrected field
    """
    assert base_col in df_all.columns, f"{base_col} column not found in DataFrame!"

    df = df_all.copy()
    logging.info(f"Starting microleveling on {len(df)} samples across {df['Line'].nunique()} lines.")

    # 1) Per-line statistics
    line_med = df.groupby('Line', sort=False)[base_col].median().astype(float)
    line_n   = df.groupby('Line', sort=False)[base_col].size().astype(float)
    line_x   = df.groupby('Line', sort=False)['Easting'].median().astype(float)  # across-track axis
    line_day = df.groupby('Line', sort=False)['date'].first() if 'date' in df.columns else pd.Series('ALL', index=line_med.index)

    # 2) Sort lines by across-track position
    order = line_x.sort_values().index
    m = line_med.loc[order].to_numpy()
    logging.info(f"Computed median per line. Window smoothing over {len(m)} lines.")

    # 3) Smooth striping trend
    win = max(5, (len(m) // 5) * 2 + 1)  # ensures odd length ~5–11
    win = min(win, len(m) - (1 - (len(m) % 2)))  # adjust if too large
    polyorder = 1
    trend = savgol_filter(m, window_length=win, polyorder=polyorder, mode='interp')
    logging.info(f"Applied Savitzky-Golay filter with window={win}, polyorder={polyorder}.")

    # 4) Per-line microlevel bias = median - smoothed trend
    bias = m - trend
    bias = pd.Series(bias, index=order)

    # 5) Zero-mean per day and clamp
    for day, idx in line_day.groupby(line_day).groups.items():
        idx = pd.Index(idx).intersection(bias.index)
        w = line_n.loc[idx].to_numpy()
        mu = float(np.average(bias.loc[idx].to_numpy(), weights=w))
        bias.loc[idx] -= mu
        logging.info(f"Day {day}: applied zero-mean leveling, mean shift={mu:.3f} nT.")

    bias = bias.clip(-clamp_val, clamp_val)
    logging.info(f"Clamped bias to ±{clamp_val} nT.")

    # 6) Map back to samples
    df_all['micro_bias_sg'] = df_all['Line'].map(bias).astype('float32').fillna(0.0)
    df_all['TMI_micro_sg']  = df_all[base_col].astype('float32') - df_all['micro_bias_sg']

    logging.info("Microleveling completed. Added columns: ['TMI_micro_sg'].")
    

    return df_all
