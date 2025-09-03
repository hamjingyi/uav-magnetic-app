import torch, time, psutil, math, logging
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter1d
from scipy.signal import detrend
import os
import streamlit as st

class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


def run_LSTM_trend_transfer(df_dalat: pd.DataFrame,
                        df_local: pd.DataFrame,
                        dalat_col: str = "N",
                        local_col: str = "Magnetic Reading",
                        seq_len: int = 60,
                        sigma_smooth: float = 10,
                        lr: float = 5e-4,
                        batch_size: int = 512,
                        epochs: int = 20,
                        plot: bool = True,
                        temp_dir: str = None):
    
    logging.info(f"LSTM: Start Trend Transfering")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Start LSTM trend transfer using device: {device}")

    # === Step 1: Smooth and scale Dalat data ===
    dalat_values = df_dalat[dalat_col].ffill().values.reshape(-1, 1)
    dalat_values_smoothed = gaussian_filter1d(dalat_values.flatten(), sigma=sigma_smooth).reshape(-1, 1)
    scaler_dalat = MinMaxScaler()
    dalat_scaled = scaler_dalat.fit_transform(dalat_values_smoothed)

    # === Step 2: Create sequences ===
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return torch.tensor(X).float(), torch.tensor(y).float()

    X_dalat, y_dalat = create_sequences(dalat_scaled, seq_len)
    X_dalat, y_dalat = X_dalat.to(device), y_dalat.to(device)

    # === Step 3: LSTM model ===
    model = LSTMNet().to(device)
    dataset = TensorDataset(X_dalat, y_dalat)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # === Step 4: Training ===
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f}")

    # === Step 5: Prediction for full range ===
    full_time = pd.date_range(start=df_local["Time"].min(), end=df_local["Time"].max(), freq="1s")
    dummy_input = dalat_scaled[:len(full_time)]
    X_pred_input, _ = create_sequences(dummy_input, seq_len)
    X_pred_input = X_pred_input.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(X_pred_input).cpu().numpy()
    pred_full = np.concatenate([dalat_scaled[:seq_len].flatten(), pred.flatten()])

    # === Step 6: Back to original Dalat scale ===
    pred_trend_dalat = scaler_dalat.inverse_transform(pred_full.reshape(-1, 1)).flatten()

    # === Step 7: Normalize shape and smooth ===
    trend_shape = (pred_trend_dalat - np.mean(pred_trend_dalat)) / np.std(pred_trend_dalat)
    trend_smooth = gaussian_filter1d(trend_shape, sigma=1)

    # === Step 8: Rescale to local range ===
    local_min, local_max = df_local[local_col].min(), df_local[local_col].max()
    scaled_pattern = (trend_smooth - np.min(trend_smooth)) / (np.max(trend_smooth) - np.min(trend_smooth))
    trend_scaled = scaled_pattern * (local_max - local_min) + local_min

    # === Step 9: Evaluation ===
    gt_dalat = dalat_values[:len(pred_trend_dalat)].flatten()
    metrics_dalat = {
        "RMSE": np.sqrt(mean_squared_error(gt_dalat, pred_trend_dalat)),
        "MAE": mean_absolute_error(gt_dalat, pred_trend_dalat),
        "R2": r2_score(gt_dalat, pred_trend_dalat)
    }

    interp_values = np.interp(df_local["Time"].astype(np.int64),
                              pd.Series(full_time).astype(np.int64),
                              trend_scaled)
    metrics_local = {
        "RMSE": np.sqrt(mean_squared_error(df_local[local_col], interp_values)),
        "MAE": mean_absolute_error(df_local[local_col], interp_values),
        "R2": r2_score(df_local[local_col], interp_values)
    }

    # === Step 10: Plot if requested ===
    if plot:
        plt.figure(figsize=(14, 5))
        plt.plot(pd.to_datetime(full_time), trend_scaled, label="LSTM Interpolated Trend (Rescaled)", linewidth=2)
        plt.plot(pd.to_datetime(df_dalat["Time_Malaysia"]), df_dalat[dalat_col], label="Dalat Base Trend", alpha=0.5)
        plt.scatter(pd.to_datetime(df_local["Time"]), df_local[local_col], color='red', label="Sparse Local Readings", zorder=3)
        plt.title("Sparse Local Interpolation Guided by LSTM Pattern from Dalat")
        plt.xlabel("Time")
        plt.ylabel("Magnetic Reading")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if temp_dir is not None:
          save_path = os.path.join(temp_dir, "LSTM_base_station_trend_plot.png")
          plt.savefig(save_path, dpi=300)
          logging.info(f"LSTM trend plot saved to {save_path}")

        plt.show()

    # === Step 11: Memory info & cleanup ===
    cpu_mem = psutil.virtual_memory()
    logging.info(f"CPU Memory Used: {(cpu_mem.total - cpu_mem.available) / 1e9:.2f} GB / {cpu_mem.total / 1e9:.2f} GB")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"CUDA Memory Allocated: {allocated:.2f} GB")
        logging.info(f"CUDA Memory Reserved: {reserved:.2f} GB")
        logging.info(f"CUDA Memory Total: {total:.2f} GB")

    end_time = time.time()
    logging.info(f"Total Time Elapsed: {end_time - start_time:.2f} seconds")

    del X_pred_input, model, optimizer, X_dalat, y_dalat
    torch.cuda.empty_cache()

    df_output = pd.DataFrame({
        "Time": full_time,
        "interpolated_magnetic_reading": trend_scaled
    })

    if temp_dir is not None:
      csv_path = os.path.join(temp_dir, "LSTM_interpolated_base_station_data.csv")
      df_output.to_csv(csv_path, index=False)
      logging.info(f"LSTM interpolated DataFrame saved to {csv_path}")

    return df_output

def run_GRU_trend_transfer(df_dalat, df_local,
                        temp_dir: str = None):

    logging.info(f"GRU: Start Trend Transfering")
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    def make_time_features(n, period=86400):
        t = np.arange(n, dtype=np.float32)
        ang = 2 * np.pi * (t % period) / period
        return np.stack([np.sin(ang), np.cos(ang)], axis=1)

    def create_seq_multifeat(series_1d, time_feats, seq_len):
        X, y = [], []
        for i in range(len(series_1d) - seq_len):
            window_vals = series_1d[i:i+seq_len].reshape(-1, 1)
            window_ts = time_feats[i:i+seq_len]
            X.append(np.concatenate([window_vals, window_ts], axis=1))
            y.append(series_1d[i+seq_len])
        return torch.tensor(np.stack(X)).float(), torch.tensor(np.array(y)).float().unsqueeze(1)

    dalat_values = df_dalat["N"].ffill().values.reshape(-1, 1)
    dalat_values_smoothed = gaussian_filter1d(dalat_values.flatten(), sigma=10).reshape(-1, 1)
    scaler_dalat = MinMaxScaler()
    dalat_scaled = scaler_dalat.fit_transform(dalat_values_smoothed).flatten()

    seq_len = 24
    time_feats = make_time_features(len(dalat_scaled), period=86400)
    X_dalat, y_dalat = create_seq_multifeat(dalat_scaled, time_feats, seq_len)

    n = len(X_dalat)
    n_val = max(1, int(0.10 * n))
    X_tr, y_tr = X_dalat[:-n_val], y_dalat[:-n_val]
    X_va, y_va = X_dalat[-n_val:], y_dalat[-n_val:]

    class GRUPlus(nn.Module):
        def __init__(self, input_size=3, hidden_size=20, num_layers=1, dropout=0.0):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    model = GRUPlus().to(device)
    batch_size = 32
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    epochs = 16
    early_patience = 4
    best_val = float("inf")
    pat = 0
    best_state = None

    logging.info("Model Training: GRU+ …")
    for ep in range(1, epochs+1):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                vloss += criterion(model(xb), yb).item()
        logging.info(f"Epoch {ep}/{epochs} - Train: {ep_loss:.6f}  Val: {vloss:.6f}")
        scheduler.step(vloss)
        if vloss < best_val - 1e-6:
            best_val = vloss
            pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= early_patience:
                logging.info("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize()

    full_time = pd.date_range(start=df_local["Time"].min(), end=df_local["Time"].max(), freq="1s")
    Nfull = len(full_time)
    dummy_scaled = dalat_scaled[:Nfull]
    dummy_feats = make_time_features(Nfull, period=86400)

    def make_X_pred(series_1d, time_feats, seq_len):
        X = []
        for i in range(len(series_1d) - seq_len):
            win_vals = series_1d[i:i+seq_len].reshape(-1, 1)
            win_ts = time_feats[i:i+seq_len]
            X.append(np.concatenate([win_vals, win_ts], axis=1))
        return torch.tensor(np.stack(X)).float()

    X_pred = make_X_pred(dummy_scaled, dummy_feats, seq_len).to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_pred).detach().cpu().numpy().flatten()

    pred_full_scaled = np.concatenate([dummy_scaled[:seq_len], pred_scaled])
    pred_trend_dalat = scaler_dalat.inverse_transform(pred_full_scaled.reshape(-1, 1)).flatten()

    trend_shape = (pred_trend_dalat - np.mean(pred_trend_dalat)) / (np.std(pred_trend_dalat) + 1e-12)
    trend_smooth = gaussian_filter1d(trend_shape, sigma=1)

    local_min = df_local["Magnetic Reading"].min()
    local_max = df_local["Magnetic Reading"].max()
    scaled_pattern = (trend_smooth - np.min(trend_smooth)) / (np.max(trend_smooth) - np.min(trend_smooth) + 1e-12)
    trend_scaled = scaled_pattern * (local_max - local_min) + local_min
    trend_scaled = (trend_scaled - trend_scaled.mean()) * 1.10 + trend_scaled.mean()

    dalat_true = dalat_values_smoothed[:len(pred_trend_dalat)].flatten()
    rmse_dalat = np.sqrt(mean_squared_error(dalat_true, pred_trend_dalat))
    mae_dalat = mean_absolute_error(dalat_true, pred_trend_dalat)
    r2_dalat = r2_score(dalat_true, pred_trend_dalat)
    logging.info("GRU+ vs. Dalat")
    logging.info(f"RMSE: {rmse_dalat:.4f}  MAE: {mae_dalat:.4f}  R²: {r2_dalat:.4f}")

    df_interp = pd.DataFrame({'Time': full_time[:len(trend_scaled)], 'Pred': trend_scaled}).set_index("Time")
    local_time = df_local["Time"]
    interpolated = df_interp.reindex(df_interp.index.union(local_time)).interpolate("time").loc[local_time]

    local_true = df_local["Magnetic Reading"].values
    local_pred = interpolated["Pred"].values

    rmse_local = np.sqrt(mean_squared_error(local_true, local_pred))
    mae_local = mean_absolute_error(local_true, local_pred)
    r2_local = r2_score(local_true, local_pred)
    logging.info("Interpolation vs. Sparse Local")
    logging.info(f"RMSE: {rmse_local:.4f}  MAE: {mae_local:.4f}  R²: {r2_local:.4f}")

    survey_date = df_local["Time"].dt.date.min() 

    plt.figure(figsize=(14, 5))
    plt.plot(full_time[:len(trend_scaled)], trend_scaled, label="GRU+ Interpolated Trend", linewidth=2)
    plt.plot(df_dalat["Time_Malaysia"], df_dalat["N"], label="Dalat Base Trend", alpha=0.5)
    plt.scatter(df_local["Time"], df_local["Magnetic Reading"], color='red', label="Sparse Local Readings", zorder=3)
    plt.title("Sparse Local Interpolation Guided by GRU+ (with time features)")
    plt.xlabel("Time")
    plt.ylabel("Magnetic Reading")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if temp_dir is not None:
      save_path = os.path.join(temp_dir, f"GRU_trend_{survey_date}.png")
      plt.savefig(save_path, dpi=300)
      logging.info(f"{survey_date} GRU trend plot saved to {save_path}")
    plt.show()

    cpu_mem = psutil.virtual_memory()
    logging.info(f"CPU Memory Used: {round((cpu_mem.total - cpu_mem.available) / 1e9, 2)} GB / {round(cpu_mem.total / 1e9, 2)} GB")
    if device.type == "cuda":
        logging.info(f"CUDA Memory Allocated: {round(torch.cuda.memory_allocated(0) / 1e9, 2)} GB")
        logging.info(f"CUDA Memory Reserved:  {round(torch.cuda.memory_reserved(0) / 1e9, 2)} GB")
        logging.info(f"CUDA Total:            {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")

    if device.type == "cuda":
        torch.cuda.synchronize()
    logging.info(f"Total Time Elapsed: {time.time() - start_time:.2f} seconds")

    df_output = pd.DataFrame({
        "Time": full_time,
        "interpolated_magnetic_reading": trend_scaled
    })

    if temp_dir is not None:
      csv_path = os.path.join(temp_dir, f"GRU_interpolated_{survey_date}.csv")
      df_output.to_csv(csv_path, index=False)
      logging.info(f"GRU interpolated DataFrame saved to {csv_path}")

    return df_output


def per_day_GRU_transfer(df_dalat_filtered, df_local, run_GRU_trend_transfer, temp_dir=None):
    """
    Apply GRU trend transfer per day and combine results.

    Parameters:
        df_dalat_filtered (pd.DataFrame): Dense Dalat station data (multi-day).
        df_local (pd.DataFrame): Sparse local base station data (multi-day).
        run_GRU_trend_transfer (func): Function to perform GRU trend transfer for a single day.
        temp_dir (str): Optional temp directory to save plots.

    Returns:
        pd.DataFrame: Combined per-day interpolated local data.
    """
    df_local['Date'] = pd.to_datetime(df_local['Date']).dt.date
    df_dalat_filtered['Date'] = pd.to_datetime(df_dalat_filtered['Time_Malaysia']).dt.date

    # Store per-day results
    dfs_interp = []

    for day in sorted(df_local['Date'].unique()):
        logging.info("Apply GRU trend transfer day by day, aligning each day individually.")
        logging.info(f"Running {day} trend tranfer...")
        df_local_day = df_local[df_local['Date'] == day].copy()
        df_dalat_day = df_dalat_filtered[df_dalat_filtered['Date'] == day].copy()

        if df_dalat_day.empty or df_local_day.empty:
            continue  # skip if no data

        # Run GRU trend transfer for this single day
        df_interp_day = run_GRU_trend_transfer(df_dalat_day, df_local_day, temp_dir=temp_dir)
        dfs_interp.append(df_interp_day)

    # Combine all days into one dataframe
    df_interp_combined = pd.concat(dfs_interp, ignore_index=True)

    return df_interp_combined


def apply_diurnal_correction(df_local, df_dalat_interp, time_col_local='Time_hh:mm:ss', time_col_dalat='Time',
                             local_mag_col='Total_field_anomaly__nT_', interp_mag_col='interpolated_magnetic_reading',
                             corrected_col_name='diurnal_correction_magnetic'):

    try:
        logging.info(f"Signaling magnetic shape: {df_local.shape}")
        df_local["Time_hh:mm:ss"] = pd.to_datetime(df_local["Time_hh:mm:ss"])
        df_dalat_interp["Time"] = pd.to_datetime(df_dalat_interp["Time"])

        df_combined = df_local.merge(
            df_dalat_interp,
            left_on=[time_col_local],
            right_on=[time_col_dalat],
            how='left'
        )
        logging.info(f"Combined shape after merge: {df_combined.shape}")

        # Apply diurnal correction
        df_combined[corrected_col_name] = (
            df_combined[local_mag_col] - df_combined[interp_mag_col]
        )
        logging.info(f"Diurnal correction applied: new column '{corrected_col_name}' added.")
        return df_combined

    except Exception as e:
        logging.error(f"Diurnal correction failed: {e}")
        raise



def partial_detrend(df, value_col, time_col, chosen_date, start_time, end_time,
                    tick_interval=300, plot=True, temp_dir: str = None):

    df_out = df.copy()

    # Ensure 'Date' and 'Time_obj' columns exist and have correct type
    df_out['Date'] = pd.to_datetime(df_out['Date']).dt.date
    df_out['Time_obj'] = pd.to_datetime(df_out[time_col], format='%H:%M:%S').dt.time

    # Ensure the partial_detrended column exists
    df_out['partial_detrended'] = df_out[value_col].ffill().astype(float)

    # Filter only rows for the chosen_date
    df_date = df_out[df_out['Date'] == chosen_date].copy()

    # Convert start/end times to datetime.time
    start_t = pd.to_datetime(start_time, format='%H:%M:%S').time()
    end_t = pd.to_datetime(end_time, format='%H:%M:%S').time()

    # Mask for detrending window
    mask = df_date['Time_obj'].between(start_t, end_t)
    if not mask.any():
        raise ValueError(f"No data found in specified time window for {chosen_date}, {start_time}-{end_time}")

    # Detrend only the selected segment
    y_segment = df_date.loc[mask, value_col].ffill().astype(float).values
    detrended_segment = detrend(y_segment, type='linear')
    df_date.loc[mask, 'partial_detrended'] = detrended_segment

    # Merge back the detrended segment into the full dataframe
    df_out.update(df_date[['partial_detrended']])

    # Plotting
    if plot:
        time_str = pd.to_datetime(df_date[time_col]).dt.strftime('%H:%M:%S')
        tick_positions = range(0, len(time_str), tick_interval)

        plt.figure(figsize=(12, 6))
        plt.plot(time_str, df_date[value_col], label='Original', alpha=0.5)
        plt.plot(time_str, df_date['partial_detrended'], label='Partially Detrended', linewidth=2)

        # Trend before detrending
        x_segment = np.arange(len(y_segment))
        coeffs_before = np.polyfit(x_segment, y_segment, 1)
        trend_before = np.polyval(coeffs_before, x_segment)
        plt.plot(time_str[mask], trend_before, 'r--', label='Trend Before Detrend')

        # Trend after detrending
        coeffs_after = np.polyfit(x_segment, detrended_segment, 1)
        trend_after = np.polyval(coeffs_after, x_segment)
        plt.plot(time_str[mask], trend_after, 'g--', label='Trend After Detrend')

        plt.axvline(time_str[mask].iloc[0], color='red', linestyle=':', label='Detrend Start')
        plt.axvline(time_str[mask].iloc[-1], color='green', linestyle=':', label='Detrend End')

        plt.xlabel('Time (hh:mm:ss)')
        plt.ylabel('Total Field Anomaly (nT)')
        plt.title(f'Partial Detrending ({start_time} – {end_time})')
        plt.xticks(tick_positions, time_str.iloc[tick_positions], rotation=45)
        plt.legend()
        plt.tight_layout()

        if temp_dir is not None:
            os.makedirs(temp_dir, exist_ok=True)
            save_path = os.path.join(temp_dir, "solar_storm_partial_detrend_plot.png")
            plt.savefig(save_path, dpi=300)
            logging.info(f"Partial detrend plot saved to {save_path}")

        plt.show()

    return df_out

    

# def partial_detrend(df, value_col, time_col, chosen_date, start_time, end_time, tick_interval=300, plot=True, temp_dir: str = None):
   
#     df_out = df.copy()

#     y_full = df_out[value_col].ffill().astype(float).values
#     x_full = np.arange(len(y_full))

#     df_out['Date'] = df_out['Date'].dt.date
#     df_out['Time_obj'] = pd.to_datetime(df_out[time_col], format='%H:%M:%S')

#     start_t = pd.to_datetime(start_time, format='%H:%M:%S').time()
#     end_t = pd.to_datetime(end_time, format='%H:%M:%S').time()

#     mask = (
#         (df_out['Date'] == chosen_date) &
#         (df_out['Time_obj'].dt.time.between(start_t, end_t))
#     )

#     if not mask.any():
#         raise ValueError(f"No data found in specified time window, with {chosen_date}, {start_t}-{end_t}")

#     start_idx = mask[mask].index.min()
#     end_idx = mask[mask].index.max()

#     y_partial = y_full.copy()
#     x_segment = np.arange(start_idx, end_idx+1)
#     segment = y_full[start_idx:end_idx+1]
#     detrended_segment = detrend(segment, type='linear')
#     y_partial[start_idx:end_idx+1] = detrended_segment

#     df_out['partial_detrended'] = y_partial

#     if plot:
#         time_str = df_out[time_col].astype(str)
#         tick_positions = range(0, len(time_str), tick_interval)

#         plt.figure(figsize=(12, 6))
#         plt.plot(time_str, y_full, label='Original Diurnally Corrected', alpha=0.5)
#         plt.plot(time_str, y_partial, label='Partially Detrended', linewidth=2)

#         # Plot trend line before detrending
#         coeffs_before = np.polyfit(x_segment, segment, 1)
#         trend_before = np.polyval(coeffs_before, x_segment)
#         plt.plot(time_str.iloc[start_idx:end_idx+1], trend_before,
#                  'r--', label='Trend Before Detrend')

#         # Plot trend line after detrending
#         coeffs_after = np.polyfit(x_segment, detrended_segment, 1)
#         trend_after = np.polyval(coeffs_after, x_segment)
#         plt.plot(time_str.iloc[start_idx:end_idx+1], trend_after,
#                  'g--', label='Trend After Detrend')

#         plt.axvline(time_str.iloc[start_idx], color='red', linestyle=':', label='Detrend Start')
#         plt.axvline(time_str.iloc[end_idx], color='green', linestyle=':', label='Detrend End')

#         plt.xlabel('Time (hh:mm:ss)')
#         plt.ylabel('Total Field Anomaly (nT)')
#         plt.title(f'Partial Detrending of Magnetic Data ({start_time} – {end_time})')
#         plt.xticks(tick_positions, time_str.iloc[tick_positions], rotation=45)
#         plt.legend()
#         plt.tight_layout()

#         if temp_dir is not None:
#           os.makedirs(temp_dir, exist_ok=True)
#           save_path = os.path.join(temp_dir, "solar_storm_partial_detrend_plot.png")
#           plt.savefig(save_path, dpi=300)
#           logging.info(f"Partial detrend plot saved to {save_path}")

#         plt.show()

#     return df_out


