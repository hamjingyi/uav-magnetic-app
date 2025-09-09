import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.linear_model import HuberRegressor
import os

def apply_directional_correction(df_all: pd.DataFrame,
                                base_col: str = 'diurnal_correction_magnetic',
                                plot: bool = True, output_dir: str = None):
    
    need = ['Line','Timestamp__ms_','Easting','Northing', base_col]
    missing = [c for c in need if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_all = df_all.copy()
    logging.info(f"Starting directional correction on {len(df_all)} samples")

    # --- Compute heading per point ---
    def compute_heading_deg(df: pd.DataFrame) -> pd.Series:
        def _per_line(g):
            g = g.sort_values('Timestamp__ms_', kind='stable')
            de = g['Easting'].diff().bfill().ffill().fillna(0.0)
            dn = g['Northing'].diff().bfill().ffill().fillna(0.0)
            ang = np.degrees(np.arctan2(de.to_numpy(), dn.to_numpy()))
            return pd.Series((ang + 360.0) % 360.0, index=g.index)

        try:
            return df.groupby('Line', group_keys=False, sort=False).apply(_per_line, include_groups=False)
        except TypeError:
            return df.groupby('Line', group_keys=False, sort=False).apply(_per_line)

    def infer_cardinal(dn, de):
        if abs(dn) > abs(de): 
            return 0 if dn > 0 else 180
        return 90 if de > 0 else 270

    df_all['Heading_deg'] = compute_heading_deg(df_all)
    logging.info("Computed per-sample headings")

    # --- Compute line-wise directions ---
    line_dirs = {}
    for line, g in df_all.groupby('Line'):
        g = g.sort_values('Timestamp__ms_', kind='stable')
        dn = float(g['Northing'].iloc[-1] - g['Northing'].iloc[0])
        de = float(g['Easting'].iloc[-1]  - g['Easting'].iloc[0])
        line_dirs[line] = infer_cardinal(dn, de)
    df_all['FlightDirection_deg'] = df_all['Line'].map(line_dirs)
    logging.info(f"Inferred cardinal directions for {len(line_dirs)} lines")

    # --- Per-line summary table ---
    L = pd.DataFrame({
        'y': df_all.groupby('Line', sort=False)[base_col].median().astype(np.float32), 
        'n': df_all.groupby('Line', sort=False)[base_col].size().astype(np.float32),
        'dir': pd.Series(line_dirs),
        'theta_mean': df_all.groupby('Line', sort=False)['Heading_deg'].mean().astype(np.float32),
    }).dropna()
    logging.info(f"Constructed per-line table with {len(L)} entries")

    # Design matrix: one-hot(dir) + sin2/cos2
    X_dir = pd.get_dummies(L['dir'].astype('category'), prefix='dir', drop_first=False)
    th = np.deg2rad(L['theta_mean'].to_numpy()) #Compute mean heading
    X_harm = pd.DataFrame({'sin2': np.sin(2*th), 'cos2': np.cos(2*th)}, index=L.index)
    X = pd.concat([X_dir, X_harm], axis=1).astype(np.float32).fillna(0.0)
    y = L['y'].to_numpy().astype(np.float32) #compute median
    w = (L['n'] / (L['n'].mean() + 1e-9)).to_numpy().astype(np.float32)
    lines = L.index.to_list()

    # --- Fit robust regression ---
    try:
        huber = HuberRegressor(alpha=1e-4, epsilon=1.35, fit_intercept=True, max_iter=1000)
        huber.fit(X, y, sample_weight=w)
        y_hat_huber = huber.predict(X).astype(np.float32)
    except Exception:
        # Weighted ridge fallback
        W = np.diag(w)
        XtW = X.to_numpy().T @ W
        coef = np.linalg.solve(XtW @ X.to_numpy() + 1e-3*np.eye(X.shape[1], dtype=np.float32),
                              XtW @ y)
        y_hat_huber = (X.to_numpy() @ coef).astype(np.float32)

    logging.info("Fitted Directional Correction Model (HuberRegressor) successfully")

    y_hat_huber -= float(y_hat_huber.mean())
    line_pred_huber = pd.Series(y_hat_huber, index=L.index).astype(np.float32)

    df_all['directional_bias_huber'] = df_all['Line'].map(line_pred_huber).astype('float32').fillna(0.0)
    df_all['direction_corrected_huber'] = (
        df_all[base_col].astype(np.float32) - df_all['directional_bias_huber']
    )
    logging.info("Applied bias correction to dataframe")

    # --- Plot if requested ---
    if plot:
        theta_rad = np.deg2rad(L['theta_mean'].values)
        X_poly2 = pd.DataFrame({
            'sin2': np.sin(2 * theta_rad),
            'cos2': np.cos(2 * theta_rad),
        })
        y = L['y'].values

        # --- Fit robust regression (directional bias curve) ---
        huber = HuberRegressor().fit(X_poly2, y)

        # Smooth grid for drawing the fitted curve
        theta_grid = np.linspace(0, 360, 500)
        theta_rad_grid = np.deg2rad(theta_grid)
        X_grid = pd.DataFrame({
            'sin2': np.sin(2 * theta_rad_grid),
            'cos2': np.cos(2 * theta_rad_grid),
        })

        bias_curve = huber.predict(X_grid)
        bias_curve -= bias_curve.mean()  # center around zero

        # --- Apply correction to your data ---
        bias_estimated = huber.predict(X_poly2)
        bias_estimated -= bias_estimated.mean()
        y_corrected = y - bias_estimated   # correction = remove fitted bias

        # --- Plot before correction ---
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.scatter(L['theta_mean'], y - y.mean(), alpha=0.6, label="Raw centered values")
        plt.plot(theta_grid, bias_curve, color='blue', linewidth=2, label="Fitted bias curve")
        plt.axhline(0, color='k', linewidth=1)
        plt.xlabel("Heading (deg)")
        plt.ylabel("Centered magnetic anomaly (nT)")
        plt.title("Before correction: bias present")
        plt.legend()
        plt.tight_layout()

        if output_dir is not None:
          fig_path = os.path.join(output_dir, "Before_directional_correction: bias_present.png")
          plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        plt.show()

        # --- Plot after correction ---
        plt.subplot(1,2,2)
        plt.scatter(L['theta_mean'], y_corrected - y_corrected.mean(), alpha=0.6, color='green', label="Corrected values")
        plt.axhline(0, color='k', linewidth=1)
        plt.xlabel("Heading (deg)")
        plt.ylabel("Corrected anomaly (nT)")
        plt.title("After correction: bias removed")
        plt.legend()
        plt.tight_layout()

        if output_dir is not None:
          fig_path = os.path.join(output_dir, "After_directional_correction: bias_removed.png")
          plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        plt.show()

        plt.hist(df_all['diurnal_correction_magnetic'], bins=100, alpha=0.5, label='Before')
        plt.hist(df_all['direction_corrected_huber'], bins=100, alpha=0.5, label='After')
        plt.legend()
        plt.title("Magnetic Value Distribution: Before vs After Direction Correction")
        plt.xlabel("nT")
        plt.ylabel("Frequency")

        if output_dir is not None:
          fig_path = os.path.join(output_dir, "Directional Correction: Magnetic Value Distribution.png")
          plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        plt.show()
        logging.info(f"Before correction mean: {df_all['diurnal_correction_magnetic'].mean()}")
        logging.info(f"After correction mean: {df_all['direction_corrected_huber'].mean()}")

    return {
        "df_corrected": df_all,
        "model": huber,
        "bias_curve": bias_curve,
        "theta_grid": theta_grid,
        "line_table": L
    }
