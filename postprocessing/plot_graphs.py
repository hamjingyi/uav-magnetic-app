import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import matplotlib.dates as mdates
import plotly.express as px
import plotly.io as pio


def plot_interactive_TMI(
    df: pd.DataFrame,
    easting_col: str,
    northing_col: str,
    magnetic_col: str = 'diurnal_correction_magnetic',
    filename: str = 'interactive_TMI_Map',
    output_folder: str = None,
    save_plot: bool = False,
):
    # Check required columns
    for col in [easting_col, northing_col, magnetic_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    df = df.dropna(subset=[magnetic_col]).copy()

    # Build coolwarm colormap
    cmap = plt.get_cmap("coolwarm", 256)
    coolwarm_colors = [
        f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
        for r, g, b, _ in cmap(np.linspace(0, 1, 256))
    ]

    df["Date_str"] = df["Date"].dt.date.astype(str)  # only the date
    df["Time_str"] = df["Time"].dt.time.astype(str)  # only the time

    # Simple scatter plot
    fig = px.scatter(
        df,
        x=easting_col,
        y=northing_col,
        color=magnetic_col,
        color_continuous_scale=coolwarm_colors,
        hover_data={"Date_str": True, "Time_str": True},
        # title=f'Interative Magnetic Map: {filename}',
        # width=1000,   # ~10 inches
        # height=1000    # ~8 inches
    )

    # Save if requested
    if save_plot:
        import os
        import plotly.io as pio
        if output_folder is None:
            output_folder = os.getcwd()
        os.makedirs(output_folder, exist_ok=True)
        html_path = os.path.join(output_folder, f"{filename}.html")
        pio.write_html(fig, html_path, include_plotlyjs='cdn', full_html=True)
        print(f"Interactive plot saved at: {html_path}")

    return fig

def plot_TMI(
    df: pd.DataFrame,
    easting_col: str,
    northing_col: str,
    magnetic_col: str = 'Total_field_anomaly__nT_',
    filename: str = 'TMI_Map',
    output_folder: str = None,
    save_plot: bool = False,
    gridsize: int = 60
):
    """
    Plots a 2D hexbin map of magnetic anomaly using Easting, Northing, and magnetic values.

    Args:
        df (pd.DataFrame): Input DataFrame containing spatial and magnetic data.
        easting_col (str): Column name for Easting (x-axis).
        northing_col (str): Column name for Northing (y-axis).
        magnetic_col (str): Column name for magnetic values to color the map.
        filename (str): Name for the plot and optional saved file.
        output_folder (str): Directory to save the plot, if enabled.
        save_plot (bool): Whether to save the plot as PNG.
        gridsize (int): Resolution of the hexbin.
    """
    try:
        x = df[easting_col]
        y = df[northing_col]
        z = df[magnetic_col]

        plt.figure(figsize=(10, 8))
        hb = plt.hexbin(x, y, z, gridsize=gridsize, cmap='coolwarm', mincnt=1)
        plt.colorbar(hb, label='Magnetic Field [nT]')
        plt.xlabel('Easting [m]')
        plt.ylabel('Northing [m]')
        plt.title(f'Magnetic Map: {filename}')
        plt.tight_layout()

        if save_plot and output_folder:
            os.makedirs(output_folder, exist_ok=True)
            save_path = os.path.join(output_folder, f'{filename}.png')
            plt.savefig(save_path, dpi=300)
            logging.info(f"Plot saved to: {save_path}")

        plt.show()
        plt.close()

    except Exception as e:
        logging.error(f"Plotting failed: {e}")
        raise


def plot_magnetic_timeseries(
    df: pd.DataFrame,
    time_col: str = "Time_Malaysia",
    title: str = "Magnetic Field Over Time",
    n_col: str = "N",
    filename: str = "magnetic_timeseries",
    output_folder: str = None,
    save_plot: bool = False,
    tick_interval: int = None,
    show_full_datetime: bool = True
):
    """
    Plots time series of magnetic field vs time.
    Handles:
      - full datetime
      - time-only + separate date column (e.g., "date")
      - time-only fallback to HH:MM:SS
    """
    try:
        t_raw = df[time_col]

        # Try to parse time column
        if np.issubdtype(t_raw.dtype, np.datetime64):
            t = pd.to_datetime(t_raw, errors="coerce")
        else:
            t = pd.to_datetime(t_raw.astype(str), errors="coerce")

        # If parsed times still missing dates, try combining with date column
        if t.dt.date.nunique() == 1 and t.dt.date.iloc[0] == pd.Timestamp.today().date():
            # Check for presence of a date column (case-insensitive)
            date_col = next((col for col in df.columns if col.lower() == "date"), None)
            if date_col:
                combined_dt = df[date_col].astype(str) + " " + df[time_col].astype(str)
                t = pd.to_datetime(combined_dt, errors="coerce")

        # Remove timezone if exists
        if hasattr(t.dt, "tz") and t.dt.tz is not None:
            t = t.dt.tz_localize(None)

        y = pd.to_numeric(df[n_col], errors="coerce")
        mask = t.notna() & y.notna()
        t = t[mask]
        y = y[mask]

        if len(t) == 0:
            logging.error("No valid points to plot.")
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(t, y, label=n_col)

        ax.set_xlabel("Time")
        ax.set_ylabel("Magnetic Field (nT)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        if tick_interval:
            tick_positions = range(0, len(t), tick_interval)
            tick_labels = t.iloc[list(tick_positions)].dt.strftime("%H:%M:%S")
            ax.set_xticks(t.iloc[list(tick_positions)])
            ax.set_xticklabels(tick_labels, rotation=45)
        else:
            fmt = "%Y-%m-%d %H:%M:%S" if show_full_datetime else "%H:%M:%S"
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            fig.autofmt_xdate()

        plt.tight_layout()

        if save_plot and output_folder:
            os.makedirs(output_folder, exist_ok=True)
            save_path = os.path.join(output_folder, f"{filename}.png")
            plt.savefig(save_path, dpi=300)
            logging.info(f"Timeseries saved to {save_path}")

        plt.show()
        plt.close()

    except Exception as e:
        logging.error(f"Timeseries plotting failed: {e}")
        raise

def plot_simple_magnetic_timeseries(
    df: pd.DataFrame,
    time_col: str = 'Time_hh:mm:ss',
    value_col: str = 'Total_field_anomaly__nT_',
    title: str = 'Total Field Anomaly vs Time',
    xlabel: str = 'Time (hh:mm:ss)',
    ylabel: str = 'Total Field Anomaly (nT)',
    tick_interval: int = 300,
    figsize: tuple = (12, 6),
    rotation: int = 45
):

    time_str = df[time_col].astype(str)
    tick_positions = range(0, len(time_str), tick_interval)

    plt.figure(figsize=figsize)
    plt.plot(time_str, df[value_col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(tick_positions, time_str.iloc[list(tick_positions)], rotation=rotation)
    plt.tight_layout()
    plt.show()
