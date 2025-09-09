import streamlit as st
import pandas as pd
import logging
import streamlit.components.v1 as components
import glob
import zipfile
import os
import tempfile
import uuid
import datetime
import numpy as np
np.random.seed(42)

from utils.msia_logger import setup_logger
from utils.file_helpers import upload_file, clean_output_folder
from data_preprocessing.preprocess_mag_data import resample_magnetic_data
from data_preprocessing.preprocess_base_station_obs import read_local_csv, preprocess_and_align
from reference_data.download_dalat_reference import download_dalat_data_fast, process_dalat_data
from corrections.diurnal_correction import run_GRU_trend_transfer, per_day_GRU_transfer, apply_diurnal_correction, partial_detrend
from corrections.directional_correction import apply_directional_correction
from postprocessing.microleveling import savgol_filter_microleveling
from postprocessing.plot_graphs import plot_TMI, plot_interactive_TMI
from inversion.susceptibility_inversion import run_3d_susceptibility_inversion, overlay_terrain_chi

# ----------------- Streamlit Layout -------------------

company_logo = "https://www.petroseis.asia/image001.gif"
st.set_page_config(
    page_title="UAV Magnetic Data Processing",
    page_icon=company_logo,
    # layout="wide",
    # initial_sidebar_state="expanded"
)
st.image(company_logo, width=300)
st.title("UAV Magnetic Data Processing and Optimized Visualization for GIS Integration")
st.markdown("---")

# ----------------- Logger -------------------
log_stream = setup_logger()
log_stream.truncate(0)
log_stream.seek(0)
log_placeholder = st.sidebar.empty()

def show_logs(tail_lines=20):
    log_content = log_stream.getvalue().splitlines()
    unique_key = f"log_text_area_{uuid.uuid4()}"
    if len(log_content) > tail_lines:
        log_content = log_content[-tail_lines:]
    log_placeholder.text_area("Logs", value="\n".join(log_content), key=unique_key, height=600)

# ----------------- Local env -----------------
# out = 'Automation Output Testing'
# input_dir = f"/content/drive/MyDrive/Research Project/P1/Dataset/1st Set Data"
# out_base_path = f"/content/drive/MyDrive/Research Project/P1/{out}"
# output_dir = f"/content/drive/MyDrive/Research Project/P1/{out}/csv files"
# tmi_dir = f"/content/drive/MyDrive/Research Project/P1/{out}/tmi"
# temp_dir = f"/content/drive/MyDrive/Research Project/P1/{out}/temp"
# suscep_dir = f"/content/drive/MyDrive/Research Project/P1/{out}/final suscep"

# 1st data
# filename = f"{input_dir}/1608_Mag_data_combine2days.csv"
# local_csv = f"{input_dir}/base_station_combine2days.csv"
# xyz_path=f"{input_dir}/dem_190825.xyz"
# photo_path=f"{input_dir}/DSM 5m.tif"

# 2nd data
# filename = "/content/drive/MyDrive/Research Project/P1/Dataset/2nd Set Data/Survey Line/1560_DATASET.csv"
# local_csv = "/content/drive/MyDrive/Research Project/P1/Dataset/2nd Set Data/Diurnal Observation/combined_diurnal_observation.csv"
# xyz_path=None
# photo_path=None
# --------------------------------------------------------------------


# ----------------- Production env -----------------
BASE_DIR = os.path.dirname(__file__)
out_base_path = os.path.join(BASE_DIR, "Automation_Output")
out = 'Automation Output Testing'
output_dir = f"{out}/csv files"
tmi_dir = f"{out}/tmi"
temp_dir = f"{out}/temp"
suscep_dir = f"{out}/final suscep"

filename = upload_file("(required) MAGNETIC DATA CSV file here:")
if filename:
    st.write(f"Uploaded: `{filename.name}`")

local_csv = upload_file("(required) BASE STATION CSV file here:")
if local_csv:
    st.write(f"Uploaded: `{local_csv.name}`")

# xyz_path = upload_file("DEM XYZ file here:")
# if xyz_path:
#     st.write(f"Uploaded: `{xyz_path.name}`")

# photo_path = upload_file("DSM TIF file here:")
# if photo_path:
#     st.write(f"Uploaded: `{photo_path.name}`")
xyz_path=None
photo_path=None
# --------------------------------------------------------------------

required_files = {
    "Magnetic CSV": filename,
    "Base Station CSV": local_csv,
    # "DEM XYZ": xyz_path,
    # "DSM TIF": photo_path
}

missing_files = [name for name, f in required_files.items() if f is None]

# ----------------- Progress bar -----------------
steps = [
    "Resample UAV Data", "Dalat Reference", "Local Base Station",
    "Diurnal Correction", "Partial Detrend", "Directional Correction",
    "Microleveling", "3D Inversion", "Terrain Overlay"
]
progress = st.progress(0)
status_placeholder = st.empty()

def update_progress(step_idx):
    pct = int((step_idx+1) / len(steps) * 100)
    progress.progress(pct)
    status_placeholder.markdown(f"Step {step_idx+1}/{len(steps)} → {steps[step_idx]}")

# ----------------- Reset downstream -----------------
def reset_downstream():
    downstream_keys = [
        "df_detrend",
        "df_directional_corrected",
        "df_microleveling",
        "mesh", "mrec", "actv",
        "terrain_overlay_done"
    ]
    for key in downstream_keys:
        if key in st.session_state:
            del st.session_state[key]


if missing_files:
    st.info(f"⚠️ Please upload all required files before running resampling: {', '.join(missing_files)}")
else:
    try:

      # clear historical files
      if "cleaned_output" not in st.session_state:
          clean_output_folder(out_base_path)
          st.session_state.cleaned_output = True

      # ----------------- Step 1: Resample UAV -----------------
      update_progress(0)
      st.header("Step 1: Resample UAV Magnetic Data to 1-sec")
      with st.spinner("Grouping UAV magnetic data..."):
          if "df_resampled" not in st.session_state:
              st.session_state.df_resampled = resample_magnetic_data(input_dir=filename, output_dir=output_dir)
          df_resampled = st.session_state.df_resampled
      show_logs()
      st.dataframe(df_resampled.head(20), width='stretch')

      if "original_tmi_path" not in st.session_state:
          plot_TMI(
              df=df_resampled,
              easting_col="Easting",
              northing_col="Northing",
              magnetic_col="Total_field_anomaly__nT_",
              filename="Original TMI",
              output_folder=tmi_dir,
              save_plot=True,
          )
          st.session_state.original_tmi_path = f"{tmi_dir}/Original TMI.png"
      st.image(st.session_state.original_tmi_path, width='stretch', caption="Original TMI Map")

      # ----------------- Step 2: Dalat Reference -----------------
      update_progress(1)
      st.header("Step 2: Download INTERMAGNET Base Station Reference")

      date_str= pd.to_datetime(df_resampled['Date'],format='%d/%m/%Y', errors='coerce')
      start_date = date_str.min().strftime("%Y-%m-%d")
      end_date = date_str.max().strftime("%Y-%m-%d")
      duration_days = (date_str.max() - date_str.min()).days + 1

      dalat_filename = f"{start_date}_{end_date}_dalat_base_station.csv"
      dalat_path = os.path.join(temp_dir, dalat_filename)
      
      if "df_dalat" not in st.session_state:
          # Delete all other *_dalat_base_station.csv files except the one we will download
          if os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
              if f.endswith("_dalat_base_station.csv") and f != dalat_filename:
                  try:
                      os.remove(os.path.join(temp_dir, f))
                      logging.info(f"Deleted old Dalat file: {f}")
                  except Exception as e:
                      logging.error(f"Failed to delete {f}: {e}")

          # Load or download the target Dalat CSV
          if os.path.exists(dalat_path):
              logging.info(f"{dalat_filename} exist!")
              st.session_state.df_dalat = pd.read_csv(dalat_path)
          else:
              with st.spinner("Downloading Dalat base station reference..."):
                  st.session_state.df_dalat = download_dalat_data_fast(
                      start_date=start_date,
                      duration_days=duration_days,
                      output_dir=temp_dir,
                      filename=dalat_filename
                  )
      df_dalat = st.session_state.df_dalat
      show_logs()

      if "processed_dalat" not in st.session_state:
          st.session_state.processed_dalat = process_dalat_data(
              df=df_dalat,
              target_tz="Asia/Ho_Chi_Minh",
              extra_offset_hours=1,
              output_csv=f"{temp_dir}/dalat_base_station_processed.csv"
          )
      processed_dalat = st.session_state.processed_dalat
      show_logs()
      st.dataframe(processed_dalat.head(20), width='stretch')

      # ----------------- Step 3: Local Base Station Alignment -----------------
      update_progress(2)
      st.header("Step 3: Local Base Station Alignment")
      with st.spinner("Aligning local base station data..."):
          if "df_local" not in st.session_state:
              df_local_raw = read_local_csv(local_csv)
              df_local, df_dalat_filtered = preprocess_and_align(
                  df_local_raw,
                  processed_dalat,
                  local_time_col="Time",
                  dalat_time_col="Time_Malaysia"
              )
              st.session_state.df_local = df_local
              st.session_state.df_dalat_filtered = df_dalat_filtered
          df_local = st.session_state.df_local
          df_dalat_filtered = st.session_state.df_dalat_filtered
      show_logs()

      with st.spinner("Running GRU Trend Transfer per day..."):
          if "df_interp_GRU" not in st.session_state:
              st.session_state.df_interp_GRU = per_day_GRU_transfer(
                  df_dalat_filtered, 
                  df_local, 
                  run_GRU_trend_transfer,
                  temp_dir=temp_dir
              )
      df_interp_GRU = st.session_state.df_interp_GRU
      show_logs()

      plot_files = sorted(glob.glob(f"{temp_dir}/GRU_trend_*.png"))

      if not plot_files:
          st.warning("No GRU trend plots found.")
      else:
          cols = st.columns(len(plot_files))
          
          for col, plot_file in zip(cols, plot_files):
              col.image(
                  plot_file,
                  width='stretch',
                  caption=os.path.basename(plot_file)
              )

      # else:
      #     selected_plot = st.selectbox(
      #         "Select a GRU trend plot to view:",
      #         plot_files,
      #         index=len(plot_files)-1,  # last one = latest
      #         format_func=lambda x: os.path.basename(x)  # show only filename
      #     )

      #     st.image(
      #         selected_plot,
      #         width='stretch',
      #         caption=f"Viewing {os.path.basename(selected_plot)}"
      #     )


      col1, col2 = st.columns(2)
      with col1: 
        st.subheader("Local Base Station Observation")
        st.dataframe(df_local, width='stretch')
      with col2: 
        st.subheader("Interpolated to 1-sec using GRU Trend Tansfer")
        st.dataframe(df_interp_GRU, width='stretch')

      # ----------------- Step 4: Diurnal Correction -----------------
      update_progress(3)
      st.header("Step 4: Diurnal Correction")

      with st.spinner("Applying diurnal correction..."):
          if "df_diurnal_corrected" not in st.session_state:
              st.session_state.df_diurnal_corrected = apply_diurnal_correction(df_resampled, df_interp_GRU)
      df_diurnal_corrected = st.session_state.df_diurnal_corrected
      show_logs()
      st.subheader("Diurnal Corrected")

      # TMI static and interactive plots
      if "diurnal_tmi_path" not in st.session_state:
          plot_TMI(
              df=df_diurnal_corrected,
              easting_col="Easting",
              northing_col="Northing",
              magnetic_col="diurnal_correction_magnetic",
              filename="Diurnal Corrected TMI",
              output_folder=tmi_dir,
              save_plot=True
          )
          st.session_state.diurnal_tmi_path = f"{tmi_dir}/Diurnal Corrected TMI.png"

      if "fig_diurnal" not in st.session_state:
          st.session_state.fig_diurnal = plot_interactive_TMI(
              df=df_diurnal_corrected,
              easting_col='Easting',
              northing_col='Northing',
              magnetic_col='diurnal_correction_magnetic',
              filename='interactive_TMI_to_check_solar_storm_period',
              output_folder=tmi_dir,
              save_plot=True
          )
      fig = st.session_state.fig_diurnal

      col1, col2 = st.columns(2)
      with col1:
          st.image(f"{tmi_dir}/Diurnal Corrected TMI.png", width='stretch', caption="Diurnal Corrected TMI")
      with col2:
          st.plotly_chart(fig, width='stretch')

      show_logs()

      # ----------------- Step 5: Partial Detrend -----------------
      update_progress(4)
      st.header("Step 5: Partial Detrend")

      # --- User choice ---
      detrend_option = st.radio(
          "Apply partial detrend?",
          ["-- Select --", "Yes", "No"],
          index=0,
          key="detrend_option",
          on_change=reset_downstream
      )

      # Stop here until user chooses
      if detrend_option == "-- Select --":
          st.info("Based on Interactive TMI → Please select whether to apply detrend before continuing.")
          st.stop()

      # --- If user wants detrend ---
      if detrend_option == "Yes":
          min_date = pd.to_datetime(df_diurnal_corrected["Date"]).min().date()
          max_date = pd.to_datetime(df_diurnal_corrected["Date"]).max().date()

          col1, col2, col3 = st.columns(3)

          with col1:
            chosen_date = st.date_input(
                "Select detrend date:",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="detrend_date",
                on_change=reset_downstream
            )

          with col2:
            start_time = st.time_input(
                "Start time:",
                value=datetime.time(10, 0),
                step=900,
                key="detrend_start",
                on_change=reset_downstream
            )

          with col3:
            end_time = st.time_input(
                "End time:",
                value=datetime.time(10, 0),
                step=900,
                key="detrend_end",
                on_change=reset_downstream
            )

          if end_time <= start_time:
              st.error("⚠️ End time must be later than start time.")
              st.stop()

          with st.spinner("Performing partial detrend..."):
              if "df_detrend" not in st.session_state:
                  st.session_state.df_detrend = partial_detrend(
                      df_diurnal_corrected,
                      value_col="diurnal_correction_magnetic",
                      time_col="Time_hh:mm:ss",
                      chosen_date=chosen_date,
                      start_time=start_time.strftime("%H:%M:%S"),
                      end_time=end_time.strftime("%H:%M:%S"),
                      tick_interval=300,
                      plot=True,
                      temp_dir=temp_dir
                  )

          df_current = st.session_state.df_detrend
          directional_base_col = "partial_detrended"

          show_logs()
          plot_TMI(
              df=df_current,
              easting_col='Easting',
              northing_col='Northing',
              magnetic_col=directional_base_col,
              filename='Solar Storm TMI',
              output_folder=tmi_dir,
              save_plot=True,
          )
          col1, col2 = st.columns(2)
          with col1:
              st.image(
                  f"{temp_dir}/solar_storm_partial_detrend_plot.png",
                  width='stretch',
                  caption="Solar Storm Trend"
              )

          with col2:
              st.image(
                  f"{tmi_dir}/Solar Storm TMI.png",
                  width='stretch',
                  caption="Solar Storm TMI"
          )
          show_logs()
      else:
          df_current = df_diurnal_corrected.copy()
          directional_base_col = "diurnal_correction_magnetic"
          st.info("TMI map same as diurnal correction TMI")

      # ----------------- Step 6: Directional Correction -----------------
      update_progress(5)
      st.header("Step 6: Directional Correction")
      with st.spinner("Applying directional correction..."):
          if "df_directional_corrected" not in st.session_state:
              results = apply_directional_correction(df_current, base_col=directional_base_col, plot=True, output_dir=temp_dir)
              st.session_state.df_directional_corrected = results["df_corrected"]
      df_directional_corrected = st.session_state.df_directional_corrected
      show_logs()

      plot_TMI(
          df=df_directional_corrected,
          easting_col='Easting',
          northing_col='Northing',
          magnetic_col='direction_corrected_huber',
          filename='Directional Corrected TMI',
          output_folder=tmi_dir,
          save_plot=True,
      )
      st.image(f"{tmi_dir}/Directional Corrected TMI.png", width='stretch', caption="Directional Corrected TMI")
      show_logs()

      # ----------------- Step 7: Microleveling -----------------
      update_progress(6)
      st.header("Step 7: Microleveling")
      with st.spinner("Running microleveling..."):
          if "df_microleveling" not in st.session_state:
              st.session_state.df_microleveling = savgol_filter_microleveling(
                  df_directional_corrected, base_col="direction_corrected_huber", clamp_val=5.0
              )
      df_microleveling = st.session_state.df_microleveling
      micro_csv = os.path.join(output_dir, "final_processed_magnetic.csv")
      df_microleveling.to_csv(micro_csv, index=False)
      show_logs()
      st.subheader("Final Processed Magnetic Values")
      st.dataframe(df_microleveling, width='stretch')

      plot_TMI(
          df=df_microleveling,
          easting_col='Easting',
          northing_col='Northing',
          magnetic_col='TMI_micro_sg',
          filename='Microleveling Corrected TMI',
          output_folder=tmi_dir,
          save_plot=True,
      )
      st.image(f"{tmi_dir}/Microleveling Corrected TMI.png", width='stretch', caption="Microleveling Corrected TMI")
      show_logs()

      # ----------------- Step 8: 3D Inversion -----------------
      update_progress(7)
      st.header("Step 8: 3D Magnetic Susceptibility Inversion")
      with st.spinner("Running 3D susceptibility inversion..."):
          if "mesh" not in st.session_state:
              COLS = {"x": "Easting", "y": "Northing", "z": "Altitude__m_", "tmi": "TMI_micro_sg", "line": "Line"}
              mrec, mesh, actv = run_3d_susceptibility_inversion(
                  csv_path=micro_csv,
                  out_dir=suscep_dir,
                  cols=COLS,
                  B_total=50000.0,
                  incl=10.0,
                  decl=0.0
              )
              st.session_state.mrec, st.session_state.mesh, st.session_state.actv = mrec, mesh, actv
      show_logs()
      st.success("3D inversion completed!")


      # # ------------------save result testing----------------------
      # import pickle
      # save_path = os.path.join(suscep_dir, "inversion_results.pkl")
      # with open(inver, "wb") as f:
      #     pickle.dump({"mrec": mrec, "mesh": mesh, "actv": actv}, f)

      # with open(os.path.join(suscep_dir, "inversion_results.pkl"), "rb") as f:
      #     results = pickle.load(f)

      # mrec = results["mrec"]
      # mesh = results["mesh"]
      # actv = results["actv"]

      # ----------------- Step 9: Terrain Overlay -----------------
      update_progress(8)
      st.header("Step 9: Terrain Overlay")
      with st.spinner("Overlaying terrain with magnetic model..."):
          if "terrain_overlay_done" not in st.session_state:
              overlay_terrain_chi(
                    xyz_path=xyz_path,
                    photo_path=photo_path,
                    html_out=suscep_dir,
                    title="terrain_overlay.html",
                    # mesh=mesh,
                    # actv=actv,
                    # mrec=mrec
                    mesh=st.session_state.mesh,
                    actv=st.session_state.actv,
                    mrec=st.session_state.mrec
                )
              st.session_state.terrain_overlay_done = True

      html_path = os.path.join(suscep_dir, "terrain_overlay.html")
      with open(html_path, "r", encoding="utf-8") as f:
          html_content = f.read()
      components.html(html_content, height=800, scrolling=True)

      # ----------------- Collect and Download -----------------
      def collect_and_download():
          st.markdown("---")
          st.header("Final Results Collector")

          temp_zip_path = os.path.join(tempfile.gettempdir(), "all_results.zip")
          with zipfile.ZipFile(temp_zip_path, "w") as zipf:
              for folder, subfolder in [
                  (tmi_dir, "TMI"),
                  (output_dir, "Magnetic_CSV"),
                  (temp_dir, "Temp"),
                  (suscep_dir, "Susceptibility")
              ]:
                  if os.path.exists(folder):
                      for f in os.listdir(folder):
                          zipf.write(os.path.join(folder, f), f"{subfolder}/{f}")
              zipf.writestr("logs.txt", log_stream.getvalue())

          with open(temp_zip_path, "rb") as f:
              st.download_button(
                  label="Download All Final Results (ZIP)",
                  data=f,
                  file_name="all_results.zip",
                  mime="application/zip"
              )

      collect_and_download()
      show_logs()

    except ValueError as e:
            st.warning("Something went wrong while reading the CSV. "
                      "Make sure the file is valid and uploaded correctly.")
            st.error(f"Details: {e}")
