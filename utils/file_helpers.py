import streamlit as st
import os
import shutil
import fnmatch
import logging

def clean_output_folder(out_base_path, keep_temp_pattern="*_dalat_base_station.csv"):

    if not os.path.exists(out_base_path):
        logging.warning(f"Output folder does not exist: {out_base_path}")
        os.makedirs(out_base_path, exist_ok=True)
        return

    for item in os.listdir(out_base_path):
        item_path = os.path.join(out_base_path, item)

        if os.path.isdir(item_path):
            # If this is the temp folder, preserve Dalat CSV files
            if item.lower() == "temp":
                for temp_file in os.listdir(item_path):
                    temp_file_path = os.path.join(item_path, temp_file)
                    if fnmatch.fnmatch(temp_file, keep_temp_pattern):
                        continue
                    try:
                        if os.path.isfile(temp_file_path) or os.path.islink(temp_file_path):
                            os.unlink(temp_file_path)
                            logging.info(f"Clear historical file: {temp_file_path}")
                        elif os.path.isdir(temp_file_path):
                            shutil.rmtree(temp_file_path)
                            logging.info(f"Clear historical folder: {temp_file_path}")
                    except Exception as e:
                        logging.error(f"Failed to delete {temp_file_path}: {e}")
            else:
                # Delete the entire folder recursively
                try:
                    shutil.rmtree(item_path)
                    logging.info(f"Clear historical folder: {item_path}")
                except Exception as e:
                    logging.error(f"Failed to delete {item_path}: {e}")
        else:
            # Delete any file directly under out_base_path
            try:
                os.unlink(item_path)
                logging.info(f"Clear historical file: {item_path}")
            except Exception as e:
                logging.error(f"Failed to delete {item_path}: {e}")


def upload_file(label: str):
    uploaded_file = st.file_uploader(label, type=None)  # type=None allows all file types
    if uploaded_file is not None:
        return uploaded_file  
    return None

# def upload_file(label: str, save_dir: str = None) -> str:

#     uploaded_file = st.file_uploader(label, type=None)  # type=None allows all file types
#     if uploaded_file is not None:
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             file_path = os.path.join(save_dir, uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             return file_path
#         else:
#             return uploaded_file
#     return None
