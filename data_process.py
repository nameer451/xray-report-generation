import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import h5py
from typing import *
from pathlib import Path

#####################################
# LOAD CSV
#####################################

def load_data(filepath):
    """
    Loads your CSV which must contain:
    - filename: absolute paths (or relative if base_dir is used)
    - impression: text of the impression
    """
    dataframe = pd.read_csv(filepath)
    return dataframe


#####################################
# PATH HANDLING
#####################################

def get_cxr_paths_list(filepath, base_dir=None):
    """
    Reads all paths from the CSV.
    If base_dir is provided, it prepends it to each relative path.
    If base_dir is empty, filenames are assumed to be absolute.

    This function does NOT modify internal folder structure.
    """
    df = load_data(filepath)

    paths = df["filename"].tolist()

    if base_dir and base_dir.strip() != "":
        paths = [os.path.join(base_dir, p) for p in paths]

    return paths


#####################################
# IMAGE PREPROCESSING
#####################################

def preprocess(img, desired_size=320):
    old_size = img.size  # (width, height)
    ratio = float(desired_size) / max(old_size)
    new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))

    # Resize using new Pillow enums
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Pad to square
    new_img = Image.new("L", (desired_size, desired_size))
    new_img.paste(
        img,
        ((desired_size - new_size[0]) // 2,
         (desired_size - new_size[1]) // 2)
    )
    return new_img


#####################################
# WRITE HDF5
#####################################

def img_to_hdf5(cxr_paths: List[str], out_filepath: str, resolution=320):
    """
    Loads all images using PIL (robust for grayscale / nonstandard JPEGs),
    preprocesses them, and saves them in a single HDF5 file.
    """

    dset_size = len(cxr_paths)
    failed_images = []

    with h5py.File(out_filepath, "w") as h5f:
        img_dset = h5f.create_dataset(
            "cxr",
            shape=(dset_size, resolution, resolution),
            dtype="uint8"
        )

        for idx, path in enumerate(tqdm(cxr_paths)):
            try:
                # OPEN WITH PIL (safe for your dataset)
                img_pil = Image.open(path).convert("L")

                # Preprocess (resize, pad)
                img = preprocess(img_pil, desired_size=resolution)

                # Convert to numpy before saving
                img_dset[idx] = np.array(img)

            except Exception as e:
                failed_images.append((path, str(e)))

    print(f"{len(failed_images)} / {len(cxr_paths)} images failed.")
    if len(failed_images) > 0:
        print("Failed images list:", failed_images)

    return failed_images


#####################################
# WRITE IMPRESSIONS CSV
#####################################

def write_report_csv(cxr_paths, csv_path, out_path):
    """
    CheXzero expects an impressions CSV with fields:
    - filename
    - impression

    Since you already have these in your CSV, we simply copy the relevant columns.
    """

    df = pd.read_csv(csv_path)

    # Only keep rows that correspond to paths we used
    df = df[df["filename"].isin(cxr_paths)]

    out_df = df[["filename", "impression"]]
    out_df.to_csv(out_path, index=False)

