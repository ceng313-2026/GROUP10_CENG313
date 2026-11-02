"""Loading the competition labels and tomograms.

Owner: Yasin (data and infrastructure).

The competition stores each tomogram as a folder of 2D JPEG slices:

    train/<tomo_id>/slice_0000.jpg, slice_0001.jpg, ...

Stacking the slices along z gives a (Z, Y, X) uint8 volume.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Short names used everywhere in the project, mapped from the original CSV headers.
LABEL_COLUMNS = {
    "Motor axis 0": "z",
    "Motor axis 1": "y",
    "Motor axis 2": "x",
    "Array shape (axis 0)": "shape_z",
    "Array shape (axis 1)": "shape_y",
    "Array shape (axis 2)": "shape_x",
    "Voxel spacing": "voxel_spacing",
    "Number of motors": "n_motors",
}

KAGGLE_DATA_DIR = Path("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025")


def load_labels(csv_path):
    """Read train_labels.csv and rename columns to short names.

    A tomogram with no motor appears once with z, y, x = -1.
    A tomogram with several motors appears once per motor.
    """
    df = pd.read_csv(csv_path).rename(columns=LABEL_COLUMNS)
    # -1 is a "no motor" sentinel, not a real coordinate.
    df["has_motor"] = df["z"] >= 0
    return df


def tomogram_table(labels):
    """One row per tomogram (instead of one row per motor)."""
    return (
        labels.groupby("tomo_id")
        .agg(
            shape_z=("shape_z", "first"),
            shape_y=("shape_y", "first"),
            shape_x=("shape_x", "first"),
            voxel_spacing=("voxel_spacing", "first"),
            n_motors=("n_motors", "first"),
            has_motor=("has_motor", "any"),
        )
        .reset_index()
    )


def list_slices(tomo_dir):
    """Slice file paths of one tomogram, sorted along z."""
    return sorted(Path(tomo_dir).glob("*.jpg"))


def tomogram_shape(tomo_dir):
    """(Z, Y, X) of a tomogram without loading all slices."""
    slices = list_slices(tomo_dir)
    with Image.open(slices[0]) as img:
        width, height = img.size
    return len(slices), height, width


def load_tomogram(tomo_dir, z_step=1, scale=1.0):
    """Load a tomogram as a uint8 array of shape (Z, Y, X).

    z_step: read every z_step-th slice (2 halves the memory and time).
    scale:  resize factor applied to each slice in y and x (0.25 = quarter size).
    """
    slices = list_slices(tomo_dir)[::z_step]
    volume = []
    for path in slices:
        with Image.open(path) as img:
            img = img.convert("L")
            if scale != 1.0:
                new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
                img = img.resize(new_size, Image.BILINEAR)
            volume.append(np.asarray(img, dtype=np.uint8))
    return np.stack(volume)
