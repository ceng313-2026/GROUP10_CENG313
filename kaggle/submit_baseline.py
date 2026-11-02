"""Kaggle submission for B0 / B1 baselines.

How to use:
  1. Kaggle -> competition page -> Code -> New Notebook
  2. Settings: Internet OFF (required for this code competition), Accelerator None
  3. Paste this whole file into one cell, set METHOD below, Run All
  4. "Save Version" -> "Save & Run All (Commit)"
  5. When the run finishes: notebook's Output tab -> "Submit" (late submission)

The script is self-contained because the submission notebook has no internet,
so it cannot clone our repository. Logic is the same as src/models.py.
"""

from pathlib import Path

import pandas as pd
from PIL import Image

METHOD = "B1"  # "B0" = always no motor, "B1" = mean normalized position

DATA = Path("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025")
OUT = Path("/kaggle/working/submission.csv")


def tomogram_shape(tomo_dir):
    slices = sorted(tomo_dir.glob("*.jpg"))
    with Image.open(slices[0]) as img:
        width, height = img.size
    return len(slices), height, width


# Mean normalized motor position from the training labels (B1).
labels = pd.read_csv(DATA / "train_labels.csv")
motors = labels[labels["Motor axis 0"] >= 0]
mean_norm = [
    (motors[f"Motor axis {i}"] / motors[f"Array shape (axis {i})"]).mean() for i in range(3)
]
print("Mean normalized position (z, y, x):", [round(float(v), 3) for v in mean_norm])

rows = []
for tomo_dir in sorted(p for p in (DATA / "test").iterdir() if p.is_dir()):
    if METHOD == "B0":
        z = y = x = -1
    else:
        shape = tomogram_shape(tomo_dir)
        z, y, x = (round(m * s) for m, s in zip(mean_norm, shape))
    rows.append({"tomo_id": tomo_dir.name, "Motor axis 0": z, "Motor axis 1": y, "Motor axis 2": x})

submission = pd.DataFrame(rows)
submission.to_csv(OUT, index=False)
print(submission.head())
print(f"Wrote {len(submission)} rows to {OUT}")
