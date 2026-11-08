"""Shrink every training tomogram 8x on all axes and save it as .npz.

Owner: Yasin. Output becomes the private Kaggle Dataset that the EDA and B2 use.

How to use:
  1. Kaggle -> competition page -> Code -> New Notebook
  2. Settings: Accelerator None (CPU is enough; GPU quota is not used)
  3. Paste this whole file into one cell, Run until the first progress line looks fine,
     then "Save Version" -> "Save & Run All (Commit)" so it finishes in the background.
  4. When the version finishes: notebook page -> Output -> "New Dataset"
     name it "group10-byu-downsampled-8x", visibility Private.
     Share it with Kaan and Salih (dataset page -> Settings -> Sharing).

Output in /kaggle/working:
  downsampled/<tomo_id>.npz     volume (uint8, Z/8 x Y/8 x X/8) + original shape, spacing, factor
  labels_downsampled.csv        train labels with an extra z_s, y_s, x_s (coords in the small volume)
  examples/<tomo_id>.npz        full-resolution motor slice and a crop around it (image EDA)

The block-averaging logic is the same as src/data.py downsample_tomogram; it is copied
here so the script runs without access to our repository.
"""

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FACTOR = 8
N_EXAMPLES_POS = 12  # tomograms with a motor saved at full resolution for image EDA
N_EXAMPLES_NEG = 4
CROP = 192  # full-resolution crop size around the motor (pixels)
OUT = Path("/kaggle/working")


def find_data_dir(root=Path("/kaggle/input")):
    for pattern in ("train_labels.csv", "*/train_labels.csv", "*/*/train_labels.csv", "*/*/*/train_labels.csv"):
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(f"train_labels.csv not found under {root}")


def list_slices(tomo_dir):
    return sorted(Path(tomo_dir).glob("*.jpg"))


def downsample_tomogram(tomo_dir, factor=FACTOR):
    slices = list_slices(tomo_dir)
    blocks = []
    for b in range(len(slices) // factor):
        acc = None
        for path in slices[b * factor:(b + 1) * factor]:
            with Image.open(path) as img:
                small = np.asarray(img.convert("L").reduce(factor), dtype=np.float32)
            acc = small if acc is None else acc + small
        blocks.append(acc / factor)
    return np.clip(np.rint(np.stack(blocks)), 0, 255).astype(np.uint8)


def process(args):
    tomo_id, tomo_dir, spacing = args
    out_path = OUT / "downsampled" / f"{tomo_id}.npz"
    if out_path.exists():  # lets a restarted run skip finished tomograms
        return tomo_id, 0.0
    t = time.time()
    volume = downsample_tomogram(tomo_dir)
    slices = list_slices(tomo_dir)
    with Image.open(slices[0]) as img:
        width, height = img.size
    np.savez_compressed(
        out_path,
        volume=volume,
        orig_z=len(slices), orig_y=height, orig_x=width,
        voxel_spacing=spacing, factor=FACTOR,
    )
    return tomo_id, time.time() - t


def save_examples(labels, train_dir):
    """Full-resolution slices for the image EDA: motor slice + crops, and some negatives."""
    (OUT / "examples").mkdir(exist_ok=True)
    pos = labels[labels["Motor axis 0"] >= 0].drop_duplicates("tomo_id").head(N_EXAMPLES_POS)
    neg = labels[labels["Motor axis 0"] < 0].drop_duplicates("tomo_id").head(N_EXAMPLES_NEG)
    for _, r in pos.iterrows():
        z, y, x = int(r["Motor axis 0"]), int(r["Motor axis 1"]), int(r["Motor axis 2"])
        slices = list_slices(train_dir / r["tomo_id"])
        full = np.asarray(Image.open(slices[z]).convert("L"))
        h = CROP // 2
        y0, x0 = max(0, y - h), max(0, x - h)
        crops = np.stack([
            np.asarray(Image.open(slices[zz]).convert("L"))[y0:y0 + CROP, x0:x0 + CROP]
            for zz in range(max(0, z - 4), min(len(slices), z + 5))
        ])
        np.savez_compressed(OUT / "examples" / f"{r['tomo_id']}.npz",
                            motor_slice=full, crops=crops, motor_zyx=np.array([z, y, x]),
                            crop_origin_yx=np.array([y0, x0]))
    for _, r in neg.iterrows():
        slices = list_slices(train_dir / r["tomo_id"])
        mid = np.asarray(Image.open(slices[len(slices) // 2]).convert("L"))
        np.savez_compressed(OUT / "examples" / f"{r['tomo_id']}.npz", middle_slice=mid)
    print(f"Saved {len(pos)} positive and {len(neg)} negative examples")


if __name__ == "__main__":
    data = find_data_dir()
    train_dir = data / "train"
    print("Competition data folder:", data)
    (OUT / "downsampled").mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(data / "train_labels.csv")
    tomos = labels.drop_duplicates("tomo_id")[["tomo_id", "Voxel spacing"]]
    jobs = [(t, train_dir / t, s) for t, s in zip(tomos["tomo_id"], tomos["Voxel spacing"])]
    print(f"{len(jobs)} tomograms, factor {FACTOR}")

    start = time.time()
    with Pool() as pool:
        for i, (tomo_id, secs) in enumerate(pool.imap_unordered(process, jobs), 1):
            if i % 25 == 0 or i == len(jobs):
                elapsed = time.time() - start
                eta = elapsed / i * (len(jobs) - i)
                print(f"{i}/{len(jobs)} done, {elapsed / 60:.1f} min elapsed, ~{eta / 60:.0f} min left", flush=True)

    # Labels with coordinates in the small volume (-1 stays -1).
    small = labels.copy()
    for i, axis in enumerate(("z_s", "y_s", "x_s")):
        col = labels[f"Motor axis {i}"]
        small[axis] = np.where(col >= 0, col / FACTOR, -1)
    small.to_csv(OUT / "labels_downsampled.csv", index=False)

    save_examples(labels, train_dir)

    sizes = [p.stat().st_size for p in (OUT / "downsampled").glob("*.npz")]
    print(f"Done: {len(sizes)} files, {sum(sizes) / 1e9:.2f} GB, {(time.time() - start) / 60:.1f} min")
