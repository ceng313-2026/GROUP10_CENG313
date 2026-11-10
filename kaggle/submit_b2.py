"""Kaggle submission for B2 (patch classifier + sliding window, Random Forest).

How to use:
  1. Kaggle -> competition page -> Code -> New Notebook
  2. Settings: Internet OFF (required for this code competition), Accelerator None
  3. Add Input -> Your Work / Datasets -> "group10-byu-downsampled-8x"
     (the competition data is already there)
  4. Paste this whole file into one cell and Run All (about 15-20 min on CPU)
  5. "Save Version" -> "Save & Run All (Commit)"
  6. When the run finishes: notebook's Output tab -> "Submit" (late submission)

The notebook has no internet, so it cannot clone our repo. The B2 code below is the
same as src/models.py, and the downsampling is the same as kaggle/downsample_train.py.

Steps: train B2 on all 648 downsampled train tomograms, downsample each test tomogram
in the same way, find the best window and write submission.csv.
Threshold is 0 (always predict a motor), this was the best on validation.
"""

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/submission.csv")
FACTOR = 8

# same settings as the notebook (B2-RF v2 + refine)
SIZE, STRIDE, Z_STEP = 16, 4, 2
NEG_PER_TOMO, HARD_PER_TOMO, NEG_MIN_DIST = 30, 10, 2000.0
MARGIN, SMOOTH = 0.1, 3


# ---------------------------------------------------------------------------
# find the input folders
# ---------------------------------------------------------------------------

def find_dir(marker):
    """Folder under /kaggle/input that contains `marker` (file or folder)."""
    for depth in range(5):
        hits = sorted(INPUT.glob("/".join(["*"] * depth + [marker])))
        if hits:
            return hits[0].parent
    for p in sorted(INPUT.glob("*")) + sorted(INPUT.glob("*/*")):
        print(p)
    raise FileNotFoundError(f"{marker} not found under {INPUT}")


DATA = find_dir("train_labels.csv")                 # competition data
VOL_DIR = find_dir("labels_downsampled.csv") / "downsampled"   # our dataset
print("Competition data:", DATA)
print("Downsampled train volumes:", VOL_DIR, len(list(VOL_DIR.glob("*.npz"))), "files")


# ---------------------------------------------------------------------------
# downsampling (same as kaggle/downsample_train.py)
# ---------------------------------------------------------------------------

def downsample_tomogram(tomo_dir, factor=FACTOR):
    slices = sorted(Path(tomo_dir).glob("*.jpg"))
    blocks = []
    for b in range(len(slices) // factor):
        acc = None
        for path in slices[b * factor:(b + 1) * factor]:
            with Image.open(path) as img:
                small = np.asarray(img.convert("L").reduce(factor), dtype=np.float32)
            acc = small if acc is None else acc + small
        blocks.append(acc / factor)
    return np.clip(np.rint(np.stack(blocks)), 0, 255).astype(np.uint8)


def load_train_volume(tomo_id):
    with np.load(VOL_DIR / f"{tomo_id}.npz") as f:
        return f["volume"], float(f["voxel_spacing"])


# ---------------------------------------------------------------------------
# B2 (same as src/models.py)
# ---------------------------------------------------------------------------

def normalize_volume(vol):
    vol = vol.astype(np.float32)
    return (vol - vol.mean()) / (vol.std() + 1e-6)


def patch_at(vol, z, y, x, size=SIZE):
    h = size // 2
    if not (0 <= z < vol.shape[0] and h <= y <= vol.shape[1] - (size - h)
            and h <= x <= vol.shape[2] - (size - h)):
        return None
    return vol[z, y - h:y - h + size, x - h:x - h + size]


def prob_map(clf, vol):
    zc = np.arange(0, vol.shape[0], Z_STEP)
    win = sliding_window_view(vol[zc], (SIZE, SIZE), axis=(1, 2))[:, ::STRIDE, ::STRIDE]
    nz, ny, nx = win.shape[:3]
    prob = clf.predict_proba(win.reshape(nz * ny * nx, SIZE * SIZE))[:, 1]
    yc = np.arange(ny) * STRIDE + SIZE // 2
    xc = np.arange(nx) * STRIDE + SIZE // 2
    return prob.reshape(nz, ny, nx), (zc, yc, xc)


def inside(shape, centers):
    keep = [(c >= MARGIN * n) & (c < (1 - MARGIN) * n) for c, n in zip(centers, shape)]
    return keep[0][:, None, None] & keep[1][None, :, None] & keep[2][None, None, :]


def refine(clf, vol, center):
    h = SIZE // 2
    z0, y0, x0 = center
    zs = np.arange(max(z0 - Z_STEP, 0), min(z0 + Z_STEP, vol.shape[0] - 1) + 1)
    ys = np.arange(max(y0 - STRIDE, h), min(y0 + STRIDE, vol.shape[1] - (SIZE - h)) + 1)
    xs = np.arange(max(x0 - STRIDE, h), min(x0 + STRIDE, vol.shape[2] - (SIZE - h)) + 1)
    patches = np.stack([vol[z, y - h:y - h + SIZE, x - h:x - h + SIZE].ravel()
                        for z in zs for y in ys for x in xs])
    prob = clf.predict_proba(patches)[:, 1].reshape(len(zs), len(ys), len(xs))
    prob = uniform_filter(prob, size=SMOOTH, mode="nearest")
    k, i, j = np.unravel_index(int(prob.argmax()), prob.shape)
    return int(zs[k]), int(ys[i]), int(xs[j])


def locate(clf, vol):
    """Best place (z, y, x) in the small volume."""
    prob, (zc, yc, xc) = prob_map(clf, vol)
    prob = uniform_filter(prob, size=SMOOTH, mode="nearest")
    prob[~inside(vol.shape, (zc, yc, xc))] = -1.0
    k, i, j = np.unravel_index(int(prob.argmax()), prob.shape)
    return refine(clf, vol, (int(zc[k]), int(yc[i]), int(xc[j])))


def training_patches(labels, rng):
    X, y = [], []
    for tomo_id, group in labels.groupby("tomo_id"):
        vol, spacing = load_train_volume(tomo_id)
        vol = normalize_volume(vol)
        motors = group[group["z"] >= 0][["z", "y", "x"]].to_numpy(dtype=float) / FACTOR
        for mz, my, mx in np.round(motors).astype(int):
            for dz in (-1, 0, 1):
                for dy, dx in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    p = patch_at(vol, mz + dz, my + dy, mx + dx)
                    if p is not None:
                        X.append(p.ravel())
                        y.append(1)
        found, tries = 0, 0
        while found < NEG_PER_TOMO and tries < NEG_PER_TOMO * 20:
            tries += 1
            c = np.array([rng.integers(0, vol.shape[0]),
                          rng.integers(SIZE // 2, vol.shape[1] - SIZE // 2),
                          rng.integers(SIZE // 2, vol.shape[2] - SIZE // 2)])
            if len(motors) and np.linalg.norm(motors - c, axis=1).min() * spacing * FACTOR < NEG_MIN_DIST:
                continue
            p = patch_at(vol, *c)
            if p is not None:
                X.append(p.ravel())
                y.append(0)
                found += 1
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)


def hard_negatives(clf, labels):
    X = []
    for tomo_id, group in labels.groupby("tomo_id"):
        vol, spacing = load_train_volume(tomo_id)
        vol = normalize_volume(vol)
        motors = group[group["z"] >= 0][["z", "y", "x"]].to_numpy(dtype=float) / FACTOR
        prob, (zc, yc, xc) = prob_map(clf, vol)
        prob[~inside(vol.shape, (zc, yc, xc))] = -1.0
        picked = []
        for idx in np.argsort(prob, axis=None)[::-1]:
            if len(picked) == HARD_PER_TOMO or prob.flat[idx] < 0.5:
                break
            k, i, j = np.unravel_index(idx, prob.shape)
            c = np.array([zc[k], yc[i], xc[j]], dtype=float)
            if len(motors) and np.linalg.norm(motors - c, axis=1).min() * spacing * FACTOR < NEG_MIN_DIST:
                continue
            if any(np.abs(c - q).max() <= 2 * STRIDE for q in picked):
                continue
            picked.append(c)
            X.append(patch_at(vol, *c.astype(int)).ravel())
    return np.array(X, dtype=np.float32).reshape(-1, SIZE * SIZE)


# ---------------------------------------------------------------------------
# 1) train on all train tomograms
# ---------------------------------------------------------------------------

start = time.time()
raw = pd.read_csv(DATA / "train_labels.csv")
labels = raw.rename(columns={"Motor axis 0": "z", "Motor axis 1": "y", "Motor axis 2": "x"})
labels = labels[labels["tomo_id"].isin([p.stem for p in VOL_DIR.glob("*.npz")])]
print(labels["tomo_id"].nunique(), "train tomograms")

clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=0)
X, y = training_patches(labels, np.random.default_rng(0))
clf.fit(X, y)
Xh = hard_negatives(clf, labels)
X = np.vstack([X, Xh])
y = np.concatenate([y, np.zeros(len(Xh), dtype=np.int8)])
clf.fit(X, y)
print(f"trained: {int(y.sum())} positive / {int(len(y) - y.sum())} negative patches "
      f"({len(Xh)} hard negatives), {(time.time() - start) / 60:.1f} min")


# ---------------------------------------------------------------------------
# 2) test tomograms: downsample, find the motor, write submission.csv
# ---------------------------------------------------------------------------

test_dirs = sorted(p for p in (DATA / "test").iterdir() if p.is_dir())
print(len(test_dirs), "test tomograms")

rows = []
with Pool() as pool:
    for n, (tomo_dir, vol) in enumerate(zip(test_dirs, pool.imap(downsample_tomogram, test_dirs)), 1):
        z, y_, x = locate(clf, normalize_volume(vol))
        n_slices = len(list(tomo_dir.glob("*.jpg")))
        with Image.open(next(tomo_dir.glob("*.jpg"))) as img:
            width, height = img.size
        # small coordinate * 8 + 4 = center of the 8x8x8 block, kept inside the tomogram
        rows.append({"tomo_id": tomo_dir.name,
                     "Motor axis 0": min(z * FACTOR + FACTOR // 2, n_slices - 1),
                     "Motor axis 1": min(y_ * FACTOR + FACTOR // 2, height - 1),
                     "Motor axis 2": min(x * FACTOR + FACTOR // 2, width - 1)})
        if n % 25 == 0 or n == len(test_dirs):
            print(f"{n}/{len(test_dirs)} test tomograms, {(time.time() - start) / 60:.1f} min")

submission = pd.DataFrame(rows, columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
submission.to_csv(OUT, index=False)
print(submission.head())
print(f"Wrote {len(submission)} rows to {OUT}")
