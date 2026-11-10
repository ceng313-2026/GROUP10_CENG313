"""Baseline models.

Owner: baseline and evaluation member.

Every model has the same interface:
    model.fit(labels)            labels = output of data.load_labels (train part only)
    model.predict(tomos)         tomos = one row per tomogram with tomo_id, shape_z, shape_y, shape_x
                                 returns DataFrame tomo_id, z, y, x  (-1 = no motor)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter

from src import data


class NoMotorBaseline:
    """B0: always predict "no motor". F2 = 0, the lower bound."""

    def fit(self, labels):
        return self

    def predict(self, tomos):
        out = tomos[["tomo_id"]].copy()
        out["z"] = out["y"] = out["x"] = -1
        return out


class MeanPositionBaseline:
    """B1: always predict a motor at the mean normalized position of training motors.

    Position is normalized by the tomogram shape (z / shape_z etc.) because tomograms
    have different sizes, then scaled back to each test tomogram's shape.
    """

    def fit(self, labels):
        motors = labels[labels["z"] >= 0]
        self.mean_ = {
            axis: (motors[axis] / motors[f"shape_{axis}"]).mean() for axis in ("z", "y", "x")
        }
        return self

    def predict(self, tomos):
        out = tomos[["tomo_id"]].copy()
        for axis in ("z", "y", "x"):
            out[axis] = (self.mean_[axis] * tomos[f"shape_{axis}"]).round().astype(int)
        return out


# ---------------------------------------------------------------------------
# B2: classical ML patch classifier + sliding window on the 8x downsampled volumes
# ---------------------------------------------------------------------------

def normalize_volume(vol):
    """Make mean 0 and std 1 for one tomogram.
    Some tomograms are brighter than others, this removes that difference."""
    vol = vol.astype(np.float32)
    return (vol - vol.mean()) / (vol.std() + 1e-6)


def patch_at(vol, z, y, x, size):
    """Cut a 2D square (size x size) with center (z, y, x) from the small volume.
    Returns None if the square goes outside the volume."""
    h = size // 2
    if not (0 <= z < vol.shape[0] and h <= y <= vol.shape[1] - (size - h)
            and h <= x <= vol.shape[2] - (size - h)):
        return None
    return vol[z, y - h:y - h + size, x - h:x - h + size]


def extract_training_patches(labels, volume_dir, size=16, neg_per_tomo=30,
                             neg_min_dist_angstrom=2000.0, seed=0):
    """Make the B2 training data from the small volumes.

    Positive patches: around each motor, on its slice and the slices above and below.
    We also move the patch 1 pixel up/down/left/right to get more motor examples
    (we have few motors).
    Negative patches: random places in each tomogram that are more than
    neg_min_dist_angstrom away from all motors.

    Returns X with shape (n, size*size) and y with shape (n,), 1 = motor, 0 = no motor.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for tomo_id, group in labels.groupby("tomo_id"):
        vol, meta = data.load_downsampled(Path(volume_dir) / f"{tomo_id}.npz")
        vol = normalize_volume(vol)
        factor = meta["factor"]
        motors = group[group["z"] >= 0][["z", "y", "x"]].to_numpy(dtype=float) / factor
        spacing_small = meta["voxel_spacing"] * factor          # Angstrom for one small voxel

        # positives
        for mz, my, mx in np.round(motors).astype(int):
            for dz in (-1, 0, 1):
                for dy, dx in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    p = patch_at(vol, mz + dz, my + dy, mx + dx, size)
                    if p is not None:
                        X.append(p.ravel())
                        y.append(1)

        # negatives
        found, tries = 0, 0
        while found < neg_per_tomo and tries < neg_per_tomo * 20:
            tries += 1
            c = np.array([rng.integers(0, vol.shape[0]),
                          rng.integers(size // 2, vol.shape[1] - size // 2),
                          rng.integers(size // 2, vol.shape[2] - size // 2)])
            if len(motors) and (np.linalg.norm(motors - c, axis=1).min() * spacing_small
                                < neg_min_dist_angstrom):
                continue
            p = patch_at(vol, *c, size)
            if p is not None:
                X.append(p.ravel())
                y.append(0)
                found += 1
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)


class PatchClassifierBaseline:
    """B2: a simple classifier (for example Logistic Regression or Random Forest)
    says "motor / no motor" for small 2D patches. We slide a window over the small
    volume and take the place with the highest motor probability.

    The first version did not work well (see notebook, Block 5.4), so we added:
    - hard negatives: after the first training, we run the model on the TRAIN
      tomograms, take places far from real motors where the model still says
      "motor", add them as negatives and train again;
    - margin: we do not search the outer part of the volume (many wrong
      predictions were near the border);
    - smoothing: we average the probability with the nearby windows, so one
      noisy window does not win;
    - refine: around the best window we check every slice and every pixel,
      because the first search only checks every 4th pixel and every 2nd slice.

    predict() returns tomo_id, z, y, x in ORIGINAL voxel coordinates and `prob`.
    with_threshold() changes low-probability rows to "no motor" (-1), so we can
    try many thresholds without running the sliding window again.
    """

    def __init__(self, classifier, volume_dir, size=16, stride=4, z_step=2,
                 neg_per_tomo=30, hard_rounds=1, hard_per_tomo=10, margin=0.1,
                 smooth=3, refine=True, neg_min_dist_angstrom=2000.0, seed=0):
        self.classifier = classifier
        self.volume_dir = Path(volume_dir)
        self.size, self.stride, self.z_step = size, stride, z_step
        self.neg_per_tomo, self.seed = neg_per_tomo, seed
        self.hard_rounds, self.hard_per_tomo = hard_rounds, hard_per_tomo
        self.margin, self.smooth, self.refine = margin, smooth, refine
        self.neg_min_dist = neg_min_dist_angstrom

    # --- sliding window -----------------------------------------------------
    def prob_map(self, vol):
        """Motor probability for every window of a normalized volume.

        Returns prob with shape (nz, ny, nx) and the window centers (zc, yc, xc)
        in small-volume coordinates. We call predict_proba only once (faster)."""
        s, st = self.size, self.stride
        zc = np.arange(0, vol.shape[0], self.z_step)
        win = sliding_window_view(vol[zc], (s, s), axis=(1, 2))[:, ::st, ::st]  # (nz, ny, nx, s, s)
        nz, ny, nx = win.shape[:3]
        prob = self.classifier.predict_proba(win.reshape(nz * ny * nx, s * s))[:, 1]
        yc = np.arange(ny) * st + s // 2
        xc = np.arange(nx) * st + s // 2
        return prob.reshape(nz, ny, nx), (zc, yc, xc)

    def _inside(self, shape, centers):
        """True for windows that are not in the border margin."""
        keep = [(c >= self.margin * n) & (c < (1 - self.margin) * n)
                for c, n in zip(centers, shape)]
        return keep[0][:, None, None] & keep[1][None, :, None] & keep[2][None, None, :]

    def locate(self, vol):
        """Best window center (z, y, x) in small-volume coordinates and its probability."""
        prob, (zc, yc, xc) = self.prob_map(vol)
        if self.smooth > 1:
            prob = uniform_filter(prob, size=self.smooth, mode="nearest")
        prob[~self._inside(vol.shape, (zc, yc, xc))] = -1.0
        k, i, j = np.unravel_index(int(prob.argmax()), prob.shape)
        best = (int(zc[k]), int(yc[i]), int(xc[j]))
        if self.refine:
            best = self._refine(vol, best)
        # we keep the probability from the first search, so the threshold works the same with refine
        return best, float(prob[k, i, j])

    def _refine(self, vol, center):
        """Check every pixel and slice near the best window (one grid step around it)."""
        s, h = self.size, self.size // 2
        z0, y0, x0 = center
        zs = np.arange(max(z0 - self.z_step, 0), min(z0 + self.z_step, vol.shape[0] - 1) + 1)
        ys = np.arange(max(y0 - self.stride, h), min(y0 + self.stride, vol.shape[1] - (s - h)) + 1)
        xs = np.arange(max(x0 - self.stride, h), min(x0 + self.stride, vol.shape[2] - (s - h)) + 1)
        patches = np.stack([vol[z, y - h:y - h + s, x - h:x - h + s].ravel()
                            for z in zs for y in ys for x in xs])
        prob = self.classifier.predict_proba(patches)[:, 1].reshape(len(zs), len(ys), len(xs))
        if self.smooth > 1:
            prob = uniform_filter(prob, size=self.smooth, mode="nearest")
        k, i, j = np.unravel_index(int(prob.argmax()), prob.shape)
        return int(zs[k]), int(ys[i]), int(xs[j])

    # --- training -----------------------------------------------------------
    def _hard_negatives(self, labels):
        """Places in the train tomograms where the model says "motor" but there is no motor near."""
        X = []
        for tomo_id, group in labels.groupby("tomo_id"):
            vol, meta = data.load_downsampled(self.volume_dir / f"{tomo_id}.npz")
            vol = normalize_volume(vol)
            f = meta["factor"]
            motors = group[group["z"] >= 0][["z", "y", "x"]].to_numpy(dtype=float) / f
            prob, (zc, yc, xc) = self.prob_map(vol)
            prob[~self._inside(vol.shape, (zc, yc, xc))] = -1.0
            picked = []
            for idx in np.argsort(prob, axis=None)[::-1]:
                if len(picked) == self.hard_per_tomo or prob.flat[idx] < 0.5:
                    break
                k, i, j = np.unravel_index(idx, prob.shape)
                c = np.array([zc[k], yc[i], xc[j]], dtype=float)
                if len(motors) and (np.linalg.norm(motors - c, axis=1).min()
                                    * meta["voxel_spacing"] * f < self.neg_min_dist):
                    continue
                # skip windows very close to one we already took (it is the same thing)
                if any(np.abs(c - q).max() <= 2 * self.stride for q in picked):
                    continue
                picked.append(c)
                X.append(patch_at(vol, *c.astype(int), self.size).ravel())
        return np.array(X, dtype=np.float32).reshape(-1, self.size * self.size)

    def fit(self, labels):
        X, y = extract_training_patches(labels, self.volume_dir, self.size, self.neg_per_tomo,
                                        self.neg_min_dist, seed=self.seed)
        self.classifier.fit(X, y)
        self.n_hard_ = 0
        for _ in range(self.hard_rounds):
            Xh = self._hard_negatives(labels)
            self.n_hard_ += len(Xh)
            X = np.vstack([X, Xh])
            y = np.concatenate([y, np.zeros(len(Xh), dtype=np.int8)])
            self.classifier.fit(X, y)
        self.n_pos_, self.n_neg_ = int(y.sum()), int(len(y) - y.sum())
        return self

    def predict(self, tomos):
        rows = []
        for tomo_id in tomos["tomo_id"]:
            vol, meta = data.load_downsampled(self.volume_dir / f"{tomo_id}.npz")
            (z, y, x), p = self.locate(normalize_volume(vol))
            f = meta["factor"]
            # small coordinate * 8 + 4 = center of the 8x8x8 block in original voxels
            rows.append({"tomo_id": tomo_id, "z": int(z * f + f // 2),
                         "y": int(y * f + f // 2), "x": int(x * f + f // 2), "prob": p})
        return pd.DataFrame(rows)


def with_threshold(predictions, threshold):
    """If prob is lower than the threshold, the prediction becomes "no motor" (-1)."""
    out = predictions.copy()
    low = out["prob"] < threshold
    out.loc[low, ["z", "y", "x"]] = -1
    return out
