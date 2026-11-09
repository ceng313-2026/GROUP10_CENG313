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
    """Zero mean, unit variance per tomogram, so brightness differences between
    tomograms do not matter to the classifier."""
    vol = vol.astype(np.float32)
    return (vol - vol.mean()) / (vol.std() + 1e-6)


def patch_at(vol, z, y, x, size):
    """2D patch of shape (size, size) centered at (z, y, x) in a downsampled volume,
    or None if it does not fit inside the volume."""
    h = size // 2
    if not (0 <= z < vol.shape[0] and h <= y <= vol.shape[1] - (size - h)
            and h <= x <= vol.shape[2] - (size - h)):
        return None
    return vol[z, y - h:y - h + size, x - h:x - h + size]


def extract_training_patches(labels, volume_dir, size=16, neg_per_tomo=30,
                             neg_min_dist_angstrom=2000.0, seed=0):
    """Build the B2 training set from the downsampled volumes.

    Positives: patches centered on each motor, on its slice and the two neighbouring
    slices, plus 1-pixel shifts (data augmentation, motors are rare).
    Negatives: random patches from every tomogram whose center is farther than
    neg_min_dist_angstrom from all motors of that tomogram.

    Returns X (n, size*size) and y (n,) with 1 = motor, 0 = background.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for tomo_id, group in labels.groupby("tomo_id"):
        vol, meta = data.load_downsampled(Path(volume_dir) / f"{tomo_id}.npz")
        vol = normalize_volume(vol)
        factor = meta["factor"]
        motors = group[group["z"] >= 0][["z", "y", "x"]].to_numpy(dtype=float) / factor
        spacing_small = meta["voxel_spacing"] * factor          # Angstrom per small voxel

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
    """B2: a classical classifier (e.g. Logistic Regression, Random Forest) decides
    "motor / no motor" for small 2D patches; a sliding window over the downsampled
    volume finds the most motor-like position.

    Three things make the window search work in practice (see notebook, Block 5.4):
    - hard negative mining: after a first fit, the most motor-like windows far away
      from real motors in the TRAINING tomograms are added as negatives and the model
      is refit, so it learns the background structures it confuses with motors;
    - margin: the outer part of each axis (artefacts near the volume border) is not searched;
    - smoothing: the probability map is averaged over neighbouring windows, so a motor
      that looks like a motor on several slices beats a single noisy window;
    - refinement: around the best coarse window, every slice and every pixel shift
      within one grid step is scored, so the position is not limited to the coarse grid.

    predict() returns tomo_id, z, y, x in ORIGINAL voxel coordinates plus `prob`
    (the best smoothed window probability). Use with_threshold() to turn
    low-probability predictions into "no motor" (-1), so the threshold can be
    tuned without running the sliding window again.
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
        """Motor probability for every window of a (normalized) volume.

        Returns prob (nz, ny, nx) and the window centers (zc, yc, xc) in downsampled
        coordinates. All windows are scored in one predict_proba call."""
        s, st = self.size, self.stride
        zc = np.arange(0, vol.shape[0], self.z_step)
        win = sliding_window_view(vol[zc], (s, s), axis=(1, 2))[:, ::st, ::st]  # (nz, ny, nx, s, s)
        nz, ny, nx = win.shape[:3]
        prob = self.classifier.predict_proba(win.reshape(nz * ny * nx, s * s))[:, 1]
        yc = np.arange(ny) * st + s // 2
        xc = np.arange(nx) * st + s // 2
        return prob.reshape(nz, ny, nx), (zc, yc, xc)

    def _inside(self, shape, centers):
        """Boolean grid: True for windows whose center is outside the border margin."""
        keep = [(c >= self.margin * n) & (c < (1 - self.margin) * n)
                for c, n in zip(centers, shape)]
        return keep[0][:, None, None] & keep[1][None, :, None] & keep[2][None, None, :]

    def locate(self, vol):
        """Best window center (z, y, x) in downsampled coordinates and its probability."""
        prob, (zc, yc, xc) = self.prob_map(vol)
        if self.smooth > 1:
            prob = uniform_filter(prob, size=self.smooth, mode="nearest")
        prob[~self._inside(vol.shape, (zc, yc, xc))] = -1.0
        k, i, j = np.unravel_index(int(prob.argmax()), prob.shape)
        best = (int(zc[k]), int(yc[i]), int(xc[j]))
        if self.refine:
            best = self._refine(vol, best)
        # the coarse (smoothed) probability is kept, so thresholds mean the same with or without refine
        return best, float(prob[k, i, j])

    def _refine(self, vol, center):
        """Fine search (step 1 in every axis) within one coarse grid step around center."""
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
        """Most motor-like windows far from every real motor, from the training tomograms."""
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
                # skip windows right next to one already picked (same structure)
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
            # center of the 8x8x8 block back in original voxels
            rows.append({"tomo_id": tomo_id, "z": int(z * f + f // 2),
                         "y": int(y * f + f // 2), "x": int(x * f + f // 2), "prob": p})
        return pd.DataFrame(rows)


def with_threshold(predictions, threshold):
    """Predictions below the probability threshold become "no motor" (-1)."""
    out = predictions.copy()
    low = out["prob"] < threshold
    out.loc[low, ["z", "y", "x"]] = -1
    return out
