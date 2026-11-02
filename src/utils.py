"""Evaluation metric, validation split and coordinate helpers.

Owner: baseline and evaluation member.

Metric (from the competition): for each tomogram,
  - motor present, prediction within 1000 Angstrom  -> TP
  - motor present, prediction too far or -1         -> FN
  - no motor, prediction given                      -> FP
  - no motor, prediction -1                         -> TN
Then F-beta with beta = 2 (recall weighs more than precision).
"""

import numpy as np
from sklearn.model_selection import train_test_split

THRESHOLD_ANGSTROM = 1000.0


def fbeta(tp, fp, fn, beta=2.0):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    b2 = beta**2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def score(labels, predictions, threshold=THRESHOLD_ANGSTROM, beta=2.0, return_counts=False):
    """F2 score of predictions against labels.

    labels:      DataFrame, one row per motor, columns tomo_id, z, y, x, voxel_spacing
                 (tomograms without a motor have z = y = x = -1). Output of data.load_labels.
    predictions: DataFrame, one row per tomogram, columns tomo_id, z, y, x (-1 = no motor).

    For a tomogram with several true motors, a prediction counts as correct if it is
    within the threshold of any of them.
    """
    preds = predictions.set_index("tomo_id")
    tp = fp = fn = 0
    for tomo_id, group in labels.groupby("tomo_id"):
        true_motors = group[group["z"] >= 0][["z", "y", "x"]].to_numpy(dtype=float)
        has_true = len(true_motors) > 0

        p = preds.loc[tomo_id, ["z", "y", "x"]].to_numpy(dtype=float)
        has_pred = p[0] >= 0

        if has_true and has_pred:
            spacing = group["voxel_spacing"].iloc[0]
            dist = np.linalg.norm((true_motors - p) * spacing, axis=1).min()
            if dist <= threshold:
                tp += 1
            else:
                fn += 1
        elif has_true:
            fn += 1
        elif has_pred:
            fp += 1

    result = fbeta(tp, fp, fn, beta)
    if return_counts:
        return result, {"tp": tp, "fp": fp, "fn": fn}
    return result


def split_tomograms(tomos, val_size=0.2, seed=42):
    """Train/validation split at tomogram level, stratified by has_motor.

    tomos: one row per tomogram (data.tomogram_table). Returns two lists of tomo_id.
    Splitting by tomogram keeps parts of one tomogram from landing on both sides.
    """
    train_ids, val_ids = train_test_split(
        tomos["tomo_id"],
        test_size=val_size,
        stratify=tomos["has_motor"],
        random_state=seed,
    )
    return list(train_ids), list(val_ids)


def voxel_to_angstrom(distance_voxels, voxel_spacing):
    return distance_voxels * voxel_spacing


def to_submission(predictions):
    """Rename short columns back to the competition's submission format."""
    out = predictions[["tomo_id", "z", "y", "x"]].copy()
    out.columns = ["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"]
    return out
