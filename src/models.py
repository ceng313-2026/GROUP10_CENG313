"""Baseline models.

Owner: baseline and evaluation member.

Every model has the same interface:
    model.fit(labels)            labels = output of data.load_labels (train part only)
    model.predict(tomos)         tomos = one row per tomogram with tomo_id, shape_z, shape_y, shape_x
                                 returns DataFrame tomo_id, z, y, x  (-1 = no motor)
"""

import pandas as pd


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


# B2 (classical ML presence classifier) comes next.
