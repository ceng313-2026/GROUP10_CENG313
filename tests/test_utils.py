"""Hand-checked test for the F2 metric in src/utils.py (score).

Run from the repository root:
    python tests/test_utils.py

Each tomogram below covers one case of the competition metric, so the
expected TP / FP / FN counts and the F2 value can be worked out by hand.
"""
import pathlib
import sys

# Make "from src import utils" work no matter where the script is started from.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from src import utils

# voxel_spacing = 10 Angstrom for every tomogram, so the 1000 Angstrom threshold
# equals 100 voxels in this example.
labels = pd.DataFrame({
    "tomo_id":       ["a",  "b",  "c",  "d",  "e",  "m",  "m"],
    "z":             [50,   50,   50,   -1,   -1,   10,   80],
    "y":             [100,  100,  100,  -1,   -1,   10,   80],
    "x":             [100,  100,  100,  -1,   -1,   10,   80],
    "voxel_spacing": [10.0] * 7,
})
predictions = pd.DataFrame({
    "tomo_id": ["a", "b", "c", "d", "e", "m"],
    "z":       [55,  50,  -1,  30,  -1,  82],
    "y":       [100, 300, -1,  30,  -1,  80],
    "x":       [100, 100, -1,  30,  -1,  80],
})
# a: motor present, prediction 5 voxels = 50 A away         -> TP
# b: motor present, prediction 200 voxels = 2000 A away     -> FN (the competition counts a far prediction as a miss only)
# c: motor present, predicted "no motor" (-1)                -> FN
# d: no motor, a motor was predicted                         -> FP
# e: no motor, predicted "no motor"                          -> TN (does not affect F2)
# m: two motors, prediction 2 voxels = 20 A from the second  -> TP (closest motor counts)

f2, counts = utils.score(labels, predictions, return_counts=True)
print(f2, counts)

assert counts == {"tp": 2, "fp": 1, "fn": 2}, counts
# precision = 2 / 3, recall = 2 / 4 = 1 / 2
# F2 = 5 * P * R / (4 * P + R) = (5/3) / (8/3 + 1/2) = 10 / 19
assert abs(f2 - 10 / 19) < 1e-9, f2

# Edge case: no true positives at all must give F2 = 0, not a division error.
f2_zero = utils.score(labels[labels.tomo_id == "c"], predictions[predictions.tomo_id == "c"])
assert f2_zero == 0.0, f2_zero

print("OK")
