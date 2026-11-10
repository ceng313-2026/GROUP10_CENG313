# GROUP10_CENG313: Locating Bacterial Flagellar Motors

CENG313 Introduction to Data Science (Gazi University), group project.

## Competition

- **Kaggle competition:** [BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025)
- **Task:** For each cryo-ET tomogram (3D image of a bacterium), decide whether it contains a flagellar motor and, if so, predict the motor center as (z, y, x) voxel coordinates.
- **Metric:** F2 score. A prediction counts as correct if it lies within 1000 Å (Euclidean distance) of the true motor center.

## Dataset

### Competition data

- **648 training tomograms**, each stored as a folder of 2D JPEG slices along z (`train/<tomo_id>/slice_XXXX.jpg`), about 70 GB in total.
- `train_labels.csv`: one row per motor with the motor center (`Motor axis 0/1/2` = z, y, x in voxels), the tomogram shape, the voxel spacing (Å per voxel) and the number of motors. Tomograms without a motor appear once with coordinates -1.
- The test set is hidden; Kaggle re-runs the submission notebook on it.
- 362 of the 648 tomograms (56%) contain at least one motor, 286 contain none; 313 contain exactly one motor, 49 contain 2–10. The label file has 451 motor coordinates (`Number of motors` sums to 455, so 4 motors are counted but not labelled).
- Voxel spacing takes 10 values between 6.5 and 19.7 Å, so the 1000 Å threshold is 51–154 voxels. Sizes: 300–800 slices in z, 924–1912 voxels in y and x.

### Key EDA findings (notebook sections 2–7)

- Motors cluster around the middle in z but are spread over most of the y–x plane; a fixed guess is within 1000 Å of only 9 of 451 motors (median distance ≈ 4,250 Å). This is why B1 scores ≈ 0.01.
- Motors look different from tomogram to tomogram, and slice brightness does not separate tomograms with and without a motor, so a model has to learn local shape.
- After 8x downsampling the cell edge and the flagellum are still visible and 1000 Å is still about 8 voxels, so the downsampled data is fine enough for B2.

### Downsampled data (our preprocessing)

The raw data does not fit in Colab, so `kaggle/downsample_train.py` shrinks every training tomogram **8x on all three axes** by block averaging (each 8 consecutive slices are averaged in z, each slice is reduced 8x8 in y and x). Averaging instead of skipping slices also lowers the noise. Because the factor is the same on every axis, coordinates convert by a single division (`z_small = z / 8`).

The script ran on Kaggle (CPU) in about 20 minutes and produced **0.35 GB**, published as the private Kaggle Dataset [`yasunl/group10-byu-downsampled-8x`](https://www.kaggle.com/datasets/yasunl/group10-byu-downsampled-8x):

| File                        | Content                                                                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `downsampled/<tomo_id>.npz` | `volume` (uint8, Z/8 x Y/8 x X/8) plus `orig_z`, `orig_y`, `orig_x`, `voxel_spacing`, `factor`. Read with `src.data.load_downsampled()`           |
| `labels_downsampled.csv`    | `train_labels.csv` plus `z_s`, `y_s`, `x_s`: motor coordinates in the small volume (-1 = no motor)                                                |
| `examples/<tomo_id>.npz`    | Full-resolution motor slice and 9-slice crops around the motor for 12 tomograms with a motor; middle slice of 4 tomograms without one (image EDA) |

## Approach

### Assignment 1: EDA and baselines

| Baseline | Idea                                                                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| B0       | Always predict "no motor" (-1, -1, -1)                                                                                                                                         |
| B1       | Always predict the mean normalized motor position from the training set                                                                                                        |
| B2       | Classical ML patch classifier (Random Forest) on the downsampled volumes; a sliding window picks the most likely location. Threshold 0 was best, so B2 always predicts a motor |

### Assignment 2: Proposed solution

_TODO_

## Results

| Model | Validation F2 | Kaggle public | Kaggle private |
| ----- | ------------- | ------------- | -------------- |
| B0    | 0.0000        | 0.00000       | 0.00000        |
| B1    | 0.0426        | 0.01043       | 0.00801        |
| B2    | 0.2582        | 0.10245       | 0.07515        |

Validation: 518 train / 130 validation tomograms (`splits/val_ids.csv`). For Kaggle, the models are trained on all 648 train tomograms. Details: `assignment1_eda_baseline.ipynb`, sections 8–11 (working version: `notebooks/drafts/salih_baseline.ipynb`).

## Kaggle submission history

| Date       | Method                                                | Public score | Private score | Notes                                                                          |
| ---------- | ----------------------------------------------------- | ------------ | ------------- | ------------------------------------------------------------------------------ |
| 2026-11-02 | B1: mean normalized position                          | 0.01043      | 0.00801       | Every tomogram predicted as positive; almost no prediction falls within 1000 Å |
| 2026-11-09 | B2: patch classifier (Random Forest) + sliding window | 0.10245      | 0.07515       | Trained on all 648 tomograms; always predicts a motor                          |
| 2026-11-10 | B0: always no motor                                   | 0.00000      | 0.00000       | Lower bound                                                                    |

## How to run

0. One-time setup (the repository is private): create a GitHub token at GitHub → Settings → Developer settings → Personal access tokens → **Tokens (classic)** with the `repo` scope. In Colab, open Secrets (key icon on the left), add it as `GITHUB_TOKEN` and turn on notebook access. Never paste the token into a notebook cell.
1. Open `assignment1_eda_baseline.ipynb` in Google Colab.
2. Run the first cell: it clones this repository and adds `src/` to the Python path.
3. Run the data download cell and upload your `kaggle.json` (Kaggle → Settings → API → Create New Token). It downloads our downsampled dataset `yasunl/group10-byu-downsampled-8x` (0.35 GB); your Kaggle account needs access to it.
4. Run the remaining cells top to bottom ("Run all" takes about 15 minutes on a Colab CPU, mostly training B2).

Scripts in `kaggle/` run as Kaggle notebooks, with instructions at the top of each file: `downsample_train.py` builds the downsampled dataset, `submit_baseline.py` makes the B0/B1 submissions, `submit_b2.py` makes the B2 submission (it also needs our downsampled dataset as input).

## Repository structure

```
GROUP10_CENG313/
├── README.md
├── requirements.txt
├── assignment1_eda_baseline.ipynb   # Assignment 1: EDA + baselines (run in Colab)
├── assignment2_solution.ipynb       # added in Assignment 2
├── src/
│   ├── data.py      # loading, downsampling and reading tomograms
│   ├── plots.py     # EDA plotting helpers
│   ├── utils.py     # F2 metric, validation split, coordinate conversion
│   └── models.py    # baselines B0, B1, B2
├── kaggle/          # scripts run as Kaggle notebooks
│   ├── downsample_train.py   # builds the downsampled dataset
│   ├── submit_baseline.py    # B0 / B1 submission
│   └── submit_b2.py          # B2 submission
├── splits/val_ids.csv        # fixed validation split (130 tomograms)
├── tests/test_utils.py       # hand-checked test of the F2 metric
└── notebooks/drafts/         # each member's working notebook (kaan_eda, salih_baseline)
```

## Team

| Name                    | Student ID  | GitHub                |
| ----------------------- | ----------- | --------------------- |
| Yasin Ünal              | 23118080001 | yasunl                |
| Abdullah Kaan Yurtseven | 23118080027 | AbdullahKaanYurtseven |
| Salih Özçam             | 23118080016 | ozcamm                |

## Progress log

| Date       | Member | Work                                                                                                 |
| ---------- | ------ | ---------------------------------------------------------------------------------------------------- |
| 2026-11-02 | Yasin  | Repository created, initial project structure                                                        |
| 2026-11-02 | Yasin  | First Kaggle late submission (B1); fixed data folder path on Kaggle                                  |
| 2026-11-04 | All    | Tasks split: Yasin data and integration, Kaan EDA, Salih baselines                                   |
| 2026-11-08 | Yasin  | Downsampled all 648 training tomograms 8x on Kaggle, published the private dataset                   |
| 2026-11-08 | Salih  | F2 metric test, validation split saved (`splits/val_ids.csv`), B0/B1 validation scores               |
| 2026-11-09 | Yasin  | Notebook downloads the downsampled dataset; README dataset section                                   |
| 2026-11-09 | Salih  | B2 patch classifier + sliding window (`src/models.py`), B2 Kaggle submission (`kaggle/submit_b2.py`) |
| 2026-11-09 | Kaan   | EDA sections 1–4: problem, data description, metadata and motor location EDA (`src/plots.py`)        |
| 2026-11-10 | Salih  | Baseline notebook (`notebooks/drafts/salih_baseline.ipynb`), results in README                       |
| 2026-11-10 | Kaan   | Image EDA (full resolution vs 8x downsampled), summary of observations                               |
| 2026-11-11 | Yasin  | Merged all work into `assignment1_eda_baseline.ipynb`, README and requirements completed             |
