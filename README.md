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
- _TODO (EDA): positive/negative split, shape and voxel spacing ranges, key findings._

### Downsampled data (our preprocessing)

The raw data does not fit in Colab, so `kaggle/downsample_train.py` shrinks every training tomogram **8x on all three axes** by block averaging (each 8 consecutive slices are averaged in z, each slice is reduced 8x8 in y and x). Averaging instead of skipping slices also lowers the noise. Because the factor is the same on every axis, coordinates convert by a single division (`z_small = z / 8`).

The script ran on Kaggle (CPU) in about 20 minutes and produced **0.35 GB**, published as the private Kaggle Dataset [`yasunl/group10-byu-downsampled-8x`](https://www.kaggle.com/datasets/yasunl/group10-byu-downsampled-8x):

| File | Content |
| --- | --- |
| `downsampled/<tomo_id>.npz` | `volume` (uint8, Z/8 x Y/8 x X/8) plus `orig_z`, `orig_y`, `orig_x`, `voxel_spacing`, `factor`. Read with `src.data.load_downsampled()` |
| `labels_downsampled.csv` | `train_labels.csv` plus `z_s`, `y_s`, `x_s`: motor coordinates in the small volume (-1 = no motor) |
| `examples/<tomo_id>.npz` | Full-resolution motor slice and 9-slice crops around the motor for 12 tomograms with a motor; middle slice of 4 tomograms without one (image EDA) |

## Approach

### Assignment 1: EDA and baselines

| Baseline | Idea                                                                         |
| -------- | ---------------------------------------------------------------------------- |
| B0       | Always predict "no motor" (-1, -1, -1)                                       |
| B1       | Always predict the mean normalized motor position from the training set      |
| B2       | Classical ML patch classifier on the downsampled volumes; a sliding window picks the most likely location, a threshold decides "no motor" |

### Assignment 2: Proposed solution

_TODO_

## Results

| Model | Validation F2 | Kaggle public | Kaggle private |
| ----- | ------------- | ------------- | -------------- |
| B0    |               |               |                |
| B1    |               | 0.01043       | 0.00801        |
| B2    |               |               |                |

## Kaggle submission history

| Date       | Method                       | Public score | Private score | Notes                                                                          |
| ---------- | ---------------------------- | ------------ | ------------- | ------------------------------------------------------------------------------ |
| 2026-11-02 | B1: mean normalized position | 0.01043      | 0.00801       | Every tomogram predicted as positive; almost no prediction falls within 1000 Å |

## How to run

0. One-time setup (the repository is private): create a GitHub token at GitHub → Settings → Developer settings → Personal access tokens → **Tokens (classic)** with the `repo` scope. In Colab, open Secrets (key icon on the left), add it as `GITHUB_TOKEN` and turn on notebook access. Never paste the token into a notebook cell.
1. Open `assignment1_eda_baseline.ipynb` in Google Colab.
2. Run the first cell: it clones this repository and adds `src/` to the Python path.
3. Run the data download cell and upload your `kaggle.json` (Kaggle → Settings → API → Create New Token). It downloads our downsampled dataset `yasunl/group10-byu-downsampled-8x` (0.35 GB); your Kaggle account needs access to it.
4. Run the remaining cells top to bottom.

Scripts in `kaggle/` run as Kaggle notebooks, with instructions at the top of each file: `downsample_train.py` builds the downsampled dataset, `submit_baseline.py` makes the B0/B1 submissions.

## Repository structure

```
GROUP10_CENG313/
├── README.md
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb   # added in Assignment 2
├── src/
│   ├── data.py      # loading and preprocessing tomograms
│   ├── models.py    # baseline and proposed models
│   ├── plots.py     # EDA plotting helpers
│   └── utils.py     # metric, validation split, coordinate conversion
├── kaggle/          # scripts run as Kaggle notebooks (submissions, preprocessing)
└── notebooks/drafts # personal working notebooks
```

## Team

| Name       | Student ID | GitHub |
| ---------- | ---------- | ------ |
| Yasin Ünal |            | yasunl |
| Kaan       |            |        |
| Salih      |            |        |

## Progress log

| Date       | Member | Work                                                                |
| ---------- | ------ | ------------------------------------------------------------------- |
| 2026-11-02 | Yasin  | Repository created, initial project structure                       |
| 2026-11-02 | Yasin  | First Kaggle late submission (B1); fixed data folder path on Kaggle |
| 2026-11-04 | All    | Tasks split: Yasin data and integration, Kaan EDA, Salih baselines  |
| 2026-11-08 | Yasin  | Downsampled all 648 training tomograms 8x on Kaggle, published the private dataset |
| 2026-11-09 | Yasin  | Notebook downloads the downsampled dataset; README dataset section  |
