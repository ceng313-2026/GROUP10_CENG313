# GROUP10_CENG313: Locating Bacterial Flagellar Motors

CENG313 Introduction to Data Science (Gazi University), group project.

## Competition

- **Kaggle competition:** [BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025)
- **Task:** For each cryo-ET tomogram (3D image of a bacterium), decide whether it contains a flagellar motor and, if so, predict the motor center as (z, y, x) voxel coordinates.
- **Metric:** F2 score. A prediction counts as correct if it lies within 1000 Å (Euclidean distance) of the true motor center.

## Dataset

_TODO (Assignment 1): number of tomograms, positive/negative split, image format, shapes, voxel spacing, key EDA findings._

## Approach

### Assignment 1: EDA and baselines

| Baseline | Idea                                                                         |
| -------- | ---------------------------------------------------------------------------- |
| B0       | Always predict "no motor" (-1, -1, -1)                                       |
| B1       | Always predict the mean normalized motor position from the training set      |
| B2       | Classical ML presence classifier on simple volume features, location from B1 |

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

1. Open `assignment1_eda_baseline.ipynb` in Google Colab.
2. Run the first cell: it clones this repository and adds `src/` to the Python path.
3. Run the data download cell and upload your `kaggle.json` (Kaggle → Settings → API → Create New Token). You must have joined the competition and accepted its rules first.
4. Run the remaining cells top to bottom.

Kaggle submissions are made from `kaggle/submit_baseline.py`; instructions are at the top of that file.

## Repository structure

```
GROUP10_CENG313/
├── README.md
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb
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
| Yasin Ünal |            |        |
| Kaan       |            |        |
| Salih      |            |        |

## Progress log

| Date       | Member | Work                                                                |
| ---------- | ------ | ------------------------------------------------------------------- |
| 2026-11-02 | Yasin  | Repository created, initial project structure                       |
| 2026-11-02 | Yasin  | First Kaggle late submission (B1); fixed data folder path on Kaggle |
