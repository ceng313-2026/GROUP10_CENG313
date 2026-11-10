"""EDA plotting helpers.

Owner: EDA member.

Keep each figure in its own function so the notebook only calls it, e.g.
    plots.motor_position_histograms(labels)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def class_balance(tomos, ax=None):
    """Bar chart of tomograms with and without a motor."""
    ax = ax or plt.gca()
    counts = tomos["has_motor"].map({True: "motor", False: "no motor"}).value_counts()
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("tomograms")
    ax.set_title("Tomograms with and without a motor")
    return ax


def voxel_spacing(tomos, ax=None):
    """Bar chart of how many tomograms have each voxel spacing."""
    ax = ax or plt.gca()
    counts = tomos["voxel_spacing"].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Voxel spacing (Å)")
    ax.set_ylabel("Number of tomograms")
    ax.set_title("Voxel spacing of the training tomograms")
    return ax


def tomogram_sizes(tomos):
    """Tomogram size along z, y and x: one bar chart per axis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["shape_z", "shape_y", "shape_x"]):
        tomos[col].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("Size (voxels)")
        ax.set_ylabel("Number of tomograms")
    fig.tight_layout()
    return fig

def normalized_positions(labels):
    """Motors only, with positions divided by tomogram size (0 = first voxel, 1 = last)."""
    motors = labels[labels["has_motor"]].copy()
    for axis in ["z", "y", "x"]:
        motors[f"{axis}_norm"] = motors[axis] / motors[f"shape_{axis}"]
    return motors


def position_histograms(labels):
    """Histograms of normalized motor positions along z, y and x."""
    motors = normalized_positions(labels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, axis in zip(axes, ["z", "y", "x"]):
        ax.hist(motors[f"{axis}_norm"], bins=20, range=(0, 1))
        ax.set_title(f"{axis} (normalized)")
        ax.set_xlabel("Position / tomogram size")
        ax.set_ylabel("Number of motors")
    fig.tight_layout()
    return fig


def position_yx(labels, ax=None):
    """Top view: normalized motor positions in the y-x plane, with B1's mean position."""
    ax = ax or plt.gca()
    motors = normalized_positions(labels)
    ax.scatter(motors["x_norm"], motors["y_norm"], s=10, alpha=0.5, label="motors")
    ax.scatter(motors["x_norm"].mean(), motors["y_norm"].mean(),
               color="red", marker="x", s=150, label="mean position (B1 prediction)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x / tomogram size")
    ax.set_ylabel("y / tomogram size")
    ax.set_title("Motor positions, top view (normalized)")
    ax.legend()
    return ax


def distances_to_mean(labels):
    """Distance in Å from each motor to B1's prediction (mean normalized position scaled to its tomogram)."""
    motors = normalized_positions(labels)
    squared = 0
    for axis in ["z", "y", "x"]:
        predicted = motors[f"{axis}_norm"].mean() * motors[f"shape_{axis}"]   # B1's guess, in voxels
        diff = (motors[axis] - predicted) * motors["voxel_spacing"]            # difference, in Å
        squared = squared + diff ** 2
    return np.sqrt(squared)


def distance_to_mean_hist(labels, threshold=1000, ax=None):
    """Histogram of motor distances to B1's prediction, with the 1000 Å threshold."""
    ax = ax or plt.gca()
    dist = distances_to_mean(labels)
    ax.hist(dist, bins=40)
    ax.axvline(threshold, color="red", linestyle="--", label=f"{threshold} Å threshold")
    ax.set_xlabel("Distance to B1's prediction (Å)")
    ax.set_ylabel("Number of motors")
    ax.set_title("How far is each motor from B1's prediction?")
    ax.legend()
    return ax


def load_examples(folder):
    """Read every .npz file in the examples/ folder of our Kaggle dataset into {tomo_id: {key: array}}."""
    examples = {}
    for path in sorted(Path(folder).glob("*.npz")):
        with np.load(path) as f:
            examples[path.stem] = {key: f[key] for key in f.files}
    return examples


def motor_vs_no_motor(motor_example, no_motor_example, voxel_spacing, threshold=1000):
    """Slice through a motor (red circle = `threshold` Å around it) next to a slice without a motor."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    z, y, x = motor_example["motor_zyx"]
    axes[0].imshow(motor_example["motor_slice"], cmap="gray")
    axes[0].add_patch(plt.Circle((x, y), threshold / voxel_spacing, color="red", fill=False, linewidth=1.5))
    axes[0].set_title(f"With a motor (slice z = {z}); red circle = {threshold} Å")
    axes[1].imshow(no_motor_example["middle_slice"], cmap="gray")
    axes[1].set_title("Without a motor (middle slice)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def crop_grid(examples, ncols=6):
    """Close-up (192 x 192 voxels) of every motor example; the motor is at the centre of each crop."""
    motors = [(name, e) for name, e in examples.items() if "crops" in e]
    nrows = int(np.ceil(len(motors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.7 * nrows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, e) in zip(axes.flat, motors):
        middle = min(4, e["motor_zyx"][0])          # the crop slice that contains the motor
        ax.imshow(e["crops"][middle], cmap="gray")
        ax.set_title(name, fontsize=9)
    fig.tight_layout()
    return fig


def full_vs_downsampled(example, volume, factor=8):
    """The same motor at full resolution and in the downsampled volume, covering the same physical area."""
    z, y, x = example["motor_zyx"]
    crop = example["crops"][min(4, z)]
    half = crop.shape[0] // (2 * factor)            # 192 voxels / 8 = 24, so 12 on each side
    sz, sy, sx = z // factor, y // factor, x // factor
    small = volume[sz, max(0, sy - half):sy + half, max(0, sx - half):sx + half]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(crop, cmap="gray")
    axes[0].set_title(f"Full resolution ({crop.shape[0]} x {crop.shape[1]} voxels)")
    axes[1].imshow(small, cmap="gray")
    axes[1].set_title(f"Downsampled {factor}x ({small.shape[0]} x {small.shape[1]} voxels)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def intensity_histograms(examples, ax=None):
    """Pixel intensity distribution of each example slice: red = with a motor, blue = without."""
    ax = ax or plt.gca()
    for e in examples.values():
        has_motor = "crops" in e
        image = e["motor_slice"] if has_motor else e["middle_slice"]
        ax.hist(image.ravel(), bins=64, range=(0, 255), density=True, histtype="step",
                color="tab:red" if has_motor else "tab:blue", alpha=0.7)
    ax.plot([], [], color="tab:red", label="with a motor")
    ax.plot([], [], color="tab:blue", label="without a motor")
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Density")
    ax.set_title("Intensity distribution of the example slices")
    ax.legend()
    return ax