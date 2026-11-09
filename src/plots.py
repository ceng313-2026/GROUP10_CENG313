"""EDA plotting helpers.

Owner: EDA member.

Keep each figure in its own function so the notebook only calls it, e.g.
    plots.motor_position_histograms(labels)
"""

import matplotlib.pyplot as plt


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