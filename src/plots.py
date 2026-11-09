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