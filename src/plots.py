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