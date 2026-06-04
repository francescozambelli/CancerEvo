# scripts/plots/plot_spatial_study.py
#
# Load spatial grids and plot a comparison of the spatial tissue state over time.

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "spatial_study"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "solid"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Define custom colors
# 0 = Wild-type (Beige/Pale Green)
# 1 = Cancer (Vibrant Red)
# 2 = Dead (Dark Charcoal)
colors = ["#E9F5EC", "#E63946", "#3A3A3A"]
cmap = ListedColormap(colors)

scenarios = [
    {"id": "dec_below", "title": "Below Boundary\n(rmax=0.8480, dmu=0.0045)"},
    {"id": "dec_on", "title": "On Boundary (Transition)\n(rmax=0.8480, dmu=0.0058)"},
    {"id": "dec_above", "title": "Above Boundary\n(rmax=0.8480, dmu=0.0070)"}
]

steps = [0, 50, 100, 200, 400]

fig, axes = plt.subplots(len(scenarios), len(steps), figsize=(15, 9.5))

for r_idx, sc in enumerate(scenarios):
    for c_idx, step in enumerate(steps):
        file_path = DATA_DIR / f"{sc['id']}_step_{step}.npz"
        ax = axes[r_idx, c_idx]
        
        if file_path.exists():
            grid = np.load(file_path)
            # Plot the 2D grid
            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
            
            # Count cell types
            n_cells = grid.size
            n_cancer = np.sum(grid == 1)
            n_dead = np.sum(grid == 2)
            cancer_frac = n_cancer / n_cells
            dead_frac = n_dead / n_cells
            
            # Print fractions on image
            ax.text(5, 195, f"C:{cancer_frac:.1%}\nD:{dead_frac:.1%}", 
                    color="white", fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"))
        else:
            # If terminated early and file is missing, show empty/grey
            ax.fill_between([0, 200], 0, 200, color="#CCCCCC")
            ax.text(100, 100, "Extinct", ha="center", va="center", fontsize=12)

        # Labels
        if r_idx == 0:
            ax.set_title(f"Step {step}", fontsize=12, fontweight="bold")
        if c_idx == 0:
            ax.set_ylabel(sc["title"], fontsize=12, fontweight="bold")
            
        ax.set_xticks([])
        ax.set_yticks([])

# Create a shared legend
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color=colors[0], label="Wild-Type"),
    mpatches.Patch(color=colors[1], label="Active Cancer"),
    mpatches.Patch(color=colors[2], label="Dead Cells (Lethal HK)")
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.01))

plt.suptitle("Spatial Evolution Comparison: Peak vs Decreasing Regimes", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])

fig_path_png = FIG_DIR / "spatial_comparison.png"
fig_path_svg = FIG_DIR / "spatial_comparison.svg"
plt.savefig(fig_path_png, dpi=200, bbox_inches="tight")
plt.savefig(fig_path_svg, bbox_inches="tight")
print(f"Saved spatial comparison figures to:\n  - {fig_path_png}\n  - {fig_path_svg}")
