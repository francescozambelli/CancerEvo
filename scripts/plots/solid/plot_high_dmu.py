# scripts/plots/plot_high_dmu.py
#
# Plot spatial comparison of Fast Increase vs Fast Decrease scenarios at high dmu = 0.0030.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "spatial_study"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "solid"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Custom colormap
# 0 = Wild-type (Pale Green)
# 1 = Cancer (Vibrant Red)
# 2 = Dead (Dark Charcoal)
colors = ["#E9F5EC", "#E63946", "#3A3A3A"]
cmap = ListedColormap(colors)

# Setup figure (2 rows, 5 columns)
fig, axes = plt.subplots(2, 5, figsize=(15, 7.5))

# Row 1: Fast Increase
inc_steps = [0, 20, 50, 100, 150]
for idx, step in enumerate(inc_steps):
    ax = axes[0, idx]
    grid = np.load(DATA_DIR / f"fast_inc_step_{step}.npz")
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    
    n_cells = grid.size
    n_cancer = np.sum(grid == 1)
    n_dead = np.sum(grid == 2)
    ax.text(5, 195, f"C:{n_cancer/n_cells:.1%}\nD:{n_dead/n_cells:.1%}", 
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"))
    
    ax.set_title(f"Step {step}", fontsize=11, fontweight="bold")
    if idx == 0:
        ax.set_ylabel("Fast Increase (Tumor_Max)\nrmax = 0.3153", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

# Row 2: Fast Decrease
dec_steps = [0, 1, 2, 3, 5]
for idx, step in enumerate(dec_steps):
    ax = axes[1, idx]
    grid = np.load(DATA_DIR / f"fast_dec_step_{step}.npz")
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    
    n_cells = grid.size
    n_cancer = np.sum(grid == 1)
    n_dead = np.sum(grid == 2)
    ax.text(5, 195, f"C:{n_cancer/n_cells:.1%}\nD:{n_dead/n_cells:.1%}", 
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"))
    
    ax.set_title(f"Step {step}", fontsize=11, fontweight="bold")
    if idx == 0:
        ax.set_ylabel("Fast Decrease (Tumor_Min)\nrmax = 0.8480", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

# Shared Legend
legend_patches = [
    mpatches.Patch(color=colors[0], label="Wild-Type"),
    mpatches.Patch(color=colors[1], label="Active Cancer"),
    mpatches.Patch(color=colors[2], label="Dead Cells (Lethal HK)")
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.01))

plt.suptitle("Spatial Transitions at High Mutation Rate (dmu = 0.0030)", fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

fig_path_png = FIG_DIR / "high_dmu_transition.png"
fig_path_svg = FIG_DIR / "high_dmu_transition.svg"
plt.savefig(fig_path_png, dpi=200, bbox_inches="tight")
plt.savefig(fig_path_svg, bbox_inches="tight")
print(f"Saved high dmu comparison figures to:\n  - {fig_path_png}\n  - {fig_path_svg}")
