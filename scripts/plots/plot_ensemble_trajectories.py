"""
plot_ensemble_trajectories.py
------------------------------
Plot the ensemble-averaged tumor-density trajectories and relative growth
speed (running slope / density) for all three ploidy conditions.

Reproduces Cells 4 and 17 of notebooks/analysis.ipynb, adapted to the
new NPZ trajectory format.

Outputs
-------
outputs/figures/ensemble_trajectories.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.analysis.loaders import load_all_ploidy, extract_field
from src.analysis.stats import plot_stats_elementwise, running_slope

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    "Diploid":   "#E63946",
    "Aneuploid": "#2A9D8F",
    "Polyploid": "#7B2D8B",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SKIP = 50   # half-window for running slope

# ---------------------------------------------------------------------------
# Load only tumor trajectories
# ---------------------------------------------------------------------------
print("Loading ensemble data (Tumor runs only) …")
all_data = load_all_ploidy(outcome_filter="Tumor")

# ---------------------------------------------------------------------------
# Build trajectory lists
# ---------------------------------------------------------------------------
density_trajs: dict[str, list] = {}
growth_trajs:  dict[str, list] = {}

for label in ["Diploid", "Aneuploid", "Polyploid"]:
    summary, trajs = all_data[label]
    td_list = extract_field(trajs, "tumor_density")
    density_trajs[label] = td_list

    # Relative growth speed: d(density)/dt / density
    gs_list = []
    for td in td_list:
        slope = running_slope(td, SKIP)
        denom = td[SKIP + 1 : len(td) - SKIP]
        # Avoid division by zero
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(denom > 1e-9, slope / denom, np.nan)
        gs_list.append(rel)
    growth_trajs[label] = gs_list

    print(f"  {label}: {len(td_list)} tumor runs")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Ensemble Tumor Dynamics", fontsize=16, fontweight="bold", y=1.02)

# ── Tumor density ──
ax = axes[0]
for label in ["Diploid", "Aneuploid", "Polyploid"]:
    plot_stats_elementwise(ax, density_trajs[label],
                           color=COLORS[label], lw=2, alpha=0.25, label=label)

ax.set_xlabel("Time step", fontsize=14)
ax.set_ylabel("Tumor cell density", fontsize=14)
ax.set_title("Tumor Density Trajectories", fontsize=14)
ax.legend(fontsize=12)
ax.yaxis.grid(True, ls="--", alpha=0.4)

# ── Relative growth speed ──
ax = axes[1]
for label in ["Diploid", "Aneuploid", "Polyploid"]:
    gs = growth_trajs[label]
    # Clip extreme values for legibility
    gs_clipped = [np.clip(g, 0, 0.02) for g in gs]
    plot_stats_elementwise(ax, gs_clipped,
                           color=COLORS[label], lw=2, alpha=0.25, label=label)

ax.set_xlabel("Time step", fontsize=14)
ax.set_ylabel("Relative growth speed  (d[density]/dt) / density", fontsize=13)
ax.set_title("Tumor Growth Speed", fontsize=14)
ax.set_ylim(bottom=0)
ax.legend(fontsize=12)
ax.yaxis.grid(True, ls="--", alpha=0.4)

plt.tight_layout()

out_dir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "ensemble_trajectories.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
plt.show()
