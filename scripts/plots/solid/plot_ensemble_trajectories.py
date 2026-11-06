"""
plot_ensemble_trajectories.py
------------------------------
Plot the ensemble-averaged tumor-density trajectories for all three ploidy conditions,
with an inset plot showing absolute growth speed in linear scale.

Outputs
-------
outputs/figures/solid/ensemble_trajectories.png
outputs/figures/solid/ensemble_trajectories.svg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from src.analysis.loaders import load_all_ploidy, extract_field
from src.analysis.stats import plot_stats_elementwise

# ---------------------------------------------------------------------------
# Style  (matches notebook colour palette)
# ---------------------------------------------------------------------------
COLORS = {
    "Diploid":   "orangered",
    "Aneuploid": "forestgreen",
    "Polyploid": "purple",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Load only tumor trajectories
# ---------------------------------------------------------------------------
print("Loading ensemble data (Tumor runs only) …")
all_data = load_all_ploidy(outcome_filter="Tumor")

density_trajs: dict[str, list] = {}
for label in ["Diploid", "Aneuploid", "Polyploid"]:
    summary, trajs = all_data[label]
    td_list = extract_field(trajs, "tumor_density")
    density_trajs[label] = td_list
    print(f"  {label}: {len(td_list)} tumor runs")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(9, 12))

SAVGOL_WIN = 121    # Savitzky-Golay filtering window length
SAVGOL_POLY = 2     # Polynomial order

def get_smooth_slope(td, win=SAVGOL_WIN, poly=SAVGOL_POLY):
    """
    Computes a clean growth velocity (drho/dt) trajectory via Savitzky-Golay
    filtering on individual tumor density time series followed by gradient evaluation.
    Preserves true peak curvature while removing discrete stochastic lattice noise.
    """
    if len(td) >= win:
        td_smooth = savgol_filter(td, window_length=win, polyorder=poly)
        slope = np.gradient(td_smooth)
        half = win // 4
        slope[:half] = np.nan
        slope[-half:] = np.nan
        return slope
    else:
        return np.full_like(td, np.nan)

for label in ["Diploid", "Polyploid", "Aneuploid"]:
    # 1. Plot raw empirical density trajectories on upper axis (median ± 1σ band across runs)
    plot_stats_elementwise(ax[0], density_trajs[label], color=COLORS[label], lw=3, label=label)
    
    # 2. Plot regularized growth speed on lower axis via Savitzky-Golay filtering
    gs_list = [get_smooth_slope(td) for td in density_trajs[label]]
    plot_stats_elementwise(ax[1], gs_list, color=COLORS[label], lw=3)

# Format main axis
fontsize_label = 20
fontsize_ticks = 18
fontsize_legend = 18

ax[0].set_xlim(left=0, right=2400)
ax[0].set_ylim(bottom=0, top=0.50)
ax[0].set_xlabel("Time", fontsize=fontsize_label)
ax[0].set_ylabel("Tumor cell density", fontsize=fontsize_label)
ax[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax[0].legend(fontsize=fontsize_legend, loc="upper left")

# Format lower axis (linear, matching extended time axis with ax[0])
ax[1].set_xlim(left=0, right=2400)
ax[1].set_ylim(bottom=0)
ax[1].set_xlabel("Time", fontsize=fontsize_label)
ax[1].set_ylabel(r"Growth speed $d\rho/dt$", fontsize=fontsize_label)
ax[1].tick_params(axis='both', which='both', labelsize=fontsize_ticks)
ax[1].grid(False)
ax[1].set_axisbelow(True)
# Put y ticks to scientific notation
ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax[1].yaxis.get_offset_text().set_fontsize(18)

plt.tight_layout()

out_dir = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)
out_path_png = out_dir / "ensemble_trajectories.png"
out_path_wt05_png = out_dir / "ensemble_trajectories_wt05.png"
out_path_svg = out_dir / "ensemble_trajectories.svg"
plt.savefig(out_path_png, dpi=150, bbox_inches="tight")
plt.savefig(out_path_wt05_png, dpi=150)
plt.savefig(out_path_svg, bbox_inches="tight")
print(f"\nSaved →\n  - {out_path_png}\n  - {out_path_wt05_png}\n  - {out_path_svg}")



