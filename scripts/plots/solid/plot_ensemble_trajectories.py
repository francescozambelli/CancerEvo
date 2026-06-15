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

from src.analysis.loaders import load_all_ploidy, extract_field
from src.analysis.stats import plot_stats_elementwise, running_slope

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

DENSITY_MAX = 0.4
N_BINS = 300

# ---------------------------------------------------------------------------
# Helper: return_stats  (replicates the notebook function exactly)
# ---------------------------------------------------------------------------

def return_stats(vec, interv):
    """
    For each density bin in *interv*, collect every timestep index across all
    trajectories in *vec* where the density falls in that bin, then return
    the mean and std of those indices.

    Parameters
    ----------
    vec   : list of 1-D arrays  – tumor_density per trajectory
    interv: 1-D array            – density bin edges

    Returns
    -------
    means : (len(interv)-1,) array  – mean timestep per bin
    stds  : (len(interv)-1,) array  – std  timestep per bin
    """
    means, stds = [], []
    for i in range(len(interv) - 1):
        provv = []
        for traj in vec:
            idx = np.where(
                np.logical_and(traj > interv[i], traj <= interv[i + 1])
            )[0]
            provv.append(idx)
        provv = np.concatenate(provv)
        if len(provv):
            means.append(np.mean(provv))
            stds.append(np.std(provv))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)

# ---------------------------------------------------------------------------
# Load only tumor trajectories
# ---------------------------------------------------------------------------
print("Loading ensemble data (Tumor runs only) …")
all_data = load_all_ploidy(outcome_filter="Tumor")

density_trajs: dict[str, list] = {}
d_lenght = {"Diploid":1200, "Aneuploid":1000, "Polyploid":1500}

for label in ["Diploid", "Aneuploid", "Polyploid"]:
    summary, trajs = all_data[label]
    td_list = extract_field(trajs, "tumor_density")
    density_trajs[label] = td_list
    print(f"  {label}: {len(td_list)} tumor runs")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

# Create inset axes in the lower right
# Position: [x0, y0, width, height] in axes coordinates
ax_inset = ax.inset_axes([0.6, 0.1, 0.48, 0.29])

interv = np.linspace(0, DENSITY_MAX, N_BINS)

SKIP_SMOOTH = 10    # running-slope half-window
WIN_AVG     = 50   # running average smoothing window

def get_smooth_slope(td, skip, win_avg):
    slope = running_slope(td, skip)
    if len(slope) >= win_avg:
        kernel = np.ones(win_avg) / win_avg
        slope_smooth = np.convolve(slope, kernel, mode="same")
        half = win_avg // 2
        slope_smooth[:half] = np.nan
        slope_smooth[-half:] = np.nan
    else:
        slope_smooth = np.full_like(slope, np.nan)
    return slope_smooth

for label in ["Diploid", "Polyploid", "Aneuploid"]:
    # 1. Plot density trajectories on main axis
    means, stds = return_stats(density_trajs[label], interv)
    # Prepend origin so the curve starts at (0, 0)
    means = np.concatenate([[0], means])
    stds  = np.concatenate([[0], stds])
    if label=="Polyploid":
        ax.plot(means[:-100], interv[:-100], color=COLORS[label], lw=3, label=label)
        ax.fill_betweenx(interv[:-100], means[:-100] - stds[:-100], means[:-100] + stds[:-100],
                        color=COLORS[label], alpha=0.3)
    else:
        ax.plot(means, interv, color=COLORS[label], lw=3, label=label)
        ax.fill_betweenx(interv, means - stds, means + stds,
                        color=COLORS[label], alpha=0.3)
    
    # 2. Plot growth speed on inset axis (linear scale)
    gs_list = [get_smooth_slope(td[:d_lenght[label]], SKIP_SMOOTH, WIN_AVG) for td in density_trajs[label]]
    
    plot_stats_elementwise(ax_inset, gs_list, color=COLORS[label], lw=1.5, alpha=0.20)
    
# Format main axis
ax.set_xlim(left=0, right=3500)
ax.set_ylim(bottom=0, top=DENSITY_MAX)
ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Tumor cell density", fontsize=14)
#ax.set_title("Tumor Density Trajectories", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
ax.grid(False)

# Format inset axis (linear)
ax_inset.set_xlim(left=0, right=1500)
ax_inset.set_ylim(bottom=0)
ax_inset.set_xlabel("Time", fontsize=8)
ax_inset.set_ylabel(r"Growth speed $d\rho/dt$", fontsize=8)
ax_inset.tick_params(axis='both', which='both', labelsize=8)
ax_inset.grid(False)
ax_inset.set_axisbelow(True)
#put y ticks to scientific notation
ax_inset.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax_inset.yaxis.get_offset_text().set_fontsize(8)

plt.tight_layout()

out_dir = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)
out_path_png = out_dir / "ensemble_trajectories.png"
out_path_svg = out_dir / "ensemble_trajectories.svg"
plt.savefig(out_path_png, dpi=150, bbox_inches="tight")
plt.savefig(out_path_svg, bbox_inches="tight")
print(f"\nSaved →\n  - {out_path_png}\n  - {out_path_svg}")



