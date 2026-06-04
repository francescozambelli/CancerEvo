"""
plot_ensemble_trajectories.py
------------------------------
Plot the ensemble-averaged tumor-density trajectories and relative growth
speed (running slope / density) for all three ploidy conditions.

Reproduces Cells 17 & 19 of notebooks/analysis.ipynb, adapted to the new
NPZ trajectory format.

Left panel  (ax[0]) — ``return_stats`` style:
    For each density level bin, pool timestep indices across all tumor
    trajectories where density falls in that bin → mean ± std of time step.
    Axes: X = mean time step, Y = tumor cell density.

Right panel (ax[1]) — smooth growth speed from the mean curve:
    Reconstruct a single smooth mean density-vs-time series from the
    return_stats means (inverting the density→time mapping), then apply
    running_slope / mean_density.  This matches notebook cell 19 ax[1] and
    avoids the early-time noise that comes from averaging per-trajectory
    slope/density ratios.

Outputs
-------
outputs/figures/ensemble_trajectories.png
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

SKIP = 50        # half-window for running slope
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
# Each entry: a single 1-D array (the smooth mean-density curve vs time)
smooth_mean_density: dict[str, np.ndarray] = {}

for label in ["Diploid", "Aneuploid", "Polyploid"]:
    summary, trajs = all_data[label]
    td_list = extract_field(trajs, "tumor_density")
    density_trajs[label] = td_list
    print(f"  {label}: {len(td_list)} tumor runs")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reconstruct_mean_curve(means: np.ndarray, interv: np.ndarray) -> np.ndarray:
    """
    Invert the return_stats density→time mapping into a time→density curve.

    ``means`` gives the mean time step at which each density bin in ``interv``
    is reached.  We build a time-indexed array where ``curve[t] = density``
    by linear interpolation between the (means, interv) pairs.

    Returns a 1-D array of length ``int(max_valid_mean) + 1``.
    """
    # Drop NaN
    valid = ~np.isnan(means)
    m = means[valid]
    d = interv[valid]          # interv has same length as means after prepend
    if len(m) < 2:
        return np.array([])
    t_max = int(m[-1]) + 1
    t_axis = np.arange(t_max)
    # Interpolate density as a function of time
    return np.interp(t_axis, m, d)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
#fig.suptitle("Ensemble Tumor Dynamics", fontsize=16, fontweight="bold", y=1.02)

interv = np.linspace(0, DENSITY_MAX, N_BINS)

# Store the mean curves so we can reuse them for the right panel
mean_curves: dict[str, np.ndarray] = {}

# ── Left: mean arrival-time per density level ──────────────────────────────
ax = axes[0]
for label in ["Diploid", "Polyploid", "Aneuploid"]:
    means, stds = return_stats(density_trajs[label], interv)
    # Prepend origin so the curve starts at (0, 0)
    means = np.concatenate([[0], means])
    stds  = np.concatenate([[0], stds])
    mean_curves[label] = (means, stds)   # save for right panel

    ax.plot(means, interv, color=COLORS[label], lw=3, label=label)
    ax.fill_betweenx(interv, means - stds, means + stds,
                     color=COLORS[label], alpha=0.3)

ax.set_xlabel("Time", fontsize=15)
ax.set_ylabel("Tumor cell density", fontsize=15)
ax.set_title("Tumor Density Trajectories", fontsize=14)
ax.legend(fontsize=13)
ax.grid(False)

# ── Right: smooth absolute growth speed with confidence bars ────────────────
# Plots d(density)/dt (absolute rate) with confidence bands.
# - We compute the running slope for each individual trajectory to preserve
#   variation, pad it to align with the original time steps, and smooth it
#   using a running average.
# - We then use plot_stats_elementwise to compute and plot the median and the
#   16th-84th percentile confidence bands across runs.
ax = axes[1]
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
    gs_list = [get_smooth_slope(td, SKIP_SMOOTH, WIN_AVG) for td in density_trajs[label]]
    plot_stats_elementwise(ax, gs_list, color=COLORS[label], lw=2, alpha=0.25, label=label)

ax.set_ylim(bottom=0)
ax.set_xlabel("Time", fontsize=15)
ax.set_ylabel(r"Tumor growth speed  $d\rho/dt$", fontsize=14)
ax.set_title("Absolute Growth Speed", fontsize=14)
ax.legend(fontsize=13)
ax.grid(False)

plt.tight_layout()

out_dir = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)
out_path_png = out_dir / "ensemble_trajectories.png"
out_path_svg = out_dir / "ensemble_trajectories.svg"
plt.savefig(out_path_png, dpi=150, bbox_inches="tight")
plt.savefig(out_path_svg, bbox_inches="tight")
print(f"\nSaved →\n  - {out_path_png}\n  - {out_path_svg}")
#plt.show()
