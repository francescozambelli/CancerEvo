"""
plot_diploid_mu_progression.py
------------------------------
Plot the progression in time of the mutation rate (mu) variable for the
ensemble of trajectories in the diploid (2CHR) case.

This script loads the diploid (2CHR) simulations, divides them into two groups
based on their final outcome (Health vs Others), and plots both individual
trajectories (as translucent lines) and their ensemble statistics (median and
16th-84th percentile bands of active runs).

Outputs
-------
outputs/figures/diploid_mu_progression.png
outputs/figures/diploid_mu_progression.svg
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
from src.analysis.loaders import load_ensemble

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    "Health": "#2A9D8F",      # teal-green for recovery
    "Others": "#E63946",      # vivid red for tumor progression
    "Health_dark": "#1C6B61", # darker teal for median line
    "Others_dark": "#B51A2B", # darker red for median line
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ---------------------------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------------------------
print("Loading ensemble data for Diploid (2CHR) …")
# The diploid case is stored in 'ensemble_results_2CHR'
summary, trajs = load_ensemble("ensemble_results_D")

health_trajs = []
other_trajs = []

# Separate trajectories based on final tumor density (0.0 -> Health, >0.0 -> Others)
# We use the actual simulation trajectories to avoid any potential CSV alignment discrepancies.
for t in trajs:
    # Keep only the active part of the trajectory (where tumor_density > 0)
    td = t["tumor_density"]
    mu = t["mu"]
    active_idx = np.where(td > 0.0)[0]
    
    if len(active_idx) > 0:
        mu_active = mu[active_idx]
        time_active = active_idx
        # Check if the final step of the active trajectory is at 0.0 density
        # (meaning it ended in extinction / Health)
        is_health = (td[-1] == 0.0)
        
        traj_data = {
            "time": time_active,
            "mu": mu_active
        }
        if is_health:
            health_trajs.append(traj_data)
        else:
            other_trajs.append(traj_data)

print(f"Loaded {len(health_trajs)} Health trajectories and {len(other_trajs)} Tumor/Other trajectories.")

# ---------------------------------------------------------------------------
# Helper: compute active statistics at each time step
# ---------------------------------------------------------------------------
def compute_stats(traj_list, min_active_count=5):
    """
    Compute median, mean, and 16th/84th percentiles of mu at each time step.
    Truncates the curves when the number of active trajectories drops below min_active_count.
    """
    if not traj_list:
        return None
        
    max_time = max(t["time"][-1] for t in traj_list) + 1
    
    times = []
    medians = []
    means = []
    p16 = []
    p84 = []
    
    for step in range(max_time):
        values = [t["mu"][step] for t in traj_list if step <= t["time"][-1]]
        if len(values) < min_active_count:
            break
        times.append(step)
        medians.append(np.median(values))
        means.append(np.mean(values))
        p16.append(np.percentile(values, 16))
        p84.append(np.percentile(values, 84))
        
    return {
        "time": np.array(times),
        "median": np.array(medians),
        "mean": np.array(means),
        "p16": np.array(p16),
        "p84": np.array(p84),
    }

health_stats = compute_stats(health_trajs, min_active_count=5)
other_stats = compute_stats(other_trajs, min_active_count=5)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6))

xlim = (0, 700)
alpha_health = 0.12
alpha_others = 0.15

ax.set_xlim(xlim)
ax.set_xlabel("Time (Steps)", fontsize=13)
ax.set_ylabel("Mutation Rate ($\\mu$)", fontsize=13)
ax.grid(True, ls=":", alpha=0.4)

# 1. Plot individual trajectories
# Health trajectories
for t in health_trajs:
    ax.plot(t["time"], t["mu"], color=COLORS["Health"], alpha=alpha_health, lw=1.0, zorder=1)
    
# Other trajectories
for t in other_trajs:
    ax.plot(t["time"], t["mu"], color=COLORS["Others"], alpha=alpha_others, lw=1.0, zorder=2)
    
# 2. Plot ensemble statistics (Median and 16th-84th percentile range)
# Health statistics
if health_stats is not None:
    ax.plot(health_stats["time"], health_stats["median"], 
            color=COLORS["Health_dark"], lw=2.5, zorder=3,
            label=f"Health Median (N={len(health_trajs)})")
    ax.fill_between(health_stats["time"], health_stats["p16"], health_stats["p84"],
                    color=COLORS["Health"], alpha=0.35, zorder=3)
                    
# Other statistics
if other_stats is not None:
    ax.plot(other_stats["time"], other_stats["median"], 
            color=COLORS["Others_dark"], lw=2.5, zorder=4,
            label=f"Tumor Median (N={len(other_trajs)})")
    ax.fill_between(other_stats["time"], other_stats["p16"], other_stats["p84"],
                    color=COLORS["Others"], alpha=0.25, zorder=4)

# Add legend
ax.legend(loc="upper left", fontsize=11, frameon=True, facecolor="white", edgecolor="none", framealpha=0.8)

plt.title("Mutation Rate ($\\mu$) Progression in Diploid (2CHR) Case (0-800 steps)", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()

# Save figures
out_dir = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)

out_path_png = out_dir / "diploid_mu_progression.png"
out_path_svg = out_dir / "diploid_mu_progression.svg"

plt.savefig(out_path_png, dpi=180, bbox_inches="tight")
plt.savefig(out_path_svg, bbox_inches="tight")

print(f"\nSaved figures to:\n  - {out_path_png}\n  - {out_path_svg}")
plt.show()
