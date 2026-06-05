"""
analyze_genes.py
----------------
Analyze the temporal evolution of mutation and activation of different gene types
(HK, I, O, S, M) in solid vs. liquid simulations for the diploid (2CHR) case.
Separates simulations into "Tumor" (Tumor_Max) and "Health" (Health) outcomes.
Saves comparison plots and text summary of stationary states.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Output directory for this task
OUT_DIR = REPO_ROOT / "gene_evolution_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Data directories
SOLID_DIR = REPO_ROOT / "data" / "simulations" / "ensemble_results_D"
LIQUID_DIR = REPO_ROOT / "data" / "simulations_liquid" / "ensemble_results_D"

GENE_TYPES = ["I", "O", "S", "M", "HK"]
GENE_LABELS = {
    "I": "Mutator (I)",
    "O": "Oncogene (O)",
    "S": "Suppressor (S)",
    "M": "Missegregation (M)",
    "HK": "Housekeeping (HK)"
}
MAX_GENES = {
    "I": 10,
    "O": 10,
    "S": 10,
    "M": 5,
    "HK": 10
}

# Style configurations
COLORS = {
    "Solid_Tumor": "#1F77B4",       # rich slate blue
    "Solid_Health": "#A9CDE2",      # light slate blue
    "Liquid_Tumor": "#FF7F0E",      # warm orange
    "Liquid_Health": "#FFD1A9",     # light orange
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

# ---------------------------------------------------------------------------
# Data Loading & Processing
# ---------------------------------------------------------------------------

def load_trajectories_by_outcome(data_dir):
    """Load trajectories separated by outcome (Tumor_Max and Health)."""
    csv_path = data_dir / "ensemble_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    tumor_ids = df[df["outcome"] == "Tumor_Max"]["sim_id"].values
    health_ids = df[df["outcome"] == "Health"]["sim_id"].values
    
    tumor_trajs = []
    health_trajs = []
    
    for sid in tumor_ids:
        sim_path = data_dir / f"sim_{sid}.npz"
        if not sim_path.exists():
            continue
        with np.load(sim_path) as data:
            traj_data = {k: data[k] for k in data.files}
            tumor_trajs.append(traj_data)
            
    for sid in health_ids:
        sim_path = data_dir / f"sim_{sid}.npz"
        if not sim_path.exists():
            continue
        with np.load(sim_path) as data:
            traj_data = {k: data[k] for k in data.files}
            # Truncate health runs to remove the extinction step (last element is 0 density/mutations)
            steps = len(traj_data["tumor_density"])
            if steps > 1:
                for k in traj_data.keys():
                    if k != "outcome_code" and len(traj_data[k]) == steps:
                        traj_data[k] = traj_data[k][:-1]
            health_trajs.append(traj_data)
            
    return tumor_trajs, health_trajs

def compute_real_time_stats(trajs, min_active=5):
    """Compute mean and std over real time steps (aligned at t=0)."""
    if not trajs:
        return {}
        
    max_steps = max(len(t["tumor_density"]) for t in trajs)
    
    stats = {
        "time": [],
        "active_count": []
    }
    for var in ["mut_I", "mut_O", "mut_S", "mut_M", "mut_HK",
                "act_I", "act_O", "act_S", "act_M", "act_HK"]:
        stats[f"{var}_mean"] = []
        stats[f"{var}_std"] = []
        
    for step in range(max_steps):
        # Find active trajectories at this step
        active_trajs = [t for t in trajs if step < len(t["tumor_density"])]
        if len(active_trajs) < min_active:
            break
            
        stats["time"].append(step)
        stats["active_count"].append(len(active_trajs))
        
        for var in ["mut_I", "mut_O", "mut_S", "mut_M", "mut_HK",
                    "act_I", "act_O", "act_S", "act_M", "act_HK"]:
            vals = [t[var][step] for t in active_trajs]
            stats[f"{var}_mean"].append(np.mean(vals))
            stats[f"{var}_std"].append(np.std(vals))
            
    # Convert lists to numpy arrays
    for k in stats.keys():
        stats[k] = np.array(stats[k])
        
    return stats

def compute_normalized_time_stats(trajs, num_points=100):
    """Interpolate each trajectory to a fixed length and average them."""
    if not trajs:
        return {}
        
    grid = np.linspace(0, 1, num_points)
    
    interpolated_data = {}
    for var in ["mut_I", "mut_O", "mut_S", "mut_M", "mut_HK",
                "act_I", "act_O", "act_S", "act_M", "act_HK"]:
        interpolated_data[var] = []
        
    for t in trajs:
        steps = len(t["tumor_density"])
        # Avoid interpolation issues with length-1 trajectories (if any)
        if steps < 2:
            continue
        x_orig = np.linspace(0, 1, steps)
        for var in interpolated_data.keys():
            y_orig = t[var]
            y_interp = np.interp(grid, x_orig, y_orig)
            interpolated_data[var].append(y_interp)
            
    stats = {"time_norm": grid * 100.0} # as percentage
    for var in interpolated_data.keys():
        arr = np.array(interpolated_data[var])
        stats[f"{var}_mean"] = np.mean(arr, axis=0)
        stats[f"{var}_std"] = np.std(arr, axis=0)
        
    return stats

def extract_stationary_states(trajs):
    """Extract mutation and activation levels at the final step."""
    if not trajs:
        return {}
    states = {}
    for var in ["mut_I", "mut_O", "mut_S", "mut_M", "mut_HK",
                "act_I", "act_O", "act_S", "act_M", "act_HK"]:
        states[var] = np.array([t[var][-1] for t in trajs])
    return states

# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------

print("Loading trajectories...")
solid_tumor, solid_health = load_trajectories_by_outcome(SOLID_DIR)
liquid_tumor, liquid_health = load_trajectories_by_outcome(LIQUID_DIR)

print(f"Solid: {len(solid_tumor)} Tumor, {len(solid_health)} Health.")
print(f"Liquid: {len(liquid_tumor)} Tumor, {len(liquid_health)} Health.")

# Compute statistics
solid_t_real = compute_real_time_stats(solid_tumor)
solid_h_real = compute_real_time_stats(solid_health)
liquid_t_real = compute_real_time_stats(liquid_tumor)
liquid_h_real = compute_real_time_stats(liquid_health)

solid_t_norm = compute_normalized_time_stats(solid_tumor)
solid_h_norm = compute_normalized_time_stats(solid_health)
liquid_t_norm = compute_normalized_time_stats(liquid_tumor)
liquid_h_norm = compute_normalized_time_stats(liquid_health)

solid_t_final = extract_stationary_states(solid_tumor)
solid_h_final = extract_stationary_states(solid_health)
liquid_t_final = extract_stationary_states(liquid_tumor)
liquid_h_final = extract_stationary_states(liquid_health)

# ---------------------------------------------------------------------------
# Plot 1: Real-Time Evolution
# ---------------------------------------------------------------------------
print("Plotting Figure 1: Real-time evolution...")
fig, axes = plt.subplots(2, 5, figsize=(16, 8.5), sharex=False)

# Adjust margins and spacing
plt.subplots_adjust(hspace=0.35, wspace=0.32)

for col, g in enumerate(GENE_TYPES):
    # Row 0: Mutation Fraction
    ax_mut = axes[0, col]
    ax_mut.set_title(GENE_LABELS[g], fontweight="bold", fontsize=12)
    
    # Solid Tumor (Solid Blue)
    if solid_t_real:
        ax_mut.plot(solid_t_real["time"], solid_t_real[f"mut_{g}_mean"], 
                    color=COLORS["Solid_Tumor"], lw=2.2, label="Solid - Tumor")
        ax_mut.fill_between(solid_t_real["time"], 
                            solid_t_real[f"mut_{g}_mean"] - solid_t_real[f"mut_{g}_std"],
                            solid_t_real[f"mut_{g}_mean"] + solid_t_real[f"mut_{g}_std"],
                            color=COLORS["Solid_Tumor"], alpha=0.15)
    
    # Solid Health (Dashed Blue)
    if solid_h_real:
        ax_mut.plot(solid_h_real["time"], solid_h_real[f"mut_{g}_mean"], 
                    color=COLORS["Solid_Tumor"], lw=1.8, ls="--", label="Solid - Health")
        
    # Liquid Tumor (Solid Orange)
    if liquid_t_real:
        ax_mut.plot(liquid_t_real["time"], liquid_t_real[f"mut_{g}_mean"], 
                    color=COLORS["Liquid_Tumor"], lw=2.2, label="Liquid - Tumor")
        ax_mut.fill_between(liquid_real_time := liquid_t_real["time"], 
                            liquid_t_real[f"mut_{g}_mean"] - liquid_t_real[f"mut_{g}_std"],
                            liquid_t_real[f"mut_{g}_mean"] + liquid_t_real[f"mut_{g}_std"],
                            color=COLORS["Liquid_Tumor"], alpha=0.15)
        
    # Liquid Health (Dashed Orange)
    if liquid_h_real:
        ax_mut.plot(liquid_h_real["time"], liquid_h_real[f"mut_{g}_mean"], 
                    color=COLORS["Liquid_Tumor"], lw=1.8, ls="--", label="Liquid - Health")
    
    ax_mut.set_ylim(-0.05, 1.05)
    ax_mut.grid(True, ls=":", color="#e0e0e0", alpha=0.5, zorder=0)
    ax_mut.set_axisbelow(True)
    if col == 0:
        ax_mut.set_ylabel("Mutation Fraction", fontsize=12, fontweight="bold")
    if col == 4:
        ax_mut.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=9)

    # Row 1: Activation Level
    ax_act = axes[1, col]
    # Solid Tumor
    if solid_t_real:
        ax_act.plot(solid_t_real["time"], solid_t_real[f"act_{g}_mean"], 
                    color=COLORS["Solid_Tumor"], lw=2.2)
        ax_act.fill_between(solid_t_real["time"], 
                            solid_t_real[f"act_{g}_mean"] - solid_t_real[f"act_{g}_std"],
                            solid_t_real[f"act_{g}_mean"] + solid_t_real[f"act_{g}_std"],
                            color=COLORS["Solid_Tumor"], alpha=0.15)
    # Solid Health
    if solid_h_real:
        ax_act.plot(solid_h_real["time"], solid_h_real[f"act_{g}_mean"], 
                    color=COLORS["Solid_Tumor"], lw=1.8, ls="--")
        
    # Liquid Tumor
    if liquid_t_real:
        ax_act.plot(liquid_t_real["time"], liquid_t_real[f"act_{g}_mean"], 
                    color=COLORS["Liquid_Tumor"], lw=2.2)
        ax_act.fill_between(liquid_t_real["time"], 
                            liquid_t_real[f"act_{g}_mean"] - liquid_t_real[f"act_{g}_std"],
                            liquid_t_real[f"act_{g}_mean"] + liquid_t_real[f"act_{g}_std"],
                            color=COLORS["Liquid_Tumor"], alpha=0.15)
    # Liquid Health
    if liquid_h_real:
        ax_act.plot(liquid_h_real["time"], liquid_h_real[f"act_{g}_mean"], 
                    color=COLORS["Liquid_Tumor"], lw=1.8, ls="--")
    
    ax_act.set_ylim(-0.05 * MAX_GENES[g], 1.05 * MAX_GENES[g])
    ax_act.grid(True, ls=":", color="#e0e0e0", alpha=0.5, zorder=0)
    ax_act.set_axisbelow(True)
    ax_act.set_xlabel("Time (Steps)", fontsize=11)
    if col == 0:
        ax_act.set_ylabel("Activation Level", fontsize=12, fontweight="bold")

# Add panel labels
for i, label in enumerate(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]):
    r = i // 5
    c = i % 5
    axes[r, c].text(-0.15, 1.12, label, transform=axes[r, c].transAxes, 
                    fontsize=14, fontweight="bold", va="top", ha="right")

plt.savefig(OUT_DIR / "gene_evolution_real_time.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "gene_evolution_real_time.svg", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: Normalized-Time Evolution
# ---------------------------------------------------------------------------
print("Plotting Figure 2: Normalized-time evolution...")
fig, axes = plt.subplots(2, 5, figsize=(16, 8.5), sharex=True)
plt.subplots_adjust(hspace=0.25, wspace=0.32)

for col, g in enumerate(GENE_TYPES):
    # Row 0: Mutation Fraction
    ax_mut = axes[0, col]
    ax_mut.set_title(GENE_LABELS[g], fontweight="bold", fontsize=12)
    
    # Solid Tumor
    if solid_t_norm:
        ax_mut.plot(solid_t_norm["time_norm"], solid_t_norm[f"mut_{g}_mean"], 
                    color=COLORS["Solid_Tumor"], lw=2.2, label="Solid - Tumor")
        ax_mut.fill_between(solid_t_norm["time_norm"], 
                            solid_t_norm[f"mut_{g}_mean"] - solid_t_norm[f"mut_{g}_std"],
                            solid_t_norm[f"mut_{g}_mean"] + solid_t_norm[f"mut_{g}_std"],
                            color=COLORS["Solid_Tumor"], alpha=0.15)
    # Solid Health
    if solid_h_norm:
        ax_mut.plot(solid_h_norm["time_norm"], solid_h_norm[f"mut_{g}_mean"], 
                    color=COLORS["Solid_Tumor"], lw=1.8, ls="--", label="Solid - Health")
        
    # Liquid Tumor
    if liquid_t_norm:
        ax_mut.plot(liquid_t_norm["time_norm"], liquid_t_norm[f"mut_{g}_mean"], 
                    color=COLORS["Liquid_Tumor"], lw=2.2, label="Liquid - Tumor")
        ax_mut.fill_between(liquid_t_norm["time_norm"], 
                            liquid_t_norm[f"mut_{g}_mean"] - liquid_t_norm[f"mut_{g}_std"],
                            liquid_t_norm[f"mut_{g}_mean"] + liquid_t_norm[f"mut_{g}_std"],
                            color=COLORS["Liquid_Tumor"], alpha=0.15)
    # Liquid Health
    if liquid_h_norm:
        ax_mut.plot(liquid_h_norm["time_norm"], liquid_h_norm[f"mut_{g}_mean"], 
                    color=COLORS["Liquid_Tumor"], lw=1.8, ls="--", label="Liquid - Health")
    
    ax_mut.set_ylim(-0.05, 1.05)
    ax_mut.grid(True, ls=":", color="#e0e0e0", alpha=0.5, zorder=0)
    ax_mut.set_axisbelow(True)
    if col == 0:
        ax_mut.set_ylabel("Mutation Fraction", fontsize=12, fontweight="bold")
    if col == 4:
        ax_mut.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=9)

    # Row 1: Activation Level
    ax_act = axes[1, col]
    # Solid Tumor
    if solid_t_norm:
        ax_act.plot(solid_t_norm["time_norm"], solid_t_norm[f"act_{g}_mean"], 
                    color=COLORS["Solid_Tumor"], lw=2.2)
        ax_act.fill_between(solid_t_norm["time_norm"], 
                            solid_t_norm[f"act_{g}_mean"] - solid_t_norm[f"act_{g}_std"],
                            solid_t_norm[f"act_{g}_mean"] + solid_t_norm[f"act_{g}_std"],
                            color=COLORS["Solid_Tumor"], alpha=0.15)
    # Solid Health
    if solid_h_norm:
        ax_act.plot(solid_h_norm["time_norm"], solid_h_norm[f"act_{g}_mean"], 
                    color=COLORS["Solid_Tumor"], lw=1.8, ls="--")
        
    # Liquid Tumor
    if liquid_t_norm:
        ax_act.plot(liquid_t_norm["time_norm"], liquid_t_norm[f"act_{g}_mean"], 
                    color=COLORS["Liquid_Tumor"], lw=2.2)
        ax_act.fill_between(liquid_t_norm["time_norm"], 
                            liquid_t_norm[f"act_{g}_mean"] - liquid_t_norm[f"act_{g}_std"],
                            liquid_t_norm[f"act_{g}_mean"] + liquid_t_norm[f"act_{g}_std"],
                            color=COLORS["Liquid_Tumor"], alpha=0.15)
    # Liquid Health
    if liquid_h_norm:
        ax_act.plot(liquid_h_norm["time_norm"], liquid_h_norm[f"act_{g}_mean"], 
                    color=COLORS["Liquid_Tumor"], lw=1.8, ls="--")
    
    ax_act.set_ylim(-0.05 * MAX_GENES[g], 1.05 * MAX_GENES[g])
    ax_act.grid(True, ls=":", color="#e0e0e0", alpha=0.5, zorder=0)
    ax_act.set_axisbelow(True)
    ax_act.set_xlabel("Tumor Progression (%)", fontsize=11)
    if col == 0:
        ax_act.set_ylabel("Activation Level", fontsize=12, fontweight="bold")

# Add panel labels
for i, label in enumerate(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]):
    r = i // 5
    c = i % 5
    axes[r, c].text(-0.15, 1.12, label, transform=axes[r, c].transAxes, 
                    fontsize=14, fontweight="bold", va="top", ha="right")

plt.savefig(OUT_DIR / "gene_evolution_normalized_time.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "gene_evolution_normalized_time.svg", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Plot 3: Stationary States Comparison
# ---------------------------------------------------------------------------
print("Plotting Figure 3: Stationary states comparison...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plt.subplots_adjust(wspace=0.35)

x = np.arange(len(GENE_TYPES))
width = 0.2  # thinner bars to fit 4 groups

# Helpers to extract lists of final means & stds
def get_mut_bar_data(final_dict):
    if not final_dict: return [0.0]*5, [0.0]*5
    means = [np.mean(final_dict[f"mut_{g}"]) for g in GENE_TYPES]
    stds = [np.std(final_dict[f"mut_{g}"]) for g in GENE_TYPES]
    return means, stds

def get_act_bar_data(final_dict):
    if not final_dict: return [0.0]*5, [0.0]*5
    means = [np.mean(final_dict[f"act_{g}"]) / MAX_GENES[g] for g in GENE_TYPES]
    stds = [np.std(final_dict[f"act_{g}"]) / MAX_GENES[g] for g in GENE_TYPES]
    return means, stds

s_t_mut_m, s_t_mut_s = get_mut_bar_data(solid_t_final)
s_h_mut_m, s_h_mut_s = get_mut_bar_data(solid_h_final)
l_t_mut_m, l_t_mut_s = get_mut_bar_data(liquid_t_final)
l_h_mut_m, l_h_mut_s = get_mut_bar_data(liquid_h_final)

s_t_act_m, s_t_act_s = get_act_bar_data(solid_t_final)
s_h_act_m, s_h_act_s = get_act_bar_data(solid_h_final)
l_t_act_m, l_t_act_s = get_act_bar_data(liquid_t_final)
l_h_act_m, l_h_act_s = get_act_bar_data(liquid_h_final)

# Panel A: Final Mutation Fraction
ax_mut = axes[0]
ax_mut.bar(x - 1.5*width, s_t_mut_m, width, yerr=s_t_mut_s, color=COLORS["Solid_Tumor"], label="Solid - Tumor", capsize=3, edgecolor="none")
ax_mut.bar(x - 0.5*width, s_h_mut_m, width, yerr=s_h_mut_s, color=COLORS["Solid_Health"], label="Solid - Health", capsize=3, edgecolor="none")
ax_mut.bar(x + 0.5*width, l_t_mut_m, width, yerr=l_t_mut_s, color=COLORS["Liquid_Tumor"], label="Liquid - Tumor", capsize=3, edgecolor="none")
ax_mut.bar(x + 1.5*width, l_h_mut_m, width, yerr=l_h_mut_s, color=COLORS["Liquid_Health"], label="Liquid - Health", capsize=3, edgecolor="none")

ax_mut.set_ylabel("Mutation Fraction at End", fontsize=12, fontweight="bold")
ax_mut.set_xticks(x)
ax_mut.set_xticklabels([GENE_LABELS[g] for g in GENE_TYPES], rotation=25, ha="right")
ax_mut.set_ylim(0, 1.1)
ax_mut.grid(True, ls=":", color="#e0e0e0", alpha=0.5, axis="y", zorder=0)
ax_mut.set_axisbelow(True)
ax_mut.legend(frameon=True, edgecolor="none", fontsize=9)

# Panel B: Final Activation Level (Normalized)
ax_act = axes[1]
ax_act.bar(x - 1.5*width, s_t_act_m, width, yerr=s_t_act_s, color=COLORS["Solid_Tumor"], label="Solid - Tumor", capsize=3, edgecolor="none")
ax_act.bar(x - 0.5*width, s_h_act_m, width, yerr=s_h_act_s, color=COLORS["Solid_Health"], label="Solid - Health", capsize=3, edgecolor="none")
ax_act.bar(x + 0.5*width, l_t_act_m, width, yerr=l_t_act_s, color=COLORS["Liquid_Tumor"], label="Liquid - Tumor", capsize=3, edgecolor="none")
ax_act.bar(x + 1.5*width, l_h_act_m, width, yerr=l_h_act_s, color=COLORS["Liquid_Health"], label="Liquid - Health", capsize=3, edgecolor="none")

ax_act.set_ylabel("Relative Activation level at End", fontsize=12, fontweight="bold")
ax_act.set_xticks(x)
ax_act.set_xticklabels([GENE_LABELS[g] for g in GENE_TYPES], rotation=25, ha="right")
ax_act.set_ylim(0, 1.1)
ax_act.grid(True, ls=":", color="#e0e0e0", alpha=0.5, axis="y", zorder=0)
ax_act.set_axisbelow(True)

# Add panel labels
axes[0].text(-0.15, 1.05, "a", transform=axes[0].transAxes, fontsize=14, fontweight="bold")
axes[1].text(-0.15, 1.05, "b", transform=axes[1].transAxes, fontsize=14, fontweight="bold")

plt.savefig(OUT_DIR / "stationary_states_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "stationary_states_comparison.svg", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Output Statistics to Text
# ---------------------------------------------------------------------------
stats_file = OUT_DIR / "stationary_states_summary.txt"
print(f"Writing statistics to {stats_file}...")
with open(stats_file, "w") as out:
    out.write("=== STATISTICAL SUMMARY OF STATIONARY STATES (DIPLOID) ===\n\n")
    out.write("This summary compares the mutation fraction and activation levels of different gene types\n")
    out.write("at the end of simulations, separated by outcome (Tumor = Tumor_Max vs Health) for both models.\n\n")
    
    out.write(f"Number of Solid tumor runs:  {len(solid_tumor)} (Tumor), {len(solid_health)} (Health)\n")
    out.write(f"Number of Liquid tumor runs: {len(liquid_tumor)} (Tumor), {len(liquid_health)} (Health)\n\n")
    
    header = f"{'Gene Class':<18} | {'Metric':<10} | {'Solid - Tumor':<18} | {'Solid - Health':<18} | {'Liquid - Tumor':<18} | {'Liquid - Health':<18}\n"
    out.write(header)
    out.write("-" * len(header) + "\n")
    
    def get_stats_strings(t_final, h_final, key):
        if not t_final or not h_final:
            return "N/A", "N/A"
        t_mean, t_std = np.mean(t_final[key]), np.std(t_final[key])
        h_mean, h_std = np.mean(h_final[key]), np.std(h_final[key])
        return f"{t_mean:.4f}±{t_std:.4f}", f"{h_mean:.4f}±{h_std:.4f}"
        
    for g in GENE_TYPES:
        s_t_mut, s_h_mut = get_stats_strings(solid_t_final, solid_h_final, f"mut_{g}")
        l_t_mut, l_h_mut = get_stats_strings(liquid_t_final, liquid_h_final, f"mut_{g}")
        
        s_t_act, s_h_act = get_stats_strings(solid_t_final, solid_h_final, f"act_{g}")
        l_t_act, l_h_act = get_stats_strings(liquid_t_final, liquid_h_final, f"act_{g}")
        
        out.write(f"{GENE_LABELS[g]:<18} | {'Mutation':<10} | {s_t_mut:<18} | {s_h_mut:<18} | {l_t_mut:<18} | {l_h_mut:<18}\n")
        out.write(f"{'':<18} | {'Activation':<10} | {s_t_act:<18} | {s_h_act:<18} | {l_t_act:<18} | {l_h_act:<18}\n")
        out.write("-" * len(header) + "\n")
        
print("Analysis completed successfully!")
