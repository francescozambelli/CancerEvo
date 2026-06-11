#!/usr/bin/env python3
"""
plot_diploid_genes_trajectory.py
--------------------------------
Create a 1x4 publication-ready plot showing the mutation and activation levels 
of the O, I, and HK gene types over time for diploid solid tumor simulations.
Compares a case ending in tumor (sim_1) and a case ending in health (sim_437),
overlaying the theoretical stationary limits of the HK and I genes.
Uses a single row layout with a shared legend at the bottom.
"""

import sys
from pathlib import Path

# Setup paths relative to script location
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import matplotlib.pyplot as plt
from src.analysis.loaders import load_sim
from calculate_asymptotic_theory import compute_asymptotic_limit

# 1. Configuration & Parameters
N_I = 10
N_O = 10
N_S = 10
N_M = 5
N_HK = 10
DMU = 0.015

# Load simulations from ensemble_results_D
SIM_T_ID = 10    # Tumor outcome
SIM_H_ID = 1  # Health outcome
sim_t = load_sim(SIM_T_ID, "ensemble_results_D")
sim_h = load_sim(SIM_H_ID, "ensemble_results_D")

# Compute theoretical limits using the Master Equation solver
theory = compute_asymptotic_limit(N_I=N_I, N_H=N_HK, dmu=DMU, remove_lower=0)
i_act_limit = 2.55#theory["asymp_level_analytical"]  # Expected active instability genes (~2.1742)
hk_mut_limit = 0.5                              # Max mutation fraction for surviving HK genes

# 2. Setup Plotting Style (with increased font sizes)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Define distinctive color palette for gene types
colors = {
    "I": "#2A9D8F",   # Teal
    "O": "#F4A261",   # Sandy Orange
    "HK": "#E63946",  # Vivid Crimson/Red
}

gene_labels = {
    "I": r"Instability ($\mathcal{I}$)",
    "O": r"Oncogenes ($\mathcal{O}$)",
    "HK": r"Housekeeping ($\mathcal{H}$)",
}

# 3. Create the Figure layout (1x4 horizontal line)
fig, axes = plt.subplot_mosaic(
    [["A", "B", "C", "D"]],
    figsize=(22, 5.0)
)

fontsize_label = 22
fontsize_letter = 28
fontsize_ticks = 20
fontsize_legend = 22
line_width = 3.5

# Helpers to plot trajectories (no internal legends)
def plot_mutation_panel(ax, sim_data, label_letter):
    steps = np.arange(len(sim_data["tumor_density"]))
    
    # Plot mutation levels for each gene type (only I, O, HK)
    
    for gtype in ["I", "O", "HK"]:
        ax.plot(steps, sim_data[f"mut_{gtype}"], color=colors[gtype], lw=line_width, label=gene_labels[gtype])
        
    # Draw theoretical limit for HK mutation level (0.5)
    ax.axhline(hk_mut_limit, color=colors["HK"], linestyle="--", lw=line_width, alpha=0.8,
               label=rf"Theory $\mathcal{{H}}$ mut limit ({hk_mut_limit})")
               
    ax.set_xlabel("Simulation step", fontsize=fontsize_label)
    ax.set_ylabel("Mutation level", fontsize=fontsize_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.set_ylim(-0.05, 1.05)
    ax.set_axisbelow(True)
    
    # Lettering label (larger font size)
    
    ax.text(-0.12, 1.10, label_letter, transform=ax.transAxes, fontsize=fontsize_letter, fontweight="bold", va="top", ha="right")

def plot_activation_panel(ax, sim_data, label_letter):
    steps = np.arange(len(sim_data["tumor_density"]))
    
    # Plot activation levels for each gene type (only I, O, HK)
    for gtype in ["I", "O", "HK"]:
        ax.plot(steps, sim_data[f"act_{gtype}"], color=colors[gtype], lw=line_width, label=gene_labels[gtype])
        
    # Draw theoretical limit for I activation level (~2.17)
    ax.axhline(i_act_limit, color=colors["I"], linestyle="--", lw=line_width, alpha=0.8,
               label=rf"Boundary $\mathcal{{I}}$ active level ({i_act_limit:.2f})")
               
    ax.set_xlabel("Simulation step", fontsize=fontsize_label)
    ax.set_ylabel("Activation level", fontsize=fontsize_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.set_ylim(-0.5, 10.5)
    ax.set_axisbelow(True)
    
    # Lettering label (larger font size)
    ax.text(-0.12, 1.10, label_letter, transform=ax.transAxes, fontsize=fontsize_letter, fontweight="bold", va="top", ha="right")

# 4. Populate the subplots (all in 1 row)
# A & B: Tumor Case (sim_1)
plot_mutation_panel(axes["A"], sim_t, "a")
plot_activation_panel(axes["B"], sim_t, "b")

# C & D: Health Case (sim_437)
plot_mutation_panel(axes["C"], sim_h, "c")
plot_activation_panel(axes["D"], sim_h, "d")

# 5. Extract handles and labels to build a single global legend
handles, labels = [], []
for ax in [axes["A"], axes["B"]]:
    h, l = ax.get_legend_handles_labels()
    for hi, li in zip(h, l):
        if li not in labels:
            handles.append(hi)
            labels.append(li)

# Precise manual subplots adjustment for 1x4 layout leaving space at bottom for legend
fig.subplots_adjust(bottom=0.22, top=0.90, left=0.05, right=0.98, wspace=0.28)

# Place the shared horizontal legend at the bottom center (larger size)
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    frameon=True,
    edgecolor="#e0e0e0",
    facecolor="white",
    fontsize=fontsize_legend,
    framealpha=0.9,
    bbox_to_anchor=(0.5, 1.15)
)

# 6. Export Figure
out_dir = REPO_ROOT / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)

for ext in ["png", "svg"]:
    out_path = out_dir / f"diploid_genes_trajectory.{ext}"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False, format=ext)
    print(f"Saved figure: {out_path}")

plt.close(fig)
print("Plotting completed successfully.")
