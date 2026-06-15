"""
plot_stability_phase_diagram_liquid.py
--------------------------------------
Plot a 2D phase diagram (heatmap) showing the fraction of liquid-tumor simulations
that end with a tumor present, as a function of (rmax_norm, dmu) under instability
gene saturation conditions.

Outputs
-------
outputs/figures/liquid/stability_phase_diagram_liquid.png
outputs/figures/liquid/stability_phase_diagram_liquid.svg
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set plotting style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fontsize_ticks = 20
fontsize_labels = 23
fontsize_cbar = 20
fontsize_legend = 18

def plot_heatmap():
    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / "data" / "stability_phase_diagram_results_liquid.csv"
    output_dir = project_root / "outputs" / "figures" / "liquid"
    output_dir.mkdir(parents=True, exist_ok=True)
    root_dir = output_dir.parent
    
    output_path_png = output_dir / "stability_phase_diagram_liquid.png"
    output_path_svg = output_dir / "stability_phase_diagram_liquid.svg"
    output_path_png_root = root_dir / "stability_phase_diagram_liquid.png"
    output_path_svg_root = root_dir / "stability_phase_diagram_liquid.svg"

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run the Julia liquid sweep script first.")
        sys.exit(1)

    # Load data
    df = pd.read_csv(csv_path)
    
    # Pivot the dataframe to get a grid for 2D plotting
    # Rows: dmu (y-axis), Columns: rmax_norm (x-axis)
    pivot_df = df.pivot(index='dmu', columns='rmax_norm', values='fraction')
    
    # Sort index so dmu goes from lowest to highest
    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

    rmax_vals = pivot_df.columns.values
    dmu_vals = pivot_df.index.values
    z = pivot_df.values

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Use pcolormesh for clean grid alignment
    rmax_edges = np.zeros(len(rmax_vals) + 1)
    rmax_edges[0] = rmax_vals[0] - (rmax_vals[1] - rmax_vals[0])/2
    rmax_edges[1:-1] = (rmax_vals[:-1] + rmax_vals[1:]) / 2
    rmax_edges[-1] = rmax_vals[-1] + (rmax_vals[-1] - rmax_vals[-2])/2

    dmu_edges = np.zeros(len(dmu_vals) + 1)
    dmu_edges[0] = dmu_vals[0] - (dmu_vals[1] - dmu_vals[0])/2
    dmu_edges[1:-1] = (dmu_vals[:-1] + dmu_vals[1:]) / 2
    dmu_edges[-1] = dmu_vals[-1] + (dmu_vals[-1] - dmu_vals[-2])/2

    im = ax.pcolormesh(rmax_edges, dmu_edges, z, cmap="plasma", vmin=0.0, vmax=1.0, edgecolors='face', linewidths=0, rasterized=True)
    
    # Theoretical curve calculation
    # P_s^* = 0.5 * (1 + 1/rmax_norm)
    # P_s^* = (1 - c*dmu*)^N_HK -> dmu* = (1 - (P_s^*)^(1/N_HK)) / c
    c = 10
    N_HK = 10
    theory_rmax = np.linspace(rmax_edges[0], rmax_edges[-1], 200)
    p_star = 0.5 * (1.0 + 1.0 / theory_rmax)
    theory_dmu = (1.0 - p_star ** (1.0 / N_HK)) / 10
    
    ax.plot(theory_rmax, theory_dmu, color='white', linestyle='--', linewidth=3.5, label='Theoretical Boundary')
    ax.plot(theory_rmax, theory_dmu, color='black', linestyle=':', linewidth=1.5)

    # Labeling
    ax.set_xlabel(r"Normalized Division Rate ($r_{max} / r_0$)", fontsize=fontsize_labels, labelpad=10)
    ax.set_ylabel(r"Mutation rate increment ($d\mu$)", fontsize=fontsize_labels, labelpad=10)
    
    ax.text(0.95, 0.05, "Expansion", 
                transform=ax.transAxes, color='black', fontsize=18, fontweight='bold',
                ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', linewidth=0.2, boxstyle='round,pad=0.4'))
    ax.text(0.05, 0.95, "Collapse", 
                transform=ax.transAxes, color='white', fontsize=18, fontweight='bold',
                ha='left', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='black', linewidth=0.2, boxstyle='round,pad=0.4'))

    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    
    ax.set_xlim(rmax_edges[0], rmax_edges[-1])
    ax.set_ylim(dmu_edges[0], dmu_edges[-1])
    ax.grid(False)

    ax.legend(loc="upper right", framealpha=0.9, facecolor='white', edgecolor='none', fontsize=fontsize_legend)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label("Frac. runs ending with tumor", fontsize=fontsize_cbar, labelpad=12)
    cbar.ax.tick_params(labelsize=fontsize_ticks)

    plt.tight_layout()
    plt.savefig(output_path_png, dpi=200, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    plt.savefig(output_path_png_root, dpi=200, bbox_inches="tight")
    plt.savefig(output_path_svg_root, bbox_inches="tight")
    print(f"Plot saved successfully to:\n  - {output_path_png}\n  - {output_path_svg}\n  - {output_path_png_root}\n  - {output_path_svg_root}")

if __name__ == "__main__":
    plot_heatmap()
