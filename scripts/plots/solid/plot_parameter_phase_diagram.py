"""
plot_parameter_phase_diagram.py
-------------------------------
Plot a 2D phase diagram (heatmap) showing the fraction of simulations that
end with a tumor present (either exceeding the mass limit 'Tumor_Max' or
persisting until the end of steps 'Done') as a function of the parameter grid (dmu, dr).

Outputs
-------
outputs/figures/parameter_phase_diagram.png
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
fontsize_legend = 20


def plot_heatmap():
    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / "outputs" / "results" / "parameter_phase_diagram_results_solid.csv"
    output_dir = project_root / "outputs" / "figures" / "solid"
    output_dir.mkdir(parents=True, exist_ok=True)
    root_dir = output_dir.parent
    output_path_png = output_dir / "parameter_phase_diagram_solid.png"
    output_path_svg = output_dir / "parameter_phase_diagram_solid.svg"
    output_path_png_root = root_dir / "parameter_phase_diagram_solid.png"
    output_path_svg_root = root_dir / "parameter_phase_diagram_solid.svg"


    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run the Julia sweep script first.")
        sys.exit(1)

    # Load data
    df = pd.read_csv(csv_path)
    
    # Pivot the dataframe to get a grid for 2D plotting
    # Rows: dr (y-axis), Columns: dmu (x-axis)
    pivot_df = df.pivot(index='dr', columns='dmu', values='fraction')
    
    # Sort index so dr goes from lowest to highest (bottom to top in imshow with origin='lower')
    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

    dmu_vals = pivot_df.columns.values
    dr_vals = pivot_df.index.values
    z = pivot_df.values

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    # Use pcolormesh for clean grid alignment
    # Shift values to represent bin edges
    dmu_edges = np.zeros(len(dmu_vals) + 1)
    dmu_edges[0] = dmu_vals[0] - (dmu_vals[1] - dmu_vals[0])/2
    dmu_edges[1:-1] = (dmu_vals[:-1] + dmu_vals[1:]) / 2
    dmu_edges[-1] = dmu_vals[-1] + (dmu_vals[-1] - dmu_vals[-2])/2

    dr_edges = np.zeros(len(dr_vals) + 1)
    dr_edges[0] = dr_vals[0] - (dr_vals[1] - dr_vals[0])/2
    dr_edges[1:-1] = (dr_vals[:-1] + dr_vals[1:]) / 2
    dr_edges[-1] = dr_vals[-1] + (dr_vals[-1] - dr_vals[-2])/2

    im = ax.pcolormesh(dmu_edges, dr_edges, z, cmap="plasma", vmin=0.0, vmax=1.0, edgecolors='face', linewidths=0, rasterized=True)
    
    # Labeling
    ax.set_xlabel(r"Mutation rate increment ($d\mu$)", fontsize=fontsize_labels, labelpad=10)
    ax.set_ylabel(r"Division rate increment ($dr$)", fontsize=fontsize_labels, labelpad=10)
    #ax.set_title("Liquid Tumor Progression / Persistence Fraction", fontsize=18, fontweight="bold", pad=15)
    
    ax.text(0.15, 0.55, "Expansion", 
                transform=ax.transAxes, color='black', fontsize=18, fontweight='bold',
                ha='left', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', linewidth=0.2, boxstyle='round,pad=0.4'))
    ax.text(0.71, 0.14, "Collapse", 
                transform=ax.transAxes, color='white', fontsize=18, fontweight='bold',
                ha='left', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='black', linewidth=0.2, boxstyle='round,pad=0.4'))

    # Automatically locate and format ticks dynamically (reducing number of ticks)
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.tick_params(axis='x', which='major', rotation=45)
    ax.grid(False)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label("Frac. runs ending with tumor", fontsize=fontsize_cbar, labelpad=12)
    cbar.ax.tick_params(labelsize=fontsize_ticks)

    # Highlight default settings from parameters_liquid.jl if within sweep range
    # default dmu = 0.045, dr = 0.008
    default_dmu, default_dr = 0.012, 0.008
    if dmu_vals.min() <= default_dmu <= dmu_vals.max() and dr_vals.min() <= default_dr <= dr_vals.max():
        ax.scatter(default_dmu, default_dr, color='cyan', marker='*', s=650, edgecolors='black', 
                   linewidths=1.5, zorder=10, label=f"Default Parameters\n($d\mu={default_dmu:.1e}$,\n $dr={default_dr:.1e}$)")
        ax.legend(loc="upper left", framealpha=0.9, facecolor='white', edgecolor='none', fontsize=fontsize_legend)
    else:
        # If outside the sweep range, add text to show default params
        ax.text(0.97, 0.95, f"Default parameters outside sweep:\n($d\mu={default_dmu:.1e}$,\n $dr={default_dr:.1e}$)", 
                transform=ax.transAxes, color='white', fontsize=11, fontweight='semibold',
                ha='right', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.4'))

    

    plt.tight_layout()
    plt.savefig(output_path_png, dpi=200, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    plt.savefig(output_path_png_root, dpi=200, bbox_inches="tight")
    plt.savefig(output_path_svg_root, bbox_inches="tight")
    print(f"Plot saved successfully to:\n  - {output_path_png}\n  - {output_path_svg}\n  - {output_path_png_root}\n  - {output_path_svg_root}")

if __name__ == "__main__":
    plot_heatmap()

