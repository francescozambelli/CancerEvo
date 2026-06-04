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

def plot_heatmap():
    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / "data" / "parameter_phase_diagram_results.csv"
    output_dir = project_root / "outputs" / "figures" / "solid"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_png = output_dir / "parameter_phase_diagram.png"
    output_path_svg = output_dir / "parameter_phase_diagram.svg"


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
    ax.set_xlabel(r"Mutation rate increment ($d\mu$)", fontsize=16, labelpad=10)
    ax.set_ylabel(r"Division rate increment ($dr$)", fontsize=16, labelpad=10)
    ax.set_title("Tumor Progression / Persistence Fraction", fontsize=18, fontweight="bold", pad=15)
    
    # Automatically locate and format ticks dynamically (reducing number of ticks)
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(False)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label("Fraction of runs ending with tumor present", fontsize=14, labelpad=12)
    cbar.ax.tick_params(labelsize=12)


    # Highlight default settings from parameters.jl if within sweep range
    # default dmu = 0.015, dr = 0.008
    default_dmu, default_dr = 0.015, 0.008
    if dmu_vals.min() <= default_dmu <= dmu_vals.max() and dr_vals.min() <= default_dr <= dr_vals.max():
        ax.scatter(default_dmu, default_dr, color='cyan', marker='*', s=150, edgecolors='black', 
                   linewidths=1.5, zorder=10, label=f"Default Parameters\n(dmu={default_dmu}, dr={default_dr})")
        ax.legend(loc="upper left", framealpha=0.9, facecolor='white', edgecolor='none')

    plt.tight_layout()
    plt.savefig(output_path_png, dpi=200, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    print(f"Plot saved successfully to:\n  - {output_path_png}\n  - {output_path_svg}")

if __name__ == "__main__":
    plot_heatmap()

