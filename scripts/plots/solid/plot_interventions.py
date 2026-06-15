# scripts/plots/solid/plot_interventions.py

import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# FONT SIZE CONFIGURATION
# Modify these variables to easily change the font sizes across the entire figure.
# ==============================================================================
FONT_SIZE_BASE = 14       # Default font size for tick labels and body text
FONT_SIZE_LABEL = 22      # Axis labels (x and y)
FONT_SIZE_TICK = 20       # Tick label sizes
FONT_SIZE_TITLE = 22      # Subplot titles
FONT_SIZE_LEGEND = 22     # Legend text size
FONT_SIZE_PANEL = 25      # Panel letters (A, B, C, D)
# ==============================================================================

def save_publication_figure(fig, name_base, output_dir="outputs/figures/solid", dpi=300):
    """
    Saves a matplotlib figure to both high-resolution PNG and vector SVG formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{name_base}.png")
    svg_path = os.path.join(output_dir, f"{name_base}.svg")
    pdf_path = os.path.join(output_dir, f"{name_base}.pdf")
    
    # Save PNG
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    # Save SVG
    fig.savefig(svg_path, bbox_inches='tight', transparent=True)
    # Save PDF
    fig.savefig(pdf_path, bbox_inches='tight', transparent=True)
    print(f"Saved figure:\n  - Raster: {png_path}\n  - Vector: {svg_path}")

def main():
    # Style setup using configured variables
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_SIZE_BASE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
    })

    # Load results
    data_path = "data/simulations/intervention_results.npz"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run the simulation script first.")
        return

    data = np.load(data_path)
    
    interventions = {
        "a": {
            "title": r"Remove cells proportionally to $r$",
            "dens": data["dens_a"],
            "mu": data["mu_a"],
            "k_int": data["k_int_a"][0]
        },
        "b": {
            "title": "Clear low-instability clones",
            "dens": data["dens_b"],
            "mu": data["mu_b"],
            "k_int": data["k_int_b"][0]
        },
        "c": {
            "title": "Clear high-instability clones",
            "dens": data["dens_c"],
            "mu": data["mu_c"],
            "k_int": data["k_int_c"][0]
        },
        "d": {
            "title":  "Double instability burden",
            "dens": data["dens_d"],
            "mu": data["mu_d"],
            "k_int": data["k_int_d"][0]
        }
    }

    # Plot everything in 1 row (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.0), facecolor='white')
    axes_flat = axes.flatten()
    
    # Colors
    color_dens = "#0f4c81"  # Deep classic blue
    color_mu = "#d95f02"    # Colorblind-safe dark orange
    
    # Store line objects for the figure-level legend
    l1, l2, l3 = None, None, None
    
    for i, (key, info) in enumerate(interventions.items()):
        ax_left = axes_flat[i]
        ax_left.set_facecolor('white')
        
        # Plot tumor density
        steps = np.arange(len(info["dens"]))
        l1, = ax_left.plot(steps, info["dens"], color=color_dens, lw=2.5, label="Tumor density")
        
        ax_left.set_title(info["title"], fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=12)
        ax_left.set_ylim(-0.02, 1.02)
        ax_left.set_xlabel("Time steps", labelpad=6, fontsize=FONT_SIZE_LABEL)
        if i == 0:
            ax_left.set_ylabel("Tumor density", color=color_dens, fontsize=FONT_SIZE_LABEL)
        ax_left.tick_params(axis='y', labelcolor=color_dens, labelsize=FONT_SIZE_TICK)
        
        # Plot mean mutation rate on the right y-axis
        ax_right = ax_left.twinx()
        l2, = ax_right.plot(steps, info["mu"], color=color_mu, lw=2.5, label=r"Mean $\mu$")
        if i == 3:
            ax_right.set_ylabel(r"Mean mutation rate $\langle \mu \rangle$", color=color_mu, fontsize=FONT_SIZE_LABEL)
        ax_right.tick_params(axis='y', labelcolor=color_mu, labelsize=FONT_SIZE_TICK)
        ax_right.set_ylim(bottom=0.0)
        
        # Add intervention vertical line
        l3 = ax_left.axvline(x=info["k_int"], color="#333333", linestyle="--", lw=2.0, label="Intervention Instant")
        
        # De-clutter & Color spines
        ax_left.spines['left'].set_color(color_dens)
        ax_left.spines['left'].set_linewidth(1.5)
        ax_left.spines['right'].set_visible(False)
        ax_left.spines['top'].set_visible(False)
        ax_left.spines['bottom'].set_color('#555555')
        ax_left.spines['bottom'].set_linewidth(1.0)
        
        ax_right.spines['right'].set_color(color_mu)
        ax_right.spines['right'].set_linewidth(1.5)
        ax_right.spines['left'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['bottom'].set_visible(False)
        
        # Gridlines (placed behind data)
        ax_left.grid(True, linestyle=":", alpha=0.5, color="#d3d3d3")
        ax_left.set_axisbelow(True)
        
        # Panel labels (A, B, C, D)
        ax_left.text(-0.16, 1.08, key.lower(), transform=ax_left.transAxes, 
                     fontsize=FONT_SIZE_PANEL, fontweight='bold', va='top', ha='right')

    # Position a single unified legend at the bottom of the figure
    handles = [l1, l2, l3]
    labels = ["Tumor density", r"Mean mutation rate $\langle \mu \rangle$", "Intervention Instant"]
    
    # Adjust layout to make room for the bottom legend
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.98),
               frameon=True, edgecolor="#d3d3d3", fontsize=FONT_SIZE_LEGEND)

    save_publication_figure(fig, "interventions_4panel")
    plt.close(fig)

if __name__ == "__main__":
    main()
