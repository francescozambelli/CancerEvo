import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# Add plotting guidelines skill to path
sys.path.append("/home/francesco/Antigravity/SKILLS/plotting-guidelines/scripts")
try:
    from plotting_utils import save_publication_figure
except ImportError:
    # Fallback if not found
    def save_publication_figure(fig, filename, output_dir="plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"), bbox_inches='tight')

def load_data(sweep_type):
    data_dir = f"../../../data/phase_transition/{sweep_type}"
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if not files:
        print(f"No .npz files found in {data_dir}")
        return None, None, None

    data = {}
    for f in files:
        try:
            res = np.load(f)
            val = float(res[sweep_type])
            time = float(res["time"])
            if val not in data:
                data[val] = []
            data[val].append(time)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not data:
        return None, None, None

    vals = sorted(list(data.keys()))
    means = []
    sems = []
    
    for v in vals:
        times = data[v]
        means.append(np.mean(times))
        sems.append(np.std(times, ddof=1) / np.sqrt(len(times)) if len(times) > 1 else 0.0)

    return np.array(vals), np.array(means), np.array(sems)

def plot_combined():
    fontsize_labels = 18
    fontsize_ticks = 15
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 9))
    
    # --- DMU PLOT ---
    vals_dmu, means_dmu, sems_dmu = load_data("dmu")
    if vals_dmu is not None:
        ax = axes[0]
        color = 'royalblue'  # Tab:blue
        
        # Scale vals_dmu by 10^3 to avoid messy decimals and match DR plot scale
        scale_factor = 1e3
        vals_dmu_scaled = vals_dmu * scale_factor
        
        # Plot data (rotated)
        ax.errorbar(means_dmu, vals_dmu_scaled, xerr=sems_dmu, fmt='-o', color=color, markersize=4, capsize=3, linewidth=1.5, zorder=3)
        ax.fill_betweenx(vals_dmu_scaled, means_dmu - sems_dmu, means_dmu + sems_dmu, color=color, alpha=0.2, zorder=2)
        
        # Horizontal line at peak
        peak_idx = np.argmax(means_dmu)
        peak_val = vals_dmu_scaled[peak_idx]
        ax.axhline(peak_val, color='#7f7f7f', linestyle='--', alpha=0.8, linewidth=1.5, zorder=1)
        
        # Shade the supercritical region (above the peak)
        ax.axhspan(peak_val, max(vals_dmu_scaled), color='#7f7f7f', alpha=0.08, zorder=0)
        
        # Add labels for subcritical and supercritical regions
        ax.text(0.40, 0.20, "subcritical", transform=ax.transAxes, ha='center', va='center', fontsize=14, color='#444444', fontweight='semibold')
        ax.text(0.40, 0.80, "supercritical", transform=ax.transAxes, ha='center', va='center', fontsize=14, color='#444444', fontweight='semibold')
        
        # Formatting
        ax.set_ylabel(r"$\Delta \mu$ ($\times 10^{-3}$)", fontsize=fontsize_labels)
        ax.set_xlabel("Mean Simulation Time (steps)", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    # --- DR PLOT ---
    vals_dr, means_dr, sems_dr = load_data("dr")
    if vals_dr is not None:
        ax = axes[1]
        color = 'tomato'  # Tab:orange
        
        # Scale vals_dr by 10^3 to avoid messy scientific notation on ticks
        scale_factor = 1e3
        vals_dr_scaled = vals_dr * scale_factor
        
        # Plot data (rotated)
        ax.errorbar(means_dr, vals_dr_scaled, xerr=sems_dr, fmt='-s', color=color, markersize=4, capsize=3, linewidth=1.5, zorder=3)
        ax.fill_betweenx(vals_dr_scaled, means_dr - sems_dr, means_dr + sems_dr, color=color, alpha=0.2, zorder=2)
        
        # Horizontal line at peak
        peak_idx = np.argmax(means_dr)
        peak_val = vals_dr_scaled[peak_idx]
        ax.axhline(peak_val, color='#7f7f7f', linestyle='--', alpha=0.8, linewidth=1.5, zorder=1)
        
        # Shade the supercritical region (above the peak)
        ax.axhspan(peak_val, max(vals_dr_scaled), color='#7f7f7f', alpha=0.08, zorder=0)
        
        # Add labels for subcritical and supercritical regions
        ax.text(0.40, 0.20, "subcritical", transform=ax.transAxes, ha='center', va='center', fontsize=14, color='#444444', fontweight='semibold')
        ax.text(0.40, 0.80, "supercritical", transform=ax.transAxes, ha='center', va='center', fontsize=14, color='#444444', fontweight='semibold')
        
        # Formatting
        ax.set_ylabel(r"$\Delta r$ ($\times 10^{-3}$)", fontsize=fontsize_labels)
        ax.set_xlabel("Mean Simulation Time (steps)", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    # Add subplot letters (g, h)
    #axes[0].text(-0.22, 1.15, "g", transform=axes[0].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    #axes[1].text(-0.22, 1.15, "h", transform=axes[1].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    
    output_dir = "../../../outputs/figures/phase_transition"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_publication_figure(fig, "extinction_time_combined", output_dir=output_dir)
    print(f"Saved combined plot to {output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    plot_combined()
