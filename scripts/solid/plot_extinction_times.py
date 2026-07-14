import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

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
    data_dir = f"../../data/phase_transition/{sweep_type}"
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
    fontsize_labels = 13
    fontsize_ticks = 11
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # --- DMU PLOT ---
    vals_dmu, means_dmu, sems_dmu = load_data("dmu")
    if vals_dmu is not None:
        ax = axes[0]
        color = '#1f77b4'  # Tab:blue
        
        # Plot data
        ax.errorbar(vals_dmu, means_dmu, yerr=sems_dmu, fmt='-o', color=color, markersize=4, capsize=3, linewidth=1.5, zorder=3)
        ax.fill_between(vals_dmu, means_dmu - sems_dmu, means_dmu + sems_dmu, color=color, alpha=0.2, zorder=2)
        
        # Vertical line at peak
        peak_idx = np.argmax(means_dmu)
        peak_val = vals_dmu[peak_idx]
        ax.axvline(peak_val, color='#7f7f7f', linestyle='--', alpha=0.8, linewidth=1.5, zorder=1)
        
        # Formatting
        ax.set_xlabel(r"Mutation rate step, $\Delta \mu$", fontsize=fontsize_labels)
        ax.set_ylabel("Mean Simulation Time (steps)", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    # --- DR PLOT ---
    vals_dr, means_dr, sems_dr = load_data("dr")
    if vals_dr is not None:
        ax = axes[1]
        color = '#ff7f0e'  # Tab:orange
        
        # Plot data
        ax.errorbar(vals_dr, means_dr, yerr=sems_dr, fmt='-s', color=color, markersize=4, capsize=3, linewidth=1.5, zorder=3)
        ax.fill_between(vals_dr, means_dr - sems_dr, means_dr + sems_dr, color=color, alpha=0.2, zorder=2)
        
        # Vertical line at peak
        peak_idx = np.argmax(means_dr)
        peak_val = vals_dr[peak_idx]
        ax.axvline(peak_val, color='#7f7f7f', linestyle='--', alpha=0.8, linewidth=1.5, zorder=1)
        
        # Formatting
        ax.set_xlabel(r"Replication rate step, $\Delta r$", fontsize=fontsize_labels)
        # Only y-axis label on the left subplot to save space, but keeping it helps readability if axes differ
        # ax.set_ylabel("Mean Simulation Time (steps)", fontsize=fontsize_labels) 
        ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    # Add subplot letters (a, b)
    axes[0].text(-0.1, 1.05, "a", transform=axes[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    axes[1].text(-0.1, 1.05, "b", transform=axes[1].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    
    output_dir = "../../outputs/figures/phase_transition"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_publication_figure(fig, "extinction_time_combined", output_dir=output_dir)
    print(f"Saved combined plot to {output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    plot_combined()
