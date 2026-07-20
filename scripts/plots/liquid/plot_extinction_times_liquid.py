import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import to_hex

parser = argparse.ArgumentParser()
parser.add_argument("--init_mass_pct", type=int, default=10)
parser.add_argument("--limit_pct", type=int, default=60)
parser.add_argument("--n_steps", type=int, default=10000)
args = parser.parse_args()

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
    steps_str = "" if args.n_steps == 10000 else f"_steps{args.n_steps}"
    data_dir = f"../../../data/phase_transition_liquid{steps_str}_init{args.init_mass_pct}_limit{args.limit_pct}/{sweep_type}"
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if not files:
        print(f"No .npz files found in {data_dir}")
        return None, None, None

    data = {}
    for f in files:
        try:
            res = np.load(f)
            val = float(np.atleast_1d(res[sweep_type])[0])
            time = float(np.atleast_1d(res["time"])[0])
            if val not in data:
                data[val] = []
            data[val].append(time)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not data:
        return None, None, None

    vals = sorted(list(data.keys()))
    means = []
    
    for v in vals:
        times = data[v]
        means.append(np.median(times))

    return np.array(vals), data, np.array(means)

def plot_combined():
    fontsize_labels = 18
    fontsize_ticks = 15
    
    fig, axes = plt.subplots(2, 1, figsize=(7, 9))
    
    # --- DMU PLOT ---
    vals_dmu, data_time_dmu, means_dmu = load_data("dmu")
    if vals_dmu is not None:
        ax = axes[0]
        scale_factor = 1e3
        vals_dmu_scaled = vals_dmu * scale_factor
        
        # Horizontal line at critical point
        sorted_indices = np.argsort(means_dmu)
        peak_val = (vals_dmu_scaled[sorted_indices[-1]] + vals_dmu_scaled[sorted_indices[-2]]) / 2.0
        ax.axhline(peak_val, color='#7f7f7f', linestyle='--', alpha=0.8, linewidth=1.5, zorder=1)
        
        # Separate branches
        sub_idx = vals_dmu_scaled <= peak_val
        sup_idx = vals_dmu_scaled >= peak_val
        
        # Plot individual replica points (small dots)
        for i, v in enumerate(vals_dmu):
            y_val = vals_dmu_scaled[i]
            times = data_time_dmu[v]
            color = '#08519c' if y_val <= peak_val else '#6baed6'
            ax.scatter(times, [y_val] * len(times), color=color, s=16, alpha=0.55, zorder=2, edgecolors='none')
        
        # Subcritical & Supercritical Mean Trend Lines
        # ax.plot(means_dmu[sub_idx], vals_dmu_scaled[sub_idx], '-', color='#08519c', linewidth=1.5, label='Subcritical', zorder=3)
        # ax.plot(means_dmu[sup_idx], vals_dmu_scaled[sup_idx], '--', color='#6baed6', linewidth=1.5, label='Supercritical', zorder=3)
        
        # Shade the supercritical region (above the peak)
        ax.axhspan(peak_val, max(vals_dmu_scaled), color='#7f7f7f', alpha=0.08, zorder=0)
        
        # Add region labels centered in their respective sub-regions
        sub_y = (min(vals_dmu_scaled) + peak_val) / 2.0
        sup_y = (peak_val + max(vals_dmu_scaled)) / 2.0
        ax.text(0.50, sub_y, "subcritical", transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=14, color='#444444', fontweight='semibold')
        ax.text(0.50, sup_y, "supercritical", transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=14, color='#444444', fontweight='semibold')
        
        # Formatting
        ax.set_ylabel(r"$\Delta \mu$ ($\times 10^{-3}$)", fontsize=fontsize_labels)
        ax.set_xlabel("Extinction time", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)

    # --- DR PLOT ---
    vals_dr, data_time_dr, means_dr = load_data("dr")
    if vals_dr is not None:
        ax = axes[1]
        scale_factor = 1e3
        vals_dr_scaled = vals_dr * scale_factor
        
        # Horizontal line at critical point
        sorted_indices = np.argsort(means_dr)
        peak_val = (vals_dr_scaled[sorted_indices[-1]] + vals_dr_scaled[sorted_indices[-2]]) / 2.0
        ax.axhline(peak_val, color='#7f7f7f', linestyle='--', alpha=0.8, linewidth=1.5, zorder=1)
        
        # Separate branches
        sub_idx = vals_dr_scaled <= peak_val
        sup_idx = vals_dr_scaled >= peak_val
        
        # Plot individual replica points (small dots)
        for i, v in enumerate(vals_dr):
            y_val = vals_dr_scaled[i]
            times = data_time_dr[v]
            color = '#a50f15' if y_val <= peak_val else '#fb6a4a'
            ax.scatter(times, [y_val] * len(times), color=color, s=16, alpha=0.55, zorder=2, edgecolors='none')
        
        # Subcritical & Supercritical Mean Trend Lines
        # ax.plot(means_dr[sub_idx], vals_dr_scaled[sub_idx], '-', color='#a50f15', linewidth=1.5, label='Subcritical', zorder=3)
        # ax.plot(means_dr[sup_idx], vals_dr_scaled[sup_idx], '--', color='#fb6a4a', linewidth=1.5, label='Supercritical', zorder=3)
        
        # Shade the supercritical region
        ax.axhspan(peak_val, max(vals_dr_scaled), color='#7f7f7f', alpha=0.08, zorder=0)
        
        # Add region labels centered in their respective sub-regions
        sub_y_dr = (min(vals_dr_scaled) + peak_val) / 2.0
        sup_y_dr = (peak_val + max(vals_dr_scaled)) / 2.0
        ax.text(0.50, sub_y_dr, "subcritical", transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=14, color='#444444', fontweight='semibold')
        ax.text(0.50, sup_y_dr, "supercritical", transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=14, color='#444444', fontweight='semibold')
        
        # Formatting
        ax.set_ylabel(r"$\Delta r$ ($\times 10^{-3}$)", fontsize=fontsize_labels)
        ax.set_xlabel("Extinction time", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)

    plt.tight_layout()
    
    steps_str = "" if args.n_steps == 10000 else f"_steps{args.n_steps}"
    output_dir = f"../../../outputs/figures/phase_transition_liquid{steps_str}/init{args.init_mass_pct}_limit{args.limit_pct}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_publication_figure(fig, "extinction_time_combined_liquid", output_dir=output_dir)
    print(f"Saved combined plot to {output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    plot_combined()
