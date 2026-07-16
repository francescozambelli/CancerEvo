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
    data_dir = f"../../../data/phase_transition_init{args.init_mass_pct}_limit{args.limit_pct}/{sweep_type}"
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
        # Scale vals_dmu by 10^3 to avoid messy decimals and match DR plot scale
        scale_factor = 1e3
        vals_dmu_scaled = vals_dmu * scale_factor
        
        # Horizontal line at critical point (midpoint of highest and second highest)
        sorted_indices = np.argsort(means_dmu)
        peak_val = (vals_dmu_scaled[sorted_indices[-1]] + vals_dmu_scaled[sorted_indices[-2]]) / 2.0
        ax.axhline(peak_val, color='#7f7f7f', linestyle='--', alpha=0.8, linewidth=1.5, zorder=1)
        
        # Separate branches
        sub_idx = vals_dmu_scaled <= peak_val
        sup_idx = vals_dmu_scaled >= peak_val
        
        # Subcritical Plot
        ax.errorbar(means_dmu[sub_idx], vals_dmu_scaled[sub_idx], xerr=sems_dmu[sub_idx], fmt='-o', color='#08519c', markersize=4, capsize=3, linewidth=1.5, zorder=3)
        ax.fill_betweenx(vals_dmu_scaled[sub_idx], means_dmu[sub_idx] - sems_dmu[sub_idx], means_dmu[sub_idx] + sems_dmu[sub_idx], color='#08519c', alpha=0.2, zorder=2)
        
        # Supercritical Plot
        ax.errorbar(means_dmu[sup_idx], vals_dmu_scaled[sup_idx], xerr=sems_dmu[sup_idx], fmt='--^', color='#6baed6', markersize=4, capsize=3, linewidth=1.5, zorder=3)
        ax.fill_betweenx(vals_dmu_scaled[sup_idx], means_dmu[sup_idx] - sems_dmu[sup_idx], means_dmu[sup_idx] + sems_dmu[sup_idx], color='#6baed6', alpha=0.2, zorder=2)
        
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
        # Scale vals_dr by 10^3 to avoid messy scientific notation on ticks
        scale_factor = 1e3
        vals_dr_scaled = vals_dr * scale_factor
        
        # Horizontal line at critical point (midpoint of highest and second highest)
        sorted_indices = np.argsort(means_dr)
        peak_val = (vals_dr_scaled[sorted_indices[-1]] + vals_dr_scaled[sorted_indices[-2]]) / 2.0
        ax.axhline(peak_val, color='#7f7f7f', linestyle='--', alpha=0.8, linewidth=1.5, zorder=1)
        
        # Separate branches
        sub_idx = vals_dr_scaled <= peak_val
        sup_idx = vals_dr_scaled >= peak_val
        
        # Subcritical Plot
        ax.errorbar(means_dr[sub_idx], vals_dr_scaled[sub_idx], xerr=sems_dr[sub_idx], fmt='-s', color='#a50f15', markersize=4, capsize=3, linewidth=1.5, zorder=3)
        ax.fill_betweenx(vals_dr_scaled[sub_idx], means_dr[sub_idx] - sems_dr[sub_idx], means_dr[sub_idx] + sems_dr[sub_idx], color='#a50f15', alpha=0.2, zorder=2)
        
        # Supercritical Plot
        ax.errorbar(means_dr[sup_idx], vals_dr_scaled[sup_idx], xerr=sems_dr[sup_idx], fmt='--d', color='#fb6a4a', markersize=4, capsize=3, linewidth=1.5, zorder=3)
        ax.fill_betweenx(vals_dr_scaled[sup_idx], means_dr[sup_idx] - sems_dr[sup_idx], means_dr[sup_idx] + sems_dr[sup_idx], color='#fb6a4a', alpha=0.2, zorder=2)
        
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
    
    output_dir = f"../../../outputs/figures/phase_transition/init{args.init_mass_pct}_limit{args.limit_pct}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_publication_figure(fig, "extinction_time_combined", output_dir=output_dir)
    print(f"Saved combined plot to {output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    plot_combined()
