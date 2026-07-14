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
    data_dir = f"../../../data/phase_transition_liquid/{sweep_type}"
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
        
        sorted_indices = np.argsort(means_dmu)
        peak_val = (vals_dmu_scaled[sorted_indices[-1]] + vals_dmu_scaled[sorted_indices[-2]]) / 2.0
        
        # Absolute distance to critical point
        dist_dmu = np.abs(vals_dmu_scaled - peak_val)
        
        # Separate branches (excluding points very close to the peak due to noise/finite-size effects)
        sub_idx = (vals_dmu_scaled < peak_val) & (dist_dmu > 0.8)
        sup_idx = (vals_dmu_scaled > peak_val) & (dist_dmu > 0.8)
        
        x_sub, y_sub = dist_dmu[sub_idx], means_dmu[sub_idx]
        x_sup, y_sup = dist_dmu[sup_idx], means_dmu[sup_idx]
        
        # Fit log-log to find the exponent and its error
        fit_sub, cov_sub = np.polyfit(np.log10(x_sub), np.log10(y_sub), 1, cov=True)
        slope_sub = fit_sub[0]
        err_sub = np.sqrt(cov_sub[0, 0])
        
        fit_sup, cov_sup = np.polyfit(np.log10(x_sup), np.log10(y_sup), 1, cov=True)
        slope_sup = fit_sup[0]
        err_sup = np.sqrt(cov_sup[0, 0])
        
        # Plot data
        ax.errorbar(x_sub, y_sub, yerr=sems_dmu[sub_idx], fmt='-o', color='#08519c', markersize=5, capsize=3, linewidth=1.5, label=rf'Subcritical ($\nu \approx {slope_sub:.2f} \pm {err_sub:.2f}$)')
        ax.errorbar(x_sup, y_sup, yerr=sems_dmu[sup_idx], fmt='--^', color='#6baed6', markersize=5, capsize=3, linewidth=1.5, label=rf'Supercritical ($\nu \approx {slope_sup:.2f} \pm {err_sup:.2f}$)')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Formatting
        ax.set_xlabel("Distance to critical point\n" + r"$|\Delta \mu - \Delta \mu_c|$ ($\times 10^{-3}$)", fontsize=fontsize_labels)
        ax.set_ylabel("Mean Simulation Time (steps)", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=15, frameon=False)

    # --- DR PLOT ---
    vals_dr, means_dr, sems_dr = load_data("dr")
    if vals_dr is not None:
        ax = axes[1]
        
        # Scale vals_dr by 10^3 to avoid messy scientific notation on ticks
        scale_factor = 1e3
        vals_dr_scaled = vals_dr * scale_factor
        
        sorted_indices = np.argsort(means_dr)
        peak_val = (vals_dr_scaled[sorted_indices[-1]] + vals_dr_scaled[sorted_indices[-2]]) / 2.0
        
        # Absolute distance to critical point
        dist_dr = np.abs(vals_dr_scaled - peak_val)
        
        # Separate branches (excluding points very close to the peak due to noise/finite-size effects)
        sub_idx = (vals_dr_scaled < peak_val) & (dist_dr > 0.2)
        sup_idx = (vals_dr_scaled > peak_val) & (dist_dr > 0.2)
        
        x_sub, y_sub = dist_dr[sub_idx], means_dr[sub_idx]
        x_sup, y_sup = dist_dr[sup_idx], means_dr[sup_idx]
        
        # Fit log-log to find the exponent and its error
        fit_sub, cov_sub = np.polyfit(np.log10(x_sub), np.log10(y_sub), 1, cov=True)
        slope_sub = fit_sub[0]
        err_sub = np.sqrt(cov_sub[0, 0])
        
        fit_sup, cov_sup = np.polyfit(np.log10(x_sup), np.log10(y_sup), 1, cov=True)
        slope_sup = fit_sup[0]
        err_sup = np.sqrt(cov_sup[0, 0])
        
        # Plot data
        ax.errorbar(x_sub, y_sub, yerr=sems_dr[sub_idx], fmt='-s', color='#a50f15', markersize=5, capsize=3, linewidth=1.5, label=rf'Subcritical ($\nu \approx {slope_sub:.2f} \pm {err_sub:.2f}$)')
        ax.errorbar(x_sup, y_sup, yerr=sems_dr[sup_idx], fmt='--d', color='#fb6a4a', markersize=5, capsize=3, linewidth=1.5, label=rf'Supercritical ($\nu \approx {slope_sup:.2f} \pm {err_sup:.2f}$)')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Formatting
        ax.set_xlabel("Distance to critical point\n" + r"$|\Delta r - \Delta r_c|$ ($\times 10^{-3}$)", fontsize=fontsize_labels)
        ax.set_ylabel("Mean Simulation Time (steps)", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=15, frameon=False)

    # Add subplot letters (g, h)
    #axes[0].text(-0.22, 1.15, "g", transform=axes[0].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    #axes[1].text(-0.22, 1.15, "h", transform=axes[1].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    
    output_dir = "../../../outputs/figures/phase_transition_liquid"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_publication_figure(fig, "extinction_time_scaling_combined_liquid", output_dir=output_dir)
    print(f"Saved combined scaling plot to {output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    plot_combined()
