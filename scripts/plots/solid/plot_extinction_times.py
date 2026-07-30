import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--init_mass_pct", type=int, default=10)
parser.add_argument("--limit_pct", type=int, default=40)
parser.add_argument("--n_steps", type=int, default=100000)
args = parser.parse_args()

# Add plotting guidelines skill to path
sys.path.append("/home/francesco/Antigravity/SKILLS/plotting-guidelines/scripts")
try:
    from plotting_utils import save_publication_figure
except ImportError:
    def save_publication_figure(fig, filename, output_dir="plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"), bbox_inches='tight')


def load_data(sweep_type, v_min=None, v_max=None):
    steps_str =f"_steps{args.n_steps}"
    data_dir = f"/home/francesco/Universita/PhD/PROJECTS/CancerEvo/data/phase_transition{steps_str}_init{args.init_mass_pct}_limit{args.limit_pct}/{sweep_type}"
    files = glob.glob(os.path.join(data_dir, "*.npz"))

    if not files:
        print(f"No .npz files found in {data_dir}")
        return None, None, None, None, None

    data = {}
    indiv_vals = []
    indiv_times = []
    ceiling_threshold = args.n_steps * 0.99

    for f in files:
        try:
            res = np.load(f)
            val = float(np.atleast_1d(res[sweep_type])[0])
            if v_min is not None and val < v_min:
                continue
            if v_max is not None and val > v_max:
                continue
            time = float(np.atleast_1d(res["time"])[0])

            # Keep only extinction trials (t < 1e5)
            if time < ceiling_threshold:
                indiv_vals.append(val)
                indiv_times.append(time)

            if val not in data:
                data[val] = []
            data[val].append(time)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not data:
        return None, None, None, None, None

    vals = sorted(list(data.keys()))
    means = []

    for v in vals:
        times = data[v]
        means.append(np.mean(times))

    return np.array(vals), data, np.array(means), np.array(indiv_vals), np.array(indiv_times)


def plot_panel(ax, vals, data_time, means, indiv_v, indiv_t, sweep_name, color,
               fontsize_labels=18, fontsize_ticks=0):
    scale_factor = 1e3
    vals_scaled = vals * scale_factor

    # Vertical line at critical point
    sorted_indices = np.argsort(means)
    peak_val = (vals_scaled[sorted_indices[-1]] + vals_scaled[sorted_indices[-2]]) / 2.0
    ax.axvline(peak_val, color='#7f7f7f', linestyle='--', alpha=0.8, linewidth=1.5, zorder=1)

    # Overlay individual replica scatter points across full domain
    for i, v in enumerate(vals):
        x_val = vals_scaled[i]
        times = data_time[v]
        ax.scatter([x_val] * len(times), times, color=color, s=28, alpha=0.35, zorder=2, edgecolors='none')

    # Fit power-law scaling curve and draw all the way UP TO 1e5 saturation ceiling
    indiv_sc = indiv_v * scale_factor
    x_fit = np.abs(indiv_sc - peak_val)
    y_fit = indiv_t
    mask_fit = x_fit > 1e-12

    if np.sum(mask_fit) >= 4:
        slope, intercept = np.polyfit(np.log10(x_fit[mask_fit]), np.log10(y_fit[mask_fit]), 1)
        x_grid = np.linspace(vals_scaled.min(), vals_scaled.max(), 2000)
        dist_grid = np.abs(x_grid - peak_val)
        y_curve = np.clip(10 ** (slope * np.log10(dist_grid) + intercept), a_min=0, a_max=args.n_steps)

        left_mask = x_grid < peak_val
        right_mask = x_grid > peak_val

        ax.plot(x_grid[left_mask], y_curve[left_mask], '-', color=color, linewidth=1.5, zorder=4)
        ax.plot(x_grid[right_mask], y_curve[right_mask], '-', color=color, linewidth=1.5, zorder=4)

    # Formatting
    ax.set_ylim(-2000, args.n_steps * 1.05)
    ax.set_xlabel(rf"${sweep_name} \ (\times 10^{{-3}})$", fontsize=fontsize_labels)
    ax.set_ylabel("Time to Extinction", fontsize=fontsize_labels)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)


def plot_combined(fontsize_labels=25, fontsize_ticks=20):
    steps_str = "" if args.n_steps == 10000 else f"_steps{args.n_steps}"
    output_dir = f"../../../outputs/figures/phase_transition{steps_str}/init{args.init_mass_pct}_limit{args.limit_pct}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(2, 1, figsize=(7, 9))

    # --- DMU PLOT ---
    vals_dmu, data_time_dmu, means_dmu, indiv_v_dmu, indiv_t_dmu = load_data("dmu", v_min=15e-3, v_max=19e-3)
    if vals_dmu is not None:
        plot_panel(axes[0], vals_dmu, data_time_dmu, means_dmu, indiv_v_dmu, indiv_t_dmu, r"\Delta \mu", "#08519c")

        # Single panel figure (no title, enlarged fonts)
        fig_s, ax_s = plt.subplots(figsize=(6.5, 6.5))
        plot_panel(ax_s, vals_dmu, data_time_dmu, means_dmu, indiv_v_dmu, indiv_t_dmu, r"\Delta \mu", "#08519c", fontsize_labels=fontsize_labels, fontsize_ticks=fontsize_ticks)
        #ax_s.set_yticks([])
        fig_s.tight_layout()
        save_publication_figure(fig_s, "solid_extinction_time_dmu", output_dir=output_dir)
        plt.close(fig_s)

    # --- DR PLOT ---
    vals_dr, data_time_dr, means_dr, indiv_v_dr, indiv_t_dr = load_data("dr", v_min=3e-3, v_max=5e-3)
    if vals_dr is not None:
        plot_panel(axes[1], vals_dr, data_time_dr, means_dr, indiv_v_dr, indiv_t_dr, r"\Delta r", "#a50f15")

        # Single panel figure (no title, enlarged fonts)
        fig_s, ax_s = plt.subplots(figsize=(6.5, 6.5))
        plot_panel(ax_s, vals_dr, data_time_dr, means_dr, indiv_v_dr, indiv_t_dr, r"\Delta r", "#a50f15", fontsize_labels=fontsize_labels, fontsize_ticks=fontsize_ticks)
        #ax_s.set_yticks([])
        fig_s.tight_layout()
        save_publication_figure(fig_s, "solid_extinction_time_dr", output_dir=output_dir)
        plt.close(fig_s)

    # Save combined pair plot
    fig.tight_layout()
    save_publication_figure(fig, "extinction_time_combined", output_dir=output_dir)
    print(f"Saved combined pair plot and single panels to {output_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    plot_combined()
