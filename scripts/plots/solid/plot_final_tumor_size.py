import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

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
    def save_publication_figure(fig, filename, output_dir="plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"), bbox_inches='tight')


def load_data(sweep_type, v_min=None, v_max=None):
    steps_str = "" if args.n_steps == 10000 else f"_steps{args.n_steps}"
    data_dir = f"../../../data/phase_transition{steps_str}_init{args.init_mass_pct}_limit{args.limit_pct}/{sweep_type}"
    files = glob.glob(os.path.join(data_dir, "*.npz"))

    if not files:
        print(f"No .npz files found in {data_dir}")
        return None, None, None, None

    data_size = {}
    data_time = {}
    for f in files:
        try:
            res = np.load(f)
            val = float(np.atleast_1d(res[sweep_type])[0])
            if v_min is not None and val < v_min:
                continue
            if v_max is not None and val > v_max:
                continue
            final_size = float(res["tumor_density"][-1]) * 6400
            time = float(np.atleast_1d(res["time"])[0])
            if val not in data_size:
                data_size[val] = []
                data_time[val] = []
            data_size[val].append(final_size)
            data_time[val].append(time)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not data_size:
        return None, None, None, None

    vals = sorted(list(data_size.keys()))
    means_size = []
    means_time = []

    for v in vals:
        sizes = data_size[v]
        times = data_time[v]
        means_size.append(np.median(sizes))
        means_time.append(np.median(times))

    return np.array(vals), data_size, np.array(means_size), np.array(means_time)


def plot_panel(ax, vals, data_size, means_time, sweep_name, color,
               fontsize_labels=18, fontsize_ticks=15):
    scale_factor = 1e3
    vals_scaled = vals * scale_factor

    # Vertical line at critical point
    sorted_indices = np.argsort(means_time)
    peak_val = (vals_scaled[sorted_indices[-1]] + vals_scaled[sorted_indices[-2]]) / 2.0
    ax.axvline(peak_val, color='#7f7f7f', linestyle='--', alpha=0.8, linewidth=1.5, zorder=1)

    # Plot individual replica scatter points
    for i, v in enumerate(vals):
        x_val = vals_scaled[i]
        sizes = data_size[v]
        ax.scatter([x_val] * len(sizes), sizes, color=color, s=20, alpha=0.35, zorder=2, edgecolors='none')

    # Formatting
    ax.set_ylim(-100, 2750)
    ax.set_xlabel(rf"${sweep_name} \ (\times 10^{{-3}})$", fontsize=fontsize_labels)
    ax.set_ylabel("Equilibrium Tumor Size (cells)", fontsize=fontsize_labels)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)


def plot_combined():
    steps_str = "" if args.n_steps == 10000 else f"_steps{args.n_steps}"
    output_dir = f"../../../outputs/figures/phase_transition{steps_str}/init{args.init_mass_pct}_limit{args.limit_pct}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(2, 1, figsize=(7, 9))

    # --- DMU PLOT ---
    vals_dmu, data_size_dmu, means_size_dmu, means_time_dmu = load_data("dmu", v_min=13e-3, v_max=21e-3)
    if vals_dmu is not None:
        plot_panel(axes[0], vals_dmu, data_size_dmu, means_time_dmu, r"\Delta \mu", "#08519c")

        # Single panel figure (no title, enlarged fonts)
        fig_s, ax_s = plt.subplots(figsize=(6.5, 5.5))
        plot_panel(ax_s, vals_dmu, data_size_dmu, means_time_dmu, r"\Delta \mu", "#08519c", fontsize_labels=20, fontsize_ticks=16)
        fig_s.tight_layout()
        save_publication_figure(fig_s, "solid_final_tumor_size_dmu", output_dir=output_dir)
        plt.close(fig_s)

    # --- DR PLOT ---
    vals_dr, data_size_dr, means_size_dr, means_time_dr = load_data("dr", v_min=1e-3, v_max=7e-3)
    if vals_dr is not None:
        plot_panel(axes[1], vals_dr, data_size_dr, means_time_dr, r"\Delta r", "#a50f15")

        # Single panel figure (no title, enlarged fonts)
        fig_s, ax_s = plt.subplots(figsize=(6.5, 5.5))
        plot_panel(ax_s, vals_dr, data_size_dr, means_time_dr, r"\Delta r", "#a50f15", fontsize_labels=20, fontsize_ticks=16)
        fig_s.tight_layout()
        save_publication_figure(fig_s, "solid_final_tumor_size_dr", output_dir=output_dir)
        plt.close(fig_s)

    # Save combined pair plot
    fig.tight_layout()
    save_publication_figure(fig, "final_tumor_size_combined", output_dir=output_dir)
    print(f"Saved combined pair plot and single panels to {output_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    plot_combined()
