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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.append(os.path.join(PROJECT_ROOT, "scripts", "plots"))
try:
    from plotting_utils import save_publication_figure
except ImportError:
    def save_publication_figure(fig, filename, output_dir="plots"):
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"), bbox_inches='tight')


# ---------------------------------------------------------------------------
# Data loading: loads individual trial points excluding max-step limit runs
# ---------------------------------------------------------------------------
def load_individual_data(is_liquid: bool, sweep_type: str):
    """Load individual simulation extinction trials (values Δ and extinction times T_e).

    Trials reaching the 1e5 simulation step ceiling (time >= 0.99 * n_steps) are
    filtered out for all models because they did not undergo extinction.
    For the liquid model, non-extinction outcomes (outcome_code != 0) are also excluded.
    """
    steps_str = "" if args.n_steps == 10000 else f"_steps{args.n_steps}"
    model_str = "phase_transition_liquid" if is_liquid else "phase_transition"
    data_dir = os.path.join(
        PROJECT_ROOT,
        "data",
        f"{model_str}{steps_str}_init{args.init_mass_pct}_limit{args.limit_pct}",
        sweep_type,
    )
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        print(f"No .npz files found in {data_dir}")
        return None, None, None, None

    raw_by_val = {}
    indiv_vals = []
    indiv_times = []

    ceiling_threshold = args.n_steps * 0.99

    for f in files:
        try:
            res = np.load(f)
            val = float(np.atleast_1d(res[sweep_type])[0])
            time = float(np.atleast_1d(res["time"])[0])
            # Filter out runs that hit the 1e5 step limit (did not reach extinction)
            if time >= ceiling_threshold:
                continue
            # For the liquid model, keep only true extinction trials (outcome_code == 0)
            if is_liquid:
                code = int(np.atleast_1d(res.get("outcome_code", 0))[0])
                if code != 0:
                    continue
            raw_by_val.setdefault(val, []).append(time)
            indiv_vals.append(val)
            indiv_times.append(time)
        except Exception:
            continue

    if not indiv_vals:
        return None, None, None, None

    vals = np.array(sorted(raw_by_val.keys()))
    means = np.array([np.mean(raw_by_val[v]) for v in vals])
    return vals, means, np.array(indiv_vals), np.array(indiv_times)


# ---------------------------------------------------------------------------
# Plot single panel: fit individual extinction trial points on log-log scale
# ---------------------------------------------------------------------------
def plot_scaling_panel(ax, vals, means, indiv_vals, indiv_times, sweep_name,
                       color, fontsize_labels=15, fontsize_ticks=13):
    scale_factor = 1e3
    vals_scaled = vals * scale_factor

    # Determine critical point as mean of the 2 parameter values with highest mean extinction times
    sorted_idx = np.argsort(means)
    peak_val = (vals_scaled[sorted_idx[-1]] + vals_scaled[sorted_idx[-2]]) / 2.0

    # Convert all individual trial points to distance from critical point
    indiv_vals_scaled = indiv_vals * scale_factor
    dist_all = np.abs(indiv_vals_scaled - peak_val)

    # Exclude points super close to critical point (dist < 1e-12) due to floating point roundoff
    mask = (dist_all > 1e-12) & (indiv_times > 0)

    x_fit = dist_all[mask]
    y_fit = indiv_times[mask]

    # Perform linear regression in log-log space across all individual extinction points
    fit, cov = np.polyfit(np.log10(x_fit), np.log10(y_fit), 1, cov=True)
    slope = fit[0]
    intercept = fit[1]
    slope_err = np.sqrt(cov[0, 0])
    exponent_nu = -slope

    # Calculate R^2 in log-log space
    y_log = np.log10(y_fit)
    y_pred = slope * np.log10(x_fit) + intercept
    ss_res = np.sum((y_log - y_pred) ** 2)
    ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Plot individual extinction points as scatter dots
    ax.scatter(
        x_fit, y_fit, color=color, alpha=0.35, s=14, linewidths=0,
        label=rf'Extinction Trials ($N={len(x_fit)}$)',
    )

    # Plot single linear regression line across the full range of points
    x_line = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 200)
    y_line = 10 ** (slope * np.log10(x_line) + intercept)
    ax.plot(
        x_line, y_line, '-', color='black', linewidth=1.8,
        label=rf'Fit: $\nu = {exponent_nu:.2f}$ ($R^2={r2:.3f}$)',
    )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(
        f"Distance to critical point\n$|{sweep_name} - {sweep_name}_c|$ ($\times 10^{{-3}}$)",
        fontsize=fontsize_labels,
    )
    ax.set_ylabel("Extinction Time $T_e$ (steps)", fontsize=fontsize_labels)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=11, frameon=False, loc='best')
    ax.grid(True, which="both", ls=":", alpha=0.3)


# ---------------------------------------------------------------------------
# Main routine: assemble 2x2 panel figure
# ---------------------------------------------------------------------------
def plot_all_scaling():
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Configuration for each of the 4 panels:
    # (is_liquid, sweep_type, sweep_name, color, ax, title)
    panels = [
        (False, "dmu", r"\Delta \mu", "#08519c", axes[0, 0],
         r"Solid Model: $\Delta \mu$ Critical Slowing Down"),
        (False, "dr", r"\Delta r", "#a50f15", axes[0, 1],
         r"Solid Model: $\Delta r$ Critical Slowing Down"),
        (True, "dmu", r"\Delta \mu", "#08519c", axes[1, 0],
         r"Liquid Model: $\Delta \mu$ Critical Slowing Down (Extinction Only)"),
        (True, "dr", r"\Delta r", "#a50f15", axes[1, 1],
         r"Liquid Model: $\Delta r$ Critical Slowing Down (Extinction Only)"),
    ]

    for is_liq, stype, sname, col, ax, title in panels:
        vals, means, indiv_vals, indiv_times = load_individual_data(is_liq, stype)
        if vals is None:
            ax.set_title(title + " [no data]", fontsize=13, fontweight='bold')
            continue
        plot_scaling_panel(ax, vals, means, indiv_vals, indiv_times, sname, col)
        ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    steps_str = "" if args.n_steps == 10000 else f"_steps{args.n_steps}"
    out_dir = os.path.join(
        PROJECT_ROOT, "outputs", "figures",
        f"scaling_analysis{steps_str}_init{args.init_mass_pct}_limit{args.limit_pct}",
    )
    os.makedirs(out_dir, exist_ok=True)
    save_publication_figure(fig, "critical_slowing_down_scaling", output_dir=out_dir)
    print(f"Saved critical slowing down scaling plot to {out_dir}")


if __name__ == "__main__":
    plot_all_scaling()
