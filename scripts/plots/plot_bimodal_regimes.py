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


def plot_bimodal_regimes():
    steps_str = "" if args.n_steps == 10000 else f"_steps{args.n_steps}"
    data_dir = os.path.join(
        PROJECT_ROOT,
        "data",
        f"phase_transition_liquid{steps_str}_init{args.init_mass_pct}_limit{args.limit_pct}",
        "dmu",
    )
    files = glob.glob(os.path.join(data_dir, "*.npz"))

    if not files:
        print(f"No files found in {data_dir}")
        return

    vals, times, outcomes = [], [], []

    for f in files:
        try:
            res = np.load(f)
            v = float(np.atleast_1d(res["dmu"])[0]) * 1e3  # scale x10^-3
            t = float(np.atleast_1d(res["time"])[0])
            code = int(np.atleast_1d(res["outcome_code"])[0])
            vals.append(v)
            times.append(t)
            outcomes.append(code)
        except Exception:
            continue

    vals = np.array(vals)
    times = np.array(times)
    outcomes = np.array(outcomes)

    # Focus exclusively on the critical window [22.0, 23.5] x 10^-3
    bimodal_left, bimodal_right = 22.0, 23.5
    crit_mask = (vals >= bimodal_left) & (vals <= bimodal_right)

    v_c = vals[crit_mask]
    t_c = times[crit_mask]
    o_c = outcomes[crit_mask]

    # Calculate critical point dc from peak mean extinction time of true extinction runs
    ext_m = o_c == 0
    sat_m = o_c == 1
    max_m = o_c == 2

    raw_by_val = {}
    for v, t in zip(v_c[ext_m], t_c[ext_m]):
        raw_by_val.setdefault(v, []).append(t)
    u_vals = np.array(sorted(raw_by_val.keys()))
    u_means = np.array([np.mean(raw_by_val[v]) for v in u_vals])
    sorted_idx = np.argsort(u_means)
    dc_val = (u_vals[sorted_idx[-1]] + u_vals[sorted_idx[-2]]) / 2.0

    # Subcritical (< dc) and Supercritical (>= dc) masks inside critical window
    sub_m = v_c < dc_val
    sup_m = v_c >= dc_val

    # Distinct color palette for outcomes (ColorBrewer Dark2):
    # Saturation: Burnt Amber (#d95f02), Extinction: Deep Teal (#1b9e77), Max Steps: Slate (#7570b3)
    color_sat = "#d95f02"
    color_ext = "#1b9e77"
    color_max = "#7570b3"

    # Set up 3-panel figure
    fig = plt.figure(figsize=(14, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.35, wspace=0.28)

    fontsize_labels = 14
    fontsize_ticks = 12

    # =========================================================================
    # PANEL A: Critical Window Individual Trial Scatter (ALL Outcomes Included)
    # =========================================================================
    ax1 = fig.add_subplot(gs[:, 0])

    # Highlight background regions before and after critical point
    ax1.axvspan(bimodal_left, dc_val, color="gold", alpha=0.12)
    ax1.axvspan(dc_val, bimodal_right, color="teal", alpha=0.08)

    # Scatter points for ALL outcomes
    ax1.scatter(
        v_c[sat_m], t_c[sat_m], color=color_sat, s=22, alpha=0.6,
        linewidths=0, label=f"Tumor Saturation ($N=2560$, n={np.sum(sat_m)})",
    )
    ax1.scatter(
        v_c[ext_m], t_c[ext_m], color=color_ext, s=22, alpha=0.6,
        linewidths=0, label=f"True Extinction ($N=0$, n={np.sum(ext_m)})",
    )
    ax1.scatter(
        v_c[max_m], t_c[max_m], color=color_max, s=18, alpha=0.5,
        linewidths=0, label=f"Max Steps Ceiling ($T = 10^5$, n={np.sum(max_m)})",
    )

    # Vertical line marking dc
    ax1.axvline(dc_val, color="black", linestyle="--", linewidth=1.5, alpha=0.8)

    ax1.set_yscale("log")
    ax1.set_xlim(bimodal_left - 0.05, bimodal_right + 0.05)
    ax1.set_xlabel(r"Liquid $\Delta \mu$ ($\times 10^{-3}$)", fontsize=fontsize_labels)
    ax1.set_ylabel("Simulation Time $T$ (steps)", fontsize=fontsize_labels)
    ax1.set_title(r"A. Critical Window ($\Delta \mu \in [22.0, 23.5]$) Outcomes", fontsize=14, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=fontsize_ticks, direction="out")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, which="both", ls=":", alpha=0.3)

    # Common binning for log-times (from 1.5 to 5.05 to capture T=10^5)
    bins = np.linspace(1.5, 5.05, 36)

    # =========================================================================
    # PANEL B: Histogram BEFORE dc (Bimodal: 2 Peaks)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.hist(
        np.log10(t_c[sub_m & sat_m]), bins=bins, color=color_sat, alpha=0.65,
        edgecolor="none", label="Fast Saturation Peak ($T \sim 10^2$)",
    )
    ax2.hist(
        np.log10(t_c[sub_m & ext_m]), bins=bins, color=color_ext, alpha=0.65,
        edgecolor="none", label="True Extinction",
    )
    ax2.hist(
        np.log10(t_c[sub_m & max_m]), bins=bins, color=color_max, alpha=0.65,
        edgecolor="none", label="Max Steps Ceiling ($T = 10^5$)",
    )

    ax2.set_title(rf"B. Before $\Delta \mu_c$ ($\Delta \mu < {dc_val:.2f}$): Bimodal (2 Peaks)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=fontsize_labels)
    ax2.tick_params(axis="both", labelsize=fontsize_ticks, direction="out")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, ls=":", alpha=0.3)

    # =========================================================================
    # PANEL C: Histogram AFTER dc (Smoother / Unimodal Extinction)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)

    ax3.hist(
        np.log10(t_c[sup_m & sat_m]), bins=bins, color=color_sat, alpha=0.65,
        edgecolor="none", label="Saturation Trials",
    )
    ax3.hist(
        np.log10(t_c[sup_m & ext_m]), bins=bins, color=color_ext, alpha=0.65,
        edgecolor="none", label="Extinction Trials",
    )
    ax3.hist(
        np.log10(t_c[sup_m & max_m]), bins=bins, color=color_max, alpha=0.65,
        edgecolor="none", label="Max Steps Ceiling",
    )

    ax3.set_title(rf"C. After $\Delta \mu_c$ ($\Delta \mu \geq {dc_val:.2f}$): Smoother Distribution", fontsize=13, fontweight="bold")
    ax3.set_xlabel(r"$\log_{10}(\mathrm{Simulation\ Time}\ T)$", fontsize=fontsize_labels)
    ax3.set_ylabel("Frequency", fontsize=fontsize_labels)
    ax3.tick_params(axis="both", labelsize=fontsize_ticks, direction="out")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(True, ls=":", alpha=0.3)

    plt.tight_layout()

    out_dir = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "figures",
        f"scaling_analysis{steps_str}_init{args.init_mass_pct}_limit{args.limit_pct}",
    )
    os.makedirs(out_dir, exist_ok=True)

    save_publication_figure(fig, "liquid_dmu_bimodal_regimes", output_dir=out_dir)
    print(f"Saved bimodal diagnostic figure to {out_dir}")


if __name__ == "__main__":
    plot_bimodal_regimes()
