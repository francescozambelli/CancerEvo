"""
plot_figure_article_bifurcations.py
----------------------------------
Generates the concluding multi-panel figure for the manuscript comparing phase transitions,
critical slowing down (extinction times), and dynamical scenarios (time series insets)
in Solid and Liquid tumors.

Layout:
- Left Column: Solid Tumors (Top: Δμ, Bottom: Δr)
- Right Column: Liquid Tumors (Top: Δμ with Bistability, Bottom: Δr)
- Extinction time insets: linear-linear plots showing critical slowing down peaks (no fitted exponents).
- Time series insets: representative trajectories showing before/after transition, long transients,
  and coexisting persistent/extinction branches in the liquid bistability region.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure clean styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 18,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Directories
SOLID_DMU_DIR = os.path.join(PROJECT_ROOT, "data", "phase_transition_steps100000_init10_limit40", "dmu")
SOLID_DR_DIR  = os.path.join(PROJECT_ROOT, "data", "phase_transition_steps100000_init10_limit40", "dr")
LIQUID_DMU_DIR = os.path.join(PROJECT_ROOT, "data", "phase_transition_liquid_steps100000_init10_limit40", "dmu")
LIQUID_DR_DIR  = os.path.join(PROJECT_ROOT, "data", "phase_transition_liquid_steps100000_init10_limit40", "dr")


def load_bifurcation_data(data_dir, param_key, scale=1e3, v_min=None, v_max=None):
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    data_size = {}
    data_time = {}
    extinct_data = {}

    for f in files:
        try:
            res = np.load(f)
            val = float(np.atleast_1d(res[param_key])[0])
            if v_min is not None and val < v_min:
                continue
            if v_max is not None and val > v_max:
                continue

            scaled_val = val * scale
            final_size = float(res["tumor_density"][-1]) * 6400.0
            time = float(np.atleast_1d(res["time"])[0])
            code = int(np.atleast_1d(res["outcome_code"])[0])

            if scaled_val not in data_size:
                data_size[scaled_val] = []
                data_time[scaled_val] = []
                extinct_data[scaled_val] = []

            data_size[scaled_val].append(final_size)
            data_time[scaled_val].append(time)

            # Extinction time for runs that extincted (code 0 or time < 99000)
            if code == 0 or time < 99000:
                extinct_data[scaled_val].append(time)

        except Exception:
            continue

    vals = np.array(sorted(data_size.keys()))
    means_time = np.array([np.mean(data_time[v]) for v in vals])

    # Find critical threshold as peak of mean extinction time
    sorted_idx = np.argsort(means_time)
    peak_val = (vals[sorted_idx[-1]] + vals[sorted_idx[-2]]) / 2.0 if len(vals) > 1 else vals[0]

    return vals, data_size, data_time, extinct_data, peak_val


def add_extinction_inset(ax_main, vals, extinct_data, peak_val, color, inset_rect=[0.16, 0.18, 0.32, 0.35]):
    """Adds a small inset for Time to Extinction vs parameter (linear scale, no exponent text)."""
    ax_ins = ax_main.inset_axes(inset_rect)

    x_points = []
    y_points = []
    for v in vals:
        if v in extinct_data and len(extinct_data[v]) > 0:
            for t in extinct_data[v]:
                x_points.append(v)
                y_points.append(t)

    x_points = np.array(x_points)
    y_points = np.array(y_points)

    if len(x_points) > 0:
        ax_ins.scatter(x_points, y_points, color=color, s=4, alpha=0.35, edgecolors="none")

        # Power-law scaling interpolation curve (like in plot_critical_scaling.py)
        dist_all = np.abs(x_points - peak_val)
        mask_fit = (dist_all > 1e-6) & (y_points > 0)

        if np.sum(mask_fit) >= 4:
            fit = np.polyfit(np.log10(dist_all[mask_fit]), np.log10(y_points[mask_fit]), 1)
            slope, intercept = fit[0], fit[1]

            x_grid = np.linspace(vals.min(), vals.max(), 1000)
            dist_grid = np.abs(x_grid - peak_val)
            mask_valid = dist_grid > 1e-6

            y_curve = np.zeros_like(x_grid)
            y_curve[mask_valid] = 10 ** (slope * np.log10(dist_grid[mask_valid]) + intercept)
            y_curve = np.clip(y_curve, a_min=0, a_max=100000)

            left_mask = x_grid < (peak_val - 1e-4)
            right_mask = x_grid > (peak_val + 1e-4)

            if np.any(left_mask):
                ax_ins.plot(x_grid[left_mask], y_curve[left_mask], '-', color=color, lw=1.2, alpha=0.85)
            if np.any(right_mask):
                ax_ins.plot(x_grid[right_mask], y_curve[right_mask], '-', color=color, lw=1.2, alpha=0.85)
        else:
            unique_x = np.array(sorted(list(set(x_points))))
            mean_y = np.array([np.mean(y_points[x_points == ux]) for ux in unique_x])
            ax_ins.plot(unique_x, mean_y, color=color, lw=1.2, alpha=0.85)

    ax_ins.axvline(peak_val, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.8)

    ax_ins.set_ylabel("Time to extinction", fontsize=9, labelpad=2)
    ax_ins.tick_params(axis="both", labelsize=8)
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)

    return ax_ins


def main():
    fig = plt.figure(figsize=(17, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

    ax_solid_dmu = fig.add_subplot(gs[0, 0])
    ax_solid_dr  = fig.add_subplot(gs[1, 0])
    ax_liquid_dmu = fig.add_subplot(gs[0, 1])
    ax_liquid_dr  = fig.add_subplot(gs[1, 1])

    color_blue = "#08519c"
    color_red  = "#a50f15"

    # =========================================================================
    # PANEL 1: SOLID Δμ
    # =========================================================================
    vals_s_dmu, size_s_dmu, time_s_dmu, ext_s_dmu, peak_s_dmu = load_bifurcation_data(
        SOLID_DMU_DIR, "dmu", scale=1e3, v_min=13e-3, v_max=21e-3
    )

    for v in vals_s_dmu:
        sizes = size_s_dmu[v]
        ax_solid_dmu.scatter([v] * len(sizes), sizes, color=color_blue, s=18, alpha=0.35, edgecolors="none")

    ax_solid_dmu.axvline(peak_s_dmu, color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_solid_dmu.set_xlim(13.0, 21.0)
    ax_solid_dmu.set_ylim(-100, 2800)
    ax_solid_dmu.set_xlabel(r"$\Delta \mu \ (\times 10^{-3})$")
    ax_solid_dmu.set_ylabel("Equilibrium Tumor Size (cells)")
    ax_solid_dmu.set_title("Solid", fontsize=16, fontweight="bold", loc="left", pad=12)

    # Inset extinction time
    ins_s_dmu = add_extinction_inset(ax_solid_dmu, vals_s_dmu, ext_s_dmu, peak_s_dmu, color_blue, inset_rect=[0.16, 0.18, 0.30, 0.35])
    ins_s_dmu.set_xlabel(r"$\Delta \mu \ (\times 10^{-3})$", fontsize=8)

    # Trajectory 1: Before transition (Δμ = 15.0)
    f_s1 = os.path.join(SOLID_DMU_DIR, "dmu_0.015_rep_1.npz")
    d_s1 = np.load(f_s1)
    td_s1 = d_s1["tumor_density"] * 6400.0

    ax_ts_s1 = ax_solid_dmu.inset_axes([0.08, 0.72, 0.32, 0.23])
    ax_ts_s1.plot(td_s1, color=color_blue, lw=1.2)
    ax_ts_s1.set_ylabel("Tumor density", fontsize=8, labelpad=1)
    ax_ts_s1.set_xlabel("Time steps", fontsize=8, labelpad=1)
    ax_ts_s1.tick_params(labelsize=7)
    ax_ts_s1.spines["top"].set_visible(False)
    ax_ts_s1.spines["right"].set_visible(False)

    ax_solid_dmu.annotate(
        "", xy=(15.0, 2560), xytext=(15.0, 2100),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.2)
    )

    # Trajectory 2: Transient cloud hitting max steps limit (Δμ = 17.17 x 10^-3, 100,000 steps)
    f_s2 = os.path.join(SOLID_DMU_DIR, "dmu_0.01717391304347826_rep_9.npz")
    d_s2 = np.load(f_s2)
    td_s2 = d_s2["tumor_density"] * 6400.0

    ax_ts_s2 = ax_solid_dmu.inset_axes([0.48, 0.58, 0.28, 0.23])
    ax_ts_s2.plot(td_s2, color=color_blue, lw=1.2)
    ax_ts_s2.set_ylabel("Tumor density", fontsize=8, labelpad=1)
    ax_ts_s2.set_xlabel("Time steps", fontsize=8, labelpad=1)
    ax_ts_s2.tick_params(labelsize=7)
    ax_ts_s2.spines["top"].set_visible(False)
    ax_ts_s2.spines["right"].set_visible(False)

    ax_solid_dmu.annotate(
        "", xy=(17.2, 560), xytext=(16.3, 1550),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.2)
    )

    # Trajectory 3: After transition (Δμ = 18.04)
    f_s3 = os.path.join(SOLID_DMU_DIR, "dmu_0.018043478260869564_rep_1.npz")
    d_s3 = np.load(f_s3)
    td_s3 = d_s3["tumor_density"] * 6400.0

    ax_ts_s3 = ax_solid_dmu.inset_axes([0.62, 0.20, 0.28, 0.23])
    ax_ts_s3.plot(td_s3, color=color_blue, lw=1.2)
    ax_ts_s3.set_ylabel("Tumor density", fontsize=8, labelpad=1)
    ax_ts_s3.set_xlabel("Time steps", fontsize=8, labelpad=1)
    ax_ts_s3.tick_params(labelsize=7)
    ax_ts_s3.spines["top"].set_visible(False)
    ax_ts_s3.spines["right"].set_visible(False)

    ax_solid_dmu.annotate(
        "", xy=(18.2, 50), xytext=(17.4, 500),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.2)
    )


    # =========================================================================
    # PANEL 2: SOLID Δr
    # =========================================================================
    vals_s_dr, size_s_dr, time_s_dr, ext_s_dr, peak_s_dr = load_bifurcation_data(
        SOLID_DR_DIR, "dr", scale=1e3, v_min=1e-3, v_max=7e-3
    )

    for v in vals_s_dr:
        sizes = size_s_dr[v]
        ax_solid_dr.scatter([v] * len(sizes), sizes, color=color_red, s=18, alpha=0.35, edgecolors="none")

    ax_solid_dr.axvline(peak_s_dr, color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_solid_dr.set_xlim(0.8, 7.2)
    ax_solid_dr.set_ylim(-100, 2800)
    ax_solid_dr.set_xlabel(r"$\Delta r \ (\times 10^{-3})$")
    ax_solid_dr.set_ylabel("Equilibrium Tumor Size (cells)")

    ins_s_dr = add_extinction_inset(ax_solid_dr, vals_s_dr, ext_s_dr, peak_s_dr, color_red, inset_rect=[0.55, 0.18, 0.32, 0.38])
    ins_s_dr.set_xlabel(r"$\Delta r \ (\times 10^{-3})$", fontsize=8)


    # =========================================================================
    # PANEL 3: LIQUID Δμ (WITH BISTABILITY & TRANSIENTS)
    # =========================================================================
    vals_l_dmu, size_l_dmu, time_l_dmu, ext_l_dmu, peak_l_dmu = load_bifurcation_data(
        LIQUID_DMU_DIR, "dmu", scale=1e3, v_min=19e-3, v_max=27e-3
    )

    # Shaded bistability / critical window
    ax_liquid_dmu.axvspan(22.7, 23.7, color="#4292c6", alpha=0.20, label="Bistability Region")

    for v in vals_l_dmu:
        sizes = size_l_dmu[v]
        ax_liquid_dmu.scatter([v] * len(sizes), sizes, color=color_blue, s=18, alpha=0.35, edgecolors="none")

    ax_liquid_dmu.axvline(peak_l_dmu, color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_liquid_dmu.set_xlim(19.0, 27.0)
    ax_liquid_dmu.set_ylim(-100, 2800)
    ax_liquid_dmu.set_xlabel(r"$\Delta \mu \ (\times 10^{-3})$")
    ax_liquid_dmu.set_ylabel("Equilibrium Tumor Size (cells)")
    ax_liquid_dmu.set_title("Liquid", fontsize=16, fontweight="bold", loc="left", pad=12)

    # Inset extinction time
    ins_l_dmu = add_extinction_inset(ax_liquid_dmu, vals_l_dmu, ext_l_dmu, peak_l_dmu, color_blue, inset_rect=[0.08, 0.18, 0.28, 0.35])
    ins_l_dmu.set_xlabel(r"$\Delta \mu \ (\times 10^{-3})$", fontsize=8)

    # Trajectory Inset 1: Bistability (Δμ = 23.2 x 10^-3) - BOTH persistence & extinction
    f_b_pers = os.path.join(LIQUID_DMU_DIR, "dmu_fine_0.0232_rep_9.npz")
    f_b_ext  = os.path.join(LIQUID_DMU_DIR, "dmu_fine_0.0232_rep_7.npz")
    td_b_pers = np.load(f_b_pers)["tumor_density"] * 6400.0
    td_b_ext  = np.load(f_b_ext)["tumor_density"] * 6400.0

    ax_ts_bistable = ax_liquid_dmu.inset_axes([0.64, 0.72, 0.33, 0.24])
    ax_ts_bistable.plot(td_b_pers, color=color_blue, lw=1.3, label="Persistence")
    ax_ts_bistable.plot(td_b_ext, color="#525252", lw=1.2, ls="--", label="Extinction")
    ax_ts_bistable.set_ylabel("Tumor density", fontsize=7.5, labelpad=1)
    ax_ts_bistable.set_xlabel("Time steps", fontsize=7.5, labelpad=1)
    ax_ts_bistable.set_title("Bistability ??", fontsize=9.5, fontweight="bold", pad=2)
    ax_ts_bistable.tick_params(labelsize=6.5)
    ax_ts_bistable.spines["top"].set_visible(False)
    ax_ts_bistable.spines["right"].set_visible(False)
    ax_ts_bistable.legend(fontsize=6.5, loc="center right", frameon=True, framealpha=0.6)

    ax_liquid_dmu.annotate(
        "", xy=(23.2, 2570), xytext=(24.9, 2350),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.1)
    )
    ax_liquid_dmu.annotate(
        "", xy=(23.2, 0), xytext=(24.9, 2050),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.1)
    )

    # Trajectory Inset 2: Transient cloud (Δμ = 23.12 x 10^-3)
    f_cloud = os.path.join(LIQUID_DMU_DIR, "dmu_0.02311594202898551_rep_9.npz")
    td_cloud = np.load(f_cloud)["tumor_density"] * 6400.0

    ax_ts_cloud = ax_liquid_dmu.inset_axes([0.64, 0.39, 0.33, 0.22])
    ax_ts_cloud.plot(td_cloud, color=color_blue, lw=1.2)
    ax_ts_cloud.set_ylabel("Tumor density", fontsize=7.5, labelpad=1)
    ax_ts_cloud.set_xlabel("Time steps", fontsize=7.5, labelpad=1)
    ax_ts_cloud.set_title("Extinction?", fontsize=9.5, fontweight="bold", pad=2)
    ax_ts_cloud.tick_params(labelsize=6.5)
    ax_ts_cloud.spines["top"].set_visible(False)
    ax_ts_cloud.spines["right"].set_visible(False)

    ax_liquid_dmu.annotate(
        "", xy=(23.0, 950), xytext=(24.9, 1250),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.1)
    )

    # Trajectory Inset 3: Post-transition Extinction (Δμ = 24.57 x 10^-3)
    f_l_post = os.path.join(LIQUID_DMU_DIR, "dmu_0.024565217391304347_rep_1.npz")
    td_l_post = np.load(f_l_post)["tumor_density"] * 6400.0

    ax_ts_lpost = ax_liquid_dmu.inset_axes([0.64, 0.06, 0.33, 0.22])
    ax_ts_lpost.plot(td_l_post, color=color_blue, lw=1.2)
    ax_ts_lpost.set_ylabel("Tumor density", fontsize=7.5, labelpad=1)
    ax_ts_lpost.set_xlabel("Time steps", fontsize=7.5, labelpad=1)
    ax_ts_lpost.tick_params(labelsize=6.5)
    ax_ts_lpost.spines["top"].set_visible(False)
    ax_ts_lpost.spines["right"].set_visible(False)

    ax_liquid_dmu.annotate(
        "", xy=(24.6, 0), xytext=(24.9, 450),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.1)
    )


    # =========================================================================
    # PANEL 4: LIQUID Δr
    # =========================================================================
    vals_l_dr, size_l_dr, time_l_dr, ext_l_dr, peak_l_dr = load_bifurcation_data(
        LIQUID_DR_DIR, "dr", scale=1e3, v_min=1.0e-3, v_max=4.0e-3
    )

    for v in vals_l_dr:
        sizes = size_l_dr[v]
        ax_liquid_dr.scatter([v] * len(sizes), sizes, color=color_red, s=18, alpha=0.35, edgecolors="none")

    ax_liquid_dr.axvline(peak_l_dr, color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_liquid_dr.set_xlim(0.9, 4.1)
    ax_liquid_dr.set_ylim(-100, 2800)
    ax_liquid_dr.set_xlabel(r"$\Delta r \ (\times 10^{-3})$")
    ax_liquid_dr.set_ylabel("Equilibrium Tumor Size (cells)")

    ins_l_dr = add_extinction_inset(ax_liquid_dr, vals_l_dr, ext_l_dr, peak_l_dr, color_red, inset_rect=[0.55, 0.18, 0.32, 0.38])
    ins_l_dr.set_xlabel(r"$\Delta r \ (\times 10^{-3})$", fontsize=8)


    # Save figure outputs
    paper_dir = os.path.join(PROJECT_ROOT, "outputs", "paper_figures")
    fig_dir   = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(paper_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    fig.tight_layout()

    for d in [paper_dir, fig_dir, os.path.join(fig_dir, "transition_final")]:
        os.makedirs(d, exist_ok=True)
        pdf_path = os.path.join(d, "Fig_Transition_Analysis.pdf")
        png_path = os.path.join(d, "Fig_Transition_Analysis.png")
        svg_path = os.path.join(d, "Fig_Transition_Analysis.svg")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"Saved figure: {pdf_path}")
        print(f"Saved figure: {png_path}")
        print(f"Saved figure: {svg_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
