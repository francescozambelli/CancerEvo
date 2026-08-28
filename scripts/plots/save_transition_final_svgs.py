"""
save_transition_final_svgs.py
------------------------------
Generates and saves standalone SVG files for every main panel, extinction time inset,
and representative trajectory time series used in the transition analysis figure into
outputs/figures/transition_final/.

Centralized Font Size Controls:
- Change FS_MAIN_LABEL, FS_MAIN_TITLE, FS_MAIN_TICK, FS_INSET_LABEL, etc. at the top
  of this file to dynamically update font sizes across ALL generated SVG files!
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# CENTRALIZED FONT SIZE CONFIGURATION
# =============================================================================
FS_MAIN_LABEL  = 20  # Main panel axis labels (Equilibrium Tumor Size, Δμ, Δr)
FS_MAIN_TITLE  = 0  # Main panel titles
FS_MAIN_TICK   = 15  # Main panel tick labels

FS_INSET_LABEL = 20  # Inset plot axis labels (Time to extinction, Tumor density, Time steps)
FS_INSET_TITLE = 0  # Inset titles (Bistability ??, Extinction?)
FS_INSET_TICK  = 18  # Inset tick labels
FS_LEGEND      = 18  # Legend text size

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": FS_MAIN_LABEL,
    "axes.titlesize": FS_MAIN_TITLE,
    "xtick.labelsize": FS_MAIN_TICK,
    "ytick.labelsize": FS_MAIN_TICK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Output directory
FINAL_SVG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures", "transition_final")
os.makedirs(FINAL_SVG_DIR, exist_ok=True)

# Data Directories
SOLID_DMU_DIR  = os.path.join(PROJECT_ROOT, "data", "phase_transition_steps100000_init10_limit40", "dmu")
SOLID_DR_DIR   = os.path.join(PROJECT_ROOT, "data", "phase_transition_steps100000_init10_limit40", "dr")
LIQUID_DMU_DIR = os.path.join(PROJECT_ROOT, "data", "phase_transition_liquid_steps100000_init10_limit40", "dmu")
LIQUID_DR_DIR  = os.path.join(PROJECT_ROOT, "data", "phase_transition_liquid_steps100000_init10_limit40", "dr")

COLOR_BLUE = "#08519c"
COLOR_RED  = "#a50f15"


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

            if code == 0 or time < 99000:
                extinct_data[scaled_val].append(time)

        except Exception:
            continue

    vals = np.array(sorted(data_size.keys()))
    means_time = np.array([np.mean(data_time[v]) for v in vals])
    sorted_idx = np.argsort(means_time)
    peak_val = (vals[sorted_idx[-1]] + vals[sorted_idx[-2]]) / 2.0 if len(vals) > 1 else vals[0]

    return vals, data_size, data_time, extinct_data, peak_val


def save_extinction_time_svg(filename, vals, extinct_data, peak_val, color, xlabel):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))

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
        ax.scatter(x_points, y_points, color=color, s=10, alpha=0.35, edgecolors="none")

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
                ax.plot(x_grid[left_mask], y_curve[left_mask], '-', color=color, lw=1.8, alpha=0.9)
            if np.any(right_mask):
                ax.plot(x_grid[right_mask], y_curve[right_mask], '-', color=color, lw=1.8, alpha=0.9)
        else:
            unique_x = np.array(sorted(list(set(x_points))))
            mean_y = np.array([np.mean(y_points[x_points == ux]) for ux in unique_x])
            ax.plot(unique_x, mean_y, color=color, lw=1.8, alpha=0.9)

    ax.axvline(peak_val, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_ylabel("Time to extinction", fontsize=FS_INSET_LABEL)
    ax.set_xlabel(xlabel, fontsize=FS_INSET_LABEL)
    ax.tick_params(axis="both", labelsize=FS_INSET_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = os.path.join(FINAL_SVG_DIR, filename)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_single_trajectory_svg(filename, time_series_dict, title=None, ylabel="Tumor density", xlabel="Time steps"):
    fig, ax = plt.subplots(figsize=(8, 3.5))

    for label, (td, color, ls) in time_series_dict.items():
        ax.plot(td, color=color, lw=1.5, ls=ls, label=label)

    ax.set_ylabel(ylabel, fontsize=FS_INSET_LABEL)
    ax.set_xlabel(xlabel, fontsize=FS_INSET_LABEL)
    ax.tick_params(axis="both", labelsize=FS_INSET_TICK)
    if title:
        ax.set_title(title, fontsize=FS_INSET_TITLE, fontweight="bold")
    if len(time_series_dict) > 1:
        ax.legend(fontsize=FS_LEGEND, loc="best", frameon=True, framealpha=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = os.path.join(FINAL_SVG_DIR, filename)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def export_all_svgs():
    print(f"Saving all SVG panels to: {FINAL_SVG_DIR}")

    # Load all bifurcation data
    vals_s_dmu, size_s_dmu, time_s_dmu, ext_s_dmu, peak_s_dmu = load_bifurcation_data(
        SOLID_DMU_DIR, "dmu", scale=1e3, v_min=13e-3, v_max=21e-3
    )
    vals_s_dr, size_s_dr, time_s_dr, ext_s_dr, peak_s_dr = load_bifurcation_data(
        SOLID_DR_DIR, "dr", scale=1e3, v_min=1e-3, v_max=7e-3
    )
    vals_l_dmu, size_l_dmu, time_l_dmu, ext_l_dmu, peak_l_dmu = load_bifurcation_data(
        LIQUID_DMU_DIR, "dmu", scale=1e3, v_min=19e-3, v_max=27e-3
    )
    vals_l_dr, size_l_dr, time_l_dr, ext_l_dr, peak_l_dr = load_bifurcation_data(
        LIQUID_DR_DIR, "dr", scale=1e3, v_min=1.0e-3, v_max=4.0e-3
    )

    # 1. Extinction Time Standalone SVGs
    save_extinction_time_svg("solid_dmu_extinction_time.svg", vals_s_dmu, ext_s_dmu, peak_s_dmu, COLOR_BLUE, r"$\Delta \mu \ (\times 10^{-3})$")
    save_extinction_time_svg("solid_dr_extinction_time.svg", vals_s_dr, ext_s_dr, peak_s_dr, COLOR_RED, r"$\Delta r \ (\times 10^{-3})$")
    save_extinction_time_svg("liquid_dmu_extinction_time.svg", vals_l_dmu, ext_l_dmu, peak_l_dmu, COLOR_BLUE, r"$\Delta \mu \ (\times 10^{-3})$")
    save_extinction_time_svg("liquid_dr_extinction_time.svg", vals_l_dr, ext_l_dr, peak_l_dr, COLOR_RED, r"$\Delta r \ (\times 10^{-3})$")

    # 2. Time Series Standalone SVGs
    # Solid trajectories
    f_s1 = os.path.join(SOLID_DMU_DIR, "dmu_0.015_rep_1.npz")
    td_s1 = np.load(f_s1)["tumor_density"] * 6400.0
    save_single_trajectory_svg("solid_dmu_ts_before.svg", {"Active Tumor": (td_s1, COLOR_BLUE, "-")})

    f_s2 = os.path.join(SOLID_DMU_DIR, "dmu_0.01717391304347826_rep_9.npz")
    td_s2 = np.load(f_s2)["tumor_density"] * 6400.0
    save_single_trajectory_svg("solid_dmu_ts_transient.svg", {"Long Transient": (td_s2, COLOR_BLUE, "-")})

    f_s3 = os.path.join(SOLID_DMU_DIR, "dmu_0.018043478260869564_rep_1.npz")
    td_s3 = np.load(f_s3)["tumor_density"] * 6400.0
    save_single_trajectory_svg("solid_dmu_ts_extinction.svg", {"Extinction": (td_s3, COLOR_BLUE, "-")})

    # Liquid trajectories
    f_b_pers = os.path.join(LIQUID_DMU_DIR, "dmu_fine_0.02285_rep_1.npz")
    f_b_ext  = os.path.join(LIQUID_DMU_DIR, "dmu_fine_0.02285_rep_10.npz")
    td_b_pers = np.load(f_b_pers)["tumor_density"] * 6400.0
    td_b_ext  = np.load(f_b_ext)["tumor_density"] * 6400.0
    save_single_trajectory_svg(
        "liquid_dmu_ts_bistable.svg",
        {"Persistence": (td_b_pers, COLOR_BLUE, "-"), "Extinction": (td_b_ext, "#525252", "--")},
        title="Bistability ??"
    )


    f_l1 = os.path.join(LIQUID_DMU_DIR, "dmu_0.022572463768115943_rep_1.npz")
    td_l1 = np.load(f_l1)["tumor_density"] * 6400.0
    save_single_trajectory_svg("liquid_dmu_ts_before.svg", {"Active Tumor": (td_l1, COLOR_BLUE, "-")})

    f_l2 = os.path.join(LIQUID_DMU_DIR, "dmu_0.022572463768115943_rep_6.npz")
    td_l2 = np.load(f_l2)["tumor_density"] * 6400.0
    save_single_trajectory_svg("liquid_dmu_ts_transient.svg", {"Long Transient": (td_l2, COLOR_BLUE, "-")})

    f_l3 = os.path.join(LIQUID_DMU_DIR, "dmu_0.025289855072463768_rep_2.npz")
    td_l3 = np.load(f_l3)["tumor_density"] * 6400.0
    save_single_trajectory_svg("liquid_dmu_ts_extinction.svg", {"Extinction": (td_l3, COLOR_BLUE, "-")})

    # 3. Main Panel SVGs (Individual Panels with Insets)
    # Solid Δμ Panel
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for v in vals_s_dmu:
        ax.scatter([v] * len(size_s_dmu[v]), size_s_dmu[v], color=COLOR_BLUE, s=18, alpha=0.35, edgecolors="none")
    ax.axvline(peak_s_dmu, color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlim(13.0, 21.0)
    ax.set_ylim(-100, 2800)
    ax.set_xlabel(r"$\Delta \mu \ (\times 10^{-3})$", fontsize=FS_MAIN_LABEL)
    ax.set_ylabel("Equilibrium Tumor Size (cells)", fontsize=FS_MAIN_LABEL)
    #ax.set_title("Solid Tumor ($\Delta \mu$ Bifurcation)", fontsize=FS_MAIN_TITLE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FS_MAIN_TICK)

    fig.tight_layout()
    fig.savefig(os.path.join(FINAL_SVG_DIR, "solid_dmu_bifurcation.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)

    # Solid Δr Panel
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for v in vals_s_dr:
        ax.scatter([v] * len(size_s_dr[v]), size_s_dr[v], color=COLOR_RED, s=18, alpha=0.35, edgecolors="none")
    ax.axvline(peak_s_dr, color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlim(0.8, 7.2)
    ax.set_ylim(-100, 2800)
    ax.set_xlabel(r"$\Delta r \ (\times 10^{-3})$", fontsize=FS_MAIN_LABEL)
    ax.set_ylabel("Equilibrium Tumor Size (cells)", fontsize=FS_MAIN_LABEL)
    #ax.set_title("Solid Tumor ($\Delta r$ Bifurcation)", fontsize=FS_MAIN_TITLE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FS_MAIN_TICK)

    fig.tight_layout()
    fig.savefig(os.path.join(FINAL_SVG_DIR, "solid_dr_bifurcation.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)

    # Liquid Δμ Panel
    fig, ax = plt.subplots(figsize=(8.5, 6))
    #ax.axvspan(22.7, 23.7, color="#4292c6", alpha=0.20, label="Bistability Region")
    for v in vals_l_dmu:
        ax.scatter([v] * len(size_l_dmu[v]), size_l_dmu[v], color=COLOR_BLUE, s=18, alpha=0.35, edgecolors="none")
    ax.axvline(peak_l_dmu, color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlim(19.0, 27.0)
    ax.set_ylim(-100, 2800)
    ax.set_xlabel(r"$\Delta \mu \ (\times 10^{-3})$", fontsize=FS_MAIN_LABEL)
    ax.set_ylabel("Equilibrium Tumor Size (cells)", fontsize=FS_MAIN_LABEL)
    #ax.set_title("Liquid Tumor ($\Delta \mu$ Bifurcation)", fontsize=FS_MAIN_TITLE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FS_MAIN_TICK)

    fig.tight_layout()
    fig.savefig(os.path.join(FINAL_SVG_DIR, "liquid_dmu_bifurcation.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)

    # Liquid Δr Panel
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for v in vals_l_dr:
        ax.scatter([v] * len(size_l_dr[v]), size_l_dr[v], color=COLOR_RED, s=18, alpha=0.35, edgecolors="none")
    ax.axvline(peak_l_dr, color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlim(0.9, 4.1)
    ax.set_ylim(-100, 2800)
    ax.set_xlabel(r"$\Delta r \ (\times 10^{-3})$", fontsize=FS_MAIN_LABEL)
    ax.set_ylabel("Equilibrium Tumor Size (cells)", fontsize=FS_MAIN_LABEL)
    #ax.set_title("Liquid Tumor ($\Delta r$ Bifurcation)", fontsize=FS_MAIN_TITLE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FS_MAIN_TICK)

    fig.tight_layout()
    fig.savefig(os.path.join(FINAL_SVG_DIR, "liquid_dr_bifurcation.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)

    print("Export completed successfully.")


if __name__ == "__main__":
    export_all_svgs()
