"""
plot_single_trajectory_liquid.py
--------------------------------
Plot individual health (recovery) and tumor (proliferation) trajectories
for the liquid-tumor model in phase space (r/r0 vs P_death), alongside
the critical boundary.

Outputs
-------
outputs/figures/single_trajectory_4panel_health_liquid.png
outputs/figures/single_trajectory_4panel_tumor_liquid.png
outputs/figures/single_trajectory_3d_health_liquid.png
outputs/figures/single_trajectory_3d_tumor_liquid.png
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import make_smoothing_spline
from matplotlib.ticker import MaxNLocator, FuncFormatter

from src.analysis.loaders import load_sim, load_all_stability_results, load_adaptive_stability_results_liquid
from src.analysis.stats import dyn_state

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
R0 = 0.15        # baseline reproduction rate
N_HK = 10        # number of HK genes
LIQUID_DIR = "data/simulations_liquid/ensemble_results"

# Load solid stability boundary
merged_solid = load_all_stability_results()
df_solid = merged_solid
# Group/groupby full solid data to fit spline safely
grp_s = df_solid.groupby("rmax_norm")["stable_dmu"]
r_unique_s = np.array(sorted(df_solid["rmax_norm"].unique()))
solid_med_mu = grp_s.median().values * 10
solid_med = dyn_state(0.0, 1.0, solid_med_mu, 0.5, N_HK)[1]
spl_s = make_smoothing_spline(r_unique_s, solid_med, lam=0.0001)

# Load liquid stability boundary
dfl = load_adaptive_stability_results_liquid()
if dfl.empty:
    print("Liquid stability sweep file not found or empty.")
    sys.exit(1)
# We clip rmax_norm > 1.0 because boundary starts above 1.0 (at 1.0 any mutation is unstable)
df_liquid_clip = dfl[dfl["rmax_norm"] > 1.0].copy()
grp_l = df_liquid_clip.groupby("rmax_norm")["stable_dmu"]
r_unique_l = np.array(sorted(df_liquid_clip["rmax_norm"].unique()))
liquid_med_mu = grp_l.median().values * 10
liquid_med = dyn_state(0.0, 1.0, liquid_med_mu, 0.5, N_HK)[1]
spl_l = make_smoothing_spline(r_unique_l, liquid_med, lam=0.0001)

x_new = np.linspace(0.95, 2.05, 200)

# ---------------------------------------------------------------------------
# Load specific trajectories (sim_81 for Health, sim_43 for Tumor)
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[3]
path_h = repo_root / LIQUID_DIR / "sim_81.npz"
path_t = repo_root / LIQUID_DIR / "sim_43.npz"

if not path_h.exists() or not path_t.exists():
    print("Could not find sim_81.npz or sim_43.npz in data/simulations_liquid/ensemble_results/")
    sys.exit(1)

sim_h = {k: v for k, v in np.load(path_h).items()}
sim_t = {k: v for k, v in np.load(path_t).items()}

print(f"Loaded Health run (sim_81): {len(sim_h['r'])} steps, max density = {np.max(sim_h['tumor_density']):.6f}")
print(f"Loaded Tumor run (sim_43): {len(sim_t['r'])} steps, max density = {np.max(sim_t['tumor_density']):.6f}")

def trajectory_phase(sim) -> np.ndarray:
    """Return array of shape (T, 2) with (r/r0, p_death) at each step."""
    pts = np.array([
        dyn_state(r=r, r0=R0, mu=mu, k=k, N=N_HK)
        for r, mu, k in zip(sim["r"], sim["mu"], sim["mut_HK"])
    ])
    return pts

pts_h = trajectory_phase(sim_h)
pts_t = trajectory_phase(sim_t)
td_h  = sim_h["tumor_density"]
td_t  = sim_t["tumor_density"]

# Filter out steps where tumor density is 0 to avoid fallback artifacts at (0,0)
valid_h = td_h > 0
pts_h = pts_h[valid_h]
td_h = td_h[valid_h]
for k in list(sim_h.keys()):
    if len(sim_h[k]) == len(valid_h):
        sim_h[k] = sim_h[k][valid_h]

valid_t = td_t > 0
pts_t = pts_t[valid_t]
td_t = td_t[valid_t]
for k in list(sim_t.keys()):
    if len(sim_t[k]) == len(valid_t):
        sim_t[k] = sim_t[k][valid_t]

# ---------------------------------------------------------------------------
# ── Figure 1: 4-panel diagnostic ──────────────────────────────────────────
# ---------------------------------------------------------------------------
def _four_panel(pts, sim, td, spl_s, spl_l, x_new, title, color):
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.7))
    fs = 18

    for axi in ax:
        axi.tick_params(axis="both", which="major", labelsize=13)
        axi.grid(ls=":", alpha=0.4)
        axi.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Panel 0: phase space
    y_spl_s = spl_s(x_new)
    y_spl_l = spl_l(x_new)
    ax[0].plot(x_new, y_spl_s, color="black", ls="--", lw=1.2, zorder=1, label="Solid Boundary")
    ax[0].plot(x_new, y_spl_l, color="black", lw=1.8, zorder=2, label="Liquid Boundary")
    ax[0].fill_between(x_new, -0.05, y_spl_s, alpha=0.15, color="#2A9D8F")
    ax[0].fill_between(x_new, y_spl_s, y_spl_l, alpha=0.15, color="#E9C46A")
    ax[0].fill_between(x_new, y_spl_l, 1.05, alpha=0.15, color="#E63946")
    sc = ax[0].scatter(*pts.T, c=np.linspace(0, 1, len(pts)),
                       cmap="magma", s=10, zorder=5)
    plt.colorbar(sc, ax=ax[0], label="Time (normalised)")
    ax[0].set_xlabel(r"$r/r_0$", fontsize=fs)
    ax[0].set_ylabel(r"$p_{\rm death}$", fontsize=fs)
    ax[0].set_xlim(0.95, 2.05)
    ax[0].set_ylim(-0.05, 1.05)
    
    # Place labels carefully
    ax[0].text(1.0,  0.95, "Global Collapse",
               bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
               fontsize=12)
    ax[0].text(1.70, 0.02, "Expansion",
               bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
               fontsize=12)

    # Panel 1: mutation rate
    ax[1].scatter(np.arange(len(sim["mu"])), sim["mu"],
                  c=np.arange(len(sim["mu"])), cmap="magma", s=8)
    ax[1].set_xlabel("Time", fontsize=fs)
    ax[1].set_ylabel(r"$\mu$", fontsize=fs)

    # Panel 2: tumor density
    ax[2].scatter(np.arange(len(td)), td,
                  c=np.arange(len(td)), cmap="magma", s=8)
    ax[2].set_xlabel("Time", fontsize=fs)
    ax[2].set_ylabel("Tumor Density", fontsize=fs)

    # Panel 3: distance to critical line
    dist = np.array([float(spl_l(r)) - p for r, p in zip(sim["r"] / R0, pts[:, 1])])
    ax[3].scatter(np.arange(len(dist)), dist,
                  c=np.arange(len(dist)), cmap="magma", s=8)
    ax[3].axhline(0, color="red", ls="--", lw=1.5)
    ax[3].set_xlabel("Time", fontsize=fs)
    ax[3].set_ylabel(r"$p_{\rm death,crit} - p_{\rm death}$", fontsize=fs)

    #fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig

fig_h = _four_panel(pts_h, sim_h, td_h, spl_s, spl_l, x_new,
                    title="Health trajectory (sim 81) - Liquid Model", color="#2A9D8F")
fig_t = _four_panel(pts_t, sim_t, td_t, spl_s, spl_l, x_new,
                    title="Tumor trajectory (sim 43) - Liquid Model", color="#E63946")

out_dir = repo_root / "outputs" / "figures" / "liquid"
out_dir.mkdir(parents=True, exist_ok=True)

fig_h.savefig(out_dir / "single_trajectory_4panel_health_liquid.png", dpi=150, bbox_inches="tight")
fig_t.savefig(out_dir / "single_trajectory_4panel_tumor_liquid.png", dpi=150, bbox_inches="tight")
fig_h.savefig(out_dir / "single_trajectory_4panel_health_liquid.svg", dpi=150, bbox_inches="tight")
fig_t.savefig(out_dir / "single_trajectory_4panel_tumor_liquid.svg", dpi=150, bbox_inches="tight")
plt.close(fig_h)
plt.close(fig_t)
print("Saved 4-panel diagnostic figures.")

# ---------------------------------------------------------------------------
# ── Figure 2: 3-D phase-space trajectories ────────────────────────────────
# ---------------------------------------------------------------------------
def _3d_plot(pts, td, spl_s, spl_l, x_new, title="", scale_str="×10⁻³"):
    # Normalise dot size to [5, 200]
    td_min, td_max = td.min(), td.max()
    if td_max > td_min:
        size = 5 + 195 * ((td - td_min) / (td_max - td_min)) ** 1.5
    else:
        size = np.full_like(td, 20.0)

    z_min = 0
    y_spline_s = spl_s(x_new)
    y_spline_l = spl_l(x_new)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Trajectory scatter
    sc = ax.scatter(pts[:, 0], pts[:, 1], td, c=td, cmap="coolwarm", s=size)

    # Phase diagram curves on floor
    ax.plot(x_new, y_spline_s, zs=z_min, zdir="z", color="k", ls="--", lw=1.2, zorder=10)
    ax.plot(x_new, y_spline_l, zs=z_min, zdir="z", color="k", lw=1.8, zorder=10)

    # Green region (Solid & Liquid Expansion)
    verts_g = list(zip(x_new, np.full_like(x_new, -0.05))) + list(zip(x_new[::-1], y_spline_s[::-1]))
    ax.add_collection3d(Poly3DCollection([[(x, y, z_min) for x, y in verts_g]],
                                          alpha=0.05, facecolor="k"))

    # Yellow region (Liquid-Only Expansion)
    verts_y = list(zip(x_new, y_spline_s)) + list(zip(x_new[::-1], y_spline_l[::-1]))
    ax.add_collection3d(Poly3DCollection([[(x, y, z_min) for x, y in verts_y]],
                                          alpha=0.25, facecolor="k"))

    # Red region (Global Collapse)
    verts_r = list(zip(x_new, y_spline_l)) + list(zip(x_new[::-1], np.full_like(x_new, 1.05)))
    ax.add_collection3d(Poly3DCollection([[(x, y, z_min) for x, y in verts_r]],
                                          alpha=0.4, facecolor="k"))

    # Place region names in the z=0 shaded plane
    x_lbl = 1.8
    y_s_val = float(spl_s(x_lbl))
    y_l_val = float(spl_l(x_lbl))
    
    y_g = (-0.05 + y_s_val) / 2.0
    y_y = (y_s_val + y_l_val) / 2.0
    y_r = (y_l_val + 1.05) / 2.0
    
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.55)
    
    ax.text(x_lbl, y_g, z_min, "Solid & Liquid\nExpansion", color="black", fontsize=9,
            ha="center", va="center", weight="semibold", bbox=bbox_props, zorder=20)
    ax.text(x_lbl, y_y-0.1, z_min, "Liquid Expansion\nSolid Collapse", color="black", fontsize=9,
            ha="center", va="center", weight="semibold", bbox=bbox_props, zorder=20)
    ax.text(x_lbl-0.5, y_r, z_min, "Global\nCollapse", color="black", fontsize=9,
            ha="center", va="center", weight="semibold", bbox=bbox_props, zorder=20)


    # Shadow on the floor
    ax.scatter(pts[:, 0], pts[:, 1], zs=0, zdir="z",
               c=td, cmap="coolwarm", s=size, alpha=0.08)

    # Vertical line at crossing point (liquid boundary)
    sign_diff = pts[:, 1] - spl_l(pts[:, 0])
    crossings = np.where(np.diff(np.sign(sign_diff)))[0]
    if len(crossings):
        ci = crossings[0]
        ax.plot([pts[ci, 0]] * 2, [pts[ci, 1]] * 2, [0, td[ci]],
                color="k", ls="--", lw=1.5)

    ax.set_xlabel(r"$r/r_0$", fontsize=16, labelpad=10)
    ax.set_ylabel(r"$P_{\rm death}$", fontsize=16, labelpad=10)
    ax.set_zlabel(f"Tumor Density ({scale_str})", fontsize=14, labelpad=26)

    ax.set_xlim(0.95, 2.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_zlim(0, max(td) * 1.1)

    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    
    def format_sci(x, pos):
        if abs(x) < 1e-9:
            return "0"
        s = f"{x:.1e}"
        return s.replace("e-0", "e-").replace("e+0", "e+")
    ax.zaxis.set_major_formatter(FuncFormatter(format_sci))
    
    # Rotate Z-axis tick labels and add padding to avoid superposition with grid/label
    ax.tick_params(axis="z", pad=6)
    plt.setp(ax.get_zticklabels(), rotation=15, ha='left')
    
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.3))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.3))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"].update(
            {"color": (0.8, 0.8, 0.8, 0.4), "linewidth": 0.8, "linestyle": "--"}
        )

    ax.set_box_aspect(None, zoom=0.85)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig

fig_3dh = _3d_plot(pts_h, td_h, spl_s, spl_l, x_new,
                   title=None,#"Health trajectory in phase space (sim 81) - Liquid Model",
                   scale_str="×10⁻³")
fig_3dt = _3d_plot(pts_t, td_t, spl_s, spl_l, x_new,
                   title=None,#"Tumor trajectory in phase space (sim 43) - Liquid Model",
                   scale_str="×10⁻¹")

fig_3dh.savefig(out_dir / "single_trajectory_3d_health_liquid.png", dpi=150, bbox_inches="tight")
fig_3dt.savefig(out_dir / "single_trajectory_3d_tumor_liquid.png", dpi=150, bbox_inches="tight")
fig_3dh.savefig(out_dir / "single_trajectory_3d_health_liquid.svg", dpi=150, bbox_inches="tight")
fig_3dt.savefig(out_dir / "single_trajectory_3d_tumor_liquid.svg", dpi=150, bbox_inches="tight")
plt.close(fig_3dh)
plt.close(fig_3dt)

print(f"Saved 3D liquid trajectory plots to: {out_dir}")
plt.show()
