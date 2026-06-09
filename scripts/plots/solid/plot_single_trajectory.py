"""
plot_single_trajectory.py
--------------------------
Plot an individual health (recovery) and tumor (proliferation) trajectory
in phase space (r/r0 vs p_death), alongside the critical boundary.

Also produces the 4-panel diagnostic figure and the 3-D phase-space plot
from Cells 27–30 of notebooks/analysis.ipynb.

New data format
~~~~~~~~~~~~~~~
A single trajectory is loaded directly from one NPZ file in
``data/simulations/ensemble_results_2CHR/``.
The simulation stores per-step values for ``r``, ``mu``, ``mut_HK``,
``tumor_density``, and ``outcome_code``.

The phase-diagram coordinate is computed as:
    x  = r / r0
    p  = 1 − (1−mu)^(2kN) · (1−mu²)^(N·(1−2k))
where k ≡ mut_HK (fraction of HK genes mutated) and N = 10 HK genes.

Outputs
-------
outputs/figures/single_trajectory_4panel.png
outputs/figures/single_trajectory_3d_health.png
outputs/figures/single_trajectory_3d_tumor.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers the projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import make_smoothing_spline

from matplotlib.ticker import MaxNLocator, FuncFormatter

from src.analysis.loaders import load_ensemble_csv, load_sim, load_adaptive_stability_results
from src.analysis.stats import dyn_state

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
})

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ENSEMBLE_DIR = "ensemble_results_D"
R0 = 0.15        # baseline reproduction rate
N_HK = 10        # number of HK genes

# Phase-diagram data (from sweep:analysis_adaptive)
dfa = load_adaptive_stability_results()
grp = dfa.groupby("rmax_norm")["stable_dmu"]
r_prop_list = np.array(sorted(dfa["rmax_norm"].unique()))
mu_med = grp.median().values * 10
p_die_data = 1.0 - (1.0 - mu_med+0.00) ** N_HK

spl = make_smoothing_spline(r_prop_list, p_die_data, lam=0.001)
x_new = np.linspace(0.8, 2.2, 200)

# ---------------------------------------------------------------------------
# Load reproducible Health and Tumor trajectories
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[3]
path_h = repo_root / "data" / "simulations" / "spatial_run" / "health_results.npz"
path_t = repo_root / "data" / "simulations" / "spatial_run" / "sim_1.npz"

if not path_h.exists() or not path_t.exists():
    print("Could not find health_results.npz or tumor_results.npz in data/simulations/spatial_run/")
    sys.exit(1)

sim_h = {k: v for k, v in np.load(path_h).items()}
sim_t = {k: v for k, v in np.load(path_t).items()}

sid_h = "9 (reproducible)"
sid_t = "7 (reproducible)"

print(f"Health run: sim_{sid_h}  ({len(sim_h['r'])} steps)")
print(f"Tumor  run: sim_{sid_t}  ({len(sim_t['r'])} steps)")


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

# ---------------------------------------------------------------------------
# ── Figure 1: 4-panel diagnostic ──────────────────────────────────────────
# ---------------------------------------------------------------------------

def _four_panel(pts, sim, td, spl, x_new, title, color):
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.7))
    fs = 18
    fontsize_letter = 24
    letters = ["a", "b", "c", "d"]

    for i, axi in enumerate(ax):
        axi.tick_params(axis="both", which="major", labelsize=13)
        axi.grid(ls=":", alpha=0.4)
        axi.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axi.text(-0.15, 1.12, letters[i], transform=axi.transAxes,
                 fontsize=fontsize_letter, fontweight="bold", va="top", ha="right")

    # Panel 0: phase space
    y_spl = spl(x_new)
    ax[0].plot(x_new, y_spl, color="k", lw=1.5, zorder=0)
    ax[0].fill_between(x_new, y_spl, max(y_spl) * 1.5, alpha=0.15, color="#2A9D8F")
    ax[0].fill_between(x_new, -0.01, y_spl, alpha=0.15, color="#E63946")
    sc = ax[0].scatter(*pts.T, c=np.linspace(0, 1, len(pts)),
                       cmap="magma", s=10, zorder=5)
    plt.colorbar(sc, ax=ax[0], label="Time (normalised)")
    ax[0].set_xlabel(r"$r/r_0$", fontsize=fs)
    ax[0].set_ylabel(r"$p_{\rm death}$", fontsize=fs)
    ax[0].set_xlim(0.95, 2.05)
    ax[0].set_ylim(-0.005, 0.5)
    ax[0].text(1.0,  0.45, "Tumor Shrinks",
               bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
               fontsize=12)
    ax[0].text(1.70, 0.02, "Tumor Grows",
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
    dist = np.array([float(spl(r)) - p for r, p in zip(sim["r"] / R0, pts[:, 1])])
    ax[3].scatter(np.arange(len(dist)), dist,
                  c=np.arange(len(dist)), cmap="magma", s=8)
    ax[3].axhline(0, color="red", ls="--", lw=1.5)
    ax[3].set_xlabel("Time", fontsize=fs)
    ax[3].set_ylabel(r"$p_{\rm death,crit} - p_{\rm death}$", fontsize=fs)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


fig_h = _four_panel(pts_h, sim_h, td_h, spl, x_new,
                    title=f"Health trajectory  (sim {sid_h})", color="#2A9D8F")
fig_t = _four_panel(pts_t, sim_t, td_t, spl, x_new,
                    title=f"Tumor trajectory  (sim {sid_t})", color="#E63946")

out_dir = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)

#fig_h.savefig(out_dir / "single_trajectory_4panel_health.png", dpi=150,
              #bbox_inches="tight")
#fig_h.savefig(out_dir / "single_trajectory_4panel_health.svg",
              #bbox_inches="tight")
#fig_t.savefig(out_dir / "single_trajectory_4panel_tumor.png", dpi=150,
              #bbox_inches="tight")
#fig_t.savefig(out_dir / "single_trajectory_4panel_tumor.svg",
              #bbox_inches="tight")
#print("Saved 4-panel figures (PNG and SVG).")

# ---------------------------------------------------------------------------
# ── Figure 2: 3-D phase-space trajectories ────────────────────────────────
# ---------------------------------------------------------------------------

def _3d_plot(pts, td, spl, x_new, lim=None, title="", scale_str="×10⁻³", label_letter=""):
    if lim:
        pts = pts[:lim]
        td  = td[:lim]

    # Normalise dot size to [5, 200] regardless of absolute density scale
    td_min, td_max = td.min(), td.max()
    if td_max > td_min:
        size = 5 + 195 * ((td - td_min) / (td_max - td_min)) ** 1.5
    else:
        size = np.full_like(td, 20.0)

    z_min = 0
    y_spline = spl(x_new)

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    if label_letter:
        ax.text2D(0.2, 0.85, label_letter, transform=ax.transAxes,
                  fontsize=24, fontweight="bold", va="top", ha="right")

    # Trajectory scatter
    ax.scatter(pts[:, 0], pts[:, 1], td, c=td, cmap="coolwarm", s=size)

    # Phase diagram curve
    ax.plot(x_new, y_spline, zs=z_min, zdir="z", color="k", lw=1.5, zorder=10)

    # Green region (healthy, above curve)
    verts_g = list(zip(x_new, y_spline)) + list(zip(x_new[::-1], np.full_like(x_new, max(y_spline) * 1.5)))
    ax.add_collection3d(Poly3DCollection([[(x, y, z_min) for x, y in verts_g]],
                                          alpha=0.05, facecolor="k"))

    # Red region (tumor, below curve)
    verts_r = list(zip(x_new, np.full_like(x_new, min(y_spline)))) + \
              list(zip(x_new[::-1], y_spline[::-1]))
    ax.add_collection3d(Poly3DCollection([[(x, y, z_min) for x, y in verts_r]],
                                          alpha=0.3, facecolor="k"))

    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.55)

    fontsize_text = 16
    ax.text(1.8, 0.08, z_min, "Expansion", color="black", fontsize=fontsize_text,
            ha="center", va="center", weight="semibold", bbox=bbox_props, zorder=20)

    ax.text(1.2, 0.35, z_min, "Collapse", color="black", fontsize=fontsize_text,
            ha="center", va="center", weight="semibold", bbox=bbox_props, zorder=0)
    

    # Shadow on the floor
    ax.scatter(pts[:, 0], pts[:, 1], zs=0, zdir="z",
               c=td, cmap="coolwarm", s=size, alpha=0.08)

    # Vertical line at crossing point
    sign_diff = pts[:, 1] - spl(pts[:, 0])
    crossings = np.where(np.diff(np.sign(sign_diff)))[0]
    if len(crossings):
        ci = crossings[0]
        ax.plot([pts[ci, 0]] * 2, [pts[ci, 1]] * 2, [0, td[ci]],
                color="k", ls="--", lw=1.5)

    fontsize_label = 18
    fontsize_tick = 14
    ax.set_xlabel(r"$r/r_0$", fontsize=fontsize_label, labelpad=15)
    ax.set_ylabel(r"$P_{\rm death}$", fontsize=fontsize_label, labelpad=15)
    ax.set_zlabel(f"Tumor Density", fontsize=fontsize_label, labelpad=40)

    ax.set_xlim(min(x_new), max(x_new))
    ax.set_ylim(min(y_spline), max(y_spline) * 1.5)
    ax.set_zlim(0, max(td) * 1.1)

    ax.tick_params(axis="both", labelsize=fontsize_tick)
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

    ax.set_box_aspect(None, zoom=0.8)
    fig.subplots_adjust(left=0.05, right=0.82, top=0.95, bottom=0.05)
    #fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


fig_3dh = _3d_plot(pts_h, td_h, spl, x_new,
                   title=f"Health trajectory in phase space (sim {sid_h})",
                   scale_str="×10⁻³", label_letter="f")
fig_3dt = _3d_plot(pts_t, td_t, spl, x_new, lim=700,
                   title=f"Tumor trajectory in phase space (sim {sid_t})",
                   scale_str="×10⁻¹", label_letter="e")

fig_3dh.savefig(out_dir / "single_trajectory_3d_health.png", dpi=150)
fig_3dh.savefig(out_dir / "single_trajectory_3d_health.svg")
fig_3dt.savefig(out_dir / "single_trajectory_3d_tumor.png",  dpi=150)
fig_3dt.savefig(out_dir / "single_trajectory_3d_tumor.svg")
print(f"Saved 3-D figures (PNG and SVG) → {out_dir}")
#plt.show()
