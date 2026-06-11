"""
plot_single_trajectory.py
--------------------------
Plot an individual health (recovery) and tumor (proliferation) trajectory
in phase space (r/r0 vs P_s), alongside the critical boundary.

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

from src.analysis.loaders import load_ensemble_csv, load_sim, load_stability_results_solid
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
dfa = load_stability_results_solid()
grp = dfa.groupby("rmax_norm")["stable_dmu"]
r_prop_list = np.array(sorted(dfa["rmax_norm"].unique()))
mu_med = grp.median().values * 10
p_die_data = (1.0 - mu_med+0.00) ** N_HK

spl = make_smoothing_spline(r_prop_list, p_die_data, lam=0.001)
x_new = np.linspace(1.0, 2.2, 200)

# ---------------------------------------------------------------------------
# Load reproducible Health and Tumor trajectories
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[3]
path_h = repo_root / "data" / "simulations" / ENSEMBLE_DIR / "sim_1.npz"
path_t = repo_root / "data" / "simulations" / ENSEMBLE_DIR / "sim_10.npz"

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
    """Return array of shape (T, 2) with (r/r0, P_s) at each step."""
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

    # Phase diagram curve (Simulation boundary)
    ax.plot(x_new, y_spline, zs=z_min, zdir="z", color="k", lw=1.5, zorder=10, label="Simulation boundary")

    # Green region (healthy, above curve)
    verts_g = list(zip(x_new, y_spline)) + list(zip(x_new[::-1], np.full_like(x_new, 0.6)))
    ax.add_collection3d(Poly3DCollection([[(x, y, z_min) for x, y in verts_g]],
                                          alpha=0.05, facecolor="k"))

    # Red region (tumor, below curve)
    verts_r = list(zip(x_new, np.full_like(x_new, 0.6))) + \
              list(zip(x_new[::-1], y_spline[::-1]))
    ax.add_collection3d(Poly3DCollection([[(x, y, z_min) for x, y in verts_r]],
                                          alpha=0.3, facecolor="k"))

    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.55)

    fontsize_text = 16
    ax.text(1.8, 0.90, z_min, "Expansion", color="black", fontsize=fontsize_text,
            ha="center", va="center", weight="semibold", bbox=bbox_props, zorder=20)

    ax.text(1.4, 0.7, z_min, "Collapse", color="black", fontsize=fontsize_text,
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
    ax.set_ylabel(r"$P_s$", fontsize=fontsize_label, labelpad=15)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(f"Tumor Density", fontsize=fontsize_label, labelpad=40, rotation=96)

    ax.set_xlim(min(x_new), max(x_new))
    ax.set_ylim(0.6, 1.0)
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
    ax.tick_params(axis="z", pad=5)
    plt.setp(ax.get_zticklabels(), rotation=15, ha='right')
    
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.3))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.3))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"].update(
            {"color": (0.8, 0.8, 0.8, 0.4), "linewidth": 0.8, "linestyle": "--"}
        )

    ax.view_init(elev=30, azim=30)
    ax.set_box_aspect(None, zoom=0.8)
    fig.subplots_adjust(left=0.05, right=0.82, top=0.95, bottom=0.05)
    #fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


out_dir = repo_root / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)

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
