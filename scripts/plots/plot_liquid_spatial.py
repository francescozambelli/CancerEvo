"""
plot_liquid_spatial.py
----------------------
Visualize the spatiotemporal progression of a liquid-tumor simulation.

Loads spatial snapshots (data/simulations_liquid/spatial_run/snapshots.npz)
produced by scripts/simulation_spatial_liquid.jl and renders:

  Panel A  – Grid of L×L lattice frames (state: WT / Cancer / Dead)
             coloured by cell type, one frame per snapshot.
  Panel B  – Tumor density time series with vertical markers at snapshot times.
  Panel C  – Spatial map of mutation rate μ at the final snapshot.
  Panel D  – Spatial map of division rate r at the final snapshot.

Outputs
-------
outputs/figures/liquid_spatial_progression.png
outputs/figures/liquid_spatial_progression.svg

Usage
-----
  python scripts/plots/plot_liquid_spatial.py [path/to/snapshots.npz]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = PROJECT_ROOT / "data" / "simulations_liquid" / "spatial_run"
OUT_DIR   = PROJECT_ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

snap_path    = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "snapshots.npz"
results_path = snap_path.parent / "results.npz"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "figure.facecolor":   "#0d1117",   # dark background
    "axes.facecolor":     "#161b22",
    "axes.labelcolor":    "#e6edf3",
    "xtick.color":        "#8b949e",
    "ytick.color":        "#8b949e",
    "text.color":         "#e6edf3",
    "axes.edgecolor":     "#30363d",
    "grid.color":         "#21262d",
})

# Cell-state colormap: WT (#1d3557 deep blue) | Cancer (#e63946 vivid red) | Dead (#6e7681 grey)
STATE_CMAP = mcolors.ListedColormap(["#1d3557", "#e63946", "#6e7681"])
STATE_NORM  = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], STATE_CMAP.N)

MU_CMAP = "plasma"
R_CMAP  = "viridis"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Loading snapshots from: {snap_path}")
snaps = np.load(snap_path, allow_pickle=True)

state_arr   = snaps["state"]           # (n_frames, L, L)  int8
mu_arr      = snaps["mu"]              # (n_frames, L, L)  float64
r_arr       = snaps["r"]              # (n_frames, L, L)  float64
snap_steps  = snaps["snapshot_steps"] # (n_frames,)        int
L           = int(snaps["L"][0])
outcome     = snaps["outcome"].tobytes().decode()  # was saved as Vector{UInt8}

n_frames = state_arr.shape[0]
print(f"  {n_frames} frames | L={L} | Outcome: {outcome}")

# Subsample to at most 12 frames so the grid stays readable
MAX_FRAMES = 12
if n_frames > MAX_FRAMES:
    idx = np.round(np.linspace(0, n_frames - 1, MAX_FRAMES)).astype(int)
    state_arr  = state_arr[idx]
    mu_arr     = mu_arr[idx]
    r_arr      = r_arr[idx]
    snap_steps = snap_steps[idx]
    n_frames   = MAX_FRAMES
    print(f"  Subsampled to {n_frames} frames for display.")

results  = np.load(results_path)
td       = results["tumor_density"]
t_axis   = np.arange(len(td))

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# Determine grid dimensions for snapshot panels
n_cols = min(n_frames, 4)
n_rows = int(np.ceil(n_frames / n_cols))

fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows + 5), facecolor="#0d1117")

outer = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[n_rows * 4, 4.5],
    hspace=0.35,
)

# --- Top: snapshot grid ---
snap_gs = gridspec.GridSpecFromSubplotSpec(
    n_rows, n_cols, subplot_spec=outer[0],
    hspace=0.08, wspace=0.06,
)

# --- Bottom: time series + final mu + final r ---
bottom_gs = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[1],
    wspace=0.38,
)
ax_ts  = fig.add_subplot(bottom_gs[0])
ax_mu  = fig.add_subplot(bottom_gs[1])
ax_r   = fig.add_subplot(bottom_gs[2])

# ---------------------------------------------------------------------------
# A – Snapshot grid
# ---------------------------------------------------------------------------
snap_axes = []
for f in range(n_frames):
    row, col = divmod(f, n_cols)
    ax = fig.add_subplot(snap_gs[row, col])
    snap_axes.append(ax)

    im = ax.imshow(
        state_arr[f],
        cmap=STATE_CMAP, norm=STATE_NORM,
        origin="upper", interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    # Frame label: step number + cancer fraction
    canc_frac = (state_arr[f] == 1).mean()
    label = f"t = {snap_steps[f]}\n{100*canc_frac:.1f}% cancer"
    ax.set_title(label, fontsize=8.5, color="#e6edf3", pad=3)

# Hide unused axes
for f in range(n_frames, n_rows * n_cols):
    row, col = divmod(f, n_cols)
    fig.add_subplot(snap_gs[row, col]).set_visible(False)

# Shared colorbar legend for state map
legend_elements = [
    Patch(facecolor="#1d3557", label="Wild-type"),
    Patch(facecolor="#e63946", label="Cancer"),
    Patch(facecolor="#6e7681", label="Dead"),
]
snap_axes[0].legend(
    handles=legend_elements,
    loc="upper left", fontsize=8,
    frameon=True, facecolor="#161b22",
    edgecolor="#30363d", framealpha=0.9,
    labelcolor="#e6edf3",
)

# ---------------------------------------------------------------------------
# B – Tumor density time series
# ---------------------------------------------------------------------------
ax_ts.plot(t_axis, td, color="#e63946", lw=1.5, label="Tumor density")
ax_ts.fill_between(t_axis, 0, td, color="#e63946", alpha=0.18)

# Mark snapshot times
for s in snap_steps:
    if s <= len(td):
        ax_ts.axvline(s, color="#f4a261", lw=0.8, alpha=0.7, ls="--")

ax_ts.set_xlabel("Simulation step", fontsize=11)
ax_ts.set_ylabel("Tumor density", fontsize=11)
ax_ts.set_ylim(0, min(1.0, td.max() * 1.15 + 0.01))
ax_ts.set_xlim(0, len(td))
ax_ts.grid(True, ls=":", alpha=0.35)
ax_ts.set_title("Tumor density over time", fontsize=11, pad=6)
outcome_color = "#2a9d8f" if outcome == "Health" else "#e63946"
ax_ts.text(
    0.97, 0.95, f"Outcome: {outcome}",
    transform=ax_ts.transAxes,
    ha="right", va="top", fontsize=9,
    color=outcome_color, fontweight="bold",
)

# ---------------------------------------------------------------------------
# C – Final μ spatial map
# ---------------------------------------------------------------------------
mu_final = mu_arr[-1]
cancer_mask = (state_arr[-1] == 1)

# Show μ only where there are cancer cells; grey-out the rest
mu_display = np.where(cancer_mask, mu_final, np.nan)
im_mu = ax_mu.imshow(
    mu_display,
    cmap=MU_CMAP, origin="upper",
    interpolation="nearest", aspect="equal",
)
# Underlay WT / dead in dark colour
ax_mu.imshow(
    state_arr[-1],
    cmap=STATE_CMAP, norm=STATE_NORM,
    alpha=0.25, origin="upper",
    interpolation="nearest", aspect="equal",
)
ax_mu.set_xticks([]); ax_mu.set_yticks([])
ax_mu.set_title(f"μ field  (t = {snap_steps[-1]})", fontsize=11, pad=6)
cbar_mu = fig.colorbar(im_mu, ax=ax_mu, fraction=0.046, pad=0.04)
cbar_mu.set_label("Mutation rate μ", fontsize=9, color="#8b949e")
cbar_mu.ax.tick_params(colors="#8b949e", labelsize=8)

# ---------------------------------------------------------------------------
# D – Final r spatial map
# ---------------------------------------------------------------------------
r_final = r_arr[-1]
r_display = np.where(cancer_mask, r_final, np.nan)
im_r = ax_r.imshow(
    r_display,
    cmap=R_CMAP, origin="upper",
    interpolation="nearest", aspect="equal",
)
ax_r.imshow(
    state_arr[-1],
    cmap=STATE_CMAP, norm=STATE_NORM,
    alpha=0.25, origin="upper",
    interpolation="nearest", aspect="equal",
)
ax_r.set_xticks([]); ax_r.set_yticks([])
ax_r.set_title(f"r field  (t = {snap_steps[-1]})", fontsize=11, pad=6)
cbar_r = fig.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
cbar_r.set_label("Division rate r", fontsize=9, color="#8b949e")
cbar_r.ax.tick_params(colors="#8b949e", labelsize=8)

# ---------------------------------------------------------------------------
# Super-title
# ---------------------------------------------------------------------------
fig.suptitle(
    "Liquid Tumor – Spatial Progression",
    fontsize=16, fontweight="bold", color="#e6edf3", y=1.005,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_png = OUT_DIR / "liquid_spatial_progression.png"
out_svg = OUT_DIR / "liquid_spatial_progression.svg"
plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(out_svg,           bbox_inches="tight", facecolor=fig.get_facecolor())

print(f"\nFigures saved to:\n  {out_png}\n  {out_svg}")
plt.show()
