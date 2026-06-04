"""
plot_liquid_examples.py
-----------------------
Side-by-side comparison of one Health and one Tumor_Max liquid-tumor run.

Loads:
  data/simulations_liquid/example_health/snapshots.npz
  data/simulations_liquid/example_tumor/snapshots.npz

Layout (two-column, one per outcome):
  Row 1–3 : Lattice snapshot grid (up to 6 frames each)
  Row 4   : Tumor density time series
  Row 5   : Final-frame μ field | Final-frame r field

Outputs
-------
outputs/figures/liquid_examples_comparison.png
outputs/figures/liquid_examples_comparison.svg
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "simulations_liquid"
OUT_DIR  = PROJECT_ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = {
    "Health":    DATA_DIR / "example_health",
    "Tumor_Max": DATA_DIR / "example_tumor",
}
CASE_LABELS = {
    "Health":    "Health (tumor clearance)",
    "Tumor_Max": "Tumor (progression)",
}
CASE_COLORS = {
    "Health":    "#2a9d8f",
    "Tumor_Max": "#e63946",
}

MAX_SNAP = 6     # frames shown per case in the grid

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "axes.edgecolor":   "#30363d",
    "grid.color":       "#21262d",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

STATE_CMAP = mcolors.ListedColormap(["#1d3557", "#e63946", "#6e7681"])
STATE_NORM  = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], STATE_CMAP.N)
MU_CMAP    = "plasma"
R_CMAP     = "viridis"

# ---------------------------------------------------------------------------
# Load both datasets
# ---------------------------------------------------------------------------
def load_case(folder):
    snaps   = np.load(folder / "snapshots.npz", allow_pickle=True)
    results = np.load(folder / "results.npz",   allow_pickle=True)
    outcome = snaps["outcome"].tobytes().decode()
    L       = int(snaps["L"][0])
    state   = snaps["state"]
    mu      = snaps["mu"]
    r       = snaps["r"]
    steps   = snaps["snapshot_steps"]
    td      = results["tumor_density"]

    # Subsample to MAX_SNAP frames (evenly spaced, always keep first and last)
    n = state.shape[0]
    if n > MAX_SNAP:
        idx    = np.round(np.linspace(0, n - 1, MAX_SNAP)).astype(int)
        state, mu, r, steps = state[idx], mu[idx], r[idx], steps[idx]

    return dict(outcome=outcome, L=L, state=state, mu=mu, r=r,
                steps=steps, td=td)

data = {name: load_case(folder) for name, folder in CASES.items()}
for name, d in data.items():
    print(f"  {name}: {d['state'].shape[0]} frames, L={d['L']}, "
          f"outcome={d['outcome']}, run_length={len(d['td'])} steps")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# 2 columns (one per case), rows:
#   0..1  : snapshot grid  (3 frames per row × 2 rows = 6)
#   2     : density time series
#   3     : mu / r final maps side by side inside each column

SNAP_ROWS = 2   # rows of snapshot frames
SNAP_COLS = MAX_SNAP // SNAP_ROWS  # = 3

fig_w = 7 * 2          # two wide columns
fig_h = 4 * SNAP_ROWS + 3.5 + 3.5

fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d1117")

# Outer: 2 equal columns
outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.10)

def build_case_column(outer_spec, case_name):
    d = data[case_name]
    col_color = CASE_COLORS[case_name]
    col_label = CASE_LABELS[case_name]

    inner = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=outer_spec,
        height_ratios=[SNAP_ROWS * 4, 3, 3, 3],
        hspace=0.45,
    )

    # --- Column title ---
    ax_title = fig.add_subplot(inner[0])
    ax_title.set_visible(False)
    fig.text(
        ax_title.get_position().x0 + ax_title.get_position().width / 2,
        ax_title.get_position().y1 + 0.01,
        col_label,
        ha="center", va="bottom",
        fontsize=13, fontweight="bold", color=col_color,
        transform=fig.transFigure,
    )

    # --- Snapshot grid ---
    snap_gs = gridspec.GridSpecFromSubplotSpec(
        SNAP_ROWS, SNAP_COLS,
        subplot_spec=inner[0],
        hspace=0.06, wspace=0.04,
    )
    state_arr = d["state"]
    snap_steps = d["steps"]

    for f in range(state_arr.shape[0]):
        row, col = divmod(f, SNAP_COLS)
        ax = fig.add_subplot(snap_gs[row, col])
        ax.imshow(state_arr[f], cmap=STATE_CMAP, norm=STATE_NORM,
                  origin="upper", interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        canc_frac = (state_arr[f] == 1).mean()
        ax.set_title(f"t={snap_steps[f]}\n{100*canc_frac:.2f}%",
                     fontsize=7.5, color="#c9d1d9", pad=2)

        # Legend only on first frame of first column
        if f == 0 and case_name == "Health":
            legend_elements = [
                Patch(facecolor="#1d3557", label="WT"),
                Patch(facecolor="#e63946", label="Cancer"),
                Patch(facecolor="#6e7681", label="Dead"),
            ]
            ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
                      frameon=True, facecolor="#161b22", edgecolor="#30363d",
                      labelcolor="#e6edf3", framealpha=0.9)

    # --- Tumor density ---
    ax_ts = fig.add_subplot(inner[1])
    td     = d["td"]
    t_axis = np.arange(len(td))
    ax_ts.plot(t_axis, td, color=col_color, lw=1.5)
    ax_ts.fill_between(t_axis, 0, td, color=col_color, alpha=0.20)
    for s in snap_steps:
        if s <= len(td):
            ax_ts.axvline(s, color="#f4a261", lw=0.7, alpha=0.6, ls="--")
    ax_ts.set_xlim(0, len(td))
    ax_ts.set_ylim(0, max(td.max() * 1.18, 1e-4))
    ax_ts.set_xlabel("Step", fontsize=9)
    ax_ts.set_ylabel("Tumor density", fontsize=9)
    ax_ts.grid(True, ls=":", alpha=0.3)
    ax_ts.set_title("Tumor density", fontsize=9, pad=4)

    # --- Final μ and r maps (inner 2-column sub-grid) ---
    final_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=inner[2], wspace=0.35,
    )
    mu_final = d["mu"][-1]
    r_final  = d["r"][-1]
    cancer_mask = (d["state"][-1] == 1)

    for ax_sub, field, field_label, cmap_ in [
        (fig.add_subplot(final_gs[0]), mu_final, "μ field", MU_CMAP),
        (fig.add_subplot(final_gs[1]), r_final,  "r field",  R_CMAP),
    ]:
        display = np.where(cancer_mask, field, np.nan)
        im = ax_sub.imshow(display, cmap=cmap_, origin="upper",
                           interpolation="nearest", aspect="equal")
        ax_sub.imshow(d["state"][-1], cmap=STATE_CMAP, norm=STATE_NORM,
                      alpha=0.22, origin="upper", interpolation="nearest",
                      aspect="equal")
        ax_sub.set_xticks([]); ax_sub.set_yticks([])
        ax_sub.set_title(f"{field_label} (t={snap_steps[-1]})",
                         fontsize=8.5, pad=3)
        cb = fig.colorbar(im, ax=ax_sub, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="#8b949e", labelsize=7)

    # Hide 4th inner row (unused)
    ax_empty = fig.add_subplot(inner[3])
    ax_empty.set_visible(False)


build_case_column(outer[0], "Health")
build_case_column(outer[1], "Tumor_Max")

fig.suptitle("Liquid Tumor — Health vs Tumor Progression",
             fontsize=15, fontweight="bold", color="#e6edf3", y=1.01)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_png = OUT_DIR / "liquid_examples_comparison.png"
out_svg = OUT_DIR / "liquid_examples_comparison.svg"
plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(out_svg,           bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved:\n  {out_png}\n  {out_svg}")
