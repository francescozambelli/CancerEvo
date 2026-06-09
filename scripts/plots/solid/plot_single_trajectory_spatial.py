# scripts/plots/plot_single_trajectory_spatial.py
#
# Visualize the spatiotemporal progression of solid tumor simulations (Health and Tumor).
# Loads spatial snapshots and renders:
#   - Snapshot grid of L×L lattice frames showing cell states (WT / Cancer / Dead) over time
#   - Tumor density time series with vertical markers at snapshot times
#   - Spatial map of mutation rate μ at the final snapshot
#   - Spatial map of division rate r at the final snapshot
#
# Outputs:
#   - outputs/figures/single_trajectory_spatial_health.png / .svg
#   - outputs/figures/single_trajectory_spatial_tumor.png / .svg

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "simulations" / "spatial_run"
OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "solid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style (Light Theme for Solid Tumor)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.labelcolor": "black",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "black",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#eeeeee",
})

# Colors for cell states (WT: Pale Green, Cancer: Red, Dead: Charcoal)
STATE_COLORS = ["#E9F5EC", "#E63946", "#3A3A3A"]
STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NORM = BoundaryNorm([0, 0.5, 1.5, 2.5], STATE_CMAP.N)

MU_CMAP = "plasma"
R_CMAP = "viridis"

def plot_spatial_progression(scenario_name, title_prefix, zoom=False):
    snap_path = DATA_DIR / f"{scenario_name}_snapshots.npz"
    results_path = DATA_DIR / f"{scenario_name}_results.npz"
    
    if not snap_path.exists() or not results_path.exists():
        print(f"Error: files for scenario '{scenario_name}' not found.")
        return
        
    print(f"Loading {scenario_name} simulation from {snap_path}...")
    snaps = np.load(snap_path, allow_pickle=True)
    results = np.load(results_path)
    
    state_arr = snaps["state"]           # (n_frames, L, L)
    mu_arr = snaps["mu"]                 # (n_frames, L, L)
    r_arr = snaps["r"]                   # (n_frames, L, L)
    snap_steps = snaps["snapshot_steps"] # (n_frames,)
    L = int(snaps["L"][0])
    
    # Handle outcome format
    outcome_raw = snaps["outcome"]
    if isinstance(outcome_raw, np.ndarray) and outcome_raw.dtype.kind in ('U', 'S'):
        outcome = str(outcome_raw[0])
    elif hasattr(outcome_raw, 'tobytes'):
        outcome = outcome_raw.tobytes().decode()
    else:
        outcome = str(outcome_raw)
        
    n_frames = state_arr.shape[0]
    print(f"  Loaded {n_frames} frames | L={L} | Outcome: {outcome}")
    
    # Subsample to at most 6 frames to keep the layout extremely clean
    MAX_FRAMES = 6
    if n_frames > MAX_FRAMES:
        idx = np.round(np.linspace(0, n_frames - 1, MAX_FRAMES)).astype(int)
        state_arr = state_arr[idx]
        mu_arr = mu_arr[idx]
        r_arr = r_arr[idx]
        snap_steps = snap_steps[idx]
        n_frames = MAX_FRAMES
        print(f"  Subsampled to {n_frames} frames for display.")
        
    # Calculate bounding box zoom if requested
    if zoom:
        non_wt_mask = (state_arr != 0)
        if np.any(non_wt_mask):
            coords = np.argwhere(non_wt_mask)
            rows = coords[:, 1]
            cols = coords[:, 2]
            
            margin = 10
            r_min = max(0, rows.min() - margin)
            r_max = min(L - 1, rows.max() + margin)
            c_min = max(0, cols.min() - margin)
            c_max = min(L - 1, cols.max() + margin)
        else:
            r_min, r_max = L // 2 - 25, L // 2 + 25
            c_min, c_max = L // 2 - 25, L // 2 + 25
        print(f"  Zooming to bounding box: rows [{r_min}, {r_max}], cols [{c_min}, {c_max}]")

    td = results["tumor_density"]
    t_axis = np.arange(len(td))
    
    # Grid layout
    n_cols = min(n_frames, 6)
    n_rows = int(np.ceil(n_frames / n_cols))
    
    fig = plt.figure(figsize=(3.2 * n_cols, 3.2 * n_rows + 4.5), facecolor="white")
    
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[n_rows * 3.2, 4.0],
        hspace=0.35,
    )
    
    # Top: Snapshots
    snap_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=outer[0],
        hspace=0.08, wspace=0.06,
    )
    
    # Bottom: density + final mu + final r
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        wspace=0.35,
    )
    
    ax_ts = fig.add_subplot(bottom_gs[0])
    ax_mu = fig.add_subplot(bottom_gs[1])
    ax_r  = fig.add_subplot(bottom_gs[2])
    
    # Draw snapshot panels
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
            spine.set_edgecolor("#dddddd")
            
        canc_frac = (state_arr[f] == 1).mean()
        label = f"t = {snap_steps[f]}\n{100*canc_frac:.1f}% cancer"
        ax.set_title(label, fontsize=10, fontweight="bold", pad=4)
        
        if zoom:
            ax.set_xlim(c_min, c_max)
            ax.set_ylim(r_max, r_min)
    
    
    # Bottom Panel 1: Time Series
    ax_ts.plot(t_axis, td, color="#e63946", lw=2, label="Tumor density")
    ax_ts.fill_between(t_axis, 0, td, color="#e63946", alpha=0.15)
    
    # Mark snapshot times
    for s in snap_steps:
        if s <= len(td):
            ax_ts.axvline(s, color="#d97706", lw=1.0, alpha=0.8, ls="--")
            
    ax_ts.set_xlabel("Simulation step", fontsize=11)
    ax_ts.set_ylabel("Tumor density", fontsize=11)
    ax_ts.set_ylim(0, min(1.0, td.max() * 1.15 + 0.05))
    ax_ts.set_xlim(0, len(td))
    ax_ts.grid(True, ls=":", alpha=0.5)
    ax_ts.set_title("Tumor density over time", fontsize=11, fontweight="bold", pad=8)
    
    # Fix the Outcome Cleared/Progressing mapping
    outcome_label = "Cleared" if outcome == "Health" else "Progressing"
    outcome_color = "#2a9d8f" if outcome == "Health" else "#e63946"
    ax_ts.text(
        0.95, 0.90, f"Outcome: {outcome_label}",
        transform=ax_ts.transAxes,
        ha="right", va="top", fontsize=10,
        color=outcome_color, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor=outcome_color, boxstyle="round,pad=0.2", alpha=0.9)
    )
    
    # Bottom Panel 2: Final mu field
    mu_final = mu_arr[-1]
    cancer_mask = (state_arr[-1] == 1)
    mu_display = np.where(cancer_mask, mu_final, np.nan)
    
    # Underlay WT / dead in light grey/beige
    ax_mu.imshow(
        state_arr[-1],
        cmap=STATE_CMAP, norm=STATE_NORM,
        alpha=0.15, origin="upper",
        interpolation="nearest", aspect="equal",
    )
    im_mu = ax_mu.imshow(
        mu_display,
        cmap=MU_CMAP, origin="upper",
        interpolation="nearest", aspect="equal",
    )
    ax_mu.set_xticks([]); ax_mu.set_yticks([])
    ax_mu.set_title(fr"$\mu$ field (t = {snap_steps[-1]})", fontsize=11, fontweight="bold", pad=8)
    cbar_mu = fig.colorbar(im_mu, ax=ax_mu, fraction=0.046, pad=0.04)
    cbar_mu.set_label(r"Mutation rate $\mu$", fontsize=10)
    cbar_mu.ax.tick_params(labelsize=9)
    
    if zoom:
        ax_mu.set_xlim(c_min, c_max)
        ax_mu.set_ylim(r_max, r_min)
    
    # Bottom Panel 3: Final r field
    r_final = r_arr[-1]
    r_display = np.where(cancer_mask, r_final, np.nan)
    
    ax_r.imshow(
        state_arr[-1],
        cmap=STATE_CMAP, norm=STATE_NORM,
        alpha=0.15, origin="upper",
        interpolation="nearest", aspect="equal",
    )
    im_r = ax_r.imshow(
        r_display,
        cmap=R_CMAP, origin="upper",
        interpolation="nearest", aspect="equal",
    )
    ax_r.set_xticks([]); ax_r.set_yticks([])
    ax_r.set_title(f"$r$ field (t = {snap_steps[-1]})", fontsize=11, fontweight="bold", pad=8)
    cbar_r = fig.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
    cbar_r.set_label("Division rate $r$", fontsize=10)
    cbar_r.ax.tick_params(labelsize=9)
    
    if zoom:
        ax_r.set_xlim(c_min, c_max)
        ax_r.set_ylim(r_max, r_min)
    
    # Main Title
    fig.suptitle(
        f"{title_prefix} – Spatial Progression",
        fontsize=15, fontweight="bold", y=0.98,
    )
    
    # Save
    out_png = OUT_DIR / f"single_trajectory_spatial_{scenario_name}.png"
    out_svg = OUT_DIR / f"single_trajectory_spatial_{scenario_name}.svg"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    print(f"Saved figures to:\n  - {out_png}\n  - {out_svg}")
    plt.close(fig)

if __name__ == "__main__":
    plot_spatial_progression("health", "Health Trajectory", zoom=True)
    plot_spatial_progression("tumor", "Tumor Trajectory", zoom=False)
