"""
plot_stability_phase_diagram_Ps_liquid.py
-----------------------------------------
Extracts the transition boundary from the 2D parameter grid data (from
stability_phase_diagram_results_liquid.csv) and plots it as a continuous line
on the (rmax_norm, Ps) plane, matching the styling of plot_stability_sweep_liquid.py.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import make_smoothing_spline

# ---------------------------------------------------------------------------
# Load and process data
# ---------------------------------------------------------------------------
r0   = 0.15
N_HK = 10
N_I  = 10
c    = 10

project_root = Path(__file__).resolve().parents[3]
csv_path = project_root / "data" / "stability_phase_diagram_results_liquid.csv"
if not csv_path.exists():
    print(f"Error: CSV file not found at {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

# Map dmu to Ps
df["Ps"] = (1.0 - c * df["dmu"]) ** N_HK

# Extract the boundary for each rmax_norm where fraction crosses 0.5
boundary_pts = []
for r, group in df.groupby("rmax_norm"):
    group = group.sort_values("Ps")
    ps_arr = group["Ps"].values
    frac_arr = group["fraction"].values
    
    # We want to find where frac_arr crosses 0.5 (goes from <0.5 to >=0.5 as Ps increases)
    if np.all(frac_arr < 0.5):
        boundary_pts.append((r, 1.0))
    elif np.all(frac_arr > 0.5):
        boundary_pts.append((r, ps_arr[0]))
    else:
        # Find first index where fraction >= 0.5
        idx = np.where(frac_arr >= 0.5)[0][0]
        if idx == 0:
            boundary_pts.append((r, ps_arr[0]))
        else:
            p1, f1 = ps_arr[idx-1], frac_arr[idx-1]
            p2, f2 = ps_arr[idx], frac_arr[idx]
            if f2 == f1:
                p_star = (p1 + p2) / 2
            else:
                p_star = p1 + (0.5 - f1) * (p2 - p1) / (f2 - f1)
            boundary_pts.append((r, p_star))

boundary_pts = np.array(boundary_pts)
r_unique_l = boundary_pts[:, 0]
liquid_med = boundary_pts[:, 1]

XMIN = 1.0
XMAX = 7.0
YMAX = 1.0

# Clip range
mask = (r_unique_l >= XMIN) & (r_unique_l <= XMAX)
r_unique_l = r_unique_l[mask]
liquid_med = liquid_med[mask]

# Sort
sort_idx = np.argsort(r_unique_l)
r_unique_l = r_unique_l[sort_idx]
liquid_med = liquid_med[sort_idx]

# Smooth curve
if len(r_unique_l) > 3:
    spl_l = make_smoothing_spline(r_unique_l, liquid_med, lam=0.05)
    r_smooth_l = np.linspace(XMIN, XMAX, 300)
    liquid_med_smooth = spl_l(r_smooth_l)
else:
    r_smooth_l = r_unique_l
    liquid_med_smooth = liquid_med

# Theoretical Liquid Boundary
theory_rmax_norm = np.linspace(XMIN, XMAX, 600)
p_star_theory_l = 0.5 * (1.0 + 1.0 / theory_rmax_norm)

# ---------------------------------------------------------------------------
# 1. Generate Static Figure (Matplotlib)
# ---------------------------------------------------------------------------
print("Generating static figures (Matplotlib) ...")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig_mpl, ax_mpl = plt.subplots(figsize=(9, 7))

# Regimes shading (matching plot_stability_sweep_liquid.py styling)
ax_mpl.fill_between(r_smooth_l, liquid_med_smooth, YMAX, alpha=0.2, color="#E63946")
ax_mpl.fill_between(r_smooth_l, -0.005, liquid_med_smooth, alpha=0.1, color="#E63946")

# Theory boundary
ax_mpl.plot(theory_rmax_norm, p_star_theory_l, color="red", ls="--", lw=2.5, alpha=0.6, zorder=3, label="Theory (Liquid boundary)")

# Liquid Simulation boundary (without scatter points)
ax_mpl.plot(r_smooth_l, liquid_med_smooth, color="#D62728", lw=3.0, zorder=6, label="Liquid Tumor (Sim.)")

# Annotate regimes (centered in white boxes)
fontsize_reg = 22
ax_mpl.text(3.0, 0.5, "Tumor Collapse",
            bbox=dict(facecolor="white", edgecolor="#E63946", boxstyle="round,pad=0.3", alpha=0.9),
            fontsize=fontsize_reg, ha="center")
ax_mpl.text(5.0, 0.8, "Tumor Expansion",
            bbox=dict(facecolor="white", edgecolor="#E63946", boxstyle="round,pad=0.3", alpha=0.9),
            fontsize=fontsize_reg, ha="center")

# Labels & Bounds
fontsize_lab = 25
fontsize_ticks = 23
fontsize_legend = 20

ax_mpl.set_xlabel(r"Normalized Division Rate ($r_{\mathrm{max}} / r_0$)", fontsize=fontsize_lab)
ax_mpl.set_ylabel(r"Survival Probability ($P_s$)", fontsize=fontsize_lab)
ax_mpl.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax_mpl.grid(False)

ax_mpl.set_xlim(1.0, XMAX)
ax_mpl.set_ylim(0.3, YMAX)
ax_mpl.legend(fontsize=fontsize_legend, loc="lower right", framealpha=0.9)

plt.tight_layout()

# ---------------------------------------------------------------------------
# 2. Save Outputs
# ---------------------------------------------------------------------------
output_dir = project_root / "outputs" / "figures" / "liquid"
output_dir.mkdir(parents=True, exist_ok=True)
root_dir = output_dir.parent

html_path = output_dir / "stability_phase_diagram_Ps_liquid.html"
png_path  = output_dir / "stability_phase_diagram_Ps_liquid.png"
svg_path  = output_dir / "stability_phase_diagram_Ps_liquid.svg"

png_path_root  = root_dir / "stability_phase_diagram_Ps_liquid.png"
svg_path_root  = root_dir / "stability_phase_diagram_Ps_liquid.svg"

# Save Matplotlib static figures
fig_mpl.savefig(png_path, dpi=200, bbox_inches="tight")
fig_mpl.savefig(svg_path, bbox_inches="tight")
fig_mpl.savefig(png_path_root, dpi=200, bbox_inches="tight")
fig_mpl.savefig(svg_path_root, bbox_inches="tight")
print(f"Plot saved successfully to:\n  - {png_path}\n  - {svg_path}\n  - {png_path_root}\n  - {svg_path_root}")
plt.close(fig_mpl)

# ---------------------------------------------------------------------------
# 3. Generate Interactive Figure (Plotly)
# ---------------------------------------------------------------------------
print("Generating interactive HTML figure (Plotly) ...")
fig_plotly = go.Figure()

# Add Liquid Sim boundary (without markers)
fig_plotly.add_trace(go.Scatter(
    x=r_smooth_l,
    y=liquid_med_smooth,
    mode="lines",
    name="Liquid Tumor (Sim.)",
    line=dict(color="#D62728", width=3.0),
    hovertemplate="rmax/r0: %{x:.3f}<br>P_s: %{y:.5f}<extra>Liquid Sim</extra>"
))

# Add Theory Liquid Curve
fig_plotly.add_trace(go.Scatter(
    x=theory_rmax_norm,
    y=p_star_theory_l,
    mode="lines",
    name="Theory (Liquid boundary)",
    line=dict(color="red", width=2.5, dash="dash"),
    hovertemplate="rmax/r0: %{x:.3f}<br>P_s theory: %{y:.5f}<extra>Theory Liquid</extra>"
))

fig_plotly.update_layout(
    title=dict(
        text="Phase Boundary: Critical Survival Probability P<sub>s</sub>*(r<sub>max</sub>) (Liquid Model)",
        font=dict(size=18),
        x=0.5,
    ),
    xaxis=dict(
        title=dict(text="Normalized Division Rate r<sub>max</sub> / r<sub>0</sub>", font=dict(size=15)),
        range=[1.0, XMAX],
        showgrid=True,
        gridcolor="#e0e0e0",
    ),
    yaxis=dict(
        title=dict(text="Critical Survival Probability P<sub>s</sub>", font=dict(size=15)),
        range=[0.3, YMAX],
        showgrid=True,
        gridcolor="#e0e0e0",
    ),
    legend=dict(
        font=dict(size=12),
        bordercolor="#cccccc",
        borderwidth=1,
    ),
    plot_bgcolor="white",
    hovermode="closest",
    width=950,
    height=600,
)

fig_plotly.write_html(str(html_path), include_plotlyjs="cdn")
print(f"Saved Plotly interactive figure to:\n  - {html_path}")
