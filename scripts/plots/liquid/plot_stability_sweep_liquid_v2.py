"""
plot_stability_sweep_liquid_v2.py
------------------------------
Phase diagram for the liquid tumor stability boundary
in P_death space across normalized division rates (rmax / r0).
Uses v2 data which features suppression of stochastic drift and finite-time artifacts.

Outputs
-------
outputs/figures/liquid/stability_sweep_liquid_v2.html
outputs/figures/liquid/stability_sweep_liquid_v2.png
outputs/figures/liquid/stability_sweep_liquid_v2.svg
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import make_smoothing_spline

from src.analysis.stats import dyn_state

# ---------------------------------------------------------------------------
# Load and process data
# ---------------------------------------------------------------------------
r0   = 0.15
N_HK = 10
N_I  = 10

data_file = Path(__file__).resolve().parents[3] / "outputs" / "results" / "stability_results_liquid_v2.csv"
if not data_file.exists():
    print(f"Results not found at {data_file}")
    sys.exit(1)

df_liquid = pd.read_csv(data_file)
if df_liquid.empty:
    print("Liquid results empty.")
    sys.exit(1)

df_liquid["rmax_norm"] = df_liquid["rmax"] / r0

XMIN = 1.0
XMAX = 7.0

# We clip rmax_norm > 1.0 for the liquid model
df_liquid_clip = df_liquid[(df_liquid["rmax_norm"] > 1.0) & (df_liquid["rmax_norm"] <= XMAX)].copy()
grp_l = df_liquid_clip.groupby("rmax_norm")["stable_dmu"]
r_unique_l = np.array(sorted(df_liquid_clip["rmax_norm"].unique()))

liquid_med_mu = grp_l.median().values * 10
liquid_med = dyn_state(0.0, 1.0, liquid_med_mu, 0.5, N_HK)[1]

if len(r_unique_l) > 3:
    spl_l = make_smoothing_spline(r_unique_l, liquid_med, lam=0.0001)
    r_smooth_l = np.linspace(min(r_unique_l), XMAX, 300)
    liquid_med_smooth = spl_l(r_smooth_l)
else:
    r_smooth_l = r_unique_l
    liquid_med_smooth = liquid_med

# ── Theoretical Liquid Boundary derived from the Unified Growth Foothold Inequality (N_I = 1) ──
theory_rmax_norm = np.linspace(XMIN, XMAX, 600)
p_star_theory_l = 0.5 * (1.0 + 1.0 / theory_rmax_norm)

# Maximum Y value for plot
YMAX = 1.0

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

# Regimes shading
ax_mpl.fill_between(r_smooth_l, liquid_med_smooth, YMAX, alpha=0.2, color="#E63946")
ax_mpl.fill_between(r_smooth_l, -0.005, liquid_med_smooth, alpha=0.1, color="#E63946")

# Theory boundary
ax_mpl.plot(theory_rmax_norm, p_star_theory_l, color="red", ls="--", lw=2.5, alpha=0.6, zorder=3, label="Theory (Liquid boundary)")

# Liquid Simulation boundary
ax_mpl.plot(r_smooth_l, liquid_med_smooth, color="#D62728", lw=3.0, zorder=6)
ax_mpl.scatter(r_unique_l, liquid_med, color="#FF9896", s=80, zorder=11, edgecolor="#D62728", linewidths=1.5, label="Sim. data")

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
# 2. Generate Interactive Figure (Plotly)
# ---------------------------------------------------------------------------
print("Generating interactive HTML figure (Plotly) ...")
fig_plotly = go.Figure()

# Add Liquid Sim boundary
fig_plotly.add_trace(go.Scatter(
    x=r_smooth_l,
    y=liquid_med_smooth,
    mode="lines",
    name="Liquid Tumor (Sim.)",
    line=dict(color="#D62728", width=3.0),
    hoverinfo="skip"
))
fig_plotly.add_trace(go.Scatter(
    x=r_unique_l,
    y=liquid_med,
    mode="markers",
    name="Sim. data",
    marker=dict(size=8, color="#FF9896", line=dict(color="#D62728", width=1.0)),
    hovertemplate="rmax/r0: %{x:.3f}<br>P_death: %{y:.5f}<extra>Liquid Sim</extra>"
))

# Add Theory Liquid Curve
fig_plotly.add_trace(go.Scatter(
    x=theory_rmax_norm,
    y=p_star_theory_l,
    mode="lines",
    name="Theory (Liquid boundary)",
    line=dict(color="red", width=2.5, dash="dash"),
    hovertemplate="rmax/r0: %{x:.3f}<br>P_death theory: %{y:.5f}<extra>Theory Liquid</extra>"
))

# Layout
fig_plotly.update_layout(
    title=dict(
        text="Phase Boundary: Critical Death Probability P<sub>death</sub>*(r<sub>max</sub>) (Liquid Model v2)",
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
        title=dict(text="Critical Death Probability P<sub>death</sub>", font=dict(size=15)),
        range=[0, YMAX],
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

# ---------------------------------------------------------------------------
# Save Outputs
# ---------------------------------------------------------------------------
out_dir = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "liquid"
out_dir.mkdir(parents=True, exist_ok=True)
root_dir = out_dir.parent

html_path = out_dir / "stability_sweep_liquid_v2.html"
png_path  = out_dir / "stability_sweep_liquid_v2.png"
svg_path  = out_dir / "stability_sweep_liquid_v2.svg"

png_path_root  = root_dir / "stability_sweep_liquid_v2.png"
svg_path_root  = root_dir / "stability_sweep_liquid_v2.svg"

# Save Matplotlib static figures
fig_mpl.savefig(png_path, dpi=200, bbox_inches="tight")
fig_mpl.savefig(svg_path, bbox_inches="tight")
fig_mpl.savefig(png_path_root, dpi=200, bbox_inches="tight")
fig_mpl.savefig(svg_path_root, bbox_inches="tight")
print(f"Saved Matplotlib static figures to:\n  - {png_path}\n  - {svg_path}\n  - {png_path_root}\n  - {svg_path_root}")
plt.close(fig_mpl)

# Save Plotly HTML figure
fig_plotly.write_html(str(html_path), include_plotlyjs="cdn")
print(f"Saved Plotly interactive figure to:\n  - {html_path}")
