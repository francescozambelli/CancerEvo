"""
plot_stability_sweep_liquid.py
------------------------------
Unified phase diagram comparing solid and liquid tumor stability boundaries
in P_death space across normalized division rates (rmax / r0).

Outputs
-------
outputs/figures/stability_sweep_liquid.html   ← interactive (open in browser)
outputs/figures/stability_sweep_liquid.png    ← static snapshot (Matplotlib)
outputs/figures/stability_sweep_liquid.svg    ← static vector (Matplotlib)
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

from src.analysis.loaders import load_all_stability_results
from src.analysis.stats import dyn_state

# ---------------------------------------------------------------------------
# Load and process data
# ---------------------------------------------------------------------------
r0   = 0.15
N_HK = 10
N_I  = 10

# Load solid sweep data
merged_solid = load_all_stability_results()
df_solid = merged_solid[merged_solid["source"] == "adaptive"]
solid_available = not df_solid.empty

# Load liquid sweep data
liquid_path = Path(__file__).resolve().parents[3] / "data" / "stability_results_liquid_adaptive.csv"
liquid_available = liquid_path.exists()

if not liquid_available:
    print(f"Liquid results not found at: {liquid_path}")
    print("Please run scripts/stability_sweep_liquid.jl first.")
    sys.exit(1)

df_liquid = pd.read_csv(liquid_path)
df_liquid["rmax_norm"] = df_liquid["rmax"] / r0

XMIN = 1.2
XMAX = 7.0

# ── Process Solid Boundary ──────────────────────────────────────────────────
if solid_available:
    df_solid_clip = df_solid[df_solid["rmax_norm"] <= XMAX].copy()
    grp_s = df_solid_clip.groupby("rmax_norm")["stable_dmu"]
    r_unique_s = np.array(sorted(df_solid_clip["rmax_norm"].unique()))
    
    solid_med_mu = grp_s.median().values * 10
    solid_med = dyn_state(0.0, 1.0, solid_med_mu, 0.5, N_HK)[1]
    
    solid_lo_mu = grp_s.quantile(0.25).values * 10
    solid_hi_mu = grp_s.quantile(0.75).values * 10
    solid_lo = dyn_state(0.0, 1.0, solid_lo_mu, 0.5, N_HK)[1]
    solid_hi = dyn_state(0.0, 1.0, solid_hi_mu, 0.5, N_HK)[1]
    
    spl_s = make_smoothing_spline(r_unique_s, solid_med, lam=0.0001)
    r_smooth_s = np.linspace(r_unique_s.min(), XMAX, 300)
    solid_med_smooth = spl_s(r_smooth_s)

# ── Process Liquid Boundary ─────────────────────────────────────────────────
# We clip rmax_norm > 1.0 for the liquid model because at rmax_norm = 1.0 (no fitness advantage),
# any positive dmu leads to extinction, so the boundary starts above 1.0.
df_liquid_clip = df_liquid[(df_liquid["rmax_norm"] > 1.0) & (df_liquid["rmax_norm"] <= XMAX)].copy()
grp_l = df_liquid_clip.groupby("rmax_norm")["stable_dmu"]
r_unique_l = np.array(sorted(df_liquid_clip["rmax_norm"].unique()))

liquid_med_mu = grp_l.median().values * 10
liquid_med = dyn_state(0.0, 1.0, liquid_med_mu, 0.5, N_HK)[1]

liquid_lo_mu = grp_l.quantile(0.25).values * 10
liquid_hi_mu = grp_l.quantile(0.75).values * 10
liquid_lo = dyn_state(0.0, 1.0, liquid_lo_mu, 0.5, N_HK)[1]
liquid_hi = dyn_state(0.0, 1.0, liquid_hi_mu, 0.5, N_HK)[1]

spl_l = make_smoothing_spline(r_unique_l, liquid_med, lam=0.0001)
r_smooth_l = np.linspace(r_unique_l.min(), XMAX, 300)
liquid_med_smooth = spl_l(r_smooth_l)

# ── Theoretical Solid Boundary ───────────────────────────────────────────────
theory_rmax_norm = np.linspace(1.001, XMAX, 600)
theory_rmax_abs  = theory_rmax_norm * r0
P_s_star         = (1.0 + r0 / theory_rmax_abs) / 2.0
mu_star          = 1.0 - P_s_star ** (1.0 / N_HK)
p_star_theory    = dyn_state(0.0, 1.0, mu_star, 0.5, N_HK)[1]

# Asymptotic Saturation
dmu_star_sat_s = float((1.0 - 0.5 ** (1.0 / N_HK)) / N_I)
mu_star_sat_s = dmu_star_sat_s * N_I
p_star_sat_s = dyn_state(0.0, 1.0, mu_star_sat_s, 0.5, N_HK)[1]

# Empirical Liquid Saturation (median of the flat asymptotic part rmax_norm >= 5)
p_star_sat_l = np.median(liquid_med[r_unique_l >= 5.0])

# Maximum Y value for plot
YMAX = 1.05

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

fig_mpl, ax_mpl = plt.subplots(figsize=(12, 7))

# Plot regimes shading
if solid_available:
    # 1. Global Collapse (above liquid boundary)
    ax_mpl.fill_between(r_smooth_l, liquid_med_smooth, YMAX, alpha=0.4, color="k")
    
    # 2. Liquid-Only Tumorigenic (between solid and liquid boundaries)
    # We interpolate the solid boundary onto the liquid rmax grid to fill cleanly
    solid_interp = np.interp(r_smooth_l, r_smooth_s, solid_med_smooth)
    ax_mpl.fill_between(r_smooth_l, solid_interp, liquid_med_smooth, alpha=0.25, color="k")
    
    # 3. Solid & Liquid Expansion (below solid boundary)
    ax_mpl.fill_between(r_smooth_s, -0.01, solid_med_smooth, alpha=0.05, color="k")

# Theory solid boundary
ax_mpl.plot(theory_rmax_norm, p_star_theory, color="blue", ls="--", lw=2.5, alpha=0.6, zorder=3, label="Theory (Solid critical boundary)")

# Solid Simulation boundary
if solid_available:
    ax_mpl.plot(r_smooth_s, solid_med_smooth, color="black", lw=2.5, zorder=5, label="Solid Tumor (Sim.)")
    ax_mpl.scatter(r_unique_s, solid_med, color="gray", s=80, zorder=10, edgecolor="black", linewidths=1.5)

# Liquid Simulation boundary
ax_mpl.plot(r_smooth_l, liquid_med_smooth, color="#D62728", lw=3.0, zorder=6, label="Liquid Tumor (Sim.)")
ax_mpl.scatter(r_unique_l, liquid_med, color="#FF9896", s=80, zorder=11, edgecolor="#D62728", linewidths=1.5)

# Asymptotic saturations
#ax_mpl.axhline(p_star_sat_s, color="orange", ls="--", lw=1.5, alpha=0.6, zorder=3)
#ax_mpl.text(XMAX * 0.98, p_star_sat_s + 0.005, 
#            f"$P_{{death, \\infty}}^* \\approx {p_star_sat_s:.3f}$ (Solid)", 
#            color="orange", ha="right", va="bottom", fontsize=11)

#ax_mpl.axhline(p_star_sat_l, color="purple", ls="--", lw=1.5, alpha=0.6, zorder=3)
#ax_mpl.text(XMAX * 0.98, p_star_sat_l + 0.005, 
#            f"$P_{{death, \\infty}}^* \\approx {p_star_sat_l:.3f}$ (Liquid)", 
#            color="purple", ha="right", va="bottom", fontsize=11)

# Annotate regimes (centered in white boxes)
fontsize_reg = 18
ax_mpl.text(3.0, 0.92, "Global Collapse)",
            bbox=dict(facecolor="white", edgecolor="#E63946", boxstyle="round,pad=0.3", alpha=0.9),
            fontsize=fontsize_reg, ha="center")
ax_mpl.text(4.0, 0.65, "Liquid Expansion / Solid Collapse",
            bbox=dict(facecolor="white", edgecolor="#E9C46A", boxstyle="round,pad=0.3", alpha=0.9),
            fontsize=fontsize_reg, ha="center")
ax_mpl.text(3.0, 0.15, "Solid & Liquid Expansion",
            bbox=dict(facecolor="white", edgecolor="#2A9D8F", boxstyle="round,pad=0.3", alpha=0.9),
            fontsize=fontsize_reg, ha="center")

# Labels & Bounds
fontsize_lab = 20
fontsize_title = 25
ax_mpl.set_xlabel(r"Normalized Division Rate ($r_{\mathrm{max}} / r_0$)", fontsize=fontsize_lab)
ax_mpl.set_ylabel(r"Death Probability ($P_{\mathrm{death}}$)", fontsize=fontsize_lab)
#ax_mpl.set_title(r"Phase Boundary Comparison: Solid vs. Liquid Tumor Stability", 
#                 fontsize=fontsize_title, fontweight="bold", pad=15)

ax_mpl.set_xlim(XMIN, XMAX)
ax_mpl.set_ylim(-0.005, YMAX)
ax_mpl.yaxis.grid(True, ls=":", alpha=0.5)
ax_mpl.legend(fontsize=18, loc="lower right", framealpha=0.9)

plt.tight_layout()

# ---------------------------------------------------------------------------
# 2. Generate Interactive Figure (Plotly)
# ---------------------------------------------------------------------------
print("Generating interactive HTML figure (Plotly) ...")
fig_plotly = go.Figure()

# Add Solid Sim boundary
if solid_available:
    fig_plotly.add_trace(go.Scatter(
        x=r_smooth_s,
        y=solid_med_smooth,
        mode="lines",
        name="Solid Tumor (Sim.)",
        line=dict(color="black", width=2.5),
        hoverinfo="skip"
    ))
    fig_plotly.add_trace(go.Scatter(
        x=r_unique_s,
        y=solid_med,
        mode="markers",
        name="Solid Data Points",
        marker=dict(size=6, color="gray", line=dict(color="black", width=0.8)),
        hovertemplate="rmax/r0: %{x:.3f}<br>P_death: %{y:.5f}<extra>Solid Sim</extra>"
    ))

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
    name="Liquid Data Points",
    marker=dict(size=7, color="#FF9896", line=dict(color="#D62728", width=1.0)),
    hovertemplate="rmax/r0: %{x:.3f}<br>P_death: %{y:.5f}<extra>Liquid Sim</extra>"
))

# Add Theory Solid Curve
fig_plotly.add_trace(go.Scatter(
    x=theory_rmax_norm,
    y=p_star_theory,
    mode="lines",
    name="Theory (Solid boundary)",
    line=dict(color="blue", width=1.5, dash="dash"),
    hovertemplate="rmax/r0: %{x:.3f}<br>P_death theory: %{y:.5f}<extra>Theory Solid</extra>"
))

# Add Saturation Lines
fig_plotly.add_hline(
    y=p_star_sat_s,
    line=dict(color="orange", width=1.5, dash="dash"),
    annotation_text=f"Solid Saturation ≈ {p_star_sat_s:.3f}",
    annotation_position="bottom right",
    annotation_font=dict(size=11, color="orange")
)
fig_plotly.add_hline(
    y=p_star_sat_l,
    line=dict(color="purple", width=1.5, dash="dash"),
    annotation_text=f"Liquid Saturation ≈ {p_star_sat_l:.3f}",
    annotation_position="top right",
    annotation_font=dict(size=11, color="purple")
)

# Layout
fig_plotly.update_layout(
    title=dict(
        text="Phase Boundary Comparison: Solid vs. Liquid Tumor Stability",
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

html_path = out_dir / "stability_sweep_liquid.html"
png_path  = out_dir / "stability_sweep_liquid.png"
svg_path  = out_dir / "stability_sweep_liquid.svg"

# Save Matplotlib static figures
fig_mpl.savefig(png_path, dpi=200, bbox_inches="tight")
fig_mpl.savefig(svg_path, bbox_inches="tight")
print(f"Saved Matplotlib static figures to:\n  - {png_path}\n  - {svg_path}")
plt.close(fig_mpl)

# Save Plotly HTML figure
fig_plotly.write_html(str(html_path), include_plotlyjs="cdn")
print(f"Saved Plotly interactive figure to:\n  - {html_path}")
