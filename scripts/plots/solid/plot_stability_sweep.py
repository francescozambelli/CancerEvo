"""
plot_stability_sweep.py
-----------------------
Unified phase diagram: all stability datasets overlaid, with
adaptive boundary median + IQR ribbon, and analytic theory curve.

Saves a beautiful, publication-quality static figure (PNG/SVG) using Matplotlib
to match outputs/figures/phase_diagram.png, and an interactive HTML figure (Plotly).

Outputs
-------
outputs/figures/solid/stability_sweep.html   ← interactive (open in browser)
outputs/figures/solid/stability_sweep.png    ← static snapshot (Matplotlib)
outputs/figures/solid/stability_sweep.svg    ← static vector (Matplotlib)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import make_smoothing_spline

from src.analysis.loaders import load_stability_results_solid
from src.analysis.stats import dyn_state

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
merged    = load_stability_results_solid()

dfa              = merged
adaptive_available = not dfa.empty

XMIN = 1.0
XMAX = 7.0
r0   = 0.15
N_HK = 10
N_I  = 10

# Maximum death probability to plot is 0.6
YMAX = 1

# ── Adaptive boundary: per-rmax_norm statistics ──────────────────────────────
if adaptive_available:
    dfa_clip = dfa[dfa["rmax_norm"] <= XMAX].copy()
    grp = dfa_clip.groupby("rmax_norm")["stable_dmu"]
    r_unique  = np.array(sorted(dfa_clip["rmax_norm"].unique()))
    adapt_med_mu = grp.median().values * 10
    
    # Map raw mutation rate to p_death space using dyn_state with k=0.5
    adapt_med = dyn_state(0.0, 1.0, adapt_med_mu, 0.5, N_HK)[1]
    
    adapt_lo_mu  = grp.quantile(0.25).values * 10
    adapt_hi_mu  = grp.quantile(0.75).values * 10
    adapt_rmax_raw = dfa_clip["rmax_norm"].values
    adapt_dmu_raw_mu  = dfa_clip["stable_dmu"].values * 10
    
    adapt_lo = dyn_state(0.0, 1.0, adapt_lo_mu, 0.5, N_HK)[1]
    adapt_hi = dyn_state(0.0, 1.0, adapt_hi_mu, 0.5, N_HK)[1]
    adapt_dmu_raw = dyn_state(0.0, 1.0, adapt_dmu_raw_mu, 0.5, N_HK)[1]

    # Fit a smoothing spline on the median points to draw a smooth boundary
    spl = make_smoothing_spline(r_unique, adapt_med, lam=0.005)
    r_smooth = np.linspace(XMIN, XMAX, 300)
    adapt_med_smooth = spl(r_smooth)

# ---------------------------------------------------------------------------
# Theoretical prediction (corrected)
# ---------------------------------------------------------------------------
theory_rmax_norm = np.linspace(XMIN, XMAX, 600)
theory_rmax_abs  = theory_rmax_norm * r0
P_s_star         = (1.0 + r0 / theory_rmax_abs) / 2.0
mu_star          = 1.0 - P_s_star ** (1.0 / N_HK)
p_star_theory    = dyn_state(0.0, 1.0, mu_star, 0.5, N_HK)[1]

# Asymptotic saturation
dmu_star_sat = float((1.0 - 0.5 ** (1.0 / N_HK)) / N_I)
mu_star_sat = dmu_star_sat * N_I
p_star_sat = dyn_state(0.0, 1.0, mu_star_sat, 0.5, N_HK)[1]


# ---------------------------------------------------------------------------
# 1. Generate Static Figure (Matplotlib) - Premium publication styling
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
if adaptive_available:
    ax_mpl.fill_between(r_smooth, adapt_med_smooth, YMAX, alpha=0.2, color="#1F77B4")
    ax_mpl.fill_between(r_smooth, -0.005, adapt_med_smooth, alpha=0.1, color="#1F77B4")

# Theory critical boundary curve (Blue dashed line)
ax_mpl.plot(theory_rmax_norm, p_star_theory, color="blue", ls="--", lw=2.5, alpha=0.6, zorder=3, label="Theory (Solid boundary)")

# Simulation boundary line (Solid black line connecting smooth spline)
if adaptive_available:
    ax_mpl.plot(r_smooth, adapt_med_smooth, color="navy", lw=3.0, zorder=5, label= "Sim. data")#, label="Solid Tumor (Sim.)")
    
    # Simulation data points (Gray markers)
    #ax_mpl.scatter(r_unique, adapt_med, color="lightblue", s=80, zorder=10, edgecolor="navy", linewidths=1.5, label="Sim. data")

# Asymptotic saturation
# ax_mpl.axhline(p_star_sat, color="orange", ls="--", lw=1.5, alpha=0.6, zorder=3)
# ax_mpl.text(XMAX * 0.98, p_star_sat + 0.01, 
#             r"$P_{\rm death, \infty}^* \approx " + f"{p_star_sat:.3f}$", 
#             color="orange", ha="right", va="bottom", fontsize=11)

# Annotate regimes (centered in white boxes matching phase_diagram.png)
fontsize_reg = 22
ax_mpl.text(3.0, 0.5, "Tumor Collapse",
            bbox=dict(facecolor="white", edgecolor="#1F77B4", boxstyle="round,pad=0.3", alpha=0.9),
            fontsize=fontsize_reg, ha="center")
ax_mpl.text(5.0, 0.8, "Tumor Expansion",
            bbox=dict(facecolor="white", edgecolor="#1F77B4", boxstyle="round,pad=0.3", alpha=0.9),
            fontsize=fontsize_reg, ha="center")

# Labels & Bounds
fontsize_lab = 25
fontsize_ticks = 23
fontsize_legend = 20

ax_mpl.set_xlabel(r"Normalized Division Rate ($r_{\mathrm{max}} / r_0$)", fontsize=fontsize_lab)
ax_mpl.set_ylabel(r"Survival Probability ($P_s$)", fontsize=fontsize_lab)
ax_mpl.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

ax_mpl.set_xlim(1.0, XMAX)
ax_mpl.set_ylim(0.3, YMAX)
ax_mpl.legend(fontsize=fontsize_legend, loc="lower right", framealpha=0.9)

plt.tight_layout()


# ---------------------------------------------------------------------------
# 2. Generate Interactive Figure (Plotly)
# ---------------------------------------------------------------------------
print("Generating interactive HTML figure (Plotly) ...")
fig_plotly = go.Figure()

# Simulation raw points + median boundary line
if adaptive_available:
    # Median data points (Gray)
    fig_plotly.add_trace(go.Scatter(
        x=r_unique,
        y=adapt_med,
        mode="markers",
        name="Sim. data",
        marker=dict(size=8, color="gray", line=dict(color="black", width=1.0)),
        hovertemplate="rmax/r0: %{x:.3f}<br>P_death: %{y:.5f}<extra>Sim. data</extra>",
    ))
    # Median boundary line (black, smooth)
    fig_plotly.add_trace(go.Scatter(
        x=r_smooth,
        y=adapt_med_smooth,
        mode="lines",
        name="Solid Tumor (Sim.)",
        line=dict(color="black", width=3.0),
        hoverinfo="skip",
    ))

# Theory critical boundary (Blue dashed line)
fig_plotly.add_trace(go.Scatter(
    x=theory_rmax_norm,
    y=p_star_theory,
    mode="lines",
    name="Theory (Solid boundary)",
    line=dict(color="blue", width=2.5, dash="dash"),
    hovertemplate="rmax/r0: %{x:.3f}<br>P_death theory: %{y:.5f}<extra>Theory</extra>",
))

# Saturation line (Orange)
fig_plotly.add_hline(
    y=p_star_sat,
    line=dict(color="rgba(255,165,0,0.45)", width=1.5, dash="dash"),
    annotation_text=f"Solid Saturation ≈ {p_star_sat:.3f}",
    annotation_position="top right",
    annotation_font=dict(size=11, color="orange"),
)

# Layout matching Plotly style with new colors and limits
fig_plotly.update_layout(
    title=dict(
        text="Phase Boundary: Critical Death Probability P<sub>death</sub>*(r<sub>max</sub>) (Solid Model)",
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
out_dir = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)
root_dir = out_dir.parent

html_path = out_dir / "stability_sweep.html"
png_path  = out_dir / "stability_sweep.png"
svg_path  = out_dir / "stability_sweep.svg"

png_path_root  = root_dir / "stability_sweep.png"
svg_path_root  = root_dir / "stability_sweep.svg"

# Save Matplotlib figures
fig_mpl.savefig(png_path, dpi=200, bbox_inches="tight")
fig_mpl.savefig(svg_path, bbox_inches="tight")
fig_mpl.savefig(png_path_root, dpi=200, bbox_inches="tight")
fig_mpl.savefig(svg_path_root, bbox_inches="tight")
print(f"Saved Matplotlib static figures to:\n  - {png_path}\n  - {svg_path}\n  - {png_path_root}\n  - {svg_path_root}")
plt.close(fig_mpl)

# Save Plotly HTML figure
fig_plotly.write_html(str(html_path), include_plotlyjs="cdn")
print(f"Saved Plotly interactive figure to:\n  - {html_path}")
