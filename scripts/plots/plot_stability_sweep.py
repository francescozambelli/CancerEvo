"""
plot_stability_sweep.py
-----------------------
Unified phase diagram: all stability datasets overlaid, with
adaptive boundary median + IQR ribbon, and analytic theory curve.

Saves a beautiful, publication-quality static figure (PNG/SVG) using Matplotlib
to match outputs/figures/phase_diagram.png, and an interactive HTML figure (Plotly).

Outputs
-------
outputs/figures/stability_sweep.html   ← interactive (open in browser)
outputs/figures/stability_sweep.png    ← static snapshot (Matplotlib)
outputs/figures/stability_sweep.svg    ← static vector (Matplotlib)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import make_smoothing_spline

from src.analysis.loaders import load_all_stability_results

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
merged    = load_all_stability_results()

dfa              = merged[merged["source"] == "adaptive"]
adaptive_available = not dfa.empty

XMAX = 7.0   # rmax/r0 cut-off (updated to 7.0)
YMAX = 0.072  # maximum mutation rate to plot

# ── Adaptive boundary: per-rmax_norm statistics ──────────────────────────────
if adaptive_available:
    dfa_clip = dfa[dfa["rmax_norm"] <= XMAX].copy()
    grp = dfa_clip.groupby("rmax_norm")["stable_dmu"]
    r_unique  = np.array(sorted(dfa_clip["rmax_norm"].unique()))
    adapt_med = grp.median().values * 10
    adapt_lo  = grp.quantile(0.25).values * 10
    adapt_hi  = grp.quantile(0.75).values * 10
    adapt_rmax_raw = dfa_clip["rmax_norm"].values
    adapt_dmu_raw  = dfa_clip["stable_dmu"].values * 10

    # Fit a smoothing spline on the median points to draw a smooth boundary
    # lam=0.0001 handles the small deviations to give a clean smooth curve
    spl = make_smoothing_spline(r_unique, adapt_med, lam=0.0001)
    r_smooth = np.linspace(r_unique.min(), XMAX, 300)
    adapt_med_smooth = spl(r_smooth)

# ---------------------------------------------------------------------------
# Theoretical prediction (corrected)
# ---------------------------------------------------------------------------
r0   = 0.15
N_I  = 10
N_HK = 10

theory_rmax_norm = np.linspace(1.001, XMAX, 600)
theory_rmax_abs  = theory_rmax_norm * r0
P_s_star         = (1.0 + r0 / theory_rmax_abs) / 2.0
mu_star          = 1.0 - P_s_star ** (1.0 / N_HK)
dmu_star         = np.clip(mu_star / N_I, 0.0, None) * 10  # Already scaled by 10 (mu_star)

# Asymptotic saturation (multiplied by N_I to get mu saturation)
dmu_star_sat = float((1.0 - 0.5 ** (1.0 / N_HK)) / N_I)
mu_star_sat = dmu_star_sat * N_I


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

fig_mpl, ax_mpl = plt.subplots(figsize=(8.5, 6))

# Theory critical boundary curve (Blue dashed line)
ax_mpl.plot(theory_rmax_norm, dmu_star, color="blue", ls="--", lw=1.5, alpha=0.7, zorder=3, label="Theory (Critical boundary)")

# Simulation boundary line (Solid black line connecting smooth spline)
if adaptive_available:
    ax_mpl.plot(r_smooth, adapt_med_smooth, color="black", lw=2.5, zorder=5, label="Tumorigenic boundary (Sim.)")
    
    # Color-filled regimes (Healthy above sweep boundary, Tumor below sweep boundary)
    ax_mpl.fill_between(r_smooth, adapt_med_smooth, YMAX, alpha=0.12, color="#2A9D8F", label="Healthy regime")
    ax_mpl.fill_between(r_smooth, -0.002, adapt_med_smooth, alpha=0.12, color="#E63946", label="Tumor regime")
    
    # Simulation data points (Gray markers)
    ax_mpl.scatter(r_unique, adapt_med, color="gray", s=50, zorder=10, edgecolor="black", linewidths=1.0, label="Sim. data")

# Asymptotic saturation
ax_mpl.axhline(mu_star_sat, color="orange", ls="--", lw=1.5, alpha=0.6, zorder=3)
ax_mpl.text(XMAX * 0.98, mu_star_sat + 0.001, 
            r"$\mu^*_{\infty} = 1 - 0.5^{1/N_{\mathrm{HK}}} \approx 0.067$", 
            color="orange", ha="right", va="bottom", fontsize=11)

# Annotate regimes (centered in white boxes matching phase_diagram.png)
ax_mpl.text(2.0, 0.055, "Tumor Collapse",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", alpha=0.9),
            fontsize=12, ha="center")
ax_mpl.text(5.0, 0.008, "Tumor Expansion",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", alpha=0.9),
            fontsize=12, ha="center")

# Labels & Bounds
ax_mpl.set_xlabel(r"$r_{\mathrm{max}} / r_0$", fontsize=16)
ax_mpl.set_ylabel(r"Phase boundary $\mu^*$", fontsize=16)
ax_mpl.set_title(r"Phase Boundary: Critical Mutation Rate $\mu^*(r_{\mathrm{max}})$", 
                 fontsize=15, fontweight="bold", pad=15)

ax_mpl.set_xlim(1.0, XMAX)
ax_mpl.set_ylim(-0.002, YMAX)
ax_mpl.yaxis.grid(True, ls=":", alpha=0.5)
ax_mpl.legend(fontsize=11, loc=[0.6, 0.3])

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
        marker=dict(size=6, color="gray", line=dict(color="black", width=0.8)),
        hovertemplate="rmax/r0: %{x:.3f}<br>μ: %{y:.5f}<extra>Sim. data</extra>",
    ))
    # Median boundary line (black, smooth)
    fig_plotly.add_trace(go.Scatter(
        x=r_smooth,
        y=adapt_med_smooth,
        mode="lines",
        name="Tumorigenic boundary (Sim.)",
        line=dict(color="black", width=2.5),
        hoverinfo="skip",
    ))

# Theory critical boundary (Blue dashed line)
fig_plotly.add_trace(go.Scatter(
    x=theory_rmax_norm,
    y=dmu_star,
    mode="lines",
    name="Theory (Critical boundary)",
    line=dict(color="blue", width=2, dash="dash"),
    hovertemplate="rmax/r0: %{x:.3f}<br>μ* theory: %{y:.5f}<extra>Theory</extra>",
))

# Saturation line (Orange)
fig_plotly.add_hline(
    y=mu_star_sat,
    line=dict(color="rgba(255,165,0,0.45)", width=1.5, dash="dash"),
    annotation_text=f"μ*<sub>∞</sub> ≈ {mu_star_sat:.4f}",
    annotation_position="top right",
    annotation_font=dict(size=11, color="orange"),
)

# Layout matching Plotly style with new colors and limits
fig_plotly.update_layout(
    title=dict(
        text="Phase Boundary: Critical Mutation Rate μ*(r<sub>max</sub>)",
        font=dict(size=18),
        x=0.5,
    ),
    xaxis=dict(
        title=dict(text="r<sub>max</sub> / r<sub>0</sub>", font=dict(size=15)),
        range=[1.0, XMAX],
        showgrid=True,
        gridcolor="#e0e0e0",
    ),
    yaxis=dict(
        title=dict(text="Mutation rate μ", font=dict(size=15)),
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
    width=900,
    height=560,
)


# ---------------------------------------------------------------------------
# Save Outputs
# ---------------------------------------------------------------------------
out_dir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

html_path = out_dir / "stability_sweep.html"
png_path  = out_dir / "stability_sweep.png"
svg_path  = out_dir / "stability_sweep.svg"

# Save Matplotlib figures
fig_mpl.savefig(png_path, dpi=200, bbox_inches="tight")
fig_mpl.savefig(svg_path, bbox_inches="tight")
print(f"Saved Matplotlib static figures to:\n  - {png_path}\n  - {svg_path}")
plt.close(fig_mpl)

# Save Plotly HTML figure
fig_plotly.write_html(str(html_path), include_plotlyjs="cdn")
print(f"Saved Plotly interactive figure to:\n  - {html_path}")
