"""
plot_stability_sweep.py
-----------------------
Interactive unified phase diagram: all stability datasets overlaid, with
adaptive boundary median + IQR ribbon, and analytic theory curve.

Opens an interactive HTML figure (Plotly) that supports zoom, pan, and hover.
Also saves a static PNG to outputs/figures/stability_sweep.png.

Outputs
-------
outputs/figures/stability_sweep.html   ← interactive (open in browser)
outputs/figures/stability_sweep.png    ← static snapshot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go

from src.analysis.loaders import load_stability_results, load_all_stability_results

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df0, df1 = load_stability_results()
merged    = load_all_stability_results()

dfa              = merged[merged["source"] == "adaptive"]
adaptive_available = not dfa.empty

XMAX = 10.0   # rmax/r0 cut-off

# ── Adaptive boundary: per-rmax_norm statistics ──────────────────────────────
if adaptive_available:
    dfa_clip = dfa[dfa["rmax_norm"] <= XMAX].copy()
    grp = dfa_clip.groupby("rmax_norm")["stable_dmu"]
    r_unique  = np.array(sorted(dfa_clip["rmax_norm"].unique()))
    adapt_med = grp.median().values
    adapt_lo  = grp.quantile(0.25).values
    adapt_hi  = grp.quantile(0.75).values
    adapt_rmax_raw = dfa_clip["rmax_norm"].values
    adapt_dmu_raw  = dfa_clip["stable_dmu"].values

# ---------------------------------------------------------------------------
# Theoretical prediction (corrected)
# ---------------------------------------------------------------------------
# Model parameters (from scripts/parameters.jl)
r0   = 0.15
N_I  = 10
N_HK = 10

# In the sweep: dr = rmax/10, so r_cancer = min(r0 + 20*(rmax/10), rmax) = rmax (no genetic cap)
#
# Stability condition: after a boundary division, BOTH mother AND daughter are mutated.
#   Net cancer gain per step at boundary:
#     r_cancer * P_s [daughter survives] - r_cancer * (1-P_s) [mother dies] = r0
#   => r_cancer * (2*P_s - 1) = r0
#   => P_s = (1 + r0/r_cancer) / 2
#   => (1 - N_I*dmu*)^N_HK = (1 + r0/r_cancer) / 2
#   => dmu* = [1 - ((1 + r0/r_cancer)/2)^(1/N_HK)] / N_I
#
# Limiting cases:
#   r_cancer = r0: P_s = 1 => dmu* = 0  (no advantage, any mutation kills)
#   r_cancer >> r0: P_s -> 0.5 => dmu*_inf = [1 - 0.5^(1/N_HK)] / N_I

theory_rmax_norm = np.linspace(1.001, XMAX, 600)
theory_rmax_abs  = theory_rmax_norm * r0          # r_cancer = rmax (dr = rmax/10)
P_s_star         = (1.0 + r0 / theory_rmax_abs) / 2.0
mu_star          = 1.0 - P_s_star ** (1.0 / N_HK)
dmu_star         = np.clip(mu_star / N_I, 0.0, None)

# Asymptotic saturation
dmu_star_sat = float((1.0 - 0.5 ** (1.0 / N_HK)) / N_I)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig = go.Figure()

# ── Adaptive data ─────────────────────────────────────────────────────────────
if adaptive_available:
    # Raw scatter
    fig.add_trace(go.Scatter(
        x=adapt_rmax_raw,
        y=adapt_dmu_raw,
        mode="markers",
        name="Adaptive (raw)",
        marker=dict(color="#6A0572", size=5, opacity=0.35),
        hovertemplate="rmax/r0: %{x:.3f}<br>δμ: %{y:.5f}<extra>Adaptive raw</extra>",
    ))

    # IQR ribbon (filled area between Q25 and Q75)
    fig.add_trace(go.Scatter(
        x=np.concatenate([r_unique, r_unique[::-1]]),
        y=np.concatenate([adapt_hi, adapt_lo[::-1]]),
        fill="toself",
        fillcolor="rgba(106,5,114,0.12)",
        line=dict(width=0),
        name="Adaptive IQR",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Median boundary line
    fig.add_trace(go.Scatter(
        x=r_unique,
        y=adapt_med,
        mode="lines+markers",
        name="Adaptive boundary (median)",
        line=dict(color="#6A0572", width=2.5),
        marker=dict(size=6),
        hovertemplate="rmax/r0: %{x:.3f}<br>δμ*: %{y:.5f}<extra>Adaptive median</extra>",
    ))

# ── Theoretical prediction ───────────────────────────────────────────────────
# Corrected formula: both mother and daughter mutate; dr=rmax/10 so r_cancer=rmax
fig.add_trace(go.Scatter(
    x=theory_rmax_norm,
    y=dmu_star,
    mode="lines",
    name="Theory: δμ* = [1−((1+r₀/rₘₐₓ)/2)^(1/N_HK)]/N_I",
    line=dict(color="#F4A261", width=3, dash="solid"),
    hovertemplate="rmax/r0: %{x:.3f}<br>δμ* theory: %{y:.5f}<extra>Theory</extra>",
))

# Horizontal line at asymptotic saturation
fig.add_hline(
    y=dmu_star_sat,
    line=dict(color="rgba(244,162,97,0.45)", width=1.5, dash="dash"),
    annotation_text=f"δμ*<sub>∞</sub> = [1−0.5<sup>1/N_HK</sup>]/N_I ≈ {dmu_star_sat:.4f}",
    annotation_position="top right",
    annotation_font=dict(size=11, color="#F4A261"),
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
fig.update_layout(
    title=dict(
        text="Phase Boundary: Critical Mutation Rate δμ*(r<sub>max</sub>)",
        font=dict(size=18),
        x=0.5,
    ),
    xaxis=dict(
        title=dict(text="r<sub>max</sub> / r<sub>0</sub>", font=dict(size=15)),
        range=[0, XMAX],
        showgrid=True,
        gridcolor="#e0e0e0",
    ),
    yaxis=dict(
        title=dict(text="Phase boundary δμ*", font=dict(size=15)),
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
# Save
# ---------------------------------------------------------------------------
out_dir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

html_path = out_dir / "stability_sweep.html"
png_path  = out_dir / "stability_sweep.png"

fig.write_html(str(html_path), include_plotlyjs="cdn")
print(f"Saved interactive → {html_path}")

try:
    fig.write_image(str(png_path), scale=2)
    print(f"Saved static     → {png_path}")
except Exception as e:
    print(f"Static PNG skipped ({e}); install kaleido: pip install kaleido")

fig.show()
