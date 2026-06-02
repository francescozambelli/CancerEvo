"""
plot_stability_sweep.py
-----------------------
Interactive unified phase diagram: all stability datasets overlaid, with
adaptive boundary median + IQR ribbon.  No analytic theory curve.

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
# Build figure
# ---------------------------------------------------------------------------
fig = go.Figure()

# ── Prior sweep 1 ────────────────────────────────────────────────────────────
mask0 = df0["rmax_norm"] <= XMAX
fig.add_trace(go.Scatter(
    x=df0.loc[mask0, "rmax_norm"],
    y=df0.loc[mask0, "stable_dmu"],
    mode="markers",
    name="Sweep 1",
    marker=dict(color="#E63946", size=6, opacity=0.45),
    hovertemplate="rmax/r0: %{x:.3f}<br>δμ: %{y:.5f}<extra>Sweep 1</extra>",
))

# ── Prior sweep 2 ────────────────────────────────────────────────────────────
mask1 = df1["rmax_norm"] <= XMAX
fig.add_trace(go.Scatter(
    x=df1.loc[mask1, "rmax_norm"],
    y=df1.loc[mask1, "stable_dmu"],
    mode="markers",
    name="Sweep 2",
    marker=dict(color="#2A9D8F", size=6, opacity=0.45),
    hovertemplate="rmax/r0: %{x:.3f}<br>δμ: %{y:.5f}<extra>Sweep 2</extra>",
))

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
