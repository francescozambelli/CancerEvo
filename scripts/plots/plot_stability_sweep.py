"""
plot_stability_sweep.py
-----------------------
Plot the stability sweep results: stable mutation rate (``stable_dmu``)
as a function of the normalised maximum reproduction rate (``rmax / r0``).

Reproduces Cells 5–7 of notebooks/analysis.ipynb.

Outputs
-------
outputs/figures/stability_sweep.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline

from src.analysis.loaders import load_stability_results

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df0, df1 = load_stability_results()

# ── Group by rmax_norm ──
grp0 = df0.groupby("rmax_norm")["stable_dmu"].mean()
grp1 = df1.groupby("rmax_norm")["stable_dmu"].mean()

# ── Analytical prediction ──
x_theory = np.linspace(1.0, 2.0, 200)
# From the notebook: y = (1 - (r)^(-1/10)) / 10
y_theory  = (1 - x_theory ** (-1.0 / 10)) / 10

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Stability Sweep: Critical Mutation Rate", fontsize=16,
             fontweight="bold", y=1.02)

# ── Panel 1: grouped mean ──
ax = axes[0]
ax.plot(grp0.index, grp0.values, "o-", color="#E63946", lw=2,
        markersize=6, label="Sweep 1")
ax.plot(grp1.index, grp1.values, "s-", color="#2A9D8F", lw=2,
        markersize=6, label="Sweep 2")
ax.plot(x_theory, y_theory, "--", color="#333333", lw=2,
        label=r"Theory: $(1-r^{-1/10})/10$")

ax.set_xlabel(r"$r_{\max}/r_0$", fontsize=15)
ax.set_ylabel(r"Mean stable $\delta\mu$", fontsize=14)
ax.set_title("Mean Stable Mutation Rate", fontsize=14)
ax.legend(fontsize=12)
ax.yaxis.grid(True, ls="--", alpha=0.4)

# ── Panel 2: scatter (all individual runs) ──
ax = axes[1]
ax.scatter(df0["rmax_norm"], df0["stable_dmu"] * 10,
           color="#E63946", alpha=0.5, s=18, label="Sweep 1 (×10)")
ax.scatter(df1["rmax_norm"], df1["stable_dmu"] * 10,
           color="#2A9D8F", alpha=0.5, s=18, label="Sweep 2 (×10)")

y_theory2 = 1.0 - x_theory ** (-1.0 / 18)
ax.plot(x_theory, y_theory2, "--", color="#333333", lw=2,
        label=r"Theory: $1 - r^{-1/18}$")

ax.set_xlabel(r"$r_{\max}/r_0$", fontsize=15)
ax.set_ylabel(r"Stable $\delta\mu \times 10$", fontsize=14)
ax.set_title("Individual Runs", fontsize=14)
ax.legend(fontsize=12)
ax.yaxis.grid(True, ls="--", alpha=0.4)

plt.tight_layout()

out_dir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "stability_sweep.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
