"""
plot_phase_diagram.py
---------------------
Plot the analytical phase diagram (p_death vs r/r0) showing the boundary
between the "Tumor Grows" and "Tumor Shrinks" regimes, together with the
data points derived from stability-sweep results.

Reproduces Cell 25 of notebooks/analysis.ipynb.

Outputs
-------
outputs/figures/phase_diagram.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline

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
# Hard-coded data from stability-sweep results (notebook Cell 24)
# These are the average stable mutation rates at each r/r0 value.
# ---------------------------------------------------------------------------
r_prop_list = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
results = np.array([
    -0.6101874390439277,
     0.6226682669930774,
     1.2917773010287472,
     1.8387745389776091,
     2.2434712898547584,
     2.6607741287392520,
     2.9356429157955044,
     3.1821620335205547,
     3.4130647650803505,
     3.6252708643165220,
     3.8084414875435573,
])

# Convert raw mu-step counts to p_death probability
# p_death = 1 - (1 - mu)^N  where N=10 HK genes, mu = results * 1e-2
p_die_data = np.array([(1.0 - (1.0 - m * 1e-2) ** 10) for m in results])
mu_data    = results * 1e-2

# ---------------------------------------------------------------------------
# Spline fits
# ---------------------------------------------------------------------------
spl_pd = make_smoothing_spline(r_prop_list, p_die_data, lam=0.001)
spl_mu = make_smoothing_spline(r_prop_list, mu_data,    lam=0.001)

x_new = np.linspace(0.8, 2.2, 200)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase Diagram: Tumor Growth Regimes", fontsize=16,
             fontweight="bold", y=1.02)


def _annotate_regions(ax, x, y_spl, y_lo, y_hi, label_shrink, label_grow,
                      pos_shrink, pos_grow):
    """Fill shrink/grow regions and add text labels."""
    ax.fill_between(x, y_spl, y_hi, alpha=0.18, color="#2A9D8F", label="Healthy regime")
    ax.fill_between(x, y_lo, y_spl, alpha=0.18, color="#E63946", label="Tumor regime")
    ax.text(*pos_shrink, label_shrink,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
            fontsize=13)
    ax.text(*pos_grow, label_grow,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
            fontsize=13)


# ── Panel 1: p_death ──
ax = axes[0]
y_spl = spl_pd(x_new)
ax.plot(x_new, y_spl, color="k", lw=2, zorder=5, label="Critical boundary")
_annotate_regions(
    ax, x_new, y_spl,
    y_lo=-0.01, y_hi=max(y_spl) * 1.5,
    label_shrink="Tumor Shrinks",
    label_grow="Tumor Grows",
    pos_shrink=(1.0, 0.40),
    pos_grow=(1.70, 0.02),
)
ax.scatter(r_prop_list, p_die_data, color="#E07B39", s=60, zorder=10,
           label="Sim. data")
ax.set_xlabel(r"$r / r_0$", fontsize=16)
ax.set_ylabel(r"$p_{\mathrm{death}}$", fontsize=16)
ax.set_xlim(0.95, 2.05)
ax.set_ylim(-0.005, 0.50)
ax.set_title(r"Death probability $p_{\rm death}$", fontsize=14)
ax.yaxis.grid(True, ls=":", alpha=0.5)
ax.legend(fontsize=11)

# ── Panel 2: mu ──
ax = axes[1]
y_spl2 = spl_mu(x_new)
ax.plot(x_new, y_spl2, color="k", lw=2, zorder=5, label="Critical boundary")
_annotate_regions(
    ax, x_new, y_spl2,
    y_lo=-0.001, y_hi=max(y_spl2) * 1.5,
    label_shrink="Tumor Shrinks",
    label_grow="Tumor Grows",
    pos_shrink=(1.0, 0.038),
    pos_grow=(1.70, 0.002),
)
ax.scatter(r_prop_list, mu_data, color="#E07B39", s=60, zorder=10,
           label="Sim. data")
ax.set_xlabel(r"$r / r_0$", fontsize=16)
ax.set_ylabel(r"$\mu$", fontsize=16)
ax.set_xlim(0.95, 2.05)
ax.set_ylim(-0.001, 0.05)
ax.set_title(r"Mutation rate $\mu$", fontsize=14)
ax.yaxis.grid(True, ls=":", alpha=0.5)
ax.legend(fontsize=11)

plt.tight_layout()

out_dir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "phase_diagram.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
