"""
plot_density_vs_actI.py
-----------------------
Plot subpopulation fractions x_k(t) by activation class of I genes,
using a liquid tumor trajectory with per-class actI_dist data.
Overlays the theoretical stationary distribution x_k* from the Master Equation.

Usage:
  python scripts/plots/liquid/plot_density_vs_actI.py \
      [npz_path] [out_suffix] [mu_label] [mu_lo] [mu_hi] [dmu] [N_H] [remove_lower]

Defaults:
  npz_path     = data/simulations_liquid/highmu_actI_dist/sim_highmu.npz
  out_suffix   = highmu
  mu_label     = 0.087
  mu_lo/hi     = 0.083 / 0.095
  dmu          = 0.023   (from parameters_liquid.jl)
  N_H          = 10      (N_HK)
  remove_lower = 0

Example (mu~0.055):
  python scripts/plots/liquid/plot_density_vs_actI.py \\
      data/simulations_liquid/highmu_actI_dist/sim_mu055.npz \\
      mu055 0.055 0.052 0.058 0.023 10 0
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

from src.sim_liquid import CorrectedLiquidTumor

# ---------------------------------------------------------------------------
# CLI args: [npz_path] [out_suffix] [mu_label] [mu_lo] [mu_hi] [dmu] [N_H] [remove_lower]
# ---------------------------------------------------------------------------
NPZ_PATH     = REPO_ROOT / (sys.argv[1] if len(sys.argv) > 1
               else "data/simulations_liquid/highmu_actI_dist/sim_highmu.npz")
OUT_SUFFIX   = sys.argv[2] if len(sys.argv) > 2 else "highmu"
MU_LABEL     = sys.argv[3] if len(sys.argv) > 3 else "0.087"
MU_BAND_LO   = float(sys.argv[4]) if len(sys.argv) > 4 else 0.083
MU_BAND_HI   = float(sys.argv[5]) if len(sys.argv) > 5 else 0.095
DMU          = float(sys.argv[6]) if len(sys.argv) > 6 else 0.023
N_H          = int(sys.argv[7])   if len(sys.argv) > 7 else 10
REMOVE_LOWER = int(sys.argv[8])   if len(sys.argv) > 8 else 0
OUT_DIR      = REPO_ROOT / "outputs" / "figures" / "liquid"

# ---------------------------------------------------------------------------
# Load simulation data
# ---------------------------------------------------------------------------
with np.load(NPZ_PATH) as d:
    tumor_density = d["tumor_density"]   # (T,)
    actI_dist     = d["actI_dist"]       # (T, N_I+1): fraction of cells in class k
    mu            = d["mu"]              # (T,)
    act_I_mean    = d["act_I"]           # (T,) population-mean act_I
    r_sim         = d["r"] if "r" in d else None

T, N_CLASSES = actI_dist.shape
N_I = N_CLASSES - 1
steps = np.arange(T)
tail_start = int(0.70 * T)

# ---------------------------------------------------------------------------
# Theoretical stationary distribution from Master Equation (sim_liquid.py)
# ---------------------------------------------------------------------------
r_tail_mean = r_sim[tail_start:].mean() if r_sim is not None else 0.25
r0 = 0.15  # standard baseline replication rate in the liquid tumor model

model = CorrectedLiquidTumor(
    NI=N_I,
    NHK=N_H,
    delta_mu=DMU,
    r=np.array([r_tail_mean] * N_I),
    r0=r0
)

sol = model.solve()
if not sol.success:
    raise RuntimeError(f"Theoretical solver failed: {sol.message}")

f_theory = sol.x[:-1]
pD_theory = sol.x[-1]

# Normalize cancer fractions to sum to 1
sum_f = np.sum(f_theory)
if sum_f > 0:
    theory_x = f_theory / sum_f
else:
    theory_x = np.zeros(N_I)

# Apply remove_lower if requested
if REMOVE_LOWER > 0:
    theory_x[:REMOVE_LOWER] = 0.0
    sum_f = np.sum(theory_x)
    if sum_f > 0:
        theory_x = theory_x / sum_f

theory_k = np.arange(1, N_I + 1)
class_mus = theory_k * DMU
mu_theory = np.sum(theory_x * class_mus)

print(f"Theory μ∞ = {mu_theory:.5f}  (dmu={DMU}, N_H={N_H}, remove_lower={REMOVE_LOWER})")
print(f"Theory x_k*: { {int(k): round(x,4) for k,x in zip(theory_k, theory_x)} }")
print(f"Loaded trajectory: T={T} steps, N_I={N_I}")
print(f"Sim mu tail mean: {mu[tail_start:].mean():.4f}")
print(f"Stationary actI_dist (last step): {actI_dist[-1].round(3)}")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "lines.linewidth":   1.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.5,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

cmap   = cm.viridis
colors = [cmap(k / N_I) for k in range(N_CLASSES)]
active_classes = [k for k in range(N_CLASSES) if actI_dist[:, k].max() > 1e-4]
legend_fontsize = 12
label_fontsize = 15
ticks_fontsize = 13
letter_fontsize = 17

# ---------------------------------------------------------------------------
# Figure layout  (4-panel mosaic)
# ---------------------------------------------------------------------------
fig, ax_dict = plt.subplot_mosaic(
    [["A", "B"], ["C", "D"]],
    figsize=(16, 7),
    gridspec_kw={"hspace": 0.42, "wspace": 0.30},
)

# ── Panel A: x_k(t) ─────────────────────────────────────────────────────────
ax_A = ax_dict["A"]
for k in active_classes:
    ax_A.plot(steps, actI_dist[:, k], color=colors[k], lw=1.5, alpha=0.9,
              label=f"$k={k}$")
ax_A.set_xlabel("Simulation step", fontsize=label_fontsize)
ax_A.set_ylabel("Subpopulation fraction $x_k$", fontsize=label_fontsize)
ax_A.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax_A.legend(loc="upper right", ncol=3, framealpha=0.6, title="$n_{\\rm act,I}$", fontsize=legend_fontsize-2)
ax_A.tick_params(axis="both", labelsize=ticks_fontsize)
ax_A.text(-0.05, 1.12, "a", transform=ax_A.transAxes,
          fontsize=letter_fontsize, fontweight="bold", va="top")

# ── Panel B: μ(t) ───────────────────────────────────────────────────────────
ax_B = ax_dict["B"]
ax_B.plot(steps, mu, color="#2A9D8F", lw=1.8, label=r"$\mu(t)$")
ax_B.axhline(y=mu_theory, color="k", lw=1.8, label=r"$\mu_\infty$", ls="--")
ax_B.set_xlabel("Simulation step", fontsize=label_fontsize)
ax_B.set_ylabel(r"Mutation rate $\mu$", fontsize=label_fontsize)
ax_B.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax_B.legend(loc="lower right", framealpha=0.6, fontsize=legend_fontsize)
ax_B.tick_params(axis="both", labelsize=ticks_fontsize)
ax_B.text(-0.13, 1.12, "b", transform=ax_B.transAxes,
          fontsize=letter_fontsize, fontweight="bold", va="top")

# ── Panel C: stationary distribution + theory overlay ───────────────────────
ax_C = ax_dict["C"]

xk_tail      = actI_dist[tail_start:, :]
xk_tail_mean = xk_tail.mean(axis=0)
xk_tail_std  = xk_tail.std(axis=0)

k_vals     = np.arange(N_CLASSES)
bar_colors = [colors[k] for k in k_vals]

ax_C.bar(k_vals, xk_tail_mean, color=bar_colors, alpha=0.80,
         edgecolor="white", linewidth=0.5, zorder=3,
         label="Tumor liquid simulation")
ax_C.errorbar(k_vals, xk_tail_mean, yerr=xk_tail_std,
              fmt="none", color="black", capsize=4, lw=1.2, zorder=4)

# Theory overlay
ax_C.plot(theory_k, theory_x, color="#E76F51", lw=2.0, ls="--",
          marker="D", ms=7, mec="white", mew=0.8, zorder=5,
          label=rf"Theory $x_k^*$")

ax_C.text(-0.05, 1.12, "c", transform=ax_C.transAxes,
          fontsize=letter_fontsize, fontweight="bold", va="top")
ax_C.set_xlabel(r"$k$ — number of active instability genes per cell", fontsize=label_fontsize)
ax_C.set_ylabel(r"Subpopulation fraction $x_k^*$", fontsize=label_fontsize)
ax_C.set_xticks(k_vals)
ax_C.set_xlim(-0.6, N_I + 0.6)
ax_C.set_axisbelow(True)
ax_C.legend(loc="upper right", framealpha=0.8, fontsize=legend_fontsize)
ax_C.tick_params(axis="both", labelsize=ticks_fontsize)

# ── Panel D: pdie & ppromote vs k ───────────────────────────────────────────
ax_D = ax_dict["D"]
k_range = np.arange(1, N_I + 1)
p_die_vals = [model.delta(k) for k in k_range]
p_promote_vals = [model.gamma(k) for k in k_range]

ax_D.plot(k_range, p_die_vals, label=r"$p_{\rm die}$", color="purple", lw=2)
ax_D.scatter(k_range, p_die_vals, color=[colors[k] for k in k_range], zorder=3, s=50, edgecolors="white", linewidths=0.5)

ax_D.plot(k_range, p_promote_vals, label=r"$p_{\rm promote}$", color="darkorange", lw=2)
ax_D.scatter(k_range, p_promote_vals, color=[colors[k] for k in k_range], zorder=3, s=50, edgecolors="white", linewidths=0.5)

ax_D.set_xlabel(r"$k$ — number of active instability genes", fontsize=label_fontsize)
ax_D.set_ylabel("Probability", fontsize=label_fontsize)
ax_D.set_xticks(k_range)
ax_D.set_xlim(0.4, N_I + 0.6)
ax_D.set_ylim(-0.05, 1.05)
ax_D.legend(loc="upper left", framealpha=0.6, fontsize=legend_fontsize)
ax_D.tick_params(axis="both", labelsize=ticks_fontsize)
ax_D.text(-0.13, 1.12, "d", transform=ax_D.transAxes,
          fontsize=letter_fontsize, fontweight="bold", va="top")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    out = OUT_DIR / f"density_vs_actI_{OUT_SUFFIX}.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=False, format=ext)
    print(f"Saved → {out}")

plt.close(fig)
print("Done.")
