"""
plot_density_vs_actI_solid.py
-----------------------------
Plot subpopulation fractions x_k(t) by activation class of I genes,
using a solid tumor trajectory with per-class actI_dist data.
Overlays the theoretical stationary distribution x_k* from the Master Equation.

Usage:
  python scripts/plots/solid/plot_density_vs_actI_solid.py \
      [npz_path] [out_suffix] [mu_label] [mu_lo] [mu_hi] [dmu] [N_H] [remove_lower]

Defaults:
  npz_path     = data/simulations/sim_solid_actI.npz
  out_suffix   = solid_mu0326
  mu_label     = 0.0326
  mu_lo/hi     = 0.031 / 0.034
  dmu          = 0.015   (from parameters.jl)
  N_H          = 10      (N_HK)
  remove_lower = 0
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

from calculate_asymptotic_theory import compute_asymptotic_limit

# ---------------------------------------------------------------------------
# CLI args: [npz_path] [out_suffix] [mu_label] [mu_lo] [mu_hi] [dmu] [N_H] [remove_lower]
# ---------------------------------------------------------------------------
NPZ_PATH     = REPO_ROOT / (sys.argv[1] if len(sys.argv) > 1
               else "data/simulations/sim_solid_actI.npz")
OUT_SUFFIX   = sys.argv[2] if len(sys.argv) > 2 else "solid_mu0326"
MU_LABEL     = sys.argv[3] if len(sys.argv) > 3 else "0.0326"
MU_BAND_LO   = float(sys.argv[4]) if len(sys.argv) > 4 else 0.031
MU_BAND_HI   = float(sys.argv[5]) if len(sys.argv) > 5 else 0.034
DMU          = float(sys.argv[6]) if len(sys.argv) > 6 else 0.015
N_H          = int(sys.argv[7])   if len(sys.argv) > 7 else 10
REMOVE_LOWER = int(sys.argv[8])   if len(sys.argv) > 8 else 0
OUT_DIR      = REPO_ROOT / "outputs" / "figures" / "solid"

# ---------------------------------------------------------------------------
# Load simulation data
# ---------------------------------------------------------------------------
with np.load(NPZ_PATH) as d:
    tumor_density = d["tumor_density"]   # (T,)
    actI_dist     = d["actI_dist"]       # (T, N_I+1): fraction of cells in class k
    mu            = d["mu"]              # (T,)
    act_I_mean    = d["act_I"]           # (T,) population-mean act_I

T, N_CLASSES = actI_dist.shape
N_I = N_CLASSES - 1
steps = np.arange(T)

# ---------------------------------------------------------------------------
# Theoretical stationary distribution from Master Equation
# ---------------------------------------------------------------------------
theory   = compute_asymptotic_limit(N_I=N_I, N_H=N_H, dmu=DMU, remove_lower=REMOVE_LOWER)
theory_k = np.array(theory["classes"])       # k values: 1..N_I
theory_x = np.array(theory["x_analytical"])  # fractions for k = 1..N_I
mu_theory = theory["asymp_mu_analytical"]

print(f"Theory μ∞ = {mu_theory:.5f}  (dmu={DMU}, N_H={N_H}, remove_lower={REMOVE_LOWER})")
print(f"Theory x_k*: { {int(k): round(x,4) for k,x in zip(theory_k, theory_x)} }")
print(f"Loaded trajectory: T={T} steps, N_I={N_I}")
print(f"Sim mu tail mean: {mu[int(0.7*T):].mean():.4f}")
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
tail_start = int(0.70 * T)
legend_fontsize = 12
label_fontsize=15
ticks_fontsize=13
letter_fontsize=17

# ---------------------------------------------------------------------------
# Figure layout  (4-panel mosaic)
# ---------------------------------------------------------------------------
fig, ax_dict = plt.subplot_mosaic(
    [["A", "B"], ["C", "D"]],
    figsize=(14, 7),
    gridspec_kw={"hspace": 0.42, "wspace": 0.30},
)

# ── Panel A: x_k(t) ─────────────────────────────────────────────────────────
ax_A = ax_dict["A"]
for k in active_classes:
    ax_A.plot(steps, actI_dist[:, k], color=colors[k], lw=1.5, alpha=0.9,
              label=f"$k={k}$")
ax_A.set_xlabel("Simulation step", fontsize=label_fontsize)
ax_A.set_ylabel("Subpopulation fraction $x_k$", fontsize=label_fontsize)
#ax_A.set_title("Per-class subpopulation fraction $x_k(t)$  [sum = 1 at each step]")
ax_A.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax_A.legend(loc="upper right", ncol=3, framealpha=0.6, title="$n_{\\rm act,I}$", fontsize=legend_fontsize-2)
ax_A.tick_params(axis="both", labelsize=ticks_fontsize)
ax_A.text(-0.05, 1.12, "a", transform=ax_A.transAxes,
          fontsize=letter_fontsize, fontweight="bold", va="top")

# ── Panel B: μ(t) ───────────────────────────────────────────────────────────
ax_B = ax_dict["B"]
ax_B.plot(steps, mu, color="#2A9D8F", lw=1.8, label=r"$\mu(t)$")
#ax_B.axhspan(MU_BAND_LO, MU_BAND_HI, color="#E63946", alpha=0.15,
#             label=f"Target [{MU_BAND_LO}, {MU_BAND_HI}]")
#ax_B.axvline(tail_start, color="grey", ls="--", lw=1.0, label="Tail region start")
ax_B.axhline(y=mu_theory, color="k", lw=1.8, label=r"$\mu_\infty$", ls="--")

ax_B.set_xlabel("Simulation step", fontsize=label_fontsize)
ax_B.set_ylabel(r"Mutation rate $\mu$", fontsize=label_fontsize)
#ax_B.set_title(r"Stabilisation of $\mu$")
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
         label="Tumor spatial simulation")
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
#ax_C.set_title(r"Stationary subpopulation distribution $x_k^*$")
ax_C.set_xticks(k_vals)
ax_C.set_xlim(-0.6, N_I + 0.6)
ax_C.set_axisbelow(True)
ax_C.legend(loc="upper right", framealpha=0.8, fontsize=legend_fontsize)
ax_C.tick_params(axis="both", labelsize=ticks_fontsize)

# ── Panel D: pdie & pmut vs k ───────────────────────────────────────────────
ax_D = ax_dict["D"]

def pdie(mu, ni, N_HK):
    return 1.0 - (1.0 - ni * mu) ** N_HK

def pmut(mu, ni, N_I, chroms=2):
    return 1.0 - (1.0 - ni * mu) ** (chroms * (N_I - ni))

mu_val = float(MU_LABEL)
k_range = np.arange(1, N_I + 1)
p_die_vals = [pdie(mu_val, k, N_H) for k in k_range]
p_mut_vals = [pmut(mu_val, k, N_I, chroms=2) for k in k_range]

ax_D.plot(k_range, p_die_vals, label=r"$p_{\rm die}$", color="purple", lw=2)
ax_D.scatter(k_range, p_die_vals, color=[colors[k] for k in k_range], zorder=3, s=50, edgecolors="white", linewidths=0.5)

ax_D.plot(k_range, p_mut_vals, label=r"$p_{\rm mut}$", color="darkorange", lw=2)
ax_D.scatter(k_range, p_mut_vals, color=[colors[k] for k in k_range], zorder=3, s=50, edgecolors="white", linewidths=0.5)

ax_D.set_xlabel(r"$k$ — number of active instability genes", fontsize=label_fontsize)
ax_D.set_ylabel("Probability", fontsize=label_fontsize)
#ax_D.set_title(rf"Death & Mutation Probabilities ($\mu = {mu_val:.4f}$)")
ax_D.set_xticks(k_range)
ax_D.set_xlim(0.4, N_I + 0.6)
ax_D.set_ylim(-0.05, 1.05)
ax_D.legend(loc="lower center", framealpha=0.6, fontsize=legend_fontsize)
ax_D.tick_params(axis="both", labelsize=ticks_fontsize)
ax_D.text(-0.13, 1.12, "d", transform=ax_D.transAxes,
          fontsize=letter_fontsize, fontweight="bold", va="top")

#fig.suptitle(
#    rf"Subpopulation fractions by $n_{{\rm act,I}}$ — solid tumor"
#    rf" ($\mu_\infty \approx {MU_LABEL}$, Tumor$_{{\rm Max}}$)",
#    fontsize=13, fontweight="bold", y=1.01
#)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    out = OUT_DIR / f"density_vs_actI_{OUT_SUFFIX}.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=False, format=ext)
    print(f"Saved → {out}")

plt.close(fig)
print("Done.")
