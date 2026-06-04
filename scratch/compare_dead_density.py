"""
Scratch: compare dead-cell density vs tumor density between solid and liquid ensembles.
Runs from project root.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path("/home/francesco/Universita/PhD/PROJECTS/CancerEvo")
DATA_ROOT    = Path("/data/UNIVERSITA/PhD/PROJECTS/CancerEvo")

LIQUID_DIR = DATA_ROOT   / "data" / "simulations_liquid" / "ensemble_results"
SOLID_DIR  = DATA_ROOT   / "data" / "simulations"        / "ensemble_results"

def load_ensemble(path):
    sims = []
    for f in sorted(path.glob("sim_*.npz")):
        d = np.load(f)
        sims.append({
            "tumor_density": d["tumor_density"],
            "death_density": d["death_density"],
            "outcome": int(d["outcome_code"][0]),
        })
    return sims

print("Loading...")
liquid = load_ensemble(LIQUID_DIR)
solid  = load_ensemble(SOLID_DIR)
print(f"  Liquid: {len(liquid)} sims | Solid: {len(solid)} sims")

# ------------------------------------------------------------------
# For each sim: compute mean(death_density) / mean(tumor_density)
# over steps where tumor_density > 0 (i.e. while cancer is active)
# ------------------------------------------------------------------
def ratio_stats(sims):
    ratios, peak_dead, peak_tumor = [], [], []
    for s in sims:
        td = s["tumor_density"]
        dd = s["death_density"]
        active = td > 1e-6
        if active.sum() < 5:
            continue
        td_a = td[active]
        dd_a = dd[active]
        ratio = dd_a / (td_a + 1e-12)
        ratios.append(ratio.mean())
        peak_dead.append(dd_a.max())
        peak_tumor.append(td_a.max())
    return np.array(ratios), np.array(peak_dead), np.array(peak_tumor)

liq_ratios, liq_peak_d, liq_peak_t = ratio_stats(liquid)
sol_ratios, sol_peak_d, sol_peak_t = ratio_stats(solid)

print(f"\n  Liquid  dead/cancer ratio  mean={liq_ratios.mean():.3f}  median={np.median(liq_ratios):.3f}")
print(f"  Solid   dead/cancer ratio  mean={sol_ratios.mean():.3f}  median={np.median(sol_ratios):.3f}")
print(f"\n  Liquid  peak dead density  mean={liq_peak_d.mean():.4f}")
print(f"  Solid   peak dead density  mean={sol_peak_d.mean():.4f}")
print(f"\n  Liquid  peak tumor density mean={liq_peak_t.mean():.4f}")
print(f"  Solid   peak tumor density mean={sol_peak_t.mean():.4f}")

# ------------------------------------------------------------------
# Plot: time-averaged dead density vs tumor density (scatter)
#       + time series of mean trajectories
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0d1117")
plt.rcParams.update({"text.color": "#e6edf3", "axes.labelcolor": "#e6edf3",
                     "xtick.color": "#8b949e", "ytick.color": "#8b949e",
                     "axes.edgecolor": "#30363d", "grid.color": "#21262d",
                     "axes.facecolor": "#161b22", "axes.spines.top": False,
                     "axes.spines.right": False})
for ax in axes:
    ax.set_facecolor("#161b22")

COL = {"liquid": "#e63946", "solid": "#2a9d8f"}

# Panel 1: scatter of peak dead vs peak tumor density
ax = axes[0]
ax.scatter(sol_peak_t, sol_peak_d, c=COL["solid"],  alpha=0.6, s=30, label="Solid")
ax.scatter(liq_peak_t, liq_peak_d, c=COL["liquid"], alpha=0.6, s=30, label="Liquid")
ax.set_xlabel("Peak tumor density")
ax.set_ylabel("Peak dead-cell density")
ax.set_title("Peak densities per run", color="#e6edf3")
ax.legend(fontsize=9, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")
ax.grid(True, ls=":", alpha=0.3)

# Panel 2: histogram of dead/cancer ratio
ax = axes[1]
bins = np.linspace(0, max(liq_ratios.max(), sol_ratios.max()) * 1.05, 30)
ax.hist(sol_ratios, bins=bins, color=COL["solid"],  alpha=0.6, label="Solid",  density=True)
ax.hist(liq_ratios, bins=bins, color=COL["liquid"], alpha=0.6, label="Liquid", density=True)
ax.set_xlabel("Mean dead / cancer density ratio")
ax.set_ylabel("Density")
ax.set_title("Dead/cancer ratio distribution", color="#e6edf3")
ax.legend(fontsize=9, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")
ax.grid(True, ls=":", alpha=0.3)

# Panel 3: mean time series of dead density for both ensembles
ax = axes[2]
def mean_trajectory(sims, key, max_len=400):
    """Align on t=0 and average, truncate at max_len."""
    arrs = [s[key][:max_len] for s in sims]
    max_t = min(max(len(a) for a in arrs), max_len)
    mat = np.full((len(arrs), max_t), np.nan)
    for i, a in enumerate(arrs):
        mat[i, :len(a)] = a
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

for label, sims, col in [("Solid", solid, COL["solid"]), ("Liquid", liquid, COL["liquid"])]:
    for key, ls in [("death_density", "-"), ("tumor_density", "--")]:
        mn, sd = mean_trajectory(sims, key)
        t = np.arange(len(mn))
        tag = "dead" if "death" in key else "tumor"
        ax.plot(t, mn, color=col, ls=ls, lw=1.8, label=f"{label} {tag}")
        ax.fill_between(t, mn - sd, mn + sd, color=col, alpha=0.1)

ax.set_xlabel("Step")
ax.set_ylabel("Density")
ax.set_title("Mean trajectories (solid=dead, dashed=tumor)", color="#e6edf3")
ax.legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3",
          edgecolor="#30363d", ncol=2)
ax.grid(True, ls=":", alpha=0.3)
ax.set_xlim(0, 400)

fig.suptitle("Dead vs Tumor density — Solid vs Liquid", fontsize=14,
             fontweight="bold", color="#e6edf3")
plt.tight_layout()

out = PROJECT_ROOT / "outputs" / "figures" / "scratch_dead_vs_tumor.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nFigure saved to {out}")
