"""
plot_karyotype_mechanisms.py
----------------------------
Generates a 4-panel internal-state mechanistic figure for the karyotype comparison
(Diploid vs. Polyploid vs. Aneuploid), directly addressing Reviewer 2 Major Issue 4:
  (a) Effective gene-copy ploidy and max copy-slot number (n_c) over time
  (b) Cryptic housekeeping mutation burden (mut_HK) buffered in living cancer cells
  (c) Copy-number state distribution (n_c = 1..5) over time (Polyploid vs. Aneuploid)
  (d) Mutation burden stratified by copy number (n_c = 2..5)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
from src.analysis.loaders import load_all_ploidy

COLORS = {
    "Diploid":   "orangered",
    "Aneuploid": "forestgreen",
    "Polyploid": "purple",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print("Loading solid ensemble data (Tumor runs)...")
all_data = load_all_ploidy(outcome_filter="Tumor")

T_MAX = 2500

def pad_and_stack(trajs, key, t_max=T_MAX):
    arrs = []
    for tr in trajs:
        if key not in tr or len(tr[key]) == 0:
            continue
        v = np.asarray(tr[key], dtype=float)
        if len(v) < t_max:
            v = np.pad(v, (0, t_max - len(v)), mode="edge")
        else:
            v = v[:t_max]
        arrs.append(v)
    return np.array(arrs)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
t_grid = np.arange(1, T_MAX + 1)

# ---------------------------------------------------------------------------
# Panel (a): Effective Ploidy & Max Copy Number (n_c) over time
# ---------------------------------------------------------------------------
ax = axes[0, 0]
for label in ["Diploid", "Polyploid", "Aneuploid"]:
    _, trajs = all_data[label]
    eff_p = pad_and_stack(trajs, "eff_ploidy")
    n_c   = pad_and_stack(trajs, "n_chrs")
    m_ep, s_ep = np.mean(eff_p, axis=0), np.std(eff_p, axis=0)
    m_nc       = np.mean(n_c, axis=0)
    ax.plot(t_grid, m_ep, color=COLORS[label], lw=2.5, label=f"{label} (Effective ploidy)")
    ax.fill_between(t_grid, m_ep - s_ep, m_ep + s_ep, color=COLORS[label], alpha=0.15)
    if label == "Aneuploid":
        ax.plot(t_grid, m_nc, color=COLORS[label], lw=1.8, ls="--", label="Aneuploid (Max copy slot $n_c$)")

ax.set_xlabel("Time (steps)")
ax.set_ylabel("Mean gene-copy ploidy")
ax.set_title("(a) Ploidy & Segmental Copy Evolution", fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(0, T_MAX)
ax.set_ylim(1.8, 5.2)

# ---------------------------------------------------------------------------
# Panel (b): Cryptic Housekeeping Damage (mut_HK) over time
# ---------------------------------------------------------------------------
ax = axes[0, 1]
for label in ["Diploid", "Polyploid", "Aneuploid"]:
    _, trajs = all_data[label]
    mut_hk = pad_and_stack(trajs, "mut_HK")
    m_hk, s_hk = np.mean(mut_hk, axis=0), np.std(mut_hk, axis=0)
    ax.plot(t_grid, m_hk, color=COLORS[label], lw=2.5, label=label)
    ax.fill_between(t_grid, m_hk - s_hk, m_hk + s_hk, color=COLORS[label], alpha=0.18)

ax.axhline(0.5, color="gray", ls=":", lw=1.2, label="Diploid saturation limit (1/2)")
ax.set_xlabel("Time (steps)")
ax.set_ylabel("Mutated fraction of present Essential (HK) alleles")
ax.set_title("(b) Buffering of Cryptic Essential-Gene Damage", fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(0, T_MAX)
ax.set_ylim(0.0, 0.90)

# ---------------------------------------------------------------------------
# Panel (c): Copy-Number Distribution (n_c = 1..5) at t = 2000
# ---------------------------------------------------------------------------
ax = axes[1, 0]
t_idx = 1999  # step 2000
x_nc = np.arange(1, 6)
width = 0.26

for i, label in enumerate(["Diploid", "Polyploid", "Aneuploid"]):
    _, trajs = all_data[label]
    fracs = []
    for k in range(1, 6):
        arr = pad_and_stack(trajs, f"nc_dist_{k}")
        fracs.append(np.mean(arr[:, t_idx]))
    stds = [np.std(pad_and_stack(trajs, f"nc_dist_{k}")[:, t_idx]) for k in range(1, 6)]
    ax.bar(x_nc + (i - 1) * width, fracs, width=width, color=COLORS[label], alpha=0.85,
           yerr=stds, capsize=3, label=label, edgecolor="black", lw=0.7)

ax.set_xticks(x_nc)
ax.set_xticklabels([f"$n_c = {k}$" for k in x_nc])
ax.set_xlabel("Maximum allele copy number ($n_c$)")
ax.set_ylabel("Fraction of cancer population (at $t=2000$)")
ax.set_title("(c) Copy-Number State Distribution", fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0.0, 1.05)

# ---------------------------------------------------------------------------
# Panel (d): Total Mutation Burden by Copy Number (n_c = 2..5)
# ---------------------------------------------------------------------------
ax = axes[1, 1]
x_nc_mut = np.arange(2, 6)
width_m = 0.32

for i, label in enumerate(["Polyploid", "Aneuploid"]):
    _, trajs = all_data[label]
    means_m = []
    stds_m  = []
    for k in range(2, 6):
        arr = pad_and_stack(trajs, f"mut_by_nc_{k}")
        # Filter runs where state k is populated at t_idx
        vals = arr[:, t_idx]
        vals = vals[vals > 0]
        means_m.append(np.mean(vals) if len(vals) > 0 else 0.0)
        stds_m.append(np.std(vals) if len(vals) > 0 else 0.0)
    ax.bar(x_nc_mut + (i - 0.5) * width_m, means_m, width=width_m, color=COLORS[label], alpha=0.85,
           yerr=stds_m, capsize=3, label=label, edgecolor="black", lw=0.7)

# Add Diploid n_c = 2 reference line
_, d_trajs = all_data["Diploid"]
d_arr = pad_and_stack(d_trajs, "mut_by_nc_2")[:, t_idx]
d_mean = np.mean(d_arr[d_arr > 0])
ax.axhline(d_mean, color=COLORS["Diploid"], ls="--", lw=2.0, label=f"Diploid ($n_c=2$) mean ({d_mean:.2f})")

ax.set_xticks(x_nc_mut)
ax.set_xticklabels([f"$n_c = {k}$" for k in x_nc_mut])
ax.set_xlabel("Subpopulation copy-number class ($n_c$)")
ax.set_ylabel("Mean fraction of mutated alleles per cell")
ax.set_title("(d) Mutation Burden Stratified by Ploidy ($t=2000$)", fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.set_ylim(0.0, 0.95)

plt.tight_layout()
out_dir = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)
png_path = out_dir / "karyotype_mechanistic_internal_states.png"
svg_path = out_dir / "karyotype_mechanistic_internal_states.svg"
fig.savefig(png_path, dpi=250, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
print(f"Saved mechanistic internal-state figure to:\n  - {png_path}\n  - {svg_path}")
