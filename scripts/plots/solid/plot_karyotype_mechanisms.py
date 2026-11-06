"""
plot_karyotype_mechanisms.py
----------------------------
Generates a focused 3-panel internal-state mechanistic figure for the
karyotype comparison (Diploid vs. Polyploid vs. Aneuploid), directly addressing
Reviewer 2 Major Comment 4:
  (a) Raw replication rate r(t) vs. effective growth propensity r_eff(t) = r(t)*(1 - p_d(t))
  (b) Per-division death probability p_d(t): Total vs. HK failure and >5n ploidy ceiling
  (c) Cryptic housekeeping mutation burden (mut_HK) buffered in living cells (with 0.50 cap)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
from src.analysis.loaders import load_all_ploidy

COLORS = {
    "Diploid":   "#E64B35",  # vermilion
    "Aneuploid": "#00A087",  # teal green
    "Polyploid": "#7E6148",  # dark mauve / purple-brown
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.2,
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

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
t_grid = np.arange(1, T_MAX + 1)

# ---------------------------------------------------------------------------
# Precompute Death Probabilities & Effective Growth Propensity for Panels (a, b)
# ---------------------------------------------------------------------------
SAT_LIMIT = {"Diploid": 0.500, "Polyploid": 0.800, "Aneuploid": 0.762}
metrics_ab = {}

for label in ["Diploid", "Polyploid", "Aneuploid"]:
    _, trajs = all_data[label]
    r_arr   = pad_and_stack(trajs, "r")
    mu_arr  = pad_and_stack(trajs, "mu")
    hk_arr  = pad_and_stack(trajs, "mut_HK")
    ep_arr  = pad_and_stack(trajs, "eff_ploidy")
    nc_arr  = pad_and_stack(trajs, "n_chrs")
    m_arr   = pad_and_stack(trajs, "m")
    nc5_arr = pad_and_stack(trajs, "nc_dist_5")
    nc1_arr = pad_and_stack(trajs, "nc_dist_1")
    
    # Buffer saturation fraction sigma(t) in [0, 1] accounting for slot-averaging of mut_HK
    ep_max = np.maximum(0.1, np.mean(ep_arr[:, -100:]) - 2.0)
    f_ceil = 0.5 + (SAT_LIMIT[label] - 0.5) * np.clip((ep_arr - 2.0) / ep_max, 0.0, 1.0)
    sigma  = np.clip(hk_arr / f_ceil, 0.0, 1.0)
    
    # Effective remaining WT copies per HK gene: starts at k, approaches 1.0 at saturation
    k_wt = 1.0 + (ep_arr - 1.0) * (1.0 - sigma)
    
    # 1a. Point-mutation HK failure
    mu = np.clip(mu_arr, 1e-6, 1.0)
    p_hk_mut = 1.0 - np.power(1.0 - np.power(mu, k_wt), 10.0)
    
    # 1b. Missegregation-induced HK unmasking (loss of the lone WT copy in mother)
    q_cut = 1.0 if label == "Polyploid" else (0.5 * (ep_arr / np.maximum(2.0, nc_arr)) if label == "Aneuploid" else 0.0)
    p_loc_unmask = q_cut * np.power(sigma, np.maximum(1.0, ep_arr - 1.0)) / ep_arr
    p_hk_unmask  = 0.5 * m_arr * (1.0 - np.power(np.clip(1.0 - p_loc_unmask, 0.0, 1.0), 10.0))
    
    p_hk = 1.0 - (1.0 - p_hk_mut) * (1.0 - p_hk_unmask)
    
    # 2. Allele-count threshold (>5n) violation in daughter
    if label == "Diploid":
        p_ploidy = np.zeros_like(p_hk)
    elif label == "Polyploid":
        p_ploidy = m_arr * (nc5_arr + nc1_arr)
    else:  # Aneuploid: chunk only overflows slot 5 if it hits a 5-copy locus
        f_at_5 = np.clip((ep_arr - 3.0) / 2.0, 0.0, 1.0) * 0.5
        p_ploidy = m_arr * (nc5_arr * f_at_5 + nc1_arr)
        
    # 3. Total death probability & Effective growth propensity r_eff = r * (1 - p_tot)
    p_tot = 1.0 - (1.0 - p_hk) * (1.0 - p_ploidy)
    r_eff = r_arr * (1.0 - p_tot)
    
    metrics_ab[label] = {
        "r": r_arr, "r_eff": r_eff,
        "p_tot": p_tot, "p_hk": p_hk, "p_ploidy": p_ploidy,
    }

# ---------------------------------------------------------------------------
# Panel (a): Replication Rate r(t) vs. Effective Growth Propensity r_eff(t)
# ---------------------------------------------------------------------------
ax_a = axes[0]

for label in ["Diploid", "Polyploid", "Aneuploid"]:
    r_arr = metrics_ab[label]["r"]
    r_eff = metrics_ab[label]["r_eff"]
    
    m_r    = np.mean(r_arr, axis=0)
    m_reff = np.mean(r_eff, axis=0)
    s_reff = np.std(r_eff, axis=0)
    
    # Solid line: Effective growth propensity r_eff(t) = r(t)*(1 - p_d(t))
    ax_a.plot(t_grid, m_reff, color=COLORS[label], lw=2.5, ls="-", label=f"{label} ($\\bar{{r}}_{{\\mathrm{{eff}}}}$)")
    ax_a.fill_between(t_grid, m_reff - s_reff, m_reff + s_reff, color=COLORS[label], alpha=0.12)
    # Dashed line: Raw cellular replication rate r(t)
    ax_a.plot(t_grid, m_r, color=COLORS[label], lw=1.6, ls="--", alpha=0.80)

ax_a.plot([], [], color="#444444", lw=1.6, ls="--", label=r"Raw replication rate $\bar{r}(t)$")

ax_a.set_xlabel("Time (steps)")
ax_a.set_ylabel(r"Replication vs. Effective growth $\bar{r}_{\mathrm{eff}}(t)$")
ax_a.set_title(r"(a) Raw Replication $\bar{r}(t)$ vs. Effective Growth $\bar{r}(1 - \bar{p}_{\mathrm{d}})$", fontweight="bold")
ax_a.set_xlim(0, T_MAX)
ax_a.set_ylim(0.14, 0.31)
ax_a.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9.0)

# ---------------------------------------------------------------------------
# Panel (b): Total Death Probability & Separate Contributions (HK vs. Ploidy Ceiling)
# ---------------------------------------------------------------------------
ax_b = axes[1]
for label in ["Diploid", "Polyploid", "Aneuploid"]:
    p_tot    = metrics_ab[label]["p_tot"]
    p_hk     = metrics_ab[label]["p_hk"]
    p_ploidy = metrics_ab[label]["p_ploidy"]
    
    m_tot, s_tot = np.mean(p_tot, axis=0), np.std(p_tot, axis=0)
    m_hk         = np.mean(p_hk, axis=0)
    m_pl         = np.mean(p_ploidy, axis=0)
    
    # Solid line: Total death probability
    ax_b.plot(t_grid, m_tot, color=COLORS[label], lw=2.5, ls="-", label=f"{label} (Total)")
    ax_b.fill_between(t_grid, m_tot - s_tot, m_tot + s_tot, color=COLORS[label], alpha=0.12)
    
    # Dashed line: HK inactivation contribution (for P and A where it differs from Total)
    if label != "Diploid":
        ax_b.plot(t_grid, m_hk, color=COLORS[label], lw=1.6, ls="--", alpha=0.85)
        # Dotted line: Allele-count threshold violation (>5n)
        ax_b.plot(t_grid, m_pl, color=COLORS[label], lw=1.8, ls=":", alpha=0.95)

# Style legend entries for the two separate contributions
ax_b.plot([], [], color="#444444", lw=1.6, ls="--", label=r"HK failure ($\bar{p}_{\mathrm{d},\mathcal{H}}$)")
ax_b.plot([], [], color="#444444", lw=1.8, ls=":", label=r"Allele threshold $>5n$ ($\bar{p}_{\mathrm{d},\mathrm{ploidy}}$)")

ax_b.set_xlabel("Time (steps)")
ax_b.set_ylabel(r"Per-division death probability $\bar{p}_{\mathrm{d}}(t)$")
ax_b.set_title("(b) Death Probability: Total vs. HK & Ploidy Contributions", fontweight="bold")
ax_b.legend(loc="center right", frameon=True, framealpha=0.9, fontsize=8.8)
ax_b.set_xlim(0, T_MAX)
ax_b.set_ylim(0.0, 0.30)

# ---------------------------------------------------------------------------
# Panel (c): Cryptic Housekeeping Damage in Surviving Cells
# ---------------------------------------------------------------------------
ax_c = axes[2]
for label in ["Diploid", "Polyploid", "Aneuploid"]:
    _, trajs = all_data[label]
    mut_hk = pad_and_stack(trajs, "mut_HK")
    m_hk, s_hk = np.mean(mut_hk, axis=0), np.std(mut_hk, axis=0)
    ax_c.plot(t_grid, m_hk, color=COLORS[label], lw=2.4, label=label)
    ax_c.fill_between(t_grid, m_hk - s_hk, m_hk + s_hk, color=COLORS[label], alpha=0.15)

ax_c.axhline(0.5, color="black", ls=":", lw=1.6, label="Diploid viability ceiling (1/2)")
ax_c.set_xlabel("Time (steps)")
ax_c.set_ylabel(r"Mutated fraction of present Essential alleles ($\mathrm{mut}_{\mathcal{H}}$)")
ax_c.set_title("(c) Cryptic Damage Accumulation in Viable Cells", fontweight="bold")
ax_c.legend(loc="lower right", frameon=True, framealpha=0.9)
ax_c.set_xlim(0, T_MAX)
ax_c.set_ylim(0.0, 0.90)

plt.tight_layout(w_pad=2.2)

out_dir = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)
paper_fig_dir = Path(__file__).resolve().parents[3] / "outputs" / "paper_figures"
paper_fig_dir.mkdir(parents=True, exist_ok=True)

png_path = out_dir / "karyotype_mechanistic_internal_states.png"
svg_path = out_dir / "karyotype_mechanistic_internal_states.svg"
pdf_path = out_dir / "karyotype_mechanistic_internal_states.pdf"
paper_pdf_path = paper_fig_dir / "karyotype_mechanistic_internal_states.pdf"

fig.savefig(png_path, dpi=250, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(paper_pdf_path, bbox_inches="tight")
print(f"Successfully generated 3-panel mechanistic figure:\n  - {png_path}\n  - {svg_path}\n  - {pdf_path}\n  - {paper_pdf_path}")
