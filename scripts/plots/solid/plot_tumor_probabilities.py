"""
plot_tumor_probabilities.py
---------------------------
Compare the probability of tumor development across ploidy conditions
(Diploid / Aneuploid / Polyploid) using the new NPZ ensemble data.

Reproduces the summary statistics shown in Cell 3 of notebooks/analysis.ipynb.

Outputs
-------
outputs/figures/tumor_probabilities.png
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from src.analysis.loaders import load_all_ploidy

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    "Diploid":   "#E63946",   # vivid red
    "Aneuploid": "#2A9D8F",   # teal-green
    "Polyploid": "#7B2D8B",   # deep purple
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading ensemble data …")
all_data = load_all_ploidy()

# ---------------------------------------------------------------------------
# Compute probabilities
# ---------------------------------------------------------------------------
labels, probs, counts, n_total = [], [], [], []
for label in ["Diploid", "Aneuploid", "Polyploid"]:
    summary, _ = all_data[label]
    n = len(summary)
    n_tumor = (summary["outcome"] != "Health").sum()
    p = n_tumor / n
    labels.append(label)
    probs.append(p)
    counts.append(n_tumor)
    n_total.append(n)
    print(f"  {label}: P(tumor) = {p:.3f}  ({n_tumor}/{n})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Tumor Development Probability by Ploidy", fontsize=16, fontweight="bold", y=1.01)

# ── Bar chart ──
ax = axes[0]
bar_colors = [COLORS[l] for l in labels]
bars = ax.bar(labels, probs, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.5)

for bar, p, n_t, n_tot in zip(bars, probs, counts, n_total):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{p:.2f}\n({n_t}/{n_tot})",
        ha="center", va="bottom", fontsize=11, color=COLORS[labels[bars.index(bar)]],
    )

ax.set_ylabel("P(Tumor)", fontsize=14)
ax.set_ylim(0, max(probs) * 1.35)
ax.yaxis.grid(True, ls="--", alpha=0.4)
ax.set_title("Tumor Probability", fontsize=14)

# ── Breakdown stacked bar ──
ax2 = axes[1]
x = np.arange(len(labels))
health_frac = [1 - p for p in probs]
tumor_frac  = probs

b1 = ax2.bar(x, health_frac, color=[c + "88" for c in bar_colors], edgecolor="white", label="Health")
b2 = ax2.bar(x, tumor_frac, bottom=health_frac, color=bar_colors, edgecolor="white", label="Tumor")

ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel("Fraction of runs", fontsize=14)
ax2.set_ylim(0, 1.05)
ax2.yaxis.grid(True, ls="--", alpha=0.4)
ax2.set_title("Outcome Breakdown", fontsize=14)

patches = [
    mpatches.Patch(facecolor="#aaaaaa88", edgecolor="white", label="Health"),
    mpatches.Patch(facecolor="#666666", edgecolor="white", label="Tumor"),
]
ax2.legend(handles=patches, loc="upper right", fontsize=11)

plt.tight_layout()

out_dir = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "solid"
out_dir.mkdir(parents=True, exist_ok=True)
out_path_png = out_dir / "tumor_probabilities.png"
out_path_svg = out_dir / "tumor_probabilities.svg"
plt.savefig(out_path_png, dpi=150, bbox_inches="tight")
plt.savefig(out_path_svg, bbox_inches="tight")
print(f"\nSaved →\n  - {out_path_png}\n  - {out_path_svg}")
plt.show()
