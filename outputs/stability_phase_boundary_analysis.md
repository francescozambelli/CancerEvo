# Stability Phase Boundary Analysis: $\delta\mu^*(r_{\max})$

**Project:** CancerEvo  
**Date:** 2026-06-03  
**Scripts:** `scripts/solid/stability_sweep.jl`, `scripts/plots/solid/plot_stability_sweep.py`

---

## 1. Overview

This document describes the investigation of the stability phase boundary in the cancer evolution spatial simulation. The boundary $\delta\mu^*(r_{\max})$ separates the parameter region where a cancerous perturbation grows (invades healthy tissue) from the region where it is eliminated by mutation-induced death.

The key observable is: for a given maximum replication rate $r_{\max}$, what is the critical per-instability-gene mutation rate $\delta\mu^*$ above which the tumor fails to persist?

---

## 2. Simulation Model

The simulation runs on a 2D lattice of $L \times L = 200 \times 200$ cells. Each cell has a state:
- **0**: Wild-type (healthy)
- **1**: Cancer
- **2**: Dead

Each cancer cell carries a chromosome with $N_\text{genes} = N_I + N_O + N_S + N_M + N_K = 45$ genes.

| Gene class | Count | Effect when activated |
|---|---|---|
| Instability ($I$) | $N_I = 10$ | Increases mutation rate: $\mu = \mu_0 + n_{I} \cdot \delta\mu$ |
| Oncogenes ($O$) | $N_O = 10$ | Increases birth rate: $r = r_0 + (n_O + n_S) \cdot dr$ |
| Suppressors ($S$) | $N_S = 10$ | Increases birth rate (same as oncogenes) |
| Missegregation ($M$) | $N_M = 5$ | Increases chromosome missegregation rate |
| Housekeeping ($HK$) | $N_{HK} = 10$ | **Lethal** if activated |

**Birth rate:** $r_\text{cancer} = \min(r_0 + (n_O + n_S) \cdot dr,\; r_{\max})$

In the stability sweep, `dr = rmax/10`, so:
$$r_\text{cancer} = \min\!\left(r_0 + 20 \cdot \frac{r_{\max}}{10},\; r_{\max}\right) = \min(r_0 + 2r_{\max}, r_{\max}) = r_{\max}$$
since $r_0 + 2r_{\max} > r_{\max}$ for all $r_{\max} > 0$.

**→ Cancer cells always divide at rate $r_\text{cancer} = r_{\max}$ in the sweep.**

**Initial perturbation:** A circular seed of radius $R = L\sqrt{0.2/\pi} \approx 50$ cells, corresponding to 20% tissue coverage. All $N_I$, $N_O$, $N_S$ genes start mutated. $HK$ and $M$ genes start unmutated.

Initial mutation rate of cancer cells:
$$\mu = \mu_0 + N_I \cdot \delta\mu = 10 \cdot \delta\mu$$

---

## 3. Stability Boundary Definition

### 3.1 Old definition (cancer cell density) — *incorrect at high $r_{\max}$*

The original sweep terminated when:
- Cancer density $> 0.24$ → `Tumor_Max`
- Cancer density $< 0.16$ → `Tumor_Min`

**Problem:** At high $r_{\max}$, the tumor core divides rapidly, accumulating lethal HK mutations and **hollowing out**. The active cancer density drops below 0.16, triggering a false `Tumor_Min` even when healthy cells were completely depleted. This artificially drove $\delta\mu^*$ to *decrease* at high $r_{\max}$, producing a non-monotonic (peaked) phase boundary.

### 3.2 New definition (healthy cell density) — *correct*

The sweep now terminates when:
- Healthy cell density $< 0.76$ → `Tumor_Max` (tumor invaded)
- Healthy cell density $> 0.84$ → `Tumor_Min` (tumor extinct)

This correctly classifies a fully hollowed-out tumor (0% active cancer, 0% healthy) as `Tumor_Max`, since no healthy cells remain.

---

## 4. Observed Phase Boundary

The adaptive stability sweep (`stability_results_adaptive.csv`) produces the following critical values:

| $r_{\max}/r_0$ | $r_{\max}$ | $\delta\mu^*$ (observed) |
|---|---|---|
| 1.00 | 0.150 | $\approx 0.000106$ |
| 1.43 | 0.214 | $0.00243$ |
| 1.86 | 0.279 | $0.00377$ |
| 2.29 | 0.343 | $0.00444$ |
| 3.14 | 0.471 | $0.00504$ |
| 4.00 | 0.600 | $0.00542$ |
| 5.29 | 0.794 | $0.00571$ |
| 7.00 | 1.050 | $0.00588$ |

**Shape:** Monotonically increasing, saturating near $\delta\mu^* \approx 0.006$ for large $r_{\max}/r_0$.

![Phase boundary diagram](figures/solid/stability_sweep.png)

---

## 5. Analytical Prediction

### 5.1 Mean-field stability condition

Consider a single cancer cell at the boundary of the tumor cluster, adjacent to a wild-type cell.

Per simulation step:
1. The cancer cell is selected to reproduce with probability $\approx r_\text{cancer}$
2. It divides, placing the daughter at the wild-type neighbor's position
3. **Both the mother and the daughter are mutated** (each unmutated gene mutates with probability $\mu = N_I \cdot \delta\mu$)
4. A cell dies if any of its $N_{HK} = 10$ housekeeping genes is mutated

Per division, the survival probability for each cell is:
$$P_s = (1 - \mu)^{N_{HK}} = (1 - N_I \cdot \delta\mu)^{N_{HK}}$$

The net change in cancer cell count from one boundary division event:
- **Both survive** ($P_s^2$): gained a cancer cell $\Rightarrow +1$
- **Mother dies, daughter survives** ($P_s(1-P_s)$): cancer cell replaced WT, original position becomes dead $\Rightarrow 0$
- **Mother survives, daughter dies** ($P_s(1-P_s)$): WT replaced by dead cell, no net cancer gain $\Rightarrow 0$
- **Both die** ($(1-P_s)^2$): cancer lost, WT became dead $\Rightarrow -1$

Expected net gain $= P_s^2 - (1-P_s)^2 = 2P_s - 1$

The wild-type tissue exerts a competitive pressure at rate $r_0$. At the stability boundary, the cancer cluster neither expands nor contracts:
$$r_\text{cancer} \cdot (2P_s^* - 1) = r_0$$

Solving for $P_s^*$:
$$\boxed{P_s^* = \frac{1 + r_0/r_\text{cancer}}{2}}$$

Substituting back and solving for the critical mutation rate:
$$(1 - N_I \cdot \delta\mu^*)^{N_{HK}} = \frac{1 + r_0/r_\text{cancer}}{2}$$

$$\boxed{\delta\mu^*(r_{\max}) = \frac{1}{N_I}\left[1 - \left(\frac{1 + r_0/r_{\max}}{2}\right)^{1/N_{HK}}\right]}$$

### 5.2 Limiting cases

**No birth advantage** ($r_{\max} = r_0$):
$$P_s^* = 1 \quad \Rightarrow \quad \delta\mu^* = 0$$
Any mutation rate causes extinction (cancer has no advantage to sustain losses).

**Large birth advantage** ($r_{\max} \gg r_0$):
$$P_s^* \to \frac{1}{2} \quad \Rightarrow \quad \delta\mu^*_\infty = \frac{1}{N_I}\left[1 - \left(\frac{1}{2}\right)^{1/N_{HK}}\right] = \frac{1-0.5^{0.1}}{10} \approx 0.0067$$

This is the **error threshold**: the maximum mutation rate any cancer cell can tolerate, regardless of how fast it divides.

### 5.3 Comparison with simulation data

| $r_{\max}/r_0$ | Observed $\delta\mu^*$ | Predicted $\delta\mu^*$ | Ratio |
|---|---|---|---|
| 1.43 | $0.00243$ | $0.00178$ | $0.73$ |
| 1.86 | $0.00377$ | $0.00270$ | $0.72$ |
| 2.29 | $0.00444$ | $0.00336$ | $0.76$ |
| 3.14 | $0.00504$ | $0.00410$ | $0.81$ |
| 4.00 | $0.00542$ | $0.00479$ | $0.88$ |
| 7.00 | $0.00588$ | $0.00560$ | $0.95$ |

The formula **correctly captures the shape and saturation** of the phase boundary. It underestimates by ~10–25% at intermediate values. This residual gap is expected: the mean-field derivation ignores spatial effects in 2D (e.g., the fact that the boundary is a 1D interface, not a single cell pair, and that dead cells provide a buffer that WT can exploit).

---

## 6. Spatial Study of the Transition

To confirm the phase boundary interpretation, we ran spatial simulations at $r_{\max} = 0.8480$ for three values of $\delta\mu$:

| Regime | $\delta\mu$ | Healthy cells at step 400 |
|---|---|---|
| Below boundary | $0.0045$ | $19.2\%$ (tumor invaded) |
| On boundary | $0.0058$ | $77.6\%$ (stable coexistence) |
| Above boundary | $0.0070$ | $100.0\%$ (tumor extinct) |

The three regimes are clearly distinguishable spatially:

![Spatial comparison of the three boundary regimes](figures/solid/spatial_comparison.png)

---

## 7. Comparison: Fast Increase vs. Fast Decrease at High $\delta\mu$

At $\delta\mu = 0.0030$ (high mutation rate), two scenarios illustrate how $r_{\max}$ determines the fate:

- **Fast Increase** ($r_{\max} = 0.3153$): $\delta\mu < \delta\mu^*$, tumor grows as a solid expanding mass. Density crosses the upper limit at step 150.
- **Fast Decrease** ($r_{\max} = 0.8480$): the initial burst of divisions causes ~26% of cells to immediately acquire lethal HK mutations. Cancer density drops from 20% to ~14% within 2 steps, triggering extinction via the healthy-cell check.

![Fast increase vs. fast decrease at high dmu](figures/solid/high_dmu_transition.png)

The fast-decrease scenario illustrates the **initial mutational shock**: at high $r_{\max}$, cancer cells are so proliferative that the first step kills a significant fraction, and the healthy cell fraction rises above 0.84 before recovery is possible.

---

## 8. Summary

| Finding | Result |
|---|---|
| Shape of phase boundary | Monotonically increasing, saturating at $\delta\mu^*_\infty \approx 0.006$ |
| Physical origin | Balance between replication advantage and mutation-induced death |
| Key correction | **Healthy cell density** (not cancer density) is the correct stability observable |
| Old (incorrect) mechanism | Cancer density check caused artificial decrease at high $r_{\max}$ due to interior hollowing |
| Analytical formula | $\delta\mu^* = \frac{1}{N_I}\left[1 - \left(\frac{1 + r_0/r_{\max}}{2}\right)^{1/N_{HK}}\right]$ |
| Error threshold (saturation) | $\delta\mu^*_\infty = \frac{1-0.5^{1/N_{HK}}}{N_I} \approx 0.0067$ |
| Agreement with simulation | Shape and saturation correct; ~10–25% underestimate (spatial correction needed) |
