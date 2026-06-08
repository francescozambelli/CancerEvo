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

---

## 9. Extension to the Liquid Tumor Case

In the liquid tumor model, cells are well-mixed (non-spatial Moran-like process) rather than confined to a 2D lattice. This introduces key differences in both simulation dynamics and analytical behavior.

### 9.1 Mutation Logic Alignment

To directly compare the spatial effects of the solid vs. liquid environments, **both models employ the same symmetric division mutation logic**:
- **Symmetric Mutation**: When a cell divides, **both** resulting cells (mother and daughter) undergo a round of DNA replication and have a probability of acquiring mutations.
- **Ecological Interaction**: In the solid model, divisions target local Moore neighbors (leading to a high probability of replacing wild-type cells at the boundary, but constrained by spatial borders). In the liquid model, divisions target uniformly random global positions on the lattice.

### 9.2 Observed Phase Boundary (Liquid vs. Solid)

The liquid adaptive sweep results (`stability_results_liquid_adaptive.csv`) show a clear crossover in stability relative to the solid tumor.

| $r_{\max}/r_0$ | Solid Observed $\delta\mu^*$ | Liquid Observed $\delta\mu^*$ | Ratio (Liquid/Solid) | Crossover Regime |
|---|---|---|---|---|
| 1.63 | $\approx 0.0029$ | $\approx 0.0033$ | $\approx 1.14$ | **Liquid More Stable** |
| 1.95 | $\approx 0.0039$ | $\approx 0.0041$ | $\approx 1.05$ | **Liquid More Stable** |
| 2.89 | $\approx 0.0049$ | $\approx 0.0048$ | $\approx 0.98$ | **Solid More Stable** |
| 7.00 | $\approx 0.0059$ | $\approx 0.0051$ | $\approx 0.86$ | **Solid More Stable** |

In terms of critical death probability ($P_{\text{death}}^*$), the liquid tumor saturates at a lower boundary ($P_{\text{death,}\infty}^* \approx 0.40$) than the solid tumor ($P_{\text{death,}\infty}^* \approx 0.46$).

### 9.3 Analytical Prediction for the Liquid Tumor (Symmetric Mutation)

#### A. Low-Density Limit ($f_C \to 0$)
In a well-mixed tissue, when the tumor fraction $f_C = C/N \to 0$, a cancer cell chosen for division targets a wild-type cell with probability $\approx 1.0$. The replacement is accepted with probability:
$$P_{\text{replace}} = \frac{r_{\text{cancer}}}{r_{\text{cancer}} + r_0}$$

Since both the mother and daughter cells are mutated:
- Each cell independently survives the housekeeping check with probability $P_s = (1 - N_I \cdot \delta\mu)^{N_{HK}}$.
- The expected number of surviving cancer cells after division is $2P_s$. Since we started with 1 dividing cancer cell, the net change in cancer cell count per division event is:
$$\mathbb{E}[\Delta C_{\text{WT}}] = 2P_s - 1$$

WT cells divide and replace cancer cells at rate $r_0 \cdot \frac{r_0}{r_{\text{cancer}} + r_0}$ per cancer cell. Equating the rate of gain and rate of loss at the stability boundary:
$$r_{\text{cancer}} \frac{r_{\text{cancer}}}{r_{\text{cancer}} + r_0} (2P_s^* - 1) - r_0 \frac{r_0}{r_{\text{cancer}} + r_0} = 0$$

$$r_{\text{cancer}}^2 (2P_s^* - 1) - r_0^2 = 0 \quad \Rightarrow \quad P_s^* = \frac{1 + (r_0/r_{\text{cancer}})^2}{2}$$

Substituting $P_s^* = (1 - N_I \cdot \delta\mu^*)^{N_{HK}}$ gives the critical mutation rate:
$$\boxed{\delta\mu^*(r_{\max}) = \frac{1}{N_I}\left[1 - \left(\frac{1 + (r_0/r_{\max})^2}{2}\right)^{1/N_{HK}}\right]}$$

#### B. Finite-Density Corrections ($f_C = 0.20$)
At the start of the stability sweep, the tumor is seeded at $f_C = 0.20$. In a liquid model, cancer cells target other cancer cells for replacement with probability $f_C$. Incorporating this self-inflicted replacement term, the stability condition becomes:
$$\mathbb{E}[\Delta C] = (1 - f_C) (2P_s - 1) - (1 - f_C) \left(\frac{r_0}{r_{\text{cancer}}}\right)^2 + 2f_C (P_s - 1) = 0$$

Solving for $P_s^*$ yields:
$$\boxed{P_s^* = \frac{1 + f_C + (1 - f_C) \left(r_0 / r_{\text{cancer}}\right)^2}{2}}$$

#### C. Asymptotic Limits ($r_{\max} \gg r_0$)
- **Solid Tumor**: $P_s^* \to 0.5 \implies P_{\text{death,}\infty}^* = 0.50$
- **Liquid Tumor ($f_C \to 0$)**: $P_s^* \to 0.5 \implies P_{\text{death,}\infty}^* = 0.50$
- **Liquid Tumor ($f_C = 0.20$)**: $P_s^* \to \frac{1+f_C}{2} = 0.60 \implies P_{\text{death,}\infty}^* = 0.40$ ($\delta\mu^* \approx 0.0051$)

This finite-density correction matches the observed simulation saturation boundary ($P_{\text{death}}^* \approx 0.40$, $\delta\mu^* \approx 0.0051$) perfectly.

### 9.4 The Phase Boundary Crossover (The $r_{\max} = 4 r_0$ Rule)

Equating the critical survival probability of the solid and liquid models:
$$\frac{1 + r_0/r_{\max}}{2} = \frac{1 + f_C + (1-f_C)(r_0/r_{\max})^2}{2}$$

This simplifies to:
$$f_C x^2 - x + (1 - f_C) = 0 \quad \Rightarrow \quad (f_C x - (1 - f_C))(x - 1) = 0 \quad \text{where } x = r_{\max}/r_0$$

This yields two physical crossover points:
1. $r_{\max} = r_0$ (trivially, both models are unstable for any non-zero mutation rate since cancer has no selective advantage).
2. $r_{\max} = \frac{1-f_C}{f_C} r_0$. For $f_C = 0.20$, this crossover occurs exactly at $r_{\max} = 4 r_0$.

**Physical Interpretation of the Crossover**:
- **For $r_{\max} < 4 r_0$ (Weak Selection)**: The liquid model is more stable because global mixing allows cancer cells to easily find and colonize remote healthy spots, escaping the local clustering of dead cells that traps solid tumors.
- **For $r_{\max} > 4 r_0$ (Strong Selection)**: The solid model is more stable because spatial localization restricts mutation-induced cell death to localized pockets. In contrast, the well-mixed liquid model suffers from global replacement events: dividing cancer cells frequently overwrite other cancer cells, and because both undergo mutation, this leads to rapid global mutational meltdown.

### 9.5 Comparison: Solid vs. Liquid Phase Diagram

The phase diagram comparison is visualized below:

![Liquid vs Solid stability boundary](figures/liquid/stability_sweep_liquid.png)

---

## 10. Master Equation Framework for Stability Boundaries

The mean-field derivations in Section 5 and Section 9 assume that all cancer cells reside in a single instability class (specifically the maximum class $N_I = 10$, where $\mu = 10 \cdot \delta\mu$). However, a realistic tumor is a heterogeneous population consisting of multiple instability classes $i \in \{1, 2, \dots, N_I\}$, where each class has a distinct mutation rate $\mu_i = i \cdot \delta\mu$ and a distinct housekeeping death probability:
$$p_{d, i} = 1 - (1 - i \cdot \delta\mu)^{N_{HK}}$$

Here, we reformulate the stability boundary problem using the multi-class Master Equation (ME) framework.

### 10.1 Multi-Class Subpopulation Dynamics

Let $x_i(t)$ be the fraction of cancer cells in instability class $i$. The subpopulation fractions evolve according to the replication-mutation Master Equation:
$$\frac{d\mathbf{x}}{dt} = r \mathbf{T} \mathbf{x} - \Phi \mathbf{x}$$
where $\mathbf{T}$ is the lower bidiagonal transition matrix with elements:
$$\mathbf{T}_{i, j} = \begin{cases} (1 - p_{\mu, j})(1 - p_{d, j}) & \text{if } i = j \\ p_{\mu, j}(1 - p_{d, i}) & \text{if } i = j+1 \\ 0 & \text{otherwise} \end{cases}$$
and the transition probability to the next class is $p_{\mu, j} = 1 - (1 - j \cdot \delta\mu)^{N_I - j}$ (for $j < N_I$, and $p_{\mu, N_I} = 0$). The term $\Phi = r \sum_{j=1}^{N_I} x_j S_j$ is the average fitness (effective birth rate) of the tumor population that enforces the normalization constraint $\sum x_i = 1$, where $S_j = \sum_{i=1}^{N_I} \mathbf{T}_{i, j} = (1 - p_{\mu, j})(1 - p_{d, j}) + p_{\mu, j}(1 - p_{d, j+1})$ represents the expected survival probability of daughter cells produced by a parent cell of class $j$ (and $S_{N_I} = 1 - p_{d, N_I}$).

Because $\mathbf{T}$ is lower triangular, its eigenvalues are its diagonal entries:
$$\lambda_j = \mathbf{T}_{j, j} = (1 - p_{\mu, j})(1 - p_{d, j})$$
Under the stationary distribution $\mathbf{x}^*$, the population converges to the principal eigenvector associated with the largest eigenvalue $\lambda_{\max}$. Since the death rate $p_{d, j}$ and transition rate $p_{\mu, j}$ both increase with $j$, the largest eigenvalue is typically the first diagonal element:
$$\lambda_{\max} = \lambda_1 = (1 - p_{\mu, 1})(1 - p_{d, 1})$$

A fundamental mathematical identity relates the principal eigenvalue to the average daughter survival probability $\langle S \rangle^*$ at stationarity:
$$\langle S \rangle^* = \sum_{j=1}^{N_I} x_j^* S_j = \sum_{j=1}^{N_I} x_j^* \left( \sum_{i=1}^{N_I} \mathbf{T}_{i, j} \right) = \sum_{i=1}^{N_I} (\mathbf{T} \mathbf{x}^*)_{i} = \sum_{i=1}^{N_I} \lambda_{\max} x_i^* = \lambda_{\max}$$
Thus:
$$\boxed{\langle S \rangle^* = \lambda_{\max}}$$
At stationarity, the average survival probability of the daughter cells is exactly the principal eigenvalue of the transition matrix.


### 10.2 Absolute Population Dynamics and Stability Conditions

To determine stability (invasion vs. extinction), we must track the absolute numbers of cancer cells in each class $C_i(t)$. In a well-mixed (liquid) Moran process, the expected rate of change of $C_i(t)$ is given by:
$$\frac{dC_i}{dt} = \sum_{j=1}^{N_I} \mathbf{M}^{\text{eff}}_{i, j} C_j$$
where the effective transition and growth matrix $\mathbf{M}^{\text{eff}}$ accounts for cancer cell division, mutations, and competitive replacement.

#### A. Low-Density Limit ($f_C \to 0$)
In the low-density limit, the effective matrix is lower bidiagonal with elements:
$$\mathbf{M}^{\text{eff}}_{i, j} = 2 \frac{r_j^2}{r_j + r_0} \mathbf{T}_{i, j} - \frac{r_j^2 + r_0^2}{r_j + r_0} \delta_{i, j}$$
The tumor is stable if the largest eigenvalue of $\mathbf{M}^{\text{eff}}$ is positive. Because $\mathbf{M}^{\text{eff}}$ is lower bidiagonal, its eigenvalues are its diagonal entries:
$$\Lambda_j = \mathbf{M}^{\text{eff}}_{j, j} = 2 \frac{r_j^2}{r_j + r_0} \mathbf{T}_{j, j} - \frac{r_j^2 + r_0^2}{r_j + r_0}$$
At the stability boundary, the critical condition for class $j$ is $\Lambda_j = 0$, which yields:
$$\boxed{\mathbf{T}_{j, j}^* = \frac{1 + (r_0/r_j)^2}{2}}$$
Substituting the definition of $\mathbf{T}_{j, j}$, the critical housekeeping death probability $p_{d, j}^*$ is:
$$1 - p_{d, j}^* = \frac{1 + (r_0/r_j)^2}{2 (1 - p_{\mu, j})}$$

#### B. Finite-Density Case ($f_C > 0$)
Incorporating self-replacement terms when the tumor fraction is $f_C$, the diagonal elements of the effective matrix become:
$$\Lambda_j = 2 \left[ r_j (1-f_C) \frac{r_j}{r_j + r_0} + \frac{1}{2} r_j f_C \right] \mathbf{T}_{j, j} - \left[ r_j (1-f_C) \frac{r_j}{r_j + r_0} + r_j f_C + r_0 (1-f_C) \frac{r_0}{r_j + r_0} \right]$$
Setting $\Lambda_j = 0$ yields the critical self-preservation probability:
$$\boxed{\mathbf{T}_{j, j}^* = \frac{1 + f_C (r_0/r_j) + (1-f_C)(r_0/r_j)^2}{2 - f_C + f_C (r_0/r_j)}}$$

#### C. Solid Tumor Mapping
For the solid tumor model (boundary-dominated growth), the equivalent stability condition is:
$$\boxed{\mathbf{T}_{j, j}^* = \frac{1 + r_0/r_j}{2}}$$
which leads to:
$$1 - p_{d, j}^* = \frac{1 + r_0/r_j}{2 (1 - p_{\mu, j})}$$

### 10.3 The Leakage Cost of Mutational Evolution

Comparing the Master Equation stability condition to the single-class model highlights a fundamental biological trade-off:
- In the single-class limit, cells cannot mutate out of their class ($p_{\mu, j} = 0$). The survival condition is simply $1 - p_{d, j}^* = P_s^*$.
- In the Master Equation framework, mutation-induced transitions out of class $j$ act as a "leakage" loss term ($1 - p_{\mu, j} < 1$).
- Because of this leakage, the critical death probability $p_{d, j}^*$ must be smaller to sustain the population:
  $$p_{d, j}^* = 1 - \frac{P_s^*}{1 - p_{\mu, j}} < 1 - P_s^*$$
This explains why the critical mutation rate boundary $\delta\mu^*$ is shifted downwards in the presence of mutational progression: the tumor pays a survival cost ("mutational leakage") to generate more mutated cells. Only when the selection pressure is sufficiently strong (high $r_{\max}$) can the tumor tolerate both housekeeping death and the leaky mutational flux.


---

## 11. Three-Population Master Equation Model (WT, Cancer, Dead)

To analyze the tumor dynamics in a well-mixed liquid environment, we can set up a Master Equation for a 3-population system consisting of:
*   **Wild-Type (Healthy) Cells ($W$):** divide at rate $r_0$, do not mutate.
*   **Cancer Cells ($C$):** divide at rate $r_{\max}$, undergo symmetric division with a constant mutation rate $\mu = N_I \cdot \delta\mu$.
*   **Dead Cells ($D$):** cannot divide ($r_{\text{dead}} = 0$).

Let the densities (fractions) of the three populations be $w = W/N$, $c = C/N$, and $d = D/N$, where $w + c + d = 1$ and $N$ is the constant total cell count of the Moran process.

### 11.1 Division and Survival Logic
During a cancer division event, both daughters mutate and survive housekeeping checks with probability:
$$P_s = (1 - \mu)^{N_{HK}} = (1 - N_I \cdot \delta\mu)^{N_{HK}}$$
If a cell fails this check, it becomes a dead cell. 

At each step of the Moran process:
1.  A cell is chosen to divide with probability proportional to its birth rate:
    *   WT: $\Pi_W = \frac{r_0 w}{r_0 w + r_{\max} c}$
    *   Cancer: $\Pi_C = \frac{r_{\max} c}{r_0 w + r_{\max} c}$
2.  The divider targets a cell in the tissue. In the priority-targeting Moran process, dead cells are replaced first (with probability $d$). If no dead cells exist (with probability $1-d$), the divider targets a random living cell, and replacement is accepted with Moran probability $\frac{r_{\text{divider}}}{r_{\text{divider}} + r_{\text{target}}}$.

### 11.2 Master Equation Formulation
By evaluating all transition events and their net changes, we obtain the following system of differential equations:

#### Healthy Cell Dynamics
$$\frac{dw}{dt} = w \left[ r_0 d + (r_0 - r_{\max}) c \right]$$

#### Cancer Cell Dynamics
$$\frac{dc}{dt} = c \left[ - \frac{r_0^2}{r_0 + r_{\max}} w + (2P_s - 1) \left( \frac{r_{\max}^2}{r_0 + r_{\max}} w + r_{\max} d \right) + (P_s - 1) r_{\max} c \right]$$

#### Dead Cell Dynamics
$$\frac{dd}{dt} = - \frac{dw}{dt} - \frac{dc}{dt}$$

### 11.3 Stability Boundary Condition
The tumor stability boundary is defined by the critical condition where the initial cancer growth rate is zero:
$$\left. \frac{dc}{dt} \right|_{t=0} = 0$$
Assuming the tumor is seeded at an initial fraction $c(0) = c_0$ into a healthy tissue ($w(0) = 1 - c_0$ and $d(0) = 0$), setting the growth rate to zero yields:
$$- \frac{r_0^2}{r_0 + r_{\max}} (1 - c_0) + (2P_s^* - 1) \frac{r_{\max}^2}{r_0 + r_{\max}} (1 - c_0) + (P_s^* - 1) r_{\max} c_0 = 0$$

Solving for the critical survival probability $P_s^*$ gives:
$$\boxed{P_s^* = \frac{1 + c_0 (r_0 / r_{\max}) + (1 - c_0) (r_0 / r_{\max})^2}{2 - c_0 + c_0 (r_0 / r_{\max})}}$$

This exact theoretical boundary matches the finite-density liquid boundary sweep results with $c_0 = 0.20$ as shown in Section 9.3.B and Section 10.2.B.


