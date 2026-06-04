# Liquid-Tumor Simulation: Model Description

> **Status:** Implemented — June 2026  
> **Relevant files:** `src/utils_liquid.jl`, `src/simulation_liquid.jl`,
> `scripts/liquid/simulation_liquid.jl`, `scripts/liquid/ensemble_run_liquid.jl`

---

## 1. Motivation

The original lattice model (`src/simulation.jl`) represents a **solid tumor**: cells
are arranged on a 2-D square lattice and a dividing cell can only displace one of its
eight Moore neighbors.  This local constraint captures the contact-inhibition and
mechanical pressure that govern solid-tumor growth, where cells are physically confined
by surrounding tissue.

Many cancers, however — most prominently haematological malignancies such as leukemias
and lymphomas — behave as **liquid tumors**: malignant cells circulate freely through
the bloodstream or lymphatic system and can engraft at arbitrary body sites far from
the original clone.  The spatial locality of the Moore-neighborhood rule is
inappropriate for this biology.  The liquid-tumor model replaces it with a
**global random-placement** rule, giving every cell in the tissue equal probability of
being targeted for replacement.

---

## 2. Shared Biology

All aspects of the cell biology are **identical** between the solid and liquid models:

| Component | Description |
|-----------|-------------|
| **Genome** | Each cell carries `N_CHR` chromosomes, each a 64-bit integer (bitmask). Bits encode mutations in five gene classes: Instability (I), Oncogenes (O), Suppressor (S), Missegregation (M), Housekeeping (HK). |
| **Mutation** | At each division, every unmutated bit flips with probability `mu` (per-gene, per-chromosome). |
| **Rate update** | Division rate `r` increases with activations of O and S genes (capped at `rmax`). Mutation rate `mu` increases with activations of I genes. |
| **Death** | A cell dies if it loses all chromosomes (`n_chrs = 0`), accumulates more than 5 (`n_chrs > 5`), or activates any HK housekeeping gene in the "all-copies" sense. |
| **Missegregation** | With probability `m` (driven by M-gene activations), one chromosome is transferred from mother to daughter during division, altering ploidy in both. |
| **Reproducers** | Each time step, the number of dividing cells is drawn from `Poisson(Σ r_i)` and cells are sampled proportionally to their rates — a continuous-time Gillespie-like scheme. |

---

## 3. The Key Difference: Substitution Kernel

### 3.1 Solid Tumor (local, neighborhood-restricted)

```
substitute_optimized!(tiss, n_chrs_init)
```

For each dividing cell $i$:

1. Collect its 8 Moore neighbors.
2. If all neighbors are wild-type, **skip** (no room to expand — contact inhibition).
3. Otherwise, select a target neighbor $j$:
   - Dead neighbors ($r_j = 0$) have priority.
   - Otherwise sample $j$ with probability $\propto 1/r_j$ (slower cells are easier to displace).
4. Copy cell $i$ into position $j$ (daughter cell).

The requirement that `!all(neigh_states .== 0)` enforces spatial confinement: a cancer
clone cannot invade a fully wild-type region without first accumulating enough dividers
to form a frontier.

---

### 3.2 Liquid Tumor (global, random placement)

```julia
substitute_liquid!(tiss, n_chrs_init)
```

For each dividing cell $i$ (with division rate $r_i$):

1. **Target Selection with Dead-Cell Priority**:
   - Let $n_{\text{dead}}$ be the current number of dead cells ($r=0$) in the tissue.
   - With probability $n_{\text{dead}}/N$, target a dead cell $j$ uniformly at random. This replacement always succeeds (mimicking the priority dead-cell clearance logic of the solid tumor).
   - Otherwise, target a living cell $j$ uniformly at random.
2. **Symmetric Moran Competition**:
   - If a living target $j$ is chosen, a replacement event occurs with probability:
   
   $$P(\text{replace}) = \frac{r_i}{r_i + r_j}$$
   
   - If the random check succeeds, cell $i$ displaces target $j$. If it fails, the division is rejected (target wins, no change). This ensures cell turnover is not artificially suppressed.
3. **Daughter-Only Mutation**:
   - In accordance with the physical Moran process, mutations and karyotypic updates are applied **only to the daughter cell** (at position $j$). The mother cell (at position $i$) retains its pre-division state. This prevents an artificial, runaway accumulation of mutations in active cancer cells.

> **No spatial frontier is required.** A single cancer cell with a fitness advantage
> can immediately seed any lattice site, mimicking haematogenous dissemination.

### 3.3 Initial Perturbation (liquid model)

The solid model seeds a **compact circular cluster** of cancer cells centred on the
lattice (`perturb_optimized!`, radius `r_pert * L`).

The liquid model instead uses `perturb_liquid!`, which:

1. Draws `n_seed` lattice indices **uniformly at random without replacement** using
   `StatsBase.sample(1:N, n_seed; replace=false)`.
2. Initialises each drawn cell as a cancer cell (`state = 1`) with the prescribed
   chromosome bitmasks.

This reflects the biology of disseminated disease: malignant cells do not originate
in one contiguous focus but are spread across the tissue from the start.

---

## 4. Biological Interpretation

| Feature | Solid model | Liquid model |
|---------|-------------|--------------|
| Cell movement | None — cells are fixed | Implicit — daughters appear globally |
| Spatial structure | Strong — clones form contiguous patches | None — clones are spatially mixed |
| Contact inhibition | Yes — WT frontier blocks invasion | No |
| Competition range | Local (Moore neighborhood) | Global (entire tissue) |
| Analogy | Epithelial / stromal tumors | Leukemia, lymphoma, metastatic circulating cells |
| Expected dynamics | Slow front propagation; spatial hollowing | Rapid takeover; well-mixed mean-field limit |

---

## 5. Expected Dynamical Consequences

Because the daughter can land **anywhere**, the effective population size felt by a
mutant clone is the entire tissue rather than just its local neighborhood.  This has
several measurable consequences:

- **Faster initial expansion**: advantageous clones spread at a rate proportional to
  their fitness advantage times the total population $N$, not just the frontier length.
- **Weaker genetic drift**: in a well-mixed system the effective population size is
  large, so neutral mutations fix more slowly.
- **Higher selective coefficient needed for tumor progression**: the global competition
  means that a cell needs a larger advantage over the mean fitness to consistently
  win replacements.
- **Different phase boundary**: the stability analysis derived for the solid tumor
  (see `outputs/stability_phase_boundary_analysis.md`) does not directly apply.
  A mean-field (Moran process) approximation is more appropriate for the liquid model.

---

## 6. Implementation Notes

### Include hierarchy

```
src/utils.jl           ← shared structs, gene masks, mutation, death, rate update
  └── src/utils_liquid.jl   ← adds substitute_liquid!
        └── src/simulation_liquid.jl  ← simulation_liquid() loop
              └── scripts/liquid/simulation_liquid.jl     (single run)
              └── scripts/liquid/ensemble_run_liquid.jl   (parallel ensemble)
```

`utils_liquid.jl` uses `include("utils.jl")` so all data structures
(`OptimizedTissue`, `OptimizedResults`) and helper functions (`mutate_optimized!`,
`update_cell_rates_optimized!`, `check_death_optimized`, `sample_reproducers`,
`perturb_optimized!`, `invert`) are shared with no duplication.

### Output compatibility

All output files follow the same schema as the solid-tumor runs:

| Key | Description |
|-----|-------------|
| `mu`, `r`, `m`, `n_chrs` | Mean trait values across cancer cells, one entry per step |
| `tumor_density` | Fraction of lattice occupied by cancer cells |
| `death_density` | Fraction of lattice occupied by dead cells |
| `outcome_code` | `0` = Health, `1` = Tumor_Max, `2` = Done (steps), `3` = Tumor_Min |
| `mut_{I,O,S,M,HK}` | Mean mutation fraction per gene class |
| `act_{I,O,S,M,HK}` | Mean activation fraction per gene class |

Output is written to `data/simulations_liquid/` (single run) or
`data/simulations_liquid/ensemble_results/` (ensemble), keeping liquid and solid
results cleanly separated.  Existing Python loaders and plotting scripts can be
reused by pointing to the new data path.

---

## 7. Running the Simulations

```bash
# Single exploratory run (with progress bar)
julia scripts/liquid/simulation_liquid.jl

# Ensemble of 100 runs, all available threads
julia --threads auto scripts/liquid/ensemble_run_liquid.jl 100
```

Parameters are controlled via `scripts/liquid/parameters_liquid.jl` (mirrors
`scripts/solid/parameters.jl`).  Key knobs for comparing solid vs. liquid behaviour:

| Parameter | Variable | Default |
|-----------|----------|---------|
| Division rate baseline | `r0` | `0.15` |
| Division rate increment per oncogene | `dr` | `0.008` |
| Max division rate | `rmax` | `2 * r0` |
| Mutation rate increment per I gene | `dmu` | `0.015` |
| Lattice side | `L` | `200` |
| Simulation steps | `n_steps` | `2500` |
| Tumor-size termination threshold | `limit` | `0.5` |
| Number of scattered seed cells | `n_seed` | `50` |

---

## 8. Suggested Comparative Analyses

1. **Tumor progression probability** — fraction of ensemble runs reaching `Tumor_Max`
   vs. `Health` across a grid of `(dmu, dr)` values, comparing solid and liquid
   phase diagrams.
2. **Time to progression** — distribution of steps until `Tumor_Max`, expected to be
   shorter for the liquid model at the same parameters.
3. **Mutation accumulation trajectories** — time series of `mu`, `r`, and gene-class
   activations, expected to show faster convergence in the liquid model.
4. **Effective population size estimation** — fit a neutral Moran model to the
   variance of clone frequencies to quantify how much the liquid kernel inflates $N_e$.
