# CancerEvo: Cancer Evolution Simulation Framework

[![Julia](https://img.shields.io/badge/Julia-1.9+-9558B2?logo=julia&logoColor=white)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Scientific Overview & Objectives

This repository implements a spatial stochastic simulation of cancer evolution on a 2D lattice tissue. The model tracks the accumulation of somatic mutations (in tumour suppressor, oncogene, survival, missegregation and housekeeping gene categories) and the emergence of malignant clones under selective pressure.

* **Hypothesis:** Tumour growth dynamics and stability are governed by the interplay between the replication rate advantage (`rmax`) and the mutation rate increment per gene (`dmu`). A critical boundary exists in this parameter space that separates stable, containable tumours from explosive or self-extinguishing ones.
* **Methodology:** The simulation uses a discrete-time lattice model with a **Struct-of-Arrays (SoA)** data layout and **bitwise `UInt64` chromosome encoding** for high-performance execution. Parameter sweeps over `rmax` and `dmu` identify stability regions where tumour density remains within a prescribed tolerance of a target value.

---

## 2. Directory Structure & Content Descriptions

```
CancerEvo/
├── README.md
├── Project.toml            # Julia package environment (dependencies)
├── Manifest.toml
├── .gitignore
│
├── src/                    # Core library modules (imported, not run directly)
│   ├── utils.jl            # OptimizedTissue struct, bitwise mutation and reproduction
│   ├── simulation.jl       # Main simulation loop: simulation_optimized()
│   └── interventions.jl    # Therapeutic intervention functions
│
├── scripts/                # Runnable entry points
│   ├── solid/              # Solid-tumor simulation scripts
│   │   ├── parameters.jl       # Centralized parameter configuration
│   │   ├── simulation.jl       # Single simulation run → saves results.npz
│   │   ├── ensemble_run.jl     # Parallel ensemble of simulations
│   │   └── stability_sweep.jl  # Parameter sweep over rmax × dmu
│   │
│   ├── liquid/             # Liquid-tumor simulation scripts
│   │   ├── parameters_liquid.jl   # Centralized parameter configuration
│   │   ├── simulation_liquid.jl   # Single simulation run
│   │   ├── ensemble_run_liquid.jl # Parallel ensemble of simulations
│   │   └── stability_sweep_liquid.jl # Parameter sweep
│   │
│   └── plots/              # Python plotting scripts
│       ├── solid/          # Solid-tumor plotting scripts
│       └── liquid/         # Liquid-tumor plotting scripts
│
├── notebooks/              # Exploratory Jupyter notebooks (sandbox)
│
└── data/                   # Git-ignored. Created at runtime.
    └── simulations/
        ├── results.npz             # Output of a single run
        ├── stability_results.csv   # Output of the stability sweep
        └── ensemble_results/       # Per-simulation NPZ files + summary CSV
```

### `src/`
Core reusable library modules.

- **`utils.jl`**: Defines `OptimizedTissue` (Struct-of-Arrays) and `OptimizedResults`. Implements:
  - `mutate_optimized!`: Bitwise stochastic mutation on `UInt64` chromosome masks.
  - `update_cell_rates_optimized!`: Updates `mu`, `r`, `m` from gene activation state.
  - `substitute_optimized!`: One full lattice update step — reproduction, mutation, death, missegregation.
  - `perturb_optimized!`: Seeds the initial cancer clone in a circular region.
- **`simulation.jl`**: Defines `simulation_optimized(tiss, n_chr_init, n_steps, ...)`. Runs the time loop and terminates early if the tumour grows above `limit`, shrinks below `lower_limit`, or goes extinct. Returns an `OptimizedResults` struct.
- **`interventions.jl`**: Ported intervention functions for the SoA layout:
  - `int_a_optimized!`: Clears cancer cells with probability proportional to replication rate.
  - `int_b_optimized!` / `int_c_optimized!`: Target cells by mutation rate threshold.
  - `int_d_optimized!`: Globally changes `dmu` and recomputes all cell rates.

### `scripts/`
Runnable entry points, organized into `solid/` and `liquid/` subfolders. Each script loads `../../src/` modules.

#### Solid Tumor (`scripts/solid/`)
- **`parameters.jl`**: Edit this file to configure solid-tumor simulation parameters (lattice size, baseline rates, steps, etc.).
- **`simulation.jl`**: Runs a single solid-tumor simulation and saves all time-series to `data/simulations/results.npz`.
- **`ensemble_run.jl`**: Runs `N` independent parallel simulations using `Base.Threads`. Saves per-simulation NPZ files and a summary CSV.
- **`stability_sweep.jl`**: Sweeps `rmax` and `dmu` to identify the stability boundary for solid tumors.

#### Liquid Tumor (`scripts/liquid/`)
- **`parameters_liquid.jl`**: Configure parameters for the liquid-tumor (global placement Moran process) model.
- **`simulation_liquid.jl`**: Runs a single liquid-tumor simulation.
- **`ensemble_run_liquid.jl`**: Runs parallel ensembles of liquid-tumor simulations.
- **`stability_sweep_liquid.jl`**: Sweeps parameter space to find the liquid-tumor stability boundary.

---

## 3. Main Features

### Simulation Model

Each lattice site holds a cell characterised by its state (wild-type `0`, cancer `1`, dead `2`), replication rate `r`, mutation rate `mu`, missegregation rate `m`, and up to 6 chromosomes encoded as `UInt64` bitmasks.

At each time step:
1. **Reproduction**: A Poisson-sampled number of cells replicate, placing a daughter cell in an adjacent site (weighted toward lower-rate neighbours).
2. **Mutation**: Each gene mutates independently with probability `mu` (using bitwise OR on `UInt64`).
3. **Rate update**: `mu`, `r`, `m` are recomputed from the bitwise activation of I/O/S/M gene masks.
4. **Death**: Cells with 0 or >5 chromosomes, activated HK genes, or zero mutation rate die.
5. **Missegregation**: With probability `m`, a chromosome is transferred from mother to daughter.

### Gene Types & Activation Logic

| Gene Type | # Genes | Activation Rule | Effect |
|-----------|---------|----------------|--------|
| **I** (Inhibitor) | 10 | ALL chromosomes mutated | Increases `mu` |
| **O** (Oncogene) | 10 | ANY chromosome mutated | Increases `r` |
| **S** (Survival) | 10 | ALL chromosomes mutated | Increases `r` |
| **M** (Missegregation) | 5 | ALL chromosomes mutated | Increases `m` |
| **HK** (Housekeeping) | 10 | ALL chromosomes mutated | Cell death |

### Output Format (NPZ)

All outputs use NumPy's compressed `.npz` format, readable in both Julia and Python:

```python
import numpy as np
data = np.load("data/simulations/results.npz")
# Available keys:
# mu, r, m, n_chrs, tumor_density, death_density
# outcome_code (0=Health, 1=Tumor_Max, 2=Done)
# mut_I, mut_O, mut_S, mut_M, mut_HK
# act_I, act_O, act_S, act_M, act_HK
```

---

## 4. How to Run

### Setup

Activate the Julia project environment once:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Run a Single Simulation
```bash
# Solid
julia scripts/solid/simulation.jl

# Liquid
julia scripts/liquid/simulation_liquid.jl
```

### Run a Parallel Ensemble (e.g. 50 simulations)
```bash
# Solid
julia -t auto scripts/solid/ensemble_run.jl 50

# Liquid
julia -t auto scripts/liquid/ensemble_run_liquid.jl 50
```

### Run the Stability Parameter Sweep
```bash
# Solid
julia -t auto scripts/solid/stability_sweep.jl

# Liquid
julia -t auto scripts/liquid/stability_sweep_liquid.jl
```

> **Tip:** Edit configuration parameters in `scripts/solid/parameters.jl` or `scripts/liquid/parameters_liquid.jl` before running.
