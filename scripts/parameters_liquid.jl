# scripts/parameters_liquid.jl
#
# Parameters for the liquid-tumor simulations.
# Inherits the same gene-count and rate structure as the solid-tumor model
# so that results can be directly compared.
# The only conceptual change is in the substitution kernel (global vs local),
# which is controlled by choosing simulation_liquid instead of simulation_optimized.

#### Genomic architecture ####
N_CHR = 2
N_I   = 10
N_O   = 10
N_S   = 10
N_M   = 5
N_HK  = 10

#### Tissue / rate parameters ####
L     = 200
mu0   = 0.0
dmu   = 4.5e-2
r0    = 1.5e-1
dr    = 0.8e-2
rmax  = 2 * r0
dm    = 0.0   # Missegregation rate (set > 0 to enable)

#### Simulation control ####
n_steps    = 2500
n_it_store = 100
limit      = 0.5    # Stop when WT density falls below (1 - limit)
n_seed     = 10     # Number of randomly scattered cancer seed cells
                    # (replaces the r_pert circular cluster of the solid model)
misseg_type = "whole" # "whole" or "chunk" chromosome missegregation

#### Initial perturbation ####
# Which genes are mutated in the seed cancer cells?
pert_vec = fill(0, N_I + N_O + N_S + N_M + N_HK)
pert_vec[1]      = 1   # 1st I gene
pert_vec[1+N_I]  = 1   # 1st O gene

function to_mask(vec)
    m = UInt64(0)
    for (i, v) in enumerate(vec)
        if v == 1; m |= (UInt64(1) << (i-1)); end
    end
    return m
end

pert_chr  = to_mask(pert_vec)
pert_chrs = [pert_chr, pert_chr]  # One chromosome copy per ploidy
