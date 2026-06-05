#### Optimized Simulation Parameters ####

N_CHR = 2
N_I   = 10
N_O   = 10
N_S   = 10
N_M   = 5
N_HK  = 10

# Tissue properties
L     = 200
mu0   = 0.0
dmu   = 1.5e-2
r0    = 1.5e-1
dr    = 0.8e-2
rmax  = 2 * r0
dm    = 0 #1e-2 # Missegregation rate

# Simulation control
n_steps    = 2500
n_it_store = 100
limit      = 0.5 # Tumor size limit to stop simulation
r_pert     = 0.005 # Perturbation radius
misseg_type = "whole" # "whole" or "chunk" chromosome missegregation

# Initial perturbation: which genes are mutated in the first cancer cells?
# Bit 0 is the 1st gene. 
# Index 1 -> bit 0, Index 11 -> bit 10, etc.
pert_vec = fill(0, N_I + N_O + N_S + N_M + N_HK)
pert_vec[1] = 1      # 1st I gene
pert_vec[1+N_I] = 1  # 1st O gene

# Helper to convert to bitmask
function to_mask(vec)
    m = UInt64(0)
    for (i, v) in enumerate(vec)
        if v == 1; m |= (UInt64(1) << (i-1)); end
    end
    return m
end

pert_chr = to_mask(pert_vec)
pert_chrs = [pert_chr, pert_chr] # Initial chromosomes for the cancer seed 