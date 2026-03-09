#### Parameters ####
###############################

#### Chromosome parameters #####
N_CHR = 2
N_I  = 10
N_O  = 10
N_S  = 10
N_M  = 5
N_HK = 10
N0_genes  = N_I+N_O+N_S+N_M+N_HK
gene_map = create_gene_map(N_I, N_O, N_S, N_M, N_HK)

#### Wild Type Chromosome ####
chrom_gene_type = fill(collect(1:N0_genes), N_CHR)
chrom_gene_mut = fill(fill(0,N0_genes), N_CHR)

#### Perturbed Chromosome ####
pert_chrom_gene_mut = Vector{Vector{Int}}()
for i in 1:N_CHR
    push!(pert_chrom_gene_mut, copy(fill(0, N0_genes)))
end

pert_chrom_gene_mut[1][1] = 1      # I genes perturbation
pert_chrom_gene_mut[2][1] = 1      # I genes perturbation
pert_chrom_gene_mut[1][1+N_I] = 1  # O genes perturbation

#### Tissue parameters ####
L       = 200
mu0     = 0.
dmu     = 15e-3
r0      = 1.5e-1
dr      = 0.8e-2
dm      = 0.

println("Parameters defined")