# src/utils.jl

import Pkg
Pkg.activate(joinpath(dirname(@__DIR__)))

using Random, Statistics, ProgressMeter, JSON, StatsBase, Distributions, GLM, LaTeXStrings, SplitApplyCombine

# Optimized Result Structure
mutable struct OptimizedResults
    state::String
    mu::Vector{Float64}
    r::Vector{Float64}
    m::Vector{Float64}
    n_chrs::Vector{Float64}
    tumor_density::Vector{Float64}
    dcells_density::Vector{Float64}
    muts::Vector{Vector{Float64}}
    activations::Vector{Vector{Float64}}
    eff_ploidy::Vector{Float64}
    n_chrs_dist::Vector{Vector{Float64}}
    mut_by_nc::Vector{Vector{Float64}}
end

function OptimizedResults(state, mu, r, m, n_chrs, tumor_density, dcells_density, muts, activations)
    return OptimizedResults(
        state, mu, r, m, n_chrs, tumor_density, dcells_density, muts, activations,
        Float64[], Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
    )
end

# Optimized Tissue using Struct of Arrays (SoA)
mutable struct OptimizedTissue
    L::Int
    N_genes::Int
    
    # Cell properties (Flat arrays)
    state::Vector{Int8} # 0: wt, 1: cancer, 2: dead
    mu::Vector{Float64}
    r::Vector{Float64}
    m::Vector{Float64}
    
    # Chromosomes: stored as a matrix [6, L*L] of UInt64 (max 6 chromosomes as per death condition)
    # Each UInt64 bit represents a gene mutation (0 = WT, 1 = mutated).
    # presence: bitmask tracking whether each gene locus is physically present (1) or deleted/absent (0).
    chromosomes::Matrix{UInt64}
    presence::Matrix{UInt64}
    n_chrs::Vector{Int8}
    
    # Parameters
    mu0::Float64
    dmu::Float64
    r0::Float64
    dr::Float64
    rmax::Float64
    dm::Float64
    
    # Bitmasks for gene types
    mask_I::UInt64
    mask_O::UInt64
    mask_S::UInt64
    mask_M::UInt64
    mask_HK::UInt64
    mask_all::UInt64
    
    # Neighbor cache
    neighbors::Vector{Vector{Int}}
end

function get_neighbors_1d(L, idx)
    r = ((idx - 1) % L) + 1
    c = ((idx - 1) ÷ L) + 1
    neighs = Int[]
    for dr in -1:1, dc in -1:1
        (dr == 0 && dc == 0) && continue
        nr, nc = r + dr, c + dc
        if 1 <= nr <= L && 1 <= nc <= L
            push!(neighs, nr + (nc - 1) * L)
        end
    end
    return neighs
end

function OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, n_chrs_init)
    N = L * L
    N_genes = N_I + N_O + N_S + N_M + N_HK
    state = zeros(Int8, N)
    mu = fill(mu0, N)
    r = fill(r0, N)
    m = zeros(Float64, N)
    chromosomes = zeros(UInt64, 6, N)
    presence = zeros(UInt64, 6, N)
    n_chrs = fill(Int8(n_chrs_init), N)
    
    # Build masks
    mask_I = (UInt64(1) << N_I) - 1
    mask_O = ((UInt64(1) << N_O) - 1) << N_I
    mask_S = ((UInt64(1) << N_S) - 1) << (N_I + N_O)
    mask_M = ((UInt64(1) << N_M) - 1) << (N_I + N_O + N_S)
    mask_HK = ((UInt64(1) << N_HK) - 1) << (N_I + N_O + N_S + N_M)
    mask_all = (UInt64(1) << N_genes) - 1
    
    for c in 1:n_chrs_init
        presence[c, :] .= mask_all
    end
    
    neighs = [get_neighbors_1d(L, i) for i in 1:N]
    
    return OptimizedTissue(L, N_genes, state, mu, r, m, chromosomes, presence, n_chrs,
                           mu0, dmu, r0, dr, rmax, dm, 
                           mask_I, mask_O, mask_S, mask_M, mask_HK, mask_all, neighs)
end

# Compute effective recessive (all_muts) and dominant (any_muts) gene masks accounting for locus presence
@inline function get_cell_masks(tiss, idx::Int)
    nc = tiss.n_chrs[idx]
    functional_wt = UInt64(0)
    any_muts = UInt64(0)
    @inbounds for c in 1:nc
        pres = tiss.presence[c, idx]
        chr  = tiss.chromosomes[c, idx]
        functional_wt |= (pres & ~chr)
        any_muts      |= (pres & chr)
    end
    # A recessive locus is inactivated/lost if NO chromosome carries a present, unmutated (WT) copy
    all_muts = ~functional_wt & tiss.mask_all
    return all_muts, any_muts
end

# Fast mutation function (only present, unmutated loci can mutate)
function mutate_optimized!(tiss::OptimizedTissue, idx::Int)
    μ = tiss.mu[idx]
    nc = tiss.n_chrs[idx]
    for c in 1:nc
        chr = tiss.chromosomes[c, idx]
        pres = tiss.presence[c, idx]
        for g in 0:(tiss.N_genes - 1)
            mask = UInt64(1) << g
            if (pres & mask) != 0 && (chr & mask) == 0
                if rand() < μ
                    chr |= mask
                end
            end
        end
        tiss.chromosomes[c, idx] = chr
    end
end

# Fast activation and rate update
function update_cell_rates_optimized!(tiss::OptimizedTissue, idx::Int)
    nc = tiss.n_chrs[idx]
    if nc == 0
        tiss.state[idx] = 2
        return
    end
    
    all_muts, any_muts = get_cell_masks(tiss, idx)
    
    n_act_I = count_ones(all_muts & tiss.mask_I)
    n_act_O = count_ones(any_muts & tiss.mask_O)
    n_act_S = count_ones(all_muts & tiss.mask_S)
    n_act_M = count_ones(all_muts & tiss.mask_M)
    
    tiss.mu[idx] = tiss.mu0 + n_act_I * tiss.dmu
    new_r = tiss.r0 + (n_act_O + n_act_S) * tiss.dr
    tiss.r[idx] = min(new_r, tiss.rmax)
    tiss.m[idx] = n_act_M * tiss.dm
end

function check_death_optimized(tiss::OptimizedTissue, idx::Int)
    nc = tiss.n_chrs[idx]
    if nc == 0 || nc > 5
        return true
    end
    if tiss.state[idx] != 0 && tiss.mu[idx] == 0.0
        return true
    end
    
    # HK death: any HK gene lacks a functional WT copy (all present copies mutated, or all copies deleted)
    all_muts, _ = get_cell_masks(tiss, idx)
    if (all_muts & tiss.mask_HK) != 0
        return true
    end
    
    return false
end

# Helper to perform whole or chunk missegregation from mother (i) to daughter (target_idx)
function missegregate_cell_pair!(tiss, i::Int, target_idx::Int, misseg_type::String)
    chr_idx = rand(1:tiss.n_chrs[i])
    chr_to_move = tiss.chromosomes[chr_idx, i]
    pres_to_move = tiss.presence[chr_idx, i]
    
    if misseg_type == "chunk"
        # Chunk-based chromosome missegregation (segmental aneuploidy)
        N_genes = tiss.N_genes
        len_cut = rand(0:N_genes)
        if len_cut >= 3 && len_cut <= N_genes - 3
            # Contiguous bitwise slice with wrap-around
            start_bit = rand(0:(N_genes - 1))
            slice_mask = UInt64(0)
            for bit_idx in 0:(len_cut - 1)
                bit = (start_bit + bit_idx) % N_genes
                slice_mask |= (UInt64(1) << bit)
            end
            actual_slice = slice_mask & pres_to_move
        else
            actual_slice = pres_to_move
        end
    else
        # Default: whole chromosome missegregation (polyploid scenario)
        actual_slice = pres_to_move
    end

    if actual_slice != 0
        chunk_muts = chr_to_move & actual_slice
        chunk_pres = actual_slice

        # 1. Remove transferred loci (both presence and mutation bits) from mother's donor slot
        tiss.chromosomes[chr_idx, i] &= ~actual_slice
        tiss.presence[chr_idx, i]    &= ~actual_slice

        # Compact mother's allele copy-slots downward so copy c always occupies slot c
        for c in 1:(tiss.n_chrs[i]-1)
            for c_above in (c+1):tiss.n_chrs[i]
                can_drop = tiss.presence[c_above, i] & ~tiss.presence[c, i]
                if can_drop != 0
                    tiss.presence[c, i]          |= can_drop
                    tiss.chromosomes[c, i]       |= (tiss.chromosomes[c_above, i] & can_drop)
                    tiss.presence[c_above, i]    &= ~can_drop
                    tiss.chromosomes[c_above, i] &= ~can_drop
                end
            end
        end
        while tiss.n_chrs[i] > 0 && tiss.presence[tiss.n_chrs[i], i] == 0
            tiss.n_chrs[i] -= 1
        end

        # 2. Pack transferred loci into daughter's lowest available allele copy-slots (1..6)
        rem_pres = chunk_pres
        rem_muts = chunk_muts
        for c in 1:6
            rem_pres == 0 && break
            can_place = rem_pres & ~tiss.presence[c, target_idx]
            if can_place != 0
                tiss.presence[c, target_idx]    |= can_place
                tiss.chromosomes[c, target_idx] |= (rem_muts & can_place)
                rem_pres &= ~can_place
                rem_muts &= ~can_place
                if c > tiss.n_chrs[target_idx]
                    tiss.n_chrs[target_idx] = Int8(c)
                end
            end
        end
    end
end

# Old optimized substitution with weighted neighbor selection (backup)
function substitute_optimized_old!(tiss::OptimizedTissue, n_chrs_init::Int, misseg_type::String="whole")
    N = tiss.L * tiss.L
    # Reproducers
    repro_indices, _ = sample_reproducers(tiss.r)
    
    dead_indices = Int[]
    
    repro_indices_list = findall(repro_indices)
    
    if length(repro_indices_list) > 1
        for i in repro_indices_list
        neighs = tiss.neighbors[i]
        neigh_states = tiss.state[neighs]
        
        if !all(neigh_states .== 0) 
            neigh_r = tiss.r[neighs]
            target_idx = -1
            dead_neighs = findall(neigh_r .== 0.0)
            if !isempty(dead_neighs)
                target_idx = neighs[rand(dead_neighs)]
            else
                inv_rates = 1.0 ./ neigh_r
                target_idx = neighs[sample(1:length(neighs), Weights(inv_rates))]
            end
            
            # Daughter copies mother
            tiss.state[target_idx] = tiss.state[i]
            tiss.n_chrs[target_idx] = tiss.n_chrs[i]
            tiss.chromosomes[:, target_idx] .= tiss.chromosomes[:, i]
            tiss.presence[:, target_idx] .= tiss.presence[:, i]
            tiss.mu[target_idx] = tiss.mu[i]
            tiss.r[target_idx] = tiss.r[i]
            tiss.m[target_idx] = tiss.m[i]
            
            for idx in (i, target_idx)
                mutate_optimized!(tiss, idx)
                
                # State transition to cancer
                if tiss.state[idx] == 0
                    has_mut = false
                    for c in 1:tiss.n_chrs[idx]
                        if tiss.chromosomes[c, idx] != 0 || tiss.presence[c, idx] != tiss.mask_all
                            has_mut = true; break
                        end
                    end
                    if has_mut || tiss.n_chrs[idx] != n_chrs_init
                        tiss.state[idx] = 1
                    end
                end
                
                if check_death_optimized(tiss, idx)
                    push!(dead_indices, idx)
                else
                    update_cell_rates_optimized!(tiss, idx)
                end
            end
            
            # Missegregation (mother to daughter)
            mother_m = tiss.m[i]
            if rand() < mother_m && tiss.n_chrs[i] > 0
                missegregate_cell_pair!(tiss, i, target_idx, misseg_type)
                
                # Re-check death and update rates for both after transfer
                for idx in (i, target_idx)
                    if check_death_optimized(tiss, idx)
                        push!(dead_indices, idx)
                    else
                        update_cell_rates_optimized!(tiss, idx)
                    end
                end
            end
        end
        end
    end
    
    for idx in unique(dead_indices)
        tiss.state[idx] = 2
        tiss.r[idx] = 0.0
        tiss.mu[idx] = 0.0
        tiss.m[idx] = 0.0
        tiss.n_chrs[idx] = 0
        tiss.chromosomes[:, idx] .= 0
        tiss.presence[:, idx] .= 0
    end
end

# Optimized substitution
function substitute_optimized!(tiss::OptimizedTissue, n_chrs_init::Int, misseg_type::String="whole")
    N = tiss.L * tiss.L
    # Reproducers
    repro_indices, _ = sample_reproducers(tiss.r)
    
    dead_indices = Int[]
    
    repro_indices_list = findall(repro_indices)
    
    if length(repro_indices_list) > 1
        for i in repro_indices_list
        neighs = tiss.neighbors[i]
        neigh_states = tiss.state[neighs]
        
        # A cell can divide only if it has at least one neighbor that is NOT wild-type
        # (either cancer or dead). If it is completely isolated (surrounded only by WT cells),
        # the division event is skipped. Interior cancer cells (surrounded by other cancer cells)
        # can freely divide and replace their neighbors.
        if !all(neigh_states .== 0) 
            # Moran-like selection within the Moore neighborhood:
            # 1. Prioritize replacing dead neighbors: with probability n_dead_neighs / n_neighs,
            #    target a dead neighbor uniformly at random (always succeeds).
            # 2. Otherwise, target a living neighbor uniformly at random, and replace it with
            #    probability r_i / (r_i + r_j) (symmetric Moran competition).
            neigh_r = tiss.r[neighs]
            target_idx = -1
            dead_neighs = findall(neigh_r .== 0.0)
            n_dead_neighs = length(dead_neighs)
            n_neighs = length(neighs)
            
            if n_dead_neighs > 0 && rand() < n_dead_neighs / n_neighs
                target_idx = neighs[rand(dead_neighs)]
            else
                living_neighs = findall(neigh_r .!= 0.0)
                if !isempty(living_neighs)
                    candidate = neighs[rand(living_neighs)]
                    r_i = tiss.r[i]
                    r_j = tiss.r[candidate]
                    denom = r_i + r_j
                    if denom > 0.0 && rand() <= r_i / denom
                        target_idx = candidate
                    end
                end
            end
            
            if target_idx == -1
                continue
            end
            
            # Daughter copies mother
            tiss.state[target_idx] = tiss.state[i]
            tiss.n_chrs[target_idx] = tiss.n_chrs[i]
            tiss.chromosomes[:, target_idx] .= tiss.chromosomes[:, i]
            tiss.presence[:, target_idx] .= tiss.presence[:, i]
            tiss.mu[target_idx] = tiss.mu[i]
            tiss.r[target_idx] = tiss.r[i]
            tiss.m[target_idx] = tiss.m[i]
            
            # Mutate and finalize both cells (i = mother, target_idx = daughter).
            for idx in (i, target_idx)
                mutate_optimized!(tiss, idx)
                
                # State transition to cancer
                if tiss.state[idx] == 0
                    has_mut = false
                    for c in 1:tiss.n_chrs[idx]
                        if tiss.chromosomes[c, idx] != 0 || tiss.presence[c, idx] != tiss.mask_all
                            has_mut = true; break
                        end
                    end
                    if has_mut || tiss.n_chrs[idx] != n_chrs_init
                        tiss.state[idx] = 1
                    end
                end
                
                if check_death_optimized(tiss, idx)
                    push!(dead_indices, idx)
                else
                    update_cell_rates_optimized!(tiss, idx)
                end
            end
            
            # Missegregation (mother to daughter)
            mother_m = tiss.m[i]
            if rand() < mother_m && tiss.n_chrs[i] > 0
                missegregate_cell_pair!(tiss, i, target_idx, misseg_type)
                
                # Re-check death and update rates for both after transfer
                for idx in (i, target_idx)
                    if check_death_optimized(tiss, idx)
                        push!(dead_indices, idx)
                    else
                        update_cell_rates_optimized!(tiss, idx)
                    end
                end
            end
        end
        end
    end
    
    for idx in unique(dead_indices)
        tiss.state[idx] = 2
        tiss.r[idx] = 0.0
        tiss.mu[idx] = 0.0
        tiss.m[idx] = 0.0
        tiss.n_chrs[idx] = 0
        tiss.chromosomes[:, idx] .= 0
        tiss.presence[:, idx] .= 0
    end
end

function sample_reproducers(rates::Vector{Float64})
    R = sum(rates)
    K = rand(Poisson(R))
    n_positive = count(>(0.0), rates)
    K = min(K, n_positive)
    if K == 0
        return falses(length(rates)), zeros(Int, length(rates))
    end
    events = sample(1:length(rates), Weights(rates), K; replace=false)
    counts = zeros(Int, length(rates))
    for i in events
        counts[i] += 1
    end
    reproduced = counts .> 0
    return reproduced, events
end

function save_to_file(file::Any, name::String, vec_of_vec::Bool)
    open(string(name), "w") do io
        if vec_of_vec      
            for mi in file
                println(io, join(mi, ", "))
            end
        else
            for mi in file
                println(io, mi)
            end
        end
    end
end

function perturb_optimized!(tiss::OptimizedTissue, r_pert::Float64, pert_chrs::Vector{UInt64})
    L = tiss.L
    for i in 1:L, j in 1:L
        if (i - L/2)^2 + (j - L/2)^2 <= (r_pert * L)^2
            idx = i + (j-1)*L
            tiss.state[idx] = 1
            tiss.n_chrs[idx] = length(pert_chrs)
            tiss.chromosomes[:, idx] .= 0
            tiss.presence[:, idx] .= 0
            for c in 1:length(pert_chrs)
                tiss.chromosomes[c, idx] = pert_chrs[c]
                tiss.presence[c, idx] = tiss.mask_all
            end
            update_cell_rates_optimized!(tiss, idx)
        end
    end
end