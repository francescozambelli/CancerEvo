# src/utils_liquid.jl
#
# Liquid-tumor variant of utils.jl.
# All cell biology (mutations, chromosome logic, death checks, rate updates)
# is identical to the solid-tumor model.
# Two differences from the solid model:
#   1. substitute_liquid!  — daughter placed at a UNIFORMLY RANDOM global site
#      instead of a Moore neighbor (non-local competition / free diffusion).
#   2. perturb_liquid!     — seeds N_SEED cancer cells at random scattered
#      positions instead of a compact circular cluster.

include("utils_solid.jl")   # Re-use all structs, masks, mutation & rate functions

# ---------------------------------------------------------------------------
# Non-spatial Liquid Tissue Struct of Arrays (SoA)
# ---------------------------------------------------------------------------
mutable struct LiquidTissue
    N::Int
    N_genes::Int
    
    # Cell properties (Flat arrays)
    state::Vector{Int8} # 0: wt, 1: cancer, 2: dead
    mu::Vector{Float64}
    r::Vector{Float64}
    m::Vector{Float64}
    
    # Chromosomes: stored as a matrix [6, N] of UInt64 (max 6 chromosomes as per death condition)
    # Each UInt64 bit represents a gene mutation.
    chromosomes::Matrix{UInt64}
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
end

function LiquidTissue(N, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, n_chrs_init)
    state = zeros(Int8, N)
    mu = fill(mu0, N)
    r = fill(r0, N)
    m = zeros(Float64, N)
    chromosomes = zeros(UInt64, 6, N)
    n_chrs = fill(Int8(n_chrs_init), N)
    
    # Build masks
    mask_I = (UInt64(1) << N_I) - 1
    mask_O = ((UInt64(1) << N_O) - 1) << N_I
    mask_S = ((UInt64(1) << N_S) - 1) << (N_I + N_O)
    mask_M = ((UInt64(1) << N_M) - 1) << (N_I + N_O + N_S)
    mask_HK = ((UInt64(1) << N_HK) - 1) << (N_I + N_O + N_S + N_M)
    
    return LiquidTissue(N, N_I+N_O+N_S+N_M+N_HK, state, mu, r, m, chromosomes, n_chrs,
                        mu0, dmu, r0, dr, rmax, dm, 
                        mask_I, mask_O, mask_S, mask_M, mask_HK)
end

# ---------------------------------------------------------------------------
# Overloaded biology functions for LiquidTissue (identical logic to solid counterparts)
# ---------------------------------------------------------------------------

function mutate_optimized!(tiss::LiquidTissue, idx::Int)
    μ = tiss.mu[idx]
    nc = tiss.n_chrs[idx]
    for c in 1:nc
        chr = tiss.chromosomes[c, idx]
        for g in 0:(tiss.N_genes - 1)
            mask = UInt64(1) << g
            if (chr & mask) == 0
                if rand() < μ
                    chr |= mask
                end
            end
        end
        tiss.chromosomes[c, idx] = chr
    end
end

function update_cell_rates_optimized!(tiss::LiquidTissue, idx::Int)
    nc = tiss.n_chrs[idx]
    if nc == 0
        tiss.state[idx] = 2
        return
    end
    
    all_muts = tiss.chromosomes[1, idx]
    any_muts = tiss.chromosomes[1, idx]
    for c in 2:nc
        all_muts &= tiss.chromosomes[c, idx]
        any_muts |= tiss.chromosomes[c, idx]
    end
    
    n_act_I = count_ones(all_muts & tiss.mask_I)
    n_act_O = count_ones(any_muts & tiss.mask_O)
    n_act_S = count_ones(all_muts & tiss.mask_S)
    n_act_M = count_ones(all_muts & tiss.mask_M)
    
    tiss.mu[idx] = tiss.mu0 + n_act_I * tiss.dmu
    new_r = tiss.r0 + (n_act_O + n_act_S) * tiss.dr
    tiss.r[idx] = min(new_r, tiss.rmax)
    tiss.m[idx] = n_act_M * tiss.dm
end

function check_death_optimized(tiss::LiquidTissue, idx::Int)
    nc = tiss.n_chrs[idx]
    if nc == 0 || nc > 5
        return true
    end
    if tiss.state[idx] != 0 && tiss.mu[idx] == 0.0
        return true
    end
    
    # HK death: any HK gene activated in mode "all"
    all_muts = tiss.chromosomes[1, idx]
    for c in 2:nc
        all_muts &= tiss.chromosomes[c, idx]
    end
    if (all_muts & tiss.mask_HK) != 0
        return true
    end
    
    return false
end

# ---------------------------------------------------------------------------
# Liquid substitution step (operating on LiquidTissue)
# ---------------------------------------------------------------------------
"""
    substitute_liquid!(tiss, n_chrs_init)

One time-step of the liquid-tumor Moran-like process.

Target-selection rule (mirrors solid-model priority):
  1. Dead cells are preferentially targeted: with probability n_dead/N a
     dividing cell replaces a uniformly random dead slot (always succeeds).
  2. Otherwise the divider targets a uniformly random LIVING cell and the
     outcome is decided by the symmetric Moran rule:
       P(replace) = r_i / (r_i + r_j)
     so every division event always results in a replacement — turnover is
     never suppressed, which prevents dead-cell accumulation.

Missegregation logic is identical to the solid model.
"""
function substitute_liquid!(tiss::LiquidTissue, n_chrs_init::Int, misseg_type::String="whole")
    N = tiss.N
    repro_indices, _ = sample_reproducers(tiss.r)

    new_dead_indices = Int[]

    repro_indices_list = findall(repro_indices)

    # Pre-compute dead-site list once per step for O(1) priority targeting.
    dead_sites = findall(tiss.state .== 2)
    n_dead     = length(dead_sites)

    if length(repro_indices_list) > 1
        for i in repro_indices_list
            r_i        = tiss.r[i]
            target_idx = -1

            # ── Priority 1: fill a dead slot (mirrors solid-model priority) ──
            if n_dead > 0 && rand() < n_dead / N
                candidate = dead_sites[rand(1:n_dead)]
                if candidate != i
                    target_idx = candidate   # dead → always replaced
                end
            end

            # ── Priority 2: Moran competition with a random living cell ──────
            if target_idx == -1
                candidate = i
                attempts  = 0
                while attempts < 20
                    candidate = rand(1:N)
                    if candidate != i && tiss.state[candidate] != 2
                        break
                    end
                    attempts += 1
                end
                if candidate == i || tiss.state[candidate] == 2
                    continue
                end
                # Symmetric Moran: always a replacement, fitness decides winner
                r_j   = tiss.r[candidate]
                denom = r_i + r_j
                if denom == 0.0 || rand() > r_i / denom
                    continue   # target wins — no change
                end
                target_idx = candidate
            end

            if target_idx == -1; continue; end

            # ── Copy mother → daughter ────────────────────────────────────
            tiss.state[target_idx]       = tiss.state[i]
            tiss.n_chrs[target_idx]      = tiss.n_chrs[i]
            tiss.chromosomes[:, target_idx] .= tiss.chromosomes[:, i]
            tiss.mu[target_idx]          = tiss.mu[i]
            tiss.r[target_idx]           = tiss.r[i]
            tiss.m[target_idx]           = tiss.m[i]

            # ── Mutate and update both cells (symmetric division model) ──
            for idx in (i, target_idx)
                mutate_optimized!(tiss, idx)

                # State transition to cancer
                if tiss.state[idx] == 0
                    has_mut = false
                    for c in 1:tiss.n_chrs[idx]
                        if tiss.chromosomes[c, idx] != 0
                            has_mut = true; break
                        end
                    end
                    if has_mut || tiss.n_chrs[idx] != n_chrs_init
                        tiss.state[idx] = 1
                    end
                end

                if check_death_optimized(tiss, idx)
                    push!(new_dead_indices, idx)
                else
                    update_cell_rates_optimized!(tiss, idx)
                end
            end

            # ── Missegregation (mother → daughter) ────────────────────────
            mother_m = tiss.m[i]
            if rand() < mother_m && tiss.n_chrs[i] > 0
                chr_idx = rand(1:tiss.n_chrs[i])
                chr_to_move = tiss.chromosomes[chr_idx, i]
                
                if misseg_type == "chunk"
                    # Chunk-based chromosome missegregation (aneuploid scenario)
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
                        
                        # Extract chunk mutations
                        chunk_muts = chr_to_move & slice_mask
                        
                        # Remove chunk mutations from mother
                        tiss.chromosomes[chr_idx, i] &= ~slice_mask
                        
                        # Push chunk to daughter as new chromosome
                        if tiss.n_chrs[target_idx] < 6
                            tiss.n_chrs[target_idx] += 1
                            tiss.chromosomes[tiss.n_chrs[target_idx], target_idx] = chunk_muts
                        end
                    else
                        # Fallback to transferring whole chromosome
                        for k in chr_idx:(tiss.n_chrs[i]-1)
                            tiss.chromosomes[k, i] = tiss.chromosomes[k+1, i]
                        end
                        tiss.chromosomes[tiss.n_chrs[i], i] = 0
                        tiss.n_chrs[i] -= 1
                        
                        if tiss.n_chrs[target_idx] < 6
                            tiss.n_chrs[target_idx] += 1
                            tiss.chromosomes[tiss.n_chrs[target_idx], target_idx] = chr_to_move
                        end
                    end
                else
                    # Default: whole chromosome missegregation (polyploid scenario)
                    for k in chr_idx:(tiss.n_chrs[i]-1)
                        tiss.chromosomes[k, i] = tiss.chromosomes[k+1, i]
                    end
                    tiss.chromosomes[tiss.n_chrs[i], i] = 0
                    tiss.n_chrs[i] -= 1
                    
                    if tiss.n_chrs[target_idx] < 6
                        tiss.n_chrs[target_idx] += 1
                        tiss.chromosomes[tiss.n_chrs[target_idx], target_idx] = chr_to_move
                    end
                end
                
                # Re-check death after chromosome transfer
                for idx in (i, target_idx)
                    if check_death_optimized(tiss, idx)
                        push!(new_dead_indices, idx)
                    end
                end
            end
        end
    end

    for idx in unique(new_dead_indices)
        tiss.state[idx]          = 2
        tiss.r[idx]              = 0.0
        tiss.mu[idx]             = 0.0
        tiss.m[idx]              = 0.0
        tiss.n_chrs[idx]         = 0
        tiss.chromosomes[:, idx] .= 0
    end
end

# ---------------------------------------------------------------------------
# Liquid-tumor initial perturbation
# ---------------------------------------------------------------------------
"""
    perturb_liquid!(tiss, n_seed, pert_chrs)

Seed `n_seed` cancer cells at uniformly random, non-repeating positions across
the entire population. This replaces the compact circular-cluster initialisation
used by `perturb_optimized!`, which is inappropriate for a liquid tumor where
malignant cells are disseminated from the start.

Arguments
---------
- `tiss`      : the tissue to modify in-place
- `n_seed`    : number of cells to initialise as cancer (≤ N)
- `pert_chrs` : vector of UInt64 chromosome bitmasks for the seed cells
"""
function perturb_liquid!(tiss::LiquidTissue, n_seed::Int,
                         pert_chrs::Vector{UInt64})
    N        = tiss.N
    n_seed   = min(n_seed, N)          # safety clamp
    n_chrs_p = length(pert_chrs)

    # Draw n_seed unique population indices without replacement
    seed_indices = sample(1:N, n_seed; replace=false)

    for idx in seed_indices
        tiss.state[idx]  = 1
        tiss.n_chrs[idx] = n_chrs_p
        for c in 1:n_chrs_p
            tiss.chromosomes[c, idx] = pert_chrs[c]
        end
        update_cell_rates_optimized!(tiss, idx)
    end
end
