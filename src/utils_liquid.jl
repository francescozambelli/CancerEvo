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

include("utils.jl")   # Re-use all structs, masks, mutation & rate functions

# ---------------------------------------------------------------------------
# Liquid substitution step
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
function substitute_liquid!(tiss::OptimizedTissue, n_chrs_init::Int, misseg_type::String="whole")
    N = tiss.L * tiss.L
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

            # ── Mutate and update DAUGHTER only ──────────────────────────
            # The mother (i) retains its current mutation state — mutations
            # arise in newborn cells, not retroactively in the parent.
            # (In the solid model the spatial gate limits how often each
            # cancer cell acts, so mother mutation there is less impactful;
            # here all cancer cells divide at full rate, so mutating the
            # mother would cause excessive accumulation.)
            mutate_optimized!(tiss, target_idx)

            # State transition to cancer (daughter only)
            if tiss.state[target_idx] == 0
                has_mut = false
                for c in 1:tiss.n_chrs[target_idx]
                    if tiss.chromosomes[c, target_idx] != 0
                        has_mut = true; break
                    end
                end
                if has_mut || tiss.n_chrs[target_idx] != n_chrs_init
                    tiss.state[target_idx] = 1
                end
            end

            if check_death_optimized(tiss, target_idx)
                push!(new_dead_indices, target_idx)
            else
                update_cell_rates_optimized!(tiss, target_idx)
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
the entire lattice.  This replaces the compact circular-cluster initialisation
used by `perturb_optimized!`, which is inappropriate for a liquid tumor where
malignant cells are disseminated from the start.

Arguments
---------
- `tiss`      : the tissue to modify in-place
- `n_seed`    : number of cells to initialise as cancer (≤ L²)
- `pert_chrs` : vector of UInt64 chromosome bitmasks for the seed cells
"""
function perturb_liquid!(tiss::OptimizedTissue, n_seed::Int,
                         pert_chrs::Vector{UInt64})
    N        = tiss.L * tiss.L
    n_seed   = min(n_seed, N)          # safety clamp
    n_chrs_p = length(pert_chrs)

    # Draw n_seed unique lattice indices without replacement
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
