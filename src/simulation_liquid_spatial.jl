# src/simulation_liquid_spatial.jl
#
# Spatial snapshot variant of the liquid-tumor simulation.
# Identical dynamics to simulation_liquid but also saves full L×L lattice
# arrays (state, mu, r) at n_snapshots equally-spaced time points.
# The snapshots are returned alongside the usual aggregate trajectories so
# that spatial and temporal analyses can be done from a single run.

include("utils_liquid.jl")

# ---------------------------------------------------------------------------
# Extended result struct that carries spatial snapshots
# ---------------------------------------------------------------------------
mutable struct SpatialResults
    # Standard aggregate trajectories (one value per step)
    base::OptimizedResults

    # Spatial snapshots: each is an L×L matrix
    snapshot_steps::Vector{Int}       # which step each snapshot was taken
    snapshots_state::Vector{Matrix{Int8}}    # 0=WT, 1=cancer, 2=dead
    snapshots_mu::Vector{Matrix{Float64}}    # mutation rate field
    snapshots_r::Vector{Matrix{Float64}}     # division rate field
end

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
"""
    simulation_liquid_spatial(tiss, n_chr_init, n_steps, n_snapshots, bar, limit, lower_limit)

Run the liquid-tumor simulation and record `n_snapshots` spatially-resolved
lattice frames distributed uniformly across the run.

Returns a `SpatialResults` with:
- `.base`            : standard `OptimizedResults` (aggregate metrics)
- `.snapshot_steps`  : actual step indices of each saved frame
- `.snapshots_state` : vector of L×L Int8 matrices (0=WT, 1=cancer, 2=dead)
- `.snapshots_mu`    : vector of L×L Float64 matrices (mutation rate per cell)
- `.snapshots_r`     : vector of L×L Float64 matrices (division rate per cell)
"""
function simulation_liquid_spatial(tiss::OptimizedTissue, n_chr_init::Int,
                                   n_steps::Int, n_snapshots::Int=10,
                                   bar=true, limit=0.5, lower_limit=0.0, misseg_type::String="whole")

    L = tiss.L
    N = L * L

    # Fire a snapshot every snap_interval steps.
    # We use n_steps ÷ (n_snapshots * 8) so that even if the run terminates early
    # (e.g. Tumor_Max at step 10% of n_steps) we still collect ~n_snapshots frames.
    snap_interval = max(1, n_steps ÷ (n_snapshots * 8))
    last_snap_step = -1   # track to avoid duplicate final-step snapshots

    res = OptimizedResults(
        "Done",
        Float64[], Float64[], Float64[], Float64[],
        Float64[], Float64[],
        Vector{Vector{Float64}}(),
        Vector{Vector{Float64}}()
    )

    spatial = SpatialResults(
        res,
        Int[],
        Matrix{Int8}[],
        Matrix{Float64}[],
        Matrix{Float64}[]
    )

    @showprogress enabled=bar for k in 1:n_steps
        substitute_liquid!(tiss, n_chr_init, misseg_type)

        # ---- Aggregate metrics (identical to simulation_liquid) ----
        cancer_idx = findall(tiss.state .== 1)
        dead_idx   = findall(tiss.state .== 2)
        n_canc     = length(cancer_idx)

        density = n_canc / N
        push!(res.tumor_density, density)
        push!(res.dcells_density, length(dead_idx) / N)

        if n_canc > 0
            push!(res.mu, mean(tiss.mu[cancer_idx]))
            push!(res.r,  mean(tiss.r[cancer_idx]))
            push!(res.m,  mean(tiss.m[cancer_idx]))
            push!(res.n_chrs, mean(tiss.n_chrs[cancer_idx]))

            mut_means = Float64[]; act_means = Float64[]
            for (mask, mode) in [(tiss.mask_I, "all"), (tiss.mask_O, "any"),
                                  (tiss.mask_S, "all"), (tiss.mask_M, "all"),
                                  (tiss.mask_HK, "all")]
                total_bits = count_ones(mask)
                m_sum = 0.0; a_sum = 0.0
                for idx in cancer_idx
                    nc = tiss.n_chrs[idx]
                    if mode == "all"
                        combined = tiss.chromosomes[1, idx]
                        for c in 2:nc; combined &= tiss.chromosomes[c, idx]; end
                    else
                        combined = tiss.chromosomes[1, idx]
                        for c in 2:nc; combined |= tiss.chromosomes[c, idx]; end
                    end
                    a_sum += count_ones(combined & mask)
                    c_sum = 0.0
                    for c in 1:nc
                        c_sum += count_ones(tiss.chromosomes[c, idx] & mask) / total_bits
                    end
                    m_sum += c_sum / nc
                end
                push!(mut_means, m_sum / n_canc)
                push!(act_means, a_sum / n_canc)
            end
            push!(res.muts, mut_means)
            push!(res.activations, act_means)
        else
            push!(res.mu, 0.0); push!(res.r, 0.0); push!(res.m, 0.0); push!(res.n_chrs, 0.0)
            push!(res.muts, zeros(5)); push!(res.activations, zeros(5))
        end

        # ---- Spatial snapshot (every snap_interval steps) ----
        if k % snap_interval == 0 || k == 1
            last_snap_step = k
            push!(spatial.snapshot_steps, k)
            push!(spatial.snapshots_state, reshape(copy(tiss.state), L, L))
            push!(spatial.snapshots_mu,    reshape(copy(tiss.mu),    L, L))
            push!(spatial.snapshots_r,     reshape(copy(tiss.r),     L, L))
        end

        # ---- Termination ----
        function capture_final()
            if last_snap_step != k
                push!(spatial.snapshot_steps, k)
                push!(spatial.snapshots_state, reshape(copy(tiss.state), L, L))
                push!(spatial.snapshots_mu,    reshape(copy(tiss.mu),    L, L))
                push!(spatial.snapshots_r,     reshape(copy(tiss.r),     L, L))
            end
        end
        if n_canc == 0
            res.state = "Health"; capture_final(); break
        end
        n_wt = count(tiss.state .== 0)
        wt_density = n_wt / N
        if wt_density < (1.0 - limit)
            println("Over"); res.state = "Tumor_Max"; capture_final(); break
        end
        if wt_density > (1.0 - lower_limit) && k > 1
            println("Under"); res.state = "Tumor_Min"; capture_final(); break
        end
    end

    if res.state == "Done"; println("Completed steps"); end

    res.muts        = invert(res.muts)
    res.activations = invert(res.activations)
    spatial.base    = res

    return spatial
end
