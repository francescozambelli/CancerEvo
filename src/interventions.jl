# src/interventions.jl

using Random

"""
    reset_cell_optimized!(tiss, idx, n_chrs_init)

Resets the cell at `idx` in the tissue to a wild-type state.
"""
function reset_cell_optimized!(tiss::OptimizedTissue, idx::Int, n_chrs_init::Int)
    tiss.state[idx] = 0
    tiss.n_chrs[idx] = n_chrs_init
    tiss.chromosomes[:, idx] .= 0
    tiss.mu[idx] = tiss.mu0
    tiss.r[idx] = tiss.r0
    tiss.m[idx] = 0.0
end

"""
    int_a_optimized!(tiss, n_chrs_init)

Intervention A: Cancer cells are cleared with a probability based on their replication rate.
"""
function int_a_optimized!(tiss::OptimizedTissue, n_chrs_init::Int)
    for idx in findall(tiss.state .== 1)
        if rand() > 1.0 - 1.0 / (tiss.r[idx] / tiss.r0)
            reset_cell_optimized!(tiss, idx, n_chrs_init)
        end
    end
end

"""
    int_b_optimized!(tiss, i_lim, p_d, n_chrs_init)

Intervention B: Cells with mutation rate below or equal to `i_lim * dmu` are cleared with probability `p_d`.
"""
function int_b_optimized!(tiss::OptimizedTissue, i_lim::Int, p_d::Float64, n_chrs_init::Int)
    for idx in findall(tiss.mu .<= i_lim * tiss.dmu)
        if rand() < p_d
            reset_cell_optimized!(tiss, idx, n_chrs_init)
        end
    end
end

"""
    int_c_optimized!(tiss, i_lim, p_d, n_chrs_init)

Intervention C: Cells with mutation rate above or equal to `i_lim * dmu` are cleared with probability `p_d`.
"""
function int_c_optimized!(tiss::OptimizedTissue, i_lim::Int, p_d::Float64, n_chrs_init::Int)
    for idx in findall(tiss.mu .>= i_lim * tiss.dmu)
        if rand() < p_d
            reset_cell_optimized!(tiss, idx, n_chrs_init)
        end
    end
end

"""
    int_d_optimized!(tiss, new_dmu)

Intervention D: Changes the dmu parameter and updates all cellular mutation rates.
"""
function int_d_optimized!(tiss::OptimizedTissue, new_dmu::Float64)
    tiss.dmu = new_dmu
    for idx in 1:(tiss.L * tiss.L)
        if tiss.state[idx] != 2
            update_cell_rates_optimized!(tiss, idx)
        end
    end
end