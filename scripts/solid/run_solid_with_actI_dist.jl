# scripts/solid/run_solid_with_actI_dist.jl
#
# Runs the solid-tumor simulation inline until a Tumor outcome is found whose
# stationary tail has mu in [MU_LO, MU_HI]. Records per-class actI_dist at
# every timestep.
#
# Usage:
#   julia scripts/solid/run_solid_with_actI_dist.jl [MU_LO] [MU_HI] [outfile]
#
# Defaults:
#   MU_LO   = 0.030
#   MU_HI   = 0.035
#   outfile = sim_solid_actI.npz

include("../../src/utils.jl")
include("../../src/simulation.jl")
include("parameters.jl")

using Random, NPZ, Statistics

# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
MU_LO     = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.030
MU_HI     = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.035
OUT_NAME  = length(ARGS) >= 3 ? ARGS[3] : "sim_solid_actI.npz"
const TAIL_FRAC = 0.70

println("Target mu range: [$MU_LO, $MU_HI]  →  $OUT_NAME")

# ---------------------------------------------------------------------------
# Per-class act_I histogram for one timestep
# Returns Float64 vector of length N_I+1, normalised (sum = 1)
# ---------------------------------------------------------------------------
function actI_class_fractions(tiss::OptimizedTissue, cancer_idx::Vector{Int}, N_I::Int)
    counts = zeros(Int, N_I + 1)
    mask   = tiss.mask_I
    for idx in cancer_idx
        nc = tiss.n_chrs[idx]
        combined = tiss.chromosomes[1, idx]
        for c in 2:nc
            combined &= tiss.chromosomes[c, idx]   # recessive: AND across copies
        end
        k = count_ones(combined & mask)             # 0 … N_I active I genes
        counts[k + 1] += 1
    end
    n = length(cancer_idx)
    return n > 0 ? Float64.(counts) ./ n : zeros(Float64, N_I + 1)
end

# ---------------------------------------------------------------------------
# Main loop: every attempt runs the FULL inline simulation and records data
# ---------------------------------------------------------------------------
function run_until_mu()
    attempt = 0
    while true
        attempt += 1
        println("Attempt $attempt …")
        Random.seed!(time_ns())

        tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
        perturb_optimized!(tiss, r_pert, pert_chrs)

        # ---- Pre-allocate accumulators ----
        mu_vec       = Float64[]
        r_vec        = Float64[]
        m_vec        = Float64[]
        nchrs_vec    = Float64[]
        tdensity_vec = Float64[]
        ddensity_vec = Float64[]
        mut_vecs     = [Float64[] for _ in 1:5]
        act_vecs     = [Float64[] for _ in 1:5]
        actI_dist_rows = Vector{Float64}[]

        final_state = "Done"

        for k in 1:n_steps
            substitute_optimized!(tiss, N_CHR, misseg_type)

            cancer_idx = findall(tiss.state .== 1)
            dead_idx   = findall(tiss.state .== 2)
            n_canc     = length(cancer_idx)

            density = n_canc / (tiss.L^2)
            push!(tdensity_vec, density)
            push!(ddensity_vec, length(dead_idx) / (tiss.L^2))

            if n_canc > 0
                push!(mu_vec,    mean(tiss.mu[cancer_idx]))
                push!(r_vec,     mean(tiss.r[cancer_idx]))
                push!(m_vec,     mean(tiss.m[cancer_idx]))
                push!(nchrs_vec, mean(tiss.n_chrs[cancer_idx]))

                for (gi, (mask, mode)) in enumerate([
                        (tiss.mask_I, "all"), (tiss.mask_O, "any"),
                        (tiss.mask_S, "all"), (tiss.mask_M, "all"),
                        (tiss.mask_HK, "all")])
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
                    push!(mut_vecs[gi], m_sum / n_canc)
                    push!(act_vecs[gi], a_sum / n_canc)
                end

                push!(actI_dist_rows, actI_class_fractions(tiss, cancer_idx, N_I))

            else
                push!(mu_vec, 0.0); push!(r_vec, 0.0)
                push!(m_vec, 0.0);  push!(nchrs_vec, 0.0)
                for gi in 1:5
                    push!(mut_vecs[gi], 0.0); push!(act_vecs[gi], 0.0)
                end
                push!(actI_dist_rows, zeros(Float64, N_I + 1))
            end

            # Termination
            if n_canc == 0
                final_state = "Health"; break
            end
            n_wt = count(tiss.state .== 0)
            if n_wt / (tiss.L^2) < (1.0 - limit)
                final_state = "Tumor_Max"; break
            end
        end

        # ---- Check criteria ----
        oc = final_state == "Tumor_Max" ? 1 : (final_state == "Health" ? 0 : 2)
        if oc ∉ [1, 2]
            println("  → $final_state, skipping.")
            continue
        end

        T    = length(mu_vec)
        tail = mu_vec[max(1, Int(floor(TAIL_FRAC * T))):end]
        mu_tail_mean = isempty(tail) ? 0.0 : mean(tail)

        if !(MU_LO <= mu_tail_mean <= MU_HI)
            println("  → $final_state, mu_tail=$(round(mu_tail_mean, digits=4)), out of range.")
            continue
        end

        println("  ✓ Accepted: $final_state, mu_tail=$(round(mu_tail_mean, digits=4)), T=$T steps.")

        # ---- Build actI_dist matrix (T × (N_I+1)) ----
        actI_dist_mat = reduce(hcat, actI_dist_rows)'  # T × (N_I+1)

        # ---- Save ----
        out_dir = joinpath(dirname(dirname(@__DIR__)), "data", "simulations")
        mkpath(out_dir)
        out_file = joinpath(out_dir, OUT_NAME)

        results_dict = Dict(
            "mu"            => mu_vec,
            "r"             => r_vec,
            "m"             => m_vec,
            "n_chrs"        => nchrs_vec,
            "tumor_density" => tdensity_vec,
            "death_density" => ddensity_vec,
            "outcome_code"  => [oc],
            "actI_dist"     => actI_dist_mat,   # T × (N_I+1)
        )
        for (i, gtype) in enumerate(["I", "O", "S", "M", "HK"])
            results_dict["mut_$(gtype)"] = mut_vecs[i]
            results_dict["act_$(gtype)"] = act_vecs[i]
        end

        npzwrite(out_file, results_dict)
        println("Saved → $out_file")
        println("actI_dist shape: $(size(actI_dist_mat))")
        break
    end
end

run_until_mu()
