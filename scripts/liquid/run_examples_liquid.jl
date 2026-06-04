# scripts/run_examples_liquid.jl
#
# Runs liquid-tumor simulations until at least one Health and one Tumor_Max
# example is found, then saves each to its own named directory.
#
# Usage:
#   julia --threads auto scripts/run_examples_liquid.jl [max_attempts]
#
# Outputs:
#   data/simulations_liquid/example_health/   results.npz + snapshots.npz
#   data/simulations_liquid/example_tumor/    results.npz + snapshots.npz

include("../../src/utils_liquid.jl")
include("../../src/simulation_liquid_spatial.jl")
include("parameters_liquid.jl")

using Random, NPZ

# ---------------------------------------------------------------------------
# Helper: save a SpatialResults to a directory
# ---------------------------------------------------------------------------
function save_spatial_run(spatial, dir, L)
    if !isdir(dir); mkpath(dir); end
    res     = spatial.base
    n_frames = length(spatial.snapshot_steps)

    # --- aggregate trajectories ---
    results_dict = Dict(
        "mu"            => res.mu,
        "r"             => res.r,
        "m"             => res.m,
        "n_chrs"        => res.n_chrs,
        "tumor_density" => res.tumor_density,
        "death_density" => res.dcells_density,
        "outcome_code"  => [res.state == "Health"    ? 0 :
                            res.state == "Tumor_Max" ? 1 :
                            res.state == "Tumor_Min" ? 3 : 2],
        "L"             => [L],
    )
    for (i, type) in enumerate(["I", "O", "S", "M", "HK"])
        results_dict["mut_$(type)"] = res.muts[i]
        results_dict["act_$(type)"] = res.activations[i]
    end
    npzwrite(joinpath(dir, "results.npz"), results_dict)

    # --- spatial snapshots ---
    state_3d = zeros(Int8,    n_frames, L, L)
    mu_3d    = zeros(Float64, n_frames, L, L)
    r_3d     = zeros(Float64, n_frames, L, L)
    for (f, (s, m, r)) in enumerate(zip(spatial.snapshots_state,
                                         spatial.snapshots_mu,
                                         spatial.snapshots_r))
        state_3d[f, :, :] = s
        mu_3d[f, :, :]    = m
        r_3d[f, :, :]     = r
    end
    snap_dict = Dict(
        "snapshot_steps" => spatial.snapshot_steps,
        "state"          => state_3d,
        "mu"             => mu_3d,
        "r"              => r_3d,
        "L"              => [L],
        "outcome"        => Vector{UInt8}(res.state),
    )
    npzwrite(joinpath(dir, "snapshots.npz"), snap_dict)
    println("  Saved to: $dir  ($(n_frames) snapshots)")
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function run_examples(max_attempts=200, n_snapshots=12)
    base_dir = joinpath(dirname(dirname(@__DIR__)), "data", "simulations_liquid")

    found_health = false
    found_tumor  = false

    println("--- Searching for Health + Tumor_Max examples ---")
    println("Parameters: L=$L, steps=$n_steps, n_seed=$n_seed")
    println("Will run up to $max_attempts simulations.\n")

    for attempt in 1:max_attempts
        if found_health && found_tumor
            break
        end

        Random.seed!(time_ns() + attempt * 999_983)
        tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
        perturb_liquid!(tiss, n_seed, pert_chrs)

        spatial = simulation_liquid_spatial(tiss, N_CHR, n_steps, n_snapshots, false, limit)
        outcome = spatial.base.state

        print("  Attempt $attempt → $outcome")

        if outcome == "Health" && !found_health
            save_spatial_run(spatial, joinpath(base_dir, "example_health"), L)
            found_health = true
            println("  ✓ Health example saved.")
        elseif outcome == "Tumor_Max" && !found_tumor
            save_spatial_run(spatial, joinpath(base_dir, "example_tumor"), L)
            found_tumor  = true
            println("  ✓ Tumor example saved.")
        else
            println("  (skipping)")
        end
    end

    if found_health && found_tumor
        println("\nBoth examples found and saved.")
    else
        println("\nWarning: could not find both outcomes in $max_attempts attempts.")
        println("  Health found: $found_health | Tumor found: $found_tumor")
    end
end

max_att = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 200
run_examples(max_att)
