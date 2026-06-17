# scripts/run_single_spatial.jl
#
# Run single-trajectory spatial simulations of the solid tumor model.
# Uses fixed seeds (9 for Health, 1 for Tumor) to ensure reproducibility.
# Saves snapshots and aggregate results to data/simulations/spatial_run/

include("../../src/utils_solid.jl")
include("../../src/simulation_solid_spatial.jl")
include("parameters_solid.jl")

using Random, NPZ

function run_scenario(name::String, seed::Int, n_snapshots::Int=12)
    println("\nRunning scenario: $name (seed=$seed)...")
    Random.seed!(seed)

    tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
    perturb_optimized!(tiss, r_pert, pert_chrs)

    spatial = simulation_spatial(tiss, N_CHR, n_steps, n_snapshots, true, limit, 0.0, misseg_type)
    res = spatial.base

    # ---- Output directory ----
    output_dir = joinpath(dirname(dirname(@__DIR__)), "data", "simulations", "spatial_run")
    if !isdir(output_dir); mkpath(output_dir); end

    # ---- Save aggregate trajectories ----
    results_dict = Dict(
        "mu"            => res.mu,
        "r"             => res.r,
        "m"             => res.m,
        "n_chrs"        => res.n_chrs,
        "tumor_density" => res.tumor_density,
        "death_density" => res.dcells_density,
        "outcome_code"  => [res.state == "Health" ? 0 :
                            (res.state == "Tumor_Max" || res.state == "Tumor_Min" || res.state == "Tumor") ? 1 : 2],
        "L"             => [L],
        "n_steps"       => [n_steps]
    )
    for (i, type) in enumerate(["I", "O", "S", "M", "HK"])
        results_dict["mut_$(type)"] = res.muts[i]
        results_dict["act_$(type)"] = res.activations[i]
    end
    npzwrite(joinpath(output_dir, "$(name)_results.npz"), results_dict)

    # ---- Save spatial snapshots ----
    n_frames = length(spatial.snapshot_steps)
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
        "outcome"        => Vector{UInt8}(res.state)
    )
    npzwrite(joinpath(output_dir, "$(name)_snapshots.npz"), snap_dict)

    println("Finished scenario: $name. Outcome: $(res.state), Steps: $(length(res.tumor_density))")
end

# Run both scenarios
run_scenario("health", 9, 12)
run_scenario("tumor", 1, 12)
