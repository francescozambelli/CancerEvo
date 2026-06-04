# scripts/simulation_spatial_liquid.jl
#
# Run a single liquid-tumor simulation and save full spatial lattice snapshots.
#
# Usage:
#   julia scripts/simulation_spatial_liquid.jl [n_snapshots]
#
# Output:
#   data/simulations_liquid/spatial_run/
#       results.npz          — standard aggregate trajectories
#       snapshots.npz        — 3-D arrays (n_snapshots × L × L) of state / mu / r
#
# The snapshots.npz file is what plot_liquid_spatial.py reads for visualization.

include("../src/utils_liquid.jl")
include("../src/simulation_liquid_spatial.jl")
include("parameters_liquid.jl")

using Random, NPZ

function run_spatial_simulation(n_snapshots=12)
    println("--- Liquid-Tumor Spatial Simulation ---")
    println("L = $L, steps = $n_steps, N_CHR = $N_CHR, n_seed = $n_seed")
    println("Will capture $n_snapshots spatial snapshots during the run.")

    Random.seed!(time_ns())

    tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
    println("Seeding $n_seed cancer cells at random positions...")
    perturb_liquid!(tiss, n_seed, pert_chrs)

    println("Running simulation loop...")
    spatial = simulation_liquid_spatial(tiss, N_CHR, n_steps, n_snapshots, true, limit)
    res     = spatial.base

    # ---- Output directory ----
    output_dir = joinpath(dirname(@__DIR__), "data", "simulations_liquid", "spatial_run")
    if !isdir(output_dir); mkpath(output_dir); end

    # ---- Save aggregate trajectories ----
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
        "n_steps"       => [n_steps],
        "n_seed"        => [n_seed],
    )
    for (i, type) in enumerate(["I", "O", "S", "M", "HK"])
        results_dict["mut_$(type)"] = res.muts[i]
        results_dict["act_$(type)"] = res.activations[i]
    end
    npzwrite(joinpath(output_dir, "results.npz"), results_dict)

    # ---- Save spatial snapshots ----
    # Stack into 3-D arrays: shape (n_frames, L, L)
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
        "outcome"        => Vector{UInt8}(res.state),   # decode in Python: arr.tobytes().decode()
    )
    npzwrite(joinpath(output_dir, "snapshots.npz"), snap_dict)

    println("\nSimulation finished. Outcome: $(res.state)")
    println("Saved $(n_frames) snapshots and aggregate trajectories to:")
    println("  $(output_dir)/")
end

n_snap = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 12
run_spatial_simulation(n_snap)
