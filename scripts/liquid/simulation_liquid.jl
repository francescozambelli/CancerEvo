# scripts/simulation_liquid.jl
#
# Single-run entry point for the LIQUID-tumor simulation.
# Usage:
#   julia scripts/simulation_liquid.jl
#
# Output:
#   data/simulations_liquid/results.npz

include("../../src/utils_liquid.jl")
include("../../src/simulation_liquid.jl")
include("parameters_liquid.jl")

using Random, NPZ

function run_full_simulation_liquid()
    println("--- Starting Liquid-Tumor Simulation ---")
    println("L = $L, steps = $n_steps, N_CHR = $N_CHR")
    println("Daughters placed at uniformly random lattice positions (liquid model).")

    Random.seed!(time_ns())

    tiss = LiquidTissue(L * L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)

    println("Seeding $n_seed cancer cells at random positions (liquid perturbation)...")
    perturb_liquid!(tiss, n_seed, pert_chrs)

    println("Running simulation loop with missegregation mechanism: ", misseg_type)
    results = simulation_liquid(tiss, N_CHR, n_steps, n_it_store, true, limit, 0.0, misseg_type)

    # ---- Storage ----
    output_dir = joinpath(dirname(dirname(@__DIR__)), "data", "simulations_liquid")
    if !isdir(output_dir); mkpath(output_dir); end

    output_file = joinpath(output_dir, "results.npz")
    println("\nSaving results to $output_file ...")

    results_dict = Dict(
        "mu"            => results.mu,
        "r"             => results.r,
        "m"             => results.m,
        "n_chrs"        => results.n_chrs,
        "tumor_density" => results.tumor_density,
        "death_density" => results.dcells_density,
        # Outcome codes: 0 = Health, 1 = Tumor_Max, 2 = Done, 3 = Tumor_Min
        "outcome_code"  => [results.state == "Health"    ? 0 :
                            results.state == "Tumor_Max" ? 1 :
                            results.state == "Tumor_Min" ? 3 : 2]
    )
    for (i, type) in enumerate(["I", "O", "S", "M", "HK"])
        results_dict["mut_$(type)"] = results.muts[i]
        results_dict["act_$(type)"] = results.activations[i]
    end

    npzwrite(output_file, results_dict)
    println("Simulation finished. Outcome: ", results.state)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_full_simulation_liquid()
end
