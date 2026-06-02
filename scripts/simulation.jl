# scripts/simulation.jl

include("../src/utils.jl")
include("../src/simulation.jl")
include("parameters.jl")

using Random, NPZ

function run_full_simulation()
    println("--- Starting Optimized Simulation ---")
    println("L = $L, steps = $n_steps, N_CHR = $N_CHR")
    
    # Initialization
    Random.seed!(time_ns()) # Use current time for unique runs, or set a fixed seed
    
    tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
    
    println("Applying initial perturbation...")
    perturb_optimized!(tiss, r_pert, pert_chrs)
    
    println("Running simulation loop...")
    results = simulation_optimized(tiss, N_CHR, n_steps, n_it_store, true, limit)
    
    # Storage
    output_dir = joinpath(dirname(@__DIR__), "data", "simulations")
    if !isdir(output_dir); mkpath(output_dir); end
    
    output_file = joinpath(output_dir, "results.npz")
    println("\nSaving results to $output_file ...")
    
    # NPZ requires a dictionary of arrays for cross-language compatibility
    results_dict = Dict(
        "mu" => results.mu,
        "r" => results.r,
        "m" => results.m,
        "n_chrs" => results.n_chrs,
        "tumor_density" => results.tumor_density,
        "death_density" => results.dcells_density,
        # Outcome codes: 0: Health, 1: Tumor, 2: Done
        "outcome_code" => [results.state == "Health" ? 0 : results.state == "Tumor" ? 1 : 2]
    )
    for (i, type) in enumerate(["I", "O", "S", "M", "HK"])
        results_dict["mut_$(type)"] = results.muts[i]
        results_dict["act_$(type)"] = results.activations[i]
    end
    
    npzwrite(output_file, results_dict)
    
    println("Simulation finished. Outcome: ", results.state)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_full_simulation()
end