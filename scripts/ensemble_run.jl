# scripts/ensemble_run.jl

include("../src/utils.jl")
include("../src/simulation.jl")
include("parameters.jl")

using Base.Threads
using Random, NPZ

# Helper function to save results in NPZ format (reusable in ensemble)
function save_results_npz(filename, results)
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
    npzwrite(filename, results_dict)
end

"""
    run_ensemble(num_sims)

Runs multiple simulations in parallel using Julia's multi-threading.
Tracks whether the tumor reached the size limit, died out, or reached max steps.
"""
function run_ensemble(num_sims=50)
    # Ensure output directory exists
    ensemble_dir = joinpath(dirname(@__DIR__), "data", "simulations", "ensemble_results")
    if !isdir(ensemble_dir); mkpath(ensemble_dir); end
    output_file = joinpath(ensemble_dir, "ensemble_results.csv")

    results_summary = Vector{Any}(undef, num_sims)
    
    println("--- Ensemble Runner ---")
    println("Running $num_sims simulations on $(Threads.nthreads()) threads.")
    println("Parameters: L=$L, steps=$n_steps, limit=$limit, N_CHR=$N_CHR")
    
    progress_lock = ReentrantLock()
    completed = 0

    @threads for i in 1:num_sims
        # Each thread gets its own RNG state for independence
        # Using the sim_id as part of the seed
        Random.seed!(time_ns() + i)
        
        # Initialize fresh tissue
        tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
        perturb_optimized!(tiss, r_pert, pert_chrs)
        
        # Run simulation (bar=false to keep terminal clean)
        res = simulation_optimized(tiss, N_CHR, n_steps, n_it_store, false, limit)
        
        # Save full NPZ result for this individual simulation
        save_results_npz(joinpath(ensemble_dir, "sim_$(i).npz"), res)
        
        # Capture outcome
        final_step = length(res.tumor_density)
        final_state = res.state
        final_size = res.tumor_density[end]
        
        results_summary[i] = (id=i, state=final_state, steps=final_step, size=final_size)
        
        lock(progress_lock) do
            completed += 1
            if completed % 5 == 0 || completed == num_sims
                println("Progress: $completed / $num_sims simulations finished.")
            end
        end
    end
    
    # Save to CSV
    open(output_file, "w") do io
        println(io, "sim_id,outcome,steps,final_size")
        for r in results_summary
            println(io, "$(r.id),$(r.state),$(r.steps),$(r.size)")
        end
    end
    
    # Print Quick Summary
    outcomes = [r.state for r in results_summary]
    health_count = count(x -> x == "Health", outcomes)
    tumor_count = count(x -> x == "Tumor", outcomes)
    done_count = count(x -> x == "Done", outcomes)
    
    println("\n--- Ensemble Summary ---")
    println("Survival (Health): $health_count")
    println("Progression (Tumor): $tumor_count")
    println("Limit reached (Steps): $done_count")
    println("Results saved to: $output_file")
end

# Check if arguments are passed via command line
if !isempty(ARGS)
    n = parse(Int, ARGS[1])
    run_ensemble(n)
else
    # Default run
    run_ensemble(20)
end
