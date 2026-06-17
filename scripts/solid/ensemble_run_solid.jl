# scripts/ensemble_run.jl

include("../../src/utils_solid.jl")
include("../../src/simulation_solid.jl")
include("parameters_solid.jl")

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
        "dm" => [dm],
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
function run_ensemble(num_sims=50, misseg_mech=misseg_type, suffix="", dm=dm)
    # Ensure output directory exists
    ensemble_dir = joinpath(dirname(dirname(@__DIR__)), "data", "simulations", "ensemble_results$(suffix)")
    if !isdir(ensemble_dir); mkpath(ensemble_dir); end
    output_file = joinpath(ensemble_dir, "ensemble_results.csv")

    results_summary = Vector{Any}(undef, num_sims)
    
    println("--- Ensemble Runner ---")
    println("Running $num_sims simulations on $(Threads.nthreads()) threads.")
    println("Parameters: L=$L, steps=$n_steps, limit=$limit, N_CHR=$N_CHR, misseg_type=$misseg_mech, dmu=$dmu, mu0=$mu0, r0=$r0, dr=$dr, rmax=$rmax, dm=$dm")
    if !isempty(suffix)
        println("Suffix: $suffix")
    end
    
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
        res = simulation_optimized(tiss, N_CHR, n_steps, n_it_store, false, limit, 0.0, misseg_mech)
        
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

suffix = ""
# Find and extract suffix if present (supports both --suffix and -s)
idx = findfirst(x -> x == "--suffix" || x == "-s", ARGS)
if !isnothing(idx)
    if idx < length(ARGS)
        suffix = ARGS[idx+1]
        deleteat!(ARGS, [idx, idx+1])
    else
        deleteat!(ARGS, idx)
    end
end

dm_val = dm
# Find and extract dm if present
idx_dm = findfirst(x -> x == "--dm", ARGS)
if !isnothing(idx_dm)
    if idx_dm < length(ARGS)
        dm_val = parse(Float64, ARGS[idx_dm+1])
        deleteat!(ARGS, [idx_dm, idx_dm+1])
    else
        deleteat!(ARGS, idx_dm)
    end
end

if length(ARGS) >= 3
    n = parse(Int, ARGS[1])
    m_mech = ARGS[2]
    # If suffix is not already set via flag, take the 3rd argument
    if isempty(suffix)
        suffix = ARGS[3]
    end
    run_ensemble(n, m_mech, suffix, dm_val)
elseif length(ARGS) == 2
    n = parse(Int, ARGS[1])
    m_mech = ARGS[2]
    run_ensemble(n, m_mech, suffix, dm_val)
elseif length(ARGS) == 1
    n = parse(Int, ARGS[1])
    run_ensemble(n, misseg_type, suffix, dm_val)
else
    # Default run
    run_ensemble(20, misseg_type, suffix, dm_val)
end
