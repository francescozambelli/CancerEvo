# scripts/ensemble_run_liquid.jl
#
# Ensemble runner for the LIQUID-tumor simulation.
# Usage:
#   julia --threads auto scripts/ensemble_run_liquid.jl [num_sims]
#
# Runs `num_sims` independent liquid-tumor simulations in parallel and saves
# per-run NPZ files plus a summary CSV, mirroring the layout of ensemble_run.jl
# so that the same Python analysis scripts can be reused with a different data path.
#
# Output directory:
#   data/simulations_liquid/ensemble_results/

include("../src/utils_liquid.jl")
include("../src/simulation_liquid.jl")
include("parameters_liquid.jl")

using Base.Threads
using Random, NPZ

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
function save_results_npz_liquid(filename, results)
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
    npzwrite(filename, results_dict)
end

# ---------------------------------------------------------------------------
# Ensemble runner
# ---------------------------------------------------------------------------
"""
    run_ensemble_liquid(num_sims)

Runs `num_sims` liquid-tumor simulations in parallel (one per thread).
Each simulation is seeded independently for reproducibility.
"""
function run_ensemble_liquid(num_sims=50)
    ensemble_dir = joinpath(dirname(@__DIR__), "data", "simulations_liquid", "ensemble_results")
    if !isdir(ensemble_dir); mkpath(ensemble_dir); end
    output_file = joinpath(ensemble_dir, "ensemble_results.csv")

    results_summary = Vector{Any}(undef, num_sims)

    println("--- Liquid-Tumor Ensemble Runner ---")
    println("Running $num_sims simulations on $(Threads.nthreads()) threads.")
    println("Parameters: L=$L, steps=$n_steps, limit=$limit, N_CHR=$N_CHR")
    println("Substitution kernel: LIQUID (global random placement)")
    println("Initial seed:        $n_seed cells at random positions")

    progress_lock = ReentrantLock()
    completed = 0

    @threads for i in 1:num_sims
        Random.seed!(time_ns() + i * 1_000_003)  # independent per-thread seeds

        tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
        perturb_liquid!(tiss, n_seed, pert_chrs)

        res = simulation_liquid(tiss, N_CHR, n_steps, n_it_store, false, limit)

        save_results_npz_liquid(joinpath(ensemble_dir, "sim_$(i).npz"), res)

        final_step  = length(res.tumor_density)
        final_state = res.state
        final_size  = isempty(res.tumor_density) ? 0.0 : res.tumor_density[end]

        results_summary[i] = (id=i, state=final_state, steps=final_step, size=final_size)

        lock(progress_lock) do
            completed += 1
            if completed % 5 == 0 || completed == num_sims
                println("Progress: $completed / $num_sims simulations finished.")
            end
        end
    end

    # ---- Save summary CSV ----
    open(output_file, "w") do io
        println(io, "sim_id,outcome,steps,final_size")
        for r in results_summary
            println(io, "$(r.id),$(r.state),$(r.steps),$(r.size)")
        end
    end

    outcomes      = [r.state for r in results_summary]
    health_count  = count(x -> x == "Health",    outcomes)
    tumor_count   = count(x -> x == "Tumor_Max", outcomes)
    tmin_count    = count(x -> x == "Tumor_Min", outcomes)
    done_count    = count(x -> x == "Done",       outcomes)

    println("\n--- Liquid-Tumor Ensemble Summary ---")
    println("Survival     (Health):    $health_count")
    println("Progression  (Tumor_Max): $tumor_count")
    println("Regression   (Tumor_Min): $tmin_count")
    println("Limit reached (Steps):    $done_count")
    println("Results saved to: $output_file")
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if !isempty(ARGS)
    n = parse(Int, ARGS[1])
    run_ensemble_liquid(n)
else
    run_ensemble_liquid(20)
end
