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

include("../../src/utils_liquid.jl")
include("../../src/simulation_liquid.jl")
include("parameters_liquid.jl")

using Base.Threads
using Random, NPZ

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
function save_results_npz_liquid(filename, results, dm_val=dm)
    results_dict = Dict(
        "mu"            => results.mu,
        "r"             => results.r,
        "m"             => results.m,
        "n_chrs"        => results.n_chrs,
        "eff_ploidy"    => results.eff_ploidy,
        "tumor_density" => results.tumor_density,
        "death_density" => results.dcells_density,
        "dm"            => [dm_val],
        # Outcome codes: 0 = Health, 1 = Tumor_Max, 2 = Done, 3 = Tumor_Min
        "outcome_code"  => [results.state == "Health"    ? 0 :
                            results.state == "Tumor_Max" ? 1 :
                            results.state == "Tumor_Min" ? 3 : 2]
    )
    for (i, type) in enumerate(["I", "O", "S", "M", "HK"])
        results_dict["mut_$(type)"] = results.muts[i]
        results_dict["act_$(type)"] = results.activations[i]
    end
    for k in 1:min(5, length(results.n_chrs_dist))
        results_dict["nc_dist_$(k)"] = results.n_chrs_dist[k]
        results_dict["mut_by_nc_$(k)"] = results.mut_by_nc[k]
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
function run_ensemble_liquid(num_sims=50, misseg_mech=misseg_type, suffix="", dm_val=dm, dmu_val=dmu)
    ensemble_dir = joinpath(dirname(dirname(@__DIR__)), "data", "simulations_liquid", "ensemble_results$(suffix)")
    if !isdir(ensemble_dir); mkpath(ensemble_dir); end
    output_file = joinpath(ensemble_dir, "ensemble_results.csv")

    results_summary = Vector{Any}(undef, num_sims)

    println("--- Liquid-Tumor Ensemble Runner ---")
    println("Running $num_sims simulations on $(Threads.nthreads()) threads.")
    println("Parameters: L=$L, steps=$n_steps, limit=$limit, N_CHR=$N_CHR, misseg_type=$misseg_mech, dmu=$dmu_val, mu0=$mu0, r0=$r0, dr=$dr, rmax=$rmax, dm=$dm_val")
    if !isempty(suffix)
        println("Suffix: $suffix")
    end
    println("Substitution kernel: LIQUID (global random placement)")
    println("Initial seed:        $n_seed cells at random positions")

    progress_lock = ReentrantLock()
    completed = 0

    @threads for i in 1:num_sims
        Random.seed!(time_ns() + i * 1_000_003)  # independent per-thread seeds

        tiss = LiquidTissue(L * L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu_val, r0, dr, rmax, dm_val, N_CHR)
        perturb_liquid!(tiss, n_seed, pert_chrs)

        res = simulation_liquid(tiss, N_CHR, n_steps, n_it_store, false, limit, 0.0, misseg_mech)

        save_results_npz_liquid(joinpath(ensemble_dir, "sim_$(i).npz"), res, dm_val)

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
if abspath(PROGRAM_FILE) == @__FILE__
    suffix = ""
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
    idx_dm = findfirst(x -> x == "--dm", ARGS)
    if !isnothing(idx_dm)
        if idx_dm < length(ARGS)
            dm_val = parse(Float64, ARGS[idx_dm+1])
            deleteat!(ARGS, [idx_dm, idx_dm+1])
        else
            deleteat!(ARGS, idx_dm)
        end
    end

    dmu_val = dmu
    idx_dmu = findfirst(x -> x == "--dmu", ARGS)
    if !isnothing(idx_dmu)
        if idx_dmu < length(ARGS)
            dmu_val = parse(Float64, ARGS[idx_dmu+1])
            deleteat!(ARGS, [idx_dmu, idx_dmu+1])
        else
            deleteat!(ARGS, idx_dmu)
        end
    end

    if length(ARGS) >= 2
        n = parse(Int, ARGS[1])
        m_mech = ARGS[2]
        run_ensemble_liquid(n, m_mech, suffix, dm_val, dmu_val)
    elseif length(ARGS) == 1
        n = parse(Int, ARGS[1])
        run_ensemble_liquid(n, misseg_type, suffix, dm_val, dmu_val)
    else
        run_ensemble_liquid(20, misseg_type, suffix, dm_val, dmu_val)
    end
end
