# ==============================================================================
# run_phase_transition_liquid.jl
# ==============================================================================
# Runs multi-threaded simulations to calculate extinction times for varying
# parameters (dmu and dr) in the LIQUID tumor model. 
# Saves results to data/phase_transition_liquid/
# ==============================================================================

include("../../src/utils_liquid.jl")
include("../../src/simulation_liquid.jl")
include("parameters_liquid.jl")

using Base.Threads
using Random, NPZ

function run_sweep_dmu(n_steps=10000, limit=0.6, L_sweep=80, N_s=10)
    fixed_dr = 0.008
    dmu_vals = collect(range(0.0175, 0.03, length=70))
    n_seed_sweep = round(Int, 0.05 * L_sweep^2) # 5% tissue mass

    output_dir = joinpath(dirname(dirname(@__DIR__)), "data", "phase_transition_liquid_5pct", "dmu")
    if !isdir(output_dir); mkpath(output_dir); end

    # Flatten tasks
    tasks = []
    for dmu in dmu_vals
        for rep in 1:N_s
            push!(tasks, (dmu=dmu, dr=fixed_dr, rep=rep))
        end
    end

    num_sims = length(tasks)
    println("--- Running dmu sweep (Liquid, 5%) ---")
    println("Total simulations: $num_sims on $(Threads.nthreads()) threads")

    progress_lock = ReentrantLock()
    completed = 0

    @threads for i in 1:num_sims
        t = tasks[i]
        Random.seed!(time_ns() + i)
        
        tiss = LiquidTissue(L_sweep * L_sweep, N_I, N_O, N_S, N_M, N_HK, mu0, t.dmu, r0, t.dr, rmax, dm, N_CHR)
        perturb_liquid!(tiss, n_seed_sweep, pert_chrs)
        
        res = simulation_liquid(tiss, N_CHR, n_steps, 1, false, limit, 0.0, misseg_type)
        
        filename = joinpath(output_dir, "dmu_$(t.dmu)_rep_$(t.rep).npz")
        
        npzwrite(filename, Dict(
            "tumor_density" => res.tumor_density,
            "dmu" => [t.dmu],
            "dr" => [t.dr],
            "outcome_code" => [res.state == "Health" ? 0 : (res.state == "Tumor_Max" ? 1 : (res.state == "Tumor_Min" ? 3 : 2))],
            "time" => [length(res.tumor_density)]
        ))
        
        lock(progress_lock) do
            completed += 1
            if completed % 10 == 0 || completed == num_sims
                println("dmu sweep progress: $completed / $num_sims")
            end
        end
    end
end

function run_sweep_dr(n_steps=10000, limit=0.6, L_sweep=80, N_s=10)
    fixed_dmu = 0.012
    dr_vals = collect(range(0.0, 0.006, length=50))
    n_seed_sweep = round(Int, 0.05 * L_sweep^2)

    output_dir = joinpath(dirname(dirname(@__DIR__)), "data", "phase_transition_liquid_5pct", "dr")
    if !isdir(output_dir); mkpath(output_dir); end

    # Flatten tasks
    tasks = []
    for dr in dr_vals
        for rep in 1:N_s
            push!(tasks, (dmu=fixed_dmu, dr=dr, rep=rep))
        end
    end

    num_sims = length(tasks)
    println("\n--- Running dr sweep (Liquid, 5%) ---")
    println("Total simulations: $num_sims on $(Threads.nthreads()) threads")

    progress_lock = ReentrantLock()
    completed = 0

    @threads for i in 1:num_sims
        t = tasks[i]
        Random.seed!(time_ns() + i)
        
        tiss = LiquidTissue(L_sweep * L_sweep, N_I, N_O, N_S, N_M, N_HK, mu0, t.dmu, r0, t.dr, rmax, dm, N_CHR)
        perturb_liquid!(tiss, n_seed_sweep, pert_chrs)
        
        res = simulation_liquid(tiss, N_CHR, n_steps, 1, false, limit, 0.0, misseg_type)
        
        filename = joinpath(output_dir, "dr_$(t.dr)_rep_$(t.rep).npz")
        npzwrite(filename, Dict(
            "tumor_density" => res.tumor_density,
            "dmu" => [t.dmu],
            "dr" => [t.dr],
            "outcome_code" => [res.state == "Health" ? 0 : (res.state == "Tumor_Max" ? 1 : (res.state == "Tumor_Min" ? 3 : 2))],
            "time" => [length(res.tumor_density)]
        ))
        
        lock(progress_lock) do
            completed += 1
            if completed % 10 == 0 || completed == num_sims
                println("dr sweep progress: $completed / $num_sims")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Compiling and running liquid phase transition sweeps...")
    # L=80 (tissue size) to match solid setup
    run_sweep_dmu(10000, 0.6, 80, 10)
    run_sweep_dr(10000, 0.6, 80, 10)
    println("All liquid sweeps finished successfully!")
end
