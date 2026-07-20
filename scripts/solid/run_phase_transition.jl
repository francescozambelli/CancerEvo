# ==============================================================================
# run_phase_transition.jl
# ==============================================================================
# Runs multi-threaded simulations to calculate extinction times for varying
# parameters (dmu and dr). Saves results to data/phase_transition/
# ==============================================================================

include("../../src/utils_solid.jl")
include("../../src/simulation_solid.jl")
include("parameters_solid.jl")

using Base.Threads
using Random, NPZ

function run_sweep_dmu(n_steps=10000, limit=0.6, L_sweep=80, N_s=10, init_mass_pct=10.0)
    fixed_dr = 0.008
    dmu_vals = collect(range(0.01, 0.025, length=70))
    init_mass_frac = init_mass_pct / 100.0
    r_pert_sweep = sqrt(init_mass_frac / pi)

    limit_pct = round(Int, limit * 100)
    steps_str = n_steps == 10000 ? "" : "_steps$(n_steps)"
    output_dir = joinpath(dirname(dirname(@__DIR__)), "data", "phase_transition$(steps_str)_init$(round(Int, init_mass_pct))_limit$(limit_pct)", "dmu")
    if !isdir(output_dir); mkpath(output_dir); end

    # Flatten tasks
    tasks = []
    for dmu in dmu_vals
        for rep in 1:N_s
            push!(tasks, (dmu=dmu, dr=fixed_dr, rep=rep))
        end
    end

    num_sims = length(tasks)
    println("--- Running dmu sweep (init: $(init_mass_pct)%, limit: $(round(Int, limit*100))%) ---")
    println("Total simulations: $num_sims on $(Threads.nthreads()) threads")

    progress_lock = ReentrantLock()
    completed = 0

    @threads for i in 1:num_sims
        t = tasks[i]
        Random.seed!(time_ns() + i)
        
        tiss = OptimizedTissue(
            L_sweep, N_I, N_O, N_S, N_M, N_HK, 
            mu0, t.dmu, r0, t.dr, rmax, dm, N_CHR
        )
        perturb_optimized!(tiss, r_pert_sweep, pert_chrs)
        
        # Store data every 1 step for precise time (n_it_store doesn't affect return arrays in optimized)
        res = simulation_optimized(tiss, N_CHR, n_steps, 1, false, limit, 0.0, misseg_type)
        
        filename = joinpath(output_dir, "dmu_$(t.dmu)_rep_$(t.rep).npz")
        
        # Just save the lengths (time) and parameter values, and final state.
        # It's lighter than full trajectory, but we can also save full trajectory as requested.
        npzwrite(filename, Dict(
            "tumor_density" => res.tumor_density,
            "dmu" => [t.dmu],
            "dr" => [t.dr],
            "outcome_code" => [res.state == "Health" ? 0 : (res.state == "Tumor_Max" ? 1 : 2)],
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

function run_sweep_dr(n_steps=10000, limit=0.6, L_sweep=80, N_s=10, init_mass_pct=10.0)
    fixed_dmu = 0.012
    dr_vals = collect(range(0.0, 0.01, length=50))
    init_mass_frac = init_mass_pct / 100.0
    r_pert_sweep = sqrt(init_mass_frac / pi)

    limit_pct = round(Int, limit * 100)
    steps_str = n_steps == 10000 ? "" : "_steps$(n_steps)"
    output_dir = joinpath(dirname(dirname(@__DIR__)), "data", "phase_transition$(steps_str)_init$(round(Int, init_mass_pct))_limit$(limit_pct)", "dr")
    if !isdir(output_dir); mkpath(output_dir); end

    # Flatten tasks
    tasks = []
    for dr in dr_vals
        for rep in 1:N_s
            push!(tasks, (dmu=fixed_dmu, dr=dr, rep=rep))
        end
    end

    num_sims = length(tasks)
    println("\n--- Running dr sweep (init: $(init_mass_pct)%, limit: $(round(Int, limit*100))%) ---")
    println("Total simulations: $num_sims on $(Threads.nthreads()) threads")

    progress_lock = ReentrantLock()
    completed = 0

    @threads for i in 1:num_sims
        t = tasks[i]
        Random.seed!(time_ns() + i)
        
        tiss = OptimizedTissue(
            L_sweep, N_I, N_O, N_S, N_M, N_HK, 
            mu0, t.dmu, r0, t.dr, rmax, dm, N_CHR
        )
        perturb_optimized!(tiss, r_pert_sweep, pert_chrs)
        
        res = simulation_optimized(tiss, N_CHR, n_steps, 1, false, limit, 0.0, misseg_type)
        
        filename = joinpath(output_dir, "dr_$(t.dr)_rep_$(t.rep).npz")
        npzwrite(filename, Dict(
            "tumor_density" => res.tumor_density,
            "dmu" => [t.dmu],
            "dr" => [t.dr],
            "outcome_code" => [res.state == "Health" ? 0 : (res.state == "Tumor_Max" ? 1 : 2)],
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

    init_mass_pct = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 10.0
    limit_pct = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 60.0
    n_steps = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 10000
    limit = limit_pct / 100.0
    
    # Warm up to compile 
    println("Compiling...")
    # L=80 (tissue size)
    run_sweep_dmu(n_steps, limit, 80, 10, init_mass_pct)
    run_sweep_dr(n_steps, limit, 80, 10, init_mass_pct)
    println("Done!")
end
