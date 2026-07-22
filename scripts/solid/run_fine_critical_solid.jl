# ==============================================================================
# run_fine_critical_solid.jl
# ==============================================================================
# Runs high-resolution simulations around the critical points (dmu_c and dr_c)
# to obtain dense data for log-log critical slowing down scaling analysis.
# ==============================================================================

include("../../src/utils_solid.jl")
include("../../src/simulation_solid.jl")
include("parameters_solid.jl")

using Base.Threads
using Random, NPZ

function run_fine_sweep_dmu(n_steps=100000, limit=0.4, L_sweep=80, N_s=10, init_mass_pct=10.0)
    fixed_dr = 0.008
    # Fine grid around dmu_c ≈ 0.0168 (16.0 to 17.5 x10^-3)
    dmu_vals = collect(range(0.0160, 0.0175, length=31))
    init_mass_frac = init_mass_pct / 100.0
    r_pert_sweep = sqrt(init_mass_frac / pi)

    limit_pct = round(Int, limit * 100)
    steps_str = n_steps == 10000 ? "" : "_steps$(n_steps)"
    output_dir = joinpath(dirname(dirname(@__DIR__)), "data", "phase_transition$(steps_str)_init$(round(Int, init_mass_pct))_limit$(limit_pct)", "dmu")
    if !isdir(output_dir); mkpath(output_dir); end

    tasks = []
    for dmu in dmu_vals
        for rep in 1:N_s
            push!(tasks, (dmu=dmu, dr=fixed_dr, rep=rep))
        end
    end

    num_sims = length(tasks)
    println("--- Running fine dmu critical sweep (init: $(init_mass_pct)%, limit: $(limit_pct)%) ---")
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
        
        filename = joinpath(output_dir, "dmu_fine_$(t.dmu)_rep_$(t.rep).npz")
        npzwrite(filename, Dict(
            "tumor_density" => res.tumor_density,
            "dmu" => [t.dmu],
            "dr" => [t.dr],
            "outcome_code" => [res.state == "Health" ? 0 : (res.state == "Tumor_Max" ? 1 : 2)],
            "time" => [length(res.tumor_density)]
        ))
        
        lock(progress_lock) do
            completed += 1
            if completed % 20 == 0 || completed == num_sims
                println("fine dmu sweep progress: $completed / $num_sims")
            end
        end
    end
end

function run_fine_sweep_dr(n_steps=100000, limit=0.4, L_sweep=80, N_s=10, init_mass_pct=10.0)
    fixed_dmu = 0.012
    # Fine grid around dr_c ≈ 0.0039 (3.5 to 4.5 x10^-3)
    dr_vals = collect(range(0.0035, 0.0045, length=31))
    init_mass_frac = init_mass_pct / 100.0
    r_pert_sweep = sqrt(init_mass_frac / pi)

    limit_pct = round(Int, limit * 100)
    steps_str = n_steps == 10000 ? "" : "_steps$(n_steps)"
    output_dir = joinpath(dirname(dirname(@__DIR__)), "data", "phase_transition$(steps_str)_init$(round(Int, init_mass_pct))_limit$(limit_pct)", "dr")
    if !isdir(output_dir); mkpath(output_dir); end

    tasks = []
    for dr in dr_vals
        for rep in 1:N_s
            push!(tasks, (dmu=fixed_dmu, dr=dr, rep=rep))
        end
    end

    num_sims = length(tasks)
    println("\n--- Running fine dr critical sweep (init: $(init_mass_pct)%, limit: $(limit_pct)%) ---")
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
        
        filename = joinpath(output_dir, "dr_fine_$(t.dr)_rep_$(t.rep).npz")
        npzwrite(filename, Dict(
            "tumor_density" => res.tumor_density,
            "dmu" => [t.dmu],
            "dr" => [t.dr],
            "outcome_code" => [res.state == "Health" ? 0 : (res.state == "Tumor_Max" ? 1 : 2)],
            "time" => [length(res.tumor_density)]
        ))
        
        lock(progress_lock) do
            completed += 1
            if completed % 20 == 0 || completed == num_sims
                println("fine dr sweep progress: $completed / $num_sims")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    init_mass_pct = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 10.0
    limit_pct = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 40.0
    n_steps = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 100000
    limit = limit_pct / 100.0
    
    println("Starting fine critical parameter sweeps for solid model...")
    run_fine_sweep_dmu(n_steps, limit, 80, 10, init_mass_pct)
    run_fine_sweep_dr(n_steps, limit, 80, 10, init_mass_pct)
    println("Fine critical sweeps complete!")
end
