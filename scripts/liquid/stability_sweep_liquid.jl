# ==============================================================================
# stability_sweep_liquid.jl
# ==============================================================================
# Description:
#   Runs a 2D parameter grid sweep over dmu (mutation rate step size) and
#   rmax_norm (normalized division rate rmax/r0) for the liquid tumor model.
#   Uses terminal class initialization and N_CHR_STAB = 1 to map the transition
#   boundary (tumor collapse vs expansion) and saves results in CSV format.
#
# Usage:
#   julia --project=. scripts/liquid/stability_sweep_liquid.jl [--n-grid value] [--reps value]
#
# Arguments:
#   --n-grid value - Size of the 2D grid parameter space (default: 20)
#   --reps value   - Number of replicate simulations per grid point (default: 25)
# ==============================================================================


include("../../src/utils_liquid.jl")
include("../../src/simulation_liquid.jl")
include("parameters_liquid.jl")

using Base.Threads
using Random, CSV, DataFrames

# Struct to store individual simulation outcomes
struct SimOutcome
    dmu::Float64
    rmax_norm::Float64
    rep::Int
    state::String
    steps::Int
end

# Ensure we use N_CHR_STAB = 1 and terminal class initialization
const N_CHR_STAB = 1
function get_initial_stab_mask()
    m = UInt64(0)
    for i in 1:N_I;             m |= (UInt64(1) << (i - 1));              end
    for i in 1:N_O;             m |= (UInt64(1) << (N_I + i - 1));        end
    for i in 1:N_S;             m |= (UInt64(1) << (N_I + N_O + i - 1)); end
    return m
end
const PERT_CHR_STAB = [get_initial_stab_mask()]

function simulation_liquid_sweep(tiss::LiquidTissue, n_chr_init::Int, n_steps::Int, limit::Float64)
    state = "Done"
    step_count = n_steps
    N = tiss.N
    for k in 1:n_steps
        substitute_liquid!(tiss, n_chr_init)
        
        n_canc = count(==(1), tiss.state)
        
        if n_canc == 0
            state = "Health"
            step_count = k
            break
        end
        
        density = n_canc / N
        if density > limit
            state = "Tumor_Max"
            step_count = k
            break
        end
    end
    
    # Resolve 'Done' cases using final density check
    if state == "Done"
        n_canc = count(==(1), tiss.state)
        density = n_canc / N
        # If the tumor density at step 15000 is still substantial, it survived.
        if density > 0.05
            state = "Tumor_Max"
        else
            state = "Health"
        end
    end
    
    return (state=state, steps=step_count)
end

function run_parameter_sweep(n_grid=20, reps=25)
    output_dir = joinpath(dirname(dirname(@__DIR__)), "data")
    if !isdir(output_dir); mkpath(output_dir); end
    output_file = joinpath(output_dir, "stability_phase_diagram_results_liquid.csv")

    dmu_range = collect(range(0.0001, 0.007, length=n_grid))
    rmax_norm_range = collect(range(1.1, 7.0, length=n_grid))

    L_sweep = 100
    n_steps_sweep = 1500
    n_seed_sweep = 100
    limit = 0.5
        
    tasks = []
    for dmu_val in dmu_range
        for r_norm in rmax_norm_range
            for r in 1:reps
                push!(tasks, (dmu=dmu_val, rmax_norm=r_norm, rep=r))
            end
        end
    end

    num_sims = length(tasks)
    results = Vector{SimOutcome}(undef, num_sims)
    
    println("--- Liquid Stability Phase Diagram Sweep ---")
    println("Grid size: $n_grid x $n_grid = $(n_grid * n_grid) points")
    println("Replicates per point: $reps")
    println("Total simulations to run: $num_sims")
    println("Using $(Threads.nthreads()) threads.")
    println("Sweep Parameters: L=$L_sweep, steps=$n_steps_sweep, limit=$limit, n_seed=$n_seed_sweep")

    progress_lock = ReentrantLock()
    completed = 0
    start_time = time()

    @threads for i in 1:num_sims
        t = tasks[i]
        
        Random.seed!(time_ns() + i)
        
        rmax_val = t.rmax_norm * r0
        dr_val = rmax_val / 10
        
        tiss = LiquidTissue(
            L_sweep * L_sweep, N_I, N_O, N_S, N_M, N_HK, 
            mu0, t.dmu, r0, dr_val, rmax_val, 0.0, N_CHR_STAB
        )
        perturb_liquid!(tiss, n_seed_sweep, PERT_CHR_STAB)
        
        res = simulation_liquid_sweep(tiss, N_CHR_STAB, n_steps_sweep, limit)
        
        results[i] = SimOutcome(t.dmu, t.rmax_norm, t.rep, res.state, res.steps)
        
        lock(progress_lock) do
            completed += 1
            if completed % 100 == 0 || completed == num_sims
                elapsed = time() - start_time
                est_total = (elapsed / completed) * num_sims
                remaining = est_total - elapsed
                
                rem_str = if remaining < 60
                    "$(round(remaining, digits=1))s"
                else
                    "$(round(remaining/60, digits=1))m"
                end
                
                println("Progress: $completed / $num_sims simulations finished. (Est. remaining: $rem_str)")
            end
        end
    end

    println("\nAggregating results...")
    
    df_raw = DataFrame(
        dmu = [r.dmu for r in results],
        rmax_norm = [r.rmax_norm for r in results],
        state = [r.state for r in results]
    )

    gd = groupby(df_raw, [:dmu, :rmax_norm])
    
    df_agg = combine(gd) do sdf
        n_total = nrow(sdf)
        n_health = count(x -> x == "Health", sdf.state)
        n_tumor_max = count(x -> x == "Tumor_Max", sdf.state)
        
        fraction = n_tumor_max / n_total
        
        (
            n_reps = n_total,
            health_count = n_health,
            tumor_max_count = n_tumor_max,
            fraction = fraction
        )
    end

    CSV.write(output_file, df_agg)
    println("Sweep complete! Aggregated results saved to: $output_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    n_grid_val = 20
    reps_val = 25
    for (i, arg) in enumerate(ARGS)
        if arg == "--n-grid" && i < length(ARGS)
            global n_grid_val = parse(Int, ARGS[i+1])
        elseif arg == "--reps" && i < length(ARGS)
            global reps_val = parse(Int, ARGS[i+1])
        end
    end
    run_parameter_sweep(n_grid_val, reps_val)
end
