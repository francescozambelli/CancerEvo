# scripts/parameter_phase_diagram.jl

include("../../src/utils.jl")
include("../../src/simulation.jl")
include("parameters.jl")

using Base.Threads
using Random, CSV, DataFrames

# Struct to store individual simulation outcomes
struct SimOutcome
    dmu::Float64
    dr::Float64
    rep::Int
    state::String
    steps::Int
end

function run_parameter_sweep(n_grid=20, reps=50)
    # Output file
    output_dir = joinpath(dirname(dirname(@__DIR__)), "data")
    if !isdir(output_dir); mkpath(output_dir); end
    output_file = joinpath(output_dir, "parameter_phase_diagram_results.csv")

    # Define the parameter sweep ranges
    dmu_range = collect(range(0.1e-3, 3.0e-2, length=n_grid))
    dr_range = collect(range(0.0, 2.4e-2, length=n_grid))

    # Setup optimized local parameters to override the default ones
    L_sweep = 80
    n_steps_sweep = 1000
    r_pert_sweep = 0.03 # Yields 17 cells to prevent stochastic extinction
    
    # Flatten the grid tasks
    tasks = []
    for dmu_val in dmu_range
        for dr_val in dr_range
            for r in 1:reps
                push!(tasks, (dmu=dmu_val, dr=dr_val, rep=r))
            end
        end
    end

    num_sims = length(tasks)
    results = Vector{SimOutcome}(undef, num_sims)
    
    println("--- Parameter Space Phase Diagram Sweep ---")
    println("Grid size: $n_grid x $n_grid = $(n_grid * n_grid) points")
    println("Replicates per point: $reps")
    println("Total simulations to run: $num_sims")
    println("Using $(Threads.nthreads()) threads.")
    println("Sweep Parameters: L=$L_sweep, steps=$n_steps_sweep, limit=$limit, r_pert=$r_pert_sweep")

    # Progress tracking
    progress_lock = ReentrantLock()
    completed = 0
    start_time = time()

    @threads for i in 1:num_sims
        t = tasks[i]
        
        # Unique seed per thread/simulation
        Random.seed!(time_ns() + i)
        
        # Initialize fresh tissue
        tiss = OptimizedTissue(
            L_sweep, N_I, N_O, N_S, N_M, N_HK, 
            mu0, t.dmu, r0, t.dr, rmax, dm, N_CHR
        )
        perturb_optimized!(tiss, r_pert_sweep, pert_chrs)
        
        # Run simulation with bar=false to prevent log cluttering
        res = simulation_optimized(tiss, N_CHR, n_steps_sweep, n_it_store, false, limit)
        
        results[i] = SimOutcome(t.dmu, t.dr, t.rep, res.state, length(res.tumor_density))
        
        lock(progress_lock) do
            completed += 1
            if completed % 100 == 0 || completed == num_sims
                elapsed = time() - start_time
                est_total = (elapsed / completed) * num_sims
                remaining = est_total - elapsed
                
                # Format remaining time
                rem_str = if remaining < 60
                    "$(round(remaining, digits=1))s"
                else
                    "$(round(remaining/60, digits=1))m"
                end
                
                println("Progress: $completed / $num_sims simulations finished. (Est. remaining: $rem_str)")
            end
        end
    end

    # Aggregate results by (dmu, dr)
    println("\nAggregating results...")
    
    # We construct a dataframe from individual outcomes
    df_raw = DataFrame(
        dmu = [r.dmu for r in results],
        dr = [r.dr for r in results],
        state = [r.state for r in results]
    )

    # Group by dmu and dr
    gd = groupby(df_raw, [:dmu, :dr])
    
    df_agg = combine(gd) do sdf
        n_total = nrow(sdf)
        n_health = count(x -> x == "Health", sdf.state)
        n_tumor_max = count(x -> x == "Tumor_Max", sdf.state)
        n_done = count(x -> x == "Done", sdf.state)
        n_tumor_min = count(x -> x == "Tumor_Min", sdf.state)
        
        # Fraction of interest: tumor mass exceeds limit (Tumor_Max) or finishes with tumor present (Done)
        fraction = (n_tumor_max + n_done) / n_total
        
        (
            n_reps = n_total,
            health_count = n_health,
            tumor_max_count = n_tumor_max,
            done_count = n_done,
            tumor_min_count = n_tumor_min,
            fraction = fraction
        )
    end

    # Save to CSV
    CSV.write(output_file, df_agg)
    println("Sweep complete! Aggregated results saved to: $output_file")
end

# Check for command line arguments
if abspath(PROGRAM_FILE) == @__FILE__
    n_grid_val = 20
    reps_val = 50
    for (i, arg) in enumerate(ARGS)
        if arg == "--n-grid" && i < length(ARGS)
            global n_grid_val = parse(Int, ARGS[i+1])
        elseif arg == "--reps" && i < length(ARGS)
            global reps_val = parse(Int, ARGS[i+1])
        end
    end
    run_parameter_sweep(n_grid_val, reps_val)
end
