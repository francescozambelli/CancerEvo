# scripts/stability_sweep.jl

include("../src/utils.jl")
include("../src/simulation.jl")
include("parameters.jl")

using Random, Statistics, Base.Threads, NPZ, ProgressMeter

# Experimental Setup
const TARGET_DENSITY = 0.2
const STABILITY_TOLERANCE = 0.2
const LOWER_LIMIT = TARGET_DENSITY * (1 - STABILITY_TOLERANCE)
const UPPER_LIMIT = TARGET_DENSITY * (1 + STABILITY_TOLERANCE)
const R_PERT_STABILITY = sqrt(TARGET_DENSITY / pi)
const MAX_STEPS_STABILITY = 500

# 1 Chromosome, All O,I,S mutated
const N_CHR_STAB = 1
function get_initial_stab_mask()
    m = UInt64(0)
    # I genes (1:10)
    for i in 1:N_I; m |= (UInt64(1) << (i-1)); end
    # O genes (11:20)
    for i in 1:N_O; m |= (UInt64(1) << (N_I + i - 1)); end
    # S genes (21:30)
    for i in 1:N_S; m |= (UInt64(1) << (N_I + N_O + i - 1)); end
    return m
end
const PERT_CHR_STAB = [get_initial_stab_mask()]

function run_stability_sweep()
    rmax_vals = range(2*r0, 4*r0, length=10)
    dmu_vals = range(0.003, 0.004, length=20) # Sweep range for dmu
    
    stable_results = Dict{Float64, Vector{Float64}}()
    
    println("Starting Stability Sweep...")
    println("Target Density: $TARGET_DENSITY (Range: $LOWER_LIMIT - $UPPER_LIMIT)")
    println("Initial Radius: $R_PERT_STABILITY")
    
    for r_max in rmax_vals
        println("Testing rmax = $(round(r_max, digits=4))...")
        stable_dmus = Float64[]
        
        # Parallelize the dmu sweep for each rmax
        dmu_outcomes = Vector{Float64}(undef, length(dmu_vals))
        dmu_states = Vector{String}(undef, length(dmu_vals))
        
        for i in 1:length(dmu_vals)
            d_mu = dmu_vals[i]
            
            # Setup tissue with current rmax and dmu
            dr = r_max/10
            tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, d_mu, r0, dr, r_max, 0.0, N_CHR_STAB)
            perturb_optimized!(tiss, R_PERT_STABILITY, PERT_CHR_STAB)
            
            # Run simulation with tight limits
            res = simulation_optimized(tiss, N_CHR_STAB, MAX_STEPS_STABILITY, 100, false, UPPER_LIMIT, LOWER_LIMIT)
            
            final_size = res.tumor_density[end]
            final_state = res.state
            dmu_states[i] = final_state
            
            # Stable if it reached the end of the simulation within the limits
            if final_state == "Done"
                dmu_outcomes[i] = final_size
            else
                dmu_outcomes[i] = -1.0 # Indicator for unstable
            end
        end
        
        # Filter for stable dmu values
        for i in 1:length(dmu_vals)
            if dmu_outcomes[i] != -1.0
                push!(stable_dmus, dmu_vals[i])
            end
        end
        
        # If no stable dmu values found, look for the transition between Over (Max) and Under (Min)
        if isempty(stable_dmus)
            println("  No stable dmu found. Looking for transition...")
            for i in 1:(length(dmu_vals)-1)
                s1 = dmu_states[i]
                s2 = dmu_states[i+1]
                # Transition from Over to Under (increasing dmu leads to more death)
                if (s1 == "Tumor_Max" && s2 == "Tumor_Min") || (s1 == "Tumor_Min" && s2 == "Tumor_Max")
                    mid_dmu = (dmu_vals[i] + dmu_vals[i+1]) / 2
                    push!(stable_dmus, mid_dmu)
                    println("  Found transition between $s1 and $s2 at dmu ≈ $(round(mid_dmu, digits=6))")
                    break # Just take the first transition found
                end
            end
        end
        
        stable_results[r_max] = stable_dmus
        println("  Found $(length(stable_dmus)) result values.")
    end
    
    # Save results to CSV
    output_path = joinpath(dirname(@__DIR__), "data", "stability_results.csv")
    open(output_path, "w") do io
        println(io, "rmax,stable_dmu")
        for r_max in sort(collect(keys(stable_results)))
            for d_mu in stable_results[r_max]
                println(io, "$r_max,$d_mu")
            end
        end
    end
    
    println("\nSweep complete. Results saved to $output_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_stability_sweep()
end
