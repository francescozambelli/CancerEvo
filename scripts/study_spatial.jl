# scripts/study_spatial.jl
#
# Study spatial structures at high mutation rate dmu = 0.0030:
#   1. Fast Increase (Tumor_Max): rmax = 0.3153, dmu = 0.0030
#      (Saves steps 0, 20, 50, 100, 150)
#   2. Fast Decrease (Tumor_Min): rmax = 0.8480, dmu = 0.0030
#      (Saves steps 0, 1, 2, 3, 5)

include("../src/utils.jl")
include("../src/simulation.jl")
include("parameters.jl")

using NPZ

const N_CHR_STAB = 1
const TARGET_DENSITY = 0.2
const STABILITY_TOLERANCE = 0.2
const LOWER_LIMIT = TARGET_DENSITY * (1 - STABILITY_TOLERANCE)
const UPPER_LIMIT = TARGET_DENSITY * (1 + STABILITY_TOLERANCE)
const R_PERT_STABILITY = sqrt(TARGET_DENSITY / pi)

function get_initial_stab_mask()
    m = UInt64(0)
    for i in 1:N_I;             m |= (UInt64(1) << (i - 1));              end
    for i in 1:N_O;             m |= (UInt64(1) << (N_I + i - 1));        end
    for i in 1:N_S;             m |= (UInt64(1) << (N_I + N_O + i - 1));  end
    return m
end
const PERT_CHR_STAB = [get_initial_stab_mask()]

function run_spatial_study()
    out_dir = joinpath(dirname(@__DIR__), "data", "spatial_study")
    mkpath(out_dir)

    # 1. Fast Increase Case
    println("\nRunning Fast Increase Scenario (rmax=0.3153, dmu=0.0030)...")
    tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, 0.0030, r0, 0.03153, 0.3153, 0.0, N_CHR_STAB)
    perturb_optimized!(tiss, R_PERT_STABILITY, PERT_CHR_STAB)
    
    inc_steps = [0, 20, 50, 100, 150]
    for step in 0:150
        if step in inc_steps
            npzwrite(joinpath(out_dir, "fast_inc_step_$(step).npz"), reshape(tiss.state, (L, L)))
            println("  Saved step $step")
        end
        substitute_optimized!(tiss, N_CHR_STAB)
    end

    # 2. Fast Decrease Case
    println("\nRunning Fast Decrease Scenario (rmax=0.8480, dmu=0.0030)...")
    tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, 0.0030, r0, 0.0848, 0.8480, 0.0, N_CHR_STAB)
    perturb_optimized!(tiss, R_PERT_STABILITY, PERT_CHR_STAB)
    
    dec_steps = [0, 1, 2, 3, 5]
    for step in 0:5
        if step in dec_steps
            npzwrite(joinpath(out_dir, "fast_dec_step_$(step).npz"), reshape(tiss.state, (L, L)))
            println("  Saved step $step")
        end
        substitute_optimized!(tiss, N_CHR_STAB)
    end
    
    println("\nSpatial study simulations complete!")
end

run_spatial_study()
