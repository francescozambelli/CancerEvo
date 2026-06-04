# scripts/study_spatial.jl
#
# Study spatial structures at the new healthy-cell-based boundary for rmax = 0.8480:
#   1. Below Boundary: rmax = 0.8480, dmu = 0.0045
#   2. On Boundary: rmax = 0.8480, dmu = 0.00577
#   3. Above Boundary: rmax = 0.8480, dmu = 0.0070

include("../../src/utils.jl")
include("../../src/simulation.jl")
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
    out_dir = joinpath(dirname(dirname(@__DIR__)), "data", "spatial_study")
    mkpath(out_dir)

    scenarios = [
        (name="dec_below", dmu=0.0045),
        (name="dec_on", dmu=0.00577),
        (name="dec_above", dmu=0.0070)
    ]

    for sc in scenarios
        println("\nRunning study for scenario: $(sc.name) (rmax=0.8480, dmu=$(sc.dmu))...")
        dr = 0.0848
        tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK,
                                mu0, sc.dmu, r0, dr, 0.8480, 0.0, N_CHR_STAB)
        perturb_optimized!(tiss, R_PERT_STABILITY, PERT_CHR_STAB)

        intervals = [0, 50, 100, 200, 400]
        for step in 0:400
            if step in intervals
                grid = reshape(tiss.state, (L, L))
                npzwrite(joinpath(out_dir, "$(sc.name)_step_$(step).npz"), grid)
                println("  Saved step $step")
            end
            substitute_optimized!(tiss, N_CHR_STAB)
        end
    end
    
    println("\nSpatial study simulations complete!")
end

run_spatial_study()
