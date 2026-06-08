# scripts/stability_sweep_liquid.jl
#
# Adaptive phase-boundary sweep for the liquid-tumor model.
#
# Strategy
# --------
# 1. Load prior stability CSV files for the liquid model (if any).
# 2. Fit a log-linear predictor dmu*(rmax) on the merged data.
# 3. For each rmax on the sweep grid:
#      a. Skip if already well-sampled.
#      b. Compute a narrow adaptive window around the prediction.
#      c. Run a coarse scan across that window.
#      d. Refine with bisection around the transition.
#      e. Append findings to the output CSV after every rmax.
# 4. Save results to data/stability_results_liquid_adaptive.csv.

include("../../src/utils_liquid.jl")
include("../../src/simulation_liquid.jl")
include("parameters_liquid.jl")

using Random, Statistics, Base.Threads, NPZ, ProgressMeter, CSV, DataFrames

# ═══════════════════════════════════════════════════════════════════════════════
# ── Configuration ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Absolute search limits for dmu
const DMU_MIN = 1e-6
const DMU_MAX = 0.02
const N_BISECT = 14  # 14 iterations gives accuracy of ~1.2e-6

const N_RMAX_DEFAULT      = 50
const TARGET_DENSITY      = 0.2
const STABILITY_TOLERANCE = 0.2
const LOWER_LIMIT         = TARGET_DENSITY * (1 - STABILITY_TOLERANCE)
const UPPER_LIMIT         = TARGET_DENSITY * (1 + STABILITY_TOLERANCE)
const MAX_STEPS_STABILITY = 500
const OUTPUT_FILE         = "stability_results_liquid_adaptive.csv"

const N_CHR_STAB = 1
function get_initial_stab_mask()
    m = UInt64(0)
    for i in 1:N_I;             m |= (UInt64(1) << (i - 1));              end
    for i in 1:N_O;             m |= (UInt64(1) << (N_I + i - 1));        end
    for i in 1:N_S;             m |= (UInt64(1) << (N_I + N_O + i - 1)); end
    return m
end
const PERT_CHR_STAB = [get_initial_stab_mask()]


# ═══════════════════════════════════════════════════════════════════════════════
# ── Simulation helpers ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

function make_tissue(rmax::Float64, dmu::Float64)::LiquidTissue
    dr   = rmax / 10
    tiss = LiquidTissue(L * L, N_I, N_O, N_S, N_M, N_HK,
                            mu0, dmu, r0, dr, rmax, 0.0, N_CHR_STAB)
    N = tiss.N
    perturb_liquid!(tiss, round(Int, TARGET_DENSITY * N), PERT_CHR_STAB)
    return tiss
end

"""
    simulation_liquid_stability(tiss, n_chr_init, n_steps, limit, lower_limit)

Lightweight simulation loop for stability sweeps that terminates based directly
on living tumor_density (ignoring dead cells to prevent false triggers).
"""
function simulation_liquid_stability(tiss::LiquidTissue, n_chr_init::Int, n_steps::Int,
                                     limit::Float64, lower_limit::Float64)
    state = "Done"
    final_density = 0.0
    N = tiss.N
    for k in 1:n_steps
        substitute_liquid!(tiss, n_chr_init)
        n_canc = count(tiss.state .== 1)
        if n_canc == 0
            state = "Health"
            final_density = 0.0
            break
        end
        density = n_canc / N
        final_density = density
        if density > limit
            state = "Tumor_Max"
            break
        end
        if density < lower_limit && k > 1
            state = "Tumor_Min"
            break
        end
    end
    return state, final_density
end

function probe(rmax::Float64, dmu::Float64)
    tiss = make_tissue(rmax, dmu)
    state, final_density = simulation_liquid_stability(tiss, N_CHR_STAB, MAX_STEPS_STABILITY,
                                                        UPPER_LIMIT, LOWER_LIMIT)
    return state, final_density
end

function coarse_scan(rmax::Float64, lo::Float64, hi::Float64, n::Int)
    dmu_vals  = collect(range(lo, hi, length=n))
    states    = Vector{String}(undef, n)
    densities = Vector{Float64}(undef, n)

    # Parallelize the simulation sweeps across available threads
    @threads for i in 1:n
        states[i], densities[i] = probe(rmax, dmu_vals[i])
    end
    return dmu_vals, states, densities
end

function bisect_boundary(rmax::Float64, lo::Float64, hi::Float64,
                          n_iters::Int)::Float64
    for _ in 1:n_iters
        mid = (lo + hi) / 2
        state, _ = probe(rmax, mid)
        if state == "Tumor_Max"
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

# ═══════════════════════════════════════════════════════════════════════════════
# ── CSV helpers ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

function init_output(path::String)
    if !isfile(path)
        open(path, "w") do io
            println(io, "rmax,stable_dmu")
        end
    end
end

function append_results(path::String, rmax::Float64, dmus::Vector{Float64})
    isempty(dmus) && return
    open(path, "a") do io
        for d in dmus
            println(io, "$rmax,$d")
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# ── Main sweep ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

function load_completed_rmaxs(path::String)::Set{Float64}
    rmaxs = Set{Float64}()
    if isfile(path)
        try
            df = CSV.read(path, DataFrame)
            if !isempty(df) && :rmax in propertynames(df)
                for r in df.rmax
                    push!(rmaxs, r)
                end
            end
        catch e
        end
    end
    return rmaxs
end

function run_stability_sweep(; n_rmax::Int = 50, dry_run::Bool = false)
    data_dir    = joinpath(dirname(dirname(@__DIR__)), "data")
    output_path = joinpath(data_dir, OUTPUT_FILE)

    println("\n═══════════════════════════════════════════════════")
    println("  Direct Bisection Stability Sweep (Liquid)")
    println("═══════════════════════════════════════════════════")

    # ── 1. Build rmax grid ────────────────────────────────────────────────────
    rmax_lo   = 1.0 * r0
    rmax_hi   = 7.0 * r0
    rmax_grid = collect(range(rmax_lo, rmax_hi, length=n_rmax))

    # Load completed rmaxs for resumption
    completed_rmaxs = load_completed_rmaxs(output_path)

    # ── 2. Dry-run: print plan ────────────────────────────────────────────────
    println("\nSweep plan  ($(length(rmax_grid)) rmax values, $(dry_run ? "DRY RUN" : "LIVE")):")
    println("  rmax range : $(round(rmax_lo, sigdigits=3)) – $(round(rmax_hi, sigdigits=3))")
    println("  dmu range  : $DMU_MIN – $DMU_MAX")
    for r in rmax_grid
        skip = r in completed_rmaxs
        tag = skip ? " [SKIP – already completed]" : ""
        println("    rmax=$(round(r,digits=4))$tag")
    end
    dry_run && return

    # ── 3. Init output file ───────────────────────────────────────────────────
    init_output(output_path)
    println("\nAppending results to: $output_path")

    # ── 4. Main loop ──────────────────────────────────────────────────────────
    for (idx, r_max) in enumerate(rmax_grid)
        println("\n── rmax = $(round(r_max, digits=4))  [$(idx)/$(length(rmax_grid))] ──")

        if r_max in completed_rmaxs
            println("  Already completed. Skipping.")
            continue
        end

        # Verify bracket endpoints
        state_lo, _ = probe(r_max, DMU_MIN)
        state_hi, _ = probe(r_max, DMU_MAX)

        if state_lo == "Tumor_Max" && state_hi != "Tumor_Max"
            println("  Transition bracketed. Starting bisection over [$DMU_MIN, $DMU_MAX]...")
            dmu_star = bisect_boundary(r_max, DMU_MIN, DMU_MAX, N_BISECT)
            append_results(output_path, r_max, [dmu_star])
            println("    → boundary dmu* ≈ $(round(dmu_star, sigdigits=5))")
            println("  Saved 1 boundary estimate for rmax=$(round(r_max, digits=4))")
        else
            println("  No transition bracketed (lo state: $state_lo, hi state: $state_hi). Skipping.")
        end
    end

    println("\n═══════════════════════════════════════════════════")
    println("  Sweep complete!")
    println("  Output: $output_path")
    if isfile(output_path)
        result = CSV.read(output_path, DataFrame)
        println("  Total rows: $(nrow(result))")
    end
    println("═══════════════════════════════════════════════════\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    dry_run = "--dry-run" in ARGS
    n_rmax_val = N_RMAX_DEFAULT
    for (i, arg) in enumerate(ARGS)
        if arg == "--n-rmax" && i < length(ARGS)
            global n_rmax_val = parse(Int, ARGS[i+1])
        end
    end
    run_stability_sweep(; n_rmax=n_rmax_val, dry_run=dry_run)
end
