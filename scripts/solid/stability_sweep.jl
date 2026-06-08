# scripts/stability_sweep.jl
#
# Adaptive phase-boundary sweep.
#
# Strategy
# --------
# 1. Load all prior stability CSV files from data/ and merge them into a
#    single empirical boundary dataset: (rmax, dmu*).
# 2. Fit a log-linear predictor dmu*(rmax) on the merged data.
# 3. For each rmax on a new (configurable) grid:
#      a. Skip if already well-sampled (≥ MIN_PRIOR_HITS points within
#         SKIP_TOLERANCE of each other).
#      b. Compute a narrow adaptive window around the prediction.
#      c. Run a coarse scan (N_COARSE points) across that window.
#      d. Refine with bisection (N_BISECT steps) around the
#         Tumor_Max ↔ stable or stable ↔ Tumor_Min transition.
#      e. Append findings to the output CSV after every rmax (crash-safe).
# 4. The output file is data/stability_results_adaptive.csv and can be
#    overlaid with prior data for a robust phase diagram.
#
# Usage
# -----
#   julia scripts/stability_sweep.jl              # full run
#   julia scripts/stability_sweep.jl --dry-run    # print plan, no sims
#   julia scripts/stability_sweep.jl --n-rmax 5   # quick smoke test

include("../../src/utils.jl")
include("../../src/simulation.jl")
include("parameters.jl")

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
const R_PERT_STABILITY    = sqrt(TARGET_DENSITY / pi)
const MAX_STEPS_STABILITY = 500
const OUTPUT_FILE         = "stability_results_adaptive.csv"

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

"""
    make_tissue(rmax, dmu) → OptimizedTissue

Build and perturb a fresh tissue for the stability sweep.
"""
function make_tissue(rmax::Float64, dmu::Float64)::OptimizedTissue
    dr   = rmax / 10
    tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK,
                            mu0, dmu, r0, dr, rmax, 0.0, N_CHR_STAB)
    perturb_optimized!(tiss, R_PERT_STABILITY, PERT_CHR_STAB)
    return tiss
end


"""
    simulation_solid_stability(tiss, n_chr_init, n_steps, limit, lower_limit)

Lightweight simulation loop for stability sweeps that terminates based directly
on living tumor_density (ignoring dead cells to prevent false triggers).
"""
function simulation_solid_stability(tiss::OptimizedTissue, n_chr_init::Int, n_steps::Int,
                                     limit::Float64, lower_limit::Float64)
    state = "Done"
    final_density = 0.0
    N = tiss.L * tiss.L
    for k in 1:n_steps
        substitute_optimized!(tiss, n_chr_init)
        n_canc = count(tiss.state .== 1)
        if n_canc == 0
            state = "Health"
            final_density = 0.0
            break
        end
        n_wt = count(tiss.state .== 0)
        wt_density = n_wt / N
        final_density = n_canc / N
        if wt_density < (1.0 - limit)
            state = "Tumor_Max"
            break
        end
        if wt_density > (1.0 - lower_limit) && k > 1
            state = "Tumor_Min"
            break
        end
    end
    return state, final_density
end


"""
    probe(rmax, dmu) → (state::String, final_density::Float64)

Run one stability simulation and return the terminal state + final density.
"""
function probe(rmax::Float64, dmu::Float64)
    tiss = make_tissue(rmax, dmu)
    state, final_density = simulation_solid_stability(tiss, N_CHR_STAB, MAX_STEPS_STABILITY,
                                                       UPPER_LIMIT, LOWER_LIMIT)
    return state, final_density
end


"""
    coarse_scan(rmax, lo, hi, n) → (dmu_vals, states, densities)

Scan n evenly-spaced dmu values in [lo, hi] and return the outcomes.
"""
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


"""
    bisect_boundary(rmax, lo, hi, n_iters) → dmu_boundary

Refine the transition between a Tumor_Max region (low dmu) and a stable or
Tumor_Min region (high dmu) using binary search.

Returns the midpoint of the final bracket as the estimated boundary.
"""
function bisect_boundary(rmax::Float64, lo::Float64, hi::Float64,
                          n_iters::Int)::Float64
    # Convention: lo → Tumor_Max (over), hi → stable or Tumor_Min (under/done)
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

"""
    init_output(path) → nothing

Create the output CSV with a header if it does not already exist.
"""
function init_output(path::String)
    if !isfile(path)
        open(path, "w") do io
            println(io, "rmax,stable_dmu")
        end
    end
end


"""
    append_results(path, rmax, dmus) → nothing

Append rmax + each discovered stable dmu to the output CSV.
"""
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
    println("  Direct Bisection Stability Sweep (Solid)")
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

    # ── 7. Summary ────────────────────────────────────────────────────────────
    println("\n═══════════════════════════════════════════════════")
    println("  Sweep complete!")
    println("  Output: $output_path")
    if isfile(output_path)
        result = CSV.read(output_path, DataFrame)
        println("  Total rows: $(nrow(result))")
        if nrow(result) > 0
            println("  rmax  range: $(round(minimum(result.rmax), sigdigits=4)) – $(round(maximum(result.rmax), sigdigits=4))")
            println("  dmu*  range: $(round(minimum(result.stable_dmu), sigdigits=4)) – $(round(maximum(result.stable_dmu), sigdigits=4))")
        end
    end
    println("═══════════════════════════════════════════════════\n")
end

# ═══════════════════════════════════════════════════════════════════════════════
# ── Entry point ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

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
