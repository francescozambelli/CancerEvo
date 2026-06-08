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

function run_stability_sweep(; n_rmax::Int = N_RMAX_DEFAULT, dry_run::Bool = false)
    data_dir    = joinpath(dirname(dirname(@__DIR__)), "data")
    output_path = joinpath(data_dir, OUTPUT_FILE)

    println("\n═══════════════════════════════════════════════════")
    println("  Liquid Stability Sweep")
    println("═══════════════════════════════════════════════════")
    println("Loading prior data...")
    prior = load_prior_data(data_dir)
    if nrow(prior) == 0
        prior = DataFrame(rmax=Float64[], stable_dmu=Float64[])
    end
    println("  Total prior points: $(nrow(prior))")

    println("\nFitting boundary model...")
    predict = fit_boundary_model(prior)

    rmax_lo   = RMAX_MIN_MULT * r0
    rmax_hi   = RMAX_MAX_MULT * r0
    rmax_grid = collect(range(rmax_lo, rmax_hi, length=n_rmax))

    println("\nSweep plan  ($(length(rmax_grid)) rmax values, $(dry_run ? "DRY RUN" : "LIVE")):")
    println("  rmax range : $(round(rmax_lo, sigdigits=3)) – $(round(rmax_hi, sigdigits=3))")
    for r in rmax_grid
        skip = is_well_sampled(r, prior)
        lo, hi = adaptive_window(r, predict, prior)
        tag = skip ? " [SKIP – well sampled]" : " window=[$(round(lo,sigdigits=3)), $(round(hi,sigdigits=3))]"
        println("    rmax=$(round(r,digits=4))$tag")
    end
    dry_run && return

    init_output(output_path)
    println("\nAppending results to: $output_path")

    window_multiplier = 1.0

    for (idx, r_max) in enumerate(rmax_grid)
        println("\n── rmax = $(round(r_max, digits=4))  [$(idx)/$(length(rmax_grid))] ──")

        if is_well_sampled(r_max, prior)
            println("  Already well-sampled. Skipping.")
            continue
        end

        lo_base, hi_base = adaptive_window(r_max, predict, prior)
        lo = max(lo_base / sqrt(window_multiplier), 1e-6)
        hi = hi_base * sqrt(window_multiplier)

        dmu_vals, states, _densities = coarse_scan(r_max, lo, hi, N_COARSE)

        stable_idxs   = findall(s -> s == "Done",      states)
        max_idxs      = findall(s -> s == "Tumor_Max",  states)
        min_idxs      = findall(s -> s == "Tumor_Min",  states)
        health_idxs   = findall(s -> s == "Health",     states)

        found_dmus = Float64[]

        if !isempty(stable_idxs)
            println("  Stable points found at dmu: $(round.(dmu_vals[stable_idxs], sigdigits=4))")

            if !isempty(max_idxs) && minimum(max_idxs) < minimum(stable_idxs)
                boundary_lo = dmu_vals[maximum(max_idxs)]
                boundary_hi = dmu_vals[minimum(stable_idxs)]
                println("  Refining lower boundary [$(round(boundary_lo,sigdigits=4)), $(round(boundary_hi,sigdigits=4))]...")
                dmu_lower = bisect_boundary(r_max, boundary_lo, boundary_hi, N_BISECT)
                push!(found_dmus, dmu_lower)
                println("    → lower boundary dmu* ≈ $(round(dmu_lower, sigdigits=5))")
            end

            upper_unstable = vcat(min_idxs, health_idxs)
            if !isempty(upper_unstable) && maximum(stable_idxs) < minimum(upper_unstable)
                boundary_lo2 = dmu_vals[maximum(stable_idxs)]
                boundary_hi2 = dmu_vals[minimum(upper_unstable)]
                println("  Refining upper boundary [$(round(boundary_lo2,sigdigits=4)), $(round(boundary_hi2,sigdigits=4))]...")
                push!(found_dmus, (boundary_lo2 + boundary_hi2) / 2)
                println("    → upper boundary dmu* ≈ $(round(found_dmus[end], sigdigits=5))")
            end

            mean_stable = mean(dmu_vals[stable_idxs])
            push!(found_dmus, mean_stable)

            window_multiplier = 1.0

        elseif !isempty(max_idxs) && !isempty(min_idxs)
            println("  No stable region. Detected Tumor_Max ↔ Tumor_Min transition.")
            for i in 1:(N_COARSE - 1)
                s1, s2 = states[i], states[i+1]
                if (s1 == "Tumor_Max" && s2 == "Tumor_Min") ||
                   (s1 == "Tumor_Min" && s2 == "Tumor_Max")
                    boundary_lo = dmu_vals[i]
                    boundary_hi = dmu_vals[i+1]
                    println("  Refining transition boundary [$(round(boundary_lo,sigdigits=4)), $(round(boundary_hi,sigdigits=4))]...")
                    dmu_star = bisect_boundary(r_max, boundary_lo, boundary_hi, N_BISECT)
                    push!(found_dmus, dmu_star)
                    println("    → transition boundary dmu* ≈ $(round(dmu_star, sigdigits=5))")
                    break
                end
            end
            window_multiplier = 1.0

        elseif !isempty(max_idxs) && isempty(min_idxs) && isempty(stable_idxs)
            println("  All Tumor_Max – window too low, widening upward.")
            window_multiplier *= FALLBACK_MULTIPLIER

        elseif (isempty(max_idxs)) && (!isempty(min_idxs) || !isempty(health_idxs))
            println("  All Tumor_Min/Health – window too high, widening downward.")
            window_multiplier *= FALLBACK_MULTIPLIER

        else
            println("  Unexpected outcome pattern: $(unique(states)). Skipping.")
        end

        unique!(sort!(found_dmus))
        append_results(output_path, r_max, found_dmus)
        println("  Saved $(length(found_dmus)) boundary estimate(s) for rmax=$(round(r_max, digits=4))")

        for d in found_dmus
            push!(prior, (rmax=r_max, stable_dmu=d))
        end
        predict = fit_boundary_model(prior)
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
