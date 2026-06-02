module OldSim
    include("../scripts/utils.jl")
    include("../scripts/interventions.jl")
    # Redefine simulation to avoid ProgressMeter collision if any, but it should be fine
    include("../scripts/simulation.jl")
end

module NewSim
    include("utils.jl")
    include("simulation.jl")
end

using Random, Statistics

# Shared Parameters
L = 100
N_CHR = 2
N_I, N_O, N_S, N_M, N_HK = 10, 10, 10, 5, 10
mu0, dmu, r0, dr, dm = 0.0, 15e-3, 1.5e-1, 0.8e-2, 1e-2
n_steps = 200

# Helper to convert old chromosome format to UInt64
function old_to_new_chr(vec)
    res = UInt64(0)
    for (i, val) in enumerate(vec)
        if val == 1
            res |= (UInt64(1) << (i-1))
        end
    end
    return res
end

pert_vec = fill(0, 45)
pert_vec[1] = 1; pert_vec[1+N_I] = 1

# 1. Benchmark Old Simulation
println("--- BENCHMARKING OLD SIMULATION ---")
Random.seed!(42)
old_tiss = OldSim.init_tissue(L, fill(fill(0, 45), N_CHR), OldSim.create_gene_map(N_I, N_O, N_S, N_M, N_HK), mu0, dmu, r0, dr, 2*r0, dm)
OldSim.perturb_init_tissue!(old_tiss, 0.05, [copy(pert_vec), copy(pert_vec)])

t_old = @elapsed res_old = OldSim.simulation(old_tiss, N_CHR, n_steps, 100, false, false, 0.4)
println("Old Simulation Time: ", round(t_old, digits=3), "s")

# 2. Benchmark New Simulation
println("\n--- BENCHMARKING NEW SIMULATION ---")
Random.seed!(42)
new_tiss = NewSim.OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, 2*r0, dm, N_CHR)
pert_chr = old_to_new_chr(pert_vec)
NewSim.perturb_optimized!(new_tiss, 0.05, [pert_chr, pert_chr])

t_new = @elapsed res_new = NewSim.simulation_optimized(new_tiss, N_CHR, n_steps, 100, false, 0.4)
println("New Simulation Time: ", round(t_new, digits=3), "s")

# 3. Compare Results
println("\n--- COMPARISON ---")
println("Old Final Tumor Density: ", res_old.tumor_density[end])
println("New Final Tumor Density: ", res_new.tumor_density[end])
println("Speedup: ", round(t_old / t_new, digits=1), "x")

# Check accuracy at some steps
len = min(length(res_old.tumor_density), length(res_new.tumor_density))
diff = mean(abs.(res_old.tumor_density[1:len] .- res_new.tumor_density[1:len]))
println("Mean Absolute Difference in Tumor Density: ", round(diff, digits=5))
