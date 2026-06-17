# scripts/solid/run_interventions.jl

include("../../src/utils_solid.jl")
include("../../src/simulation_solid.jl")
include("../../src/interventions.jl")
include("parameters_solid.jl")

using Random, NPZ, Statistics

function run_single_intervention_sim(intervention_type::String)
    println("\nRunning simulation for Intervention $intervention_type...")
    
    local tiss
    local k_int = -1
    local pre_density = Float64[]
    local pre_mu = Float64[]
    local post_density = Float64[]
    local post_mu = Float64[]
    
    attempt = 0
    while true
        attempt += 1
        # Re-initialize tissue
        tiss = OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
        perturb_optimized!(tiss, 0.05, pert_chrs)
        
        empty!(pre_density)
        empty!(pre_mu)
        k_int = -1
        
        # Phase 1: Pre-intervention
        for k in 1:2000
            substitute_optimized!(tiss, N_CHR, misseg_type)
            cancer_idx = findall(tiss.state .== 1)
            n_canc = length(cancer_idx)
            density = n_canc / (L^2)
            
            push!(pre_density, density)
            push!(pre_mu, n_canc > 0 ? mean(tiss.mu[cancer_idx]) : 0.0)
            
            if density >= 0.3
                k_int = k
                break
            end
            
            if n_canc == 0
                break # Extinct before reaching trigger
            end
        end
        
        if k_int != -1
            println("Attempt $attempt: Reached 30% density at step $k_int. Applying intervention...")
            break
        else
            println("Attempt $attempt: Tumor went extinct or did not reach 30%. Retrying...")
        end
    end
    
    # Phase 2: Apply Intervention
    if intervention_type == "A"
        int_a_optimized!(tiss, N_CHR)
    elseif intervention_type == "B"
        int_b_optimized!(tiss, 1, 1.0, N_CHR) # i_lim=1 (mu <= dmu), p_d=1.0
    elseif intervention_type == "C"
        int_c_optimized!(tiss, 3, 1.0, N_CHR) # i_lim=3 (mu >= 3*dmu), p_d=1.0
    elseif intervention_type == "D"
        int_d_optimized!(tiss, 2.4e-2)       # new_dmu = 2.4e-2 (double dmu)
    else
        error("Unknown intervention type: $intervention_type")
    end
    
    # Phase 3: Post-intervention
    for k in 1:1000
        substitute_optimized!(tiss, N_CHR, misseg_type)
        cancer_idx = findall(tiss.state .== 1)
        n_canc = length(cancer_idx)
        density = n_canc / (L^2)
        
        push!(post_density, density)
        push!(post_mu, n_canc > 0 ? mean(tiss.mu[cancer_idx]) : 0.0)
    end
    
    total_density = vcat(pre_density, post_density)
    total_mu = vcat(pre_mu, post_mu)
    
    return total_density, total_mu, k_int
end

function run_all_interventions()
    # Ensure reproducibility
    Random.seed!(42)
    
    dens_a, mu_a, k_int_a = run_single_intervention_sim("A")
    dens_b, mu_b, k_int_b = run_single_intervention_sim("B")
    dens_c, mu_c, k_int_c = run_single_intervention_sim("C")
    dens_d, mu_d, k_int_d = run_single_intervention_sim("D")
    
    output_dir = joinpath(dirname(dirname(@__DIR__)), "data", "simulations")
    if !isdir(output_dir); mkpath(output_dir); end
    
    output_file = joinpath(output_dir, "intervention_results.npz")
    println("\nSaving intervention results to $output_file ...")
    
    results_dict = Dict(
        "dens_a" => dens_a, "mu_a" => mu_a, "k_int_a" => [k_int_a],
        "dens_b" => dens_b, "mu_b" => mu_b, "k_int_b" => [k_int_b],
        "dens_c" => dens_c, "mu_c" => mu_c, "k_int_c" => [k_int_c],
        "dens_d" => dens_d, "mu_d" => mu_d, "k_int_d" => [k_int_d],
    )
    
    npzwrite(output_file, results_dict)
    println("Done!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_interventions()
end
