include("utils.jl")
include("parameters.jl")
include("interventions.jl")

import Pkg

# List of packages you want
packages = [
    "Random", "Statistics", "Plots", "ProgressMeter", "JSON", 
    "StatsBase", "Distributions", "GLM", "LaTeXStrings", 
    "SplitApplyCombine", "StatsPlots"
]

for pkg in packages
    # Check if the package can be loaded
    try
        @eval using $(Symbol(pkg))
    catch
        println("Installing $pkg ...")
        Pkg.add(pkg)
        @eval using $(Symbol(pkg))
    end
end

using Random, Statistics, Plots, ProgressMeter, JSON, StatsBase, Distributions, GLM, LaTeXStrings, SplitApplyCombine, StatsPlots

# Simulation loop

mutable struct Results
    state::String
    mu::Vector{Float64}
    r::Vector{Float64}
    m::Vector{Float64}
    n_chrs::Vector{Float64}
    tumor_density::Vector{Float64}
    dcells_density::Vector{Float64}
    tissues::Vector{Tissue}
    muts::Vector{Vector{Float64}}
    activations::Vector{Vector{Float64}}
    I_populations_density::Vector{Vector{Float64}}
end

function simulation(tiss::Tissue, n_chr::Int, n_steps::Int, n_it_store::Int, track_state::Bool, bar=true, limit=0.5)
    ##
    res = Results(
    "",                         ## state
    Float64[],                  ## mu
    Float64[],                  ## r
    Float64[],                  ## m
    Float64[],                  ## chrs
    Float64[],                  ## tumor
    Float64[],                  ## dcells
    Tissue[],                   ## tissues
    Vector{Vector{Float64}}(),  ## muts
    Vector{Vector{Float64}}(),  ## acts
    Vector{Vector{Float64}}())  ## pop_dens
    
    res.state = "Done"
    
    @showprogress enabled=bar for k in 1:n_steps
        substitute!(tiss, n_chr)  # Update system stat
        # Save important quantities
        push!(res.mu, get_mu_canc(tiss))
        push!(res.r, get_r_canc(tiss))
        push!(res.m, get_m_canc(tiss))
        push!(res.n_chrs, get_Nchrs_canc(tiss))
        
        tumorsize, death_cells = get_cancercells(tiss) #number of cell not wild type
        push!(res.tumor_density, mean(tumorsize))
        push!(res.dcells_density, mean(death_cells))
        
        push!(res.muts, [get_avg_genemutation_mean(tiss, ["I"])[1],
                         get_avg_genemutation_mean(tiss, ["O"])[1],
                         get_avg_genemutation_mean(tiss, ["S"])[1],
                         get_avg_genemutation_mean(tiss, ["M"])[1],
                         get_avg_genemutation_mean(tiss, ["HK"])[1]])

        push!(res.activations, [get_activation_pergene_canc(tiss, "I","all")[1],
                                get_activation_pergene_canc(tiss, "O","any")[1],
                                get_activation_pergene_canc(tiss, "S","all")[1],
                                get_activation_pergene_canc(tiss, "M","all")[1],
                                get_activation_pergene_canc(tiss, "HK","all")[1]])
        
        sizz = length(findall(x-> x==1, get_state(tiss)))
        i_state = get_mu(tiss)[findall(x-> x==1, get_state(tiss))]
        cm = sort(collect(countmap(i_state)); by=first)
        countss = [(cm[i][2]/sizz) for i in 1:length(cm)] 
        push!(res.I_populations_density, countss)
        
        # store a tissue object every x iterations
        if rem(k,n_it_store)==0
            push!(res.tissues, copy_tissue(tiss)) 
        end
    
        # Conditions to end the simulation

        # 1. all the cells are wt (tumor exthinguished)
        if all(tumorsize.==0)
            println("All normal cells at iteration ", k)
            res.state = "Health"
            break
        end
        # 2. if the tumors size if larger than 50%
        if mean(tumorsize)>limit
            println("Tumor size limit reached at iteration ", k)
            res.state="Tumor"
            break
        end
    end
    res.muts = invert(res.muts)
    res.activations = invert(res.activations)
    push!(res.tissues, copy_tissue(tiss))
    
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Initializing simulation...")
    tiss = init_tissue(L, chrom_gene_mut, gene_map, mu0, dmu, r0, dr, 2*r0, dm)
    perturb_init_tissue!(tiss, 0.005, pert_chrom_gene_mut)
    println("Running simulation...")
    res_0 = simulation(tiss, length(chrom_gene_mut), 2500, 100, false, true, 0.4)
    println("Saving results...")
    
    # Save the results to data/simulations/
    out_dir = joinpath(dirname(@__DIR__), "data", "simulations")
    mkpath(out_dir)
    save_to_file(res_0.mu, joinpath(out_dir, "0ch_m.txt"), false)
    save_to_file(res_0.r, joinpath(out_dir, "0ch_r.txt"), false)
    save_to_file(res_0.activations[1], joinpath(out_dir, "0ch_ia.txt"), false)
    save_to_file(res_0.tumor_density, joinpath(out_dir, "0ch_t.txt"), false)
    save_to_file(res_0.muts[5], joinpath(out_dir, "0ch_k.txt"), false)
    println("Simulation finished and results saved to $out_dir")
end