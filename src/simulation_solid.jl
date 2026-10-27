# src/simulation.jl

include("utils_solid.jl")

function simulation_optimized(tiss::OptimizedTissue, n_chr_init::Int, n_steps::Int, n_it_store::Int, bar=true, limit=0.5, lower_limit=0.0, misseg_type::String="whole")
    res = OptimizedResults(
        "Done",
        Float64[], # mu
        Float64[], # r
        Float64[], # m
        Float64[], # n_chrs
        Float64[], # tumor_density
        Float64[], # dcells_density
        Vector{Vector{Float64}}(), # muts
        Vector{Vector{Float64}}()  # activations
    )

    @showprogress enabled=bar for k in 1:n_steps
        substitute_optimized!(tiss, n_chr_init, misseg_type)
        
        # Extract metrics
        cancer_idx = findall(tiss.state .== 1)
        dead_idx = findall(tiss.state .== 2)
        n_canc = length(cancer_idx)
        n_dead = length(dead_idx)
        
        density = n_canc / (tiss.L^2)
        push!(res.tumor_density, density)
        push!(res.dcells_density, n_dead / (tiss.L^2))
        
        if n_canc > 0
            push!(res.mu, mean(tiss.mu[cancer_idx]))
            push!(res.r, mean(tiss.r[cancer_idx]))
            push!(res.m, mean(tiss.m[cancer_idx]))
            push!(res.n_chrs, mean(tiss.n_chrs[cancer_idx]))
            
            # Internal-state ploidy distribution & mutation burden by n_chrs
            nc_counts = zeros(Int, 5)
            nc_mut_sums = zeros(Float64, 5)
            eff_p_sum = 0.0
            for idx in cancer_idx
                nc = Int(tiss.n_chrs[idx])
                pres_tot = 0
                mut_tot = 0
                for c in 1:nc
                    p_c = tiss.presence[c, idx]
                    pres_tot += count_ones(p_c)
                    mut_tot  += count_ones(tiss.chromosomes[c, idx] & p_c)
                end
                eff_p_sum += pres_tot / tiss.N_genes
                if 1 <= nc <= 5
                    nc_counts[nc] += 1
                    if pres_tot > 0
                        nc_mut_sums[nc] += mut_tot / pres_tot
                    end
                end
            end
            push!(res.eff_ploidy, eff_p_sum / n_canc)
            push!(res.n_chrs_dist, Float64.(nc_counts) ./ n_canc)
            push!(res.mut_by_nc, [nc_counts[k] > 0 ? nc_mut_sums[k] / nc_counts[k] : 0.0 for k in 1:5])
            
            # Mutations and Activations (per type)
            mut_means = Float64[]
            act_means = Float64[]
            
            # Helper for metrics
            for (mask, mode) in [(tiss.mask_I, "all"), (tiss.mask_O, "any"), 
                                 (tiss.mask_S, "all"), (tiss.mask_M, "all"), (tiss.mask_HK, "all")]
                
                # Mutation mean: average fraction of set bits in mask across all chromosomes of cancer cells
                total_bits = count_ones(mask)
                m_sum = 0.0
                a_sum = 0.0
                for idx in cancer_idx
                    nc = tiss.n_chrs[idx]
                    all_muts, any_muts = get_cell_masks(tiss, idx)
                    
                    # Activation part
                    combined = (mode == "all") ? all_muts : any_muts
                    a_sum += count_ones(combined & mask)
                    
                    # Mutation part (average fraction of present loci in class that are mutated)
                    c_sum = 0.0
                    valid_chrs = 0
                    for c in 1:nc
                        pres_bits = count_ones(tiss.presence[c, idx] & mask)
                        if pres_bits > 0
                            c_sum += count_ones(tiss.chromosomes[c, idx] & tiss.presence[c, idx] & mask) / pres_bits
                            valid_chrs += 1
                        end
                    end
                    if valid_chrs > 0
                        m_sum += c_sum / valid_chrs
                    end
                end
                push!(mut_means, m_sum / n_canc)
                push!(act_means, a_sum / n_canc)
            end
            push!(res.muts, mut_means)
            push!(res.activations, act_means)
        else
            push!(res.mu, 0.0); push!(res.r, 0.0); push!(res.m, 0.0); push!(res.n_chrs, 0.0)
            push!(res.eff_ploidy, 0.0)
            push!(res.n_chrs_dist, zeros(5))
            push!(res.mut_by_nc, zeros(5))
            push!(res.muts, zeros(5)); push!(res.activations, zeros(5))
        end

        # Termination
        if n_canc == 0
            res.state = "Health"
            break
        end
        n_wt = count(tiss.state .== 0)
        wt_density = n_wt / (tiss.L^2)
        if density >= limit
            println("Over")
            res.state = "Tumor_Max"
            break
        end
        if wt_density > (1.0 - lower_limit) && k > 1 # Allow one step to stabilize if needed
            println("Under")
            res.state = "Tumor_Min"
            break
        end
    
    end
    
    if res.state == "Done"
        println("Completed steps")
    end
    
    # Post-process to match old format (invert list of vectors to list of type-trajectories)
    res.muts = invert(res.muts)
    res.activations = invert(res.activations)
    res.n_chrs_dist = invert(res.n_chrs_dist)
    res.mut_by_nc   = invert(res.mut_by_nc)
    
    return res
end
