import Pkg
Pkg.activate(joinpath(dirname(@__DIR__)))

using Random, Statistics, ProgressMeter, JSON, StatsBase, Distributions, GLM, LaTeXStrings, SplitApplyCombine

### Save data to file ###

function save_to_file(file::Any, name::String, vec_of_vec::Bool)
    open(string(name), "w") do io
        if vec_of_vec      
            for mi in file
                println(io, join(mi, ", "))  # comma-separated values
            end
        else
            for mi in file
                println(io, mi)  # comma-separated values
            end
        end
    end
end

# Chromosomes Type and Functions
# -------------------------------
mutable struct Chromosomes
#    chromosomes_gene_type::Vector{Vector{Int}}
    chromosomes_gene_mut::Vector{Vector{Int}}
    gene_map::Dict{Int,String}
end

function get_Nchrs(ch::Chromosomes)
    return length(ch.chromosomes_gene_mut)
end

function get_Ngenes(ch::Chromosomes)
    ngenes = 0
    if length(ch.chromosomes_gene_type) == length(ch.chromosomes_gene_mut)
        for i in 1:length(ch.chromosomes_gene_type)
            if length(ch.chromosomes_gene_type[i]) == length(ch.chromosomes_gene_mut[i])
                ngenes = ngenes+length(ch.chromosomes_gene_type[i])
            else
                error("Mismatched genes lenghts")
            end
        end
        return ngenes
    else
        error("Mismatched chromosome lists.")
    end
end

function mutate!(ch::Chromosomes, mu::Float64)
    n_mut = 0
    for i in 1:get_Nchrs(ch)
        chrom = copy(ch.chromosomes_gene_mut[i])
        # For each gene in the chromosome, mutate with probability mu.
        for j in 1:length(chrom)
            if rand() < mu
                chrom[j] = 1
            end
        end
        ch.chromosomes_gene_mut[i] = copy(chrom)
    end
end

function get_genes_by_type(ch::Chromosomes, gene_type::String)
#    restr_gene_mut_gt = Vector{Vector{Int}}()
    restr_gene_mut_gm = Vector{Vector{Int}}()
    for i in 1:length(ch.chromosomes_gene_mut)
        mapped = [ch.gene_map[x] for x in 1:length(ch.chromosomes_gene_mut[i])]
        inds = findall(x -> x == gene_type, mapped)
#        push!(restr_gene_mut_gt, ch.chromosomes_gene_type[i][inds])
        push!(restr_gene_mut_gm, ch.chromosomes_gene_mut[i][inds])
    end
    return restr_gene_mut_gm #restr_gene_mut_gt, 
end


# Cell Type and Methods
# ---------------------
mutable struct Cell
    x::Int
    y::Int
    chromosomes::Chromosomes
    state::Int
    mu0::Float64
    dmu::Float64
    mu::Float64
    r0::Float64
    dr::Float64
    rmax::Float64
    r::Float64
    dm::Float64
    m::Float64
end

get_Nchrs(cell::Cell) = get_Nchrs(cell.chromosomes)

function dead_cell(x::Int, y::Int)
    return Cell(x,y,Chromosomes([], Dict()),2,0,0,0,0,0,0,0,0,0)
end
"""
    compute_activation(cell, gene_type; mode="any")

Helper function that computes, for a given gene type, an activation Boolean vector
across the unique genes. For mode "any", a gene is activated if at least one chromosome
has it mutated; for mode "all", only if all chromosomes show the mutation.
"""
function compute_activation(cell::Cell, gene_type::String; mode::String="any")
    gene_muts = get_genes_by_type(cell.chromosomes, gene_type)
    #if all(x -> isempty(x), gene_types)
    #    return Bool[]
    #end
    #unique_genes = unique(vcat(gene_types...))
    # For each chromosome, build a Boolean vector indicating for each unique gene if any matching gene is mutated.
    activation = [ [ gene_muts[i][j]==1 for j in 1:length(gene_muts[i]) ] for i in 1:length(gene_muts) ]
    if mode == "any"
        # A gene is activated if any chromosome shows it.
        return [ any(activation[i][j] for i in 1:length(activation)) for j in 1:length(gene_muts[1]) ]
    elseif mode == "all"
        # A gene is activated only if every chromosome shows it.
        return [ all(activation[i][j] for i in 1:length(activation)) for j in 1:length(gene_muts[1]) ]
    else
        error("Invalid mode. Use \"any\" or \"all\".")
    end
end

function update_mu!(cell::Cell)
    act = compute_activation(cell, "I"; mode="all")
    cell.mu = cell.mu0 + sum(act) * cell.dmu
end

function update_r!(cell::Cell)
    act_o = compute_activation(cell, "O"; mode="any")
    act_s = compute_activation(cell, "S"; mode="all")
    new_r = cell.r0 + (sum(act_o) + sum(act_s)) * cell.dr
    cell.r = new_r < cell.rmax ? new_r : cell.rmax
end

function update_m!(cell::Cell)
    act_m = compute_activation(cell, "M"; mode="all")
    cell.m = sum(act_m) * cell.dm
end


function check_cell_death(cell::Cell)
    if length(cell.chromosomes.chromosomes_gene_mut) == 0
        return true
    elseif cell.state!=0 && cell.mu==0.
        return true
    elseif get_Nchrs(cell)>5
        return true
    elseif any(compute_activation(cell, "HK"; mode="all"))
        return true
    else
        return false
    end
end

"""
    update!(cell)

Updates the cell state: first checks for cell death, then applies mutation, and updates
the mutation and reproduction rates. Returns 0 if the cell dies, 1 if it reproduces,
or 2 if nothing happens.
"""
function sample_reproducers(rates::Vector{Float64})
    R = sum(rates)
    K = rand(Poisson(R))

    # if no events, short‐circuit:
    if K == 0
        return falses(length(rates)), zeros(Int, length(rates))
    end

    # pick K events (with replacement) weighted by rates:
    events = sample(1:length(rates), Weights(rates), K; replace=false)

    # tabulate
    counts = zeros(Int, length(rates))
    for i in events
        counts[i] += 1
    end
    reproduced = counts .> 0
    return reproduced, events
end


function copy_cell(cell::Cell)
    new_chromosomes = Chromosomes([copy(arr) for arr in cell.chromosomes.chromosomes_gene_mut],cell.chromosomes.gene_map)
    return Cell(cell.x, cell.y, new_chromosomes, cell.state, cell.mu0, cell.dmu, cell.mu, 
                cell.r0, cell.dr, cell.rmax, cell.r, cell.dm, cell.m)
end

# Tissue Type and Functions
# -------------------------
mutable struct Tissue
    L::Int
    cells::Array{Cell,2}
    gene_map::Dict{Int,String}
    neighbor_cache::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}
end

function compute_neighbors(L::Int, row::Int, col::Int)
    neighbors = Vector{Tuple{Int,Int}}()
    for dr in -1:1
        for dc in -1:1
            if dr == 0 && dc == 0
                continue
            end
            new_row = row + dr
            new_col = col + dc
            if new_row ≥ 1 && new_row ≤ L && new_col ≥ 1 && new_col ≤ L
                push!(neighbors, (new_row, new_col))
            end
        end
    end
    return neighbors
end

function Tissue(L::Int, init_cell_state::Array{Cell,2})
    gene_map = init_cell_state[1,1].chromosomes.gene_map
    neighbor_cache = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()
    for i in 1:L, j in 1:L
        neighbor_cache[(i,j)] = compute_neighbors(L, i, j)
    end
    return Tissue(L, init_cell_state, gene_map, neighbor_cache)
end

get_neighbors(tiss::Tissue, row::Int, col::Int) = tiss.neighbor_cache[(row, col)]

function initialize_tiss_indx!(tiss::Tissue)
    for i in 1:tiss.L, j in 1:tiss.L
        tiss.cells[i,j].x = i
        tiss.cells[i,j].y = j
    end
end

function get_idx(tiss::Tissue)
    idx = zeros(Int, tiss.L, tiss.L)
    for i in 1:tiss.L, j in 1:tiss.L
        idx[i,j] = (tiss.cells[i,j].x) + (tiss.L*(tiss.cells[i,j].y-1))
    end
    return idx
end

function copy_tissue(orig::Tissue)
    L = orig.L
    # Deep-copy every cell
    newcells = [ copy_cell(orig.cells[i,j]) for i in 1:L, j in 1:L ]
    # Rebuild a fresh Tissue (this will re-extract gene_map and rebuild neighbor_cache)
    return Tissue(L, newcells)
end

# Tissue Initialization
# --------------------
function init_tissue(L::Int, chrom_gene_mut::Vector{Vector{Int}}, 
    gene_map::Dict{Int,String}, 
    mu0::Float64, dmu::Float64, 
    r0::Float64, dr::Float64, rmax::Float64, dm::Float64)
    cells = Array{Cell}(undef, L, L)
    for i in 1:L, j in 1:L
        ch = Chromosomes(copy(chrom_gene_mut), gene_map)
        cells[i,j] = Cell(i, j, ch, 0, mu0, dmu, mu0,
                                    r0, dr, rmax, r0, dm, 0)
    end
    return Tissue(L, cells)
end

# Perturbation of the initial state
# ---------------------------------
function perturb_init_tissue!(tiss::Tissue, r::Float64, pert_chrom_gene_mut::Vector{Vector{Int}})
    for i in 1:tiss.L
        for j in 1:tiss.L
            if (i-tiss.L//2)^2+(j-tiss.L//2)^2 <= (r*tiss.L)^2
                tiss.cells[i,j].chromosomes.chromosomes_gene_mut = copy(pert_chrom_gene_mut)
                tiss.cells[i,j].state=1
                update_mu!(tiss.cells[i,j])
                update_r!(tiss.cells[i,j])
                update_m!(tiss.cells[i,j])
            end
        end
    end
end


### Tissue update ###

# Helper function for weighted random selection.
function weighted_choice(neighbors::Vector{Tuple{Int,Int}}, weights::Vector{Float64})
    total = sum(weights)
    r = rand() * total
    cum = 0.0
    for (i, w) in enumerate(weights)
        cum += w
        if r < cum
            return neighbors[i]
        end
    end
    return neighbors[end]
end

# Helper function to sample an integer from a power law distribution.
function sample_powerlaw_int(min_length::Int, max_length::Int, alpha::Float64)
    # Generate a uniform random number between 0 and 1.
    u = rand()
    # Inverse transform sampling for a power law:
    #   F(x) = (x^(1-α) - min_length^(1-α)) / (max_length^(1-α) - min_length^(1-α))
    # Solving for x gives:
    x = ((u * (max_length^(1 - alpha) - min_length^(1 - alpha))) + min_length^(1 - alpha))^(1 / (1 - alpha))
    # Ensure the returned length is at least min_length and at most max_length.
    slice_length = max(min_length, min(max_length, round(Int, x)))
    return slice_length
end


# small helper to finish off a cell after placement
function _finalize!(cell::Cell, dead_pos::Vector{Tuple{Int,Int}}, i::Int, j::Int, n_chrs::Int)
    mutate!(cell.chromosomes, cell.mu)
    if cell.state == 0
        if get_Nchrs(cell)!=n_chrs
            cell.state=1
        elseif sum(sum(cell.chromosomes.chromosomes_gene_mut))!=0
            cell.state=1
        end
    end
    
    if check_cell_death(cell)
        push!(dead_pos, (i,j))
    else
        update_mu!(cell)
        update_r!(cell)
        update_m!(cell)
    end
end

# exchange of chromosome from mother to daugther during replication
function _combine!(cell_mother::Cell, i::Int, j::Int, cell_daughter::Cell, ci::Int, cj::Int, dead_pos::Vector{Tuple{Int,Int}})
    if rand()<cell_mother.m && length(cell_mother.chromosomes.chromosomes_gene_mut)>0
        i = sample(1:get_Nchrs(cell_mother))
        chunk = popat!(cell_mother.chromosomes.chromosomes_gene_mut, i)
        push!(cell_daughter.chromosomes.chromosomes_gene_mut, deepcopy(chunk))
    end

    if check_cell_death(cell_mother) # if the cell has 0 chromosomes dies
        push!(dead_pos, (i,j))
    end
    
    if check_cell_death(cell_daughter) # if the cell has 0 chromosomes dies
        push!(dead_pos, (ci,cj))
    end 
    
    return nothing
end


# HUGE FUNCTION for the cell lattice update
function substitute!(tiss::Tissue, n_chrs::Int)
    L = tiss.L
    dead_pos = Vector{Tuple{Int,Int}}()
    # 1) reproducing cells (action == 1)
    repro_positions = findall(reshape(sample_reproducers(vec(get_r(tiss)))[1],tiss.L,tiss.L).==1)
    
    if length(repro_positions)>1
        for idx in eachindex(repro_positions)
            pos = repro_positions[idx]
            i,j = Tuple(pos)            # unpack the CartesianIndex
            cell = tiss.cells[i,j]
    
            # pick neighbor once
            neighs = get_neighbors(tiss, i, j)
            states = [ tiss.cells[r,c].state for (r,c) in neighs ]
            if !all(states.==0)
                rates = [ tiss.cells[r,c].r for (r,c) in neighs ]
                if any(rates.==0.)
                    ci, cj = neighs[rand(findall(rates.==0.))]
                else
                    inv_rates = 1.0 ./ rates
                    ci, cj = weighted_choice(neighs, inv_rates)
                end
                
                ############################################
                copyc = copy_cell(cell)
                tiss.cells[ci,cj] = copyc
                _finalize!(tiss.cells[i,j], dead_pos, i, j, n_chrs)
                _finalize!(tiss.cells[ci,cj], dead_pos, ci, cj, n_chrs)
                _combine!(tiss.cells[i,j],i,j,tiss.cells[ci,cj],ci,cj,dead_pos)
                #############################################
            end
        end

        for pos in dead_pos
            i,j = Tuple(pos)
            tiss.cells[i,j] = dead_cell(i,j)
        end
    end
end





### Exctract important quantities ###


### GROWTH RATE ###
function get_r(tiss::Tissue)
    r_mat = zeros(Float64, tiss.L, tiss.L)
    for i in 1:tiss.L, j in 1:tiss.L
        r_mat[i, j] = tiss.cells[i,j].r
    end
    return r_mat
end

function get_r_canc(tiss::Tissue)
    r_mat = Float64[]
    for i in 1:tiss.L, j in 1:tiss.L
        if tiss.cells[i,j].state==1
            push!(r_mat, tiss.cells[i,j].r)
        end
    end
    return mean(r_mat)
end

### MUTATION RATE ###
function get_mu(tiss::Tissue)
    mu_mat = zeros(Float64, tiss.L, tiss.L)
    for i in 1:tiss.L, j in 1:tiss.L
        mu_mat[i, j] = tiss.cells[i,j].mu
    end
    return mu_mat
end

function get_mu_canc(tiss::Tissue)
    mu_mat = Float64[]
    for i in 1:tiss.L, j in 1:tiss.L
        if tiss.cells[i,j].state==1
            push!(mu_mat, tiss.cells[i,j].mu)
        end
    end
    return mean(mu_mat)
end

### MISSEGREGATION RATE ###
function get_m(tiss::Tissue)
    m_mat = zeros(Float64, tiss.L, tiss.L)
    for i in 1:tiss.L, j in 1:tiss.L
        m_mat[i, j] = tiss.cells[i,j].m
    end
    return m_mat
end

function get_m_canc(tiss::Tissue)
    m_mat = Float64[]
    for i in 1:tiss.L, j in 1:tiss.L
        if tiss.cells[i,j].state==1
            push!(m_mat, tiss.cells[i,j].m)
        end
    end
    return mean(m_mat)
end


### NUMBER OF CHROMOSOMES ###
function get_Nchrs(tiss::Tissue)
    n_mat = zeros(Float64, tiss.L, tiss.L)
    for i in 1:tiss.L, j in 1:tiss.L
        n_mat[i, j] = length(tiss.cells[i,j].chromosomes.chromosomes_gene_mut)
    end
    return n_mat
end

function get_Nchrs_canc(tiss::Tissue)
    n_mat = Float64[]
    for i in 1:tiss.L, j in 1:tiss.L
        if tiss.cells[i,j].state==1
            push!(n_mat, length(tiss.cells[i,j].chromosomes.chromosomes_gene_mut))
        end
    end
    return mean(n_mat)
end

### DENSITY OF CANCER CELLS ###
function get_cancercells(tiss::Tissue)
    n_mat = zeros(Int, tiss.L, tiss.L)
    for i in 1:tiss.L, j in 1:tiss.L
        n_mat[i, j] = tiss.cells[i,j].state
    end
    return sum(n_mat.==1)/tiss.L^2, sum(n_mat.==2)/tiss.L^2
end

### STATE OF EACH CELL ###
function get_state(tiss::Tissue)
    n_mat = zeros(Int, tiss.L, tiss.L)
    for i in 1:tiss.L, j in 1:tiss.L
        n_mat[i, j] = tiss.cells[i,j].state
    end
    return n_mat
end


### AVERGAGE GENE TYPE MUTATION ###
function get_avg_genemutation(tiss::Tissue, gene_types::Vector{String})
    # Determine keys (indices) of genes matching the provided gene_types.
    selected_keys = Int[]
    for gt in gene_types
        for (k, v) in tiss.gene_map
            if v == gt
                push!(selected_keys, k)
            end
        end
    end
    selected_keys = unique(selected_keys)
    
    # Create an output matrix filled with NaN.
    mut_state = fill(NaN, tiss.L, tiss.L)
    
    for i in 1:tiss.L, j in 1:tiss.L
        cell = tiss.cells[i,j]
        if get_Nchrs(cell) == 0
            mut_state[i,j] = NaN
        else
            means = Float64[]
            for c in 1:length(cell.chromosomes.chromosomes_gene_mut)
                # Get the gene types and mutation values for chromosome c.
            #    gene_types_vec = cell.chromosomes.chromosomes_gene_type[c]
                gene_mut_vec = cell.chromosomes.chromosomes_gene_mut[c]
                # Find indices where the gene type is one of the selected keys.
            #    idxs = findall(x -> x in selected_keys, gene_types_vec)
                #if !isempty(idxs)
                push!(means, mean(gene_mut_vec[selected_keys]))#[idxs]))
                #end
            end
            mut_state[i,j] = mean(means)
        end
    end
    return mut_state
end

function get_avg_genemutation_mean(tiss::Tissue, gene_types::Vector{String})
    # Determine keys (indices) of genes matching the provided gene_types.
    selected_keys = Int[]
    for gt in gene_types
        for (k, v) in tiss.gene_map
            if v == gt
                push!(selected_keys, k)
            end
        end
    end
    selected_keys = unique(selected_keys)
    
    # Create an output matrix filled with NaN.
    mut_state = Float64[]
    
    for i in 1:tiss.L, j in 1:tiss.L
        cell = tiss.cells[i,j]
        if cell.state==1
            means = Float64[]
            for c in 1:length(cell.chromosomes.chromosomes_gene_mut)
                # Get the gene types and mutation values for chromosome c.
                #gene_types_vec = cell.chromosomes.chromosomes_gene_type[c]
                gene_mut_vec = cell.chromosomes.chromosomes_gene_mut[c]
                # Find indices where the gene type is one of the selected keys.
                #idxs = findall(x -> x in selected_keys, gene_types_vec)
                #if !isempty(idxs)
                push!(means, mean(gene_mut_vec[selected_keys]))
                #end
            end
            push!(mut_state, mean(means))
        end
    end
    return mean(mut_state), std(mut_state)
end


##### ACTIVATION #####
function get_activation_pergene(tiss::Tissue, gene_type::String, mode::String)
    
    # Create an output matrix filled with NaN.
    mut_state = fill(NaN, tiss.L, tiss.L)
    
    for i in 1:tiss.L, j in 1:tiss.L
        cell = tiss.cells[i,j]
        if get_Nchrs(cell) == 0
            mut_state[i,j] = NaN
        else
            mut_state[i,j] = sum(compute_activation(cell, gene_type; mode=mode))
        end
    end
    return mut_state
end

function get_activation_pergene_canc(tiss::Tissue, gene_type::String, mode::String)
    
    # Create an output matrix filled with NaN.
    mut_state = Float64[]
    
    for i in 1:tiss.L, j in 1:tiss.L
        cell = tiss.cells[i,j]
        if cell.state==1
            if get_Nchrs(cell) != 0
                push!(mut_state, sum(compute_activation(cell, gene_type; mode=mode)))
            end
        end
    end
    return mean(mut_state), std(mut_state)
end

###### NUMBER OF NOT MUTATED GENES #########

function get_nomutgenes(tiss::Tissue, gene_types::Vector{String})
    # Determine keys (indices) of genes matching the provided gene_types.
    selected_keys = Int[]
    for gt in gene_types
        for (k, v) in tiss.gene_map
            if v == gt
                push!(selected_keys, k)
            end
        end
    end
    selected_keys = unique(selected_keys)
    
    # Create an output matrix filled with NaN.
    mut_state = fill(NaN, tiss.L, tiss.L)
    
    for i in 1:tiss.L, j in 1:tiss.L
        cell = tiss.cells[i,j]
        if get_Nchrs(cell) != 0
            v = sum(hcat([(chi[selected_keys].-1).*-1 for chi in cell.chromosomes.chromosomes_gene_mut]))
            mut_state[i,j] = length(v[v.==1])
        end
    end
    return mut_state
end


function get_nomutgenes_mean(tiss::Tissue)
    # Create an output matrix filled with NaN.
    mut_state = Float64[]
    
    for i in 1:tiss.L, j in 1:tiss.L
        cell = tiss.cells[i,j]
        if cell.state==1 && length(cell.chromosomes.chromosomes_gene_mut)>0
            v = sum(hcat([(chi[36:45].-1).*-1 for chi in cell.chromosomes.chromosomes_gene_mut]))
            nvirg = length(v[v.==1])
            nvirg_2 = length(v[v.==2])
            push!(mut_state, 1-(1-cell.mu)^nvirg*(1-cell.mu^2)^nvirg_2)
        end
    end
    return mut_state
end

# Create Gene Map 
# ---------------

function create_gene_map(n_I::Int, n_O::Int, n_S::Int, n_M::Int, n_HK::Int)
    types = vcat(fill("I", n_I), fill("O", n_O), fill("S", n_S), fill("M", n_M), fill("HK", n_HK))
    return Dict(i => types[i] for i in 1:length(types))
end