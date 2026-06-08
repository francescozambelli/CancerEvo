# scratch/benchmark_init.jl

function get_neighbors_1d(L, idx)
    r = ((idx - 1) % L) + 1
    c = ((idx - 1) ÷ L) + 1
    neighs = Int[]
    for dr in -1:1, dc in -1:1
        (dr == 0 && dc == 0) && continue
        nr, nc = r + dr, c + dc
        if 1 <= nr <= L && 1 <= nc <= L
            push!(neighs, nr + (nc - 1) * L)
        end
    end
    return neighs
end

mutable struct OptimizedTissue
    L::Int
    N_genes::Int
    state::Vector{Int8}
    mu::Vector{Float64}
    r::Vector{Float64}
    m::Vector{Float64}
    chromosomes::Matrix{UInt64}
    n_chrs::Vector{Int8}
    mu0::Float64
    dmu::Float64
    r0::Float64
    dr::Float64
    rmax::Float64
    dm::Float64
    mask_I::UInt64
    mask_O::UInt64
    mask_S::UInt64
    mask_M::UInt64
    mask_HK::UInt64
    neighbors::Vector{Vector{Int}}
end

function OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, n_chrs_init)
    N = L * L
    state = zeros(Int8, N)
    mu = fill(mu0, N)
    r = fill(r0, N)
    m = zeros(Float64, N)
    chromosomes = zeros(UInt64, 6, N)
    n_chrs = fill(Int8(n_chrs_init), N)
    mask_I = (UInt64(1) << N_I) - 1
    mask_O = ((UInt64(1) << N_O) - 1) << N_I
    mask_S = ((UInt64(1) << N_S) - 1) << (N_I + N_O)
    mask_M = ((UInt64(1) << N_M) - 1) << (N_I + N_O + N_S)
    mask_HK = ((UInt64(1) << N_HK) - 1) << (N_I + N_O + N_S + N_M)
    neighs = [get_neighbors_1d(L, i) for i in 1:N]
    return OptimizedTissue(L, N_I+N_O+N_S+N_M+N_HK, state, mu, r, m, chromosomes, n_chrs,
                           mu0, dmu, r0, dr, rmax, dm, 
                           mask_I, mask_O, mask_S, mask_M, mask_HK, neighs)
end

mutable struct LiquidTissue
    N::Int
    N_genes::Int
    state::Vector{Int8}
    mu::Vector{Float64}
    r::Vector{Float64}
    m::Vector{Float64}
    chromosomes::Matrix{UInt64}
    n_chrs::Vector{Int8}
    mu0::Float64
    dmu::Float64
    r0::Float64
    dr::Float64
    rmax::Float64
    dm::Float64
    mask_I::UInt64
    mask_O::UInt64
    mask_S::UInt64
    mask_M::UInt64
    mask_HK::UInt64
end

function LiquidTissue(N, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, n_chrs_init)
    state = zeros(Int8, N)
    mu = fill(mu0, N)
    r = fill(r0, N)
    m = zeros(Float64, N)
    chromosomes = zeros(UInt64, 6, N)
    n_chrs = fill(Int8(n_chrs_init), N)
    mask_I = (UInt64(1) << N_I) - 1
    mask_O = ((UInt64(1) << N_O) - 1) << N_I
    mask_S = ((UInt64(1) << N_S) - 1) << (N_I + N_O)
    mask_M = ((UInt64(1) << N_M) - 1) << (N_I + N_O + N_S)
    mask_HK = ((UInt64(1) << N_HK) - 1) << (N_I + N_O + N_S + N_M)
    return LiquidTissue(N, N_I+N_O+N_S+N_M+N_HK, state, mu, r, m, chromosomes, n_chrs,
                        mu0, dmu, r0, dr, rmax, dm, 
                        mask_I, mask_O, mask_S, mask_M, mask_HK)
end

# Warmup
L = 80
N = L * L
N_I, N_O, N_S, N_M, N_HK = 10, 10, 10, 5, 10
mu0, dmu, r0, dr, rmax, dm, N_CHR = 0.0, 0.023, 0.15, 0.005, 0.30, 0.0, 2

OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
LiquidTissue(N, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)

println("--- L = 80, 1000 repetitions ---")
print("OptimizedTissue: ")
@time for _ in 1:1000
    OptimizedTissue(L, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
end

print("LiquidTissue:    ")
@time for _ in 1:1000
    LiquidTissue(N, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
end

L2 = 200
N2 = L2 * L2
println("\n--- L = 200, 100 repetitions ---")
print("OptimizedTissue: ")
@time for _ in 1:100
    OptimizedTissue(L2, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
end

print("LiquidTissue:    ")
@time for _ in 1:100
    LiquidTissue(N2, N_I, N_O, N_S, N_M, N_HK, mu0, dmu, r0, dr, rmax, dm, N_CHR)
end
