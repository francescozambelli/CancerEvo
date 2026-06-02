include("utils.jl")

using Random, Statistics, ProgressMeter, JSON, StatsBase, Distributions, GLM, LaTeXStrings, SplitApplyCombine

# Interventions #
function int_a!(tiss, chrom_gene_mut, gene_map, mu0, dmu, r0, dr, dm)
    for p in findall(x->x==1 ,get_state(tiss))
        i,j = Tuple(p)
        if rand()>1-1/(tiss.cells[i,j].r/(r0))
            tiss.cells[i,j] = Cell(i, j, Chromosomes(copy(chrom_gene_mut), gene_map), 0, mu0, dmu, mu0,
                                    r0, dr, 2*r0, r0, dm, 0)
        end
    end
end

function int_b!(tiss, i_lim, p_d, chrom_gene_mut, gene_map, mu0, dmu, r0, dr, dm)
    for p in findall(x->x<=i_lim*dmu ,get_mu(tiss))
        i,j = Tuple(p)
        if rand()<p_d
            tiss.cells[i,j] = Cell(i, j, Chromosomes(copy(chrom_gene_mut), gene_map), 0, mu0, dmu, mu0,
                                    r0, dr, 2*r0, r0, dm, 0)
        end
    end
end

function int_c!(tiss, i_lim, p_d, chrom_gene_mut, gene_map, mu0, dmu, r0, dr, dm)
    for p in findall(x->x>=i_lim*dmu ,get_mu(tiss))
        i,j = Tuple(p)
        if rand()<p_d
            tiss.cells[i,j] = Cell(i, j, Chromosomes(copy(chrom_gene_mut), gene_map), 0, mu0, dmu, mu0,
                                    r0, dr, 2*r0, r0, dm, 0)
        end
    end
end

function int_d!(tiss, new_dmu, chrom_gene_mut, gene_map, mu0, dmu, r0, dr, dm)
    [tc.dmu=new_dmu for tc in tiss.cells]
end