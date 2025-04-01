import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from graph_tool.all import *
import scipy as sp
import sklearn
from tqdm import tqdm
import imageio
import shutil

class Chromosomes:
    def __init__(self, chromosomes_gene_type, chromosomes_gene_mut, gene_map):
        # Convert lists to NumPy arrays for vectorized operations.
        self.chromosomes_gene_type = [np.array(chrom) for chrom in chromosomes_gene_type]
        self.chromosomes_gene_mut = [np.array(chrom) for chrom in chromosomes_gene_mut]
        self.gene_map = gene_map

    def get_Nchrs(self):
        if len(self.chromosomes_gene_type) == len(self.chromosomes_gene_mut):
            return len(self.chromosomes_gene_type)
        else:
            raise ValueError("Mismatched chromosome lists.")

    def mutate(self, mu):
        n_mut = 0
        for i in range(self.get_Nchrs()):
            chrom = self.chromosomes_gene_mut[i]
            rand_vals = np.random.rand(len(chrom))
            mutations = rand_vals < mu
            n_mut += np.sum(mutations)
            # Set genes to 1 where mutation occurs.
            chrom[mutations] = 1
            self.chromosomes_gene_mut[i] = chrom
        return n_mut
        
    def crossover(self, ext_chr_gt, ext_chr_gm):
        idx_chr = np.random.randint(0, self.get_Nchrs())
        self.chromosomes_gene_type[idx_chr] = np.concatenate([
            self.chromosomes_gene_type[idx_chr],
            np.array(ext_chr_gt)
        ])
        self.chromosomes_gene_mut[idx_chr] = np.concatenate([
            self.chromosomes_gene_mut[idx_chr],
            np.array(ext_chr_gm)
        ])

    def get_genes_by_type(self, gene_type):
        # Map gene codes to their types using gene_map.
        mapped_chromosomes = [np.array([self.gene_map[x] for x in chrom])
                              for chrom in self.chromosomes_gene_type]
        restr_gene_mut_gt = []
        restr_gene_mut_gm = []
        for i, mapped in enumerate(mapped_chromosomes):
            mask = mapped == gene_type
            restr_gene_mut_gt.append(self.chromosomes_gene_type[i][mask])
            restr_gene_mut_gm.append(self.chromosomes_gene_mut[i][mask])
        return restr_gene_mut_gt, restr_gene_mut_gm


class Cell:
    def __init__(self, x, y, chromosomes, mu0, dmu, mumax, r0, dr, rmax):
        self.x = x
        self.y = y
        
        self.mu0 = mu0
        self.dmu = dmu
        self.mumax = mumax
        self.mu = mu0
        
        self.r0 = r0
        self.dr = dr
        self.rmax = rmax
        self.r = r0
        
        self.chromosomes = chromosomes

    def get_Nchrs(self):
        return self.chromosomes.get_Nchrs()

    def _compute_activation(self, gene_type, mode='any'):
        """
        Computes the activation status for a given gene type.
        mode 'any' returns a Boolean array per unique gene where at least one chromosome is activated,
        mode 'all' requires all chromosomes to be activated.
        """
        gene_types, gene_muts = self.chromosomes.get_genes_by_type(gene_type)
        if not gene_types or all(len(gt) == 0 for gt in gene_types):
            return np.array([])
        unique_genes = np.unique(np.concatenate(gene_types))
        activation = np.array([
            [int(np.any(gene_muts[i][gene_types[i] == ug])) for ug in unique_genes]
            for i in range(len(gene_types))
        ])
        if mode == 'any':
            return np.any(activation, axis=0)
        elif mode == 'all':
            return np.all(activation, axis=0)
        else:
            raise ValueError("Invalid mode. Use 'any' or 'all'.")

    def update_mu(self):
        act = self._compute_activation("I", mode='any')
        new_mu = self.mu0 + np.sum(act) * self.dmu
        self.mu = new_mu if (self.mumax is None or new_mu < self.mumax) else self.mumax

    def update_r(self):
        act_o = self._compute_activation("O", mode='any')
        act_s = self._compute_activation("S", mode='all')
        new_r = self.r0 + (np.sum(act_o) + np.sum(act_s)) * self.dr
        self.r = new_r if (self.rmax is None or new_r < self.rmax) else self.rmax

    def reproduce(self):
        return np.random.rand() < self.r

    def check_cell_death(self):
        if self.get_Nchrs() == 0:
            return True
        act_hk = self._compute_activation("HK", mode='all')
        # If any housekeeping gene is fully mutated, the cell is marked for death.
        return np.any(act_hk)

    def update(self):
        if self.check_cell_death():
            return 0  # Death
        
        self.chromosomes.mutate(self.mu)
        self.update_mu()
        self.update_r()
        return 1 if self.reproduce() else 2

    def copy(self):
        new_chromosomes = Chromosomes(
            [chrom.copy() for chrom in self.chromosomes.chromosomes_gene_type],
            [chrom.copy() for chrom in self.chromosomes.chromosomes_gene_mut],
            self.chromosomes.gene_map
        )
        return Cell(self.x, self.y, new_chromosomes, self.mu0, self.dmu, self.mumax, self.r0, self.dr, self.rmax)

    def replicate_split_chromosomes(self):
        idx_chr = np.random.choice(np.arange(self.get_Nchrs()))
        mother = self.copy()
        daughter = self.copy()
        # Remove the selected chromosome from the mother.
        mother.chromosomes.chromosomes_gene_type.pop(idx_chr)
        mother.chromosomes.chromosomes_gene_mut.pop(idx_chr)
        # Append a copy of the selected chromosome to the daughter.
        daughter.chromosomes.chromosomes_gene_type.append(
            daughter.chromosomes.chromosomes_gene_type[idx_chr].copy()
        )
        daughter.chromosomes.chromosomes_gene_mut.append(
            daughter.chromosomes.chromosomes_gene_mut[idx_chr].copy()
        )
        return mother, daughter


class Tissue:
    def __init__(self, L, init_cell_state, p_poly=1e-3):
        self.L = L
        self.cells = init_cell_state
        self.p_poly = p_poly
        self.gene_map = init_cell_state[0][0].chromosomes.gene_map
        # Precompute neighbors for each grid cell.
        self.neighbor_cache = {
            (i, j): self._compute_neighbors(i, j)
            for i in range(L) for j in range(L)
        }
    
    def _compute_neighbors(self, row, col):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.L and 0 <= new_col < self.L:
                    neighbors.append((new_row, new_col))
        return neighbors

    def get_neighbors(self, row, col):
        return self.neighbor_cache[(row, col)]
    
    def get_actions(self):
        actions = np.empty((self.L, self.L), dtype=int)
        for i in range(self.L):
            for j in range(self.L):
                actions[i, j] = self.cells[i][j].update()
        return actions
    
    def substitute(self, action):
        # Replace dead cells using neighbors weighted inversely by their reproduction rate.
        death_cells = np.argwhere(action == 0)
        for row, col in death_cells:
            neighs = self.get_neighbors(row, col)
            rates = np.array([self.cells[n[0]][n[1]].r for n in neighs])
            inv_rates = 1 / rates
            prob = inv_rates / np.sum(inv_rates)
            selected = neighs[np.random.choice(len(neighs), p=prob)]
            self.cells[row][col] = self.cells[selected[0]][selected[1]].copy()
        
        # For cells that reproduced, copy them into a neighboring cell.
        reproducing_cells = np.argwhere(action == 1)
        for row, col in reproducing_cells:
            neighs = self.get_neighbors(row, col)
            rates = np.array([self.cells[n[0]][n[1]].r for n in neighs])
            prob = rates / np.sum(rates)
            selected = neighs[np.random.choice(len(neighs), p=prob)]
            if np.random.rand() < self.p_poly:
                old_cell, new_cell = self.cells[row][col].replicate_split_chromosomes()
            else:
                old_cell, new_cell = self.cells[row][col], self.cells[row][col].copy()
            self.cells[row][col] = old_cell
            self.cells[selected[0]][selected[1]] = new_cell

    def update(self):
        action = self.get_actions()
        self.substitute(action)
        return action
    
    def get_r(self):
        r_mat = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(self.L):
                r_mat[i, j] = self.cells[i][j].r
        return r_mat

    def get_mu(self):
        mu_mat = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(self.L):
                mu_mat[i, j] = self.cells[i][j].mu
        return mu_mat

    def get_Nchrs(self):
        n_mat = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(self.L):
                n_mat[i, j] = self.cells[i][j].get_Nchrs()
        return n_mat
    
    def get_avg_genemutation(self, gene_types):
        # Precompute indices for each gene type.
        gene_keys = np.array(list(self.gene_map.keys()))
        gene_vals = np.array(list(self.gene_map.values()))
        selected_indices = np.concatenate([np.where(gene_vals == gt)[0] for gt in gene_types])
        
        mut_state = np.full((self.L, self.L), np.nan)
        for i in range(self.L):
            for j in range(self.L):
                if self.cells[i][j].get_Nchrs() == 0:
                    continue
                chrom_mut = self.cells[i][j].chromosomes.chromosomes_gene_mut
                mean_vals = [
                    np.mean(chrom[selected_indices]) if len(chrom[selected_indices]) > 0 else np.nan
                    for chrom in chrom_mut
                ]
                mut_state[i, j] = np.nanmean(mean_vals)
        return mut_state


def init_tissue(L, chrom_gene_type, chrom_gene_mut, gene_map, mu0, dmu, mumax, r0, dr, rmax):
    cell_init_state = []
    for i in range(L):
        row = []
        for j in range(L):
            ch = Chromosomes(chrom_gene_type, chrom_gene_mut, gene_map=gene_map)
            row.append(Cell(i, j, chromosomes=ch,
                            mu0=mu0, dmu=dmu, mumax=mumax,
                            r0=r0, dr=dr, rmax=rmax))
        cell_init_state.append(row)
    return cell_init_state

# Model parameters
chrom_gene_type = [np.arange(16), np.arange(16)]
chrom_gene_mut  = [np.zeros(16, dtype=int), np.zeros(16, dtype=int)]
L = 50
mu0 = 1e-8
dmu = 1e-4
mumax = 10 * mu0
r0 = 5e-3
dr = 8e-5
rmax = 2 * r0
p_poly = 5e-5

# Plot parameters (unchanged)
nframes = 2
repsperframe = 300
p1args = {"vmin": r0, "vmax": rmax, "cbar": True}
p2args = {"vmin": 0, "vmax": 5, "cbar": True}
gtype_p3 = ["I", "O", "S"]
p3args = {"vmin": 0, "vmax": 1, "cbar": True}
gtype_p4 = ["HK"]
p4args = {"vmin": 0, "vmax": 1, "cbar": True}

add_tag = "_pert"
gifname = ("cancer_L%d_r%.0e_dr%.0e_rmax%.0e_mu%.0e_dmu%.0e_mumax%.0e_ppoly%.0e_frames%d" %
           (L, r0, dr, rmax, mu0, dmu, mumax, p_poly, nframes)) + add_tag + ".gif"

# Dummy gene_map (adjust as needed)
gene_map = {i: "I" for i in range(16)}
init_state = init_tissue(L, chrom_gene_type, chrom_gene_mut, gene_map,
                         mu0=mu0, dmu=dmu, mumax=mumax, r0=r0, dr=dr, rmax=rmax)

# Introduce a perturbation in the central cell.
pert_chrom_gene_type = [np.arange(16), np.arange(16), np.arange(16)]
pert_chrom_gene_mut  = [np.zeros(16, dtype=int), np.zeros(16, dtype=int),
                        np.zeros(16, dtype=int)]
pert_chrom_gene_mut[2][1] = 1  # Introduce a mutation in one gene.
ch_pert = Chromosomes(pert_chrom_gene_type, pert_chrom_gene_mut, gene_map=gene_map)
init_state[L//2][L//2] = Cell(L//2, L//2, chromosomes=ch_pert,
                               mu0=mu0, dmu=dmu, mumax=mumax, r0=r0, dr=dr, rmax=rmax)

tiss = Tissue(L, init_cell_state=init_state, p_poly=p_poly)

for k in tqdm(range(400)):
    tiss.update()