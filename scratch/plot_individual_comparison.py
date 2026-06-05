import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load empirical data
csv_path = Path("data/simulations_liquid/ensemble_results/ensemble_results.csv")
df = pd.read_csv(csv_path)
tumor_ids = df[df["outcome"] == "Tumor_Max"]["sim_id"].values

empirical_muts = []
empirical_acts = []
empirical_rs = []

for sid in tumor_ids:
    sim_path = Path("data/simulations_liquid/ensemble_results") / f"sim_{sid}.npz"
    with np.load(sim_path) as f:
        empirical_muts.append(f["mut_I"])
        empirical_acts.append(f["act_I"])
        empirical_rs.append(f["r"])

# Find the maximum length to align
max_len = max(len(m) for m in empirical_muts)

# Pad with NaN and compute mean
arr_mut = np.full((len(empirical_muts), max_len), np.nan)
arr_act = np.full((len(empirical_acts), max_len), np.nan)
arr_r = np.full((len(empirical_rs), max_len), np.nan)

for idx in range(len(empirical_muts)):
    arr_mut[idx, :len(empirical_muts[idx])] = empirical_muts[idx]
    arr_act[idx, :len(empirical_acts[idx])] = empirical_acts[idx]
    arr_r[idx, :len(empirical_rs[idx])] = empirical_rs[idx]

# Calculate mean only where we have at least 5 simulations to avoid noisy tails
num_valid = np.sum(~np.isnan(arr_mut), axis=0)
valid_mask = num_valid >= 5
mean_mut_emp = np.nanmean(arr_mut, axis=0)[valid_mask]
mean_act_emp = np.nanmean(arr_act, axis=0)[valid_mask]
mean_r_emp = np.nanmean(arr_r, axis=0)[valid_mask]

# 2. Setup the theoretical ODE
N_I = 10
N_HK = 10
dmu = 2.3e-2

states = []
for n2 in range(1, N_I + 1):
    for n1 in range(0, N_I - n2 + 1):
        states.append((n1, n2))

state_to_idx = {state: idx for idx, state in enumerate(states)}
num_states = len(states)

T = np.zeros((num_states, num_states))
for idx_from, (n1_from, n2_from) in enumerate(states):
    mu = n2_from * dmu
    p_survival = (1.0 - mu**2)**N_HK
    wt_from = N_I - n1_from - n2_from
    
    for k in range(n1_from + 1):
        p_k = np.math.comb(n1_from, k) * (mu**k) * ((1.0 - mu)**(n1_from - k))
        for h1 in range(wt_from + 1):
            for h2 in range(wt_from - h1 + 1):
                p_h = (np.math.comb(wt_from, h1) * np.math.comb(wt_from - h1, h2) *
                       (2.0 * mu * (1.0 - mu))**h1 *
                       (mu**2)**h2 *
                       ((1.0 - mu)**2)**(wt_from - h1 - h2))
                
                n1_to = n1_from - k + h1
                n2_to = n2_from + k + h2
                state_to = (n1_to, n2_to)
                if state_to in state_to_idx:
                    idx_to = state_to_idx[state_to]
                    T[idx_to, idx_from] += p_survival * p_k * p_h

p = np.zeros(num_states)
p[state_to_idx[(0, 1)]] = 1.0  # Seed has 1 active (n2=1) and 0 heterozygous (n1=0)

ode_muts = [1.0 / N_I]
ode_acts = [1.0]

valid_len = len(mean_mut_emp)

for i in range(1, valid_len):
    # Each stored step corresponds to 1 simulation step!
    dt_div = 1.0 * mean_r_emp[i-1]
    
    # We integrate the ODE
    n_substeps = 10
    sub_dt = dt_div / n_substeps
    for _ in range(n_substeps):
        T_p = T @ p
        phi = np.sum(T_p)
        dp = T_p - phi * p
        p = p + dp * sub_dt
        p = p / np.sum(p)
    
    mean_n2 = np.sum(p * [state[1] for state in states])
    mean_mut = np.sum(p * [state[0] + 2*state[1] for state in states]) / (2.0 * N_I)
    
    ode_muts.append(mean_mut)
    ode_acts.append(mean_n2)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

steps_theory = np.arange(valid_len)

# Plot individual simulation trajectories in light grey
for idx in range(len(empirical_muts)):
    s = np.arange(len(empirical_muts[idx]))
    ax[0].plot(s, empirical_muts[idx], color='lightgray', alpha=0.3, lw=1)
    ax[1].plot(s, empirical_acts[idx], color='lightgray', alpha=0.3, lw=1)

# Plot empirical mean and ODE prediction
ax[0].plot(steps_theory, mean_mut_emp, 'k-', lw=3, label='Simulation Mean')
ax[0].plot(steps_theory, ode_muts, 'r--', lw=3, label='Master Equation ODE')
ax[0].set_xlabel('Simulation Steps', fontsize=12)
ax[0].set_ylabel('Mutation Fraction (I)', fontsize=12)
ax[0].legend(fontsize=11)
ax[0].grid(True, ls=':')
ax[0].set_ylim(0.0, 1.0)

ax[1].plot(steps_theory, mean_act_emp, 'k-', lw=3, label='Simulation Mean')
ax[1].plot(steps_theory, ode_acts, 'r--', lw=3, label='Master Equation ODE')
ax[1].set_xlabel('Simulation Steps', fontsize=12)
ax[1].set_ylabel('Activation Level (I)', fontsize=12)
ax[1].legend(fontsize=11)
ax[1].grid(True, ls=':')
ax[1].set_ylim(0.0, 10.0)

plt.suptitle('Mutator (I) Dynamics: Liquid Simulations vs. Diploid Master Equation', fontsize=14, fontweight='bold')
plt.savefig('gene_evolution_analysis/comparison_theory_individual.png', dpi=200, bbox_inches='tight')
print("Saved figure to gene_evolution_analysis/comparison_theory_individual.png")
