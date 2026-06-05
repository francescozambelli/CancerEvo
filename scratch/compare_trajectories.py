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
def mean_aligned(vs):
    arr = np.full((len(vs), max_len), np.nan)
    for idx, v in enumerate(vs):
        arr[idx, :len(v)] = v
    # compute mean ignoring nan
    return np.nanmean(arr, axis=0)

mean_mut_emp = mean_aligned(empirical_muts)
mean_act_emp = mean_aligned(empirical_acts)
mean_r_emp = mean_aligned(empirical_rs)

# Print some characteristics of the empirical trajectories
print(f"Empirical mean final step: {max_len}")
print(f"Empirical final mean mut_I: {mean_mut_emp[~np.isnan(mean_mut_emp)][-1]:.4f}")
print(f"Empirical final mean act_I: {mean_act_emp[~np.isnan(mean_act_emp)][-1]:.4f}")

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

# 3. Simulate the ODE using the empirical division rate r(t)
# In the simulation, each step is storing interval n_it_store = 100 steps.
# Wait! Let's check n_it_store in parameters_liquid.jl.
# In parameters_liquid.jl: n_it_store = 100.
# So each element in the saved vectors corresponds to 100 simulation steps!
# Let's check: if each step corresponds to 100 simulation steps, then the total number of simulation steps is:
# step_index * 100.
# So if the length of the vector is 5, it means 500 simulation steps.
# The division rate stored in f["r"] is the average at that store step.
# In each simulation step, the total number of divisions is Poisson(sum(r)).
# So for a single cell, the number of divisions per simulation step is r_cell.
# For 100 simulation steps, a cell divides 100 * r_cell times.
# Let's check: in the ODE, the time step corresponding to 100 simulation steps with division rate r is:
# delta_t = 100 * r.
# Let's run the ODE using this variable step size!

p = np.zeros(num_states)
p[state_to_idx[(0, 1)]] = 1.0  # Seed has 1 active (n2=1) and 0 heterozygous (n1=0)

ode_muts = [1.0 / N_I]
ode_acts = [1.0]

valid_len = len(mean_mut_emp[~np.isnan(mean_mut_emp)])

for i in range(1, valid_len):
    # Get the average division rate in this interval
    r_val = mean_r_emp[i-1]
    # Time step in division units for 100 simulation steps
    dt_div = 100.0 * r_val
    
    # We integrate the ODE from t=0 to t=dt_div using small substeps
    n_substeps = 100
    sub_dt = dt_div / n_substeps
    
    for _ in range(n_substeps):
        T_p = T @ p
        # Dilution factor to keep sum(p) = 1
        phi = np.sum(T_p)
        dp = T_p - phi * p
        p = p + dp * sub_dt
        p = p / np.sum(p)
    
    mean_n2 = np.sum(p * [state[1] for state in states])
    mean_mut = np.sum(p * [state[0] + 2*state[1] for state in states]) / (2.0 * N_I)
    
    ode_muts.append(mean_mut)
    ode_acts.append(mean_n2)

print("\nComparison at the final step:")
print(f"Simulation final step (aligned): {valid_len}")
print(f"Simulation mut_I                : {mean_mut_emp[valid_len-1]:.4f}")
print(f"ODE mut_I                       : {ode_muts[-1]:.4f}")
print(f"Simulation act_I                : {mean_act_emp[valid_len-1]:.4f}")
print(f"ODE act_I                       : {ode_acts[-1]:.4f}")

# Plot and save comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
steps = np.arange(valid_len) * 100

ax[0].plot(steps, mean_mut_emp[:valid_len], 'k-', label='Simulation (Liquid)')
ax[0].plot(steps, ode_muts, 'r--', label='Master Equation ODE')
ax[0].set_xlabel('Simulation Steps')
ax[0].set_ylabel('Mutation Fraction (I)')
ax[0].legend()
ax[0].grid(True, ls=':')

ax[1].plot(steps, mean_act_emp[:valid_len], 'k-', label='Simulation (Liquid)')
ax[1].plot(steps, ode_acts, 'r--', label='Master Equation ODE')
ax[1].set_xlabel('Simulation Steps')
ax[1].set_ylabel('Activation Level (I)')
ax[1].legend()
ax[1].grid(True, ls=':')

plt.suptitle('Comparison: Liquid Simulation vs. Analytical Master Equation', fontsize=14)
plt.savefig('gene_evolution_analysis/comparison_theory.png', dpi=200, bbox_inches='tight')
print("\nSaved comparison plot to gene_evolution_analysis/comparison_theory.png")
