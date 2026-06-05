import numpy as np
import scipy.linalg as la

# Parameters
N_I = 10
N_HK = 10
dmu = 2.3e-2
r_mean = 0.3  # division rate of cancer cells when O is mostly mutated (rmax = 0.3)

# State representation: (n1, n2)
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

# The transition rate matrix W for the population vector p is:
# dp_i / dt = r_mean * ( sum_j T_ij p_j - p_i * sum_k T_ki p_j )
# Wait, this is equivalent to:
# W = r_mean * (T - I)
# where I is the identity matrix, and then we normalize the vector at each step to sum to 1.
# Or we can write the ODE as:
# dp/dt = W p - phi p where phi = sum(W p)
# Let's solve this numerically.

# Initial condition: all cells in (n1=0, n2=1)
p = np.zeros(num_states)
p[state_to_idx[(0, 1)]] = 1.0

# Run ODE simulation
dt = 0.05
steps_sim = 491
time_final = steps_sim * r_mean
nsteps = int(time_final / dt)

print(f"Running ODE simulation for {steps_sim} steps (final time = {time_final:.2f}, dt = {dt})")

time_points = [0.0]
mean_n2_points = [1.0]
mean_mut_fraction_points = [1.0 / N_I]

for step in range(nsteps):
    # dp/dt = r_mean * (T p) - phi * p
    # In the Moran process, the rate of change is:
    # dp_i / dt = r_mean * (sum_j T_ij p_j - p_i * sum_j T_ji p_j)  - wait, T_ji sum is the survival probability of state j.
    # Let's calculate:
    survival_probs = np.array([(1.0 - (state[1] * dmu)**2)**N_HK for state in states])
    # The growth term for state i is:
    # birth of daughter into state i from j: T_ij p_j
    # plus mother staying in state i: p_i * survival_prob_i (if we divide, mother survives, but wait: in Moran, the mother stays. 
    # Let's use the exact growth rate:
    # For a state j, division rate is r_mean. 
    # It produces a daughter in state i with rate r_mean * T_ij.
    # The mother stays in state j (so no net change for mother, except if j=i).
    # Also, at each Moran step, a cell is replaced. The replacement is random from the population, which acts as a dilution term -phi * p_i.
    # So:
    # dp_i / dt = r_mean * sum_j T_ij p_j - phi * p_i
    # To keep sum(p) = 1, we must have:
    # sum(dp_i/dt) = 0 => phi = r_mean * sum_i sum_j T_ij p_j
    
    T_p = T @ p
    phi = r_mean * np.sum(T_p)
    dp = r_mean * T_p - phi * p
    p = p + dp * dt
    # Keep normalized
    p = p / np.sum(p)
    
    time_points.append((step + 1) * dt)
    mean_n2 = np.sum(p * [state[1] for state in states])
    mean_mut = np.sum(p * [state[0] + 2*state[1] for state in states]) / (2.0 * N_I)
    mean_n2_points.append(mean_n2)
    mean_mut_fraction_points.append(mean_mut)

print("\n--- Simulation vs Theory comparison at key steps ---")
print(f"{'Sim Step':<10} | {'Time':<6} | {'Mean n2 (Active)':<18} | {'Mean Mut Fraction':<18}")
print("-" * 60)
for step_check in [50, 100, 200, 300, 400, 491]:
    t_check = step_check * r_mean
    idx_check = int(t_check / dt)
    if idx_check < len(mean_n2_points):
        print(f"{step_check:<10} | {t_check:<6.1f} | {mean_n2_points[idx_check]:<18.4f} | {mean_mut_fraction_points[idx_check]:<18.4f}")

# Compare with asymptotic limit
eigenvalues, eigenvectors = la.eig(T)
principal_idx = np.argmax(np.real(eigenvalues))
principal_vec = np.real(eigenvectors[:, principal_idx])
principal_vec = principal_vec / np.sum(principal_vec)
asymp_n2 = np.sum(principal_vec * [state[1] for state in states])
asymp_mut = np.sum(principal_vec * [state[0] + 2*state[1] for state in states]) / (2.0 * N_I)
print("-" * 60)
print(f"{'Asymptotic':<10} | {'inf':<6} | {asymp_n2:<18.4f} | {asymp_mut:<18.4f}")
