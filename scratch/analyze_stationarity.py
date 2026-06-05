import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load summary CSV
csv_path = Path("data/simulations_liquid/ensemble_results_D/ensemble_results.csv")
df = pd.read_csv(csv_path)

# Filter for Tumor_Max outcomes
tumor_df = df[df["outcome"] == "Tumor_Max"]
print(f"Total tumor simulations: {len(tumor_df)}")

durations = []
final_mus = []
final_acts = []

for _, row in tumor_df.iterrows():
    sid = int(row["sim_id"])
    steps = int(row["steps"])
    sim_path = Path("data/simulations_liquid/ensemble_results_D") / f"sim_{sid}.npz"
    with np.load(sim_path) as f:
        # mu is stored at every step
        mu_traj = f["mu"]
        act_traj = f["act_I"]
        durations.append(steps)
        final_mus.append(mu_traj[-1])
        final_acts.append(act_traj[-1])

durations = np.array(durations)
final_mus = np.array(final_mus)
final_acts = np.array(final_acts)

# Print correlation
corr_mu = np.corrcoef(durations, final_mus)[0, 1]
corr_act = np.corrcoef(durations, final_acts)[0, 1]
print(f"Correlation between duration and final mu: {corr_mu:.4f}")
print(f"Correlation between duration and final activation level: {corr_act:.4f}")

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Scatter plot
ax[0].scatter(durations, final_mus, alpha=0.6, color="blue", edgecolors="k")
ax[0].axhline(0.054, color="red", ls="--", label="Theoretical Asymptote (0.054)")
ax[0].set_xlabel("Simulation Steps (Duration)")
ax[0].set_ylabel("Final Mutation Rate (mu)")
ax[0].set_title("Final Mutation Rate vs. Duration")
ax[0].legend()
ax[0].grid(True, ls=":")

# Plot 2: Selected individual trajectories to illustrate the two regimes
# Let's find some short duration and some long duration runs
short_idx = np.where(durations < 450)[0][:5]
long_idx = np.where(durations > 800)[0][:5]

for idx in short_idx:
    sid = int(tumor_df.iloc[idx]["sim_id"])
    sim_path = Path("data/simulations_liquid/ensemble_results_D") / f"sim_{sid}.npz"
    with np.load(sim_path) as f:
        ax[1].plot(f["mu"], color="orange", alpha=0.7, label="Short-lived (rapid progression)" if idx == short_idx[0] else "")

for idx in long_idx:
    sid = int(tumor_df.iloc[idx]["sim_id"])
    sim_path = Path("data/simulations_liquid/ensemble_results_D") / f"sim_{sid}.npz"
    with np.load(sim_path) as f:
        ax[1].plot(f["mu"], color="green", alpha=0.7, label="Long-lived (delayed progression)" if idx == long_idx[0] else "")

ax[1].axhline(0.054, color="red", ls="--", label="Theoretical Asymptote (0.054)")
ax[1].set_xlabel("Simulation Steps")
ax[1].set_ylabel("Mutation Rate (mu)")
ax[1].set_title("Temporal Trajectories of Mutation Rate")
ax[1].legend()
ax[1].grid(True, ls=":")

plt.tight_layout()
output_fig = Path("artifacts/analyze_stationarity.png")
output_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_fig, dpi=150, bbox_inches="tight")
print(f"Saved analysis plot to {output_fig}")
