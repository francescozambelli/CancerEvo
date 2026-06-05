import numpy as np
from pathlib import Path

sim_path = Path("/home/francesco/Universita/PhD/PROJECTS/CancerEvo/data/simulations_liquid/ensemble_results_D/sim_1.npz")
with np.load(sim_path) as data:
    for k in sorted(data.files):
        print(f"Key: {k}")
        print(f"  Shape: {data[k].shape}")
        print(f"  Type: {data[k].dtype}")
        if data[k].ndim == 1:
            print(f"  Sample values (first 5): {data[k][:5]}")
            print(f"  Sample values (last 5): {data[k][-5:]}")
        else:
            print(f"  Dimensions: {data[k].ndim}")
            print(f"  Sample row:\n{data[k][:2]}")
        print("-" * 30)
