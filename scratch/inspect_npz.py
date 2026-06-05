import numpy as np
from pathlib import Path

data_dir = Path("/home/francesco/Universita/PhD/PROJECTS/CancerEvo/data/simulations_liquid")
for folder in data_dir.iterdir():
    if folder.is_dir():
        npz_files = list(folder.glob("*.npz"))
        if npz_files:
            print(f"Folder: {folder.name}")
            print(f"Sample file: {npz_files[0].name}")
            with np.load(npz_files[0]) as data:
                print("Keys:", data.files)
                for k in list(data.files)[:10]:
                    print(f"  {k}: shape {data[k].shape}, type {data[k].dtype}")
            break
