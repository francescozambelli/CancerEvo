import pandas as pd
from pathlib import Path

base_dir = Path("/home/francesco/Universita/PhD/PROJECTS/CancerEvo/data/simulations_liquid")
for folder in base_dir.iterdir():
    if folder.is_dir() and folder.name.startswith("ensemble_results_"):
        csv_path = folder / "ensemble_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            npz_count = len(list(folder.glob("*.npz")))
            print(f"Folder: {folder.name}")
            print(f"  NPZ files: {npz_count}")
            print(f"  CSV rows: {len(df)}")
            print("  CSV Head:")
            print(df.head(2))
            print("  Outcome counts:")
            print(df["outcome"].value_counts())
            print("-" * 40)
