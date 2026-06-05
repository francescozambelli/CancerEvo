"""
loaders.py
----------
Data-loading utilities for the CancerEvo project.

New data format
~~~~~~~~~~~~~~~
Trajectories are stored as `.npz` files in
``data/simulations/<ensemble_dir>/sim_<id>.npz``.

Each file contains 1-D arrays indexed by time step:
  - ``tumor_density``   : fraction of tumor cells
  - ``death_density``   : fraction of dead cells
  - ``r``               : mean reproduction rate
  - ``mu``              : mean mutation rate
  - ``m``               : mean chromosome number deviation
  - ``n_chrs``          : mean number of chromosomes per cell
  - ``mut_I``           : fraction of mutated I (mutator) genes
  - ``mut_O``           : fraction of mutated O (oncogene) genes
  - ``mut_S``           : fraction of mutated S (suppressor) genes
  - ``mut_HK``          : fraction of mutated HK (housekeeping) genes
  - ``mut_M``           : fraction of mutated M genes
  - ``act_I``           : activation level of I genes
  - ``act_O``           : activation level of O genes
  - ``act_S``           : activation level of S genes
  - ``act_HK``          : activation level of HK genes
  - ``act_M``           : activation level of M genes
  - ``outcome_code``    : scalar  (0 = Health, 1 = Tumor)

Ensemble summary CSVs
~~~~~~~~~~~~~~~~~~~~~
``data/simulations/<ensemble_dir>/ensemble_results.csv`` with columns:
  ``sim_id``, ``outcome``, ``steps``, ``final_size``

Ploidy mapping
~~~~~~~~~~~~~~
+------------------------------+-----------+
| Directory                    | Label     |
+==============================+===========+
| ensemble_results             | Diploid   |
| ensemble_results_2CHR        | Diploid   |
| ensemble_results_2CHR_recomb | Aneuploid |
| ensemble_results_4CHR        | Polyploid |
+------------------------------+-----------+
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIM_DIR = _REPO_ROOT / "data" / "simulations"
_SIM_DIR_LIQUID = _REPO_ROOT / "data" / "simulations_liquid"

ENSEMBLE_DIRS: Dict[str, str] = {
    "Diploid":   "ensemble_results_D",
    "Aneuploid": "ensemble_results_A",
    "Polyploid": "ensemble_results_P",
}

# Default (largest) ensemble
DEFAULT_ENSEMBLE = "ensemble_results"


# ---------------------------------------------------------------------------
# Low-level loader
# ---------------------------------------------------------------------------

def load_sim(
    sim_id: int,
    ensemble_dir: str = DEFAULT_ENSEMBLE,
    *,
    sim_dir: Optional[Path] = None,
) -> dict:
    """Load a single simulation NPZ and return it as a plain dict of arrays."""
    base_dir = sim_dir if sim_dir is not None else _SIM_DIR
    path = base_dir / ensemble_dir / f"sim_{sim_id}.npz"
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def load_ensemble_csv(
    ensemble_dir: str = DEFAULT_ENSEMBLE,
    *,
    sim_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the ensemble summary CSV for *ensemble_dir*."""
    base_dir = sim_dir if sim_dir is not None else _SIM_DIR
    path = base_dir / ensemble_dir / "ensemble_results.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def load_ensemble(
    ensemble_dir: str = DEFAULT_ENSEMBLE,
    *,
    outcome_filter: Optional[str] = None,
    max_sims: Optional[int] = None,
    sim_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Load ensemble summary + all trajectory NPZ files.

    Parameters
    ----------
    ensemble_dir:
        Subdirectory of the simulation directory.
    outcome_filter:
        If given (e.g. ``"Tumor"`` or ``"Health"``), only return rows with
        that outcome.
    max_sims:
        Limit the number of simulations loaded (useful for quick tests).
    sim_dir:
        Base simulation directory (default is solid simulations).

    Returns
    -------
    summary : pd.DataFrame
    trajs   : list[dict]  – parallel to summary rows
    """
    summary = load_ensemble_csv(ensemble_dir, sim_dir=sim_dir)
    if outcome_filter is not None:
        if outcome_filter == "Tumor":
            summary = summary[summary["outcome"] != "Health"].reset_index(drop=True)
        elif outcome_filter == "Health":
            summary = summary[summary["outcome"] == "Health"].reset_index(drop=True)
        else:
            summary = summary[summary["outcome"] == outcome_filter].reset_index(drop=True)
    if max_sims is not None:
        summary = summary.iloc[:max_sims]

    trajs = []
    for sid in summary["sim_id"]:
        trajs.append(load_sim(sid, ensemble_dir, sim_dir=sim_dir))

    return summary, trajs


def load_ensemble_liquid(
    ensemble_dir: str = ENSEMBLE_DIRS["Diploid"],
    *,
    outcome_filter: Optional[str] = None,
    max_sims: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Load a specific liquid simulation ensemble (summary + trajectory NPZ files)."""
    return load_ensemble(
        ensemble_dir,
        outcome_filter=outcome_filter,
        max_sims=max_sims,
        sim_dir=_SIM_DIR_LIQUID,
    )


def load_all_ploidy(
    *,
    outcome_filter: Optional[str] = None,
    max_sims: Optional[int] = None,
) -> Dict[str, Tuple[pd.DataFrame, List[dict]]]:
    """
    Load data for all three ploidy conditions (solid simulations).

    Returns
    -------
    dict with keys ``"Diploid"``, ``"Aneuploid"``, ``"Polyploid"``,
    each value is ``(summary_df, trajs_list)``.
    """
    result = {}
    for label, edir in ENSEMBLE_DIRS.items():
        result[label] = load_ensemble(
            edir,
            outcome_filter=outcome_filter,
            max_sims=max_sims,
        )
    return result


def load_all_ploidy_liquid(
    *,
    outcome_filter: Optional[str] = None,
    max_sims: Optional[int] = None,
) -> Dict[str, Tuple[pd.DataFrame, List[dict]]]:
    """
    Load data for all three ploidy conditions (liquid simulations).

    Returns
    -------
    dict with keys ``"Diploid"``, ``"Aneuploid"``, ``"Polyploid"``,
    each value is ``(summary_df, trajs_list)``.
    """
    result = {}
    for label, edir in ENSEMBLE_DIRS.items():
        result[label] = load_ensemble(
            edir,
            outcome_filter=outcome_filter,
            max_sims=max_sims,
            sim_dir=_SIM_DIR_LIQUID,
        )
    return result


def extract_field(trajs: List[dict], field: str) -> List[np.ndarray]:
    """Extract a named field from every trajectory as a list of 1-D arrays."""
    return [t[field] for t in trajs]


# ---------------------------------------------------------------------------
# Stability results
# ---------------------------------------------------------------------------

_R0 = 0.15  # reference reproduction rate used to normalise rmax


def load_stability_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stability_results, stability_results_1) data frames."""
    df0 = pd.read_csv(_REPO_ROOT / "data" / "stability_results.csv")
    df1 = pd.read_csv(_REPO_ROOT / "data" / "stability_results_1.csv")
    df0["rmax_norm"] = df0["rmax"] / _R0
    df1["rmax_norm"] = df1["rmax"] / _R0
    return df0, df1


def load_adaptive_stability_results() -> pd.DataFrame:
    """
    Load ``data/stability_results_adaptive.csv`` produced by the adaptive
    sweep script.  Returns an empty DataFrame if the file does not yet exist.

    Adds a ``rmax_norm`` column (rmax / r0).
    """
    path = _REPO_ROOT / "data" / "stability_results_adaptive.csv"
    if not path.exists():
        import warnings
        warnings.warn(
            f"{path.name} not found – run scripts/stability_sweep.jl first.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["rmax", "stable_dmu", "rmax_norm"])
    df = pd.read_csv(path)
    df["rmax_norm"] = df["rmax"] / _R0
    return df


def load_all_stability_results() -> pd.DataFrame:
    """
    Merge all stability CSV files (prior + adaptive) into one DataFrame.

    Each row carries a ``source`` column identifying its origin:
    ``"sweep_1"``, ``"sweep_2"``, or ``"adaptive"``.

    Adds a ``rmax_norm`` column (rmax / r0).
    """
    df0, df1 = load_stability_results()
    df0["source"] = "sweep_1"
    df1["source"] = "sweep_2"

    dfa = load_adaptive_stability_results()
    if not dfa.empty:
        dfa["source"] = "adaptive"

    frames = [df for df in [df0, df1, dfa] if not df.empty]
    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values(["rmax", "stable_dmu"], inplace=True, ignore_index=True)
    return merged


def load_external_tumor_mu() -> List[np.ndarray]:
    """
    Load mutation rate trajectories for 'Tumor' runs from the external data directory.
    """
    external_dir = Path("/data/UNIVERSITA/PhD/PROJECTS/Data/CancerEvo")
    mu_path = external_dir / "0ch_mu.txt"
    state_path = external_dir / "0ch_state.txt"

    data = []
    with open(mu_path) as f:
        for line in f:
            data.append(np.array([float(n) for n in line.strip().split(", ")]))

    states = np.loadtxt(state_path, dtype=str)
    tumor_indices = np.where(states == "Tumor")[0]
    return [data[i] for i in tumor_indices]

