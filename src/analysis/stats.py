"""
stats.py
--------
Statistical helper functions for CancerEvo trajectory analysis.

All functions operate on *lists of 1-D numpy arrays* of possibly
different lengths (one array per simulation run).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Element-wise statistics across uneven sequences
# ---------------------------------------------------------------------------

def stats_elementwise(
    vs: List[np.ndarray],
    q_lo: float = 0.159,
    q_hi: float = 0.841,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute element-wise median and 68 % credible interval across sequences
    of possibly different lengths (shorter sequences simply contribute fewer
    points).

    Parameters
    ----------
    vs    : list of 1-D arrays
    q_lo  : lower quantile (default = 16th ≈ −1σ)
    q_hi  : upper quantile (default = 84th ≈ +1σ)

    Returns
    -------
    medians : (N,) array – element-wise median
    lo      : (N,) array – lower quantile
    hi      : (N,) array – upper quantile
    """
    n = max(len(v) for v in vs)
    medians = np.full(n, np.nan)
    lo      = np.full(n, np.nan)
    hi      = np.full(n, np.nan)

    for i in range(n):
        vals = [v[i] for v in vs if i < len(v) and not np.isnan(v[i])]
        if vals:
            arr = np.array(vals, dtype=float)
            medians[i] = np.quantile(arr, 0.5)
            lo[i]      = np.quantile(arr, q_lo)
            hi[i]      = np.quantile(arr, q_hi)

    return medians, lo, hi


def plot_stats_elementwise(ax, vs, color="C0", lw=2, alpha=0.25, label=""):
    """
    Plot element-wise median ± 1σ band on *ax*.

    Parameters
    ----------
    ax     : matplotlib Axes
    vs     : list of 1-D arrays
    color  : line / fill colour
    lw     : line width
    alpha  : fill transparency
    label  : legend label
    """
    medians, lo, hi = stats_elementwise(vs)
    t = np.arange(len(medians))
    ax.plot(t, medians, color=color, lw=lw, label=label)
    ax.fill_between(t, lo, hi, color=color, alpha=alpha)
    return medians, lo, hi


# ---------------------------------------------------------------------------
# Running slope (discrete derivative)
# ---------------------------------------------------------------------------

def running_slope(x: np.ndarray, skip: int = 50) -> np.ndarray:
    """
    Compute a centred running slope of *x* with half-window *skip*.

    Returns an array of length ``len(x) - 2*skip - 1``.
    """
    vec = []
    for i in range(skip, len(x) - skip - 1):
        vec.append((x[i + skip] - x[i]) / skip)
    return np.array(vec)


# ---------------------------------------------------------------------------
# Phase-diagram helpers
# ---------------------------------------------------------------------------

def dyn_state(r: float, r0: float, mu: float, k: float, N: int, exponent: float = 1.0):
    """
    Map a cell state (r, mu, k) to phase-diagram coordinates.

    Parameters
    ----------
    r   : current reproduction rate
    r0  : baseline reproduction rate
    mu  : current mutation rate
    k   : fraction of HK genes mutated
    N   : number of HK genes

    Returns
    -------
    (r/r0, p_death)  where
    p_death = 1 - (1-mu)^(2kN) * (1-mu^2)^(N*(1-2k))
    """
    x = r / r0
    p = (1.0 - mu) ** (2 * k * N)# * (1.0 - mu ** 2) ** (N * (1.0 - 2 * k))) ** exponent
    return x, p
