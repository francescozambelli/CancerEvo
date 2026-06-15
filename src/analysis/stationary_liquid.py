"""
stationary_liquid.py
--------------------
Analytical stationary-distribution solver for the liquid-tumor Moran model.

Implements the sequential quadratic equations derived in
``outputs/liquid_ODE.tex`` (§1.3: Stationary State Distributions).

The key equations solved are:

    A_k (f_k*)^2 - B_k f_k* - Q_{k-1}* = 0

with:
    A_k = r_max / 2
    B_k = r_max phi_k^{≠k} alpha_k - r_max(delta_k + gamma_k)
          - sum_{j≠k} r_max f_j beta_cc - r0 f_W beta_wc
    Q_{k-1} = r_max f_{k-1} gamma_{k-1} (1 + phi_{k-1})

Solution: f_k* = (B_k + sqrt(B_k^2 + 4 A_k Q_{k-1})) / (2 A_k)

Since phi_k, f_W, p_D all depend on {f_k*}, we solve self-consistently
via fixed-point iteration.

Usage
-----
    from src.analysis.stationary_liquid import solve_stationary

    result = solve_stationary(N_I=10, N_HK=10, dmu=0.023, r0=0.15, rmax=0.30)
    print(result["f_k"])       # stationary cancer fractions
    print(result["f_W"])       # stationary wild-type fraction
    print(result["p_D"])       # stationary dead fraction
    print(result["invadable"]) # whether the tumor can invade
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.optimize import root, root_scalar


# ═══════════════════════════════════════════════════════════════════════
# Microscopic probability functions
# ═══════════════════════════════════════════════════════════════════════

import math

def survival_prob(k: int, dmu: float, N_HK: int) -> float:
    """P_{s,k} = (1 - k*dmu)^{N_HK}."""
    base = 1.0 - k * dmu
    if base <= 0:
        return 0.0
    return base ** N_HK


def alpha(k: int, dmu: float, N_HK: int, N_I: int) -> float:
    """Faithful cloning probability: alpha_k = (1 - k*dmu)^{N_HK + N_I - k}."""
    base = 1.0 - k * dmu
    if base <= 0:
        return 0.0
    return base ** (N_HK + N_I - k)


def transition_prob(j_from: int, k_to: int, dmu: float, N_HK: int, N_I: int) -> float:
    """Probability of transition from class j_from to class k_to."""
    if k_to < j_from:
        return 0.0
    mu_j = j_from * dmu
    if mu_j >= 1.0:
        return 0.0
    P_s = (1.0 - mu_j) ** N_HK
    steps = k_to - j_from
    binom = math.comb(N_I - j_from, steps)
    p_mut = binom * (mu_j ** steps) * ((1.0 - mu_j) ** (N_I - k_to))
    return P_s * p_mut


def gamma(k: int, dmu: float, N_HK: int, N_I: int) -> float:
    """Promotion probability: gamma_k = (N_I - k) * k * dmu * alpha_{k+1}."""
    if k >= N_I:
        return 0.0
    return (N_I - k) * k * dmu * alpha(k + 1, dmu, N_HK, N_I)


def delta(k: int, dmu: float, N_HK: int) -> float:
    """Lethal mutation probability: delta_k = 1 - P_{s,k}."""
    return 1.0 - survival_prob(k, dmu, N_HK)


def beta_cw(rmax: float, r0: float) -> float:
    """Moran win prob: cancer vs wild-type."""
    return rmax / (rmax + r0)


def beta_cc() -> float:
    """Moran win prob: cancer vs cancer (symmetric)."""
    return 0.5


def beta_wc(rmax: float, r0: float) -> float:
    """Moran win prob: wild-type vs cancer."""
    return r0 / (rmax + r0)


# ═══════════════════════════════════════════════════════════════════════
# Vectorised probability arrays
# ═══════════════════════════════════════════════════════════════════════

def compute_probabilities(
    N_I: int, N_HK: int, dmu: float
) -> Dict[str, np.ndarray]:
    """
    Compute all class-dependent probabilities for k = 1, ..., N_I.

    Returns
    -------
    dict with keys 'alpha', 'gamma', 'delta', 'P_s' — each (N_I,) arrays
    indexed so that arr[i] corresponds to class k = i + 1.
    """
    classes = np.arange(1, N_I + 1)
    alpha_arr = np.array([alpha(k, dmu, N_HK, N_I) for k in classes])
    gamma_arr = np.array([gamma(k, dmu, N_HK, N_I) for k in classes])
    delta_arr = np.array([delta(k, dmu, N_HK) for k in classes])
    Ps_arr = np.array([survival_prob(k, dmu, N_HK) for k in classes])

    # Precompute full transition matrix
    P_matrix = np.zeros((N_I, N_I))
    for r_idx in range(N_I):
        for c_idx in range(N_I):
            P_matrix[r_idx, c_idx] = transition_prob(r_idx + 1, c_idx + 1, dmu, N_HK, N_I)

    return {
        "alpha": alpha_arr,
        "gamma": gamma_arr,
        "delta": delta_arr,
        "P_s": Ps_arr,
        "P_matrix": P_matrix,
        "classes": classes,
    }


# ═══════════════════════════════════════════════════════════════════════
# Growth foothold inequality
# ═══════════════════════════════════════════════════════════════════════

def growth_foothold(
    N_I: int, N_HK: int, dmu: float, r0: float, rmax: float, remove_lower: int = 0
) -> Dict[str, float]:
    """
    Evaluate the exact unified growth foothold inequality for the founding class c = remove_lower + 1:

        rmax * (2 * alpha_c - 1) > r0

    Returns
    -------
    dict with 'lhs', 'rhs', 'margin' (lhs - rhs), and 'invadable' (bool).
    """
    c = remove_lower + 1
    ac = alpha(c, dmu, N_HK, N_I)
    gc = gamma(c, dmu, N_HK, N_I)
    dc = delta(c, dmu, N_HK)

    lhs = rmax * (2.0 * ac - 1.0)
    rhs = r0
    return {
        "lhs": lhs,
        "rhs": rhs,
        "margin": lhs - rhs,
        "invadable": lhs > rhs,
        "alpha_c": ac,
        "gamma_c": gc,
        "delta_c": dc,
        "c": c,
    }


# ═══════════════════════════════════════════════════════════════════════
# Stationary distribution solver
# ═══════════════════════════════════════════════════════════════════════

def solve_stationary(
    N_I: int,
    N_HK: int,
    dmu: float,
    r0: float,
    rmax: float,
    remove_lower: int = 0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> Dict:
    """
    Self-consistent hybrid 2D/1D root solver for the liquid-tumor stationary state.

    First checks invasion condition. If invadable, finds the root of the 2D
    residual system for (f_W, p_D) using an inner self-consistent loop to compute
    the cancer class profile. Handles the wild-type extinction regime (f_W = 0)
    using a 1D root finder for p_D.

    Parameters
    ----------
    N_I         : number of instability genes (= max cancer classes)
    N_HK        : number of housekeeping genes
    dmu         : per-locus mutation rate increment
    r0          : wild-type division rate
    rmax        : cancer division rate
    remove_lower: number of lower cancer classes to assume absent (default: 0)
    tol         : convergence tolerance (passed to scipy root finder)
    max_iter    : maximum iterations for the inner self-consistent f_k profile loop

    Returns
    -------
    dict with keys:
        'f_k'       : (N_I,) array of stationary cancer class fractions
        'f_W'       : stationary wild-type fraction
        'p_D'       : stationary dead fraction
        'f_cancer'  : total cancer fraction sum(f_k)
        'mean_class' : average cancer class <k>
        'mean_mu'    : average chromosomal mutation rate <mu>
        'invadable' : whether the tumor can invade the healthy state
        'converged' : whether the solver converged successfully
        'regime'    : 'Coexistence', 'WT Extinction', or 'None'
        'probs'     : dict of microscopic probabilities
        'foothold'  : growth foothold inequality results
    """
    # --- Check invasion condition ---
    foothold = growth_foothold(N_I, N_HK, dmu, r0, rmax, remove_lower)
    probs = compute_probabilities(N_I, N_HK, dmu)

    alpha_arr = probs["alpha"]
    gamma_arr = probs["gamma"]
    delta_arr = probs["delta"]
    P_matrix = probs["P_matrix"]

    b_cw = beta_cw(rmax, r0)
    b_cc = beta_cc()
    b_wc = beta_wc(rmax, r0)

    if not foothold["invadable"]:
        dmu_star = find_critical_dmu(N_I, N_HK, r0, rmax, remove_lower)
        # Healthy fixed point is stable — no tumor
        return {
            "f_k": np.zeros(N_I),
            "f_W": 1.0,
            "p_D": 0.0,
            "f_cancer": 0.0,
            "mean_class": 0.0,
            "mean_mu": 0.0,
            "invadable": False,
            "converged": True,
            "regime": "None",
            "probs": probs,
            "foothold": foothold,
            "dmu_star": dmu_star,
        }

    # Inner helper to solve for f_k profile given f_W, p_D
    def get_self_consistent_f_k(f_W_val: float, p_D_val: float) -> np.ndarray:
        f_k = np.zeros(N_I)
        f_k[remove_lower:] = 0.05
        
        for _ in range(max_iter):
            f_k_new = np.zeros(N_I)
            f_C_temp = np.sum(f_k)
            phi_val = p_D_val + f_W_val * b_cw + f_C_temp * b_cc
            
            for i in range(N_I):
                if i < remove_lower:
                    continue
                
                # Exact A_k = rmax * (1.0 - alpha_k)
                A_k = rmax * (1.0 - alpha_arr[i])
                
                # phi excluding self-displacement
                phi_neq_k = phi_val - f_k[i] * b_cc
                
                # B_k = rmax * phi_neq_k * (2 * alpha_k - 1) - sum_{j!=k} rmax * f_j * b_cc - r0 * f_W * b_wc
                sum_other_cc = f_C_temp * b_cc - f_k[i] * b_cc
                B_k = (rmax * phi_neq_k * (2.0 * alpha_arr[i] - 1.0)
                       - sum_other_cc * rmax
                       - r0 * f_W_val * b_wc)
                
                # Influx from all lower classes j < i
                Q_prev = 0.0
                for j in range(i):
                    Q_prev += 2.0 * rmax * f_k_new[j] * phi_val * P_matrix[j, i]
                    
                disc = B_k**2 + 4.0 * A_k * Q_prev
                if disc < 0:
                    f_k_new[i] = max(0.0, B_k / (2.0 * A_k))
                else:
                    f_k_new[i] = (B_k + np.sqrt(disc)) / (2.0 * A_k)
                f_k_new[i] = max(0.0, f_k_new[i])
                
            f_k = 0.5 * f_k + 0.5 * f_k_new
            
        return f_k

    # 2D Residuals for Coexistence solver
    def residuals_2d(x):
        f_W_val, p_D_val = x
        f_W_val = max(0.0, min(1.0, f_W_val))
        p_D_val = max(0.0, min(1.0, p_D_val))
        
        f_k_val = get_self_consistent_f_k(f_W_val, p_D_val)
        f_C_val = np.sum(f_k_val)
        phi_val = p_D_val + f_W_val * b_cw + f_C_val * b_cc
        
        res1 = 1.0 - f_W_val - p_D_val - f_C_val
        res2 = 2.0 * rmax * phi_val * np.sum(f_k_val * delta_arr) - p_D_val * (r0 * f_W_val + rmax * f_C_val)
        return [res1, res2]

    # 1D Residual for WT Extinction solver (f_W = 0)
    def residual_1d(p_D_val: float) -> float:
        f_k_val = get_self_consistent_f_k(0.0, p_D_val)
        f_C_val = np.sum(f_k_val)
        return 1.0 - p_D_val - f_C_val

    # 1. Attempt to solve 2D system (Coexistence)
    sol2d = root(residuals_2d, [0.3, 0.3], method="hybr", tol=tol)
    f_W_opt, p_D_opt = sol2d.x

    if sol2d.success and f_W_opt >= -1e-6:
        f_W_opt = max(0.0, f_W_opt)
        p_D_opt = max(0.0, p_D_opt)
        f_k_opt = get_self_consistent_f_k(f_W_opt, p_D_opt)
        regime = "Coexistence"
        converged = True
    else:
        # 2. WT is extinct, solve 1D system for p_D
        sol1d = root_scalar(residual_1d, bracket=[0.0, 1.0], method="brentq", xtol=tol)
        f_W_opt = 0.0
        p_D_opt = sol1d.root
        f_k_opt = get_self_consistent_f_k(0.0, p_D_opt)
        regime = "WT Extinction"
        converged = sol1d.converged

    f_cancer = np.sum(f_k_opt)
    classes = np.arange(1, N_I + 1)
    mean_class = np.sum(f_k_opt * classes) / f_cancer if f_cancer > 0 else 0.0
    mean_mu = mean_class * dmu
    dmu_star = find_critical_dmu(N_I, N_HK, r0, rmax, remove_lower)

    return {
        "f_k": f_k_opt,
        "f_W": f_W_opt,
        "p_D": p_D_opt,
        "f_cancer": f_cancer,
        "mean_class": mean_class,
        "mean_mu": mean_mu,
        "invadable": True,
        "converged": converged,
        "regime": regime,
        "probs": probs,
        "foothold": foothold,
        "dmu_star": dmu_star,
    }


# ═══════════════════════════════════════════════════════════════════════
# Critical mutation rate finder
# ═══════════════════════════════════════════════════════════════════════

def find_critical_dmu(
    N_I: int,
    N_HK: int,
    r0: float,
    rmax: float,
    remove_lower: int = 0,
    dmu_lo: float = 1e-5,
    dmu_hi: float = 0.1,
    tol: float = 1e-8,
) -> float:
    """
    Find the critical delta-mu at which the growth foothold inequality
    transitions from satisfied to violated (bisection on the margin).

    Returns
    -------
    dmu_star : critical mutation rate increment
    """
    for _ in range(200):
        dmu_mid = (dmu_lo + dmu_hi) / 2.0
        fh = growth_foothold(N_I, N_HK, dmu_mid, r0, rmax, remove_lower)
        if fh["invadable"]:
            dmu_lo = dmu_mid
        else:
            dmu_hi = dmu_mid
        if (dmu_hi - dmu_lo) < tol:
            break

    return (dmu_lo + dmu_hi) / 2.0


# ═══════════════════════════════════════════════════════════════════════
# Unified High-Level analysis function for programmatic/notebook imports
# ═══════════════════════════════════════════════════════════════════════

def solve_stationary_system(
    N_I: int,
    N_HK: int,
    dmu: float,
    r0: float,
    rmax: float,
    remove_lower: int = 0,
    tol: float = 1e-10,
    max_iter: int = 100,
    verbose: bool = False,
) -> Dict:
    """
    Solve for the stationary state distribution of the liquid tumor Moran model.

    This function wraps:
      1. Growth foothold check (foothold)
      2. Critical mutation rate calculation (dmu_star)
      3. Self-consistent stationary state solving (f_k, f_W, p_D)

    Parameters
    ----------
    N_I          : number of instability genes (max mutation classes)
    N_HK         : number of housekeeping genes
    dmu          : mutation rate step size (delta mu)
    r0           : wild-type division rate
    rmax         : cancer division rate
    remove_lower : number of lower cancer classes to assume absent (default: 0)
    tol          : tolerance for numerical root-finding
    max_iter     : maximum iterations for inner self-consistent f_k profile loop
    verbose      : if True, prints a complete formatted report to stdout

    Returns
    -------
    dict with keys:
        'f_k'        : (N_I,) array of stationary cancer class fractions
        'f_W'        : stationary wild-type fraction
        'p_D'        : stationary dead fraction
        'f_cancer'   : total cancer fraction sum(f_k)
        'mean_class' : average cancer class <k>
        'mean_mu'    : average chromosomal mutation rate <mu>
        'invadable'  : whether the tumor can invade the healthy state
        'converged'  : whether the solver converged successfully
        'regime'     : 'Coexistence', 'WT Extinction', or 'None'
        'probs'      : dict of microscopic probabilities
        'foothold'   : growth foothold inequality results
        'dmu_star'   : critical mutation rate increment
    """
    res = solve_stationary(
        N_I=N_I,
        N_HK=N_HK,
        dmu=dmu,
        r0=r0,
        rmax=rmax,
        remove_lower=remove_lower,
        tol=tol,
        max_iter=max_iter,
    )

    if verbose:
        fh = res["foothold"]
        print("=" * 60)
        print("  LIQUID TUMOR — STATIONARY STATE ANALYSIS")
        print("=" * 60)
        print(f"  N_I          = {N_I}")
        print(f"  N_HK         = {N_HK}")
        print(f"  dmu          = {dmu:.5f}")
        print(f"  r0           = {r0:.4f}")
        print(f"  rmax         = {rmax:.4f}")
        print(f"  remove-lower = {remove_lower}")
        print("-" * 60)
        print(f"\n  Growth foothold: LHS = {fh['lhs']:.6f}, RHS = {fh['rhs']:.6f}")
        print(f"  Margin = {fh['margin']:.6f}  =>  {'INVADABLE' if fh['invadable'] else 'STABLE (no tumor)'}")
        print(f"  Critical dmu* = {res['dmu_star']:.6f}")
        print(f"  Theoretical mean mu = {res['mean_mu']:.3e}")
        print(f"\n  Converged: {res['converged']} (Regime: {res['regime']})")
        print(f"  f_W  = {res['f_W']:.6f}")
        print(f"  p_D  = {res['p_D']:.6f}")
        print(f"  f_C  = {res['f_cancer']:.6f} (total cancer)")
        if res["f_cancer"] > 0:
            print(f"  <k>  = {res['mean_class']:.4f}")
            print(f"  <mu> = {res['mean_mu']:.6f}")

        print("\n  Class | f_k*        | alpha_k    | gamma_k    | delta_k")
        print("  " + "-" * 56)
        probs = res["probs"]
        for i in range(N_I):
            print(f"  {i+1:5d} | {res['f_k'][i]:11.6f} | "
                  f"{probs['alpha'][i]:10.6f} | "
                  f"{probs['gamma'][i]:10.6f} | "
                  f"{probs['delta'][i]:10.6f}")
        print("=" * 60)

    return res


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute liquid-tumor stationary distributions."
    )
    parser.add_argument("--N_I", type=int, default=10, help="Instability genes")
    parser.add_argument("--N_HK", type=int, default=10, help="Housekeeping genes")
    parser.add_argument("--dmu", type=float, default=0.023, help="Mutation rate step")
    parser.add_argument("--r0", type=float, default=0.15, help="WT division rate")
    parser.add_argument("--rmax", type=float, default=0.30, help="Cancer division rate")
    parser.add_argument("--remove-lower", type=int, default=0, help="Number of lower cancer classes to assume absent")
    args = parser.parse_args()

    solve_stationary_system(
        N_I=args.N_I,
        N_HK=args.N_HK,
        dmu=args.dmu,
        r0=args.r0,
        rmax=args.rmax,
        remove_lower=args.remove_lower,
        verbose=True,
    )


if __name__ == "__main__":
    main()
