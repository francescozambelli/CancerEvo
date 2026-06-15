# src/analysis/compute_stability_line.py
"""
Compute the theoretical stability boundary (dmu*) and corresponding survival probability (P_s*)
for the liquid tumor model starting from the Unified Growth Foothold Inequality:

    r_max * (2 * alpha_c - 1) > r_0

where:
    c = remove_lower + 1
    alpha_c = (1 - c * dmu)^(N_HK + N_I - c)

Setting to equality at the phase boundary:
    r_max * (2 * alpha_c - 1) = r_0
    2 * alpha_c - 1 = r_0 / r_max
    alpha_c = 0.5 * (1.0 + r_0 / r_max)
    (1 - c * dmu*)^(N_HK + N_I - c) = 0.5 * (1.0 + r_0 / r_max)
    dmu* = (1 / c) * [ 1.0 - (0.5 * (1.0 + r_0 / r_max))^(1 / (N_HK + N_I - c)) ]

Usage:
  python src/analysis/compute_stability_line.py --rmax 0.30
  python src/analysis/compute_stability_line.py --rmax-norm 2.0 --remove-lower 1
  python src/analysis/compute_stability_line.py --save-csv outputs/liquid_stability_theory.csv --rmax-min 1.0 --rmax-max 7.0 --steps 100
"""

import argparse
import numpy as np
import pandas as pd
import sys

def get_critical_dmu(rmax: float, r0: float, N_I: int, N_HK: int, remove_lower: int = 0) -> float:
    """
    Compute the analytical critical locus mutation rate increment (dmu*) 
    by solving the Unified Growth Foothold Inequality for class c = remove_lower + 1:
        r_max * (2 * alpha_c - 1) = r_0
    """
    if rmax <= r0:
        return 0.0
    c = remove_lower + 1
    exponent = N_HK + N_I - c
    if exponent <= 0:
        return 0.0
    val = 0.5 * (1.0 + r0 / rmax)
    if val >= 1.0 or val <= 0.0:
        return 0.0
    return (1.0 / c) * (1.0 - (val ** (1.0 / exponent)))

def get_critical_survival(dmu_star: float, N_HK: int, remove_lower: int = 0) -> float:
    """
    Compute the corresponding survival probability P_s* at the critical boundary.
    For class c, P_s = (1 - c * dmu*)^N_HK.
    """
    c = remove_lower + 1
    base = 1.0 - c * dmu_star
    if base <= 0.0:
        return 0.0
    return base ** N_HK

def main():
    parser = argparse.ArgumentParser(description="Compute theoretical stability boundary line.")
    parser.add_argument("--rmax", type=float, default=None, help="Maximum division rate (rmax)")
    parser.add_argument("--rmax-norm", type=float, default=None, help="Normalized division rate (rmax / r0)")
    parser.add_argument("--r0", type=float, default=0.15, help="Baseline division rate r0 (default: 0.15)")
    parser.add_argument("--N-I", type=int, default=10, help="Number of instability genes N_I (default: 10)")
    parser.add_argument("--N-HK", type=int, default=10, help="Number of housekeeping genes N_HK (default: 10)")
    parser.add_argument("--remove-lower", type=int, default=0, help="Number of lower classes removed (default: 0)")
    parser.add_argument("--save-csv", type=str, default=None, help="Path to save theoretical curve as CSV")
    parser.add_argument("--rmax-min", type=float, default=1.0, help="Minimum normalized division rate for CSV sweep")
    parser.add_argument("--rmax-max", type=float, default=7.0, help="Maximum normalized division rate for CSV sweep")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps for CSV sweep")

    args = parser.parse_args()

    # If single rmax value requested
    if args.rmax is not None or args.rmax_norm is not None:
        if args.rmax is not None:
            rmax = args.rmax
            rmax_norm = rmax / args.r0
        else:
            rmax_norm = args.rmax_norm
            rmax = rmax_norm * args.r0

        dmu_star = get_critical_dmu(rmax, args.r0, args.N_I, args.N_HK, args.remove_lower)
        ps_star = get_critical_survival(dmu_star, args.N_HK, args.remove_lower)

        print(f"Parameters:")
        print(f"  r0           = {args.r0:.4f}")
        print(f"  N_I          = {args.N_I}")
        print(f"  N_HK         = {args.N_HK}")
        print(f"  remove_lower = {args.remove_lower} (founding class c = {args.remove_lower + 1})")
        print(f"Input:")
        print(f"  rmax         = {rmax:.4f} (rmax/r0 = {rmax_norm:.4f})")
        print(f"Results:")
        print(f"  Critical dmu* = {dmu_star:.6f}")
        print(f"  Survival P_s* = {ps_star:.6f}")
        return

    # If CSV saving requested or no args provided (default to sweep print)
    rmax_norm_vals = np.linspace(args.rmax_min, args.rmax_max, args.steps)
    rows = []
    for rn in rmax_norm_vals:
        rmax = rn * args.r0
        dmu_star = get_critical_dmu(rmax, args.r0, args.N_I, args.N_HK, args.remove_lower)
        ps_star = get_critical_survival(dmu_star, args.N_HK, args.remove_lower)
        rows.append({
            "rmax_norm": rn,
            "rmax": rmax,
            "dmu_star": dmu_star,
            "P_s_star": ps_star
        })

    df = pd.DataFrame(rows)

    if args.save_csv is not None:
        df.to_csv(args.save_csv, index=False)
        print(f"Saved theoretical stability line to {args.save_csv}")
    else:
        print(f"Founding class c = {args.remove_lower + 1}")
        print(f"{'rmax/r0':<10} | {'rmax':<8} | {'dmu*':<9} | {'P_s*':<9}")
        print("-" * 45)
        for _, row in df.iloc[::max(1, args.steps // 15)].iterrows():
            print(f"{row['rmax_norm']:<10.4f} | {row['rmax']:<8.4f} | {row['dmu_star']:<9.6f} | {row['P_s_star']:<9.6f}")

if __name__ == "__main__":
    main()
