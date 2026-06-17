#!/usr/bin/env python3
"""
calculate_asymptotic_theory.py
------------------------------
A script to calculate the asymptotic (stationary) subpopulation distributions 
and the expected asymptotic mutation rate, based on the analytical Master 
Equation solution derived in outputs/ansymptotic_behavior.md.

Usage:
  python scripts/calculate_asymptotic_theory.py --N_I 10 --N_H 10 --dmu 0.023
"""

import argparse
import sys
import json
import numpy as np
import scipy.linalg as la
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compute_asymptotic_limit(N_I: int, N_H: int, dmu: float, remove_lower: int = 0):
    """
    Computes the asymptotic subpopulation fractions and mutation rate
    using both the analytical recursive product formulas and a numerical 
    eigenvalue/eigenvector solver on the bidiagonal transition matrix.
    Optionally, removes the lower X classes of mutation.

    Parameters
    ----------
    N_I : int
        Number of instability genes.
    N_H : int
        Number of housekeeping genes.
    dmu : float
        Mutation rate step (delta mu).
    remove_lower : int
        Number of lower mutation classes to remove (default: 0).

    Returns
    -------
    dict
        A dictionary containing the computed results.
    """
    if remove_lower < 0 or remove_lower >= N_I:
        raise ValueError(f"remove_lower must be in [0, {N_I - 1}]")
        
    c = remove_lower + 1  # 1-based index of the first remaining class
    num_remaining = N_I - remove_lower
    
    # 1. Initialize probability vectors (1-indexed for alignment with paper formulas)
    p_d = np.zeros(N_I + 1)
    p_mu = np.zeros(N_I + 1)
    
    for i in range(1, N_I + 1):
        # p_d,i = 1 - (1 - i * dmu)^N_H
        p_d[i] = 1.0 - (1.0 - i * dmu) ** N_H
        
        # p_mu,i = 1 - (1 - i * dmu)^(N_I - i)
        if i < N_I:
            p_mu[i] = 1.0 - (1.0 - i * dmu) ** (N_I - i)
        else:
            p_mu[i] = 0.0  # No transition out of the final instability class

    # 2. Analytical calculation using the product formulas in the paper starting from class c
    prod_terms = np.zeros(num_remaining)
    prod_terms[0] = 1.0
    denom_c = p_d[c] + p_mu[c]
    
    for i in range(c + 1, N_I + 1):
        term = 1.0
        for j in range(c + 1, i + 1):
            denom_j = p_d[j] + p_mu[j]
            term *= p_mu[j-1] / (denom_j - denom_c)
        prod_terms[i - c] = term
        
    x_c_analytical = 1.0 / np.sum(prod_terms)
    x_analytical = x_c_analytical * prod_terms

    # 3. Numerical validation: Find the principal eigenvector of A_reduced
    A = np.zeros((num_remaining, num_remaining))
    for k in range(num_remaining):
        i = k + c
        A[k, k] = 1.0 - p_d[i] - p_mu[i]
        if k < num_remaining - 1:
            A[k + 1, k] = p_mu[i]
            
    eigenvalues, eigenvectors = la.eig(A)
    # The principal eigenvector is associated with the largest eigenvalue.
    # For a bidiagonal matrix, the eigenvalues are the diagonal entries.
    # The largest is the first diagonal entry: A_0,0 = 1 - p_d,c - p_mu,c
    principal_idx = np.argmax(np.real(eigenvalues))
    x_numerical = np.real(eigenvectors[:, principal_idx])
    
    # Ensure positive sign alignment and normalize
    if np.sum(x_numerical) < 0:
        x_numerical = -x_numerical
    x_numerical = x_numerical / np.sum(x_numerical)

    # 4. Embed into the full class structure (1 to N_I), setting removed classes to 0.0
    full_classes = np.arange(1, N_I + 1)
    full_class_mus = full_classes * dmu
    
    x_analytical_full = np.zeros(N_I)
    x_numerical_full = np.zeros(N_I)
    x_analytical_full[remove_lower:] = x_analytical
    x_numerical_full[remove_lower:] = x_numerical
    
    asymp_level_analytical = np.sum(x_analytical_full * full_classes)
    asymp_level_numerical = np.sum(x_numerical_full * full_classes)
    
    asymp_mu_analytical = np.sum(x_analytical_full * full_class_mus)
    asymp_mu_numerical = np.sum(x_numerical_full * full_class_mus)
    
    # Pad p_d and p_mu with zeros for removed classes to keep array lengths consistent
    p_d_full = np.zeros(N_I)
    p_mu_full = np.zeros(N_I)
    p_d_full[remove_lower:] = p_d[c:]
    p_mu_full[remove_lower:] = p_mu[c:]
    
    max_diff = np.max(np.abs(x_analytical - x_numerical))
    
    return {
        "N_I": N_I,
        "N_H": N_H,
        "dmu": dmu,
        "remove_lower": remove_lower,
        "classes": full_classes.tolist(),
        "class_mus": full_class_mus.tolist(),
        "p_d": p_d_full.tolist(),
        "p_mu": p_mu_full.tolist(),
        "x_analytical": x_analytical_full.tolist(),
        "x_numerical": x_numerical_full.tolist(),
        "asymp_level_analytical": float(asymp_level_analytical),
        "asymp_level_numerical": float(asymp_level_numerical),
        "asymp_mu_analytical": float(asymp_mu_analytical),
        "asymp_mu_numerical": float(asymp_mu_numerical),
        "asymp_pd_analytical": float(np.sum(x_analytical_full * p_d[1:])),
        "asymp_pd_numerical": float(np.sum(x_numerical_full * p_d[1:])),
        "max_diff": float(max_diff),
        "eigenvalues": np.real(eigenvalues).tolist(),
        "principal_eigenval": float(np.real(eigenvalues[principal_idx]))
    }


def generate_plot(res: dict, plot_path: Path):
    """
    Generates a publication-quality plot of the asymptotic subpopulation distribution.
    """
    import matplotlib.pyplot as plt
    
    # Plotting styles
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    classes = np.array(res["classes"])
    x_val = np.array(res["x_analytical"])
    
    # Draw bar chart of subpopulation densities
    bars = ax.bar(classes, x_val, color="#1f77b4", alpha=0.8, edgecolor="black", 
                  linewidth=1.2, width=0.6, label=r"Analytical $x_i^*$")
    
    # Draw a line plot as a guide/comparison
    ax.plot(classes, x_val, color="#ff7f0e", marker="o", ls="--", lw=1.5, 
            markersize=6, label="Trendline")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0.01:
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
            
    # Labels and titles
    ax.set_xlabel(r"Instability Class ($i$)", fontsize=12)
    ax.set_ylabel(r"Stationary Fraction ($x_i^*$)", fontsize=12)
    ax.set_xticks(classes)
    ax.set_ylim(0, max(x_val) * 1.15)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, ls=":", alpha=0.5)
    
    # Text annotation with summary parameters
    info_text = (
        f"$N_{{\\mathcal{{I}}}} = {res['N_I']}$\n"
        f"$N_{{\\mathcal{{H}}}} = {res['N_H']}$\n"
        f"$\\Delta\\mu = {res['dmu']:.4f}$\n"
    )
    if res["remove_lower"] > 0:
        info_text += f"Removed classes: {res['remove_lower']}\n"
    info_text += f"$\\langle i\\rangle^* \\approx {res['asymp_level_analytical']:.2f}$\n"
    info_text += f"$\\langle\\mu\\rangle^* \\approx {res['asymp_mu_analytical']:.4f}$"
    
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8))
            
    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.72))
    
    plt.tight_layout()
    
    # Dual PNG/SVG export
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(plot_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(plot_path.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved plots to:\n  - {plot_path.with_suffix('.png')}\n  - {plot_path.with_suffix('.svg')}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate asymptotic distributions and expected mutation rate."
    )
    parser.add_argument("-i", "--N_I", type=int, default=10, help="Number of instability genes (default: 10)")
    parser.add_argument("--N_H", type=int, default=10, help="Number of housekeeping genes (default: 10)")
    parser.add_argument("-d", "--dmu", type=float, default=0.023, help="Mutation rate step size dmu (default: 0.023)")
    parser.add_argument("-r", "--remove_lower", type=int, default=0, help="Number of lower mutation classes to remove (default: 0)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to save output table (markdown/json/csv)")
    parser.add_argument("-p", "--plot", type=str, default=None, help="Prefix/path to save distribution plot (saves .png and .svg)")
    parser.add_argument("--json", action="store_true", help="Print result summary as JSON to stdout")
    
    args = parser.parse_args()
    
    # Run calculation
    res = compute_asymptotic_limit(args.N_I, args.N_H, args.dmu, args.remove_lower)
    
    # Print basic output
    if args.json:
        print(json.dumps(res, indent=2))
        return
        
    print("=" * 60)
    print("      ASYMPTOTIC MASTER EQUATION ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Instability Genes (N_I)    : {res['N_I']}")
    print(f"Housekeeping Genes (N_H)   : {res['N_H']}")
    print(f"Mutation Step Size (dmu)   : {res['dmu']:.5f}")
    if res["remove_lower"] > 0:
        print(f"Removed Lower Classes      : {res['remove_lower']}")
    print(f"Principal Eigenvalue       : {res['principal_eigenval']:.6f}")
    print("-" * 60)
    print(f"Asymptotic Mutation Level  : {res['asymp_level_analytical']:.4f}")
    print(f"Asymptotic Mutation Rate   : {res['asymp_mu_analytical']:.6f}")
    print(f"Asymptotic Death Prob      : {res['asymp_pd_analytical']:.6f}")
    print(f"Max Analytical-Numerical Diff: {res['max_diff']:.2e}")
    print("-" * 60)
    
    # Create results table
    table_lines = []
    table_lines.append(f"| Class (i) | Cell Mu (i*dmu) | Fraction x_i* (Analytical) | Fraction x_i* (Numerical) | Death Prob (p_d,i) | Transition Prob (p_mu,i) |")
    table_lines.append(f"|:---------:|:---------------:|:--------------------------:|:-------------------------:|:------------------:|:------------------------:|")
    
    for idx in range(len(res["classes"])):
        i = res["classes"][idx]
        mu_val = res["class_mus"][idx]
        x_a = res["x_analytical"][idx]
        x_n = res["x_numerical"][idx]
        pd_val = res["p_d"][idx]
        p_mu_val = res["p_mu"][idx]
        table_lines.append(
            f"| {i:9d} | {mu_val:15.5f} | {x_a:26.6f} | {x_n:25.6f} | {pd_val:18.6f} | {p_mu_val:24.6f} |"
        )
        
    for line in table_lines:
        print(line)
        
    print("=" * 60)
    
    # Handle optional file saving
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if out_path.suffix == ".json":
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"Saved JSON results to {out_path}")
        else:
            # Default to Markdown output
            with open(out_path, "w") as f:
                f.write(f"# Asymptotic Master Equation Analysis Results\n\n")
                f.write(f"## Parameter Setup\n")
                f.write(f"- **Number of Instability Genes ($N_\\mathcal{{I}}$)**: {res['N_I']}\n")
                f.write(f"- **Number of Housekeeping Genes ($N_\\mathcal{{H}}$)**: {res['N_H']}\n")
                f.write(f"- **Mutation Step Size ($\\Delta\\mu$)**: {res['dmu']}\n")
                if res["remove_lower"] > 0:
                    f.write(f"- **Removed Lower Mutation Classes**: {res['remove_lower']}\n")
                f.write(f"\n## Summary Metrics\n")
                f.write(f"- **Asymptotic Mutation Level ($\\langle i\\rangle^*$)**: {res['asymp_level_analytical']:.4f}\n")
                f.write(f"- **Asymptotic Mutation Rate ($\\langle\\mu\\rangle^*$)**: {res['asymp_mu_analytical']:.6f}\n")
                f.write(f"- **Asymptotic Death Probability ($\\langle P_\\text{{death}}\\rangle^*$)**: {res['asymp_pd_analytical']:.6f}\n")
                f.write(f"- **Principal Eigenvalue ($\\lambda_{{\\max}}$)**: {res['principal_eigenval']:.6f}\n")
                f.write(f"- **Max Analytical vs Numerical Deviation**: {res['max_diff']:.2e}\n\n")
                f.write(f"## Subpopulation Distribution Table\n\n")
                for line in table_lines:
                    f.write(line + "\n")
            print(f"Saved Markdown report to {out_path}")
            
    # Handle optional plotting
    if args.plot:
        generate_plot(res, Path(args.plot))


if __name__ == "__main__":
    main()
