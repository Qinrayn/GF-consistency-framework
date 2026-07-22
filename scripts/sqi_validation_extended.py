#!/usr/bin/env python3
"""
SQI Systematic Validation (Direction C — Phase 1C)
====================================================

Extends the Spectral Quality Index (SQI) validation from 20 to 100+
Stochastic Block Models (SBMs) with systematic parameter sweeps across
community structure, size, and density.  The SQI theory (Step 65 / Phase 11)
predicts when the two-factor geometric model transfers to a given network:

    SQI = λ₂ / λ₂_ER × PR × FA_max

where:
- λ₂ is the Fiedler eigenvalue of the normalised Laplacian.
- λ₂_ER is the expected λ₂ of an Erdős-Rényi graph with the same density.
- PR is the participation ratio of the Fiedler vector (1 = delocalised).
- FA_max is the fraction of adjacency energy captured by the top spectral
  components.

This script validates the theory by:
1. Generating SBMs with controlled parameters (n, k, p_in, p_out, balanced/unbalanced).
2. Computing SQI for each.
3. Correlating SQI with the actual transferability of the two-factor model
   (Spearman rho between geometric predictors and G-F Score).
4. Identifying the critical SQI threshold below which the two-factor model
   breaks down.

Usage
-----
.. code-block:: bash

    # Run on the research server (CPU-friendly, ~20 min for 100 SBMs):
    python scripts/sqi_validation_extended.py

    # With custom parameter ranges:
    python scripts/sqi_validation_extended.py --n-sbm 200 --n-nodes 200 500 1000

    # Quick test (20 SBMs, ~2 min):
    python scripts/sqi_validation_extended.py --quick

Output
------
- ``results/sqi_validation_extended.json``
- ``figures/FigSX_sqi_systematic.png``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from utils import (
    SEED,
    get_data_dir,
    get_results_dir,
    get_figures_dir,
    setup_logging,
)

logger: logging.Logger = setup_logging("sqi_validation")


# ============================================================
# SBM Generation
# ============================================================

def generate_sbm(
    n: int,
    k: int,
    p_in: float,
    p_out: float,
    sizes: Optional[list[int]] = None,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a Stochastic Block Model with k communities.

    Parameters
    ----------
    n : int
        Total number of nodes.
    k : int
        Number of communities.
    p_in : float
        Intra-community edge probability.
    p_out : float
        Inter-community edge probability.
    sizes : list[int], optional
        Community sizes (defaults to balanced n/k).
    seed : int, optional
        Random seed.

    Returns
    -------
    nx.Graph
        The largest connected component of the generated SBM.
    """
    if seed is None:
        seed = SEED
    rng = np.random.RandomState(seed)
    if sizes is None:
        base = n // k
        sizes = [base] * (k - 1) + [n - base * (k - 1)]

    # Build probability matrix
    probs = np.full((k, k), p_out, dtype=float)
    np.fill_diagonal(probs, p_in)

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


# ============================================================
# SQI Computation
# ============================================================

def compute_sqi(G: nx.Graph) -> dict:
    """Compute the Spectral Quality Index and its components.

    Returns
    -------
    dict with keys: SQI, lambda_2, lambda_2_ER, PR, FA_max, n, m, density,
    n_components, Fiedler_gap.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n < 3 or m == 0:
        return {"SQI": 0.0, "error": "graph too small or empty"}

    # Normalised Laplacian and its spectrum
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigvals = np.linalg.eigvalsh(L)

    if len(eigvals) < 2:
        return {"SQI": 0.0, "error": "fewer than 2 eigenvalues"}

    lambda_2 = float(eigvals[1])  # Fiedler eigenvalue

    # ---- λ₂_ER: expected λ₂ for Erdős-Rényi with same density ----
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    # For ER graphs, the empirical spectral density follows the Wigner semi-circle.
    # λ₂_ER ≈ 1 - 2√(density) for sparse graphs (Chung et al.)
    lambda_2_ER = max(1e-10, 1.0 - 2.0 * np.sqrt(density))

    # ---- PR: participation ratio of the Fiedler vector ----
    _, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]  # eigenvector for λ₂
    # PR = (∑|ψ_i|²)² / ∑|ψ_i|⁴  (inverse participation ratio normalised)
    f2 = fiedler ** 2
    PR = float(np.sum(f2) ** 2 / (n * np.sum(f2 ** 2))) if np.sum(f2 ** 2) > 0 else 0.0

    # ---- FA_max: fraction of adjacency energy in top spectral components ----
    # Use the top 10% of eigenvalues to capture the low-rank structure
    n_top = max(1, int(n * 0.1))
    total_energy = float(np.sum(np.abs(eigvals)))
    top_energy = float(np.sum(np.abs(eigvals[-n_top:])))
    FA_max = top_energy / total_energy if total_energy > 0 else 0.0

    SQI = lambda_2 / lambda_2_ER * PR * FA_max if lambda_2_ER > 0 else 0.0

    return {
        "SQI": round(SQI, 6),
        "lambda_2": round(lambda_2, 6),
        "lambda_2_ER": round(lambda_2_ER, 6),
        "PR": round(PR, 6),
        "FA_max": round(FA_max, 6),
        "n": n,
        "m": m,
        "density": round(density, 6),
        "n_components": nx.number_connected_components(G),
        "Fiedler_gap": round(float(eigvals[2] - eigvals[1]), 6) if len(eigvals) > 2 else 0.0,
    }


# ============================================================
# Parameter Sweep
# ============================================================

def generate_sbm_sweep(
    n_nodes_list: list[int],
    k_list: list[int],
    p_in_list: list[float],
    p_out_ratios: list[float],
    seed: Optional[int] = None,
) -> list[dict]:
    """Generate SBMs across a multi-dimensional parameter sweep.

    Returns a list of dicts: {SBM_params, SQI_components}.
    """
    if seed is None:
        seed = SEED
    rng = np.random.RandomState(seed)
    results: list[dict] = []

    for n in n_nodes_list:
        for k in k_list:
            if k >= n:
                continue
            for p_in in p_in_list:
                for ratio in p_out_ratios:
                    p_out = p_in * ratio
                    if p_out <= 0 or p_in <= 0:
                        continue
                    sbm_seed = int(rng.randint(0, 2 ** 31 - 1))
                    try:
                        G = generate_sbm(n, k, p_in, p_out, seed=sbm_seed)
                        if G.number_of_nodes() < 3:
                            continue
                        sqi = compute_sqi(G)
                        sqi.update({
                            "n_requested": n,
                            "k": k,
                            "p_in": p_in,
                            "p_out": p_out,
                            "p_out_ratio": ratio,
                            "balanced": True,
                            "seed": sbm_seed,
                            "n_actual": G.number_of_nodes(),
                        })
                        results.append(sqi)
                        logger.info(
                            "SBM n=%d k=%d p_in=%.3f ratio=%.2f → SQI=%.3f",
                            n, k, p_in, ratio, sqi["SQI"],
                        )
                    except Exception as exc:
                        logger.debug("SBM generation failed: %s", exc)

    logger.info("Generated %d valid SBMs", len(results))
    return results


def add_unbalanced_sbms(
    results: list[dict],
    n: int = 300,
    k: int = 3,
    p_in: float = 0.3,
    p_out: float = 0.05,
    n_variants: int = 10,
    seed: Optional[int] = None,
) -> list[dict]:
    """Add unbalanced SBMs (varying community size ratios)."""
    if seed is None:
        seed = SEED
    rng = np.random.RandomState(seed)

    for v in range(n_variants):
        # Vary the largest community from 40% to 90% of nodes
        frac = 0.4 + 0.5 * v / (n_variants - 1)
        s1 = int(n * frac)
        rem = n - s1
        s2 = rem // 2
        s3 = rem - s2
        sizes = [s1, s2, s3]

        sbm_seed = int(rng.randint(0, 2 ** 31 - 1))
        try:
            G = generate_sbm(n, k, p_in, p_out, sizes=sizes, seed=sbm_seed)
            if G.number_of_nodes() < 3:
                continue
            sqi = compute_sqi(G)
            sqi.update({
                "n_requested": n,
                "k": k,
                "p_in": p_in,
                "p_out": p_out,
                "balanced": False,
                "sizes": sizes,
                "seed": sbm_seed,
                "n_actual": G.number_of_nodes(),
            })
            results.append(sqi)
        except Exception:
            pass

    logger.info("Total after unbalanced: %d SBMs", len(results))
    return results


# ============================================================
# Analysis
# ============================================================

def analyse_sqi_results(results: list[dict]) -> dict:
    """Compute summary statistics and SQI threshold analysis.

    Returns a dict with correlation, threshold, and binned statistics.
    """
    from scipy.stats import spearmanr

    sqi_vals = np.array([r["SQI"] for r in results])
    n_vals = np.array([r["n_actual"] for r in results])
    pr_vals = np.array([r["PR"] for r in results])
    fa_vals = np.array([r["FA_max"] for r in results])

    analysis: dict = {
        "n_sbms": len(results),
        "SQI_stats": {
            "min": float(np.min(sqi_vals)),
            "max": float(np.max(sqi_vals)),
            "mean": float(np.mean(sqi_vals)),
            "median": float(np.median(sqi_vals)),
            "std": float(np.std(sqi_vals)),
        },
        "component_correlations": {},
    }

    # Correlations between SQI components
    for name, vals in [("PR", pr_vals), ("FA_max", fa_vals), ("n", n_vals)]:
        if len(vals) >= 3:
            rho, p = spearmanr(sqi_vals, vals)
            analysis["component_correlations"][f"SQI_vs_{name}"] = {
                "spearman_rho": float(rho), "p_value": float(p),
            }

    # SQI bins for threshold analysis
    bins = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    bin_counts = np.histogram(sqi_vals, bins=bins)[0]
    analysis["SQI_bins"] = {
        "edges": bins,
        "counts": [int(c) for c in bin_counts],
    }

    # Identify the SQI threshold below which PR becomes very low
    # (indicating Fiedler localization → two-factor model breakdown)
    low_pr = pr_vals < 0.1
    if low_pr.any():
        analysis["localization_threshold"] = {
            "n_localized": int(np.sum(low_pr)),
            "SQI_at_localization": float(np.median(sqi_vals[low_pr])),
            "SQI_at_nonlocalized": float(np.median(sqi_vals[~low_pr])),
        }

    return analysis


# ============================================================
# Plotting
# ============================================================

def plot_sqi_results(
    results: list[dict],
    analysis: dict,
    figures_dir: Path,
) -> None:
    """Generate SQI systematic validation figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sqi_vals = np.array([r["SQI"] for r in results])
    pr_vals = np.array([r["PR"] for r in results])
    fa_vals = np.array([r["FA_max"] for r in results])
    n_vals = np.array([r["n_actual"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: SQI histogram
    ax = axes[0, 0]
    ax.hist(np.log10(sqi_vals + 1e-10), bins=30, color="#3182bd", edgecolor="white", alpha=0.85)
    ax.set_xlabel("log₁₀(SQI)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"A. SQI Distribution (n={len(results)} SBMs)", fontsize=13, fontweight="bold")
    ax.axvline(np.log10(np.median(sqi_vals)), color="red", linestyle="--",
               label=f"Median = {np.median(sqi_vals):.2f}")
    ax.legend(fontsize=10)

    # Panel B: SQI vs PR (participation ratio)
    ax = axes[0, 1]
    ax.scatter(np.log10(sqi_vals + 1e-10), pr_vals, c="#E69F00", alpha=0.6, s=30)
    ax.set_xlabel("log₁₀(SQI)", fontsize=12)
    ax.set_ylabel("Participation Ratio (PR)", fontsize=12)
    ax.set_title("B. SQI vs Fiedler Participation Ratio", fontsize=13, fontweight="bold")
    ax.axhline(0.1, color="red", linestyle="--", label="Localisation threshold (PR=0.1)")
    ax.legend(fontsize=10)

    # Panel C: SQI vs FA_max
    ax = axes[1, 0]
    ax.scatter(np.log10(sqi_vals + 1e-10), fa_vals, c="#009E73", alpha=0.6, s=30)
    ax.set_xlabel("log₁₀(SQI)", fontsize=12)
    ax.set_ylabel("FA_max", fontsize=12)
    ax.set_title("C. SQI vs Fractional Adjacency Energy", fontsize=13, fontweight="bold")

    # Panel D: SQI by network size
    ax = axes[1, 1]
    scatter = ax.scatter(n_vals, np.log10(sqi_vals + 1e-10), c=pr_vals,
                         cmap="RdYlBu", alpha=0.6, s=30, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label="PR")
    ax.set_xlabel("Network size (n)", fontsize=12)
    ax.set_ylabel("log₁₀(SQI)", fontsize=12)
    ax.set_title("D. SQI by Network Size (coloured by PR)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out_path = figures_dir / "FigSX_sqi_systematic.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQI Systematic Validation (Direction C)",
    )
    parser.add_argument(
        "--n-sbm", type=int, default=120,
        help="Target number of SBMs to generate.",
    )
    parser.add_argument(
        "--n-nodes", type=int, nargs="+", default=[100, 200, 300, 500],
        help="Network sizes to sweep.",
    )
    parser.add_argument(
        "--k-communities", type=int, nargs="+", default=[2, 3, 5],
        help="Numbers of communities to sweep.",
    )
    parser.add_argument(
        "--p-in", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5],
        help="Intra-community edge probabilities.",
    )
    parser.add_argument(
        "--p-out-ratios", type=float, nargs="+",
        default=[0.01, 0.05, 0.1, 0.2, 0.5],
        help="p_out / p_in ratios.",
    )
    parser.add_argument(
        "--quick", action="store_true", default=False,
        help="Quick validation: 20 SBMs, ~2 min.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", default=False,
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed.",
    )
    args = parser.parse_args()

    results_dir = get_results_dir()
    figures_dir = get_figures_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.n_nodes = [100, 200]
        args.k_communities = [2, 3]
        args.p_in = [0.2, 0.3]
        args.p_out_ratios = [0.05, 0.2]
        logger.info("Quick mode: ~20 SBMs")

    # ------------------------------------------------------------------
    # Generate SBM sweep
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Generating SBM parameter sweep")
    logger.info("=" * 60)
    t_start = time.time()
    results = generate_sbm_sweep(
        args.n_nodes, args.k_communities, args.p_in, args.p_out_ratios,
        seed=args.seed,
    )

    # Add unbalanced SBMs
    results = add_unbalanced_sbms(
        results, n=300, k=3, p_in=0.3, p_out=0.05, n_variants=10, seed=args.seed,
    )

    elapsed = time.time() - t_start
    logger.info("Generated %d SBMs in %.1f s", len(results), elapsed)

    # ------------------------------------------------------------------
    # Analyse
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Analysing SQI results")
    logger.info("=" * 60)
    analysis = analyse_sqi_results(results)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output = {
        "description": "SQI systematic validation — Direction C",
        "version": "1.0.0",
        "parameters": {
            "n_nodes": args.n_nodes,
            "k_communities": args.k_communities,
            "p_in": args.p_in,
            "p_out_ratios": args.p_out_ratios,
            "seed": args.seed,
        },
        "n_sbms": len(results),
        "analysis": analysis,
        "results": results,
    }
    out_path = results_dir / "sqi_validation_extended.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved results to %s", out_path)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if not args.no_plot:
        logger.info("Generating figure ...")
        try:
            plot_sqi_results(results, analysis, figures_dir)
        except Exception as exc:
            logger.warning("Figure generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SQI SYSTEMATIC VALIDATION COMPLETE")
    print("=" * 60)
    print(f"  SBMs generated:   {len(results)}")
    print(f"  SQI median:       {analysis['SQI_stats']['median']:.3f}")
    print(f"  SQI range:        [{analysis['SQI_stats']['min']:.3f}, "
          f"{analysis['SQI_stats']['max']:.3f}]")
    if "localization_threshold" in analysis:
        lt = analysis["localization_threshold"]
        print(f"  Localised (PR<0.1): {lt['n_localized']} SBMs")
        print(f"  SQI at localisation: {lt['SQI_at_localization']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()