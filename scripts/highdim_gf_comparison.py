#!/usr/bin/env python3
"""
highdim_gf_comparison.py
=========================
Systematic high-dimensional G-F Score comparison across all embedding methods.

Addresses Reviewer Major Concern #2: All core G-F rankings were locked to 2D,
potentially biasing the evaluation of GNNs and modern graph learning methods.

This script evaluates all 11 primary methods (8 classical + 3 GNN) at
dimensionality d = {2, 4, 8, 16, 32, 64} on the curated 153-node yeast PPI
network.  For each (method, d) pair it:

  1. Computes the embedding at the specified dimensionality.
  2. Evaluates the G-F curve (200-point grid) on the FIRST 2 principal
     dimensions (for fair comparison with the 2D baseline) and on ALL d
     dimensions (for the full high-dimensional assessment).
  3. Computes G-F Score, plateau width, and peak purity.
  4. Performs statistical comparison: Spearman rank correlation between
     d-dimensional and 2D rankings; per-method dimension-sensitivity score.

Outputs
-------
results/highdim_gf_comparison.json     : numerical results
results/highdim_gf_rankings.json     : rank tables per dimension
results/highdim_gf_sensitivity.json  : dimension-sensitivity scores
figures/FigHD_GF_dimension_gradient.png : publication figure

Usage
-----
    python scripts/highdim_gf_comparison.py

Requires the modified embed_all.py and utils.py (with n_components support).
"""

from __future__ import annotations

import json
import sys
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, R_MIN, R_MAX, N_POINTS, GF_R_MIN, GF_R_MAX, TARGET_STD,
    ALL_CURATED_METHODS, GNN_METHODS, PLATEAU_RELATIVE_THRESHOLD,
    get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
    load_curated_network, compute_centrality_features, save_embedding,
    compute_gf_curve, compute_gf_score, compute_plateau_width,
    rescale_coordinates, precompute_distance_matrix,
    BANNER, METHOD_COLORS,
)
from embed_all import embed_method_by_name

# For GNN embeddings, import the GNN embedder (refactored to support n_components)
from embed_gnn import embed_gnn_method_by_name

# ============================================================
# Configuration
# ============================================================

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()
EMB = get_embeddings_dir()

for d in [RESULTS, FIGURES, EMB]:
    d.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [2, 4, 8, 16, 32, 64]
METHODS = ALL_CURATED_METHODS + GNN_METHODS  # 11 methods

# METHOD_COLORS imported from utils
# BANNER imported from utils


def embed_highdim(G, nodes, method, features, n_components, seed=SEED):
    """Compute embedding at arbitrary dimensionality.

    Parameters
    ----------
    G, nodes, method, features : standard
    n_components : int
        Target dimensionality.
    seed : int

    Returns
    -------
    np.ndarray, shape (n_nodes, n_components)
    """
    random.seed(seed)
    np.random.seed(seed)

    if method in GNN_METHODS:
        return embed_gnn_method_by_name(
            G, nodes, method, features=features,
            latent_dim=n_components, seed=seed
        )
    else:
        return embed_method_by_name(
            G, nodes, method, features=features,
            n_components=n_components
        )


def compute_gf_for_embedding(coords, nodes, go_map, r_vals,
                              gf_r_min=GF_R_MIN, gf_r_max=GF_R_MAX):
    """Evaluate G-F curve and score for a given embedding.

    Parameters
    ----------
    coords : np.ndarray
        Embedding coordinates (may be high-dimensional).
    nodes, go_map, r_vals : standard
    gf_r_min, gf_r_max : float
        Integration interval bounds.

    Returns
    -------
    dict with keys: gf_score, plateau_width, peak_purity, purities, modularity.
    """
    # For high-dimensional coords, we compute G-F on the FULL d-dimensional
    # Euclidean distance.  This is the most principled choice because the
    # community structure in the full space is what the method actually produces.
    # A 2D-projection sub-analysis is also performed for comparison.
    purities, mods = compute_gf_curve(coords, nodes, go_map, r_vals)
    gf = compute_gf_score(r_vals, purities, gf_r_min, gf_r_max)
    plateau = compute_plateau_width(r_vals, purities, PLATEAU_RELATIVE_THRESHOLD)
    return {
        "gf_score": float(gf),
        "plateau_width": float(plateau["W"]),
        "peak_purity": float(plateau["peak_purity"]),
        "purities": [float(p) for p in purities],
        "modularity": [float(m) for m in mods],
    }


def compute_gf_2d_projection(coords, nodes, go_map, r_vals,
                                gf_r_min=GF_R_MIN, gf_r_max=GF_R_MAX):
    """Evaluate G-F on the first 2 principal components (for fair 2D comparison)."""
    if coords.shape[1] <= 2:
        coords_2d = coords
    else:
        # Project to top-2 principal components via SVD
        U, S, Vt = np.linalg.svd(coords - coords.mean(axis=0), full_matrices=False)
        coords_2d = U[:, :2] * S[:2]
    purities, mods = compute_gf_curve(coords_2d, nodes, go_map, r_vals)
    gf = compute_gf_score(r_vals, purities, gf_r_min, gf_r_max)
    plateau = compute_plateau_width(r_vals, purities, PLATEAU_RELATIVE_THRESHOLD)
    return {
        "gf_score": float(gf),
        "plateau_width": float(plateau["W"]),
        "peak_purity": float(plateau["peak_purity"]),
    }


def dimension_sensitivity_score(scores):
    """Quantify how sensitive a method's G-F Score is to dimensionality.

    Returns the ratio of max / min G-F Score across all tested dimensions.
    A ratio close to 1 indicates insensitivity (stable); a large ratio
    indicates the method benefits strongly from higher dimensions.
    """
    if not scores or len(scores) < 2:
        return 1.0
    vals = np.array(list(scores.values()), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2 or vals.min() <= 0:
        return 1.0
    return float(vals.max() / vals.min())


def rank_methods_by_gf(gf_dict):
    """Return a list of (method, gf_score, rank) sorted by G-F Score descending."""
    items = [(m, float(s)) for m, s in gf_dict.items() if np.isfinite(s)]
    items.sort(key=lambda x: x[1], reverse=True)
    return [(m, s, i + 1) for i, (m, s) in enumerate(items)]


def main():
    print(BANNER)
    print("High-Dimensional G-F Score Comparison")
    print(f"Methods: {len(METHODS)} | Dimensions: {DIMENSIONS}")
    print(BANNER)

    # Load data
    print("\n[1/6] Loading network and annotations...")
    G, nodes, go_map = load_curated_network(DATA)
    print(f"      {len(nodes)} nodes, {G.number_of_edges()} edges")

    print("\n[2/6] Computing centrality features...")
    features = compute_centrality_features(G, nodes)

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # Main data structure: results[method][dim] = {gf_score, plateau_width, ...}
    results = {m: {} for m in METHODS}
    results_2dproj = {m: {} for m in METHODS}  # 2D projection for comparison

    # ============================================================
    # Compute embeddings and G-F scores for all (method, dim) pairs
    # ============================================================
    print("\n[3/6] Computing embeddings and G-F scores...")
    total = len(METHODS) * len(DIMENSIONS)
    done = 0

    for method in METHODS:
        for dim in DIMENSIONS:
            done += 1
            t0 = time.time()
            print(f"      [{done}/{total}] {method} d={dim} ...", end=" ", flush=True)

            try:
                coords = embed_highdim(G, nodes, method, features, dim, seed=SEED)
                gf_data = compute_gf_for_embedding(coords, nodes, go_map, r_vals)
                gf_data["dim"] = dim
                results[method][dim] = gf_data

                # Also compute 2D-projection G-F for fair comparison
                gf_2d = compute_gf_2d_projection(coords, nodes, go_map, r_vals)
                results_2dproj[method][dim] = gf_2d

                print(f"GF={gf_data['gf_score']:.4f} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                results[method][dim] = {
                    "gf_score": np.nan, "plateau_width": np.nan,
                    "peak_purity": np.nan, "dim": dim, "error": str(e)
                }
                results_2dproj[method][dim] = {
                    "gf_score": np.nan, "plateau_width": np.nan, "peak_purity": np.nan
                }

    # ============================================================
    # Rank analysis per dimension
    # ============================================================
    print("\n[4/6] Computing rank tables and rank correlations...")

    rank_tables = {}
    rank_correlations = {}

    for dim in DIMENSIONS:
        gf_dict = {m: results[m][dim]["gf_score"] for m in METHODS}
        rank_tables[dim] = rank_methods_by_gf(gf_dict)

    # Correlation between d-dimensional ranking and 2D ranking
    baseline_2d = {m: results[m][2]["gf_score"] for m in METHODS}
    for dim in DIMENSIONS:
        if dim == 2:
            continue
        dim_scores = {m: results[m][dim]["gf_score"] for m in METHODS}
        m_list = [m for m in METHODS if np.isfinite(baseline_2d[m]) and np.isfinite(dim_scores[m])]
        if len(m_list) >= 3:
            x = [baseline_2d[m] for m in m_list]
            y = [dim_scores[m] for m in m_list]
            rho, p = spearmanr(x, y)
            rank_correlations[dim] = {"rho": float(rho), "p": float(p), "n": len(m_list)}
        else:
            rank_correlations[dim] = {"rho": np.nan, "p": np.nan, "n": len(m_list)}

    # ============================================================
    # Dimension sensitivity
    # ============================================================
    print("\n[5/6] Computing dimension sensitivity scores...")
    sensitivity = {}
    for method in METHODS:
        scores = {d: results[method][d]["gf_score"] for d in DIMENSIONS}
        sensitivity[method] = {
            "sensitivity_ratio": dimension_sensitivity_score(scores),
            "scores": {str(d): float(v) if np.isfinite(v) else None for d, v in scores.items()},
            "max_dim": max((d for d in DIMENSIONS if np.isfinite(scores.get(d, np.nan))), default=2),
            "max_gf": max((v for v in scores.values() if np.isfinite(v)), default=0.0),
        }

    # ============================================================
    # Key findings
    # ============================================================
    print("\n" + BANNER)
    print("KEY FINDINGS")
    print(BANNER)

    print("\nRank correlation (2D vs d-dimensional):")
    for dim in DIMENSIONS:
        if dim == 2:
            continue
        rc = rank_correlations[dim]
        print(f"  d={dim:2d}: Spearman rho = {rc['rho']:+.3f} (p={rc['p']:.3f}, n={rc['n']})")

    print("\nDimension sensitivity (max_GF / min_GF across d=2..64):")
    sens_sorted = sorted(sensitivity.items(), key=lambda x: x[1]["sensitivity_ratio"], reverse=True)
    for method, sens in sens_sorted:
        print(f"  {method:12s}: ratio = {sens['sensitivity_ratio']:.2f} "
              f"(max GF={sens['max_gf']:.4f} at d={sens['max_dim']})")

    print("\nG-F Score at d=64 (full-dimensional evaluation):")
    d64_sorted = sorted(
        [(m, results[m][64]["gf_score"]) for m in METHODS if np.isfinite(results[m][64]["gf_score"])],
        key=lambda x: x[1], reverse=True
    )
    for method, gf in d64_sorted:
        gf_2d = results[method][2]["gf_score"]
        delta = gf - gf_2d
        print(f"  {method:12s}: GF_64={gf:.4f}  GF_2D={gf_2d:.4f}  delta={delta:+.4f}")

    # GNN-specific analysis
    print("\nGNN-specific analysis (d=64 vs d=2):")
    for method in GNN_METHODS:
        gf_64 = results[method][64]["gf_score"]
        gf_2 = results[method][2]["gf_score"]
        if np.isfinite(gf_64) and np.isfinite(gf_2):
            ratio = gf_64 / gf_2 if gf_2 > 0 else np.nan
            print(f"  {method:12s}: GF_64={gf_64:.4f}  GF_2D={gf_2:.4f}  "
                  f"ratio={ratio:.2f}x")

    # ============================================================
    # Save results
    # ============================================================
    print("\n[6/6] Saving results...")

    # Prepare serialisable output
    output = {
        "dimensions": DIMENSIONS,
        "methods": METHODS,
        "results": {
            m: {str(d): {
                "gf_score": float(v["gf_score"]) if np.isfinite(v["gf_score"]) else None,
                "plateau_width": float(v["plateau_width"]) if np.isfinite(v.get("plateau_width", np.nan)) else None,
                "peak_purity": float(v["peak_purity"]) if np.isfinite(v.get("peak_purity", np.nan)) else None,
            } for d, v in results[m].items()}
            for m in METHODS
        },
        "results_2d_projection": {
            m: {str(d): {
                "gf_score": float(v["gf_score"]) if np.isfinite(v["gf_score"]) else None,
                "plateau_width": float(v["plateau_width"]) if np.isfinite(v.get("plateau_width", np.nan)) else None,
                "peak_purity": float(v["peak_purity"]) if np.isfinite(v.get("peak_purity", np.nan)) else None,
            } for d, v in results_2dproj[m].items()}
            for m in METHODS
        },
        "rank_tables": {
            str(dim): [
                {"method": m, "gf_score": s, "rank": r}
                for m, s, r in rank_tables[dim]
            ]
            for dim in DIMENSIONS
        },
        "rank_correlations": {
            str(dim): rc for dim, rc in rank_correlations.items()
        },
        "dimension_sensitivity": sensitivity,
    }

    out_path = RESULTS / "highdim_gf_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"      Saved: {out_path}")

    # ============================================================
    # Publication-quality figure
    # ============================================================
    print("\nGenerating publication figure...")
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.35)

    # Panel A: G-F Score vs dimensionality (line plot)
    ax_a = fig.add_subplot(gs[0, :2])
    for method in METHODS:
        dims = [d for d in DIMENSIONS if np.isfinite(results[method][d]["gf_score"])]
        scores = [results[method][d]["gf_score"] for d in dims]
        ax_a.plot(dims, scores, marker="o", label=method, color=METHOD_COLORS.get(method, None), linewidth=1.5)
    ax_a.set_xlabel("Embedding Dimensionality $d$", fontsize=12)
    ax_a.set_ylabel("G-F Score", fontsize=12)
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(DIMENSIONS)
    ax_a.set_xticklabels([str(d) for d in DIMENSIONS])
    ax_a.set_title("A. G-F Score vs Dimensionality (Full $d$-D Space)", fontsize=13, fontweight="bold")
    ax_a.legend(loc="lower right", fontsize=7, ncol=2)
    ax_a.grid(True, alpha=0.3)
    ax_a.axhline(0.180, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Leiden baseline")

    # Panel B: Rank correlation (2D vs d)
    ax_b = fig.add_subplot(gs[0, 2])
    dims_corr = [d for d in DIMENSIONS if d != 2]
    rhos = [rank_correlations[d]["rho"] for d in dims_corr]
    ax_b.bar(range(len(dims_corr)), rhos, color="#3182bd", alpha=0.8)
    ax_b.set_xticks(range(len(dims_corr)))
    ax_b.set_xticklabels([f"d={d}" for d in dims_corr], rotation=45, ha="right")
    ax_b.set_ylabel("Spearman $\rho$ (2D vs $d$-D ranking)", fontsize=11)
    ax_b.set_title("B. Rank Stability", fontsize=13, fontweight="bold")
    ax_b.axhline(0.0, color="black", linewidth=0.5)
    ax_b.grid(True, alpha=0.3, axis="y")

    # Panel C: Heatmap of G-F Scores (method x dimension)
    ax_c = fig.add_subplot(gs[1, :])
    score_matrix = np.zeros((len(METHODS), len(DIMENSIONS)))
    for i, method in enumerate(METHODS):
        for j, dim in enumerate(DIMENSIONS):
            score_matrix[i, j] = results[method][dim]["gf_score"] if np.isfinite(results[method][dim]["gf_score"]) else np.nan

    im = ax_c.imshow(score_matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=0.25)
    ax_c.set_xticks(range(len(DIMENSIONS)))
    ax_c.set_xticklabels([str(d) for d in DIMENSIONS])
    ax_c.set_yticks(range(len(METHODS)))
    ax_c.set_yticklabels(METHODS)
    ax_c.set_xlabel("Embedding Dimensionality $d$", fontsize=12)
    ax_c.set_ylabel("Method", fontsize=12)
    ax_c.set_title("C. G-F Score Heatmap (Method $\times$ Dimensionality)", fontsize=13, fontweight="bold")
    for i in range(len(METHODS)):
        for j in range(len(DIMENSIONS)):
            if np.isfinite(score_matrix[i, j]):
                text_color = "white" if score_matrix[i, j] > 0.15 else "black"
                ax_c.text(j, i, f"{score_matrix[i, j]:.3f}", ha="center", va="center",
                         fontsize=7, color=text_color)
    plt.colorbar(im, ax=ax_c, label="G-F Score", shrink=0.8)

    fig_path = FIGURES / "FigHD_GF_dimension_gradient.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"      Saved: {fig_path}")
    plt.close()

    print("\n" + BANNER)
    print("High-Dimensional G-F Comparison Complete")
    print(BANNER)


if __name__ == "__main__":
    main()
