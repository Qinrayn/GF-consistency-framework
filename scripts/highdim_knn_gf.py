#!/usr/bin/env python3
"""
High-dimensional G-F Score: kNN vs Euclidean comparison
=======================================================
Direction A: Validates that kNN-GF Score is stable across dimensions
while Euclidean GF Score degrades (distance concentration effect).

Computes both scores for Spectral embeddings at d=2,4,8,16,32,64
on the curated 153-node yeast PPI network.

Output: results/highdim_knn_gf_comparison.json
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, set_seed,
    get_data_dir, get_results_dir,
    rescale_coordinates, compute_gf_curve, compute_gf_score,
    compute_gf_curve_knn, compute_knn_gf_score,
    GF_R_MIN, GF_R_MAX, N_POINTS, BANNER,
    KNN_DEFAULT_RANGE,
)

DATA = get_data_dir()
RESULTS = get_results_dir()

EDGELIST = DATA / "curated_153_ppi.edgelist"
GO_MAP_FILE = DATA / "gene_go_map.json"


def load_network():
    G = nx.read_edgelist(str(EDGELIST))
    nodes = sorted(G.nodes())
    return G, nodes


def load_go_map():
    with open(GO_MAP_FILE, encoding="utf-8") as f:
        return json.load(f)


def spectral_embedding(G, nodes, dim):
    """Compute Spectral embedding at arbitrary dimension."""
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    # Skip first (zero) eigenvector, take next `dim`
    coords = eigvecs[:, 1:dim+1].real
    return coords


def main():
    print(BANNER)
    print("  High-dimensional G-F Score: kNN vs Euclidean")
    print(BANNER)

    set_seed(SEED)
    G, nodes = load_network()
    go_map = load_go_map()
    r_vals = np.linspace(0.05, 0.55, N_POINTS)
    k_vals = list(range(KNN_DEFAULT_RANGE[0], KNN_DEFAULT_RANGE[1] + 1))

    print(f"  Network: {len(nodes)} nodes, {G.number_of_edges()} edges")
    print(f"  Dimensions: [2, 4, 8, 16, 32, 64]")
    print(f"  k range: {k_vals[0]}..{k_vals[-1]}")
    print(f"  r range: [{GF_R_MIN}, {GF_R_MAX}] ({N_POINTS} points)")

    dimensions = [2, 4, 8, 16, 32, 64]
    results = {}

    for d in dimensions:
        print(f"\n  d={d}...")
        t0 = time.time()

        # Compute Spectral embedding at dimension d
        coords = spectral_embedding(G, nodes, d)
        rescaled = rescale_coordinates(coords, target_std=0.3)

        # Euclidean GF Score
        purities_euc, _ = compute_gf_curve(rescaled, nodes, go_map, r_vals)
        gf_euc = compute_gf_score(r_vals, purities_euc, GF_R_MIN, GF_R_MAX)

        # kNN GF Score
        purities_knn, k_used = compute_gf_curve_knn(rescaled, nodes, go_map, k_vals)
        gf_knn = compute_knn_gf_score(purities_knn, k_used)

        # Distance concentration metrics
        from scipy.spatial.distance import pdist
        dists = pdist(rescaled)
        dist_mean = float(np.mean(dists))
        dist_std = float(np.std(dists))
        dist_cv = dist_std / dist_mean if dist_mean > 0 else 0  # coefficient of variation

        results[f"d{d}"] = {
            "dim": d,
            "gf_euclidean": float(gf_euc),
            "gf_knn": float(gf_knn),
            "purity_euc_curve": [float(p) for p in purities_euc],
            "purity_knn_curve": [float(p) for p in purities_knn],
            "dist_mean": dist_mean,
            "dist_std": dist_std,
            "dist_cv": dist_cv,  # lower = more concentrated
            "time_sec": time.time() - t0,
        }

        print(f"    Euclidean GF: {gf_euc:.4f}")
        print(f"    kNN GF:       {gf_knn:.4f}")
        print(f"    Distance CV:  {dist_cv:.4f} (lower=more concentrated)")
        print(f"    Time: {time.time()-t0:.1f}s")

    # Analysis
    print("\n  " + "=" * 60)
    print("  ANALYSIS")
    print("=" * 60)

    gf_euc_vals = [results[f"d{d}"]["gf_euclidean"] for d in dimensions]
    gf_knn_vals = [results[f"d{d}"]["gf_knn"] for d in dimensions]
    dist_cvs = [results[f"d{d}"]["dist_cv"] for d in dimensions]

    # Stability: coefficient of variation across dimensions
    euc_cv = float(np.std(gf_euc_vals) / np.mean(gf_euc_vals)) if np.mean(gf_euc_vals) > 0 else 0
    knn_cv = float(np.std(gf_knn_vals) / np.mean(gf_knn_vals)) if np.mean(gf_knn_vals) > 0 else 0

    print(f"\n  Euclidean GF across dims: {[f'{v:.4f}' for v in gf_euc_vals]}")
    print(f"  kNN GF across dims:       {[f'{v:.4f}' for v in gf_knn_vals]}")
    print(f"  Distance CV across dims:  {[f'{v:.4f}' for v in dist_cvs]}")
    print(f"\n  Euclidean GF stability (CV across dims): {euc_cv:.4f}")
    print(f"  kNN GF stability (CV across dims):       {knn_cv:.4f}")
    print(f"\n  kNN is {'MORE' if knn_cv < euc_cv else 'LESS'} stable than Euclidean across dimensions")
    print(f"  Improvement factor: {euc_cv/knn_cv:.2f}x" if knn_cv > 0 else "")

    # Distance concentration trend
    rho_conc, p_conc = spearmanr(dimensions, dist_cvs)
    print(f"\n  Dimension vs Distance CV: Spearman rho={rho_conc:.3f} (p={p_conc:.4f})")
    if rho_conc < 0:
        print(f"  -> Distance concentration INCREASES with dimension (confirmed)")

    # Save
    output = {
        "description": "High-dimensional G-F Score: kNN vs Euclidean comparison",
        "method": "Spectral",
        "dimensions": dimensions,
        "k_range": [k_vals[0], k_vals[-1]],
        "r_range": [GF_R_MIN, GF_R_MAX],
        "results": results,
        "summary": {
            "gf_euclidean_by_dim": dict(zip(dimensions, gf_euc_vals)),
            "gf_knn_by_dim": dict(zip(dimensions, gf_knn_vals)),
            "dist_cv_by_dim": dict(zip(dimensions, dist_cvs)),
            "euc_stability_cv": euc_cv,
            "knn_stability_cv": knn_cv,
            "knn_more_stable": knn_cv < euc_cv,
            "improvement_factor": euc_cv / knn_cv if knn_cv > 0 else None,
            "dim_vs_dist_cv_rho": float(rho_conc),
            "dim_vs_dist_cv_p": float(p_conc),
        },
    }

    out_file = RESULTS / "highdim_knn_gf_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to: {out_file}")
    print(BANNER)


if __name__ == "__main__":
    main()
