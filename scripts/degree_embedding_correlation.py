#!/usr/bin/env python3
"""
degree_embedding_correlation.py
================================
Formally test the hypothesis that the DP null model z-score dichotomy
arises from how much each embedding method encodes degree information.

Hypothesis:
  Spectral methods (Laplacian eigenvectors) directly encode degree structure.
  DP randomization preserves degrees, so spectral GF is "expected" under null.
  Random-walk methods capture transition-path structure beyond degrees,
  which is destroyed by rewiring, so they exceed DP null.

Test: correlate each method's degree-embedding similarity with its DP null z-score.
"""
import json
import sys
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import SEED, get_results_dir, get_data_dir, load_curated_network, TARGET_STD, rescale_coordinates

RESULTS = get_results_dir()
DATA = get_data_dir()

METHODS = ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE",
           "PCA", "VGAE-feat", "GraphSAGE", "GAT", "GIN"]


def load_embedding(method):
    """Load 2D embedding for the curated 153-node network."""
    fpath = Path("embeddings") / f"{method}_153_nodes.json"
    if not fpath.exists():
        return None, None
    with open(fpath) as f:
        data = json.load(f)
    if isinstance(data, dict):
        nodes = sorted(data.keys())
        coords = np.array([[data[n]["x"], data[n]["y"]] for n in nodes])
    else:
        fpath_npy = Path("embeddings") / f"{method}_153.npy"
        nodes_fpath = Path("embeddings") / f"{method}_153_nodes.json"
        if fpath_npy.exists() and nodes_fpath.exists():
            coords = np.load(fpath_npy)
            with open(nodes_fpath) as f:
                nodes = json.load(f)
        else:
            return None, None
    return coords, nodes


def compute_degree_similarity(coords, nodes, G):
    """Compute how much the embedding encodes degree information.

    Method: Spearman correlation between node degree rank and
    first embedding dimension (which captures the dominant variance).
    """
    degrees = np.array([G.degree(n) for n in nodes], dtype=float)
    # Use the embedding dimension with largest variance
    var = np.var(coords, axis=0)
    primary_dim = np.argmax(var)
    emb_primary = coords[:, primary_dim]

    # Correlation between degree and primary embedding dimension
    rho, p = spearmanr(degrees, emb_primary)

    # Also compute mean pairwise distance correlation with degree difference
    n = len(nodes)
    sample = min(n, 153)
    idx = np.random.default_rng(SEED).choice(n, sample, replace=False)
    emb_dists = cdist(coords[idx], coords[idx])[np.triu_indices(sample, k=1)]
    deg_diffs = np.abs(np.subtract.outer(degrees[idx], degrees[idx]))[np.triu_indices(sample, k=1)]

    dist_deg_rho, _ = spearmanr(emb_dists, deg_diffs)

    return abs(rho), dist_deg_rho


def main():
    G, nodes, _ = load_curated_network(DATA)
    print(f"Network: {len(nodes)} nodes, {G.number_of_edges()} edges")

    # Load DP null z-scores
    dp = json.load(open(RESULTS / "degree_preserving_null.json"))
    z_scores = {}
    for m in METHODS:
        if m in dp["results"]:
            z_scores[m] = dp["results"][m]["z_score"]

    # Compute degree-embedding similarity for each method
    results = []
    print(f"\n{'Method':<15} {'z-score':>10} {'|ρ(deg,dim1)|':>15} {'ρ(dist,Δdeg)':>15}")
    print("-" * 60)

    for method in METHODS:
        if method not in z_scores:
            continue
        coords, emb_nodes = load_embedding(method)
        if coords is None:
            print(f"  {method}: no embedding found")
            continue

        # Align node ordering
        if emb_nodes != nodes:
            node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
            idx = [node_to_idx.get(n) for n in nodes]
            if None in idx:
                continue
            coords = coords[idx]

        abs_rho_dim, rho_dist_deg = compute_degree_similarity(coords, nodes, G)
        z = z_scores[method]
        results.append({
            "method": method,
            "z_score": z,
            "degree_dim_correlation": float(abs_rho_dim),
            "distance_degree_correlation": float(rho_dist_deg),
        })
        print(f"  {method:<15} {z:10.2f} {abs_rho_dim:15.4f} {rho_dist_deg:15.4f}")

    # Cross-method correlation: degree similarity vs z-score
    if len(results) >= 5:
        z_arr = np.array([r["z_score"] for r in results])
        deg_arr = np.array([r["degree_dim_correlation"] for r in results])
        dist_arr = np.array([r["distance_degree_correlation"] for r in results])

        rho_z_deg, p_z_deg = spearmanr(z_arr, deg_arr)
        rho_z_dist, p_z_dist = spearmanr(z_arr, dist_arr)

        print(f"\n=== Cross-Method Correlation ===")
        print(f"  z-score vs |ρ(degree, dim1)|:  rho = {rho_z_deg:+.4f} (p = {p_z_deg:.4f})")
        print(f"  z-score vs ρ(dist, Δdegree):  rho = {rho_z_dist:+.4f} (p = {p_z_dist:.4f})")

        # Categorize
        spectral_methods = [r for r in results if r["method"] in ["Spectral", "MDS", "DM", "PCA"]]
        rw_methods = [r for r in results if r["method"] in ["DeepWalk", "Node2Vec", "GIN"]]

        if spectral_methods and rw_methods:
            spec_z = np.mean([r["z_score"] for r in spectral_methods])
            rw_z = np.mean([r["z_score"] for r in rw_methods])
            spec_deg = np.mean([r["degree_dim_correlation"] for r in spectral_methods])
            rw_deg = np.mean([r["degree_dim_correlation"] for r in rw_methods])

            print(f"\n=== Class Averages ===")
            print(f"  Spectral methods: mean z = {spec_z:+.2f}, mean |ρ(deg)| = {spec_deg:.4f}")
            print(f"  Random-walk methods: mean z = {rw_z:+.2f}, mean |ρ(deg)| = {rw_deg:.4f}")

    # Save results
    output = {
        "hypothesis": "Spectral methods encode degree structure; DP null preserves degrees, so spectral GF is expected under null. Random-walk methods capture path-based structure beyond degrees.",
        "per_method": results,
        "cross_method_correlation": {
            "z_vs_degree_dim": {"rho": float(rho_z_deg), "p": float(p_z_deg)},
            "z_vs_distance_degree": {"rho": float(rho_z_dist), "p": float(p_z_dist)},
        } if len(results) >= 5 else None,
    }
    out_path = RESULTS / "degree_embedding_correlation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
