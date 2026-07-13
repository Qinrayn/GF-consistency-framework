#!/usr/bin/env python3
"""
degree_controlled_gf.py -- Degree-Controlled Geometric-Functional Analysis
============================================================================
Tests whether the G-F Score ranking is confounded by degree-annotation
correlation.

Motivation:
  Step 42 (degree_preserving_null.py) showed that spectral methods
  (MDS z=-18.4, Spectral z=-11.9) score BELOW the degree-preserving
  null, meaning their G-F Scores are largely determined by degree
  sequence.  Since GO annotations correlate with degree (well-studied
  proteins have both high degree and rich annotations), the spectral
  "optimality" may be an artifact.

  This script performs a pair-level test: for each protein pair (i,j),
  compute:
    D_ij   = embedding distance
    S_ij   = GO Jaccard similarity
    Delta  = |log(deg_i) - log(deg_j)|  (degree dissimilarity)

  Then compare:
    rho(D, S)             standard geometry-function correlation
    rho(D, S | Delta)     partial correlation, controlling for degree

  If the method ranking changes after degree control, the standard
  G-F Score is confounded by degree.

  This is distinct from:
    - Step 42b (method-level z-score vs degree-encoding correlation)
    - Supplementary Note S6 (partial correlation controlling for
      effective rank, not degree)

Output: results/degree_controlled_gf.json
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, set_seed, ALL_METHODS,
    load_curated_network, load_embedding,
    rescale_coordinates, get_results_dir,
)

set_seed(SEED)


# ===================================================================
# Pairwise matrices
# ===================================================================

def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances."""
    return squareform(pdist(coords, metric="euclidean"))


def compute_go_jaccard_similarity(nodes: list, go_map: dict) -> np.ndarray:
    """Pairwise GO Jaccard similarity.

    Jaccard(i,j) = |T_i ∩ T_j| / |T_i ∪ T_j|
    where T_i is the set of GO terms annotated to protein i.
    Returns 0 for pairs with no shared terms.
    """
    n = len(nodes)
    term_sets = []
    for node in nodes:
        node_str = str(node)
        terms = set(go_map.get(node_str, go_map.get(node, [])))
        term_sets.append(terms)

    S = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ti = term_sets[i]
        if not ti:
            continue
        for j in range(i + 1, n):
            tj = term_sets[j]
            if not tj:
                continue
            union = ti | tj
            if len(union) > 0:
                S[i, j] = len(ti & tj) / len(union)
                S[j, i] = S[i, j]
    return S


def compute_degree_dissimilarity(G: nx.Graph, nodes: list) -> np.ndarray:
    """Pairwise degree dissimilarity: |log10(deg+1)_i - log10(deg+1)_j|.

    Log scale handles the power-law degree distribution of PPI networks.
    """
    n = len(nodes)
    log_deg = np.array([np.log10(G.degree(node) + 1) for node in nodes])
    Delta = np.abs(log_deg[:, None] - log_deg[None, :])
    return Delta


# ===================================================================
# Correlation analysis
# ===================================================================

def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle values (excluding diagonal)."""
    n = matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return matrix[idx]


def spearman_partial(x, y, z):
    """Partial Spearman correlation of x and y, controlling for z.

    Uses the rank-residual method:
      1. Convert x, y, z to ranks
      2. Regress rank(x) on rank(z), get residuals
      3. Regress rank(y) on rank(z), get residuals
      4. Spearman correlation of the two residual vectors

    This is equivalent to the standard partial correlation formula
    applied to ranks.
    """
    rx = _rank(x)
    ry = _rank(y)
    rz = _rank(z)

    # Residualize rx on rz
    res_x = _residualize(rx, rz)
    # Residualize ry on rz
    res_y = _residualize(ry, rz)

    rho, p = spearmanr(res_x, res_y)
    return float(rho), float(p)


def _rank(x):
    """Convert to ranks (average ties)."""
    from scipy.stats import rankdata
    return rankdata(x)


def _residualize(y, x):
    """Return residuals of y after linear regression on x."""
    # OLS: y = a + b*x + epsilon
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    return y - y_pred


def compute_attenuation(rho_standard, rho_partial):
    """Fraction of geometry-function correlation explained by degree.

    attenuation = 1 - rho_partial / rho_standard

    Returns None when the standard correlation is too weak to interpret
    (|rho_standard| < 0.02), in which case "attenuation" is meaningless.
    """
    if abs(rho_standard) < 0.02:
        return None  # signal too weak to quantify attenuation
    return float(1.0 - rho_partial / rho_standard)


# ===================================================================
# Permutation test
# ===================================================================

def permutation_test_partial(D_vals, S_vals, Delta_vals, n_perm=1000, seed=42):
    """Permutation test for partial correlation rho(D,S|Delta).

    Shuffles S values across pairs (breaking the D-S link while
    preserving the S-Delta and D-Delta marginal relationships).

    Returns (observed_rho, null_mean, null_std, z_score, p_value).
    """
    rng = np.random.RandomState(seed)

    rho_obs, _ = spearman_partial(D_vals, S_vals, Delta_vals)

    null_rhos = []
    for _ in range(n_perm):
        S_perm = rng.permutation(S_vals)
        rho_perm, _ = spearman_partial(D_vals, S_perm, Delta_vals)
        null_rhos.append(rho_perm)

    null_rhos = np.array(null_rhos)
    null_mean = float(np.mean(null_rhos))
    null_std = float(np.std(null_rhos))
    z_score = float((rho_obs - null_mean) / (null_std + 1e-10))

    # Two-sided p-value
    p_value = float(np.mean(np.abs(null_rhos - null_mean) >= abs(rho_obs - null_mean)))

    return float(rho_obs), null_mean, null_std, z_score, p_value, null_rhos.tolist()


# ===================================================================
# Main
# ===================================================================

def main():
    t_start = time.time()
    print("=" * 72)
    print("  Degree-Controlled Geometric-Functional Analysis")
    print("  Tests whether G-F ranking is confounded by degree")
    print("=" * 72)
    print()

    # ----------------------------------------------------------------
    # Load network and GO annotations
    # ----------------------------------------------------------------
    print("[1/6] Loading curated 153-node yeast network ...")
    G, nodes, go_map = load_curated_network()
    n = len(nodes)
    print(f"  Network: {n} nodes, {G.number_of_edges()} edges")
    n_annotated = sum(1 for nd in nodes if str(nd) in go_map or nd in go_map)
    print(f"  GO-annotated: {n_annotated}/{n}")
    print()

    # ----------------------------------------------------------------
    # Compute pairwise matrices (shared across all methods)
    # ----------------------------------------------------------------
    print("[2/6] Computing pairwise matrices ...")
    S_matrix = compute_go_jaccard_similarity(nodes, go_map)
    Delta_matrix = compute_degree_dissimilarity(G, nodes)

    S_vals = upper_triangle_values(S_matrix)
    Delta_vals = upper_triangle_values(Delta_matrix)

    # Report distribution statistics
    n_pairs = len(S_vals)
    n_shared = np.sum(S_vals > 0)
    print(f"  Total pairs: {n_pairs}")
    print(f"  Pairs sharing >= 1 GO term: {n_shared} ({100*n_shared/n_pairs:.1f}%)")
    print(f"  Mean GO Jaccard: {np.mean(S_vals):.6f}")
    print(f"  Mean degree dissimilarity: {np.mean(Delta_vals):.4f}")
    print()

    # Check degree-similarity correlation (the confound we are testing)
    rho_deg_sim, p_deg_sim = spearmanr(Delta_vals, S_vals)
    print(f"  Degree dissimilarity vs GO Jaccard: "
          f"rho={rho_deg_sim:+.4f} (p={p_deg_sim:.2e})")
    if rho_deg_sim < 0 and p_deg_sim < 0.05:
        print(f"  -> Proteins with SIMILAR degrees share MORE GO terms.")
        print(f"     This is the confound: degree-annotation correlation.")
    print()

    # ----------------------------------------------------------------
    # Load standard G-F scores for comparison
    # ----------------------------------------------------------------
    print("[3/6] Loading standard G-F scores ...")
    results_dir = get_results_dir()
    # Prefer the 11-method file; fall back to 8-method
    gf_file = results_dir / "gf_scores_all11.json"
    if not gf_file.exists():
        gf_file = results_dir / "gf_scores.json"
    with open(gf_file, encoding="utf-8") as f:
        gf_data = json.load(f)
    gf_scores = gf_data.get("scores", gf_data.get("scores_paper_interval", {}))
    print(f"  Loaded {len(gf_scores)} method scores from {gf_file.name}")
    print()

    # ----------------------------------------------------------------
    # For each embedding method, compute correlations
    # ----------------------------------------------------------------
    print("[4/6] Computing degree-controlled correlations ...")
    print("=" * 80)
    print()

    header = (f"{'Method':<14s} {'GF_std':>7s} {'rho(D,S)':>9s} "
              f"{'rho(D,S|d)':>11s} {'Atten':>7s} {'p_perm':>8s}")
    print(header)
    print("-" * len(header))

    method_results = []
    for method in ALL_METHODS:
        try:
            coords, emb_nodes = load_embedding(method, subset="153")
        except FileNotFoundError:
            print(f"  {method:14s}  (embedding not found, skipping)")
            continue

        # Align embedding to network nodes
        node_to_idx = {nd: i for i, nd in enumerate(emb_nodes)}
        common = [nd for nd in nodes if nd in node_to_idx]
        if len(common) < 10:
            print(f"  {method:14s}  (insufficient overlap: {len(common)})")
            continue

        emb_idx = [node_to_idx[nd] for nd in common]
        net_idx = [nodes.index(nd) for nd in common]

        Y = coords[emb_idx]
        Y = rescale_coordinates(Y.copy())

        # Compute distance matrix for this method
        D_matrix = compute_distance_matrix(Y)
        D_vals = upper_triangle_values(D_matrix)

        # Extract matching S and Delta values for common nodes
        S_sub = S_matrix[np.ix_(net_idx, net_idx)]
        Delta_sub = Delta_matrix[np.ix_(net_idx, net_idx)]
        S_sub_vals = upper_triangle_values(S_sub)
        Delta_sub_vals = upper_triangle_values(Delta_sub)

        # Standard correlation
        rho_std, p_std = spearmanr(D_vals, S_sub_vals)

        # Partial correlation (controlling for degree)
        rho_part, p_part = spearman_partial(D_vals, S_sub_vals, Delta_sub_vals)

        # Attenuation
        atten = compute_attenuation(rho_std, rho_part)

        # Permutation test (100 permutations for speed; increase for publication)
        n_perm = 200
        _, null_mean, null_std, z_score, p_perm, _ = permutation_test_partial(
            D_vals, S_sub_vals, Delta_sub_vals, n_perm=n_perm, seed=SEED
        )

        gf_std = float(gf_scores.get(method, 0.0))

        atten = compute_attenuation(rho_std, rho_part)
        atten_str = f"{atten:7.1%}" if atten is not None else "   N/A"

        print(f"  {method:14s} {gf_std:7.4f} {rho_std:+9.4f} "
              f"{rho_part:+11.4f} {atten_str} {p_perm:8.4f}")

        method_results.append({
            "method": method,
            "gf_standard": gf_std,
            "rho_ds": float(rho_std),
            "rho_ds_p": float(p_std),
            "rho_ds_partial": float(rho_part),
            "rho_ds_partial_p": float(p_part),
            "attenuation": atten,
            "permutation_p": float(p_perm),
            "permutation_z": float(z_score),
            "permutation_null_mean": float(null_mean),
            "permutation_null_std": float(null_std),
            "n_perm": n_perm,
        })

    print()

    # ----------------------------------------------------------------
    # Rank comparison
    # ----------------------------------------------------------------
    print("[5/6] Rank comparison ...")
    print("-" * 50)

    if len(method_results) < 4:
        print("  Insufficient methods for rank comparison.")
        return

    # Three rankings
    from scipy.stats import spearmanr as sp

    gf_rank = sorted(method_results, key=lambda x: -x["gf_standard"])
    rho_rank = sorted(method_results, key=lambda x: x["rho_ds"])  # negative = closer = better
    partial_rank = sorted(method_results, key=lambda x: x["rho_ds_partial"])

    print("\n  Ranking by standard G-F Score (higher = better):")
    for i, mr in enumerate(gf_rank, 1):
        print(f"    {i:2d}. {mr['method']:12s}  GF={mr['gf_standard']:.4f}")

    print("\n  Ranking by rho(D, S) (more negative = closer proteins share more GO):")
    for i, mr in enumerate(rho_rank, 1):
        print(f"    {i:2d}. {mr['method']:12s}  rho={mr['rho_ds']:+.4f}")

    print("\n  Ranking by rho(D, S | degree) (degree-controlled):")
    for i, mr in enumerate(partial_rank, 1):
        atten_str = f"(atten={mr['attenuation']:.1%})" if mr['attenuation'] is not None else "(rho too weak)"
        print(f"    {i:2d}. {mr['method']:12s}  rho={mr['rho_ds_partial']:+.4f}  "
              f"{atten_str}")

    # Spearman correlations between rankings
    gf_vals = [mr["gf_standard"] for mr in method_results]
    rho_vals = [-mr["rho_ds"] for mr in method_results]  # negate: higher = better
    partial_vals = [-mr["rho_ds_partial"] for mr in method_results]

    rho_gf_vs_rho, p1 = sp(gf_vals, rho_vals)
    rho_gf_vs_part, p2 = sp(gf_vals, partial_vals)
    rho_rho_vs_part, p3 = sp(rho_vals, partial_vals)

    print(f"\n  Rank correlations (Spearman):")
    print(f"    G-F vs rho(D,S):           {rho_gf_vs_rho:+.4f} (p={p1:.4f})")
    print(f"    G-F vs rho(D,S|degree):    {rho_gf_vs_part:+.4f} (p={p2:.4f})")
    print(f"    rho(D,S) vs rho(D,S|deg):  {rho_rho_vs_part:+.4f} (p={p3:.4f})")
    print()

    # Key question: did the ranking change?
    top3_gf = set(mr["method"] for mr in gf_rank[:3])
    top3_partial = set(mr["method"] for mr in partial_rank[:3])
    overlap = top3_gf & top3_partial

    print(f"  Top-3 by G-F Score:      {sorted(top3_gf)}")
    print(f"  Top-3 by rho(D,S|degree): {sorted(top3_partial)}")
    print(f"  Overlap: {len(overlap)}/3")

    if top3_gf != top3_partial:
        print(f"  >> RANKING CHANGED after degree control.")
        print(f"     Standard G-F ranking is confounded by degree.")
    else:
        print(f"  >> Top-3 stable after degree control.")
    print()

    # Attenuation summary
    print(f"  Attenuation by method (fraction of correlation explained by degree):")
    valid_atten = [mr for mr in method_results if mr["attenuation"] is not None]
    no_signal = [mr for mr in method_results if mr["attenuation"] is None]
    for mr in sorted(valid_atten, key=lambda x: -x["attenuation"]):
        bar = "#" * int(max(mr["attenuation"], 0) * 40)
        print(f"    {mr['method']:12s}  {mr['attenuation']:6.1%}  {bar}")
    for mr in no_signal:
        print(f"    {mr['method']:12s}    N/A  (rho(D,S) too weak to quantify)")
    print()

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    print("[6/6] Saving results ...")

    output = {
        "analysis": "Degree-Controlled Geometric-Functional Analysis",
        "description": (
            "Pair-level test of whether G-F Score ranking is confounded "
            "by degree-annotation correlation. Computes partial Spearman "
            "correlation of embedding distance and GO Jaccard similarity, "
            "controlling for degree dissimilarity."
        ),
        "network": {
            "n_nodes": n,
            "n_edges": G.number_of_edges(),
            "n_annotated": n_annotated,
            "n_pairs": n_pairs,
            "n_pairs_shared_go": int(n_shared),
        },
        "confound_check": {
            "degree_sim_vs_go_jaccard_rho": float(rho_deg_sim),
            "degree_sim_vs_go_jaccard_p": float(p_deg_sim),
            "interpretation": (
                "Proteins with similar degrees share more GO terms. "
                "This is the degree-annotation confound."
                if rho_deg_sim < 0 and p_deg_sim < 0.05
                else "No significant degree-annotation confound detected."
            ),
        },
        "method_results": method_results,
        "rank_correlations": {
            "gf_vs_rho_ds": {"rho": float(rho_gf_vs_rho), "p": float(p1)},
            "gf_vs_rho_partial": {"rho": float(rho_gf_vs_part), "p": float(p2)},
            "rho_ds_vs_rho_partial": {"rho": float(rho_rho_vs_part), "p": float(p3)},
        },
        "ranking_stability": {
            "top3_gf": sorted(top3_gf),
            "top3_partial": sorted(top3_partial),
            "top3_overlap": len(overlap),
            "ranking_changed": top3_gf != top3_partial,
        },
    }

    out_path = results_dir / "degree_controlled_gf.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")
    print()

    elapsed = time.time() - t_start
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Analyzed {len(method_results)} methods in {elapsed:.1f}s")
    print()
    print(f"  Degree-annotation confound: rho={rho_deg_sim:+.4f} (p={rho_deg_sim and p_deg_sim:.2e})")
    # Count methods with significant degree-controlled signal
    sig_methods = [mr for mr in method_results if mr["permutation_p"] < 0.05]
    print(f"  Methods with significant degree-controlled signal: {len(sig_methods)}/{len(method_results)}")
    for mr in sig_methods:
        print(f"    {mr['method']:12s}  rho(D,S|deg)={mr['rho_ds_partial']:+.4f}  "
              f"p={mr['permutation_p']:.4f}")
    print(f"  Ranking changed: {'YES' if top3_gf != top3_partial else 'NO'}")
    print()
    if top3_gf != top3_partial:
        print("  FINDING: Standard G-F ranking IS confounded by degree.")
        print("  Degree control changes the top-3 method ranking.")
    else:
        print("  FINDING: G-F top-3 ranking is robust to degree control.")
        if sig_methods:
            print(f"  But only {len(sig_methods)}/{len(method_results)} methods retain")
            print(f"  significant geometry-function signal after degree control.")
    print()
    print("  Done.")

    return output


if __name__ == "__main__":
    main()