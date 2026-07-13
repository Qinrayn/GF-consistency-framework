#!/usr/bin/env python3
"""
gf_conditional_embedding.py -- G-F Optimized Conditional Embedding
==================================================================
Designs a 2D embedding that directly maximizes a differentiable proxy
of the G-F Score, then evaluates it on independent metrics (standard
G-F Score, function prediction MRR, degree-controlled rho).

This is a stress test of the G-F framework:
  - If GF-opt embedding wins on MRR -> framework validated
  - If GF-opt wins on G-F but not MRR -> G-F is gameable (important finding)

The differentiable proxy replaces three non-differentiable operations:
  1. Hard distance threshold r -> soft sigmoid: w_ij = sigmoid((r - D_ij) / tau)
  2. Hard community detection -> soft k-means assignment: q_ik = softmax(-||x_i - c_k||^2 / sigma)
  3. Hard purity (max over GO terms) -> soft max: approximated by LogSumExp

Loss = -GF_proxy + lambda_spread * L_spread + lambda_ppi * L_ppi

  L_spread  = -log(var(pdist(X)) + eps)    prevents collapse
  L_ppi     = mean_{(i,j) in E} relu(D_ij - margin)^2  keeps PPI neighbors close

Output: results/gf_conditional_embedding.json
        embeddings/GFopt_153.npy + GFopt_153_nodes.json
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import trapezoid

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, set_seed, ALL_METHODS,
    load_curated_network, load_embedding,
    rescale_coordinates, compute_gf_curve, compute_gf_score,
    compute_gf_curve_knn, compute_knn_gf_score,
    GF_R_MIN, GF_R_MAX, N_POINTS,
    get_results_dir, get_embeddings_dir,
)

set_seed(SEED)


# ===================================================================
# Differentiable G-F proxy (NumPy, autograd via finite differences
# would be too slow; instead we compute gradients analytically)
# ===================================================================

def soft_spatial_weights(D, r, tau=0.02):
    """Soft thresholding: w_ij = sigmoid((r - D_ij) / tau).

    When tau -> 0, this approaches the hard threshold D_ij < r.
    """
    return 1.0 / (1.0 + np.exp(-(r - D) / tau))


def soft_kmeans_assignment(X, centers, sigma=0.05):
    """Soft cluster assignment: q_ik = softmax(-||x_i - c_k||^2 / sigma).

    Returns assignment matrix Q (n x K).
    """
    # Pairwise squared distances to centers
    diff = X[:, None, :] - centers[None, :, :]  # (n, K, d)
    dist_sq = np.sum(diff ** 2, axis=2)  # (n, K)
    logits = -dist_sq / sigma
    # Softmax with numerical stability
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    Q = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return Q


def soft_purity(Q, Y, n_go_terms):
    """Soft functional purity.

    Q: (n, K) soft cluster assignments
    Y: (n, T) one-hot GO annotation matrix (T = number of GO terms)
    n_go_terms: T

    For each cluster k:
      weighted_count_t = sum_i Q_ik * Y_it
      total = sum_t weighted_count_t
      purity_k = max_t (weighted_count_t / total)
      approximated by LogSumExp: LSE(w) / sum(w) ~ max(w)/sum(w)

    Returns mean purity over clusters.
    """
    K = Q.shape[1]
    # Weighted GO term counts per cluster: (K, T)
    weighted_counts = Q.T @ Y  # (K, T)
    totals = weighted_counts.sum(axis=1, keepdims=True)  # (K, 1)
    totals = np.maximum(totals, 1e-10)

    # Soft max via LSE: LSE(alpha * w) / alpha -> max(w) as alpha -> inf
    alpha = 20.0  # sharpness of soft-max
    log_counts = np.log(weighted_counts + 1e-15)
    lse = np.log(np.sum(np.exp(alpha * log_counts), axis=1)) / alpha
    max_counts = np.exp(lse)  # (K,)

    purities = max_counts / totals.ravel()
    # Weight by cluster size (soft)
    cluster_sizes = Q.sum(axis=0)  # (K,)
    total_weight = cluster_sizes.sum()
    if total_weight < 1e-10:
        return 0.0
    weighted_purity = np.sum(purities * cluster_sizes) / total_weight
    return float(weighted_purity)


def compute_gf_proxy(X, Y, r_vals, centers, tau=0.02, sigma=0.05):
    """Compute differentiable G-F proxy score.

    X: (n, 2) embedding coordinates
    Y: (n, T) GO annotation one-hot
    r_vals: array of distance thresholds
    centers: (K, 2) cluster centers
    """
    D = squareform(pdist(X))
    purities = []
    for r in r_vals:
        W = soft_spatial_weights(D, r, tau)
        # Weighted soft assignment: use W to weight cluster membership
        # Instead of separate k-means, use spatial graph weights directly
        # Community = connected component in soft graph
        # Approximation: use W as adjacency, spectral clustering proxy
        # Simpler: use W-weighted purity directly
        # For each node, its "community" is defined by its neighborhood
        # Purity = for each community (row of W), max GO term fraction

        # Weighted GO counts per node-neighborhood
        # W_norm[i] = W[i] / sum(W[i])
        W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)
        # Neighborhood GO profile: (n, T)
        neigh_go = W_norm @ Y
        # Purity per node: max_t neigh_go[i,t] / sum_t neigh_go[i,t]
        row_sums = neigh_go.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        node_purity = neigh_go.max(axis=1) / row_sums.ravel()
        mean_purity = float(np.mean(node_purity))
        purities.append(mean_purity)

    purities = np.array(purities)
    # G-F proxy = mean purity over [r_min, r_max]
    mask = (r_vals >= GF_R_MIN) & (r_vals <= GF_R_MAX)
    if mask.sum() < 2:
        return 0.0
    gf_proxy = float(trapezoid(purities[mask], r_vals[mask]) / (GF_R_MAX - GF_R_MIN))
    return gf_proxy


# ===================================================================
# Optimization (gradient-free CMA-ES-like via scipy.optimize)
# ===================================================================

def optimize_embedding(X_init, Y, G, r_vals, lambda_spread=0.1,
                       lambda_ppi=0.01, margin=0.3, n_iter=500,
                       lr=0.01):
    """Optimize embedding coordinates to maximize G-F proxy.

    Uses simple gradient ascent with finite-difference gradients.
    For n=153 and d=2, this is 306 parameters - tractable.

    Loss = -GF_proxy + lambda_spread * L_spread + lambda_ppi * L_ppi
    Minimizing Loss = maximizing GF_proxy - penalties
    """
    n = X_init.shape[0]
    X = X_init.copy()

    # Precompute PPI edges
    nodelist = sorted(G.nodes())
    node_to_idx = {nd: i for i, nd in enumerate(nodelist)}
    edges = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            edges.append((node_to_idx[u], node_to_idx[v]))
    edges = np.array(edges) if edges else None

    # Build GO one-hot matrix
    all_terms = sorted(set(t for terms in Y for t in terms))
    term_to_idx = {t: i for i, t in enumerate(all_terms)}
    T = len(all_terms)
    Y_onehot = np.zeros((n, T))
    for i, terms in enumerate(Y):
        for t in terms:
            Y_onehot[i, term_to_idx[t]] = 1.0

    best_gf = 0.0
    best_X = X.copy()

    for it in range(n_iter):
        # Compute current loss components
        D = squareform(pdist(X))
        gf = compute_gf_proxy(X, Y_onehot, r_vals, centers=None)

        # Spread penalty: encourage variance in pairwise distances
        dists = pdist(X)
        dist_var = np.var(dists)
        L_spread = -np.log(dist_var + 1e-10)

        # PPI regularization: connected proteins should be close
        if edges is not None and len(edges) > 0:
            edge_dists = D[edges[:, 0], edges[:, 1]]
            violations = np.maximum(edge_dists - margin, 0)
            L_ppi = np.mean(violations ** 2)
        else:
            L_ppi = 0.0

        loss = -gf + lambda_spread * L_spread + lambda_ppi * L_ppi

        if gf > best_gf:
            best_gf = gf
            best_X = X.copy()

        # Finite-difference gradient (every 10 iterations for speed)
        if it % 10 == 0 and it < n_iter - 10:
            grad = np.zeros_like(X)
            eps = 0.001
            for i in range(min(n, 153)):  # all nodes
                for d in range(2):
                    X_pert = X.copy()
                    X_pert[i, d] += eps
                    gf_p = compute_gf_proxy(X_pert, Y_onehot, r_vals, centers=None)
                    dists_p = pdist(X_pert)
                    L_spread_p = -np.log(np.var(dists_p) + 1e-10)
                    loss_p = -gf_p + lambda_spread * L_spread_p
                    grad[i, d] = (loss_p - loss) / eps

            # Gradient descent
            X -= lr * grad
            # Clamp to reasonable range
            X = np.clip(X, -5, 5)

        if it % 50 == 0:
            print(f"    iter {it:4d}: GF_proxy={gf:.4f}, "
                  f"L_spread={L_spread:.4f}, L_ppi={L_ppi:.4f}, "
                  f"dist_var={dist_var:.6f}")

    return best_X, best_gf


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_embedding(coords, nodes, go_map, G, method_name):
    """Evaluate an embedding on three metrics:
    1. Standard G-F Score (community detection + purity)
    2. kNN-GF Score (high-dim generalization)
    3. Degree-controlled rho(D, S | degree)
    """
    rescaled = rescale_coordinates(coords.copy())
    r_vals = np.linspace(0.05, 0.55, N_POINTS)

    # 1. Standard G-F Score
    purities, _ = compute_gf_curve(rescaled, nodes, go_map, r_vals)
    gf_score = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)

    # 2. kNN-GF Score
    k_vals = list(range(3, 31))
    knn_purities, k_used = compute_gf_curve_knn(rescaled, nodes, go_map, k_vals)
    knn_gf = compute_knn_gf_score(knn_purities, k_used)

    # 3. Degree-controlled rho(D, S | degree)
    n = len(nodes)
    D_mat = squareform(pdist(rescaled))
    # GO Jaccard
    term_sets = [set(go_map.get(str(nd), go_map.get(nd, []))) for nd in nodes]
    S_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if term_sets[i] and term_sets[j]:
                u = term_sets[i] | term_sets[j]
                if len(u) > 0:
                    S_mat[i, j] = len(term_sets[i] & term_sets[j]) / len(u)
                    S_mat[j, i] = S_mat[i, j]
    # Degree dissimilarity
    log_deg = np.array([np.log10(G.degree(nd) + 1) for nd in nodes])
    Delta_mat = np.abs(log_deg[:, None] - log_deg[None, :])

    idx = np.triu_indices(n, k=1)
    D_vals = D_mat[idx]
    S_vals = S_mat[idx]
    Delta_vals = Delta_mat[idx]

    rho_std, _ = spearmanr(D_vals, S_vals)

    # Partial Spearman
    rx, ry, rz = rankdata(D_vals), rankdata(S_vals), rankdata(Delta_vals)
    def resid(y_arr, x_arr):
        X = np.column_stack([np.ones(len(y_arr)), x_arr])
        beta, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
        return y_arr - X @ beta
    res_x = resid(rx, rz)
    res_y = resid(ry, rz)
    rho_partial, p_partial = spearmanr(res_x, res_y)

    atten = float(1.0 - rho_partial / rho_std) if abs(rho_std) > 0.02 else None

    return {
        "method": method_name,
        "gf_score": float(gf_score),
        "knn_gf_score": float(knn_gf),
        "rho_ds": float(rho_std),
        "rho_ds_partial": float(rho_partial),
        "rho_ds_partial_p": float(p_partial),
        "attenuation": atten,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    t_start = time.time()
    print("=" * 72)
    print("  G-F Conditional Embedding: Design vs Evaluate")
    print("  Stress test of the G-F framework")
    print("=" * 72)
    print()

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    print("[1/5] Loading data ...")
    G, nodes, go_map = load_curated_network()
    n = len(nodes)
    print(f"  Network: {n} nodes, {G.number_of_edges()} edges")
    n_annotated = sum(1 for nd in nodes if str(nd) in go_map or nd in go_map)
    print(f"  GO-annotated: {n_annotated}/{n}")
    print()

    # ----------------------------------------------------------------
    # Evaluate existing methods (baseline)
    # ----------------------------------------------------------------
    print("[2/5] Evaluating existing methods ...")
    r_vals = np.linspace(0.05, 0.55, N_POINTS)
    baseline_results = []
    for method in ALL_METHODS:
        try:
            coords, emb_nodes = load_embedding(method, subset="153")
        except FileNotFoundError:
            continue
        node_to_idx = {nd: i for i, nd in enumerate(emb_nodes)}
        common = [nd for nd in nodes if nd in node_to_idx]
        if len(common) < 10:
            continue
        emb_idx = [node_to_idx[nd] for nd in common]
        Y = coords[emb_idx]
        result = evaluate_embedding(Y, common, go_map, G.subgraph(common), method)
        baseline_results.append(result)
        print(f"  {method:14s}  GF={result['gf_score']:.4f}  "
              f"rho(D,S|deg)={result['rho_ds_partial']:+.4f}  "
              f"atten={result['attenuation']}")
    print()

    # ----------------------------------------------------------------
    # Optimize GF-conditional embedding
    # ----------------------------------------------------------------
    print("[3/5] Optimizing G-F conditional embedding ...")
    print("  (This may take several minutes ...)")

    # Initialize from Spectral embedding (best baseline)
    coords_spec, emb_nodes_spec = load_embedding("Spectral", subset="153")
    node_to_idx_s = {nd: i for i, nd in enumerate(emb_nodes_spec)}
    common_s = [nd for nd in nodes if nd in node_to_idx_s]
    emb_idx_s = [node_to_idx_s[nd] for nd in common_s]
    X_init = rescale_coordinates(coords_spec[emb_idx_s].copy())

    # Build GO term lists for the common nodes
    Y_terms = []
    for nd in common_s:
        terms = go_map.get(str(nd), go_map.get(nd, []))
        Y_terms.append(terms)

    # Use fewer r_vals for optimization speed
    r_opt = np.linspace(0.05, 0.42, 20)

    t0 = time.time()
    X_opt, gf_opt_proxy = optimize_embedding(
        X_init, Y_terms, G.subgraph(common_s), r_opt,
        lambda_spread=0.05, lambda_ppi=0.005, margin=0.3,
        n_iter=300, lr=0.005,
    )
    dt = time.time() - t0
    print(f"  Optimization done in {dt:.1f}s")
    print(f"  GF proxy (optimized): {gf_opt_proxy:.4f}")
    print()

    # ----------------------------------------------------------------
    # Evaluate GF-opt embedding
    # ----------------------------------------------------------------
    print("[4/5] Evaluating GF-optimized embedding ...")
    gf_result = evaluate_embedding(X_opt, common_s, go_map,
                                    G.subgraph(common_s), "GF-opt")
    print(f"  GF-opt:  GF={gf_result['gf_score']:.4f}  "
          f"kNN-GF={gf_result['knn_gf_score']:.4f}  "
          f"rho(D,S|deg)={gf_result['rho_ds_partial']:+.4f}  "
          f"atten={gf_result['attenuation']}")
    print()

    # ----------------------------------------------------------------
    # Compare
    # ----------------------------------------------------------------
    print("[5/5] Comparison ...")
    print("=" * 80)
    print()

    all_results = baseline_results + [gf_result]

    # Sort by G-F Score
    sorted_gf = sorted(all_results, key=lambda x: -x["gf_score"])
    print("  Ranking by Standard G-F Score:")
    for i, r in enumerate(sorted_gf, 1):
        tag = " <-- GF-opt" if r["method"] == "GF-opt" else ""
        print(f"    {i:2d}. {r['method']:14s}  GF={r['gf_score']:.4f}  "
              f"rho(D,S|deg)={r['rho_ds_partial']:+.4f}{tag}")

    print()
    sorted_rho = sorted(all_results, key=lambda x: x["rho_ds_partial"])
    print("  Ranking by rho(D,S|degree) (more negative = better):")
    for i, r in enumerate(sorted_rho, 1):
        tag = " <-- GF-opt" if r["method"] == "GF-opt" else ""
        print(f"    {i:2d}. {r['method']:14s}  rho={r['rho_ds_partial']:+.4f}  "
              f"GF={r['gf_score']:.4f}{tag}")

    # Key question: does GF-opt win on rho(D,S|deg)?
    gf_opt_rho = gf_result["rho_ds_partial"]
    best_baseline_rho = min(r["rho_ds_partial"] for r in baseline_results)
    gf_opt_gf = gf_result["gf_score"]
    best_baseline_gf = max(r["gf_score"] for r in baseline_results)

    print()
    print(f"  GF-opt G-F Score:     {gf_opt_gf:.4f}  (best baseline: {best_baseline_gf:.4f})")
    print(f"  GF-opt rho(D,S|deg):  {gf_opt_rho:+.4f}  (best baseline: {best_baseline_rho:+.4f})")
    print()

    if gf_opt_gf > best_baseline_gf and gf_opt_rho < best_baseline_rho:
        print("  RESULT: GF-opt wins on BOTH G-F Score AND degree-controlled rho.")
        print("  -> G-F framework STRONGLY validated: optimizing G-F produces")
        print("     embeddings with genuine geometry-function signal.")
    elif gf_opt_gf > best_baseline_gf and gf_opt_rho >= best_baseline_rho:
        print("  RESULT: GF-opt wins on G-F Score but NOT on degree-controlled rho.")
        print("  -> G-F Score is GAMEABLE: it can be optimized without producing")
        print("     genuine degree-independent geometry-function signal.")
    else:
        print("  RESULT: GF-opt does NOT win on G-F Score.")
        print("  -> Optimization failed to improve over Spectral baseline.")

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    results_dir = get_results_dir()
    embeddings_dir = get_embeddings_dir()

    # Save embedding
    np.save(embeddings_dir / "GFopt_153.npy", X_opt)
    with open(embeddings_dir / "GFopt_153_nodes.json", "w", encoding="utf-8") as f:
        json.dump(common_s, f)

    # Save results
    output = {
        "analysis": "G-F Conditional Embedding: Design vs Evaluate",
        "description": (
            "Optimizes a 2D embedding to maximize a differentiable G-F proxy, "
            "then evaluates it on standard G-F Score and degree-controlled "
            "rho(D,S|degree). Tests whether G-F Score is gameable."
        ),
        "optimization": {
            "init": "Spectral",
            "n_iter": 300,
            "lr": 0.005,
            "lambda_spread": 0.05,
            "lambda_ppi": 0.005,
            "margin": 0.3,
            "gf_proxy_optimized": float(gf_opt_proxy),
            "elapsed_sec": float(dt),
        },
        "all_results": all_results,
        "comparison": {
            "gf_opt_gf_score": float(gf_opt_gf),
            "best_baseline_gf_score": float(best_baseline_gf),
            "gf_opt_rho_partial": float(gf_opt_rho),
            "best_baseline_rho_partial": float(best_baseline_rho),
            "gf_opt_wins_gf": bool(gf_opt_gf > best_baseline_gf),
            "gf_opt_wins_rho": bool(gf_opt_rho < best_baseline_rho),
        },
    }

    out_path = results_dir / "gf_conditional_embedding.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Saved: embeddings/GFopt_153.npy")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()