#!/usr/bin/env python3
"""
prone_harp_gf.py
================
Implement ProNE (Zhang et al., IJCAI 2019) and HARP (Chen et al., KDD 2018)
embedding methods from scratch, then evaluate their G-F Scores on the curated
153-node yeast PPI network and the full 5936-node network.

ProNE: spectral propagation with Chebyshev polynomial approximation and
       information-theoretic enhancement.
HARP:  hierarchical graph coarsening + spectral embedding + refinement.

Outputs:
  results/prone_harp_gf_scores.json    -- curated 153-node results
  results/prone_harp_full_network.json -- full 5936-node results
"""
from __future__ import annotations

import sys
import json
import random
import numpy as np
import networkx as nx
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.integrate import trapezoid
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, GF_R_MIN, GF_R_MAX, R_MIN, R_MAX, N_POINTS, TARGET_STD,
    get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, compute_centrality_features,
    rescale_coordinates, compute_gf_curve, compute_gf_score,
    spectral_embedding_from_graph,
)
from compute_gf import compute_random_baseline


# ============================================================
# Part 1: ProNE Implementation
# ============================================================

def prone_embedding(G, nodes, embedding_dim=2, chebyshev_order=5):
    """ProNE: Spectral propagation with Chebyshev approximation.

    Algorithm (Zhang et al., IJCAI 2019):
    1. Compute normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}
    2. Spectral propagation: first k eigenvectors of L_norm
    3. Chebyshev polynomial approximation of graph filter (order m)
    4. Information enhancement: log(|x| + eps) + sparse filtering
    5. Return first `embedding_dim` columns

    Parameters
    ----------
    G : nx.Graph
    nodes : list of node labels (ordered)
    embedding_dim : int, output dimensionality (default 2)
    chebyshev_order : int, order of Chebyshev approximation (default 5)

    Returns
    -------
    np.ndarray of shape (n_nodes, embedding_dim)
    """
    n = len(nodes)
    k = max(embedding_dim + 2, 8)  # compute a few extra eigenvectors for stability
    k = min(k, n - 2)

    # Step 1: Normalized Laplacian (sparse)
    A = nx.adjacency_matrix(G, nodelist=nodes, weight=None).astype(np.float64)
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0  # avoid division by zero for isolated nodes
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    I = sparse.eye(n, format="csr")
    L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt

    # Step 2: Spectral propagation -- first k eigenvectors of L_norm
    # Use smallest eigenvalues (near 0) which capture community structure
    try:
        eigvals, eigvecs = eigsh(L_norm, k=k, which="SM", tol=1e-6)
    except Exception as e:
        # Fallback to dense if sparse solver fails
        L_dense = L_norm.toarray()
        eigvals_all, eigvecs_all = np.linalg.eigh(L_dense)
        eigvals = eigvals_all[:k]
        eigvecs = eigvecs_all[:, :k]

    # Sort by eigenvalue (ascending)
    sort_idx = np.argsort(eigvals)
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    # Step 3: Initial features from centrality (6-dim)
    features = compute_centrality_features(G, nodes)

    # Chebyshev polynomial approximation of graph filter
    # T_0(L) X = X
    # T_1(L) X = L X
    # T_m(L) X = 2 L T_{m-1}(L) X - T_{m-2}(L) X
    # Rescale L_norm to [-1, 1] range: L_scaled = 2*L_norm - I (eigenvalues of L_norm are in [0,2])
    # Actually, for Chebyshev on [0, 2], we use L_scaled = L_norm - I so eigenvalues are in [-1, 1]
    L_scaled = L_norm - I

    T_prev2 = features.copy()               # T_0(L) X = X
    T_prev1 = (L_scaled @ features)          # T_1(L) X = L_scaled * X
    filtered = T_prev2 + T_prev1             # accumulate: sum of T_0 + T_1

    for m in range(2, chebyshev_order + 1):
        T_curr = 2.0 * (L_scaled @ T_prev1) - T_prev2
        filtered += T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr

    # Step 4: Information enhancement
    # Apply log(|x| + eps) element-wise
    eps = 1e-8
    enhanced = np.log(np.abs(filtered) + eps)

    # Sparse filtering: keep only top-k values per row
    top_k = min(embedding_dim + 2, enhanced.shape[1])
    for i in range(n):
        row = enhanced[i]
        if len(row) > top_k:
            threshold = np.sort(np.abs(row))[-(top_k)]
            mask = np.abs(row) < threshold
            row[mask] = 0.0

    # Step 5: SVD to get final embedding
    # Use SVD on the enhanced matrix to extract the best `embedding_dim` components
    U, S, Vt = np.linalg.svd(enhanced, full_matrices=False)
    coords = U[:, :embedding_dim] * np.sqrt(S[:embedding_dim])

    return coords


# ============================================================
# Part 2: HARP Implementation
# ============================================================

def _coarsen_graph(G, nodes, target_nodes=None):
    """One level of graph coarsening via maximum weight matching.

    Merges matched node pairs into supernodes. Unmatched nodes remain
    as singleton supernodes.

    Returns
    -------
    G_coarse : nx.Graph  -- coarsened graph
    node_to_super : dict -- maps original node index -> supernode index
    super_to_nodes : dict -- maps supernode index -> list of original node indices
    coarse_nodes : list  -- supernode labels (tuples of original node indices)
    """
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    # Build a weighted graph for matching (uniform weights = 1)
    G_weighted = nx.Graph()
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        G_weighted.add_edge(i, j, weight=1.0)

    # Add isolated nodes so they appear in the matching
    for i in range(n):
        if i not in G_weighted:
            G_weighted.add_node(i)

    # Maximum weight matching
    matching = nx.algorithms.matching.max_weight_matching(G_weighted, maxcardinality=True)

    # Build supernode mapping
    matched = set()
    node_to_super = {}
    super_to_nodes = {}
    super_idx = 0

    for u, v in matching:
        node_to_super[u] = super_idx
        node_to_super[v] = super_idx
        super_to_nodes[super_idx] = [u, v]
        matched.add(u)
        matched.add(v)
        super_idx += 1

    # Unmatched nodes become singleton supernodes
    for i in range(n):
        if i not in matched:
            node_to_super[i] = super_idx
            super_to_nodes[super_idx] = [i]
            super_idx += 1

    n_super = super_idx

    # Build coarsened graph
    G_coarse = nx.Graph()
    G_coarse.add_nodes_from(range(n_super))

    # Aggregate edge weights between supernodes
    edge_weights = {}
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        su, sv = node_to_super[i], node_to_super[j]
        if su != sv:
            key = (min(su, sv), max(su, sv))
            edge_weights[key] = edge_weights.get(key, 0) + 1

    for (su, sv), w in edge_weights.items():
        G_coarse.add_edge(su, sv, weight=w)

    return G_coarse, node_to_super, super_to_nodes, n_super


def harp_embedding(G, nodes, embedding_dim=2):
    """HARP: Hierarchical graph coarsening + embedding + refinement.

    Algorithm (Chen et al., KDD 2018):
    1. Iteratively coarsen the graph using maximum weight matching
       until the graph has ~sqrt(n) nodes
    2. Embed the coarsened graph using Spectral embedding
    3. Refine: propagate embeddings from supernodes back to original nodes

    Parameters
    ----------
    G : nx.Graph
    nodes : list of node labels (ordered)
    embedding_dim : int, output dimensionality (default 2)

    Returns
    -------
    np.ndarray of shape (n_nodes, embedding_dim)
    """
    n = len(nodes)
    target_size = max(int(np.sqrt(n)), 4)  # coarsen down to ~sqrt(n) nodes

    # Hierarchical coarsening: record all levels
    levels = []  # list of (node_to_super, super_to_nodes) at each level
    current_G = G
    current_nodes = nodes
    all_node_to_super = list(range(n))  # maps original node idx -> current supernode idx

    current_n = n
    max_levels = 20  # safety limit

    for level in range(max_levels):
        coarse_G, node_to_super, super_to_nodes, n_super = _coarsen_graph(
            current_G, current_nodes
        )
        levels.append((node_to_super, super_to_nodes, current_n))

        # Update the mapping chain: original -> current supernode
        new_mapping = [0] * n
        for orig_idx in range(n):
            prev_super = all_node_to_super[orig_idx]
            if prev_super in node_to_super:
                new_mapping[orig_idx] = node_to_super[prev_super]
            else:
                new_mapping[orig_idx] = prev_super
        all_node_to_super = new_mapping

        if n_super <= target_size or n_super >= current_n:
            # Stop coarsening if we reached target or can't coarsen further
            current_G = coarse_G
            break

        # Prepare for next level
        coarse_nodes = list(range(n_super))
        current_G = coarse_G
        current_nodes = coarse_nodes
        current_n = n_super

    # Embed the coarsened graph using Spectral embedding
    coarse_n = current_G.number_of_nodes()
    coarse_node_list = sorted(current_G.nodes())

    if coarse_n <= embedding_dim + 1:
        # Graph is too small for spectral embedding, use random
        np.random.seed(SEED)
        coarse_coords = np.random.randn(coarse_n, embedding_dim)
    else:
        coarse_coords = spectral_embedding_from_graph(current_G, nodelist=coarse_node_list)

    # Build supernode index -> coordinate mapping
    coarse_node_to_idx = {nd: i for i, nd in enumerate(coarse_node_list)}

    # Refine: propagate embeddings back to original nodes
    coords = np.zeros((n, embedding_dim))
    for orig_idx in range(n):
        super_idx = all_node_to_super[orig_idx]
        if super_idx in coarse_node_to_idx:
            coords[orig_idx] = coarse_coords[coarse_node_to_idx[super_idx]]
        else:
            # Fallback: use mean of coarse embedding
            coords[orig_idx] = coarse_coords.mean(axis=0)

    return coords


# ============================================================
# Part 3: G-F Score Evaluation on Curated 153-node Network
# ============================================================

def evaluate_on_curated_network():
    """Evaluate ProNE and HARP on the curated 153-node yeast PPI network.

    Returns dict with all results.
    """
    random.seed(SEED)
    np.random.seed(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load network and annotations
    print("Loading curated 153-node network...")
    G, nodes, go_map = load_curated_network(data_dir)
    n = len(nodes)
    print(f"  Network: {n} nodes, {G.number_of_edges()} edges")

    # Load existing GF Scores for comparison
    with open(results_dir / "gf_scores.json", encoding="utf-8") as f:
        existing_data = json.load(f)
    existing_scores = existing_data.get("scores", {})

    # Load GNN GF Scores
    gnn_scores = {}
    gnn_file = results_dir / "gnn_gf_scores.json"
    if gnn_file.exists():
        with open(gnn_file, encoding="utf-8") as f:
            gnn_data = json.load(f)
        gnn_scores = gnn_data.get("gf_scores", {})

    # Merge all existing scores
    all_scores = {}
    all_scores.update(existing_scores)
    all_scores.update(gnn_scores)

    random_baseline = existing_data.get("random_baseline", 0.1348)

    # Generate r values (200 points)
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # Compute ProNE embedding
    print("\nComputing ProNE embedding...")
    random.seed(SEED)
    np.random.seed(SEED)
    prone_coords_raw = prone_embedding(G, nodes, embedding_dim=2, chebyshev_order=5)
    prone_coords = rescale_coordinates(prone_coords_raw, TARGET_STD)
    print(f"  ProNE shape: {prone_coords.shape}")
    print(f"  ProNE std: {np.std(prone_coords):.4f}")

    # Compute HARP embedding
    print("\nComputing HARP embedding...")
    random.seed(SEED)
    np.random.seed(SEED)
    harp_coords_raw = harp_embedding(G, nodes, embedding_dim=2)
    harp_coords = rescale_coordinates(harp_coords_raw, TARGET_STD)
    print(f"  HARP shape: {harp_coords.shape}")
    print(f"  HARP std: {np.std(harp_coords):.4f}")

    # Compute G-F curves
    print("\nComputing G-F curves (200 points)...")

    print("  ProNE G-F curve...")
    prone_purities, prone_mods = compute_gf_curve(prone_coords, nodes, go_map, r_vals)
    prone_gf_score = compute_gf_score(r_vals, prone_purities, GF_R_MIN, GF_R_MAX)
    print(f"  ProNE GF Score: {prone_gf_score:.4f}")

    print("  HARP G-F curve...")
    harp_purities, harp_mods = compute_gf_curve(harp_coords, nodes, go_map, r_vals)
    harp_gf_score = compute_gf_score(r_vals, harp_purities, GF_R_MIN, GF_R_MAX)
    print(f"  HARP GF Score: {harp_gf_score:.4f}")

    # Add new scores to the comparison
    all_scores["ProNE"] = float(prone_gf_score)
    all_scores["HARP"] = float(harp_gf_score)

    # Print summary table: all 13 methods ranked by G-F Score
    print("\n" + "=" * 60)
    print("  G-F Score Ranking (All Methods, Curated 153-node Network)")
    print("=" * 60)
    print(f"  {'Rank':<6}{'Method':<15}{'GF Score':>10}  {'Status'}")
    print("  " + "-" * 54)

    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (method, score) in enumerate(ranked, 1):
        if method in ("ProNE", "HARP"):
            status = "<-- NEW"
        elif score >= random_baseline:
            status = "above random"
        else:
            status = "below random"
        print(f"  {i:<6}{method:<15}{score:>10.4f}  {status}")

    print(f"\n  Random baseline: {random_baseline:.4f}")
    print("=" * 60)

    # Save results
    results = {
        "curated_network": {
            "n_nodes": int(n),
            "n_edges": int(G.number_of_edges()),
        },
        "prone": {
            "gf_score": float(prone_gf_score),
            "purity_range": [float(min(prone_purities)), float(max(prone_purities))],
            "modularity_range": [float(min(prone_mods)), float(max(prone_mods))],
            "purity_curve": [float(p) for p in prone_purities],
            "modularity_curve": [float(m) for m in prone_mods],
        },
        "harp": {
            "gf_score": float(harp_gf_score),
            "purity_range": [float(min(harp_purities)), float(max(harp_purities))],
            "modularity_range": [float(min(harp_mods)), float(max(harp_mods))],
            "purity_curve": [float(p) for p in harp_purities],
            "modularity_curve": [float(m) for m in harp_mods],
        },
        "r_vals": [float(r) for r in r_vals],
        "random_baseline": float(random_baseline),
        "all_scores_ranked": {m: float(s) for m, s in ranked},
        "existing_scores": {m: float(s) for m, s in existing_scores.items()},
        "gnn_scores": {m: float(s) for m, s in gnn_scores.items()},
        "integration_interval": [float(GF_R_MIN), float(GF_R_MAX)],
    }

    output_file = results_dir / "prone_harp_gf_scores.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved curated results to: {output_file}")

    return results


# ============================================================
# Part 4: Full 5936-node Network
# ============================================================

def evaluate_on_full_network():
    """Evaluate ProNE and HARP on the full 5936-node yeast PPI network.

    Returns dict with all results.
    """
    random.seed(SEED)
    np.random.seed(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load full network
    print("\nLoading full 5936-node network...")
    G = nx.Graph()
    edgelist_file = data_dir / "yeast_ppi_5936.edgelist"
    with open(edgelist_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])

    # Keep largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    nodes = sorted(G.nodes())
    n = len(nodes)
    n_edges = G.number_of_edges()
    print(f"  Full network: {n} nodes, {n_edges} edges")

    # Load GO annotations for the annotated subset
    with open(data_dir / "gene_go_map.json", encoding="utf-8") as f:
        go_map = json.load(f)
    annotated_nodes = sorted(set(go_map.keys()) & set(nodes))
    print(f"  Annotated subset: {len(annotated_nodes)} nodes")

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # ---- ProNE on full network ----
    print("\nComputing ProNE on full network...")
    random.seed(SEED)
    np.random.seed(SEED)
    prone_coords_full = prone_embedding(G, nodes, embedding_dim=2, chebyshev_order=5)
    prone_coords_full = rescale_coordinates(prone_coords_full, TARGET_STD)
    print(f"  ProNE full shape: {prone_coords_full.shape}")

    # Evaluate on annotated subset
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}
    ann_indices = [node_to_idx[nd] for nd in annotated_nodes]
    prone_subset = prone_coords_full[ann_indices]

    print("  Computing ProNE GF curve on annotated subset...")
    prone_purities_full, prone_mods_full = compute_gf_curve(
        prone_subset, annotated_nodes, go_map, r_vals
    )
    prone_gf_full = compute_gf_score(r_vals, prone_purities_full, GF_R_MIN, GF_R_MAX)
    print(f"  ProNE full-network GF Score: {prone_gf_full:.4f}")

    # ---- HARP on full network ----
    print("\nComputing HARP on full network...")
    random.seed(SEED)
    np.random.seed(SEED)
    harp_coords_full = harp_embedding(G, nodes, embedding_dim=2)
    harp_coords_full = rescale_coordinates(harp_coords_full, TARGET_STD)
    print(f"  HARP full shape: {harp_coords_full.shape}")

    harp_subset = harp_coords_full[ann_indices]

    print("  Computing HARP GF curve on annotated subset...")
    harp_purities_full, harp_mods_full = compute_gf_curve(
        harp_subset, annotated_nodes, go_map, r_vals
    )
    harp_gf_full = compute_gf_score(r_vals, harp_purities_full, GF_R_MIN, GF_R_MAX)
    print(f"  HARP full-network GF Score: {harp_gf_full:.4f}")

    # Save results
    results = {
        "full_network": {
            "n_nodes": int(n),
            "n_edges": int(n_edges),
            "n_annotated": int(len(annotated_nodes)),
        },
        "prone": {
            "gf_score": float(prone_gf_full),
            "purity_range": [float(min(prone_purities_full)),
                             float(max(prone_purities_full))],
            "modularity_range": [float(min(prone_mods_full)),
                                 float(max(prone_mods_full))],
        },
        "harp": {
            "gf_score": float(harp_gf_full),
            "purity_range": [float(min(harp_purities_full)),
                             float(max(harp_purities_full))],
            "modularity_range": [float(min(harp_mods_full)),
                                 float(max(harp_mods_full))],
        },
        "integration_interval": [float(GF_R_MIN), float(GF_R_MAX)],
    }

    output_file = results_dir / "prone_harp_full_network.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full-network results to: {output_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("  Full Network GF Scores (5936 nodes)")
    print("=" * 50)
    print(f"  {'Method':<15}{'GF Score':>10}")
    print("  " + "-" * 25)
    print(f"  {'ProNE':<15}{prone_gf_full:>10.4f}")
    print(f"  {'HARP':<15}{harp_gf_full:>10.4f}")
    print("=" * 50)

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  ProNE & HARP G-F Score Evaluation")
    print("  SEED = 42")
    print("=" * 60)

    # Part 3: Curated 153-node network
    curated_results = evaluate_on_curated_network()

    # Part 4: Full 5936-node network
    full_results = evaluate_on_full_network()

    # Final combined summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  Curated 153-node network:")
    print(f"    ProNE GF Score: {curated_results['prone']['gf_score']:.4f}")
    print(f"    HARP  GF Score: {curated_results['harp']['gf_score']:.4f}")
    print(f"\n  Full 5936-node network:")
    print(f"    ProNE GF Score: {full_results['prone']['gf_score']:.4f}")
    print(f"    HARP  GF Score: {full_results['harp']['gf_score']:.4f}")

    # Rank comparison
    ranked = curated_results["all_scores_ranked"]
    prone_rank = list(ranked.keys()).index("ProNE") + 1 if "ProNE" in ranked else -1
    harp_rank = list(ranked.keys()).index("HARP") + 1 if "HARP" in ranked else -1
    spectral_score = ranked.get("Spectral", 0.0)

    print(f"\n  ProNE rank: {prone_rank} of {len(ranked)} (GF={ranked.get('ProNE', 0):.4f})")
    print(f"  HARP  rank: {harp_rank} of {len(ranked)} (GF={ranked.get('HARP', 0):.4f})")
    print(f"  Spectral (#1): GF={spectral_score:.4f}")

    if ranked.get("ProNE", 0) > spectral_score:
        print("\n  -> ProNE BEATS Spectral!")
    else:
        print("\n  -> Spectral still leads.")

    if ranked.get("HARP", 0) > spectral_score:
        print("  -> HARP BEATS Spectral!")
    else:
        print("  -> HARP does not surpass Spectral.")

    print("\nDone.")


if __name__ == "__main__":
    main()
