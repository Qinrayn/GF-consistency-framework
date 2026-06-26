#!/usr/bin/env python3
"""
degree_preserving_null.py
=========================
Degree-preserving null model for the G-F Score.

Instead of shuffling embedding coordinates (the current baseline), this
script randomizes the PPI network edges while preserving the degree
distribution (via networkx.double_edge_swap), then computes GF curves
on the INTERSECTION of the spatial graph with the (randomized) PPI
network.

The current coordinate-shuffle baseline tests:
  "Is the embedding more structured than random coordinates?"

This degree-preserving null tests:
  "Does the SPECIFIC topology of the PPI network create the observed
   geometric-functional alignment, or would ANY network with the same
   degree distribution produce it?"

Design
------
The original GF curve builds communities from the spatial graph alone
(purely embedding-based proximity), which is independent of PPI
topology.  To make the null model sensitive to PPI structure, we
compute communities on the intersection of the spatial graph with the
PPI network: at each radius r, an edge exists iff nodes are within
distance r in the embedding AND connected in the PPI.

Under this formulation:
  - Same embedding coordinates  ->  same pool of candidate edges
  - Randomized PPI edges         ->  different subset selected
  - Same GO annotations          ->  same purity computation

A method significantly above this null indicates that the specific PPI
edges that fall within geometric proximity carry more functional signal
than expected from degree distribution alone.

Uses greedy_modularity_communities to match the main pipeline (utils.py),
ensuring methodological consistency between the null model and the primary
GF Score analysis.
"""
from __future__ import annotations

import sys
import json
import time
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, R_MIN, R_MAX, N_POINTS, GF_R_MIN, GF_R_MAX,
    ALL_METHODS,
    get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_embedding, compute_gf_score,
    precompute_distance_matrix, functional_purity,
)
from networkx.algorithms.community import greedy_modularity_communities

# ============================================================
# Configuration
# ============================================================
N_RANDOMIZATIONS = 50
R_MIN_OVERRIDE = 0.05
R_MAX_OVERRIDE = 0.422
N_POINTS_OVERRIDE = 200


# ============================================================
# Degree-preserving network randomization
# ============================================================

def generate_randomized_networks(G, n_randomizations=50, seed=42):
    """Generate degree-preserving randomized networks using double_edge_swap.

    Parameters
    ----------
    G : nx.Graph
        Original PPI network.
    n_randomizations : int
        Number of randomized copies to generate.
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    list of nx.Graph
        Randomized network copies with identical degree sequences.
    """
    nswap = 10 * G.number_of_edges()
    max_tries = 100 * nswap
    randomized = []

    for i in range(n_randomizations):
        rng = np.random.RandomState(seed + i)
        G_rand = G.copy()
        try:
            nx.double_edge_swap(G_rand, nswap=nswap, max_tries=max_tries, seed=rng)
        except nx.NetworkXError as e:
            print(f"    Warning: randomization {i} failed ({e}), retrying with fewer swaps")
            G_rand = G.copy()
            nx.double_edge_swap(G_rand, nswap=nswap // 2, max_tries=max_tries // 2, seed=rng)

        randomized.append(G_rand)

    return randomized


# ============================================================
# Pre-computation helpers
# ============================================================

def precompute_spatial_edges(coords, nodes):
    """Sort spatial edges by distance and build node-index mapping.

    Returns
    -------
    sorted_rows, sorted_cols, sorted_d : np.ndarray
        Upper-triangle edges sorted by ascending distance.
    node_to_idx : dict
        Node name -> integer index mapping.
    """
    dist_matrix = precompute_distance_matrix(coords)
    n = dist_matrix.shape[0]
    node_to_idx = {name: i for i, name in enumerate(nodes)}

    iu = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[iu]
    sort_idx = np.argsort(edge_dists)
    sorted_rows = iu[0][sort_idx]
    sorted_cols = iu[1][sort_idx]
    sorted_d = edge_dists[sort_idx]

    return sorted_rows, sorted_cols, sorted_d, node_to_idx


def ppi_to_edge_set(ppi_graph, node_to_idx):
    """Convert PPI graph to a set of (min_idx, max_idx) edge tuples."""
    edges = set()
    for u, v in ppi_graph.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            edges.add((min(i, j), max(i, j)))
    return edges


# ============================================================
# Fast intersection-based GF curve
# ============================================================

def compute_gf_curve_intersection_fast(sorted_rows, sorted_cols, sorted_d,
                                        n_nodes, nodes, go_map, r_vals,
                                        ppi_edges):
    """Compute GF curve on spatial-PPI intersection (optimised).

    Uses pre-sorted spatial edges and greedy_modularity_communities for
    community detection (~16ms per call).  Pre-filters to only
    intersection edges for speed.

    Parameters
    ----------
    sorted_rows, sorted_cols, sorted_d : np.ndarray
        Pre-sorted upper-triangle spatial edges.
    n_nodes : int
        Number of nodes.
    nodes : list
        Ordered node identifiers.
    go_map : dict
        Node-to-GO-term mapping.
    r_vals : np.ndarray
        Radius thresholds.
    ppi_edges : set of (int, int)
        PPI edge set (min, max index pairs).

    Returns
    -------
    purities_out : list of float
    """
    # Pre-filter: keep only spatial edges that are also in the PPI
    mask = np.array([(int(sorted_rows[k]), int(sorted_cols[k])) in ppi_edges
                      for k in range(len(sorted_d))])
    isec_rows = sorted_rows[mask]
    isec_cols = sorted_cols[mask]
    isec_d = sorted_d[mask]

    r_order = np.argsort(r_vals)
    purities_out = [0.0] * len(r_vals)

    G_r = nx.Graph()
    G_r.add_nodes_from(range(n_nodes))
    edge_ptr = 0
    n_isec = len(isec_d)
    _cache = {}

    for _, orig_idx in enumerate(r_order):
        r = float(r_vals[orig_idx])

        # Incrementally add intersection edges (already filtered)
        while edge_ptr < n_isec and isec_d[edge_ptr] < r:
            G_r.add_edge(int(isec_rows[edge_ptr]), int(isec_cols[edge_ptr]))
            edge_ptr += 1

        ne = G_r.number_of_edges()
        if ne == 0:
            continue

        if ne in _cache:
            communities = _cache[ne]
        else:
            communities = list(greedy_modularity_communities(G_r))
            _cache[ne] = communities

        purities_out[orig_idx] = functional_purity(communities, go_map, nodes)

    return purities_out


# ============================================================
# Coordinate-shuffle baseline (fast, using same community method)
# ============================================================

def compute_coord_shuffle_baseline_fast(sorted_rows, sorted_cols, sorted_d,
                                         n_nodes, nodes, go_map, r_vals,
                                         ppi_edges,
                                         n_shuffles=50, seed=42):
    """Shuffle node-coordinate mapping, compute intersection-based GF score.

    For each shuffle, applies a random permutation to node labels, remaps
    spatial edges accordingly, filters through PPI intersection, and
    computes purity using label_propagation communities.

    Both null models (DP and shuffle) use the same intersection approach
    so they are directly comparable:
      - DP null:   fixed coords, randomized PPI edges
      - Shuffle:   randomized coords (via node permutation), fixed PPI edges

    Parameters
    ----------
    sorted_rows, sorted_cols, sorted_d : np.ndarray
        Pre-sorted upper-triangle spatial edges from original coords.
    n_nodes : int
        Number of nodes.
    nodes : list
        Ordered node identifiers.
    go_map : dict
        Node-to-GO-term mapping.
    r_vals : np.ndarray
        Radius thresholds.
    ppi_edges : set of (int, int)
        PPI edge set (fixed, not randomized).
    n_shuffles : int
        Number of random permutations.
    seed : int
        Base random seed.

    Returns
    -------
    np.ndarray of GF scores (one per shuffle).
    """
    n = n_nodes
    r_order = np.argsort(r_vals)
    n_edges_total = len(sorted_d)
    all_scores = []

    for s in range(n_shuffles):
        rng = np.random.RandomState(seed + s + 1000)
        perm = rng.permutation(n)
        perm_inv = np.argsort(perm)

        G_r = nx.Graph()
        G_r.add_nodes_from(range(n))
        edge_ptr = 0
        _cache = {}
        purities_out = [0.0] * len(r_vals)

        for _, orig_idx in enumerate(r_order):
            r = float(r_vals[orig_idx])

            while edge_ptr < n_edges_total and sorted_d[edge_ptr] < r:
                # Map original edge indices through permutation
                i_orig = int(sorted_rows[edge_ptr])
                j_orig = int(sorted_cols[edge_ptr])
                i_shuf = int(perm_inv[i_orig])
                j_shuf = int(perm_inv[j_orig])
                e = (min(i_shuf, j_shuf), max(i_shuf, j_shuf))
                # Intersection: only add if also in PPI
                if e in ppi_edges:
                    G_r.add_edge(e[0], e[1])
                edge_ptr += 1

            ne = G_r.number_of_edges()
            if ne == 0:
                continue

            if ne in _cache:
                communities = _cache[ne]
            else:
                communities = list(greedy_modularity_communities(G_r))
                _cache[ne] = communities

            purities_out[orig_idx] = functional_purity(communities, go_map, nodes)

        score = compute_gf_score(r_vals, purities_out, GF_R_MIN, GF_R_MAX)
        all_scores.append(score)

    return np.array(all_scores)


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()

    # Load network
    print("Loading curated 153-node yeast PPI network...")
    G, nodes, go_map = load_curated_network(data_dir)
    n_nodes = len(nodes)
    n_edges = G.number_of_edges()
    print(f"  Network: {n_nodes} nodes, {n_edges} edges")

    # Generate r values
    r_vals = np.linspace(R_MIN_OVERRIDE, R_MAX_OVERRIDE, N_POINTS_OVERRIDE)

    # --------------------------------------------------------
    # Step 1: Generate 50 degree-preserving randomized networks
    # --------------------------------------------------------
    print(f"\nGenerating {N_RANDOMIZATIONS} degree-preserving randomized networks...")
    print(f"  nswap = {10 * n_edges}, max_tries = {100 * 10 * n_edges}")
    t0 = time.time()
    randomized_networks = generate_randomized_networks(
        G, N_RANDOMIZATIONS, seed=SEED,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Verify degree preservation
    orig_deg = dict(G.degree())
    for idx, G_rand in enumerate(randomized_networks[:3]):
        match = all(orig_deg[n] == G_rand.degree(n) for n in orig_deg)
        print(f"  Randomization {idx}: edges={G_rand.number_of_edges()}, "
              f"degree_preserved={match}, connected={nx.is_connected(G_rand)}")

    # Pre-convert randomized networks to edge sets (will merge with node_to_idx per method)
    # We'll do this per-method since node_to_idx depends on common_nodes

    # --------------------------------------------------------
    # Step 2: Compute actual GF Score + null for each method
    # --------------------------------------------------------
    results = {}
    print(f"\nComputing GF curves (intersection-based, greedy_modularity) for {len(ALL_METHODS)} methods...")
    print(f"  {N_RANDOMIZATIONS} randomizations per method")
    print(f"  Community detection: greedy_modularity_communities (~16ms/call)")
    print(f"  Total: {len(ALL_METHODS) * (1 + N_RANDOMIZATIONS)} GF curve computations\n")

    t_total = time.time()

    for method in ALL_METHODS:
        print(f"  {method}...", end=" ", flush=True)
        t_method = time.time()
        try:
            coords, emb_nodes = load_embedding(method, "153", embeddings_dir=emb_dir)

            # Align embedding to network nodes
            node_to_idx_emb = {n: i for i, n in enumerate(emb_nodes)}
            common_nodes = sorted(set(node_to_idx_emb) & set(nodes) & set(go_map))
            node_indices = [node_to_idx_emb[n] for n in common_nodes]
            aligned_coords = coords[node_indices]
            n_common = len(common_nodes)

            # Pre-compute sorted spatial edges (shared across all randomizations)
            s_rows, s_cols, s_d, nmap = precompute_spatial_edges(aligned_coords, common_nodes)

            # PPI edge sets
            ppi_real = ppi_to_edge_set(G, nmap)
            ppi_rands = [ppi_to_edge_set(G_r, nmap) for G_r in randomized_networks]

            # --- Actual GF curve (intersection with real PPI) ---
            actual_purities = compute_gf_curve_intersection_fast(
                s_rows, s_cols, s_d, n_common, common_nodes, go_map, r_vals, ppi_real,
            )
            actual_score = compute_gf_score(r_vals, actual_purities, GF_R_MIN, GF_R_MAX)

            # --- Null GF curves (intersection with randomized PPIs) ---
            null_scores = []
            for i, ppi_rand in enumerate(ppi_rands):
                null_purities = compute_gf_curve_intersection_fast(
                    s_rows, s_cols, s_d, n_common, common_nodes, go_map, r_vals, ppi_rand,
                )
                null_score = compute_gf_score(r_vals, null_purities, GF_R_MIN, GF_R_MAX)
                null_scores.append(null_score)
                # Progress indicator every 10 randomizations
                if (i + 1) % 10 == 0:
                    print(f"[{i+1}/{N_RANDOMIZATIONS}]", end=" ", flush=True)

            null_scores = np.array(null_scores)
            null_mean = float(np.mean(null_scores))
            null_std = float(np.std(null_scores, ddof=1))

            # Z-score and p-value (one-sided: actual > null)
            if null_std > 1e-10:
                z_score = float((actual_score - null_mean) / null_std)
                p_value = float(1 - norm.cdf(z_score))
            else:
                if actual_score > null_mean:
                    z_score = float("inf")
                    p_value = 0.0
                else:
                    z_score = 0.0
                    p_value = 1.0

            # Coordinate-shuffle baseline (intersection-based, same approach)
            shuffle_scores = compute_coord_shuffle_baseline_fast(
                s_rows, s_cols, s_d, n_common, common_nodes, go_map, r_vals,
                ppi_real,
                n_shuffles=N_RANDOMIZATIONS, seed=SEED,
            )
            shuffle_mean = float(np.mean(shuffle_scores))
            shuffle_std = float(np.std(shuffle_scores, ddof=1))
            if shuffle_std > 1e-10:
                shuffle_z = float((actual_score - shuffle_mean) / shuffle_std)
                shuffle_p = float(1 - norm.cdf(shuffle_z))
            else:
                shuffle_z = float("inf") if actual_score > shuffle_mean else 0.0
                shuffle_p = 0.0 if actual_score > shuffle_mean else 1.0

            elapsed = time.time() - t_method
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"actual={actual_score:.4f}  null={null_mean:.4f}+/-{null_std:.4f}  "
                  f"z={z_score:+.2f}  p={p_value:.4f} {sig}  [{elapsed:.1f}s]")

            results[method] = {
                "actual_gf_score": round(actual_score, 6),
                "null_mean": round(null_mean, 6),
                "null_std": round(null_std, 6),
                "z_score": round(z_score, 4),
                "p_value": round(p_value, 6),
                "significant_0.05": p_value < 0.05,
                "null_scores": [round(s, 6) for s in null_scores.tolist()],
                "coord_shuffle_mean": round(shuffle_mean, 6),
                "coord_shuffle_std": round(shuffle_std, 6),
                "coord_shuffle_z": round(shuffle_z, 4),
                "coord_shuffle_p": round(shuffle_p, 6),
                "n_common_nodes": n_common,
            }

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {"error": str(e)}

    total_elapsed = time.time() - t_total
    print(f"\nTotal computation time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # --------------------------------------------------------
    # Step 3: Print summary table
    # --------------------------------------------------------
    print("\n" + "=" * 130)
    print("DEGREE-PRESERVING NULL MODEL RESULTS")
    print("=" * 130)

    hdr = (f"{'Method':<12} {'Actual':>8} {'Null Mean':>10} {'Null Std':>9} "
           f"{'Z-score':>8} {'P-value':>8} {'Sig':>4}   "
           f"{'Shuf Mean':>10} {'Shuf Z':>8} {'Shuf P':>8}")
    print(hdr)
    print("-" * len(hdr))

    for method in ALL_METHODS:
        if method not in results or "error" in results[method]:
            err = results.get(method, {}).get("error", "missing")
            print(f"  {method:<12} -- ERROR: {err}")
            continue
        r = results[method]
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"  {method:<12} {r['actual_gf_score']:>8.4f} "
              f"{r['null_mean']:>10.4f} {r['null_std']:>9.4f} "
              f"{r['z_score']:>+8.2f} {r['p_value']:>8.4f} {sig:>4}   "
              f"{r['coord_shuffle_mean']:>10.4f} "
              f"{r['coord_shuffle_z']:>+8.2f} {r['coord_shuffle_p']:>8.4f}")

    # Significant methods
    sig_methods = [m for m in ALL_METHODS
                   if m in results and "error" not in results[m]
                   and results[m]["significant_0.05"]]
    print(f"\nMethods significantly above degree-preserving null (p < 0.05): "
          f"{len(sig_methods)}/{len(ALL_METHODS)}")
    for m in sig_methods:
        r = results[m]
        print(f"  {m}: z={r['z_score']:+.2f}, p={r['p_value']:.4f}")

    non_sig = [m for m in ALL_METHODS
               if m in results and "error" not in results[m]
               and not results[m]["significant_0.05"]]
    if non_sig:
        print(f"\nMethods NOT significantly above null (p >= 0.05): {len(non_sig)}")
        for m in non_sig:
            r = results[m]
            print(f"  {m}: z={r['z_score']:+.2f}, p={r['p_value']:.4f}")

    # --------------------------------------------------------
    # Step 4: Comparison with coordinate-shuffle baseline
    # --------------------------------------------------------
    print("\n" + "=" * 100)
    print("COMPARISON: Degree-Preserving Null vs Coordinate-Shuffle Baseline")
    print("=" * 100)
    print(f"  {'Method':<12} {'Actual':>8}  "
          f"{'DP Null Z':>10} {'DP P':>8}  "
          f"{'Shuf Z':>10} {'Shuf P':>8}  "
          f"{'Stricter':>10}")
    print("  " + "-" * 82)

    for method in ALL_METHODS:
        if method not in results or "error" in results[method]:
            continue
        r = results[method]
        dp_z = r["z_score"]
        sh_z = r["coord_shuffle_z"]
        # "Stricter" = which null model gives a LOWER z-score for the actual result
        # A lower z-score means the null is harder to beat
        if dp_z < sh_z:
            stricter = "DP null"
        elif sh_z < dp_z:
            stricter = "Coord shuf"
        else:
            stricter = "~same"
        print(f"  {method:<12} {r['actual_gf_score']:>8.4f}  "
              f"{dp_z:>+10.2f} {r['p_value']:>8.4f}  "
              f"{sh_z:>+10.2f} {r['coord_shuffle_p']:>8.4f}  "
              f"{stricter:>10}")

    # --------------------------------------------------------
    # Step 5: Save results
    # --------------------------------------------------------
    output = {
        "description": (
            "Degree-preserving null model for GF Score. "
            "PPI edges randomized via double_edge_swap while preserving "
            "degree distribution. GF curves computed on intersection of "
            "spatial graph with (randomized) PPI network. "
            "Community detection uses greedy_modularity_communities."
        ),
        "parameters": {
            "seed": SEED,
            "n_randomizations": N_RANDOMIZATIONS,
            "r_min": R_MIN_OVERRIDE,
            "r_max": R_MAX_OVERRIDE,
            "n_points": N_POINTS_OVERRIDE,
            "gf_r_min": GF_R_MIN,
            "gf_r_max": GF_R_MAX,
            "nswap": 10 * n_edges,
            "max_tries": 100 * 10 * n_edges,
            "gf_curve_type": "intersection (spatial AND PPI)",
            "community_detection": "greedy_modularity_communities",
        },
        "network": {
            "nodes": n_nodes,
            "edges": n_edges,
        },
        "results": results,
        "significant_methods_0.05": sig_methods,
        "summary": {
            "n_methods_tested": len([m for m in ALL_METHODS if m in results and "error" not in results[m]]),
            "n_significant": len(sig_methods),
            "significant_methods": sig_methods,
            "total_runtime_seconds": round(total_elapsed, 1),
        },
    }

    output_file = results_dir / "degree_preserving_null.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
