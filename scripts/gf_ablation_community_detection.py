#!/usr/bin/env python3
"""
gf_ablation_community_detection.py -- G-F Score Ablation Study
==============================================================

Tests the sensitivity of G-F Scores to the choice of community detection
algorithm. The main pipeline uses greedy_modularity_communities; this
script compares it against:
  1. Louvain (community.best_partition)
  2. Label Propagation (label_propagation_communities)
  3. Connected Components (fast approximation)

If the method rankings are consistent across community detection algorithms,
this demonstrates that the G-F Score captures genuine functional-geometric
structure rather than an artifact of a specific clustering method.

Outputs:
  - results/gf_ablation_community_detection.json
"""
from __future__ import annotations

import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.integrate import trapezoid
from scipy.stats import spearmanr, kendalltau
import networkx as nx
from networkx.algorithms.community import (
    greedy_modularity_communities,
    label_propagation_communities,
    modularity,
)

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from utils import (
    SEED, ALL_METHODS, CLASSICAL_METHODS,
    rescale_coordinates, load_curated_network, load_embedding,
    precompute_distance_matrix,
    get_data_dir, get_embeddings_dir, get_results_dir,
    GF_R_MIN, GF_R_MAX, R_MIN, R_MAX, N_POINTS, TARGET_STD,
    _community_purity,
)

DATA = get_data_dir()
EMB = get_embeddings_dir()
RES = get_results_dir()

# Use fewer r-points for speed (ablation, not main result)
N_POINTS_ABLATION = 50

# Try to import Louvain
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    try:
        from community import best_partition as louvain_best_partition
        HAS_LOUVAIN = True
    except ImportError:
        HAS_LOUVAIN = False

# Try Leiden
try:
    import leidenalg
    import igraph as ig
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False


# ================================================================
# Community Detection Wrappers
# ================================================================

def detect_greedy_modularity(G):
    """Standard greedy modularity communities."""
    if G.number_of_edges() == 0:
        return [set(G.nodes())]
    try:
        comms = list(greedy_modularity_communities(G))
        if not comms:
            return [set(G.nodes())]
        return comms
    except Exception as e:
        return [set(G.nodes())]


def detect_louvain(G, seed=SEED):
    """Louvain community detection."""
    if not HAS_LOUVAIN:
        return None
    if G.number_of_edges() == 0:
        return [set(G.nodes())]
    try:
        partition = community_louvain.best_partition(G, random_state=seed)
        # Convert partition dict to list of sets
        comm_dict = {}
        for node, comm_id in partition.items():
            comm_dict.setdefault(comm_id, set()).add(node)
        return list(comm_dict.values())
    except Exception as e:
        return [set(G.nodes())]


def detect_label_propagation(G):
    """Label propagation communities."""
    if G.number_of_edges() == 0:
        return [set(G.nodes())]
    try:
        comms = list(label_propagation_communities(G))
        if not comms:
            return [set(G.nodes())]
        return comms
    except Exception as e:
        return [set(G.nodes())]


def detect_connected_components(G):
    """Connected components (fastest, coarsest)."""
    return [set(c) for c in nx.connected_components(G)]


def detect_leiden(G, seed=SEED):
    """Leiden community detection via igraph."""
    if not HAS_LEIDEN:
        return None
    if G.number_of_edges() == 0:
        return [set(G.nodes())]
    try:
        # Convert to igraph
        node_list = sorted(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        g_ig = ig.Graph(n=len(node_list), edges=edges, directed=False)
        partition = leidenalg.find_partition(
            g_ig, leidenalg.ModularityVertexPartition, seed=seed
        )
        comms = []
        for comm_indices in partition:
            comms.append({node_list[i] for i in comm_indices})
        return comms
    except Exception as e:
        return [set(G.nodes())]


# ================================================================
# G-F Curve with Custom Community Detection
# ================================================================

def compute_gf_curve_custom_community(coords, nodes, go_map, r_vals,
                                       community_func):
    """Compute G-F curve using a custom community detection function.

    Same algorithm as utils.compute_gf_curve but with pluggable
    community detection.
    """
    dist_matrix = precompute_distance_matrix(coords)
    n = dist_matrix.shape[0]

    # Pre-sort edges by distance
    iu = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[iu]
    sort_idx = np.argsort(edge_dists)
    sorted_rows = iu[0][sort_idx]
    sorted_cols = iu[1][sort_idx]
    sorted_d = edge_dists[sort_idx]

    r_order = np.argsort(r_vals)
    purities_out = [0.0] * len(r_vals)
    mods_out = [0.0] * len(r_vals)

    G_r = nx.Graph()
    G_r.add_nodes_from(range(n))
    edge_ptr = 0
    n_edges_total = len(sorted_d)

    _cache = {}

    for rank, orig_idx in enumerate(r_order):
        r = float(r_vals[orig_idx])

        while edge_ptr < n_edges_total and sorted_d[edge_ptr] < r:
            G_r.add_edge(int(sorted_rows[edge_ptr]), int(sorted_cols[edge_ptr]))
            edge_ptr += 1

        ne = G_r.number_of_edges()
        if ne == 0:
            continue

        if ne in _cache:
            communities, mod_val = _cache[ne]
        else:
            communities = community_func(G_r)
            # Compute modularity if possible
            try:
                if len(communities) > 1:
                    mod_val = modularity(G_r, communities)
                else:
                    mod_val = 0.0
            except Exception as e:
                mod_val = 0.0
            _cache[ne] = (communities, mod_val)

        # Compute purity
        purities = []
        for comm in communities:
            if not comm:
                continue
            # Convert index-based to name-based
            comm_names = [nodes[idx] for idx in comm if idx < len(nodes)]
            purities.append(_community_purity(comm_names, go_map))

        purities_out[orig_idx] = float(np.mean(purities)) if purities else 0.0
        mods_out[orig_idx] = mod_val

    return purities_out, mods_out


# ================================================================
# Main Ablation Study
# ================================================================

def run_ablation():
    """Run G-F Score computation with all community detection methods."""

    # Load network and annotations
    G, nodes, go_map = load_curated_network(DATA)
    print(f"  Network: {len(nodes)} nodes, {G.number_of_edges()} edges")

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS_ABLATION)

    # Community detection methods to test
    community_methods = {
        "greedy_modularity": detect_greedy_modularity,
        "label_propagation": detect_label_propagation,
        "connected_components": detect_connected_components,
    }

    if HAS_LOUVAIN:
        community_methods["louvain"] = detect_louvain
    else:
        print("  WARNING: python-louvain not installed, skipping Louvain")

    if HAS_LEIDEN:
        community_methods["leiden"] = detect_leiden
    else:
        print("  WARNING: leidenalg not installed, skipping Leiden")

    # Results: {community_method: {embedding_method: {gf_score, purities}}}
    all_results = {}
    method_gf_scores = {}  # {community_method: {embedding_method: gf_score}}

    for comm_name, comm_func in community_methods.items():
        print(f"\n  Community detection: {comm_name}")
        all_results[comm_name] = {}
        method_gf_scores[comm_name] = {}

        for emb_method in ALL_METHODS:
            try:
                coords, emb_nodes = load_embedding(emb_method, "153",
                                                    embeddings_dir=EMB)
            except FileNotFoundError:
                continue

            coords = rescale_coordinates(coords.copy(), target_std=TARGET_STD)

            # Align nodes
            node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
            common = sorted(set(node_to_idx) & set(nodes) & set(go_map))
            idx = [node_to_idx[n] for n in common]
            aligned_coords = coords[idx]

            # Compute G-F curve with this community detection
            t0 = time.time()
            purities, modularities = compute_gf_curve_custom_community(
                aligned_coords, common, go_map, r_vals, comm_func
            )
            elapsed = time.time() - t0

            # Compute G-F Score
            gf = float(trapezoid(
                np.array(purities)[
                    (r_vals >= GF_R_MIN) & (r_vals <= GF_R_MAX)
                ],
                r_vals[(r_vals >= GF_R_MIN) & (r_vals <= GF_R_MAX)]
            ) / (GF_R_MAX - GF_R_MIN))

            all_results[comm_name][emb_method] = {
                "gf_score": gf,
                "mean_purity": float(np.mean(purities)),
                "max_purity": float(np.max(purities)),
                "time_seconds": elapsed,
            }
            method_gf_scores[comm_name][emb_method] = gf

            print(f"    {emb_method:12s}: GF={gf:.4f} ({elapsed:.1f}s)")

    return all_results, method_gf_scores


def analyze_rank_consistency(method_gf_scores):
    """Analyze rank consistency across community detection methods."""

    # Get rankings for each community detection method
    rankings = {}
    for comm_name, scores in method_gf_scores.items():
        sorted_methods = sorted(scores.keys(), key=lambda m: scores[m],
                                reverse=True)
        rankings[comm_name] = {m: i + 1 for i, m in enumerate(sorted_methods)}

    # Find common embedding methods
    all_emb_methods = set()
    for scores in method_gf_scores.values():
        all_emb_methods |= set(scores.keys())
    common_methods = sorted(all_emb_methods)

    # Pairwise Spearman correlations between community detection rankings
    comm_names = sorted(method_gf_scores.keys())
    pairwise_correlations = {}

    for i, c1 in enumerate(comm_names):
        for c2 in comm_names[i + 1:]:
            common = sorted(set(rankings[c1].keys()) & set(rankings[c2].keys()))
            if len(common) >= 4:
                ranks1 = [rankings[c1][m] for m in common]
                ranks2 = [rankings[c2][m] for m in common]
                rho, p = spearmanr(ranks1, ranks2)
                tau, p_tau = kendalltau(ranks1, ranks2)
                pairwise_correlations[f"{c1}_vs_{c2}"] = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "kendall_tau": float(tau),
                    "kendall_p": float(p_tau),
                    "n_methods": len(common),
                }

    # Kendall's W across all community detection methods
    if len(comm_names) >= 2 and len(common_methods) >= 3:
        # Build rank matrix: rows = community methods, cols = embedding methods
        rank_matrix = []
        for c in comm_names:
            row = [rankings[c].get(m, len(common_methods) + 1)
                   for m in common_methods]
            rank_matrix.append(row)
        rank_matrix = np.array(rank_matrix)

        k = len(comm_names)  # number of judges
        n = len(common_methods)  # number of items
        rank_sums = rank_matrix.sum(axis=0)
        mean_rank_sum = rank_sums.mean()
        S = np.sum((rank_sums - mean_rank_sum) ** 2)
        W = 12 * S / (k**2 * (n**3 - n))

        kendall_W = float(W)
    else:
        kendall_W = None

    # Check specific claims:
    # 1. Is Spectral consistently in top 3?
    spectral_ranks = [rankings[c].get("Spectral", 99) for c in comm_names]

    # 2. Is the top method consistent?
    top_methods = {c: sorted(rankings[c].keys(),
                              key=lambda m: rankings[c][m])[0]
                   for c in comm_names}

    # 3. Do GNN methods consistently rank low?
    gnn_methods = ["GAT", "GraphSAGE", "GIN", "VGAE", "VGAE-feat"]
    gnn_ranks = {}
    for gnn in gnn_methods:
        gnn_ranks[gnn] = [rankings[c].get(gnn, 99) for c in comm_names]

    return {
        "rankings": rankings,
        "pairwise_correlations": pairwise_correlations,
        "kendall_W": kendall_W,
        "spectral_ranks_across_methods": {
            c: rankings[c].get("Spectral", None) for c in comm_names
        },
        "spectral_always_top3": all(r <= 3 for r in spectral_ranks),
        "top_method_per_community": top_methods,
        "gnn_ranks_across_methods": gnn_ranks,
        "gnn_consistently_low": all(
            np.mean(ranks) >= 5 for ranks in gnn_ranks.values()
            if len(ranks) > 0
        ),
    }


def main():
    print("=" * 70)
    print("G-F Score Ablation: Community Detection Algorithm Sensitivity")
    print("=" * 70)

    print("\n[1/3] Running ablation study...")
    t_start = time.time()
    all_results, method_gf_scores = run_ablation()
    total_time = time.time() - t_start
    print(f"\n  Total ablation time: {total_time:.1f}s")

    print("\n[2/3] Analyzing rank consistency...")
    consistency = analyze_rank_consistency(method_gf_scores)

    print(f"  Kendall's W: {consistency['kendall_W']}")
    print(f"  Spectral always top-3: {consistency['spectral_always_top3']}")
    print(f"  GNN consistently low: {consistency['gnn_consistently_low']}")

    for pair, corr in consistency["pairwise_correlations"].items():
        print(f"  {pair}: rho={corr['spearman_rho']:.3f} (p={corr['spearman_p']:.3f})")

    # Save results
    print("\n[3/3] Saving results...")
    output = {
        "analysis": "G-F Score Ablation: Community Detection Sensitivity",
        "version": "1.0",
        "n_points": N_POINTS_ABLATION,
        "r_range": [R_MIN, R_MAX],
        "gf_interval": [GF_R_MIN, GF_R_MAX],
        "community_methods_available": list(method_gf_scores.keys()),
        "detailed_results": all_results,
        "gf_scores_by_community": method_gf_scores,
        "rank_consistency": consistency,
        "conclusion": {
            "kendall_W": consistency["kendall_W"],
            "spectral_robust": consistency["spectral_always_top3"],
            "gnn_consistently_low": consistency["gnn_consistently_low"],
            "overall": (
                f"Kendall's W = {consistency['kendall_W']:.3f} across "
                f"{len(method_gf_scores)} community detection methods. "
                f"Spectral consistently top-3: {consistency['spectral_always_top3']}. "
                f"GNN methods consistently low: {consistency['gnn_consistently_low']}. "
                f"The G-F Score ranking is {'robust' if consistency['kendall_W'] and consistency['kendall_W'] > 0.5 else 'somewhat sensitive'} "
                f"to community detection algorithm choice."
            ),
        },
    }

    output_path = RES / "gf_ablation_community_detection.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: Community Detection Ablation")
    print("=" * 70)

    # Print ranking table
    comm_names = sorted(method_gf_scores.keys())
    emb_methods = sorted(set().union(*[set(s.keys()) for s in method_gf_scores.values()]))

    header = f"  {'Method':12s}"
    for c in comm_names:
        header += f"  {c:>18s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for m in emb_methods:
        row = f"  {m:12s}"
        for c in comm_names:
            gf = method_gf_scores[c].get(m, None)
            if gf is not None:
                rank = sorted(method_gf_scores[c].values(), reverse=True).index(gf) + 1
                row += f"  {gf:8.4f} (#{rank})"
            else:
                row += f"  {'N/A':>18s}"
        print(row)

    W = consistency["kendall_W"]
    if W is not None:
        print(f"\n  Kendall's W = {W:.3f}")
        if W > 0.7:
            print("  Interpretation: STRONG rank agreement across community methods")
        elif W > 0.5:
            print("  Interpretation: MODERATE rank agreement")
        else:
            print("  Interpretation: WEAK rank agreement -- G-F Score may be sensitive")

    print("=" * 70)


if __name__ == "__main__":
    main()
