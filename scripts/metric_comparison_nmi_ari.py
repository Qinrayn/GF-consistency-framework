#!/usr/bin/env python3
"""
metric_comparison_nmi_ari.py -- Compare G-F Score to NMI and ARI
=================================================================
Computes Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI)
between spatial-graph communities (at optimal r) and GO-based functional
modules, for all 11 embedding methods.

This answers: is G-F Score consistent with standard clustering metrics?

For each method:
  1. Build spatial graph at r* (peak purity radius)
  2. Detect communities via greedy modularity
  3. Build GO ground-truth partition: cluster by dominant GO term
  4. Compute NMI and ARI between spatial and GO partitions
  5. Compare rankings: G-F Score vs NMI vs ARI

Output: results/metric_comparison_nmi_ari.json
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, set_seed, ALL_METHODS,
    load_curated_network, load_embedding,
    rescale_coordinates, compute_gf_curve, compute_gf_score,
    precompute_distance_matrix, build_spatial_graph_fast,
    functional_purity,
    GF_R_MIN, GF_R_MAX, N_POINTS,
    get_results_dir,
)

set_seed(SEED)


def build_go_ground_truth(nodes, go_map):
    """Build GO-based ground-truth partition.

    Each node is assigned to its most specific (least frequent) GO term.
    Nodes without annotations get their own singleton cluster.
    """
    # Count term frequencies
    term_counts = Counter()
    for nd in nodes:
        terms = go_map.get(str(nd), go_map.get(nd, []))
        for t in terms:
            term_counts[t] += 1

    # Assign each node to its rarest GO term
    labels = []
    for nd in nodes:
        terms = go_map.get(str(nd), go_map.get(nd, []))
        if not terms:
            labels.append(-1)  # unannotated -> singleton
        else:
            # Pick the rarest term (most specific)
            rarest = min(terms, key=lambda t: term_counts[t])
            labels.append(hash(rarest) % (10**8))  # numeric label

    # Ensure unique labels for unannotated
    base = max(labels) + 1 if labels else 0
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = base
            base += 1

    return np.array(labels)


def compute_nmi_ari_at_r(coords, nodes, go_map, go_labels, r):
    """Compute NMI and ARI between spatial communities and GO ground truth."""
    D = precompute_distance_matrix(coords)
    G_r = build_spatial_graph_fast(D, r)

    if G_r.number_of_edges() == 0:
        return 0.0, 0.0, 0

    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G_r))

    # Convert community partition to labels
    spatial_labels = np.zeros(len(nodes), dtype=int)
    for ci, comm in enumerate(communities):
        for idx in comm:
            spatial_labels[idx] = ci

    nmi = normalized_mutual_info_score(go_labels, spatial_labels)
    ari = adjusted_rand_score(go_labels, spatial_labels)

    return float(nmi), float(ari), len(communities)


def main():
    t_start = time.time()
    print("=" * 72)
    print("  Metric Comparison: G-F Score vs NMI vs ARI")
    print("=" * 72)
    print()

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    print("[1/4] Loading data ...")
    G, nodes, go_map = load_curated_network()
    n = len(nodes)
    print(f"  Network: {n} nodes, {G.number_of_edges()} edges")
    print()

    # Build GO ground truth
    go_labels = build_go_ground_truth(nodes, go_map)
    n_go_clusters = len(set(go_labels))
    print(f"  GO ground truth: {n_go_clusters} clusters")
    print()

    # Load existing G-F scores
    results_dir = get_results_dir()
    with open(results_dir / "gf_scores_all11.json", encoding="utf-8") as f:
        gf_data = json.load(f)
    gf_scores = gf_data.get("scores", {})
    print(f"  Loaded {len(gf_scores)} G-F scores")
    print()

    # ----------------------------------------------------------------
    # For each method, compute NMI/ARI at multiple r values
    # ----------------------------------------------------------------
    print("[2/4] Computing NMI/ARI for all methods ...")
    print("=" * 90)
    print()

    r_vals = np.linspace(0.05, 0.55, N_POINTS)

    header = (f"{'Method':<14s} {'GF_Score':>8s} {'r_peak':>7s} "
              f"{'NMI_peak':>9s} {'ARI_peak':>9s} {'n_comm':>6s}")
    print(header)
    print("-" * len(header))

    method_results = []
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
        net_idx = [nodes.index(nd) for nd in common]

        Y = rescale_coordinates(coords[emb_idx].copy())
        go_labels_sub = go_labels[net_idx]

        # Compute G-F curve to find peak r
        purities, _ = compute_gf_curve(Y, common, go_map, r_vals)
        gf_score = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)

        # Find peak purity r
        peak_idx = np.argmax(purities)
        r_peak = float(r_vals[peak_idx])

        # Compute NMI/ARI at peak r and a few surrounding r values
        r_test = [r_peak]
        # Also test at 0.1, 0.2, 0.3 (commonly used thresholds)
        for r_add in [0.1, 0.15, 0.2, 0.25, 0.3]:
            if r_add not in r_test:
                r_test.append(r_add)

        nmi_vals = []
        ari_vals = []
        for r in r_test:
            nmi, ari, n_comm = compute_nmi_ari_at_r(
                Y, common, go_map, go_labels_sub, r
            )
            nmi_vals.append(nmi)
            ari_vals.append(ari)

        # Use NMI/ARI at r_peak
        nmi_peak = nmi_vals[0]
        ari_peak = ari_vals[0]

        # Also compute mean NMI/ARI across all r values
        nmi_mean = float(np.mean(nmi_vals))
        ari_mean = float(np.mean(ari_vals))

        print(f"  {method:14s} {gf_score:8.4f} {r_peak:7.3f} "
              f"{nmi_peak:9.4f} {ari_peak:9.4f} {n_comm:6d}")

        method_results.append({
            "method": method,
            "gf_score": float(gf_score),
            "r_peak": r_peak,
            "nmi_peak": float(nmi_peak),
            "ari_peak": float(ari_peak),
            "nmi_mean": nmi_mean,
            "ari_mean": ari_mean,
            "n_communities_at_peak": int(n_comm),
        })

    print()

    # ----------------------------------------------------------------
    # Rank comparison
    # ----------------------------------------------------------------
    print("[3/4] Rank comparison ...")
    print("-" * 50)

    if len(method_results) < 4:
        print("  Insufficient methods.")
        return

    gf_vals = [r["gf_score"] for r in method_results]
    nmi_vals = [r["nmi_peak"] for r in method_results]
    ari_vals = [r["ari_peak"] for r in method_results]

    rho_gf_nmi, p_gf_nmi = spearmanr(gf_vals, nmi_vals)
    rho_gf_ari, p_gf_ari = spearmanr(gf_vals, ari_vals)
    rho_nmi_ari, p_nmi_ari = spearmanr(nmi_vals, ari_vals)

    print(f"  G-F vs NMI:  rho={rho_gf_nmi:+.4f} (p={p_gf_nmi:.4f})")
    print(f"  G-F vs ARI:  rho={rho_gf_ari:+.4f} (p={p_gf_ari:.4f})")
    print(f"  NMI vs ARI:  rho={rho_nmi_ari:+.4f} (p={p_nmi_ari:.4f})")
    print()

    # Rankings
    rank_gf = sorted(method_results, key=lambda x: -x["gf_score"])
    rank_nmi = sorted(method_results, key=lambda x: -x["nmi_peak"])
    rank_ari = sorted(method_results, key=lambda x: -x["ari_peak"])

    print("  Top-3 by G-F Score:  ", [r["method"] for r in rank_gf[:3]])
    print("  Top-3 by NMI:        ", [r["method"] for r in rank_nmi[:3]])
    print("  Top-3 by ARI:        ", [r["method"] for r in rank_ari[:3]])
    print()

    top3_gf = set(r["method"] for r in rank_gf[:3])
    top3_nmi = set(r["method"] for r in rank_nmi[:3])
    top3_ari = set(r["method"] for r in rank_ari[:3])
    print(f"  Top-3 overlap (G-F vs NMI): {len(top3_gf & top3_nmi)}/3")
    print(f"  Top-3 overlap (G-F vs ARI): {len(top3_gf & top3_ari)}/3")
    print()

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    print("[4/4] Saving ...")

    output = {
        "analysis": "Metric Comparison: G-F Score vs NMI vs ARI",
        "description": (
            "Compares G-F Score ranking with NMI and ARI between "
            "spatial-graph communities and GO-based ground-truth "
            "partitions. Tests whether G-F Score is consistent with "
            "standard clustering quality metrics."
        ),
        "network": {"n_nodes": n, "n_edges": G.number_of_edges()},
        "go_ground_truth": {"n_clusters": int(n_go_clusters)},
        "method_results": method_results,
        "rank_correlations": {
            "gf_vs_nmi": {"rho": float(rho_gf_nmi), "p": float(p_gf_nmi)},
            "gf_vs_ari": {"rho": float(rho_gf_ari), "p": float(p_gf_ari)},
            "nmi_vs_ari": {"rho": float(rho_nmi_ari), "p": float(p_nmi_ari)},
        },
        "top3_overlap": {
            "gf_vs_nmi": len(top3_gf & top3_nmi),
            "gf_vs_ari": len(top3_gf & top3_ari),
        },
    }

    out_path = results_dir / "metric_comparison_nmi_ari.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()