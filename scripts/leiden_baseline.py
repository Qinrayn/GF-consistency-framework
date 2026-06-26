#!/usr/bin/env python3
"""
leiden_baseline.py
Step 4: Community-detection baseline on the original 153-node PPI network.

Reports functional purity for every community-detection algorithm so that the
embedding G-F Scores (computed with ``greedy_modularity_communities`` in
``utils.compute_gf_curve``) are always compared against an *apples-to-apples*
baseline.  The headline value is the algorithm that matches the G-F pipeline
(``greedy_modularity``); Leiden / Louvain are reported alongside so reviewers
can see the baseline is not an artefact of a single algorithm.

The purity formula is identical everywhere: ``_community_purity`` from
``utils.py`` (most-common GO term / total GO terms in community).
"""
from __future__ import annotations

import sys
import json
import random as _random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir,
    load_curated_network, _community_purity,
)

# Purity formula note kept for the JSON output so downstream readers know the
# comparison is methodologically consistent with the G-F curve.
PURITY_NOTE = (
    "mean over communities of (most-common GO term count / total GO terms). "
    "Identical formula to utils._community_purity used in compute_gf_curve."
)


def _purity_from_named_communities(communities, rev, go_map):
    """Functional purity for a partition whose elements are integer index sets."""
    purities = []
    for cluster in communities:
        cluster_nodes = [rev[i] for i in cluster]
        purities.append(_community_purity(cluster_nodes, go_map))
    return float(np.mean(purities)) if purities else 0.0, purities


def main():
    import igraph as ig
    from networkx.algorithms.community import (
        greedy_modularity_communities, label_propagation_communities,
    )

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    G, nodes, go_map = load_curated_network(data_dir)
    print(f"Network: {len(nodes)} nodes, {G.number_of_edges()} edges")

    # Shared node indexing for both networkx and igraph partitions.
    mapping = {n: i for i, n in enumerate(nodes)}
    rev = {i: n for n, i in mapping.items()}

    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    g.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])

    _random.seed(SEED)
    np.random.seed(SEED)

    # ---- Algorithm 1: greedy_modularity_communities ----------------------
    # THIS is the algorithm the G-F pipeline uses, so it is the apples-to-
    # apples baseline for the 0.163 (Spectral) vs baseline comparison.
    gm_comm = list(greedy_modularity_communities(G))
    gm_comm_idx = [{mapping[n] for n in comm} for comm in gm_comm]
    gm_purity, gm_purities = _purity_from_named_communities(gm_comm_idx, rev, go_map)
    print(f"greedy_modularity baseline: {gm_purity:.4f} ({len(gm_comm)} communities)")

    # ---- Algorithm 2: Leiden (igraph) ------------------------------------
    leiden_part = g.community_leiden(objective_function="modularity", n_iterations=10)
    leiden_purity, leiden_purities = _purity_from_named_communities(leiden_part, rev, go_map)
    print(f"Leiden baseline:            {leiden_purity:.4f} ({len(leiden_part)} communities)")

    # ---- Algorithm 3: Louvain (igraph) -----------------------------------
    louvain_part = g.community_multilevel()
    louvain_purity, louvain_purities = _purity_from_named_communities(louvain_part, rev, go_map)
    print(f"Louvain baseline:           {louvain_purity:.4f} ({len(louvain_part)} communities)")

    # ---- Algorithm 4: Label propagation ----------------------------------
    lp_comm = list(label_propagation_communities(G))
    lp_comm_idx = [{mapping[n] for n in comm} for comm in lp_comm]
    lp_purity, lp_purities = _purity_from_named_communities(lp_comm_idx, rev, go_map)
    print(f"Label propagation baseline: {lp_purity:.4f} ({len(lp_comm)} communities)")

    result = {
        # Headline value = the algorithm the G-F pipeline actually uses.
        "baseline_purity": gm_purity,
        "baseline_algorithm": "greedy_modularity",
        "purity_note": PURITY_NOTE,
        "n_communities": len(gm_comm),
        "community_purities": gm_purities,
        "community_sizes": [len(c) for c in gm_comm],
        # Per-algorithm values (all use the same purity formula).
        "per_algorithm": {
            "greedy_modularity": {
                "purity": gm_purity, "n_communities": len(gm_comm),
            },
            "leiden": {
                "purity": leiden_purity, "n_communities": len(leiden_part),
            },
            "louvain": {
                "purity": louvain_purity, "n_communities": len(louvain_part),
            },
            "label_propagation": {
                "purity": lp_purity, "n_communities": len(lp_comm),
            },
        },
        # Backward-compatible aliases so existing readers keep working.
        "leiden_baseline_purity": leiden_purity,
    }

    output_file = results_dir / "leiden_baseline.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nHeadline baseline (greedy_modularity): {gm_purity:.4f}")
    print(f"Baseline range across 4 algorithms: "
          f"[{min(gm_purity, leiden_purity, louvain_purity, lp_purity):.4f}, "
          f"{max(gm_purity, leiden_purity, louvain_purity, lp_purity):.4f}]")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
