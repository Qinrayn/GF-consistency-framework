#!/usr/bin/env python3
"""
leiden_baseline.py
Step 4: Leiden community detection on original 153-node PPI network.
Expected purity ≈ 0.180
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import SEED, get_data_dir, get_results_dir, load_curated_network, _community_purity

def main():
    import igraph as ig
    
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    
    G, nodes, go_map = load_curated_network(data_dir)
    print(f"Network: {len(nodes)} nodes, {G.number_of_edges()} edges")
    
    # Build igraph graph
    mapping = {n: i for i, n in enumerate(nodes)}
    rev = {i: n for n, i in mapping.items()}
    
    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    g.add_edges(edges)
    
    # Run Leiden community detection (seeded for reproducibility)
    import random as _random
    _random.seed(SEED)
    np.random.seed(SEED)
    partition = g.community_leiden(objective_function="modularity", n_iterations=10)
    print(f"Leiden communities: {len(partition)}")
    
    # Compute functional purity (consistent with utils._community_purity)
    purities = []
    for cluster in partition:
        cluster_nodes = [rev[i] for i in cluster]
        purities.append(_community_purity(cluster_nodes, go_map))
    
    baseline_purity = float(np.mean(purities))
    print(f"Leiden baseline purity: {baseline_purity:.4f}")
    
    # Save result
    result = {
        "leiden_baseline_purity": baseline_purity,
        "n_communities": len(partition),
        "community_purities": purities,
        "community_sizes": [len(c) for c in partition],
    }
    
    output_file = results_dir / "leiden_baseline.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
