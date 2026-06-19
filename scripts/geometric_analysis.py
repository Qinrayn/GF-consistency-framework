#!/usr/bin/env python3
"""
geometric_analysis.py
Step 7: Geometric analysis - d_intra and d_inter (Proposition 1).
Expected: DM d_inter - d_intra = -1.640, VGAE gap also large.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, ALL_CURATED_METHODS, get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_embedding, compute_centrality_features,
    precompute_distance_matrix,
)


def compute_geometric_margins(coords, nodes, go_map):
    """Compute d_intra and d_inter for functional modules.
    
    d_intra: average Euclidean distance between nodes sharing the same most-frequent GO term.
    d_inter: average Euclidean distance between nodes from different modules.

    Optimised: uses vectorised NumPy indexing instead of nested Python loops.
    """
    from collections import Counter
    
    n = len(nodes)
    
    # Assign each node to its most frequent GO term
    node_to_term = {}
    for node in nodes:
        if node in go_map and go_map[node]:
            term_counts = Counter(go_map[node])
            node_to_term[node] = term_counts.most_common(1)[0][0]
    
    # Group nodes by term
    modules = defaultdict(list)
    for i, node in enumerate(nodes):
        if node in node_to_term:
            modules[node_to_term[node]].append(i)
    
    # Compute pairwise distances using the optimised utility
    dist_matrix = precompute_distance_matrix(coords)
    
    # d_intra: vectorised — extract within-module distances via fancy indexing
    intra_dists = []
    for term, member_indices in modules.items():
        if len(member_indices) < 2:
            continue
        idx = np.asarray(member_indices)
        # Extract upper-triangle distances for this module
        iu = np.triu_indices(len(idx), k=1)
        intra_dists.append(dist_matrix[idx[iu[0]], idx[iu[1]]])
    
    if intra_dists:
        d_intra = float(np.mean(np.concatenate(intra_dists)))
    else:
        d_intra = 0.0
    
    # d_inter: vectorised — extract between-module distances
    inter_dists = []
    module_keys = list(modules.keys())
    for m1_idx in range(len(module_keys)):
        members1 = np.asarray(modules[module_keys[m1_idx]])
        for m2_idx in range(m1_idx + 1, len(module_keys)):
            members2 = np.asarray(modules[module_keys[m2_idx]])
            # Broadcast to get all pairwise distances between two modules
            inter_dists.append(dist_matrix[np.ix_(members1, members2)].ravel())
    
    if inter_dists:
        d_inter = float(np.mean(np.concatenate(inter_dists)))
    else:
        d_inter = 0.0
    
    return {
        "d_intra": d_intra,
        "d_inter": d_inter,
        "margin": d_inter - d_intra,
        "n_modules": len(modules),
    }


def main():
    np.random.seed(SEED)
    
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()
    
    G, nodes, go_map = load_curated_network(data_dir)
    
    methods = ALL_CURATED_METHODS
    
    results = {}
    for method in methods:
        print(f"\nComputing geometric margins for {method}...")
        try:
            coords, emb_nodes = load_embedding(method, "153", embeddings_dir=emb_dir)
            # Align using dict lookup (O(1) per node)
            common = sorted(set(emb_nodes) & set(nodes))
            emb_map = {n: i for i, n in enumerate(emb_nodes)}
            idx_map = [emb_map[n] for n in common]
            aligned_coords = coords[idx_map]
            
            margins = compute_geometric_margins(aligned_coords, common, go_map)
            results[method] = margins
            print(f"  d_intra={margins['d_intra']:.4f}, d_inter={margins['d_inter']:.4f}, "
                  f"margin={margins['margin']:.4f}")
        except Exception as e:
            print(f"  {method} FAILED: {e}")
    
    # Save results
    output_file = results_dir / "geometric_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
