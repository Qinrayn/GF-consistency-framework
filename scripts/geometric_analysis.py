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
    SEED, get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_embedding, compute_centrality_features,
)


def compute_geometric_margins(coords, nodes, go_map):
    """Compute d_intra and d_inter for functional modules.
    
    d_intra: average Euclidean distance between nodes sharing the same most-frequent GO term.
    d_inter: average Euclidean distance between nodes from different modules.
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
    
    # Compute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    
    # d_intra: average distance within modules
    intra_dists = []
    for term, member_indices in modules.items():
        if len(member_indices) < 2:
            continue
        for i_idx in range(len(member_indices)):
            for j_idx in range(i_idx + 1, len(member_indices)):
                intra_dists.append(dist_matrix[member_indices[i_idx], member_indices[j_idx]])
    
    d_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
    
    # d_inter: average distance between modules
    inter_dists = []
    module_keys = list(modules.keys())
    for m1_idx in range(len(module_keys)):
        for m2_idx in range(m1_idx + 1, len(module_keys)):
            members1 = modules[module_keys[m1_idx]]
            members2 = modules[module_keys[m2_idx]]
            for i in members1:
                for j in members2:
                    inter_dists.append(dist_matrix[i, j])
    
    d_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    
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
    
    methods = ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE", "PCA", "VGAE-feat"]
    
    results = {}
    for method in methods:
        print(f"\nComputing geometric margins for {method}...")
        try:
            coords, emb_nodes = load_embedding(method, "153", embeddings_dir=emb_dir)
            # Align
            common = sorted(set(emb_nodes) & set(nodes))
            idx_map = [emb_nodes.index(n) for n in common]
            aligned_coords = coords[idx_map]
            
            margins = compute_geometric_margins(aligned_coords, common, go_map)
            results[method] = margins
            print(f"  d_intra={margins['d_intra']:.4f}, d_inter={margins['d_inter']:.4f}, "
                  f"margin={margins['margin']:.4f}")
        except Exception as e:
            print(f"  {method} FAILED: {e}")
    
    # Save results
    output_file = results_dir / "geometric_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
