#!/usr/bin/env python3
"""
full_network.py
Step 6: Full network validation.
- Embed on the full ~5,936-node STRING topology
- Evaluate G-F curves only on the 153 GO-annotated nodes
Methods: DM, MDS, Node2Vec, VGAE
"""
from __future__ import annotations

import sys
import json
import random
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_embeddings_dir,
    load_full_STRING_network, compute_centrality_features,
    rescale_coordinates, save_embedding, compute_gf_curve,
    build_similarity_matrix, diffusion_map_from_similarity,
    classical_mds_from_distances, node2vec_from_graph, vgae_from_graph,
)

R_MIN = 0.05
R_MAX = 0.55
N_POINTS = 200


def embed_dm_full(G, nodes):
    """DM on full network."""
    print("  Computing centrality features on full network (this may take a while)...")
    features = compute_centrality_features(G, nodes)
    sim = build_similarity_matrix(features)
    coords = diffusion_map_from_similarity(sim)
    return rescale_coordinates(coords, target_std=0.3)


def embed_mds_full(G, nodes):
    """MDS on full network using scipy sparse shortest paths."""
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix
    
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    
    # Build sparse adjacency matrix
    row, col, data = [], [], []
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        row.extend([i, j])
        col.extend([j, i])
        data.extend([1, 1])
    
    adj_sparse = csr_matrix((data, (row, col)), shape=(n, n))
    D = shortest_path(adj_sparse, directed=False, unweighted=True)
    D[np.isinf(D)] = n  # Replace disconnected pairs
    
    coords = classical_mds_from_distances(D)
    return rescale_coordinates(coords, target_std=0.3)


def embed_node2vec_full(G, nodes, walk_length=20, walks_per_node=10, window=5, p=0.5, q=2.0):
    """Node2Vec on full network."""
    print(f"  Generating Node2Vec embedding for {len(nodes)} nodes...")
    coords = node2vec_from_graph(G, walk_length=walk_length,
                                  walks_per_node=walks_per_node,
                                  window_size=window, p=p, q=q, seed=SEED)
    return rescale_coordinates(coords, target_std=0.3)


def embed_vgae_full(G, nodes, hidden_dim=4, latent_dim=2, epochs=300, lr=0.01):
    """VGAE on full network (structure-only, identity features)."""
    print(f"  Training VGAE for {epochs} epochs on {len(nodes)} nodes...")
    coords = vgae_from_graph(G, hidden_dim=hidden_dim, latent_dim=latent_dim,
                              epochs=epochs, lr=lr, seed=SEED)
    return rescale_coordinates(coords, target_std=0.3)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full network
    print("Loading full STRING network...")
    G_full = load_full_STRING_network(data_dir)
    nodes_full = sorted(G_full.nodes())
    print(f"Full network: {len(nodes_full)} nodes, {G_full.number_of_edges()} edges")
    
    # Load GO map for annotated subset
    with open(data_dir / "gene_go_map.json") as f:
        go_map = json.load(f)
    annotated_nodes = sorted(set(go_map.keys()) & set(nodes_full))
    print(f"Annotated subset: {len(annotated_nodes)} nodes")
    
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    
    methods_to_run = {
        "DM": lambda: embed_dm_full(G_full, nodes_full),
        "MDS": lambda: embed_mds_full(G_full, nodes_full),
        "Node2Vec": lambda: embed_node2vec_full(G_full, nodes_full),
        "VGAE": lambda: embed_vgae_full(G_full, nodes_full),
    }
    
    results = {"r": r_vals.tolist(), "n_nodes_full": len(nodes_full),
               "n_nodes_evaluated": len(annotated_nodes)}
    
    for method_name, embed_fn in methods_to_run.items():
        print(f"\nComputing {method_name} on full network...")
        random.seed(SEED)
        np.random.seed(SEED)
        try:
            coords = embed_fn()
            # Save full embedding
            save_embedding(coords, nodes_full, method_name, "full", emb_dir)
            print(f"  Saved {method_name} full embedding: {coords.shape}")
            
            # Evaluate on annotated subset only (dict lookup O(1))
            full_node_map = {n: i for i, n in enumerate(nodes_full)}
            ann_indices = [full_node_map[n] for n in annotated_nodes]
            subset_coords = coords[ann_indices]
            purities, modularities = compute_gf_curve(
                subset_coords, annotated_nodes, go_map, r_vals
            )
            results[f"{method_name}_purity"] = purities
            results[f"{method_name}_modularity"] = modularities
            print(f"  Purity range: [{min(purities):.3f}, {max(purities):.3f}]")
        except Exception as e:
            print(f"  {method_name} FAILED: {e}")
    
    # Save results
    output_file = results_dir / "full_network_validation.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
