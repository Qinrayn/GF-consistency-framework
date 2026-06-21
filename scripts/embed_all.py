#!/usr/bin/env python3
"""
embed_all.py
Step 2: Compute all 8 embedding methods on the curated 153-node network.
Methods: DM, MDS, Spectral, DeepWalk, Node2Vec, VGAE, VGAE-feat, PCA
"""

import sys
import json
import random
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, TARGET_STD, get_data_dir, get_embeddings_dir, load_curated_network,
    compute_centrality_features, rescale_coordinates, save_embedding,
    build_similarity_matrix, diffusion_map_from_similarity,
    classical_mds_from_distances, spectral_embedding_from_graph,
    deepwalk_from_graph, node2vec_from_graph, vgae_from_graph,
)

# Seeds are set inside main() to avoid side-effects on import.


def embed_diffusion_map(G, nodes, features=None, n_components=2):
    """Diffusion Map: 6 centrality features -> similarity -> Markov -> eigendecomposition.

    Parameters
    ----------
    features : np.ndarray, optional
        Pre-computed centrality features.  When *None*, they are computed
        internally (backward compatible).
    n_components : int, default 2
        Number of dimensions to return.
    """
    if features is None:
        features = compute_centrality_features(G, nodes)
    sim = build_similarity_matrix(features)
    coords = diffusion_map_from_similarity(sim, n_components=n_components)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_mds(G, nodes, n_components=2):
    """Classical MDS on shortest-path distances.

    Optimised: uses vectorised NumPy filling instead of nested Python loops.

    Parameters
    ----------
    n_components : int, default 2
        Number of dimensions to return.
    """
    n = len(nodes)
    lengths = dict(nx.shortest_path_length(G))
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    D = np.full((n, n), n, dtype=float)  # default = disconnected
    for u, dists in lengths.items():
        i = node_to_idx[u]
        for v, d in dists.items():
            D[i, node_to_idx[v]] = d
    coords = classical_mds_from_distances(D, n_components=n_components)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_spectral(G, nodes, n_components=2):
    """Spectral embedding from normalized Laplacian.

    Parameters
    ----------
    n_components : int, default 2
        Number of dimensions (Fiedler and subsequent eigenvectors).
    """
    coords = spectral_embedding_from_graph(G, nodelist=nodes, n_components=n_components)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_deepwalk(G, nodes, walk_length=20, walks_per_node=10, window=5, n_components=2):
    """DeepWalk: uniform random walks + co-occurrence matrix + SVD."""
    coords = deepwalk_from_graph(G, walk_length=walk_length,
                                  walks_per_node=walks_per_node,
                                  window_size=window, dimensions=n_components, seed=SEED)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_node2vec(G, nodes, walk_length=20, walks_per_node=10, window=5, p=0.5, q=2.0, n_components=2):
    """Node2Vec: biased random walks + co-occurrence matrix + SVD."""
    coords = node2vec_from_graph(G, walk_length=walk_length,
                                  walks_per_node=walks_per_node,
                                  window_size=window, p=p, q=q,
                                  dimensions=n_components, seed=SEED)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_vgae(G, nodes, features=None, hidden_dim=4, latent_dim=2, epochs=300, lr=0.01):
    """VGAE: Variational Graph Autoencoder with 2-layer GCN encoder."""
    coords = vgae_from_graph(G, hidden_dim=hidden_dim, latent_dim=latent_dim,
                              epochs=epochs, lr=lr, features=features, seed=SEED)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_pca(G, nodes, features=None, n_components=2):
    """PCA control: PCA on 6 centrality features.

    Parameters
    ----------
    features : np.ndarray, optional
        Pre-computed centrality features.  When *None*, they are computed
        internally (backward compatible).
    n_components : int, default 2
        Number of principal components to return.
    """
    if features is None:
        features = compute_centrality_features(G, nodes)
    features_centered = features - features.mean(axis=0)
    cov = features_centered.T @ features_centered / (len(nodes) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    coords = features_centered @ eigvecs[:, -n_components:]
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_method_by_name(G, nodes, method_name, features=None, n_components=2, **kwargs):
    """Factory to compute any supported embedding at arbitrary dimensionality.

    Parameters
    ----------
    G : nx.Graph
    nodes : list
    method_name : str
        One of DM, MDS, Spectral, DeepWalk, Node2Vec, VGAE, VGAE-feat, PCA.
    features : np.ndarray, optional
        Pre-computed centrality features (for DM, VGAE-feat, PCA).
    n_components : int, default 2
        Embedding dimensionality.
    **kwargs
        Additional method-specific parameters (e.g., p, q for Node2Vec).

    Returns
    -------
    np.ndarray
        Embedding coordinates (n_nodes, n_components).
    """
    method_name = method_name.strip()
    if method_name == "DM":
        return embed_diffusion_map(G, nodes, features=features, n_components=n_components)
    elif method_name == "MDS":
        return embed_mds(G, nodes, n_components=n_components)
    elif method_name == "Spectral":
        return embed_spectral(G, nodes, n_components=n_components)
    elif method_name == "DeepWalk":
        return embed_deepwalk(G, nodes, n_components=n_components, **kwargs)
    elif method_name == "Node2Vec":
        return embed_node2vec(G, nodes, n_components=n_components, **kwargs)
    elif method_name == "VGAE":
        return embed_vgae(G, nodes, features=None, latent_dim=n_components, **kwargs)
    elif method_name == "VGAE-feat":
        return embed_vgae(G, nodes, features=features, latent_dim=n_components, **kwargs)
    elif method_name == "PCA":
        return embed_pca(G, nodes, features=features, n_components=n_components)
    else:
        raise ValueError(f"Unknown embedding method: {method_name}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    data_dir = get_data_dir()
    emb_dir = get_embeddings_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    # Load network
    print("Loading curated 153-node network...")
    G, nodes, go_map = load_curated_network(data_dir)
    print(f"Network: {len(nodes)} nodes, {G.number_of_edges()} edges")
    
    # Compute centrality features once (used by DM, VGAE-feat, PCA)
    print("Computing centrality features...")
    features = compute_centrality_features(G, nodes)
    
    # Compute all embeddings
    methods = {
        "DM": lambda: embed_diffusion_map(G, nodes, features=features),
        "MDS": lambda: embed_mds(G, nodes),
        "Spectral": lambda: embed_spectral(G, nodes),
        "DeepWalk": lambda: embed_deepwalk(G, nodes),
        "Node2Vec": lambda: embed_node2vec(G, nodes),
        "VGAE": lambda: embed_vgae(G, nodes, features=None),
        "VGAE-feat": lambda: embed_vgae(G, nodes, features=features),
        "PCA": lambda: embed_pca(G, nodes, features=features),
    }
    
    for method_name, embed_fn in methods.items():
        print(f"\nComputing {method_name}...")
        random.seed(SEED)
        np.random.seed(SEED)
        try:
            coords = embed_fn()
            save_embedding(coords, nodes, method_name, "153", emb_dir)
            print(f"  {method_name}: std={np.std(coords):.4f}, shape={coords.shape}")
        except Exception as e:
            print(f"  {method_name} FAILED: {e}")
    
    print("\nAll embeddings computed!")


if __name__ == "__main__":
    main()
