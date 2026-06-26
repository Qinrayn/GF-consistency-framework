#!/usr/bin/env python3
"""
Extended Embedding Methods (n≥25 for statistical power)
=======================================================
P0-1 follow-up: Expand method panel from n=11 to n≥25 for adequate
statistical power. Uses scikit-learn and scipy for methods that don't
require additional libraries.

New methods (14):
  12. Laplacian Eigenmap (scipy eigendecomposition)
  13. Isomap (sklearn.manifold)
  14. LLE (sklearn.manifold)
  15. Modified LLE (sklearn.manifold)
  16. Kernel PCA (sklearn.decomposition, RBF kernel)
  17. t-SNE 2D (sklearn.manifold)
  18. Random Projection (sklearn.random_projection)
  19. Degree Encoding (networkx centrality → 2D)
  20. SE+ (Spectral with eig 3,4 instead of 1,2)
  21. SE- (Spectral with eig 4,5)
  22. Graph Factorization (truncated SVD on adjacency)
  23. Heat Kernel Embedding (exp(-tL) top-2)
  24. PageRank Embedding (PageRank + eigenvector centrality → 2D)
  25. Gravity Embedding (degree * distance product)

Output: embeddings/{method}_153.npy + embeddings/{method}_153_nodes.json
        results/extended_gf_scores.json
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
import networkx as nx
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, set_seed,
    get_data_dir, get_embeddings_dir, get_results_dir,
    rescale_coordinates, compute_gf_curve, compute_gf_score,
    GF_R_MIN, GF_R_MAX, N_POINTS,
    BANNER,
)

DATA = get_data_dir()
EMB = get_embeddings_dir()
RESULTS = get_results_dir()

EDGELIST = DATA / "curated_153_ppi.edgelist"
GO_MAP = DATA / "gene_go_map.json"


def load_network():
    G = nx.read_edgelist(str(EDGELIST))
    nodes = sorted(G.nodes())
    return G, nodes


def load_go_map():
    with open(GO_MAP, encoding="utf-8") as f:
        return json.load(f)


def save_embedding(method, coords, nodes):
    np.save(str(EMB / f"{method}_153.npy"), coords)
    with open(EMB / f"{method}_153_nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f)


def compute_and_save_gf(method, coords, nodes, go_map, r_vals):
    rescaled = rescale_coordinates(coords, target_std=0.3)
    purities, _ = compute_gf_curve(rescaled, nodes, go_map, r_vals)
    score = compute_gf_score(r_vals, purities, r_min=GF_R_MIN, r_max=GF_R_MAX)
    return score, purities


# ============================================================
# New embedding methods
# ============================================================

def embed_laplacian_eigenmap(G, nodes, dim=2):
    """Laplacian Eigenmap via normalized Laplacian eigendecomposition."""
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    # Skip first (zero) eigenvector, take next 2
    coords = eigvecs[:, 1:dim+1].real
    return coords


def embed_isomap(G, nodes, dim=2):
    """Isomap using shortest-path distances."""
    from sklearn.manifold import Isomap
    dist = nx.floyd_warshall_numpy(G, nodelist=nodes)
    dist = np.array(dist, dtype=float)
    np.fill_diagonal(dist, 0)
    iso = Isomap(n_components=dim, metric="precomputed")
    coords = iso.fit_transform(dist)
    return coords


def embed_lle(G, nodes, dim=2):
    """Locally Linear Embedding."""
    from sklearn.manifold import LocallyLinearEmbedding
    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(float)
    lle = LocallyLinearEmbedding(n_components=dim, n_neighbors=min(10, len(nodes)-1),
                                  random_state=SEED)
    coords = lle.fit_transform(A)
    return coords


def embed_modified_lle(G, nodes, dim=2):
    """Modified LLE."""
    from sklearn.manifold import LocallyLinearEmbedding
    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(float)
    lle = LocallyLinearEmbedding(n_components=dim, n_neighbors=min(10, len(nodes)-1),
                                  method="modified", random_state=SEED)
    coords = lle.fit_transform(A)
    return coords


def embed_kernel_pca(G, nodes, dim=2):
    """Kernel PCA with RBF kernel on adjacency."""
    from sklearn.decomposition import KernelPCA
    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(float)
    kpca = KernelPCA(n_components=dim, kernel="rbf", random_state=SEED)
    coords = kpca.fit_transform(A)
    return coords


def embed_tsne(G, nodes, dim=2):
    """t-SNE 2D on adjacency."""
    from sklearn.manifold import TSNE
    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(float)
    tsne = TSNE(n_components=dim, random_state=SEED, perplexity=min(30, len(nodes)-1))
    coords = tsne.fit_transform(A)
    return coords


def embed_random_projection(G, nodes, dim=2):
    """Random projection of adjacency matrix."""
    from sklearn.random_projection import GaussianRandomProjection
    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(float)
    rp = GaussianRandomProjection(n_components=dim, random_state=SEED)
    coords = rp.fit_transform(A)
    return coords


def embed_degree_encoding(G, nodes, dim=2):
    """Degree-based 2D encoding: (degree, clustering_coefficient)."""
    degrees = np.array([G.degree(n) for n in nodes], dtype=float)
    clustering = np.array([nx.clustering(G, n) for n in nodes], dtype=float)
    coords = np.column_stack([degrees, clustering])
    return coords


def embed_spectral_plus(G, nodes, dim=2):
    """Spectral with eigenvectors 3,4 instead of 1,2."""
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    coords = eigvecs[:, 3:dim+3].real
    return coords


def embed_spectral_minus(G, nodes, dim=2):
    """Spectral with eigenvectors 4,5."""
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    coords = eigvecs[:, 4:dim+4].real
    return coords


def embed_graph_factorization(G, nodes, dim=2):
    """Truncated SVD on adjacency matrix."""
    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(float)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    coords = U[:, :dim] * np.sqrt(S[:dim])
    return coords


def embed_heat_kernel(G, nodes, dim=2, t=1.0):
    """Heat kernel embedding: exp(-tL) top-2 eigenvectors."""
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    heat = eigvecs * np.exp(-t * eigvals)
    coords = heat[:, 1:dim+1]  # skip first
    return coords


def embed_pagerank(G, nodes, dim=2):
    """PageRank + eigenvector centrality → 2D."""
    pr = np.array([nx.pagerank(G).get(n, 0) for n in nodes], dtype=float)
    ec = np.array([nx.eigenvector_centrality_numpy(G).get(n, 0) for n in nodes], dtype=float)
    coords = np.column_stack([pr, ec])
    return coords


def embed_gravity(G, nodes, dim=2):
    """Gravity embedding: degree × inverse shortest-path distance."""
    n = len(nodes)
    degrees = np.array([G.degree(n) for n in nodes], dtype=float)
    dist = nx.floyd_warshall_numpy(G, nodelist=nodes)
    dist = np.array(dist, dtype=float)
    np.fill_diagonal(dist, 1)  # avoid division by zero
    gravity = np.outer(degrees, degrees) / dist
    U, S, Vt = np.linalg.svd(gravity, full_matrices=False)
    coords = U[:, :dim] * np.sqrt(S[:dim])
    return coords


# ============================================================
# Main
# ============================================================

NEW_METHODS = {
    "Laplacian-EM": embed_laplacian_eigenmap,
    "Isomap": embed_isomap,
    "LLE": embed_lle,
    "Mod-LLE": embed_modified_lle,
    "KPCA": embed_kernel_pca,
    "tSNE-2D": embed_tsne,
    "RandProj": embed_random_projection,
    "DegreeEnc": embed_degree_encoding,
    "Spectral+": embed_spectral_plus,
    "Spectral-": embed_spectral_minus,
    "GraphFact": embed_graph_factorization,
    "HeatKernel": embed_heat_kernel,
    "PageRank": embed_pagerank,
    "Gravity": embed_gravity,
}


def main():
    print(BANNER)
    print("  Extended Embedding Methods (n→25 for statistical power)")
    print(BANNER)

    set_seed(SEED)
    G, nodes = load_network()
    go_map = load_go_map()
    r_vals = np.linspace(0.05, 0.55, N_POINTS)

    print(f"  Network: {len(nodes)} nodes, {G.number_of_edges()} edges")
    print(f"  GO annotations: {len(go_map)} genes")
    print(f"  New methods: {len(NEW_METHODS)}")

    results = {}
    for method_name, embed_fn in NEW_METHODS.items():
        print(f"\n  Computing {method_name}...")
        t0 = time.time()
        try:
            coords = embed_fn(G, nodes)
            if coords.shape[1] != 2:
                coords = coords[:, :2]
            # Handle NaN/Inf
            coords = np.nan_to_num(coords, nan=0.0, posinf=10.0, neginf=-10.0)
            save_embedding(method_name, coords, nodes)
            score, purities = compute_and_save_gf(method_name, coords, nodes, go_map, r_vals)
            results[method_name] = {
                "gf_score": float(score),
                "peak_purity": float(max(purities)),
                "shape": list(coords.shape),
                "time_sec": time.time() - t0,
            }
            print(f"    GF Score: {score:.4f}, peak: {max(purities):.4f}, "
                  f"shape: {coords.shape}, time: {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[method_name] = {"error": str(e)}

    # Save
    output = {
        "description": "Extended embedding methods for statistical power (n→25)",
        "n_new_methods": len(NEW_METHODS),
        "methods": list(NEW_METHODS.keys()),
        "results": results,
    }
    out_file = RESULTS / "extended_gf_scores.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to: {out_file}")

    # Summary
    successful = {m: r for m, r in results.items() if "gf_score" in r}
    print(f"\n  {len(successful)}/{len(NEW_METHODS)} methods computed successfully")
    if successful:
        sorted_methods = sorted(successful.items(), key=lambda x: -x[1]["gf_score"])
        print(f"\n  G-F Score ranking (new methods):")
        for m, r in sorted_methods:
            print(f"    {m}: {r['gf_score']:.4f}")

    print(BANNER)


if __name__ == "__main__":
    main()
