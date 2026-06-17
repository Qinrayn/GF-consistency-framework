#!/usr/bin/env python3
"""
Step 52: Compute high-dimensional spectral embeddings for human and mouse PPI networks.

Current limitation: human/mouse spectral embeddings are 2D only (Fiedler pair).
This script computes d=64 spectral embeddings (eigenvectors 1..64 of normalized Laplacian)
to enable higher-dimensional cross-species functional conservation analysis.

Outputs:
    embeddings/human_spectral_d64.npy       (n_human, 64)
    embeddings/human_spectral_d64_nodes.json
    embeddings/mouse_spectral_d64.npy       (n_mouse, 64)
    embeddings/mouse_spectral_d64_nodes.json
    results/highdim_spectral_embeddings.json (summary statistics)
"""

import sys
import os
import json
import time
import gzip

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh as sparse_eigsh
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEED = 42
DIM = 64  # target embedding dimension (eigenvectors 1..64)
MIN_SCORE = 700  # STRING confidence threshold


def load_string_network(filepath, min_score=700):
    """Load PPI network from STRING gzipped file, extract largest CC."""
    print(f"  Loading {os.path.basename(filepath)}...")
    t0 = time.time()

    edges = []
    with gzip.open(filepath, "rt") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            p1, p2, score = parts[0], parts[1], int(parts[2])
            if score >= min_score and p1 != p2:
                edges.append((p1, p2, score))

    print(f"  {len(edges)} edges (score >= {min_score}) in {time.time()-t0:.1f}s")

    G_full = nx.Graph()
    for p1, p2, w in edges:
        G_full.add_edge(p1, p2, weight=w)

    # Largest connected component
    components = list(nx.connected_components(G_full))
    components.sort(key=len, reverse=True)
    lcc_nodes = components[0]
    G = G_full.subgraph(lcc_nodes).copy()

    print(f"  LCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"({len(components)} components)")
    return G


def compute_spectral_highdim(G, dim=64):
    """Compute d-dimensional spectral embedding via sparse eigendecomposition."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    k = dim + 1  # need eigenvectors 0..dim, use 1..dim

    print(f"  Computing normalized Laplacian ({n}x{n})...")
    t0 = time.time()
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).astype(np.float64)
    print(f"  Laplacian: {time.time()-t0:.1f}s, nnz={L.nnz}")

    print(f"  Sparse eigendecomposition (k={k})...")
    t0 = time.time()
    # shift-invert mode to find eigenvalues closest to 0
    eigvals, eigvecs = sparse_eigsh(L, k=k, sigma=0, which="LM")
    print(f"  Eigendecomposition: {time.time()-t0:.1f}s")

    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Skip trivial eigenvector (index 0), take eigenvectors 1..dim
    embedding = eigvecs[:, 1:dim+1]
    eigenvalues = eigvals[1:dim+1]

    # Verify: trivial eigenvalue should be ~0
    print(f"  Eigenvalue[0] (trivial): {eigvals[0]:.2e}")
    print(f"  Eigenvalue range used: [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]")

    # Spectral gap analysis
    gaps = np.diff(eigenvalues)
    print(f"  Fiedler value (lambda_2): {eigenvalues[0]:.6f}")
    print(f"  Min spectral gap: {gaps.min():.6f} (at k={gaps.argmin()+1})")
    print(f"  Max spectral gap: {gaps.max():.6f} (at k={gaps.argmax()+1})")

    # Effective dimensionality via participation ratio
    variances = np.var(embedding, axis=0)
    total_var = variances.sum()
    if total_var > 0:
        norm_var = variances / total_var
        pr = 1.0 / np.sum(norm_var ** 2)
        print(f"  Participation ratio: {pr:.2f} (of {dim} dims)")
    else:
        pr = 0

    return embedding, nodes, eigenvalues, pr


def save_results(embedding, nodes, eigenvalues, pr, species, output_dir):
    """Save embedding, nodes, and summary statistics."""
    npy_path = os.path.join(output_dir, f"{species}_spectral_d{DIM}.npy")
    nodes_path = os.path.join(output_dir, f"{species}_spectral_d{DIM}_nodes.json")

    np.save(npy_path, embedding)
    with open(nodes_path, "w") as f:
        json.dump(nodes, f)

    print(f"  Saved: {npy_path} ({embedding.shape})")
    print(f"  Saved: {nodes_path} ({len(nodes)} nodes)")

    return {
        "npy_path": npy_path,
        "nodes_path": nodes_path,
        "shape": list(embedding.shape),
        "n_nodes": len(nodes),
        "eigenvalues": eigenvalues.tolist(),
        "participation_ratio": float(pr),
        "fiedler_value": float(eigenvalues[0]),
        "spectral_gaps": np.diff(eigenvalues).tolist(),
    }


def main():
    np.random.seed(SEED)
    project_dir = os.path.join(os.path.dirname(__file__), "..")
    emb_dir = os.path.join(project_dir, "embeddings")
    res_dir = os.path.join(project_dir, "results")
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Human network
    human_file = os.path.join(project_dir, "human_validation",
                              "9606.protein.links.v12.0.txt.gz")
    # Mouse network
    mouse_file = os.path.join(project_dir, "data",
                              "10090.protein.links.v11.5.txt.gz")

    summary = {}
    total_t0 = time.time()

    # ---- Human ----
    print("=" * 60)
    print("HUMAN (Homo sapiens) - STRING v12.0")
    print("=" * 60)
    t0 = time.time()
    G_human = load_string_network(human_file, MIN_SCORE)
    emb_h, nodes_h, eigvals_h, pr_h = compute_spectral_highdim(G_human, DIM)
    summary["human"] = save_results(emb_h, nodes_h, eigvals_h, pr_h,
                                     "human", emb_dir)
    summary["human"]["load_time_s"] = time.time() - t0
    print(f"  Total human: {time.time()-t0:.1f}s\n")

    # ---- Mouse ----
    print("=" * 60)
    print("MOUSE (Mus musculus) - STRING v11.5")
    print("=" * 60)
    t0 = time.time()
    G_mouse = load_string_network(mouse_file, MIN_SCORE)
    emb_m, nodes_m, eigvals_m, pr_m = compute_spectral_highdim(G_mouse, DIM)
    summary["mouse"] = save_results(emb_m, nodes_m, eigvals_m, pr_m,
                                     "mouse", emb_dir)
    summary["mouse"]["load_time_s"] = time.time() - t0
    print(f"  Total mouse: {time.time()-t0:.1f}s\n")

    # ---- Cross-species comparison ----
    print("=" * 60)
    print("Cross-species spectral diagnostics")
    print("=" * 60)

    # Compare Fiedler values and participation ratios
    print(f"  Human: Fiedler={summary['human']['fiedler_value']:.6f}, "
          f"PR={summary['human']['participation_ratio']:.2f}, "
          f"N={summary['human']['n_nodes']}")
    print(f"  Mouse: Fiedler={summary['mouse']['fiedler_value']:.6f}, "
          f"PR={summary['mouse']['participation_ratio']:.2f}, "
          f"N={summary['mouse']['n_nodes']}")

    # Compare eigenvalue decay
    eig_h = np.array(summary["human"]["eigenvalues"])
    eig_m = np.array(summary["mouse"]["eigenvalues"])
    # Spectral entropy (normalized)
    def spectral_entropy(eigenvalues):
        """Shannon entropy of normalized eigenvalue distribution."""
        e = np.abs(eigenvalues)
        e = e / e.sum()
        e = e[e > 0]
        return -np.sum(e * np.log(e))

    se_h = spectral_entropy(eig_h)
    se_m = spectral_entropy(eig_m)
    max_entropy = np.log(DIM)
    print(f"  Spectral entropy: human={se_h:.4f}, mouse={se_m:.4f} "
          f"(max={max_entropy:.4f})")

    summary["cross_species"] = {
        "human_spectral_entropy": float(se_h),
        "mouse_spectral_entropy": float(se_m),
        "max_entropy": float(max_entropy),
        "fiedler_ratio": float(summary["human"]["fiedler_value"] /
                               max(summary["mouse"]["fiedler_value"], 1e-10)),
        "pr_ratio": float(summary["human"]["participation_ratio"] /
                          max(summary["mouse"]["participation_ratio"], 1e-10)),
    }

    # Save summary
    summary_path = os.path.join(res_dir, "highdim_spectral_embeddings.json")
    # Convert any numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=convert)
    print(f"\n  Summary saved: {summary_path}")
    print(f"  Total time: {time.time()-total_t0:.1f}s")


if __name__ == "__main__":
    main()
