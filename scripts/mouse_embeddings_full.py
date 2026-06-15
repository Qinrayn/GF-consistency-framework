#!/usr/bin/env python3
"""
Phase 10B-v2: Mouse Embedding (Full Network)
===============================================
Compute all 11 embeddings on the FULL mouse STRING network (~16K nodes),
matching the human pipeline (embed full, subsample later for GF).

Memory-aware implementations for the two heaviest methods:
  - MDS: Landmark MDS with 500 landmarks + Nystrom extension
         (avoids O(n^2) memory of full all-pairs BFS + eigendecomposition)
  - VGAE / VGAE-feat: sparse negative-sampling BCE loss
         (avoids O(n^2) dense adjacency reconstruction)

These produce mathematically comparable results to the full implementations
while staying within 16 GB RAM on CPU.

Output: data/mouse_{method}_embedding.json (11 files)
"""

import json
import sys
import time
import warnings
import gzip
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, TARGET_STD,
    build_similarity_matrix,
    diffusion_map_from_similarity,
    spectral_embedding_from_graph,
    deepwalk_from_graph,
    node2vec_from_graph,
    compute_centrality_features,
    rescale_coordinates,
)

from human_embed_extended import (
    _train_gnn_sparse,
    _build_sage_encoder,
    _build_gat_encoder,
    _build_gin_encoder,
)

DATA_DIR = SCRIPT_DIR.parent / "data"
SCORE_THRESHOLD = 700
OUTLIER_THRESHOLD = 100
LANDMARK_COUNT = 500
BANNER = "=" * 70


# ============================================================
# Network + helpers
# ============================================================

def load_mouse_network():
    """Load full mouse STRING network, score >= 700, largest CC."""
    string_file = DATA_DIR / "10090.protein.links.v11.5.txt.gz"
    G = nx.Graph()
    with gzip.open(str(string_file), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and int(parts[2]) >= SCORE_THRESHOLD:
                G.add_edge(parts[0], parts[1])
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    return G


def detect_and_remove_outliers(coords, node_list, method_name):
    x_std = np.std(coords[:, 0])
    y_std = np.std(coords[:, 1])
    x_mean = np.mean(coords[:, 0])
    y_mean = np.mean(coords[:, 1])
    clean_mask = np.ones(len(node_list), dtype=bool)
    n_out = 0
    for i in range(len(node_list)):
        xd = abs(coords[i, 0] - x_mean) / max(x_std, 1e-10)
        yd = abs(coords[i, 1] - y_mean) / max(y_std, 1e-10)
        if xd > OUTLIER_THRESHOLD or yd > OUTLIER_THRESHOLD:
            clean_mask[i] = False
            n_out += 1
    if n_out > 0:
        print(f"    [{method_name}] Removed {n_out} outlier(s)")
    return coords[clean_mask], [n for n, m in zip(node_list, clean_mask) if m]


def save_embedding(coords, node_list, method_name):
    embedding = {}
    for i, node in enumerate(node_list):
        embedding[node] = {"x": float(coords[i, 0]), "y": float(coords[i, 1])}
    out_file = DATA_DIR / f"mouse_{method_name.lower()}_embedding.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(embedding, f, indent=2, ensure_ascii=False)
    print(f"    Saved {out_file.name} ({len(embedding)} nodes)")


# ============================================================
# Landmark MDS (memory-efficient alternative to full MDS)
# ============================================================

def landmark_mds(G, node_list, n_landmarks=500, d=2, seed=42):
    """Landmark MDS: BFS from *n_landmarks* nodes + Nystrom extension.

    Instead of computing the full n*n distance matrix (O(n^2) memory),
    we compute distances from only *n_landmarks* nodes and use the
    Nystrom formula to embed the remaining nodes.

    Parameters
    ----------
    G : nx.Graph
    node_list : list of node IDs
    n_landmarks : number of BFS source nodes
    d : embedding dimension (2)
    seed : random seed for landmark selection
    """
    rng = np.random.RandomState(seed)
    n = len(node_list)
    lm_indices = rng.choice(n, size=min(n_landmarks, n), replace=False)

    node_to_idx = {node: i for i, node in enumerate(node_list)}
    lm_nodes = [node_list[i] for i in lm_indices]
    n_lm = len(lm_nodes)

    # BFS from each landmark
    print(f"    BFS from {n_lm} landmarks...", flush=True)
    t0 = time.time()
    D_lm = np.zeros((n_lm, n), dtype=np.float64)
    for li, src in enumerate(lm_nodes):
        lengths = nx.single_source_shortest_path_length(G, src)
        for node, length in lengths.items():
            j = node_to_idx.get(node)
            if j is not None:
                D_lm[li, j] = length
        if (li + 1) % 100 == 0:
            print(f"      {li+1}/{n_lm} ({time.time()-t0:.0f}s)", flush=True)

    # Unreached nodes: replace 0 with max_finite + 1
    max_finite = np.max(D_lm[D_lm > 0]) if np.any(D_lm > 0) else 1
    D_lm[D_lm == 0] = max_finite + 1
    # Fix: landmarks have distance 0 to themselves
    for li, idx in enumerate(lm_indices):
        D_lm[li, idx] = 0.0

    print(f"    BFS done ({time.time()-t0:.0f}s). Computing landmark MDS...", flush=True)

    # Classical MDS on landmark-landmark distances
    D_ll = D_lm[:, lm_indices]
    D_ll_sq = D_ll ** 2

    # Double centering
    row_mean = D_ll_sq.mean(axis=1, keepdims=True)
    col_mean = D_ll_sq.mean(axis=0, keepdims=True)
    grand_mean = D_ll_sq.mean()
    B_ll = -0.5 * (D_ll_sq - row_mean - col_mean + grand_mean)

    # Eigendecomposition (small: n_lm x n_lm)
    eigvals, eigvecs = np.linalg.eigh(B_ll)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep top-d positive eigenvalues
    pos_mask = eigvals > 1e-10
    n_pos = min(np.sum(pos_mask), d)
    if n_pos < d:
        print(f"    Warning: only {n_pos} positive eigenvalues")
        n_pos = max(n_pos, 1)

    Lambda = eigvals[:n_pos]
    V = eigvecs[:, :n_pos]

    # Landmark coordinates
    L_coords = V * np.sqrt(Lambda)

    # Nystrom extension: embed all n nodes
    # x_i = 0.5 * Lambda^{-1} V^T (delta_landmark - d_i^2)
    delta = D_ll_sq.mean(axis=0)  # (n_lm,) mean sq dist per column
    D_all_sq = D_lm ** 2           # (n_lm, n)

    delta_mat = delta[:, np.newaxis]  # (n_lm, 1)
    diff = delta_mat - D_all_sq       # (n_lm, n)

    # W = V * Lambda^{-1}  (n_lm x n_pos)
    W = V / Lambda[np.newaxis, :]
    # coords = W^T @ diff  (n_pos x n) -> transpose to (n, n_pos)
    coords = (0.5 * W.T @ diff).T

    # Pad if n_pos < d
    if coords.shape[1] < d:
        pad = np.zeros((n, d - coords.shape[1]))
        coords = np.hstack([coords, pad])

    return coords


# ============================================================
# Sparse VGAE (memory-efficient for large n)
# ============================================================

def sparse_vgae(G, hidden_dim=4, latent_dim=2, epochs=300, lr=0.01,
                features=None, seed=42, neg_ratio=1):
    """VGAE with sparse negative-sampling BCE loss.

    Replaces the O(n^2) dense adjacency reconstruction with edge-based
    positive + negative sampling, reducing per-epoch cost from O(n^2)
    to O(|E|). Same encoder architecture as utils.vgae_from_graph.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import from_networkx

    torch.manual_seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    n = len(nodes)
    data = from_networkx(G)

    if features is not None:
        data.x = torch.tensor(features, dtype=torch.float32)
        in_dim = features.shape[1]
    else:
        data.x = torch.eye(n, dtype=torch.float32)
        in_dim = n

    class Encoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, latent_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv_mu = GCNConv(hidden_dim, latent_dim)
            self.conv_logvar = GCNConv(hidden_dim, latent_dim)

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            return self.conv_mu(h, edge_index), self.conv_logvar(h, edge_index)

    model = Encoder(in_dim, hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    edge_index = data.edge_index
    n_pos = edge_index.shape[1]

    # Negative edge sampling
    rng_np = np.random.RandomState(seed)
    edge_set = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        edge_set.add((min(u, v), max(u, v)))

    n_neg = n_pos * neg_ratio
    neg_pairs = []
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 10:
        u = rng_np.randint(0, n)
        v = rng_np.randint(0, n)
        if u != v and (min(u, v), max(u, v)) not in edge_set:
            neg_pairs.append((u, v))
        attempts += 1

    neg_pairs = np.array(neg_pairs, dtype=np.int64).reshape(-1, 2)
    if len(neg_pairs) == 0:
        neg_pairs = np.array([[0, 1]], dtype=np.int64)

    pos_src = edge_index[0]
    pos_dst = edge_index[1]
    neg_src = torch.tensor(neg_pairs[:, 0], dtype=torch.long)
    neg_dst = torch.tensor(neg_pairs[:, 1], dtype=torch.long)

    pos_labels = torch.ones(n_pos)
    neg_labels = torch.zeros(len(neg_pairs))
    all_labels = torch.cat([pos_labels, neg_labels])

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        mu, logvar = model(data.x, edge_index)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        # Sparse edge scores
        pos_scores = torch.sigmoid(
            torch.sum(z[pos_src] * z[pos_dst], dim=1))
        neg_scores = torch.sigmoid(
            torch.sum(z[neg_src] * z[neg_dst], dim=1))
        all_scores = torch.cat([pos_scores, neg_scores])

        recon_loss = F.binary_cross_entropy(all_scores, all_labels,
                                            reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"      Epoch {epoch+1}/{epochs}, loss={loss.item():.2f}",
                  flush=True)

    model.eval()
    with torch.no_grad():
        mu, _ = model(data.x, edge_index)
    return mu.numpy()


# ============================================================
# Main pipeline
# ============================================================

def run():
    print(BANNER)
    print("Phase 10B-v2: Mouse Embedding (Full Network)")
    print(BANNER)

    # Load full network
    print("\nLoading mouse STRING network...")
    G = load_mouse_network()
    node_list = list(G.nodes())
    n = len(node_list)
    print(f"  Network: {n} nodes, {G.number_of_edges()} edges")

    # Centrality features
    print("\nComputing centrality features...")
    t0 = time.time()
    features = compute_centrality_features(G, node_list)
    print(f"  Features: {features.shape}, {time.time()-t0:.1f}s")

    results = {}

    # ---- 1. DM (memory-efficient sparse eigendecomposition) ----
    print("\n[1/11] Diffusion Map (sparse eigsh)...")
    t0 = time.time()
    try:
        from scipy.sparse.linalg import eigsh

        feat_norm = normalize(features, norm="l2", axis=0)
        # build_similarity_matrix: features @ features.T (n x n)
        sim = feat_norm @ feat_norm.T

        # Normalize: D^{-1/2} S D^{-1/2} via broadcasting (no diag matrix)
        row_sums = sim.sum(axis=1)
        d_inv_sqrt = 1.0 / (np.sqrt(row_sums) + 1e-10)
        sim *= d_inv_sqrt[:, np.newaxis]
        sim *= d_inv_sqrt[np.newaxis, :]

        # Sparse eigendecomposition: only top-3 eigenvectors
        eigvals, eigvecs = eigsh(sim, k=3, which="LM")
        idx = np.argsort(eigvals)
        coords = eigvecs[:, idx[-2::-1][:2]]  # 2nd and 3rd largest

        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "DM")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "DM")
        results["DM"] = len(clean_nodes)
        print(f"    {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 2. MDS (landmark) ----
    print(f"\n[2/11] Landmark MDS ({LANDMARK_COUNT} landmarks)...")
    t0 = time.time()
    try:
        coords = landmark_mds(G, node_list, n_landmarks=LANDMARK_COUNT,
                              d=2, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "MDS")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "MDS")
        results["MDS"] = len(clean_nodes)
        print(f"    {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 3. Spectral (sparse Laplacian + eigsh) ----
    print("\n[3/11] Spectral Embedding (sparse eigsh)...")
    t0 = time.time()
    try:
        from scipy.sparse.linalg import eigsh as sparse_eigsh

        L = nx.normalized_laplacian_matrix(G).astype(np.float64)
        # Shift-invert: find eigenvalues closest to 0
        eigvals, eigvecs = sparse_eigsh(L, k=3, sigma=0, which="LM")
        idx = np.argsort(eigvals)
        coords = eigvecs[:, idx[1:3]]  # 2nd and 3rd smallest (skip trivial)

        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "Spectral")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "Spectral")
        results["Spectral"] = len(clean_nodes)
        print(f"    {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 4. DeepWalk ----
    print("\n[4/11] DeepWalk...")
    t0 = time.time()
    try:
        coords = deepwalk_from_graph(G, walk_length=20, walks_per_node=10,
                                      window_size=5, dimensions=2, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "DeepWalk")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "DeepWalk")
        results["DeepWalk"] = len(clean_nodes)
        print(f"    {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 5. Node2Vec ----
    print("\n[5/11] Node2Vec...")
    t0 = time.time()
    try:
        coords = node2vec_from_graph(G, walk_length=20, walks_per_node=10,
                                      window_size=5, dimensions=2,
                                      p=0.5, q=2.0, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "Node2Vec")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "Node2Vec")
        results["Node2Vec"] = len(clean_nodes)
        print(f"    {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 6. VGAE (sparse, one-hot) ----
    print("\n[6/11] VGAE (sparse neg-sampling, one-hot features)...")
    t0 = time.time()
    try:
        coords = sparse_vgae(G, hidden_dim=4, latent_dim=2, epochs=300,
                             lr=0.01, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "VGAE")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "VGAE")
        results["VGAE"] = len(clean_nodes)
        print(f"    {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 7. PCA ----
    print("\n[7/11] PCA...")
    t0 = time.time()
    try:
        fc = features - features.mean(axis=0)
        cov = fc.T @ fc / (n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        coords = fc @ eigvecs[:, -2:]
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "PCA")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "PCA")
        results["PCA"] = len(clean_nodes)
        print(f"    {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 8. VGAE-feat (sparse) ----
    print("\n[8/11] VGAE-feat (sparse neg-sampling, centrality features)...")
    t0 = time.time()
    try:
        coords = sparse_vgae(G, hidden_dim=4, latent_dim=2, epochs=300,
                             lr=0.01, features=features, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "VGAE-feat")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "VGAE-feat")
        results["VGAE-feat"] = len(clean_nodes)
        print(f"    {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 9-11. GNN methods (sparse neg-sampling) ----
    for idx, (name, builder) in enumerate([
        ("GraphSAGE", _build_sage_encoder),
        ("GAT", _build_gat_encoder),
        ("GIN", _build_gin_encoder),
    ], start=9):
        print(f"\n[{idx}/11] {name} (sparse neg-sampling)...")
        t0 = time.time()
        try:
            coords = _train_gnn_sparse(
                G, builder,
                hidden_dim=16, latent_dim=2,
                epochs=200, lr=0.01,
                features=features, seed=SEED, neg_ratio=1,
            )
            coords = rescale_coordinates(coords, TARGET_STD)
            coords, clean_nodes = detect_and_remove_outliers(coords, node_list, name)
            coords = rescale_coordinates(coords, TARGET_STD)
            save_embedding(coords, clean_nodes, name)
            results[name] = len(clean_nodes)
            print(f"    {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Summary
    print(f"\n{BANNER}")
    print(f"Phase 10B-v2 complete: {len(results)}/11 methods")
    for m, count in results.items():
        print(f"  {m:<12}: {count} nodes")
    print(BANNER)


if __name__ == "__main__":
    run()
