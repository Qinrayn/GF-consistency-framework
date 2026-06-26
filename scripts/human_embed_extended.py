#!/usr/bin/env python3
"""
human_embed_extended.py
Step 33a: Extended Human Embedding — Compute 5 additional methods
(PCA, VGAE-feat, GraphSAGE, GAT, GIN) on the human STRING network.

Combined with the existing 6 methods (DM, MDS, Spectral, DeepWalk,
Node2Vec, VGAE), this brings the human network to **11 methods**,
enabling a full cross-species comparison (11 shared methods).

Design decisions:
  - PCA: centrality features → top-2 principal components.
  - VGAE-feat: VGAE with 6 centrality features (vs topology-only VGAE).
  - GNN methods (GraphSAGE, GAT, GIN): sparse negative-sampling BCE
    loss to avoid the O(n²) dense adjacency matrix used in the
    original training loop (prohibitive at n ≈ 16 000).

Output
------
  data/human_pca_embedding.json
  data/human_vgae-feat_embedding.json
  data/human_graphsage_embedding.json
  data/human_gat_embedding.json
  data/human_gin_embedding.json
"""
from __future__ import annotations

import os
import sys
import json
import gzip
import time
import numpy as np
import networkx as nx
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import (
    SEED,
    TARGET_STD,
    compute_centrality_features,
    rescale_coordinates,
    vgae_from_graph,
)

# ---- Configuration ----
HUMAN_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "human_validation")
LINKS_FILE = os.path.join(HUMAN_DATA_DIR, "9606.protein.links.v12.0.txt.gz")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCORE_THRESHOLD = 700

# GNN hyper-parameters
GNN_HIDDEN_DIM = 16
GNN_LATENT_DIM = 2
GNN_EPOCHS = 200          # slightly fewer epochs for speed
GNN_LR = 0.01
NEG_SAMPLES_PER_EDGE = 1  # ratio of negative : positive samples


# ============================================================
# Network loading (same as human_embed_all.py)
# ============================================================

def load_human_network():
    """Load human STRING network, filter by score, take largest CC."""
    print("Loading human STRING network...")
    edges = []
    with gzip.open(LINKS_FILE, "rt", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                score = int(parts[2])
                if score >= SCORE_THRESHOLD:
                    edges.append((parts[0], parts[1], score))
    print(f"  Loaded {len(edges)} edges with score >= {SCORE_THRESHOLD}")

    G = nx.Graph()
    for p1, p2, score in edges:
        G.add_edge(p1, p2, weight=score)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"  Largest CC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ============================================================
# Embedding helpers
# ============================================================

def detect_and_remove_outliers(coords, node_list, method_name, threshold=100):
    """Remove nodes whose coordinates exceed *threshold* standard deviations."""
    x_std = np.std(coords[:, 0])
    y_std = np.std(coords[:, 1])
    clean_mask = np.ones(len(node_list), dtype=bool)
    outliers = []
    for i, node in enumerate(node_list):
        x_dev = abs(coords[i, 0] - np.mean(coords[:, 0])) / max(x_std, 1e-10)
        y_dev = abs(coords[i, 1] - np.mean(coords[:, 1])) / max(y_std, 1e-10)
        if x_dev > threshold or y_dev > threshold:
            outliers.append(node)
            clean_mask[i] = False
    if outliers:
        print(f"  [{method_name}] Removed {len(outliers)} outlier(s)")
    return coords[clean_mask], [n for n, m in zip(node_list, clean_mask) if m]


def save_embedding(coords, node_list, method_name):
    """Save embedding in the standard human JSON format."""
    embedding = {}
    for i, node in enumerate(node_list):
        embedding[node] = {"x": float(coords[i, 0]), "y": float(coords[i, 1])}
    out_file = os.path.join(OUTPUT_DIR, f"human_{method_name.lower()}_embedding.json")
    with open(out_file, "w") as f:
        json.dump(embedding, f, indent=2)
    print(f"  Saved: {out_file} ({len(embedding)} nodes)")
    return out_file


# ============================================================
# Sparse GNN training (memory-efficient for large graphs)
# ============================================================

def _train_gnn_sparse(G, encoder_builder, hidden_dim=16, latent_dim=2,
                      epochs=200, lr=0.01, features=None, seed=42,
                      neg_ratio=1):
    """Train a 2-layer GNN encoder with sparse negative-sampling BCE loss.

    Instead of computing the full n×n adjacency reconstruction loss
    (which is O(n²) memory and compute), this function:
      1.  Uses edge_index as positive samples.
      2.  Samples an equal number of negative (non-edge) pairs.
      3.  Computes BCE only on the sampled pairs.

    This reduces per-epoch cost from O(n²) to O(|E|).
    """
    import torch
    import torch.nn.functional as F
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
        data.x = torch.eye(n)
        in_dim = n

    model = encoder_builder(in_dim, hidden_dim, latent_dim, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    edge_index = data.edge_index
    n_pos = edge_index.shape[1]
    n_neg = n_pos * neg_ratio

    # Pre-generate negative sample pool (random node pairs, not in edges)
    rng_np = np.random.RandomState(seed)
    edge_set = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        edge_set.add((min(u, v), max(u, v)))

    neg_pairs = []
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 10:
        u = rng_np.randint(0, n)
        v = rng_np.randint(0, n)
        if u != v and (min(u, v), max(u, v)) not in edge_set:
            neg_pairs.append((u, v))
        attempts += 1
    neg_pairs = np.array(neg_pairs, dtype=np.int64)
    neg_edge_index = torch.tensor(neg_pairs.T, dtype=torch.long)

    # Labels
    pos_labels = torch.ones(n_pos)
    neg_labels = torch.zeros(len(neg_pairs))

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, edge_index)

        # Positive scores
        pos_src, pos_dst = edge_index[0], edge_index[1]
        pos_scores = torch.sigmoid(torch.sum(z[pos_src] * z[pos_dst], dim=1))

        # Negative scores
        neg_src, neg_dst = neg_edge_index[0], neg_edge_index[1]
        neg_scores = torch.sigmoid(torch.sum(z[neg_src] * z[neg_dst], dim=1))

        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])
        loss = F.binary_cross_entropy(all_scores, all_labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model(data.x, edge_index)
        coords = z.numpy()
    return coords


def _build_sage_encoder(in_dim, hidden_dim, latent_dim, use_bn=True):
    """Build GraphSAGE encoder (same architecture as embed_gnn.py)."""
    import torch.nn as nn
    import torch.nn.functional as F_func
    from torch_geometric.nn import SAGEConv

    class SAGEEncoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, latent_dim, use_bn=True):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
            self.conv2 = SAGEConv(hidden_dim, latent_dim, aggr="mean")
            self.use_bn = use_bn
            if use_bn:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(latent_dim)

        def forward(self, x, edge_index):
            h = self.conv1(x, edge_index)
            if self.use_bn:
                h = self.bn1(h)
            h = F_func.relu(h)
            z = self.conv2(h, edge_index)
            if self.use_bn:
                z = self.bn2(z)
            return z

    return SAGEEncoder(in_dim, hidden_dim, latent_dim, use_bn)


def _build_gat_encoder(in_dim, hidden_dim, latent_dim, use_bn=True):
    """Build GAT encoder (same architecture as embed_gnn.py)."""
    import torch.nn as nn
    import torch.nn.functional as F_func
    from torch_geometric.nn import GATConv

    class GATEncoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, latent_dim, use_bn=True):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden_dim, heads=1, concat=False)
            self.conv2 = GATConv(hidden_dim, latent_dim, heads=1, concat=False)
            self.use_bn = use_bn
            if use_bn:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(latent_dim)

        def forward(self, x, edge_index):
            h = self.conv1(x, edge_index)
            if self.use_bn:
                h = self.bn1(h)
            h = F_func.relu(h)
            z = self.conv2(h, edge_index)
            if self.use_bn:
                z = self.bn2(z)
            return z

    return GATEncoder(in_dim, hidden_dim, latent_dim, use_bn)


def _build_gin_encoder(in_dim, hidden_dim, latent_dim, use_bn=True):
    """Build GIN encoder (same architecture as embed_gnn.py)."""
    import torch.nn as nn
    import torch.nn.functional as F_func
    from torch_geometric.nn import GINConv

    class GINEncoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, latent_dim, use_bn=True):
            super().__init__()
            mlp1_layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ]
            mlp1 = nn.Sequential(*[l for l in mlp1_layers
                                   if not isinstance(l, nn.Identity)])
            mlp2_layers = [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            ]
            mlp2 = nn.Sequential(*[l for l in mlp2_layers
                                   if not isinstance(l, nn.Identity)])
            self.conv1 = GINConv(mlp1)
            self.conv2 = GINConv(mlp2)
            self.use_bn = use_bn
            if use_bn:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(latent_dim)

        def forward(self, x, edge_index):
            h = self.conv1(x, edge_index)
            if self.use_bn:
                h = self.bn1(h)
            h = F_func.relu(h)
            z = self.conv2(h, edge_index)
            if self.use_bn:
                z = self.bn2(z)
            return z

    return GINEncoder(in_dim, hidden_dim, latent_dim, use_bn)


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    G = load_human_network()
    node_list = list(G.nodes())
    n = len(node_list)

    # ---- Compute centrality features (shared by PCA, VGAE-feat, GNNs) ----
    print("\nComputing centrality features (degree, eigenvector, PageRank, ...) ...")
    t0 = time.time()
    features = compute_centrality_features(G, node_list)
    print(f"  Features shape: {features.shape}, elapsed: {time.time()-t0:.1f}s")

    results = {}

    # ---- 1. PCA ----
    print("\n[1/5] PCA on centrality features...")
    t0 = time.time()
    features_centered = features - features.mean(axis=0)
    cov = features_centered.T @ features_centered / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    coords = features_centered @ eigvecs[:, -2:]
    coords = rescale_coordinates(coords, TARGET_STD)
    coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "PCA")
    coords = rescale_coordinates(coords, TARGET_STD)
    save_embedding(coords, clean_nodes, "PCA")
    results["PCA"] = (coords, clean_nodes)
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- 2. VGAE-feat ----
    print("\n[2/5] VGAE-feat (VGAE with centrality features)...")
    t0 = time.time()
    try:
        coords = vgae_from_graph(G, hidden_dim=4, latent_dim=2, epochs=300,
                                  lr=0.01, features=features, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "VGAE-feat")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "VGAE-feat")
        results["VGAE-feat"] = (coords, clean_nodes)
    except Exception as e:
        print(f"  ERROR: {e}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- 3. GraphSAGE ----
    print("\n[3/5] GraphSAGE (sparse negative sampling)...")
    t0 = time.time()
    try:
        coords = _train_gnn_sparse(
            G, _build_sage_encoder,
            hidden_dim=GNN_HIDDEN_DIM, latent_dim=GNN_LATENT_DIM,
            epochs=GNN_EPOCHS, lr=GNN_LR,
            features=features, seed=SEED, neg_ratio=NEG_SAMPLES_PER_EDGE,
        )
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "GraphSAGE")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "GraphSAGE")
        results["GraphSAGE"] = (coords, clean_nodes)
    except Exception as e:
        print(f"  ERROR: {e}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- 4. GAT ----
    print("\n[4/5] GAT (sparse negative sampling)...")
    t0 = time.time()
    try:
        coords = _train_gnn_sparse(
            G, _build_gat_encoder,
            hidden_dim=GNN_HIDDEN_DIM, latent_dim=GNN_LATENT_DIM,
            epochs=GNN_EPOCHS, lr=GNN_LR,
            features=features, seed=SEED, neg_ratio=NEG_SAMPLES_PER_EDGE,
        )
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "GAT")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "GAT")
        results["GAT"] = (coords, clean_nodes)
    except Exception as e:
        print(f"  ERROR: {e}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- 5. GIN ----
    print("\n[5/5] GIN (sparse negative sampling)...")
    t0 = time.time()
    try:
        coords = _train_gnn_sparse(
            G, _build_gin_encoder,
            hidden_dim=GNN_HIDDEN_DIM, latent_dim=GNN_LATENT_DIM,
            epochs=GNN_EPOCHS, lr=GNN_LR,
            features=features, seed=SEED, neg_ratio=NEG_SAMPLES_PER_EDGE,
        )
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "GIN")
        coords = rescale_coordinates(coords, TARGET_STD)
        save_embedding(coords, clean_nodes, "GIN")
        results["GIN"] = (coords, clean_nodes)
    except Exception as e:
        print(f"  ERROR: {e}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("HUMAN EXTENDED EMBEDDINGS SUMMARY")
    print("=" * 60)
    for method, (coords, nodes) in results.items():
        print(f"  {method}: {len(nodes)} nodes, "
              f"x_std={np.std(coords[:, 0]):.4f}, y_std={np.std(coords[:, 1]):.4f}")
    print(f"\n{len(results)}/5 methods completed successfully.")


if __name__ == "__main__":
    main()
