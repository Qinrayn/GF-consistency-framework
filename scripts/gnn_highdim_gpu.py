#!/usr/bin/env python3
"""
gnn_highdim_gpu.py
===================
GPU-accelerated high-dimensional GNN embedding training + G-F Score evaluation.

Trains VGAE, GraphSAGE, GAT, GIN at d = {8, 16, 32, 64} using CUDA (A800).
Previously these were stuck at d=2 due to laptop CPU limitations.
With A800 GPU, training at d=64 takes seconds instead of hours.

Outputs:
  results/gnn_highdim_gpu.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, GF_R_MIN, GF_R_MAX, TARGET_STD,
    get_data_dir, get_results_dir, get_figures_dir,
    load_curated_network, compute_centrality_features,
    compute_gf_curve, compute_gf_score, compute_plateau_width,
    rescale_coordinates,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()

GNN_METHODS = ["VGAE", "GraphSAGE", "GAT", "GIN"]
DIMENSIONS = [8, 16, 32, 64]
EPOCHS = 300
LR = 0.01
HIDDEN_DIM = 64  # larger hidden dim for high-d training


def get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  WARNING: No GPU available, falling back to CPU")
    return device


def train_gnn_autoencoder(G, nodes, features, latent_dim, method_name, device,
                          epochs=EPOCHS, lr=LR, hidden_dim=HIDDEN_DIM, seed=SEED):
    """Train a GNN autoencoder at specified latent dimension using GPU.

    Parameters
    ----------
    G : nx.Graph
    nodes : list
    features : np.ndarray or None
    latent_dim : int - output embedding dimension
    method_name : str - one of VGAE, GraphSAGE, GAT, GIN
    device : torch.device

    Returns
    -------
    np.ndarray - embedding coordinates (n_nodes, latent_dim)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.utils import from_networkx

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(nodes)
    data = from_networkx(G)

    if features is not None:
        data.x = torch.tensor(features, dtype=torch.float32)
        in_dim = features.shape[1]
    else:
        data.x = torch.eye(n)
        in_dim = n

    # Move data to GPU
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)

    # Build encoder
    if method_name == "GraphSAGE":
        from torch_geometric.nn import SAGEConv
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv(in_dim, hidden_dim)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, latent_dim)
            def forward(self, x, ei):
                h = F.relu(self.bn1(self.conv1(x, ei)))
                return self.conv2(h, ei)

    elif method_name == "GAT":
        from torch_geometric.nn import GATConv
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GATConv(in_dim, hidden_dim, heads=1, concat=False)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.conv2 = GATConv(hidden_dim, latent_dim, heads=1, concat=False)
            def forward(self, x, ei):
                h = F.relu(self.bn1(self.conv1(x, ei)))
                return self.conv2(h, ei)

    elif method_name == "GIN":
        from torch_geometric.nn import GINConv
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                mlp1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
                self.conv1 = GINConv(mlp1)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.conv2 = GINConv(mlp2)
            def forward(self, x, ei):
                h = F.relu(self.bn1(self.conv1(x, ei)))
                return self.conv2(h, ei)

    elif method_name == "VGAE":
        from torch_geometric.nn import GCNConv
        class VGAEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden_dim)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.conv_mu = GCNConv(hidden_dim, latent_dim)
                self.conv_logvar = GCNConv(hidden_dim, latent_dim)
            def forward(self, x, ei):
                h = F.relu(self.bn1(self.conv1(x, ei)))
                return self.conv_mu(h, ei), self.conv_logvar(h, ei)

        model = VGAEModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Adjacency target for reconstruction loss
        adj_target = torch.zeros(n, n, device=device)
        ei = data.edge_index
        adj_target[ei[0], ei[1]] = 1.0

        model.train()
        for ep in range(epochs):
            optimizer.zero_grad()
            mu, logvar = model(data.x, data.edge_index)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            adj_recon = torch.sigmoid(z @ z.T)
            recon_loss = F.binary_cross_entropy(adj_recon, adj_target, reduction="sum")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mu, _ = model(data.x, data.edge_index)
            coords = mu.cpu().numpy()
        return coords

    else:
        raise ValueError(f"Unknown method: {method_name}")

    # For non-VGAE methods (autoencoder with reconstruction loss)
    model = Encoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Adjacency target
    adj_target = torch.zeros(n, n, device=device)
    ei = data.edge_index
    adj_target[ei[0], ei[1]] = 1.0

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        adj_recon = torch.sigmoid(z @ z.T)
        loss = F.binary_cross_entropy(adj_recon, adj_target, reduction="sum")
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        coords = z.cpu().numpy()
    return coords


def main():
    import torch

    print("=" * 70)
    print("GNN High-Dimensional GPU Training")
    print(f"Methods: {GNN_METHODS} | Dimensions: {DIMENSIONS}")
    print("=" * 70)

    device = get_device()

    # Load network
    print("\n[1/3] Loading curated 153-node network...")
    G, nodes, go_map = load_curated_network(DATA)
    print(f"  {len(nodes)} nodes, {G.number_of_edges()} edges")

    # Centrality features
    features = compute_centrality_features(G, nodes)

    # r-grid for GF curves
    r_vals = np.linspace(0.05, 0.55, 100)

    # Train all (method, dim) combinations
    print(f"\n[2/3] Training {len(GNN_METHODS)} methods x {len(DIMENSIONS)} dimensions = "
          f"{len(GNN_METHODS) * len(DIMENSIONS)} embeddings on GPU...")

    all_results = {}
    total_t0 = time.time()

    for method in GNN_METHODS:
        all_results[method] = {}
        for d in DIMENSIONS:
            t0 = time.time()
            print(f"  {method} d={d}...", end="", flush=True)

            try:
                coords = train_gnn_autoencoder(
                    G, nodes, features, latent_dim=d,
                    method_name=method, device=device,
                    epochs=EPOCHS, lr=LR, hidden_dim=HIDDEN_DIM,
                )

                # For GF Score, we use the first 2 dimensions (for fair 2D comparison)
                # AND compute full-d GF Score
                coords_2d = coords[:, :2]
                coords_2d = rescale_coordinates(coords_2d, target_std=TARGET_STD)

                # GF curve on 2D projection
                purities_2d, _ = compute_gf_curve(coords_2d, nodes, go_map, r_vals)
                gf_2d = compute_gf_score(r_vals, purities_2d, GF_R_MIN, GF_R_MAX)

                # GF curve on full-d (first 2 PCA components of full embedding)
                from sklearn.decomposition import PCA
                if d > 2:
                    coords_pca2d = PCA(n_components=2).fit_transform(coords)
                    coords_pca2d = rescale_coordinates(coords_pca2d, target_std=TARGET_STD)
                    purities_full, _ = compute_gf_curve(coords_pca2d, nodes, go_map, r_vals)
                    gf_full = compute_gf_score(r_vals, purities_full, GF_R_MIN, GF_R_MAX)
                else:
                    gf_full = gf_2d

                elapsed = time.time() - t0
                print(f" GF(2d)={gf_2d:.4f}, GF(PCA-2d)={gf_full:.4f}  ({elapsed:.1f}s)")

                all_results[method][str(d)] = {
                    "gf_score_2d_slice": round(float(gf_2d), 6),
                    "gf_score_pca_2d": round(float(gf_full), 6),
                    "elapsed_s": round(elapsed, 1),
                }

            except Exception as e:
                elapsed = time.time() - t0
                print(f" FAILED: {e}  ({elapsed:.1f}s)")
                all_results[method][str(d)] = {"error": str(e)}

    total_elapsed = time.time() - total_t0
    print(f"\n  Total training time: {total_elapsed:.1f}s")

    # Load existing 2D scores for comparison
    existing_2d = {}
    for fname in ["results/gnn_gf_scores.json"]:
        fpath = RESULTS.parent / fname if not os.path.isabs(fname) else Path(fname)
        if fpath.exists():
            d = json.load(open(fpath))
            existing_2d.update(d.get("gf_scores", {}))

    # Save results
    output = {
        "analysis": "GNN High-Dimensional GPU Training",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "methods": GNN_METHODS,
        "dimensions": DIMENSIONS,
        "epochs": EPOCHS,
        "hidden_dim": HIDDEN_DIM,
        "network": {"nodes": len(nodes), "edges": G.number_of_edges()},
        "gf_interval": [GF_R_MIN, GF_R_MAX],
        "results": all_results,
        "existing_2d_scores": existing_2d,
        "total_time_s": round(total_elapsed, 1),
    }

    out_path = RESULTS / "gnn_highdim_gpu.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[3/3] Saved: {out_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Method':<12}", end="")
    for d in DIMENSIONS:
        print(f"  d={d:>2}(2d)  d={d:>2}(PCA)", end="")
    print(f"  d=2(orig)")
    print("-" * 100)
    for method in GNN_METHODS:
        print(f"{method:<12}", end="")
        for d in DIMENSIONS:
            r = all_results[method].get(str(d), {})
            gf_2d = r.get("gf_score_2d_slice", -1)
            gf_pca = r.get("gf_score_pca_2d", -1)
            print(f"  {gf_2d:8.4f}  {gf_pca:8.4f}", end="")
        orig = existing_2d.get(method, -1)
        print(f"  {orig:8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
