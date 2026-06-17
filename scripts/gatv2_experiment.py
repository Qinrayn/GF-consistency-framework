#!/usr/bin/env python3
"""
gatv2_experiment.py -- GATv2 vs GAT Collapse Comparison
========================================================

Tests whether GATv2 (Brody et al., 2022) avoids the collapse that GAT
(Velickovic et al., 2018) exhibits on the degree-heterogeneous PPI network.

Paper theorems predict for GAT:
  1. Attention degeneration: H_norm >= 0.97 (near-uniform attention)
  2. Effective rank collapse: eff_rank ~ 1.045
  3. GF Score bound: GF ~ 1.0 * random baseline (collapsed embeddings)

GATv2 modifies attention to a^T * LeakyReLU(W * [h_i || h_j]) which is
data-dependent rather than static, potentially avoiding degree-driven
uniformity.

Design
------
  - Load curated 153-node yeast PPI subnetwork
  - Train GATv2 and GAT with matching architectures (2 layers, 32 hidden)
  - Sweep output dimensions d in {2, 4, 8, 16, 32}
  - Test single-head (1h) and multi-head (4h) variants for GATv2
  - Compute: attention entropy, effective rank, G-F Score
  - Compare against random baseline (0.135) and spectral baseline (0.163)

Output
------
  results/gatv2_comparison.json
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir,
    load_curated_network, compute_centrality_features,
    rescale_coordinates, compute_gf_curve, compute_gf_score,
    check_embedding_collapse,
    GF_R_MIN, GF_R_MAX, R_MIN, R_MAX, TARGET_STD,
)

# Experiment-specific constants
DIMENSIONS = [2, 4, 8, 16, 32]
HIDDEN_DIM = 32
EPOCHS = 300
LR = 0.01
N_POINTS = 25
RANDOM_BASELINE_GF = 0.135
SPECTRAL_BASELINE_GF = 0.163


# ============================================================
# Attention Entropy (per-node, normalized by degree)
# ============================================================

def compute_per_node_attention_entropy(edge_index_np, alpha_np, n_nodes, degrees):
    """Compute per-node normalized attention entropy.

    For each node i with degree d_i, compute H(alpha_i) / log(d_i),
    where H is Shannon entropy of attention weights from node i's neighbors.

    Parameters
    ----------
    edge_index_np : (2, E) array
        edge_index[0] = source, edge_index[1] = target
    alpha_np : (E,) or (E, heads) array
        Attention weights (already softmax-ed by the GAT layer).
    n_nodes : int
    degrees : dict
        Node index -> degree (number of neighbors in the message-passing graph
        including self-loops if present).

    Returns
    -------
    float
        Mean normalized entropy across all nodes (1.0 = uniform attention).
    """
    if alpha_np.ndim == 2:
        # Multi-head: average entropy across heads
        entropies_per_head = []
        for h in range(alpha_np.shape[1]):
            ent = _per_node_entropy(edge_index_np, alpha_np[:, h], n_nodes, degrees)
            entropies_per_head.append(ent)
        return float(np.mean(entropies_per_head))
    else:
        return float(_per_node_entropy(edge_index_np, alpha_np, n_nodes, degrees))


def _per_node_entropy(edge_index_np, alpha_1d, n_nodes, degrees):
    """Per-node normalized entropy for a single attention head."""
    target_nodes = edge_index_np[1]  # messages flow TO target
    source_nodes = edge_index_np[0]  # messages flow FROM source

    # Group attention by target node (the node receiving messages)
    # For each target node i, collect attention weights from all sources
    node_entropies = []
    for i in range(n_nodes):
        mask = target_nodes == i
        if not mask.any():
            continue
        a = alpha_1d[mask]
        a = np.abs(a) + 1e-12
        a = a / a.sum()
        d_i = mask.sum()
        if d_i <= 1:
            continue
        h = -np.sum(a * np.log(a + 1e-12))
        h_norm = h / np.log(d_i)
        node_entropies.append(h_norm)

    if not node_entropies:
        return 1.0
    return float(np.mean(node_entropies))


# ============================================================
# Effective Rank (via SVD)
# ============================================================

def compute_effective_rank(coords):
    """Compute effective rank of an embedding matrix via SVD.

    effective_rank = exp(H(p))
    where p_i = s_i / sum(s_j), s = singular values, H = Shannon entropy.

    Returns 1.0 for rank-1 matrices, d for full-rank isotropic matrices.
    """
    s = np.linalg.svd(coords, compute_uv=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


# ============================================================
# GATv2 Model
# ============================================================

def train_gatv2(G, features, latent_dim, n_heads=1, hidden_dim=32,
                epochs=300, lr=0.01, seed=SEED):
    """Train a GATv2 encoder.

    Architecture:
      Layer 1: GATv2Conv(in_dim -> hidden_dim, heads=n_heads, concat if >1)
      BatchNorm + ReLU
      Layer 2: GATv2Conv(h_dim -> latent_dim, heads=1, concat=False)
      BatchNorm

    Training: adjacency reconstruction via BCE loss, gradient clipping (1.0),
    linear LR warmup (10 epochs).

    Returns
    -------
    dict with: coords, attention_entropy, effective_rank, final_loss
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv
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

    use_concat = n_heads > 1

    class GATv2Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=n_heads,
                                   concat=use_concat)
            self.bn1 = nn.BatchNorm1d(hidden_dim * n_heads if use_concat
                                      else hidden_dim)
            h_dim = hidden_dim * n_heads if use_concat else hidden_dim
            self.conv2 = GATv2Conv(h_dim, latent_dim, heads=1, concat=False)
            self.bn2 = nn.BatchNorm1d(latent_dim)

        def forward(self, x, edge_index, return_attention=False):
            h, alpha1 = self.conv1(x, edge_index,
                                   return_attention_weights=True)
            h = self.bn1(h)
            h = F.relu(h)
            z, alpha2 = self.conv2(h, edge_index,
                                   return_attention_weights=True)
            z = self.bn2(z)
            if return_attention:
                return z, alpha1, alpha2
            return z

    model = GATv2Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Adjacency target for reconstruction loss
    adj_target = torch.zeros(n, n)
    ei = data.edge_index
    adj_target[ei[0], ei[1]] = 1.0

    warmup_epochs = 10
    final_loss = None

    for epoch in range(epochs):
        # LR warmup
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        elif epoch == warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        adj_recon = torch.sigmoid(z @ z.T)
        loss = F.binary_cross_entropy(adj_recon, adj_target, reduction="sum")
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = float(loss.item())

    # Extract embeddings and attention weights
    model.eval()
    with torch.no_grad():
        z, alpha1, alpha2 = model(data.x, data.edge_index,
                                  return_attention=True)
        coords = z.numpy()

    # Compute degrees for normalization
    degrees = {}
    ei_np = data.edge_index.numpy()
    for i in range(n):
        degrees[i] = int((ei_np[1] == i).sum())

    # Per-node attention entropy (average across layers)
    ei_np_1 = alpha1[0].numpy()
    a_np_1 = alpha1[1].numpy().flatten() if alpha1[1].ndim > 1 else alpha1[1].numpy()
    ei_np_2 = alpha2[0].numpy()
    a_np_2 = alpha2[1].numpy().flatten() if alpha2[1].ndim > 1 else alpha2[1].numpy()

    # Handle multi-head attention weights
    a1_raw = alpha1[1].numpy()
    a2_raw = alpha2[1].numpy()

    ent_l1 = compute_per_node_attention_entropy(ei_np_1, a1_raw, n, degrees)
    ent_l2 = compute_per_node_attention_entropy(ei_np_2, a2_raw, n, degrees)
    avg_entropy = (ent_l1 + ent_l2) / 2.0

    # Effective rank
    eff_rank = compute_effective_rank(coords)

    return {
        "coords": coords,
        "attention_entropy": avg_entropy,
        "attention_entropy_l1": ent_l1,
        "attention_entropy_l2": ent_l2,
        "effective_rank": eff_rank,
        "final_loss": final_loss,
    }


# ============================================================
# GAT Model (baseline, same architecture)
# ============================================================

def train_gat(G, features, latent_dim, n_heads=1, hidden_dim=32,
              epochs=300, lr=0.01, seed=SEED):
    """Train a GAT (original) encoder with identical training protocol."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
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

    use_concat = n_heads > 1

    class GATModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden_dim, heads=n_heads,
                                 concat=use_concat)
            self.bn1 = nn.BatchNorm1d(hidden_dim * n_heads if use_concat
                                      else hidden_dim)
            h_dim = hidden_dim * n_heads if use_concat else hidden_dim
            self.conv2 = GATConv(h_dim, latent_dim, heads=1, concat=False)
            self.bn2 = nn.BatchNorm1d(latent_dim)

        def forward(self, x, edge_index, return_attention=False):
            h, alpha1 = self.conv1(x, edge_index,
                                   return_attention_weights=True)
            h = self.bn1(h)
            h = F.relu(h)
            z, alpha2 = self.conv2(h, edge_index,
                                   return_attention_weights=True)
            z = self.bn2(z)
            if return_attention:
                return z, alpha1, alpha2
            return z

    model = GATModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    adj_target = torch.zeros(n, n)
    ei = data.edge_index
    adj_target[ei[0], ei[1]] = 1.0

    warmup_epochs = 10
    final_loss = None

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        elif epoch == warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        adj_recon = torch.sigmoid(z @ z.T)
        loss = F.binary_cross_entropy(adj_recon, adj_target, reduction="sum")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        z, alpha1, alpha2 = model(data.x, data.edge_index,
                                  return_attention=True)
        coords = z.numpy()

    # Degrees
    degrees = {}
    ei_np = data.edge_index.numpy()
    for i in range(n):
        degrees[i] = int((ei_np[1] == i).sum())

    # Per-node attention entropy
    ei_np_1 = alpha1[0].numpy()
    ei_np_2 = alpha2[0].numpy()
    a1_raw = alpha1[1].numpy()
    a2_raw = alpha2[1].numpy()

    ent_l1 = compute_per_node_attention_entropy(ei_np_1, a1_raw, n, degrees)
    ent_l2 = compute_per_node_attention_entropy(ei_np_2, a2_raw, n, degrees)
    avg_entropy = (ent_l1 + ent_l2) / 2.0

    eff_rank = compute_effective_rank(coords)

    return {
        "coords": coords,
        "attention_entropy": avg_entropy,
        "attention_entropy_l1": ent_l1,
        "attention_entropy_l2": ent_l2,
        "effective_rank": eff_rank,
        "final_loss": final_loss,
    }


# ============================================================
# GF Score computation for an embedding
# ============================================================

def compute_gf_for_embedding(coords, nodes, go_map_sub, r_vals, latent_dim):
    """Compute G-F Score for an embedding, projecting to 2D if needed.

    Returns (gf_score, gf_raw, collapse_info).
    """
    # For GF curve we need 2D spatial graph. Project to 2D if dim > 2.
    if latent_dim > 2:
        from sklearn.decomposition import PCA
        coords_2d = PCA(n_components=2).fit_transform(coords)
    else:
        coords_2d = coords

    coords_2d = rescale_coordinates(coords_2d, TARGET_STD)
    collapse_info = check_embedding_collapse(coords_2d, "embedding")

    purities, _ = compute_gf_curve(coords_2d, nodes, go_map_sub, r_vals)
    gf_2d = compute_gf_score(r_vals, purities, r_min=GF_R_MIN, r_max=GF_R_MAX)

    # Also compute GF on raw high-dim coords if dim > 2
    gf_raw = None
    if latent_dim > 2:
        coords_raw = rescale_coordinates(coords, TARGET_STD)
        purities_raw, _ = compute_gf_curve(coords_raw, nodes, go_map_sub, r_vals)
        gf_raw = compute_gf_score(r_vals, purities_raw,
                                  r_min=GF_R_MIN, r_max=GF_R_MAX)

    return float(gf_2d), gf_raw, collapse_info


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load network ----
    print("Loading curated yeast PPI network...")
    G, nodes, go_map = load_curated_network(data_dir)
    print("  {} nodes, {} edges, {} annotated".format(
        len(nodes), G.number_of_edges(), len(go_map)))

    # ---- Compute centrality features ----
    print("Computing centrality features...")
    features = compute_centrality_features(G, nodes)

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    go_map_sub = {n: go_map[n] for n in nodes if n in go_map}

    # ---- Experiment grid ----
    configs = [
        # (label, model_type, n_heads, latent_dim)
        # GATv2 single-head
        ("gatv2_1h_d2",  "gatv2", 1,  2),
        ("gatv2_1h_d4",  "gatv2", 1,  4),
        ("gatv2_1h_d8",  "gatv2", 1,  8),
        ("gatv2_1h_d16", "gatv2", 1, 16),
        ("gatv2_1h_d32", "gatv2", 1, 32),
        # GATv2 multi-head (4 heads)
        ("gatv2_4h_d2",  "gatv2", 4,  2),
        ("gatv2_4h_d4",  "gatv2", 4,  4),
        ("gatv2_4h_d8",  "gatv2", 4,  8),
        ("gatv2_4h_d16", "gatv2", 4, 16),
        ("gatv2_4h_d32", "gatv2", 4, 32),
        # GAT single-head (baseline)
        ("gat_1h_d2",  "gat", 1,  2),
        ("gat_1h_d4",  "gat", 1,  4),
        ("gat_1h_d8",  "gat", 1,  8),
        ("gat_1h_d16", "gat", 1, 16),
        ("gat_1h_d32", "gat", 1, 32),
    ]

    print("")
    print("=" * 78)
    print("GATv2 vs GAT Collapse Comparison")
    print("  {} configurations x {} epochs, seed={}".format(
        len(configs), EPOCHS, SEED))
    print("  Hidden dim: {}, Dimensions: {}".format(HIDDEN_DIM, DIMENSIONS))
    print("  GF interval: [{}, {}], N_POINTS={}".format(
        GF_R_MIN, GF_R_MAX, N_POINTS))
    print("  Random baseline GF: {:.3f}".format(RANDOM_BASELINE_GF))
    print("  Spectral baseline GF: {:.3f}".format(SPECTRAL_BASELINE_GF))
    print("=" * 78)

    all_results = {}
    t_total = time.time()

    for label, model_type, n_heads, latent_dim in configs:
        print("")
        print("--- {} (model={}, heads={}, dim={}) ---".format(
            label, model_type, n_heads, latent_dim))
        t0 = time.time()

        if model_type == "gatv2":
            result = train_gatv2(
                G, features, latent_dim=latent_dim, n_heads=n_heads,
                hidden_dim=HIDDEN_DIM, epochs=EPOCHS, lr=LR, seed=SEED)
        else:
            result = train_gat(
                G, features, latent_dim=latent_dim, n_heads=n_heads,
                hidden_dim=HIDDEN_DIM, epochs=EPOCHS, lr=LR, seed=SEED)

        coords = result["coords"]

        # Compute GF Score
        gf_2d, gf_raw, collapse_info = compute_gf_for_embedding(
            coords, nodes, go_map_sub, r_vals, latent_dim)

        elapsed = time.time() - t0

        print("  Attn entropy: {:.4f} (L1={:.4f}, L2={:.4f})".format(
            result["attention_entropy"],
            result["attention_entropy_l1"],
            result["attention_entropy_l2"]))
        print("  Effective rank: {:.4f}".format(result["effective_rank"]))
        print("  GF Score (2D): {:.4f}".format(gf_2d))
        if gf_raw is not None:
            print("  GF Score (raw {}D): {:.4f}".format(latent_dim, gf_raw))
        print("  Collapsed: {}".format(
            "YES" if collapse_info["collapsed"] else "no"))
        print("  Final loss: {:.2f}".format(result["final_loss"]))
        print("  Elapsed: {:.1f}s".format(elapsed))

        # Best GF (max of 2D and raw)
        best_gf = gf_2d
        if gf_raw is not None and gf_raw > best_gf:
            best_gf = gf_raw

        all_results[label] = {
            "model_type": model_type,
            "n_heads": n_heads,
            "latent_dim": latent_dim,
            "gf_score_2d": float(gf_2d),
            "gf_score_raw": float(gf_raw) if gf_raw is not None else None,
            "best_gf_score": float(best_gf),
            "attention_entropy": float(result["attention_entropy"]),
            "attention_entropy_l1": float(result["attention_entropy_l1"]),
            "attention_entropy_l2": float(result["attention_entropy_l2"]),
            "effective_rank": float(result["effective_rank"]),
            "collapsed": collapse_info["collapsed"],
            "median_dist": float(collapse_info["median_dist"]),
            "dist_cv": float(collapse_info["cv"]),
            "final_loss": float(result["final_loss"]),
            "elapsed_s": round(elapsed, 1),
        }

    total_elapsed = time.time() - t_total

    # ================================================================
    # Summary Table
    # ================================================================
    print("")
    print("=" * 78)
    print("RESULTS SUMMARY")
    print("=" * 78)
    print("")

    header = ("{:<16} {:>5} {:>4} {:>9} {:>9} {:>9} {:>9} {:>8}"
              .format("Variant", "Model", "Hds", "Dim",
                      "GF(2D)", "GF(best)", "AttnEnt", "EffRank"))
    print(header)
    print("-" * 78)

    for label, model_type, n_heads, latent_dim in configs:
        v = all_results[label]
        gf_2d_s = "{:.4f}".format(v["gf_score_2d"])
        gf_best_s = "{:.4f}".format(v["best_gf_score"])
        marker = ""
        if v["best_gf_score"] > RANDOM_BASELINE_GF:
            marker = " *"
        print("{:<16} {:>5} {:>4} {:>4} {:>9} {:>9} {:>9.4f} {:>9.4f}{}".format(
            label, model_type, n_heads, latent_dim,
            gf_2d_s, gf_best_s,
            v["attention_entropy"], v["effective_rank"],
            marker))

    print("-" * 78)
    print("  * = exceeds random baseline ({:.3f})".format(RANDOM_BASELINE_GF))

    # ================================================================
    # Analysis
    # ================================================================
    print("")
    print("=" * 78)
    print("ANALYSIS")
    print("=" * 78)

    # Compare GATv2 vs GAT at each dimension (single-head)
    print("")
    print("Single-head comparison (GATv2 vs GAT):")
    print("{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Dim", "GATv2_GF", "GAT_GF", "GATv2_Ent", "GAT_Ent",
        "GATv2_Rank", "GAT_Rank"))
    print("-" * 72)
    for d in DIMENSIONS:
        v2_key = "gatv2_1h_d{}".format(d)
        v1_key = "gat_1h_d{}".format(d)
        v2 = all_results[v2_key]
        v1 = all_results[v1_key]
        print("{:>6} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            d, v2["best_gf_score"], v1["best_gf_score"],
            v2["attention_entropy"], v1["attention_entropy"],
            v2["effective_rank"], v1["effective_rank"]))

    # Q1: Does GATv2 achieve lower attention entropy?
    print("")
    gatv2_entropies = [all_results["gatv2_1h_d{}".format(d)]["attention_entropy"]
                       for d in DIMENSIONS]
    gat_entropies = [all_results["gat_1h_d{}".format(d)]["attention_entropy"]
                     for d in DIMENSIONS]
    avg_gatv2_ent = np.mean(gatv2_entropies)
    avg_gat_ent = np.mean(gat_entropies)
    print("Q1: Does GATv2 achieve lower attention entropy than GAT?")
    print("    GATv2 avg entropy: {:.4f}".format(avg_gatv2_ent))
    print("    GAT   avg entropy: {:.4f}".format(avg_gat_ent))
    if avg_gatv2_ent < avg_gat_ent - 0.02:
        print("    YES -- dynamic attention reduces degree-driven uniformity.")
    elif avg_gatv2_ent > avg_gat_ent + 0.02:
        print("    NO  -- GATv2 attention is actually MORE uniform than GAT.")
    else:
        print("    MARGINAL -- entropy difference is small ({:.4f}).".format(
            avg_gatv2_ent - avg_gat_ent))

    # Q2: Does GATv2 achieve higher effective rank?
    print("")
    gatv2_ranks = [all_results["gatv2_1h_d{}".format(d)]["effective_rank"]
                   for d in DIMENSIONS]
    gat_ranks = [all_results["gat_1h_d{}".format(d)]["effective_rank"]
                 for d in DIMENSIONS]
    print("Q2: Does GATv2 achieve higher effective rank?")
    print("    GATv2 avg eff_rank: {:.4f}".format(np.mean(gatv2_ranks)))
    print("    GAT   avg eff_rank: {:.4f}".format(np.mean(gat_ranks)))
    if np.mean(gatv2_ranks) > np.mean(gat_ranks) * 1.05:
        print("    YES -- GATv2 avoids rank collapse.")
    else:
        print("    NO  -- GATv2 still exhibits rank collapse similar to GAT.")

    # Q3: Does GATv2 achieve higher G-F Score?
    print("")
    gatv2_gfs = [all_results["gatv2_1h_d{}".format(d)]["best_gf_score"]
                 for d in DIMENSIONS]
    gat_gfs = [all_results["gat_1h_d{}".format(d)]["best_gf_score"]
               for d in DIMENSIONS]
    print("Q3: Does GATv2 achieve higher G-F Score?")
    print("    GATv2 GF scores: {}".format(
        ", ".join("{:.4f}".format(g) for g in gatv2_gfs)))
    print("    GAT   GF scores: {}".format(
        ", ".join("{:.4f}".format(g) for g in gat_gfs)))
    print("    GATv2 max GF: {:.4f}, GAT max GF: {:.4f}".format(
        max(gatv2_gfs), max(gat_gfs)))
    if max(gatv2_gfs) > max(gat_gfs) * 1.1:
        print("    YES -- GATv2 produces more geometrically meaningful embeddings.")
    else:
        print("    NO  -- GATv2 GF Score is comparable to GAT.")

    # Q4: At what dimension does GATv2 first exceed the random baseline?
    print("")
    print("Q4: At what dimension does GATv2 first exceed random baseline ({:.3f})?".format(
        RANDOM_BASELINE_GF))
    first_exceed = None
    for d in DIMENSIONS:
        v2_key = "gatv2_1h_d{}".format(d)
        if all_results[v2_key]["best_gf_score"] > RANDOM_BASELINE_GF:
            first_exceed = d
            break
    if first_exceed is not None:
        print("    d={} (GF={:.4f})".format(
            first_exceed,
            all_results["gatv2_1h_d{}".format(first_exceed)]["best_gf_score"]))
    else:
        print("    GATv2 does NOT exceed random baseline at any tested dimension.")

    # Multi-head GATv2 results
    print("")
    print("Multi-head (4h) GATv2 results:")
    print("{:>6} {:>10} {:>10} {:>10}".format("Dim", "GF(best)", "AttnEnt", "EffRank"))
    print("-" * 40)
    for d in DIMENSIONS:
        v = all_results["gatv2_4h_d{}".format(d)]
        print("{:>6} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            d, v["best_gf_score"], v["attention_entropy"], v["effective_rank"]))

    print("")
    print("Total elapsed: {:.1f}s".format(total_elapsed))

    # ================================================================
    # Save results
    # ================================================================
    output = {
        "experiment": "gatv2_vs_gat_collapse",
        "description": ("GATv2 (Brody et al., 2022) vs GAT (Velickovic et al., 2018) "
                        "collapse comparison on curated 153-node yeast PPI."),
        "random_baseline": float(RANDOM_BASELINE_GF),
        "spectral_baseline": float(SPECTRAL_BASELINE_GF),
        "gf_interval": [float(GF_R_MIN), float(GF_R_MAX)],
        "n_points": int(N_POINTS),
        "n_nodes": int(len(nodes)),
        "n_edges": int(G.number_of_edges()),
        "n_annotated": int(len(go_map_sub)),
        "training": {
            "epochs": int(EPOCHS),
            "lr": float(LR),
            "hidden_dim": int(HIDDEN_DIM),
            "clip_grad": 1.0,
            "warmup_epochs": 10,
            "seed": int(SEED),
        },
        "gatv2": {
            "1h": {str(d): {
                "gf_score": float(all_results["gatv2_1h_d{}".format(d)]["best_gf_score"]),
                "gf_score_2d": float(all_results["gatv2_1h_d{}".format(d)]["gf_score_2d"]),
                "gf_score_raw": all_results["gatv2_1h_d{}".format(d)]["gf_score_raw"],
                "attention_entropy": float(all_results["gatv2_1h_d{}".format(d)]["attention_entropy"]),
                "effective_rank": float(all_results["gatv2_1h_d{}".format(d)]["effective_rank"]),
                "collapsed": bool(all_results["gatv2_1h_d{}".format(d)]["collapsed"]),
            } for d in DIMENSIONS},
            "4h": {str(d): {
                "gf_score": float(all_results["gatv2_4h_d{}".format(d)]["best_gf_score"]),
                "gf_score_2d": float(all_results["gatv2_4h_d{}".format(d)]["gf_score_2d"]),
                "gf_score_raw": all_results["gatv2_4h_d{}".format(d)]["gf_score_raw"],
                "attention_entropy": float(all_results["gatv2_4h_d{}".format(d)]["attention_entropy"]),
                "effective_rank": float(all_results["gatv2_4h_d{}".format(d)]["effective_rank"]),
                "collapsed": bool(all_results["gatv2_4h_d{}".format(d)]["collapsed"]),
            } for d in DIMENSIONS},
        },
        "gat": {
            "1h": {str(d): {
                "gf_score": float(all_results["gat_1h_d{}".format(d)]["best_gf_score"]),
                "gf_score_2d": float(all_results["gat_1h_d{}".format(d)]["gf_score_2d"]),
                "gf_score_raw": all_results["gat_1h_d{}".format(d)]["gf_score_raw"],
                "attention_entropy": float(all_results["gat_1h_d{}".format(d)]["attention_entropy"]),
                "effective_rank": float(all_results["gat_1h_d{}".format(d)]["effective_rank"]),
                "collapsed": bool(all_results["gat_1h_d{}".format(d)]["collapsed"]),
            } for d in DIMENSIONS},
        },
        "analysis": {
            "gatv2_avg_entropy_1h": float(avg_gatv2_ent),
            "gat_avg_entropy_1h": float(avg_gat_ent),
            "entropy_reduction": bool(avg_gatv2_ent < avg_gat_ent - 0.02),
            "gatv2_avg_eff_rank_1h": float(np.mean(gatv2_ranks)),
            "gat_avg_eff_rank_1h": float(np.mean(gat_ranks)),
            "rank_improvement": bool(float(np.mean(gatv2_ranks)) > float(np.mean(gat_ranks)) * 1.05),
            "gatv2_max_gf_1h": float(max(gatv2_gfs)),
            "gat_max_gf_1h": float(max(gat_gfs)),
            "gf_improvement": bool(max(gatv2_gfs) > max(gat_gfs) * 1.1),
            "first_dim_exceeding_random": int(first_exceed) if first_exceed else None,
        },
        "detailed_results": {k: v for k, v in all_results.items()},
    }

    out_file = results_dir / "gatv2_comparison.json"
    with open(str(out_file), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("")
    print("Saved: {}".format(out_file))


if __name__ == "__main__":
    main()
