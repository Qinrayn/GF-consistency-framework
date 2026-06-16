#!/usr/bin/env python3
"""
multihead_gat_experiment.py
Multi-Head GAT Fairness Experiment for GF-Consistency Framework.

Addresses reviewer concern about GNN fairness: does the GAT architecture
perform poorly simply because of under-tuned hyperparameters (single head,
low dimension)?  This script systematically sweeps:

  - Attention heads: 1, 4, 8
  - Latent dimension: 2, 16, 32

on the curated 153-node yeast PPI network, using the same GF Score pipeline
as the main experiments (greedy_modularity_communities, GF interval
[0.05, 0.422]).

Key question: can multi-head + higher dimension GAT exceed the random
baseline GF Score of 0.135?

Output
------
  results/multihead_gat_experiment.json
"""

import sys
import json
import time
import numpy as np
import networkx as nx
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir,
    load_curated_network, compute_centrality_features,
    rescale_coordinates, compute_gf_curve, compute_gf_score,
    check_embedding_collapse,
    GF_R_MIN, GF_R_MAX, R_MIN, R_MAX, N_POINTS, TARGET_STD,
)


def train_multihead_gat(G, features, n_heads, latent_dim,
                        epochs=300, lr=0.01, seed=SEED,
                        clip_grad=1.0, warmup_epochs=10,
                        hidden_dim=16):
    """Train a multi-head GAT encoder with given heads and latent dim.

    Architecture
    ------------
      Layer 1: GATConv(in_dim -> hidden_dim, heads=n_heads, concat if heads>1)
      BatchNorm + ReLU
      Layer 2: GATConv(h_dim -> latent_dim, heads=1, concat=False)
      BatchNorm

    Training: adjacency reconstruction via BCE loss, gradient clipping,
    linear LR warmup.

    Returns
    -------
    dict with keys: coords, loss_history, attention_entropy_layer1,
    attention_entropy_layer2, final_loss
    """
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

    # Adjacency target
    adj_target = torch.zeros(n, n)
    ei = data.edge_index
    adj_target[ei[0], ei[1]] = 1.0

    loss_history = []
    final_loss = None

    for epoch in range(epochs):
        # LR warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        elif warmup_epochs > 0 and epoch == warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        adj_recon = torch.sigmoid(z @ z.T)
        recon_loss = F.binary_cross_entropy(adj_recon, adj_target,
                                            reduction="sum")
        loss = recon_loss
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad)

        optimizer.step()
        final_loss = float(loss.item())

        if (epoch + 1) % 50 == 0 or epoch == 0:
            loss_history.append({
                "epoch": epoch + 1,
                "loss": final_loss,
            })

    # Extract embeddings and attention
    model.eval()
    with torch.no_grad():
        z, alpha1, alpha2 = model(data.x, data.edge_index,
                                  return_attention=True)
        coords = z.numpy()

    # Compute attention entropy
    def attention_entropy(alpha_tuple):
        edge_index, alpha = alpha_tuple
        alpha_np = alpha.cpu().numpy().flatten()
        if len(alpha_np) == 0:
            return 0.0, 0.0
        alpha_pos = np.abs(alpha_np) + 1e-10
        alpha_norm = alpha_pos / alpha_pos.sum()
        entropy = -np.sum(alpha_norm * np.log(alpha_norm + 1e-10))
        max_entropy = np.log(len(alpha_norm))
        return float(entropy), float(max_entropy)

    ent1 = attention_entropy(alpha1)
    ent2 = attention_entropy(alpha2)

    return {
        "coords": coords,
        "loss_history": loss_history,
        "final_loss": final_loss,
        "attention_entropy_layer1": {
            "entropy": ent1[0],
            "max_entropy": ent1[1],
            "normalized": ent1[0] / (ent1[1] + 1e-10),
        },
        "attention_entropy_layer2": {
            "entropy": ent2[0],
            "max_entropy": ent2[1],
            "normalized": ent2[0] / (ent2[1] + 1e-10),
        },
    }


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

    # ---- Define variants ----
    # (label, n_heads, latent_dim)
    variants = [
        ("1h_d2",   1,  2),
        ("4h_d2",   4,  2),
        ("8h_d2",   8,  2),
        ("4h_d16",  4, 16),
        ("8h_d16",  8, 16),
        ("4h_d32",  4, 32),
        ("8h_d32",  8, 32),
    ]

    RANDOM_BASELINE_GF = 0.135

    print("")
    print("=" * 72)
    print("Multi-Head GAT Fairness Experiment")
    print("  {} variants x 300 epochs, seed={}".format(len(variants), SEED))
    print("  GF interval: [{}, {}]".format(GF_R_MIN, GF_R_MAX))
    print("  Random baseline GF: {:.3f}".format(RANDOM_BASELINE_GF))
    print("=" * 72)

    variant_results = {}

    for label, n_heads, latent_dim in variants:
        print("")
        print("--- {} (heads={}, dim={}) ---".format(
            label, n_heads, latent_dim))
        t0 = time.time()

        result = train_multihead_gat(
            G, features,
            n_heads=n_heads,
            latent_dim=latent_dim,
            epochs=300, lr=0.01, seed=SEED,
            clip_grad=1.0, warmup_epochs=10,
            hidden_dim=16,
        )

        coords = result["coords"]

        # For GF computation, we need 2D coordinates.
        # If latent_dim > 2, use PCA to project to 2D for the standard
        # spatial-graph pipeline (same as the paper's protocol).
        if latent_dim > 2:
            from sklearn.decomposition import PCA
            coords_2d = PCA(n_components=2).fit_transform(coords)
        else:
            coords_2d = coords

        coords_2d = rescale_coordinates(coords_2d, TARGET_STD)

        # Collapse diagnostics
        collapse_info = check_embedding_collapse(coords_2d, label)
        print("  Collapsed: {}".format(
            "YES" if collapse_info["collapsed"] else "no"))
        if collapse_info["reasons"]:
            for r in collapse_info["reasons"]:
                print("    {}".format(r))
        print("  Median dist: {:.4f}".format(collapse_info["median_dist"]))
        print("  Dist CV: {:.4f}".format(collapse_info["cv"]))

        # Attention entropy
        ent1 = result["attention_entropy_layer1"]["normalized"]
        ent2 = result["attention_entropy_layer2"]["normalized"]
        print("  Attn entropy L1: {:.4f} (1.0 = uniform)".format(ent1))
        print("  Attn entropy L2: {:.4f}".format(ent2))

        # GF Score (on 2D projection)
        purities, _ = compute_gf_curve(coords_2d, nodes, go_map_sub, r_vals)
        gf = compute_gf_score(r_vals, purities,
                              r_min=GF_R_MIN, r_max=GF_R_MAX)
        print("  GF Score: {:.4f}".format(gf))

        # Also compute GF on raw high-dim coords if dim > 2
        gf_raw = None
        if latent_dim > 2:
            coords_raw = rescale_coordinates(coords, TARGET_STD)
            purities_raw, _ = compute_gf_curve(
                coords_raw, nodes, go_map_sub, r_vals)
            gf_raw = compute_gf_score(r_vals, purities_raw,
                                      r_min=GF_R_MIN, r_max=GF_R_MAX)
            print("  GF Score (raw {}D): {:.4f}".format(latent_dim, gf_raw))

        elapsed = time.time() - t0
        print("  Final loss: {:.2f}".format(result["final_loss"]))
        print("  Elapsed: {:.1f}s".format(elapsed))

        variant_results[label] = {
            "n_heads": n_heads,
            "latent_dim": latent_dim,
            "gf_score_2d": float(gf),
            "gf_score_raw": float(gf_raw) if gf_raw is not None else None,
            "collapsed": collapse_info["collapsed"],
            "median_dist": collapse_info["median_dist"],
            "dist_cv": collapse_info["cv"],
            "attention_entropy_layer1": result["attention_entropy_layer1"],
            "attention_entropy_layer2": result["attention_entropy_layer2"],
            "final_loss": result["final_loss"],
            "loss_history": result["loss_history"],
            "elapsed_s": round(elapsed, 1),
        }

    # ---- Summary Table ----
    print("")
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print("")

    header = ("{:<12} {:>6} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}"
              .format("Variant", "Heads", "Dim",
                      "GF(2D)", "GF(raw)", "Collapse",
                      "MedDist", "AttnE1", "AttnE2"))
    print(header)
    print("-" * 72)

    best_gf = 0.0
    best_label = ""
    for label, n_heads, latent_dim in variants:
        v = variant_results[label]
        gf_2d = v["gf_score_2d"]
        gf_raw_str = "{:.4f}".format(v["gf_score_raw"]) if (
            v["gf_score_raw"] is not None) else "  --  "
        collapse_str = "YES" if v["collapsed"] else "no"

        best_for_variant = gf_2d
        if v["gf_score_raw"] is not None and v["gf_score_raw"] > best_for_variant:
            best_for_variant = v["gf_score_raw"]
        if best_for_variant > best_gf:
            best_gf = best_for_variant
            best_label = label

        marker = ""
        if best_for_variant > RANDOM_BASELINE_GF:
            marker = " <-- EXCEEDS RANDOM"

        print("{:<12} {:>6} {:>5} {:>9.4f} {:>9} {:>9} {:>9.4f} "
              "{:>9.4f} {:>9.4f}{}".format(
                  label, n_heads, latent_dim,
                  gf_2d, gf_raw_str, collapse_str,
                  v["median_dist"],
                  v["attention_entropy_layer1"]["normalized"],
                  v["attention_entropy_layer2"]["normalized"],
                  marker))

    print("-" * 72)
    print("")
    print("Random baseline GF Score:   {:.3f}".format(RANDOM_BASELINE_GF))
    print("Best GAT GF Score:          {:.4f} ({})".format(
        best_gf, best_label))
    if best_gf > RANDOM_BASELINE_GF:
        print("VERDICT: Multi-head GAT EXCEEDS random baseline.")
    else:
        print("VERDICT: Multi-head GAT does NOT exceed random baseline.")
        print("         GAT attention degeneration is structural, not a")
        print("         hyperparameter issue. This supports the paper's")
        print("         claim that GNN encoders are not universally")
        print("         superior to classical methods.")

    # ---- Save results ----
    save_data = {
        "experiment": "multihead_gat_fairness",
        "description": ("Systematic sweep of GAT heads x latent dims to "
                        "test whether multi-head + high-dim GAT can exceed "
                        "random baseline GF Score."),
        "random_baseline_gf": RANDOM_BASELINE_GF,
        "gf_interval": [GF_R_MIN, GF_R_MAX],
        "n_nodes": len(nodes),
        "n_edges": G.number_of_edges(),
        "n_annotated": len(go_map_sub),
        "training": {
            "epochs": 300,
            "lr": 0.01,
            "clip_grad": 1.0,
            "warmup_epochs": 10,
            "hidden_dim": 16,
            "seed": SEED,
        },
        "variants": {label: {
            "n_heads": v["n_heads"],
            "latent_dim": v["latent_dim"],
        } for label, v in variant_results.items()},
        "results": {label: {
            k: val for k, val in v.items()
            if k not in ("loss_history",)
        } for label, v in variant_results.items()},
        "best_variant": best_label,
        "best_gf_score": best_gf,
        "exceeds_random_baseline": best_gf > RANDOM_BASELINE_GF,
    }

    out_file = results_dir / "multihead_gat_experiment.json"
    with open(str(out_file), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print("")
    print("Saved: {}".format(out_file))


if __name__ == "__main__":
    main()
