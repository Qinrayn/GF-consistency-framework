#!/usr/bin/env python3
"""
gat_collapse_diagnosis.py
Step 39: GAT Embedding Collapse Root-Cause Analysis.

GAT consistently produces poor G-F Scores on both yeast (0.069) and human
(~0.069) PPI networks — comparable to random baseline.  This script
diagnoses the root cause by testing three hypotheses:

H1: **Gradient explosion** — Without gradient clipping, GAT attention
    parameters receive large gradients, causing unstable training.
H2: **No LR warmup** — Starting at full LR (0.01) may overshoot the
    attention parameter optimum.
H3: **Attention degeneration** — GAT attention weights converge to
    near-uniform values, making GAT equivalent to mean-aggregation GCN.

Design
------
  1. Baseline GAT (no clipping, no warmup) — reproduce the collapse.
  2. GAT + gradient clipping (max_norm=1.0).
  3. GAT + linear LR warmup (10 epochs).
  4. GAT + clipping + warmup.
  5. GAT with multi-head attention (heads=4, concat=True).
  6. For each variant, extract attention weights and compute entropy.
  7. Compare all variants on: embedding collapse diagnostics, GF Score,
     attention entropy, training loss trajectory.

Output
------
  results/gat_collapse_diagnosis.json
  figures/Fig25_gat_diagnosis.png
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


def train_gat_variant(G, features, variant_name, epochs=300, lr=0.01,
                      seed=SEED, clip_grad=None, warmup_epochs=0,
                      n_heads=1, hidden_dim=16, latent_dim=2):
    """Train a GAT encoder with specified hyperparameters.

    Returns
    -------
    dict with keys: coords, loss_history, attention_entropy_layer1,
    attention_entropy_layer2, model
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

    # Build model
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
            h, alpha1 = self.conv1(x, edge_index, return_attention_weights=True)
            h = self.bn1(h)
            h = F.relu(h)
            z, alpha2 = self.conv2(h, edge_index, return_attention_weights=True)
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

        # Gradient clipping
        grad_norm = None
        if clip_grad is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad)

        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            loss_history.append({
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm) if grad_norm is not None
                             else None,
            })

    # Extract attention weights
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
            return 0.0
        # Normalize to probability distribution
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
        "attention_entropy_layer1": {"entropy": ent1[0], "max_entropy": ent1[1],
                                     "normalized": ent1[0] / (ent1[1] + 1e-10)},
        "attention_entropy_layer2": {"entropy": ent2[0], "max_entropy": ent2[1],
                                     "normalized": ent2[0] / (ent2[1] + 1e-10)},
    }


def generate_figure(variant_results, figures_dir):
    """Generate multi-panel diagnostic figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.35)

    # Panel A: Embedding scatter (baseline vs best fix)
    ax = fig.add_subplot(gs[0, 0])
    baseline = variant_results.get("baseline", {})
    if baseline and "coords" in baseline:
        c = baseline["coords"]
        ax.scatter(c[:, 0], c[:, 1], s=8, alpha=0.5, c="#e74c3c",
                   label="Baseline")
    best_key = None
    best_gf = 0
    for k, v in variant_results.items():
        if k != "baseline" and "gf_score" in v and v["gf_score"] > best_gf:
            best_gf = v["gf_score"]
            best_key = k
    if best_key and "coords" in variant_results[best_key]:
        c = variant_results[best_key]["coords"]
        ax.scatter(c[:, 0], c[:, 1], s=8, alpha=0.5, c="#2ecc71",
                   label=best_key)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title("(A) Embedding: Baseline vs Best Fix")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel B: Loss trajectories
    ax = fig.add_subplot(gs[0, 1])
    for name, v in variant_results.items():
        if "loss_history" in v and v["loss_history"]:
            epochs = [h["epoch"] for h in v["loss_history"]]
            losses = [h["loss"] for h in v["loss_history"]]
            ax.plot(epochs, losses, "-o", label=name, markersize=3,
                    linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("(B) Training Loss")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Panel C: GF Score comparison
    ax = fig.add_subplot(gs[0, 2])
    names = list(variant_results.keys())
    gf_scores = [variant_results[n].get("gf_score", 0) for n in names]
    colors = ["#e74c3c" if n == "baseline" else "#2ecc71" for n in names]
    bars = ax.barh(names, gf_scores, color=colors,
                   edgecolor="black", linewidth=0.5)
    ax.set_xlabel("G-F Score")
    ax.set_title("(C) G-F Score per Variant")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, s in zip(bars, gf_scores):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{s:.4f}", va="center", fontsize=8)

    # Panel D: Attention entropy
    ax = fig.add_subplot(gs[0, 3])
    ent_l1 = [variant_results[n].get("attention_entropy_layer1", {}).get(
        "normalized", 0) for n in names]
    ent_l2 = [variant_results[n].get("attention_entropy_layer2", {}).get(
        "normalized", 0) for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, ent_l1, width, label="Layer 1", color="#3498db")
    ax.bar(x + width / 2, ent_l2, width, label="Layer 2", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Normalized Attention Entropy")
    ax.set_title("(D) Attention Weight Distribution")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5,
               label="Uniform (max entropy)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_file = figures_dir / "Fig25_gat_diagnosis.png"
    plt.savefig(str(out_file), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_file}")


def main():
    np.random.seed(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    figures_dir = get_figures_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load network
    print("Loading curated yeast PPI network...")
    G, nodes, go_map = load_curated_network(data_dir)
    print(f"  {len(nodes)} nodes, {G.number_of_edges()} edges, "
          f"{len(go_map)} annotated")

    # Compute centrality features
    print("Computing centrality features...")
    features = compute_centrality_features(G, nodes)

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # ---- Define variants ----
    variants = {
        "baseline": {
            "clip_grad": None, "warmup_epochs": 0, "n_heads": 1,
        },
        "clip_norm1": {
            "clip_grad": 1.0, "warmup_epochs": 0, "n_heads": 1,
        },
        "warmup10": {
            "clip_grad": None, "warmup_epochs": 10, "n_heads": 1,
        },
        "clip+warmup": {
            "clip_grad": 1.0, "warmup_epochs": 10, "n_heads": 1,
        },
        "heads4_concat": {
            "clip_grad": 1.0, "warmup_epochs": 10, "n_heads": 4,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"GAT Collapse Diagnosis: {len(variants)} variants")
    print(f"{'=' * 60}")

    variant_results = {}

    for name, config in variants.items():
        print(f"\n--- {name} ---")
        t0 = time.time()

        result = train_gat_variant(
            G, features, name,
            epochs=300, lr=0.01, seed=SEED,
            clip_grad=config["clip_grad"],
            warmup_epochs=config["warmup_epochs"],
            n_heads=config["n_heads"],
        )

        coords = result["coords"]
        coords = rescale_coordinates(coords, TARGET_STD)

        # Collapse diagnostics
        collapse_info = check_embedding_collapse(coords, name)
        print(f"  Collapsed: {collapse_info['collapsed']}")
        if collapse_info["reasons"]:
            for r in collapse_info["reasons"]:
                print(f"    {r}")
        print(f"  Median dist: {collapse_info['median_dist']:.4f}")
        print(f"  Dist CV: {collapse_info['cv']:.4f}")

        # Attention entropy
        ent1 = result["attention_entropy_layer1"]["normalized"]
        ent2 = result["attention_entropy_layer2"]["normalized"]
        print(f"  Attention entropy L1: {ent1:.4f} (1.0 = uniform)")
        print(f"  Attention entropy L2: {ent2:.4f}")

        # GF Score
        go_map_sub = {n: go_map[n] for n in nodes if n in go_map}
        purities, _ = compute_gf_curve(coords, nodes, go_map_sub, r_vals)
        gf = compute_gf_score(r_vals, purities, r_min=GF_R_MIN, r_max=GF_R_MAX)
        print(f"  GF Score: {gf:.4f}")

        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.1f}s")

        variant_results[name] = {
            "coords": coords,
            "gf_score": float(gf),
            "collapsed": collapse_info["collapsed"],
            "median_dist": collapse_info["median_dist"],
            "dist_cv": collapse_info["cv"],
            "attention_entropy_layer1": result["attention_entropy_layer1"],
            "attention_entropy_layer2": result["attention_entropy_layer2"],
            "loss_history": result["loss_history"],
            "elapsed_s": round(elapsed, 1),
        }

    # ---- Comparison ----
    print(f"\n{'=' * 60}")
    print("Diagnosis Summary")
    print(f"{'=' * 60}")
    print(f"\n{'Variant':<18} {'GF':>8} {'Collapsed':>10} {'MedDist':>8} "
          f"{'DistCV':>8} {'AttnEnt1':>9} {'AttnEnt2':>9}")
    print("-" * 80)
    baseline_gf = variant_results["baseline"]["gf_score"]
    for name, v in variant_results.items():
        marker = " *" if v["gf_score"] > baseline_gf * 1.1 else ""
        print(f"{name:<18} {v['gf_score']:>8.4f} "
              f"{'YES' if v['collapsed'] else 'no':>10} "
              f"{v['median_dist']:>8.4f} {v['dist_cv']:>8.4f} "
              f"{v['attention_entropy_layer1']['normalized']:>9.4f} "
              f"{v['attention_entropy_layer2']['normalized']:>9.4f}"
              f"{marker}")

    # Determine root cause
    print("\nRoot cause analysis:")
    baseline_ent1 = variant_results["baseline"]["attention_entropy_layer1"]["normalized"]
    if baseline_ent1 > 0.95:
        print("  -> Attention degeneration CONFIRMED: baseline attention "
              f"entropy = {baseline_ent1:.4f} (near-uniform)")
    else:
        print("  -> Attention degeneration NOT confirmed: baseline attention "
              f"entropy = {baseline_ent1:.4f}")

    clip_fix = variant_results.get("clip_norm1", {})
    if clip_fix.get("gf_score", 0) > baseline_gf * 1.1:
        print(f"  -> Gradient clipping HELPS: GF {baseline_gf:.4f} -> "
              f"{clip_fix['gf_score']:.4f}")
    else:
        print(f"  -> Gradient clipping does NOT help significantly")

    warmup_fix = variant_results.get("warmup10", {})
    if warmup_fix.get("gf_score", 0) > baseline_gf * 1.1:
        print(f"  -> LR warmup HELPS: GF {baseline_gf:.4f} -> "
              f"{warmup_fix['gf_score']:.4f}")
    else:
        print(f"  -> LR warmup does NOT help significantly")

    # ---- Save results ----
    save_results = {}
    for name, v in variant_results.items():
        save_results[name] = {k: val for k, val in v.items() if k != "coords"}

    output = {
        "variants_tested": list(variants.keys()),
        "variant_configs": {k: {kk: vv for kk, vv in v.items()}
                            for k, v in variants.items()},
        "results": save_results,
        "baseline_gf": baseline_gf,
        "n_nodes": len(nodes),
        "n_edges": G.number_of_edges(),
        "n_annotated": len(go_map),
        "gf_interval": [GF_R_MIN, GF_R_MAX],
    }

    out_file = results_dir / "gat_collapse_diagnosis.json"
    with open(str(out_file), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")

    # ---- Generate figure ----
    generate_figure(variant_results, figures_dir)


if __name__ == "__main__":
    main()
