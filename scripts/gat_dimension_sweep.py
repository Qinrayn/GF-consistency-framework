#!/usr/bin/env python3
"""
gat_dimension_sweep.py -- Phase 5B: GAT Latent Dimension Sweep
===============================================================

Tests whether increasing GAT's output dimension breaks the collapse
causal chain identified in Phase 4:

  Degree heterogeneity (CV=0.644)
    -> Attention degeneration (entropy=0.973)
    -> GAT equivalent to GCN
    -> Over-smoothing in 2-D bottleneck (rank -> 1)
    -> G-F Score below random baseline

Key question: Does latent_dim > 2 rescue GAT from collapse?

Design
------
  Train GAT with latent_dim in {2, 4, 8, 16, 32} on the curated
  153-node yeast PPI.  For each dimension measure:
    - G-F Score (functional-geometric consistency)
    - Attention entropy (degeneration indicator)
    - Effective dimensionality (participation ratio)
    - Matrix rank of embedding
    - Embedding collapse diagnostics
    - Reconstruction loss trajectory

  Control: GraphSAGE at the same dimensions (to separate GAT-specific
  effects from generic high-dimensional behaviour).

Outputs
-------
  results/gat_dimension_sweep.json
  figures/Fig41_gat_dimension_sweep.png
"""

import sys
import json
import time
import warnings
import numpy as np
import networkx as nx
from pathlib import Path

warnings.filterwarnings("ignore")

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir,
    load_curated_network, compute_centrality_features,
    rescale_coordinates, compute_gf_curve, compute_gf_score,
    check_embedding_collapse,
    GF_R_MIN, GF_R_MAX, R_MIN, R_MAX, N_POINTS, TARGET_STD,
)

# Import GAT training function with attention extraction
from gat_collapse_diagnosis import train_gat_variant

DIMENSIONS = [2, 4, 8, 16, 32]
HIDDEN_DIM = 16
EPOCHS = 300
LR = 0.01


# ============================================================
# GraphSAGE Control (same architecture, no attention)
# ============================================================

def train_sage_variant(G, features, latent_dim=2, hidden_dim=16,
                       epochs=300, lr=0.01, seed=SEED):
    """Train GraphSAGE encoder at specified latent dimension."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
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

    class SAGEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, latent_dim, aggr="mean")
            self.bn2 = nn.BatchNorm1d(latent_dim)

        def forward(self, x, edge_index):
            h = self.conv1(x, edge_index)
            h = self.bn1(h)
            h = F.relu(h)
            z = self.conv2(h, edge_index)
            z = self.bn2(z)
            return z

    model = SAGEModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    adj_target = torch.zeros(n, n)
    ei = data.edge_index
    adj_target[ei[0], ei[1]] = 1.0

    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        adj_recon = torch.sigmoid(z @ z.T)
        loss = F.binary_cross_entropy(adj_recon, adj_target, reduction="sum")
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            loss_history.append({"epoch": epoch + 1, "loss": float(loss.item())})

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        coords = z.numpy()

    return {"coords": coords, "loss_history": loss_history}


# ============================================================
# Metrics
# ============================================================

def compute_effective_dim(coords):
    """Participation ratio of PCA eigenvalues."""
    c = coords - coords.mean(axis=0)
    cov = c.T @ c / c.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    s = eigvals.sum()
    if s < 1e-10:
        return 1.0
    return float(s ** 2 / max((eigvals ** 2).sum(), 1e-10))


def compute_matrix_rank(coords, tol=1e-6):
    """Numerical rank via SVD."""
    s = np.linalg.svd(coords, compute_uv=False)
    return int(np.sum(s > tol))


def compute_gf_for_coords(coords, nodes, go_map, r_vals):
    """Compute G-F Score for given coordinates."""
    common = sorted(set(nodes) & set(go_map.keys()))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx = [node_to_idx[n] for n in common]
    aligned = coords[idx]
    aligned = rescale_coordinates(aligned.copy(), target_std=TARGET_STD)
    purities, _ = compute_gf_curve(aligned, common, go_map, r_vals)
    score = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)
    return score, purities


# ============================================================
# Visualization
# ============================================================

def generate_figure(results, figures_dir):
    """Fig41: GAT dimension sweep results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    dims = DIMENSIONS
    gat_data = results["GAT"]
    sage_data = results.get("GraphSAGE", {})

    # Panel A: G-F Score vs dimension
    ax = axes[0, 0]
    gat_gf = [gat_data[d]["gf_score"] for d in dims]
    sage_gf = [sage_data[d]["gf_score"] for d in dims] if sage_data else []
    ax.plot(dims, gat_gf, "o-", color="#E63946", linewidth=2, markersize=8,
            label="GAT", zorder=5)
    if sage_gf:
        ax.plot(dims, sage_gf, "s--", color="#457B9D", linewidth=2, markersize=8,
                label="GraphSAGE", zorder=5)
    ax.axhline(results.get("random_baseline", 0.1), color="gray", linestyle=":",
               alpha=0.5, label="Random baseline")
    ax.set_xlabel("Latent dimension", fontsize=12)
    ax.set_ylabel("G-F Score", fontsize=12)
    ax.set_title("A. G-F Score vs Latent Dimension", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dims)

    # Panel B: Attention entropy vs dimension
    ax2 = axes[0, 1]
    gat_ent_l1 = [gat_data[d]["attention_entropy_layer1"]["normalized"] for d in dims]
    gat_ent_l2 = [gat_data[d]["attention_entropy_layer2"]["normalized"] for d in dims]
    ax2.plot(dims, gat_ent_l1, "o-", color="#E63946", linewidth=2, markersize=8,
             label="Layer 1")
    ax2.plot(dims, gat_ent_l2, "^--", color="#E63946", linewidth=2, markersize=8,
             alpha=0.6, label="Layer 2")
    ax2.axhline(1.0, color="red", linestyle=":", alpha=0.4, label="Uniform (1.0)")
    ax2.set_xlabel("Latent dimension", fontsize=12)
    ax2.set_ylabel("Normalized attention entropy", fontsize=12)
    ax2.set_title("B. Attention Entropy vs Dimension", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(dims)

    # Panel C: Effective dimensionality vs latent dim
    ax3 = axes[0, 2]
    gat_ed = [gat_data[d]["effective_dim"] for d in dims]
    sage_ed = [sage_data[d]["effective_dim"] for d in dims] if sage_data else []
    ax3.plot(dims, gat_ed, "o-", color="#E63946", linewidth=2, markersize=8,
             label="GAT")
    if sage_ed:
        ax3.plot(dims, sage_ed, "s--", color="#457B9D", linewidth=2, markersize=8,
                 label="GraphSAGE")
    ax3.plot(dims, dims, "k:", alpha=0.3, label="y=x (full rank)")
    ax3.set_xlabel("Latent dimension", fontsize=12)
    ax3.set_ylabel("Effective dimensionality", fontsize=12)
    ax3.set_title("C. Effective Dim vs Latent Dim", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(dims)

    # Panel D: Matrix rank vs dimension
    ax4 = axes[1, 0]
    gat_rank = [gat_data[d]["matrix_rank"] for d in dims]
    sage_rank = [sage_data[d]["matrix_rank"] for d in dims] if sage_data else []
    ax4.plot(dims, gat_rank, "o-", color="#E63946", linewidth=2, markersize=8,
             label="GAT")
    if sage_rank:
        ax4.plot(dims, sage_rank, "s--", color="#457B9D", linewidth=2, markersize=8,
                 label="GraphSAGE")
    ax4.plot(dims, dims, "k:", alpha=0.3, label="y=x (full rank)")
    ax4.set_xlabel("Latent dimension", fontsize=12)
    ax4.set_ylabel("Matrix rank", fontsize=12)
    ax4.set_title("D. Matrix Rank vs Latent Dim", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(dims)

    # Panel E: Final loss vs dimension
    ax5 = axes[1, 1]
    gat_loss = [gat_data[d]["final_loss"] for d in dims]
    sage_loss = [sage_data[d]["final_loss"] for d in dims] if sage_data else []
    ax5.plot(dims, gat_loss, "o-", color="#E63946", linewidth=2, markersize=8,
             label="GAT")
    if sage_loss:
        ax5.plot(dims, sage_loss, "s--", color="#457B9D", linewidth=2, markersize=8,
                 label="GraphSAGE")
    ax5.set_xlabel("Latent dimension", fontsize=12)
    ax5.set_ylabel("Final reconstruction loss", fontsize=12)
    ax5.set_title("E. Final Loss vs Latent Dim", fontsize=13, fontweight="bold")
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(dims)

    # Panel F: GAT embedding scatter at each dimension (2x2 grid inset)
    ax6 = axes[1, 2]
    for i, d in enumerate([2, 4, 8, 16]):
        coords = gat_data[d]["coords_raw"]
        if coords.shape[1] >= 2:
            alpha_val = 0.4 + 0.15 * i
            ax6.scatter(coords[:, 0], coords[:, 1], s=8, alpha=alpha_val,
                        label=f"d={d}")
    ax6.set_xlabel("dim 1", fontsize=12)
    ax6.set_ylabel("dim 2", fontsize=12)
    ax6.set_title("F. GAT Embedding (dim 1 vs 2)", fontsize=13, fontweight="bold")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = figures_dir / "Fig41_gat_dimension_sweep.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Phase 5B: GAT Latent Dimension Sweep")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading curated network...")
    G, nodes, go_map = load_curated_network(get_data_dir())
    features = compute_centrality_features(G, nodes)
    print(f"  Network: {len(nodes)} nodes, {G.number_of_edges()} edges")

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # Load existing G-F scores for reference
    data_dir = get_data_dir()
    res_dir = get_results_dir()
    fig_dir = get_figures_dir()

    # Random baseline
    rng = np.random.default_rng(SEED)
    random_coords = rng.standard_normal((len(nodes), 2))
    random_gf, _ = compute_gf_for_coords(random_coords, nodes, go_map, r_vals)
    print(f"  Random baseline G-F Score: {random_gf:.4f}")

    results = {"GAT": {}, "GraphSAGE": {}, "random_baseline": random_gf}

    # [2/4] GAT sweep
    print("\n[2/4] GAT dimension sweep...")
    for d in DIMENSIONS:
        print(f"\n  --- GAT latent_dim={d} ---")
        t0 = time.time()
        out = train_gat_variant(
            G, features, variant_name=f"gat_d{d}",
            epochs=EPOCHS, lr=LR, seed=SEED,
            hidden_dim=HIDDEN_DIM, latent_dim=d,
        )
        elapsed = time.time() - t0
        coords = out["coords"]

        gf_score, _ = compute_gf_for_coords(coords, nodes, go_map, r_vals)
        eff_dim = compute_effective_dim(coords)
        rank = compute_matrix_rank(coords)
        collapse = check_embedding_collapse(coords, f"GAT_d{d}")
        final_loss = out["loss_history"][-1]["loss"] if out["loss_history"] else None

        print(f"    G-F Score:     {gf_score:.4f}")
        print(f"    Eff dim:       {eff_dim:.3f}")
        print(f"    Matrix rank:   {rank}")
        print(f"    Attn entropy:  L1={out['attention_entropy_layer1']['normalized']:.3f}"
              f"  L2={out['attention_entropy_layer2']['normalized']:.3f}")
        print(f"    Collapse:      {collapse['collapsed']}")
        print(f"    Final loss:    {final_loss:.2f}")
        print(f"    Time:          {elapsed:.1f}s")

        results["GAT"][d] = {
            "gf_score": gf_score,
            "effective_dim": eff_dim,
            "matrix_rank": rank,
            "attention_entropy_layer1": out["attention_entropy_layer1"],
            "attention_entropy_layer2": out["attention_entropy_layer2"],
            "collapse": collapse,
            "final_loss": final_loss,
            "train_time_s": round(elapsed, 1),
            "coords_raw": coords,  # kept in memory for plotting, not saved to JSON
        }

    # [3/4] GraphSAGE control
    print("\n[3/4] GraphSAGE control sweep...")
    for d in DIMENSIONS:
        print(f"\n  --- GraphSAGE latent_dim={d} ---")
        t0 = time.time()
        out = train_sage_variant(
            G, features, latent_dim=d, hidden_dim=HIDDEN_DIM,
            epochs=EPOCHS, lr=LR, seed=SEED,
        )
        elapsed = time.time() - t0
        coords = out["coords"]

        gf_score, _ = compute_gf_for_coords(coords, nodes, go_map, r_vals)
        eff_dim = compute_effective_dim(coords)
        rank = compute_matrix_rank(coords)
        collapse = check_embedding_collapse(coords, f"SAGE_d{d}")
        final_loss = out["loss_history"][-1]["loss"] if out["loss_history"] else None

        print(f"    G-F Score:     {gf_score:.4f}")
        print(f"    Eff dim:       {eff_dim:.3f}")
        print(f"    Matrix rank:   {rank}")
        print(f"    Collapse:      {collapse['collapsed']}")
        print(f"    Final loss:    {final_loss:.2f}")
        print(f"    Time:          {elapsed:.1f}s")

        results["GraphSAGE"][d] = {
            "gf_score": gf_score,
            "effective_dim": eff_dim,
            "matrix_rank": rank,
            "collapse": collapse,
            "final_loss": final_loss,
            "train_time_s": round(elapsed, 1),
            "coords_raw": coords,
        }

    # [4/4] Generate figure
    print("\n[4/4] Generating figure...")
    generate_figure(results, fig_dir)

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<15s} {'d':>3s} {'G-F':>8s} {'EffDim':>8s} {'Rank':>6s} "
          f"{'AttnEnt':>9s} {'Collapse':>10s} {'Loss':>10s}")
    print("-" * 80)
    for method in ["GAT", "GraphSAGE"]:
        for d in DIMENSIONS:
            r = results[method][d]
            attn = r.get("attention_entropy_layer1", {}).get("normalized", -1)
            col = "YES" if r["collapse"]["collapsed"] else "no"
            loss_str = f"{r['final_loss']:.1f}" if r["final_loss"] else "N/A"
            print(f"{method:<15s} {d:3d} {r['gf_score']:8.4f} {r['effective_dim']:8.3f} "
                  f"{r['matrix_rank']:6d} {attn:9.3f} {col:>10s} {loss_str:>10s}")
    print("=" * 80)

    # Save results (exclude coords_raw for JSON)
    save_data = {
        "analysis": "Phase 5B: GAT Latent Dimension Sweep",
        "version": "1.0",
        "dimensions": DIMENSIONS,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "lr": LR,
        "random_baseline": random_gf,
        "GAT": {},
        "GraphSAGE": {},
    }
    for method in ["GAT", "GraphSAGE"]:
        for d in DIMENSIONS:
            r = results[method][d]
            save_data[method][str(d)] = {
                k: v for k, v in r.items() if k != "coords_raw"
            }

    output_path = res_dir / "gat_dimension_sweep.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved {output_path}")

    print("\n" + "=" * 70)
    print("Phase 5B complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
