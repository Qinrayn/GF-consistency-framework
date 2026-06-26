#!/usr/bin/env python3
"""
scale_gradient.py
Step 31: Scale-Dependent Topological Coupling Analysis.

Subsamples the yeast full STRING network at multiple scales (500, 1000, 2000,
4000 nodes) and computes G-F Scores to study how method rankings change with
network size.  This addresses the question: "Is embedding quality a scale-
dependent property, or do method rankings remain stable across network sizes?"

Design:
  - Annotated nodes (153 curated GO-annotated proteins) are always retained.
  - Non-annotated nodes are added uniformly at random to reach target size.
  - Four fast embedding methods are used (Spectral, DeepWalk, Node2Vec, PCA).
  - GF curves computed on the 153 annotated nodes at each scale.
  - Kendall's W measures rank stability across scales.

Output:
  - results/scale_gradient.json
  - figures/Fig18_scale_gradient.png
"""
from __future__ import annotations

import sys
import json
import time
import random
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir,
    load_full_STRING_network,
    spectral_embedding_from_graph, deepwalk_from_graph,
    node2vec_from_graph, rescale_coordinates, compute_gf_curve,
    compute_gf_score,
)
from sklearn.decomposition import PCA

# ---- Configuration ----
SCALE_SIZES = [500, 1000, 2000, 4000]
R_MIN = 0.05
R_MAX = 0.55
N_POINTS = 200
GF_R_MIN = 0.05
GF_R_MAX = 0.422
TARGET_STD = 0.3

# Fast methods only (DM and MDS are O(n^2+) — too slow at 4000)
METHODS = ["Spectral", "DeepWalk", "Node2Vec", "PCA"]


def subsample_network(G_full, annotated_nodes, target_size, rng):
    """Subsample G_full to target_size nodes, retaining all annotated nodes.

    Returns the subgraph and sorted node list.
    """
    annotated_set = set(annotated_nodes)
    full_nodes = sorted(G_full.nodes())
    non_annotated = [n for n in full_nodes if n not in annotated_set]

    n_keep = min(target_size - len(annotated_nodes), len(non_annotated))
    if n_keep < 0:
        n_keep = 0

    kept_non_ann = set(rng.choice(non_annotated, size=n_keep, replace=False))
    kept_nodes = sorted(annotated_set | kept_non_ann)

    G_sub = G_full.subgraph(kept_nodes).copy()

    # Take largest connected component that contains most annotated nodes
    ccs = sorted(nx.connected_components(G_sub), key=len, reverse=True)
    if len(ccs) > 1:
        # Merge small components into main if they contain annotated nodes
        main_cc = ccs[0]
        for cc in ccs[1:]:
            if cc & annotated_set:
                main_cc = main_cc | cc
        G_sub = G_sub.subgraph(main_cc).copy()

    return G_sub, sorted(G_sub.nodes())


def embed_spectral(G, nodes):
    """Spectral embedding."""
    coords = spectral_embedding_from_graph(G)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_deepwalk(G, nodes):
    """DeepWalk embedding."""
    coords = deepwalk_from_graph(G, walk_length=20, walks_per_node=10,
                                  window_size=5, dimensions=2, seed=SEED)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_node2vec(G, nodes):
    """Node2Vec embedding."""
    coords = node2vec_from_graph(G, walk_length=20, walks_per_node=10,
                                  window_size=5, dimensions=2, p=0.5, q=2.0, seed=SEED)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_pca(G, nodes):
    """PCA on adjacency matrix."""
    node_list = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    adj = np.zeros((n, n))
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i, j] = adj[j, i] = 1.0

    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(adj)
    return rescale_coordinates(coords, target_std=TARGET_STD)


EMBED_FUNCTIONS = {
    "Spectral": embed_spectral,
    "DeepWalk": embed_deepwalk,
    "Node2Vec": embed_node2vec,
    "PCA": embed_pca,
}


def kendalls_w(rank_matrix):
    """Kendall's W coefficient of concordance.

    Parameters
    ----------
    rank_matrix : (k, n) array
        k rankings of n items; each row is a ranking (1 = best).
    """
    k, n = rank_matrix.shape
    rank_sums = rank_matrix.sum(axis=0)
    mean_sum = rank_sums.mean()
    S = ((rank_sums - mean_sum) ** 2).sum()
    W = 12.0 * S / (k ** 2 * n * (n ** 2 - 1))
    return float(W)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = get_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load full network ----
    print("Loading full yeast STRING network...")
    G_full = load_full_STRING_network()
    full_nodes = sorted(G_full.nodes())
    print(f"  Full network: {len(full_nodes)} nodes, {G_full.number_of_edges()} edges")

    # ---- Load GO annotations ----
    data_dir = get_data_dir()
    with open(data_dir / "gene_go_map.json", encoding="utf-8") as f:
        go_map = json.load(f)

    annotated_nodes = sorted(set(go_map.keys()) & set(full_nodes))
    print(f"  Annotated nodes: {len(annotated_nodes)}")

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # ---- Run at each scale ----
    all_results = {}  # scale -> method -> {gf_score, purities, ...}

    for scale in SCALE_SIZES:
        print(f"\n{'='*60}")
        print(f"Scale: {scale} nodes")
        print(f"{'='*60}")

        if scale >= len(full_nodes):
            # Use full network
            G_sub, sub_nodes = G_full, full_nodes
            print(f"  Using full network ({len(sub_nodes)} nodes)")
        else:
            G_sub, sub_nodes = subsample_network(
                G_full, annotated_nodes, scale, rng
            )
            print(f"  Subsampled to {len(sub_nodes)} nodes")

        # Identify annotated nodes in this subsample
        sub_ann = sorted(set(annotated_nodes) & set(sub_nodes))
        print(f"  Annotated in subsample: {len(sub_ann)}")

        if len(sub_ann) < 20:
            print("  Skipping: too few annotated nodes")
            continue

        node_to_idx = {n: i for i, n in enumerate(sub_nodes)}
        ann_indices = [node_to_idx[n] for n in sub_ann]

        scale_results = {}

        for method_name in METHODS:
            print(f"\n  [{method_name}] Computing embedding...")
            t0 = time.time()
            random.seed(SEED)
            np.random.seed(SEED)

            try:
                embed_fn = EMBED_FUNCTIONS[method_name]
                coords = embed_fn(G_sub, sub_nodes)
                ann_coords = coords[ann_indices]

                # Compute GF curve
                purities, modularities = compute_gf_curve(
                    ann_coords, sub_ann, go_map, r_vals
                )

                # Compute GF score
                gf_score = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)

                elapsed = time.time() - t0
                scale_results[method_name] = {
                    "gf_score": float(gf_score),
                    "peak_purity": float(max(purities)),
                    "elapsed_s": round(elapsed, 1),
                }
                print(f"    G-F Score = {gf_score:.4f}, "
                      f"Peak purity = {max(purities):.4f}, "
                      f"Time = {elapsed:.1f}s")

            except Exception as e:
                print(f"    FAILED: {e}")
                scale_results[method_name] = {
                    "gf_score": None,
                    "error": str(e),
                }

        all_results[scale] = scale_results

    # ---- Rank analysis across scales ----
    print(f"\n{'='*60}")
    print("Cross-Scale Rank Analysis")
    print(f"{'='*60}")

    # Build score matrix
    score_matrix = {}
    for scale in SCALE_SIZES:
        if scale in all_results:
            score_matrix[scale] = {}
            for m in METHODS:
                if m in all_results[scale] and all_results[scale][m]["gf_score"] is not None:
                    score_matrix[scale][m] = all_results[scale][m]["gf_score"]

    # Compute ranks at each scale
    rank_matrix = []
    valid_scales = []
    for scale in SCALE_SIZES:
        if scale not in score_matrix or len(score_matrix[scale]) < 2:
            continue
        scores = score_matrix[scale]
        sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranks = {m: i + 1 for i, (m, _) in enumerate(sorted_methods)}
        rank_row = [ranks.get(m, 0) for m in METHODS if m in ranks]
        if rank_row:
            rank_matrix.append(rank_row)
            valid_scales.append(scale)
            rank_str = ", ".join(f"{m}#{ranks[m]}" for m in METHODS if m in ranks)
            print(f"  Scale {scale:>5d}: {rank_str}")

    # Kendall's W
    W = 0.0
    if len(rank_matrix) >= 2:
        rank_arr = np.array(rank_matrix)
        W = kendalls_w(rank_arr)
        print(f"\n  Kendall's W = {W:.4f}")

    # ---- Rank stability heatmap data ----
    # Spearman between consecutive scales
    from scipy.stats import spearmanr
    pairwise_correlations = []
    for i in range(len(valid_scales) - 1):
        s1, s2 = valid_scales[i], valid_scales[i + 1]
        shared = sorted(set(score_matrix[s1].keys()) & set(score_matrix[s2].keys()))
        if len(shared) >= 3:
            v1 = [score_matrix[s1][m] for m in shared]
            v2 = [score_matrix[s2][m] for m in shared]
            rho, p = spearmanr(v1, v2)
            pairwise_correlations.append({
                "scale_from": s1,
                "scale_to": s2,
                "spearman_rho": float(rho),
                "p_value": float(p),
            })
            print(f"  {s1} -> {s2}: rho = {rho:.4f} (P = {p:.4f})")

    # ---- Save results ----
    output = {
        "scale_sizes": SCALE_SIZES,
        "methods": METHODS,
        "n_annotated": len(annotated_nodes),
        "full_network_size": len(full_nodes),
        "results": {
            str(scale): {
                m: all_results[scale][m] for m in all_results.get(scale, {})
            } for scale in all_results
        },
        "kendalls_w": W,
        "valid_scales": valid_scales,
        "pairwise_correlations": pairwise_correlations,
        "gf_r_interval": [GF_R_MIN, GF_R_MAX],
    }

    output_file = results_dir / "scale_gradient.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_file}")

    # ---- Generate figure ----
    _generate_figure(all_results, valid_scales, METHODS, W,
                     pairwise_correlations, figures_dir)


def _generate_figure(all_results, valid_scales, methods, W,
                     pairwise_corr, figures_dir):
    """Generate Fig18: Scale gradient analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.35)

    # (A) G-F Score vs network size (line plot)
    ax1 = fig.add_subplot(gs[0, 0])
    colours = {"Spectral": "#e74c3c", "DeepWalk": "#3498db",
               "Node2Vec": "#2ecc71", "PCA": "#9b59b6"}
    markers = {"Spectral": "o", "DeepWalk": "s", "Node2Vec": "^", "PCA": "D"}

    for method in methods:
        x_vals, y_vals = [], []
        for scale in valid_scales:
            if method in all_results.get(scale, {}):
                score = all_results[scale][method].get("gf_score")
                if score is not None:
                    x_vals.append(scale)
                    y_vals.append(score)
        if x_vals:
            ax1.plot(x_vals, y_vals, f"{markers[method]}-",
                     color=colours.get(method, "gray"), linewidth=2,
                     markersize=8, label=method)

    ax1.set_xlabel("Network Size (nodes)")
    ax1.set_ylabel("G-F Score")
    ax1.set_title(f"(A) G-F Score vs Network Scale  (W={W:.3f})")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(valid_scales)
    ax1.set_xticklabels([str(s) for s in valid_scales])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # (B) Rank heatmap across scales
    ax2 = fig.add_subplot(gs[0, 1])
    rank_data = []
    for scale in valid_scales:
        scores = {}
        for m in methods:
            if m in all_results.get(scale, {}):
                s = all_results[scale][m].get("gf_score")
                if s is not None:
                    scores[m] = s
        if scores:
            sorted_m = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranks = {m: i + 1 for i, (m, _) in enumerate(sorted_m)}
            rank_data.append([ranks.get(m, 0) for m in methods])

    if rank_data:
        rank_arr = np.array(rank_data)
        im = ax2.imshow(rank_arr, cmap="RdYlGn_r", aspect="auto", vmin=1,
                        vmax=len(methods))
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax2.set_yticks(range(len(valid_scales)))
        ax2.set_yticklabels([str(s) for s in valid_scales])
        ax2.set_xlabel("Method")
        ax2.set_ylabel("Network Size")

        for i in range(len(valid_scales)):
            for j in range(len(methods)):
                ax2.text(j, i, str(rank_arr[i, j]),
                         ha="center", va="center", fontsize=10, fontweight="bold")

        plt.colorbar(im, ax=ax2, label="Rank", shrink=0.8)

    ax2.set_title("(B) Method Rank Heatmap")

    # (C) Pairwise Spearman between consecutive scales
    ax3 = fig.add_subplot(gs[0, 2])
    if pairwise_corr:
        x_labels = [f"{c['scale_from']}\n→\n{c['scale_to']}"
                    for c in pairwise_corr]
        rho_vals = [c["spearman_rho"] for c in pairwise_corr]
        bar_colors = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0
                      else "#e74c3c" for r in rho_vals]
        bars = ax3.bar(range(len(rho_vals)), rho_vals, color=bar_colors,
                       edgecolor="black", linewidth=0.5)
        ax3.set_xticks(range(len(rho_vals)))
        ax3.set_xticklabels(x_labels, fontsize=7)
        ax3.set_ylabel("Spearman ρ")
        ax3.set_title("(C) Consecutive-Scale Rank Correlation")
        ax3.axhline(y=0, color="black", linewidth=0.8)
        ax3.set_ylim(-1.1, 1.1)
        ax3.grid(True, alpha=0.3, axis="y")

        for bar, r in zip(bars, rho_vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, r + 0.03,
                     f"{r:.2f}", ha="center", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                 transform=ax3.transAxes)

    fig.suptitle("Fig 18 — Scale-Dependent Topological Coupling",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig_path = figures_dir / "Fig18_scale_gradient.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
