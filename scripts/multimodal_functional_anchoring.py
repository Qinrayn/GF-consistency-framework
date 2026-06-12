#!/usr/bin/env python3
"""
multimodal_functional_anchoring.py
Step 34: Multi-Modal Functional Anchoring Analysis.

Evaluates whether G-F Score rankings are consistent across different
STRING evidence channels (interaction types).  This addresses:
"Do method rankings depend on the type of biological evidence
underlying the PPI network, or are they robust across evidence modalities?"

Design
------
  1. **STRING threshold gradient** (no extra data needed):
     Re-builds the yeast STRING network at combined_score thresholds
     {400, 500, 600, 700, 800, 900}, computing GF scores with 4 fast
     methods (Spectral, DeepWalk, Node2Vec, PCA) at each threshold.
     This probes how network density affects method quality.

  2. **STRING channel-specific networks** (requires full links file):
     If ``4932.protein.links.full.v11.5.txt.gz`` is available, builds
     networks from individual channels (experiments, coexpression,
     textmining, databases) and computes GF scores.

Output
------
  results/multimodal_anchoring.json
  figures/Fig20_multimodal_anchoring.png
"""

import sys
import os
import json
import gzip
import time
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir,
    load_curated_network,
    spectral_embedding_from_graph, deepwalk_from_graph,
    node2vec_from_graph, rescale_coordinates, compute_gf_curve,
    compute_gf_score, compute_centrality_features,
    GF_R_MIN, GF_R_MAX,
)
from sklearn.decomposition import PCA as SkPCA

# ---- Configuration ----
THRESHOLDS = [400, 500, 600, 700, 800, 900]
METHODS = ["Spectral", "DeepWalk", "Node2Vec", "PCA"]
R_MIN = 0.05
R_MAX = 0.55
N_POINTS = 100
TARGET_STD = 0.3

# Channel names and their column indices in STRING full links file
# Columns: protein1 protein2 neighborhood(nbr) neighborhood_transferred
#          cooccurrence coexpression coexpression_transferred
#          experiments experiments_transferred database database_transferred
#          textmining textmining_transferred combined_score
STRING_CHANNELS = {
    "coexpression": 5,       # coexpression
    "experiments": 7,        # experiments
    "database": 9,           # database
    "textmining": 11,        # textmining
}
CHANNEL_THRESHOLD = 400     # minimum channel-specific score


def load_string_full_links(data_dir):
    """Try to load the STRING full links file (with per-channel scores).

    Returns None if the file is not available.
    """
    full_file = data_dir / "4932.protein.links.full.v11.5.txt.gz"
    if not full_file.exists():
        # Try v12.0
        full_file = data_dir / "4932.protein.links.full.v12.0.txt.gz"
    if not full_file.exists():
        return None

    print(f"  Found full links file: {full_file.name}")
    channel_edges = {ch: [] for ch in STRING_CHANNELS}

    with gzip.open(str(full_file), "rt", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 14:
                continue
            p1 = parts[0].split(".")[1]
            p2 = parts[1].split(".")[1]
            for ch_name, col_idx in STRING_CHANNELS.items():
                score = int(parts[col_idx])
                if score >= CHANNEL_THRESHOLD:
                    channel_edges[ch_name].append((p1, p2, score))

    for ch, edges in channel_edges.items():
        print(f"    {ch}: {len(edges)} edges (score >= {CHANNEL_THRESHOLD})")
    return channel_edges


def build_network_at_threshold(data_dir, threshold):
    """Build yeast STRING network at a given combined_score threshold."""
    string_file = data_dir / "4932.protein.links.v11.5.txt.gz"
    G = nx.Graph()
    with gzip.open(str(string_file), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p1, p2, score = parts
            if int(score) >= threshold:
                p1_clean = p1.split(".")[1]
                p2_clean = p2.split(".")[1]
                G.add_edge(p1_clean, p2_clean)
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def build_channel_network(edges, min_nodes=50):
    """Build a network from channel-specific edges."""
    G = nx.Graph()
    for p1, p2, score in edges:
        G.add_edge(p1, p2, weight=score)
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def compute_embeddings_for_network(G, go_map):
    """Compute 4 fast embeddings and return GF scores."""
    # Restrict to annotated nodes
    annotated = sorted(set(G.nodes()) & set(go_map.keys()))
    if len(annotated) < 20:
        return {}

    G_ann = G.subgraph(annotated).copy()
    nodes_ann = sorted(G_ann.nodes())
    go_map_sub = {n: go_map[n] for n in nodes_ann if n in go_map}

    if G_ann.number_of_nodes() < 10:
        return {}

    results = {}
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    for method in METHODS:
        try:
            if method == "Spectral":
                coords = spectral_embedding_from_graph(G_ann, nodelist=nodes_ann)
            elif method == "DeepWalk":
                coords = deepwalk_from_graph(G_ann, walk_length=20,
                                              walks_per_node=10, window_size=5,
                                              dimensions=2, seed=SEED)
            elif method == "Node2Vec":
                coords = node2vec_from_graph(G_ann, walk_length=20,
                                              walks_per_node=10, window_size=5,
                                              dimensions=2, p=0.5, q=2.0, seed=SEED)
            elif method == "PCA":
                features = compute_centrality_features(G_ann, nodes_ann)
                features_c = features - features.mean(axis=0)
                cov = features_c.T @ features_c / (len(nodes_ann) - 1)
                eigvals, eigvecs = np.linalg.eigh(cov)
                coords = features_c @ eigvecs[:, -2:]
            else:
                continue

            coords = rescale_coordinates(coords, TARGET_STD)
            purities, _ = compute_gf_curve(coords, nodes_ann, go_map_sub, r_vals)
            score = compute_gf_score(r_vals, purities)
            results[method] = float(score)
        except Exception as e:
            print(f"    [{method}] Error: {e}")
    return results


def kendalls_w(rankings):
    """Compute Kendall's coefficient of concordance."""
    k = len(rankings)
    n = len(rankings[0])
    if n <= 1 or k <= 1:
        return 1.0
    rank_sums = np.sum(rankings, axis=0)
    mean_rank_sum = np.mean(rank_sums)
    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = (12.0 * S) / (k ** 2 * n * (n ** 2 - 1))
    return float(W)


def ranks_from_scores(score_dict, methods):
    """Convert scores to ranks (1 = best/highest)."""
    sorted_methods = sorted(methods, key=lambda m: score_dict.get(m, 0), reverse=True)
    return {m: i + 1 for i, m in enumerate(sorted_methods)}


def generate_figure(threshold_results, channel_results, figures_dir):
    """Generate multi-panel figure for multi-modal anchoring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: GF Score vs threshold
    ax = axes[0]
    thresholds = sorted(threshold_results.keys())
    for method in METHODS:
        scores = [threshold_results[t].get(method, 0) for t in thresholds]
        ax.plot(thresholds, scores, "o-", label=method, markersize=5)
    ax.set_xlabel("STRING Combined Score Threshold")
    ax.set_ylabel("G-F Score")
    ax.set_title("(A) G-F Score vs Network Density Threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Rank heatmap across thresholds
    ax = axes[1]
    all_methods_sorted = METHODS
    rank_matrix = np.zeros((len(thresholds), len(METHODS)))
    for ti, t in enumerate(thresholds):
        scores = threshold_results[t]
        sorted_m = sorted(METHODS, key=lambda m: scores.get(m, 0), reverse=True)
        for mi, m in enumerate(sorted_m):
            rank_matrix[ti, METHODS.index(m)] = mi + 1

    im = ax.imshow(rank_matrix, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=45, ha="right")
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([str(t) for t in thresholds])
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            ax.text(j, i, f"{int(rank_matrix[i, j])}",
                    ha="center", va="center", fontsize=9)
    ax.set_title("(B) Method Rank Across Thresholds")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel C: Channel-specific scores (if available)
    ax = axes[2]
    if channel_results:
        channels = sorted(channel_results.keys())
        x = np.arange(len(channels))
        width = 0.18
        for mi, method in enumerate(METHODS):
            scores = [channel_results[ch].get(method, 0) for ch in channels]
            ax.bar(x + mi * width, scores, width, label=method)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(channels, rotation=30, ha="right")
        ax.set_ylabel("G-F Score")
        ax.set_title("(C) G-F Score by STRING Evidence Channel")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "Channel data not available\n"
                "(requires STRING full links file)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.set_title("(C) Channel-Specific Analysis")
        ax.axis("off")

    plt.tight_layout()
    out_file = figures_dir / "Fig20_multimodal_anchoring.png"
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

    # Load GO annotations
    print("Loading curated GO annotations...")
    _, _, go_map = load_curated_network(data_dir)
    print(f"  {len(go_map)} annotated genes")

    # ---- Part 1: Threshold gradient ----
    print(f"\n{'=' * 60}")
    print("Part 1: STRING threshold gradient analysis")
    print(f"{'=' * 60}")

    threshold_results = {}
    for threshold in THRESHOLDS:
        print(f"\n--- Threshold {threshold} ---")
        t0 = time.time()
        G = build_network_at_threshold(data_dir, threshold)
        annotated = sorted(set(G.nodes()) & set(go_map.keys()))
        print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"{len(annotated)} annotated")

        if len(annotated) < 20:
            print("  Skipping (too few annotated nodes)")
            continue

        scores = compute_embeddings_for_network(G, go_map)
        threshold_results[threshold] = scores
        elapsed = time.time() - t0
        for m, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"    {m}: {s:.4f}")
        print(f"  Elapsed: {elapsed:.1f}s")

    # Compute Kendall's W across thresholds
    valid_thresholds = sorted([t for t in threshold_results
                               if len(threshold_results[t]) == len(METHODS)])
    if len(valid_thresholds) >= 2:
        rankings = []
        for t in valid_thresholds:
            scores = threshold_results[t]
            sorted_m = sorted(METHODS, key=lambda m: scores.get(m, 0), reverse=True)
            rankings.append([sorted_m.index(m) + 1 for m in METHODS])
        W_threshold = kendalls_w(rankings)
    else:
        W_threshold = None
    print(f"\nKendall's W across thresholds: {W_threshold}")

    # ---- Part 2: Channel-specific networks ----
    print(f"\n{'=' * 60}")
    print("Part 2: STRING channel-specific analysis")
    print(f"{'=' * 60}")

    channel_results = {}
    full_links = load_string_full_links(data_dir)
    if full_links is not None:
        for ch_name, edges in full_links.items():
            if not edges:
                continue
            print(f"\n--- Channel: {ch_name} ---")
            G_ch = build_channel_network(edges)
            annotated = sorted(set(G_ch.nodes()) & set(go_map.keys()))
            print(f"  Network: {G_ch.number_of_nodes()} nodes, "
                  f"{G_ch.number_of_edges()} edges, {len(annotated)} annotated")

            if len(annotated) < 20:
                print("  Skipping (too few annotated nodes)")
                continue

            scores = compute_embeddings_for_network(G_ch, go_map)
            channel_results[ch_name] = scores
            for m, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"    {m}: {s:.4f}")
    else:
        print("  Full links file not found. Skipping channel analysis.")
        print("  To enable: download 4932.protein.links.full.v11.5.txt.gz from STRING")

    # Channel Kendall's W
    W_channel = None
    if channel_results:
        valid_channels = [ch for ch in channel_results
                          if len(channel_results[ch]) == len(METHODS)]
        if len(valid_channels) >= 2:
            rankings = []
            for ch in valid_channels:
                scores = channel_results[ch]
                sorted_m = sorted(METHODS, key=lambda m: scores.get(m, 0), reverse=True)
                rankings.append([sorted_m.index(m) + 1 for m in METHODS])
            W_channel = kendalls_w(rankings)
        print(f"\nKendall's W across channels: {W_channel}")

    # ---- Save results ----
    print(f"\n{'=' * 60}")
    print("Saving results...")

    output = {
        "threshold_gradient": {
            str(t): scores for t, scores in threshold_results.items()
        },
        "thresholds_tested": THRESHOLDS,
        "valid_thresholds": valid_thresholds,
        "methods": METHODS,
        "kendalls_w_threshold": W_threshold,
        "channel_analysis": {
            ch: scores for ch, scores in channel_results.items()
        } if channel_results else None,
        "kendalls_w_channel": W_channel,
        "n_annotated_genes": len(go_map),
    }
    out_file = results_dir / "multimodal_anchoring.json"
    with open(str(out_file), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_file}")

    # ---- Generate figure ----
    generate_figure(threshold_results, channel_results, figures_dir)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("MULTI-MODAL FUNCTIONAL ANCHORING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Threshold gradient: {len(valid_thresholds)} thresholds, "
          f"W={W_threshold}")
    if channel_results:
        print(f"Channel analysis: {len(channel_results)} channels, "
              f"W={W_channel}")
    else:
        print("Channel analysis: not available (no full links file)")


if __name__ == "__main__":
    main()
