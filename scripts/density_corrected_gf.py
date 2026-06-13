#!/usr/bin/env python3
"""
density_corrected_gf.py
Step 36: Density-Corrected G-F Score Analysis.

At each STRING confidence threshold the network has a different density,
which affects community detection and therefore functional purity.  A
dense network may yield inflated G-F Scores even for random embeddings
simply because communities are easier to detect.

This script computes a **null-model baseline** at each threshold by
permuting GO annotations across nodes (destroying the embedding-function
relationship while preserving network structure).  The density-corrected
G-F Score is then:

    GF_corrected = (GF_method - GF_random) / (1 - GF_random)

where GF_random is the mean GF Score over multiple GO-label permutations.

A corrected Kendall's W is computed to evaluate whether rank agreement
across thresholds strengthens after removing the density confound.

Design
------
  * Reuses threshold gradient networks from multimodal_functional_anchoring.
  * 4 fast methods: Spectral, DeepWalk, Node2Vec, PCA.
  * 6 thresholds: {400, 500, 600, 700, 800, 900}.
  * 20 GO-label permutations per threshold for stable null estimate.
  * Integration interval: [0.05, 0.422] (same as yeast paper default).

Output
------
  results/density_corrected_gf.json
  figures/Fig22_density_corrected.png
"""

import sys
import os
import json
import gzip
import time
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir,
    load_curated_network, load_full_STRING_network,
    spectral_embedding_from_graph, deepwalk_from_graph,
    node2vec_from_graph, rescale_coordinates, compute_gf_curve,
    compute_gf_score, compute_centrality_features,
    GF_R_MIN, GF_R_MAX,
)
from sklearn.decomposition import PCA as SkPCA

# ---- Configuration ----
THRESHOLDS = [400, 500, 600, 700, 800, 900]
METHODS = ["Spectral", "DeepWalk", "Node2Vec", "PCA"]
R_MIN_GRID = 0.05
R_MAX_GRID = 0.55
N_POINTS = 100
TARGET_STD = 0.3
N_PERMUTATIONS = 20


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


def compute_embedding(G, nodes_ann, method):
    """Compute 2-D embedding for a single method. Returns (n, 2) array."""
    if method == "Spectral":
        coords = spectral_embedding_from_graph(G, nodelist=nodes_ann)
    elif method == "DeepWalk":
        coords = deepwalk_from_graph(G, walk_length=20, walks_per_node=10,
                                     window_size=5, dimensions=2, seed=SEED)
    elif method == "Node2Vec":
        coords = node2vec_from_graph(G, walk_length=20, walks_per_node=10,
                                     window_size=5, dimensions=2,
                                     p=0.5, q=2.0, seed=SEED)
    elif method == "PCA":
        features = compute_centrality_features(G, nodes_ann)
        features_c = features - features.mean(axis=0)
        cov = features_c.T @ features_c / (len(nodes_ann) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        coords = features_c @ eigvecs[:, -2:]
    else:
        raise ValueError(f"Unknown method: {method}")
    return rescale_coordinates(coords, TARGET_STD)


def compute_random_baseline(coords, nodes_ann, go_map_sub, r_vals,
                            n_perm, rng):
    """Compute mean random baseline GF Score via GO-label permutation.

    At each permutation, GO annotations are shuffled across nodes,
    destroying the embedding-function relationship while preserving
    network and embedding structure.

    Returns (mean_gf, std_gf, list_of_gf_scores).
    """
    gf_scores = []
    node_list = list(nodes_ann)
    all_go_terms = [go_map_sub.get(n, []) for n in node_list]

    for _ in range(n_perm):
        # Permute GO assignments
        perm = rng.permutation(len(node_list))
        perm_go_map = {}
        for i, n in enumerate(node_list):
            perm_go_map[n] = all_go_terms[perm[i]]

        purities, _ = compute_gf_curve(coords, node_list, perm_go_map, r_vals)
        score = compute_gf_score(r_vals, purities,
                                 r_min=GF_R_MIN, r_max=GF_R_MAX)
        gf_scores.append(float(score))

    return float(np.mean(gf_scores)), float(np.std(gf_scores)), gf_scores


def density_correction(gf_method, gf_random):
    """Apply density correction formula.

    GF_corrected = (GF_method - GF_random) / (1 - GF_random)

    Edge cases:
      - If gf_random >= 1.0: returns 0.0 (ceiling effect)
      - If result < 0: clips to 0.0 (method no better than random)
    """
    if gf_random >= 1.0 - 1e-10:
        return 0.0
    corrected = (gf_method - gf_random) / (1.0 - gf_random)
    return max(0.0, float(corrected))


def kendalls_w(rankings):
    """Compute Kendall's coefficient of concordance.

    Parameters
    ----------
    rankings : list of lists
        Each inner list is a ranking (1 = best) for one condition.
    """
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
    sorted_methods = sorted(methods, key=lambda m: score_dict.get(m, 0),
                            reverse=True)
    return {m: i + 1 for i, m in enumerate(sorted_methods)}


def generate_figure(threshold_results, figures_dir):
    """Generate 3-panel figure for density-corrected analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)
    thresholds = sorted(threshold_results.keys())

    # Panel A: Raw vs corrected GF Scores
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(thresholds))
    width = 0.18
    for mi, method in enumerate(METHODS):
        raw_scores = [threshold_results[t]["raw"][method] for t in thresholds]
        corr_scores = [threshold_results[t]["corrected"][method]
                       for t in thresholds]
        offset = (mi - 1.5) * width
        ax.bar(x + offset - width / 4, raw_scores, width / 2,
               label=f"{method} (raw)", alpha=0.5,
               color=plt.cm.Set2(mi / len(METHODS)))
        ax.bar(x + offset + width / 4, corr_scores, width / 2,
               label=f"{method} (corr)", alpha=1.0,
               color=plt.cm.Set2(mi / len(METHODS)),
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel("STRING Combined Score Threshold")
    ax.set_ylabel("G-F Score")
    ax.set_title("(A) Raw vs Density-Corrected G-F Scores")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Random baseline vs threshold
    ax = fig.add_subplot(gs[0, 1])
    random_means = [threshold_results[t]["random_mean"] for t in thresholds]
    random_stds = [threshold_results[t]["random_std"] for t in thresholds]
    random_means_arr = np.array(random_means)
    random_stds_arr = np.array(random_stds)
    ax.errorbar(thresholds, random_means_arr, yerr=random_stds_arr,
                fmt="o-", color="#e74c3c", markersize=7, capsize=4,
                linewidth=2, label="Random baseline")
    # Also plot mean method GF for reference
    mean_method_gf = []
    for t in thresholds:
        scores = list(threshold_results[t]["raw"].values())
        mean_method_gf.append(np.mean(scores))
    ax.plot(thresholds, mean_method_gf, "s--", color="#3498db", markersize=7,
            linewidth=2, label="Mean method GF")
    ax.set_xlabel("STRING Combined Score Threshold")
    ax.set_ylabel("G-F Score")
    ax.set_title("(B) Random Baseline vs Network Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Corrected rank heatmap
    ax = fig.add_subplot(gs[0, 2])
    rank_matrix = np.zeros((len(thresholds), len(METHODS)))
    for ti, t in enumerate(thresholds):
        corr = threshold_results[t]["corrected"]
        sorted_m = sorted(METHODS, key=lambda m: corr.get(m, 0), reverse=True)
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
                    ha="center", va="center", fontsize=10,
                    fontweight="bold")
    ax.set_title("(C) Corrected Method Rank Across Thresholds")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out_file = figures_dir / "Fig22_density_corrected.png"
    plt.savefig(str(out_file), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_file}")


def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    figures_dir = get_figures_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load GO annotations
    print("Loading curated GO annotations...")
    _, _, go_map = load_curated_network(data_dir)
    print(f"  {len(go_map)} annotated genes")

    r_vals = np.linspace(R_MIN_GRID, R_MAX_GRID, N_POINTS)

    # ---- Process each threshold ----
    print(f"\n{'=' * 60}")
    print(f"Density-Corrected G-F Score Analysis")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Methods: {METHODS}")
    print(f"  Permutations: {N_PERMUTATIONS}")
    print(f"  Integration interval: [{GF_R_MIN}, {GF_R_MAX}]")
    print(f"{'=' * 60}")

    threshold_results = {}

    for threshold in THRESHOLDS:
        print(f"\n--- Threshold {threshold} ---")
        t0 = time.time()

        # Build network
        G = build_network_at_threshold(data_dir, threshold)
        annotated = sorted(set(G.nodes()) & set(go_map.keys()))
        print(f"  Network: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, {len(annotated)} annotated")

        if len(annotated) < 20:
            print("  Skipping (too few annotated nodes)")
            continue

        G_ann = G.subgraph(annotated).copy()
        nodes_ann = sorted(G_ann.nodes())
        go_map_sub = {n: go_map[n] for n in nodes_ann if n in go_map}

        # Compute embeddings and real GF scores
        raw_scores = {}
        embeddings_cache = {}
        for method in METHODS:
            try:
                coords = compute_embedding(G_ann, nodes_ann, method)
                embeddings_cache[method] = coords
                purities, _ = compute_gf_curve(
                    coords, nodes_ann, go_map_sub, r_vals)
                score = compute_gf_score(r_vals, purities,
                                         r_min=GF_R_MIN, r_max=GF_R_MAX)
                raw_scores[method] = float(score)
                print(f"  [{method}] raw GF = {score:.4f}")
            except Exception as e:
                print(f"  [{method}] Error: {e}")
                raw_scores[method] = 0.0

        # Compute random baseline (GO-label permutation)
        # Use first successful method's coordinates for null model
        # (all methods share the same node set, so random baseline is
        #  identical in expectation regardless of which coords we use)
        random_gf_scores = {}
        perm_gf_lists = []
        for method in METHODS:
            if method not in embeddings_cache:
                continue
            coords = embeddings_cache[method]
            mean_gf, std_gf, gf_list = compute_random_baseline(
                coords, nodes_ann, go_map_sub, r_vals, N_PERMUTATIONS, rng)
            random_gf_scores[method] = {
                "mean": mean_gf, "std": std_gf, "scores": gf_list
            }
            perm_gf_lists.append(gf_list)
            print(f"  [{method}] random GF = {mean_gf:.4f} "
                  f"(std={std_gf:.4f})")

        # Average random baseline across methods for a single threshold-level
        # baseline (method-independent property of network density)
        all_perm_scores = np.array(perm_gf_lists)  # (n_methods, n_perm)
        random_mean = float(np.mean(all_perm_scores))
        random_std = float(np.std(np.mean(all_perm_scores, axis=0)))

        # Density correction
        corrected_scores = {}
        for method in METHODS:
            gf_raw = raw_scores.get(method, 0.0)
            # Use method-specific random baseline if available,
            # else use cross-method average
            if method in random_gf_scores:
                gf_rand = random_gf_scores[method]["mean"]
            else:
                gf_rand = random_mean
            corrected_scores[method] = density_correction(gf_raw, gf_rand)
            print(f"  [{method}] corrected GF = "
                  f"{corrected_scores[method]:.4f}  "
                  f"(raw={gf_raw:.4f}, rand={gf_rand:.4f})")

        elapsed = time.time() - t0
        threshold_results[threshold] = {
            "raw": raw_scores,
            "random_per_method": {
                m: {"mean": v["mean"], "std": v["std"]}
                for m, v in random_gf_scores.items()
            },
            "random_mean": random_mean,
            "random_std": random_std,
            "corrected": corrected_scores,
            "n_nodes": G_ann.number_of_nodes(),
            "n_edges": G_ann.number_of_edges(),
            "n_annotated": len(annotated),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  Elapsed: {elapsed:.1f}s")

    # ---- Kendall's W comparison ----
    print(f"\n{'=' * 60}")
    print("Kendall's W Comparison")
    print(f"{'=' * 60}")

    valid_thresholds = sorted([
        t for t in threshold_results
        if len(threshold_results[t]["raw"]) == len(METHODS)
        and len(threshold_results[t]["corrected"]) == len(METHODS)
    ])

    W_raw = None
    W_corrected = None
    if len(valid_thresholds) >= 2:
        raw_rankings = []
        corr_rankings = []
        for t in valid_thresholds:
            raw = threshold_results[t]["raw"]
            corr = threshold_results[t]["corrected"]
            raw_sorted = sorted(METHODS, key=lambda m: raw.get(m, 0),
                                reverse=True)
            corr_sorted = sorted(METHODS, key=lambda m: corr.get(m, 0),
                                 reverse=True)
            raw_rankings.append([raw_sorted.index(m) + 1 for m in METHODS])
            corr_rankings.append([corr_sorted.index(m) + 1 for m in METHODS])

        W_raw = kendalls_w(raw_rankings)
        W_corrected = kendalls_w(corr_rankings)
        print(f"  Raw Kendall's W:       {W_raw:.4f}")
        print(f"  Corrected Kendall's W: {W_corrected:.4f}")
        delta_W = W_corrected - W_raw
        print(f"  Delta W: {delta_W:+.4f}")
        if delta_W > 0:
            print("  -> Density correction INCREASES rank agreement")
            print("     (density was masking true method differences)")
        elif delta_W < 0:
            print("  -> Density correction DECREASES rank agreement")
            print("     (density was contributing to apparent agreement)")
        else:
            print("  -> Density correction has no effect on rank agreement")
    else:
        print("  Not enough valid thresholds for Kendall's W")

    # ---- Save results ----
    output = {
        "thresholds_tested": THRESHOLDS,
        "valid_thresholds": valid_thresholds,
        "methods": METHODS,
        "n_permutations": N_PERMUTATIONS,
        "gf_integration_interval": [GF_R_MIN, GF_R_MAX],
        "threshold_results": {
            str(t): {
                "raw": v["raw"],
                "random_per_method": v["random_per_method"],
                "random_mean": v["random_mean"],
                "random_std": v["random_std"],
                "corrected": v["corrected"],
                "n_nodes": v["n_nodes"],
                "n_edges": v["n_edges"],
                "n_annotated": v["n_annotated"],
                "elapsed_s": v["elapsed_s"],
            }
            for t, v in threshold_results.items()
        },
        "kendalls_w_raw": W_raw,
        "kendalls_w_corrected": W_corrected,
        "delta_w": float(W_corrected - W_raw) if (W_raw is not None and
                                                     W_corrected is not None
                                                     ) else None,
        "n_annotated_genes": len(go_map),
    }

    out_file = results_dir / "density_corrected_gf.json"
    with open(str(out_file), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")

    # ---- Generate figure ----
    generate_figure(threshold_results, figures_dir)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("DENSITY-CORRECTED G-F SCORE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Thresholds: {len(valid_thresholds)} valid / "
          f"{len(THRESHOLDS)} tested")
    print(f"Permutations per threshold: {N_PERMUTATIONS}")
    if W_raw is not None:
        print(f"Raw W = {W_raw:.4f}  ->  Corrected W = {W_corrected:.4f}  "
              f"(delta = {W_corrected - W_raw:+.4f})")

    # Print corrected ranking summary at each threshold
    print(f"\nCorrected G-F Score summary:")
    for t in valid_thresholds:
        corr = threshold_results[t]["corrected"]
        ranking = sorted(corr.items(), key=lambda x: x[1], reverse=True)
        rank_str = ", ".join(f"{m}={s:.4f}" for m, s in ranking)
        print(f"  T={t}: {rank_str}")


if __name__ == "__main__":
    main()
