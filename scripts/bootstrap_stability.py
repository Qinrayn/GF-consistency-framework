#!/usr/bin/env python3
"""
bootstrap_stability.py
Step 32: Bootstrap Stability Analysis for G-F Score Rankings.

Tests whether the observed G-F Score rank ordering is statistically robust
by computing GF Scores on 30 bootstrap resamples (80% sampling with
replacement) of the 153 curated GO-annotated genes.

Uses Louvain community detection for O(n log n) performance.

Analyses:
  1. 95% bootstrap confidence intervals for each method's G-F Score.
  2. Coefficient of variation (CV = sigma/mu) per method.
  3. Pairwise rank stability: P(method_i > method_j) across resamples.
  4. Significance of adjacent rank differences (e.g., #1 vs #2).

Output:
  - results/bootstrap_stability.json
  - figures/Fig19_bootstrap_stability.png
"""

import sys
import json
import time
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir,
    get_embeddings_dir, load_curated_network,
    load_embedding, compute_gf_score,
    ALL_METHODS, GF_R_MIN, GF_R_MAX,
    precompute_distance_matrix,
)

N_BOOTSTRAP = 30
SAMPLE_FRAC = 0.80
R_MIN = 0.05
R_MAX = 0.55
N_POINTS = 100
MAX_EDGES = 200_000


def compute_gf_curve_louvain(coords, nodes, go_map, r_vals):
    """Fast G-F curve using Louvain community detection.

    Mirrors compute_gf_curve but uses python-louvain for O(n log n)
    community detection instead of greedy_modularity_communities.
    """
    import community as community_louvain
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(coords))
    n = dist_matrix.shape[0]
    purities = np.zeros(len(r_vals))
    modularities = np.zeros(len(r_vals))

    # Pre-sort edges by distance for incremental graph building
    iu = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[iu]
    sort_idx = np.argsort(edge_dists)
    sorted_rows = iu[0][sort_idx]
    sorted_cols = iu[1][sort_idx]
    sorted_d = edge_dists[sort_idx]

    r_order = np.argsort(r_vals)
    G_r = nx.Graph()
    G_r.add_nodes_from(range(n))
    edge_ptr = 0
    n_edges_total = len(sorted_d)
    _cache = {}

    for rank, orig_idx in enumerate(r_order):
        r = float(r_vals[orig_idx])

        while edge_ptr < n_edges_total and sorted_d[edge_ptr] < r:
            G_r.add_edge(int(sorted_rows[edge_ptr]), int(sorted_cols[edge_ptr]))
            edge_ptr += 1

        ne = G_r.number_of_edges()
        if ne == 0:
            continue

        if ne in _cache:
            communities, mod_val = _cache[ne]
        else:
            if ne > MAX_EDGES:
                # Too dense — use connected components
                communities = [frozenset(c) for c in nx.connected_components(G_r)]
                mod_val = 0.0
            else:
                try:
                    partition = community_louvain.best_partition(G_r, random_state=SEED)
                    groups = defaultdict(set)
                    for node, comm in partition.items():
                        groups[comm].add(node)
                    communities = [frozenset(g) for g in groups.values()]
                    mod_val = community_louvain.modularity(partition, G_r)
                except Exception:
                    communities = [frozenset(c) for c in nx.connected_components(G_r)]
                    mod_val = 0.0
            _cache[ne] = (communities, mod_val)

        # Compute purity (standard formula: all GO terms)
        comm_purities = []
        for comm in communities:
            all_terms = []
            for idx in comm:
                node = nodes[idx]
                if node in go_map:
                    terms = go_map[node]
                    if terms:
                        all_terms.extend(terms)
            if not all_terms:
                continue
            counts = Counter(all_terms)
            comm_purities.append(counts.most_common(1)[0][1] / len(all_terms))

        if comm_purities:
            purities[orig_idx] = np.mean(comm_purities)
        modularities[orig_idx] = mod_val

    return purities.tolist(), modularities.tolist()


def load_all_embeddings(methods, emb_dir):
    """Load embeddings for all methods."""
    embeddings = {}
    for method in methods:
        try:
            coords, nodes = load_embedding(method, "153", emb_dir)
            if coords is not None:
                embeddings[method] = (coords, nodes)
        except Exception as e:
            print(f"  Warning: could not load {method}: {e}")
    return embeddings


def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = get_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()

    # ---- Load data ----
    print("Loading curated network and GO annotations...")
    G, nodes, go_map = load_curated_network()
    print(f"  Network: {len(nodes)} nodes, {G.number_of_edges()} edges")

    # ---- Load embeddings ----
    print("Loading embeddings...")
    embeddings = load_all_embeddings(ALL_METHODS, emb_dir)
    methods = sorted(embeddings.keys())
    print(f"  Loaded: {methods}")

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # ---- Observed GF Scores ----
    print("\nComputing observed G-F Scores...")
    observed_scores = {}
    for method in methods:
        coords, method_nodes = embeddings[method]
        node_to_idx = {n_i: i for i, n_i in enumerate(method_nodes)}
        valid_nodes = [n_i for n_i in nodes if n_i in node_to_idx and n_i in go_map]
        valid_coords = coords[[node_to_idx[n_i] for n_i in valid_nodes]]

        t0 = time.time()
        purities, _ = compute_gf_curve_louvain(valid_coords, valid_nodes, go_map, r_vals)
        gf_score = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)
        elapsed = time.time() - t0
        observed_scores[method] = float(gf_score)
        print(f"  {method}: {gf_score:.4f} ({elapsed:.1f}s)")

    # ---- Bootstrap resampling ----
    n = len(nodes)
    sample_size = max(20, int(n * SAMPLE_FRAC))
    print(f"\nRunning {N_BOOTSTRAP} bootstrap resamples ({SAMPLE_FRAC*100:.0f}% = {sample_size} genes)...")

    boot_scores = {m: [] for m in methods}
    t_start = time.time()

    for b in range(N_BOOTSTRAP):
        boot_idx = rng.choice(n, size=sample_size, replace=True)
        boot_idx_unique = np.unique(boot_idx)
        boot_nodes = [nodes[i] for i in boot_idx_unique]
        boot_go = {n_i: go_map[n_i] for n_i in boot_nodes if n_i in go_map}

        if len(boot_go) < 10:
            continue

        for method in methods:
            coords, method_nodes = embeddings[method]
            method_node_to_idx = {n_i: i for i, n_i in enumerate(method_nodes)}
            valid_coords_idx = []
            valid_boot_nodes = []
            for bn in boot_nodes:
                if bn in boot_go and bn in method_node_to_idx:
                    valid_coords_idx.append(method_node_to_idx[bn])
                    valid_boot_nodes.append(bn)

            if len(valid_boot_nodes) < 10:
                continue

            valid_coords = coords[valid_coords_idx]
            valid_go = {n_i: boot_go[n_i] for n_i in valid_boot_nodes if n_i in boot_go}

            try:
                purities, _ = compute_gf_curve_louvain(
                    valid_coords, valid_boot_nodes, valid_go, r_vals
                )
                gf = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)
                boot_scores[method].append(float(gf))
            except Exception:
                pass

        if (b + 1) % 5 == 0 or b == 0:
            elapsed = time.time() - t_start
            rate = elapsed / (b + 1)
            remaining = rate * (N_BOOTSTRAP - b - 1)
            print(f"    Resample {b+1}/{N_BOOTSTRAP} done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed_total = time.time() - t_start
    print(f"Bootstrap completed in {elapsed_total:.1f}s")

    # ---- Statistical analysis ----
    print("\nBootstrap Statistics:")
    stats = {}
    for method in methods:
        scores = boot_scores[method]
        if len(scores) < 5:
            continue
        arr = np.array(scores)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        ci_low = float(np.percentile(arr, 2.5))
        ci_high = float(np.percentile(arr, 97.5))
        cv = std / mean if mean > 0 else 0.0

        stats[method] = {
            "observed": observed_scores.get(method, mean),
            "boot_mean": mean,
            "boot_std": std,
            "ci_95": [ci_low, ci_high],
            "cv": cv,
            "n_resamples": len(scores),
        }
        print(f"  {method:12s}: mean={mean:.4f} +/- {std:.4f}, "
              f"95%CI=[{ci_low:.4f}, {ci_high:.4f}], CV={cv:.3f}")

    # ---- Pairwise rank stability ----
    print("\nPairwise Rank Stability (P(row > col)):")
    sorted_methods = sorted(stats.keys(), key=lambda m: -stats[m]["boot_mean"])
    n_methods = len(sorted_methods)
    pairwise = np.zeros((n_methods, n_methods))

    for i, m1 in enumerate(sorted_methods):
        s1 = np.array(boot_scores[m1])
        for j, m2 in enumerate(sorted_methods):
            if i == j:
                pairwise[i, j] = 0.5
                continue
            s2 = np.array(boot_scores[m2])
            min_len = min(len(s1), len(s2))
            if min_len > 0:
                pairwise[i, j] = float(np.mean(s1[:min_len] > s2[:min_len]))

    print("  Adjacent rank P(higher > lower):")
    for i in range(n_methods - 1):
        m1, m2 = sorted_methods[i], sorted_methods[i + 1]
        p = pairwise[i, i + 1]
        sig = "***" if p > 0.99 else "**" if p > 0.95 else "*" if p > 0.9 else "ns"
        print(f"    {m1} > {m2}: P = {p:.3f} ({sig})")

    # ---- Save results ----
    output = {
        "n_bootstrap": N_BOOTSTRAP,
        "sample_fraction": SAMPLE_FRAC,
        "n_genes": len(nodes),
        "n_points": N_POINTS,
        "gf_r_interval": [GF_R_MIN, GF_R_MAX],
        "observed_scores": observed_scores,
        "bootstrap_stats": stats,
        "method_ranking": sorted_methods,
        "pairwise_rank_stability": {
            sorted_methods[i]: {
                sorted_methods[j]: float(pairwise[i, j])
                for j in range(n_methods)
            }
            for i in range(n_methods)
        },
        "adjacent_significance": [
            {
                "higher": sorted_methods[i],
                "lower": sorted_methods[i + 1],
                "p_higher_wins": float(pairwise[i, i + 1]),
                "significant_0.95": bool(pairwise[i, i + 1] > 0.95),
            }
            for i in range(n_methods - 1)
        ],
    }

    output_file = results_dir / "bootstrap_stability.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_file}")

    # ---- Generate figure ----
    _generate_figure(sorted_methods, stats, pairwise, observed_scores, figures_dir)


def _generate_figure(methods, stats, pairwise, observed, figures_dir):
    """Generate Fig19: Bootstrap stability analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.35)

    # (A) Forest plot: observed scores with 95% CI
    ax1 = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(methods))
    means = [stats[m]["boot_mean"] for m in methods]
    ci_lows = [stats[m]["ci_95"][0] for m in methods]
    ci_highs = [stats[m]["ci_95"][1] for m in methods]
    errors_low = [m - l for m, l in zip(means, ci_lows)]
    errors_high = [h - m for m, h in zip(means, ci_highs)]

    ax1.errorbar(means, y_pos,
                 xerr=[errors_low, errors_high],
                 fmt="o", color="#2c3e50", markersize=8,
                 ecolor="#3498db", elinewidth=2, capsize=4, capthick=2)

    obs_vals = [observed.get(m, stats[m]["observed"]) for m in methods]
    ax1.scatter(obs_vals, y_pos, marker="D", color="#e74c3c", s=60,
                zorder=5, label="Observed")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=8)
    ax1.set_xlabel("G-F Score")
    ax1.set_title("(A) Bootstrap 95% CI for G-F Scores")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3, axis="x")

    # (B) Pairwise rank stability heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(pairwise, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods, fontsize=7)

    for i in range(len(methods)):
        for j in range(len(methods)):
            val = pairwise[i, j]
            color = "white" if val > 0.7 or val < 0.3 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6, color=color)

    plt.colorbar(im, ax=ax2, label="P(row > col)", shrink=0.8)
    ax2.set_title("(B) Pairwise Rank Stability")

    # (C) CV bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    cvs = [stats[m]["cv"] for m in methods]
    bar_colors = ["#2ecc71" if cv < 0.1 else "#f39c12" if cv < 0.2
                  else "#e74c3c" for cv in cvs]
    bars = ax3.barh(methods, cvs, color=bar_colors, edgecolor="black",
                    linewidth=0.5)
    ax3.set_xlabel("Coefficient of Variation (CV)")
    ax3.set_title("(C) Score Stability (lower = more stable)")
    ax3.grid(True, alpha=0.3, axis="x")

    for bar, cv in zip(bars, cvs):
        ax3.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{cv:.3f}", va="center", fontsize=8)

    fig.suptitle("Fig 19 — Bootstrap Stability of G-F Score Rankings (30 resamples, 80% sampling)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig_path = figures_dir / "Fig19_bootstrap_stability.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
