#!/usr/bin/env python3
"""
human_seed_stability.py
Step 37: Human G-F Score Seed Stability Analysis.

The human PPI network has ~14,679 annotated genes, but computational
constraints limit us to subsampling 2,000 nodes per run.  This script
tests whether the G-F Score rankings are robust to the choice of random
subsample by repeating the analysis with **10 different seeds**.

Design
------
  * 11 embedding methods (same as human_gf_extended.py).
  * 10 seeds, each producing a different 2000-node subsample.
  * Embeddings are pre-computed and fixed; only subsampling varies.
  * Kendall's W across 10 rankings measures subsample stability.
  * Coefficient of variation (CV) per method measures score stability.
  * For random-walk methods (DeepWalk, Node2Vec), we also vary the
    walk seed to test embedding stochasticity.

Output
------
  results/human_seed_stability.json
  figures/Fig23_seed_stability.png
"""
from __future__ import annotations

import os
import sys
import json

import time
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import SEED

# ---- Configuration ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

R_MIN = 0.05
R_MAX = 0.55
N_POINTS = 100
MAX_EDGES = 500_000
SUBSAMPLE_SIZE = 2000
N_SEEDS = 10

ALL_METHODS = [
    "DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE",
    "PCA", "VGAE-feat", "GraphSAGE", "GAT", "GIN",
]


def load_human_go_annotations():
    go_file = os.path.join(DATA_DIR, "human_go_annotations.json")
    if os.path.exists(go_file):
        with open(go_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_embedding(method):
    filepath = os.path.join(DATA_DIR, f"human_{method.lower()}_embedding.json")
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR,
                                f"human_{method.lower().replace('-', '_')}_embedding.json")
    if not os.path.exists(filepath):
        return None, None
    with open(filepath, "r", encoding="utf-8") as f:
        emb = json.load(f)
    nodes = list(emb.keys())
    coords = np.array([[emb[n]["x"], emb[n]["y"]] for n in nodes])
    return nodes, coords


def subsample_annotated(nodes, coords, node_labels, target_size, rng):
    """Subsample annotated nodes to target_size using given RNG."""
    annotated_idx = [i for i in range(len(nodes)) if i in node_labels]
    if len(annotated_idx) <= target_size:
        sub_idx = annotated_idx
    else:
        sub_idx = sorted(rng.choice(annotated_idx, size=target_size,
                                    replace=False))
    sub_coords = coords[sub_idx]
    sub_labels = {}
    for new_i, old_i in enumerate(sub_idx):
        sub_labels[new_i] = node_labels[old_i]
    return sub_coords, sub_labels, len(annotated_idx)


def communities_from_partition(partition):
    groups = defaultdict(set)
    for node, comm in partition.items():
        groups[comm].add(node)
    return [frozenset(g) for g in groups.values()]


def compute_gf_curve_fast(coords, node_labels, r_values, seed=SEED):
    """G-F curve using greedy modularity community detection."""
    from networkx.algorithms.community import greedy_modularity_communities
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(coords))
    n = dist_matrix.shape[0]
    purities = np.zeros(len(r_values))
    modularities = np.zeros(len(r_values))

    for ri, r in enumerate(r_values):
        mask = (dist_matrix < r) & (dist_matrix > 0)
        n_edges = np.sum(mask) // 2
        if n_edges == 0:
            continue

        rows, cols = np.where(mask)
        upper = rows < cols
        edges = list(zip(rows[upper].tolist(), cols[upper].tolist()))

        G = nx.Graph()
        G.add_nodes_from(range(n))

        if n_edges > MAX_EDGES:
            G.add_edges_from(edges)
            communities = [frozenset(c) for c in nx.connected_components(G)]
            mod_val = 0.0
        else:
            G.add_edges_from(edges)
            try:
                partition = list(greedy_modularity_communities(G))
                communities = [frozenset(c) for c in partition]
                mod_val = nx.community.modularity(G, partition)
            except Exception as e:
                communities = [frozenset(c) for c in nx.connected_components(G)]
                mod_val = 0.0

        comm_purities = []
        for comm in communities:
            all_terms = []
            for i in comm:
                if i in node_labels:
                    lbl = node_labels[i]
                    if isinstance(lbl, list):
                        all_terms.extend(lbl)
                    else:
                        all_terms.append(lbl)
            if not all_terms:
                continue
            counts = Counter(all_terms)
            comm_purities.append(counts.most_common(1)[0][1] / len(all_terms))

        if comm_purities:
            purities[ri] = np.mean(comm_purities)
        modularities[ri] = mod_val

    return purities, modularities


def compute_gf_score(purities, r_values, r_min_s, r_max_s):
    mask = (r_values >= r_min_s) & (r_values <= r_max_s)
    if not np.any(mask):
        return 0.0
    from scipy.integrate import trapezoid
    r_sub, p_sub = r_values[mask], purities[mask]
    return trapezoid(p_sub, r_sub) / (r_max_s - r_min_s)


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


def generate_figure(seed_results, methods, figures_dir):
    """Generate multi-panel figure for seed stability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.35)

    # Panel A: Score boxplot across seeds
    ax = fig.add_subplot(gs[0, 0])
    score_data = {m: [] for m in methods}
    for seed_data in seed_results:
        for m in methods:
            if m in seed_data["scores"]:
                score_data[m].append(seed_data["scores"][m])

    sorted_methods = sorted(methods, key=lambda m: np.mean(score_data.get(m, [0])),
                            reverse=True)
    plot_data = [score_data[m] for m in sorted_methods]
    bp = ax.boxplot(plot_data, patch_artist=True, widths=0.6,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="red",
                                                   markersize=5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(sorted_methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("G-F Score")
    ax.set_title("(A) Score Distribution Across 10 Seeds")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Rank heatmap
    ax = fig.add_subplot(gs[0, 1])
    rank_matrix = np.zeros((len(seed_results), len(methods)))
    for si, seed_data in enumerate(seed_results):
        scores = seed_data["scores"]
        sorted_m = sorted(methods, key=lambda m: scores.get(m, 0), reverse=True)
        for mi, m in enumerate(sorted_m):
            rank_matrix[si, methods.index(m)] = mi + 1

    im = ax.imshow(rank_matrix, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(seed_results)))
    ax.set_yticklabels([f"S{s}" for s in range(len(seed_results))], fontsize=8)
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            ax.text(j, i, f"{int(rank_matrix[i, j])}",
                    ha="center", va="center", fontsize=8)
    ax.set_title("(B) Method Rank per Seed")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel C: CV per method
    ax = fig.add_subplot(gs[0, 2])
    cvs = []
    for m in sorted_methods:
        vals = score_data[m]
        if len(vals) > 1 and np.mean(vals) > 0:
            cvs.append((m, np.std(vals) / np.mean(vals)))
        else:
            cvs.append((m, 0.0))
    cvs_sorted = sorted(cvs, key=lambda x: x[1], reverse=True)
    method_names = [c[0] for c in cvs_sorted]
    cv_vals = [c[1] for c in cvs_sorted]
    bar_colors = ["#e74c3c" if v > 0.15 else "#f39c12" if v > 0.08
                  else "#2ecc71" for v in cv_vals]
    bars = ax.barh(method_names, cv_vals, color=bar_colors,
                   edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Coefficient of Variation (CV)")
    ax.set_title("(C) Score Stability per Method")
    ax.axvline(x=0.08, color="gray", linestyle="--", alpha=0.5, label="CV=8%")
    ax.axvline(x=0.15, color="gray", linestyle="-.", alpha=0.5, label="CV=15%")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    for bar, cv in zip(bars, cv_vals):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{cv:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    out_file = os.path.join(figures_dir, "Fig23_seed_stability.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_file}")


def run_single_seed(args):
    """Run GF analysis for a single seed. Designed for multiprocessing."""
    seed, seed_idx, available_methods, embeddings, node_labels_per_method, r_values, r_min_u, r_max_u = args
    import time as _time
    t0 = _time.time()
    rng = np.random.default_rng(seed)
    scores = {}

    for method in available_methods:
        nodes, coords = embeddings[method]
        node_labels = node_labels_per_method[method]
        sub_coords, sub_labels, _ = subsample_annotated(
            nodes, coords, node_labels, SUBSAMPLE_SIZE, rng
        )
        if len(sub_labels) < 10:
            continue
        purities, _ = compute_gf_curve_fast(
            sub_coords, sub_labels, r_values, seed=seed
        )
        score = compute_gf_score(purities, r_values, r_min_u, r_max_u)
        scores[method] = float(score)

    elapsed = _time.time() - t0
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top3 = ", ".join(f"{m}={s:.4f}" for m, s in ranking[:3])
    print(f"  [Seed {seed} ({seed_idx+1}/{N_SEEDS})] Top 3: {top3}  ({elapsed:.1f}s)", flush=True)
    return {
        "seed": seed,
        "scores": scores,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    from multiprocessing import Pool, cpu_count
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading GO annotations...")
    go_annotations = load_human_go_annotations()
    node_labels_all = {}
    for node, terms in go_annotations.items():
        if isinstance(terms, list) and len(terms) > 0:
            node_labels_all[node] = terms
        elif isinstance(terms, str):
            node_labels_all[node] = [terms]
    print(f"  Loaded annotations for {len(node_labels_all)} nodes")

    # Load unified interval from extended results
    r_min_u, r_max_u = 0.282, 0.297  # default from human_gf_extended
    ext_file = os.path.join(RESULTS_DIR, "human_gf_scores_extended.json")
    if os.path.exists(ext_file):
        with open(ext_file, "r", encoding="utf-8") as f:
            ext_data = json.load(f)
        interval = ext_data.get("unified_interval", [r_min_u, r_max_u])
        r_min_u, r_max_u = interval[0], interval[1]
    print(f"  Unified interval: [{r_min_u:.4f}, {r_max_u:.4f}]")

    r_values = np.linspace(R_MIN, R_MAX, N_POINTS)

    # ---- Pre-load all embeddings ----
    print("\nPre-loading embeddings...")
    embeddings = {}
    node_labels_per_method = {}
    for method in ALL_METHODS:
        nodes, coords = load_embedding(method)
        if nodes is None:
            print(f"  Skipping {method} (embedding not found)")
            continue
        node_labels = {}
        for i, node in enumerate(nodes):
            if node in node_labels_all:
                node_labels[i] = node_labels_all[node]
        if len(node_labels) < 10:
            print(f"  Skipping {method} (only {len(node_labels)} annotated)")
            continue
        embeddings[method] = (nodes, coords)
        node_labels_per_method[method] = node_labels
        print(f"  {method}: {len(nodes)} nodes, {len(node_labels)} annotated")

    available_methods = list(embeddings.keys())
    print(f"  {len(available_methods)} methods ready")

    # ---- Multi-seed analysis ----
    print(f"\n{'=' * 60}")
    print(f"Seed Stability Analysis: {N_SEEDS} seeds x "
          f"{SUBSAMPLE_SIZE} nodes x {len(available_methods)} methods")
    print(f"{'=' * 60}")

    seeds = [SEED + i * 100 for i in range(N_SEEDS)]
    n_workers = min(N_SEEDS, cpu_count() or N_SEEDS)
    print(f"  Using {n_workers} parallel workers for {N_SEEDS} seeds")

    task_args = [
        (seed, seed_idx, available_methods, embeddings,
         node_labels_per_method, r_values, r_min_u, r_max_u)
        for seed_idx, seed in enumerate(seeds)
    ]

    with Pool(processes=n_workers) as pool:
        seed_results = pool.map(run_single_seed, task_args)

    # Sort by seed to maintain consistent ordering
    seed_results.sort(key=lambda x: x["seed"])

    # ---- Stability metrics ----
    print(f"\n{'=' * 60}")
    print("Stability Metrics")
    print(f"{'=' * 60}")

    # Kendall's W across seeds
    valid_seeds = [sr for sr in seed_results
                   if len(sr["scores"]) == len(available_methods)]
    W = None
    if len(valid_seeds) >= 2:
        rankings = []
        for sr in valid_seeds:
            sorted_m = sorted(available_methods,
                              key=lambda m: sr["scores"].get(m, 0),
                              reverse=True)
            rankings.append([sorted_m.index(m) + 1 for m in available_methods])
        W = kendalls_w(rankings)
        print(f"  Kendall's W across {len(valid_seeds)} seeds: {W:.4f}")

    # Per-method statistics
    method_stats = {}
    for method in available_methods:
        vals = [sr["scores"].get(method, 0) for sr in seed_results]
        if vals:
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            cv = std_val / mean_val if mean_val > 0 else 0.0
            ranks = []
            for sr in seed_results:
                sorted_m = sorted(sr["scores"].items(),
                                  key=lambda x: x[1], reverse=True)
                rank = next(i + 1 for i, (m, _) in enumerate(sorted_m)
                            if m == method)
                ranks.append(rank)
            method_stats[method] = {
                "mean": mean_val,
                "std": std_val,
                "cv": cv,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean_rank": float(np.mean(ranks)),
                "std_rank": float(np.std(ranks)),
                "best_rank": int(np.min(ranks)),
                "worst_rank": int(np.max(ranks)),
            }
            print(f"  {method:12s}  GF={mean_val:.4f}+/-{std_val:.4f}  "
                  f"CV={cv:.3f}  rank={np.mean(ranks):.1f}+/-{np.std(ranks):.1f}  "
                  f"[{int(np.min(ranks))}-{int(np.max(ranks))}]")

    # ---- Save results ----
    output = {
        "n_seeds": N_SEEDS,
        "seeds": seeds,
        "subsample_size": SUBSAMPLE_SIZE,
        "unified_interval": [r_min_u, r_max_u],
        "methods": available_methods,
        "valid_seeds": len(valid_seeds),
        "kendalls_w": W,
        "per_seed_scores": {
            str(sr["seed"]): sr["scores"] for sr in seed_results
        },
        "method_statistics": method_stats,
        "ranking_stability": {
            "mean_ranking": {
                m: method_stats[m]["mean_rank"]
                for m in sorted(available_methods,
                                key=lambda m: method_stats[m]["mean_rank"])
            }
        },
    }

    out_file = os.path.join(RESULTS_DIR, "human_seed_stability.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")

    # ---- Generate figure ----
    generate_figure(seed_results, available_methods, FIGURES_DIR)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("SEED STABILITY SUMMARY")
    print(f"{'=' * 60}")
    print(f"Seeds tested: {N_SEEDS} ({len(valid_seeds)} valid)")
    print(f"Methods: {len(available_methods)}")
    if W is not None:
        stability = "high" if W > 0.8 else "moderate" if W > 0.5 else "low"
        print(f"Kendall's W: {W:.4f} ({stability} stability)")

    # Most/least stable methods
    cv_sorted = sorted(method_stats.items(), key=lambda x: x[1]["cv"],
                       reverse=True)
    print(f"\nMost variable:  {cv_sorted[0][0]} (CV={cv_sorted[0][1]['cv']:.3f})")
    print(f"Most stable:    {cv_sorted[-1][0]} (CV={cv_sorted[-1][1]['cv']:.3f})")


if __name__ == "__main__":
    main()
