#!/usr/bin/env python3
"""
human_ic_weighted_gf.py
Step 38: IC-Weighted G-F Score on Human PPI Network.

Standard G-F purity treats all GO terms as equal tokens.  After True Path
Rule propagation, general ancestor terms (e.g. "biological_process") appear
in every gene, diluting the denominator and inflating purity towards 1.0.

IC-weighted purity down-weights non-specific terms:

    purity_IC(C) = max_t [count(t,C) * IC(t)] / sum_t [count(t,C) * IC(t)]

where IC(t) = -log(freq(t) / N) is the corpus-based Information Content.

This script computes IC-weighted G-F Scores for all 11 methods on the
human STRING network and compares rankings with standard G-F Scores.

Design
------
  * Same subsample (seed=42, 2000 nodes) as human_gf_extended.py for
    direct comparability.
  * IC computed from the full human GO annotation corpus.
  * Louvain community detection at each r threshold.
  * Spearman correlation between standard and IC-weighted rankings.

Output
------
  results/human_ic_weighted_gf.json
  figures/Fig24_ic_weighted_gf.png
"""

import os
import sys
import json
import math
import pickle
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
EPS = 1e-12

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


def compute_ic(go_map):
    """Compute corpus-based Information Content.

    IC(t) = -log(freq(t) / N)

    Parameters
    ----------
    go_map : dict
        gene -> list of GO terms

    Returns
    -------
    dict : go_term -> IC value
    """
    term_gene_count = Counter()
    for gene, terms in go_map.items():
        seen = set(terms)
        for t in seen:
            term_gene_count[t] += 1
    n_genes = len(go_map)
    ic = {}
    for term, count in term_gene_count.items():
        p = count / max(n_genes, 1)
        ic[term] = -math.log(p + EPS)
    return ic


def subsample_annotated(nodes, coords, node_labels, target_size, rng):
    annotated_idx = [i for i in range(len(nodes)) if i in node_labels]
    if len(annotated_idx) <= target_size:
        sub_idx = annotated_idx
    else:
        sub_idx = sorted(rng.choice(annotated_idx, size=target_size,
                                    replace=False))
    sub_coords = coords[sub_idx]
    sub_labels = {}
    sub_node_names = {}
    for new_i, old_i in enumerate(sub_idx):
        sub_labels[new_i] = node_labels[old_i]
        sub_node_names[new_i] = nodes[old_i]
    return sub_coords, sub_labels, sub_node_names, len(annotated_idx)


def communities_from_partition(partition):
    groups = defaultdict(set)
    for node, comm in partition.items():
        groups[comm].add(node)
    return [frozenset(g) for g in groups.values()]


def purity_ic_weighted(comm_indices, node_labels, ic):
    """IC-weighted purity for a single community.

    purity_IC = max_t [count(t) * IC(t)] / sum_t [count(t) * IC(t)]
    """
    all_terms = []
    for i in comm_indices:
        if i in node_labels:
            lbl = node_labels[i]
            if isinstance(lbl, list):
                all_terms.extend(lbl)
            else:
                all_terms.append(lbl)
    if not all_terms:
        return 0.0
    counts = Counter(all_terms)
    weighted_sum = 0.0
    max_weighted = 0.0
    for term, cnt in counts.items():
        w = cnt * ic.get(term, 0.0)
        weighted_sum += w
        if w > max_weighted:
            max_weighted = w
    if weighted_sum < EPS:
        return 0.0
    return max_weighted / weighted_sum


def purity_standard(comm_indices, node_labels):
    """Standard purity (count-based).

    purity = max_t count(t) / total_terms
    """
    all_terms = []
    for i in comm_indices:
        if i in node_labels:
            lbl = node_labels[i]
            if isinstance(lbl, list):
                all_terms.extend(lbl)
            else:
                all_terms.append(lbl)
    if not all_terms:
        return 0.0
    counts = Counter(all_terms)
    return counts.most_common(1)[0][1] / len(all_terms)


def compute_gf_curve_dual(coords, node_labels, ic, r_values, seed=SEED):
    """Compute both standard and IC-weighted G-F curves simultaneously.

    Returns (std_purities, ic_purities, modularities).
    """
    import community as community_louvain
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(coords))
    n = dist_matrix.shape[0]
    std_purities = np.zeros(len(r_values))
    ic_purities = np.zeros(len(r_values))
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
                partition = community_louvain.best_partition(G, random_state=seed)
                communities = communities_from_partition(partition)
                mod_val = community_louvain.modularity(partition, G)
            except Exception:
                communities = [frozenset(c) for c in nx.connected_components(G)]
                mod_val = 0.0

        std_purs = []
        ic_purs = []
        for comm in communities:
            comm_list = list(comm)
            sp = purity_standard(comm_list, node_labels)
            ip = purity_ic_weighted(comm_list, node_labels, ic)
            if sp > 0:
                std_purs.append(sp)
            if ip > 0:
                ic_purs.append(ip)

        if std_purs:
            std_purities[ri] = np.mean(std_purs)
        if ic_purs:
            ic_purities[ri] = np.mean(ic_purs)
        modularities[ri] = mod_val

    return std_purities, ic_purities, modularities


def compute_gf_score(purities, r_values, r_min_s, r_max_s):
    mask = (r_values >= r_min_s) & (r_values <= r_max_s)
    if not np.any(mask):
        return 0.0
    from scipy.integrate import trapezoid
    r_sub, p_sub = r_values[mask], purities[mask]
    return trapezoid(p_sub, r_sub) / (r_max_s - r_min_s)


def spearman_correlation(x, y):
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def generate_figure(results, methods, figures_dir):
    """Generate multi-panel figure for IC-weighted analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)

    # Panel A: Standard vs IC-weighted scores
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))
    width = 0.35
    std_scores = [results[m]["standard_gf"] for m in methods]
    ic_scores = [results[m]["ic_weighted_gf"] for m in methods]
    ax.bar(x - width / 2, std_scores, width, label="Standard",
           color="#3498db", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, ic_scores, width, label="IC-weighted",
           color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("G-F Score")
    ax.set_title("(A) Standard vs IC-Weighted G-F Scores")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Rank comparison
    ax = fig.add_subplot(gs[0, 1])
    std_sorted = sorted(methods, key=lambda m: results[m]["standard_gf"],
                        reverse=True)
    ic_sorted = sorted(methods, key=lambda m: results[m]["ic_weighted_gf"],
                       reverse=True)
    std_ranks = {m: i + 1 for i, m in enumerate(std_sorted)}
    ic_ranks = {m: i + 1 for i, m in enumerate(ic_sorted)}

    for m in methods:
        ax.annotate("", xy=(ic_ranks[m], 1), xytext=(std_ranks[m], 0),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50",
                                    lw=1.5))
    ax.set_xlim(0.5, len(methods) + 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Standard", "IC-weighted"])
    ax.set_ylabel("Rank (1 = best)")
    ax.set_ylim(len(methods) + 0.5, 0.5)
    ax.set_title("(B) Rank Trajectory")
    # Add method labels
    for m in methods:
        ax.plot([std_ranks[m], ic_ranks[m]], [0, 1], 'o-',
                markersize=4, linewidth=1, alpha=0.7)
    ax.grid(True, alpha=0.3)

    # Panel C: IC distribution
    ax = fig.add_subplot(gs[0, 2])
    ratios = [results[m]["ic_weighted_gf"] / (results[m]["standard_gf"] + 1e-10)
              for m in methods]
    colors = ["#e74c3c" if r < 0.9 else "#2ecc71" if r > 1.1 else "#95a5a6"
              for r in ratios]
    bars = ax.barh(methods, ratios, color=colors,
                   edgecolor="black", linewidth=0.5)
    ax.axvline(x=1.0, color="black", linewidth=0.8)
    ax.set_xlabel("IC-weighted / Standard ratio")
    ax.set_title("(C) IC Correction Factor")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{r:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    out_file = os.path.join(figures_dir, "Fig24_ic_weighted_gf.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_file}")


def main():
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---- Load GO annotations and compute IC ----
    print("Loading GO annotations...")
    go_annotations = load_human_go_annotations()
    node_labels_all = {}
    for node, terms in go_annotations.items():
        if isinstance(terms, list) and len(terms) > 0:
            node_labels_all[node] = terms
        elif isinstance(terms, str):
            node_labels_all[node] = [terms]
    print(f"  Loaded annotations for {len(node_labels_all)} nodes")

    # Compute IC from full corpus
    print("\nComputing Information Content (IC)...")
    ic = compute_ic(node_labels_all)
    print(f"  {len(ic)} unique GO terms")
    ic_vals = list(ic.values())
    print(f"  IC range: [{min(ic_vals):.2f}, {max(ic_vals):.2f}], "
          f"mean={np.mean(ic_vals):.2f}")

    # Load unified interval
    r_min_u, r_max_u = 0.282, 0.297
    ext_file = os.path.join(RESULTS_DIR, "human_gf_scores_extended.json")
    if os.path.exists(ext_file):
        with open(ext_file, "r", encoding="utf-8") as f:
            ext_data = json.load(f)
        interval = ext_data.get("unified_interval", [r_min_u, r_max_u])
        r_min_u, r_max_u = interval[0], interval[1]
    print(f"\n  Unified interval: [{r_min_u:.4f}, {r_max_u:.4f}]")

    r_values = np.linspace(R_MIN, R_MAX, N_POINTS)

    # ---- Compute dual curves for each method ----
    print(f"\n{'=' * 60}")
    print("IC-Weighted G-F Score Analysis (11 methods)")
    print(f"{'=' * 60}")

    results = {}
    for method in ALL_METHODS:
        print(f"\n[{method}]")
        t0 = time.time()
        nodes, coords = load_embedding(method)
        if nodes is None:
            print(f"  Skipping (embedding not found)")
            continue

        node_labels = {}
        for i, node in enumerate(nodes):
            if node in node_labels_all:
                node_labels[i] = node_labels_all[node]
        if len(node_labels) < 10:
            print(f"  Skipping (only {len(node_labels)} annotated)")
            continue

        sub_coords, sub_labels, sub_names, total_ann = subsample_annotated(
            nodes, coords, node_labels, SUBSAMPLE_SIZE, rng
        )
        print(f"  Subsampled {len(sub_labels)}/{total_ann} annotated nodes")

        std_purs, ic_purs, mods = compute_gf_curve_dual(
            sub_coords, sub_labels, ic, r_values, seed=SEED
        )
        elapsed = time.time() - t0

        std_gf = compute_gf_score(std_purs, r_values, r_min_u, r_max_u)
        ic_gf = compute_gf_score(ic_purs, r_values, r_min_u, r_max_u)

        results[method] = {
            "standard_gf": float(std_gf),
            "ic_weighted_gf": float(ic_gf),
            "ratio": float(ic_gf / (std_gf + 1e-10)),
            "std_peak": float(np.max(std_purs)),
            "ic_peak": float(np.max(ic_purs)),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  Standard: GF={std_gf:.4f}, peak={np.max(std_purs):.4f}")
        print(f"  IC-weighted: GF={ic_gf:.4f}, peak={np.max(ic_purs):.4f}")
        print(f"  Ratio: {ic_gf / (std_gf + 1e-10):.3f}  ({elapsed:.1f}s)")

    if not results:
        print("\nNo results computed.")
        return

    available_methods = list(results.keys())

    # ---- Rank comparison ----
    print(f"\n{'=' * 60}")
    print("Rank Comparison")
    print(f"{'=' * 60}")

    std_sorted = sorted(available_methods,
                        key=lambda m: results[m]["standard_gf"], reverse=True)
    ic_sorted = sorted(available_methods,
                       key=lambda m: results[m]["ic_weighted_gf"], reverse=True)

    std_ranks = {m: i + 1 for i, m in enumerate(std_sorted)}
    ic_ranks = {m: i + 1 for i, m in enumerate(ic_sorted)}

    print(f"\n{'Method':<15} {'Std Rank':>8} {'IC Rank':>8} {'Shift':>8}")
    print("-" * 45)
    for m in std_sorted:
        shift = std_ranks[m] - ic_ranks[m]
        direction = "up" if shift > 0 else ("down" if shift < 0 else "---")
        print(f"{m:<15} {std_ranks[m]:>8} {ic_ranks[m]:>8} "
              f"{shift:>+4d} ({direction})")

    # Spearman correlation
    std_vals = [results[m]["standard_gf"] for m in available_methods]
    ic_vals = [results[m]["ic_weighted_gf"] for m in available_methods]
    rho, p_val = spearman_correlation(std_vals, ic_vals)
    print(f"\nSpearman correlation (standard vs IC-weighted): "
          f"rho={rho:.4f}, P={p_val:.4f}")

    # ---- Save results ----
    output = {
        "unified_interval": [r_min_u, r_max_u],
        "subsample_size": SUBSAMPLE_SIZE,
        "n_ic_terms": len(ic),
        "ic_stats": {
            "min": float(min(ic_vals)),
            "max": float(max(ic_vals)),
            "mean": float(np.mean(ic_vals)),
            "median": float(np.median(ic_vals)),
        },
        "methods": available_methods,
        "results": results,
        "standard_ranking": [
            {"method": m, "rank": std_ranks[m], "score": results[m]["standard_gf"]}
            for m in std_sorted
        ],
        "ic_weighted_ranking": [
            {"method": m, "rank": ic_ranks[m], "score": results[m]["ic_weighted_gf"]}
            for m in ic_sorted
        ],
        "spearman_correlation": {
            "rho": rho,
            "p_value": p_val,
        },
    }

    out_file = os.path.join(RESULTS_DIR, "human_ic_weighted_gf.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")

    # ---- Generate figure ----
    generate_figure(results, available_methods, FIGURES_DIR)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("IC-WEIGHTED G-F SCORE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Methods: {len(available_methods)}")
    print(f"Spearman rho (std vs IC): {rho:.4f} (P={p_val:.4f})")
    print(f"\nTop 5 IC-weighted methods:")
    for i, m in enumerate(ic_sorted[:5]):
        print(f"  {i+1}. {m}: GF_IC={results[m]['ic_weighted_gf']:.4f}  "
              f"(std={results[m]['standard_gf']:.4f})")


if __name__ == "__main__":
    main()
