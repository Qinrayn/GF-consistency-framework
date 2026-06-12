#!/usr/bin/env python3
"""
human_gf_extended.py
Step 33b: Compute G-F curves, G-F scores, and plateau widths
for **all 11 embedding methods** on the human STRING network.

Combines the existing 6 methods (DM, MDS, Spectral, DeepWalk, Node2Vec,
VGAE) with 5 newly computed methods (PCA, VGAE-feat, GraphSAGE, GAT, GIN)
to produce a unified 11-method human ranking, enabling a full cross-species
comparison.

Output
------
  results/human_gf_scores_extended.json   — 11-method scores & ranking
  results/human_gf_curves_extended.pkl    — raw curve data
"""

import os
import sys
import json
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

R_MIN = 0.05
R_MAX = 0.55
N_POINTS = 100
MAX_EDGES = 500_000
SUBSAMPLE_SIZE = 2000

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
        print(f"  Warning: embedding not found for {method}, skipping")
        return None, None
    with open(filepath, "r", encoding="utf-8") as f:
        emb = json.load(f)
    nodes = list(emb.keys())
    coords = np.array([[emb[n]["x"], emb[n]["y"]] for n in nodes])
    return nodes, coords


def subsample_annotated(nodes, coords, node_labels, target_size, rng):
    annotated_idx = [i for i in range(len(nodes)) if i in node_labels]
    if len(annotated_idx) <= target_size:
        sub_idx = annotated_idx
    else:
        sub_idx = sorted(rng.choice(annotated_idx, size=target_size, replace=False))
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


def compute_gf_curve_fast(coords, node_labels, r_values):
    """G-F curve using Louvain community detection."""
    import community as community_louvain
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
                partition = community_louvain.best_partition(G, random_state=SEED)
                communities = communities_from_partition(partition)
                mod_val = community_louvain.modularity(partition, G)
            except Exception:
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


def compute_plateau_width(purities, r_values, window_frac=0.05, tolerance=0.05):
    n = len(r_values)
    window = max(3, int(n * window_frac))
    rolling_mean = np.convolve(purities, np.ones(window) / window, mode="same")
    in_plateau = np.abs(purities - rolling_mean) <= tolerance
    max_width, start = 0.0, None
    for i in range(n):
        if in_plateau[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                max_width = max(max_width, r_values[i - 1] - r_values[start])
                start = None
    if start is not None:
        max_width = max(max_width, r_values[-1] - r_values[start])
    return max_width


def determine_unified_interval(all_results, r_values):
    valid_mask = np.zeros(len(r_values), dtype=bool)
    for _, (purities, modularities) in all_results.items():
        valid_mask |= (modularities > 0.3) & (purities > 0.40)
    if not np.any(valid_mask):
        return R_MIN, R_MAX
    idx = np.where(valid_mask)[0]
    return r_values[idx[0]], r_values[idx[-1]]


def main():
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading GO annotations...")
    go_annotations = load_human_go_annotations()
    node_labels_all = {}
    for node, terms in go_annotations.items():
        if isinstance(terms, list) and len(terms) > 0:
            node_labels_all[node] = terms  # store all GO terms
        elif isinstance(terms, str):
            node_labels_all[node] = [terms]  # wrap single term in list
    print(f"  Loaded annotations for {len(node_labels_all)} nodes")

    r_values = np.linspace(R_MIN, R_MAX, N_POINTS)
    all_results = {}

    for method in ALL_METHODS:
        print(f"\n{'=' * 60}")
        print(f"Computing G-F curve for {method} ...")
        t0 = time.time()

        nodes, coords = load_embedding(method)
        if nodes is None:
            continue

        node_labels = {}
        for i, node in enumerate(nodes):
            if node in node_labels_all:
                node_labels[i] = node_labels_all[node]

        annotated_count = len(node_labels)
        print(f"  Loaded {len(nodes)} nodes, {annotated_count} annotated")

        if annotated_count < 10:
            print("  Skipping (too few annotations)")
            continue

        sub_coords, sub_labels, total_ann = subsample_annotated(
            nodes, coords, node_labels, SUBSAMPLE_SIZE, rng
        )
        print(f"  Subsampled {len(sub_labels)}/{total_ann} annotated nodes")

        purities, modularities = compute_gf_curve_fast(sub_coords, sub_labels, r_values)
        elapsed = time.time() - t0
        all_results[method] = (purities, modularities)

        peak_idx = np.argmax(purities)
        print(f"  Peak purity: {purities[peak_idx]:.4f} at r={r_values[peak_idx]:.4f}")
        print(f"  Elapsed: {elapsed:.1f}s")

    if not all_results:
        print("\nNo results computed.")
        return

    # ---- Unified interval ----
    print(f"\n{'=' * 60}")
    print("Determining unified interval...")
    r_min_u, r_max_u = determine_unified_interval(all_results, r_values)
    print(f"  Unified interval: [{r_min_u:.4f}, {r_max_u:.4f}]")

    # ---- Scores ----
    print(f"\n{'=' * 60}")
    print("Computing G-F scores and plateau widths...")
    gf_scores, plateau_widths = {}, {}
    for method, (purities, modularities) in all_results.items():
        score = compute_gf_score(purities, r_values, r_min_u, r_max_u)
        gf_scores[method] = float(score)
        width = compute_plateau_width(purities, r_values)
        plateau_widths[method] = float(width)
        print(f"  {method}: G-F Score={score:.4f}, Plateau W={width:.4f}")

    # ---- Save results ----
    print(f"\n{'=' * 60}")
    print("Saving results...")

    gf_curves_data = {
        "r_values": r_values.tolist(),
        "curves": {
            m: {"purity": p.tolist(), "modularity": q.tolist()}
            for m, (p, q) in all_results.items()
        },
        "unified_interval": [r_min_u, r_max_u],
        "subsample_size": SUBSAMPLE_SIZE,
        "n_points": N_POINTS,
    }
    curves_file = os.path.join(RESULTS_DIR, "human_gf_curves_extended.pkl")
    with open(curves_file, "wb") as f:
        pickle.dump(gf_curves_data, f)
    print(f"  Saved: {curves_file}")

    scores_data = {
        "unified_interval": [r_min_u, r_max_u],
        "scores": gf_scores,
        "ranking": sorted(gf_scores.items(), key=lambda x: x[1], reverse=True),
        "subsample_size": SUBSAMPLE_SIZE,
        "n_methods": len(gf_scores),
        "methods": list(gf_scores.keys()),
    }
    scores_file = os.path.join(RESULTS_DIR, "human_gf_scores_extended.json")
    with open(scores_file, "w") as f:
        json.dump(scores_data, f, indent=2)
    print(f"  Saved: {scores_file}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("HUMAN EXTENDED G-F ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Method':<15} {'G-F Score':>10} {'Plateau W':>12} {'Peak Purity':>12}")
    print("-" * 55)
    for method, score in sorted(gf_scores.items(), key=lambda x: x[1], reverse=True):
        peak = np.max(all_results[method][0])
        print(f"{method:<15} {score:>10.4f} {plateau_widths[method]:>12.4f} {peak:>12.4f}")
    print(f"\nUnified interval: [{r_min_u:.4f}, {r_max_u:.4f}]")
    print(f"Methods computed: {len(gf_scores)}/11")


if __name__ == "__main__":
    main()
