#!/usr/bin/env python3
"""
Compare G-F Score with traditional embedding evaluation metrics:
  1. Link prediction AUC (distance-based)
  2. Node classification micro-F1 (k-NN in embedding space)

Then compute Spearman correlations between G-F Score and each traditional
metric to assess whether G-F Score captures complementary information.

Output: JSON summary + scatter plot figure.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
EMB_DIR = REPO / "embeddings"
RESULTS_DIR = REPO / "results"
FIGURES_DIR = REPO / "figures"

EDGELIST = DATA_DIR / "curated_153_ppi.edgelist"
GO_MAP_FILE = DATA_DIR / "gene_go_map.json"

METHODS_153 = [
    "DM", "MDS", "Spectral", "DeepWalk", "Node2Vec",
    "VGAE", "VGAE-feat", "GraphSAGE", "GAT", "GIN", "PCA",
]

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_edgelist(path):
    """Load edge list as set of frozensets."""
    edges = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.add(frozenset([parts[0], parts[1]]))
    return edges


def load_embedding(method_name):
    """Load embedding coordinates and node list for a method."""
    npy_path = EMB_DIR / f"{method_name}_153.npy"
    nodes_path = EMB_DIR / f"{method_name}_153_nodes.json"

    if not npy_path.exists():
        return None, None

    coords = np.load(npy_path)
    with open(nodes_path) as f:
        nodes = json.load(f)

    return coords, nodes


def load_go_map(path):
    """Load gene -> GO term mapping."""
    with open(path) as f:
        raw = json.load(f)

    # Map gene -> dominant GO term (most frequent)
    go_labels = {}
    for gene, terms in raw.items():
        if terms:
            counter = Counter(terms)
            go_labels[gene] = counter.most_common(1)[0][0]
    return go_labels


# ---------------------------------------------------------------------------
# Metric 1: Link Prediction AUC
# ---------------------------------------------------------------------------

def compute_link_prediction_auc(coords, nodes, edges, n_neg=None, seed=42):
    """
    Compute link prediction AUC using embedding distances.
    
    For each edge (positive) and non-edge (negative), compute the Euclidean
    distance between the two nodes in embedding space. AUC = probability that
    a random positive has shorter distance than a random negative.
    
    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
    nodes : list of str
    edges : set of frozenset
    n_neg : int, number of negative samples (default = number of edges)
    seed : int
    
    Returns
    -------
    auc : float
    """
    rng = np.random.RandomState(seed)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Positive edges (both nodes must be in the embedding)
    pos_pairs = []
    for e in edges:
        e_list = list(e)
        if e_list[0] in node_to_idx and e_list[1] in node_to_idx:
            pos_pairs.append((node_to_idx[e_list[0]], node_to_idx[e_list[1]]))

    if len(pos_pairs) == 0:
        return 0.5

    # Negative edges (non-edges)
    edge_set = set()
    for e in edges:
        e_list = list(e)
        if e_list[0] in node_to_idx and e_list[1] in node_to_idx:
            i, j = node_to_idx[e_list[0]], node_to_idx[e_list[1]]
            edge_set.add((min(i, j), max(i, j)))

    if n_neg is None:
        n_neg = len(pos_pairs)

    neg_pairs = []
    max_attempts = n_neg * 10
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < max_attempts:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i != j:
            pair = (min(i, j), max(i, j))
            if pair not in edge_set:
                neg_pairs.append(pair)
                edge_set.add(pair)  # avoid duplicates
        attempts += 1

    if len(neg_pairs) == 0:
        return 0.5

    # Compute distances
    pos_dists = np.array([np.linalg.norm(coords[i] - coords[j]) for i, j in pos_pairs])
    neg_dists = np.array([np.linalg.norm(coords[i] - coords[j]) for i, j in neg_pairs])

    # AUC via Mann-Whitney U statistic
    n_pos = len(pos_dists)
    n_neg_actual = len(neg_dists)
    
    # Count: for each positive, how many negatives have larger distance
    auc_sum = 0
    for pd in pos_dists:
        auc_sum += np.sum(neg_dists > pd) + 0.5 * np.sum(neg_dists == pd)
    
    auc = auc_sum / (n_pos * n_neg_actual)
    return auc


# ---------------------------------------------------------------------------
# Metric 2: Node Classification (k-NN)
# ---------------------------------------------------------------------------

def compute_knn_classification(coords, nodes, go_labels, k=5, n_folds=5, seed=42):
    """
    Compute k-NN node classification micro-F1 via cross-validation.
    
    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
    nodes : list of str
    go_labels : dict, gene -> dominant GO term
    k : int
    n_folds : int
    seed : int
    
    Returns
    -------
    micro_f1 : float
    """
    rng = np.random.RandomState(seed)

    # Filter to nodes with GO labels
    labeled_indices = [i for i, n in enumerate(nodes) if n in go_labels]
    if len(labeled_indices) < k + n_folds:
        return 0.0

    labels = np.array([go_labels[nodes[i]] for i in labeled_indices])
    coords_labeled = coords[labeled_indices]

    # Filter to GO terms with at least 3 instances
    term_counts = Counter(labels)
    valid_terms = {t for t, c in term_counts.items() if c >= 3}
    valid_mask = np.array([l in valid_terms for l in labels])
    
    if np.sum(valid_mask) < k + n_folds:
        return 0.0

    coords_labeled = coords_labeled[valid_mask]
    labels = labels[valid_mask]
    n = len(labels)

    # Shuffle indices
    indices = np.arange(n)
    rng.shuffle(indices)

    # K-fold CV
    fold_size = n // n_folds
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        if len(train_idx) < k:
            continue

        # Compute distances from test to train
        test_coords = coords_labeled[test_idx]
        train_coords = coords_labeled[train_idx]
        train_labels = labels[train_idx]

        dists = cdist(test_coords, train_coords, metric='euclidean')

        # For each test point, find k nearest neighbors
        for ti in range(len(test_idx)):
            nn_indices = np.argsort(dists[ti])[:k]
            nn_labels = train_labels[nn_indices]
            predicted = Counter(nn_labels).most_common(1)[0][0]
            actual = labels[test_idx[ti]]

            if predicted == actual:
                tp_total += 1
            else:
                fp_total += 1
                fn_total += 1

    if tp_total + fp_total == 0 or tp_total + fn_total == 0:
        return 0.0

    precision = tp_total / (tp_total + fp_total)
    recall = tp_total / (tp_total + fn_total)
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return micro_f1


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Embedding Metric Comparison: G-F Score vs Traditional Metrics")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    edges = load_edgelist(EDGELIST)
    go_labels = load_go_map(GO_MAP_FILE)
    print(f"  Edges: {len(edges)}")
    print(f"  GO-annotated genes: {len(go_labels)}")

    # Load G-F Scores
    with open(RESULTS_DIR / "gf_scores.json") as f:
        gf_data = json.load(f)
    gf_scores = gf_data["scores"]

    # Also load GNN G-F scores
    with open(RESULTS_DIR / "gnn_gf_scores.json") as f:
        gnn_data = json.load(f)
    gf_scores.update(gnn_data["gf_scores"])

    # Compute metrics for all methods
    results = {}
    for method in METHODS_153:
        print(f"\nProcessing {method}...")
        coords, nodes = load_embedding(method)
        if coords is None:
            print(f"  Skipping {method}: embedding not found")
            continue

        print(f"  Shape: {coords.shape}, Nodes: {len(nodes)}")

        # Link prediction AUC
        auc = compute_link_prediction_auc(coords, nodes, edges, seed=SEED)
        print(f"  Link prediction AUC: {auc:.4f}")

        # Node classification F1
        f1 = compute_knn_classification(coords, nodes, go_labels, k=5, n_folds=5, seed=SEED)
        print(f"  k-NN micro-F1: {f1:.4f}")

        # G-F Score
        gf = gf_scores.get(method, None)
        if gf is None:
            # Try alternate names
            for alt in [method.lower(), method.replace("-", "_")]:
                gf = gf_scores.get(alt, None)
                if gf is not None:
                    break
        print(f"  G-F Score: {gf}")

        results[method] = {
            "link_pred_auc": round(auc, 4),
            "knn_micro_f1": round(f1, 4),
            "gf_score": round(gf, 4) if gf is not None else None,
        }

    # Correlation analysis
    print("\n" + "=" * 60)
    print("Correlation Analysis")
    print("=" * 60)

    # Collect paired data
    methods_with_all = [m for m, r in results.items()
                        if r["gf_score"] is not None and r["link_pred_auc"] > 0]

    gf_vals = np.array([results[m]["gf_score"] for m in methods_with_all])
    auc_vals = np.array([results[m]["link_pred_auc"] for m in methods_with_all])
    f1_vals = np.array([results[m]["knn_micro_f1"] for m in methods_with_all])

    if len(methods_with_all) >= 3:
        rho_gf_auc, p_gf_auc = spearmanr(gf_vals, auc_vals)
        rho_gf_f1, p_gf_f1 = spearmanr(gf_vals, f1_vals)
        rho_auc_f1, p_auc_f1 = spearmanr(auc_vals, f1_vals)

        print(f"\nG-F Score vs Link Pred AUC:  rho = {rho_gf_auc:+.3f}, P = {p_gf_auc:.3f} (n = {len(methods_with_all)})")
        print(f"G-F Score vs k-NN F1:      rho = {rho_gf_f1:+.3f}, P = {p_gf_f1:.3f} (n = {len(methods_with_all)})")
        print(f"Link Pred AUC vs k-NN F1:  rho = {rho_auc_f1:+.3f}, P = {p_auc_f1:.3f} (n = {len(methods_with_all)})")
    else:
        rho_gf_auc = p_gf_auc = rho_gf_f1 = p_gf_f1 = rho_auc_f1 = p_auc_f1 = None
        print("Not enough methods for correlation analysis.")

    # Save results
    output = {
        "description": "Comparison of G-F Score with traditional embedding evaluation metrics",
        "n_methods": len(methods_with_all),
        "methods": methods_with_all,
        "per_method": results,
        "correlations": {
            "gf_vs_link_pred_auc": {
                "spearman_rho": round(rho_gf_auc, 3) if rho_gf_auc is not None else None,
                "p_value": round(p_gf_auc, 3) if p_gf_auc is not None else None,
            },
            "gf_vs_knn_f1": {
                "spearman_rho": round(rho_gf_f1, 3) if rho_gf_f1 is not None else None,
                "p_value": round(p_gf_f1, 3) if p_gf_f1 is not None else None,
            },
            "link_pred_auc_vs_knn_f1": {
                "spearman_rho": round(rho_auc_f1, 3) if rho_auc_f1 is not None else None,
                "p_value": round(p_auc_f1, 3) if p_auc_f1 is not None else None,
            },
        },
    }

    output_path = RESULTS_DIR / "metric_comparison.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate scatter plot
    if len(methods_with_all) >= 3:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            method_colors = {
                "DM": "#1f77b4", "MDS": "#ff7f0e", "Spectral": "#2ca02c",
                "DeepWalk": "#d62728", "Node2Vec": "#9467bd", "VGAE": "#8c564b",
                "VGAE-feat": "#e377c2", "GraphSAGE": "#7f7f7f", "GAT": "#bcbd22",
                "GIN": "#17becf", "PCA": "#aec7e8",
            }

            # Panel A: G-F Score vs Link Pred AUC
            ax = axes[0]
            for m in methods_with_all:
                ax.scatter(results[m]["gf_score"], results[m]["link_pred_auc"],
                          c=method_colors.get(m, "gray"), s=80, zorder=5,
                          edgecolors="black", linewidth=0.5)
                ax.annotate(m, (results[m]["gf_score"], results[m]["link_pred_auc"]),
                           fontsize=7, ha="center", va="bottom",
                           xytext=(0, 5), textcoords="offset points")
            ax.set_xlabel("G-F Score")
            ax.set_ylabel("Link Prediction AUC")
            ax.set_title(f"(A) G-F Score vs Link Pred AUC\nρ = {rho_gf_auc:+.3f}, P = {p_gf_auc:.3f}")
            ax.grid(True, alpha=0.3)

            # Panel B: G-F Score vs k-NN F1
            ax = axes[1]
            for m in methods_with_all:
                ax.scatter(results[m]["gf_score"], results[m]["knn_micro_f1"],
                          c=method_colors.get(m, "gray"), s=80, zorder=5,
                          edgecolors="black", linewidth=0.5)
                ax.annotate(m, (results[m]["gf_score"], results[m]["knn_micro_f1"]),
                           fontsize=7, ha="center", va="bottom",
                           xytext=(0, 5), textcoords="offset points")
            ax.set_xlabel("G-F Score")
            ax.set_ylabel("k-NN Micro-F1")
            ax.set_title(f"(B) G-F Score vs k-NN Classification F1\nρ = {rho_gf_f1:+.3f}, P = {p_gf_f1:.3f}")
            ax.grid(True, alpha=0.3)

            # Panel C: Link Pred AUC vs k-NN F1
            ax = axes[2]
            for m in methods_with_all:
                ax.scatter(results[m]["link_pred_auc"], results[m]["knn_micro_f1"],
                          c=method_colors.get(m, "gray"), s=80, zorder=5,
                          edgecolors="black", linewidth=0.5)
                ax.annotate(m, (results[m]["link_pred_auc"], results[m]["knn_micro_f1"]),
                           fontsize=7, ha="center", va="bottom",
                           xytext=(0, 5), textcoords="offset points")
            ax.set_xlabel("Link Prediction AUC")
            ax.set_ylabel("k-NN Micro-F1")
            ax.set_title(f"(C) Link Pred AUC vs k-NN F1\nρ = {rho_auc_f1:+.3f}, P = {p_auc_f1:.3f}")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_path = FIGURES_DIR / "metric_comparison_scatter.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Figure saved to: {fig_path}")
        except ImportError:
            print("matplotlib not available; skipping figure generation.")

    print("\nDone.")
    return output


if __name__ == "__main__":
    main()
