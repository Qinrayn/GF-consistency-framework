#!/usr/bin/env python3
"""
Expand the GF Score vs downstream task correlation from n=11 to n=15
by computing link prediction AUC and k-NN F1 for UMAP-adj, UMAP, TSNE, TSNE-sp.
"""
from __future__ import annotations

import json
import sys
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, bootstrap
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import SEED, get_results_dir, get_data_dir

RESULTS = get_results_dir()
DATA = get_data_dir()

EXTRA_METHODS = ["UMAP-adj", "UMAP", "TSNE", "TSNE-sp"]


def load_network():
    """Load the curated 153-node yeast PPI network."""
    edges = []
    with open(DATA / "curated_153_ppi.edgelist") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
    G = nx.Graph()
    G.add_edges_from(edges)
    nodes = sorted(G.nodes())
    return G, nodes


def load_embedding(method):
    """Load 2D embedding for the 153-node network."""
    fpath = Path("embeddings") / f"{method}_153_nodes.json"
    if not fpath.exists():
        fpath_npy = Path("embeddings") / f"{method}_153.npy"
        nodes_fpath = Path("embeddings") / f"{method}_153_nodes.json"
        if fpath_npy.exists() and nodes_fpath.exists():
            coords = np.load(fpath_npy)
            with open(nodes_fpath) as f:
                data = json.load(f)
            nodes = data if isinstance(data, list) else sorted(data.keys())
            return coords, nodes
        return None, None
    with open(fpath) as f:
        data = json.load(f)
    if isinstance(data, list):
        nodes = data
    else:
        nodes = sorted(data.keys())
    coords = np.array([[data[n]["x"], data[n]["y"]] for n in nodes]) if isinstance(data, dict) else None
    if coords is None:
        # If data is a list of node names, load from .npy
        fpath_npy = Path("embeddings") / f"{method}_153.npy"
        if fpath_npy.exists():
            coords = np.load(fpath_npy)
        else:
            return None, None
    return coords, nodes


def compute_link_prediction_auc(G, coords, nodes):
    """Compute link prediction AUC using cosine similarity on held-out edges."""
    rng = np.random.default_rng(SEED)
    edges = list(G.edges())
    non_edges = list(nx.non_edges(G))

    # Hold out 20% of edges and 20% of non-edges
    n_holdout = max(1, len(edges) // 5)
    n_neg_holdout = max(1, len(non_edges) // 5)

    rng.shuffle(edges)
    rng.shuffle(non_edges)
    test_pos = edges[:n_holdout]
    test_neg = non_edges[:n_neg_holdout]

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = coords / norms

    scores = []
    labels = []
    for u, v in test_pos:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            sim = float(normed[i] @ normed[j])
            scores.append(sim)
            labels.append(1)
    for u, v in test_neg:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            sim = float(normed[i] @ normed[j])
            scores.append(sim)
            labels.append(0)

    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_knn_f1(G, coords, nodes, k=5):
    """Compute k-NN function prediction micro-F1 using GO annotations."""
    go_file = DATA / "gene_go_map.json"
    if not go_file.exists():
        return 0.0
    with open(go_file) as f:
        go_map = json.load(f)

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    dist_mat = cdist(coords, coords)
    np.fill_diagonal(dist_mat, np.inf)

    correct = 0
    total = 0
    for i, node in enumerate(nodes):
        if node not in go_map or not go_map[node]:
            continue
        true_terms = set(go_map[node]) if isinstance(go_map[node], list) else {go_map[node]}
        nn_idx = np.argsort(dist_mat[i])[:k]
        pred_terms = set()
        for j in nn_idx:
            nn_node = nodes[j]
            if nn_node in go_map:
                terms = go_map[nn_node]
                if isinstance(terms, list):
                    pred_terms.update(terms)
                else:
                    pred_terms.add(terms)
        overlap = len(true_terms & pred_terms)
        correct += overlap
        total += len(true_terms)

    return correct / total if total > 0 else 0.0


def main():
    G, net_nodes = load_network()
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load existing 11-method metrics
    mc = json.load(open(RESULTS / "metric_comparison.json"))
    per_method = mc["per_method"]

    all_data = {}
    for m, d in per_method.items():
        all_data[m] = {
            "gf_score": d["gf_score"],
            "link_pred_auc": d["link_pred_auc"],
            "knn_f1": d["knn_micro_f1"],
        }

    # Compute for extra methods
    for method in EXTRA_METHODS:
        coords, emb_nodes = load_embedding(method)
        if coords is None:
            print(f"  {method}: embeddings not found, skipping")
            continue

        # Load GF score
        gf = None
        for fname in ["umap_tsne_gf_scores.json", "position_encoding_comparison.json"]:
            fpath = RESULTS / fname
            if fpath.exists():
                d = json.load(open(fpath))
                c153 = d.get("curated_153", {})
                if method in c153:
                    gf = c153[method].get("gf_score")
                    break
        if gf is None:
            print(f"  {method}: GF score not found, skipping")
            continue

        lp_auc = compute_link_prediction_auc(G, coords, emb_nodes)
        knn_f1 = compute_knn_f1(G, coords, emb_nodes)
        all_data[method] = {
            "gf_score": gf,
            "link_pred_auc": lp_auc,
            "knn_f1": knn_f1,
        }
        print(f"  {method}: GF={gf:.4f}, LP_AUC={lp_auc:.4f}, kNN_F1={knn_f1:.4f}")

    # Run correlations
    methods = sorted(all_data.keys())
    gf_arr = np.array([all_data[m]["gf_score"] for m in methods])
    lp_arr = np.array([all_data[m]["link_pred_auc"] for m in methods])
    knn_arr = np.array([all_data[m]["knn_f1"] for m in methods])

    n = len(methods)
    print(f"\n=== Correlation Analysis (n={n} methods) ===")

    for label, y_arr in [("Link Pred AUC", lp_arr), ("k-NN F1", knn_arr)]:
        rho, p = spearmanr(gf_arr, y_arr)
        print(f"\n  GF Score vs {label}:")
        print(f"    Spearman rho = {rho:+.4f} (p = {p:.4f})")

        # Bootstrap CI
        rng = np.random.default_rng(SEED)
        boot_rhos = []
        for _ in range(10000):
            idx = rng.choice(n, n, replace=True)
            if len(set(idx)) < 3:
                continue
            r, _ = spearmanr(gf_arr[idx], y_arr[idx])
            boot_rhos.append(r)
        ci_lo = np.percentile(boot_rhos, 2.5)
        ci_hi = np.percentile(boot_rhos, 97.5)
        print(f"    95% CI = [{ci_lo:+.3f}, {ci_hi:+.3f}]")

        # Compare with n=11
        core_11 = [m for m in methods if m not in EXTRA_METHODS]
        gf_11 = np.array([all_data[m]["gf_score"] for m in core_11])
        y_11 = np.array([all_data[m]["link_pred_auc"] if label == "Link Pred AUC"
                         else all_data[m]["knn_f1"] for m in core_11])
        rho_11, p_11 = spearmanr(gf_11, y_11)
        print(f"    (n=11 comparison: rho={rho_11:+.4f}, p={p_11:.4f})")

    # Save results
    output = {
        "n_methods": n,
        "methods": methods,
        "per_method": all_data,
        "correlations": {
            "gf_vs_link_pred_auc": {
                "spearman_rho": float(spearmanr(gf_arr, lp_arr)[0]),
                "p_value": float(spearmanr(gf_arr, lp_arr)[1]),
                "n": n,
            },
            "gf_vs_knn_f1": {
                "spearman_rho": float(spearmanr(gf_arr, knn_arr)[0]),
                "p_value": float(spearmanr(gf_arr, knn_arr)[1]),
                "n": n,
            },
        },
    }
    out_path = RESULTS / "metric_comparison_extended_15.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
