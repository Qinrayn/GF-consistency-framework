#!/usr/bin/env python3
"""
Extended method correlation + FDR correction (n=25)
=====================================================
Merges 11 original + 14 new methods, computes Spearman correlations
for GF Score vs AUC and GF Score vs F1, then applies BH FDR correction
at n=25 for adequate statistical power.

Output: results/extended_25method_correlation.json
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_data_dir, get_embeddings_dir, get_results_dir, SEED

DATA = get_data_dir()
EMB = get_embeddings_dir()
RESULTS = get_results_dir()


def load_embedding(method):
    npy = EMB / f"{method}_153.npy"
    nodes_json = EMB / f"{method}_153_nodes.json"
    if not npy.exists():
        return None, None
    coords = np.load(str(npy))
    with open(nodes_json, encoding="utf-8") as f:
        nodes = json.load(f)
    return coords, nodes


def load_edges():
    edges = set()
    with open(DATA / "curated_153_ppi.edgelist", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.add(frozenset([parts[0], parts[1]]))
    return edges


def load_go_labels():
    with open(DATA / "gene_go_map.json", encoding="utf-8") as f:
        go_map = json.load(f)
    labels = {}
    for gene, terms in go_map.items():
        if terms:
            labels[gene] = Counter(terms).most_common(1)[0][0]
    return labels


def compute_auc(coords, nodes, edges, seed=SEED):
    """Embedding-distance AUC (same as metric_comparison.py)."""
    rng = np.random.RandomState(seed)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    pos = []
    edge_set = set()
    for e in edges:
        el = list(e)
        if el[0] in node_to_idx and el[1] in node_to_idx:
            i, j = node_to_idx[el[0]], node_to_idx[el[1]]
            pos.append((i, j))
            edge_set.add((min(i, j), max(i, j)))

    if not pos:
        return 0.5

    neg = []
    attempts = 0
    while len(neg) < len(pos) and attempts < len(pos) * 10:
        i, j = rng.randint(0, n), rng.randint(0, n)
        if i != j:
            pair = (min(i, j), max(i, j))
            if pair not in edge_set:
                neg.append(pair)
                edge_set.add(pair)
        attempts += 1

    if not neg:
        return 0.5

    pos_d = np.array([np.linalg.norm(coords[i] - coords[j]) for i, j in pos])
    neg_d = np.array([np.linalg.norm(coords[i] - coords[j]) for i, j in neg])

    auc_sum = sum(np.sum(neg_d > pd) + 0.5 * np.sum(neg_d == pd) for pd in pos_d)
    return float(auc_sum / (len(pos_d) * len(neg_d)))


def compute_knn_f1(coords, nodes, go_labels, k=5, seed=SEED):
    """5-fold CV k-NN micro-F1."""
    labeled = [(i, go_labels[n]) for i, n in enumerate(nodes) if n in go_labels]
    if len(labeled) < 10:
        return 0.0

    indices = [x[0] for x in labeled]
    labels = [x[1] for x in labeled]

    X = coords[indices]
    unique_labels = list(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels])

    # Filter classes with >= 3 samples
    counts = Counter(labels)
    valid = [i for i, l in enumerate(labels) if counts[l] >= 3]
    if len(valid) < 10:
        return 0.0

    X = X[valid]
    y = y[valid]

    clf = KNeighborsClassifier(n_neighbors=min(k, len(X)-1))
    try:
        scores = cross_val_score(clf, X, y, cv=min(5, len(X)//3), scoring="f1_micro")
        return float(scores.mean())
    except Exception:
        return 0.0


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(m)
    adjusted[-1] = ranked[-1]
    for k in range(m-2, -1, -1):
        rank = k + 1
        val = ranked[k] * m / rank
        adjusted[k] = min(val, adjusted[k+1])
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty(m)
    result[order] = adjusted
    return result.tolist()


def main():
    print("=" * 70)
    print("  Extended 25-Method Correlation + FDR Correction")
    print("=" * 70)

    # Load original 11 methods
    with open(RESULTS / "metric_comparison.json", encoding="utf-8") as f:
        orig = json.load(f)

    # Load new 14 methods
    with open(RESULTS / "extended_gf_scores.json", encoding="utf-8") as f:
        ext = json.load(f)

    edges = load_edges()
    go_labels = load_go_labels()

    # Build merged table
    methods = []
    gf_scores = []
    aucs = []
    f1s = []

    # Original 11
    for m in orig["methods"]:
        pm = orig["per_method"][m]
        methods.append(m)
        gf_scores.append(pm["gf_score"])
        aucs.append(pm["link_pred_auc"])
        f1s.append(pm["knn_micro_f1"])

    # New 14
    for m, r in ext["results"].items():
        if "error" in r:
            continue
        coords, nodes = load_embedding(m)
        if coords is None:
            continue
        auc = compute_auc(coords, nodes, edges)
        f1 = compute_knn_f1(coords, nodes, go_labels)
        methods.append(m)
        gf_scores.append(r["gf_score"])
        aucs.append(auc)
        f1s.append(f1)
        print(f"  {m}: GF={r['gf_score']:.4f}, AUC={auc:.4f}, F1={f1:.4f}")

    n = len(methods)
    print(f"\n  Total methods: {n}")

    # Spearman correlations
    rho_auc, p_auc = spearmanr(gf_scores, aucs)
    rho_f1, p_f1 = spearmanr(gf_scores, f1s)

    print(f"\n  GF vs AUC (n={n}): rho={rho_auc:.3f}, p={p_auc:.4f}")
    print(f"  GF vs F1  (n={n}): rho={rho_f1:.3f}, p={p_f1:.4f}")

    # FDR correction (2 tests: GF-AUC, GF-F1)
    pvals = [p_auc, p_f1]
    bh = benjamini_hochberg(pvals)

    print(f"\n  FDR-corrected (BH, 2 tests):")
    print(f"    GF vs AUC: p_BH={bh[0]:.4f} {'***' if bh[0] < 0.05 else 'ns'}")
    print(f"    GF vs F1:  p_BH={bh[1]:.4f} {'***' if bh[1] < 0.05 else 'ns'}")

    # Compare with n=11
    orig_rho_auc = orig["correlations"]["gf_vs_link_pred_auc"]["spearman_rho"]
    orig_p_auc = orig["correlations"]["gf_vs_link_pred_auc"]["p_value"]
    orig_rho_f1 = orig["correlations"]["gf_vs_knn_f1"]["spearman_rho"]
    orig_p_f1 = orig["correlations"]["gf_vs_knn_f1"]["p_value"]

    print(f"\n  Comparison (n=11 → n={n}):")
    print(f"    GF vs AUC: rho {orig_rho_auc:.3f} → {rho_auc:.3f}, p {orig_p_auc:.4f} → {p_auc:.4f}")
    print(f"    GF vs F1:  rho {orig_rho_f1:.3f} → {rho_f1:.3f}, p {orig_p_f1:.4f} → {p_f1:.4f}")

    # Save
    output = {
        "description": f"Extended {n}-method correlation analysis",
        "n_methods": n,
        "methods": methods,
        "per_method": {
            m: {"gf_score": float(g), "auc": float(a), "f1": float(f)}
            for m, g, a, f in zip(methods, gf_scores, aucs, f1s)
        },
        "correlations": {
            "gf_vs_auc": {"rho": float(rho_auc), "p": float(p_auc), "p_bh": float(bh[0])},
            "gf_vs_f1": {"rho": float(rho_f1), "p": float(p_f1), "p_bh": float(bh[1])},
        },
        "comparison_n11_vs_n25": {
            "auc_rho_11": float(orig_rho_auc), "auc_rho_25": float(rho_auc),
            "auc_p_11": float(orig_p_auc), "auc_p_25": float(p_auc),
            "f1_rho_11": float(orig_rho_f1), "f1_rho_25": float(rho_f1),
            "f1_p_11": float(orig_p_f1), "f1_p_25": float(p_f1),
        },
    }

    out_file = RESULTS / "extended_25method_correlation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
