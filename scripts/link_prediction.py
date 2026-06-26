#!/usr/bin/env python3
"""
link_prediction.py
Step 8: Link prediction using 5-fold CV with logistic regression.
Input: node pair Hadamard product of embeddings.
Expected ranking: Spectral > MDS > DM > DeepWalk ~ Node2Vec > VGAE
Observed Spearman rho(AUROC, G-F Score) ~ +0.943 (positive: higher G-F score ~ higher AUROC)
"""
from __future__ import annotations

import sys
import json
import math
import random
import numpy as np
import networkx as nx
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (SEED, CV_FOLDS, CLASSICAL_METHODS,
                    get_data_dir, get_results_dir, get_embeddings_dir,
                    load_curated_network, load_embedding)

# Minimum number of methods required for a meaningful Spearman correlation
MIN_METHODS_FOR_SPEARMAN: int = 5

METHODS = CLASSICAL_METHODS


def hadamard_product(emb_u, emb_v):
    """Compute Hadamard product of two embedding vectors."""
    return emb_u * emb_v


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()
    
    # Load network
    print("Loading network...")
    G, nodes, go_map = load_curated_network(data_dir)
    node_set = set(G.nodes())
    print(f"Network: {len(nodes)} nodes, {G.number_of_edges()} edges")
    
    # Load G-F scores for comparison
    gf_scores = {}
    scores_file = results_dir / "gf_scores.json"
    if scores_file.exists():
        with open(scores_file, encoding="utf-8") as f:
            scores_data = json.load(f)
        gf_scores = scores_data.get("scores", {})
    else:
        print("Warning: gf_scores.json not found. "
              "Spearman correlation will not be computed. "
              "Run compute_gf.py first.")
    
    # Prepare positive and negative edges
    all_edges = list(G.edges())
    all_non_edges = []
    nodes_list = list(G.nodes())
    n = len(nodes_list)
    
    # Generate negative samples
    np.random.seed(SEED)
    edge_set = set(frozenset([u, v]) for u, v in all_edges)
    max_neg = len(all_edges)
    neg_count = 0
    attempts = 0
    while neg_count < max_neg and attempts < max_neg * 20:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j and frozenset([nodes_list[i], nodes_list[j]]) not in edge_set:
            all_non_edges.append((nodes_list[i], nodes_list[j]))
            edge_set.add(frozenset([nodes_list[i], nodes_list[j]]))
            neg_count += 1
        attempts += 1
    
    print(f"Positive edges: {len(all_edges)}, Negative edges: {len(all_non_edges)}")
    
    # Evaluate each method
    auroc_results = {}
    
    for method in METHODS:
        print(f"\nEvaluating {method}...")
        try:
            coords, emb_nodes = load_embedding(method, "153", embeddings_dir=emb_dir)
            emb_dict = {emb_nodes[i]: coords[i] for i in range(len(emb_nodes))}
            
            # Filter edges to those with embeddings
            pos_edges = [(u, v) for u, v in all_edges if u in emb_dict and v in emb_dict]
            neg_edges = [(u, v) for u, v in all_non_edges if u in emb_dict and v in emb_dict]
            min_len = min(len(pos_edges), len(neg_edges))
            pos_edges = pos_edges[:min_len]
            neg_edges = neg_edges[:min_len]
            
            # Create features and labels
            features = []
            labels = []
            for u, v in pos_edges:
                features.append(hadamard_product(emb_dict[u], emb_dict[v]))
                labels.append(1)
            for u, v in neg_edges:
                features.append(hadamard_product(emb_dict[u], emb_dict[v]))
                labels.append(0)
            
            X = np.array(features)
            y = np.array(labels)
            
            # 5-fold CV
            skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
            aurocs = []
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                clf = LogisticRegression(random_state=SEED, max_iter=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_test)[:, 1]
                aurocs.append(roc_auc_score(y_test, y_pred))
            
            auroc_mean = float(np.mean(aurocs))
            auroc_std = float(np.std(aurocs))
            auroc_results[method] = {"auroc_mean": auroc_mean, "auroc_std": auroc_std}
            print(f"  AUROC: {auroc_mean:.4f} +/- {auroc_std:.4f}")
        
        except Exception as e:
            print(f"  {method} FAILED: {e}")
    
    # Compute Spearman correlation between AUROC and G-F Score
    auroc_list = []
    gf_list = []
    for method in auroc_results:
        if method in gf_scores:
            auroc_list.append(auroc_results[method]["auroc_mean"])
            gf_list.append(gf_scores[method])
    
    if len(auroc_list) >= MIN_METHODS_FOR_SPEARMAN:
        rho, p_val = stats.spearmanr(auroc_list, gf_list)
        print(f"\nSpearman correlation (AUROC vs G-F Score): rho={rho:.4f}, p={p_val:.4f}")
    else:
        rho, p_val = math.nan, math.nan
        print(f"\nWARNING: Only {len(auroc_list)} methods with both AUROC and G-F score "
              f"(< {MIN_METHODS_FOR_SPEARMAN} required). "
              f"Spearman correlation reported as NaN.")

    # Leave-one-out sensitivity analysis
    loo_results = {}
    if len(auroc_list) >= MIN_METHODS_FOR_SPEARMAN:
        method_names = [m for m in auroc_results if m in gf_scores]
        loo_rhos = []
        for excluded in method_names:
            sub_auroc = [auroc_results[m]["auroc_mean"]
                         for m in method_names if m != excluded]
            sub_gf = [gf_scores[m] for m in method_names if m != excluded]
            if len(sub_auroc) >= 4:
                loo_rho, _ = stats.spearmanr(sub_auroc, sub_gf)
                loo_results[excluded] = round(float(loo_rho), 4)
                loo_rhos.append(float(loo_rho))
        if loo_rhos:
            print(f"  Leave-one-out: rho range [{min(loo_rhos):.4f}, "
                  f"{max(loo_rhos):.4f}], mean={np.mean(loo_rhos):.4f}")

    # Save results
    result = {
        "auroc_results": auroc_results,
        "spearman_rho_auroc_gf": rho,
        "spearman_p_value": p_val,
        "gf_scores_used": gf_scores,
        "spearman_leave_one_out": loo_results,
    }
    
    output_file = results_dir / "link_prediction.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_file}")
    
    # Print ranking
    print("\n=== AUROC Ranking ===")
    ranked = sorted(auroc_results.items(), key=lambda x: x[1]["auroc_mean"], reverse=True)
    for i, (method, data) in enumerate(ranked, 1):
        gf = gf_scores.get(method, "N/A")
        gf_str = f"{gf:.4f}" if isinstance(gf, float) else gf
        print(f"  {i}. {method}: AUROC={data['auroc_mean']:.4f}, G-F={gf_str}")


if __name__ == "__main__":
    main()
