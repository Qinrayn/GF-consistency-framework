#!/usr/bin/env python3
"""
downstream_knn.py
Step 9: k-NN GO term prediction downstream task.
Filter: GO terms appearing >= 3 times -> 119 nodes, 12 classes.
5-fold CV, 5-NN, micro-F1.
Expected: DM ≈ 0.505, MDS ≈ 0.587
"""
from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, CLASSICAL_METHODS, MIN_LABEL_COUNT, K_NEIGHBORS, CV_FOLDS,
    get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_embedding,
)


def main():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    
    np.random.seed(SEED)
    
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()
    
    G, nodes, go_map = load_curated_network(data_dir)
    
    # Get most frequent GO term per protein
    labels = {}
    for node, terms in go_map.items():
        if terms:
            term_counts = Counter(terms)
            labels[node] = term_counts.most_common(1)[0][0]
    
    # Filter: labels appearing >= min_label_count times
    label_counts = Counter(labels.values())
    valid_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= MIN_LABEL_COUNT}
    valid_nodes = sorted([n for n in nodes if n in labels and labels[n] in valid_labels])
    
    print(f"Valid nodes: {len(valid_nodes)}")
    print(f"Valid categories: {len(valid_labels)}")
    
    y_raw = [labels[n] for n in valid_nodes]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    results = {}
    methods = CLASSICAL_METHODS
    
    for method in methods:
        print(f"\nEvaluating {method}...")
        try:
            coords, emb_nodes = load_embedding(method, "153", embeddings_dir=emb_dir)
            emb_dict = {emb_nodes[i]: coords[i] for i in range(len(emb_nodes))}
            
            # Filter to valid nodes with embeddings
            mask = [n in emb_dict for n in valid_nodes]
            X = np.array([emb_dict[n] for n, m in zip(valid_nodes, mask) if m])
            y_filtered = np.array([yi for yi, m in zip(y, mask) if m])
            
            knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
            scores = cross_val_score(knn, X, y_filtered, cv=CV_FOLDS,
                                     scoring='f1_micro')
            
            f1_mean = float(np.mean(scores))
            f1_std = float(np.std(scores))
            
            results[method] = {
                "micro_f1_mean": round(f1_mean, 4),
                "micro_f1_std": round(f1_std, 4),
            }
            print(f"  {method}: micro-F1 = {f1_mean:.3f} +/- {f1_std:.3f}")
        except Exception as e:
            print(f"  {method} FAILED: {e}")
    
    # Save results
    output = {
        "description": "Downstream k-NN GO term prediction (5-fold CV)",
        "n_nodes": len(valid_nodes),
        "n_categories": len(valid_labels),
        "k": K_NEIGHBORS,
        "cv_folds": CV_FOLDS,
        "min_label_count": MIN_LABEL_COUNT,
        "results": results,
    }
    
    output_file = results_dir / "downstream_knn.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
