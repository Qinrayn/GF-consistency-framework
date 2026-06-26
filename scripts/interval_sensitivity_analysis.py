#!/usr/bin/env python3
"""
Interval Sensitivity Analysis (P1-2)
=====================================
Addresses: The unified integration interval [0.05, 0.422] was determined
after seeing the data (via adaptive_interval.py), posing a data snooping
risk. This script tests rank stability across multiple pre-specified
intervals to demonstrate that method rankings are not an artifact of
interval choice.

Test intervals:
  1. [0.05, 0.422] — current unified interval (baseline)
  2. [0.05, 0.30]  — short interval
  3. [0.05, 0.50]  — wide interval
  4. [0.05, 0.55]  — full range (r_max from config)
  5. [0.10, 0.40]  — narrow central
  6. [0.15, 0.35]  — tight central
  7. [0.20, 0.422] — offset interval

Output: results/interval_sensitivity_analysis.json
"""
from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    compute_gf_curve,
    compute_gf_score,
    rescale_coordinates,
    align_embedding_to_nodes,
    get_data_dir, get_embeddings_dir, get_results_dir,
    GF_R_MIN, GF_R_MAX, PLATEAU_RELATIVE_THRESHOLD,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = get_data_dir()
EMB_DIR = get_embeddings_dir()
RESULTS_DIR = get_results_dir()

# Load curated network and GO annotations
GO_MAP_FILE = DATA_DIR / "gene_go_map.json"
EDGELIST = DATA_DIR / "curated_153_ppi.edgelist"

# 11 methods with 153-node embeddings
METHODS = [
    "DM", "MDS", "Spectral", "DeepWalk", "Node2Vec",
    "VGAE", "VGAE-feat", "PCA", "GraphSAGE", "GAT", "GIN",
]

# Test intervals
INTERVALS = [
    ("current", 0.05, 0.422),
    ("short", 0.05, 0.30),
    ("wide", 0.05, 0.50),
    ("full_range", 0.05, 0.55),
    ("narrow_central", 0.10, 0.40),
    ("tight_central", 0.15, 0.35),
    ("offset", 0.20, 0.422),
]


def load_go_map():
    with open(GO_MAP_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_network_nodes():
    nodes = set()
    with open(EDGELIST, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                nodes.add(parts[0])
                nodes.add(parts[1])
    return sorted(nodes)


def load_embedding(method):
    npy_path = EMB_DIR / f"{method}_153.npy"
    nodes_path = EMB_DIR / f"{method}_153_nodes.json"
    if not npy_path.exists():
        return None, None
    coords = np.load(str(npy_path))
    with open(nodes_path, encoding="utf-8") as f:
        emb_nodes = json.load(f)
    return coords, emb_nodes


def main():
    print("=" * 64)
    print("  P1-2: Interval Sensitivity Analysis")
    print("=" * 64)

    go_map = load_go_map()
    target_nodes = load_network_nodes()
    print(f"  Network: {len(target_nodes)} nodes")
    print(f"  GO annotations: {len(go_map)} genes")

    # 200-point r grid (same as pipeline)
    r_vals = np.linspace(0.05, 0.55, 200)

    # Compute G-F curves for each method (once)
    method_curves = {}
    method_nodes_aligned = {}

    for method in METHODS:
        coords, emb_nodes = load_embedding(method)
        if coords is None:
            print(f"  WARNING: {method} embedding not found, skipping")
            continue

        aligned, common = align_embedding_to_nodes(coords, emb_nodes, target_nodes)
        if len(common) == 0:
            print(f"  WARNING: {method} has no common nodes, skipping")
            continue

        rescaled = rescale_coordinates(aligned, target_std=0.3)
        purities, _ = compute_gf_curve(rescaled, common, go_map, r_vals)

        method_curves[method] = purities
        method_nodes_aligned[method] = common
        print(f"  {method}: {len(common)} aligned nodes, "
              f"peak purity={max(purities):.4f}")

    available_methods = list(method_curves.keys())
    n_methods = len(available_methods)
    print(f"\n  {n_methods} methods with valid curves")

    # Compute G-F Score for each method x interval combination
    results = {}
    rankings = {}

    for interval_name, r_min, r_max in INTERVALS:
        print(f"\n  Interval [{r_min}, {r_max}] ({interval_name}):")
        scores = {}
        for method in available_methods:
            purities = method_curves[method]
            score = compute_gf_score(r_vals, purities, r_min=r_min, r_max=r_max)
            scores[method] = score

        # Sort by score descending
        ranking = sorted(scores.keys(), key=lambda m: -scores[m])
        results[interval_name] = {
            "r_min": r_min,
            "r_max": r_max,
            "scores": scores,
            "ranking": ranking,
        }
        rankings[interval_name] = ranking

        # Print top-5
        print(f"    Top-5: {', '.join(ranking[:5])}")
        print(f"    Scores: " +
              ", ".join(f"{m}={scores[m]:.4f}" for m in ranking[:5]))

    # Compute rank stability across intervals
    print("\n  --- Rank Stability ---")

    # Pairwise Spearman between current interval and each other
    baseline = "current"
    baseline_rank = rankings[baseline]

    stability = {}
    for interval_name, _, _ in INTERVALS:
        if interval_name == baseline:
            continue
        other_rank = rankings[interval_name]
        # Convert rankings to rank arrays
        baseline_ranks = [baseline_rank.index(m) + 1 for m in available_methods]
        other_ranks = [other_rank.index(m) + 1 for m in available_methods]
        rho, p = spearmanr(baseline_ranks, other_ranks)
        stability[interval_name] = {
            "spearman_rho": float(rho),
            "p_value": float(p),
        }
        print(f"    {interval_name} vs current: rho={rho:.3f} (p={p:.4f})")

    # Kendall's W via rank-sum formula
    # W = 12 * S / (k^2 * (n^3 - n))
    # where S = sum_j (R_j - R_bar)^2, R_j = rank sum for method j,
    # R_bar = k(n+1)/2
    all_ranks = []
    for interval_name, _, _ in INTERVALS:
        rank = rankings[interval_name]
        all_ranks.append([rank.index(m) + 1 for m in available_methods])

    rank_matrix = np.array(all_ranks)  # (k, n) = (n_intervals, n_methods)
    k = len(INTERVALS)
    n = n_methods
    rank_sums = rank_matrix.sum(axis=0)  # R_j for each method
    R_bar = k * (n + 1) / 2
    S = np.sum((rank_sums - R_bar) ** 2)
    W = 12 * S / (k**2 * (n**3 - n))
    print(f"\n  Kendall's W across all {len(INTERVALS)} intervals: {W:.4f}")

    # Top-1 consistency
    top1_counts = {}
    for interval_name, _, _ in INTERVALS:
        top1 = rankings[interval_name][0]
        top1_counts[top1] = top1_counts.get(top1, 0) + 1
    print(f"\n  Top-1 consistency: {top1_counts}")

    # Save results
    output = {
        "description": "P1-2: Interval sensitivity analysis for G-F Score",
        "n_methods": n_methods,
        "n_intervals": len(INTERVALS),
        "intervals": {name: {"r_min": rmin, "r_max": rmax}
                      for name, rmin, rmax in INTERVALS},
        "gf_scores": {name: results[name]["scores"] for name in results},
        "rankings": {name: results[name]["ranking"] for name in results},
        "rank_stability_vs_current": stability,
        "kendalls_w_all_intervals": float(W),
        "top1_consistency": {k: v for k, v in top1_counts.items()},
        "conclusion": f"Kendall's W={W:.3f} across {len(INTERVALS)} intervals. "
                       f"Top-1 method: {max(top1_counts, key=top1_counts.get)} "
                       f"({max(top1_counts.values())}/{len(INTERVALS)} intervals).",
    }

    out_file = RESULTS_DIR / "interval_sensitivity_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_file}")
    print("=" * 64)


if __name__ == "__main__":
    main()
