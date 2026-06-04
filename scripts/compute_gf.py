#!/usr/bin/env python3
"""
compute_gf.py
Step 3: Compute G-F curves (200-point grid) and G-F scores for all methods.
Also compute PCA comparison and randomization control.

Optional --adaptive-interval flag integrates the adaptive interval algorithm
from adaptive_interval.py to determine a data-driven consensus integration
interval instead of (or in addition to) the fixed [0.05, 0.422] interval.
"""

import sys
import json
import csv
import pickle
import random
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_embedding, compute_gf_curve,
    compute_gf_score, compute_plateau_width, rescale_coordinates,
    precompute_distance_matrix,
)

R_MIN = 0.05
R_MAX = 0.55
N_POINTS = 200
GF_R_MIN = 0.05
GF_R_MAX = 0.422

METHODS = ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE"]


def compute_random_baseline(coords, nodes, go_map, r_vals, n_shuffles=10):
    """Shuffle node-coordinate mapping, recompute G-F curve, return mean purity."""
    n = len(nodes)
    all_purities = []
    for s in range(n_shuffles):
        np.random.seed(SEED + s + 1000)
        perm = np.random.permutation(n)
        shuffled_coords = coords[perm]
        purities, _ = compute_gf_curve(shuffled_coords, nodes, go_map, r_vals)
        all_purities.append(purities)
    return np.mean(all_purities, axis=0).tolist()


def compute_random_baseline_with_stats(coords, nodes, go_map, r_vals, n_shuffles=10):
    """Return (mean_purity, std_of_mean) from shuffled baselines.

    Runs *n_shuffles* independent random shuffles of the node-coordinate
    mapping, computes a G-F purity curve for each, then returns the
    element-wise mean curve **and** the standard deviation of the
    per-shuffle mean purities (a single scalar summarising how much the
    overall purity level varies from one random labelling to another).

    Parameters
    ----------
    coords : np.ndarray
        Embedding coordinates (n_nodes, dim).
    nodes : list
        Ordered node identifiers matching *coords*.
    go_map : dict
        Node-to-GO-term annotation mapping.
    r_vals : np.ndarray
        Array of radius values at which to sample purity.
    n_shuffles : int, optional
        Number of independent shuffles (default 10).

    Returns
    -------
    tuple of (list, float)
        ``(mean_curve, std_of_mean)`` where *mean_curve* is a plain
        Python list of floats (same length as *r_vals*) and
        *std_of_mean* is the sample standard deviation of the
        per-shuffle mean purities.
    """
    n = len(nodes)
    all_purities = []
    for s in range(n_shuffles):
        np.random.seed(SEED + s + 1000)
        perm = np.random.permutation(n)
        shuffled_coords = coords[perm]
        purities, _ = compute_gf_curve(shuffled_coords, nodes, go_map, r_vals)
        all_purities.append(purities)
    arr = np.array(all_purities)  # shape: (n_shuffles, n_points)
    mean_curve = arr.mean(axis=0)
    # Per-shuffle mean purity (average across all r values per shuffle)
    shuffle_means = arr.mean(axis=1)
    std_of_mean = float(np.std(shuffle_means, ddof=1))
    return mean_curve.tolist(), std_of_mean


def main():
    # ------------------------------------------------------------------
    # CLI argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Compute G-F curves and scores for embedding methods. "
            "Optionally determine a data-driven adaptive integration "
            "interval via --adaptive-interval."
        ),
    )
    parser.add_argument(
        "--adaptive-interval", action="store_true", default=False,
        help=(
            "Enable adaptive interval determination.  Runs the sliding-"
            "window algorithm from adaptive_interval.py to find a "
            "consensus integration interval instead of using the fixed "
            "[0.05, 0.422] interval only."
        ),
    )
    parser.add_argument(
        "--n-points", type=int, default=N_POINTS,
        help=f"Number of r-axis sample points (default: {N_POINTS}).",
    )
    parser.add_argument(
        "--r-min", type=float, default=R_MIN,
        help=f"Minimum r value for the sampling grid (default: {R_MIN}).",
    )
    parser.add_argument(
        "--r-max", type=float, default=R_MAX,
        help=f"Maximum r value for the sampling grid (default: {R_MAX}).",
    )
    parser.add_argument(
        "--cv-threshold", type=float, default=0.1,
        help="CV threshold for the adaptive stability constraint (default: 0.1).",
    )
    parser.add_argument(
        "--min-width", type=float, default=0.15,
        help="Minimum adaptive interval width (default: 0.15).",
    )
    parser.add_argument(
        "--significance-sigma", type=int, default=2,
        help=(
            "Number of standard deviations above the random baseline "
            "required for significance in the adaptive algorithm "
            "(default: 2)."
        ),
    )
    args = parser.parse_args()

    # Allow CLI to override grid parameters
    n_points = args.n_points
    r_min_grid = args.r_min
    r_max_grid = args.r_max

    random.seed(SEED)
    np.random.seed(SEED)
    
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()
    
    # Load network
    print("Loading network...")
    G, nodes, go_map = load_curated_network(data_dir)
    print(f"Network: {len(nodes)} nodes")
    
    # Generate r values
    r_vals = np.linspace(r_min_grid, r_max_grid, n_points)
    
    # Compute G-F curves for all methods
    all_purities = {}
    all_modularities = {}
    
    for method in METHODS:
        print(f"\nComputing G-F curve for {method}...")
        try:
            coords, emb_nodes = load_embedding(method, "153", embeddings_dir=emb_dir)
            # Align nodes: embedding might use different order
            # Ensure nodes match go_map
            common_nodes = sorted(set(emb_nodes) & set(nodes) & set(go_map.keys()))
            node_indices = [emb_nodes.index(n) for n in common_nodes]
            aligned_coords = coords[node_indices]
            
            purities, modularities = compute_gf_curve(aligned_coords, common_nodes, go_map, r_vals)
            all_purities[method] = purities
            all_modularities[method] = modularities
            print(f"  Purity range: [{min(purities):.3f}, {max(purities):.3f}]")
            print(f"  Modularity range: [{min(modularities):.3f}, {max(modularities):.3f}]")
        except Exception as e:
            print(f"  {method} FAILED: {e}")
    
    # Compute PCA comparison
    print("\nComputing PCA comparison...")
    try:
        pca_coords, pca_nodes = load_embedding("PCA", "153", embeddings_dir=emb_dir)
        common_nodes_pca = sorted(set(pca_nodes) & set(nodes) & set(go_map.keys()))
        node_indices_pca = [pca_nodes.index(n) for n in common_nodes_pca]
        aligned_pca = pca_coords[node_indices_pca]
        pca_purities, pca_modularities = compute_gf_curve(aligned_pca, common_nodes_pca, go_map, r_vals)
        all_purities["PCA"] = pca_purities
        all_modularities["PCA"] = pca_modularities
    except Exception as e:
        print(f"  PCA comparison FAILED: {e}")
    
    # Compute VGAE-feat comparison
    print("\nComputing VGAE-feat G-F curve...")
    try:
        vgae_feat_coords, vgae_feat_nodes = load_embedding("VGAE-feat", "153", embeddings_dir=emb_dir)
        common_nodes_vf = sorted(set(vgae_feat_nodes) & set(nodes) & set(go_map.keys()))
        node_indices_vf = [vgae_feat_nodes.index(n) for n in common_nodes_vf]
        aligned_vf = vgae_feat_coords[node_indices_vf]
        vf_purities, vf_modularities = compute_gf_curve(aligned_vf, common_nodes_vf, go_map, r_vals)
        all_purities["VGAE-feat"] = vf_purities
        all_modularities["VGAE-feat"] = vf_modularities
    except Exception as e:
        print(f"  VGAE-feat FAILED: {e}")
    
    # Compute random baseline
    print("\nComputing random baseline...")
    random_baseline_std = None
    try:
        dm_coords, dm_nodes = load_embedding("DM", "153", embeddings_dir=emb_dir)
        common_nodes_dm = sorted(set(dm_nodes) & set(nodes) & set(go_map.keys()))
        dm_indices = [dm_nodes.index(n) for n in common_nodes_dm]
        aligned_dm = dm_coords[dm_indices]

        if args.adaptive_interval:
            # Need per-shuffle statistics for the adaptive algorithm
            random_baseline, random_baseline_std = compute_random_baseline_with_stats(
                aligned_dm, common_nodes_dm, go_map, r_vals,
            )
            print(f"  Random baseline mean purity: {np.mean(random_baseline):.4f}")
            print(f"  Random baseline std (across shuffles): {random_baseline_std:.4f}")
        else:
            random_baseline = compute_random_baseline(
                aligned_dm, common_nodes_dm, go_map, r_vals,
            )
            print(f"  Random baseline mean purity: {np.mean(random_baseline):.4f}")
    except Exception as e:
        random_baseline = [0.0] * n_points
        print(f"  Random baseline FAILED: {e}")
    
    # Compute G-F Scores (fixed interval)
    print("\nComputing G-F Scores (fixed interval)...")
    gf_scores = {}
    for method in all_purities:
        score = compute_gf_score(r_vals, all_purities[method], GF_R_MIN, GF_R_MAX)
        gf_scores[method] = score
        print(f"  {method}: {score:.4f}")

    # ------------------------------------------------------------------
    # Adaptive interval determination (optional)
    # ------------------------------------------------------------------
    gf_scores_adaptive = None
    consensus_interval = None

    if args.adaptive_interval:
        from adaptive_interval import find_adaptive_interval, _compute_random_baseline_std

        # If the std was not computed above (e.g. random baseline failed),
        # fall back to estimating it from the stored curve.
        if random_baseline_std is None:
            # Build a minimal curves dict so _compute_random_baseline_std
            # can extract what it needs.
            _curves_stub = {"random_baseline_purity": random_baseline}
            _rb_mean, random_baseline_std = _compute_random_baseline_std(_curves_stub)
            print(f"  Estimated random baseline std from curve: {random_baseline_std:.4f}")

        rb_mean = float(np.mean(random_baseline))

        print("\n=== Adaptive Interval Determination ===")
        adaptive_intervals = {}
        for method, purity_list in all_purities.items():
            r_min_ai, r_max_ai, diag = find_adaptive_interval(
                r_vals, np.array(purity_list),
                rb_mean, random_baseline_std,
                cv_threshold=args.cv_threshold,
                min_width=args.min_width,
                significance_sigma=args.significance_sigma,
            )
            adaptive_intervals[method] = {
                "r_min": round(r_min_ai, 4),
                "r_max": round(r_max_ai, 4),
                "cv": round(diag["cv"], 4),
                "width": round(diag["width"], 4),
                "mean_purity": round(diag["mean_purity"], 4),
            }
            relaxed_tag = " [relaxed CV]" if diag.get("relaxed") else ""
            fallback_tag = " [FALLBACK]" if diag.get("fallback") else ""
            print(
                f"  {method:12s}  [{r_min_ai:.4f}, {r_max_ai:.4f}]  "
                f"width={diag['width']:.4f}  cv={diag['cv']:.4f}  "
                f"mean_p={diag['mean_purity']:.4f}"
                f"{relaxed_tag}{fallback_tag}"
            )

        # Consensus interval: median of per-method r_min and r_max
        r_min_vals = [v["r_min"] for v in adaptive_intervals.values()]
        r_max_vals = [v["r_max"] for v in adaptive_intervals.values()]
        consensus_r_min = float(np.median(r_min_vals)) if r_min_vals else GF_R_MIN
        consensus_r_max = float(np.median(r_max_vals)) if r_max_vals else GF_R_MAX
        consensus_interval = [round(consensus_r_min, 4), round(consensus_r_max, 4)]

        print(f"\n  Consensus interval: [{consensus_r_min:.4f}, {consensus_r_max:.4f}]")
        print(f"  Fixed (paper):      [{GF_R_MIN}, {GF_R_MAX}]")

        # Recompute G-F scores using the adaptive consensus interval
        print("\n=== G-F Scores (Adaptive Consensus Interval) ===")
        gf_scores_adaptive = {}
        for method in all_purities:
            score = compute_gf_score(
                r_vals, all_purities[method],
                consensus_r_min, consensus_r_max,
            )
            gf_scores_adaptive[method] = round(score, 4)
            print(f"  {method}: {score:.4f}")

        # Print comparison table: fixed vs adaptive
        print("\n=== Score Comparison: Fixed vs Adaptive ===")
        header = f"  {'Method':<12s} {'Fixed':>8s} {'Adaptive':>10s} {'|Diff|%':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        pct_diffs = []
        for method in all_purities:
            fixed_s = gf_scores.get(method, 0.0)
            adapt_s = gf_scores_adaptive.get(method, 0.0)
            abs_pct = (
                abs(adapt_s - fixed_s) / abs(fixed_s) * 100.0
                if abs(fixed_s) > 1e-12 else float("nan")
            )
            pct_diffs.append(abs_pct)
            print(
                f"  {method:<12s} {fixed_s:>8.4f} {adapt_s:>10.4f} "
                f"{abs_pct:>7.2f}%"
            )
        if pct_diffs:
            mean_abs_pct = float(np.mean(pct_diffs))
            max_abs_pct = float(np.max(pct_diffs))
            print(f"\n  Mean |diff|: {mean_abs_pct:.2f}%")
            print(f"  Max  |diff|: {max_abs_pct:.2f}%")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    # Save G-F curves
    curves_data = {"r": r_vals.tolist(), "n_points": n_points}
    for method in all_purities:
        curves_data[f"{method}_purity"] = all_purities[method]
        curves_data[f"{method}_modularity"] = all_modularities[method]
    curves_data["random_baseline_purity"] = random_baseline
    
    # Save as JSON
    curves_file = results_dir / "gf_curves_200pts.json"
    with open(curves_file, "w") as f:
        json.dump(curves_data, f, indent=2)
    print(f"\nSaved G-F curves (JSON) to: {curves_file}")
    
    # Save as pickle (for plot_figures.py Figure 1)
    pkl_file = results_dir / "gf_curves_200pts.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(curves_data, f)
    print(f"Saved G-F curves (PKL) to: {pkl_file}")
    
    # Save scores with consistent key names
    scores_data = {
        "unified_interval": [GF_R_MIN, GF_R_MAX],
        "unified_interval_paper": [GF_R_MIN, GF_R_MAX],
        "scores": gf_scores,
        "scores_paper_interval": gf_scores,
        "random_baseline": float(np.mean(random_baseline)),
        "random_baseline_mean": float(np.mean(random_baseline)),
    }

    # Extend scores_data when adaptive interval was computed
    if args.adaptive_interval and gf_scores_adaptive is not None:
        scores_data["adaptive_interval"] = consensus_interval
        scores_data["scores_adaptive"] = gf_scores_adaptive

    scores_file = results_dir / "gf_scores.json"
    with open(scores_file, "w") as f:
        json.dump(scores_data, f, indent=2)
    print(f"Saved G-F scores to: {scores_file}")

    # Save plateau width CSV (Supplementary Table S2)
    plateau_file = results_dir / "plateau_width_v3_200pts.csv"
    with open(plateau_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "W", "r_min", "r_max", "GF_Score"])
        for method in all_purities:
            w = compute_plateau_width(r_vals, all_purities[method], threshold=0.5)
            score = gf_scores.get(method, 0.0)
            # Find r_min and r_max of plateau
            p = np.array(all_purities[method])
            mask = p >= 0.5
            if mask.any():
                r_plateau = r_vals[mask]
                r_min_p, r_max_p = float(r_plateau[0]), float(r_plateau[-1])
            else:
                r_min_p, r_max_p = 0.0, 0.0
            writer.writerow([method, f"{w:.4f}", f"{r_min_p:.4f}",
                             f"{r_max_p:.4f}", f"{score:.4f}"])
    print(f"Saved plateau widths to: {plateau_file}")
    
    # Print ranking
    print("\n=== G-F Score Ranking ===")
    ranked = sorted(gf_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (method, score) in enumerate(ranked, 1):
        print(f"  {i}. {method}: {score:.4f}")

    if args.adaptive_interval and gf_scores_adaptive is not None:
        print("\n=== G-F Score Ranking (Adaptive Consensus Interval) ===")
        ranked_adaptive = sorted(
            gf_scores_adaptive.items(), key=lambda x: x[1], reverse=True,
        )
        for i, (method, score) in enumerate(ranked_adaptive, 1):
            print(f"  {i}. {method}: {score:.4f}")


if __name__ == "__main__":
    main()
