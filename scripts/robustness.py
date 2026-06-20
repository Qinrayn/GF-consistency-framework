#!/usr/bin/env python3
"""
robustness.py
Step 5: Robustness subset analysis.
- Multiple random subsets at multiple size levels
- Compute G-F curves for each embedding method on each subset
- Paired t-test with Bonferroni correction
- Extended output with per-size and overall summary

Backward compatible: running with no arguments produces the same output
files as the original 10-subset x 150-node version, plus new extended files.
"""

import sys
import json
import time
import random
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, load_curated_network,
    compute_centrality_features, rescale_coordinates,
    compute_gf_curve, save_embedding,
    build_similarity_matrix, diffusion_map_from_similarity,
    classical_mds_from_distances, spectral_embedding_from_graph,
)

R_MIN = 0.05
R_MAX = 0.55

# Legacy defaults (used when no CLI args are supplied)
DEFAULT_N_SUBSETS = 30
DEFAULT_SUBSET_SIZES = "50,100,150,200,all"
DEFAULT_N_POINTS = 30
DEFAULT_METHODS = "DM,MDS,Spectral"

# The size used for Bonferroni testing (must be <= 153, the curated network size)
BONFERRONI_REFERENCE_SIZE = 150


# ----------------------------------------------------------------
# Embedding methods
# ----------------------------------------------------------------

def embed_diffusion_map(G, nodes):
    """DM embedding for a subset graph."""
    features = compute_centrality_features(G, nodes)
    sim = build_similarity_matrix(features)
    coords = diffusion_map_from_similarity(sim)
    return rescale_coordinates(coords, target_std=0.3)


def embed_mds_subset(G, nodes):
    """MDS embedding for a subset graph."""
    n = len(nodes)
    lengths = dict(nx.shortest_path_length(G))
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    D = np.zeros((n, n))
    for u, dists in lengths.items():
        i = node_to_idx[u]
        D[i, :] = [dists.get(v, n) for v in nodes]
    coords = classical_mds_from_distances(D)
    return rescale_coordinates(coords, target_std=0.3)


def embed_spectral_subset(G, nodes):
    """Spectral embedding for a subset graph."""
    coords = spectral_embedding_from_graph(G, nodelist=nodes)
    return rescale_coordinates(coords, target_std=0.3)


# Generic dispatcher: method name -> embedding function
EMBED_METHODS = {
    "DM": embed_diffusion_map,
    "MDS": embed_mds_subset,
    "SPECTRAL": embed_spectral_subset,
}


def get_embed_function(method_name):
    """Return the embedding function for *method_name* (case-insensitive)."""
    key = method_name.upper()
    if key not in EMBED_METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Available: {sorted(EMBED_METHODS.keys())}"
        )
    return EMBED_METHODS[key]


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Robustness subset analysis with multiple sizes and methods."
    )
    p.add_argument(
        "--n-subsets", type=int, default=DEFAULT_N_SUBSETS,
        help=f"Number of random subsets per size level (default: {DEFAULT_N_SUBSETS})."
    )
    p.add_argument(
        "--subset-sizes", type=str, default=DEFAULT_SUBSET_SIZES,
        help=(
            "Comma-separated subset sizes. Use 'all' for the full network. "
            f"(default: '{DEFAULT_SUBSET_SIZES}')"
        ),
    )
    p.add_argument(
        "--n-points", type=int, default=DEFAULT_N_POINTS,
        help=f"Number of r-value sample points (default: {DEFAULT_N_POINTS})."
    )
    p.add_argument(
        "--methods", type=str, default=DEFAULT_METHODS,
        help=f"Comma-separated embedding methods to test (default: '{DEFAULT_METHODS}')."
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick test mode: 5 subsets, size=150 only."
    )
    return p.parse_args()


def parse_sizes(raw: str):
    """Parse the --subset-sizes string into a list of ints / 'all' tokens."""
    sizes = []
    for token in raw.split(","):
        token = token.strip().lower()
        if token == "all":
            sizes.append("all")
        else:
            sizes.append(int(token))
    return sizes


# ----------------------------------------------------------------
# Core analysis helpers
# ----------------------------------------------------------------

def run_subset(G_full, all_nodes, go_map, subset_nodes, methods, r_vals):
    """Run all requested embedding methods on one subset.

    Returns a dict keyed by method name, each value is
    ``{"purity": [...], "modularity": [...]}``.
    """
    G_sub = G_full.subgraph(subset_nodes).copy()

    # Ensure connectivity
    if not nx.is_connected(G_sub):
        comp = max(nx.connected_components(G_sub), key=len)
        G_sub = G_sub.subgraph(comp).copy()
    subset_nodes_sorted = sorted(G_sub.nodes())

    results = {}
    for method in methods:
        embed_fn = get_embed_function(method)
        pos = embed_fn(G_sub, subset_nodes_sorted)
        pur, mod = compute_gf_curve(pos, subset_nodes_sorted, go_map, r_vals)
        results[method] = {"purity": pur, "modularity": mod}

    return subset_nodes_sorted, results


def compute_bonferroni(purity_matrices, methods, r_vals, n_points):
    """Bonferroni-corrected paired t-tests between all method pairs.

    *purity_matrices* is a dict  method_name -> np.ndarray (n_subsets x n_points).
    Tests every pair of methods with Bonferroni correction applied per pair.
    """
    if len(methods) < 2:
        return None

    from itertools import combinations

    alpha = 0.05
    pair_results = []

    for m1, m2 in combinations(methods, 2):
        mat1 = purity_matrices[m1]
        mat2 = purity_matrices[m2]

        p_values = []
        for j in range(n_points):
            _, p_val = ttest_rel(mat1[:, j], mat2[:, j])
            p_values.append(p_val)

        p_values = np.array(p_values)
        bonf_threshold = alpha / n_points

        significant_raw = (p_values < alpha).tolist()
        significant_corrected = (p_values < bonf_threshold).tolist()
        n_sig_corrected = sum(significant_corrected)

        sig_r_values = [float(r_vals[j]) for j in range(n_points) if significant_corrected[j]]

        # Plateau region (r in roughly [0.30, 0.43])
        plateau_mask = (r_vals >= 0.30) & (r_vals <= 0.43)
        plateau_sig = sum(
            significant_corrected[j] for j in range(n_points) if plateau_mask[j]
        )

        pair_results.append({
            "methods_compared": [m1, m2],
            "n_tests": n_points,
            "alpha": alpha,
            "bonferroni_threshold": bonf_threshold,
            "p_values": p_values.tolist(),
            "significant_raw": significant_raw,
            "significant_corrected": significant_corrected,
            "n_significant_corrected": n_sig_corrected,
            "significant_r_values": sig_r_values,
            "n_significant_in_plateau": plateau_sig,
        })

    # Legacy: return the first pair result as the top-level object
    # (for backward compatibility with code that expects single-pair output)
    legacy = pair_results[0] if pair_results else None

    return {
        "pairs": pair_results,
        "n_pairs": len(pair_results),
        # Backward-compatible fields (from first pair)
        "methods_compared": legacy["methods_compared"] if legacy else [],
        "n_tests": legacy["n_tests"] if legacy else 0,
        "alpha": alpha,
        "bonferroni_threshold": legacy["bonferroni_threshold"] if legacy else alpha,
        "p_values": legacy["p_values"] if legacy else [],
        "significant_raw": legacy["significant_raw"] if legacy else [],
        "significant_corrected": legacy["significant_corrected"] if legacy else [],
        "n_significant_corrected": legacy["n_significant_corrected"] if legacy else 0,
        "significant_r_values": legacy["significant_r_values"] if legacy else [],
        "n_significant_in_plateau": legacy["n_significant_in_plateau"] if legacy else 0,
    }


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        n_subsets = 5
        sizes = [150]
        methods = [m.strip().upper() for m in args.methods.split(",")]
        n_points = args.n_points
    else:
        n_subsets = args.n_subsets
        sizes = parse_sizes(args.subset_sizes)
        methods = [m.strip().upper() for m in args.methods.split(",")]
        n_points = args.n_points

    random.seed(SEED)
    np.random.seed(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load network
    print("Loading curated network...")
    G_full, nodes, go_map = load_curated_network(data_dir)
    total_nodes = len(nodes)
    print(f"Network: {total_nodes} nodes")

    r_vals = np.linspace(R_MIN, R_MAX, n_points)

    # Resolve "all" and clamp sizes
    resolved_sizes = []
    for s in sizes:
        if s == "all":
            resolved_sizes.append(total_nodes)
        elif s >= total_nodes:
            resolved_sizes.append(total_nodes)
        else:
            resolved_sizes.append(s)
    # Remove duplicates while preserving order
    seen = set()
    unique_sizes = []
    for s in resolved_sizes:
        if s not in seen:
            seen.add(s)
            unique_sizes.append(s)
    resolved_sizes = unique_sizes

    total_embeddings = n_subsets * len(resolved_sizes) * len(methods)
    print(f"\nConfiguration:")
    print(f"  Subsets per size : {n_subsets}")
    print(f"  Size levels      : {resolved_sizes}")
    print(f"  Methods           : {methods}")
    print(f"  r-value points    : {n_points}")
    print(f"  Total embeddings  : {total_embeddings}")
    print()

    t_start = time.time()
    completed_embeddings = 0

    # Storage:  size_label -> method -> list of purity vectors
    all_results = {}       # size_label -> list of per-subset dicts
    size_summaries = {}     # size_label -> {method -> {mean, std}}
    bonf_result = None

    for size_val in resolved_sizes:
        size_label = str(size_val)
        print(f"{'='*60}")
        print(f"  Size level: {size_val} nodes")
        print(f"{'='*60}")

        purity_lists = {m: [] for m in methods}   # method -> list of purity vectors
        subset_records = []

        for subset_idx in range(n_subsets):
            random.seed(SEED + subset_idx)
            np.random.seed(SEED + subset_idx)

            # Sample nodes
            if size_val >= total_nodes:
                subset_nodes = list(nodes)
            else:
                subset_nodes = random.sample(nodes, size_val)

            t_sub_start = time.time()

            sorted_nodes, method_results = run_subset(
                G_full, nodes, go_map, subset_nodes, methods, r_vals
            )

            # Collect purities
            record = {
                "seed": subset_idx,
                "n_nodes": len(sorted_nodes),
            }
            for m in methods:
                pur = method_results[m]["purity"]
                mod = method_results[m]["modularity"]
                purity_lists[m].append(pur)
                record[f"purity_{m.lower()}"] = pur
                record[f"modularity_{m.lower()}"] = mod

            subset_records.append(record)
            completed_embeddings += len(methods)

            elapsed = time.time() - t_start
            sub_elapsed = time.time() - t_sub_start
            remaining = (elapsed / completed_embeddings) * (total_embeddings - completed_embeddings) if completed_embeddings else 0

            print(
                f"  Subset {subset_idx+1:3d}/{n_subsets}  "
                f"({len(sorted_nodes)} nodes)  "
                f"{sub_elapsed:.1f}s  "
                f"[elapsed {elapsed:.0f}s, est. remaining {remaining:.0f}s]"
            )

        # Per-size summary statistics
        size_summary = {}
        for m in methods:
            mat = np.array(purity_lists[m])
            size_summary[m] = {
                "mean_purity": mat.mean(axis=0).tolist(),
                "std_purity": mat.std(axis=0).tolist(),
            }
        size_summaries[size_label] = size_summary

        # Save per-size raw results
        per_size_file = results_dir / f"robustness_size_{size_label}.json"
        with open(per_size_file, "w", encoding="utf-8") as f:
            json.dump({
                "size": size_val,
                "n_subsets": n_subsets,
                "r_values": r_vals.tolist(),
                "subsets": subset_records,
            }, f, indent=2)
        print(f"  -> Saved {per_size_file.name}")

        # Bonferroni test: use the first size <= 153 (typically 150)
        if bonf_result is None and size_val <= BONFERRONI_REFERENCE_SIZE:
            purity_matrices = {m: np.array(purity_lists[m]) for m in methods}
            bonf_result = compute_bonferroni(purity_matrices, methods, r_vals, n_points)
            if bonf_result is not None:
                print(f"\n  Bonferroni test (all pairs, size={size_val}):")
                for pair in bonf_result.get("pairs", []):
                    m1, m2 = pair["methods_compared"]
                    print(f"    {m1} vs {m2}: "
                          f"{pair['n_significant_corrected']}/{n_points} significant "
                          f"(plateau: {pair['n_significant_in_plateau']})")

        all_results[size_label] = subset_records

    # ----------------------------------------------------------------
    # If no size was <= BONFERRONI_REFERENCE_SIZE, run Bonferroni on the
    # smallest available size as a fallback.
    # ----------------------------------------------------------------
    if bonf_result is None and len(resolved_sizes) > 0:
        fallback_size = resolved_sizes[0]
        fallback_label = str(fallback_size)
        fallback_records = all_results[fallback_label]
        purity_matrices = {}
        for m in methods:
            purity_matrices[m] = np.array([
                rec[f"purity_{m.lower()}"] for rec in fallback_records
            ])
        bonf_result = compute_bonferroni(purity_matrices, methods, r_vals, n_points)
        if bonf_result is not None:
            print(f"\n  Bonferroni test (fallback, all pairs, size={fallback_size}):")
            for pair in bonf_result.get("pairs", []):
                m1, m2 = pair["methods_compared"]
                print(f"    {m1} vs {m2}: "
                      f"{pair['n_significant_corrected']}/{n_points} significant")

    total_time = time.time() - t_start

    # ----------------------------------------------------------------
    # Backward-compatible output (subset_summary.json, bonferroni_results.json)
    # Use size=150 results if available, otherwise the first size level.
    # ----------------------------------------------------------------
    compat_size = BONFERRONI_REFERENCE_SIZE if str(BONFERRONI_REFERENCE_SIZE) in size_summaries else resolved_sizes[0]
    compat_label = str(compat_size)
    compat_summary = size_summaries[compat_label]

    # Build the legacy summary dict (mirrors the original structure)
    legacy_summary = {
        "r_values": r_vals.tolist(),
        "n_points": n_points,
        "n_subsets": n_subsets,
    }
    # Add per-method fields using the original naming convention
    for m in methods:
        suffix = m.lower()
        legacy_summary[f"mean_purity_{suffix}"] = compat_summary[m]["mean_purity"]
        legacy_summary[f"std_purity_{suffix}"] = compat_summary[m]["std_purity"]

    summary_file = results_dir / "subset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(legacy_summary, f, indent=2)

    bonf_file = results_dir / "bonferroni_results.json"
    if bonf_result is not None:
        with open(bonf_file, "w", encoding="utf-8") as f:
            json.dump(bonf_result, f, indent=2)

    # Also save backward-compatible raw subset data
    raw_file = results_dir / "subset_robustness.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(all_results.get(compat_label, []), f, indent=2)

    # ----------------------------------------------------------------
    # Extended overall summary
    # ----------------------------------------------------------------
    extended = {
        "parameters": {
            "n_subsets": n_subsets,
            "subset_sizes": resolved_sizes,
            "n_points": n_points,
            "methods": methods,
        },
    }
    for size_label in size_summaries:
        key = f"size_{size_label}"
        extended[key] = {}
        for m in methods:
            extended[key][m] = size_summaries[size_label][m]

    if bonf_result is not None:
        extended["bonferroni"] = bonf_result

    extended["total_time_seconds"] = round(total_time, 1)

    extended_file = results_dir / "robustness_extended.json"
    with open(extended_file, "w", encoding="utf-8") as f:
        json.dump(extended, f, indent=2)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Done in {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"  Saved (backward-compatible):")
    print(f"    {summary_file.name}")
    if bonf_result is not None:
        print(f"    {bonf_file.name}")
    print(f"    {raw_file.name}")
    print(f"  Saved (extended):")
    for size_val in resolved_sizes:
        print(f"    robustness_size_{size_val}.json")
    print(f"    {extended_file.name}")


if __name__ == "__main__":
    main()
