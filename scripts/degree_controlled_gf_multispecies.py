#!/usr/bin/env python3
"""
degree_controlled_gf_multispecies.py
=====================================
Extends the pair-level degree-controlled G-F analysis (Step 42c) from
yeast to human and mouse PPI networks.

For each species:
  1. Load PPI network, 2D embeddings (all 11 methods), and GO annotations
  2. Subsample to a tractable number of annotated proteins (2000 for
     human/mouse; 153 for yeast)
  3. Compute pairwise: embedding distance, GO Jaccard, degree dissimilarity
  4. Partial Spearman correlation rho(D, S | Delta) with permutation test
  5. Compare method rankings across species

The cross-species question: does the "3/11 methods retain significant
signal" finding replicate on human and mouse, or is it yeast-specific?

Output: results/degree_controlled_gf_multispecies.json
"""

from __future__ import annotations

import sys
import json
import gzip
import time
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, set_seed, ALL_METHODS,
    load_curated_network, rescale_coordinates,
    get_data_dir, get_results_dir,
)

set_seed(SEED)

SUBSAMPLE_SIZE = 500   # for human/mouse (keeps permutation test tractable)
N_PERM = 200           # permutation test iterations


# ===================================================================
# Data loading
# ===================================================================

def load_yeast():
    """Load yeast curated network, GO map, and .npy embeddings."""
    G, nodes, go_map = load_curated_network()

    embeddings = {}
    for method in ALL_METHODS:
        npy = Path(SCRIPT_DIR).parent / "embeddings" / f"{method}_153.npy"
        nodes_f = Path(SCRIPT_DIR).parent / "embeddings" / f"{method}_153_nodes.json"
        if npy.exists():
            coords = np.load(npy)
            with open(nodes_f, encoding="utf-8") as f:
                emb_nodes = json.load(f)
            embeddings[method] = (coords, emb_nodes)
    return G, nodes, go_map, embeddings, "Yeast_curated"


def load_species_from_json(species, ppi_loader, go_file, emb_pattern):
    """Generic loader for species with JSON-format embeddings.

    Parameters
    ----------
    species : str
        Species name for display.
    ppi_loader : callable
        Function that returns (nx.Graph, nodelist).
    go_file : Path
        Path to GO annotations JSON (gene -> [GO terms]).
    emb_pattern : str
        Pattern for embedding files, e.g. "data/human_{method}_embedding.json"
        with {method} placeholder.
    """
    data_dir = get_data_dir()
    results_dir = get_results_dir()

    # Load PPI
    G = ppi_loader()

    # Load GO
    with open(go_file, encoding="utf-8") as f:
        go_map = json.load(f)

    # Load embeddings (JSON format: {node_id: {"x": ..., "y": ...}})
    embeddings = {}
    for method in ALL_METHODS:
        method_lower = method.lower()
        emb_file = data_dir / emb_pattern.format(method=method_lower)
        if not emb_file.exists():
            # Try original case
            emb_file = data_dir / emb_pattern.format(method=method)
        if not emb_file.exists():
            continue
        with open(emb_file, encoding="utf-8") as f:
            raw = json.load(f)
        # Convert {node: {"x":.., "y":..}} to (coords, nodes)
        emb_nodes = list(raw.keys())
        coords = np.array([[raw[n]["x"], raw[n]["y"]] for n in emb_nodes],
                          dtype=np.float64)
        embeddings[method] = (coords, emb_nodes)

    nodes = sorted(G.nodes())
    return G, nodes, go_map, embeddings, species


def load_human_ppi():
    """Load human PPI from STRING v12.0."""
    links_file = PROJECT_ROOT / "human_validation" / "9606.protein.links.v12.0.txt.gz"
    G = nx.Graph()
    with gzip.open(str(links_file), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and int(parts[2]) >= 700:
                G.add_edge(parts[0], parts[1])
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def load_mouse_ppi():
    """Load mouse PPI from edgelist."""
    data_dir = get_data_dir()
    edgelist = data_dir / "mouse_ppi.edgelist"
    G = nx.Graph()
    with open(edgelist, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


# ===================================================================
# Pairwise computation
# ===================================================================

def compute_go_jaccard(nodes, go_map):
    """Pairwise GO Jaccard similarity matrix."""
    n = len(nodes)
    term_sets = []
    for node in nodes:
        node_str = str(node)
        terms = set(go_map.get(node_str, go_map.get(node, [])))
        term_sets.append(terms)

    S = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ti = term_sets[i]
        if not ti:
            continue
        for j in range(i + 1, n):
            tj = term_sets[j]
            if not tj:
                continue
            union = ti | tj
            if len(union) > 0:
                S[i, j] = len(ti & tj) / len(union)
                S[j, i] = S[i, j]
    return S


def compute_degree_dissimilarity(G, nodes):
    """Pairwise degree dissimilarity: |log10(deg+1)_i - log10(deg+1)_j|."""
    log_deg = np.array([np.log10(G.degree(node) + 1) for node in nodes])
    return np.abs(log_deg[:, None] - log_deg[None, :])


def upper_tri(matrix):
    n = matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return matrix[idx]


def partial_spearman(x, y, z):
    """Partial Spearman: residualize rank(x) and rank(y) on rank(z)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(y_arr, x_arr):
        X = np.column_stack([np.ones(len(y_arr)), x_arr])
        beta, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
        return y_arr - X @ beta

    res_x = resid(rx, rz)
    res_y = resid(ry, rz)
    rho, p = spearmanr(res_x, res_y)
    return float(rho), float(p)


def permutation_test(D, S, Delta, n_perm=N_PERM, seed=42):
    """Permutation test for partial Spearman rho(D, S | Delta)."""
    rng = np.random.RandomState(seed)
    rho_obs, _ = partial_spearman(D, S, Delta)
    null = []
    for _ in range(n_perm):
        S_perm = rng.permutation(S)
        r, _ = partial_spearman(D, S_perm, Delta)
        null.append(r)
    null = np.array(null)
    null_mean = float(np.mean(null))
    null_std = float(np.std(null))
    z = float((rho_obs - null_mean) / (null_std + 1e-10))
    p = float(np.mean(np.abs(null - null_mean) >= abs(rho_obs - null_mean)))
    return float(rho_obs), null_mean, null_std, z, p


def compute_attenuation(rho_std, rho_part):
    if abs(rho_std) < 0.02:
        return None
    return float(1.0 - rho_part / rho_std)


# ===================================================================
# Subsampling
# ===================================================================

def subsample_annotated(G, nodes, go_map, embeddings, target_n, seed=42):
    """Subsample to target_n annotated nodes with embeddings.

    Selects nodes that: (a) have GO annotations, (b) are in the network,
    (c) appear in all embedding methods.
    """
    rng = np.random.RandomState(seed)

    # Find nodes with GO annotations
    annotated = [n for n in nodes if str(n) in go_map or n in go_map]
    annotated_set = set(annotated)

    # Find nodes present in all embeddings
    common_all = None
    for method, (coords, emb_nodes) in embeddings.items():
        emb_set = set(emb_nodes)
        if common_all is None:
            common_all = emb_set
        else:
            common_all = common_all & emb_set

    usable = sorted(annotated_set & common_all)
    if len(usable) <= target_n:
        return usable

    selected = rng.choice(usable, size=target_n, replace=False)
    return sorted(selected)


# ===================================================================
# Per-species analysis
# ===================================================================

def analyze_species(G, nodes, go_map, embeddings, species_name,
                    target_n=SUBSAMPLE_SIZE):
    """Run degree-controlled analysis for one species."""
    t0 = time.time()
    print(f"\n  [{species_name}] Starting analysis ...")
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Subsample
    if len(nodes) > target_n:
        selected = subsample_annotated(G, nodes, go_map, embeddings, target_n)
    else:
        selected = nodes
    n = len(selected)
    print(f"  Subsampled: {n} annotated nodes with embeddings")

    # Compute shared matrices
    G_sub = G.subgraph(selected)
    S = compute_go_jaccard(selected, go_map)
    Delta = compute_degree_dissimilarity(G, selected)

    S_vals = upper_tri(S)
    Delta_vals = upper_tri(Delta)

    n_pairs = len(S_vals)
    n_shared = int(np.sum(S_vals > 0))
    print(f"  Pairs: {n_pairs}, sharing GO: {n_shared} ({100*n_shared/n_pairs:.1f}%)")

    # Confound check
    rho_conf, p_conf = spearmanr(Delta_vals, S_vals)
    print(f"  Degree-annotation confound: rho={rho_conf:+.4f} (p={p_conf:.2e})")

    # Per-method analysis
    node_to_idx = {n: i for i, n in enumerate(selected)}
    method_results = []

    for method in ALL_METHODS:
        if method not in embeddings:
            continue
        coords, emb_nodes = embeddings[method]

        emb_idx_map = {nd: i for i, nd in enumerate(emb_nodes)}
        common = [nd for nd in selected if nd in emb_idx_map]
        if len(common) < 20:
            continue

        emb_idx = [emb_idx_map[nd] for nd in common]
        net_idx = [node_to_idx[nd] for nd in common]

        Y = rescale_coordinates(coords[emb_idx].copy())
        D_mat = squareform(pdist(Y))
        D_vals = upper_tri(D_mat)

        S_sub = S[np.ix_(net_idx, net_idx)]
        Delta_sub = Delta[np.ix_(net_idx, net_idx)]
        S_sub_vals = upper_tri(S_sub)
        Delta_sub_vals = upper_tri(Delta_sub)

        rho_std, p_std = spearmanr(D_vals, S_sub_vals)
        rho_part, p_part = partial_spearman(D_vals, S_sub_vals, Delta_sub_vals)
        atten = compute_attenuation(rho_std, rho_part)

        # Permutation test (fewer perms for large species)
        n_perm = N_PERM if len(common) > 500 else 200
        rho_perm, null_m, null_s, z, p_perm = permutation_test(
            D_vals, S_sub_vals, Delta_sub_vals, n_perm=n_perm, seed=SEED
        )

        method_results.append({
            "method": method,
            "n_common": len(common),
            "rho_ds": float(rho_std),
            "rho_ds_p": float(p_std),
            "rho_ds_partial": float(rho_part),
            "rho_ds_partial_p": float(p_part),
            "attenuation": atten,
            "permutation_p": float(p_perm),
            "permutation_z": float(z),
            "n_perm": n_perm,
        })

        atten_str = f"{atten:.1%}" if atten is not None else "N/A"
        print(f"    {method:14s}  rho(D,S)={rho_std:+.4f}  "
              f"rho(D,S|deg)={rho_part:+.4f}  atten={atten_str:>6s}  "
              f"p={p_perm:.4f}")

    dt = time.time() - t0
    print(f"  [{species_name}] Done in {dt:.1f}s")

    return {
        "species": species_name,
        "n_nodes_network": G.number_of_nodes(),
        "n_nodes_analyzed": n,
        "n_pairs": n_pairs,
        "n_pairs_shared_go": n_shared,
        "confound_rho": float(rho_conf),
        "confound_p": float(p_conf),
        "method_results": method_results,
        "elapsed_sec": dt,
    }


# ===================================================================
# Cross-species comparison
# ===================================================================

def cross_species_comparison(species_results):
    """Compare degree-controlled findings across species."""
    print("\n" + "=" * 80)
    print("  CROSS-SPECIES COMPARISON")
    print("=" * 80)

    # 1. Confound replication
    print("\n  Degree-annotation confound:")
    for sr in species_results:
        print(f"    {sr['species']:18s}  rho={sr['confound_rho']:+.4f}  "
              f"(p={sr['confound_p']:.2e})")

    # 2. Significant methods per species
    print("\n  Methods with significant degree-controlled signal (p < 0.05):")
    for sr in species_results:
        sig = [m for m in sr["method_results"] if m["permutation_p"] < 0.05]
        sig_names = [m["method"] for m in sig]
        print(f"    {sr['species']:18s}  {len(sig)}/{len(sr['method_results'])}  "
              f"methods: {sig_names}")

    # 3. MDS/Spectral/DM across species
    print("\n  Key methods across species:")
    print(f"    {'Method':<14s}", end="")
    for sr in species_results:
        print(f"  {sr['species'][-8:]:>10s}", end="")
    print()
    for method in ["Spectral", "MDS", "DM", "DeepWalk", "PCA"]:
        print(f"    {method:<14s}", end="")
        for sr in species_results:
            mr = [m for m in sr["method_results"] if m["method"] == method]
            if mr:
                rho = mr[0]["rho_ds_partial"]
                p = mr[0]["permutation_p"]
                tag = "*" if p < 0.05 else " "
                print(f"  {rho:+8.4f}{tag}", end="")
            else:
                print(f"  {'N/A':>10s}", end="")
        print()
    print("    (* = permutation p < 0.05)")

    # 4. Top-3 stability
    print("\n  Top-3 by rho(D,S|degree) per species:")
    for sr in species_results:
        ranked = sorted(sr["method_results"], key=lambda x: x["rho_ds_partial"])
        top3 = [m["method"] for m in ranked[:3]]
        print(f"    {sr['species']:18s}  {top3}")

    # 5. Cross-species rank concordance
    print("\n  Cross-species rank concordance (rho(D,S|deg) rankings):")
    method_names = ["Spectral", "MDS", "DM", "DeepWalk", "Node2Vec",
                    "PCA", "VGAE", "VGAE-feat", "GraphSAGE", "GAT", "GIN"]
    rankings = []
    species_labels = []
    for sr in species_results:
        method_map = {m["method"]: m["rho_ds_partial"] for m in sr["method_results"]}
        vals = [-method_map.get(m, 0) for m in method_names]  # negate: higher = better
        if any(m in method_map for m in method_names):
            rankings.append(vals)
            species_labels.append(sr["species"])

    if len(rankings) >= 2:
        # Kendall's W via manual computation (not all scipy versions have kendallw)
        def kendalls_w(rankings_arr):
            """Kendall's coefficient of concordance W."""
            m, n = rankings_arr.shape
            # Rank each judge's ratings
            from scipy.stats import rankdata
            R = np.array([rankdata(r) for r in rankings_arr])
            R_sum = R.sum(axis=0)
            R_mean = R_sum.mean()
            S = np.sum((R_sum - R_mean) ** 2)
            W = 12 * S / (m ** 2 * (n ** 3 - n))
            return W

        rankings_arr = np.array(rankings)
        if len(rankings) >= 3:
            W = kendalls_w(rankings_arr)
            print(f"    Kendall's W = {W:.4f} (n={len(species_labels)} species)")
        for i in range(len(species_labels)):
            for j in range(i + 1, len(species_labels)):
                r, pp = spearmanr(rankings[i], rankings[j])
                print(f"    {species_labels[i][-8:]} vs {species_labels[j][-8:]}: "
                      f"rho={r:+.4f} (p={pp:.4f})")


# ===================================================================
# Main
# ===================================================================

def main():
    t_start = time.time()
    print("=" * 72)
    print("  Degree-Controlled G-F: Cross-Species Extension")
    print("  Yeast + Human + Mouse")
    print("=" * 72)

    # ----------------------------------------------------------------
    # Load all species
    # ----------------------------------------------------------------
    data_dir = get_data_dir()

    print("\n[1/3] Loading species data ...")

    # Yeast
    print("  Loading yeast ...")
    G_y, nodes_y, go_y, emb_y, name_y = load_yeast()
    print(f"    {G_y.number_of_nodes()} nodes, {len(emb_y)} methods")

    # Human
    print("  Loading human ...")
    G_h, nodes_h, go_h, emb_h, name_h = load_species_from_json(
        "Human",
        load_human_ppi,
        data_dir / "human_go_annotations.json",
        "human_{method}_embedding.json",
    )
    print(f"    {G_h.number_of_nodes()} nodes, {len(emb_h)} methods")

    # Mouse
    print("  Loading mouse ...")
    G_m, nodes_m, go_m, emb_m, name_m = load_species_from_json(
        "Mouse",
        load_mouse_ppi,
        data_dir / "mouse_go_annotations.json",
        "mouse_{method}_embedding.json",
    )
    print(f"    {G_m.number_of_nodes()} nodes, {len(emb_m)} methods")

    # ----------------------------------------------------------------
    # Run analysis per species
    # ----------------------------------------------------------------
    print("\n[2/3] Running degree-controlled analysis per species ...")

    all_results = []

    # Yeast (153 nodes, no subsampling needed)
    sr_y = analyze_species(G_y, nodes_y, go_y, emb_y, name_y, target_n=153)
    all_results.append(sr_y)

    # Human (15882 -> 2000)
    sr_h = analyze_species(G_h, nodes_h, go_h, emb_h, name_h, target_n=SUBSAMPLE_SIZE)
    all_results.append(sr_h)

    # Mouse (16180 -> 2000)
    sr_m = analyze_species(G_m, nodes_m, go_m, emb_m, name_m, target_n=SUBSAMPLE_SIZE)
    all_results.append(sr_m)

    # ----------------------------------------------------------------
    # Cross-species comparison
    # ----------------------------------------------------------------
    print("\n[3/3] Cross-species comparison ...")
    cross_species_comparison(all_results)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    results_dir = get_results_dir()
    output = {
        "analysis": "Degree-Controlled G-F: Cross-Species Extension",
        "description": (
            "Pair-level partial Spearman correlation of embedding distance "
            "and GO Jaccard, controlling for degree dissimilarity, across "
            "three species. Tests whether the yeast finding (3/11 methods "
            "retain significant signal) replicates on human and mouse."
        ),
        "parameters": {
            "subsample_size": SUBSAMPLE_SIZE,
            "n_permutations": N_PERM,
            "seed": SEED,
        },
        "species_results": all_results,
    }

    out_path = results_dir / "degree_controlled_gf_multispecies.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("  Done.")

    return output


if __name__ == "__main__":
    main()