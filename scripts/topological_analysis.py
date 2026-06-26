#!/usr/bin/env python3
"""
topological_analysis.py
=======================

Higher-order topological analysis of PPI network embeddings using
persistent homology (Vietoris-Rips complexes).

This module extends the G-F Consistency Framework by computing
topological invariants (Betti numbers) at each distance threshold r,
providing a richer characterization of embedding geometry than
pairwise graph-based methods alone.

Core concepts
-------------
- **Betti number β₀(r):** Number of connected components (= number of
  communities in the spatial graph G(r)).
- **Betti number β₁(r):** Number of 1-dimensional holes (loops) in the
  Vietoris-Rips complex VR(r).  These capture cyclic arrangements of
  proteins that pairwise graphs cannot detect.
- **Betti number β₂(r):** Number of 2-dimensional voids (enclosed
  cavities) in VR(r).  These indicate shell-like functional clusters.
- **Persistence diagram:** A multiset of (birth, death) pairs for each
  topological feature, recording at which r a feature appears and
  disappears.  Long-lived features are "signal"; short-lived ones are
  "noise".

Output files
------------
- ``results/topological_analysis.json``
- ``figures/Fig8_betti_curves.png``
- ``figures/Fig9_topo_vs_standard_purity.png``
- ``figures/Fig10_persistence_diagrams.png``

CLI usage
---------
::

    python scripts/topological_analysis.py
    python scripts/topological_analysis.py --max-dim 2 --method DM
"""
from __future__ import annotations

import sys
import json
import argparse
import logging
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, R_MIN, R_MAX, N_POINTS, GF_R_MIN, GF_R_MAX,
    ALL_METHODS, PLATEAU_RELATIVE_THRESHOLD,
    get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
    load_curated_network, load_embedding, compute_gf_curve,
    compute_gf_score, rescale_coordinates, precompute_distance_matrix,
    build_spatial_graph_fast, functional_purity,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Persistent Homology Computation
# ===========================================================================

def compute_persistent_homology(coords, max_dim=1, max_radius=None):
    """Compute persistent homology of a point cloud using Ripser.

    Constructs the Vietoris-Rips complex on the given 2-D coordinates
    and computes persistence diagrams for homology dimensions 0
    (connected components), 1 (loops), and optionally 2 (voids).

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        Embedding coordinates.
    max_dim : int
        Maximum homology dimension.  Default 1 (H₀ and H₁).
        Set to 2 to also compute H₂ (slower for large n).
    max_radius : float or None
        Maximum filtration radius.  If None, uses the diameter of the
        point set.

    Returns
    -------
    dict
        ``{dim: np.ndarray}`` where each array has shape (k, 2) with
        columns (birth, death).  Infinite deaths are clipped to
        ``max_radius``.
    """
    try:
        from ripser import ripser
    except ImportError:
        raise ImportError(
            "ripser is required. Install with: pip install ripser"
        )

    if max_radius is None:
        # Use a generous upper bound based on the coordinate range
        max_radius = np.max(np.ptp(coords, axis=0)) * 1.5

    # Ripser expects a point cloud or distance matrix
    result = ripser(
        coords,
        maxdim=max_dim,
        thresh=max_radius,
        do_cocycles=False,
    )

    diagrams = {}
    for dim in range(max_dim + 1):
        if dim < len(result["dgms"]):
            dgm = result["dgms"][dim].copy()
            # Replace infinite deaths with max_radius
            dgm[np.isinf(dgm[:, 1]), 1] = max_radius
            diagrams[dim] = dgm
        else:
            diagrams[dim] = np.empty((0, 2))

    return diagrams


def compute_betti_curves(diagrams, r_vals):
    """Extract Betti number curves from persistence diagrams.

    For each distance threshold r in ``r_vals``, count how many
    topological features of each dimension are "alive" (i.e.,
    birth ≤ r < death).

    Parameters
    ----------
    diagrams : dict
        Persistence diagrams from :func:`compute_persistent_homology`.
    r_vals : np.ndarray
        Distance thresholds to evaluate.

    Returns
    -------
    dict
        ``{dim: np.ndarray}`` of Betti number values at each r.
    """
    betti_curves = {}
    for dim, dgm in diagrams.items():
        betti = np.zeros(len(r_vals), dtype=int)
        for birth, death in dgm:
            alive = (r_vals >= birth) & (r_vals < death)
            betti[alive] += 1
        betti_curves[dim] = betti
    return betti_curves


def compute_persistence_statistics(diagrams):
    """Compute summary statistics from persistence diagrams.

    Parameters
    ----------
    diagrams : dict
        Persistence diagrams.

    Returns
    -------
    dict
        ``{dim: {n_features, mean_persistence, max_persistence,
        persistence_entropy, topological_complexity}}``.
    """
    stats = {}
    for dim, dgm in diagrams.items():
        if len(dgm) == 0:
            stats[dim] = {
                "n_features": 0,
                "mean_persistence": 0.0,
                "max_persistence": 0.0,
                "persistence_entropy": 0.0,
                "topological_complexity": 0.0,
            }
            continue

        persistence = dgm[:, 1] - dgm[:, 0]
        persistence = persistence[persistence > 0]

        n_features = len(persistence)
        mean_pers = float(np.mean(persistence)) if n_features > 0 else 0.0
        max_pers = float(np.max(persistence)) if n_features > 0 else 0.0

        # Persistence entropy: normalized Shannon entropy of persistence values
        if n_features > 0 and np.sum(persistence) > 0:
            p_norm = persistence / np.sum(persistence)
            p_norm = p_norm[p_norm > 0]
            entropy = -float(np.sum(p_norm * np.log(p_norm + 1e-12)))
        else:
            entropy = 0.0

        # Topological complexity: sum of all persistence values
        complexity = float(np.sum(persistence))

        stats[dim] = {
            "n_features": n_features,
            "mean_persistence": mean_pers,
            "max_persistence": max_pers,
            "persistence_entropy": entropy,
            "topological_complexity": complexity,
        }

    return stats


# ===========================================================================
# 2. Topological Purity Metrics
# ===========================================================================

def compute_topological_purity_at_r(coords, nodes, go_map, dist_matrix, r):
    """Compute topological purity at a single distance threshold r.

    At each r, we:
    1. Build the spatial graph G(r) (1-skeleton of VR complex)
    2. Detect communities via greedy modularity
    3. Compute functional purity per community
    4. Weight each community's purity by its local topological
       complexity (number of 1-cycles in the VR complex restricted
       to that community)

    The topological purity rewards communities that are both
    functionally pure AND topologically rich (containing loops),
    while penalizing communities that are functionally pure but
    topologically trivial (no higher-order structure).

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
    nodes : list of str
    go_map : dict
    dist_matrix : np.ndarray, shape (n, n)
    r : float

    Returns
    -------
    dict
        ``{standard_purity, topo_purity, beta_0, beta_1,
        community_topo_details}``
    """
    G_r = build_spatial_graph_fast(dist_matrix, r)

    if G_r.number_of_edges() == 0:
        return {
            "standard_purity": 0.0,
            "topo_purity": 0.0,
            "beta_0": G_r.number_of_nodes(),
            "beta_1": 0,
            "community_topo_details": [],
        }

    # Community detection (same as standard G-F)
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G_r))
    standard_purity = functional_purity(communities, go_map, nodes)

    # For each community, count the number of triangles (proxy for
    # local topological richness in 2D — exact β₁ computation per
    # subcomplex is expensive, but triangle count is a fast proxy)
    community_details = []
    weighted_purities = []

    for comm in communities:
        comm_list = sorted(comm)
        comm_names = [nodes[idx] for idx in comm_list]

        # Functional purity for this community
        from utils import _community_purity
        comm_purity = _community_purity(comm_names, go_map)

        # Local topological complexity: count triangles in the
        # subgraph induced by this community
        G_comm = G_r.subgraph(comm_list)
        n_triangles = _count_triangles(G_comm)

        # Topological weight: 1 + log(1 + n_triangles) / log(1 + max_possible)
        # This gives weight ~1 for trivial topology, higher for rich topology
        max_tri = max(len(comm_list) * (len(comm_list) - 1) *
                      (len(comm_list) - 2) / 6, 1)
        topo_weight = 1.0 + np.log1p(n_triangles) / np.log1p(max_tri)

        weighted_purities.append(comm_purity * topo_weight)
        community_details.append({
            "n_nodes": len(comm_list),
            "purity": comm_purity,
            "n_triangles": n_triangles,
            "topo_weight": topo_weight,
            "weighted_purity": comm_purity * topo_weight,
        })

    # Topological purity: mean of weighted purities, renormalized
    # so that topo_purity ∈ [0, 1]
    if weighted_purities:
        mean_weighted = float(np.mean(weighted_purities))
        # The maximum possible weight is bounded; normalize
        max_weight = 2.0  # empirical upper bound for topo_weight
        topo_purity = min(mean_weighted / max_weight, 1.0)
    else:
        topo_purity = 0.0

    # Betti numbers (global for this r)
    beta_0 = nx.number_connected_components(G_r)

    # β₁ = edges - vertices + connected_components (Euler characteristic)
    n_edges = G_r.number_of_edges()
    n_vertices = G_r.number_of_nodes()
    beta_1 = n_edges - n_vertices + beta_0

    return {
        "standard_purity": standard_purity,
        "topo_purity": topo_purity,
        "beta_0": beta_0,
        "beta_1": max(beta_1, 0),  # Ensure non-negative
        "community_topo_details": community_details,
    }


def _count_triangles(G):
    """Count the number of triangles in a graph.

    Uses the trace of A³ / 6 formula for efficiency.

    Parameters
    ----------
    G : nx.Graph

    Returns
    -------
    int
        Number of triangles.
    """
    if G.number_of_nodes() < 3:
        return 0
    # NetworkX built-in triangle counting
    triangles_per_node = nx.triangles(G)
    total = sum(triangles_per_node.values()) // 3
    return total


def compute_topo_gf_curve(coords, nodes, go_map, r_vals):
    """Compute the topological G-F curve.

    Extends :func:`utils.compute_gf_curve` with topological dimensions.
    Betti numbers are computed via persistent homology (exact), not
    via the Euler formula (which overcounts in dense graphs).

    Optimised: community detection is run once per unique graph
    structure and the result is reused for both purity and modularity.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
    nodes : list of str
    go_map : dict
    r_vals : np.ndarray

    Returns
    -------
    dict
        ``{standard_purities, topo_purities, modularities,
        beta_0, beta_1}``
    """
    dist_matrix = precompute_distance_matrix(coords)
    n = dist_matrix.shape[0]

    # ---- Compute persistent homology ONCE for the full point cloud ----
    # This gives exact Betti numbers at every r via birth/death pairs.
    max_r = float(np.max(r_vals)) * 1.1
    diagrams = compute_persistent_homology(coords, max_dim=1, max_radius=max_r)
    betti_curves = compute_betti_curves(diagrams, r_vals)

    # Pre-sort upper-triangle edges by distance (incremental graph build)
    iu = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[iu]
    sort_idx = np.argsort(edge_dists)
    sorted_rows = iu[0][sort_idx]
    sorted_cols = iu[1][sort_idx]
    sorted_d = edge_dists[sort_idx]

    r_order = np.argsort(r_vals)
    n_r = len(r_vals)
    std_pur = [0.0] * n_r
    topo_pur = [0.0] * n_r
    mods = [0.0] * n_r
    beta_0_arr = [0] * n_r
    beta_1_arr = [0] * n_r

    from networkx.algorithms.community import greedy_modularity_communities
    from networkx.algorithms.community import modularity as nx_modularity
    from utils import _community_purity

    G_r = nx.Graph()
    G_r.add_nodes_from(range(n))
    edge_ptr = 0
    n_edges_total = len(sorted_d)
    _cache = {}  # n_edges -> (standard_purity, topo_purity, modularity)

    for rank, orig_idx in enumerate(r_order):
        r = float(r_vals[orig_idx])

        # Incrementally add edges with distance < r
        while edge_ptr < n_edges_total and sorted_d[edge_ptr] < r:
            G_r.add_edge(int(sorted_rows[edge_ptr]), int(sorted_cols[edge_ptr]))
            edge_ptr += 1

        ne = G_r.number_of_edges()
        if ne == 0:
            beta_0_arr[orig_idx] = G_r.number_of_nodes()
            continue

        # Betti numbers from persistent homology (exact)
        beta_0_arr[orig_idx] = int(betti_curves[0][orig_idx]) if 0 in betti_curves else 0
        beta_1_arr[orig_idx] = int(betti_curves[1][orig_idx]) if 1 in betti_curves else 0

        if ne in _cache:
            sp, tp, m = _cache[ne]
        else:
            # Community detection (run ONCE per unique graph)
            communities = list(greedy_modularity_communities(G_r))
            sp = functional_purity(communities, go_map, nodes)

            # Topological purity (triangle-weighted)
            weighted_purities = []
            for comm in communities:
                comm_list = sorted(comm)
                comm_names = [nodes[idx] for idx in comm_list]
                comm_purity = _community_purity(comm_names, go_map)
                G_comm = G_r.subgraph(comm_list)
                n_tri = _count_triangles(G_comm)
                max_tri = max(len(comm_list) * (len(comm_list) - 1) *
                              (len(comm_list) - 2) / 6, 1)
                topo_w = 1.0 + np.log1p(n_tri) / np.log1p(max_tri)
                weighted_purities.append(comm_purity * topo_w)

            if weighted_purities:
                mean_w = float(np.mean(weighted_purities))
                tp = min(mean_w / 2.0, 1.0)
            else:
                tp = 0.0

            # Modularity
            m = nx_modularity(G_r, communities) if len(communities) > 1 else 0.0

            _cache[ne] = (sp, tp, m)

        std_pur[orig_idx] = sp
        topo_pur[orig_idx] = tp
        mods[orig_idx] = m

    return {
        "standard_purities": std_pur,
        "topo_purities": topo_pur,
        "modularities": mods,
        "beta_0": beta_0_arr,
        "beta_1": beta_1_arr,
    }


# ===========================================================================
# 3. Topological Consistency Score
# ===========================================================================

def topological_consistency_score(diagrams, standard_purities, r_vals):
    """Compute a topological consistency score for an embedding.

    Measures how well the persistence of topological features aligns
    with the functional purity profile.  A high score means that
    topological transitions (birth/death of features) coincide with
    significant changes in functional purity.

    Parameters
    ----------
    diagrams : dict
        Persistence diagrams.
    standard_purities : list of float
        Standard G-F purity values at each r.
    r_vals : np.ndarray

    Returns
    -------
    float
        Topological consistency score in [0, 1].
    """
    purities = np.array(standard_purities)

    # Compute purity gradient (rate of change)
    if len(purities) < 3:
        return 0.0
    purity_gradient = np.abs(np.gradient(purities, r_vals))

    # Collect all birth and death events from H₁
    events = []
    if 1 in diagrams:
        for birth, death in diagrams[1]:
            events.append(birth)
            events.append(death)

    if not events:
        return 0.0

    events = np.array(events)

    # For each topological event, find the nearest r value and
    # check the purity gradient there
    scores = []
    for event_r in events:
        idx = np.argmin(np.abs(r_vals - event_r))
        # Normalize gradient to [0, 1] range
        if np.max(purity_gradient) > 0:
            local_gradient = purity_gradient[idx] / np.max(purity_gradient)
        else:
            local_gradient = 0.0
        scores.append(local_gradient)

    # Topological consistency = mean gradient at event points
    # High = topological transitions happen where purity changes a lot
    return float(np.mean(scores)) if scores else 0.0


# ===========================================================================
# 4. Visualization
# ===========================================================================

def plot_betti_curves(all_betti_curves, r_vals, figures_dir):
    """Plot Betti number curves for all embedding methods.

    Creates a multi-panel figure showing β₀(r) and β₁(r) for each
    method, enabling visual comparison of topological complexity.

    Parameters
    ----------
    all_betti_curves : dict
        ``{method: {0: array, 1: array}}``
    r_vals : np.ndarray
    figures_dir : Path
    """
    methods = list(all_betti_curves.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Betti Number Curves Across Embedding Methods",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for method, color in zip(methods, colors):
        curves = all_betti_curves[method]
        # β₀ (connected components)
        if 0 in curves:
            axes[0].plot(r_vals, curves[0], '-', color=color,
                         linewidth=1.5, label=method)
        # β₁ (loops)
        if 1 in curves:
            axes[1].plot(r_vals, curves[1], '-', color=color,
                         linewidth=1.5, label=method)

    axes[0].set_ylabel("β₀ (components)", fontsize=12)
    axes[0].set_title("Connected Components (H₀)", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right", ncol=3)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Distance threshold r", fontsize=12)
    axes[1].set_ylabel("β₁ (loops)", fontsize=12)
    axes[1].set_title("1-Dimensional Holes (H₁)", fontsize=11)
    axes[1].legend(fontsize=8, loc="upper right", ncol=3)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig8_betti_curves.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig8: Betti curves saved")


def plot_topo_vs_standard_purity(all_topo_results, r_vals, figures_dir):
    """Plot topological purity vs standard purity for all methods.

    Parameters
    ----------
    all_topo_results : dict
        ``{method: {standard_purities, topo_purities}}``
    r_vals : np.ndarray
    figures_dir : Path
    """
    methods = list(all_topo_results.keys())
    n_methods = len(methods)

    # Select top 6 methods for clarity
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Topological Purity vs Standard G-F Purity",
                 fontsize=14, fontweight="bold")

    display_methods = methods[:6]
    for ax, method in zip(axes.flat, display_methods):
        data = all_topo_results[method]
        std_p = data["standard_purities"]
        topo_p = data["topo_purities"]

        ax.plot(r_vals, std_p, '-', color='#4E79A7', linewidth=2,
                label="Standard purity")
        ax.plot(r_vals, topo_p, '--', color='#E15759', linewidth=2,
                label="Topological purity")

        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.set_xlabel("r", fontsize=10)
        ax.set_ylabel("Purity", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for ax in axes.flat[n_methods:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig9_topo_vs_standard_purity.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig9: Topo vs standard purity saved")


def plot_persistence_diagrams(all_diagrams, figures_dir):
    """Plot persistence diagrams as scatter plots.

    Parameters
    ----------
    all_diagrams : dict
        ``{method: {dim: np.ndarray}}``
    figures_dir : Path
    """
    methods = list(all_diagrams.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Persistence Diagrams (H₁: Loops)",
                 fontsize=14, fontweight="bold")

    display_methods = methods[:6]
    for ax, method in zip(axes.flat, display_methods):
        dgm = all_diagrams[method].get(1, np.empty((0, 2)))
        if len(dgm) > 0:
            ax.scatter(dgm[:, 0], dgm[:, 1], s=20, alpha=0.7,
                       c='#4E79A7', edgecolors='black', linewidths=0.3)
            # Diagonal reference line
            max_val = max(np.max(dgm), 0.01)
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3,
                    linewidth=1)

        ax.set_title(f"{method}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Birth", fontsize=10)
        ax.set_ylabel("Death", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    for ax in axes.flat[n_methods:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig10_persistence_diagrams.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig10: Persistence diagrams saved")


# ===========================================================================
# 5. Main Pipeline
# ===========================================================================

def main():
    """Run the full topological analysis pipeline.

    Workflow:
    1. Load curated yeast PPI network and embeddings
    2. For each method: compute persistent homology + Betti curves
    3. For each method: compute topological G-F curve
    4. Compute topological consistency scores
    5. Generate figures (Fig8, Fig9, Fig10)
    6. Save results to JSON
    """
    np.random.seed(SEED)

    # ---- Setup logging ----
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # ---- Directories ----
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = get_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()

    # ---- CLI arguments ----
    parser = argparse.ArgumentParser(
        description="Higher-order topological analysis of PPI embeddings"
    )
    parser.add_argument(
        "--max-dim", type=int, default=1,
        help="Maximum homology dimension (default: 1 for H₀, H₁)"
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to analyze (default: all available)"
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation"
    )
    args = parser.parse_args()

    # ---- Load network ----
    print("=" * 70)
    print("Higher-Order Topological Analysis of PPI Embeddings")
    print("=" * 70)

    print("\nLoading curated yeast PPI network...")
    G, nodes, go_map = load_curated_network(data_dir)
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  GO annotations: {len(go_map)} genes")

    # ---- Distance threshold grid ----
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # ---- Determine methods to process ----
    target_methods = args.methods if args.methods else ALL_METHODS

    # ---- Process each method ----
    all_betti_curves = {}
    all_topo_results = {}
    all_diagrams = {}
    all_persistence_stats = {}
    all_consistency_scores = {}

    print(f"\nAnalyzing {len(target_methods)} methods "
          f"(max_dim={args.max_dim}, {N_POINTS} points)...")

    for method in target_methods:
        print(f"\n{'-' * 50}")
        print(f"  [{method}]")
        print(f"{'-' * 50}")

        try:
            # Load embedding
            coords, emb_nodes = load_embedding(
                method, "153", embeddings_dir=emb_dir
            )
            common = sorted(set(emb_nodes) & set(nodes) & set(go_map.keys()))
            if len(common) < 10:
                print(f"    SKIP: only {len(common)} common nodes")
                continue

            emb_node_map = {n: i for i, n in enumerate(emb_nodes)}
            idx_map = [emb_node_map[n] for n in common]
            aligned_coords = coords[idx_map]
            aligned_coords = rescale_coordinates(aligned_coords)

            print(f"    Nodes: {len(common)}, Computing persistent homology...")

            # ---- Persistent homology ----
            diagrams = compute_persistent_homology(
                aligned_coords, max_dim=args.max_dim
            )

            # ---- Betti curves ----
            betti_curves = compute_betti_curves(diagrams, r_vals)
            all_betti_curves[method] = betti_curves

            # ---- Persistence statistics ----
            p_stats = compute_persistence_statistics(diagrams)
            all_persistence_stats[method] = p_stats

            n_h0 = p_stats[0]["n_features"]
            n_h1 = p_stats[1]["n_features"] if 1 in p_stats else 0
            max_pers_h1 = p_stats[1]["max_persistence"] if 1 in p_stats else 0
            print(f"    H0 features: {n_h0}")
            print(f"    H1 features: {n_h1}, max persistence: {max_pers_h1:.4f}")

            # ---- Topological G-F curve ----
            print(f"    Computing topological G-F curve...")
            topo_result = compute_topo_gf_curve(
                aligned_coords, common, go_map, r_vals
            )
            all_topo_results[method] = topo_result
            all_diagrams[method] = diagrams

            # ---- Standard G-F Score (for comparison) ----
            std_score = compute_gf_score(
                r_vals, topo_result["standard_purities"],
                GF_R_MIN, GF_R_MAX
            )

            # ---- Topological consistency score ----
            consistency = topological_consistency_score(
                diagrams, topo_result["standard_purities"], r_vals
            )
            all_consistency_scores[method] = consistency

            # ---- Report ----
            mean_topo_purity = float(np.mean(topo_result["topo_purities"]))
            max_beta1 = max(topo_result["beta_1"]) if topo_result["beta_1"] else 0
            print(f"    Standard G-F Score: {std_score:.4f}")
            print(f"    Mean topo purity:   {mean_topo_purity:.4f}")
            print(f"    Max b1:             {max_beta1}")
            print(f"    Topo consistency:   {consistency:.4f}")

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ---- Generate figures ----
    if not args.skip_figures and all_betti_curves:
        print(f"\n{'=' * 50}")
        print("Generating figures...")
        print(f"{'=' * 50}")

        plot_betti_curves(all_betti_curves, r_vals, figures_dir)
        plot_topo_vs_standard_purity(all_topo_results, r_vals, figures_dir)
        plot_persistence_diagrams(all_diagrams, figures_dir)

    # ---- Save results ----
    output = {
        "methods": list(all_betti_curves.keys()),
        "r_vals": r_vals.tolist(),
        "persistence_statistics": all_persistence_stats,
        "consistency_scores": all_consistency_scores,
        "topo_gf_curves": {},
        "betti_curves": {},
    }

    for method in all_topo_results:
        output["topo_gf_curves"][method] = {
            "standard_purities": all_topo_results[method]["standard_purities"],
            "topo_purities": all_topo_results[method]["topo_purities"],
            "beta_0": all_topo_results[method]["beta_0"],
            "beta_1": all_topo_results[method]["beta_1"],
        }

    for method in all_betti_curves:
        output["betti_curves"][method] = {
            str(dim): curve.tolist()
            for dim, curve in all_betti_curves[method].items()
        }

    output_file = results_dir / "topological_analysis.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_file}")

    # ---- Summary table ----
    print(f"\n{'=' * 90}")
    print(f"{'Method':<12} {'GF Score':>10} {'H1 feat':>8} "
          f"{'Max pers':>10} {'Topo cons':>11} {'Max b1':>7}")
    print(f"{'-' * 90}")

    for method in sorted(all_persistence_stats.keys(),
                         key=lambda m: all_consistency_scores.get(m, 0),
                         reverse=True):
        h1_n = all_persistence_stats[method].get(1, {}).get("n_features", 0)
        h1_max = all_persistence_stats[method].get(1, {}).get(
            "max_persistence", 0)
        cons = all_consistency_scores.get(method, 0)
        max_b1 = max(all_topo_results[method]["beta_1"]) if (
            method in all_topo_results and all_topo_results[method]["beta_1"]
        ) else 0

        # Get standard GF score
        if method in all_topo_results:
            std_score = compute_gf_score(
                r_vals, all_topo_results[method]["standard_purities"],
                GF_R_MIN, GF_R_MAX
            )
        else:
            std_score = 0.0

        print(f"{method:<12} {std_score:>10.4f} {h1_n:>8d} "
              f"{h1_max:>10.4f} {cons:>11.4f} {max_b1:>7d}")

    print(f"{'=' * 90}")
    print("\nTopological analysis complete.")


if __name__ == "__main__":
    main()
