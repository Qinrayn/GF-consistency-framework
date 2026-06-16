#!/usr/bin/env python3
"""
coexpression_gf.py
==================
Run the G-F Score framework on a co-expression network derived from
the yeast STRING v11.5 full-links file.

Demonstrates that the G-F Score findings (in particular, the Spectral
ranking) generalise beyond physical PPI networks to functional
co-expression networks.

Pipeline
--------
  1. Load STRING full-links file and extract the coexpression channel
     (column index 7 in the header, score >= 400).
  2. Build an undirected graph, take the largest connected component.
  3. Build a broad GO annotation map from the SGD GAF file (mapped to
     STRING protein IDs via the aliases file), giving ~1500 annotated
     nodes instead of the 34 from the curated 153-node PPI GO map.
  4. Intersect with the annotated node set and take the largest CC.
  5. Compute 2-D embeddings for 7 methods:
       Spectral, DM, MDS, DeepWalk, Node2Vec, PCA, GIN
  6. Compute G-F curves (200 points, r in [0.05, 0.55]) using the
     connected-components fast approximation for community detection
     (validated to preserve method rankings; see gf-curve-fast-
     approximation skill).
  7. Compute G-F Scores (integrated over [0.05, 0.422]).
  8. Compute random baseline (10 shuffles).
  9. Compare method ranking with the PPI results.

Output
------
  results/coexpression_gf.json
"""

import sys
import json
import gzip
import time
import random
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components as cc_components
from scipy.sparse import csr_matrix
from scipy.integrate import trapezoid
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir,
    compute_centrality_features,
    build_similarity_matrix, diffusion_map_from_similarity,
    classical_mds_from_distances,
    spectral_embedding_from_graph, deepwalk_from_graph,
    node2vec_from_graph, rescale_coordinates,
    GF_R_MIN, GF_R_MAX, R_MIN, R_MAX, N_POINTS, TARGET_STD,
)

# GIN encoder from the existing GNN module
from embed_gnn import gin_from_graph

# ---- Configuration ----
COEXPRESSION_COL = 7        # column index in STRING full links file
CHANNEL_THRESHOLD = 400     # minimum coexpression score
RANDOM_SHUFFLES = 10


# ============================================================
# Data Loading
# ============================================================

def load_coexpression_network(data_dir, min_score=CHANNEL_THRESHOLD):
    """Load co-expression edges from the STRING full links file.

    The STRING v11.5 full links file columns (0-indexed):
      [0] protein1   [1] protein2
      [2] neighborhood   [3] neighborhood_transferred
      [4] fusion
      [5] cooccurence
      [6] homology
      [7] coexpression   [8] coexpression_transferred
      [9] experiments    [10] experiments_transferred
      [11] database      [12] database_transferred
      [13] textmining    [14] textmining_transferred
      [15] combined_score

    Returns
    -------
    (nx.Graph, int)
        Co-expression network (largest CC) and raw edge count.
    """
    full_file = data_dir / "4932.protein.links.full.v11.5.txt.gz"
    if not full_file.exists():
        raise FileNotFoundError(
            f"STRING full links file not found: {full_file}\n"
            "Download from: https://stringdb.org/downloads"
        )

    print(f"  Loading: {full_file.name}")
    G = nx.Graph()
    n_edges_raw = 0

    with gzip.open(str(full_file), "rt", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 16:
                continue
            coex_score = int(parts[COEXPRESSION_COL])
            if coex_score >= min_score:
                p1 = parts[0].split(".")[1]  # strip species prefix
                p2 = parts[1].split(".")[1]
                G.add_edge(p1, p2, weight=coex_score)
                n_edges_raw += 1

    print(f"  Raw coexpression edges (score >= {min_score}): {n_edges_raw}")
    print(f"  Nodes before CC filter: {G.number_of_nodes()}")

    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    return G, n_edges_raw


def build_gaf_go_map(data_dir):
    """Build a broad GO annotation map from the SGD GAF file.

    Extracts yeast systematic names (e.g. YAL001C, Q0010) from
    column 10 of the GAF file (pipe-separated synonyms), then
    maps each name to its GO terms.  This produces ~6000 annotated
    genes covering the full yeast genome.

    Returns
    -------
    dict
        {systematic_name: [go_term_1, go_term_2, ...]}
    """
    import re
    gaf_file = data_dir / "gene_association.sgd.gaf.gz"
    go_map = defaultdict(set)
    # Yeast systematic name pattern: Yxx999W/C or Q999
    sys_name_re = re.compile(r"[YQ][A-Z]{2}\d{3}[WC](?:-[A-Z])?")

    with gzip.open(str(gaf_file), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 11:
                go_term = parts[4]
                if not go_term.startswith("GO:"):
                    continue
                # Column 10: pipe-separated synonyms/aliases
                synonyms = parts[10].split("|")
                for syn in synonyms:
                    syn = syn.strip()
                    if sys_name_re.match(syn):
                        go_map[syn].add(go_term)

    go_map = {k: sorted(v) for k, v in go_map.items()}
    print(f"  GAF-based GO annotations (systematic names): {len(go_map)}")
    return go_map


# ============================================================
# Fast GF Curve (connected-components based)
# ============================================================

def compute_gf_curve_fast(coords, nodes, go_map, r_vals):
    """Compute G-F purity curve using connected components.

    Uses connected components of the spatial graph (distance < r)
    instead of greedy_modularity_communities. This is ~1000x faster
    on graphs with 500+ nodes while preserving method rankings
    (Spearman rho > 0.95 vs exact method; see gf-curve-fast-
    approximation skill).

    Parameters
    ----------
    coords : (n, d) embedding coordinates
    nodes : ordered node labels
    go_map : gene -> [GO terms]
    r_vals : distance thresholds

    Returns
    -------
    (purities, modularities)
    """
    n = len(nodes)
    dist_matrix = squareform(pdist(coords, metric="euclidean"))

    # Build GO term index for sparse purity computation
    all_terms = sorted(set(
        t for node in nodes if node in go_map for t in go_map[node]
    ))
    term_idx = {t: i for i, t in enumerate(all_terms)}
    n_terms = len(all_terms)

    # Build annotation matrix A: (n x n_terms)
    ann_rows, ann_cols = [], []
    for i, node in enumerate(nodes):
        if node in go_map:
            for term in go_map[node]:
                if term in term_idx:
                    ann_rows.append(i)
                    ann_cols.append(term_idx[term])
    A = csr_matrix(
        (np.ones(len(ann_rows)), (ann_rows, ann_cols)),
        shape=(n, n_terms),
    )

    # Pre-sort edges by distance for incremental construction
    iu = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[iu]
    sort_idx = np.argsort(edge_dists)
    sorted_rows = iu[0][sort_idx]
    sorted_cols = iu[1][sort_idx]
    sorted_d = edge_dists[sort_idx]

    # Process r values in ascending order
    r_order = np.argsort(r_vals)
    purities_out = [0.0] * len(r_vals)
    mods_out = [0.0] * len(r_vals)

    # Build sparse adjacency incrementally
    edge_ptr = 0
    n_edges_total = len(sorted_d)

    # Cache by edge count
    _cache = {}

    for rank, orig_idx in enumerate(r_order):
        r = float(r_vals[orig_idx])

        # Add edges with distance < r
        new_rows = []
        new_cols = []
        while edge_ptr < n_edges_total and sorted_d[edge_ptr] < r:
            new_rows.append(int(sorted_rows[edge_ptr]))
            new_cols.append(int(sorted_cols[edge_ptr]))
            edge_ptr += 1

        ne = edge_ptr  # total edges so far
        if ne == 0:
            continue

        if ne in _cache:
            purities_out[orig_idx] = _cache[ne]
            continue

        # Build sparse adjacency for connected components
        rows_arr = np.array(sorted_rows[:ne], dtype=np.int32)
        cols_arr = np.array(sorted_cols[:ne], dtype=np.int32)
        data_arr = np.ones(ne, dtype=np.float64)
        # Symmetric
        adj = csr_matrix(
            (np.concatenate([data_arr, data_arr]),
             (np.concatenate([rows_arr, cols_arr]),
              np.concatenate([cols_arr, rows_arr]))),
            shape=(n, n),
        )

        # Connected components
        n_comp, labels = cc_components(adj, directed=False)

        # Sparse purity computation
        n_clusters = labels.max() + 1
        C = csr_matrix(
            (np.ones(n), (labels, np.arange(n))),
            shape=(n_clusters, n),
        )
        counts = C @ A  # (n_clusters x n_terms)
        cluster_sizes = np.array(C.sum(axis=1)).ravel()

        # Mean functional purity: for each cluster, max_count / size
        purity_per_cluster = np.zeros(n_clusters)
        nonzero = cluster_sizes > 0
        if nonzero.any():
            counts_sub = counts[nonzero]
            # max(axis=1) on CSR returns (n,1) sparse matrix
            max_counts = np.asarray(
                counts_sub.max(axis=1).todense()
            ).ravel().astype(float)
            purity_per_cluster[nonzero] = (
                max_counts / cluster_sizes[nonzero]
            )

        # Mean purity across all non-empty clusters
        valid = cluster_sizes > 0
        mean_purity = float(
            purity_per_cluster[valid].mean()
        ) if valid.any() else 0.0

        purities_out[orig_idx] = mean_purity
        _cache[ne] = mean_purity

    return purities_out, mods_out


def compute_gf_score_fast(r_vals, purity_vals,
                          r_min=GF_R_MIN, r_max=GF_R_MAX):
    """Compute the G-F Score as mean purity over [r_min, r_max]."""
    r = np.asarray(r_vals)
    p = np.asarray(purity_vals)
    mask = (r >= r_min) & (r <= r_max)
    r_sub = r[mask]
    p_sub = p[mask]
    if len(r_sub) < 2:
        return 0.0
    return float(trapezoid(p_sub, r_sub) / (r_max - r_min))


# ============================================================
# Embedding Functions
# ============================================================

def embed_diffusion_map(G, nodes, features=None):
    """Diffusion Map: centrality features -> similarity -> Markov -> eigen."""
    if features is None:
        features = compute_centrality_features(G, nodes)
    sim = build_similarity_matrix(features)
    coords = diffusion_map_from_similarity(sim)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_mds(G, nodes):
    """Classical MDS on shortest-path distances."""
    n = len(nodes)
    # For large graphs, use BFS-based shortest paths (more efficient)
    lengths = dict(nx.shortest_path_length(G))
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    D = np.full((n, n), n, dtype=float)
    for u, dists in lengths.items():
        if u not in node_to_idx:
            continue
        i = node_to_idx[u]
        for v, d in dists.items():
            if v in node_to_idx:
                D[i, node_to_idx[v]] = d
    coords = classical_mds_from_distances(D)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_spectral(G, nodes):
    """Spectral embedding from normalised Laplacian."""
    coords = spectral_embedding_from_graph(G, nodelist=nodes)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_deepwalk(G, nodes):
    """DeepWalk: uniform random walks + co-occurrence SVD."""
    coords = deepwalk_from_graph(
        G, walk_length=20, walks_per_node=10,
        window_size=5, dimensions=2, seed=SEED,
    )
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_node2vec(G, nodes):
    """Node2Vec: biased random walks + co-occurrence SVD."""
    coords = node2vec_from_graph(
        G, walk_length=20, walks_per_node=10,
        window_size=5, dimensions=2, p=0.5, q=2.0, seed=SEED,
    )
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_pca(G, nodes, features=None):
    """PCA on centrality features (non-structural baseline)."""
    if features is None:
        features = compute_centrality_features(G, nodes)
    features_c = features - features.mean(axis=0)
    cov = features_c.T @ features_c / (len(nodes) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    coords = features_c @ eigvecs[:, -2:]
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_gin(G, nodes, features=None):
    """GIN: 2-layer Graph Isomorphism Network encoder."""
    coords = gin_from_graph(
        G, hidden_dim=16, latent_dim=2,
        epochs=300, lr=0.01, features=features, seed=SEED,
    )
    return rescale_coordinates(coords, target_std=TARGET_STD)


# ============================================================
# Random Baseline
# ============================================================

def compute_random_baseline(coords, nodes, go_map, r_vals,
                            n_shuffles=RANDOM_SHUFFLES):
    """Shuffle node-coordinate mapping, return mean GF Score."""
    n = len(nodes)
    scores = []
    for s in range(n_shuffles):
        rng = np.random.RandomState(SEED + s + 5000)
        perm = rng.permutation(n)
        shuffled_coords = coords[perm]
        purities, _ = compute_gf_curve_fast(
            shuffled_coords, nodes, go_map, r_vals,
        )
        score = compute_gf_score_fast(r_vals, purities)
        scores.append(score)
        print(f"    Shuffle {s + 1}/{n_shuffles}: score = {score:.4f}")
    return float(np.mean(scores)), float(np.std(scores))


# ============================================================
# Main Pipeline
# ============================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Step 1: Load co-expression network
    # ============================================================
    print("=" * 60)
    print("Step 1: Load co-expression network from STRING")
    print("=" * 60)

    G_coex, n_raw_edges = load_coexpression_network(
        data_dir, min_score=CHANNEL_THRESHOLD,
    )
    print(f"  Largest CC: {G_coex.number_of_nodes()} nodes, "
          f"{G_coex.number_of_edges()} edges")

    # ============================================================
    # Step 2: Build broad GO annotations from GAF
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Step 2: Build GO annotations from SGD GAF file")
    print("=" * 60)

    go_map = build_gaf_go_map(data_dir)

    # Intersect coexpression network with annotated nodes
    annotated = sorted(set(G_coex.nodes()) & set(go_map.keys()))
    print(f"  Annotated nodes in coexpression network: {len(annotated)}")

    if len(annotated) < 20:
        print("ERROR: Too few annotated nodes to proceed.")
        sys.exit(1)

    G_ann = G_coex.subgraph(annotated).copy()
    # Take largest CC of the annotated subgraph
    if G_ann.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G_ann), key=len)
        G_ann = G_ann.subgraph(largest_cc).copy()
        annotated = sorted(G_ann.nodes())

    nodes = annotated
    go_map_sub = {n: go_map[n] for n in nodes}
    n_nodes = len(nodes)
    n_edges = G_ann.number_of_edges()
    density = nx.density(G_ann)

    print(f"\n  --- Network Statistics ---")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {n_edges}")
    print(f"  Density: {density:.6f}")
    print(f"  Connected components: {nx.number_connected_components(G_ann)}")

    # ============================================================
    # Step 3: Compute embeddings
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Step 3: Compute 2-D embeddings (7 methods)")
    print("=" * 60)

    print("  Computing centrality features...")
    t0 = time.time()
    features = compute_centrality_features(G_ann, nodes)
    print(f"  Centrality features computed in {time.time() - t0:.1f}s")

    methods = {
        "Spectral": lambda: embed_spectral(G_ann, nodes),
        "DM": lambda: embed_diffusion_map(G_ann, nodes, features=features),
        "MDS": lambda: embed_mds(G_ann, nodes),
        "DeepWalk": lambda: embed_deepwalk(G_ann, nodes),
        "Node2Vec": lambda: embed_node2vec(G_ann, nodes),
        "PCA": lambda: embed_pca(G_ann, nodes, features=features),
        "GIN": lambda: embed_gin(G_ann, nodes, features=features),
    }

    method_coords = {}
    for method_name, embed_fn in methods.items():
        print(f"\n  Computing {method_name}...")
        random.seed(SEED)
        np.random.seed(SEED)
        t0 = time.time()
        try:
            coords = embed_fn()
            elapsed = time.time() - t0
            method_coords[method_name] = coords
            print(f"    Shape: {coords.shape}, "
                  f"std: {np.std(coords):.4f}, "
                  f"time: {elapsed:.1f}s")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Successfully computed: "
          f"{len(method_coords)}/{len(methods)} methods")

    if not method_coords:
        print("ERROR: No embeddings computed. Exiting.")
        sys.exit(1)

    # ============================================================
    # Step 4: Compute G-F curves and scores
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Step 4: Compute G-F curves and scores (fast approx)")
    print("=" * 60)

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    gf_scores = {}
    gf_curves = {}

    for method_name, coords in method_coords.items():
        print(f"  {method_name}...")
        t0 = time.time()
        purities, modularities = compute_gf_curve_fast(
            coords, nodes, go_map_sub, r_vals,
        )
        score = compute_gf_score_fast(r_vals, purities)
        gf_scores[method_name] = score
        gf_curves[method_name] = {
            "purities": [round(p, 6) for p in purities],
        }
        elapsed = time.time() - t0
        peak_purity = max(purities)
        print(f"    GF Score: {score:.4f}  "
              f"(peak purity: {peak_purity:.4f}, "
              f"time: {elapsed:.1f}s)")

    # ============================================================
    # Step 5: Random baseline
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Step 5: Random baseline (10 shuffles)")
    print("=" * 60)

    baseline_method = list(method_coords.keys())[0]
    baseline_coords = method_coords[baseline_method]
    random_mean, random_std = compute_random_baseline(
        baseline_coords, nodes, go_map_sub, r_vals,
    )
    print(f"  Random baseline GF Score: {random_mean:.4f} +/- {random_std:.4f}")

    # ============================================================
    # Step 6: Load PPI results for comparison
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Step 6: Comparison with PPI results")
    print("=" * 60)

    ppi_scores = {}
    ppi_results_file = results_dir / "gf_scores.json"
    gnn_results_file = results_dir / "gnn_gf_scores.json"

    if ppi_results_file.exists():
        with open(ppi_results_file, encoding="utf-8") as f:
            ppi_data = json.load(f)
        ppi_scores.update(ppi_data.get("scores", {}))
        ppi_random = ppi_data.get("random_baseline", None)
        print(f"  Loaded PPI results: {len(ppi_scores)} methods")
    else:
        ppi_random = None
        print("  WARNING: PPI results file not found")

    if gnn_results_file.exists():
        with open(gnn_results_file, encoding="utf-8") as f:
            gnn_data = json.load(f)
        ppi_scores.update(gnn_data.get("gf_scores", {}))
        print(f"  Loaded GNN results: "
              f"{len(gnn_data.get('gf_scores', {}))} methods")

    # Build ranking comparison for methods present in both
    common_methods = sorted(
        set(gf_scores.keys()) & set(ppi_scores.keys()),
    )

    coex_ranking = sorted(
        common_methods, key=lambda m: gf_scores[m], reverse=True,
    )
    ppi_ranking = sorted(
        common_methods, key=lambda m: ppi_scores[m], reverse=True,
    )

    coex_ranks = {m: i + 1 for i, m in enumerate(coex_ranking)}
    ppi_ranks = {m: i + 1 for i, m in enumerate(ppi_ranking)}

    print(f"\n  {'Method':<12} {'PPI Score':>10} {'CoEx Score':>11} "
          f"{'PPI Rank':>9} {'CoEx Rank':>10}")
    print(f"  {'-' * 52}")
    for m in coex_ranking:
        print(f"  {m:<12} {ppi_scores[m]:>10.4f} {gf_scores[m]:>11.4f} "
              f"{ppi_ranks[m]:>9} {coex_ranks[m]:>10}")

    # Rank correlation
    rho, p_val = None, None
    if len(common_methods) >= 3:
        from scipy import stats
        ppi_rank_vals = [ppi_ranks[m] for m in common_methods]
        coex_rank_vals = [coex_ranks[m] for m in common_methods]
        rho, p_val = stats.spearmanr(ppi_rank_vals, coex_rank_vals)
        rho = float(rho)
        p_val = float(p_val)
        print(f"\n  Spearman rank correlation (PPI vs CoEx): "
              f"rho={rho:.4f}, p={p_val:.4f}")

    # Check if Spectral still ranks first
    coex_top = coex_ranking[0] if coex_ranking else "N/A"
    ppi_top = ppi_ranking[0] if ppi_ranking else "N/A"
    print(f"\n  PPI top method:          {ppi_top}")
    print(f"  CoExpression top method: {coex_top}")
    print(f"  Spectral ranks first in PPI?  "
          f"{'YES' if ppi_top == 'Spectral' else 'NO'}")
    print(f"  Spectral ranks first in CoEx? "
          f"{'YES' if coex_top == 'Spectral' else 'NO'}")

    # ============================================================
    # Step 7: Save results
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Step 7: Save results")
    print("=" * 60)

    full_ranking = sorted(
        gf_scores.items(), key=lambda x: x[1], reverse=True,
    )

    output = {
        "network_type": "coexpression",
        "source": ("STRING v11.5 coexpression channel "
                   "(column 7, score >= 400)"),
        "go_annotation_source": "SGD GAF file (broad annotations)",
        "community_detection": ("connected_components "
                                "(fast approximation)"),
        "network_statistics": {
            "nodes": n_nodes,
            "edges": n_edges,
            "density": round(density, 6),
            "connected_components": 1,
            "annotated_genes_in_network": n_nodes,
            "raw_coexpression_edges_before_cc": n_raw_edges,
        },
        "gf_parameters": {
            "r_min": GF_R_MIN,
            "r_max": GF_R_MAX,
            "curve_r_min": R_MIN,
            "curve_r_max": R_MAX,
            "n_points": N_POINTS,
        },
        "gf_scores": {m: round(s, 6) for m, s in full_ranking},
        "ranking": [m for m, _ in full_ranking],
        "random_baseline": {
            "mean": round(random_mean, 6),
            "std": round(random_std, 6),
        },
        "methods_above_random": [
            m for m, s in full_ranking if s > random_mean
        ],
        "comparison_with_ppi": {
            "common_methods": common_methods,
            "coexpression_ranks": coex_ranks,
            "ppi_ranks": {m: ppi_ranks[m] for m in common_methods},
            "ppi_scores": {
                m: round(ppi_scores[m], 6) for m in common_methods
            },
            "coexpression_scores": {
                m: round(gf_scores[m], 6) for m in common_methods
            },
            "spearman_rho": round(rho, 4) if rho is not None else None,
            "spearman_p_value": (round(p_val, 4)
                                 if p_val is not None else None),
            "ppi_top_method": ppi_top,
            "coexpression_top_method": coex_top,
            "spectral_first_ppi": ppi_top == "Spectral",
            "spectral_first_coexpression": coex_top == "Spectral",
        },
        "ppi_random_baseline": ppi_random,
        "gf_curves": gf_curves,
        "r_values": [round(r, 6) for r in r_vals.tolist()],
    }

    out_file = results_dir / "coexpression_gf.json"
    with open(str(out_file), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_file}")

    # ============================================================
    # Final Summary
    # ============================================================
    print(f"\n{'=' * 60}")
    print("CO-EXPRESSION G-F SCORE SUMMARY")
    print("=" * 60)
    print(f"  Network: {n_nodes} nodes, {n_edges} edges, "
          f"density={density:.6f}")
    print(f"\n  {'Rank':<5} {'Method':<12} {'GF Score':>10}")
    print(f"  {'-' * 27}")
    for i, (m, s) in enumerate(full_ranking, 1):
        above = " *" if s > random_mean else ""
        print(f"  {i:<5} {m:<12} {s:>10.4f}{above}")
    print(f"\n  Random baseline: {random_mean:.4f} +/- {random_std:.4f}")
    print(f"  (* = above random baseline)")
    print(f"\n  Generalisation check:")
    print(f"    PPI top method:          {ppi_top}")
    print(f"    Co-expression top method: {coex_top}")
    if rho is not None:
        print(f"    Rank correlation (rho):   "
              f"{rho:.4f} (p={p_val:.4f})")


if __name__ == "__main__":
    main()
