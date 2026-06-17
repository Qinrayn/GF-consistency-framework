#!/usr/bin/env python3
"""
string_threshold_sensitivity.py
================================
Test whether core findings (method ranking, dark matter count) are robust
to the STRING combined-score threshold used to define the PPI network.

For each threshold in [600, 700, 800]:

1. Load the full yeast STRING v11.5 network at that threshold.
2. Extract the subgraph induced by the curated 153-node set (same nodes
   across all thresholds -- edges differ).
3. Compute 2-D embeddings (Spectral, MDS, DM, PCA, Node2Vec) on each
   induced subgraph.
4. Compute G-F Scores using BP annotations.
5. Count "functional dark matter" pairs: protein pairs that are
   >= 5 BFS hops apart in the full STRING network, appear in the top-50
   KNN of the Spectral embedding, are not directly connected at the
   given threshold, and share at least one experimental GO BP annotation.
6. Compute Spearman rank correlation of method G-F Scores between
   threshold=700 (reference) and each other threshold.

Results are saved as JSON and printed as a summary table.

Usage
-----
    python scripts/string_threshold_sensitivity.py
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.integrate import trapezoid
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from networkx.algorithms.community import greedy_modularity_communities

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    TARGET_STD,
    get_data_dir,
    get_results_dir,
    load_curated_network,
    load_full_STRING_network,
    rescale_coordinates,
    precompute_distance_matrix,
    compute_centrality_features,
    build_similarity_matrix,
    diffusion_map_from_similarity,
    classical_mds_from_distances,
    spectral_embedding_from_graph,
    node2vec_from_graph,
)

# Import GAF parsing from our sibling script
from go_mf_cc_gf_scores import (
    build_alias_mapping,
    parse_gaf_all_aspects,
    EXPERIMENTAL_CODES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA = get_data_dir()
RESULTS = get_results_dir()
RESULTS.mkdir(parents=True, exist_ok=True)

STRING_FILE = DATA / "4932.protein.links.v11.5.txt.gz"
GAF_FILE = DATA / "gene_association.sgd.gaf.gz"
ALIAS_FILE = DATA / "4932.protein.aliases.v11.5.txt.gz"

THRESHOLDS = [600, 700, 800]
REFERENCE_THRESHOLD = 700

METHODS = ["Spectral", "MDS", "DM", "PCA", "Node2Vec"]

# G-F Score parameters
GF_R_MIN = 0.05
GF_R_MAX = 0.422
N_POINTS = 25

# Dark matter parameters
MIN_NETWORK_DIST = 5   # BFS hops
MAX_EMB_RANK = 50      # top-K KNN in Spectral embedding

BANNER = "=" * 64


# ---------------------------------------------------------------------------
# Network loading helpers
# ---------------------------------------------------------------------------

def load_string_at_threshold(min_score):
    """Load yeast STRING network at a given score threshold.

    Returns the largest connected component.

    Parameters
    ----------
    min_score : int
        Minimum combined STRING score.

    Returns
    -------
    G : nx.Graph
    """
    G = nx.Graph()
    with gzip.open(str(STRING_FILE), "rt", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p1, p2, score = parts
            if int(score) >= min_score:
                p1_clean = p1.split(".")[1]
                p2_clean = p2.split(".")[1]
                G.add_edge(p1_clean, p2_clean)

    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    return G


def extract_curated_subgraph(G_full, curated_nodes):
    """Extract the subgraph of G_full induced by curated_nodes.

    Nodes in curated_nodes that are not in G_full are still included
    as isolated nodes so that the node ordering remains consistent.

    Parameters
    ----------
    G_full : nx.Graph
        Full STRING network.
    curated_nodes : list[str]
        Ordered list of curated node IDs.

    Returns
    -------
    G_sub : nx.Graph
        Induced subgraph on curated nodes.
    present_nodes : list[str]
        Subset of curated_nodes actually present in G_full.
    """
    present = [n for n in curated_nodes if n in G_full]
    G_sub = G_full.subgraph(present).copy()

    # Add isolated nodes for curated nodes not in G_full
    for n in curated_nodes:
        if n not in G_sub:
            G_sub.add_node(n)

    return G_sub, present


# ---------------------------------------------------------------------------
# Embedding computation (reused from embed_all.py patterns)
# ---------------------------------------------------------------------------

def embed_spectral(G, nodes):
    """Spectral embedding (2-D) from normalised Laplacian."""
    coords = spectral_embedding_from_graph(G, nodelist=nodes)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_mds(G, nodes):
    """Classical MDS (2-D) on shortest-path distances."""
    n = len(nodes)
    lengths = dict(nx.shortest_path_length(G))
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    D = np.full((n, n), float(n), dtype=float)
    for u, dists in lengths.items():
        if u not in node_to_idx:
            continue
        i = node_to_idx[u]
        for v, d in dists.items():
            if v in node_to_idx:
                D[i, node_to_idx[v]] = d
    coords = classical_mds_from_distances(D)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_dm(G, nodes):
    """Diffusion Map (2-D) from centrality features."""
    features = compute_centrality_features(G, nodes)
    sim = build_similarity_matrix(features)
    coords = diffusion_map_from_similarity(sim)
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_pca(G, nodes):
    """PCA (2-D) on centrality features."""
    features = compute_centrality_features(G, nodes)
    features_centered = features - features.mean(axis=0)
    cov = features_centered.T @ features_centered / (len(nodes) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    coords = features_centered @ eigvecs[:, -2:]
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_node2vec(G, nodes):
    """Node2Vec (2-D) with default p=0.5, q=2.0."""
    coords = node2vec_from_graph(G, walk_length=20, walks_per_node=10,
                                 window_size=5, dimensions=2,
                                 p=0.5, q=2.0, seed=SEED)
    return rescale_coordinates(coords, target_std=TARGET_STD)


EMBED_FNS = {
    "Spectral": embed_spectral,
    "MDS": embed_mds,
    "DM": embed_dm,
    "PCA": embed_pca,
    "Node2Vec": embed_node2vec,
}


# ---------------------------------------------------------------------------
# G-F Score computation (standalone)
# ---------------------------------------------------------------------------

def compute_gf_curve_standalone(coords, nodes, go_map, r_vals):
    """Compute G-F purity curve using greedy modularity communities."""
    dist_matrix = precompute_distance_matrix(coords)
    n = dist_matrix.shape[0]

    iu = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[iu]
    sort_idx = np.argsort(edge_dists)
    sorted_rows = iu[0][sort_idx]
    sorted_cols = iu[1][sort_idx]
    sorted_d = edge_dists[sort_idx]

    r_order = np.argsort(r_vals)
    purities_out = [0.0] * len(r_vals)

    G_r = nx.Graph()
    G_r.add_nodes_from(range(n))
    edge_ptr = 0
    n_edges_total = len(sorted_d)
    _cache = {}

    for rank, orig_idx in enumerate(r_order):
        r = float(r_vals[orig_idx])

        while edge_ptr < n_edges_total and sorted_d[edge_ptr] < r:
            G_r.add_edge(int(sorted_rows[edge_ptr]),
                         int(sorted_cols[edge_ptr]))
            edge_ptr += 1

        ne = G_r.number_of_edges()
        if ne == 0:
            continue

        if ne in _cache:
            communities = _cache[ne]
        else:
            communities = list(greedy_modularity_communities(G_r))
            _cache[ne] = communities

        purities = []
        for comm in communities:
            if not comm:
                continue
            comm_names = [nodes[idx] for idx in comm]
            go_terms = []
            for name in comm_names:
                if name in go_map:
                    go_terms.extend(go_map[name])
            if not go_terms:
                continue
            term_counts = Counter(go_terms)
            most_common_count = term_counts.most_common(1)[0][1]
            purity = most_common_count / len(go_terms)
            purities.append(purity)

        purities_out[orig_idx] = float(np.mean(purities)) if purities else 0.0

    return purities_out


def compute_gf_score(r_vals, purity_vals, r_min=GF_R_MIN, r_max=GF_R_MAX):
    """G-F Score = mean purity over [r_min, r_max]."""
    r = np.asarray(r_vals)
    p = np.asarray(purity_vals)
    mask = (r >= r_min) & (r <= r_max)
    r_sub = r[mask]
    p_sub = p[mask]
    if len(r_sub) < 2:
        return 0.0
    return float(trapezoid(p_sub, r_sub) / (r_max - r_min))


# ---------------------------------------------------------------------------
# Dark matter detection
# ---------------------------------------------------------------------------

def count_dark_matter_pairs(G_full_string, spectral_coords, spectral_nodes,
                            go_map_bp, curated_nodes_set):
    """Count functional dark matter pairs at a given STRING threshold.

    A pair (u, v) is "functional dark matter" if ALL of:
      1. BFS distance >= MIN_NETWORK_DIST hops in G_full_string
         (or in different connected components).
      2. v is among the top-MAX_EMB_RANK nearest neighbours of u in
         the Spectral embedding space (or vice versa).
      3. (u, v) is NOT directly connected in G_full_string.
      4. u and v share at least one experimental GO BP annotation.

    Parameters
    ----------
    G_full_string : nx.Graph
        Full STRING network at the given threshold.
    spectral_coords : np.ndarray
        Spectral embedding coordinates for curated nodes.
    spectral_nodes : list[str]
        Ordered curated node labels matching spectral_coords.
    go_map_bp : dict
        {orf_name: [GO BP terms]}
    curated_nodes_set : set[str]
        Set of curated node IDs.

    Returns
    -------
    n_dark_matter : int
        Number of unique dark matter pairs.
    dark_matter_pairs : list[tuple[str, str]]
        The pairs themselves.
    """
    # Build KNN from Spectral embedding
    n_nodes = len(spectral_nodes)
    if n_nodes < MAX_EMB_RANK + 1:
        k = n_nodes - 1
    else:
        k = MAX_EMB_RANK

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(spectral_coords)
    distances, indices = nn.kneighbors(spectral_coords)

    # Node name lookup
    node_to_idx = {n: i for i, n in enumerate(spectral_nodes)}

    # Set of edges in the full STRING network (undirected)
    string_edges = set()
    for u, v in G_full_string.edges():
        string_edges.add((min(u, v), max(u, v)))

    # Compute BFS distances from each curated node in the full STRING network
    # Only for nodes present in G_full_string
    bfs_dists = {}
    string_nodes_set = set(G_full_string.nodes())
    for node in spectral_nodes:
        if node in string_nodes_set:
            bfs_dists[node] = dict(
                nx.single_source_shortest_path_length(G_full_string, node)
            )
        else:
            bfs_dists[node] = {}

    # Find dark matter pairs
    dark_matter_set = set()

    for i in range(n_nodes):
        u = spectral_nodes[i]
        u_terms = set(go_map_bp.get(u, []))
        if not u_terms:
            continue

        # Top-K neighbours in embedding space (skip self at index 0)
        for rank in range(1, k + 1):
            j = indices[i, rank]
            v = spectral_nodes[j]

            # Canonical pair ordering
            pair = (min(u, v), max(u, v))
            if pair in dark_matter_set:
                continue

            # Condition 3: not directly connected in STRING
            if pair in string_edges:
                continue

            # Condition 1: BFS distance >= MIN_NETWORK_DIST (or disconnected)
            u_bfs = bfs_dists.get(u, {})
            dist_uv = u_bfs.get(v, float("inf"))
            if dist_uv < MIN_NETWORK_DIST:
                continue

            # Condition 4: shared GO BP annotation
            v_terms = set(go_map_bp.get(v, []))
            if not u_terms.intersection(v_terms):
                continue

            dark_matter_set.add(pair)

    return len(dark_matter_set), sorted(dark_matter_set)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    np.random.seed(SEED)

    print(BANNER)
    print("STRING Threshold Sensitivity Analysis")
    print(BANNER)

    # -----------------------------------------------------------------------
    # Step 1: Load curated network nodes and BP annotations
    # -----------------------------------------------------------------------
    print("\n[1/6] Loading curated 153-node network and BP annotations ...")
    G_curated, curated_nodes_list, go_map_default = load_curated_network(DATA)
    curated_nodes_set = set(curated_nodes_list)
    print(f"  Curated network: {G_curated.number_of_nodes()} nodes, "
          f"{G_curated.number_of_edges()} edges")

    # Parse GAF for BP annotations using the alias mapping
    print("\n  Parsing GAF for BP annotations ...")
    sgd_to_orf, symbol_to_orf, _ = build_alias_mapping()
    annotations, ann_stats = parse_gaf_all_aspects(
        sgd_to_orf, symbol_to_orf, curated_nodes_set,
    )
    go_map_bp = annotations["BP"]
    # Restrict to curated nodes
    go_map_bp = {n: go_map_bp[n] for n in curated_nodes_list if n in go_map_bp}
    print(f"  BP: {len(go_map_bp)} annotated nodes in curated network")

    # -----------------------------------------------------------------------
    # Step 2: Process each threshold
    # -----------------------------------------------------------------------
    r_vals = np.linspace(GF_R_MIN, GF_R_MAX, N_POINTS)
    all_results = {}

    for thr in THRESHOLDS:
        print(f"\n{'='*64}")
        print(f"  Threshold = {thr}")
        print(f"{'='*64}")

        # 2a: Load STRING network at this threshold
        print(f"\n  [2a] Loading STRING network (score >= {thr}) ...")
        t0 = time.time()
        G_string = load_string_at_threshold(thr)
        print(f"    STRING network: {G_string.number_of_nodes()} nodes, "
              f"{G_string.number_of_edges()} edges "
              f"({time.time()-t0:.1f}s)")

        # 2b: Extract curated subgraph
        print(f"\n  [2b] Extracting curated subgraph ...")
        G_sub, present_nodes = extract_curated_subgraph(
            G_string, curated_nodes_list,
        )
        n_present = len(present_nodes)
        n_isolated = len(curated_nodes_list) - n_present
        print(f"    {n_present} curated nodes in STRING LCC, "
              f"{n_isolated} isolated")
        print(f"    Subgraph edges: {G_sub.number_of_edges()}")

        # 2c: Compute embeddings
        print(f"\n  [2c] Computing embeddings ...")
        embeddings = {}
        for method in METHODS:
            np.random.seed(SEED)
            try:
                coords = EMBED_FNS[method](G_sub, curated_nodes_list)
                embeddings[method] = coords
                print(f"    {method}: OK  "
                      f"(std={np.std(coords):.4f})")
            except Exception as e:
                print(f"    {method}: FAILED -- {e}")

        # 2d: Compute G-F Scores
        print(f"\n  [2d] Computing G-F Scores ...")
        gf_scores = {}

        # Filter to annotated nodes for GF computation
        annotated_mask = [i for i, n in enumerate(curated_nodes_list)
                         if n in go_map_bp]
        ann_nodes = [curated_nodes_list[i] for i in annotated_mask]

        for method in METHODS:
            if method not in embeddings:
                continue
            ann_coords = embeddings[method][annotated_mask]
            purities = compute_gf_curve_standalone(
                ann_coords, ann_nodes, go_map_bp, r_vals,
            )
            score = compute_gf_score(r_vals, purities)
            gf_scores[method] = round(score, 4)
            print(f"    {method}: GF Score = {score:.4f}")

        # 2e: Count dark matter pairs
        print(f"\n  [2e] Counting dark matter pairs ...")
        if "Spectral" in embeddings:
            t0 = time.time()
            n_dm, dm_pairs = count_dark_matter_pairs(
                G_string, embeddings["Spectral"], curated_nodes_list,
                go_map_bp, curated_nodes_set,
            )
            print(f"    Dark matter pairs: {n_dm}  ({time.time()-t0:.1f}s)")
        else:
            n_dm = 0
            dm_pairs = []
            print(f"    Dark matter: skipped (Spectral embedding unavailable)")

        # Store results for this threshold
        all_results[thr] = {
            "string_nodes": G_string.number_of_nodes(),
            "string_edges": G_string.number_of_edges(),
            "curated_present": n_present,
            "curated_isolated": n_isolated,
            "subgraph_edges": G_sub.number_of_edges(),
            "gf_scores": gf_scores,
            "dark_matter_count": n_dm,
        }

    # -----------------------------------------------------------------------
    # Step 3: Spearman rank correlations vs reference threshold
    # -----------------------------------------------------------------------
    print(f"\n[3/6] Spearman rank correlations vs threshold={REFERENCE_THRESHOLD} ...")

    ref_scores = all_results[REFERENCE_THRESHOLD]["gf_scores"]
    ref_methods = sorted(ref_scores.keys(), key=lambda m: ref_scores[m],
                         reverse=True)

    for thr in THRESHOLDS:
        thr_scores = all_results[thr]["gf_scores"]
        common_methods = [m for m in ref_methods if m in thr_scores]

        if thr == REFERENCE_THRESHOLD:
            all_results[thr]["spearman_vs_700"] = 1.0
            all_results[thr]["method_ranking"] = common_methods
            continue

        if len(common_methods) < 3:
            all_results[thr]["spearman_vs_700"] = None
            all_results[thr]["method_ranking"] = common_methods
            print(f"  Threshold {thr}: too few common methods for correlation")
            continue

        ref_ranks = [ref_scores[m] for m in common_methods]
        thr_ranks = [thr_scores[m] for m in common_methods]
        rho, p_val = spearmanr(ref_ranks, thr_ranks)

        all_results[thr]["spearman_vs_700"] = round(float(rho), 4)
        all_results[thr]["spearman_pvalue"] = round(float(p_val), 4)
        all_results[thr]["method_ranking"] = sorted(
            common_methods, key=lambda m: thr_scores[m], reverse=True,
        )
        print(f"  Threshold {thr}: rho={rho:.4f}, p={p_val:.4f}")

    # Add ranking for reference too
    all_results[REFERENCE_THRESHOLD]["method_ranking"] = sorted(
        ref_methods, key=lambda m: ref_scores[m], reverse=True,
    )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    # Convert threshold keys to strings for JSON
    output = {
        "description": (
            "STRING threshold sensitivity analysis: G-F Scores, method "
            "rankings, and dark matter counts at thresholds 600, 700, 800."
        ),
        "thresholds": THRESHOLDS,
        "reference_threshold": REFERENCE_THRESHOLD,
        "integration_interval": [GF_R_MIN, GF_R_MAX],
        "n_points": N_POINTS,
        "methods": METHODS,
        "dark_matter_params": {
            "min_bfs_hops": MIN_NETWORK_DIST,
            "max_emb_rank": MAX_EMB_RANK,
        },
        "results": {str(k): v for k, v in all_results.items()},
    }

    out_file = RESULTS / "string_threshold_sensitivity.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {out_file}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print(f"\n{BANNER}")
    print("Summary: STRING Threshold Sensitivity")
    print(BANNER)

    # Table 1: G-F Scores
    print(f"\n  G-F Scores by Method and Threshold:")
    print(f"  {'Method':<12s}", end="")
    for thr in THRESHOLDS:
        print(f" {'thr='+str(thr):>10s}", end="")
    print()
    print(f"  {'-'*42}")
    for method in METHODS:
        print(f"  {method:<12s}", end="")
        for thr in THRESHOLDS:
            score = all_results[thr]["gf_scores"].get(method, None)
            if score is not None:
                print(f" {score:>10.4f}", end="")
            else:
                print(f" {'N/A':>10s}", end="")
        print()

    # Table 2: Method rankings
    print(f"\n  Method Rankings (descending GF Score):")
    for thr in THRESHOLDS:
        ranking = all_results[thr].get("method_ranking", [])
        print(f"    thr={thr}: {' > '.join(ranking)}")

    # Table 3: Dark matter counts
    print(f"\n  Dark Matter Counts:")
    print(f"  {'Threshold':<12s} {'DM Pairs':>10s} {'Spearman vs 700':>18s}")
    print(f"  {'-'*42}")
    for thr in THRESHOLDS:
        dm = all_results[thr]["dark_matter_count"]
        rho = all_results[thr].get("spearman_vs_700")
        rho_str = f"{rho:.4f}" if rho is not None else "N/A"
        print(f"  {thr:<12d} {dm:>10d} {rho_str:>18s}")

    print(f"\n{BANNER}")
    print("Done.")


if __name__ == "__main__":
    main()
