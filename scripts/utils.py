#!/usr/bin/env python3
"""
G-F Consistency Framework — Core Utilities
==========================================
Shared constants, data I/O, spatial graph construction, functional purity,
G-F curve/score computation, centrality features, coordinate rescaling,
and reusable embedding primitives.

All pipeline scripts import from this module.
"""

from __future__ import annotations

import json
import gzip
import random
import logging
from typing import Optional

import numpy as np
import networkx as nx
from collections import Counter
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import trapezoid
from networkx.algorithms.community import greedy_modularity_communities, modularity

# ============================================================
# Project-wide Constants
# ============================================================

SEED: int = 42


def set_seed(new_seed: int) -> None:
    """Update the global SEED and re-seed all RNGs.

    Call this once at the start of a pipeline run so that every
    downstream function that reads ``SEED`` (or uses ``np.random`` /
    ``random``) honours the user-requested seed.
    """
    global SEED
    SEED = new_seed
    random.seed(new_seed)
    np.random.seed(new_seed)
    try:
        import torch
        torch.manual_seed(new_seed)
    except ImportError:
        pass


# Integration interval for G-F Score (paper default)
GF_R_MIN: float = 0.05
GF_R_MAX: float = 0.422

# Sampling grid for G-F curves
R_MIN: float = 0.05
R_MAX: float = 0.55
N_POINTS: int = 200

# Embedding standardisation target
TARGET_STD: float = 0.3

# Standardised method lists (single source of truth)
CLASSICAL_METHODS: list[str] = [
    "DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE",
]
ALL_CURATED_METHODS: list[str] = [
    "DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE", "PCA", "VGAE-feat",
]
GNN_METHODS: list[str] = ["GraphSAGE", "GAT", "GIN"]
ALL_METHODS: list[str] = ALL_CURATED_METHODS + GNN_METHODS

# Evaluation constants
CV_FOLDS: int = 5
K_NEIGHBORS: int = 5
MIN_LABEL_COUNT: int = 3
STRING_MIN_SCORE: int = 700

# Plateau detection (relative threshold: fraction of peak purity)
PLATEAU_RELATIVE_THRESHOLD: float = 0.80

# ============================================================
# Shared display constants (single source of truth)
# ============================================================
# Unified banner string for all scripts (was inconsistent: some used 64, some 70)
BANNER: str = "=" * 70

# Okabe-Ito colourblind-safe palette for all 11 embedding methods
METHOD_COLORS: dict[str, str] = {
    "Spectral": "#E69F00", "DM": "#0072B2", "MDS": "#009E73",
    "Node2Vec": "#CC79A7", "PCA": "#56B4E9", "VGAE-feat": "#F0E442",
    "DeepWalk": "#D55E00", "GIN": "#949494", "GAT": "#000000",
    "GraphSAGE": "#8B4513", "VGAE": "#808080",
}

# Baseline colours for function prediction comparisons
BASELINE_COLORS: dict[str, str] = {
    "PPI-Neighbors": "#636363", "2-Hop Diffusion": "#969696", "Random": "#d9d9d9",
}

# ============================================================
# JSON Serialization (numpy-safe)
# ============================================================

def json_default(obj):
    """JSON serializer for numpy types.

    Usage::

        from utils import json_default
        json.dump(data, f, indent=2, default=json_default)
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_json(data, path, **kwargs):
    """Write JSON with numpy-safe serialization and UTF-8 encoding.

    Wraps json.dump with default=json_default, indent=2, ensure_ascii=False
    unless overridden by **kwargs.
    """
    kwargs.setdefault("default", json_default)
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("ensure_ascii", False)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)

# ============================================================
# Directory Helpers
# ============================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_data_dir() -> Path:
    return get_project_root() / "data"

def get_results_dir() -> Path:
    return get_project_root() / "results"

def get_figures_dir() -> Path:
    return get_project_root() / "figures"

def get_embeddings_dir() -> Path:
    return get_project_root() / "embeddings"

# ============================================================
# Data Loading
# ============================================================

def load_curated_network(data_dir: Optional[Path] = None):
    """Load the curated 153-node yeast PPI network with GO annotations.

    Returns
    -------
    tuple of (nx.Graph, list[str], dict)
        ``(graph, sorted_nodes, gene_go_map)``
    """
    if data_dir is None:
        data_dir = get_data_dir()
    data_dir = Path(data_dir)

    G = nx.Graph()
    edgelist_file = data_dir / "curated_153_ppi.edgelist"
    if not edgelist_file.exists():
        edgelist_file = data_dir / "yeast_ppi_final_clean.edgelist"
    with open(edgelist_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])

    with open(data_dir / "gene_go_map.json", encoding="utf-8") as f:
        go_map = json.load(f)

    valid = sorted(set(G.nodes()) & set(go_map.keys()))
    G = G.subgraph(valid).copy()
    nodes = sorted(G.nodes())
    return G, nodes, go_map


def load_full_STRING_network(data_dir: Optional[Path] = None,
                             min_score: int = STRING_MIN_SCORE) -> nx.Graph:
    """Load the full yeast STRING v11.5 network (score >= *min_score*)."""
    if data_dir is None:
        data_dir = get_data_dir()
    data_dir = Path(data_dir)
    string_file = data_dir / "4932.protein.links.v11.5.txt.gz"
    if not string_file.exists():
        raise FileNotFoundError(f"STRING file not found: {string_file}")

    G = nx.Graph()
    with gzip.open(str(string_file), "rt", encoding="utf-8") as f:
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


# ============================================================
# Embedding I/O
# ============================================================

def load_embedding(method: str, subset: str = "153",
                   embeddings_dir: Optional[Path] = None):
    """Load a pre-computed embedding (.npy + _nodes.json)."""
    if embeddings_dir is None:
        embeddings_dir = get_embeddings_dir()
    embeddings_dir = Path(embeddings_dir)

    npy_file = embeddings_dir / f"{method}_{subset}.npy"
    nodes_file = embeddings_dir / f"{method}_{subset}_nodes.json"

    if npy_file.exists():
        coords = np.load(npy_file)
        if nodes_file.exists():
            with open(nodes_file, encoding="utf-8") as f:
                nodes = json.load(f)
        else:
            nodes = [str(i) for i in range(len(coords))]
        return coords, nodes

    # Fallback: JSON format in data dir
    data_dir = get_data_dir()
    json_file = data_dir / f"embeddings_{method.lower()}.json"
    if json_file.exists():
        with open(json_file, encoding="utf-8") as f:
            pos = json.load(f)
        nodes = sorted(pos.keys())
        coords = np.array([pos[n] for n in nodes])
        return coords, nodes

    raise FileNotFoundError(f"No embedding found for {method}_{subset}")


def save_embedding(coords: np.ndarray, nodes: list, method: str,
                   subset: str = "153",
                   embeddings_dir: Optional[Path] = None) -> None:
    """Save embedding coordinates and node labels."""
    if embeddings_dir is None:
        embeddings_dir = get_embeddings_dir()
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / f"{method}_{subset}.npy", coords)
    with open(embeddings_dir / f"{method}_{subset}_nodes.json", "w") as f:
        json.dump(nodes, f)


# ============================================================
# Node Alignment
# ============================================================

def align_embedding_to_nodes(coords: np.ndarray, emb_nodes: list,
                             target_nodes: list) -> tuple[np.ndarray, list]:
    """Align embedding coordinates to a target node list.

    Returns the subset of coordinates matching *target_nodes* and the
    list of common nodes actually used.
    """
    node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
    common = [n for n in target_nodes if n in node_to_idx]
    indices = [node_to_idx[n] for n in common]
    return coords[indices], common


# ============================================================
# Spatial Graph Construction (Optimised)
# ============================================================

def precompute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix using scipy pdist.

    Uses only the upper triangle (O(n^2/2) memory) and symmetrises.
    """
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, n))
    return squareform(pdist(coords, metric="euclidean"))


def build_spatial_graph_fast(dist_matrix: np.ndarray, r: float) -> nx.Graph:
    """Build a spatial graph: edge (i, j) iff 0 < D[i,j] < r."""
    n = dist_matrix.shape[0]
    G_r = nx.Graph()
    G_r.add_nodes_from(range(n))
    mask = (dist_matrix < r) & (dist_matrix > 0)
    rows, cols = np.where(mask)
    # Only add upper-triangle edges to avoid duplicates
    valid = rows < cols
    edges = list(zip(rows[valid].tolist(), cols[valid].tolist()))
    G_r.add_edges_from(edges)
    return G_r


# ============================================================
# Functional Purity (FIXED: denominator = total GO terms)
# ============================================================

def _community_purity(comm_node_names: list, go_map: dict) -> float:
    """Compute functional purity for a single community.

    Purity = (count of most-common GO term) / (total GO terms in community).
    Returns 0.0 if no GO annotations are found.
    Guarantees purity in [0, 1].
    """
    go_terms: list[str] = []
    for node_name in comm_node_names:
        if node_name in go_map:
            go_terms.extend(go_map[node_name])
    if not go_terms:
        return 0.0
    term_counts = Counter(go_terms)
    most_common_count = term_counts.most_common(1)[0][1]
    total_terms = len(go_terms)
    return most_common_count / total_terms


def functional_purity(communities, go_map: dict, nodes: list) -> float:
    """Mean functional purity across all communities (index-based nodes)."""
    purities = []
    for comm in communities:
        if not comm:
            continue
        comm_list = list(comm)
        comm_names = [nodes[idx] for idx in comm_list]
        purities.append(_community_purity(comm_names, go_map))
    return float(np.mean(purities)) if purities else 0.0


def functional_purity_named(communities, go_map: dict) -> float:
    """Mean functional purity across all communities (named nodes)."""
    purities = []
    for comm in communities:
        if not comm:
            continue
        purities.append(_community_purity(list(comm), go_map))
    return float(np.mean(purities)) if purities else 0.0


# ============================================================
# G-F Curve Computation
# ============================================================

def compute_gf_curve(coords: np.ndarray, nodes: list, go_map: dict,
                     r_vals: np.ndarray) -> tuple[list[float], list[float]]:
    """Compute the G-F purity and modularity curves.

    Optimised implementation:
    - Edges are sorted by distance once; graphs are built incrementally.
    - Community detection results are cached when the graph structure
      (edge count) has not changed between consecutive *r* thresholds.
    - Numerical output is **identical** to the naive per-r rebuild.

    Parameters
    ----------
    coords : (n, d) embedding coordinates
    nodes : ordered node labels
    go_map : gene -> [GO terms]
    r_vals : distance thresholds to evaluate

    Returns
    -------
    (purities, modularities) — two parallel lists of floats
    """
    dist_matrix = precompute_distance_matrix(coords)
    n = dist_matrix.shape[0]

    # Pre-sort all unique upper-triangle edges by distance
    iu = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[iu]
    sort_idx = np.argsort(edge_dists)
    sorted_rows = iu[0][sort_idx]
    sorted_cols = iu[1][sort_idx]
    sorted_d = edge_dists[sort_idx]

    # Process r values in ascending order; map back to original order
    r_order = np.argsort(r_vals)
    purities_out: list[float] = [0.0] * len(r_vals)
    mods_out: list[float] = [0.0] * len(r_vals)

    G_r = nx.Graph()
    G_r.add_nodes_from(range(n))
    edge_ptr = 0
    n_edges_total = len(sorted_d)

    # Cache: keyed by edge count (graph structure only changes when edges are added)
    _cache: dict[int, tuple[list, float]] = {}  # n_edges -> (communities, mod)

    for rank, orig_idx in enumerate(r_order):
        r = float(r_vals[orig_idx])

        # Incrementally add edges with distance < r
        while edge_ptr < n_edges_total and sorted_d[edge_ptr] < r:
            G_r.add_edge(int(sorted_rows[edge_ptr]), int(sorted_cols[edge_ptr]))
            edge_ptr += 1

        ne = G_r.number_of_edges()
        if ne == 0:
            # purities_out / mods_out already initialised to 0.0
            continue

        if ne in _cache:
            communities, mod_val = _cache[ne]
        else:
            communities = list(greedy_modularity_communities(G_r))
            if len(communities) > 1:
                mod_val = modularity(G_r, communities)
            else:
                mod_val = 0.0
            _cache[ne] = (communities, mod_val)

        purities_out[orig_idx] = functional_purity(communities, go_map, nodes)
        mods_out[orig_idx] = mod_val

    return purities_out, mods_out


# ============================================================
# G-F Score
# ============================================================

def compute_gf_score(r_vals, purity_vals,
                     r_min: float = GF_R_MIN,
                     r_max: float = GF_R_MAX) -> float:
    """Compute the G-F Score as the mean purity over [r_min, r_max].

    Uses the trapezoidal rule for numerical integration.
    """
    r = np.asarray(r_vals)
    p = np.asarray(purity_vals)
    mask = (r >= r_min) & (r <= r_max)
    r_sub = r[mask]
    p_sub = p[mask]
    if len(r_sub) < 2:
        return 0.0
    return float(trapezoid(p_sub, r_sub) / (r_max - r_min))


# ============================================================
# kNN-GF Score: High-dimensional generalisation
# ============================================================
# The standard G-F Score uses Euclidean distance thresholds to build
# spatial graphs. In high dimensions, distances concentrate (all pairwise
# distances become similar), causing the spatial graph to degenerate.
# The kNN-GF Score replaces distance thresholds with k-nearest-neighbour
# graphs, which depend only on distance *rankings* (order statistics),
# not absolute distances. This makes the score invariant to the curse
# of dimensionality.
#
# See: Theorem 5 (Concentration Invariance) in Supplementary Materials.

KNN_DEFAULT_RANGE: tuple[int, int] = (3, 30)


def build_knn_graph(dist_matrix: np.ndarray, k: int) -> nx.Graph:
    """Build a symmetric k-nearest-neighbour graph from a distance matrix.

    Parameters
    ----------
    dist_matrix : (n, n) pairwise distances
    k : number of nearest neighbours per node

    Returns
    -------
    nx.Graph with n nodes and edges connecting each node to its k
    nearest neighbours (union of both directions).
    """
    n = dist_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    k = min(k, n - 1)

    for i in range(n):
        # argsort gives sorted indices; skip self (index 0 after sort)
        neighbours = np.argsort(dist_matrix[i])[1:k + 1]
        for j in neighbours:
            j = int(j)
            if not G.has_edge(i, j):
                G.add_edge(i, j)
    return G


def compute_gf_curve_knn(
    coords: np.ndarray, nodes: list, go_map: dict,
    k_vals: Optional[list[int]] = None,
) -> tuple[list[float], list[int]]:
    """Compute the kNN-GF purity curve over a range of k values.

    For each k, builds a kNN graph, runs community detection, and computes
    functional purity. This is the high-dimensional generalisation of
    compute_gf_curve — it depends only on distance rankings, not absolute
    distances, making it robust to the curse of dimensionality.

    Parameters
    ----------
    coords : (n, d) embedding coordinates (any dimension d)
    nodes : ordered node labels
    go_map : gene -> [GO terms]
    k_vals : list of k values to evaluate (default: 3..30)

    Returns
    -------
    (purities, k_values) — parallel lists
    """
    if k_vals is None:
        k_vals = list(range(KNN_DEFAULT_RANGE[0], KNN_DEFAULT_RANGE[1] + 1))

    dist_matrix = precompute_distance_matrix(coords)
    n = dist_matrix.shape[0]

    purities: list[float] = []
    for k in k_vals:
        k_eff = min(k, n - 1)
        G_k = build_knn_graph(dist_matrix, k_eff)
        if G_k.number_of_edges() == 0:
            purities.append(0.0)
            continue
        communities = list(greedy_modularity_communities(G_k))
        purities.append(functional_purity(communities, go_map, nodes))
    return purities, list(k_vals)


def compute_knn_gf_score(
    purities: list[float], k_vals: list[int],
    k_min: int = KNN_DEFAULT_RANGE[0],
    k_max: int = KNN_DEFAULT_RANGE[1],
) -> float:
    """Compute the kNN-GF Score as mean purity over [k_min, k_max].

    Unlike the Euclidean G-F Score (which uses trapezoidal integration
    over continuous r), the kNN-GF Score uses simple averaging over
    discrete k values, since k is an integer parameter.
    """
    k_arr = np.asarray(k_vals)
    p_arr = np.asarray(purities)
    mask = (k_arr >= k_min) & (k_arr <= k_max)
    p_sub = p_arr[mask]
    if len(p_sub) == 0:
        return 0.0
    return float(np.mean(p_sub))


# ============================================================
# Centrality Features
# ============================================================

def compute_centrality_features(G: nx.Graph,
                                nodes: Optional[list] = None) -> np.ndarray:
    """Compute 6 normalised centrality features for each node.

    Features: degree, eigenvector, PageRank, clustering,
    average neighbour degree, core number.
    """
    if nodes is None:
        nodes = list(G.nodes())
    n = len(nodes)
    deg = nx.degree_centrality(G)
    eig = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-5)
    pr = nx.pagerank(G)
    clust = nx.clustering(G)
    avg_deg = nx.average_neighbor_degree(G)
    kcore = nx.core_number(G)

    features = np.zeros((n, 6))
    for i, u in enumerate(nodes):
        features[i, 0] = deg[u]
        features[i, 1] = eig[u]
        features[i, 2] = pr[u]
        features[i, 3] = clust[u]
        features[i, 4] = avg_deg[u]
        features[i, 5] = kcore[u]

    norms = np.linalg.norm(features, axis=0)
    norms[norms < 1e-10] = 1.0
    features = features / norms
    return features


# ============================================================
# Coordinate Rescaling
# ============================================================

def rescale_coordinates(coords: np.ndarray,
                        target_std: float = TARGET_STD) -> np.ndarray:
    """Rescale embedding coordinates to a target global standard deviation.

    Note: this is NOT z-score standardisation (mean is not centred).
    It only adjusts the scale so that all methods operate on a comparable
    distance regime.
    """
    current_std = np.std(coords)
    if current_std < 1e-10:
        return coords
    return coords / current_std * target_std


def check_embedding_collapse(coords: np.ndarray,
                             method_name: str = "") -> dict:
    """Detect collapsed embeddings via pairwise distance statistics.

    Returns a diagnostic dict with ``collapsed`` (bool) and details.
    """
    dists = pdist(coords)
    if len(dists) == 0:
        return {"collapsed": True, "method": method_name,
                "reason": "empty or single-point embedding"}

    median_d = float(np.median(dists))
    mean_d = float(np.mean(dists))
    std_d = float(np.std(dists))
    cv = std_d / mean_d if mean_d > 1e-10 else 0.0

    collapsed = False
    reasons: list[str] = []
    if median_d < 1e-6:
        collapsed = True
        reasons.append(f"median distance {median_d:.2e} ≈ 0 (point collapse)")
    if cv < 0.01 and len(dists) > 10:
        collapsed = True
        reasons.append(f"distance CV = {cv:.4f} < 0.01 (all equidistant)")

    return {
        "collapsed": collapsed,
        "method": method_name,
        "median_dist": median_d,
        "mean_dist": mean_d,
        "cv": cv,
        "reasons": reasons,
    }


def coords_to_dict(coords: np.ndarray, nodes: list) -> dict:
    return {nodes[i]: coords[i].tolist() for i in range(len(nodes))}


# ============================================================
# Plateau Width
# ============================================================

def compute_plateau_width(r_vals, purity_vals,
                          relative_threshold: float = PLATEAU_RELATIVE_THRESHOLD) -> dict:
    """Width of the r-interval where purity >= relative_threshold * peak_purity.

    Uses a *relative* threshold so that the plateau is defined with respect
    to each method's own peak purity rather than a fixed absolute cutoff.

    Returns
    -------
    dict with keys W, r_min, r_max, peak_purity, effective_threshold.
    Returns all-zero dict when no points exceed the threshold.
    """
    r = np.asarray(r_vals, dtype=float)
    p = np.asarray(purity_vals, dtype=float)
    if len(p) == 0:
        return {"W": 0.0, "r_min": 0.0, "r_max": 0.0,
                "peak_purity": 0.0, "effective_threshold": 0.0}

    peak = float(p.max())
    effective_thr = peak * relative_threshold

    if peak <= 0.0 or effective_thr <= 0.0:
        return {"W": 0.0, "r_min": 0.0, "r_max": 0.0,
                "peak_purity": peak, "effective_threshold": effective_thr}

    mask = p >= effective_thr
    if not mask.any():
        return {"W": 0.0, "r_min": 0.0, "r_max": 0.0,
                "peak_purity": peak, "effective_threshold": effective_thr}

    r_plateau = r[mask]
    return {"W": float(r_plateau[-1] - r_plateau[0]),
            "r_min": float(r_plateau[0]),
            "r_max": float(r_plateau[-1]),
            "peak_purity": peak,
            "effective_threshold": float(effective_thr)}


# ============================================================
# Logging
# ============================================================

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ============================================================
# Reusable Embedding Primitives
# ============================================================

def build_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Inner-product similarity from normalised feature matrix."""
    return features @ features.T


def diffusion_map_from_similarity(sim: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Diffusion Map coordinates from a similarity matrix.
    
    Parameters
    ----------
    sim : np.ndarray
        Similarity matrix (n x n).
    n_components : int, default 2
        Number of dimensions to return (eigenvectors 2..n_components+1).
    """
    row_sums = sim.sum(axis=1, keepdims=True)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(row_sums.flatten()) + 1e-10))
    norm_sim = D_inv_sqrt @ sim @ D_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(norm_sim)
    idx = np.argsort(eigvals)
    # Take the 2nd, 3rd, ... (n_components+1)-th largest eigenvectors
    # Skip the largest (constant) eigenvector.
    selected = idx[-(n_components + 1):-1][::-1]
    coords = eigvecs[:, selected]
    return coords


def classical_mds_from_distances(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical MDS from a square distance matrix.
    
    Parameters
    ----------
    D : np.ndarray
        Square distance matrix (n x n).
    n_components : int, default 2
        Number of dimensions to return.
    """
    n = D.shape[0]
    D_sq = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D_sq @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    coords = eigvecs[:, idx[:n_components]] * np.sqrt(np.abs(eigvals[idx[:n_components]]))
    return coords


def spectral_embedding_from_graph(G: nx.Graph,
                                 nodelist=None, n_components: int = 2) -> np.ndarray:
    """Spectral embedding from normalised Laplacian.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph.
    nodelist : list, optional
        Ordering of nodes.
    n_components : int, default 2
        Number of dimensions (eigenvectors starting from Fiedler).
    """
    L = nx.normalized_laplacian_matrix(G, nodelist=nodelist).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvecs[:, 1:1 + n_components]


def deepwalk_from_graph(G: nx.Graph, walk_length: int = 20,
                        walks_per_node: int = 10, window_size: int = 5,
                        dimensions: int = 2, seed: Optional[int] = None) -> np.ndarray:
    """DeepWalk embedding via uniform random walks + SVD."""
    if seed is None:
        seed = SEED
    random.seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    adj = {i: [] for i in range(n)}
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i].append(j)
        adj[j].append(i)

    walks = []
    for start in range(n):
        for _ in range(walks_per_node):
            walk = [start]
            for _ in range(walk_length - 1):
                cur = walk[-1]
                if not adj[cur]:
                    break
                walk.append(np.random.choice(adj[cur]))
            if len(walk) >= 2:
                walks.append(walk)

    if n > 1000:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds

        cooc_dict: dict[tuple[int, int], float] = {}
        for walk in walks:
            for i, ni in enumerate(walk):
                for j in range(max(0, i - window_size),
                               min(len(walk), i + window_size + 1)):
                    if i != j:
                        key = (ni, walk[j])
                        cooc_dict[key] = cooc_dict.get(key, 0) + 1

        rows, cols, vals = [], [], []
        for (i, j), val in cooc_dict.items():
            rows.append(i)
            cols.append(j)
            vals.append(float(val))
        cooc_sparse = csr_matrix((vals, (rows, cols)), shape=(n, n))
        k = min(dimensions + 1, n - 1)
        U, S, _ = svds(cooc_sparse.astype(np.float64), k=k)
        idx = np.argsort(-S)
        U = U[:, idx]
        S = S[idx]
        return U[:, :dimensions] * np.sqrt(S[:dimensions])
    else:
        cooc = np.zeros((n, n))
        for walk in walks:
            for i, ni in enumerate(walk):
                for j in range(max(0, i - window_size),
                               min(len(walk), i + window_size + 1)):
                    if i != j:
                        cooc[ni, walk[j]] += 1
        U, S, _ = np.linalg.svd(cooc, full_matrices=False)
        return U[:, :dimensions] * np.sqrt(S[:dimensions])


def node2vec_from_graph(G: nx.Graph, walk_length: int = 20,
                        walks_per_node: int = 10, window_size: int = 5,
                        dimensions: int = 2, p: float = 0.5, q: float = 2.0,
                        seed: Optional[int] = None) -> np.ndarray:
    """Node2Vec embedding via biased random walks + SVD."""
    if seed is None:
        seed = SEED
    random.seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    adj = {i: [] for i in range(n)}
    adj_sets = {i: set() for i in range(n)}
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i].append(j)
        adj[j].append(i)
        adj_sets[i].add(j)
        adj_sets[j].add(i)

    walks = []
    for start in range(n):
        for _ in range(walks_per_node):
            walk = [start]
            for _ in range(walk_length - 1):
                cur = walk[-1]
                if not adj[cur]:
                    break
                if len(walk) == 1:
                    walk.append(np.random.choice(adj[cur]))
                else:
                    prev = walk[-2]
                    probs = []
                    for nbr in adj[cur]:
                        if nbr == prev:
                            probs.append(1.0 / p)
                        elif nbr in adj_sets[prev]:
                            probs.append(1.0)
                        else:
                            probs.append(1.0 / q)
                    probs = np.array(probs)
                    probs /= probs.sum()
                    walk.append(np.random.choice(adj[cur], p=probs))
            if len(walk) >= 2:
                walks.append(walk)

    if n > 1000:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds

        cooc_dict: dict[tuple[int, int], float] = {}
        for walk in walks:
            for i, ni in enumerate(walk):
                for j in range(max(0, i - window_size),
                               min(len(walk), i + window_size + 1)):
                    if i != j:
                        key = (ni, walk[j])
                        cooc_dict[key] = cooc_dict.get(key, 0) + 1

        rows, cols, vals = [], [], []
        for (i, j), val in cooc_dict.items():
            rows.append(i)
            cols.append(j)
            vals.append(float(val))
        cooc_sparse = csr_matrix((vals, (rows, cols)), shape=(n, n))
        k = min(dimensions + 1, n - 1)
        U, S, _ = svds(cooc_sparse.astype(np.float64), k=k)
        idx = np.argsort(-S)
        U = U[:, idx]
        S = S[idx]
        return U[:, :dimensions] * np.sqrt(S[:dimensions])
    else:
        cooc_matrix = np.zeros((n, n))
        for walk in walks:
            for i, ni in enumerate(walk):
                for j in range(max(0, i - window_size),
                               min(len(walk), i + window_size + 1)):
                    if i != j:
                        cooc_matrix[ni, walk[j]] += 1
        U, S, _ = np.linalg.svd(cooc_matrix, full_matrices=False)
        return U[:, :dimensions] * np.sqrt(S[:dimensions])


def vgae_from_graph(G: nx.Graph, hidden_dim: int = 4, latent_dim: int = 2,
                    epochs: int = 300, lr: float = 0.01,
                    features: Optional[np.ndarray] = None,
                    seed: Optional[int] = None) -> np.ndarray:
    """VGAE embedding (2-D latent) with 2-layer GCN encoder."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import from_networkx

    if seed is None:
        seed = SEED
    torch.manual_seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    n = len(nodes)
    data = from_networkx(G)

    if features is not None:
        data.x = torch.tensor(features, dtype=torch.float32)
        in_dim = features.shape[1]
    else:
        data.x = torch.eye(n)
        in_dim = n

    class Encoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, latent_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv_mu = GCNConv(hidden_dim, latent_dim)
            self.conv_logvar = GCNConv(hidden_dim, latent_dim)

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            return self.conv_mu(h, edge_index), self.conv_logvar(h, edge_index)

    model = Encoder(in_dim, hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    adj_target = torch.zeros(n, n)
    ei = data.edge_index
    adj_target[ei[0], ei[1]] = 1.0

    for _ in range(epochs):
        optimizer.zero_grad()
        mu, logvar = model(data.x, data.edge_index)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        adj_recon = torch.sigmoid(z @ z.T)
        recon_loss = F.binary_cross_entropy(adj_recon, adj_target, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        mu, _ = model(data.x, data.edge_index)
        coords = mu.numpy()
    return coords
