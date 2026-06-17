#!/usr/bin/env python3
"""
umap_tsne_gf.py
===============
Compute UMAP and t-SNE 2D embeddings for the yeast PPI network and evaluate
their G-F Scores, comparing against the existing 11 embedding methods.

Two evaluation modes:
  1. Curated 153-node network  (directly comparable to gf_scores.json)
  2. Full 5936-node STRING network (comparable to full_network_validation.json)

Usage:
    python scripts/umap_tsne_gf.py
"""

import sys
import json
import time
import random
import warnings
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, TARGET_STD, GF_R_MIN, GF_R_MAX,
    R_MIN, R_MAX, N_POINTS,
    get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_full_STRING_network,
    compute_gf_curve, compute_gf_score, rescale_coordinates,
    save_embedding, precompute_distance_matrix,
)

# ============================================================
# Dependency check
# ============================================================

def check_dependencies():
    """Check that umap-learn and sklearn are available."""
    missing = []
    try:
        import umap  # noqa: F401
    except ImportError:
        missing.append("umap-learn")
    try:
        from sklearn.manifold import TSNE  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        sys.exit(1)
    import umap
    from sklearn.manifold import TSNE
    print(f"  umap-learn version: {umap.__version__}")
    print(f"  sklearn TSNE: available")
    return umap, TSNE


# ============================================================
# Embedding Functions
# ============================================================

def compute_shortest_path_distances(G, nodes):
    """Compute all-pairs shortest path distances using sparse Dijkstra."""
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    row, col, data = [], [], []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            row.extend([i, j])
            col.extend([j, i])
            data.extend([1.0, 1.0])

    adj_sparse = csr_matrix((data, (row, col)), shape=(n, n))
    print(f"    Computing shortest-path distances ({n} nodes)...")
    t0 = time.time()
    D = shortest_path(adj_sparse, directed=False, unweighted=True)
    D[np.isinf(D)] = n  # cap disconnected pairs
    print(f"    Shortest-path distances: {time.time() - t0:.1f}s")
    return D


def build_adjacency_features(G, nodes):
    """Build dense adjacency matrix rows as node features."""
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    adj = np.zeros((n, n))
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


def embed_umap_precomputed(D, n_neighbors=15, min_dist=0.1, seed=SEED):
    """UMAP 2D embedding using precomputed distance matrix."""
    import umap
    np.random.seed(seed)
    n_neighbors = min(n_neighbors, D.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=seed,
        verbose=False,
    )
    coords = reducer.fit_transform(D)
    return coords


def embed_umap_euclidean(X, n_neighbors=15, min_dist=0.1, seed=SEED):
    """UMAP 2D embedding using euclidean metric on feature matrix."""
    import umap
    np.random.seed(seed)
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        verbose=False,
    )
    coords = reducer.fit_transform(X)
    return coords


def embed_tsne(X, perplexity=30, seed=SEED):
    """t-SNE 2D embedding on feature matrix."""
    from sklearn.manifold import TSNE
    import sklearn
    np.random.seed(seed)
    perplexity = min(perplexity, max(5, X.shape[0] // 3))
    # sklearn >= 1.2 renamed n_iter to max_iter
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        verbose=0,
    )
    if hasattr(sklearn, "__version__") and tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 2):
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    reducer = TSNE(**tsne_kwargs)
    coords = reducer.fit_transform(X)
    return coords


def embed_tsne_precomputed(D, perplexity=30, seed=SEED):
    """t-SNE 2D embedding using precomputed distance matrix."""
    from sklearn.manifold import TSNE
    import sklearn
    np.random.seed(seed)
    perplexity = min(perplexity, max(5, D.shape[0] // 3))
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        metric="precomputed",
        random_state=seed,
        init="random",
        learning_rate="auto",
        verbose=0,
    )
    if hasattr(sklearn, "__version__") and tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 2):
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    reducer = TSNE(**tsne_kwargs)
    coords = reducer.fit_transform(D)
    return coords


# ============================================================
# Evaluation
# ============================================================

def evaluate_gf(coords, nodes, go_map, label="Method"):
    """Rescale, compute GF curve and score, return dict."""
    coords = rescale_coordinates(coords, TARGET_STD)
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    purities, modularities = compute_gf_curve(coords, nodes, go_map, r_vals)
    gf_score = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)
    print(f"    {label}: GF Score = {gf_score:.4f}, "
          f"purity range = [{min(purities):.3f}, {max(purities):.3f}]")
    return {
        "gf_score": gf_score,
        "r_vals": r_vals.tolist(),
        "purities": purities,
        "modularities": modularities,
    }


def load_existing_gf_scores():
    """Load all 11 existing method GF scores for comparison."""
    results_dir = get_results_dir()
    scores = {}

    # 8 curated methods from gf_scores.json
    gf_file = results_dir / "gf_scores.json"
    if gf_file.exists():
        with open(gf_file, encoding="utf-8") as f:
            data = json.load(f)
        for method, score in data.get("scores", {}).items():
            scores[method] = score

    # 3 GNN methods from gnn_gf_scores.json
    gnn_file = results_dir / "gnn_gf_scores.json"
    if gnn_file.exists():
        with open(gnn_file, encoding="utf-8") as f:
            data = json.load(f)
        for method, score in data.get("gf_scores", {}).items():
            scores[method] = score

    return scores


# ============================================================
# Part 1: Curated 153-node Network
# ============================================================

def run_curated_153():
    """Compute UMAP and t-SNE on the curated 153-node network."""
    print("\n" + "=" * 70)
    print("PART 1: Curated 153-node Yeast PPI Network")
    print("=" * 70)

    data_dir = get_data_dir()
    emb_dir = get_embeddings_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Load network
    print("\nLoading curated 153-node network...")
    G, nodes, go_map = load_curated_network(data_dir)
    n = len(nodes)
    print(f"  Network: {n} nodes, {G.number_of_edges()} edges")
    annotated = [nd for nd in nodes if nd in go_map]
    print(f"  GO-annotated: {len(annotated)} nodes")

    # Compute shortest-path distances
    D = compute_shortest_path_distances(G, nodes)

    # Build adjacency features
    adj = build_adjacency_features(G, nodes)

    results_153 = {}

    # --- UMAP (precomputed shortest-path) ---
    print("\n  [UMAP] Computing on precomputed shortest-path distances...")
    t0 = time.time()
    coords_umap_sp = embed_umap_precomputed(D, n_neighbors=15, min_dist=0.1)
    print(f"    UMAP (SP): {time.time() - t0:.1f}s, shape={coords_umap_sp.shape}")
    save_embedding(coords_umap_sp, nodes, "UMAP", "153", emb_dir)
    results_153["UMAP"] = evaluate_gf(coords_umap_sp, nodes, go_map, "UMAP (SP)")

    # --- UMAP (euclidean on adjacency) ---
    print("\n  [UMAP] Computing on adjacency features (euclidean)...")
    t0 = time.time()
    coords_umap_adj = embed_umap_euclidean(adj, n_neighbors=15, min_dist=0.1)
    print(f"    UMAP (adj): {time.time() - t0:.1f}s, shape={coords_umap_adj.shape}")
    save_embedding(coords_umap_adj, nodes, "UMAP-adj", "153", emb_dir)
    results_153["UMAP-adj"] = evaluate_gf(coords_umap_adj, nodes, go_map, "UMAP (adj)")

    # --- t-SNE (on adjacency features) ---
    print("\n  [t-SNE] Computing on adjacency features...")
    t0 = time.time()
    coords_tsne_adj = embed_tsne(adj, perplexity=30)
    print(f"    t-SNE (adj): {time.time() - t0:.1f}s, shape={coords_tsne_adj.shape}")
    save_embedding(coords_tsne_adj, nodes, "TSNE", "153", emb_dir)
    results_153["TSNE"] = evaluate_gf(coords_tsne_adj, nodes, go_map, "t-SNE (adj)")

    # --- t-SNE (precomputed shortest-path) ---
    print("\n  [t-SNE] Computing on precomputed shortest-path distances...")
    t0 = time.time()
    coords_tsne_sp = embed_tsne_precomputed(D, perplexity=30)
    print(f"    t-SNE (SP): {time.time() - t0:.1f}s, shape={coords_tsne_sp.shape}")
    save_embedding(coords_tsne_sp, nodes, "TSNE-sp", "153", emb_dir)
    results_153["TSNE-sp"] = evaluate_gf(coords_tsne_sp, nodes, go_map, "t-SNE (SP)")

    return results_153


# ============================================================
# Part 2: Full 5936-node STRING Network
# ============================================================

def run_full_network():
    """Compute UMAP and t-SNE on the full STRING network."""
    print("\n" + "=" * 70)
    print("PART 2: Full 5936-node STRING Network")
    print("=" * 70)

    data_dir = get_data_dir()
    emb_dir = get_embeddings_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Load full network
    print("\nLoading full STRING network (score >= 700)...")
    t0 = time.time()
    G = load_full_STRING_network(data_dir)
    nodes = sorted(G.nodes())
    n = len(nodes)
    m = G.number_of_edges()
    print(f"  Network: {n} nodes, {m} edges ({time.time() - t0:.1f}s)")

    # Load GO annotations
    with open(data_dir / "gene_go_map.json", encoding="utf-8") as f:
        go_map = json.load(f)
    annotated = sorted(set(go_map.keys()) & set(nodes))
    print(f"  GO-annotated subset: {len(annotated)} nodes")

    # Shortest-path distances for full network
    print("\n  Computing shortest-path distances for full network...")
    print(f"    (This is O(n * (n+m)) via BFS from all nodes, ~{n} nodes)")
    t_sp = time.time()

    # Use sparse shortest_path (Dijkstra)
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    row, col, data = [], [], []
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        row.extend([i, j])
        col.extend([j, i])
        data.extend([1.0, 1.0])
    adj_sparse = csr_matrix((data, (row, col)), shape=(n, n))

    D_full = shortest_path(adj_sparse, directed=False, unweighted=True)
    D_full[np.isinf(D_full)] = n
    print(f"    Shortest-path distances: {time.time() - t_sp:.1f}s")

    results_full = {}

    # --- UMAP on full network (precomputed) ---
    print("\n  [UMAP] Full network with precomputed distances...")
    t0 = time.time()
    coords_umap_full = embed_umap_precomputed(D_full, n_neighbors=15, min_dist=0.1)
    print(f"    UMAP full: {time.time() - t0:.1f}s, shape={coords_umap_full.shape}")
    save_embedding(coords_umap_full, nodes, "UMAP", "full", emb_dir)

    # Evaluate on annotated subset
    full_map = {nd: i for i, nd in enumerate(nodes)}
    ann_idx = [full_map[nd] for nd in annotated]
    subset_coords = coords_umap_full[ann_idx]
    results_full["UMAP"] = evaluate_gf(subset_coords, annotated, go_map, "UMAP (full)")

    # --- t-SNE on full network ---
    # For t-SNE with 5936 nodes, use adjacency features + PCA reduction
    print("\n  [t-SNE] Full network with adjacency features...")
    t0 = time.time()
    # Build sparse adjacency and convert to dense for TSNE
    adj_full = adj_sparse.toarray()
    # Use PCA to reduce to 50 dims first for faster t-SNE
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=SEED)
    adj_pca = pca.fit_transform(adj_full)
    print(f"    PCA(adj): {time.time() - t0:.1f}s, explained_var={pca.explained_variance_ratio_.sum():.3f}")

    t0 = time.time()
    coords_tsne_full = embed_tsne(adj_pca, perplexity=30)
    print(f"    t-SNE full: {time.time() - t0:.1f}s, shape={coords_tsne_full.shape}")
    save_embedding(coords_tsne_full, nodes, "TSNE", "full", emb_dir)

    subset_coords_tsne = coords_tsne_full[ann_idx]
    results_full["TSNE"] = evaluate_gf(subset_coords_tsne, annotated, go_map, "t-SNE (full)")

    return results_full


# ============================================================
# Part 3: Comparison & Results
# ============================================================

def print_comparison_table(results_153, results_full):
    """Print a comparison table of all methods."""
    print("\n" + "=" * 70)
    print("RESULTS: G-F Score Comparison (all methods)")
    print("=" * 70)

    existing = load_existing_gf_scores()

    print("\n--- Curated 153-node Network GF Scores ---")
    print(f"  {'Rank':<5} {'Method':<15} {'GF Score':>10}  {'Notes'}")
    print(f"  {'-'*55}")

    # Combine existing + new (153-node)
    all_153 = dict(existing)
    new_methods_153 = {}
    for name, res in results_153.items():
        all_153[name] = res["gf_score"]
        new_methods_153[name] = res["gf_score"]

    ranked = sorted(all_153.items(), key=lambda x: x[1], reverse=True)
    for rank, (method, score) in enumerate(ranked, 1):
        marker = " <-- NEW" if method in new_methods_153 else ""
        print(f"  {rank:<5} {method:<15} {score:>10.4f}  {marker}")

    print(f"\n  Random baseline: ~0.1348")

    # Print specific new method scores
    print("\n--- New Methods Summary (153-node) ---")
    for name in results_153:
        score = results_153[name]["gf_score"]
        rank_pos = next(i for i, (m, _) in enumerate(ranked, 1) if m == name)
        print(f"  {name:<15}: GF Score = {score:.4f} (rank {rank_pos}/{len(ranked)})")

    # Full network results
    print("\n--- Full 5936-node Network GF Scores (UMAP/t-SNE only) ---")
    for name, res in results_full.items():
        print(f"  {name:<15}: GF Score = {res['gf_score']:.4f}")

    return all_153, ranked


def save_results(results_153, results_full, all_153, ranked):
    """Save results to JSON."""
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "analysis": "UMAP and t-SNE G-F Score Evaluation",
        "unified_interval": [GF_R_MIN, GF_R_MAX],
        "curated_153": {},
        "full_network": {},
        "comparison": {
            "all_methods_ranked": [
                {"rank": i, "method": m, "gf_score": round(s, 6)}
                for i, (m, s) in enumerate(ranked, 1)
            ],
        },
    }

    for name, res in results_153.items():
        output["curated_153"][name] = {
            "gf_score": round(res["gf_score"], 6),
            "purities": res["purities"],
            "modularities": res["modularities"],
        }

    for name, res in results_full.items():
        output["full_network"][name] = {
            "gf_score": round(res["gf_score"], 6),
            "purities": res["purities"],
            "modularities": res["modularities"],
        }

    output_file = results_dir / "umap_tsne_gf_scores.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {output_file}")


# ============================================================
# Main
# ============================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("UMAP & t-SNE G-F Score Evaluation")
    print("Yeast PPI Network (STRING v11.5, score >= 700)")
    print("=" * 70)

    # Check dependencies
    print("\nChecking dependencies...")
    check_dependencies()

    # Part 1: Curated 153-node network
    results_153 = run_curated_153()

    # Part 2: Full 5936-node network
    results_full = run_full_network()

    # Part 3: Comparison
    all_153, ranked = print_comparison_table(results_153, results_full)

    # Save
    save_results(results_153, results_full, all_153, ranked)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
