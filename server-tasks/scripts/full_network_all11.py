#!/usr/bin/env python3
"""
full_network_all11.py -- Full-Network G-F for All 11 Methods (Yeast 5936)
==========================================================================
Computes G-F Scores for all 11 embedding methods on the full yeast STRING
network (5936 nodes), not just the curated 153-node subset.

Currently, only 4 methods (DM, MDS, Node2Vec, VGAE) have full-network
embeddings. This script computes the remaining 7 (Spectral, DeepWalk,
PCA, VGAE-feat, GraphSAGE, GAT, GIN) and evaluates G-F on all 11.

Uses the fast server-tasks scripts for embedding computation.

Output: results/full_network_all11.json
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import numpy as np
import networkx as nx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, set_seed, ALL_METHODS,
    load_full_STRING_network, rescale_coordinates,
    compute_gf_curve, compute_gf_score,
    GF_R_MIN, GF_R_MAX, N_POINTS,
    get_results_dir, get_data_dir, get_embeddings_dir,
    spectral_embedding_from_graph, deepwalk_from_graph,
    node2vec_from_graph, vgae_from_graph,
    compute_centrality_features,
    precompute_distance_matrix,
)

set_seed(SEED)


def main():
    t_start = time.time()
    print("=" * 72)
    print("  Full-Network 11-Method G-F (Yeast 5936)")
    print("=" * 72)
    print()

    data_dir = get_data_dir()
    embeddings_dir = get_embeddings_dir()
    results_dir = get_results_dir()

    # ----------------------------------------------------------------
    # Load network
    # ----------------------------------------------------------------
    print("[1/4] Loading full yeast STRING network ...")
    G = load_full_STRING_network()
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    print(f"  Network: {n} nodes, {G.number_of_edges()} edges")

    # Load GO map - prefer full GO map (5909 genes), fall back to curated (154)
    go_file = data_dir / "gene_go_map_full.json"
    if not go_file.exists():
        go_file = data_dir / "gene_go_map.json"
        print(f"  WARNING: gene_go_map_full.json not found, using {go_file.name}")
    with open(go_file, encoding="utf-8") as f:
        go_map_full = json.load(f)

    # Filter to nodes that have GO annotations
    annotated = [nd for nd in nodes if nd in go_map_full]
    print(f"  GO-annotated in full network: {len(annotated)}")

    # Use full network for embedding, subsample for G-F
    # (G-F on 5936 nodes with greedy_modularity is too slow)
    SUBSAMPLE_GF = 1000
    rng = np.random.RandomState(SEED)
    if len(annotated) > SUBSAMPLE_GF:
        gf_nodes = sorted(rng.choice(annotated, SUBSAMPLE_GF, replace=False))
    else:
        gf_nodes = annotated
    print(f"  G-F subsample: {len(gf_nodes)} nodes")
    print()

    # ----------------------------------------------------------------
    # Load or compute embeddings for all 11 methods
    # ----------------------------------------------------------------
    print("[2/4] Loading/computing embeddings ...")

    embeddings = {}

    # Try loading existing full-network embeddings
    for method in ALL_METHODS:
        npy = embeddings_dir / f"{method}_full.npy"
        nodes_f = embeddings_dir / f"{method}_full_nodes.json"
        if npy.exists():
            coords = np.load(npy)
            with open(nodes_f, encoding="utf-8") as f:
                emb_nodes = json.load(f)
            embeddings[method] = (coords, emb_nodes)
            print(f"  {method:14s}: loaded ({coords.shape})")

    # Compute missing embeddings
    # For large networks, use sparse/fast methods
    centrality_features = None

    for method in ALL_METHODS:
        if method in embeddings:
            continue

        print(f"  {method:14s}: computing ...", end=" ", flush=True)
        t0 = time.time()

        try:
            if method == "Spectral":
                coords = spectral_embedding_from_graph(G, nodelist=nodes, n_components=2)
                emb_nodes = nodes

            elif method == "PCA":
                if centrality_features is None:
                    centrality_features = compute_centrality_features(G, nodes)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=SEED)
                coords = pca.fit_transform(centrality_features)
                emb_nodes = nodes

            elif method == "DeepWalk":
                coords = deepwalk_from_graph(G, walk_length=20, walks_per_node=5,
                                             window_size=5, dimensions=2, seed=SEED)
                emb_nodes = nodes

            elif method == "Node2Vec":
                coords = node2vec_from_graph(G, walk_length=20, walks_per_node=5,
                                             window_size=5, dimensions=2,
                                             p=0.5, q=2.0, seed=SEED)
                emb_nodes = nodes

            elif method == "DM":
                if centrality_features is None:
                    centrality_features = compute_centrality_features(G, nodes)
                from utils import build_similarity_matrix, diffusion_map_from_similarity
                sim = build_similarity_matrix(centrality_features)
                coords = diffusion_map_from_similarity(sim, n_components=2)
                emb_nodes = nodes

            elif method == "MDS":
                from scipy.sparse.csgraph import shortest_path
                from scipy.sparse import csr_matrix
                A = nx.adjacency_matrix(G, nodelist=nodes).astype(np.float64)
                D = shortest_path(A, method="D", directed=False)
                max_finite = np.max(D[np.isfinite(D)])
                D[~np.isfinite(D)] = max_finite * 2
                from utils import classical_mds_from_distances
                coords = classical_mds_from_distances(D, n_components=2)
                emb_nodes = nodes

            elif method in ("VGAE", "VGAE-feat", "GraphSAGE", "GAT", "GIN"):
                # GNN methods require torch - skip on server if not available
                if method == "VGAE":
                    coords = vgae_from_graph(G, features=None, seed=SEED)
                elif method == "VGAE-feat":
                    if centrality_features is None:
                        centrality_features = compute_centrality_features(G, nodes)
                    coords = vgae_from_graph(G, features=centrality_features, seed=SEED)
                else:
                    # GraphSAGE/GAT/GIN - import from embed_gnn
                    from embed_gnn import embed_sage, embed_gat, embed_gin
                    if centrality_features is None:
                        centrality_features = compute_centrality_features(G, nodes)
                    if method == "GraphSAGE":
                        coords = embed_sage(G, centrality_features, seed=SEED)
                    elif method == "GAT":
                        coords = embed_gat(G, centrality_features, seed=SEED)
                    elif method == "GIN":
                        coords = embed_gin(G, centrality_features, seed=SEED)
                emb_nodes = nodes

            else:
                print(f"SKIP (unknown method)")
                continue

            # Save
            np.save(embeddings_dir / f"{method}_full.npy", coords)
            with open(embeddings_dir / f"{method}_full_nodes.json", "w") as f:
                json.dump(emb_nodes, f)
            embeddings[method] = (coords, emb_nodes)
            print(f"done ({time.time()-t0:.1f}s)")

        except Exception as e:
            print(f"FAILED: {str(e)[:60]}")

    print()

    # ----------------------------------------------------------------
    # Compute G-F Scores
    # ----------------------------------------------------------------
    print("[3/4] Computing G-F Scores ...")
    print("-" * 60)

    r_vals = np.linspace(0.05, 0.55, 100)  # fewer points for speed
    results = {}

    for method, (coords, emb_nodes) in embeddings.items():
        print(f"  {method:14s} ...", end=" ", flush=True)
        t0 = time.time()

        # Align to G-F subsample nodes
        node_to_idx = {nd: i for i, nd in enumerate(emb_nodes)}
        common = [nd for nd in gf_nodes if nd in node_to_idx]
        if len(common) < 20:
            print(f"SKIP (only {len(common)} common nodes)")
            continue

        emb_idx = [node_to_idx[nd] for nd in common]
        Y = rescale_coordinates(coords[emb_idx].copy())

        # Build GO map for common nodes
        go_sub = {nd: go_map_full[nd] for nd in common if nd in go_map_full}

        try:
            purities, _ = compute_gf_curve(Y, common, go_sub, r_vals)
            gf = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)
            results[method] = {
                "gf_score": float(gf),
                "peak_purity": float(max(purities)),
                "n_common": len(common),
            }
            print(f"GF={gf:.4f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"FAILED: {str(e)[:60]}")

    print()

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    print("[4/4] Saving ...")

    # Ranking
    ranked = sorted(results.items(), key=lambda x: -x[1]["gf_score"])
    print("\n  G-F Score ranking (full network):")
    for i, (m, r) in enumerate(ranked, 1):
        print(f"    {i:2d}. {m:14s}  GF={r['gf_score']:.4f}")

    output = {
        "analysis": "Full-Network 11-Method G-F (Yeast 5936)",
        "network": {"n_nodes": n, "n_edges": G.number_of_edges()},
        "gf_subsample_size": SUBSAMPLE_GF,
        "n_r_points": 100,
        "results": results,
        "ranking": [m for m, _ in ranked],
    }

    out_path = results_dir / "full_network_all11.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    elapsed = time.time() - t_start
    print(f"  Total time: {elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()