#!/usr/bin/env python3
"""
function_prediction_full.py
============================
Expand LOTO-CV function prediction from 5 methods to all 11 methods
on the full yeast PPI network (5,936 nodes).

Steps:
  1. Compute missing full-network embeddings (PCA, DeepWalk, VGAE-feat,
     GIN, GAT, GraphSAGE)
  2. Load all 11 full-network embeddings
  3. Run LOTO-CV with KNN prediction for all 11 methods
  4. Compute MRR, GF Scores, Spearman correlation, permutation p-value
  5. Save results to results/function_prediction_full.json
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
from scipy import stats
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, TARGET_STD, get_data_dir, get_results_dir, get_embeddings_dir,
    load_embedding, save_embedding, rescale_coordinates,
    compute_centrality_features, check_embedding_collapse,
    build_similarity_matrix, diffusion_map_from_similarity,
    classical_mds_from_distances, spectral_embedding_from_graph,
    deepwalk_from_graph, node2vec_from_graph, vgae_from_graph,
)

# ============================================================
# Constants
# ============================================================

DATA = get_data_dir()
RESULTS = get_results_dir()
EMB = get_embeddings_dir()

# RESULTS.mkdir(parents=True, exist_ok=True)  # deferred to run() — P1-4b

# All 11 methods
ALL_METHODS = [
    "DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE",
    "PCA", "VGAE-feat", "GIN", "GAT", "GraphSAGE",
]

# Methods that already have full-network embeddings
EXISTING_FULL = ["DM", "MDS", "Spectral", "Node2Vec", "VGAE"]

# Methods that need full-network embeddings computed
MISSING_FULL = ["PCA", "DeepWalk", "VGAE-feat", "GIN", "GAT", "GraphSAGE"]

# Files
GAF_FILE = DATA / "gene_association.sgd.gaf.gz"
ALIAS_FILE = DATA / "4932.protein.aliases.v11.5.txt.gz"
NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"
GF_SCORES_FILE = RESULTS / "gf_scores.json"
GNN_GF_FILE = RESULTS / "gnn_gf_scores.json"

# Evidence codes considered experimental
EXPERIMENTAL_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}

# k values for precision@k
K_VALUES = [3, 5, 7, 10, 15, 20, 30]
K_MAX = max(K_VALUES)

BANNER = "=" * 64


# ============================================================
# 1. Network Loading
# ============================================================

def load_full_network():
    """Load the 5936-node yeast PPI network (largest connected component)."""
    G = nx.Graph()
    with open(str(NETWORK_FILE), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    print(f"  Full network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ============================================================
# 2. Compute Missing Full-Network Embeddings
# ============================================================

def embed_pca_full(G, nodes, features):
    """PCA on centrality features (full network)."""
    features_centered = features - features.mean(axis=0)
    cov = features_centered.T @ features_centered / (len(nodes) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    coords = features_centered @ eigvecs[:, -2:]
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_deepwalk_full(G, nodes):
    """DeepWalk on full network (sparse SVD path for n > 1000)."""
    coords = deepwalk_from_graph(
        G, walk_length=20, walks_per_node=10,
        window_size=5, dimensions=2, seed=SEED,
    )
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_vgae_feat_full(G, nodes, features):
    """VGAE with centrality features on full network."""
    coords = vgae_from_graph(
        G, hidden_dim=4, latent_dim=2, epochs=300,
        lr=0.01, features=features, seed=SEED,
    )
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_gnn_full(G, nodes, features, method_name,
                   hidden_dim=32, latent_dim=2, epochs=300, lr=0.01):
    """Generic GNN embedding for full network using embed_gnn builders."""
    from embed_gnn import (
        _build_sage_encoder, _build_gat_encoder, _build_gin_encoder,
    )

    builders = {
        "GraphSAGE": _build_sage_encoder,
        "GAT": _build_gat_encoder,
        "GIN": _build_gin_encoder,
    }
    builder = builders[method_name]

    import torch
    import torch.nn.functional as F
    from torch_geometric.utils import from_networkx

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    node_list = list(G.nodes())
    n = len(node_list)
    data = from_networkx(G)

    if features is not None:
        data.x = torch.tensor(features, dtype=torch.float32)
        in_dim = features.shape[1]
    else:
        data.x = torch.eye(n)
        in_dim = n

    model = builder(in_dim, hidden_dim, latent_dim, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Adjacency target for BCE loss
    adj_target = torch.zeros(n, n)
    ei = data.edge_index
    adj_target[ei[0], ei[1]] = 1.0

    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        adj_recon = torch.sigmoid(z @ z.T)
        recon_loss = F.binary_cross_entropy(adj_recon, adj_target, reduction="sum")
        loss = recon_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss={loss.item():.2f}")

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        coords = z.numpy()

    collapse_info = check_embedding_collapse(coords, method_name)
    if collapse_info["collapsed"]:
        print(f"  WARNING: {method_name} collapsed: {collapse_info['reasons']}")

    return rescale_coordinates(coords, target_std=TARGET_STD)


def compute_missing_embeddings(G, nodes, features):
    """Compute full-network embeddings for the 6 missing methods."""
    print("\n[Step 1] Computing missing full-network embeddings...")

    for method in MISSING_FULL:
        emb_file = EMB / f"{method}_full.npy"
        if emb_file.exists():
            print(f"  {method}: already exists, skipping")
            continue

        print(f"\n  Computing {method} on full network ({len(nodes)} nodes)...")
        random.seed(SEED)
        np.random.seed(SEED)
        t0 = time.time()

        try:
            if method == "PCA":
                coords = embed_pca_full(G, nodes, features)
            elif method == "DeepWalk":
                coords = embed_deepwalk_full(G, nodes)
            elif method == "VGAE-feat":
                coords = embed_vgae_feat_full(G, nodes, features)
            elif method in ("GraphSAGE", "GAT", "GIN"):
                coords = embed_gnn_full(
                    G, nodes, features, method,
                    hidden_dim=32, latent_dim=2, epochs=300, lr=0.01,
                )
            else:
                print(f"  Unknown method: {method}")
                continue

            save_embedding(coords, nodes, method, "full", EMB)
            elapsed = time.time() - t0
            print(f"  {method}: saved ({coords.shape}), std={np.std(coords):.4f}, "
                  f"time={elapsed:.1f}s")
        except Exception as e:
            print(f"  {method} FAILED: {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# 3. Alias Mapping and GAF Parsing (from function_prediction.py)
# ============================================================

def build_alias_mapping():
    """Build SGD_ID -> STRING_ID mapping."""
    sgd_to_strings = defaultdict(set)
    orf_to_strings = defaultdict(set)
    network_nodes = set()

    with gzip.open(str(ALIAS_FILE), "rt", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            raw_string_id = parts[0]
            alias = parts[1]
            source = parts[2]
            string_id = raw_string_id.split(".", 1)[1] if "." in raw_string_id else raw_string_id
            network_nodes.add(string_id)

            if source == "SGD_ID":
                sgd_to_strings[alias].add(string_id)
            elif source in ("Ensembl_SGD_GENE", "SGD_SYNONYM"):
                if alias and len(alias) >= 3 and alias[0] == "Y":
                    orf_to_strings[alias].add(string_id)
            elif source == "Ensembl_SGD_TRANSCRIPT":
                if alias and len(alias) >= 3 and alias[0] == "Y":
                    orf_to_strings[alias].add(string_id)

    sgd_to_string = {sgd: min(ids, key=len) for sgd, ids in sgd_to_strings.items() if ids}
    orf_to_string = {orf: min(ids, key=len) for orf, ids in orf_to_strings.items() if ids}

    print(f"  Alias mapping: {len(sgd_to_string)} SGD IDs, "
          f"{len(orf_to_string)} ORF names -> STRING IDs")
    return sgd_to_string, orf_to_string, network_nodes


def parse_gaf_experimental(sgd_to_string, orf_to_string, network_nodes):
    """Parse SGD GAF file, keeping only experimental BP annotations."""
    annotations = defaultdict(set)
    total_lines = 0
    experimental_bp = 0
    mapped = 0

    with gzip.open(str(GAF_FILE), "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("!") or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            total_lines += 1
            evidence = cols[6]
            aspect = cols[8]
            if evidence not in EXPERIMENTAL_CODES or aspect != "P":
                continue
            experimental_bp += 1
            go_term = cols[4]
            if not go_term.startswith("GO:"):
                continue

            sgd_id = cols[1]
            gene_sym = cols[2]
            orf_name = cols[10] if len(cols) > 10 else ""

            string_id = None
            if sgd_id in sgd_to_string:
                candidate = sgd_to_string[sgd_id]
                if candidate in network_nodes:
                    string_id = candidate
            if string_id is None and orf_name and orf_name in orf_to_string:
                candidate = orf_to_string[orf_name]
                if candidate in network_nodes:
                    string_id = candidate
            if string_id is None and gene_sym in orf_to_string:
                candidate = orf_to_string[gene_sym]
                if candidate in network_nodes:
                    string_id = candidate
            if string_id is None and orf_name and orf_name in network_nodes:
                string_id = orf_name

            if string_id is None:
                continue
            mapped += 1
            annotations[string_id].add(go_term)

    ann_stats = {
        "total_gaf_lines": total_lines,
        "experimental_bp_lines": experimental_bp,
        "mapped_to_network": mapped,
        "proteins_annotated": len(annotations),
        "unique_terms": len(set(t for ts in annotations.values() for t in ts)),
        "terms_per_protein_mean": round(
            np.mean([len(ts) for ts in annotations.values()]), 1
        ) if annotations else 0,
    }
    print(f"  GAF: {total_lines} lines, {experimental_bp} experimental BP, "
          f"{mapped} mapped -> {len(annotations)} proteins")
    return dict(annotations), ann_stats


# ============================================================
# 4. Load All Full-Network Embeddings
# ============================================================

def load_all_full_embeddings(network_nodes):
    """Load full-network embeddings for all 11 methods."""
    embeddings = {}
    for method in ALL_METHODS:
        try:
            coords, emb_nodes = load_embedding(method, "full", embeddings_dir=EMB)
            node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
            common = [n for n in emb_nodes if n in network_nodes]
            if len(common) < 100:
                print(f"  WARNING: {method} has only {len(common)} common nodes, skipping")
                continue
            indices = [node_to_idx[n] for n in common]
            filtered_coords = coords[indices]
            embeddings[method] = {
                "coords": filtered_coords,
                "nodes": common,
                "node_to_idx": {n: i for i, n in enumerate(common)},
            }
            print(f"  {method}: {len(common)} nodes loaded (dim={filtered_coords.shape[1]})")
        except Exception as e:
            print(f"  {method} FAILED: {e}")
    return embeddings


# ============================================================
# 5. KNN Prediction (from function_prediction.py)
# ============================================================

def build_knn_index(coords):
    """Pre-build a NearestNeighbors index."""
    nn = NearestNeighbors(n_neighbors=min(K_MAX + 1, len(coords)), metric="euclidean")
    nn.fit(coords)
    return nn


def knn_predict_fast(query_idx, nn_model, coords, nodes, annotations, k=10,
                     hidden_term=None):
    """Fast KNN prediction using pre-built index."""
    query_coord = coords[query_idx:query_idx + 1]
    n_neighbors = min(k + 1, len(coords))
    distances, indices = nn_model.kneighbors(query_coord, n_neighbors=n_neighbors)

    term_scores = Counter()
    for dist, idx in zip(distances[0], indices[0]):
        if idx == query_idx:
            continue
        neighbor_id = nodes[idx]
        neighbor_terms = annotations.get(neighbor_id, set())
        weight = 1.0 / (dist + 1e-10)
        for term in neighbor_terms:
            term_scores[term] += weight

    return term_scores.most_common()


# ============================================================
# 6. LOTO-CV Evaluation
# ============================================================

def run_loto_cv(embeddings, graph, annotations):
    """Run leave-one-term-out CV for all methods."""
    # Filter proteins with >= 2 terms
    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in graph
    }
    n_proteins = len(query_proteins)
    n_trials = sum(len(terms) for terms in query_proteins.values())
    print(f"\n  Query proteins: {n_proteins} (>= 2 terms each)")
    print(f"  Total LOTO trials: {n_trials}")

    # Pre-build KNN indices
    knn_indices = {}
    for method, emb in embeddings.items():
        knn_indices[method] = build_knn_index(emb["coords"])
        print(f"  KNN index built for {method}: {len(emb['nodes'])} nodes")

    # Run predictions
    rank_results = []  # list of {method: rank}
    completed = 0
    t0 = time.time()

    for pid, terms in sorted(query_proteins.items()):
        terms_list = sorted(terms)
        for hidden_term in terms_list:
            trial_rank = {}
            for method, emb in embeddings.items():
                node_to_idx = emb["node_to_idx"]
                if pid not in node_to_idx:
                    continue
                query_idx = node_to_idx[pid]
                preds = knn_predict_fast(
                    query_idx, knn_indices[method],
                    emb["coords"], emb["nodes"],
                    annotations, k=K_MAX,
                    hidden_term=hidden_term,
                )
                pred_terms = [t for t, _ in preds]
                try:
                    rank = pred_terms.index(hidden_term) + 1
                except ValueError:
                    rank = 0
                trial_rank[method] = rank

            rank_results.append(trial_rank)
            completed += 1
            if completed % 5000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_trials - completed) / rate if rate > 0 else 0
                print(f"    Progress: {completed}/{n_trials} "
                      f"({100*completed/n_trials:.0f}%) "
                      f"-- {rate:.0f} trials/s -- ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Completed {completed} trials in {elapsed:.1f}s "
          f"({completed/elapsed:.0f} trials/s)")
    return rank_results, {
        "query_proteins": n_proteins,
        "total_trials": n_trials,
        "completed": completed,
        "elapsed_seconds": round(elapsed, 1),
    }


# ============================================================
# 7. Metrics
# ============================================================

def compute_mrr(rank_results):
    """Compute MRR for each method."""
    method_ranks = defaultdict(list)
    for trial in rank_results:
        for method, rank in trial.items():
            if rank > 0:
                method_ranks[method].append(1.0 / rank)
            else:
                method_ranks[method].append(0.0)
    return {m: float(np.mean(rs)) if rs else 0.0 for m, rs in method_ranks.items()}


def load_gf_scores():
    """Load GF scores from both gf_scores.json and gnn_gf_scores.json."""
    # Classical + VGAE-feat scores
    with open(str(GF_SCORES_FILE), encoding="utf-8") as f:
        gf_data = json.load(f)
    gf_scores = gf_data.get("scores", {})

    # GNN scores
    with open(str(GNN_GF_FILE), encoding="utf-8") as f:
        gnn_data = json.load(f)
    gnn_scores = gnn_data.get("gf_scores", {})

    # Merge
    all_scores = {**gf_scores, **gnn_scores}
    return all_scores


def compute_correlation(method_mrr, gf_scores):
    """Compute Spearman correlation between GF Score and MRR across methods."""
    methods = sorted(set(method_mrr.keys()) & set(gf_scores.keys()))
    # Only include methods that have both MRR and GF score
    methods = [m for m in methods if m in method_mrr and m in gf_scores]

    if len(methods) < 3:
        return {"error": f"Too few common methods: {len(methods)}"}

    x = np.array([gf_scores[m] for m in methods])
    y = np.array([method_mrr[m] for m in methods])

    rho, p_val = stats.spearmanr(x, y)
    r_pearson, p_pearson = stats.pearsonr(x, y)

    # Permutation test (10,000 shuffles)
    rng = np.random.RandomState(SEED)
    n_perm = 10000
    perm_rhos = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        perm_rhos[i], _ = stats.spearmanr(x, y_perm)

    # Two-sided permutation p-value
    perm_p = float(np.mean(np.abs(perm_rhos) >= abs(rho)))

    # Bootstrap CI for Spearman
    boot_rhos = []
    n_boot = 10000
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
            continue
        br, _ = stats.spearmanr(x[idx], y[idx])
        boot_rhos.append(br)

    ci_lo = float(np.percentile(boot_rhos, 2.5)) if boot_rhos else rho
    ci_hi = float(np.percentile(boot_rhos, 97.5)) if boot_rhos else rho

    # Leave-one-out sensitivity
    loo_rhos = {}
    for m in methods:
        mask = [mi for mi in methods if mi != m]
        if len(mask) >= 3:
            lr, _ = stats.spearmanr(
                [gf_scores[mi] for mi in mask],
                [method_mrr[mi] for mi in mask],
            )
            loo_rhos[m] = round(float(lr), 4)

    result = {
        "methods": methods,
        "n": len(methods),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p_val), 6),
        "spearman_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "pearson_r": round(float(r_pearson), 4),
        "pearson_p": round(float(p_pearson), 6),
        "permutation_p": round(perm_p, 6),
        "permutation_n": n_perm,
        "loo_sensitivity": loo_rhos,
        "significant_at_005": bool(p_val < 0.05 or perm_p < 0.05),
        "per_method": [
            {"method": m, "gf_score": round(gf_scores[m], 4),
             "mrr": round(method_mrr[m], 4)}
            for m in methods
        ],
    }

    print(f"\n  GF Score vs MRR correlation (n={len(methods)} methods):")
    print(f"    Spearman rho = {rho:.4f} (parametric P = {p_val:.6f})")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    Pearson r  = {r_pearson:.4f} (P = {p_pearson:.6f})")
    print(f"    Permutation P (10,000 shuffles) = {perm_p:.6f}")
    print(f"    Significant at p < 0.05: {p_val < 0.05 or perm_p < 0.05}")

    return result


# ============================================================
# 8. Main
# ============================================================

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(BANNER)
    print("Full 11-Method LOTO-CV Function Prediction Analysis")
    print(BANNER)

    np.random.seed(SEED)
    random.seed(SEED)

    # --- Stage 1: Load network ---
    print("\n[Step 1] Loading full PPI network...")
    graph = load_full_network()
    nodes = sorted(graph.nodes())

    # --- Stage 2: Compute centrality features ---
    print("\n[Step 2] Computing centrality features...")
    features = compute_centrality_features(graph, nodes)
    print(f"  Features shape: {features.shape}")

    # --- Stage 3: Compute missing embeddings ---
    compute_missing_embeddings(graph, nodes, features)

    # --- Stage 4: Build alias mapping and parse GAF ---
    print("\n[Step 3] Building alias mapping...")
    sgd_to_string, orf_to_string, network_nodes = build_alias_mapping()

    print("\n[Step 4] Parsing GAF (experimental BP)...")
    annotations, ann_stats = parse_gaf_experimental(
        sgd_to_string, orf_to_string, network_nodes
    )

    if len(annotations) < 50:
        print(f"ERROR: Only {len(annotations)} proteins annotated, aborting.")
        return

    # Term frequencies
    term_freq = Counter()
    for terms in annotations.values():
        term_freq.update(terms)

    # --- Stage 5: Load all embeddings ---
    print("\n[Step 5] Loading all full-network embeddings...")
    embeddings = load_all_full_embeddings(set(graph.nodes()))
    if not embeddings:
        print("ERROR: No embeddings loaded, aborting.")
        return
    print(f"  Loaded {len(embeddings)} methods: {sorted(embeddings.keys())}")

    # --- Stage 6: Run LOTO-CV ---
    print("\n[Step 6] Running LOTO-CV predictions...")
    rank_results, pred_stats = run_loto_cv(embeddings, graph, annotations)

    # --- Stage 7: Compute metrics ---
    print("\n[Step 7] Computing metrics...")

    # MRR
    method_mrr = compute_mrr(rank_results)
    print("\n  Mean Reciprocal Rank:")
    for m in sorted(method_mrr, key=method_mrr.get, reverse=True):
        print(f"    {m:15s}: MRR={method_mrr[m]:.4f}")

    # GF Scores
    gf_scores = load_gf_scores()
    print("\n  GF Scores (from 153-node curated network):")
    for m in ALL_METHODS:
        if m in gf_scores:
            print(f"    {m:15s}: GF={gf_scores[m]:.4f}")
        else:
            print(f"    {m:15s}: GF=N/A")

    # Correlation
    print("\n  Computing GF Score vs MRR correlation...")
    corr_result = compute_correlation(method_mrr, gf_scores)

    # --- Stage 8: Save results ---
    print("\n[Step 8] Saving results...")
    output = {
        "description": "Full 11-Method LOTO-CV Function Prediction Analysis",
        "methods_evaluated": sorted(embeddings.keys()),
        "n_methods": len(embeddings),
        "annotation_stats": ann_stats,
        "prediction_stats": pred_stats,
        "mean_reciprocal_rank": {
            m: round(v, 6) for m, v in method_mrr.items()
        },
        "gf_scores": {
            m: round(gf_scores[m], 6) for m in ALL_METHODS if m in gf_scores
        },
        "gf_correlation": corr_result,
    }

    result_file = RESULTS / "function_prediction_full.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved {result_file}")

    # --- Final summary ---
    print(f"\n{BANNER}")
    print("SUMMARY")
    print(BANNER)
    print(f"\n  Methods evaluated: {len(embeddings)}")
    print(f"  Query proteins: {pred_stats['query_proteins']}")
    print(f"  Total LOTO trials: {pred_stats['total_trials']}")
    print(f"\n  MRR for all methods:")
    for m in sorted(method_mrr, key=method_mrr.get, reverse=True):
        gf = gf_scores.get(m, None)
        gf_str = f"{gf:.4f}" if gf is not None else "N/A"
        print(f"    {m:15s}: MRR={method_mrr[m]:.4f}  GF={gf_str}")

    if "spearman_rho" in corr_result:
        print(f"\n  Spearman rho = {corr_result['spearman_rho']:.4f}")
        print(f"  Parametric P = {corr_result['spearman_p']:.6f}")
        print(f"  Permutation P = {corr_result['permutation_p']:.6f}")
        print(f"  Significant at p < 0.05: {corr_result['significant_at_005']}")
    print(f"\n{BANNER}")


if __name__ == "__main__":
    main()
