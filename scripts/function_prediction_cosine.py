#!/usr/bin/env python3
"""
Cosine Similarity Voting Baseline for Protein Function Prediction
=================================================================

Tests whether cosine similarity weighted voting (instead of Euclidean KNN)
improves embedding-based function prediction on the full yeast STRING network.

For each of 5 embedding methods (Spectral, MDS, DM, Node2Vec, VGAE):
  - Euclidean KNN (k=10): same as function_prediction.py, for comparison
  - Cosine similarity voting: top-100 most cosine-similar proteins,
    votes weighted by max(cosine_sim, 0)

Evaluation: Leave-one-term-out cross-validation (LOTO-CV) over proteins
with >= 2 experimental BP GO annotations.

Metrics: MRR, P@10, Spearman correlation between GF Score and MRR.
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
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    get_data_dir, get_results_dir, get_embeddings_dir,
    load_embedding,
)

# ============================================================
# Constants
# ============================================================

DATA = get_data_dir()
RESULTS = get_results_dir()
EMB = get_embeddings_dir()

# RESULTS.mkdir(parents=True, exist_ok=True)  # deferred to run() — P1-4b

BANNER = "=" * 70

# Files
GAF_FILE = DATA / "gene_association.sgd.gaf.gz"
ALIAS_FILE = DATA / "4932.protein.aliases.v11.5.txt.gz"
NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"
GF_SCORES_FILE = RESULTS / "gf_scores.json"

# Methods with full-network embeddings
FULL_METHODS = ["Spectral", "MDS", "DM", "Node2Vec", "VGAE"]

# Evidence codes considered experimental
EXPERIMENTAL_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}

# Euclidean KNN parameter
K_EUCLIDEAN = 10

# Cosine voting parameter
K_COSINE = 100

# Precision@k thresholds
K_VALUES = [3, 5, 7, 10, 15, 20, 30]


# ============================================================
# 1. Alias Mapping: SGD_ID -> STRING network node ID
# ============================================================

def build_alias_mapping():
    """Build SGD_ID -> STRING_ID mapping from the yeast aliases file.

    Returns
    -------
    sgd_to_string : dict[str, str]
    orf_to_string : dict[str, str]
    network_nodes : set[str]
    """
    sgd_to_strings = defaultdict(set)
    orf_to_strings = defaultdict(set)
    network_nodes = set()

    with gzip.open(str(ALIAS_FILE), "rt", encoding="utf-8") as fh:
        fh.readline()  # skip header
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
    print(f"  Network nodes in aliases: {len(network_nodes)}")

    return sgd_to_string, orf_to_string, network_nodes


# ============================================================
# 2. GAF Parsing (Experimental Evidence, BP Only)
# ============================================================

def parse_gaf_experimental(sgd_to_string, orf_to_string, network_nodes):
    """Parse SGD GAF file, keeping only experimental BP annotations.

    Returns
    -------
    annotations : dict[str, set[str]]
        STRING_ID -> set of GO BP terms.
    ann_stats : dict
    """
    annotations = defaultdict(set)
    total_lines = 0
    experimental_bp = 0
    mapped = 0
    unmapped_ids = Counter()

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
                unmapped_ids[sgd_id] += 1
                continue

            mapped += 1
            annotations[string_id].add(go_term)

    ann_stats = {
        "total_gaf_lines": int(total_lines),
        "experimental_bp_lines": int(experimental_bp),
        "mapped_to_network": int(mapped),
        "unmapped": int(sum(unmapped_ids.values())),
        "proteins_annotated": int(len(annotations)),
        "unique_terms": int(len(set(t for ts in annotations.values() for t in ts))),
        "terms_per_protein_mean": round(
            float(np.mean([len(ts) for ts in annotations.values()])), 1
        ) if annotations else 0,
    }

    print(f"  GAF parsed: {total_lines} lines, {experimental_bp} experimental BP")
    print(f"  Mapped to network: {mapped} annotations -> {len(annotations)} proteins")
    print(f"  Unique GO BP terms: {ann_stats['unique_terms']}")
    print(f"  Mean terms/protein: {ann_stats['terms_per_protein_mean']}")

    return dict(annotations), ann_stats


# ============================================================
# 3. Network and Embedding Loading
# ============================================================

def load_full_network():
    """Load the 5936-node yeast PPI network."""
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


def load_all_full_embeddings(network_nodes):
    """Load full-network embeddings for all 5 methods.

    Returns
    -------
    embeddings : dict[str, dict]
        {method: {"coords": ndarray, "nodes": list, "node_to_idx": dict}}
    """
    embeddings = {}
    for method in FULL_METHODS:
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
            print(f"  {method}: {len(common)} nodes loaded, dim={filtered_coords.shape[1]}")
        except Exception as e:
            print(f"  {method} FAILED: {e}")

    return embeddings


# ============================================================
# 4. Prediction Strategies
# ============================================================

def build_euclidean_knn_index(coords, k=K_EUCLIDEAN):
    """Pre-build a Euclidean NearestNeighbors index."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), metric="euclidean")
    nn.fit(coords)
    return nn


def euclidean_knn_predict(query_idx, nn_model, coords, nodes, annotations,
                         k=K_EUCLIDEAN, hidden_term=None):
    """Predict GO terms using Euclidean KNN (k=10), weighted by 1/distance.

    Returns list of (go_term, score) sorted by descending score.
    """
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


def compute_cosine_similarity_matrix(coords):
    """Compute full pairwise cosine similarity matrix.

    Parameters
    ----------
    coords : ndarray of shape (n, d)

    Returns
    -------
    sim_matrix : ndarray of shape (n, n)
        Cosine similarity values in [-1, 1].
    """
    # sklearn cosine_similarity handles normalization internally
    sim_matrix = cosine_similarity(coords)
    return sim_matrix


def cosine_predict(query_idx, sim_matrix, nodes, annotations, k=K_COSINE,
                   hidden_term=None):
    """Predict GO terms using cosine similarity weighted voting.

    Uses the top-k most cosine-similar proteins (excluding self),
    weighting each vote by max(cosine_similarity, 0) so that only
    positively correlated neighbors contribute.

    Parameters
    ----------
    query_idx : int
        Index of query protein in the embedding.
    sim_matrix : ndarray (n, n)
        Precomputed cosine similarity matrix.
    nodes : list
        Node IDs.
    annotations : dict
        {STRING_ID: set(go_terms)}.
    k : int
        Number of top cosine-similar neighbors to use.
    hidden_term : str or None
        GO term to hide from neighbor annotations (for LOTO).

    Returns
    -------
    list of (go_term, score) sorted by descending score.
    """
    n = sim_matrix.shape[0]
    sims = sim_matrix[query_idx]

    # Get top-k indices (excluding self)
    # Use argpartition for efficiency: partition so that the k+1 largest
    # are at the end, then sort just those.
    actual_k = min(k, n - 1)

    # Indices of all other nodes
    # Set self-similarity to -inf temporarily so it won't be in top-k
    sims_copy = sims.copy()
    sims_copy[query_idx] = -np.inf

    if actual_k < n - 1:
        # Partial sort: get indices of the top actual_k values
        partition_idx = np.argpartition(-sims_copy, actual_k)[:actual_k]
        # Sort these by descending similarity
        top_indices = partition_idx[np.argsort(-sims_copy[partition_idx])]
    else:
        top_indices = np.argsort(-sims_copy)[:actual_k]

    term_scores = Counter()
    for idx in top_indices:
        sim_val = float(sims[idx])
        weight = max(sim_val, 0.0)
        if weight <= 0.0:
            continue
        neighbor_id = nodes[idx]
        neighbor_terms = annotations.get(neighbor_id, set())
        for term in neighbor_terms:
            term_scores[term] += weight

    return term_scores.most_common()


# ============================================================
# 5. Evaluation Helpers
# ============================================================

def evaluate_precision_at_k(predictions, actual_term, k_values):
    """Check if actual_term appears in top-k predictions.

    Returns dict[int, int]: {k: 1 if found in top-k, else 0}
    """
    pred_terms = [t for t, _ in predictions]
    return {k: 1 if actual_term in pred_terms[:k] else 0 for k in k_values}


def get_rank(predictions, actual_term):
    """Get 1-based rank of actual_term in predictions. 0 = not found."""
    pred_terms = [t for t, _ in predictions]
    try:
        return pred_terms.index(actual_term) + 1
    except ValueError:
        return 0


def compute_mrr(all_rank_results, method_key):
    """Compute MRR from a list of per-trial rank dicts."""
    reciprocals = []
    for trial in all_rank_results:
        rank = trial.get(method_key, 0)
        if rank > 0:
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)
    return float(np.mean(reciprocals)) if reciprocals else 0.0


def compute_precision_at_k_agg(all_precision_results, method_key, k_values):
    """Aggregate precision@k across all trials for a given method."""
    k_hits = {k: [] for k in k_values}
    for trial in all_precision_results:
        trial_data = trial.get(method_key, {})
        for k in k_values:
            if k in trial_data:
                k_hits[k].append(trial_data[k])
    return {k: float(np.mean(hits)) if hits else 0.0 for k, hits in k_hits.items()}


# ============================================================
# 6. Main LOTO-CV Loop
# ============================================================

def run_loto_cv(embeddings, annotations):
    """Run Leave-One-Term-Out CV with both Euclidean KNN and Cosine voting.

    For each protein with >= 2 GO terms, hide each term in turn and
    predict using both strategies for every embedding method.

    Returns
    -------
    precision_results : list of dict
        Per-trial {method_strategy: {k: hit_or_miss}}
    rank_results : list of dict
        Per-trial {method_strategy: rank}
    stats : dict
    """
    # Filter proteins with >= 2 terms
    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2
    }

    n_proteins = len(query_proteins)
    n_trials = sum(len(terms) for terms in query_proteins.values())
    print(f"\n  Query proteins: {n_proteins} (>= 2 terms each)")
    print(f"  Total LOTO trials: {n_trials}")

    # Pre-build Euclidean KNN indices and cosine similarity matrices
    euclidean_indices = {}
    cosine_matrices = {}
    for method, emb in embeddings.items():
        euclidean_indices[method] = build_euclidean_knn_index(emb["coords"], K_EUCLIDEAN)
        print(f"  Euclidean KNN index built for {method}")

        print(f"  Computing cosine similarity matrix for {method} "
              f"({emb['coords'].shape[0]} x {emb['coords'].shape[1]})...")
        t0_cos = time.time()
        cosine_matrices[method] = compute_cosine_similarity_matrix(emb["coords"])
        dt_cos = time.time() - t0_cos
        print(f"    Done in {dt_cos:.1f}s")

    precision_results = []
    rank_results = []
    completed = 0
    t0 = time.time()

    for pid, terms in sorted(query_proteins.items()):
        terms_list = sorted(terms)

        for hidden_term in terms_list:
            trial_precision = {}
            trial_rank = {}

            for method, emb in embeddings.items():
                node_to_idx = emb["node_to_idx"]
                if pid not in node_to_idx:
                    continue

                query_idx = node_to_idx[pid]

                # Strategy A: Euclidean KNN (k=10)
                euc_key = f"{method}_euclidean"
                euc_preds = euclidean_knn_predict(
                    query_idx, euclidean_indices[method],
                    emb["coords"], emb["nodes"],
                    annotations, k=K_EUCLIDEAN,
                    hidden_term=hidden_term,
                )
                trial_precision[euc_key] = evaluate_precision_at_k(
                    euc_preds, hidden_term, K_VALUES
                )
                trial_rank[euc_key] = get_rank(euc_preds, hidden_term)

                # Strategy B: Cosine similarity voting (top-100)
                cos_key = f"{method}_cosine"
                cos_preds = cosine_predict(
                    query_idx, cosine_matrices[method],
                    emb["nodes"], annotations, k=K_COSINE,
                    hidden_term=hidden_term,
                )
                trial_precision[cos_key] = evaluate_precision_at_k(
                    cos_preds, hidden_term, K_VALUES
                )
                trial_rank[cos_key] = get_rank(cos_preds, hidden_term)

            precision_results.append(trial_precision)
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
    pred_stats = {
        "query_proteins": int(n_proteins),
        "total_trials": int(n_trials),
        "completed": int(completed),
        "elapsed_seconds": round(elapsed, 1),
    }
    print(f"\n  Completed {completed} trials in {elapsed:.1f}s "
          f"({completed/elapsed:.0f} trials/s)")

    return precision_results, rank_results, pred_stats


# ============================================================
# 7. GF Score Correlation
# ============================================================

def compute_gf_correlation(method_mrr, gf_scores, strategy_label):
    """Correlate per-method MRR with curated GF Scores.

    Parameters
    ----------
    method_mrr : dict
        {method_name: MRR} (only the base method names, no suffix).
    gf_scores : dict
        {method_name: GF Score}.
    strategy_label : str
        "euclidean" or "cosine", for display.

    Returns
    -------
    dict with correlation statistics.
    """
    methods = sorted(set(method_mrr.keys()) & set(gf_scores.keys()))
    if len(methods) < 3:
        return {"error": f"Too few common methods: {len(methods)}"}

    x = np.array([gf_scores[m] for m in methods])
    y = np.array([method_mrr[m] for m in methods])

    rho, p_val = stats.spearmanr(x, y)
    r_pearson, p_pearson = stats.pearsonr(x, y)

    # Bootstrap CI for Spearman
    rng = np.random.RandomState(SEED)
    boot_rhos = []
    n_boot = 10000
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
            continue
        br, _ = stats.spearmanr(x[idx], y[idx])
        boot_rhos.append(br)

    if boot_rhos:
        ci_lo = float(np.percentile(boot_rhos, 2.5))
        ci_hi = float(np.percentile(boot_rhos, 97.5))
    else:
        ci_lo, ci_hi = float(rho), float(rho)

    result = {
        "strategy": strategy_label,
        "methods": methods,
        "n": int(len(methods)),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p_val), 6),
        "spearman_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "pearson_r": round(float(r_pearson), 4),
        "pearson_p": round(float(p_pearson), 6),
        "per_method": [
            {"method": m, "gf_score": round(float(gf_scores[m]), 4),
             "mrr": round(float(method_mrr[m]), 4)}
            for m in methods
        ],
    }

    print(f"\n  GF Score vs MRR ({strategy_label}):")
    print(f"    Spearman rho = {rho:.4f} (P = {p_val:.4f})")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    Pearson r  = {r_pearson:.4f} (P = {p_pearson:.4f})")

    return result


# ============================================================
# 8. Results and Reporting
# ============================================================

def print_comparison_table(results_by_method):
    """Print a clear comparison table of Euclidean vs Cosine results.

    Parameters
    ----------
    results_by_method : dict
        {method: {"euc_mrr", "cos_mrr", "euc_p10", "cos_p10"}}
    """
    print(f"\n{BANNER}")
    print("  EUCLIDEAN KNN (k=10) vs COSINE SIMILARITY VOTING (top-100)")
    print(BANNER)
    print()
    header = (f"  {'Method':<12s} | {'Euc MRR':>8s} | {'Cos MRR':>8s} | "
              f"{'Euc P@10':>9s} | {'Cos P@10':>9s} | {'MRR delta':>10s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for method in FULL_METHODS:
        if method not in results_by_method:
            continue
        r = results_by_method[method]
        euc_mrr = r["euc_mrr"]
        cos_mrr = r["cos_mrr"]
        euc_p10 = r["euc_p10"]
        cos_p10 = r["cos_p10"]
        delta = cos_mrr - euc_mrr
        delta_str = f"{delta:+.4f}"
        print(f"  {method:<12s} | {euc_mrr:>8.4f} | {cos_mrr:>8.4f} | "
              f"{euc_p10:>9.4f} | {cos_p10:>9.4f} | {delta_str:>10s}")

    print()


def build_results_json(results_by_method, ann_stats, pred_stats,
                       euc_corr, cos_corr, precision_by_method):
    """Build JSON-serializable results dict with proper type casting."""

    method_comparison = {}
    for method in FULL_METHODS:
        if method not in results_by_method:
            continue
        r = results_by_method[method]
        method_comparison[method] = {
            "euclidean_mrr": round(float(r["euc_mrr"]), 6),
            "cosine_mrr": round(float(r["cos_mrr"]), 6),
            "euclidean_p10": round(float(r["euc_p10"]), 6),
            "cosine_p10": round(float(r["cos_p10"]), 6),
            "mrr_improvement": round(float(r["cos_mrr"] - r["euc_mrr"]), 6),
            "p10_improvement": round(float(r["cos_p10"] - r["euc_p10"]), 6),
        }

    # Precision@k for both strategies
    precision_at_k = {}
    for method in FULL_METHODS:
        euc_key = f"{method}_euclidean"
        cos_key = f"{method}_cosine"
        if euc_key in precision_by_method:
            precision_at_k[f"{method}_euclidean"] = {
                str(k): round(float(v), 6)
                for k, v in precision_by_method[euc_key].items()
            }
        if cos_key in precision_by_method:
            precision_at_k[f"{method}_cosine"] = {
                str(k): round(float(v), 6)
                for k, v in precision_by_method[cos_key].items()
            }

    output = {
        "description": "Cosine similarity voting baseline for function prediction",
        "parameters": {
            "euclidean_k": int(K_EUCLIDEAN),
            "cosine_k": int(K_COSINE),
            "seed": int(SEED),
        },
        "annotation_stats": {k: _cast(v) for k, v in ann_stats.items()},
        "prediction_stats": {k: _cast(v) for k, v in pred_stats.items()},
        "method_comparison": method_comparison,
        "precision_at_k": precision_at_k,
        "gf_correlation_euclidean": euc_corr,
        "gf_correlation_cosine": cos_corr,
    }

    return output


def _cast(val):
    """Cast numpy types to Python native for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


# ============================================================
# 9. Entry Point
# ============================================================

def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(BANNER)
    print("  Cosine Similarity Voting Baseline")
    print("  Protein Function Prediction: Euclidean KNN vs Cosine Voting")
    print(BANNER)

    np.random.seed(SEED)

    # --- Stage 1: Alias mapping ---
    print("\n[1/7] Building alias mapping...")
    sgd_to_string, orf_to_string, network_nodes = build_alias_mapping()

    # --- Stage 2: Parse GAF ---
    print("\n[2/7] Parsing GAF (experimental BP)...")
    annotations, ann_stats = parse_gaf_experimental(
        sgd_to_string, orf_to_string, network_nodes
    )

    if len(annotations) < 50:
        print(f"ERROR: Only {len(annotations)} proteins annotated, aborting.")
        return

    # --- Stage 3: Load network ---
    print("\n[3/7] Loading full PPI network...")
    graph = load_full_network()

    # --- Stage 4: Load embeddings ---
    print("\n[4/7] Loading full-network embeddings...")
    embeddings = load_all_full_embeddings(set(graph.nodes()))

    if not embeddings:
        print("ERROR: No embeddings loaded, aborting.")
        return

    # --- Stage 5: Load GF scores ---
    print("\n[5/7] Loading GF scores...")
    with open(str(GF_SCORES_FILE), encoding="utf-8") as f:
        gf_data = json.load(f)
    gf_scores = gf_data.get("scores", {})
    print(f"  GF scores loaded: {len(gf_scores)} methods")
    for m in FULL_METHODS:
        if m in gf_scores:
            print(f"    {m}: {gf_scores[m]:.4f}")

    # --- Stage 6: Run LOTO-CV ---
    print("\n[6/7] Running LOTO-CV with Euclidean KNN and Cosine voting...")
    precision_results, rank_results, pred_stats = run_loto_cv(
        embeddings, annotations
    )

    # --- Stage 7: Evaluate and report ---
    print("\n[7/7] Evaluating results...")

    # Collect per-method MRR and P@10 for both strategies
    results_by_method = {}
    precision_by_method = {}

    print("\n  Per-method results:")
    for method in FULL_METHODS:
        euc_key = f"{method}_euclidean"
        cos_key = f"{method}_cosine"

        euc_mrr = compute_mrr(rank_results, euc_key)
        cos_mrr = compute_mrr(rank_results, cos_key)
        euc_prec = compute_precision_at_k_agg(precision_results, euc_key, K_VALUES)
        cos_prec = compute_precision_at_k_agg(precision_results, cos_key, K_VALUES)

        precision_by_method[euc_key] = euc_prec
        precision_by_method[cos_key] = cos_prec

        results_by_method[method] = {
            "euc_mrr": euc_mrr,
            "cos_mrr": cos_mrr,
            "euc_p10": euc_prec.get(10, 0.0),
            "cos_p10": cos_prec.get(10, 0.0),
        }

        print(f"    {method}: Euc MRR={euc_mrr:.4f}, Cos MRR={cos_mrr:.4f} | "
              f"Euc P@10={euc_prec.get(10, 0):.4f}, Cos P@10={cos_prec.get(10, 0):.4f}")

    # Print comparison table
    print_comparison_table(results_by_method)

    # GF Score correlations
    euc_mrr_for_corr = {m: results_by_method[m]["euc_mrr"] for m in FULL_METHODS if m in results_by_method}
    cos_mrr_for_corr = {m: results_by_method[m]["cos_mrr"] for m in FULL_METHODS if m in results_by_method}

    print("\n  Computing GF Score correlations...")
    euc_corr = compute_gf_correlation(euc_mrr_for_corr, gf_scores, "euclidean")
    cos_corr = compute_gf_correlation(cos_mrr_for_corr, gf_scores, "cosine")

    # Save results
    print("\n  Saving results...")
    output = build_results_json(
        results_by_method, ann_stats, pred_stats,
        euc_corr, cos_corr, precision_by_method,
    )

    result_file = RESULTS / "function_prediction_cosine.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved {result_file}")

    # Final summary
    print(f"\n{BANNER}")
    print("  SUMMARY")
    print(BANNER)
    n_improved = 0
    for method in FULL_METHODS:
        if method not in results_by_method:
            continue
        r = results_by_method[method]
        if r["cos_mrr"] > r["euc_mrr"]:
            n_improved += 1

    print(f"  Cosine voting improved MRR for {n_improved}/{len(results_by_method)} methods")
    print(f"  Euclidean GF-MRR Spearman rho = {euc_corr.get('spearman_rho', 'N/A')}")
    print(f"  Cosine GF-MRR Spearman rho    = {cos_corr.get('spearman_rho', 'N/A')}")
    print(BANNER)


if __name__ == "__main__":
    run()
