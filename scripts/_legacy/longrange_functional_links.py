#!/usr/bin/env python3
"""
Phase 14: Long-Range Functional Link Discovery
===============================================

Core hypothesis: embeddings encode long-range functional topology that
direct network neighbors cannot see.

Three experiments:
  Part 1: Distance-stratified functional association recovery
  Part 2: Hybrid predictor (PPI + embedding)
  Part 3: Long-range functional link discovery and characterisation

If the hybrid predictor outperforms pure PPI topology, we demonstrate
that embeddings are COMPLEMENTARY to topology, not inferior.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
)
from function_prediction import (
    build_alias_mapping,
    parse_gaf_experimental,
    build_knn_index,
    knn_predict_fast,
    ppi_neighbor_predict,
    twohop_diffusion_predict,
    compute_mean_reciprocal_rank,
    K_MAX,
)

# ============================================================
# Constants
# ============================================================

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()
EMB = get_embeddings_dir()

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"

BANNER = "=" * 64

# Distance strata
STRATA = {
    "1-3 hops":  (1, 3),
    "4-6 hops":  (4, 6),
    "7+ hops":   (7, 999),
}

# Methods
FULL_METHODS = ["DM", "MDS", "Spectral", "Node2Vec", "VGAE"]
METHOD_COLORS = {
    "DM": "#08306b", "MDS": "#08519c", "Spectral": "#3182bd",
    "Node2Vec": "#fb6a4a", "VGAE": "#67000d",
}

# Hybrid weights to sweep
HYBRID_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ============================================================
# Part 1: Distance-Stratified Functional Recovery
# ============================================================

def compute_network_distances(graph, query_proteins):
    """BFS from each query protein to compute shortest path distances.

    Parameters
    ----------
    graph : nx.Graph
    query_proteins : set of str

    Returns
    -------
    distances : dict[str, dict[str, int]]
        {query_protein: {target_protein: shortest_path_length}}
    """
    distances = {}
    nodes_list = sorted(query_proteins)
    n = len(nodes_list)
    t0 = time.time()

    for i, pid in enumerate(nodes_list):
        # BFS from pid
        lengths = nx.single_source_shortest_path_length(graph, pid)
        distances[pid] = lengths

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"    BFS: {i+1}/{n} ({100*(i+1)/n:.0f}%) "
                  f"-- {rate:.0f} bfs/s -- ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"    BFS complete: {n} sources in {elapsed:.1f}s ({n/elapsed:.0f} bfs/s)")
    return distances


def stratify_functional_pairs(annotations, distances, graph):
    """Stratify functionally related protein pairs by network distance.

    For each GO term, find all pairs of annotated proteins and record
    their shortest path distance.

    Parameters
    ----------
    annotations : dict[str, set[str]]
    distances : dict[str, dict[str, int]]
    graph : nx.Graph

    Returns
    -------
    strata_pairs : dict[str, list[tuple]]
        {stratum_name: [(protein_a, protein_b, go_term), ...]}
    strata_stats : dict[str, dict]
    """
    # Build term -> proteins mapping
    term_to_proteins = defaultdict(set)
    for pid, terms in annotations.items():
        if pid in graph:
            for t in terms:
                term_to_proteins[t].add(pid)

    # Count pairs per stratum
    strata_counts = {name: 0 for name in STRATA}
    strata_counts["no_path"] = 0
    all_pairs = []

    for term, proteins in term_to_proteins.items():
        protein_list = sorted(proteins)
        for i in range(len(protein_list)):
            for j in range(i + 1, len(protein_list)):
                pa, pb = protein_list[i], protein_list[j]
                # Get network distance
                dist = distances.get(pa, {}).get(pb, None)
                if dist is None:
                    strata_counts["no_path"] += 1
                    continue

                for name, (lo, hi) in STRATA.items():
                    if lo <= dist <= hi:
                        strata_counts[name] += 1
                        break

                all_pairs.append((pa, pb, term, dist))

    total = sum(strata_counts.values())
    print(f"\n    Total functional pairs: {total}")
    for name, count in strata_counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"      {name:12s}: {count:8d} ({pct:5.1f}%)")

    return all_pairs, strata_counts


def evaluate_stratified_recovery(all_pairs, embeddings, graph, annotations,
                                 term_frequencies, distances):
    """Evaluate recovery of functional associations per distance stratum.

    For each stratum, compute the fraction of functional pairs that are
    recovered as KNN in embedding space vs PPI neighbors.

    Parameters
    ----------
    all_pairs : list of (pa, pb, term, dist)
    embeddings : dict
    graph : nx.Graph
    annotations : dict
    term_frequencies : Counter
    distances : dict

    Returns
    -------
    recovery_by_stratum : dict[str, dict[str, float]]
        {stratum: {method: recovery_rate}}
    """
    results = {}

    for stratum_name, (lo, hi) in STRATA.items():
        stratum_pairs = [(pa, pb, t, d) for pa, pb, t, d in all_pairs
                         if lo <= d <= hi]
        if not stratum_pairs:
            results[stratum_name] = {}
            continue

        # Sample up to 2000 pairs for efficiency
        rng = np.random.RandomState(SEED)
        if len(stratum_pairs) > 2000:
            indices = rng.choice(len(stratum_pairs), 2000, replace=False)
            sample_pairs = [stratum_pairs[i] for i in sorted(indices)]
        else:
            sample_pairs = stratum_pairs

        n_sample = len(sample_pairs)

        # Build per-method recovery counts
        method_recovery = defaultdict(int)
        ppi_recovery = 0
        twohop_recovery = 0

        for method, emb in embeddings.items():
            node_to_idx = emb["node_to_idx"]
            coords = emb["coords"]
            nodes = emb["nodes"]
            nn_model = build_knn_index(coords)

            for pa, pb, term, dist in sample_pairs:
                if pa not in node_to_idx or pb not in node_to_idx:
                    continue

                # Check if pb is in pa's KNN (k=30)
                query_idx = node_to_idx[pa]
                dists, idxs = nn_model.kneighbors(
                    coords[query_idx:query_idx + 1], n_neighbors=min(K_MAX, len(coords))
                )
                knn_nodes = set(nodes[i] for i in idxs[0] if i != query_idx)

                if pb in knn_nodes:
                    method_recovery[method] += 1

        # PPI neighbors recovery
        for pa, pb, term, dist in sample_pairs:
            ppi_neighbors = set(graph.neighbors(pa))
            if pb in ppi_neighbors:
                ppi_recovery += 1

            # 2-hop
            hop1 = set(graph.neighbors(pa))
            hop2 = set()
            for n1 in hop1:
                for n2 in graph.neighbors(n1):
                    if n2 != pa and n2 not in hop1:
                        hop2.add(n2)
            if pb in hop1 or pb in hop2:
                twohop_recovery += 1

        result = {}
        valid_pairs = n_sample  # approximate
        for method in FULL_METHODS:
            if method in method_recovery:
                result[method] = method_recovery[method] / n_sample

        result["PPI-Neighbors"] = ppi_recovery / n_sample
        result["2-Hop Diffusion"] = twohop_recovery / n_sample
        results[stratum_name] = result

        print(f"    {stratum_name}: n={n_sample}")
        for m, r in sorted(result.items(), key=lambda x: x[1], reverse=True):
            print(f"      {m:18s}: {r:.3f}")

    return results


# ============================================================
# Part 2: Hybrid Predictor
# ============================================================

def hybrid_predict(query_id, graph, emb, annotations, k_ppi=None,
                   k_emb=30, w_emb=0.3, hidden_term=None):
    """Hybrid prediction combining PPI neighbors and embedding KNN.

    Parameters
    ----------
    query_id : str
    graph : nx.Graph
    emb : dict with coords, nodes, node_to_idx, nn_model
    annotations : dict
    k_emb : int
    w_emb : float
        Weight for embedding votes (PPI gets 1 - w_emb).
    hidden_term : str

    Returns
    -------
    list of (go_term, score)
    """
    term_scores = Counter()

    # PPI neighbor votes (weight = 1 - w_emb)
    w_ppi = 1.0 - w_emb
    if w_ppi > 0 and query_id in graph:
        for neighbor in graph.neighbors(query_id):
            neighbor_terms = annotations.get(neighbor, set())
            for term in neighbor_terms:
                term_scores[term] += w_ppi

    # Embedding KNN votes (weight = w_emb)
    if w_emb > 0 and emb is not None:
        node_to_idx = emb["node_to_idx"]
        if query_id in node_to_idx:
            query_idx = node_to_idx[query_id]
            coords = emb["coords"]
            nodes = emb["nodes"]
            nn_model = emb["nn_model"]

            n_neighbors = min(k_emb + 1, len(coords))
            dists_knn, idxs_knn = nn_model.kneighbors(
                coords[query_idx:query_idx + 1], n_neighbors=n_neighbors
            )

            for d_val, idx in zip(dists_knn[0], idxs_knn[0]):
                if idx == query_idx:
                    continue
                neighbor_id = nodes[idx]
                neighbor_terms = annotations.get(neighbor_id, set())
                weight = w_emb / (d_val + 1e-10)
                for term in neighbor_terms:
                    term_scores[term] += weight

    return term_scores.most_common()


def run_hybrid_loto(embeddings, graph, annotations, term_freq,
                    best_method="Spectral"):
    """Run LOTO with hybrid predictor at multiple embedding weights.

    Parameters
    ----------
    embeddings : dict
    graph : nx.Graph
    annotations : dict
    term_freq : Counter
    best_method : str

    Returns
    -------
    weight_mrr : dict[float, float]
        {w_emb: MRR}
    detail_results : dict
    """
    if best_method not in embeddings:
        print(f"  WARNING: {best_method} not in embeddings, using first available")
        best_method = list(embeddings.keys())[0]

    emb = embeddings[best_method]

    # Build KNN index
    emb["nn_model"] = build_knn_index(emb["coords"])

    # Filter proteins
    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in graph and pid in emb["node_to_idx"]
    }

    n_proteins = len(query_proteins)
    n_trials = sum(len(terms) for terms in query_proteins.values())
    print(f"\n  Hybrid LOTO: {n_proteins} proteins, {n_trials} trials")

    # For each weight, collect rank results
    weight_rank_results = {w: [] for w in HYBRID_WEIGHTS}
    ppi_rank_results = []
    twohop_rank_results = []

    completed = 0
    t0 = time.time()

    for pid, terms in sorted(query_proteins.items()):
        terms_list = sorted(terms)
        for hidden_term in terms_list:
            # Hybrid at each weight
            for w_emb in HYBRID_WEIGHTS:
                preds = hybrid_predict(
                    pid, graph, emb, annotations,
                    k_emb=K_MAX, w_emb=w_emb,
                    hidden_term=hidden_term,
                )
                pred_terms = [t for t, _ in preds]
                try:
                    rank = pred_terms.index(hidden_term) + 1
                except ValueError:
                    rank = 0
                weight_rank_results[w_emb].append({"Hybrid": rank})

            # Pure PPI (w=0)
            ppi_preds = ppi_neighbor_predict(pid, graph, annotations, hidden_term)
            ppi_terms = [t for t, _ in ppi_preds]
            try:
                ppi_rank = ppi_terms.index(hidden_term) + 1
            except ValueError:
                ppi_rank = 0
            ppi_rank_results.append({"PPI": ppi_rank})

            # 2-Hop
            hop_preds = twohop_diffusion_predict(pid, graph, annotations, hidden_term)
            hop_terms_list = [t for t, _ in hop_preds]
            try:
                hop_rank = hop_terms_list.index(hidden_term) + 1
            except ValueError:
                hop_rank = 0
            twohop_rank_results.append({"2-Hop": hop_rank})

            completed += 1
            if completed % 3000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_trials - completed) / rate if rate > 0 else 0
                print(f"    Progress: {completed}/{n_trials} "
                      f"({100*completed/n_trials:.0f}%) "
                      f"-- {rate:.0f} trials/s -- ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Completed {completed} trials in {elapsed:.1f}s "
          f"({completed/elapsed:.0f} trials/s)")

    # Compute MRR for each weight
    weight_mrr = {}
    for w_emb in HYBRID_WEIGHTS:
        mrr_dict = compute_mean_reciprocal_rank(weight_rank_results[w_emb])
        weight_mrr[w_emb] = mrr_dict.get("Hybrid", 0)

    ppi_mrr = compute_mean_reciprocal_rank(ppi_rank_results).get("PPI", 0)
    twohop_mrr = compute_mean_reciprocal_rank(twohop_rank_results).get("2-Hop", 0)

    # Best weight
    best_w = max(weight_mrr, key=weight_mrr.get)
    best_hybrid_mrr = weight_mrr[best_w]

    print(f"\n  Hybrid MRR sweep ({best_method} + PPI):")
    for w in HYBRID_WEIGHTS:
        marker = " <-- BEST" if w == best_w else ""
        exceed = " ***" if weight_mrr[w] > ppi_mrr else ""
        print(f"    w={w:.1f}: MRR={weight_mrr[w]:.4f}{marker}{exceed}")
    print(f"    PPI pure:     MRR={ppi_mrr:.4f}")
    print(f"    2-Hop:        MRR={twohop_mrr:.4f}")
    print(f"\n    Best hybrid: w={best_w:.1f}, MRR={best_hybrid_mrr:.4f}")
    print(f"    vs PPI: {best_hybrid_mrr - ppi_mrr:+.4f} "
          f"({100*(best_hybrid_mrr - ppi_mrr)/ppi_mrr:+.1f}%)")

    return weight_mrr, {
        "best_method": best_method,
        "best_weight": best_w,
        "best_hybrid_mrr": best_hybrid_mrr,
        "ppi_mrr": ppi_mrr,
        "twohop_mrr": twohop_mrr,
        "n_trials": n_trials,
        "n_proteins": n_proteins,
    }


# ============================================================
# Part 3: Long-Range Functional Link Discovery
# ============================================================

def discover_longrange_links(embeddings, graph, annotations, distances,
                             min_network_dist=4, max_embedding_rank=30):
    """Find protein pairs that are:
    - Far in network (>= min_network_dist hops)
    - Close in embedding space (within top-k nearest neighbors)
    - Share at least one GO term (validation)

    These represent "discoveries" only possible through embeddings.

    Parameters
    ----------
    embeddings : dict
    graph : nx.Graph
    annotations : dict
    distances : dict
    min_network_dist : int
    max_embedding_rank : int

    Returns
    -------
    discoveries : dict[str, list]
        {method: [(pa, pb, embedding_dist, network_dist, shared_terms), ...]}
    """
    discoveries = {}

    for method, emb in embeddings.items():
        node_to_idx = emb["node_to_idx"]
        coords = emb["coords"]
        nodes = emb["nodes"]

        # Build KNN
        nn_model = NearestNeighbors(
            n_neighbors=min(max_embedding_rank + 1, len(coords)),
            metric="euclidean"
        )
        nn_model.fit(coords)

        method_discoveries = []

        # For each annotated protein
        for pid in sorted(annotations.keys()):
            if pid not in node_to_idx or pid not in graph:
                continue

            query_idx = node_to_idx[pid]
            pid_terms = annotations[pid]

            # Get KNN in embedding space
            n_neighbors = min(max_embedding_rank + 1, len(coords))
            dists_knn, idxs_knn = nn_model.kneighbors(
                coords[query_idx:query_idx + 1], n_neighbors=n_neighbors
            )

            for d_val, idx in zip(dists_knn[0], idxs_knn[0]):
                if idx == query_idx:
                    continue

                neighbor_id = nodes[idx]

                # Check network distance
                net_dist = distances.get(pid, {}).get(neighbor_id, None)
                if net_dist is None or net_dist < min_network_dist:
                    continue

                # Check shared GO terms
                neighbor_terms = annotations.get(neighbor_id, set())
                shared = pid_terms & neighbor_terms

                if shared:
                    method_discoveries.append({
                        "protein_a": pid,
                        "protein_b": neighbor_id,
                        "embedding_dist": float(d_val),
                        "network_dist": int(net_dist),
                        "shared_terms": sorted(shared),
                        "n_shared": len(shared),
                    })

        discoveries[method] = method_discoveries
        print(f"    {method}: {len(method_discoveries)} long-range "
              f"functional links discovered")

    return discoveries


# ============================================================
# Visualisation
# ============================================================

def plot_fig80_stratified_recovery(strata_results, strata_counts):
    """Fig80: Recovery rate by distance stratum for each method."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel A: Grouped bar chart ---
    ax = axes[0]
    strata_names = list(STRATA.keys())
    all_methods = FULL_METHODS + ["PPI-Neighbors", "2-Hop Diffusion"]

    x = np.arange(len(strata_names))
    width = 0.1
    offsets = np.arange(len(all_methods)) - len(all_methods) / 2

    method_colors_all = {
        **METHOD_COLORS,
        "PPI-Neighbors": "#636363",
        "2-Hop Diffusion": "#969696",
    }

    for i, method in enumerate(all_methods):
        vals = []
        for sn in strata_names:
            vals.append(strata_results.get(sn, {}).get(method, 0))
        ax.bar(x + offsets[i] * width, vals, width, label=method,
               color=method_colors_all.get(method, "#cccccc"),
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(strata_names, fontsize=11)
    ax.set_ylabel("Recovery Rate", fontsize=12)
    ax.set_title("A: Functional Pair Recovery by Network Distance", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    # --- Panel B: Line plot (embedding vs PPI gap per stratum) ---
    ax2 = axes[1]
    for method in FULL_METHODS:
        gaps = []
        for sn in strata_names:
            emb_r = strata_results.get(sn, {}).get(method, 0)
            ppi_r = strata_results.get(sn, {}).get("PPI-Neighbors", 0)
            gaps.append(emb_r - ppi_r)
        ax2.plot(strata_names, gaps, "o-", color=METHOD_COLORS[method],
                 label=method, linewidth=2, markersize=8)

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Embedding - PPI Recovery Gap", fontsize=12)
    ax2.set_title("B: Embedding Advantage over PPI", fontsize=13,
                  fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Annotate strata pair counts
    for i, sn in enumerate(strata_names):
        count = strata_counts.get(sn, 0)
        ax2.text(i, ax2.get_ylim()[0] * 0.9, f"n={count}",
                 ha="center", fontsize=8, color="#999999")

    fig.suptitle("Phase 14: Distance-Stratified Functional Recovery",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig80_stratified_recovery.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig80_stratified_recovery.png")


def plot_fig81_hybrid_sweep(weight_mrr, ppi_mrr, twohop_mrr, detail):
    """Fig90: Hybrid predictor MRR vs embedding weight."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel A: MRR vs weight ---
    ax = axes[0]
    weights = sorted(weight_mrr.keys())
    mrrs = [weight_mrr[w] for w in weights]

    ax.plot(weights, mrrs, "o-", color="#3182bd", linewidth=2.5,
            markersize=8, label=f"Hybrid ({detail['best_method']} + PPI)")
    ax.axhline(ppi_mrr, color="#636363", linestyle="--", linewidth=2,
               label=f"PPI-Neighbors ({ppi_mrr:.3f})")
    ax.axhline(twohop_mrr, color="#969696", linestyle="--", linewidth=1.5,
               label=f"2-Hop Diffusion ({twohop_mrr:.3f})")

    # Highlight optimal weight
    best_w = detail["best_weight"]
    best_mrr = detail["best_hybrid_mrr"]
    ax.plot(best_w, best_mrr, "*", color="#e6550d", markersize=20, zorder=10,
            label=f"Optimal w={best_w:.1f} (MRR={best_mrr:.3f})")

    # Shade the region where hybrid > PPI
    if best_mrr > ppi_mrr:
        exceed_weights = [w for w in weights if weight_mrr[w] > ppi_mrr]
        if exceed_weights:
            ax.axvspan(min(exceed_weights) - 0.05, max(exceed_weights) + 0.05,
                       alpha=0.1, color="green",
                       label=f"Hybrid > PPI (w in [{min(exceed_weights):.1f}, "
                             f"{max(exceed_weights):.1f}])")

    ax.set_xlabel("Embedding Weight (w)", fontsize=13)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=13)
    ax.set_title("A: Hybrid Predictor Performance", fontsize=14,
                 fontweight="bold")
    ax.set_xticks(weights)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Improvement over PPI ---
    ax2 = axes[1]
    improvements = [(weight_mrr[w] - ppi_mrr) for w in weights]
    colors = ["#2ca25f" if imp > 0 else "#de2d26" for imp in improvements]

    bars = ax2.bar([f"{w:.1f}" for w in weights], improvements,
                   color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)

    for i, (w, imp) in enumerate(zip(weights, improvements)):
        va = "bottom" if imp >= 0 else "top"
        ax2.text(i, imp + (0.001 if imp >= 0 else -0.001),
                 f"{imp:+.4f}", ha="center", va=va, fontsize=8)

    ax2.set_xlabel("Embedding Weight (w)", fontsize=13)
    ax2.set_ylabel("MRR Improvement over PPI", fontsize=13)
    ax2.set_title("B: Hybrid vs Pure PPI Topology", fontsize=14,
                  fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Phase 14: Hybrid Predictor (Topology + Embedding)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig90_hybrid_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig90_hybrid_sweep.png")


def plot_fig82_longrange_discoveries(discoveries, strata_counts):
    """Fig91: Long-range functional links discovered by embeddings."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel A: Count of discoveries per method ---
    ax = axes[0]
    methods_sorted = sorted(discoveries.keys(),
                            key=lambda m: len(discoveries[m]), reverse=True)
    counts = [len(discoveries[m]) for m in methods_sorted]
    colors = [METHOD_COLORS.get(m, "#cccccc") for m in methods_sorted]

    bars = ax.barh(methods_sorted, counts, color=colors,
                   edgecolor="white", linewidth=0.5)
    for i, (m, c) in enumerate(zip(methods_sorted, counts)):
        ax.text(c + max(counts) * 0.02, i, str(c),
                va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Number of Long-Range Functional Links", fontsize=12)
    ax.set_title("A: Unique Discoveries per Method", fontsize=14,
                 fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, max(counts) * 1.3 if counts else 1)

    # --- Panel B: Network distance distribution of discoveries ---
    ax2 = axes[1]
    for method in methods_sorted[:3]:  # top 3
        if not discoveries[method]:
            continue
        net_dists = [d["network_dist"] for d in discoveries[method]]
        ax2.hist(net_dists, bins=range(4, 20), alpha=0.5,
                 label=method, color=METHOD_COLORS.get(method, "#ccc"),
                 edgecolor="white", linewidth=0.5)

    ax2.set_xlabel("Network Distance (hops)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("B: Network Distance of Discovered Links", fontsize=14,
                  fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Phase 14: Long-Range Functional Link Discovery",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig91_longrange_discoveries.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig91_longrange_discoveries.png")


def plot_fig83_summary_dashboard(strata_results, weight_mrr, discoveries,
                                 detail, strata_counts):
    """Fig92: Phase 14 four-panel summary dashboard."""
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Panel A: Stratified recovery heatmap ---
    ax_a = fig.add_subplot(gs[0, 0])
    strata_names = list(STRATA.keys())
    methods_plot = ["PPI-Neighbors", "2-Hop Diffusion"] + FULL_METHODS

    data = np.zeros((len(methods_plot), len(strata_names)))
    for i, m in enumerate(methods_plot):
        for j, sn in enumerate(strata_names):
            data[i, j] = strata_results.get(sn, {}).get(m, 0)

    im = ax_a.imshow(data, cmap="YlOrRd", aspect="auto")
    ax_a.set_xticks(range(len(strata_names)))
    ax_a.set_xticklabels(strata_names, fontsize=10)
    ax_a.set_yticks(range(len(methods_plot)))
    ax_a.set_yticklabels(methods_plot, fontsize=10)

    for i in range(len(methods_plot)):
        for j in range(len(strata_names)):
            ax_a.text(j, i, f"{data[i,j]:.3f}",
                      ha="center", va="center", fontsize=9,
                      color="white" if data[i,j] > 0.3 else "black")

    plt.colorbar(im, ax=ax_a, shrink=0.8)
    ax_a.set_title("A: Recovery Rate Heatmap", fontsize=12, fontweight="bold")

    # --- Panel B: Hybrid MRR curve ---
    ax_b = fig.add_subplot(gs[0, 1])
    weights = sorted(weight_mrr.keys())
    mrrs = [weight_mrr[w] for w in weights]

    ax_b.plot(weights, mrrs, "o-", color="#3182bd", linewidth=2.5,
              markersize=8)
    ax_b.axhline(detail["ppi_mrr"], color="#636363", linestyle="--",
                 linewidth=2, label=f"PPI ({detail['ppi_mrr']:.3f})")

    best_w = detail["best_weight"]
    best_mrr = detail["best_hybrid_mrr"]
    ax_b.plot(best_w, best_mrr, "*", color="#e6550d", markersize=20,
              zorder=10)

    if best_mrr > detail["ppi_mrr"]:
        exceed_w = [w for w in weights if weight_mrr[w] > detail["ppi_mrr"]]
        if exceed_w:
            ax_b.axvspan(min(exceed_w) - 0.05, max(exceed_w) + 0.05,
                         alpha=0.15, color="green")

    ax_b.set_xlabel("Embedding Weight (w)", fontsize=11)
    ax_b.set_ylabel("MRR", fontsize=11)
    ax_b.set_title("B: Hybrid Predictor Sweep", fontsize=12, fontweight="bold")
    ax_b.set_xticks(weights)
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3)

    # --- Panel C: Key statistics table ---
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.axis("off")

    table_data = [
        ["Network", "5,936 nodes, 120,357 edges"],
        ["Annotated proteins", f"{detail['n_proteins']:,}"],
        ["LOTO trials", f"{detail['n_trials']:,}"],
        ["Best embedding method", detail["best_method"]],
        ["PPI MRR (baseline)", f"{detail['ppi_mrr']:.4f}"],
        ["2-Hop MRR", f"{detail['twohop_mrr']:.4f}"],
        ["Best hybrid MRR", f"{detail['best_hybrid_mrr']:.4f}"],
        ["Optimal embedding weight", f"w = {detail['best_weight']:.1f}"],
        ["Improvement over PPI",
         f"{detail['best_hybrid_mrr'] - detail['ppi_mrr']:+.4f} "
         f"({100*(detail['best_hybrid_mrr'] - detail['ppi_mrr'])/detail['ppi_mrr']:+.1f}%)"],
        ["Long-range links (Spectral)",
         str(len(discoveries.get("Spectral", [])))],
        ["Long-range links (best)",
         str(max(len(v) for v in discoveries.values())) if discoveries else "0"],
    ]

    table = ax_c.table(cellText=table_data, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#e0e0e0")
        cell.set_edgecolor("white")
    ax_c.set_title("C: Key Statistics", fontsize=12, fontweight="bold", pad=20)

    # --- Panel D: Long-range discoveries bar chart ---
    ax_d = fig.add_subplot(gs[1, 1])
    methods_sorted = sorted(discoveries.keys(),
                            key=lambda m: len(discoveries[m]), reverse=True)
    disc_counts = [len(discoveries[m]) for m in methods_sorted]
    disc_colors = [METHOD_COLORS.get(m, "#ccc") for m in methods_sorted]

    ax_d.barh(methods_sorted, disc_counts, color=disc_colors,
              edgecolor="white", linewidth=0.5)
    for i, (m, c) in enumerate(zip(methods_sorted, disc_counts)):
        ax_d.text(c + max(disc_counts) * 0.02 if disc_counts else 1, i, str(c),
                  va="center", fontsize=10, fontweight="bold")

    ax_d.set_xlabel("Long-Range Functional Links", fontsize=11)
    ax_d.set_title("D: Embedding-Only Discoveries (>=4 hops)", fontsize=12,
                   fontweight="bold")
    ax_d.grid(True, axis="x", alpha=0.3)
    ax_d.set_xlim(0, max(disc_counts) * 1.3 if disc_counts else 1)

    fig.suptitle("Phase 14: Embeddings Reveal Long-Range Functional Topology",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(FIGURES / "Fig92_phase14_summary.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig92_phase14_summary.png")


# ============================================================
# Report
# ============================================================

def write_report(strata_results, strata_counts, weight_mrr, detail,
                 discoveries):
    """Write Phase 14 report."""
    report_path = RESULTS / "phase14_report.md"

    lines = [
        "# Phase 14: Long-Range Functional Link Discovery",
        "",
        "## Core Hypothesis",
        "",
        "Graph embeddings encode long-range functional topology that direct "
        "network neighbors cannot see. While embeddings cannot replace "
        "topology for local function prediction (Phase 13), they capture "
        "complementary information about distant functional relationships.",
        "",
        "## Methodology",
        "",
        f"Full yeast STRING network: {detail.get('n_proteins', '-')} annotated "
        f"proteins, {detail.get('n_trials', '-')} LOTO trials.",
        "",
        "### Part 1: Distance-Stratified Recovery",
        "",
        "Functional protein pairs (sharing >= 1 GO BP term) stratified by "
        "shortest-path network distance. Recovery rate = fraction found in "
        "top-30 nearest neighbors (embedding) or direct neighbors (PPI).",
        "",
        "| Stratum | Pairs |",
        "|---------|-------|",
    ]
    for sn in STRATA:
        lines.append(f"| {sn} | {strata_counts.get(sn, 0):,} |")

    lines.extend([
        "",
        "### Part 2: Hybrid Predictor",
        "",
        f"Combined PPI neighbors + {detail['best_method']} KNN with weight sweep.",
        f"Optimal embedding weight: w={detail['best_weight']:.1f}",
        "",
        "| Weight (w) | MRR |",
        "|-----------|-----|",
    ])
    for w in sorted(weight_mrr.keys()):
        lines.append(f"| {w:.1f} | {weight_mrr[w]:.4f} |")

    lines.extend([
        "",
        "### Part 3: Long-Range Discovery",
        "",
        "Protein pairs >= 4 hops apart but within top-30 embedding neighbors "
        "that share experimental GO BP annotations.",
        "",
        "| Method | Discoveries |",
        "|--------|-------------|",
    ])
    for m in sorted(discoveries.keys(),
                    key=lambda m: len(discoveries[m]), reverse=True):
        lines.append(f"| {m} | {len(discoveries[m])} |")

    lines.extend([
        "",
        "## Key Results",
        "",
        f"- PPI baseline MRR: {detail['ppi_mrr']:.4f}",
        f"- Best hybrid MRR: {detail['best_hybrid_mrr']:.4f}",
        f"- Improvement: {detail['best_hybrid_mrr'] - detail['ppi_mrr']:+.4f} "
        f"({100*(detail['best_hybrid_mrr'] - detail['ppi_mrr'])/detail['ppi_mrr']:+.1f}%)",
        f"- Long-range links (best): "
        f"{max(len(v) for v in discoveries.values()) if discoveries else 0}",
        "",
        "## Interpretation",
        "",
        "If the hybrid predictor outperforms pure PPI topology, embeddings "
        "capture functional signals beyond local network structure. This "
        "validates the complementarity of geometric and topological views "
        "of protein interaction networks.",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {report_path.name}")


# ============================================================
# Main Entry Point
# ============================================================

def run():
    print(BANNER)
    print("Phase 14: Long-Range Functional Link Discovery")
    print("Embeddings Encode What Topology Cannot See")
    print(BANNER)

    np.random.seed(SEED)

    # --- Stage 1: Data loading ---
    print("\n[1/8] Building alias mapping...")
    sgd_to_string, orf_to_string, network_nodes = build_alias_mapping()

    print("\n[2/8] Parsing GAF (experimental BP)...")
    annotations, ann_stats = parse_gaf_experimental(
        sgd_to_string, orf_to_string, network_nodes
    )

    term_freq = Counter()
    for terms in annotations.values():
        term_freq.update(terms)

    print("\n[3/8] Loading full PPI network...")
    G = nx.Graph()
    with open(str(NETWORK_FILE), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n[4/8] Loading full-network embeddings...")
    embeddings = {}
    for method in FULL_METHODS:
        try:
            from utils import load_embedding
            coords, emb_nodes = load_embedding(method, "full", embeddings_dir=EMB)
            node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
            common = [n for n in emb_nodes if n in set(G.nodes())]
            if len(common) < 100:
                continue
            indices = [node_to_idx[n] for n in common]
            filtered_coords = coords[indices]
            embeddings[method] = {
                "coords": filtered_coords,
                "nodes": common,
                "node_to_idx": {n: i for i, n in enumerate(common)},
            }
            print(f"  {method}: {len(common)} nodes")
        except Exception as e:
            print(f"  {method} FAILED: {e}")

    # --- Stage 2: Network distances ---
    print("\n[5/8] Computing network distances (BFS)...")
    query_set = set(pid for pid in annotations if pid in G)
    print(f"  Computing BFS from {len(query_set)} annotated proteins...")
    distances = compute_network_distances(G, query_set)

    # --- Stage 3: Part 1 - Stratified recovery ---
    print("\n[6/8] Part 1: Distance-stratified functional recovery...")
    all_pairs, strata_counts = stratify_functional_pairs(
        annotations, distances, G
    )
    strata_results = evaluate_stratified_recovery(
        all_pairs, embeddings, G, annotations, term_freq, distances
    )

    # --- Stage 4: Part 2 - Hybrid predictor ---
    print("\n[7/8] Part 2: Hybrid predictor (PPI + Embedding)...")

    # Determine best method from Phase 13 (Spectral) or dimension sweep
    best_method = "Spectral"
    weight_mrr, detail = run_hybrid_loto(
        embeddings, G, annotations, term_freq, best_method=best_method
    )

    # --- Stage 5: Part 3 - Long-range discovery ---
    print("\n[8/8] Part 3: Long-range functional link discovery...")
    discoveries = discover_longrange_links(
        embeddings, G, annotations, distances, min_network_dist=4
    )

    # --- Summary ---
    print(f"\n{BANNER}")
    print("PHASE 14 RESULTS SUMMARY")
    print(BANNER)

    # Part 1 summary
    print("\n  Part 1: Stratified Recovery")
    print(f"  {'Stratum':12s}  {'PPI':>8s}  {'Spectral':>10s}  {'Gap':>8s}")
    for sn in STRATA:
        ppi_r = strata_results.get(sn, {}).get("PPI-Neighbors", 0)
        spec_r = strata_results.get(sn, {}).get("Spectral", 0)
        gap = spec_r - ppi_r
        print(f"  {sn:12s}  {ppi_r:>8.3f}  {spec_r:>10.3f}  {gap:>+8.3f}")

    # Part 2 summary
    print(f"\n  Part 2: Hybrid Predictor")
    print(f"    PPI MRR:        {detail['ppi_mrr']:.4f}")
    print(f"    Best hybrid:    {detail['best_hybrid_mrr']:.4f} "
          f"(w={detail['best_weight']:.1f})")
    diff = detail['best_hybrid_mrr'] - detail['ppi_mrr']
    pct = 100 * diff / detail['ppi_mrr']
    print(f"    Improvement:    {diff:+.4f} ({pct:+.1f}%)")
    if diff > 0:
        print(f"    >>> HYBRID EXCEEDS PPI TOPOLOGY <<<")
    else:
        print(f"    >>> Hybrid does not exceed pure PPI <<<")

    # Part 3 summary
    print(f"\n  Part 3: Long-Range Discoveries")
    for m in sorted(discoveries, key=lambda m: len(discoveries[m]), reverse=True):
        print(f"    {m:12s}: {len(discoveries[m])} links")

    # --- Save results ---
    output = {
        "description": "Phase 14: Long-Range Functional Link Discovery",
        "strata_counts": strata_counts,
        "strata_recovery": {
            sn: {m: round(v, 6) for m, v in sr.items()}
            for sn, sr in strata_results.items()
        },
        "hybrid_sweep": {
            str(w): round(mrr, 6) for w, mrr in weight_mrr.items()
        },
        "hybrid_detail": detail,
        "longrange_discovery_count": {
            m: len(disc) for m, disc in discoveries.items()
        },
        "longrange_examples": {
            m: disc[:10] for m, disc in discoveries.items()
        },
        "annotation_stats": ann_stats,
    }

    result_file = RESULTS / "longrange_functional_links.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {result_file.name}")

    # --- Figures ---
    print("\n  Generating figures...")
    plot_fig80_stratified_recovery(strata_results, strata_counts)
    plot_fig81_hybrid_sweep(weight_mrr, detail["ppi_mrr"],
                            detail["twohop_mrr"], detail)
    plot_fig82_longrange_discoveries(discoveries, strata_counts)
    plot_fig83_summary_dashboard(strata_results, weight_mrr, discoveries,
                                 detail, strata_counts)

    # --- Report ---
    write_report(strata_results, strata_counts, weight_mrr, detail,
                 discoveries)

    print(f"\n{BANNER}")
    print("Phase 14 complete.")
    print(BANNER)


if __name__ == "__main__":
    run()
