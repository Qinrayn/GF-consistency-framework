#!/usr/bin/env python3
"""
Phase 14b: Fixed Hybrid Predictor + Discovery Quality Analysis
===============================================================

Fixes the score-normalization issue in the original hybrid by using:
  1. Rank-based fallback: PPI first, embedding fills gaps
  2. Rank aggregation (Borda count): scale-invariant combination
  3. Long-range discovery quality analysis
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
)
from function_prediction import (
    build_alias_mapping,
    parse_gaf_experimental,
    ppi_neighbor_predict,
    compute_mean_reciprocal_rank,
    K_MAX,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()
EMB = get_embeddings_dir()
NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"
BANNER = "=" * 64

FULL_METHODS = ["DM", "MDS", "Spectral", "Node2Vec", "VGAE"]
METHOD_COLORS = {
    "DM": "#08306b", "MDS": "#08519c", "Spectral": "#3182bd",
    "Node2Vec": "#fb6a4a", "VGAE": "#67000d",
}


# ============================================================
# Hybrid Predictors
# ============================================================

def rank_fallback_predict(query_id, graph, emb, annotations,
                          k_emb=30, hidden_term=None):
    """Hybrid via rank fallback: PPI predicts first, embedding fills gaps.

    The final ranking is:
      1. PPI neighbor predictions (ranked by vote count)
      2. Embedding KNN predictions that PPI DIDN'T make (appended)

    This guarantees: if PPI finds the term, it's found.
    If only embedding finds it, it's still recovered (at a lower rank).
    """
    # PPI predictions
    ppi_preds = ppi_neighbor_predict(query_id, graph, annotations, hidden_term)
    ppi_terms = [t for t, _ in ppi_preds]
    ppi_set = set(ppi_terms)

    # Embedding predictions
    emb_preds = []
    if emb is not None:
        node_to_idx = emb["node_to_idx"]
        if query_id in node_to_idx:
            query_idx = node_to_idx[query_id]
            nn_model = emb["nn_model"]
            coords = emb["coords"]
            nodes = emb["nodes"]
            n_neighbors = min(k_emb + 1, len(coords))
            dists, idxs = nn_model.kneighbors(
                coords[query_idx:query_idx + 1], n_neighbors=n_neighbors
            )
            term_scores = Counter()
            for d_val, idx in zip(dists[0], idxs[0]):
                if idx == query_idx:
                    continue
                neighbor_id = nodes[idx]
                weight = 1.0 / (d_val + 1e-10)
                for term in annotations.get(neighbor_id, set()):
                    term_scores[term] += weight
            emb_preds = term_scores.most_common()

    # Combine: PPI terms first, then embedding-only terms
    combined = list(ppi_preds)
    emb_only = [(t, s) for t, s in emb_preds if t not in ppi_set]
    combined.extend(emb_only)

    return combined


def rank_aggregation_predict(query_id, graph, emb, annotations,
                             k_emb=30, w_emb=0.3, hidden_term=None):
    """Hybrid via Borda rank aggregation.

    Each term gets a combined rank score:
      score = (1-w) * (1/rank_ppi) + w * (1/rank_emb)

    rank = 0 (not found) contributes 0.
    """
    # PPI predictions -> term to rank
    ppi_preds = ppi_neighbor_predict(query_id, graph, annotations, hidden_term)
    ppi_rank = {}
    for i, (t, _) in enumerate(ppi_preds):
        ppi_rank[t] = i + 1  # 1-indexed

    # Embedding predictions -> term to rank
    emb_rank = {}
    if emb is not None:
        node_to_idx = emb["node_to_idx"]
        if query_id in node_to_idx:
            query_idx = node_to_idx[query_id]
            nn_model = emb["nn_model"]
            coords = emb["coords"]
            nodes = emb["nodes"]
            n_neighbors = min(k_emb + 1, len(coords))
            dists, idxs = nn_model.kneighbors(
                coords[query_idx:query_idx + 1], n_neighbors=n_neighbors
            )
            term_scores = Counter()
            for d_val, idx in zip(dists[0], idxs[0]):
                if idx == query_idx:
                    continue
                neighbor_id = nodes[idx]
                weight = 1.0 / (d_val + 1e-10)
                for term in annotations.get(neighbor_id, set()):
                    term_scores[term] += weight
            emb_preds = term_scores.most_common()
            for i, (t, _) in enumerate(emb_preds):
                emb_rank[t] = i + 1

    # Combine all terms
    all_terms = set(ppi_rank.keys()) | set(emb_rank.keys())
    combined_scores = []
    w_ppi = 1.0 - w_emb
    for t in all_terms:
        score = 0.0
        if t in ppi_rank:
            score += w_ppi / ppi_rank[t]
        if t in emb_rank:
            score += w_emb / emb_rank[t]
        combined_scores.append((t, score))

    combined_scores.sort(key=lambda x: x[1], reverse=True)
    return combined_scores


# ============================================================
# Main Experiment
# ============================================================

def run():
    print(BANNER)
    print("Phase 14b: Fixed Hybrid + Discovery Quality")
    print(BANNER)

    np.random.seed(SEED)

    # Load data
    print("\n[1/6] Loading data...")
    sgd_to_string, orf_to_string, network_nodes = build_alias_mapping()
    annotations, ann_stats = parse_gaf_experimental(
        sgd_to_string, orf_to_string, network_nodes
    )
    term_freq = Counter()
    for terms in annotations.values():
        term_freq.update(terms)

    G = nx.Graph()
    with open(str(NETWORK_FILE), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load embeddings
    print("\n[2/6] Loading embeddings...")
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
            # Pre-build KNN
            nn_model = NearestNeighbors(
                n_neighbors=min(K_MAX + 1, len(filtered_coords)),
                metric="euclidean"
            )
            nn_model.fit(filtered_coords)
            embeddings[method]["nn_model"] = nn_model
            print(f"  {method}: {len(common)} nodes")
        except Exception as e:
            print(f"  {method} FAILED: {e}")

    # Query proteins
    query_proteins = {}
    for method in FULL_METHODS:
        if method in embeddings:
            emb = embeddings[method]
            for pid, terms in annotations.items():
                if (len(terms) >= 2 and pid in G
                        and pid in emb["node_to_idx"]
                        and pid not in query_proteins):
                    query_proteins[pid] = terms

    n_trials = sum(len(terms) for terms in query_proteins.values())
    print(f"\n  Query proteins: {len(query_proteins)}")
    print(f"  LOTO trials: {n_trials}")

    # ================================================================
    # Part 1: Rank-based fallback hybrid (per method)
    # ================================================================
    print("\n[3/6] Rank-based fallback hybrid...")

    method_results = {}

    for method in FULL_METHODS:
        if method not in embeddings:
            continue

        emb = embeddings[method]
        rank_results_fallback = []
        rank_results_ppi = []
        rank_results_emb = []

        completed = 0
        t0 = time.time()

        for pid, terms in sorted(query_proteins.items()):
            if pid not in emb["node_to_idx"]:
                continue
            for hidden_term in sorted(terms):
                # Fallback hybrid
                preds = rank_fallback_predict(
                    pid, G, emb, annotations,
                    k_emb=K_MAX, hidden_term=hidden_term,
                )
                pred_terms = [t for t, _ in preds]
                try:
                    rank_fb = pred_terms.index(hidden_term) + 1
                except ValueError:
                    rank_fb = 0
                rank_results_fallback.append({"Fallback": rank_fb})

                # Pure PPI
                ppi_preds = ppi_neighbor_predict(pid, G, annotations, hidden_term)
                ppi_t = [t for t, _ in ppi_preds]
                try:
                    rank_ppi = ppi_t.index(hidden_term) + 1
                except ValueError:
                    rank_ppi = 0
                rank_results_ppi.append({"PPI": rank_ppi})

                # Pure embedding
                emb_t = [t for t, _ in preds[len(ppi_preds):]]  # emb-only
                # Actually, use full embedding predictions
                node_to_idx = emb["node_to_idx"]
                query_idx = node_to_idx[pid]
                nn_model = emb["nn_model"]
                coords = emb["coords"]
                nodes = emb["nodes"]
                n_neighbors = min(K_MAX + 1, len(coords))
                dists_knn, idxs_knn = nn_model.kneighbors(
                    coords[query_idx:query_idx + 1], n_neighbors=n_neighbors
                )
                ts = Counter()
                for d_val, idx in zip(dists_knn[0], idxs_knn[0]):
                    if idx == query_idx:
                        continue
                    nid = nodes[idx]
                    w = 1.0 / (d_val + 1e-10)
                    for term in annotations.get(nid, set()):
                        ts[term] += w
                emb_preds_ranked = ts.most_common()
                emb_t_full = [t for t, _ in emb_preds_ranked]
                try:
                    rank_emb = emb_t_full.index(hidden_term) + 1
                except ValueError:
                    rank_emb = 0
                rank_results_emb.append({"Embedding": rank_emb})

                completed += 1
                if completed % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (n_trials - completed) / rate
                    print(f"    {method}: {completed}/{n_trials} "
                          f"({100*completed/n_trials:.0f}%) "
                          f"-- {rate:.0f}/s -- ETA {eta:.0f}s")

        elapsed = time.time() - t0
        mrr_fb = compute_mean_reciprocal_rank(rank_results_fallback).get("Fallback", 0)
        mrr_ppi = compute_mean_reciprocal_rank(rank_results_ppi).get("PPI", 0)
        mrr_emb = compute_mean_reciprocal_rank(rank_results_emb).get("Embedding", 0)

        method_results[method] = {
            "fallback_mrr": mrr_fb,
            "ppi_mrr": mrr_ppi,
            "embedding_mrr": mrr_emb,
            "improvement": mrr_fb - mrr_ppi,
            "improvement_pct": 100 * (mrr_fb - mrr_ppi) / mrr_ppi if mrr_ppi > 0 else 0,
        }

        marker = " ***" if mrr_fb > mrr_ppi else ""
        print(f"  {method:12s}: Fallback={mrr_fb:.4f}, PPI={mrr_ppi:.4f}, "
              f"Emb={mrr_emb:.4f}, gain={mrr_fb - mrr_ppi:+.4f} "
              f"({method_results[method]['improvement_pct']:+.1f}%){marker}")

    # ================================================================
    # Part 2: Rank aggregation sweep (best method)
    # ================================================================
    print("\n[4/6] Rank aggregation sweep (Borda count)...")

    best_method = max(method_results,
                      key=lambda m: method_results[m]["fallback_mrr"])
    emb = embeddings[best_method]

    weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    weight_mrr = {}

    completed = 0
    t0 = time.time()

    # Collect predictions once, aggregate at different weights
    all_ppi_preds = []
    all_emb_preds = []
    all_hidden = []

    for pid, terms in sorted(query_proteins.items()):
        if pid not in emb["node_to_idx"]:
            continue
        for hidden_term in sorted(terms):
            ppi_preds = ppi_neighbor_predict(pid, G, annotations, hidden_term)
            ppi_rank = {t: i + 1 for i, (t, _) in enumerate(ppi_preds)}

            node_to_idx = emb["node_to_idx"]
            query_idx = node_to_idx[pid]
            nn_model = emb["nn_model"]
            coords = emb["coords"]
            nodes = emb["nodes"]
            n_neighbors = min(K_MAX + 1, len(coords))
            dists_knn, idxs_knn = nn_model.kneighbors(
                coords[query_idx:query_idx + 1], n_neighbors=n_neighbors
            )
            ts = Counter()
            for d_val, idx in zip(dists_knn[0], idxs_knn[0]):
                if idx == query_idx:
                    continue
                nid = nodes[idx]
                w = 1.0 / (d_val + 1e-10)
                for term in annotations.get(nid, set()):
                    ts[term] += w
            emb_preds_ranked = ts.most_common()
            emb_rank = {t: i + 1 for i, (t, _) in enumerate(emb_preds_ranked)}

            all_ppi_preds.append(ppi_rank)
            all_emb_preds.append(emb_rank)
            all_hidden.append(hidden_term)

            completed += 1
            if completed % 5000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (n_trials - completed) / rate
                print(f"    Collecting: {completed}/{n_trials} "
                      f"({100*completed/n_trials:.0f}%) "
                      f"-- {rate:.0f}/s -- ETA {eta:.0f}s")

    print(f"  Collected {completed} predictions in {time.time()-t0:.1f}s")

    # Now sweep weights
    t1 = time.time()
    for w_emb in weights:
        w_ppi = 1.0 - w_emb
        rank_results = []

        for i in range(len(all_hidden)):
            hidden = all_hidden[i]
            ppi_r = all_ppi_preds[i]
            emb_r = all_emb_preds[i]

            # All candidate terms
            all_terms = set(ppi_r.keys()) | set(emb_r.keys())
            if not all_terms:
                rank_results.append({"Agg": 0})
                continue

            scores = {}
            for t in all_terms:
                s = 0.0
                if t in ppi_r:
                    s += w_ppi / ppi_r[t]
                if t in emb_r:
                    s += w_emb / emb_r[t]
                scores[t] = s

            # Rank hidden term
            sorted_terms = sorted(scores, key=scores.get, reverse=True)
            try:
                rank = sorted_terms.index(hidden) + 1
            except ValueError:
                rank = 0
            rank_results.append({"Agg": rank})

        mrr = compute_mean_reciprocal_rank(rank_results).get("Agg", 0)
        weight_mrr[w_emb] = mrr

    elapsed_agg = time.time() - t1
    print(f"  Aggregation sweep ({len(weights)} weights) in {elapsed_agg:.1f}s")

    best_agg_w = max(weight_mrr, key=weight_mrr.get)
    best_agg_mrr = weight_mrr[best_agg_w]
    ppi_base = weight_mrr[0.0]

    print(f"\n  Rank aggregation results ({best_method} + PPI):")
    for w in weights:
        marker = " <-- BEST" if w == best_agg_w else ""
        exceed = " ***" if weight_mrr[w] > ppi_base else ""
        print(f"    w={w:.2f}: MRR={weight_mrr[w]:.4f}{marker}{exceed}")
    print(f"\n  Best aggregation: w={best_agg_w:.2f}, MRR={best_agg_mrr:.4f}")
    print(f"  vs PPI (w=0): {best_agg_mrr - ppi_base:+.4f} "
          f"({100*(best_agg_mrr - ppi_base)/ppi_base:+.1f}%)")

    # ================================================================
    # Part 3: Recovery analysis - which trials benefit from embedding?
    # ================================================================
    print("\n[5/6] Analysing embedding-rescued trials...")

    # For best_method, categorize each trial:
    # - PPI finds it, embedding doesn't (PPI-only)
    # - Both find it (Both)
    # - Embedding finds it, PPI doesn't (Embedding-rescue)
    # - Neither finds it (Miss)

    categories = {"PPI-only": 0, "Both": 0, "Emb-rescue": 0, "Miss": 0}
    rescue_proteins = set()
    rescue_terms = Counter()

    for i in range(len(all_hidden)):
        hidden = all_hidden[i]
        ppi_r = all_ppi_preds[i]
        emb_r = all_emb_preds[i]

        ppi_found = hidden in ppi_r
        emb_found = hidden in emb_r

        if ppi_found and emb_found:
            categories["Both"] += 1
        elif ppi_found:
            categories["PPI-only"] += 1
        elif emb_found:
            categories["Emb-rescue"] += 1
        else:
            categories["Miss"] += 1

    total = sum(categories.values())
    print(f"\n  Trial categories ({best_method}, k={K_MAX}):")
    for cat, count in categories.items():
        print(f"    {cat:14s}: {count:6d} ({100*count/total:.1f}%)")

    print(f"\n  Embedding-rescued trials: {categories['Emb-rescue']}")
    if categories["Emb-rescue"] > 0:
        print(f"  These are cases where PPI FAILS but embedding SUCCEEDS.")
        print(f"  This is the UNIQUE VALUE of embeddings.")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{BANNER}")
    print("PHASE 14b FINAL RESULTS")
    print(BANNER)

    print(f"\n  Method comparison (rank fallback):")
    print(f"  {'Method':12s} {'Fallback':>10s} {'PPI':>8s} {'Emb':>8s} {'Gain':>10s}")
    print(f"  " + "-" * 52)
    for m in FULL_METHODS:
        if m in method_results:
            r = method_results[m]
            print(f"  {m:12s} {r['fallback_mrr']:>10.4f} {r['ppi_mrr']:>8.4f} "
                  f"{r['embedding_mrr']:>8.4f} {r['improvement']:>+10.4f}")

    print(f"\n  Rank aggregation ({best_method}):")
    print(f"    Best w={best_agg_w:.2f}: MRR={best_agg_mrr:.4f}")
    print(f"    Pure PPI (w=0): MRR={ppi_base:.4f}")
    print(f"    Gain: {best_agg_mrr - ppi_base:+.4f}")

    print(f"\n  Embedding-rescue trials: {categories['Emb-rescue']}")
    print(f"  (= cases where PPI fails but embedding succeeds)")

    # ================================================================
    # Save results
    # ================================================================
    output = {
        "description": "Phase 14b: Fixed Hybrid + Discovery Quality",
        "method_results": {
            m: {k: round(v, 6) for k, v in r.items()}
            for m, r in method_results.items()
        },
        "rank_aggregation": {
            "method": best_method,
            "weights": {str(w): round(mrr, 6) for w, mrr in weight_mrr.items()},
            "best_weight": best_agg_w,
            "best_mrr": round(best_agg_mrr, 6),
            "ppi_mrr": round(ppi_base, 6),
            "gain": round(best_agg_mrr - ppi_base, 6),
        },
        "trial_categories": categories,
        "total_trials": total,
    }

    result_file = RESULTS / "longrange_hybrid_fixed.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {result_file.name}")

    # ================================================================
    # Figures
    # ================================================================
    print("\n  Generating figures...")

    # Fig75: Hybrid comparison dashboard
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel A: Rank fallback per method
    ax = axes[0]
    methods_sorted = sorted(method_results.keys(),
                            key=lambda m: method_results[m]["fallback_mrr"],
                            reverse=True)
    x_pos = np.arange(len(methods_sorted))
    w_bar = 0.25

    fallback_vals = [method_results[m]["fallback_mrr"] for m in methods_sorted]
    ppi_vals = [method_results[m]["ppi_mrr"] for m in methods_sorted]
    emb_vals = [method_results[m]["embedding_mrr"] for m in methods_sorted]

    ax.bar(x_pos - w_bar, ppi_vals, w_bar, label="PPI-Neighbors",
           color="#636363", edgecolor="white")
    ax.bar(x_pos, fallback_vals, w_bar, label="Rank Fallback",
           color="#3182bd", edgecolor="white")
    ax.bar(x_pos + w_bar, emb_vals, w_bar, label="Embedding Only",
           color="#fb6a4a", edgecolor="white")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods_sorted, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("MRR", fontsize=12)
    ax.set_title("A: Rank Fallback Hybrid", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    for i, (f, p) in enumerate(zip(fallback_vals, ppi_vals)):
        ax.text(i, f + 0.003, f"{f:.4f}", ha="center", fontsize=8,
                fontweight="bold", color="#3182bd")
        if f > p:
            ax.annotate("", xy=(i, f), xytext=(i, p),
                        arrowprops=dict(arrowstyle="->", color="green",
                                        lw=2))

    # Panel B: Rank aggregation sweep
    ax2 = axes[1]
    agg_weights = sorted(weight_mrr.keys())
    agg_mrrs = [weight_mrr[w] for w in agg_weights]
    ax2.plot(agg_weights, agg_mrrs, "o-", color="#3182bd", linewidth=2.5,
             markersize=8)
    ax2.axhline(ppi_base, color="#636363", linestyle="--", linewidth=2,
                label=f"PPI (MRR={ppi_base:.3f})")

    if best_agg_mrr > ppi_base:
        exceed_w = [w for w in agg_weights if weight_mrr[w] > ppi_base]
        if exceed_w:
            ax2.axvspan(min(exceed_w) - 0.02, max(exceed_w) + 0.02,
                        alpha=0.15, color="green")
            ax2.text(0.5, 0.95, f"Hybrid > PPI for w in "
                     f"[{min(exceed_w):.2f}, {max(exceed_w):.2f}]",
                     transform=ax2.transAxes, ha="center", fontsize=10,
                     color="green", fontweight="bold")

    ax2.plot(best_agg_w, best_agg_mrr, "*", color="#e6550d",
             markersize=20, zorder=10)
    ax2.set_xlabel("Embedding Weight (w) in Borda Count", fontsize=12)
    ax2.set_ylabel("MRR", fontsize=12)
    ax2.set_title("B: Rank Aggregation Sweep", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel C: Trial categories pie chart
    ax3 = axes[2]
    labels = list(categories.keys())
    sizes = [categories[l] for l in labels]
    colors_pie = ["#636363", "#3182bd", "#2ca25f", "#d9d9d9"]
    explode = [0, 0, 0.1, 0]

    wedges, texts, autotexts = ax3.pie(
        sizes, explode=explode, labels=labels, colors=colors_pie,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10}
    )
    # Highlight rescue slice
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    ax3.set_title("C: Trial Outcomes", fontsize=13, fontweight="bold")

    # Add rescue count annotation
    ax3.text(0, -1.3, f"Embedding-rescued: {categories['Emb-rescue']} trials "
             f"({100*categories['Emb-rescue']/total:.1f}%)",
             ha="center", fontsize=11, fontweight="bold", color="#2ca25f")

    fig.suptitle("Phase 14b: Hybrid Predictor (Fixed Rank-Based)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig75_hybrid_fixed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig75_hybrid_fixed.png")

    print(f"\n{BANNER}")
    print("Phase 14b complete.")
    print(BANNER)


if __name__ == "__main__":
    run()
