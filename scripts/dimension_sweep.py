#!/usr/bin/env python3
"""
Dimension Sweep: Spectral Embedding at d = 2, 4, 8, 16, 32, 64
================================================================

Quick diagnostic: does higher-dimensional Spectral embedding outperform
PPI topology for function prediction?

If MRR(d=64) > MRR(PPI-Neighbors), the 2D constraint is the bottleneck
and the NC narrative can be revived.

Uses the same LOTO-CV framework as Phase 13 (function_prediction.py).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
    TARGET_STD,
)

# Import reusable functions from Phase 13
from function_prediction import (
    build_alias_mapping,
    parse_gaf_experimental,
    ppi_neighbor_predict,
    twohop_diffusion_predict,
    build_knn_index,
    knn_predict_fast,
    evaluate_precision_at_k,
    compute_mean_reciprocal_rank,
    K_VALUES, K_MAX,
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

DIMENSIONS = [2, 4, 8, 16, 32, 64]

BANNER = "=" * 64


# ============================================================
# Spectral Embedding at Arbitrary Dimension
# ============================================================

def compute_spectral_embedding(graph, dim):
    """Compute Spectral embedding at given dimension.

    Uses the first `dim` non-trivial eigenvectors of the normalised
    graph Laplacian (smallest eigenvalues, excluding the constant
    eigenvector at eigenvalue 0).

    Parameters
    ----------
    graph : nx.Graph
        The PPI network.
    dim : int
        Target embedding dimension.

    Returns
    -------
    coords : ndarray, shape (n_nodes, dim)
        Embedding coordinates.
    nodes : list[str]
        Node IDs in the order matching coords rows.
    eigenvalues : ndarray
        The dim+1 smallest eigenvalues (including the trivial one).
    """
    nodes = sorted(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Build adjacency matrix
    adj = nx.adjacency_matrix(graph, nodelist=nodes, weight=None).astype(float)

    # Normalised Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0  # avoid division by zero
    d_inv_sqrt = 1.0 / np.sqrt(degrees)

    from scipy.sparse import diags
    D_inv_sqrt = diags(d_inv_sqrt)
    L_norm = D_inv_sqrt @ (diags(degrees) - adj) @ D_inv_sqrt

    # Compute smallest dim+1 eigenvectors (skip the trivial constant one)
    n_eigs = dim + 1
    # eigsh needs n_eigs < n-1
    if n_eigs >= n - 1:
        n_eigs = n - 2

    eigenvalues, eigenvectors = eigsh(L_norm, k=n_eigs, which="SM", tol=1e-6)

    # Sort by eigenvalue (should already be sorted, but be safe)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip the first eigenvector (constant, eigenvalue ~0)
    coords = eigenvectors[:, 1:dim + 1]

    # Standardise to target std (matching framework convention)
    for j in range(coords.shape[1]):
        col_std = coords[:, j].std()
        if col_std > 1e-10:
            coords[:, j] = coords[:, j] / col_std * TARGET_STD

    return coords, nodes, eigenvalues[:dim + 1]


# ============================================================
# LOTO Prediction for One Dimension
# ============================================================

def run_loto_for_dimension(coords, nodes, graph, annotations, term_freq, dim):
    """Run LOTO predictions for a single embedding dimension.

    Parameters
    ----------
    coords : ndarray
        (n, dim) embedding coordinates.
    nodes : list[str]
        Node IDs.
    graph : nx.Graph
        PPI network.
    annotations : dict
        {STRING_ID: set(go_terms)}
    term_freq : Counter
        Global term frequencies.
    dim : int
        Embedding dimension (for logging).

    Returns
    -------
    dict
        {method_name: MRR}
    """
    node_set = set(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Filter proteins: in network + in embedding + >= 2 terms
    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in node_set and pid in graph
    }

    n_trials = sum(len(terms) for terms in query_proteins.values())

    # Build KNN index for this dimension
    n_neighbors = min(K_MAX + 1, len(coords))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(coords)

    # Random baseline ranking
    random_ranking = term_freq.most_common()

    rank_results = []
    completed = 0
    t0 = time.time()

    for pid, terms in sorted(query_proteins.items()):
        terms_list = sorted(terms)
        for hidden_term in terms_list:
            trial_rank = {}

            # Embedding KNN
            if pid in node_to_idx:
                query_idx = node_to_idx[pid]
                preds = knn_predict_fast(
                    query_idx, nn_model, coords, nodes,
                    annotations, k=K_MAX, hidden_term=hidden_term,
                )
                pred_terms = [t for t, _ in preds]
                try:
                    trial_rank[f"Spectral-d{dim}"] = pred_terms.index(hidden_term) + 1
                except ValueError:
                    trial_rank[f"Spectral-d{dim}"] = 0

            # PPI baseline (only compute once per trial, reuse across dims)
            if dim == DIMENSIONS[0]:  # only on first dim to avoid redundant computation
                ppi_preds = ppi_neighbor_predict(pid, graph, annotations, hidden_term)
                ppi_terms = [t for t, _ in ppi_preds]
                try:
                    trial_rank["PPI-Neighbors"] = ppi_terms.index(hidden_term) + 1
                except ValueError:
                    trial_rank["PPI-Neighbors"] = 0

                hop_preds = twohop_diffusion_predict(pid, graph, annotations, hidden_term)
                hop_terms = [t for t, _ in hop_preds]
                try:
                    trial_rank["2-Hop Diffusion"] = hop_terms.index(hidden_term) + 1
                except ValueError:
                    trial_rank["2-Hop Diffusion"] = 0

                rand_terms = [t for t, _ in random_ranking]
                try:
                    trial_rank["Random"] = rand_terms.index(hidden_term) + 1
                except ValueError:
                    trial_rank["Random"] = 0

            rank_results.append(trial_rank)
            completed += 1

            if completed % 3000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_trials - completed) / rate if rate > 0 else 0
                print(f"      d={dim}: {completed}/{n_trials} "
                      f"({100*completed/n_trials:.0f}%) "
                      f"-- {rate:.0f} trials/s -- ETA {eta:.0f}s")

    elapsed = time.time() - t0
    mrr = compute_mean_reciprocal_rank(rank_results)
    print(f"      d={dim}: {completed} trials in {elapsed:.1f}s "
          f"({completed/elapsed:.0f} trials/s)")

    return mrr


# ============================================================
# Visualisation
# ============================================================

def plot_dimension_sweep(dim_mrr, ppi_mrr, twohop_mrr, random_mrr, eigenvalue_info):
    """Plot MRR vs embedding dimension with baselines."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel A: MRR vs dimension ---
    ax = axes[0]
    dims = list(dim_mrr.keys())
    mrrs = [dim_mrr[d] for d in dims]

    ax.plot(dims, mrrs, "o-", color="#3182bd", linewidth=2.5,
            markersize=10, label="Spectral-d", zorder=5)

    # Baselines
    if ppi_mrr > 0:
        ax.axhline(ppi_mrr, color="#636363", linestyle="--", linewidth=2,
                   label=f"PPI-Neighbors ({ppi_mrr:.3f})")
    if twohop_mrr > 0:
        ax.axhline(twohop_mrr, color="#969696", linestyle="--", linewidth=1.5,
                   label=f"2-Hop Diffusion ({twohop_mrr:.3f})")
    if random_mrr > 0:
        ax.axhline(random_mrr, color="#d9d9d9", linestyle="--", linewidth=1,
                   label=f"Random ({random_mrr:.3f})")

    # Phase 13 Spectral 2D result
    phase13_spectral_mrr = mrrs[0] if dims[0] == 2 else None

    # Annotate each point
    for d, mrr in zip(dims, mrrs):
        ax.annotate(f"{mrr:.3f}", (d, mrr), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")

    # Check if any dim crosses PPI baseline
    if ppi_mrr > 0:
        crossed = [d for d, m in zip(dims, mrrs) if m > ppi_mrr]
        if crossed:
            ax.axvspan(min(crossed) - 1, max(dims) + 1, alpha=0.1, color="green",
                       label=f"Exceeds PPI at d>={min(crossed)}")
        else:
            ax.text(0.95, 0.05, "No dimension exceeds PPI baseline",
                    transform=ax.transAxes, ha="right", fontsize=10,
                    color="red", fontstyle="italic",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Embedding Dimension (d)", fontsize=13)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=13)
    ax.set_title("Dimension Sweep: Spectral Embedding MRR", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel B: Eigenvalue spectrum ---
    ax2 = axes[1]
    all_eigs = eigenvalue_info.get(64, None)
    if all_eigs is not None:
        # Plot first 64 eigenvalues
        eig_vals = all_eigs[1:]  # skip trivial eigenvalue
        ax2.plot(range(1, len(eig_vals) + 1), eig_vals, "o-",
                 color="#3182bd", markersize=4, linewidth=1)
        ax2.set_xlabel("Eigenvector Index", fontsize=13)
        ax2.set_ylabel("Eigenvalue (L_sym)", fontsize=13)
        ax2.set_title("Laplacian Eigenvalue Spectrum", fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Mark dimension boundaries
        for d in dims:
            ax2.axvline(d, color="#cccccc", linestyle=":", linewidth=0.8)
            ax2.text(d, ax2.get_ylim()[1] * 0.95, f"d={d}",
                     ha="center", fontsize=8, color="#999999")

        # Spectral gap
        if len(eig_vals) >= 2:
            gap = eig_vals[1] - eig_vals[0]
            ax2.annotate(f"Spectral gap = {gap:.4f}",
                         xy=(1, eig_vals[0]),
                         xytext=(10, eig_vals[0] + 0.1),
                         fontsize=10, color="#e6550d",
                         arrowprops=dict(arrowstyle="->", color="#e6550d"))

    fig.suptitle("Phase 13b: Dimension Sweep Diagnostic",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig69_dimension_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig69_dimension_sweep.png")


def plot_dimension_detail(dim_mrr, ppi_mrr, eigenvalue_info):
    """Detailed plot: MRR gain per dimension + information captured."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    dims = list(dim_mrr.keys())
    mrrs = [dim_mrr[d] for d in dims]

    # --- Panel A: MRR gain over d=2 ---
    ax = axes[0]
    baseline_mrr = mrrs[0]  # d=2
    gains = [m - baseline_mrr for m in mrrs]

    bar_colors = ["#3182bd"] * len(dims)
    for i, g in enumerate(gains):
        if g > 0 and mrrs[i] > ppi_mrr:
            bar_colors[i] = "#2ca25f"  # green if exceeds PPI

    bars = ax.bar([str(d) for d in dims], gains, color=bar_colors,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    if ppi_mrr > baseline_mrr:
        ax.axhline(ppi_mrr - baseline_mrr, color="#636363",
                   linestyle="--", linewidth=2,
                   label=f"PPI gain over d=2 ({ppi_mrr - baseline_mrr:+.3f})")
        ax.legend(fontsize=9)

    for i, (d, g) in enumerate(zip(dims, gains)):
        ax.text(i, g + (0.002 if g >= 0 else -0.008),
                f"{g:+.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Embedding Dimension", fontsize=12)
    ax.set_ylabel("MRR Gain over d=2", fontsize=12)
    ax.set_title("A: Marginal MRR per Dimension", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # --- Panel B: Cumulative eigenvalue information ---
    ax2 = axes[1]
    all_eigs = eigenvalue_info.get(64, None)
    if all_eigs is not None:
        eig_vals = all_eigs[1:]  # skip trivial
        # Cumulative sum of eigenvalues as proxy for information captured
        cumsum = np.cumsum(eig_vals)
        total = cumsum[-1]
        cumfrac = cumsum / total

        ax2.bar(range(1, len(cumfrac) + 1), cumfrac,
                color="#3182bd", alpha=0.7, edgecolor="white", linewidth=0.3)
        ax2.axhline(1.0, color="black", linewidth=0.5)

        for d in dims:
            if d <= len(cumfrac):
                ax2.axvline(d, color="#e6550d", linestyle="--", linewidth=1.2)
                ax2.text(d + 0.5, cumfrac[min(d - 1, len(cumfrac) - 1)],
                         f"d={d}\n{cumfrac[min(d-1, len(cumfrac)-1)]:.1%}",
                         fontsize=8, color="#e6550d")

        ax2.set_xlabel("Number of Eigenvectors", fontsize=12)
        ax2.set_ylabel("Cumulative Eigenvalue Fraction", fontsize=12)
        ax2.set_title("B: Information Captured vs Dimension", fontsize=13,
                      fontweight="bold")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "Fig70_dimension_detail.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig70_dimension_detail.png")


# ============================================================
# Main
# ============================================================

def run():
    print(BANNER)
    print("Dimension Sweep: Spectral Embedding d = 2, 4, 8, 16, 32, 64")
    print(BANNER)

    np.random.seed(SEED)

    # --- Stage 1: Data loading (reuse Phase 13 pipeline) ---
    print("\n[1/5] Building alias mapping...")
    sgd_to_string, orf_to_string, network_nodes = build_alias_mapping()

    print("\n[2/5] Parsing GAF (experimental BP)...")
    annotations, ann_stats = parse_gaf_experimental(
        sgd_to_string, orf_to_string, network_nodes
    )

    term_freq = Counter()
    for terms in annotations.values():
        term_freq.update(terms)

    print("\n[3/5] Loading full PPI network...")
    G = nx.Graph()
    with open(str(NETWORK_FILE), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- Stage 2: Compute Spectral embeddings at each dimension ---
    print("\n[4/5] Computing Spectral embeddings...")
    embeddings_by_dim = {}
    eigenvalue_info = {}

    for d in DIMENSIONS:
        print(f"  Computing d={d}...")
        t0 = time.time()
        coords, nodes, eigs = compute_spectral_embedding(G, d)
        elapsed = time.time() - t0
        embeddings_by_dim[d] = {"coords": coords, "nodes": nodes}
        eigenvalue_info[d] = eigs
        print(f"    d={d}: {coords.shape} in {elapsed:.1f}s, "
              f"eigenvalues[0:3]={eigs[:3].round(4)}")

    # --- Stage 3: LOTO predictions ---
    print("\n[5/5] Running LOTO predictions for each dimension...")

    all_mrr = {}
    ppi_mrr_all = []
    twohop_mrr_all = []
    random_mrr_all = []

    for d in DIMENSIONS:
        print(f"\n  === Dimension d={d} ===")
        emb = embeddings_by_dim[d]
        mrr = run_loto_for_dimension(
            emb["coords"], emb["nodes"], G,
            annotations, term_freq, d,
        )
        all_mrr[d] = mrr.get(f"Spectral-d{d}", 0)

        # Collect baselines from first dimension
        if d == DIMENSIONS[0]:
            ppi_mrr_all.append(mrr.get("PPI-Neighbors", 0))
            twohop_mrr_all.append(mrr.get("2-Hop Diffusion", 0))
            random_mrr_all.append(mrr.get("Random", 0))

    ppi_mrr = ppi_mrr_all[0] if ppi_mrr_all else 0
    twohop_mrr = twohop_mrr_all[0] if twohop_mrr_all else 0
    random_mrr = random_mrr_all[0] if random_mrr_all else 0

    # --- Summary ---
    print(f"\n{BANNER}")
    print("DIMENSION SWEEP RESULTS")
    print(BANNER)
    print(f"\n{'Dimension':>12s}  {'MRR':>8s}  {'vs PPI':>10s}  {'vs d=2':>10s}")
    print("-" * 48)

    d2_mrr = all_mrr.get(2, 0)
    for d in DIMENSIONS:
        m = all_mrr[d]
        vs_ppi = m - ppi_mrr
        vs_d2 = m - d2_mrr
        marker = " ***" if vs_ppi > 0 else ""
        print(f"  d={d:<8d}  {m:>8.4f}  {vs_ppi:>+10.4f}  {vs_d2:>+10.4f}{marker}")

    print(f"\n  PPI-Neighbors: {ppi_mrr:.4f}")
    print(f"  2-Hop Diffusion: {twohop_mrr:.4f}")
    print(f"  Random: {random_mrr:.4f}")

    crossed = [d for d in DIMENSIONS if all_mrr[d] > ppi_mrr]
    if crossed:
        print(f"\n  >>> PPI baseline EXCEEDED at d >= {min(crossed)} <<<")
    else:
        print(f"\n  >>> No dimension exceeds PPI baseline <<<")

    # Peak improvement
    best_d = max(DIMENSIONS, key=lambda d: all_mrr[d])
    print(f"  Best dimension: d={best_d} (MRR={all_mrr[best_d]:.4f})")
    print(f"  Improvement over d=2: {all_mrr[best_d] - d2_mrr:+.4f} "
          f"({100*(all_mrr[best_d] - d2_mrr)/d2_mrr:+.1f}%)")

    # --- Save results ---
    output = {
        "description": "Dimension Sweep: Spectral Embedding d=2..64",
        "dimensions": DIMENSIONS,
        "mrr_by_dimension": {str(d): round(all_mrr[d], 6) for d in DIMENSIONS},
        "baselines": {
            "PPI-Neighbors": round(ppi_mrr, 6),
            "2-Hop Diffusion": round(twohop_mrr, 6),
            "Random": round(random_mrr, 6),
        },
        "eigenvalues": {
            str(d): [round(float(v), 6) for v in eigenvalue_info[d]]
            for d in DIMENSIONS
        },
        "best_dimension": best_d,
        "best_mrr": round(all_mrr[best_d], 6),
        "ppi_exceeded": len(crossed) > 0,
        "crossed_at": min(crossed) if crossed else None,
        "annotation_stats": ann_stats,
    }

    result_file = RESULTS / "dimension_sweep.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {result_file.name}")

    # --- Plot ---
    print("\n  Generating figures...")
    plot_dimension_sweep(all_mrr, ppi_mrr, twohop_mrr, random_mrr, eigenvalue_info)
    plot_dimension_detail(all_mrr, ppi_mrr, eigenvalue_info)

    print(f"\n{BANNER}")
    print("Dimension sweep complete.")
    print(BANNER)


if __name__ == "__main__":
    run()
