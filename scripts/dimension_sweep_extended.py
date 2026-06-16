#!/usr/bin/env python3
"""
Extended Dimension Sweep: Spectral Embedding d = 128, 256 (Step 47 / Phase 18)
==============================================================================

Extend the Phase 13b dimension sweep to d = 128 and d = 256.

Current results (d = 2..64):
  MRR grows logarithmically from 0.066 (d=2) to 0.205 (d=64).
  PPI-Neighbors baseline = 0.219.
  The curve has NOT plateaued.

If MRR(d=128) > 0.219, this is a landmark result: embeddings surpass
network topology for function prediction for the first time.

Output
------
- results/dimension_sweep_extended.json
- figures/Fig71_dimension_sweep_extended.png
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
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
    TARGET_STD,
)
from function_prediction import (
    build_alias_mapping,
    parse_gaf_experimental,
    ppi_neighbor_predict,
    twohop_diffusion_predict,
    knn_predict_fast,
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

# New dimensions to test
NEW_DIMENSIONS = [128, 256]
# All dimensions (including previous)
ALL_DIMENSIONS = [2, 4, 8, 16, 32, 64, 128, 256]

BANNER = "=" * 64


# ============================================================
# Spectral Embedding at Arbitrary Dimension
# ============================================================

def compute_spectral_embedding(graph, dim):
    """Compute Spectral embedding at given dimension.

    Uses the first `dim` non-trivial eigenvectors of the normalised
    graph Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.
    """
    nodes = sorted(graph.nodes())
    n = len(nodes)

    adj = nx.adjacency_matrix(graph, nodelist=nodes, weight=None).astype(float)
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(degrees)

    D_inv_sqrt = diags(d_inv_sqrt)
    L_norm = D_inv_sqrt @ (diags(degrees) - adj) @ D_inv_sqrt

    # eigsh needs k < n-1
    n_eigs = min(dim + 1, n - 2)

    print(f"      Computing {n_eigs} eigenvectors for d={dim} ...")
    eigenvalues, eigenvectors = eigsh(L_norm, k=n_eigs, which="SM", tol=1e-6)

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip trivial eigenvector
    coords = eigenvectors[:, 1:dim + 1]

    # Standardise each column to TARGET_STD
    for j in range(coords.shape[1]):
        col_std = coords[:, j].std()
        if col_std > 1e-10:
            coords[:, j] = coords[:, j] / col_std * TARGET_STD

    return coords, nodes, eigenvalues[:dim + 1]


# ============================================================
# LOTO Prediction for One Dimension
# ============================================================

def run_loto_for_dimension(coords, nodes, graph, annotations, term_freq, dim,
                           compute_ppi=False):
    """Run LOTO predictions for a single embedding dimension."""
    node_set = set(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in node_set and pid in graph
    }

    n_trials = sum(len(terms) for terms in query_proteins.values())

    n_neighbors = min(K_MAX + 1, len(coords))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(coords)

    random_ranking = term_freq.most_common()

    rank_results = []
    completed = 0
    t0 = time.time()

    for pid, terms in sorted(query_proteins.items()):
        terms_list = sorted(terms)
        for hidden_term in terms_list:
            trial_rank = {}

            # Embedding KNN prediction
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

            # PPI baseline (compute on first dimension only)
            if compute_ppi:
                ppi_preds = ppi_neighbor_predict(pid, graph, annotations, hidden_term)
                ppi_terms_list = [t for t, _ in ppi_preds]
                try:
                    trial_rank["PPI-Neighbors"] = ppi_terms_list.index(hidden_term) + 1
                except ValueError:
                    trial_rank["PPI-Neighbors"] = 0

                hop_preds = twohop_diffusion_predict(pid, graph, annotations, hidden_term)
                hop_terms_list = [t for t, _ in hop_preds]
                try:
                    trial_rank["2-Hop Diffusion"] = hop_terms_list.index(hidden_term) + 1
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
# Main
# ============================================================

def run():
    """Run the extended dimension sweep."""
    t_start = time.time()
    print(BANNER)
    print("  Phase 18: Extended Dimension Sweep (d = 128, 256)")
    print(BANNER)

    np.random.seed(SEED)

    # ---- Load previous results ----
    prev_file = RESULTS / "dimension_sweep.json"
    prev_mrr = {}
    prev_eigenvalues = {}
    prev_baselines = {}
    if prev_file.exists():
        with open(prev_file, encoding="utf-8") as f:
            prev_data = json.load(f)
        prev_mrr = {int(k): v for k, v in prev_data["mrr_by_dimension"].items()}
        prev_eigenvalues = {int(k): v for k, v in prev_data["eigenvalues"].items()}
        prev_baselines = prev_data.get("baselines", {})
        print(f"  Loaded previous results: d = {sorted(prev_mrr.keys())}")
        print(f"  Previous MRR: {prev_mrr}")

    # ---- Load network ----
    print(f"\n[1/4] Loading network from {NETWORK_FILE.name} ...")
    G = nx.read_edgelist(str(NETWORK_FILE))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---- Load annotations ----
    print(f"\n[2/4] Loading GO annotations (experimental BP only) ...")
    sgd_map, orf_map, net_nodes = build_alias_mapping()
    annotations, ann_stats = parse_gaf_experimental(sgd_map, orf_map, net_nodes)

    # Build term frequency
    term_freq = Counter()
    for terms in annotations.values():
        term_freq.update(terms)

    # Query proteins
    node_set = set(G.nodes())
    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in node_set
    }
    n_trials = sum(len(terms) for terms in query_proteins.values())
    print(f"  Proteins: {len(query_proteins)}, Trials: {n_trials}")
    print(f"  Annotation stats: {ann_stats}")

    # ---- Restore PPI baselines from previous sweep ----
    ppi_mrr_val = prev_baselines.get("PPI-Neighbors", 0.0)
    twohop_mrr_val = prev_baselines.get("2-Hop Diffusion", 0.0)
    random_mrr_val = prev_baselines.get("Random", 0.0)
    print(f"  PPI baseline MRR: {ppi_mrr_val:.4f}")

    # ---- Compute new dimensions ----
    print(f"\n[3/4] Computing Spectral embeddings at d = {NEW_DIMENSIONS} ...")

    new_mrr = {}
    new_eigenvalues = {}
    is_first_dim = True

    for dim in NEW_DIMENSIONS:
        print(f"\n  --- Dimension {dim} ---")

        # Check if node count allows this many eigenvectors
        n_nodes = G.number_of_nodes()
        if dim + 1 >= n_nodes - 1:
            print(f"  SKIP: d={dim} requires {dim+1} eigenvectors "
                  f"but only {n_nodes-2} available")
            continue

        # Compute embedding
        t_emb = time.time()
        coords, nodes, eigenvalues = compute_spectral_embedding(G, dim)
        emb_time = time.time() - t_emb
        print(f"  Embedding: {coords.shape} in {emb_time:.1f}s")
        print(f"  Eigenvalue range: [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]")

        # Save embedding
        emb_path = EMB / f"Spectral_d{dim}_full.npy"
        nodes_path = EMB / f"Spectral_d{dim}_full_nodes.json"
        np.save(str(emb_path), coords)
        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(nodes, f)

        # Run LOTO
        t_loto = time.time()
        mrr_dict = run_loto_for_dimension(
            coords, nodes, G, annotations, term_freq, dim,
            compute_ppi=is_first_dim,
        )
        loto_time = time.time() - t_loto

        # Extract Spectral MRR for this dimension
        spectral_key = f"Spectral-d{dim}"
        mrr = float(mrr_dict.get(spectral_key, 0.0))
        new_mrr[dim] = mrr
        new_eigenvalues[dim] = eigenvalues.tolist()

        print(f"  MRR(d={dim}) = {mrr:.6f}")
        print(f"  vs PPI ({ppi_mrr_val:.6f}): "
              f"{'EXCEEDS' if mrr > ppi_mrr_val else 'below'} by "
              f"{abs(mrr - ppi_mrr_val):.6f}")
        print(f"  Time: embedding {emb_time:.1f}s + LOTO {loto_time:.1f}s")

        # Update baselines from fresh computation on first dimension
        if is_first_dim:
            ppi_mrr_val = float(mrr_dict.get("PPI-Neighbors", ppi_mrr_val))
            twohop_mrr_val = float(mrr_dict.get("2-Hop Diffusion", twohop_mrr_val))
            random_mrr_val = float(mrr_dict.get("Random", random_mrr_val))
            is_first_dim = False

    # ---- Combine results ----
    print(f"\n[4/4] Combining results and generating output ...")

    all_mrr = {**prev_mrr, **new_mrr}
    all_eigenvalues = {**prev_eigenvalues, **new_eigenvalues}

    # Determine crossing point
    crossed_at = None
    for d in sorted(all_mrr.keys()):
        if all_mrr[d] > ppi_mrr_val:
            crossed_at = d
            break

    # Logarithmic fit and extrapolation
    from scipy.stats import linregress
    log_dims = np.log2(list(all_mrr.keys()))
    mrr_vals = [all_mrr[d] for d in sorted(all_mrr.keys())]
    slope, intercept, r_value, p_value, std_err = linregress(log_dims, mrr_vals)

    # Predict at d=512 and d=1024
    predicted_512 = slope * np.log2(512) + intercept
    predicted_1024 = slope * np.log2(1024) + intercept

    print(f"\n  Log-linear fit: MRR = {slope:.4f} * log2(d) + {intercept:.4f}")
    print(f"  R^2 = {r_value**2:.4f}")
    print(f"  Predicted MRR(d=512)  = {predicted_512:.4f}")
    print(f"  Predicted MRR(d=1024) = {predicted_1024:.4f}")
    if crossed_at:
        print(f"  *** PPI BASELINE CROSSED at d = {crossed_at} ***")
    else:
        # Extrapolate crossing point
        if slope > 0:
            log2_cross = (ppi_mrr_val - intercept) / slope
            d_cross = 2 ** log2_cross
            print(f"  Extrapolated crossing at d ~ {d_cross:.0f}")

    # Save results
    output = {
        "description": "Extended Dimension Sweep: Spectral Embedding d=2..256",
        "dimensions": sorted(all_mrr.keys()),
        "mrr_by_dimension": {str(d): all_mrr[d] for d in sorted(all_mrr.keys())},
        "baselines": {
            "PPI-Neighbors": ppi_mrr_val,
            "2-Hop Diffusion": twohop_mrr_val,
            "Random": random_mrr_val,
        },
        "eigenvalues": {str(d): all_eigenvalues[d] for d in sorted(all_eigenvalues.keys())},
        "best_dimension": max(all_mrr, key=all_mrr.get),
        "best_mrr": max(all_mrr.values()),
        "ppi_exceeded": crossed_at is not None,
        "crossed_at": crossed_at,
        "log_linear_fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "R_squared": float(r_value ** 2),
            "predicted_d512": float(predicted_512),
            "predicted_d1024": float(predicted_1024),
        },
        "annotation_stats": ann_stats,
    }

    out_file = RESULTS / "dimension_sweep_extended.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_file}")

    # ---- Plot ----
    plot_extended_sweep(all_mrr, ppi_mrr_val, twohop_mrr_val, random_mrr_val,
                        slope, intercept, crossed_at)

    elapsed = time.time() - t_start
    print(f"\nPhase 18 completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return output


# ============================================================
# Figure
# ============================================================

def plot_extended_sweep(all_mrr, ppi_mrr, twohop_mrr, random_mrr,
                        slope, intercept, crossed_at):
    """Extended dimension sweep figure with extrapolation."""
    fig, ax = plt.subplots(figsize=(12, 8))

    dims = sorted(all_mrr.keys())
    mrrs = [all_mrr[d] for d in dims]

    # Previous dimensions (d <= 64) as circles
    prev_dims = [d for d in dims if d <= 64]
    prev_mrrs = [all_mrr[d] for d in prev_dims]
    ax.plot(prev_dims, prev_mrrs, "o-", color="#3182bd", linewidth=2.5,
            markersize=10, label="Phase 13b (d=2..64)", zorder=5)

    # New dimensions (d > 64) as stars
    new_dims = [d for d in dims if d > 64]
    new_mrrs = [all_mrr[d] for d in new_dims]
    if new_dims:
        ax.plot(new_dims, new_mrrs, "*-", color="#d62728", linewidth=2.5,
                markersize=18, label="Phase 18 (d=128, 256)", zorder=6)

    # Log-linear fit line
    fit_x = np.logspace(np.log2(2), np.log2(512), 50, base=2)
    fit_y = slope * np.log2(fit_x) + intercept
    ax.plot(fit_x, fit_y, "--", color="grey", linewidth=1.5, alpha=0.6,
            label=f"Log-linear fit (R$^2$={slope:.3f}*log2(d)+{intercept:.3f})")

    # Extrapolation to d=512
    extrap_x = np.logspace(np.log2(256), np.log2(512), 20, base=2)
    extrap_y = slope * np.log2(extrap_x) + intercept
    ax.plot(extrap_x, extrap_y, ":", color="grey", linewidth=1.5, alpha=0.4,
            label="Extrapolation")

    # Baselines
    ax.axhline(ppi_mrr, color="#636363", linestyle="--", linewidth=2,
               label=f"PPI-Neighbors ({ppi_mrr:.3f})")
    if twohop_mrr > 0:
        ax.axhline(twohop_mrr, color="#969696", linestyle="--", linewidth=1.5,
                   label=f"2-Hop Diffusion ({twohop_mrr:.3f})")
    if random_mrr > 0:
        ax.axhline(random_mrr, color="#d9d9d9", linestyle="--", linewidth=1,
                   label=f"Random ({random_mrr:.3f})")

    # Annotate points
    for d, mrr in zip(dims, mrrs):
        color = "#d62728" if d > 64 else "#3182bd"
        ax.annotate(f"{mrr:.3f}", (d, mrr), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=10,
                    fontweight="bold", color=color)

    # Crossing region
    if crossed_at:
        ax.axvspan(crossed_at - 5, max(dims) + 10, alpha=0.1, color="green",
                   label=f"Exceeds PPI at d>={crossed_at}")
        ax.text(crossed_at, ppi_mrr + 0.005, f"d={crossed_at}",
                fontsize=12, fontweight="bold", color="green",
                ha="center")
    else:
        # Show extrapolated crossing
        if slope > 0:
            log2_cross = (ppi_mrr - intercept) / slope
            d_cross = 2 ** log2_cross
            if d_cross < 2000:
                ax.axvline(d_cross, color="orange", linestyle=":",
                           alpha=0.5, linewidth=2)
                ax.text(d_cross, ppi_mrr + 0.01,
                        f"Predicted crossing\nd~{d_cross:.0f}",
                        fontsize=11, ha="center", color="orange",
                        fontweight="bold")

    ax.set_xlabel("Embedding Dimension (d)", fontsize=14)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=14)
    ax.set_title("Extended Dimension Sweep: Spectral Embedding vs PPI Topology",
                 fontsize=15, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256", "512"])
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES / "Fig71_dimension_sweep_extended.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run()
