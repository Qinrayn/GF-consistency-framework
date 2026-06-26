#!/usr/bin/env python3
"""
Dimension Sweep Extension: Spectral Embedding d = 512, 1024 (Step 66 / Phase 20)
=================================================================================

Extend the dimension sweep to d = 512 and d = 1024.

Current results (d = 2..256):
  MRR grows logarithmically from 0.066 (d=2) to 0.230 (d=256).
  PPI-Neighbors baseline = 0.219.
  d=256 EXCEEDS PPI by 5.3%.
  Log-linear fit: R^2 = 0.961, slope = 0.0208, intercept = 0.0276.
  Predicted MRR(d=512) = 0.271, MRR(d=1024) = 0.295.

This script tests whether the log-linear scaling law holds at d=512/1024,
and whether performance continues to improve or plateaus.

Output
------
- results/dimension_sweep_512.json
- figures/Fig76_dimension_sweep_512.png
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

# RESULTS.mkdir(parents=True, exist_ok=True)  # deferred to run() — P1-4b
# FIGURES.mkdir(parents=True, exist_ok=True)  # deferred to run() — P1-4b

NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"

NEW_DIMENSIONS = [512]
OPTIONAL_DIMENSIONS = [1024]

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

    n_eigs = min(dim + 1, n - 2)

    print(f"      Computing {n_eigs} eigenvectors for d={dim} ...")
    t0 = time.time()
    eigenvalues, eigenvectors = eigsh(L_norm, k=n_eigs, which="SM", tol=1e-6)
    elapsed = time.time() - t0
    print(f"      eigsh completed in {elapsed:.1f}s")

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

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
    """Run the d=512/1024 dimension sweep."""
    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(BANNER)
    print("  Phase 20: Dimension Sweep Extension (d = 512, 1024)")
    print(BANNER)

    np.random.seed(SEED)

    # ---- Load previous results ----
    prev_file = RESULTS / "dimension_sweep_extended.json"
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
        print(f"  Previous log-linear fit: {prev_data.get('log_linear_fit', {})}")
    else:
        print("  WARNING: No previous extended results found.")

    ppi_mrr_val = prev_baselines.get("PPI-Neighbors", 0.219)
    twohop_mrr_val = prev_baselines.get("2-Hop Diffusion", 0.0)
    random_mrr_val = prev_baselines.get("Random", 0.0)

    # ---- Load network ----
    print(f"\n[1/4] Loading network from {NETWORK_FILE.name} ...")
    G = nx.read_edgelist(str(NETWORK_FILE))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    n_nodes = G.number_of_nodes()
    print(f"  Network: {n_nodes} nodes, {G.number_of_edges()} edges")

    # ---- Load annotations ----
    print(f"\n[2/4] Loading GO annotations (experimental BP only) ...")
    sgd_map, orf_map, net_nodes = build_alias_mapping()
    annotations, ann_stats = parse_gaf_experimental(sgd_map, orf_map, net_nodes)

    term_freq = Counter()
    for terms in annotations.values():
        term_freq.update(terms)

    node_set = set(G.nodes())
    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in node_set
    }
    n_trials = sum(len(terms) for terms in query_proteins.values())
    print(f"  Proteins: {len(query_proteins)}, Trials: {n_trials}")
    print(f"  Annotation stats: {ann_stats}")
    print(f"  PPI baseline MRR: {ppi_mrr_val:.4f}")

    # ---- Compute new dimensions ----
    dims_to_test = list(NEW_DIMENSIONS)
    # Check if d=1024 is feasible
    if n_nodes - 2 > 1025:
        dims_to_test.extend(OPTIONAL_DIMENSIONS)
        print(f"\n[3/4] Computing Spectral embeddings at d = {dims_to_test} ...")
    else:
        print(f"\n[3/4] Computing Spectral embeddings at d = {dims_to_test} ...")
        print(f"  (d=1024 skipped: need 1025 eigenvectors but only {n_nodes-2} available)")

    new_mrr = {}
    new_eigenvalues = {}
    is_first_dim = True

    for dim in dims_to_test:
        print(f"\n  --- Dimension {dim} ---")

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

        # Spectral gap analysis
        if len(eigenvalues) > 2:
            gap = eigenvalues[2] - eigenvalues[1]
            print(f"  Spectral gap (lambda_3 - lambda_2): {gap:.6f}")
            # Participation ratio of Fiedler vector
            # (we don't have eigenvectors here, just eigenvalues)

        # Save embedding
        emb_path = EMB / f"Spectral_d{dim}_full.npy"
        nodes_path = EMB / f"Spectral_d{dim}_full_nodes.json"
        np.save(str(emb_path), coords)
        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(nodes, f)
        print(f"  Saved embedding to {emb_path}")

        # Run LOTO
        t_loto = time.time()
        mrr_dict = run_loto_for_dimension(
            coords, nodes, G, annotations, term_freq, dim,
            compute_ppi=is_first_dim,
        )
        loto_time = time.time() - t_loto

        # Extract Spectral MRR
        spectral_key = f"Spectral-d{dim}"
        mrr = float(mrr_dict.get(spectral_key, 0.0))
        new_mrr[dim] = mrr
        new_eigenvalues[dim] = eigenvalues.tolist()

        print(f"  MRR(d={dim}) = {mrr:.6f}")
        print(f"  vs PPI ({ppi_mrr_val:.6f}): "
              f"{'EXCEEDS' if mrr > ppi_mrr_val else 'below'} by "
              f"{abs(mrr - ppi_mrr_val):.6f} ({100*abs(mrr - ppi_mrr_val)/ppi_mrr_val:.1f}%)")

        # Check against log-linear prediction
        if prev_mrr:
            from scipy.stats import linregress
            old_log_dims = np.log2(sorted(prev_mrr.keys()))
            old_mrr_vals = [prev_mrr[d] for d in sorted(prev_mrr.keys())]
            old_slope, old_intercept, _, _, _ = linregress(old_log_dims, old_mrr_vals)
            predicted_mrr = old_slope * np.log2(dim) + old_intercept
            deviation = mrr - predicted_mrr
            print(f"  vs predicted ({predicted_mrr:.4f}): "
                  f"{'above' if deviation > 0 else 'below'} by {abs(deviation):.4f}")

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

    # Logarithmic fit
    from scipy.stats import linregress
    sorted_dims = sorted(all_mrr.keys())
    log_dims = np.log2(sorted_dims)
    mrr_vals = [all_mrr[d] for d in sorted_dims]
    slope, intercept, r_value, p_value, std_err = linregress(log_dims, mrr_vals)

    # Predicted values
    predicted_2048 = slope * np.log2(2048) + intercept

    # Determine crossing point
    crossed_at = None
    for d in sorted_dims:
        if all_mrr[d] > ppi_mrr_val:
            crossed_at = d
            break

    # Check log-linearity: fit only d <= 256 vs all
    if len(sorted_dims) > 3:
        mask_256 = np.array([d <= 256 for d in sorted_dims])
        if mask_256.sum() > 2:
            slope_256, intercept_256, r_256, _, _ = linregress(
                log_dims[mask_256], np.array(mrr_vals)[mask_256]
            )
            # Check if new points deviate from d<=256 fit
            for dim in sorted(new_mrr.keys()):
                pred = slope_256 * np.log2(dim) + intercept_256
                actual = new_mrr[dim]
                print(f"  d={dim}: actual={actual:.4f}, "
                      f"d<=256 fit prediction={pred:.4f}, "
                      f"deviation={actual - pred:+.4f}")

    print(f"\n  Full log-linear fit: MRR = {slope:.4f} * log2(d) + {intercept:.4f}")
    print(f"  R^2 = {r_value**2:.4f} (was 0.961 for d=2..256)")
    print(f"  Predicted MRR(d=2048) = {predicted_2048:.4f}")
    if crossed_at:
        print(f"  *** PPI BASELINE CROSSED at d = {crossed_at} ***")

    # Save results
    output = {
        "description": "Dimension Sweep Extension: Spectral Embedding d=2..1024",
        "dimensions": sorted_dims,
        "mrr_by_dimension": {str(d): all_mrr[d] for d in sorted_dims},
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
            "p_value": float(p_value),
            "std_err": float(std_err),
            "predicted_d2048": float(predicted_2048),
        },
        "annotation_stats": ann_stats,
        "n_trials": int(n_trials),
        "n_query_proteins": len(query_proteins),
    }

    out_file = RESULTS / "dimension_sweep_512.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_file}")

    # ---- Plot ----
    plot_512_sweep(all_mrr, ppi_mrr_val, twohop_mrr_val, random_mrr_val,
                   slope, intercept, r_value**2, crossed_at)

    elapsed = time.time() - t_start
    print(f"\nPhase 20 completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return output


# ============================================================
# Figure
# ============================================================

def plot_512_sweep(all_mrr, ppi_mrr, twohop_mrr, random_mrr,
                   slope, intercept, r_squared, crossed_at):
    """Dimension sweep figure including d=512/1024."""
    fig, ax = plt.subplots(figsize=(14, 9))

    dims = sorted(all_mrr.keys())
    mrrs = [all_mrr[d] for d in dims]

    # Phase 13b dimensions (d <= 64)
    phase1_dims = [d for d in dims if d <= 64]
    phase1_mrrs = [all_mrr[d] for d in phase1_dims]
    ax.plot(phase1_dims, phase1_mrrs, "o-", color="#3182bd", linewidth=2.5,
            markersize=10, label="Phase 13b (d=2..64)", zorder=5)

    # Phase 18 dimensions (d = 128, 256)
    phase2_dims = [d for d in dims if 64 < d <= 256]
    phase2_mrrs = [all_mrr[d] for d in phase2_dims]
    if phase2_dims:
        ax.plot(phase2_dims, phase2_mrrs, "s-", color="#e6550d", linewidth=2.5,
                markersize=12, label="Phase 18 (d=128, 256)", zorder=6)

    # Phase 20 dimensions (d = 512, 1024)
    phase3_dims = [d for d in dims if d > 256]
    phase3_mrrs = [all_mrr[d] for d in phase3_dims]
    if phase3_dims:
        ax.plot(phase3_dims, phase3_mrrs, "*-", color="#d62728", linewidth=3,
                markersize=20, label="Phase 20 (d=512, 1024)", zorder=7)

    # Log-linear fit line
    fit_x = np.logspace(np.log2(2), np.log2(max(dims) * 1.5), 100, base=2)
    fit_y = slope * np.log2(fit_x) + intercept
    ax.plot(fit_x, fit_y, "--", color="grey", linewidth=1.5, alpha=0.6,
            label=f"Log-linear fit (R$^2$={r_squared:.3f})")

    # Baselines
    ax.axhline(ppi_mrr, color="#636363", linestyle="--", linewidth=2,
               label=f"PPI-Neighbors ({ppi_mrr:.3f})")
    if twohop_mrr > 0:
        ax.axhline(twohop_mrr, color="#969696", linestyle="--", linewidth=1.5,
                   label=f"2-Hop Diffusion ({twohop_mrr:.3f})")
    if random_mrr > 0:
        ax.axhline(random_mrr, color="#d9d9d9", linestyle="--", linewidth=1,
                   label=f"Random ({random_mrr:.3f})")

    # Annotate all points
    for d, mrr in zip(dims, mrrs):
        if d <= 64:
            color = "#3182bd"
        elif d <= 256:
            color = "#e6550d"
        else:
            color = "#d62728"
        ax.annotate(f"{mrr:.3f}", (d, mrr), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=10,
                    fontweight="bold", color=color)

    # Crossing region
    if crossed_at:
        ax.axvspan(crossed_at - 5, max(dims) * 1.2, alpha=0.08, color="green")
        ax.text(crossed_at, ppi_mrr + 0.005, f"Exceeds PPI\nd>={crossed_at}",
                fontsize=11, fontweight="bold", color="green", ha="center")

    ax.set_xlabel("Embedding Dimension (d)", fontsize=14)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=14)
    ax.set_title("Spectral Embedding Dimension Sweep: d = 2 to " + str(max(dims)),
                 fontsize=15, fontweight="bold")
    ax.set_xscale("log", base=2)
    tick_vals = [d for d in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
                 if d <= max(dims) * 1.5]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([str(d) for d in tick_vals])
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES / "Fig76_dimension_sweep_512.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run()
