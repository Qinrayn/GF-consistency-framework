#!/usr/bin/env python3
"""
heat_kernel_multiscale.py
=========================
Heat Kernel Multi-Scale Analysis on the Yeast PPI Network.

The normalized Laplacian L is a discrete diffusion operator.  The heat
kernel K(t) = exp(-tL) describes how information diffuses across the
network over continuous time t.  Short t captures local structure;
long t captures global structure.

This script answers:
  1. At what diffusion time scale t does the PPI network's functional
     organisation become most visible?
  2. Does the Spectral embedding capture all scales simultaneously?
  3. Is the optimal time scale governed by the spectral gap?

Output
------
results/heat_kernel_multiscale.json
"""

import sys
import json
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh
from scipy.integrate import trapezoid
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, TARGET_STD, GF_R_MIN, GF_R_MAX,
    get_data_dir, get_results_dir,
    compute_gf_score, rescale_coordinates,
    _community_purity, precompute_distance_matrix,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
RESULTS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
TIME_SCALES = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
K_EIG = 50           # number of eigenvectors for heat kernel
N_GF_POINTS = 25     # r-grid resolution for GF curve
R_MIN_GF = 0.05
R_MAX_GF = 0.55

BANNER = "=" * 70


# ==================================================================
# GF curve with connected_components (fast, for large networks)
# ==================================================================

def compute_gf_curve_cc(coords, nodes, go_map, r_vals):
    """Compute G-F purity curve using connected components.

    For large networks greedy_modularity_communities is prohibitively
    slow.  Connected components give a fast, coarse community structure
    that still captures functional organisation at the appropriate
    distance scale.

    Parameters
    ----------
    coords : (n, d) array
    nodes  : list of node labels (with GO annotations)
    go_map : dict  gene -> [GO terms]
    r_vals : array of distance thresholds

    Returns
    -------
    purities : list[float]
    """
    dist_matrix = precompute_distance_matrix(coords)
    n = dist_matrix.shape[0]

    # Pre-sort upper-triangle edges by distance
    iu = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[iu]
    sort_idx = np.argsort(edge_dists)
    sorted_rows = iu[0][sort_idx]
    sorted_cols = iu[1][sort_idx]
    sorted_d = edge_dists[sort_idx]

    r_order = np.argsort(r_vals)
    purities_out = [0.0] * len(r_vals)

    G_r = nx.Graph()
    G_r.add_nodes_from(range(n))
    edge_ptr = 0
    n_edges_total = len(sorted_d)

    _cache = {}  # n_edges -> (communities, purity)

    for orig_idx in r_order:
        r = float(r_vals[orig_idx])

        while edge_ptr < n_edges_total and sorted_d[edge_ptr] < r:
            G_r.add_edge(int(sorted_rows[edge_ptr]),
                         int(sorted_cols[edge_ptr]))
            edge_ptr += 1

        ne = G_r.number_of_edges()
        if ne == 0:
            continue

        if ne in _cache:
            purities_out[orig_idx] = _cache[ne]
        else:
            communities = [set(c) for c in nx.connected_components(G_r)]
            purities = []
            for comm in communities:
                if not comm:
                    continue
                comm_names = [nodes[idx] for idx in comm if idx < len(nodes)]
                purities.append(_community_purity(comm_names, go_map))
            mean_p = float(np.mean(purities)) if purities else 0.0
            _cache[ne] = mean_p
            purities_out[orig_idx] = mean_p

    return purities_out


# ==================================================================
# Step 1: Load network and annotations
# ==================================================================

def load_network_and_annotations():
    """Load yeast PPI, GO annotations; return LCC intersected with
    annotated nodes plus the normalised Laplacian spectrum."""

    print("[1/7] Loading network and annotations ...")

    # Load PPI
    G = nx.Graph()
    edgelist_file = DATA / "yeast_ppi_5936.edgelist"
    with open(edgelist_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])

    # Largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # Load GO annotations
    with open(DATA / "gene_go_map.json", encoding="utf-8") as f:
        go_map = json.load(f)

    # Intersect LCC with annotated nodes
    annotated = sorted(set(go_map.keys()) & set(G.nodes()))
    G = G.subgraph(annotated).copy()

    # Re-take LCC after subgraph extraction
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    nodes = sorted(G.nodes())
    n = len(nodes)
    n_edges = G.number_of_edges()
    print(f"  Working network: {n} nodes, {n_edges} edges")
    print(f"  Annotated nodes in LCC: {n}")

    # Build sparse normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    rows, cols, vals = [], [], []
    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([1.0, 1.0])
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))

    deg = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-10))
    D_inv_sqrt_mat = diags(d_inv_sqrt)
    L_norm = identity(n, format="csr") - D_inv_sqrt_mat @ A @ D_inv_sqrt_mat

    return G, nodes, go_map, L_norm, A, deg


# ==================================================================
# Step 2: Eigendecomposition + heat kernel embeddings
# ==================================================================

def compute_spectrum_and_embeddings(L_norm, nodes, go_map):
    """Compute k smallest eigenpairs and heat kernel embeddings at all
    time scales.  Returns eigenvalues, eigenvectors, and a dict keyed
    by t with embedding arrays."""

    n = len(nodes)
    k = min(K_EIG, n - 2)
    print(f"\n[2/7] Computing eigendecomposition (k={k}) ...")
    t0 = time.time()

    try:
        eigenvalues, eigenvectors = eigsh(L_norm, k=k, which="SM", tol=1e-8)
    except Exception:
        # Fallback: shift-invert
        eigenvalues, eigenvectors = eigsh(L_norm, k=k, sigma=0, which="LM")

    # Sort ascending
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp tiny negative eigenvalues from numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)

    print(f"  Eigendecomposition: {time.time()-t0:.1f}s")
    print(f"  lambda_1 = {eigenvalues[0]:.8f}")
    print(f"  lambda_2 = {eigenvalues[1]:.8f}")
    print(f"  lambda_k = {eigenvalues[-1]:.8f}")

    return eigenvalues, eigenvectors, k


def heat_kernel_embedding(eigenvalues, eigenvectors, t, k):
    """Compute the heat kernel embedding at diffusion time t.

    h_i = [exp(-t*lambda_1)*v_1(i), ..., exp(-t*lambda_k)*v_k(i)]

    Returns the full k-dimensional embedding (n x k).
    """
    decay = np.exp(-t * eigenvalues[:k])   # (k,)
    # Weight each eigenvector column by its decay factor
    embedding = eigenvectors[:, :k] * decay[np.newaxis, :]  # (n, k)
    return embedding


# ==================================================================
# Step 3: G-F Score computation across time scales
# ==================================================================

def evaluate_time_scales(eigenvalues, eigenvectors, nodes, go_map, k):
    """For each time scale t, compute 2D and kD GF Scores."""

    r_vals = np.linspace(R_MIN_GF, R_MAX_GF, N_GF_POINTS)
    print(f"\n[3/7] Evaluating {len(TIME_SCALES)} time scales ...")
    print(f"  GF curve: {N_GF_POINTS} points, r in [{R_MIN_GF}, {R_MAX_GF}]")
    print(f"  GF Score: [{GF_R_MIN}, {GF_R_MAX}]")
    print()

    records = []
    header = (f"  {'t':>8s}  {'GF(2D)':>9s}  {'GF(kD)':>9s}  "
              f"{'Peak':>7s}  {'#Comm':>6s}  {'TopEigC':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for t in TIME_SCALES:
        t_start = time.time()

        # Full k-dimensional heat kernel embedding
        emb_kd = heat_kernel_embedding(eigenvalues, eigenvectors, t, k)
        emb_kd = rescale_coordinates(emb_kd, TARGET_STD)

        # 2D: skip trivial first eigenvector (lambda_1 ~ 0 -> constant),
        # take dimensions corresponding to lambda_2, lambda_3
        emb_2d = emb_kd[:, 1:3].copy()
        emb_2d = rescale_coordinates(emb_2d, TARGET_STD)

        # GF curve (2D) with connected components
        purities_2d = compute_gf_curve_cc(emb_2d, nodes, go_map, r_vals)
        gf_2d = compute_gf_score(r_vals, purities_2d, GF_R_MIN, GF_R_MAX)

        # GF curve (kD) with connected components
        purities_kd = compute_gf_curve_cc(emb_kd, nodes, go_map, r_vals)
        gf_kd = compute_gf_score(r_vals, purities_kd, GF_R_MIN, GF_R_MAX)

        # Peak purity and community count (from 2D, at best r)
        peak_purity = float(max(purities_2d))

        # Number of communities at the r with peak purity
        best_r_idx = int(np.argmax(purities_2d))
        best_r = float(r_vals[best_r_idx])
        dist_mat = precompute_distance_matrix(emb_2d)
        G_at_peak = nx.Graph()
        G_at_peak.add_nodes_from(range(len(nodes)))
        mask = (dist_mat < best_r) & (dist_mat > 0)
        ii, jj = np.where(mask)
        upper = ii < jj
        G_at_peak.add_edges_from(
            list(zip(ii[upper].tolist(), jj[upper].tolist()))
        )
        n_comm = nx.number_connected_components(G_at_peak)

        # Top eigenvalue contribution: exp(-t * lambda_2)
        top_eig_contrib = float(np.exp(-t * eigenvalues[1]))

        elapsed = time.time() - t_start

        marker = ""
        record = {
            "t": float(t),
            "gf_score_2d": float(gf_2d),
            "gf_score_kd": float(gf_kd),
            "peak_purity": peak_purity,
            "n_communities": int(n_comm),
            "top_eigenvalue_contribution": top_eig_contrib,
            "elapsed_s": float(elapsed),
        }
        records.append(record)

        print(f"  {t:8.2f}  {gf_2d:9.5f}  {gf_kd:9.5f}  "
              f"{peak_purity:7.4f}  {n_comm:6d}  {top_eig_contrib:8.5f}"
              f"  ({elapsed:.1f}s)")

    return records, r_vals


# ==================================================================
# Step 4: Compare with Spectral embedding
# ==================================================================

def compare_with_spectral(eigenvalues, eigenvectors, nodes, go_map, records):
    """Compare heat kernel at optimal t with standard Spectral embedding."""

    print(f"\n[4/7] Comparing with Spectral embedding ...")

    r_vals = np.linspace(R_MIN_GF, R_MAX_GF, N_GF_POINTS)

    # Spectral embedding: eigvecs[:, 1:3] (standard 2D)
    spectral_2d = eigenvectors[:, 1:3].copy()
    spectral_2d = rescale_coordinates(spectral_2d, TARGET_STD)

    purities_spectral = compute_gf_curve_cc(spectral_2d, nodes, go_map, r_vals)
    gf_spectral = compute_gf_score(r_vals, purities_spectral, GF_R_MIN, GF_R_MAX)
    print(f"  Spectral (2D, this network) GF Score: {gf_spectral:.5f}")

    # Load curated 153-node Spectral GF Score for reference
    curated_spectral_gf = None
    gf_file = RESULTS / "gf_scores.json"
    if gf_file.exists():
        with open(gf_file, encoding="utf-8") as f:
            gf_data = json.load(f)
        curated_spectral_gf = gf_data.get("scores", {}).get("Spectral", None)
        print(f"  Spectral (curated 153-node)  GF Score: "
              f"{curated_spectral_gf:.5f}" if curated_spectral_gf else
              "  Spectral (curated 153-node)  GF Score: N/A")

    # Best heat kernel
    best_rec = max(records, key=lambda r: r["gf_score_2d"])
    print(f"\n  Best heat kernel (t={best_rec['t']:.2f}) GF Score: "
          f"{best_rec['gf_score_2d']:.5f}")
    print(f"  Improvement over Spectral (this network): "
          f"{best_rec['gf_score_2d'] - gf_spectral:+.5f}")

    return gf_spectral, purities_spectral, curated_spectral_gf


# ==================================================================
# Step 5: Multi-scale optimal time identification
# ==================================================================

def identify_optimal_time(records, eigenvalues, eigenvectors, nodes,
                          go_map, k):
    """Find optimal t*, check for phase transitions, compute correlation
    between decay and GO coherence."""

    print(f"\n[5/7] Multi-scale optimal time identification ...")

    # GF Score vs log(t)
    t_vals = np.array([r["t"] for r in records])
    gf_2d_vals = np.array([r["gf_score_2d"] for r in records])
    gf_kd_vals = np.array([r["gf_score_kd"] for r in records])

    # Optimal t for 2D
    idx_opt_2d = int(np.argmax(gf_2d_vals))
    t_opt_2d = float(t_vals[idx_opt_2d])
    gf_opt_2d = float(gf_2d_vals[idx_opt_2d])
    print(f"  Optimal t* (2D) = {t_opt_2d:.4f}  (GF = {gf_opt_2d:.5f})")

    # Optimal t for kD
    idx_opt_kd = int(np.argmax(gf_kd_vals))
    t_opt_kd = float(t_vals[idx_opt_kd])
    gf_opt_kd = float(gf_kd_vals[idx_opt_kd])
    print(f"  Optimal t* (kD) = {t_opt_kd:.4f}  (GF = {gf_opt_kd:.5f})")

    # Phase transition detection: look for sharp changes in GF Score
    # between consecutive time scales (on log scale)
    log_t = np.log10(t_vals)
    dGF_dlogt = np.diff(gf_2d_vals) / np.diff(log_t)
    max_jump_idx = int(np.argmax(np.abs(dGF_dlogt)))
    phase_transition = {
        "t_before": float(t_vals[max_jump_idx]),
        "t_after": float(t_vals[max_jump_idx + 1]),
        "delta_GF": float(dGF_dlogt[max_jump_idx]),
        "is_sharp": bool(abs(dGF_dlogt[max_jump_idx]) > 0.01),
    }
    print(f"  Largest GF Score jump: between t="
          f"{phase_transition['t_before']:.2f} and "
          f"t={phase_transition['t_after']:.2f} "
          f"(dGF/d(log t) = {phase_transition['delta_GF']:.5f})")

    # Correlation: exp(-t* lambda_i) decay profile vs GO term coherence
    # For each eigenvector, measure GO coherence (functional alignment)
    decay_at_opt = np.exp(-t_opt_2d * eigenvalues[:k])
    go_coherence = _compute_mode_go_coherence(
        eigenvectors[:, :k], nodes, go_map, k
    )
    if len(go_coherence) >= 3 and np.std(go_coherence) > 1e-10:
        rho_decay_coherence, p_decay = spearmanr(decay_at_opt, go_coherence)
        rho_decay_coherence = float(rho_decay_coherence)
        p_decay = float(p_decay)
    else:
        rho_decay_coherence = None
        p_decay = None
    print(f"  Correlation (decay vs GO coherence): "
          f"rho = {rho_decay_coherence}"
          f"{' (p=' + f'{p_decay:.4f})' if p_decay is not None else ''}")

    return (t_opt_2d, gf_opt_2d, t_opt_kd, gf_opt_kd,
            phase_transition, rho_decay_coherence)


def _compute_mode_go_coherence(eigenvectors, nodes, go_map, k):
    """For each eigenvector, measure how well it separates GO groups.
    Returns array of length k."""
    n = len(nodes)
    # Build GO-term groups (only terms with >= 3 members)
    go_groups = {}
    for i, nd in enumerate(nodes):
        for term in go_map.get(nd, []):
            go_groups.setdefault(term, []).append(i)
    go_groups = {t: idxs for t, idxs in go_groups.items() if len(idxs) >= 3}

    if not go_groups:
        return np.zeros(k)

    coherence = np.zeros(k)
    for j in range(k):
        v = eigenvectors[:, j]
        total_var = np.var(v)
        if total_var < 1e-12:
            continue
        within_vars = [np.var(v[idxs]) for idxs in go_groups.values()]
        coherence[j] = 1.0 - np.mean(within_vars) / total_var
    return coherence


# ==================================================================
# Step 6: Spectral gap and characteristic diffusion time
# ==================================================================

def spectral_gap_analysis(eigenvalues, t_opt):
    """Analyse the relationship between optimal diffusion time and
    the spectral gap of the normalised Laplacian."""

    print(f"\n[6/7] Spectral gap and diffusion time analysis ...")

    # For normalised Laplacian, lambda_1 ~ 0
    lambda_1 = float(eigenvalues[0])
    lambda_2 = float(eigenvalues[1])
    lambda_3 = float(eigenvalues[2])

    spectral_gap = lambda_2 - lambda_1   # ~ lambda_2
    t_char = 1.0 / lambda_2 if lambda_2 > 1e-12 else float("inf")
    cheeger_estimate = float(np.sqrt(lambda_2 / 2.0))

    # Ratio: does t_opt ~ t_char?
    ratio = t_opt / t_char if t_char > 0 and t_char != float("inf") else None

    print(f"  lambda_1  = {lambda_1:.8f}")
    print(f"  lambda_2  = {lambda_2:.8f}  (spectral gap)")
    print(f"  lambda_3  = {lambda_3:.8f}")
    print(f"  t_char    = 1/lambda_2 = {t_char:.4f}")
    print(f"  t_opt     = {t_opt:.4f}")
    print(f"  t_opt / t_char = {ratio:.4f}" if ratio is not None else
          "  t_opt / t_char = N/A")
    print(f"  Cheeger h ~ sqrt(lambda_2/2) = {cheeger_estimate:.6f}")

    # Interpretation
    if ratio is not None:
        if 0.5 <= ratio <= 2.0:
            verdict = ("YES -- t_opt is within a factor of 2 of t_char. "
                       "The optimal diffusion time is governed by the "
                       "spectral gap.")
        else:
            verdict = (f"NO -- t_opt / t_char = {ratio:.2f}. The optimal "
                       f"time scale deviates significantly from the "
                       f"characteristic diffusion time.")
        print(f"  t_opt ~ t_char? {verdict}")
    else:
        verdict = "Could not determine (degenerate spectral gap)."
        print(f"  t_opt ~ t_char? {verdict}")

    return {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "lambda_3": lambda_3,
        "spectral_gap": spectral_gap,
        "t_char": float(t_char) if t_char != float("inf") else None,
        "t_opt": float(t_opt),
        "t_opt_over_t_char": float(ratio) if ratio is not None else None,
        "cheeger_estimate": cheeger_estimate,
        "verdict": verdict,
    }


# ==================================================================
# Step 7: Cross-scale consistency
# ==================================================================

def cross_scale_consistency(eigenvalues, eigenvectors, t_opt,
                            spectral_purities, nodes, go_map, k):
    """Compare the GF curve of heat kernel at optimal t with the
    Spectral GF curve."""

    print(f"\n[7/7] Cross-scale consistency ...")

    r_vals = np.linspace(R_MIN_GF, R_MAX_GF, N_GF_POINTS)

    # Heat kernel at optimal t
    emb_hk = heat_kernel_embedding(eigenvalues, eigenvectors, t_opt, k)
    emb_hk_2d = emb_hk[:, 1:3].copy()
    emb_hk_2d = rescale_coordinates(emb_hk_2d, TARGET_STD)

    purities_hk = compute_gf_curve_cc(emb_hk_2d, nodes, go_map, r_vals)

    # Spearman correlation between the two GF curves
    rho, p_val = spearmanr(purities_hk, spectral_purities)
    rho = float(rho)
    p_val = float(p_val)

    gf_hk = compute_gf_score(r_vals, purities_hk, GF_R_MIN, GF_R_MAX)
    gf_sp = compute_gf_score(r_vals, spectral_purities, GF_R_MIN, GF_R_MAX)

    print(f"  Heat kernel (t={t_opt:.2f}) GF Score: {gf_hk:.5f}")
    print(f"  Spectral           GF Score: {gf_sp:.5f}")
    print(f"  Spearman rho (GF curves):  {rho:+.4f}  (p={p_val:.4e})")

    # Shape comparison: peak locations
    r_peak_hk = float(r_vals[np.argmax(purities_hk)])
    r_peak_sp = float(r_vals[np.argmax(spectral_purities)])
    print(f"  Peak r (heat kernel): {r_peak_hk:.4f}")
    print(f"  Peak r (Spectral):    {r_peak_sp:.4f}")

    return {
        "spearman_rho": rho,
        "spearman_p": p_val,
        "gf_heat_kernel": float(gf_hk),
        "gf_spectral": float(gf_sp),
        "peak_r_heat_kernel": r_peak_hk,
        "peak_r_spectral": r_peak_sp,
        "purities_heat_kernel": [float(x) for x in purities_hk],
        "purities_spectral": [float(x) for x in spectral_purities],
    }


# ==================================================================
# Main
# ==================================================================

def main():
    np.random.seed(SEED)

    print(BANNER)
    print("  Heat Kernel Multi-Scale Analysis")
    print("  Yeast PPI Network")
    print(f"  SEED = {SEED}")
    print(BANNER)

    t_total = time.time()

    # ---- Step 1 ----
    G, nodes, go_map, L_norm, A, deg = load_network_and_annotations()

    # ---- Step 2 ----
    eigenvalues, eigenvectors, k = compute_spectrum_and_embeddings(
        L_norm, nodes, go_map
    )

    # ---- Step 3 ----
    records, r_vals = evaluate_time_scales(
        eigenvalues, eigenvectors, nodes, go_map, k
    )

    # ---- Step 4 ----
    gf_spectral, purities_spectral, curated_spectral_gf = \
        compare_with_spectral(eigenvalues, eigenvectors, nodes, go_map,
                              records)

    # ---- Step 5 ----
    (t_opt_2d, gf_opt_2d, t_opt_kd, gf_opt_kd,
     phase_transition, rho_decay_coherence) = \
        identify_optimal_time(records, eigenvalues, eigenvectors, nodes,
                              go_map, k)

    # ---- Step 6 ----
    gap_analysis = spectral_gap_analysis(eigenvalues, t_opt_2d)

    # ---- Step 7 ----
    cross_scale = cross_scale_consistency(
        eigenvalues, eigenvectors, t_opt_2d, purities_spectral,
        nodes, go_map, k
    )

    # =============================================================
    # Save results
    # =============================================================
    print(f"\nSaving results ...")

    output = {
        "analysis": "Heat Kernel Multi-Scale Analysis (yeast PPI)",
        "network": {
            "n_nodes": len(nodes),
            "n_edges": G.number_of_edges(),
        },
        "parameters": {
            "k_eigenvectors": k,
            "n_gf_points": N_GF_POINTS,
            "r_range": [R_MIN_GF, R_MAX_GF],
            "gf_interval": [GF_R_MIN, GF_R_MAX],
            "time_scales": TIME_SCALES,
            "seed": SEED,
        },
        "time_scale_results": records,
        "spectral_comparison": {
            "spectral_gf_this_network": float(gf_spectral),
            "spectral_gf_curated_153": curated_spectral_gf,
            "best_heat_kernel_t": float(t_opt_2d),
            "best_heat_kernel_gf_2d": float(gf_opt_2d),
            "best_heat_kernel_gf_kd": float(gf_opt_kd),
            "improvement_2d": float(gf_opt_2d - gf_spectral),
        },
        "optimal_time": {
            "t_optimal_2d": float(t_opt_2d),
            "gf_optimal_2d": float(gf_opt_2d),
            "t_optimal_kd": float(t_opt_kd),
            "gf_optimal_kd": float(gf_opt_kd),
            "phase_transition": phase_transition,
            "decay_go_coherence_rho": rho_decay_coherence,
        },
        "spectral_gap": float(gap_analysis["spectral_gap"]),
        "t_char": gap_analysis["t_char"],
        "t_optimal": float(t_opt_2d),
        "cheeger_estimate": float(gap_analysis["cheeger_estimate"]),
        "gap_analysis": gap_analysis,
        "cross_scale_correlation": cross_scale["spearman_rho"],
        "cross_scale_detail": cross_scale,
        "r_vals": [float(x) for x in r_vals],
        "summary": _build_summary(
            records, t_opt_2d, gf_opt_2d, gf_spectral,
            curated_spectral_gf, gap_analysis, cross_scale,
            rho_decay_coherence
        ),
    }

    out_file = RESULTS / "heat_kernel_multiscale.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {out_file}")

    # =============================================================
    # Final summary table
    # =============================================================
    elapsed_total = time.time() - t_total
    _print_summary_table(records, t_opt_2d, gf_opt_2d, gf_spectral,
                         gap_analysis, cross_scale, elapsed_total)

    print(f"\nTotal runtime: {elapsed_total:.1f}s")
    print(BANNER)
    return output


# ==================================================================
# Helpers: summary text and table
# ==================================================================

def _build_summary(records, t_opt, gf_opt, gf_spectral,
                    curated_spectral, gap_analysis, cross_scale,
                    rho_decay):
    """Build a human-readable summary string."""
    lines = [
        "Heat Kernel Multi-Scale Analysis Summary",
        "=" * 50,
        "",
        f"Optimal diffusion time t* = {t_opt:.4f}",
        f"  GF Score at t* (2D): {gf_opt:.5f}",
        f"  Spectral GF Score (this network): {gf_spectral:.5f}",
    ]
    if curated_spectral is not None:
        lines.append(
            f"  Spectral GF Score (curated 153-node): {curated_spectral:.5f}"
        )
    lines += [
        "",
        f"  Improvement over Spectral: {gf_opt - gf_spectral:+.5f}",
        "",
        "Spectral gap analysis:",
        f"  lambda_2 = {gap_analysis['lambda_2']:.8f}",
        f"  t_char = 1/lambda_2 = {gap_analysis['t_char']:.4f}",
        f"  t_opt / t_char = "
        f"{gap_analysis['t_opt_over_t_char']:.4f}"
        if gap_analysis['t_opt_over_t_char'] is not None
        else "  t_opt / t_char = N/A",
        f"  Cheeger h ~ {gap_analysis['cheeger_estimate']:.6f}",
        f"  Verdict: {gap_analysis['verdict']}",
        "",
        "Cross-scale consistency:",
        f"  Spearman rho (HK vs Spectral GF curves): "
        f"{cross_scale['spearman_rho']:+.4f}",
        f"  (p = {cross_scale['spearman_p']:.4e})",
        "",
    ]
    if rho_decay is not None:
        lines.append(
            f"  Decay-GO coherence correlation: {rho_decay:+.4f}"
        )

    return "\n".join(lines)


def _print_summary_table(records, t_opt, gf_opt, gf_spectral,
                         gap_analysis, cross_scale, elapsed):
    """Print a clear summary table showing GF Score vs time scale."""
    print()
    print(BANNER)
    print("  HEAT KERNEL MULTI-SCALE ANALYSIS -- SUMMARY")
    print(BANNER)

    # Time-scale table
    print()
    print(f"  {'t':>8s}  {'GF(2D)':>10s}  {'GF(kD)':>10s}  "
          f"{'PeakPur':>9s}  {'#Comm':>6s}")
    print("  " + "-" * 52)

    for rec in records:
        marker = " <-- OPTIMAL" if abs(rec["t"] - t_opt) < 1e-6 else ""
        print(f"  {rec['t']:8.2f}  {rec['gf_score_2d']:10.5f}  "
              f"{rec['gf_score_kd']:10.5f}  "
              f"{rec['peak_purity']:9.4f}  "
              f"{rec['n_communities']:6d}{marker}")

    # Comparison block
    print()
    print("  " + "-" * 52)
    print(f"  Spectral (2D, this network): GF = {gf_spectral:.5f}")
    print(f"  Heat kernel (t*={t_opt:.2f}):    GF = {gf_opt:.5f}")
    print(f"  Difference:                    "
          f"{gf_opt - gf_spectral:+.5f}")
    print()

    # Spectral gap
    print(f"  Spectral gap (lambda_2):    "
          f"{gap_analysis['lambda_2']:.8f}")
    print(f"  Characteristic time t_char: "
          f"{gap_analysis['t_char']:.4f}")
    print(f"  Optimal time t_opt:         {t_opt:.4f}")
    ratio = gap_analysis.get("t_opt_over_t_char")
    if ratio is not None:
        print(f"  Ratio t_opt / t_char:       {ratio:.4f}")
    print(f"  Cheeger h estimate:         "
          f"{gap_analysis['cheeger_estimate']:.6f}")
    print()

    # Cross-scale
    print(f"  Cross-scale Spearman rho:   "
          f"{cross_scale['spearman_rho']:+.4f} "
          f"(p={cross_scale['spearman_p']:.4e})")
    print()
    print(f"  Total runtime: {elapsed:.1f}s")


# ==================================================================
# Entry point
# ==================================================================

if __name__ == "__main__":
    main()
