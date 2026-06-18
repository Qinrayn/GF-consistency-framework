#!/usr/bin/env python3
"""
Position Encoding Comparison for Graph Transformers
=====================================================
Systematic benchmark of graph transformer position encoding (PE) methods
on the curated 153-node yeast PPI network using the G-F Score framework.

PE methods evaluated:
  1. Laplacian Eigenvector PE ("Spectral" -- our existing baseline)
  2. Random Walk Positional Encoding (RWPE, diagonal return probabilities)
  3. Random Walk Landing Probabilities (full off-diagonal, PCA-reduced)
  4. Sign-Invariant PE (simplified SignNet, Lim et al. ICLR 2023)
  5. Multi-dimension sweep: Laplacian + RWPE at k = 2, 4, 8, 16, 32

Additional analyses:
  - Eigenvector sign-flip robustness (4 unique sign combinations for 2D PE)
  - Cross-reference with existing 11 embedding methods

Key finding: when projecting to 2D, the first 2 PE dimensions are invariant
to the total number of dimensions computed. This means higher-k PEs do not
improve GF Score when evaluated in 2D -- the benefit of higher-dimensional
PEs only manifests when the GF curve is computed in the full k-D space.

References:
  - Dwivedi et al., "Generalization in Graph Neural Networks" (AISTATS 2022)
  - Rampasek et al., "Recipe for a General, Powerful, Scalable Graph
    Transformer" (NeurIPS 2022) -- GraphGPS
  - Lim et al., "Equivariant and Stable PE Using Basis Invariant
    Sign-Invariant Features" (ICLR 2023) -- SignNet
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.integrate import trapezoid
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_curated_network,
    compute_gf_curve,
    compute_gf_score,
    rescale_coordinates,
    compute_plateau_width,
    TARGET_STD,
    SEED,
    GF_R_MIN,
    GF_R_MAX,
    R_MIN,
    R_MAX,
    N_POINTS,
    get_results_dir,
    get_data_dir,
    setup_logging,
)

# ============================================================
# Constants
# ============================================================

RESULTS = get_results_dir()
RESULTS.mkdir(parents=True, exist_ok=True)

logger = setup_logging("position_encoding_comparison")

BANNER = "=" * 72
R_VALS = np.linspace(R_MIN, R_MAX, N_POINTS)

# Random walk parameters (GraphGPS defaults)
RWPE_STEPS = 16       # r steps for diagonal RWPE
RWPE_FULL_STEPS = 4   # r steps for full landing probabilities
SIGNNET_K = 16        # eigenvectors for SignNet-like features

# Dimensions to sweep for multi-dimension experiment
DIM_SWEEP = [4, 8, 16, 32]

# Coarser grid for sign flip robustness (still sufficient for GF Score)
SIGN_FLIP_N_POINTS = 50


# ============================================================
# GF Curve Cache
# ============================================================

_gf_cache = {}  # key: bytes hash of coords -> (purities, mods)


def _coords_key(coords_2d):
    """Create a hashable key from 2D coordinates for caching."""
    return coords_2d.tobytes()


def compute_gf_curve_cached(coords_2d, nodes, go_map, r_vals):
    """Cached version of compute_gf_curve: skip if identical coords seen."""
    key = _coords_key(coords_2d)
    if key in _gf_cache:
        return _gf_cache[key]
    result = compute_gf_curve(coords_2d, nodes, go_map, r_vals)
    _gf_cache[key] = result
    return result


# ============================================================
# Step 1: Load Network
# ============================================================

def load_data():
    """Load curated 153-node PPI, GO annotations, and existing GF scores."""
    logger.info("Loading curated 153-node yeast PPI network...")
    G, nodes, go_map = load_curated_network()
    logger.info("  Nodes: %d, Edges: %d", G.number_of_nodes(), G.number_of_edges())

    # Existing GF scores for cross-reference
    gf_file = RESULTS / "gf_scores.json"
    with open(gf_file, encoding="utf-8") as f:
        existing_data = json.load(f)
    existing_scores = existing_data.get("scores", {})

    # GNN GF scores
    gnn_file = RESULTS / "gnn_gf_scores.json"
    if gnn_file.exists():
        with open(gnn_file, encoding="utf-8") as f:
            gnn_data = json.load(f)
        existing_scores.update(gnn_data.get("gf_scores", {}))

    logger.info("  Existing methods: %d", len(existing_scores))
    return G, nodes, go_map, existing_scores


# ============================================================
# Step 2: Position Encoding Methods
# ============================================================

def compute_normalized_laplacian(G, nodes):
    """Compute the normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}."""
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray()
    return L


def compute_laplacian_eigendecomposition(L, k):
    """Compute first k non-trivial eigenvectors of normalized Laplacian.

    Returns eigenvalues and eigenvectors sorted by ascending eigenvalue,
    skipping the trivial constant eigenvector (eigenvalue ~ 0).
    """
    eigvals, eigvecs = np.linalg.eigh(L)
    # Skip the first (trivial) eigenvector
    return eigvals[1:k + 1], eigvecs[:, 1:k + 1]


def method_laplacian_pe(G, nodes, k=2):
    """Method 1: Laplacian Eigenvector PE (a.k.a. Spectral).

    Uses the first k Fiedler vectors of the normalized Laplacian.
    """
    L = compute_normalized_laplacian(G, nodes)
    _, eigvecs = compute_laplacian_eigendecomposition(L, k)
    return eigvecs


def method_rwpe_diagonal(G, nodes, r_steps=RWPE_STEPS, k=2):
    """Method 2: Random Walk Positional Encoding (diagonal return probs).

    PE_i = [T^1(i,i), T^2(i,i), ..., T^r(i,i)]
    Then take first k columns for k-D embedding.

    Reference: Dwivedi et al. 2020, GraphGPS (Rampasek 2022).
    """
    n = len(nodes)
    A = nx.adjacency_matrix(G, nodelist=nodes, weight=None).toarray().astype(np.float64)
    deg = A.sum(axis=1)
    deg_inv = np.zeros(n)
    deg_inv[deg > 0] = 1.0 / deg[deg > 0]
    T = np.diag(deg_inv) @ A  # Transition matrix T = D^{-1} A

    pe = np.zeros((n, r_steps))
    T_power = T.copy()
    for step in range(r_steps):
        pe[:, step] = np.diag(T_power)
        if step < r_steps - 1:
            T_power = T_power @ T

    return pe[:, :k]


def method_rwpe_full_landing(G, nodes, r_steps=RWPE_FULL_STEPS):
    """Method 3: Random Walk Landing Probabilities (full rows, PCA to 2D).

    PE_i = concatenation of full rows T^1(i,:), T^2(i,:), ..., T^r(i,:)
    Dimensionality: r_steps * n -> PCA to 2D.

    This captures richer structural information than just diagonal elements.
    """
    n = len(nodes)
    A = nx.adjacency_matrix(G, nodelist=nodes, weight=None).toarray().astype(np.float64)
    deg = A.sum(axis=1)
    deg_inv = np.zeros(n)
    deg_inv[deg > 0] = 1.0 / deg[deg > 0]
    T = np.diag(deg_inv) @ A

    # Concatenate full landing probability rows
    features = []
    T_power = T.copy()
    for step in range(r_steps):
        features.append(T_power)
        if step < r_steps - 1:
            T_power = T_power @ T

    # features: list of (n, n) matrices -> concatenate to (n, r_steps*n)
    pe_full = np.hstack(features)

    # PCA to 2D
    n_components = min(2, n - 1, pe_full.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    coords = pca.fit_transform(pe_full)
    return coords


def method_sign_invariant_pe(G, nodes, k=SIGNNET_K):
    """Method 4: Sign-Invariant PE (simplified SignNet, Lim et al. 2023).

    For each eigenvector v_j of L_norm, compute sign-invariant features:
      - |v_j(i)|              (absolute values)
      - v_j(i)^2              (squared)
      - sum_{nbr k} |v_j(i) - v_j(k)|  (sign-invariant gradient)
    Concatenate across first k eigenvectors -> PCA to 2D.

    The full SignNet uses a neural network over these features; here we
    use the handcrafted features directly, which captures the essence of
    sign-invariance without requiring training.
    """
    n = len(nodes)
    L = compute_normalized_laplacian(G, nodes)
    _, eigvecs = compute_laplacian_eigendecomposition(L, k)

    # Build adjacency list for gradient computation (vectorized)
    A = nx.adjacency_matrix(G, nodelist=nodes, weight=None).toarray()

    features = []
    for j in range(min(k, eigvecs.shape[1])):
        v = eigvecs[:, j]
        # Feature 1: absolute values
        feat_abs = np.abs(v)
        # Feature 2: squared values
        feat_sq = v ** 2
        # Feature 3: sign-invariant gradient (sum of abs differences over neighbors)
        # Vectorized: for each node i, sum |v(i) - v(k)| for all neighbors k
        # Use broadcasting: diff[i, k] = |v[i] - v[k]|, then mask by adjacency
        diff = np.abs(v[:, np.newaxis] - v[np.newaxis, :])  # (n, n)
        feat_grad = (A * diff).sum(axis=1)  # sum over neighbors

        features.extend([feat_abs, feat_sq, feat_grad])

    # Stack: (n, 3*k) feature matrix
    pe = np.column_stack(features)

    # PCA to 2D
    n_components = min(2, n - 1, pe.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    coords = pca.fit_transform(pe)
    return coords


# ============================================================
# Step 3: G-F Score Evaluation
# ============================================================

def evaluate_pe_method(coords, nodes, go_map, method_name):
    """Evaluate a PE method: rescale, compute GF curve, score, plateau.

    Uses a coordinate cache to avoid recomputing GF curves when the 2D
    projection is identical (e.g., Laplacian PE at k=2 vs k=32 both
    project to the same first 2 eigenvectors).
    """
    coords_2d = coords[:, :2] if coords.shape[1] > 2 else coords
    if coords_2d.shape[1] < 2:
        # Pad with zeros if only 1D
        coords_2d = np.column_stack([coords_2d, np.zeros(len(coords_2d))])

    # Rescale to TARGET_STD
    coords_scaled = rescale_coordinates(coords_2d, target_std=TARGET_STD)

    # Compute GF curve (cached)
    purities, mods = compute_gf_curve_cached(coords_scaled, nodes, go_map, R_VALS)

    # Compute GF Score
    gf_score = compute_gf_score(R_VALS, purities, r_min=GF_R_MIN, r_max=GF_R_MAX)

    # Compute plateau width
    plateau = compute_plateau_width(R_VALS, purities)

    return {
        "method": method_name,
        "gf_score": float(gf_score),
        "peak_purity": float(plateau["peak_purity"]),
        "plateau_width": float(plateau["W"]),
    }


# ============================================================
# Step 5: Eigenvector Sign-Flip Robustness
# ============================================================

def sign_flip_robustness(G, nodes, go_map, k=16):
    """Test whether Laplacian PE GF Score is robust to eigenvector sign ambiguity.

    For 2D Laplacian PE (projecting to first 2 eigenvectors), there are only
    4 unique sign combinations: (++), (+-), (-+), (--). We compute all 4
    and verify they yield identical GF Scores.

    Theoretical guarantee: Euclidean pairwise distances are invariant to
    column sign flips. For coords = [v1, v2], flipping v1 gives [-v1, v2].
    The distance d(i,j)^2 = (v1_i - v1_j)^2 + (v2_i - v2_j)^2 is unchanged
    because (-v1_i + v1_j)^2 = (v1_i - v1_j)^2. Therefore, the spatial graph
    at every radius r is identical, and the GF curve is exactly preserved.

    This function empirically verifies that property and also tests 20 random
    sign flips of all k=16 eigenvectors (only the first 2 matter for 2D).
    """
    L = compute_normalized_laplacian(G, nodes)
    _, eigvecs = compute_laplacian_eigendecomposition(L, k)

    # Use coarser grid for sign flip test (still sufficient for GF Score)
    r_vals_flip = np.linspace(R_MIN, R_MAX, SIGN_FLIP_N_POINTS)

    # Phase 1: Exhaustive 4 combinations for the 2 relevant eigenvectors
    sign_combos = [
        np.array([1.0, 1.0]),
        np.array([1.0, -1.0]),
        np.array([-1.0, 1.0]),
        np.array([-1.0, -1.0]),
    ]

    gf_scores_exhaustive = []
    for signs_2 in sign_combos:
        coords_2d = eigvecs[:, :2] * signs_2[np.newaxis, :]
        coords_scaled = rescale_coordinates(coords_2d, target_std=TARGET_STD)
        purities, _ = compute_gf_curve(coords_scaled, nodes, go_map, r_vals_flip)
        score = compute_gf_score(r_vals_flip, purities, r_min=GF_R_MIN, r_max=GF_R_MAX)
        gf_scores_exhaustive.append(float(score))

    # Phase 2: 20 random sign flips of all k eigenvectors (only first 2 matter)
    rng = np.random.RandomState(SEED)
    gf_scores_random = []
    for _ in range(20):
        signs = rng.choice([-1.0, 1.0], size=k)
        flipped = eigvecs * signs[np.newaxis, :]
        coords_2d = flipped[:, :2]
        coords_scaled = rescale_coordinates(coords_2d, target_std=TARGET_STD)
        purities, _ = compute_gf_curve(coords_scaled, nodes, go_map, r_vals_flip)
        score = compute_gf_score(r_vals_flip, purities, r_min=GF_R_MIN, r_max=GF_R_MAX)
        gf_scores_random.append(float(score))

    all_scores = gf_scores_exhaustive + gf_scores_random
    return {
        "mean_gf": float(np.mean(all_scores)),
        "std_gf": float(np.std(all_scores)),
        "min_gf": float(np.min(all_scores)),
        "max_gf": float(np.max(all_scores)),
        "exhaustive_4_scores": gf_scores_exhaustive,
        "random_20_scores": gf_scores_random,
        "n_exhaustive": 4,
        "n_random": 20,
        "k_eigenvectors": k,
        "theoretical_note": (
            "Euclidean distances are invariant to eigenvector sign flips, "
            "so all sign combinations yield identical GF curves when "
            "projecting to 2D. Std should be 0 or near-zero (floating point)."
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    np.random.seed(SEED)

    print(BANNER)
    print("  Position Encoding Comparison -- Graph Transformer PE Benchmark")
    print(BANNER)

    # ----------------------------------------------------------
    # Step 1: Load data
    # ----------------------------------------------------------
    G, nodes, go_map, existing_scores = load_data()

    all_results = []

    # ----------------------------------------------------------
    # Step 2a: Method 1 -- Laplacian Eigenvector PE (k=2)
    # ----------------------------------------------------------
    print("\n[1/6] Laplacian Eigenvector PE (k=2, a.k.a. Spectral)...")
    coords_lap2 = method_laplacian_pe(G, nodes, k=2)
    result = evaluate_pe_method(coords_lap2, nodes, go_map, "Laplacian_PE_k2")
    all_results.append(result)
    print("  GF Score = %.4f  (peak purity = %.4f, plateau W = %.4f)"
          % (result["gf_score"], result["peak_purity"], result["plateau_width"]))

    # ----------------------------------------------------------
    # Step 2b: Method 2 -- RWPE Diagonal (k=2, r=16)
    # ----------------------------------------------------------
    print("\n[2/6] Random Walk PE -- Diagonal (r=16, k=2)...")
    coords_rwpe = method_rwpe_diagonal(G, nodes, r_steps=RWPE_STEPS, k=2)
    result = evaluate_pe_method(coords_rwpe, nodes, go_map, "RWPE_diagonal_k2")
    all_results.append(result)
    print("  GF Score = %.4f  (peak purity = %.4f, plateau W = %.4f)"
          % (result["gf_score"], result["peak_purity"], result["plateau_width"]))

    # ----------------------------------------------------------
    # Step 2c: Method 3 -- RWPE Full Landing Probabilities (r=4, PCA to 2D)
    # ----------------------------------------------------------
    print("\n[3/6] Random Walk Landing Probabilities (r=4, PCA->2D)...")
    coords_rwpe_full = method_rwpe_full_landing(G, nodes, r_steps=RWPE_FULL_STEPS)
    result = evaluate_pe_method(coords_rwpe_full, nodes, go_map, "RWPE_full_r4_PCA2D")
    all_results.append(result)
    print("  GF Score = %.4f  (peak purity = %.4f, plateau W = %.4f)"
          % (result["gf_score"], result["peak_purity"], result["plateau_width"]))

    # ----------------------------------------------------------
    # Step 2d: Method 4 -- Sign-Invariant PE (SignNet, k=16, PCA to 2D)
    # ----------------------------------------------------------
    print("\n[4/6] Sign-Invariant PE (SignNet, k=16, PCA->2D)...")
    coords_signnet = method_sign_invariant_pe(G, nodes, k=SIGNNET_K)
    result = evaluate_pe_method(coords_signnet, nodes, go_map, "SignNet_k16_PCA2D")
    all_results.append(result)
    print("  GF Score = %.4f  (peak purity = %.4f, plateau W = %.4f)"
          % (result["gf_score"], result["peak_purity"], result["plateau_width"]))

    # ----------------------------------------------------------
    # Step 2e: Method 5 -- Multi-dimension sweep
    # ----------------------------------------------------------
    print("\n[5/6] Multi-dimension sweep: Laplacian PE + RWPE at k = %s..." % DIM_SWEEP)
    print("  NOTE: projecting to 2D means first 2 dims are identical across k.")
    print("  GF Scores will be the same for all k (coordinate cache active).")
    for k in DIM_SWEEP:
        # Laplacian PE at dimension k, evaluate first 2
        coords_lap_k = method_laplacian_pe(G, nodes, k=max(k, 2))
        name_lap = "Laplacian_PE_k%d" % k
        result_lap = evaluate_pe_method(coords_lap_k, nodes, go_map, name_lap)
        all_results.append(result_lap)
        cached = "(cached)" if _coords_key(rescale_coordinates(
            coords_lap_k[:, :2], target_std=TARGET_STD)) in _gf_cache else ""
        print("  %-28s  GF Score = %.4f  %s" % (name_lap, result_lap["gf_score"], cached))

        # RWPE at dimension k, evaluate first 2
        coords_rwpe_k = method_rwpe_diagonal(G, nodes, r_steps=max(k, RWPE_STEPS),
                                              k=max(k, 2))
        name_rwpe = "RWPE_diagonal_k%d" % k
        result_rwpe = evaluate_pe_method(coords_rwpe_k, nodes, go_map, name_rwpe)
        all_results.append(result_rwpe)
        cached = "(cached)" if _coords_key(rescale_coordinates(
            coords_rwpe_k[:, :2], target_std=TARGET_STD)) in _gf_cache else ""
        print("  %-28s  GF Score = %.4f  %s" % (name_rwpe, result_rwpe["gf_score"], cached))

    # ----------------------------------------------------------
    # Step 5: Eigenvector sign-flip robustness
    # ----------------------------------------------------------
    print("\n[6/6] Eigenvector sign-flip robustness (4 exhaustive + 20 random)...")
    sign_flip = sign_flip_robustness(G, nodes, go_map, k=16)
    print("  Exhaustive (4 combos): %s" %
          ", ".join("%.6f" % s for s in sign_flip["exhaustive_4_scores"]))
    print("  Mean GF = %.6f, Std = %.8f, Range = [%.6f, %.6f]"
          % (sign_flip["mean_gf"], sign_flip["std_gf"],
             sign_flip["min_gf"], sign_flip["max_gf"]))
    print("  Theoretical: std should be ~0 (Euclidean distance invariance)")

    # ----------------------------------------------------------
    # Step 4: Comparison with existing methods
    # ----------------------------------------------------------
    print("\n" + BANNER)
    print("  COMPARISON TABLE: All Methods Ranked by G-F Score")
    print(BANNER)

    # Merge PE results with existing scores
    combined = []
    for method, score in existing_scores.items():
        combined.append({
            "method": method,
            "dimension": "various",
            "gf_score": float(score),
            "category": "existing",
        })
    for r in all_results:
        combined.append({
            "method": r["method"],
            "dimension": "2D (from PE)",
            "gf_score": float(r["gf_score"]),
            "category": "PE_method",
        })

    # Sort by GF Score descending
    combined.sort(key=lambda x: x["gf_score"], reverse=True)

    # Assign ranks
    print("\n  %-4s  %-30s  %-12s  %-14s  %s"
          % ("Rank", "Method", "GF Score", "Dimension", "Category"))
    print("  " + "-" * 72)
    for rank, entry in enumerate(combined, 1):
        print("  %-4d  %-30s  %-12.4f  %-14s  %s"
              % (rank, entry["method"], entry["gf_score"],
                 entry["dimension"], entry["category"]))

    # ----------------------------------------------------------
    # Step 6: Save results
    # ----------------------------------------------------------
    output = {
        "pe_methods": [],
        "sign_flip_robustness": {
            k_key: (v_val if not isinstance(v_val, (np.floating, np.integer))
                    else float(v_val) if isinstance(v_val, np.floating)
                    else int(v_val))
            for k_key, v_val in sign_flip.items()
            if not isinstance(v_val, list)
        },
        "sign_flip_exhaustive_scores": sign_flip["exhaustive_4_scores"],
        "sign_flip_random_scores": sign_flip["random_20_scores"],
        "comparison_ranking": [],
        "existing_methods": {m: float(s) for m, s in existing_scores.items()},
        "dimension_invariance_note": (
            "When projecting to 2D, the first 2 PE dimensions are identical "
            "regardless of k (total dimensions computed). This means "
            "Laplacian_PE_k2 and Laplacian_PE_k32 have the same GF Score "
            "when evaluated in 2D. The benefit of higher-k PEs only "
            "manifests when the GF curve is computed in the full k-D space."
        ),
        "parameters": {
            "rwpe_steps": RWPE_STEPS,
            "rwpe_full_steps": RWPE_FULL_STEPS,
            "signnet_k": SIGNNET_K,
            "sign_flip_exhaustive": 4,
            "sign_flip_random": 20,
            "dim_sweep": DIM_SWEEP,
            "target_std": TARGET_STD,
            "gf_r_min": GF_R_MIN,
            "gf_r_max": GF_R_MAX,
            "n_points": N_POINTS,
            "seed": SEED,
        },
    }

    for r in all_results:
        output["pe_methods"].append({
            "method": r["method"],
            "gf_score": float(r["gf_score"]),
            "peak_purity": float(r["peak_purity"]),
            "plateau_width": float(r["plateau_width"]),
        })

    for rank, entry in enumerate(combined, 1):
        output["comparison_ranking"].append({
            "rank": rank,
            "method": entry["method"],
            "gf_score": float(entry["gf_score"]),
            "category": entry["category"],
        })

    out_path = RESULTS / "position_encoding_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nResults saved to: %s" % out_path)

    elapsed = time.time() - t0
    print("Total time: %.1f seconds" % elapsed)
    print("GF curve cache hits: %d computations avoided" %
          (len(all_results) + 24 - len(_gf_cache)))
    print(BANNER)


if __name__ == "__main__":
    main()
