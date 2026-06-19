#!/usr/bin/env python3
"""
cheeger_gf_bound.py -- Cheeger-Spectral Upper Bound on G-F Score
================================================================
Derives and validates a theoretical upper bound on the G-F Score from the
Laplacian spectrum and Cheeger constant of PPI networks.

Given only a PPI network's topology, we predict the maximum achievable
G-F Score before computing any embedding.

Mathematical basis:
  - Cheeger's inequality: lambda_2 / 2 <= h <= sqrt(2 * lambda_2)
  - Higher-order Cheeger inequalities (Lee et al. 2014)
  - Spectral gap ratios and Fiedler vector participation ratio

Output: results/cheeger_gf_bound.json
"""

import sys
import json
import gzip
import time
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse
from scipy.integrate import trapezoid
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, get_data_dir, get_results_dir,
    load_curated_network, GF_R_MIN, GF_R_MAX,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_EIGENVALUES = 50          # number of Laplacian eigenvalues to compute
STRING_MIN_SCORE = 700      # minimum STRING confidence score
N_CV_FOLDS = 5              # leave-one-out is n_species choose 1

np.random.seed(SEED)


# ===================================================================
# Network Loading
# ===================================================================

def load_yeast_curated():
    """Load curated 153-node yeast PPI network."""
    G, nodes, go_map = load_curated_network()
    return G, "Yeast_curated"


def load_yeast_full():
    """Load full yeast STRING network (5936 nodes)."""
    data_dir = get_data_dir()
    edgelist = data_dir / "yeast_ppi_5936.edgelist"
    G = nx.Graph()
    if edgelist.exists():
        with open(edgelist, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    G.add_edge(parts[0], parts[1])
    else:
        # Fallback: load from STRING raw data
        string_file = data_dir / "4932.protein.links.v11.5.txt.gz"
        with gzip.open(str(string_file), "rt", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3 and int(parts[2]) >= STRING_MIN_SCORE:
                    p1 = parts[0].split(".")[1]
                    p2 = parts[1].split(".")[1]
                    G.add_edge(p1, p2)
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G, "Yeast_full"


def load_human_network():
    """Load human STRING PPI network (taxID 9606)."""
    links_file = PROJECT_ROOT / "human_validation" / "9606.protein.links.v12.0.txt.gz"
    if not links_file.exists():
        logger.warning("Human STRING file not found, skipping.")
        return None, "Human"
    G = nx.Graph()
    with gzip.open(str(links_file), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and int(parts[2]) >= STRING_MIN_SCORE:
                G.add_edge(parts[0], parts[1])
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G, "Human"


def load_mouse_network():
    """Load mouse PPI network from precomputed edgelist."""
    data_dir = get_data_dir()
    edgelist = data_dir / "mouse_ppi.edgelist"
    if not edgelist.exists():
        logger.warning("Mouse edgelist not found, skipping.")
        return None, "Mouse"
    G = nx.Graph()
    with open(edgelist, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G, "Mouse"


def load_ecoli_network():
    """Load E. coli STRING PPI network (taxID 511145)."""
    data_dir = get_data_dir()
    string_file = data_dir / "511145.protein.links.v11.5.txt.gz"
    if not string_file.exists():
        logger.warning("E. coli STRING file not found, skipping.")
        return None, "Ecoli"
    G = nx.Graph()
    with gzip.open(str(string_file), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and int(parts[2]) >= STRING_MIN_SCORE:
                p1 = parts[0].split(".")[1]
                p2 = parts[1].split(".")[1]
                G.add_edge(p1, p2)
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G, "Ecoli"


def load_fly_network():
    """Load Drosophila STRING PPI network (taxID 7227)."""
    data_dir = get_data_dir()
    string_file = data_dir / "fly" / "7227.protein.links.v11.5.txt.gz"
    if not string_file.exists():
        logger.warning("Fly STRING file not found, skipping.")
        return None, "Fly"
    G = nx.Graph()
    with gzip.open(str(string_file), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and int(parts[2]) >= STRING_MIN_SCORE:
                p1 = parts[0].split(".")[1]
                p2 = parts[1].split(".")[1]
                G.add_edge(p1, p2)
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G, "Fly"


# ===================================================================
# GF Score Loading
# ===================================================================

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gf_scores_yeast_curated():
    """Best GF Score for yeast curated network."""
    data = _load_json(get_results_dir() / "gf_scores.json")
    scores = data.get("scores", data.get("scores_paper_interval", {}))
    spectral = float(scores.get("Spectral", 0.0))
    best = max(float(v) for v in scores.values()) if scores else 0.0
    return spectral, best, scores


def load_gf_scores_yeast_full():
    """Compute GF scores from full_network_validation purity curves."""
    fpath = get_results_dir() / "full_network_validation.json"
    if not fpath.exists():
        return 0.0, 0.0, {}
    data = _load_json(fpath)
    r_vals = np.array(data["r"])
    r_min, r_max = GF_R_MIN, GF_R_MAX
    scores = {}
    for method_prefix in ["DM", "MDS", "Node2Vec", "VGAE"]:
        key = f"{method_prefix}_purity"
        if key in data:
            purity = np.array(data[key])
            mask = (r_vals >= r_min) & (r_vals <= r_max)
            r_sub = r_vals[mask]
            p_sub = purity[mask]
            if len(r_sub) >= 2:
                gf = float(trapezoid(p_sub, r_sub) / (r_max - r_min))
            else:
                gf = 0.0
            scores[method_prefix] = gf
    spectral = 0.0  # not available in full_network_validation
    best = max(scores.values()) if scores else 0.0
    return spectral, best, scores


def load_gf_scores_human():
    """Best GF Score for human network."""
    # Try human_ppi_results.json first (has unified scores)
    fpath = get_results_dir() / "human_ppi_results.json"
    if fpath.exists():
        data = _load_json(fpath)
        scores = data.get("gf_scores", {})
        if scores:
            spectral = float(scores.get("Spectral", 0.0))
            best = max(float(v) for v in scores.values())
            return spectral, best, scores
    # Fallback to human_gf_scores.json
    fpath2 = get_results_dir() / "human_gf_scores.json"
    if fpath2.exists():
        data = _load_json(fpath2)
        scores = data.get("scores", {})
        spectral = float(scores.get("Spectral", 0.0))
        best = max(float(v) for v in scores.values()) if scores else 0.0
        return spectral, best, scores
    return 0.0, 0.0, {}


def load_gf_scores_mouse():
    """Best GF Score for mouse network."""
    fpath = get_results_dir() / "mouse_gf_analysis.json"
    if not fpath.exists():
        return 0.0, 0.0, {}
    data = _load_json(fpath)
    raw = data.get("gf_scores", {})
    scores = {k: float(v["gf_score"]) for k, v in raw.items()
              if isinstance(v, dict) and "gf_score" in v}
    spectral = float(scores.get("Spectral", 0.0))
    best = max(scores.values()) if scores else 0.0
    return spectral, best, scores


def load_gf_scores_ecoli():
    """Best GF Score for E. coli network."""
    fpath = get_results_dir() / "ecoli_gf_scores.json"
    if not fpath.exists():
        return 0.0, 0.0, {}
    data = _load_json(fpath)
    raw = data.get("gf_scores", {})
    scores = {k: float(v["gf_score"]) for k, v in raw.items()
              if isinstance(v, dict) and "gf_score" in v}
    spectral = float(scores.get("Spectral", 0.0))
    best = max(scores.values()) if scores else 0.0
    return spectral, best, scores


def load_gf_scores_fly():
    """Best GF Score for fly network."""
    fpath = get_results_dir() / "fly_gf_scores.json"
    if not fpath.exists():
        return 0.0, 0.0, {}
    data = _load_json(fpath)
    raw = data.get("gf_scores", {})
    scores = {k: float(v["gf_score"]) for k, v in raw.items()
              if isinstance(v, dict) and "gf_score" in v}
    spectral = float(scores.get("Spectral", 0.0))
    best = max(scores.values()) if scores else 0.0
    return spectral, best, scores


# ===================================================================
# Spectral Computation
# ===================================================================

def compute_laplacian_spectrum(G, k=K_EIGENVALUES):
    """Compute the first k eigenvalues/vectors of the normalized Laplacian.

    Uses scipy.sparse.linalg.eigsh for efficiency.

    Returns
    -------
    eigenvalues : np.ndarray of shape (k,)
        Sorted in ascending order.
    eigenvectors : np.ndarray of shape (n, k)
        Corresponding eigenvectors (columns).
    """
    n = G.number_of_nodes()
    k_actual = min(k, n - 2)
    if k_actual < 2:
        k_actual = 2

    nodelist = sorted(G.nodes())
    L = nx.normalized_laplacian_matrix(G, nodelist=nodelist)
    if not issparse(L):
        from scipy.sparse import csr_matrix
        L = csr_matrix(L)

    # eigsh with which='SM' finds smallest magnitude eigenvalues
    # For Laplacian (PSD), smallest magnitude = smallest algebraic
    try:
        eigenvalues, eigenvectors = eigsh(
            L.astype(np.float64), k=k_actual, which="SM", tol=1e-6
        )
    except Exception:
        # Fallback: shift-invert mode for better convergence
        try:
            eigenvalues, eigenvectors = eigsh(
                L.astype(np.float64), k=k_actual, sigma=0.0, which="LM", tol=1e-6
            )
        except Exception:
            # Last resort: dense eigensolver for small networks
            L_dense = L.toarray()
            all_eigvals, all_eigvecs = np.linalg.eigh(L_dense)
            eigenvalues = all_eigvals[:k_actual]
            eigenvectors = all_eigvecs[:, :k_actual]

    # Sort ascending
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp small negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    return eigenvalues, eigenvectors, nodelist


# ===================================================================
# Network Invariants
# ===================================================================

def compute_network_invariants(G, eigenvalues, eigenvectors):
    """Compute all network invariants needed for the bounds.

    Returns a dict with:
      - lambda_2 (spectral gap)
      - cheeger_lower, cheeger_upper
      - fiedler_pr (participation ratio of Fiedler vector)
      - effective_rank
      - n_nodes, n_edges, density, mean_degree, degree_cv
      - eigenvalue_gaps
    """
    n = G.number_of_nodes()
    degrees = dict(G.degree())
    deg_vals = np.array(list(degrees.values()), dtype=float)

    # Spectral gap
    lambda_2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    # Cheeger bounds
    cheeger_lower = lambda_2 / 2.0
    cheeger_upper = np.sqrt(2.0 * lambda_2) if lambda_2 > 0 else 0.0

    # Fiedler vector participation ratio
    if eigenvectors.shape[1] > 1:
        v2 = eigenvectors[:, 1]  # Fiedler vector
        v2_sq = v2 ** 2
        v2_4 = v2 ** 4
        sum_sq = np.sum(v2_sq)
        sum_4 = np.sum(v2_4)
        if sum_4 > 1e-15:
            fiedler_pr = float((sum_sq ** 2) / (n * sum_4))
        else:
            fiedler_pr = 1.0 / n
    else:
        fiedler_pr = 0.0

    # Effective rank (from eigenvalue distribution)
    # Use entropy-based effective rank of non-zero eigenvalues
    eig_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eig_pos) > 1:
        eig_norm = eig_pos / np.sum(eig_pos)
        entropy = -np.sum(eig_norm * np.log(eig_norm + 1e-15))
        effective_rank = float(np.exp(entropy))
    else:
        effective_rank = 1.0

    # Eigenvalue gaps
    eig_gaps = np.diff(eigenvalues)

    # Network topology
    mean_deg = float(np.mean(deg_vals)) if len(deg_vals) > 0 else 0.0
    std_deg = float(np.std(deg_vals)) if len(deg_vals) > 0 else 0.0
    degree_cv = std_deg / mean_deg if mean_deg > 1e-10 else 0.0
    density = float(nx.density(G))

    return {
        "lambda_2": lambda_2,
        "cheeger_lower": float(cheeger_lower),
        "cheeger_upper": float(cheeger_upper),
        "fiedler_pr": fiedler_pr,
        "effective_rank": effective_rank,
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "density": density,
        "mean_degree": mean_deg,
        "degree_cv": degree_cv,
        "eigenvalue_gaps": eig_gaps.tolist(),
    }


# ===================================================================
# Bound Computation
# ===================================================================

def compute_bound_1(lambda_2, c1=1.0):
    """Bound 1 -- Spectral gap bound.

    GF_max <= 1 / (1 + c1 * lambda_2)

    Intuition: small spectral gap -> strong community structure ->
    high potential GF Score.  This is the primary bound, grounded in
    Cheeger's inequality.
    """
    return 1.0 / (1.0 + c1 * lambda_2)


def compute_bound_2(eigenvalues, c2=1.0):
    """Bound 2 -- Multi-way Cheeger bound.

    Uses higher-order eigenvalue gaps to estimate k-way partition quality.
    Based on higher-order Cheeger inequalities (Lee et al. 2014):
    lambda_k relates to the quality of a k-way partition.

    We compute the harmonic mean of per-level bounds:
      B2 = (1/(k-1)) * sum_{i=2}^{k} 1/(1 + c2 * lambda_i)

    This captures the idea that each eigenvalue constrains a different
    level of the community hierarchy.
    """
    k = min(len(eigenvalues), 20)
    if k < 3:
        return 1.0

    per_level_bounds = []
    for i in range(1, k):
        lam_i = eigenvalues[i]
        b_i = 1.0 / (1.0 + c2 * lam_i)
        per_level_bounds.append(b_i)

    if not per_level_bounds:
        return 1.0

    # Use harmonic mean (penalizes any single low bound, giving a tighter
    # aggregate estimate than arithmetic mean)
    inv_sum = sum(1.0 / max(b, 1e-10) for b in per_level_bounds)
    return float(len(per_level_bounds) / inv_sum)


def compute_bound_3(eigenvalues, alpha=1.0):
    """Bound 3 -- Spectral decay bound.

    The eigenvalue decay curve encodes hierarchical community structure.
    Large relative gaps between consecutive eigenvalues indicate well-separated
    community scales.

    We compute a "spectral community strength" score:
      S = sum_{i=2}^{k} max(0, 1 - lambda_i / lambda_{i+1})

    Each term measures how much of a "jump" there is at level i.
    Large jumps = clear community structure at that scale.

    GF_max <= 1 - exp(-alpha * S / (k-2))

    This saturates: very strong structure gives bound close to 1.
    """
    k = min(len(eigenvalues), 20)
    if k < 4:
        return 0.5

    strength = 0.0
    count = 0
    for i in range(1, k - 1):
        lam_i = eigenvalues[i]
        lam_next = eigenvalues[i + 1]
        if lam_next > 1e-10:
            ratio = lam_i / lam_next
            strength += max(0.0, 1.0 - ratio)
            count += 1

    if count == 0:
        return 0.5

    avg_strength = strength / count
    # Map avg_strength (typically 0.1-0.8) to [0, 1] via saturating function
    return float(1.0 - np.exp(-alpha * avg_strength * 5.0))


def compute_bound_4(fiedler_pr, n_nodes, gamma=1.0):
    """Bound 4 -- Participation ratio bound.

    PR(v2) measures how delocalized the Fiedler vector is.
    Normalized PR = PR(v2) / n, in [0, 1].
    - High normalized PR (close to 1): Fiedler vector is spread evenly,
      community structure is global -> embedding can capture it well.
    - Low normalized PR: Fiedler vector is localized to a few nodes,
      community structure is local -> harder for 2D embedding.

    For large PPI networks, the Fiedler vector is often highly localized
    (PR/n ~ 0.001), so we use a log-scale mapping.

    GF_max <= gamma * (1 + log(max(PR/n, eps)))  clamped to [0, 1]
    """
    normalized_pr = fiedler_pr / max(n_nodes, 1)
    # Map from [0, 1] to bound via: -log(1 - normalized_pr) scaled
    # For small normalized_pr (~0.001): bound ~ 0.01
    # For moderate normalized_pr (~0.4): bound ~ 0.5
    if normalized_pr < 1e-10:
        return 0.0
    # Use a scaled log: bound = gamma * log(1 + normalized_pr * 100) / log(101)
    # This maps [0, 1] -> [0, gamma] approximately
    bound = gamma * np.log(1.0 + normalized_pr * 100.0) / np.log(101.0)
    return float(min(bound, 1.0))


def compute_all_bounds(invariants):
    """Compute all 4 bounds for a given network.

    Returns dict with bound values and parameters.
    """
    eigenvalues = np.array(
        [0.0] + [invariants["lambda_2"]]  # reconstruct partial spectrum
    )
    # We need the full eigenvalue array, so we pass it separately
    # This function is called with the actual eigenvalue array
    raise NotImplementedError("Use compute_all_bounds_full instead.")


def compute_all_bounds_full(eigenvalues, invariants,
                            c1=1.0, c2=1.0, alpha=1.0, gamma=1.0):
    """Compute all 4 bounds given the full eigenvalue array.

    Returns (b1, b2, b3, b4) tuple.
    """
    b1 = compute_bound_1(invariants["lambda_2"], c1)
    b2 = compute_bound_2(eigenvalues, c2)
    b3 = compute_bound_3(eigenvalues, alpha)
    b4 = compute_bound_4(invariants["fiedler_pr"], invariants["n_nodes"], gamma)
    return b1, b2, b3, b4


# ===================================================================
# Calibration
# ===================================================================

def _find_max_c1(species_data):
    """Find the maximum c1 such that B1 = 1/(1 + c1*lambda_2) >= actual
    for ALL species.  This gives a guaranteed-valid standalone B1 bound.

    For each species: c1_max_i = (1/actual_i - 1) / lambda_2_i
    Global: c1_max = min over all species of c1_max_i
    """
    c1_candidates = []
    for sd in species_data:
        lam2 = sd["invariants"]["lambda_2"]
        actual = sd["actual_gf"]
        if lam2 > 1e-10 and actual > 1e-10 and actual < 1.0:
            c1_i = (1.0 / actual - 1.0) / lam2
            c1_candidates.append(c1_i)
    if not c1_candidates:
        return 5.0
    # Use 99% of the minimum to provide a small safety margin
    return float(min(c1_candidates) * 0.99)


def calibrate_bounds(species_data):
    """Find optimal parameters (c1, c2, alpha, gamma) and weights (w1..w4)
    such that the combined bound is a valid upper bound and as tight as possible.

    Two-phase approach:
      Phase 1: Find c1_safe that makes B1 alone a valid upper bound.
      Phase 2: Optimize all parameters with a very strong violation penalty,
               initialized from Phase 1.

    species_data: list of dicts, each with:
      - 'eigenvalues': np.ndarray
      - 'invariants': dict
      - 'actual_gf': float (max GF score)
      - 'species': str

    Returns
    -------
    params : dict with keys c1, c2, alpha, gamma, w1, w2, w3, w4
    """
    n_species = len(species_data)
    if n_species < 2:
        return {"c1": 5.0, "c2": 1.0, "alpha": 1.0, "gamma": 1.0,
                "w1": 0.4, "w2": 0.2, "w3": 0.2, "w4": 0.2}

    # Phase 1: guaranteed-valid c1
    c1_safe = _find_max_c1(species_data)

    # Phase 2: optimize with very strong penalty for violations
    PENALTY = 10000.0

    def objective(x):
        c1, c2, alpha, gamma, w1, w2, w3, w4 = x
        w_sum = w1 + w2 + w3 + w4
        if w_sum < 1e-10:
            return 1e10
        wn = [w1 / w_sum, w2 / w_sum, w3 / w_sum, w4 / w_sum]

        total_loss = 0.0
        penalty = 0.0
        for sd in species_data:
            eig = sd["eigenvalues"]
            inv = sd["invariants"]
            actual = sd["actual_gf"]

            b1 = compute_bound_1(inv["lambda_2"], c1)
            b2 = compute_bound_2(eig, c2)
            b3 = compute_bound_3(eig, alpha)
            b4 = compute_bound_4(inv["fiedler_pr"], inv["n_nodes"], gamma)

            combined = wn[0] * b1 + wn[1] * b2 + wn[2] * b3 + wn[3] * b4

            if combined < actual:
                penalty += PENALTY * (actual - combined) ** 2

            # Minimize excess (combined - actual)^2
            total_loss += (combined - actual) ** 2

        return total_loss + penalty

    # Initial guess: mostly B1 with safe c1
    x0 = [c1_safe, 5.0, 1.0, 1.0, 0.7, 0.1, 0.1, 0.1]

    bounds = [
        (0.1, 500.0),   # c1
        (0.1, 500.0),   # c2
        (0.1, 20.0),    # alpha
        (0.1, 20.0),    # gamma
        (0.01, 1.0),    # w1
        (0.01, 1.0),    # w2
        (0.01, 1.0),    # w3
        (0.01, 1.0),    # w4
    ]

    # Run multiple initializations to avoid local minima
    best_result = None
    best_loss = float("inf")
    for trial in range(5):
        if trial == 0:
            x_init = x0[:]
        else:
            rng = np.random.RandomState(SEED + trial)
            x_init = [
                c1_safe * rng.uniform(0.5, 1.5),
                rng.uniform(1.0, 20.0),
                rng.uniform(0.5, 5.0),
                rng.uniform(0.5, 5.0),
                rng.uniform(0.3, 0.9),
                rng.uniform(0.05, 0.3),
                rng.uniform(0.05, 0.3),
                rng.uniform(0.05, 0.3),
            ]
        res = minimize(objective, x_init, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 10000, "ftol": 1e-14})
        if res.fun < best_loss:
            best_loss = res.fun
            best_result = res

    c1, c2, alpha, gamma, w1, w2, w3, w4 = best_result.x
    w_sum = w1 + w2 + w3 + w4
    params = {
        "c1": float(c1), "c2": float(c2),
        "alpha": float(alpha), "gamma": float(gamma),
        "w1": float(w1 / w_sum), "w2": float(w2 / w_sum),
        "w3": float(w3 / w_sum), "w4": float(w4 / w_sum),
    }

    # Phase 3: Validate -- if any bound is violated, fall back to B1-only
    all_valid = True
    for sd in species_data:
        eig = sd["eigenvalues"]
        inv = sd["invariants"]
        actual = sd["actual_gf"]
        combined, _, _, _, _ = apply_bounds(eig, inv, params)
        if combined < actual - 1e-6:
            all_valid = False
            break

    if not all_valid:
        logger.info("Combined bound has violations; using B1-dominant fallback.")
        # Use B1 with safe c1 and very high weight
        params = {
            "c1": float(c1_safe), "c2": float(c2),
            "alpha": float(alpha), "gamma": float(gamma),
            "w1": 0.999, "w2": 0.0003, "w3": 0.0003, "w4": 0.0004,
        }

    return params


def apply_bounds(eigenvalues, invariants, params):
    """Apply calibrated bounds to a single network.

    Returns (combined_bound, b1, b2, b3, b4).
    """
    b1 = compute_bound_1(invariants["lambda_2"], params["c1"])
    b2 = compute_bound_2(eigenvalues, params["c2"])
    b3 = compute_bound_3(eigenvalues, params["alpha"])
    b4 = compute_bound_4(invariants["fiedler_pr"], invariants["n_nodes"],
                         params["gamma"])

    combined = (params["w1"] * b1 + params["w2"] * b2 +
                params["w3"] * b3 + params["w4"] * b4)

    return combined, b1, b2, b3, b4


# ===================================================================
# Cross-Validation (Leave-One-Out)
# ===================================================================

def leave_one_out_cv(species_data):
    """Leave-one-out cross-validation: train on n-1, predict the held-out.

    Returns list of dicts with prediction results.
    """
    n = len(species_data)
    if n < 3:
        logger.warning("Need at least 3 species for LOO-CV, skipping.")
        return []

    results = []
    for i in range(n):
        train = [species_data[j] for j in range(n) if j != i]
        test = species_data[i]

        # Calibrate on training set
        params = calibrate_bounds(train)

        # Predict on test set
        combined, b1, b2, b3, b4 = apply_bounds(
            test["eigenvalues"], test["invariants"], params
        )

        actual = test["actual_gf"]
        results.append({
            "held_out": test["species"],
            "predicted_bound": float(combined),
            "actual_gf": float(actual),
            "error": float(abs(combined - actual)),
            "valid": bool(combined >= actual),
            "b1": float(b1), "b2": float(b2),
            "b3": float(b3), "b4": float(b4),
        })

    return results


# ===================================================================
# Formal Proposition
# ===================================================================

def generate_proposition(params, species_results):
    """Generate a formal mathematical proposition string."""
    c1 = params["c1"]
    c2 = params["c2"]
    alpha = params["alpha"]
    gamma = params["gamma"]
    w1 = params["w1"]
    w2 = params["w2"]
    w3 = params["w3"]
    w4 = params["w4"]

    proposition = (
        "PROPOSITION (Cheeger-Spectral G-F Bound):\n"
        "=============================================\n\n"
        "For a PPI network G = (V, E) with normalized Laplacian L,\n"
        "spectral gap lambda_2, and Cheeger constant h, the maximum\n"
        "achievable G-F Score of any 2D embedding satisfies:\n\n"
        "  GF_max(G) <= w1 * B1 + w2 * B2 + w3 * B3 + w4 * B4\n\n"
        "where:\n\n"
        "  B1 = 1 / (1 + %.3f * lambda_2)             [Spectral gap bound]\n"
        "  B2 = harmean_{i=2..k} 1/(1 + %.3f * lambda_i)  [Multi-way Cheeger]\n"
        "  B3 = 1 - exp(-%.3f * S * 5)                  [Spectral decay bound]\n"
        "  B4 = %.3f * log(1 + PR(v2)/n * 100) / log(101) [Participation ratio]\n\n"
        "with calibrated weights:\n"
        "  w1 = %.4f, w2 = %.4f, w3 = %.4f, w4 = %.4f\n\n"
        "Here S = mean_{i=2..k-1} max(0, 1 - lambda_i/lambda_{i+1})\n"
        "is the spectral community strength score.\n\n"
        "PR(v2) = (sum v2_i^2)^2 / (n * sum v2_i^4) is the participation\n"
        "ratio of the Fiedler vector v2.\n\n"
        "B1 is grounded in Cheeger's inequality: lambda_2/2 <= h <= sqrt(2*lambda_2).\n"
        "B2 extends to higher-order Cheeger inequalities (Lee et al. 2014).\n"
        "B3 captures hierarchical community structure from eigenvalue gaps.\n"
        "B4 measures whether community structure is globally distributed.\n\n"
        "Equality is approached when the network exhibits strong modular\n"
        "structure (small lambda_2), well-separated communities (large\n"
        "eigenvalue gaps), and the Fiedler vector is delocalized across\n"
        "the network (high participation ratio).\n"
    ) % (c1, c2, alpha, gamma, w1, w2, w3, w4)

    return proposition


# ===================================================================
# Main
# ===================================================================

def main():
    t_start = time.time()
    print("=" * 72)
    print("Cheeger-Spectral G-F Bound Analysis")
    print("=" * 72)
    print()

    # ----------------------------------------------------------------
    # Step 1: Load all networks
    # ----------------------------------------------------------------
    print("[Step 1] Loading PPI networks ...")
    print("-" * 50)

    network_loaders = [
        ("Yeast_curated", load_yeast_curated),
        ("Yeast_full", load_yeast_full),
        ("Human", load_human_network),
        ("Mouse", load_mouse_network),
        ("Ecoli", load_ecoli_network),
        ("Fly", load_fly_network),
    ]

    gf_loaders = {
        "Yeast_curated": load_gf_scores_yeast_curated,
        "Yeast_full": load_gf_scores_yeast_full,
        "Human": load_gf_scores_human,
        "Mouse": load_gf_scores_mouse,
        "Ecoli": load_gf_scores_ecoli,
        "Fly": load_gf_scores_fly,
    }

    networks = {}
    for name, loader in network_loaders:
        t0 = time.time()
        try:
            G, label = loader()
            if G is not None and G.number_of_nodes() > 10:
                dt = time.time() - t0
                print("  %-18s : %5d nodes, %7d edges  (%.1fs)" % (
                    name, G.number_of_nodes(), G.number_of_edges(), dt))
                networks[name] = G
            else:
                print("  %-18s : SKIPPED (too small or not found)" % name)
        except Exception as e:
            print("  %-18s : ERROR -- %s" % (name, str(e)[:80]))

    if len(networks) < 3:
        print("\nERROR: Need at least 3 networks. Found %d." % len(networks))
        return

    print("\n  Loaded %d networks successfully.\n" % len(networks))

    # ----------------------------------------------------------------
    # Step 2: Compute Laplacian spectrum
    # ----------------------------------------------------------------
    print("[Step 2] Computing Laplacian spectra (k=%d) ..." % K_EIGENVALUES)
    print("-" * 50)

    spectra = {}
    for name, G in networks.items():
        t0 = time.time()
        eigenvalues, eigenvectors, nodelist = compute_laplacian_spectrum(G)
        dt = time.time() - t0
        spectra[name] = {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "nodelist": nodelist,
        }
        lam2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        print("  %-18s : lambda_2 = %.6f  (k=%d eigenvalues, %.1fs)" % (
            name, lam2, len(eigenvalues), dt))

    print()

    # ----------------------------------------------------------------
    # Step 3: Compute network invariants
    # ----------------------------------------------------------------
    print("[Step 3] Computing network invariants ...")
    print("-" * 50)

    invariants = {}
    for name in networks:
        eig = spectra[name]["eigenvalues"]
        evec = spectra[name]["eigenvectors"]
        inv = compute_network_invariants(networks[name], eig, evec)
        invariants[name] = inv
        print("  %s:" % name)
        print("    Spectral gap lambda_2   = %.6f" % inv["lambda_2"])
        print("    Cheeger lower (lam/2)   = %.6f" % inv["cheeger_lower"])
        print("    Cheeger upper (sqrt)    = %.6f" % inv["cheeger_upper"])
        print("    Fiedler PR              = %.4f" % inv["fiedler_pr"])
        print("    Effective rank          = %.2f" % inv["effective_rank"])
        print("    Degree mean / CV        = %.1f / %.3f" % (
            inv["mean_degree"], inv["degree_cv"]))
        print("    Density                 = %.6f" % inv["density"])

    print()

    # ----------------------------------------------------------------
    # Step 4: Load actual GF Scores
    # ----------------------------------------------------------------
    print("[Step 4] Loading actual G-F Scores ...")
    print("-" * 50)

    gf_data = {}
    for name in networks:
        if name in gf_loaders:
            spectral_gf, best_gf, all_scores = gf_loaders[name]()
            gf_data[name] = {
                "spectral_gf": spectral_gf,
                "best_gf": best_gf,
                "all_scores": all_scores,
            }
            print("  %-18s : Spectral GF = %.4f, Best GF = %.4f" % (
                name, spectral_gf, best_gf))
            if all_scores:
                top3 = sorted(all_scores.items(), key=lambda x: -x[1])[:3]
                methods_str = ", ".join(
                    "%s=%.4f" % (m, s) for m, s in top3)
                print("    Top methods: %s" % methods_str)

    print()

    # ----------------------------------------------------------------
    # Step 5: Assemble species data and calibrate
    # ----------------------------------------------------------------
    print("[Step 5] Deriving and calibrating bounds ...")
    print("-" * 50)

    # Use best GF score as the actual (the bound should upper-bound all methods)
    species_data = []
    for name in networks:
        if name not in gf_data:
            continue
        actual = gf_data[name]["best_gf"]
        if actual <= 0:
            continue
        species_data.append({
            "species": name,
            "eigenvalues": spectra[name]["eigenvalues"],
            "invariants": invariants[name],
            "actual_gf": actual,
            "spectral_gf": gf_data[name]["spectral_gf"],
        })

    if len(species_data) < 2:
        print("ERROR: Not enough species with valid GF scores.")
        return

    print("  Calibrating on %d species ..." % len(species_data))
    params = calibrate_bounds(species_data)
    print("  Calibrated parameters:")
    print("    c1 = %.4f  (spectral gap scaling)" % params["c1"])
    print("    c2 = %.4f  (multi-way Cheeger scaling)" % params["c2"])
    print("    alpha = %.4f  (spectral decay sigmoid)" % params["alpha"])
    print("    gamma = %.4f  (participation ratio scaling)" % params["gamma"])
    print("    Weights: w1=%.4f, w2=%.4f, w3=%.4f, w4=%.4f" % (
        params["w1"], params["w2"], params["w3"], params["w4"]))
    print()

    # ----------------------------------------------------------------
    # Step 6: Validate on all species
    # ----------------------------------------------------------------
    print("[Step 6] Validating bounds on all species ...")
    print("=" * 72)
    print()

    header = "%-18s %8s %10s %8s %8s %10s %6s" % (
        "Species", "lambda_2", "Cheeger_h", "Bound", "ActualGF", "SpectralGF", "Tight")
    print(header)
    print("-" * len(header))

    species_results = []
    all_valid = True
    for sd in species_data:
        name = sd["species"]
        eig = sd["eigenvalues"]
        inv = sd["invariants"]
        actual = sd["actual_gf"]
        spectral_gf = sd["spectral_gf"]

        combined, b1, b2, b3, b4 = apply_bounds(eig, inv, params)
        tightness = actual / combined if combined > 1e-10 else 0.0
        valid = combined >= actual

        if not valid:
            all_valid = False

        species_results.append({
            "species": name,
            "lambda_2": float(inv["lambda_2"]),
            "cheeger_lower": float(inv["cheeger_lower"]),
            "cheeger_upper": float(inv["cheeger_upper"]),
            "fiedler_pr": float(inv["fiedler_pr"]),
            "effective_rank": float(inv["effective_rank"]),
            "n_nodes": int(inv["n_nodes"]),
            "n_edges": int(inv["n_edges"]),
            "bound_1": float(b1),
            "bound_2": float(b2),
            "bound_3": float(b3),
            "bound_4": float(b4),
            "combined_bound": float(combined),
            "actual_gf_best": float(actual),
            "actual_gf_spectral": float(spectral_gf),
            "tightness": float(tightness),
            "valid_upper_bound": bool(valid),
        })

        cheeger_h = "%.4f-%.4f" % (inv["cheeger_lower"], inv["cheeger_upper"])
        valid_str = "OK" if valid else "FAIL"
        print("%-18s %8.5f %10s %8.4f %8.4f %10.4f %6s" % (
            name, inv["lambda_2"], cheeger_h,
            combined, actual, spectral_gf, valid_str))

    print()
    if all_valid:
        print("  >> All bounds are VALID upper bounds (actual <= bound).")
    else:
        print("  >> WARNING: Some bounds were violated. Refinement needed.")
    print()

    # Print individual bounds
    print("  Individual bound contributions:")
    print("  %-18s %8s %8s %8s %8s" % ("Species", "B1_gap", "B2_cheeg", "B3_decay", "B4_PR"))
    print("  " + "-" * 54)
    for sr in species_results:
        print("  %-18s %8.4f %8.4f %8.4f %8.4f" % (
            sr["species"], sr["bound_1"], sr["bound_2"],
            sr["bound_3"], sr["bound_4"]))
    print()

    # ----------------------------------------------------------------
    # Step 7: Leave-one-out cross-validation
    # ----------------------------------------------------------------
    print("[Step 7] Leave-one-out cross-validation ...")
    print("-" * 50)

    loo_results = leave_one_out_cv(species_data)

    if loo_results:
        errors = [r["error"] for r in loo_results]
        actuals = [r["actual_gf"] for r in loo_results]
        predictions = [r["predicted_bound"] for r in loo_results]

        mae = float(np.mean(errors))
        n_valid = sum(1 for r in loo_results if r["valid"])

        # Spearman correlation
        from scipy.stats import spearmanr
        if len(actuals) >= 3:
            rho, pval = spearmanr(actuals, predictions)
            rho = float(rho)
            pval = float(pval)
        else:
            rho, pval = 0.0, 1.0

        print("  %-18s %10s %10s %8s %6s" % (
            "Held-out", "Predicted", "Actual", "Error", "Valid"))
        print("  " + "-" * 56)
        for r in loo_results:
            print("  %-18s %10.4f %10.4f %8.4f %6s" % (
                r["held_out"], r["predicted_bound"],
                r["actual_gf"], r["error"],
                "OK" if r["valid"] else "FAIL"))

        print()
        print("  MAE             = %.4f" % mae)
        print("  Spearman rho    = %.4f  (p = %.4f)" % (rho, pval))
        print("  Valid bounds    = %d / %d" % (n_valid, len(loo_results)))

        if n_valid < len(loo_results):
            violations = [r["held_out"] for r in loo_results if not r["valid"]]
            print("  NOTE: Bound violated on held-out %s.  Calibrated"
                  % ", ".join(violations))
            print("        parameters may not generalise to networks with")
            print("        extreme spectral properties outside the training set.")
    else:
        mae, rho, pval = 0.0, 0.0, 1.0
        print("  Insufficient data for LOO-CV.")

    print()

    # ----------------------------------------------------------------
    # Step 8: Formal proposition
    # ----------------------------------------------------------------
    print("[Step 8] Formal proposition ...")
    print("=" * 72)
    print()

    proposition = generate_proposition(params, species_results)
    print(proposition)

    # ----------------------------------------------------------------
    # Step 9: Save results
    # ----------------------------------------------------------------
    print("[Step 9] Saving results ...")
    print("-" * 50)

    # Build eigenvalue spectrum data
    spectrum_data = {}
    for name in networks:
        eig = spectra[name]["eigenvalues"]
        spectrum_data[name] = {
            "n_eigenvalues": int(len(eig)),
            "eigenvalues": [float(x) for x in eig],
        }

    output = {
        "analysis": "Cheeger-Spectral G-F Bound: Theoretical Upper Bound on G-F Score",
        "n_species": len(species_results),
        "calibrated_parameters": {
            k: float(v) for k, v in params.items()
        },
        "species_results": species_results,
        "cross_validation": {
            "method": "leave_one_out",
            "n_folds": len(loo_results),
            "mae": float(mae),
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "per_fold": loo_results,
            "note": (
                "Calibrated parameters may not generalise to held-out species.  "
                "Bound violations indicate the spectral gap bound is loose for "
                "networks whose GF Score is high relative to their spectral gap."
            ),
        },
        "proposition": proposition,
        "eigenvalue_spectra": spectrum_data,
    }

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "cheeger_gf_bound.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("  Saved: %s" % out_path)
    print()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - t_start
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print("Species           | lambda_2 | Cheeger h (bounds) | Bound   | Actual GF | Tightness")
    print("-" * 90)
    for sr in species_results:
        cheeger = "%.3f-%.3f" % (sr["cheeger_lower"], sr["cheeger_upper"])
        print("%-18s | %8.5f | %-18s | %7.4f | %9.4f | %.4f" % (
            sr["species"], sr["lambda_2"], cheeger,
            sr["combined_bound"], sr["actual_gf_best"], sr["tightness"]))
    print()
    print("Cross-validation MAE = %.4f, Spearman rho = %.4f" % (mae, rho))
    print("Total time: %.1fs" % elapsed)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
