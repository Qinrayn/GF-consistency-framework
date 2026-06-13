#!/usr/bin/env python3
"""
deep_geometric_analysis.py — Phase 1: Multi-Scale Geometric Fingerprint
========================================================================

Goes beyond the scalar G-F Score to understand *why* and *at what scales*
each embedding method captures (or fails to capture) functional organisation.

Three complementary analyses:
  1. Distance-Function Correspondence (DFC): at each distance scale, how well
     does geometric proximity predict functional co-annotation?
  2. Geometric Feature Extraction: spectral gap, distance CV, spatial uniformity,
     local density variance, effective dimensionality.
  3. G-F Curve Shape Decomposition: peak location, peak height, decay rate,
     plateau width — the curve as a multi-scale fingerprint.

Outputs:
  - results/deep_geometric_analysis.json  (all numerical results)
  - figures/Fig26_dfc_curves.png          (distance-function correspondence)
  - figures/Fig27_geometric_radar.png     (method fingerprint radar)
  - figures/Fig28_curve_decomposition.png (G-F curve shape features)
"""

import sys
import json
import math
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import trapezoid
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ============================================================
# Project path setup
# ============================================================
PROJECT_ROOT = Path(r"C:\Users\云丘\GF-consistency-framework")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

sys.path.insert(0, str(SCRIPTS_DIR))
from utils import (
    SEED, GF_R_MIN, GF_R_MAX, N_POINTS, R_MIN, R_MAX,
    ALL_METHODS, CLASSICAL_METHODS, GNN_METHODS, ALL_CURATED_METHODS,
    rescale_coordinates, load_curated_network,
)

# ============================================================
# Configuration
# ============================================================
METHOD_CATEGORIES = {
    "Classical":     ["DM", "MDS", "Spectral"],
    "Linear":        ["PCA"],
    "Random Walk":   ["DeepWalk", "Node2Vec"],
    "Deep Generative": ["VGAE", "VGAE-feat"],
    "GNN":           ["GraphSAGE", "GAT", "GIN"],
}
METHOD_COLORS = {
    "Spectral":    "#E69F00",  # Okabe-Ito orange
    "DM":          "#0072B2",  # blue
    "MDS":         "#009E73",  # green
    "Node2Vec":    "#CC79A7",  # pink
    "PCA":         "#56B4E9",  # sky blue
    "VGAE-feat":   "#F0E442",  # yellow
    "DeepWalk":    "#D55E00",  # vermillion
    "GIN":         "#949494",  # grey
    "GAT":         "#000000",  # black
    "GraphSAGE":   "#8B4513",  # brown
    "VGAE":        "#808080",  # dark grey
}
N_DFC_BINS = 30  # number of distance bins for DFC curves


# ============================================================
# 1. Data loading
# ============================================================

def load_embedding(method: str):
    """Load embedding coordinates and node list for a method."""
    if method in ALL_CURATED_METHODS:
        suffix = "153"
    else:
        suffix = "153"  # GNN methods also on curated 153
    npy_path = EMBEDDINGS_DIR / f"{method}_{suffix}.npy"
    nodes_path = EMBEDDINGS_DIR / f"{method}_{suffix}_nodes.json"
    if not npy_path.exists():
        return None, None
    coords = np.load(npy_path)
    with open(nodes_path, encoding="utf-8") as f:
        nodes = json.load(f)
    return coords, nodes


def load_go_map():
    """Load GO annotations and return gene -> [GO terms] dict."""
    with open(DATA_DIR / "gene_go_map.json", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. Core computations
# ============================================================

def compute_go_coannotation_matrix(nodes, go_map):
    """Binary matrix: pair (i,j) shares >= 1 GO term."""
    n = len(nodes)
    go_sets = [set(go_map.get(nd, [])) for nd in nodes]
    mat = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        if not go_sets[i]:
            continue
        for j in range(i + 1, n):
            if go_sets[i] & go_sets[j]:
                mat[i, j] = 1
                mat[j, i] = 1
    return mat


def compute_go_term_ic(nodes, go_map):
    """Corpus-based IC for each GO term: -log2(freq/total_genes)."""
    n_total = len(nodes)
    term_counts = Counter()
    for nd in nodes:
        for term in go_map.get(nd, []):
            term_counts[term] += 1
    ic = {}
    for term, count in term_counts.items():
        ic[term] = -math.log2(max(count, 1) / n_total)
    return ic, term_counts


def compute_distance_function_correspondence(dist_vec, coann_vec, n_bins):
    """
    Bin pairwise distances; compute co-annotation fraction per bin.
    Returns (bin_centers, coann_fractions, bin_counts).
    """
    # Remove zero distances (self-pairs excluded by pdist)
    mask = dist_vec > 1e-10
    d = dist_vec[mask]
    c = coann_vec[mask]
    
    if len(d) < n_bins:
        n_bins = max(5, len(d) // 10)
    
    # Use quantile-based bins for even distribution
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(d, percentiles)
    # Ensure unique edges
    edges = np.unique(edges)
    if len(edges) < 3:
        return None, None, None
    
    bin_centers = []
    coann_fracs = []
    bin_counts = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        if k == len(edges) - 2:
            sel = (d >= lo) & (d <= hi)
        else:
            sel = (d >= lo) & (d < hi)
        if sel.sum() < 5:
            continue
        bin_centers.append((lo + hi) / 2)
        coann_fracs.append(c[sel].mean())
        bin_counts.append(int(sel.sum()))
    
    return np.array(bin_centers), np.array(coann_fracs), np.array(bin_counts)


def compute_geometric_features(coords, dist_matrix):
    """Extract interpretable geometric features from embedding."""
    n = coords.shape[0]
    d = coords.shape[1]
    
    # 1. Distance statistics
    dist_upper = dist_matrix[np.triu_indices(n, k=1)]
    dist_mean = float(np.mean(dist_upper))
    dist_std = float(np.std(dist_upper))
    dist_cv = dist_std / max(dist_mean, 1e-10)
    dist_median = float(np.median(dist_upper))
    
    # 2. Spectral gap of the distance-based Laplacian
    # Build a k-NN graph (k=10) for spectral analysis
    k = min(10, n - 1)
    W = np.zeros((n, n))
    for i in range(n):
        knn_idx = np.argsort(dist_matrix[i])[1:k+1]
        for j in knn_idx:
            W[i, j] = 1.0
            W[j, i] = 1.0
    
    # Graph Laplacian
    D_deg = np.diag(W.sum(axis=1))
    L = D_deg - W
    # Normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-10)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))
    # Spectral gap = λ_2 - λ_1 (should be λ_1 ≈ 0 for connected graph)
    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0
    # Fiedler value = λ_2 (algebraic connectivity)
    fiedler = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    # Spectral spread = λ_n - λ_2
    spectral_spread = float(eigenvalues[-1] - eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    
    # 3. Spatial uniformity (how evenly distributed points are)
    # Coefficient of variation of local density (k-NN distance)
    knn_dists = []
    for i in range(n):
        sorted_d = np.sort(dist_matrix[i])
        knn_dists.append(sorted_d[k])  # k-th nearest neighbor distance
    knn_dists = np.array(knn_dists)
    spatial_uniformity = float(np.std(knn_dists) / max(np.mean(knn_dists), 1e-10))
    
    # 4. Effective dimensionality via PCA eigenvalue decay
    coords_centered = coords - coords.mean(axis=0)
    cov = coords_centered.T @ coords_centered / n
    pca_eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
    pca_eigs = np.maximum(pca_eigs, 0)
    total_var = pca_eigs.sum()
    if total_var > 0:
        eigs_norm = pca_eigs / total_var
        # Participation ratio: (sum λ)^2 / sum(λ^2)
        eff_dim = float((pca_eigs.sum()**2) / max((pca_eigs**2).sum(), 1e-10))
        # Energy in first component
        first_pc_energy = float(eigs_norm[0])
    else:
        eff_dim = 0.0
        first_pc_energy = 0.0
    
    # 5. Collapse diagnostics
    min_dist = float(np.min(dist_upper))
    max_dist = float(np.max(dist_upper))
    dist_range = max_dist - min_dist
    collapse_score = float(1.0 - dist_cv) if dist_cv < 0.1 else 0.0
    
    return {
        "dist_mean": dist_mean,
        "dist_std": dist_std,
        "dist_cv": dist_cv,
        "dist_median": dist_median,
        "dist_range": dist_range,
        "spectral_gap": spectral_gap,
        "fiedler_value": fiedler,
        "spectral_spread": spectral_spread,
        "spatial_uniformity_cv": spatial_uniformity,
        "effective_dimensionality": eff_dim,
        "first_pc_energy": first_pc_energy,
        "collapse_score": collapse_score,
    }


def compute_per_community_purity_decomposition(communities, nodes, go_map, go_term_counts, n_total_genes):
    """
    For each community, compute which GO terms contribute to purity.
    Returns list of dicts with community-level stats.
    """
    results = []
    for comm in communities:
        comm_names = [nodes[idx] for idx in comm]
        go_terms = []
        for nd in comm_names:
            go_terms.extend(go_map.get(nd, []))
        if not go_terms:
            continue
        term_counts = Counter(go_terms)
        total = len(go_terms)
        most_common_term, most_common_count = term_counts.most_common(1)[0]
        purity = most_common_count / total
        # IC of dominant term
        freq = go_term_counts.get(most_common_term, 1)
        ic = -math.log2(max(freq, 1) / n_total_genes)
        results.append({
            "size": len(comm_names),
            "purity": purity,
            "dominant_term": most_common_term,
            "dominant_count": most_common_count,
            "total_terms": total,
            "n_unique_terms": len(term_counts),
            "dominant_ic": ic,
        })
    return results


# ============================================================
# 3. G-F curve shape analysis
# ============================================================

def analyze_gf_curve_shape(r_vals, purity_vals, gf_r_min=GF_R_MIN, gf_r_max=GF_R_MAX):
    """
    Decompose the G-F purity curve into interpretable shape features.
    """
    r = np.array(r_vals)
    p = np.array(purity_vals)
    
    # Restrict to integration interval
    mask = (r >= gf_r_min) & (r <= gf_r_max)
    r_sub = r[mask]
    p_sub = p[mask]
    
    if len(r_sub) < 5:
        return None
    
    # 1. Peak analysis
    peak_idx = np.argmax(p_sub)
    peak_r = float(r_sub[peak_idx])
    peak_purity = float(p_sub[peak_idx])
    
    # 2. Plateau width (80% of peak)
    threshold = 0.80 * peak_purity
    above = p_sub >= threshold
    plateau_width = 0.0
    if above.any():
        contiguous = np.where(above)[0]
        if len(contiguous) > 0:
            # Find widest contiguous block
            splits = np.where(np.diff(contiguous) > 1)[0]
            blocks = np.split(contiguous, splits + 1)
            widest = max(blocks, key=len)
            plateau_width = float(r_sub[widest[-1]] - r_sub[widest[0]])
    
    # 3. Rising slope (from start to peak)
    if peak_idx > 0:
        rising_slope = float((p_sub[peak_idx] - p_sub[0]) / max(r_sub[peak_idx] - r_sub[0], 1e-10))
    else:
        rising_slope = 0.0
    
    # 4. Decay rate (from peak to end)
    if peak_idx < len(p_sub) - 1:
        decay_rate = float((p_sub[peak_idx] - p_sub[-1]) / max(r_sub[-1] - r_sub[peak_idx], 1e-10))
    else:
        decay_rate = 0.0
    
    # 5. Curve entropy (how "spread out" the purity is)
    p_norm = p_sub / max(p_sub.sum(), 1e-10)
    p_norm = p_norm[p_norm > 0]
    curve_entropy = float(-np.sum(p_norm * np.log2(p_norm)))
    
    # 6. Asymmetry (skewness of the purity distribution)
    mean_r = np.average(r_sub, weights=np.maximum(p_sub, 1e-10))
    var_r = np.average((r_sub - mean_r)**2, weights=np.maximum(p_sub, 1e-10))
    std_r = np.sqrt(max(var_r, 1e-10))
    skewness = float(np.average(((r_sub - mean_r) / std_r)**3, weights=np.maximum(p_sub, 1e-10)))
    
    # 7. Multi-scale stability (CV of purity across scales)
    purity_cv = float(np.std(p_sub) / max(np.mean(p_sub), 1e-10))
    
    # 8. Score above baseline (need baseline passed in or computed)
    gf_score = float(trapezoid(p_sub, r_sub) / (gf_r_max - gf_r_min))
    
    return {
        "gf_score": gf_score,
        "peak_r": peak_r,
        "peak_purity": peak_purity,
        "plateau_width": plateau_width,
        "rising_slope": rising_slope,
        "decay_rate": decay_rate,
        "curve_entropy": curve_entropy,
        "asymmetry_skewness": skewness,
        "purity_cv": purity_cv,
        "mean_purity": float(np.mean(p_sub)),
        "std_purity": float(np.std(p_sub)),
    }


# ============================================================
# 4. Visualization
# ============================================================

def plot_dfc_curves(dfc_results, go_coann_fraction):
    """
    Fig 26: Distance-Function Correspondence curves.
    For each method, plot co-annotation fraction vs embedding distance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left panel: absolute curves
    ax = axes[0]
    for method in ALL_METHODS:
        if method not in dfc_results:
            continue
        res = dfc_results[method]
        ax.plot(res["bin_centers"], res["coann_fractions"],
                color=METHOD_COLORS.get(method, "#333333"),
                linewidth=2.0 if method in ["Spectral", "DM", "MDS"] else 1.2,
                linestyle="-" if method in CLASSICAL_METHODS + ["PCA"] else "--",
                label=method, alpha=0.85)
    
    # Random baseline: expected co-annotation fraction
    ax.axhline(y=go_coann_fraction, color="grey", linestyle=":", linewidth=1.5,
               label=f"Random baseline ({go_coann_fraction:.3f})")
    
    ax.set_xlabel("Pairwise embedding distance", fontsize=12)
    ax.set_ylabel("GO co-annotation fraction", fontsize=12)
    ax.set_title("A. Distance-Function Correspondence", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, min(1.0, ax.get_ylim()[1] * 1.1))
    
    # Right panel: normalized DFC (divided by baseline)
    ax2 = axes[1]
    for method in ALL_METHODS:
        if method not in dfc_results:
            continue
        res = dfc_results[method]
        normalized = res["coann_fractions"] / max(go_coann_fraction, 1e-10)
        ax2.plot(res["bin_centers"], normalized,
                 color=METHOD_COLORS.get(method, "#333333"),
                 linewidth=2.0 if method in ["Spectral", "DM", "MDS"] else 1.2,
                 linestyle="-" if method in CLASSICAL_METHODS + ["PCA"] else "--",
                 label=method, alpha=0.85)
    
    ax2.axhline(y=1.0, color="grey", linestyle=":", linewidth=1.5, label="Baseline")
    ax2.set_xlabel("Pairwise embedding distance", fontsize=12)
    ax2.set_ylabel("Co-annotation enrichment (vs random)", fontsize=12)
    ax2.set_title("B. Normalised DFC (enrichment over random)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "Fig26_dfc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig26_dfc_curves.png")


def plot_geometric_radar(geo_features):
    """
    Fig 27: Geometric feature radar chart for all methods.
    """
    feature_names = [
        "Spectral gap",
        "Distance CV",
        "Spatial uniformity",
        "Effective dim",
        "Fiedler value",
        "1st PC energy (inv)",
    ]
    feature_keys = [
        "spectral_gap",
        "dist_cv",
        "spatial_uniformity_cv",
        "effective_dimensionality",
        "fiedler_value",
        "first_pc_energy",
    ]
    # Higher = better for display (invert where needed)
    invert = {
        "dist_cv": True,          # Low CV = more uniform = better
        "first_pc_energy": True,  # Low 1st PC energy = less collapsed = better
        "spatial_uniformity_cv": True,  # Low = more uniform = better
    }
    
    n_features = len(feature_names)
    
    # Collect raw values for normalization
    raw_values = {}
    for method in ALL_METHODS:
        if method not in geo_features:
            continue
        vals = []
        for i, key in enumerate(feature_keys):
            v = geo_features[method].get(key, 0.0)
            vals.append(v)
        raw_values[method] = vals
    
    # Normalize each feature to [0, 1] across methods
    methods_list = list(raw_values.keys())
    n_methods = len(methods_list)
    if n_methods == 0:
        return
    
    all_vals = np.array([raw_values[m] for m in methods_list])  # (n_methods, n_features)
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-10] = 1.0
    
    # Normalize and optionally invert
    normalized = {}
    for method in methods_list:
        vals = []
        for i, key in enumerate(feature_keys):
            v = (raw_values[method][i] - mins[i]) / ranges[i]
            if feature_keys[i] in invert:
                v = 1.0 - v
            vals.append(v)
        normalized[method] = vals
    
    # Plot radar chart
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # close polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Highlight top methods with thicker lines
    top_methods = ["Spectral", "DM", "MDS"]
    other_methods = [m for m in methods_list if m not in top_methods]
    
    for method in other_methods + top_methods:
        vals = normalized[method] + normalized[method][:1]
        lw = 2.5 if method in top_methods else 1.2
        ls = "-" if method in CLASSICAL_METHODS + ["PCA"] else "--"
        ax.plot(angles, vals, linewidth=lw, linestyle=ls,
                label=method, color=METHOD_COLORS.get(method, "#333"),
                alpha=0.85)
        ax.fill(angles, vals, alpha=0.04, color=METHOD_COLORS.get(method, "#333"))
    
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")
    ax.set_title("Geometric Feature Fingerprint (normalised)", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="lower left", bbox_to_anchor=(1.15, 0.0), fontsize=9,
              framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "Fig27_geometric_radar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig27_geometric_radar.png")


def plot_curve_decomposition(gf_curves_data, shape_features):
    """
    Fig 28: G-F curve shape decomposition.
    Multi-panel: purity curves + shape feature comparison.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Panel A: Full purity curves (all methods)
    ax1 = fig.add_subplot(gs[0, 0])
    r = np.array(gf_curves_data["r"])
    for method in ALL_METHODS:
        key = f"{method}_purity"
        if key not in gf_curves_data:
            continue
        p = gf_curves_data[key]
        lw = 2.0 if method in ["Spectral", "DM", "MDS"] else 1.2
        ls = "-" if method in CLASSICAL_METHODS + ["PCA"] else "--"
        ax1.plot(r, p, color=METHOD_COLORS.get(method, "#333"),
                 linewidth=lw, linestyle=ls, label=method, alpha=0.85)
    
    # Random baseline
    if "random_baseline_purity" in gf_curves_data:
        ax1.plot(r, gf_curves_data["random_baseline_purity"],
                 color="grey", linestyle=":", linewidth=1.5, label="Random")
    
    ax1.axvspan(GF_R_MIN, GF_R_MAX, alpha=0.08, color="blue", label="Integration interval")
    ax1.set_xlabel("Radius r")
    ax1.set_ylabel("Functional purity")
    ax1.set_title("A. G-F Purity Curves", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Peak purity vs peak location
    ax2 = fig.add_subplot(gs[0, 1])
    for method in ALL_METHODS:
        if method not in shape_features:
            continue
        sf = shape_features[method]
        ax2.scatter(sf["peak_r"], sf["peak_purity"],
                    color=METHOD_COLORS.get(method, "#333"), s=80, zorder=5,
                    edgecolors="white", linewidth=1.0)
        ax2.annotate(method, (sf["peak_r"], sf["peak_purity"]),
                     fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    ax2.set_xlabel("Peak location (r)")
    ax2.set_ylabel("Peak purity")
    ax2.set_title("B. Peak Location vs Height", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Plateau width vs G-F Score
    ax3 = fig.add_subplot(gs[0, 2])
    for method in ALL_METHODS:
        if method not in shape_features:
            continue
        sf = shape_features[method]
        ax3.scatter(sf["plateau_width"], sf["gf_score"],
                    color=METHOD_COLORS.get(method, "#333"), s=80, zorder=5,
                    edgecolors="white", linewidth=1.0)
        ax3.annotate(method, (sf["plateau_width"], sf["gf_score"]),
                     fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    ax3.set_xlabel("Plateau width (Δr at 80% peak)")
    ax3.set_ylabel("G-F Score")
    ax3.set_title("C. Plateau Width vs Score", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    # Panel D: DFC enrichment at short range vs long range
    ax4 = fig.add_subplot(gs[1, 0])
    for method in ALL_METHODS:
        if method not in shape_features:
            continue
        sf = shape_features[method]
        # Use decay rate vs rising slope as proxy
        ax4.barh(method, sf["rising_slope"], color=METHOD_COLORS.get(method, "#333"),
                 alpha=0.7, height=0.4, label="Rising slope" if method == ALL_METHODS[0] else "")
        ax4.barh(method, -sf["decay_rate"], color=METHOD_COLORS.get(method, "#333"),
                 alpha=0.35, height=0.4, label="Decay rate (neg)" if method == ALL_METHODS[0] else "")
    ax4.set_xlabel("Slope magnitude")
    ax4.set_title("D. Curve Dynamics (Rise vs Decay)", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis="x")
    
    # Panel E: Purity CV vs Asymmetry
    ax5 = fig.add_subplot(gs[1, 1])
    for method in ALL_METHODS:
        if method not in shape_features:
            continue
        sf = shape_features[method]
        ax5.scatter(sf["asymmetry_skewness"], sf["purity_cv"],
                    color=METHOD_COLORS.get(method, "#333"), s=80, zorder=5,
                    edgecolors="white", linewidth=1.0)
        ax5.annotate(method, (sf["asymmetry_skewness"], sf["purity_cv"]),
                     fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    ax5.set_xlabel("Asymmetry (skewness)")
    ax5.set_ylabel("Purity CV (multi-scale stability)")
    ax5.set_title("E. Curve Shape: Stability vs Symmetry", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    
    # Panel F: Summary bar chart of G-F scores with geometric features overlay
    ax6 = fig.add_subplot(gs[1, 2])
    methods_sorted = sorted(
        [(m, shape_features[m]["gf_score"]) for m in ALL_METHODS if m in shape_features],
        key=lambda x: x[1], reverse=True
    )
    names = [m[0] for m in methods_sorted]
    scores = [m[1] for m in methods_sorted]
    colors = [METHOD_COLORS.get(m, "#333") for m in names]
    bars = ax6.bar(names, scores, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax6.set_ylabel("G-F Score")
    ax6.set_title("F. G-F Score Ranking", fontsize=12, fontweight="bold")
    ax6.tick_params(axis="x", rotation=45)
    ax6.grid(True, alpha=0.3, axis="y")
    
    fig.suptitle("G-F Curve Shape Decomposition — Multi-Scale Geometric Fingerprint",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.savefig(FIGURES_DIR / "Fig28_curve_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig28_curve_decomposition.png")


def plot_dfc_heatmap(dfc_results, go_coann_fraction):
    """
    Fig 29: DFC heatmap — methods x distance quantiles, showing enrichment.
    """
    methods_ordered = ["Spectral", "DM", "MDS", "Node2Vec", "PCA",
                       "VGAE-feat", "DeepWalk", "GIN", "GraphSAGE", "GAT", "VGAE"]
    
    # Collect all DFC data with consistent binning
    # Use common quantile edges
    all_centers = []
    for m in methods_ordered:
        if m in dfc_results and dfc_results[m]["bin_centers"] is not None:
            all_centers.extend(dfc_results[m]["bin_centers"].tolist())
    
    if not all_centers:
        return
    
    # Create common bin edges
    global_min = min(dfc_results[m]["bin_centers"].min() for m in methods_ordered if m in dfc_results and dfc_results[m]["bin_centers"] is not None)
    global_max = max(dfc_results[m]["bin_centers"].max() for m in methods_ordered if m in dfc_results and dfc_results[m]["bin_centers"] is not None)
    common_edges = np.linspace(global_min, global_max, 21)
    common_centers_arr = (common_edges[:-1] + common_edges[1:]) / 2
    
    # Interpolate each method's DFC onto common grid
    heatmap = np.full((len(methods_ordered), len(common_centers_arr)), np.nan)
    for i, method in enumerate(methods_ordered):
        if method not in dfc_results or dfc_results[method]["bin_centers"] is None:
            continue
        bc = dfc_results[method]["bin_centers"]
        cf = dfc_results[method]["coann_fractions"] / max(go_coann_fraction, 1e-10)
        # Interpolate
        heatmap[i] = np.interp(common_centers_arr, bc, cf, left=np.nan, right=np.nan)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlBu_r", interpolation="nearest",
                   vmin=0.3, vmax=3.0)
    
    ax.set_yticks(range(len(methods_ordered)))
    ax.set_yticklabels(methods_ordered, fontsize=10)
    ax.set_xlabel("Embedding distance (quantile bins)", fontsize=12)
    ax.set_title("Distance-Function Correspondence Enrichment Heatmap\n(normalised by random baseline)",
                 fontsize=14, fontweight="bold")
    
    # Add x-tick labels for distance
    n_ticks = 5
    tick_idx = np.linspace(0, len(common_centers_arr) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{common_centers_arr[i]:.3f}" for i in tick_idx], fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Enrichment factor (vs random)", fontsize=10)
    cbar.ax.axhline(y=1.0, color="black", linewidth=1, linestyle="--")
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "Fig29_dfc_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig29_dfc_heatmap.png")


# ============================================================
# 5. Main orchestration
# ============================================================

def main():
    print("=" * 70)
    print("Phase 1: Multi-Scale Geometric Fingerprint Analysis")
    print("=" * 70)
    
    # -- Load data --
    print("\n[1/6] Loading network, GO annotations, and embeddings...")
    G, nodes, go_map = load_curated_network()
    print(f"  Network: {len(nodes)} nodes, {G.number_of_edges()} edges")
    
    n_annotated = sum(1 for nd in nodes if nd in go_map)
    print(f"  GO-annotated nodes: {n_annotated}/{len(nodes)}")
    
    # Load all embeddings
    embeddings = {}
    for method in ALL_METHODS:
        coords, emb_nodes = load_embedding(method)
        if coords is not None:
            embeddings[method] = (coords, emb_nodes)
            print(f"  {method}: {coords.shape[0]} nodes, {coords.shape[1]}D")
        else:
            print(f"  {method}: NOT FOUND")
    
    # -- Compute GO co-annotation matrix --
    print("\n[2/6] Computing GO co-annotation matrix...")
    coann_matrix = compute_go_coannotation_matrix(nodes, go_map)
    total_pairs = len(nodes) * (len(nodes) - 1) // 2
    coann_pairs = int(coann_matrix.sum() // 2)
    go_coann_fraction = coann_pairs / max(total_pairs, 1)
    print(f"  Co-annotated pairs: {coann_pairs}/{total_pairs} ({go_coann_fraction:.4f})")
    
    # -- Compute GO term IC --
    print("\n[3/6] Computing GO term information content...")
    go_ic, go_term_counts = compute_go_term_ic(nodes, go_map)
    print(f"  Unique GO terms: {len(go_ic)}")
    if go_ic:
        ic_vals = list(go_ic.values())
        print(f"  IC range: [{min(ic_vals):.2f}, {max(ic_vals):.2f}], median {np.median(ic_vals):.2f}")
    
    # -- Analysis 1: Distance-Function Correspondence --
    print("\n[4/6] Computing Distance-Function Correspondence (DFC)...")
    dfc_results = {}
    coann_upper = coann_matrix[np.triu_indices(len(nodes), k=1)]
    
    for method in ALL_METHODS:
        if method not in embeddings:
            continue
        coords_raw, emb_nodes = embeddings[method]
        
        # Align to common nodes
        node_set = set(nodes) & set(emb_nodes) & set(go_map.keys())
        common = sorted(node_set)
        if len(common) < 10:
            print(f"  {method}: too few common nodes ({len(common)}), skipping")
            continue
        
        emb_idx = [emb_nodes.index(nd) for nd in common]
        net_idx = [nodes.index(nd) for nd in common]
        
        coords = coords_raw[emb_idx]
        coords = rescale_coordinates(coords)
        
        # Pairwise distances
        dist_vec = pdist(coords, metric="euclidean")
        
        # Co-annotation for common nodes (recompute for aligned subset)
        go_sets = [set(go_map.get(nd, [])) for nd in common]
        n_c = len(common)
        coann_vec = np.zeros(len(dist_vec))
        idx = 0
        for i in range(n_c):
            for j in range(i + 1, n_c):
                if go_sets[i] & go_sets[j]:
                    coann_vec[idx] = 1.0
                idx += 1
        
        bc, cf, counts = compute_distance_function_correspondence(
            dist_vec, coann_vec, N_DFC_BINS
        )
        
        if bc is not None:
            # Compute DFC-AUC (area under DFC curve as summary metric)
            dfc_auc = float(trapezoid(cf, bc) / (bc[-1] - bc[0]))
            # Short-range enrichment (first 3 bins vs baseline)
            short_enrichment = float(cf[:3].mean() / max(go_coann_fraction, 1e-10))
            # Long-range depletion (last 3 bins vs baseline)
            long_depletion = float(cf[-3:].mean() / max(go_coann_fraction, 1e-10))
            
            dfc_results[method] = {
                "bin_centers": bc,
                "coann_fractions": cf,
                "bin_counts": counts,
                "dfc_auc": dfc_auc,
                "short_range_enrichment": short_enrichment,
                "long_range_depletion": long_depletion,
                "n_common_nodes": n_c,
            }
            print(f"  {method}: DFC-AUC={dfc_auc:.4f}, short enrich={short_enrichment:.2f}x, "
                  f"long deplete={long_depletion:.2f}x")
    
    # -- Analysis 2: Geometric Features --
    print("\n[5/6] Computing geometric features...")
    geo_features = {}
    
    for method in ALL_METHODS:
        if method not in embeddings:
            continue
        coords_raw, emb_nodes = embeddings[method]
        node_set = set(nodes) & set(emb_nodes) & set(go_map.keys())
        common = sorted(node_set)
        if len(common) < 10:
            continue
        
        emb_idx = [emb_nodes.index(nd) for nd in common]
        coords = coords_raw[emb_idx]
        coords = rescale_coordinates(coords)
        
        dist_matrix = squareform(pdist(coords, metric="euclidean"))
        features = compute_geometric_features(coords, dist_matrix)
        geo_features[method] = features
        
        print(f"  {method}: spec_gap={features['spectral_gap']:.4f}, "
              f"fiedler={features['fiedler_value']:.4f}, "
              f"dist_cv={features['dist_cv']:.3f}, "
              f"eff_dim={features['effective_dimensionality']:.2f}")
    
    # -- Analysis 3: G-F Curve Shape Decomposition --
    print("\n[6/6] Decomposing G-F curve shapes...")
    
    # Load existing G-F curves
    curves_path = RESULTS_DIR / "gf_curves_200pts.json"
    with open(curves_path, encoding="utf-8") as f:
        gf_curves_data = json.load(f)
    
    # Also load GNN curves if available
    gnn_curves_path = RESULTS_DIR / "gnn_gf_scores.json"
    
    shape_features = {}
    for method in ALL_METHODS:
        key = f"{method}_purity"
        if key not in gf_curves_data:
            continue
        r_vals = gf_curves_data["r"]
        purity_vals = gf_curves_data[key]
        sf = analyze_gf_curve_shape(r_vals, purity_vals)
        if sf:
            shape_features[method] = sf
            print(f"  {method}: peak_r={sf['peak_r']:.3f}, peak_p={sf['peak_purity']:.3f}, "
                  f"plateau={sf['plateau_width']:.3f}, cv={sf['purity_cv']:.3f}")
    
    # -- Cross-analysis: Correlate geometric features with G-F Score --
    print("\n" + "=" * 70)
    print("Cross-Analysis: Geometric Features vs G-F Score")
    print("=" * 70)
    
    feature_gf_correlations = {}
    common_methods = [m for m in ALL_METHODS if m in geo_features and m in shape_features]
    gf_scores_arr = np.array([shape_features[m]["gf_score"] for m in common_methods])
    
    for feat_name in ["spectral_gap", "fiedler_value", "dist_cv", "effective_dimensionality",
                       "spatial_uniformity_cv", "first_pc_energy"]:
        feat_vals = np.array([geo_features[m][feat_name] for m in common_methods])
        rho, p_val = spearmanr(feat_vals, gf_scores_arr)
        feature_gf_correlations[feat_name] = {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
        }
        print(f"  {feat_name:30s}: rho={rho:+.3f} (p={p_val:.3f})")
    
    # Also correlate DFC metrics with G-F Score
    dfc_methods = [m for m in common_methods if m in dfc_results]
    if len(dfc_methods) >= 4:
        gf_arr2 = np.array([shape_features[m]["gf_score"] for m in dfc_methods])
        for dfc_key in ["dfc_auc", "short_range_enrichment", "long_range_depletion"]:
            dfc_vals = np.array([dfc_results[m][dfc_key] for m in dfc_methods])
            rho, p_val = spearmanr(dfc_vals, gf_arr2)
            feature_gf_correlations[f"dfc_{dfc_key}"] = {
                "spearman_rho": float(rho),
                "p_value": float(p_val),
            }
            print(f"  dfc_{dfc_key:25s}: rho={rho:+.3f} (p={p_val:.3f})")
    
    # -- Generate figures --
    print("\n" + "=" * 70)
    print("Generating figures...")
    print("=" * 70)
    
    plot_dfc_curves(dfc_results, go_coann_fraction)
    plot_geometric_radar(geo_features)
    plot_curve_decomposition(gf_curves_data, shape_features)
    plot_dfc_heatmap(dfc_results, go_coann_fraction)
    
    # -- Save results --
    print("\nSaving results...")
    
    # Serialize DFC results (convert numpy arrays to lists)
    dfc_serializable = {}
    for method, res in dfc_results.items():
        dfc_serializable[method] = {
            "bin_centers": res["bin_centers"].tolist(),
            "coann_fractions": res["coann_fractions"].tolist(),
            "bin_counts": res["bin_counts"].tolist(),
            "dfc_auc": res["dfc_auc"],
            "short_range_enrichment": res["short_range_enrichment"],
            "long_range_depletion": res["long_range_depletion"],
            "n_common_nodes": res["n_common_nodes"],
        }
    
    output = {
        "analysis": "Phase 1: Multi-Scale Geometric Fingerprint",
        "version": "1.0",
        "network": {
            "n_nodes": len(nodes),
            "n_edges": G.number_of_edges(),
            "n_annotated": n_annotated,
            "go_coann_fraction": go_coann_fraction,
            "coann_pairs": coann_pairs,
            "total_pairs": total_pairs,
        },
        "go_statistics": {
            "n_unique_terms": len(go_ic),
            "ic_mean": float(np.mean(list(go_ic.values()))) if go_ic else 0,
            "ic_median": float(np.median(list(go_ic.values()))) if go_ic else 0,
        },
        "dfc_results": dfc_serializable,
        "geometric_features": geo_features,
        "shape_features": shape_features,
        "feature_gf_correlations": feature_gf_correlations,
    }
    
    output_path = RESULTS_DIR / "deep_geometric_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved {output_path}")
    
    # -- Print summary --
    print("\n" + "=" * 70)
    print("SUMMARY: Key Findings")
    print("=" * 70)
    
    # Rank methods by DFC-AUC
    if dfc_results:
        dfc_ranked = sorted(dfc_results.items(), key=lambda x: x[1]["dfc_auc"], reverse=True)
        print("\nDistance-Function Correspondence (DFC-AUC) ranking:")
        for i, (m, r) in enumerate(dfc_ranked, 1):
            print(f"  {i:2d}. {m:12s}  DFC-AUC={r['dfc_auc']:.4f}  "
                  f"short={r['short_range_enrichment']:.2f}x  long={r['long_range_depletion']:.2f}x")
    
    # Most predictive geometric feature
    if feature_gf_correlations:
        best_feat = max(feature_gf_correlations.items(),
                        key=lambda x: abs(x[1]["spearman_rho"]))
        print(f"\nMost predictive geometric feature: {best_feat[0]} "
              f"(rho={best_feat[1]['spearman_rho']:+.3f}, p={best_feat[1]['p_value']:.3f})")
    
    # Curve shape diversity
    if shape_features:
        peak_locations = [sf["peak_r"] for sf in shape_features.values()]
        print(f"\nPeak location range: [{min(peak_locations):.3f}, {max(peak_locations):.3f}]")
        print("  → Methods diverge most at this scale range")
    
    print("\n" + "=" * 70)
    print("Phase 1 complete. Results ready for interpretation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
