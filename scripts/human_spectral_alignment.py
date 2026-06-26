#!/usr/bin/env python3
"""
human_spectral_alignment.py -- Phase 5A: Human Network Spectral Alignment
==========================================================================

Tests whether the Phase 3 two-factor model (spectral alignment + effective
dimensionality) generalises from yeast (153 nodes) to human PPI (~2000 nodes).

Key questions:
  1. Does the human Laplacian have a functional frequency band analogous
     to yeast's modes 1-4?
  2. Do spectral alignment scores predict human G-F Score rankings?
  3. Does effective dimensionality predict human G-F Score on the human
     Laplacian (Phase 2 showed it does NOT on the raw features)?
  4. Does the combined two-factor model transfer cross-network?

Outputs:
  - results/human_spectral_alignment.json
  - figures/Fig39_human_spectral_decomposition.png
  - figures/Fig40_human_alignment_summary.png
"""
from __future__ import annotations

import sys
import json
import gzip
import math
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# Paths (portable)
# ============================================================
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from utils import (
    SEED, ALL_METHODS, rescale_coordinates,
    get_data_dir, get_results_dir, get_figures_dir,
)

DATA = get_data_dir()
RES = get_results_dir()
FIG = get_figures_dir()
HUMAN_STRING = _SCRIPTS.parent / "human_validation" / "9606.protein.links.v12.0.txt.gz"

SUBSAMPLE = 2000
SCORE_THRESHOLD = 700
K_MODES = 50  # number of Laplacian modes to compute

METHOD_COLORS = {
    "Spectral": "#E69F00", "DM": "#0072B2", "MDS": "#009E73",
    "Node2Vec": "#CC79A7", "PCA": "#56B4E9", "VGAE-feat": "#F0E442",
    "DeepWalk": "#D55E00", "GIN": "#949494", "GAT": "#000000",
    "GraphSAGE": "#8B4513", "VGAE": "#808080",
}


# ============================================================
# Data Loading
# ============================================================

def load_human_network():
    """Load human STRING PPI network (score >= 700), return largest CC."""
    print("  Loading human STRING v12.0...")
    G = nx.Graph()
    with gzip.open(str(HUMAN_STRING), "rt", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p1, p2, score = parts
            if int(score) >= SCORE_THRESHOLD:
                p1_clean = p1.strip()  # keep full ID "9606.ENSP..."
                p2_clean = p2.strip()  # to match embedding/GO node names
                G.add_edge(p1_clean, p2_clean)
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    print(f"  Human network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_human_go():
    """Load human GO annotations."""
    with open(DATA / "human_go_annotations.json", encoding="utf-8") as f:
        return json.load(f)


def load_human_embeddings(go_map):
    """Load all 11 human embeddings from JSON files."""
    embeddings = {}
    for method in ALL_METHODS:
        fpath = DATA / f"human_{method.lower()}_embedding.json"
        if not fpath.exists():
            # Try with hyphen
            alt = method.lower().replace("-", "-")
            fpath = DATA / f"human_{alt}_embedding.json"
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        # raw is {node_id: {"x": float, "y": float}} or {node_id: [x, y]}
        if raw:
            first_val = next(iter(raw.values()))
            if isinstance(first_val, dict):
                nodes = sorted(raw.keys())
                coords = np.array([[raw[n]["x"], raw[n]["y"]] for n in nodes])
            elif isinstance(first_val, list):
                nodes = sorted(raw.keys())
                coords = np.array([raw[n] for n in nodes])
            else:
                continue
            # Filter to annotated nodes
            annotated = set(go_map.keys())
            mask = [n in annotated for n in nodes]
            coords_ann = coords[mask]
            nodes_ann = [n for n, m in zip(nodes, mask) if m]
            embeddings[method] = (coords_ann, nodes_ann)
    return embeddings


def subsample_nodes(embeddings, go_map, n=SUBSAMPLE, seed=SEED):
    """Get consistent subsample of annotated nodes across all methods."""
    rng = np.random.default_rng(seed)
    all_annot = set(go_map.keys())
    for method in embeddings:
        all_annot &= set(embeddings[method][1])
    all_annot = sorted(all_annot)
    if len(all_annot) <= n:
        return all_annot
    return sorted(rng.choice(all_annot, n, replace=False))


# ============================================================
# Laplacian Eigenbasis
# ============================================================

def compute_laplacian_eigenbasis(G, nodes, k=K_MODES):
    """Compute normalized Laplacian eigenbasis for the largest CC of subgraph on `nodes`."""
    subG = G.subgraph(nodes).copy()
    # Restrict to largest connected component (essential for meaningful spectrum)
    if not nx.is_connected(subG):
        largest_cc = max(nx.connected_components(subG), key=len)
        n_cc = nx.number_connected_components(subG)
        print(f"    Subgraph has {n_cc} components; restricting to largest CC "
              f"({len(largest_cc)} nodes)")
        subG = subG.subgraph(largest_cc).copy()
    n = subG.number_of_nodes()
    node_list = sorted(subG.nodes())
    
    A = nx.adjacency_matrix(subG, nodelist=node_list).astype(float)
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = csr_matrix((deg_inv_sqrt, (range(n), range(n))), shape=(n, n))
    I = csr_matrix(np.eye(n))
    L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Compute smallest k eigenvalues/vectors
    k_actual = min(k, n - 2)
    eigenvalues, eigenvectors = eigsh(L_norm, k=k_actual, which="SM")
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors, node_list


# ============================================================
# Functional Frequency Band
# ============================================================

def compute_functional_frequency_band(eigenvectors, node_list, go_map, k=None):
    """Measure each eigenvector's ability to separate GO modules."""
    if k is None:
        k = eigenvectors.shape[1]
    
    # Assign GO module labels (most specific = most common GO term)
    labels = []
    for nd in node_list:
        terms = go_map.get(nd, [])
        if terms:
            labels.append(terms[0])  # first GO term as label
        else:
            labels.append("__unannotated__")
    
    unique_labels = list(set(labels) - {"__unannotated__"})
    if len(unique_labels) < 2:
        return np.zeros(k)
    
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    
    # For each eigenvector, compute between-class / total variance ratio
    func_alignment = np.zeros(k)
    for j in range(k):
        vec = eigenvectors[:, j]
        
        # Class means
        class_means = {}
        class_counts = {}
        for i, lab in enumerate(labels):
            if lab == "__unannotated__":
                continue
            li = label_to_idx[lab]
            class_means[li] = class_means.get(li, 0.0) + vec[i]
            class_counts[li] = class_counts.get(li, 0) + 1
        
        if len(class_means) < 2:
            continue
        
        for li in class_means:
            class_means[li] /= class_counts[li]
        
        global_mean = np.mean(vec)
        
        # Between-class variance
        ss_between = sum(class_counts[li] * (class_means[li] - global_mean) ** 2
                         for li in class_means)
        # Total variance
        ss_total = np.sum((vec - global_mean) ** 2)
        
        if ss_total > 1e-10:
            func_alignment[j] = ss_between / ss_total
    
    return func_alignment


# ============================================================
# Spectral Decomposition of Embeddings
# ============================================================

def decompose_embedding_in_eigenbasis(coords, emb_nodes, eigenvalues, eigenvectors,
                                       lap_nodes, func_band):
    """Project embedding coords onto Laplacian eigenbasis."""
    # Align nodes
    common = sorted(set(emb_nodes) & set(lap_nodes))
    if len(common) < 50:
        return None
    
    emb_idx = {n: i for i, n in enumerate(emb_nodes)}
    lap_idx = {n: i for i, n in enumerate(lap_nodes)}
    
    emb_sub = np.array([emb_idx[n] for n in common])
    lap_sub = np.array([lap_idx[n] for n in common])
    
    Y = coords[emb_sub]
    Y = rescale_coordinates(Y.copy())
    Y_centered = Y - Y.mean(axis=0)
    
    Phi = eigenvectors[lap_sub, :]  # (n_common, k_modes)
    
    # Project: coefficients for each dimension
    n_dims = Y_centered.shape[1]
    k_modes = Phi.shape[1]
    energy_spectrum = np.zeros(k_modes)
    
    for d in range(n_dims):
        c = Phi.T @ Y_centered[:, d]  # (k_modes,)
        energy_spectrum += c ** 2
    
    # Normalize
    total_energy = energy_spectrum.sum()
    if total_energy > 1e-10:
        energy_spectrum /= total_energy
    
    # Spectral alignment score: weighted overlap with functional band
    alignment = float(np.dot(energy_spectrum, func_band))
    
    # Energy in top-3 modes
    top3_energy = float(energy_spectrum[:3].sum())
    
    # Number of modes for 90% energy
    cumsum = np.cumsum(energy_spectrum)
    modes_90 = int(np.searchsorted(cumsum, 0.90) + 1)
    
    return {
        "alignment_score": alignment,
        "top3_energy": top3_energy,
        "modes_90pct": modes_90,
        "energy_spectrum": energy_spectrum.tolist(),
    }


# ============================================================
# Effective Dimensionality
# ============================================================

def compute_effective_dimensionality(coords):
    """Participation ratio of PCA eigenvalues."""
    coords = rescale_coordinates(coords.copy())
    coords_c = coords - coords.mean(axis=0)
    cov = coords_c.T @ coords_c / coords_c.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    s = eigvals.sum()
    if s < 1e-10:
        return 1.0
    return float(s ** 2 / max((eigvals ** 2).sum(), 1e-10))


# ============================================================
# Visualization
# ============================================================

def plot_human_spectral_decomposition(all_profiles, eigenvalues, func_band, gf_scores):
    """Fig39: Human spectral decomposition."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    methods_sorted = sorted(all_profiles.keys(),
                            key=lambda m: all_profiles[m]["alignment_score"],
                            reverse=True)
    
    # Panel A: Functional frequency band
    ax = axes[0, 0]
    k = min(30, len(func_band))
    ax.bar(range(k), func_band[:k], color="#0072B2", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Laplacian mode index", fontsize=12)
    ax.set_ylabel("Functional alignment (SS_between/SS_total)", fontsize=11)
    ax.set_title("A. Human Functional Frequency Band", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Panel B: Spectral alignment scores bar chart
    ax2 = axes[0, 1]
    colors = [METHOD_COLORS.get(m, "#333") for m in methods_sorted]
    ax2.barh(methods_sorted[::-1],
             [all_profiles[m]["alignment_score"] for m in methods_sorted[::-1]],
             color=colors[::-1], alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Spectral alignment score", fontsize=11)
    ax2.set_title("B. Human Spectral Alignment Scores", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    
    # Panel C: Energy spectrum for top methods
    ax3 = axes[1, 0]
    top_methods = methods_sorted[:5]
    for m in top_methods:
        spectrum = np.array(all_profiles[m]["energy_spectrum"])
        k_plot = min(20, len(spectrum))
        ax3.plot(range(k_plot), spectrum[:k_plot], "o-",
                 color=METHOD_COLORS.get(m, "#333"), label=m, alpha=0.8, markersize=4)
    ax3.set_xlabel("Laplacian mode index", fontsize=12)
    ax3.set_ylabel("Energy fraction", fontsize=12)
    ax3.set_title("C. Embedding Energy Spectra (Top 5)", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Spectral alignment vs G-F Score
    ax4 = axes[1, 1]
    methods_c = sorted(set(all_profiles.keys()) & set(gf_scores.keys()))
    align_vals = [all_profiles[m]["alignment_score"] for m in methods_c]
    gf_vals = [gf_scores[m] for m in methods_c]
    for m in methods_c:
        ax4.scatter(all_profiles[m]["alignment_score"], gf_scores[m],
                    color=METHOD_COLORS.get(m, "#333"), s=100, zorder=5,
                    edgecolors="white", linewidth=1.0)
        ax4.annotate(m, (all_profiles[m]["alignment_score"], gf_scores[m]),
                     fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    if len(methods_c) >= 4:
        rho, p = spearmanr(align_vals, gf_vals)
        ax4.set_title(f"D. Human Alignment vs G-F Score\n(rho={rho:.3f}, p={p:.3f})",
                      fontsize=13, fontweight="bold")
    ax4.set_xlabel("Spectral alignment score", fontsize=11)
    ax4.set_ylabel("Human G-F Score", fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIG / "Fig39_human_spectral_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig39_human_spectral_decomposition.png")


def plot_human_alignment_summary(all_profiles, eff_dims, gf_scores, yeast_profiles):
    """Fig40: Human vs yeast alignment comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    methods_c = sorted(set(all_profiles.keys()) & set(gf_scores.keys()) & set(eff_dims.keys()))
    
    # Panel A: Eff dim vs G-F Score on human
    ax = axes[0]
    for m in methods_c:
        ax.scatter(eff_dims[m], gf_scores[m], color=METHOD_COLORS.get(m, "#333"),
                   s=100, zorder=5, edgecolors="white", linewidth=1.0)
        ax.annotate(m, (eff_dims[m], gf_scores[m]), fontsize=8, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")
    if len(methods_c) >= 4:
        rho, p = spearmanr([eff_dims[m] for m in methods_c],
                           [gf_scores[m] for m in methods_c])
        ax.set_title(f"A. Human: Eff Dim vs G-F Score\n(rho={rho:.3f}, p={p:.3f})",
                     fontsize=13, fontweight="bold")
    ax.axvline(1.3, color="red", linestyle="--", alpha=0.4, label="Threshold (1.3)")
    ax.set_xlabel("Effective dimensionality", fontsize=11)
    ax.set_ylabel("Human G-F Score", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Two-factor model on human
    ax2 = axes[1]
    align_h = np.array([all_profiles[m]["alignment_score"] for m in methods_c])
    edim_h = np.array([eff_dims[m] for m in methods_c])
    gf_h = np.array([gf_scores[m] for m in methods_c])
    
    # Normalize factors to [0,1]
    align_norm = (align_h - align_h.min()) / max(align_h.max() - align_h.min(), 1e-10)
    edim_norm = (edim_h - edim_h.min()) / max(edim_h.max() - edim_h.min(), 1e-10)
    combined = 0.5 * align_norm + 0.5 * edim_norm
    
    for i, m in enumerate(methods_c):
        ax2.scatter(combined[i], gf_h[i], color=METHOD_COLORS.get(m, "#333"),
                    s=100, zorder=5, edgecolors="white", linewidth=1.0)
        ax2.annotate(m, (combined[i], gf_h[i]), fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    if len(methods_c) >= 4:
        rho2, p2 = spearmanr(combined, gf_h)
        ax2.set_title(f"B. Human Two-Factor Model\n(rho={rho2:.3f}, p={p2:.3f})",
                      fontsize=13, fontweight="bold")
    ax2.set_xlabel("Combined score (0.5*align + 0.5*eff_dim)", fontsize=11)
    ax2.set_ylabel("Human G-F Score", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Yeast vs human alignment comparison
    ax3 = axes[2]
    methods_both = sorted(set(methods_c) & set(yeast_profiles.keys()))
    for m in methods_both:
        yeast_a = yeast_profiles[m]["alignment_score"]
        human_a = all_profiles[m]["alignment_score"]
        ax3.scatter(yeast_a, human_a, color=METHOD_COLORS.get(m, "#333"),
                    s=100, zorder=5, edgecolors="white", linewidth=1.0)
        ax3.annotate(m, (yeast_a, human_a), fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    if len(methods_both) >= 4:
        rho3, p3 = spearmanr([yeast_profiles[m]["alignment_score"] for m in methods_both],
                             [all_profiles[m]["alignment_score"] for m in methods_both])
        ax3.set_title(f"C. Yeast vs Human Alignment\n(rho={rho3:.3f}, p={p3:.3f})",
                      fontsize=13, fontweight="bold")
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    ax3.set_xlabel("Yeast spectral alignment", fontsize=11)
    ax3.set_ylabel("Human spectral alignment", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIG / "Fig40_human_alignment_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig40_human_alignment_summary.png")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Phase 5A: Human Network Spectral Alignment")
    print("=" * 70)
    
    # [1/7] Load data
    print("\n[1/7] Loading human data...")
    G = load_human_network()
    go_map = load_human_go()
    print(f"  GO annotations: {len(go_map)} genes")
    
    # [2/7] Load embeddings
    print("\n[2/7] Loading human embeddings...")
    embeddings = load_human_embeddings(go_map)
    print(f"  Loaded {len(embeddings)} methods")
    
    # [3/7] Subsample
    print("\n[3/7] Subsampling to 2000 nodes...")
    sample_nodes = subsample_nodes(embeddings, go_map)
    print(f"  Subsample: {len(sample_nodes)} nodes")
    
    # [4/7] Compute Laplacian eigenbasis
    print("\n[4/7] Computing Laplacian eigenbasis (k=50)...")
    eigenvalues, eigenvectors, lap_nodes = compute_laplacian_eigenbasis(G, sample_nodes)
    print(f"  Eigenvalues[0:5] = {eigenvalues[:5].tolist()}")
    print(f"  Spectral gap = {eigenvalues[1] - eigenvalues[0]:.6f}")
    
    # [5/7] Identify functional frequency band
    print("\n[5/7] Identifying functional frequency band...")
    func_band = compute_functional_frequency_band(eigenvectors, lap_nodes, go_map)
    print(f"  Functional alignment (first 10 modes):")
    for j in range(min(10, len(func_band))):
        print(f"    Mode {j}: {func_band[j]:.4f}")
    top_func_modes = np.argsort(func_band)[::-1][:5]
    print(f"  Top 5 functional modes: {top_func_modes.tolist()}")
    
    # [6/7] Decompose embeddings
    print("\n[6/7] Spectral decomposition of human embeddings...")
    
    # Load human G-F scores
    with open(RES / "human_gf_scores_extended.json", encoding="utf-8") as f:
        hgf_data = json.load(f)
    gf_scores = {}
    scores_raw = hgf_data.get("scores", hgf_data.get("gf_scores", {}))
    if isinstance(scores_raw, dict):
        gf_scores = dict(scores_raw)
    elif isinstance(scores_raw, list):
        gf_scores = {entry[0]: entry[1] if isinstance(entry[1], float) else entry[1].get("gf_score", 0)
                     for entry in scores_raw}
    
    # Load yeast Phase 3 results for comparison
    yeast_profiles = {}
    try:
        with open(RES / "spectral_alignment.json", encoding="utf-8") as f:
            ydata = json.load(f)
        for m, prof in ydata.get("alignment_results", {}).items():
            yeast_profiles[m] = {
                "alignment_score": prof.get("alignment_score", 0),
                "top3_energy": prof.get("top3_energy", 0),
            }
    except Exception as e:
        print("  WARNING: Could not load yeast Phase 3 results")
    
    all_profiles = {}
    eff_dims = {}
    
    print(f"  {'Method':12s} {'Align':>8s} {'Top3':>7s} {'EffDim':>8s} {'G-F':>8s}")
    for method in ALL_METHODS:
        if method not in embeddings:
            continue
        coords_raw, emb_nodes = embeddings[method]
        
        # Filter to subsample
        sub_idx = {n: i for i, n in enumerate(emb_nodes)}
        valid = [n for n in sample_nodes if n in sub_idx]
        if len(valid) < 100:
            continue
        
        idx = [sub_idx[n] for n in valid]
        coords_sub = coords_raw[idx]
        
        # Effective dimensionality
        ed = compute_effective_dimensionality(coords_sub)
        eff_dims[method] = ed
        
        # Spectral decomposition
        profile = decompose_embedding_in_eigenbasis(
            coords_raw, emb_nodes, eigenvalues, eigenvectors, lap_nodes, func_band)
        
        if profile is None:
            continue
        
        all_profiles[method] = profile
        gf = gf_scores.get(method, float("nan"))
        print(f"  {method:12s} {profile['alignment_score']:8.4f} "
              f"{profile['top3_energy']:7.3f} {ed:8.3f} {gf:8.4f}")
    
    # [7/7] Generate figures
    print("\n[7/7] Generating figures...")
    plot_human_spectral_decomposition(all_profiles, eigenvalues, func_band, gf_scores)
    plot_human_alignment_summary(all_profiles, eff_dims, gf_scores, yeast_profiles)
    
    # -- Save results --
    print("\nSaving results...")
    
    # Compute key correlations
    methods_c = sorted(set(all_profiles.keys()) & set(gf_scores.keys()) & set(eff_dims.keys()))
    
    align_vals = [all_profiles[m]["alignment_score"] for m in methods_c]
    edim_vals = [eff_dims[m] for m in methods_c]
    gf_vals = [gf_scores[m] for m in methods_c]
    
    rho_align, p_align = spearmanr(align_vals, gf_vals) if len(methods_c) >= 4 else (0, 1)
    rho_edim, p_edim = spearmanr(edim_vals, gf_vals) if len(methods_c) >= 4 else (0, 1)
    
    # Two-factor combined
    align_arr = np.array(align_vals)
    edim_arr = np.array(edim_vals)
    align_norm = (align_arr - align_arr.min()) / max(align_arr.max() - align_arr.min(), 1e-10)
    edim_norm = (edim_arr - edim_arr.min()) / max(edim_arr.max() - edim_arr.min(), 1e-10)
    combined = 0.5 * align_norm + 0.5 * edim_norm
    rho_combined, p_combined = spearmanr(combined, gf_vals) if len(methods_c) >= 4 else (0, 1)
    
    print(f"\n  Human spectral alignment vs G-F: rho={rho_align:.3f} (p={p_align:.3f})")
    print(f"  Human eff_dim vs G-F:           rho={rho_edim:.3f} (p={p_edim:.3f})")
    print(f"  Human two-factor model:         rho={rho_combined:.3f} (p={p_combined:.3f})")
    
    output = {
        "analysis": "Phase 5A: Human Network Spectral Alignment",
        "version": "1.0",
        "network": {
            "species": "human",
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "subsample_size": len(sample_nodes),
            "score_threshold": SCORE_THRESHOLD,
        },
        "laplacian": {
            "k_modes": len(eigenvalues),
            "eigenvalues": eigenvalues.tolist(),
            "spectral_gap": float(eigenvalues[1] - eigenvalues[0]),
        },
        "functional_frequency_band": func_band.tolist(),
        "top_functional_modes": top_func_modes.tolist(),
        "spectral_profiles": {m: {k: v for k, v in prof.items() if k != "energy_spectrum"}
                              for m, prof in all_profiles.items()},
        "energy_spectra": {m: prof["energy_spectrum"] for m, prof in all_profiles.items()},
        "effective_dimensionality": eff_dims,
        "gf_scores": {m: gf_scores.get(m, None) for m in methods_c},
        "correlations": {
            "spectral_alignment_vs_gf": {"rho": float(rho_align), "p": float(p_align)},
            "eff_dim_vs_gf": {"rho": float(rho_edim), "p": float(p_edim)},
            "two_factor_vs_gf": {"rho": float(rho_combined), "p": float(p_combined)},
        },
        "yeast_comparison": {
            "methods_available": list(yeast_profiles.keys()),
            "note": "Cross-network comparison of spectral alignment scores",
        },
    }
    
    output_path = RES / "human_spectral_alignment.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved {output_path}")
    
    print("\n" + "=" * 70)
    print("Phase 5A complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
