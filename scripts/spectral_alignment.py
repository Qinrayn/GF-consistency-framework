#!/usr/bin/env python3
"""
spectral_alignment.py — Phase 3: Network-Aware Spectral Alignment
===============================================================

Tests the revised proposition from Phase 2: G-F consistency depends on
the MATCH between embedding geometry and host network spectral structure.

For each embedding method, decompose its 2D coordinates in the network's
Laplacian eigenbasis, then measure how well the captured spectral modes
align with the network's "functional frequency band".

Three analysis modules:
  1. Spectral decomposition of embeddings in Laplacian eigenbasis
  2. Functional frequency band identification (which eigenvectors separate GO modules)
  3. Spectral alignment score: overlap between embedding spectrum and functional band

Outputs:
  - results/spectral_alignment.json
  - figures/Fig34_spectral_decomposition.png
  - figures/Fig35_spectral_alignment.png

Note on terminology:
  The "effective dimensionality" used here is the PCA eigenvalue
  participation ratio from Phase 1 (deep_geometric_analysis.json).
  Phase 4/7 use a different metric: SVD singular-value participation
  ratio ("effective rank").  Both measure how evenly the embedding
  uses its dimensions, but they are computed from different spectra.
"""
from __future__ import annotations

import sys
import json
import math
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# Paths (portable via utils helpers)
# ============================================================
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from utils import (
    SEED, ALL_METHODS, ALL_CURATED_METHODS,
    rescale_coordinates, load_curated_network, load_embedding,
    get_data_dir, get_embeddings_dir, get_results_dir, get_figures_dir,
)

# Portable directory aliases
DATA = get_data_dir()
EMB = get_embeddings_dir()
RES = get_results_dir()
FIG = get_figures_dir()

METHOD_COLORS = {
    "Spectral": "#E69F00", "DM": "#0072B2", "MDS": "#009E73",
    "Node2Vec": "#CC79A7", "PCA": "#56B4E9", "VGAE-feat": "#F0E442",
    "DeepWalk": "#D55E00", "GIN": "#949494", "GAT": "#000000",
    "GraphSAGE": "#8B4513", "VGAE": "#808080",
}


# ============================================================
# 1. Laplacian eigendecomposition
# ============================================================

def compute_laplacian_eigenbasis(G, nodes, k=30):
    """
    Compute the first k eigenvectors of the normalised graph Laplacian.
    Returns (eigenvalues, eigenvectors) where eigenvectors[:, i] is the i-th.
    """
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    
    A = np.zeros((n, n))
    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            A[i, j] = 1; A[j, i] = 1
    
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-10))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Symmetric — use eigh for numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    
    # Sort ascending (should already be, but ensure)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    k_actual = min(k, n)
    return eigenvalues[:k_actual], eigenvectors[:, :k_actual], A, deg


def compute_functional_frequency_band(eigenvectors, nodes, go_map, k=30):
    """
    For each Laplacian eigenvector, measure how well it separates
    GO-annotated functional modules.
    
    Metric: For each eigenvector v_i, compute the variance of v_i
    within GO-term groups vs total variance. Eigenvectors that assign
    similar values to co-annotated nodes are "functionally aligned".
    
    Returns array of "functional alignment scores" for each eigenvector.
    """
    k_actual = eigenvectors.shape[1]
    n = len(nodes)
    
    # Build GO-term groups: for each GO term, list of node indices
    go_groups = {}
    for i, nd in enumerate(nodes):
        for term in go_map.get(nd, []):
            if term not in go_groups:
                go_groups[term] = []
            go_groups[term].append(i)
    
    # Filter to groups with >= 3 nodes
    go_groups = {t: idxs for t, idxs in go_groups.items() if len(idxs) >= 3}
    
    if not go_groups:
        return np.zeros(k_actual)
    
    func_alignment = np.zeros(k_actual)
    for j in range(k_actual):
        v = eigenvectors[:, j]
        total_var = np.var(v)
        if total_var < 1e-12:
            continue
        
        # Within-group variance (averaged over GO groups)
        within_vars = []
        for term, idxs in go_groups.items():
            within_vars.append(np.var(v[idxs]))
        mean_within = np.mean(within_vars)
        
        # Alignment = 1 - (within-group variance / total variance)
        # Higher = eigenvector separates GO groups well
        func_alignment[j] = 1.0 - mean_within / total_var
    
    return func_alignment


# ============================================================
# 2. Embedding spectral decomposition
# ============================================================

def decompose_embedding_in_eigenbasis(coords, eigenvectors, k=30):
    """
    Project embedding coordinates onto the Laplacian eigenbasis.
    
    For embedding Y (n x d), compute projection coefficients:
        c_j = v_j^T Y  (for each eigenvector v_j)
    
    Returns:
        coefficients: (k, d) — projection of each embedding dim onto each eigvec
        spectral_energy: (k,) — total energy captured from each eigenvector
    """
    k_actual = min(k, eigenvectors.shape[1])
    V = eigenvectors[:, :k_actual]  # (n, k)
    
    # Project: C = V^T Y  (k x d)
    C = V.T @ coords
    
    # Energy per eigenvector: sum of squared projections across all dims
    spectral_energy = (C ** 2).sum(axis=1)  # (k,)
    
    # Normalize to total energy
    total_energy = spectral_energy.sum()
    if total_energy > 1e-12:
        spectral_profile = spectral_energy / total_energy
    else:
        spectral_profile = np.zeros_like(spectral_energy)
    
    return C, spectral_energy, spectral_profile


def compute_spectral_alignment_score(spectral_profile, func_band, k=30):
    """
    Compute the overlap between embedding's spectral profile and the
    network's functional frequency band.
    
    alignment = Σ_j profile(j) * func_band(j)
    
    This is a weighted average of functional alignment, weighted by how
    much of each eigenvector the embedding captures.
    """
    k_actual = min(k, len(spectral_profile), len(func_band))
    profile = spectral_profile[:k_actual]
    band = func_band[:k_actual]
    
    # Weighted overlap
    alignment = float(np.dot(profile, band))
    
    # Also compute: cumulative alignment for first m eigenvectors
    cumulative = np.cumsum(profile * band)
    
    # Top-k alignment: how much alignment comes from the top 3 modes
    top3 = float(cumulative[min(2, k_actual - 1)])
    
    # Spectral concentration: how much energy in top 3 modes
    top3_energy = float(profile[:3].sum())
    
    return {
        "alignment_score": alignment,
        "cumulative_alignment": cumulative[:min(10, k_actual)].tolist(),
        "top3_alignment": top3,
        "top3_energy": top3_energy,
        "n_modes_90pct": int(np.searchsorted(np.cumsum(profile), 0.9) + 1),
    }


# ============================================================
# 3. Visualization
# ============================================================

def plot_spectral_decomposition(all_profiles, eigenvalues, func_band, gf_scores):
    """Fig 34: Spectral decomposition of each method in Laplacian eigenbasis."""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    k_show = 15  # show first 15 modes
    
    # Panel A: Functional frequency band
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(min(k_show, len(func_band))), func_band[:k_show],
           color="#0072B2", alpha=0.7)
    ax.set_xlabel("Laplacian eigenvector index", fontsize=11)
    ax.set_ylabel("Functional alignment score", fontsize=11)
    ax.set_title("A. Functional Frequency Band\n(GO module separation per eigenvector)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Panel B: Spectral profiles (all methods, overlaid)
    ax2 = fig.add_subplot(gs[0, 1])
    for method in sorted(all_profiles.keys()):
        profile = all_profiles[method]
        k_m = min(k_show, len(profile))
        lw = 2.0 if method in ["Spectral", "DM", "MDS"] else 1.0
        ls = "-" if method in ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE", "PCA", "VGAE-feat"] else "--"
        ax2.plot(range(k_m), profile[:k_m], color=METHOD_COLORS.get(method, "#333"),
                 linewidth=lw, linestyle=ls, label=method, alpha=0.8)
    ax2.set_xlabel("Laplacian eigenvector index", fontsize=11)
    ax2.set_ylabel("Spectral energy (normalised)", fontsize=11)
    ax2.set_title("B. Embedding Spectral Profiles", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=6.5, loc="upper right", ncol=2, framealpha=0.85)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Spectral alignment score vs G-F Score
    ax3 = fig.add_subplot(gs[0, 2])
    for method in sorted(gf_scores.keys()):
        if method not in all_profiles:
            continue
        profile = all_profiles[method]
        k_m = min(len(profile), len(func_band))
        align = float(np.dot(profile[:k_m], func_band[:k_m]))
        gf = gf_scores[method]
        ax3.scatter(align, gf, color=METHOD_COLORS.get(method, "#333"),
                    s=100, zorder=5, edgecolors="white", linewidth=1.0)
        ax3.annotate(method, (align, gf), fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    
    # Correlation
    methods_c = sorted(set(gf_scores.keys()) & set(all_profiles.keys()))
    if len(methods_c) >= 4:
        aligns = []
        gfs = []
        for m in methods_c:
            p = all_profiles[m]
            k_m = min(len(p), len(func_band))
            aligns.append(float(np.dot(p[:k_m], func_band[:k_m])))
            gfs.append(gf_scores[m])
        rho, p_val = spearmanr(aligns, gfs)
        ax3.set_title(f"C. Spectral Alignment vs G-F Score\n(Spearman rho = {rho:.3f}, p = {p_val:.3f})",
                      fontsize=12, fontweight="bold")
    else:
        ax3.set_title("C. Spectral Alignment vs G-F Score", fontsize=12, fontweight="bold")
    
    ax3.set_xlabel("Spectral alignment score", fontsize=11)
    ax3.set_ylabel("G-F Score", fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Spectral profile heatmap
    ax4 = fig.add_subplot(gs[1, 0])
    methods_ordered = sorted(all_profiles.keys(),
                              key=lambda m: gf_scores.get(m, 0), reverse=True)
    heatmap_data = []
    for m in methods_ordered:
        p = all_profiles[m]
        padded = np.zeros(k_show)
        padded[:min(k_show, len(p))] = p[:min(k_show, len(p))]
        heatmap_data.append(padded)
    heatmap_data = np.array(heatmap_data)
    
    im = ax4.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax4.set_yticks(range(len(methods_ordered)))
    ax4.set_yticklabels(methods_ordered, fontsize=9)
    ax4.set_xlabel("Laplacian eigenvector index", fontsize=11)
    ax4.set_title("D. Spectral Profile Heatmap\n(methods ranked by G-F Score)",
                  fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax4, shrink=0.8, label="Normalised energy")
    
    # Panel E: Top-3 mode energy vs G-F Score
    ax5 = fig.add_subplot(gs[1, 1])
    for method in methods_c:
        p = all_profiles[method]
        top3 = float(p[:3].sum())
        gf = gf_scores[method]
        ax5.scatter(top3, gf, color=METHOD_COLORS.get(method, "#333"),
                    s=100, zorder=5, edgecolors="white", linewidth=1.0)
        ax5.annotate(method, (top3, gf), fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    if len(methods_c) >= 4:
        top3s = [float(all_profiles[m][:3].sum()) for m in methods_c]
        rho5, p5 = spearmanr(top3s, [gf_scores[m] for m in methods_c])
        ax5.set_title(f"E. Top-3 Mode Energy vs G-F Score\n(rho = {rho5:.3f}, p = {p5:.3f})",
                      fontsize=12, fontweight="bold")
    ax5.set_xlabel("Energy in first 3 Laplacian modes", fontsize=11)
    ax5.set_ylabel("G-F Score", fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # Panel F: Eigenvalues spectrum (for reference)
    ax6 = fig.add_subplot(gs[1, 2])
    k_eig = min(30, len(eigenvalues))
    ax6.bar(range(k_eig), eigenvalues[:k_eig], color="#009E73", alpha=0.7)
    ax6.set_xlabel("Eigenvalue index", fontsize=11)
    ax6.set_ylabel("Laplacian eigenvalue", fontsize=11)
    ax6.set_title("F. Network Laplacian Spectrum (reference)", fontsize=12, fontweight="bold")
    ax6.grid(True, alpha=0.3, axis="y")
    
    fig.suptitle("Spectral Alignment: Embedding x Network Eigenbasis",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.savefig(FIG / "Fig34_spectral_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig34_spectral_decomposition.png")


def plot_spectral_alignment_summary(alignment_results, gf_scores, func_band):
    """Fig 35: Summary of spectral alignment analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods_sorted = sorted(alignment_results.keys(),
                            key=lambda m: alignment_results[m]["alignment_score"],
                            reverse=True)
    
    # Panel A: Alignment scores bar chart
    ax = axes[0]
    colors = [METHOD_COLORS.get(m, "#333") for m in methods_sorted]
    ax.barh(methods_sorted,
            [alignment_results[m]["alignment_score"] for m in methods_sorted],
            color=colors, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Spectral alignment score", fontsize=11)
    ax.set_title("A. Spectral Alignment Ranking", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Panel B: Cumulative alignment curves
    ax2 = axes[1]
    for method in methods_sorted:
        cum = alignment_results[method]["cumulative_alignment"]
        ax2.plot(range(len(cum)), cum, color=METHOD_COLORS.get(method, "#333"),
                 linewidth=2.0 if method in ["Spectral", "DM", "MDS"] else 1.0,
                 label=method, alpha=0.85)
    ax2.set_xlabel("Number of Laplacian modes", fontsize=11)
    ax2.set_ylabel("Cumulative spectral alignment", fontsize=11)
    ax2.set_title("B. Cumulative Alignment Build-up", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.85)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Comparison of predictors
    ax3 = axes[2]
    methods_c = sorted(set(alignment_results.keys()) & set(gf_scores.keys()))
    if len(methods_c) >= 4:
        # Compute three predictors for comparison
        align_scores = [alignment_results[m]["alignment_score"] for m in methods_c]
        gf_vals = [gf_scores[m] for m in methods_c]
        
        # Load eff_dim from Phase 1
        with open(RES / "deep_geometric_analysis.json", encoding="utf-8") as f:
            p1 = json.load(f)
        eff_dims = [p1["geometric_features"][m]["effective_dimensionality"]
                    for m in methods_c]
        
        rho_a, p_a = spearmanr(align_scores, gf_vals)
        rho_e, p_e = spearmanr(eff_dims, gf_vals)
        
        x = np.arange(len(methods_c))
        width = 0.35
        ax3.barh(x - width/2, 
                 [(a - min(align_scores)) / (max(align_scores) - min(align_scores) + 1e-10) 
                  for a in align_scores],
                 width, color="#E69F00", alpha=0.8, 
                 label=f"Spectral align (rho={rho_a:.3f})")
        ax3.barh(x + width/2,
                 [(e - min(eff_dims)) / (max(eff_dims) - min(eff_dims) + 1e-10) 
                  for e in eff_dims],
                 width, color="#0072B2", alpha=0.8,
                 label=f"Eff dimensionality (rho={rho_e:.3f})")
        ax3.set_yticks(x)
        ax3.set_yticklabels(methods_c, fontsize=9)
        ax3.set_xlabel("Normalised predictor value", fontsize=11)
        ax3.set_title("C. Predictor Comparison: Spectral Align vs Eff Dim",
                      fontsize=13, fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    fig.savefig(FIG / "Fig35_spectral_alignment_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig35_spectral_alignment_summary.png")


# ============================================================
# 4. Main
# ============================================================

def main():
    print("=" * 70)
    print("Phase 3: Network-Aware Spectral Alignment Analysis")
    print("=" * 70)
    
    # -- Load data --
    print("\n[1/5] Loading network and GO annotations...")
    G, nodes, go_map = load_curated_network()
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    print(f"  Network: {n} nodes, {G.number_of_edges()} edges")
    n_annotated = sum(1 for nd in nodes if nd in go_map)
    print(f"  GO-annotated: {n_annotated}/{n}")
    
    # -- Compute Laplacian eigenbasis --
    print("\n[2/5] Computing Laplacian eigenbasis (k=30)...")
    k = min(30, n)
    eigenvalues, eigenvectors, A, deg = compute_laplacian_eigenbasis(G, nodes, k=k)
    print(f"  Eigenvalues[0:5] = {eigenvalues[:5].tolist()}")
    print(f"  Spectral gap = {eigenvalues[1] - eigenvalues[0]:.6f}")
    
    # -- Identify functional frequency band --
    print("\n[3/5] Identifying functional frequency band...")
    func_band = compute_functional_frequency_band(eigenvectors, nodes, go_map, k=k)
    print(f"  Functional alignment per mode (first 10):")
    for j in range(min(10, len(func_band))):
        print(f"    Mode {j}: {func_band[j]:.4f}")
    
    # Find the functional band — which modes best separate GO modules?
    top_func_modes = np.argsort(func_band)[::-1][:5]
    print(f"  Top 5 functional modes: {top_func_modes.tolist()}")
    print(f"  Functional band energy in modes 0-2: {func_band[:3].sum():.4f}")
    print(f"  Functional band energy in modes 0-9: {func_band[:10].sum():.4f}")
    
    # -- Load embeddings and compute spectral decomposition --
    print("\n[4/5] Spectral decomposition of embeddings...")
    
    # Load G-F scores
    with open(RES / "deep_geometric_analysis.json", encoding="utf-8") as f:
        p1_data = json.load(f)
    gf_scores = {m: sf["gf_score"] for m, sf in p1_data.get("shape_features", {}).items()}
    
    all_profiles = {}
    all_alignments = {}
    all_coefficients = {}
    
    for method in ALL_METHODS:
        npy = EMB / f"{method}_153.npy"
        nodes_f = EMB / f"{method}_153_nodes.json"
        if not npy.exists():
            continue
        
        coords = np.load(npy)
        with open(nodes_f, encoding="utf-8") as f:
            emb_nodes = json.load(f)
        
        emb_node_idx = {n: i for i, n in enumerate(emb_nodes)}
        # Align
        common = sorted(set(nodes) & set(emb_nodes) & set(go_map.keys()))
        if len(common) < 10:
            continue
        
        emb_idx = [emb_node_idx[nd] for nd in common]
        net_idx = [node_idx[nd] for nd in common]
        
        Y = coords[emb_idx]
        Y = rescale_coordinates(Y.copy())
        
        # Use full eigenbasis for common nodes
        V = eigenvectors[net_idx, :k]
        
        C, energy, profile = decompose_embedding_in_eigenbasis(Y, V, k=k)
        align = compute_spectral_alignment_score(profile, func_band[:k], k=k)
        
        all_profiles[method] = profile
        all_alignments[method] = align
        all_coefficients[method] = C
        
        print(f"  {method:12s}: top3_energy={align['top3_energy']:.3f}, "
              f"alignment={align['alignment_score']:.4f}, "
              f"modes_90%={align['n_modes_90pct']}")
    
    # -- Cross-analysis --
    print("\n[5/5] Cross-analysis: Spectral alignment vs G-F Score...")
    methods_c = sorted(set(gf_scores.keys()) & set(all_alignments.keys()))
    if len(methods_c) >= 4:
        align_scores = [all_alignments[m]["alignment_score"] for m in methods_c]
        top3_align = [all_alignments[m]["top3_alignment"] for m in methods_c]
        top3_energy = [all_alignments[m]["top3_energy"] for m in methods_c]
        gf_vals = [gf_scores[m] for m in methods_c]
        
        rho_a, p_a = spearmanr(align_scores, gf_vals)
        rho_t3, p_t3 = spearmanr(top3_align, gf_vals)
        rho_e, p_e = spearmanr(top3_energy, gf_vals)
        
        # Load effective dimensionality for comparison
        eff_dims = [p1_data["geometric_features"][m]["effective_dimensionality"]
                    for m in methods_c]
        rho_ed, p_ed = spearmanr(eff_dims, gf_vals)
        
        print(f"\n  Predictor comparison (n={len(methods_c)} methods):")
        print(f"    Spectral alignment : rho={rho_a:+.3f} (p={p_a:.3f})")
        print(f"    Top-3 alignment    : rho={rho_t3:+.3f} (p={p_t3:.3f})")
        print(f"    Top-3 energy       : rho={rho_e:+.3f} (p={p_e:.3f})")
        print(f"    Effective dim      : rho={rho_ed:+.3f} (p={p_ed:.3f})  [Phase 1 baseline]")
        
        # Combined predictor: spectral alignment + eff_dim
        from scipy.optimize import minimize
        def combined_mse(w):
            w_a, w_e = w
            pred = w_a * np.array(align_scores) + w_e * np.array(eff_dims)
            return -spearmanr(pred, gf_vals)[0]
        
        res = minimize(combined_mse, [0.5, 0.5], bounds=[(0, 1), (0, 1)])
        w_a, w_e = res.x
        combined = w_a * np.array(align_scores) + w_e * np.array(eff_dims)
        rho_comb, p_comb = spearmanr(combined, gf_vals)
        print(f"    Combined (w_a={w_a:.2f}, w_e={w_e:.2f}): rho={rho_comb:+.3f} (p={p_comb:.3f})")
    else:
        rho_a, p_a = 0, 1
        rho_ed, p_ed = 0, 1
        rho_comb, p_comb = 0, 1
    
    # -- Generate figures --
    print("\n" + "=" * 70)
    print("Generating figures...")
    print("=" * 70)
    
    plot_spectral_decomposition(all_profiles, eigenvalues, func_band, gf_scores)
    plot_spectral_alignment_summary(all_alignments, gf_scores, func_band)
    
    # -- Save results --
    print("\nSaving results...")
    
    output = {
        "analysis": "Phase 3: Network-Aware Spectral Alignment",
        "version": "1.0",
        "network_spectrum": {
            "eigenvalues_first_15": eigenvalues[:15].tolist(),
            "functional_band_first_15": func_band[:15].tolist(),
            "top_functional_modes": top_func_modes.tolist(),
            "functional_band_energy_0_2": float(func_band[:3].sum()),
            "functional_band_energy_0_9": float(func_band[:10].sum()),
        },
        "spectral_profiles": {m: p.tolist() for m, p in all_profiles.items()},
        "alignment_results": {
            m: {
                "alignment_score": a["alignment_score"],
                "top3_alignment": a["top3_alignment"],
                "top3_energy": a["top3_energy"],
                "n_modes_90pct": a["n_modes_90pct"],
                "cumulative_alignment": a["cumulative_alignment"],
            }
            for m, a in all_alignments.items()
        },
        "predictor_comparison": {
            "spectral_alignment": {"rho": float(rho_a), "p": float(p_a)},
            "effective_dimensionality": {"rho": float(rho_ed), "p": float(p_ed)},
            "combined": {"rho": float(rho_comb), "p": float(p_comb)},
        } if len(methods_c) >= 4 else {},
    }
    
    output_path = RES / "spectral_alignment.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved {output_path}")
    
    # -- Summary --
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_alignments:
        ranked = sorted(all_alignments.items(), key=lambda x: x[1]["alignment_score"], reverse=True)
        print("\nSpectral alignment ranking:")
        for i, (m, a) in enumerate(ranked, 1):
            gf = gf_scores.get(m, float("nan"))
            print(f"  {i:2d}. {m:12s}  align={a['alignment_score']:.4f}  "
                  f"top3={a['top3_energy']:.3f}  modes_90%={a['n_modes_90pct']}  "
                  f"G-F={gf:.4f}")
    
    print("\n" + "=" * 70)
    print("Phase 3 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
