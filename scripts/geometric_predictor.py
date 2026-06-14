#!/usr/bin/env python3
"""
geometric_predictor.py — Phase 2: Geometric Predictability Model
==================================================================

Builds a cross-species geometric predictor for G-F Score and establishes
the spectral-graph-theoretic foundation for *why* geometric features predict.

Five analysis modules:
  1. Yeast geometric predictor  — linear model from 6 geometric features
  2. Human cross-species validation — predict human G-F from yeast-trained model
  3. Spectral theory — Laplacian spectrum & participation ratio of PPI network
  4. Collapse diagnostics — geometric signature of collapsed vs healthy methods
  5. Method clustering — hierarchical clustering on geometric fingerprints

Outputs:
  - results/geometric_predictor.json
  - figures/Fig30_cross_species_prediction.png
  - figures/Fig31_spectral_theory.png
  - figures/Fig32_collapse_diagnostics.png
  - figures/Fig33_method_clustering.png
"""

import sys
import json
import math
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# Path setup (portable via utils helpers)
# ============================================================
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from utils import (
    SEED, GF_R_MIN, GF_R_MAX, ALL_METHODS, CLASSICAL_METHODS,
    GNN_METHODS, ALL_CURATED_METHODS,
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
HUMAN_SUBSAMPLE = 2000  # match human G-F analysis subsample size


# ============================================================
# 1. Data loading
# ============================================================

def load_yeast_embeddings():
    """Load all 11 yeast embeddings (153-node, .npy format)."""
    embeddings = {}
    for method in ALL_METHODS:
        npy = EMB / f"{method}_153.npy"
        nodes_f = EMB / f"{method}_153_nodes.json"
        if npy.exists():
            coords = np.load(npy)
            with open(nodes_f, encoding="utf-8") as f:
                nodes = json.load(f)
            embeddings[method] = (coords, nodes)
    return embeddings


def load_human_embeddings():
    """Load all 11 human embeddings (.json {node: {x, y}} format)."""
    name_map = {
        "DM": "dm", "MDS": "mds", "Spectral": "spectral",
        "DeepWalk": "deepwalk", "Node2Vec": "node2vec", "VGAE": "vgae",
        "PCA": "pca", "VGAE-feat": "vgae-feat", "GraphSAGE": "graphsage",
        "GAT": "gat", "GIN": "gin",
    }
    embeddings = {}
    for method in ALL_METHODS:
        fname = f"human_{name_map[method]}_embedding.json"
        fpath = DATA / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        nodes = sorted(data.keys())
        coords = np.array([[data[n]["x"], data[n]["y"]] for n in nodes])
        embeddings[method] = (coords, nodes)
    return embeddings


def load_human_gf_scores():
    """Load human extended G-F scores."""
    with open(RES / "human_gf_scores_extended.json", encoding="utf-8") as f:
        data = json.load(f)
    # Format: {"scores": {"Method": score, ...}, ...}
    return dict(data.get("scores", {}))


def load_yeast_gf_scores():
    """Load yeast G-F scores from Phase 1 results."""
    with open(RES / "deep_geometric_analysis.json", encoding="utf-8") as f:
        data = json.load(f)
    scores = {}
    for method, sf in data.get("shape_features", {}).items():
        scores[method] = sf["gf_score"]
    return scores


def load_yeast_phase1_features():
    """Load geometric features from Phase 1."""
    with open(RES / "deep_geometric_analysis.json", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("geometric_features", {})


# ============================================================
# 2. Geometric feature computation (shared for yeast & human)
# ============================================================

def compute_geometric_features(coords, k=10):
    """Compute 6 geometric features from embedding coordinates."""
    n, d = coords.shape
    coords = rescale_coordinates(coords.copy())
    dist_matrix = squareform(pdist(coords, metric="euclidean"))
    dist_upper = dist_matrix[np.triu_indices(n, k=1)]
    
    # Distance statistics
    dist_mean = float(np.mean(dist_upper))
    dist_std = float(np.std(dist_upper))
    dist_cv = dist_std / max(dist_mean, 1e-10)
    
    # k-NN graph Laplacian spectrum
    k_actual = min(k, n - 1)
    W = np.zeros((n, n))
    for i in range(n):
        knn_idx = np.argsort(dist_matrix[i])[1:k_actual+1]
        for j in knn_idx:
            W[i, j] = 1.0; W[j, i] = 1.0
    D_deg = np.diag(W.sum(axis=1))
    L = D_deg - W
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-10)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))
    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0
    fiedler = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    
    # Spatial uniformity
    knn_dists = np.array([np.sort(dist_matrix[i])[k_actual] for i in range(n)])
    spatial_cv = float(np.std(knn_dists) / max(np.mean(knn_dists), 1e-10))
    
    # Effective dimensionality
    coords_c = coords - coords.mean(axis=0)
    cov = coords_c.T @ coords_c / n
    pca_eigs = np.maximum(np.linalg.eigvalsh(cov), 0)
    total = pca_eigs.sum()
    eff_dim = float((total**2) / max((pca_eigs**2).sum(), 1e-10)) if total > 0 else 0.0
    first_pc = float(pca_eigs.max() / total) if total > 0 else 0.0
    
    return {
        "dist_mean": dist_mean, "dist_std": dist_std, "dist_cv": dist_cv,
        "spectral_gap": spectral_gap, "fiedler_value": fiedler,
        "spatial_uniformity_cv": spatial_cv,
        "effective_dimensionality": eff_dim, "first_pc_energy": first_pc,
    }


# ============================================================
# 3. Geometric predictor
# ============================================================

FEATURE_KEYS = [
    "effective_dimensionality", "first_pc_energy", "dist_cv",
    "spectral_gap", "fiedler_value", "spatial_uniformity_cv",
]

def build_geometric_predictor(yeast_features, yeast_scores):
    """
    Build a geometric predictor for G-F Score.
    
    Strategy: Use leave-one-out cross-validation (LOOCV) to evaluate
    several candidate predictors, select the best by LOO-R².
    """
    methods = sorted(set(yeast_features.keys()) & set(yeast_scores.keys()))
    n = len(methods)
    
    # Feature matrix (n x p) and target vector (n,)
    X = np.array([[yeast_features[m][k] for k in FEATURE_KEYS] for m in methods])
    y = np.array([yeast_scores[m] for m in methods])
    
    # Normalize features to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Invert features where lower = better (for interpretability)
    X_norm[:, 1] = 1.0 - X_norm[:, 1]  # first_pc_energy
    X_norm[:, 2] = 1.0 - X_norm[:, 2]  # dist_cv
    X_norm[:, 5] = 1.0 - X_norm[:, 5]  # spatial_uniformity_cv
    
    # --- Candidate predictors ---
    candidates = {}
    
    # A) Single-feature: effective dimensionality only
    eff_dim_col = 0  # effective_dimensionality is first in FEATURE_KEYS
    x_ed = X_norm[:, eff_dim_col]
    slope, intercept = np.polyfit(x_ed, y, 1)
    pred_ed = slope * x_ed + intercept
    ss_res = np.sum((y - pred_ed)**2)
    ss_tot = np.sum((y - y.mean())**2)
    candidates["eff_dim_linear"] = {
        "predict_fn": lambda x_new, s=slope, i=intercept: s * x_new[:, eff_dim_col] + i,
        "r2": float(1 - ss_res / ss_tot),
        "slope": slope, "intercept": intercept,
        "description": "Linear model on effective dimensionality alone",
    }
    
    # B) Two-feature: eff_dim + first_pc_energy
    fp_col = 1
    X2 = X_norm[:, [eff_dim_col, fp_col]]
    X2_aug = np.column_stack([X2, np.ones(n)])
    beta2 = np.linalg.lstsq(X2_aug, y, rcond=None)[0]
    pred2 = X2_aug @ beta2
    ss_res2 = np.sum((y - pred2)**2)
    candidates["eff_dim_plus_first_pc"] = {
        "predict_fn": lambda x_new, b=beta2: np.column_stack([
            x_new[:, [eff_dim_col, fp_col]], np.ones(x_new.shape[0])
        ]) @ b,
        "r2": float(1 - ss_res2 / ss_tot),
        "coefficients": beta2.tolist(),
        "description": "Two-feature model (eff_dim + inverted first_pc_energy)",
    }
    
    # C) Weighted average of all features (equal weight, normalized)
    X_avg = X_norm.mean(axis=1)
    slope_avg, intercept_avg = np.polyfit(X_avg, y, 1)
    pred_avg = slope_avg * X_avg + intercept_avg
    ss_res_avg = np.sum((y - pred_avg)**2)
    candidates["weighted_average"] = {
        "predict_fn": lambda x_new, s=slope_avg, i=intercept_avg: (
            s * x_new.mean(axis=1) + i
        ),
        "r2": float(1 - ss_res_avg / ss_tot),
        "description": "Equal-weighted average of all 6 normalized features",
    }
    
    # D) Ridge regression on all features (LOO-CV to pick lambda)
    from scipy.optimize import minimize_scalar
    def loo_mse(lam):
        mse = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            X_tr = X_norm[mask]; y_tr = y[mask]
            X_te = X_norm[i:i+1]; y_te = y[i]
            X_tr_aug = np.column_stack([X_tr, np.ones(mask.sum())])
            X_te_aug = np.column_stack([X_te, np.ones(1)])
            # Ridge: (X^T X + λI)^{-1} X^T y
            XtX = X_tr_aug.T @ X_tr_aug + lam * np.eye(X_tr_aug.shape[1])
            Xty = X_tr_aug.T @ y_tr
            beta = np.linalg.solve(XtX, Xty)
            pred_i = X_te_aug @ beta
            mse += (y_te - pred_i)**2
        return mse / n
    
    result = minimize_scalar(loo_mse, bounds=(0.001, 100), method="bounded")
    best_lam = float(np.atleast_1d(result.x)[0])
    
    # Fit final model with best lambda on all data
    X_aug = np.column_stack([X_norm, np.ones(n)])
    XtX = X_aug.T @ X_aug + best_lam * np.eye(X_aug.shape[1])
    Xty = X_aug.T @ y
    beta_ridge = np.linalg.solve(XtX, Xty)
    pred_ridge = X_aug @ beta_ridge
    ss_res_ridge = np.sum((y - pred_ridge)**2)
    
    candidates["ridge_regression"] = {
        "predict_fn": lambda x_new, b=beta_ridge: np.column_stack([
            x_new, np.ones(x_new.shape[0])
        ]) @ b,
        "r2": float(1 - ss_res_ridge / ss_tot),
        "lambda": best_lam,
        "coefficients": {k: float(v) for k, v in zip(FEATURE_KEYS + ["intercept"], beta_ridge)},
        "description": f"Ridge regression (λ={best_lam:.3f}) on all 6 features",
    }
    
    # --- LOO-CV evaluation ---
    loo_results = {}
    for name, cand in candidates.items():
        loo_preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            X_tr = X_norm[mask]; y_tr = y[mask]
            
            if name == "eff_dim_linear":
                x_ed_tr = X_tr[:, eff_dim_col]
                s, ic = np.polyfit(x_ed_tr, y_tr, 1)
                loo_preds[i] = s * X_norm[i, eff_dim_col] + ic
            elif name == "eff_dim_plus_first_pc":
                X2_tr = np.column_stack([X_tr[:, [eff_dim_col, fp_col]], np.ones(mask.sum())])
                b = np.linalg.lstsq(X2_tr, y_tr, rcond=None)[0]
                loo_preds[i] = np.array([*X_norm[i, [eff_dim_col, fp_col]], 1]) @ b
            elif name == "weighted_average":
                x_avg_tr = X_tr.mean(axis=1)
                s, ic = np.polyfit(x_avg_tr, y_tr, 1)
                loo_preds[i] = s * X_norm[i].mean() + ic
            elif name == "ridge_regression":
                X_tr_aug = np.column_stack([X_tr, np.ones(mask.sum())])
                XtX_i = X_tr_aug.T @ X_tr_aug + best_lam * np.eye(X_tr_aug.shape[1])
                b = np.linalg.solve(XtX_i, X_tr_aug.T @ y_tr)
                loo_preds[i] = np.array([*X_norm[i], 1]) @ b
        
        ss_res_loo = np.sum((y - loo_preds)**2)
        loo_r2 = float(1 - ss_res_loo / ss_tot)
        loo_rho, loo_p = spearmanr(loo_preds, y)
        loo_results[name] = {
            "loo_r2": loo_r2,
            "loo_spearman": float(loo_rho),
            "loo_p_value": float(loo_p),
            "train_r2": cand["r2"],
            "predictions": {m: float(loo_preds[j]) for j, m in enumerate(methods)},
        }
    
    return candidates, loo_results, methods, X_norm, y, X_min, X_max, X_range


# ============================================================
# 4. Cross-species prediction
# ============================================================

def cross_species_predict(human_features, yeast_model, yeast_X_min, yeast_X_max,
                          yeast_X_range, human_scores):
    """Apply yeast-trained predictor to human geometric features."""
    methods = sorted(set(human_features.keys()) & set(human_scores.keys()))
    
    # Normalize human features using yeast normalization
    X_human = np.array([[human_features[m][k] for k in FEATURE_KEYS] for m in methods])
    X_human_norm = (X_human - yeast_X_min) / yeast_X_range
    X_human_norm = np.clip(X_human_norm, 0, 1)
    # Invert same features
    X_human_norm[:, 1] = 1.0 - X_human_norm[:, 1]
    X_human_norm[:, 2] = 1.0 - X_human_norm[:, 2]
    X_human_norm[:, 5] = 1.0 - X_human_norm[:, 5]
    
    y_human = np.array([human_scores[m] for m in methods])
    
    predictions = {}
    for model_name, model in yeast_model.items():
        pred = model["predict_fn"](X_human_norm)
        rho, p = spearmanr(pred, y_human)
        mse = float(np.mean((pred - y_human)**2))
        predictions[model_name] = {
            "method_predictions": {m: float(pred[i]) for i, m in enumerate(methods)},
            "actual_scores": {m: float(y_human[i]) for i, m in enumerate(methods)},
            "spearman_rho": float(rho),
            "p_value": float(p),
            "mse": mse,
        }
    
    return predictions, methods, X_human_norm, y_human


# ============================================================
# 5. Spectral theory analysis
# ============================================================

def compute_network_spectrum(G, nodes):
    """Compute Laplacian spectrum of the PPI network."""
    import networkx as nx
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    
    # Adjacency matrix
    A = np.zeros((n, n))
    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            A[i, j] = 1; A[j, i] = 1
    
    # Degree matrix
    D = np.diag(A.sum(axis=1))
    # Unnormalized Laplacian
    L = D - A
    # Normalized Laplacian
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-10))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))
    
    # Participation ratio: (sum λ)^2 / sum(λ^2)
    pos_eigs = eigenvalues[eigenvalues > 1e-8]
    pr = float((pos_eigs.sum()**2) / (pos_eigs**2).sum()) if len(pos_eigs) > 0 else 0
    
    # Spectral gap (λ_2 - λ_1, where λ_1 ≈ 0 for connected graph)
    spec_gap = float(eigenvalues[1] - eigenvalues[0]) if n > 1 else 0
    
    return {
        "eigenvalues": eigenvalues.tolist(),
        "n_nodes": n,
        "participation_ratio": pr,
        "spectral_gap": spec_gap,
        "fiedler_value": float(eigenvalues[1]) if n > 1 else 0,
        "n_eigs_below_01": int(np.sum(eigenvalues < 0.1)),
        "n_eigs_below_05": int(np.sum(eigenvalues < 0.5)),
    }


# ============================================================
# 6. Collapse diagnostics
# ============================================================

def analyze_collapse(geo_features, gf_scores):
    """Identify geometric signatures of collapsed vs healthy methods."""
    methods = sorted(geo_features.keys())
    
    # Classify: collapsed (eff_dim < 1.2) vs healthy (eff_dim >= 1.5)
    collapsed = [m for m in methods if geo_features[m]["effective_dimensionality"] < 1.2]
    healthy = [m for m in methods if geo_features[m]["effective_dimensionality"] >= 1.5]
    borderline = [m for m in methods if m not in collapsed and m not in healthy]
    
    diagnostics = {
        "collapsed": collapsed,
        "healthy": healthy,
        "borderline": borderline,
        "group_statistics": {},
    }
    
    for group_name, group in [("collapsed", collapsed), ("healthy", healthy), ("borderline", borderline)]:
        if not group:
            continue
        stats = {}
        for feat in FEATURE_KEYS:
            vals = [geo_features[m][feat] for m in group]
            stats[feat] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        if all(m in gf_scores for m in group):
            scores = [gf_scores[m] for m in group]
            stats["gf_score"] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
        diagnostics["group_statistics"][group_name] = stats
    
    return diagnostics


# ============================================================
# 7. Method clustering
# ============================================================

def cluster_methods(geo_features):
    """Hierarchical clustering of methods based on geometric fingerprint."""
    methods = sorted(geo_features.keys())
    X = np.array([[geo_features[m][k] for k in FEATURE_KEYS] for m in methods])
    
    # Normalize
    X_min = X.min(axis=0); X_max = X.max(axis=0)
    X_range = X_max - X_min; X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Ward's method clustering
    Z = linkage(X_norm, method="ward")
    
    return Z, methods, X_norm


# ============================================================
# 8. Visualization
# ============================================================

def plot_cross_species(cross_preds, human_scores, yeast_scores, yeast_features,
                       human_features, methods_human):
    """Fig 30: Cross-species geometric prediction."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Panel A: Yeast training fit (eff_dim vs G-F Score)
    ax = axes[0, 0]
    for method in sorted(yeast_features.keys()):
        if method not in yeast_scores:
            continue
        ed = yeast_features[method]["effective_dimensionality"]
        gf = yeast_scores[method]
        ax.scatter(ed, gf, color=METHOD_COLORS.get(method, "#333"),
                   s=100, zorder=5, edgecolors="white", linewidth=1.0)
        ax.annotate(method, (ed, gf), fontsize=8, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")
    
    # Fit line
    methods_y = sorted(set(yeast_features.keys()) & set(yeast_scores.keys()))
    ed_vals = np.array([yeast_features[m]["effective_dimensionality"] for m in methods_y])
    gf_vals = np.array([yeast_scores[m] for m in methods_y])
    x_fit = np.linspace(0.8, 2.2, 100)
    slope, intercept = np.polyfit(ed_vals, gf_vals, 1)
    ax.plot(x_fit, slope * x_fit + intercept, color="grey", linestyle="--",
            linewidth=1.5, alpha=0.7, label=f"Linear fit (R² = {1 - np.sum((gf_vals - (slope*ed_vals+intercept))**2) / np.sum((gf_vals - gf_vals.mean())**2):.3f})")
    
    # Overlay human data (different markers)
    for method in methods_human:
        if method not in human_features or method not in human_scores:
            continue
        ed = human_features[method]["effective_dimensionality"]
        gf = human_scores[method]
        ax.scatter(ed, gf, color=METHOD_COLORS.get(method, "#333"),
                   s=100, marker="^", zorder=5, edgecolors="white", linewidth=1.0)
        ax.annotate(f"{method}*", (ed, gf), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points", fontstyle="italic")
    
    ax.set_xlabel("Effective dimensionality", fontsize=12)
    ax.set_ylabel("G-F Score", fontsize=12)
    ax.set_title("A. Effective Dimensionality vs G-F Score\n(circles=yeast, triangles=human)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Cross-species prediction (predicted vs actual for human)
    ax2 = axes[0, 1]
    best_model = "eff_dim_linear"  # use simplest model
    preds = cross_preds[best_model]
    for method in sorted(preds["method_predictions"].keys()):
        pred = preds["method_predictions"][method]
        actual = preds["actual_scores"][method]
        ax2.scatter(actual, pred, color=METHOD_COLORS.get(method, "#333"),
                    s=100, zorder=5, edgecolors="white", linewidth=1.0)
        ax2.annotate(method, (actual, pred), fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    
    lims = [0, max(max(preds["actual_scores"].values()), max(preds["method_predictions"].values())) * 1.1]
    ax2.plot(lims, lims, color="grey", linestyle="--", linewidth=1.5, alpha=0.7, label="Perfect prediction")
    rho = preds["spearman_rho"]
    ax2.set_xlabel("Actual human G-F Score", fontsize=12)
    ax2.set_ylabel("Predicted G-F Score (from yeast model)", fontsize=12)
    ax2.set_title(f"B. Cross-Species Prediction\n(Spearman ρ = {rho:.3f})", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Yeast rank vs Human rank
    ax3 = axes[1, 0]
    yeast_ranked = sorted(yeast_scores.items(), key=lambda x: x[1], reverse=True)
    human_ranked = sorted(human_scores.items(), key=lambda x: x[1], reverse=True)
    yeast_ranks = {m: i+1 for i, (m, _) in enumerate(yeast_ranked)}
    human_ranks = {m: i+1 for i, (m, _) in enumerate(human_ranked)}
    
    common = sorted(set(yeast_ranks.keys()) & set(human_ranks.keys()))
    for method in common:
        yr = yeast_ranks[method]
        hr = human_ranks[method]
        ax3.scatter(yr, hr, color=METHOD_COLORS.get(method, "#333"),
                    s=100, zorder=5, edgecolors="white", linewidth=1.0)
        ax3.annotate(method, (yr, hr), fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    
    ax3.plot([0, 12], [0, 12], color="grey", linestyle="--", linewidth=1.5, alpha=0.7)
    rho_rank, p_rank = spearmanr([yeast_ranks[m] for m in common],
                                  [human_ranks[m] for m in common])
    ax3.set_xlabel("Yeast G-F Rank", fontsize=12)
    ax3.set_ylabel("Human G-F Rank", fontsize=12)
    ax3.set_title(f"C. Cross-Species Rank Consistency\n(Spearman ρ = {rho_rank:.3f}, p = {p_rank:.3f})",
                  fontsize=12, fontweight="bold")
    ax3.set_xlim(0, 12); ax3.set_ylim(0, 12)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Geometric feature vs G-F Score comparison (yeast & human)
    ax4 = axes[1, 1]
    feature_names_short = ["Eff dim", "1st PC (inv)", "Dist CV (inv)",
                           "Spec gap", "Fiedler", "Uniformity (inv)"]
    
    yeast_corrs = []
    human_corrs = []
    methods_y = sorted(set(yeast_features.keys()) & set(yeast_scores.keys()))
    methods_h = sorted(set(human_features.keys()) & set(human_scores.keys()))
    
    gf_y = np.array([yeast_scores[m] for m in methods_y])
    gf_h = np.array([human_scores[m] for m in methods_h])
    
    for feat in FEATURE_KEYS:
        fy = np.array([yeast_features[m][feat] for m in methods_y])
        fh = np.array([human_features[m][feat] for m in methods_h])
        rho_y, _ = spearmanr(fy, gf_y)
        rho_h, _ = spearmanr(fh, gf_h)
        yeast_corrs.append(rho_y)
        human_corrs.append(rho_h)
    
    x_pos = np.arange(len(feature_names_short))
    width = 0.35
    ax4.barh(x_pos - width/2, yeast_corrs, width, color="#0072B2", alpha=0.8, label="Yeast")
    ax4.barh(x_pos + width/2, human_corrs, width, color="#E69F00", alpha=0.8, label="Human")
    ax4.set_yticks(x_pos)
    ax4.set_yticklabels(feature_names_short, fontsize=9)
    ax4.set_xlabel("Spearman ρ with G-F Score", fontsize=11)
    ax4.set_title("D. Feature-GF Correlation: Yeast vs Human", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="x")
    ax4.axvline(0, color="black", linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig(FIG / "Fig30_cross_species_prediction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig30_cross_species_prediction.png")


def plot_spectral_theory(spectrum_yeast, yeast_features):
    """Fig 31: Spectral theory visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    eigs = np.array(spectrum_yeast["eigenvalues"])
    
    # Panel A: Laplacian eigenvalue distribution
    ax = axes[0]
    ax.bar(range(len(eigs)), eigs, color="#0072B2", alpha=0.7, width=0.8)
    ax.axhline(0.1, color="red", linestyle="--", alpha=0.5, label=f"λ < 0.1: {spectrum_yeast['n_eigs_below_01']} modes")
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label=f"λ < 0.5: {spectrum_yeast['n_eigs_below_05']} modes")
    ax.set_xlabel("Eigenvalue index", fontsize=11)
    ax.set_ylabel("Normalised Laplacian eigenvalue", fontsize=11)
    ax.set_title("A. Laplacian Spectrum of Yeast PPI", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Cumulative spectral energy
    ax2 = axes[1]
    cumsum = np.cumsum(eigs) / eigs.sum()
    ax2.plot(range(len(eigs)), cumsum, color="#E69F00", linewidth=2)
    ax2.axhline(0.5, color="grey", linestyle="--", alpha=0.5)
    ax2.axhline(0.9, color="grey", linestyle=":", alpha=0.5)
    n_50 = np.searchsorted(cumsum, 0.5) + 1
    n_90 = np.searchsorted(cumsum, 0.9) + 1
    ax2.annotate(f"50% energy: {n_50} modes", xy=(n_50, 0.5),
                 xytext=(n_50+10, 0.45), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="grey"))
    ax2.annotate(f"90% energy: {n_90} modes", xy=(n_90, 0.9),
                 xytext=(n_90+5, 0.85), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="grey"))
    ax2.set_xlabel("Number of eigenvalues", fontsize=11)
    ax2.set_ylabel("Cumulative spectral energy", fontsize=11)
    ax2.set_title(f"B. Spectral Energy Distribution\n(Participation ratio = {spectrum_yeast['participation_ratio']:.1f})",
                  fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Network spectral PR vs embedding effective dimensionality
    ax3 = axes[2]
    pr_net = spectrum_yeast["participation_ratio"]
    for method in sorted(yeast_features.keys()):
        ed = yeast_features[method]["effective_dimensionality"]
        ax3.scatter(pr_net, ed, color=METHOD_COLORS.get(method, "#333"),
                    s=100, zorder=5, edgecolors="white", linewidth=1.0)
        ax3.annotate(method, (pr_net, ed), fontsize=8, ha="left", va="center",
                     xytext=(8, 0), textcoords="offset points")
    
    ax3.axvline(pr_net, color="red", linestyle="--", alpha=0.5,
                label=f"Network PR = {pr_net:.1f}")
    ax3.set_xlabel("Network spectral participation ratio", fontsize=11)
    ax3.set_ylabel("Embedding effective dimensionality", fontsize=11)
    ax3.set_title("C. Network Spectrum vs Embedding Geometry", fontsize=13, fontweight="bold")
    ax3.set_xlim(pr_net - 5, pr_net + 5)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIG / "Fig31_spectral_theory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig31_spectral_theory.png")


def plot_collapse_diagnostics(collapse_data, yeast_features, yeast_scores):
    """Fig 32: Collapse diagnostics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel A: Effective dimensionality bar chart (sorted)
    ax = axes[0]
    methods_sorted = sorted(yeast_features.keys(),
                            key=lambda m: yeast_features[m]["effective_dimensionality"],
                            reverse=True)
    colors = []
    for m in methods_sorted:
        ed = yeast_features[m]["effective_dimensionality"]
        if ed < 1.2:
            colors.append("#D55E00")   # collapsed: vermillion
        elif ed < 1.5:
            colors.append("#F0E442")   # borderline: yellow
        else:
            colors.append("#009E73")   # healthy: green
    
    bars = ax.barh(methods_sorted,
                   [yeast_features[m]["effective_dimensionality"] for m in methods_sorted],
                   color=colors, alpha=0.8, edgecolor="white")
    ax.axvline(1.2, color="red", linestyle="--", alpha=0.5, label="Collapse threshold")
    ax.axvline(1.5, color="green", linestyle="--", alpha=0.5, label="Healthy threshold")
    ax.set_xlabel("Effective dimensionality", fontsize=11)
    ax.set_title("A. Embedding Dimensionality Health", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Panel B: First PC energy vs G-F Score (collapse trajectory)
    ax2 = axes[1]
    for method in sorted(yeast_features.keys()):
        if method not in yeast_scores:
            continue
        fp = yeast_features[method]["first_pc_energy"]
        gf = yeast_scores[method]
        ax2.scatter(fp, gf, color=METHOD_COLORS.get(method, "#333"),
                    s=100, zorder=5, edgecolors="white", linewidth=1.0)
        ax2.annotate(method, (fp, gf), fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    
    ax2.axvline(0.95, color="red", linestyle="--", alpha=0.5, label="Collapse line (PC1 > 95%)")
    ax2.set_xlabel("First principal component energy", fontsize=11)
    ax2.set_ylabel("G-F Score", fontsize=11)
    ax2.set_title("B. Collapse Signature: PC1 Energy vs G-F Score", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Group comparison (collapsed vs healthy)
    ax3 = axes[2]
    group_data = collapse_data["group_statistics"]
    groups = ["healthy", "borderline", "collapsed"]
    group_labels = ["Healthy\n(eff_dim ≥ 1.5)", "Borderline\n(1.2-1.5)", "Collapsed\n(eff_dim < 1.2)"]
    group_colors = ["#009E73", "#F0E442", "#D55E00"]
    
    gf_means = []
    gf_stds = []
    for g in groups:
        if g in group_data and "gf_score" in group_data[g]:
            gf_means.append(group_data[g]["gf_score"]["mean"])
            gf_stds.append(group_data[g]["gf_score"]["std"])
        else:
            gf_means.append(0); gf_stds.append(0)
    
    bars = ax3.bar(group_labels, gf_means, yerr=gf_stds,
                   color=group_colors, alpha=0.8, edgecolor="white",
                   capsize=5, error_kw={"linewidth": 1.5})
    ax3.set_ylabel("Mean G-F Score", fontsize=11)
    ax3.set_title("C. G-F Score by Collapse Status", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    fig.savefig(FIG / "Fig32_collapse_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig32_collapse_diagnostics.png")


def plot_method_clustering(Z, methods, X_norm):
    """Fig 33: Method clustering dendrogram + heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1.5]})
    
    # Panel A: Dendrogram
    from scipy.cluster.hierarchy import dendrogram
    ax = axes[0]
    dn = dendrogram(Z, labels=methods, ax=ax, orientation="left",
                    leaf_font_size=10, color_threshold=0.5,
                    above_threshold_color="grey")
    ax.set_title("A. Method Clustering (Ward's linkage)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Ward distance", fontsize=11)
    
    # Panel B: Feature heatmap
    ax2 = axes[1]
    feature_labels = ["Eff dim", "1st PC\n(inv)", "Dist CV\n(inv)",
                      "Spec gap", "Fiedler", "Uniformity\n(inv)"]
    
    # Reorder methods by dendrogram
    dendro_order = dn["leaves"]
    X_ordered = X_norm[dendro_order]
    methods_ordered = [methods[i] for i in dendro_order]
    
    im = ax2.imshow(X_ordered, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
    ax2.set_yticks(range(len(methods_ordered)))
    ax2.set_yticklabels(methods_ordered, fontsize=10)
    ax2.set_xticks(range(len(feature_labels)))
    ax2.set_xticklabels(feature_labels, fontsize=9)
    ax2.set_title("B. Geometric Fingerprint Heatmap", fontsize=13, fontweight="bold")
    
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("Normalised feature value", fontsize=10)
    
    plt.tight_layout()
    fig.savefig(FIG / "Fig33_method_clustering.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig33_method_clustering.png")


# ============================================================
# 9. Main orchestration
# ============================================================

def main():
    print("=" * 70)
    print("Phase 2: Geometric Predictability Model")
    print("=" * 70)
    
    # -- Load data --
    print("\n[1/7] Loading data...")
    G, nodes, go_map = load_curated_network()
    yeast_emb = load_yeast_embeddings()
    human_emb = load_human_embeddings()
    yeast_scores = load_yeast_gf_scores()
    human_scores = load_human_gf_scores()
    yeast_features = load_yeast_phase1_features()
    print(f"  Yeast: {len(yeast_emb)} methods, {len(nodes)} nodes")
    print(f"  Human: {len(human_emb)} methods, ~15882 nodes")
    print(f"  Yeast G-F scores: {len(yeast_scores)} methods")
    print(f"  Human G-F scores: {len(human_scores)} methods")
    
    # -- Compute human geometric features (subsampled) --
    print("\n[2/7] Computing human geometric features (subsampled to 2000 nodes)...")
    np.random.seed(SEED)
    human_features = {}
    for method in ALL_METHODS:
        if method not in human_emb:
            continue
        coords, emb_nodes = human_emb[method]
        emb_node_idx = {n: i for i, n in enumerate(emb_nodes)}
        
        # Filter to nodes with GO annotations
        go_nodes = set(human_scores.keys())  # we just need the node set
        # Load GO map for filtering
        with open(DATA / "human_go_annotations.json", encoding="utf-8") as f:
            human_go = json.load(f)
        go_set = set(human_go.keys())
        common = sorted(set(emb_nodes) & go_set)
        
        if len(common) < HUMAN_SUBSAMPLE:
            # Use all available
            sample_nodes = common
        else:
            sample_nodes = sorted(np.random.choice(common, HUMAN_SUBSAMPLE, replace=False))
        
        idx = [emb_node_idx[n] for n in sample_nodes]
        coords_sub = coords[idx]
        
        features = compute_geometric_features(coords_sub)
        human_features[method] = features
        print(f"  {method}: eff_dim={features['effective_dimensionality']:.3f}, "
              f"first_pc={features['first_pc_energy']:.3f}, "
              f"dist_cv={features['dist_cv']:.3f} ({len(sample_nodes)} nodes)")
    
    # -- Build geometric predictor on yeast --
    print("\n[3/7] Building geometric predictor (yeast)...")
    yeast_model, loo_results, yeast_methods, X_norm, y, X_min, X_max, X_range = \
        build_geometric_predictor(yeast_features, yeast_scores)
    
    print("\n  LOO-CV Results:")
    for name, res in sorted(loo_results.items(), key=lambda x: x[1]["loo_r2"], reverse=True):
        print(f"    {name:30s}: LOO-R2={res['loo_r2']:+.3f}, "
              f"LOO-rho={res['loo_spearman']:+.3f} (p={res['loo_p_value']:.3f}), "
              f"Train-R2={res['train_r2']:.3f}")
    
    # -- Cross-species prediction --
    print("\n[4/7] Cross-species prediction (yeast -> human)...")
    cross_preds, methods_human, X_human, y_human = cross_species_predict(
        human_features, yeast_model, X_min, X_max, X_range, human_scores
    )
    
    for model_name, res in cross_preds.items():
        print(f"  {model_name:30s}: rho={res['spearman_rho']:+.3f} (p={res['p_value']:.3f}), "
              f"MSE={res['mse']:.4f}")
    
    # -- Spectral theory --
    print("\n[5/7] Computing network spectral properties...")
    spectrum = compute_network_spectrum(G, nodes)
    print(f"  Network PR (participation ratio): {spectrum['participation_ratio']:.2f}")
    print(f"  Spectral gap: {spectrum['spectral_gap']:.6f}")
    print(f"  Fiedler value: {spectrum['fiedler_value']:.6f}")
    print(f"  Eigenvalues < 0.1: {spectrum['n_eigs_below_01']}")
    print(f"  Eigenvalues < 0.5: {spectrum['n_eigs_below_05']}")
    
    # -- Collapse diagnostics --
    print("\n[6/7] Collapse diagnostics...")
    collapse = analyze_collapse(yeast_features, yeast_scores)
    print(f"  Healthy ({len(collapse['healthy'])}): {collapse['healthy']}")
    print(f"  Borderline ({len(collapse['borderline'])}): {collapse['borderline']}")
    print(f"  Collapsed ({len(collapse['collapsed'])}): {collapse['collapsed']}")
    for group, stats in collapse["group_statistics"].items():
        if "gf_score" in stats:
            print(f"    {group:12s}: mean G-F = {stats['gf_score']['mean']:.3f} ± {stats['gf_score']['std']:.3f}")
    
    # -- Method clustering --
    print("\n[7/7] Method clustering...")
    Z, cluster_methods_list, X_cluster = cluster_methods(yeast_features)
    
    # -- Generate figures --
    print("\n" + "=" * 70)
    print("Generating figures...")
    print("=" * 70)
    
    plot_cross_species(cross_preds, human_scores, yeast_scores,
                       yeast_features, human_features, methods_human)
    plot_spectral_theory(spectrum, yeast_features)
    plot_collapse_diagnostics(collapse, yeast_features, yeast_scores)
    plot_method_clustering(Z, cluster_methods_list, X_cluster)
    
    # -- Save results --
    print("\nSaving results...")
    
    # Clean up for JSON serialization
    output = {
        "analysis": "Phase 2: Geometric Predictability Model",
        "version": "1.0",
        "predictor_comparison": {
            name: {
                "loo_r2": res["loo_r2"],
                "loo_spearman": res["loo_spearman"],
                "loo_p_value": res["loo_p_value"],
                "train_r2": res["train_r2"],
                "predictions": res["predictions"],
            }
            for name, res in loo_results.items()
        },
        "cross_species_prediction": {
            name: {
                "spearman_rho": res["spearman_rho"],
                "p_value": res["p_value"],
                "mse": res["mse"],
                "method_predictions": res["method_predictions"],
                "actual_scores": res["actual_scores"],
            }
            for name, res in cross_preds.items()
        },
        "network_spectrum": {
            "participation_ratio": spectrum["participation_ratio"],
            "spectral_gap": spectrum["spectral_gap"],
            "fiedler_value": spectrum["fiedler_value"],
            "n_eigs_below_01": spectrum["n_eigs_below_01"],
            "n_eigs_below_05": spectrum["n_eigs_below_05"],
            "eigenvalues_first_20": spectrum["eigenvalues"][:20],
        },
        "collapse_diagnostics": {
            "classification": {
                "collapsed": collapse["collapsed"],
                "healthy": collapse["healthy"],
                "borderline": collapse["borderline"],
            },
            "group_statistics": collapse["group_statistics"],
        },
        "human_geometric_features": human_features,
        "yeast_geometric_features": yeast_features,
    }
    
    output_path = RES / "geometric_predictor.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved {output_path}")
    
    # -- Print summary --
    print("\n" + "=" * 70)
    print("SUMMARY: Key Findings")
    print("=" * 70)
    
    best_model = max(loo_results.items(), key=lambda x: x[1]["loo_r2"])
    print(f"\nBest yeast predictor: {best_model[0]}")
    print(f"  LOO-R2 = {best_model[1]['loo_r2']:.3f}, LOO-rho = {best_model[1]['loo_spearman']:.3f}")
    
    best_cross = max(cross_preds.items(), key=lambda x: abs(x[1]["spearman_rho"]))
    print(f"\nBest cross-species model: {best_cross[0]}")
    print(f"  Human Spearman rho = {best_cross[1]['spearman_rho']:.3f} (p = {best_cross[1]['p_value']:.3f})")
    
    print(f"\nNetwork spectral PR = {spectrum['participation_ratio']:.1f}")
    print(f"  Embedding effective dims range: "
          f"[{min(f['effective_dimensionality'] for f in yeast_features.values()):.2f}, "
          f"{max(f['effective_dimensionality'] for f in yeast_features.values()):.2f}]")
    
    print("\n" + "=" * 70)
    print("Phase 2 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
