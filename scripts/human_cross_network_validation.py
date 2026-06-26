#!/usr/bin/env python3
"""
Phase 8: Cross-Network Validation & Bootstrap Confidence Intervals
==================================================================
Two extensions to strengthen Phase 7 conclusions:

(A) Validate the two/three-factor models on human PPI (out-of-sample).
    - 11 methods: two-factor model (spectral alignment + effective rank)
    - 6 methods: three-factor model (+ H1 max persistence)

(B) Bootstrap confidence intervals for Phase 7 partial correlations.
    - 10,000 resamples of the yeast 11-method feature matrix
    - 95% CI for single-factor Spearman correlations
    - 95% CI for partial correlations (controlling for spectral + eff_rank)

Generates:
    Fig46_human_cross_network_validation.png  (4 panels)
    Fig47_bootstrap_confidence_intervals.png   (3 panels)
    results/human_cross_network_validation.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    ALL_METHODS, SEED,
    get_data_dir, get_results_dir, get_figures_dir,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()

METHODS_11 = list(ALL_METHODS)  # 11 methods
METHODS_6_WITH_TDA = ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE"]

BANNER = "=" * 70


# ============================================================
# Helpers
# ============================================================

def compute_effective_rank(coords: np.ndarray) -> float:
    """Participation ratio of squared singular values."""
    if coords.ndim != 2 or coords.shape[0] < 2:
        return 1.0
    S = np.linalg.svd(coords, compute_uv=False)
    S_sq = S ** 2
    total = S_sq.sum()
    if total < 1e-12:
        return 1.0
    return float((total ** 2) / max((S_sq ** 2).sum(), 1e-10))


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {path}")


def partial_corr_spearman(x, y, z1, z2):
    """Spearman partial correlation of x,y controlling for z1,z2 via OLS residualization."""
    from scipy.stats import rankdata
    n = len(x)
    rx = rankdata(x)
    ry = rankdata(y)
    rz1 = rankdata(z1)
    rz2 = rankdata(z2)
    # Residualize rx on (rz1, rz2)
    A = np.column_stack([rz1, rz2, np.ones(n)])
    beta_x, _, _, _ = np.linalg.lstsq(A, rx, rcond=None)
    res_x = rx - A @ beta_x
    beta_y, _, _, _ = np.linalg.lstsq(A, ry, rcond=None)
    res_y = ry - A @ beta_y
    rho, p = spearmanr(res_x, res_y)
    return rho, p


# ============================================================
# Part A: Human Cross-Network Validation
# ============================================================

def load_human_gf_scores() -> dict:
    """Load standard G-F scores for all 11 methods on human PPI."""
    data = load_json(RESULTS / "human_gf_scores_extended.json")
    scores = data.get("scores", {})
    if not scores:
        # Fallback: parse ranking list [[method, score], ...]
        for entry in data.get("ranking", []):
            if isinstance(entry, list) and len(entry) == 2:
                scores[entry[0]] = entry[1]
    return scores


def load_human_spectral_alignment() -> dict:
    """Load spectral alignment scores for all 11 methods on human PPI."""
    data = load_json(RESULTS / "human_spectral_alignment.json")
    profiles = data.get("spectral_profiles", {})
    eff_dims = data.get("effective_dimensionality", {})
    alignment = {}
    for method in METHODS_11:
        p = profiles.get(method, {})
        if p:
            alignment[method] = {
                "spectral_alignment": p.get("alignment_score", 0),
                "effective_dim": eff_dims.get(method, 0),
                "top3_energy": p.get("top3_energy", 0),
            }
    return alignment


def load_human_embeddings_for_rank() -> dict:
    """Load human embeddings and compute effective rank for each method."""
    ranks = {}
    for method in METHODS_11:
        fname = f"human_{method.lower()}_embedding.json"
        fpath = DATA / fname
        if not fpath.exists():
            # Try alternative naming
            alt = method.lower().replace("-", "-")
            fpath = DATA / f"human_{alt}_embedding.json"
        if not fpath.exists():
            print(f"    WARNING: No human embedding for {method}")
            continue
        try:
            raw = load_json(fpath)
            if not raw:
                continue
            first_val = next(iter(raw.values()))
            if isinstance(first_val, dict):
                nodes = sorted(raw.keys())
                coords = np.array([[raw[n]["x"], raw[n]["y"]] for n in nodes])
            elif isinstance(first_val, list):
                nodes = sorted(raw.keys())
                coords = np.array([raw[n] for n in nodes])
            else:
                continue
            ranks[method] = compute_effective_rank(coords)
        except Exception as e:
            print(f"    WARNING: Failed to load {method} embedding: {e}")
    return ranks


def load_human_tda_features() -> dict:
    """Load human TDA features for the 6 available methods."""
    data = load_json(RESULTS / "human_topological_analysis.json")
    tda = {}
    for method in data.get("methods", []):
        mdata = data.get("results", {}).get(method, {})
        if mdata:
            h1_stats = mdata.get("persistence_stats", {}).get("H1", {})
            tda[method] = {
                "h1_max_persistence": h1_stats.get("max_persistence", 0),
                "h1_n_features": h1_stats.get("n_features", 0),
                "h1_mean_persistence": h1_stats.get("mean_persistence", 0),
                "topo_consistency": mdata.get("topo_consistency", 0),
                "topo_consistency_h1": mdata.get("topo_consistency_h1", 0),
            }
    return tda


def run_human_validation():
    """Part A: Validate two/three-factor models on human PPI."""
    print("\n[A] Human cross-network validation")
    print("-" * 50)

    # Load all data sources
    print("  Loading human G-F scores...")
    gf_scores = load_human_gf_scores()
    print(f"    Methods: {len(gf_scores)}")

    print("  Loading human spectral alignment...")
    spectral = load_human_spectral_alignment()
    print(f"    Methods: {len(spectral)}")

    print("  Computing human effective rank...")
    eff_ranks = load_human_embeddings_for_rank()
    print(f"    Methods: {len(eff_ranks)}")

    print("  Loading human TDA features...")
    tda = load_human_tda_features()
    print(f"    Methods: {len(tda)}")

    # ---- 11-method analysis: two-factor model ----
    common_11 = [m for m in METHODS_11
                 if m in gf_scores and m in spectral and m in eff_ranks]
    print(f"\n  11-method two-factor validation (n={len(common_11)}):")
    gf_11 = [gf_scores[m] for m in common_11]
    sa_11 = [spectral[m]["spectral_alignment"] for m in common_11]
    er_11 = [eff_ranks[m] for m in common_11]
    ed_11 = [spectral[m]["effective_dim"] for m in common_11]

    # Single-factor correlations
    rho_sa, p_sa = spearmanr(sa_11, gf_11)
    rho_er, p_er = spearmanr(er_11, gf_11)
    rho_ed, p_ed = spearmanr(ed_11, gf_11)

    print(f"    Spectral alignment vs GF:  rho={rho_sa:.3f} (p={p_sa:.3f})")
    print(f"    Effective rank vs GF:      rho={rho_er:.3f} (p={p_er:.3f})")
    print(f"    Effective dim vs GF:       rho={rho_ed:.3f} (p={p_ed:.3f})")

    # Two-factor model (equal weight)
    from scipy.stats import rankdata
    sa_rank = rankdata(sa_11)
    er_rank = rankdata(er_11)
    two_factor_11 = 0.5 * sa_rank + 0.5 * er_rank
    rho_2f, p_2f = spearmanr(two_factor_11, gf_11)
    print(f"    Two-factor (spec+eff_rank): rho={rho_2f:.3f} (p={p_2f:.3f})")

    # Two-factor model (spectral + eff_dim, original Phase 3 recipe)
    ed_rank = rankdata(ed_11)
    two_factor_ed = 0.5 * sa_rank + 0.5 * ed_rank
    rho_2f_ed, p_2f_ed = spearmanr(two_factor_ed, gf_11)
    print(f"    Two-factor (spec+eff_dim):  rho={rho_2f_ed:.3f} (p={p_2f_ed:.3f})")

    # ---- 6-method analysis: three-factor model ----
    common_6 = [m for m in METHODS_6_WITH_TDA
                if m in gf_scores and m in spectral and m in eff_ranks and m in tda]
    print(f"\n  6-method three-factor validation (n={len(common_6)}):")
    gf_6 = [gf_scores[m] for m in common_6]
    sa_6 = [spectral[m]["spectral_alignment"] for m in common_6]
    er_6 = [eff_ranks[m] for m in common_6]
    h1_6 = [tda[m]["h1_max_persistence"] for m in common_6]

    # Single-factor
    rho_sa6, p_sa6 = spearmanr(sa_6, gf_6)
    rho_er6, p_er6 = spearmanr(er_6, gf_6)
    rho_h16, p_h16 = spearmanr(h1_6, gf_6)
    print(f"    Spectral alignment vs GF:   rho={rho_sa6:.3f} (p={p_sa6:.3f})")
    print(f"    Effective rank vs GF:       rho={rho_er6:.3f} (p={p_er6:.3f})")
    print(f"    H1 max persistence vs GF:   rho={rho_h16:.3f} (p={p_h16:.3f})")

    # Two-factor
    sa6_r = rankdata(sa_6)
    er6_r = rankdata(er_6)
    two_factor_6 = 0.5 * sa6_r + 0.5 * er6_r
    rho_2f6, p_2f6 = spearmanr(two_factor_6, gf_6)
    print(f"    Two-factor (spec+eff_rank): rho={rho_2f6:.3f} (p={p_2f6:.3f})")

    # Three-factor
    h16_r = rankdata(h1_6)
    three_factor_6 = (1/3) * sa6_r + (1/3) * er6_r + (1/3) * h16_r
    rho_3f6, p_3f6 = spearmanr(three_factor_6, gf_6)
    print(f"    Three-factor (+h1_max):     rho={rho_3f6:.3f} (p={p_3f6:.3f})")

    # Per-method detail table
    print(f"\n  Per-method detail (11 methods):")
    print(f"    {'Method':<12} {'GF':>8} {'SpecAlign':>10} {'EffRank':>8} {'EffDim':>8}")
    for m in sorted(common_11, key=lambda x: gf_scores[x], reverse=True):
        print(f"    {m:<12} {gf_scores[m]:8.4f} {spectral[m]['spectral_alignment']:10.4f} "
              f"{eff_ranks[m]:8.3f} {spectral[m]['effective_dim']:8.3f}")

    # Comparison with yeast
    yeast_comparison = {
        "human_11": {
            "spectral_alignment_rho": round(rho_sa, 3),
            "effective_rank_rho": round(rho_er, 3),
            "effective_dim_rho": round(rho_ed, 3),
            "two_factor_spec_effrank_rho": round(rho_2f, 3),
            "two_factor_spec_effdim_rho": round(rho_2f_ed, 3),
        },
        "human_6": {
            "spectral_alignment_rho": round(rho_sa6, 3),
            "effective_rank_rho": round(rho_er6, 3),
            "h1_max_persistence_rho": round(rho_h16, 3),
            "two_factor_rho": round(rho_2f6, 3),
            "three_factor_rho": round(rho_3f6, 3),
        },
        "yeast_reference": {
            "spectral_alignment_rho": 0.609,
            "effective_rank_rho": 0.873,
            "two_factor_rho": 0.809,
            "three_factor_with_topo_rho": 0.909,
        },
    }

    # Per-method data for plotting
    per_method_11 = {}
    for m in common_11:
        per_method_11[m] = {
            "gf_score": gf_scores[m],
            "spectral_alignment": spectral[m]["spectral_alignment"],
            "effective_rank": eff_ranks[m],
            "effective_dim": spectral[m]["effective_dim"],
        }
    per_method_6 = {}
    for m in common_6:
        per_method_6[m] = {
            "gf_score": gf_scores[m],
            "spectral_alignment": spectral[m]["spectral_alignment"],
            "effective_rank": eff_ranks[m],
            "h1_max_persistence": tda[m]["h1_max_persistence"],
        }

    return {
        "comparison": yeast_comparison,
        "per_method_11": per_method_11,
        "per_method_6": per_method_6,
    }


# ============================================================
# Part B: Bootstrap CI for Phase 7
# ============================================================

def run_bootstrap_ci(n_boot: int = 10000):
    """Part B: Bootstrap CIs for Phase 7 single-factor and partial correlations."""
    print(f"\n[B] Bootstrap confidence intervals ({n_boot} resamples)")
    print("-" * 50)

    # Load Phase 7 feature matrix
    phase7 = load_json(RESULTS / "tda_geometry_bridge.json")
    fm = phase7["feature_matrix"]
    methods = phase7["methods"]
    n = len(methods)

    gf = np.array([fm[m]["gf_score"] for m in methods])
    features_to_test = {
        "topo_gf_score":      np.array([fm[m]["topo_gf_score"] for m in methods]),
        "effective_rank":     np.array([fm[m]["effective_rank"] for m in methods]),
        "h1_max_persistence": np.array([fm[m]["h1_max_persistence"] for m in methods]),
        "spectral_alignment": np.array([fm[m]["spectral_alignment"] for m in methods]),
        "h1_topological_complexity": np.array([fm[m]["h1_topological_complexity"] for m in methods]),
        "topo_consistency":   np.array([fm[m]["topo_consistency"] for m in methods]),
    }

    rng = np.random.default_rng(SEED)

    # ---- Single-factor bootstrap CIs ----
    print("  Single-factor Spearman bootstrap...")
    single_boot = {}
    for fname, fvals in features_to_test.items():
        rhos = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            if len(set(idx)) < 3:
                continue
            r, _ = spearmanr(fvals[idx], gf[idx])
            rhos.append(r)
        rhos = np.array(rhos)
        ci_lo = float(np.percentile(rhos, 2.5))
        ci_hi = float(np.percentile(rhos, 97.5))
        median_rho = float(np.median(rhos))
        # Original point estimate
        orig_rho, orig_p = spearmanr(fvals, gf)
        single_boot[fname] = {
            "rho": round(float(orig_rho), 3),
            "p": round(float(orig_p), 4),
            "boot_median": round(median_rho, 3),
            "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
            "significant": ci_lo > 0 or ci_hi < 0,
        }
        sig = "***" if single_boot[fname]["significant"] else "ns"
        print(f"    {fname:<30}: rho={orig_rho:+.3f}  95%CI [{ci_lo:+.3f}, {ci_hi:+.3f}]  {sig}")

    # ---- Partial correlation bootstrap CIs ----
    print("\n  Partial correlation bootstrap (controlling for spectral + eff_rank)...")
    sa = features_to_test["spectral_alignment"]
    er = features_to_test["effective_rank"]

    partial_targets = {
        "topo_gf_score": features_to_test["topo_gf_score"],
        "h1_max_persistence": features_to_test["h1_max_persistence"],
        "h1_topological_complexity": features_to_test["h1_topological_complexity"],
        "topo_consistency": features_to_test["topo_consistency"],
    }

    partial_boot = {}
    for fname, fvals in partial_targets.items():
        rhos = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            if len(set(idx)) < 5:
                continue
            try:
                r, _ = partial_corr_spearman(
                    fvals[idx], gf[idx], sa[idx], er[idx])
                if not np.isnan(r):
                    rhos.append(r)
            except Exception as e:
                import logging; logging.warning(f"Exception in {__name__}: {e}")
                continue
        rhos = np.array(rhos)
        ci_lo = float(np.percentile(rhos, 2.5))
        ci_hi = float(np.percentile(rhos, 97.5))
        median_rho = float(np.median(rhos))
        # Original
        orig_rho, orig_p = partial_corr_spearman(fvals, gf, sa, er)
        partial_boot[fname] = {
            "partial_rho": round(float(orig_rho), 3),
            "partial_p": round(float(orig_p), 4),
            "boot_median": round(median_rho, 3),
            "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
            "significant": ci_lo > 0 or ci_hi < 0,
        }
        sig = "***" if partial_boot[fname]["significant"] else "ns"
        print(f"    {fname:<30}: partial_rho={orig_rho:+.3f}  95%CI [{ci_lo:+.3f}, {ci_hi:+.3f}]  {sig}")

    return {
        "n_methods": n,
        "n_bootstrap": n_boot,
        "single_factor_ci": single_boot,
        "partial_correlation_ci": partial_boot,
    }


# ============================================================
# Figures
# ============================================================

def generate_figures(human_data: dict, boot_data: dict):
    """Generate Fig46 (human validation) and Fig47 (bootstrap CIs)."""
    print("\n  Generating figures...")

    pm11 = human_data["per_method_11"]
    pm6 = human_data["per_method_6"]
    comp = human_data["comparison"]

    # ---- Fig46: Human cross-network validation (4 panels) ----
    fig46, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Human G-F Score vs spectral alignment
    ax = axes[0, 0]
    methods_11 = sorted(pm11.keys(), key=lambda m: pm11[m]["gf_score"], reverse=True)
    gf_vals = [pm11[m]["gf_score"] for m in methods_11]
    sa_vals = [pm11[m]["spectral_alignment"] for m in methods_11]
    ax.scatter(sa_vals, gf_vals, s=80, c="#2171B5", edgecolors="k", linewidth=0.5, zorder=3)
    for m in methods_11:
        ax.annotate(m, (pm11[m]["spectral_alignment"], pm11[m]["gf_score"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    # Add trend line
    x = np.array(sa_vals)
    y = np.array(gf_vals)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.3, lw=1)
    rho_sa = comp["human_11"]["spectral_alignment_rho"]
    ax.set_title(f"A. Human: Spectral Alignment vs G-F\n(ρ={rho_sa:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Spectral Alignment")
    ax.set_ylabel("G-F Score")
    ax.grid(True, alpha=0.3)

    # Panel B: Human G-F Score vs effective rank
    ax = axes[0, 1]
    er_vals = [pm11[m]["effective_rank"] for m in methods_11]
    ax.scatter(er_vals, gf_vals, s=80, c="#E6550D", edgecolors="k", linewidth=0.5, zorder=3)
    for m in methods_11:
        ax.annotate(m, (pm11[m]["effective_rank"], pm11[m]["gf_score"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    x = np.array(er_vals)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.3, lw=1)
    rho_er = comp["human_11"]["effective_rank_rho"]
    ax.set_title(f"B. Human: Effective Rank vs G-F\n(ρ={rho_er:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Effective Rank")
    ax.set_ylabel("G-F Score")
    ax.grid(True, alpha=0.3)

    # Panel C: Yeast vs Human correlation comparison
    ax = axes[1, 0]
    features = ["Spectral\nAlignment", "Effective\nRank", "Two-Factor\nModel"]
    yeast_rhos = [0.609, 0.873, 0.809]
    human_rhos = [comp["human_11"]["spectral_alignment_rho"],
                  comp["human_11"]["effective_rank_rho"],
                  comp["human_11"]["two_factor_spec_effrank_rho"]]
    x_pos = np.arange(len(features))
    w = 0.35
    bars1 = ax.bar(x_pos - w/2, yeast_rhos, w, label="Yeast (n=11)", color="#3182BD", edgecolor="k", linewidth=0.5)
    bars2 = ax.bar(x_pos + w/2, human_rhos, w, label="Human (n=11)", color="#E6550D", edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Spearman ρ with G-F Score")
    ax.set_title("C. Yeast vs Human: Factor Correlations", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features, fontsize=9)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    # Panel D: 6-method three-factor comparison (human vs yeast)
    ax = axes[1, 1]
    models = ["Spectral\nAlone", "Eff Rank\nAlone", "H1 Max\nAlone",
              "Two-Factor", "Three-Factor"]
    yeast_6 = [0.609, 0.873, 0.764, 0.809, 0.909]  # reference (11-method)
    human_6 = [comp["human_6"]["spectral_alignment_rho"],
               comp["human_6"]["effective_rank_rho"],
               comp["human_6"]["h1_max_persistence_rho"],
               comp["human_6"]["two_factor_rho"],
               comp["human_6"]["three_factor_rho"]]
    x_pos = np.arange(len(models))
    bars1 = ax.bar(x_pos - w/2, yeast_6, w, label="Yeast (n=11)", color="#3182BD",
                   edgecolor="k", linewidth=0.5)
    bars2 = ax.bar(x_pos + w/2, human_6, w, label="Human (n=6)", color="#E6550D",
                   edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Spearman ρ with G-F Score")
    ax.set_title("D. Yeast vs Human: Model Comparison\n(6 methods with TDA)", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    fig46.tight_layout()
    fig46.savefig(FIGURES / "Fig46_human_cross_network_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig46)
    print(f"  Saved Fig46_human_cross_network_validation.png")

    # ---- Fig47: Bootstrap confidence intervals (3 panels) ----
    fig47, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Single-factor CIs (horizontal error bars)
    ax = axes[0]
    sf = boot_data["single_factor_ci"]
    feat_names = sorted(sf.keys(), key=lambda k: sf[k]["rho"], reverse=True)
    y_pos = np.arange(len(feat_names))
    colors = ["#2CA25F" if sf[k]["significant"] else "#969696" for k in feat_names]
    for i, k in enumerate(feat_names):
        ci = sf[k]["ci_95"]
        ax.plot([ci[0], ci[1]], [i, i], color=colors[i], lw=2.5, solid_capstyle="round")
        ax.plot(sf[k]["rho"], i, "o", color=colors[i], markersize=8, markeredgecolor="k",
                markeredgewidth=0.5, zorder=5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([k.replace("_", " ") for k in feat_names], fontsize=8)
    ax.set_xlabel("Spearman ρ (95% CI)")
    ax.set_title("A. Single-Factor Correlations with G-F Score\n(Bootstrap 95% CI, n=11)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel B: Partial correlation CIs
    ax = axes[1]
    pc = boot_data["partial_correlation_ci"]
    pfeat_names = sorted(pc.keys(), key=lambda k: pc[k]["partial_rho"], reverse=True)
    y_pos = np.arange(len(pfeat_names))
    colors_p = ["#2CA25F" if pc[k]["significant"] else "#969696" for k in pfeat_names]
    for i, k in enumerate(pfeat_names):
        ci = pc[k]["ci_95"]
        ax.plot([ci[0], ci[1]], [i, i], color=colors_p[i], lw=2.5, solid_capstyle="round")
        ax.plot(pc[k]["partial_rho"], i, "o", color=colors_p[i], markersize=8,
                markeredgecolor="k", markeredgewidth=0.5, zorder=5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([k.replace("_", " ") for k in pfeat_names], fontsize=9)
    ax.set_xlabel("Partial ρ (95% CI)")
    ax.set_title("B. Partial Correlations\n(Controlling for Spectral + Eff Rank)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel C: Summary table as text
    ax = axes[2]
    ax.axis("off")
    lines = ["Phase 7 Bootstrap Summary", "=" * 40, ""]
    lines.append("Single-factor (n=11, 10k resamples):")
    for k in sorted(sf.keys(), key=lambda x: sf[x]["rho"], reverse=True):
        ci = sf[k]["ci_95"]
        sig = "✓" if sf[k]["significant"] else "✗"
        lines.append(f"  {sig} {k:<28} ρ={sf[k]['rho']:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    lines.append("")
    lines.append("Partial correlations:")
    for k in sorted(pc.keys(), key=lambda x: pc[x]["partial_rho"], reverse=True):
        ci = pc[k]["ci_95"]
        sig = "✓" if pc[k]["significant"] else "✗"
        lines.append(f"  {sig} {k:<28} ρ={pc[k]['partial_rho']:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    lines.append("")
    lines.append("✓ = CI excludes 0 (significant)")
    lines.append("✗ = CI includes 0 (not significant)")
    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=8, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax.set_title("C. Bootstrap CI Summary", fontsize=10, fontweight="bold")

    fig47.tight_layout()
    fig47.savefig(FIGURES / "Fig47_bootstrap_confidence_intervals.png", dpi=300, bbox_inches="tight")
    plt.close(fig47)
    print(f"  Saved Fig47_bootstrap_confidence_intervals.png")


# ============================================================
# Main
# ============================================================

def main():
    print(BANNER)
    print("Phase 8: Cross-Network Validation & Bootstrap Confidence Intervals")
    print(BANNER)

    # Part A: Human validation
    human_data = run_human_validation()

    # Part B: Bootstrap CIs
    boot_data = run_bootstrap_ci(n_boot=10000)

    # Figures
    generate_figures(human_data, boot_data)

    # Save combined results
    results = {
        "analysis": "Phase 8: Cross-Network Validation & Bootstrap CIs",
        "version": "1.0",
        "human_validation": human_data["comparison"],
        "human_per_method_11": human_data["per_method_11"],
        "human_per_method_6": human_data["per_method_6"],
        "bootstrap_ci": boot_data,
    }
    save_json(results, RESULTS / "human_cross_network_validation.json")

    print(f"\n{BANNER}")
    print("Phase 8 complete.")
    print(BANNER)


if __name__ == "__main__":
    main()
