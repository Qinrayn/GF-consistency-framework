#!/usr/bin/env python3
"""
Phase 16: Metric Comparison & Statistical Rigor
=================================================

Systematic comparison of GF Score against traditional embedding evaluation
metrics (link prediction AUROC, kNN micro-F1). Demonstrates that GF Score
captures unique aspects of embedding quality not captured by existing metrics.

Three analyses:
  1. Pairwise metric correlations with bootstrap CIs
  2. Discordance analysis: what GF Score sees that others miss
  3. Permutation test for GF Score vs MRR correlation robustness
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import SEED, get_results_dir, get_figures_dir

RESULTS = get_results_dir()
FIGURES = get_figures_dir()
BANNER = "=" * 64


def load_all_metrics():
    """Load all metrics for all methods from existing result files."""
    # Metric comparison (curated network)
    with open(RESULTS / "metric_comparison.json", encoding="utf-8") as f:
        mc = json.load(f)

    # Function prediction MRR (full network)
    with open(RESULTS / "function_prediction.json", encoding="utf-8") as f:
        fp = json.load(f)

    # GF scores
    with open(RESULTS / "gf_scores.json", encoding="utf-8") as f:
        gf = json.load(f)

    # Merge into unified table
    methods = mc["methods"]
    data = {}
    for m in methods:
        entry = mc["per_method"].get(m, {})
        data[m] = {
            "gf_score": entry.get("gf_score", gf.get("scores", {}).get(m, None)),
            "link_pred_auc": entry.get("link_pred_auc", None),
            "knn_f1": entry.get("knn_micro_f1", None),
            "full_mrr": fp.get("mean_reciprocal_rank", {}).get(m, None),
        }

    return data, methods


def pairwise_correlations(data, methods):
    """Compute pairwise Spearman correlations between all metrics."""
    metrics = ["gf_score", "link_pred_auc", "knn_f1"]
    metric_labels = ["GF Score", "Link Pred AUROC", "kNN micro-F1"]

    results = {}
    rng = np.random.RandomState(SEED)

    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            m1, m2 = metrics[i], metrics[j]
            l1, l2 = metric_labels[i], metric_labels[j]

            # Get paired values
            pairs = [(data[m][m1], data[m][m2]) for m in methods
                     if data[m][m1] is not None and data[m][m2] is not None]

            if len(pairs) < 4:
                continue

            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])

            rho, p_val = stats.spearmanr(x, y)
            r, rp = stats.pearsonr(x, y)

            # Bootstrap CI
            boot_rhos = []
            for _ in range(10000):
                idx = rng.choice(len(x), size=len(x), replace=True)
                if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
                    continue
                br, _ = stats.spearmanr(x[idx], y[idx])
                boot_rhos.append(br)

            ci_lo = np.percentile(boot_rhos, 2.5) if boot_rhos else rho
            ci_hi = np.percentile(boot_rhos, 97.5) if boot_rhos else rho

            key = f"{l1} vs {l2}"
            results[key] = {
                "spearman_rho": round(rho, 4),
                "spearman_p": round(p_val, 6),
                "ci_95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
                "pearson_r": round(r, 4),
                "n": len(pairs),
            }

            print(f"  {key}: rho={rho:.3f}, p={p_val:.4f}, "
                  f"95%CI=[{ci_lo:.3f}, {ci_hi:.3f}], n={len(pairs)}")

    return results


def discordance_analysis(data, methods):
    """Identify methods where metrics disagree — these are most informative."""

    # Compute ranks for each metric
    valid_methods = [m for m in methods
                     if data[m]["gf_score"] is not None
                     and data[m]["link_pred_auc"] is not None
                     and data[m]["knn_f1"] is not None]

    gf_ranks = {m: r for r, m in enumerate(
        sorted(valid_methods, key=lambda m: data[m]["gf_score"]))}
    auc_ranks = {m: r for r, m in enumerate(
        sorted(valid_methods, key=lambda m: data[m]["link_pred_auc"]))}
    f1_ranks = {m: r for r, m in enumerate(
        sorted(valid_methods, key=lambda m: data[m]["knn_f1"]))}

    # Rank discrepancy between GF Score and other metrics
    discrepancies = {}
    for m in valid_methods:
        gf_vs_auc = abs(gf_ranks[m] - auc_ranks[m])
        gf_vs_f1 = abs(gf_ranks[m] - f1_ranks[m])
        discrepancies[m] = {
            "gf_rank": gf_ranks[m],
            "auc_rank": auc_ranks[m],
            "f1_rank": f1_ranks[m],
            "gf_vs_auc_disc": gf_vs_auc,
            "gf_vs_f1_disc": gf_vs_f1,
            "max_disc": max(gf_vs_auc, gf_vs_f1),
        }

    # Sort by max discrepancy
    sorted_methods = sorted(discrepancies.keys(),
                            key=lambda m: discrepancies[m]["max_disc"],
                            reverse=True)

    print(f"\n  Rank discrepancies (GF Score vs other metrics):")
    print(f"  {'Method':12s} {'GF rank':>8s} {'AUC rank':>8s} {'F1 rank':>8s} "
          f"{'Max disc':>8s}")
    for m in sorted_methods:
        d = discrepancies[m]
        print(f"  {m:12s} {d['gf_rank']:>8d} {d['auc_rank']:>8d} "
              f"{d['f1_rank']:>8d} {d['max_disc']:>8d}")

    return discrepancies


def permutation_test_gf_mrr(data, methods, n_permutations=10000):
    """Permutation test: is the GF Score vs full-network MRR correlation
    significant against random label permutations?

    This tests robustness beyond the small-n (n=5) Spearman test.
    """
    # Get methods that have both GF score and full MRR
    valid = [m for m in methods
             if data[m]["gf_score"] is not None
             and data[m]["full_mrr"] is not None]

    if len(valid) < 3:
        print(f"  Only {len(valid)} methods with both GF Score and MRR")
        return None

    x = np.array([data[m]["gf_score"] for m in valid])
    y = np.array([data[m]["full_mrr"] for m in valid])

    # Observed correlation
    obs_rho, obs_p = stats.spearmanr(x, y)

    # Permutation test
    rng = np.random.RandomState(SEED)
    perm_rhos = []
    n_extreme = 0

    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_rho, _ = stats.spearmanr(x, y_perm)
        perm_rhos.append(perm_rho)
        if abs(perm_rho) >= abs(obs_rho):
            n_extreme += 1

    perm_p = n_extreme / n_permutations

    print(f"\n  Permutation test (n={len(valid)} methods, "
          f"{n_permutations} permutations):")
    print(f"  Observed Spearman rho: {obs_rho:.4f}")
    print(f"  Parametric p-value:    {obs_p:.4f}")
    print(f"  Permutation p-value:   {perm_p:.4f}")
    print(f"  Mean |perm rho|:       {np.mean(np.abs(perm_rhos)):.4f}")
    print(f"  Max |perm rho|:        {np.max(np.abs(perm_rhos)):.4f}")

    return {
        "n_methods": len(valid),
        "methods": valid,
        "observed_rho": round(obs_rho, 4),
        "parametric_p": round(obs_p, 6),
        "permutation_p": round(perm_p, 6),
        "perm_rho_mean_abs": round(float(np.mean(np.abs(perm_rhos))), 4),
        "perm_rho_max_abs": round(float(np.max(np.abs(perm_rhos))), 4),
        "perm_rhos": [round(float(r), 4) for r in perm_rhos[:100]],  # sample
    }


def unique_variance_analysis(data, methods):
    """Show what fraction of MRR variance is uniquely explained by GF Score
    vs other metrics, using partial correlation."""

    # Use curated network metrics (11 methods) for unique variance
    valid_curated = [m for m in methods
                     if data[m]["gf_score"] is not None
                     and data[m]["link_pred_auc"] is not None
                     and data[m]["knn_f1"] is not None]

    if len(valid_curated) < 4:
        return None

    gf = np.array([data[m]["gf_score"] for m in valid_curated])
    auc = np.array([data[m]["link_pred_auc"] for m in valid_curated])
    f1 = np.array([data[m]["knn_f1"] for m in valid_curated])

    # Inter-metric correlations
    rho_gf_auc, _ = stats.spearmanr(gf, auc)
    rho_gf_f1, _ = stats.spearmanr(gf, f1)
    rho_auc_f1, _ = stats.spearmanr(auc, f1)

    # Variance explained (R²) by each metric of the others
    r2_gf_from_auc = rho_gf_auc ** 2
    r2_gf_from_f1 = rho_gf_f1 ** 2
    r2_auc_from_gf = rho_gf_auc ** 2
    r2_auc_from_f1 = rho_auc_f1 ** 2

    unique_gf = 1 - max(r2_gf_from_auc, r2_gf_from_f1)
    unique_auc = 1 - max(r2_auc_from_gf, r2_auc_from_f1)

    print(f"\n  Unique variance analysis (n={len(valid_curated)} methods):")
    print(f"  rho(GF, AUC) = {rho_gf_auc:.3f} -> shared R^2 = {rho_gf_auc**2:.3f}")
    print(f"  rho(GF, F1)  = {rho_gf_f1:.3f} -> shared R^2 = {rho_gf_f1**2:.3f}")
    print(f"  rho(AUC, F1) = {rho_auc_f1:.3f} -> shared R^2 = {rho_auc_f1**2:.3f}")
    print(f"  GF Score unique variance: {unique_gf:.1%}")
    print(f"  Link Pred AUC unique variance: {unique_auc:.1%}")

    return {
        "rho_gf_auc": round(rho_gf_auc, 4),
        "rho_gf_f1": round(rho_gf_f1, 4),
        "rho_auc_f1": round(rho_auc_f1, 4),
        "unique_variance_gf": round(unique_gf, 4),
        "unique_variance_auc": round(unique_auc, 4),
        "n_methods": len(valid_curated),
    }


# ============================================================
# Visualisation
# ============================================================

def plot_fig78_metric_comparison(data, methods, correlations, discrepancies,
                                 unique_var):
    """Fig78: Comprehensive metric comparison dashboard."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    valid = [m for m in methods
             if data[m]["gf_score"] is not None
             and data[m]["link_pred_auc"] is not None]

    # --- Panel A: GF Score vs Link Prediction AUROC ---
    ax_a = fig.add_subplot(gs[0, 0])
    gf = [data[m]["gf_score"] for m in valid]
    auc = [data[m]["link_pred_auc"] for m in valid]
    colors = [plt.cm.Set2(i / len(valid)) for i in range(len(valid))]

    ax_a.scatter(gf, auc, c=colors, s=120, zorder=5, edgecolors="white",
                 linewidth=1.5)
    for m, x, y in zip(valid, gf, auc):
        ax_a.annotate(m, (x, y), textcoords="offset points",
                      xytext=(5, 5), fontsize=8, fontweight="bold")

    # Regression line
    z = np.polyfit(gf, auc, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(gf) * 0.9, max(gf) * 1.1, 100)
    ax_a.plot(x_line, p(x_line), "--", color="#666", alpha=0.7)

    rho = correlations.get("GF Score vs Link Pred AUROC", {}).get("spearman_rho", 0)
    p_val = correlations.get("GF Score vs Link Pred AUROC", {}).get("spearman_p", 1)
    ax_a.text(0.05, 0.95, f"rho = {rho:.3f}\nP = {p_val:.3f}\nn = {len(valid)}",
              transform=ax_a.transAxes, va="top", fontsize=10,
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax_a.set_xlabel("GF Score", fontsize=12)
    ax_a.set_ylabel("Link Prediction AUROC", fontsize=12)
    ax_a.set_title("A: GF Score vs Link Prediction AUROC", fontsize=13,
                   fontweight="bold")
    ax_a.grid(True, alpha=0.3)

    # --- Panel B: GF Score vs kNN micro-F1 ---
    ax_b = fig.add_subplot(gs[0, 1])
    valid_f1 = [m for m in methods
                if data[m]["gf_score"] is not None
                and data[m]["knn_f1"] is not None]
    gf_f1 = [data[m]["gf_score"] for m in valid_f1]
    f1 = [data[m]["knn_f1"] for m in valid_f1]
    colors_f1 = [plt.cm.Set2(i / len(valid_f1)) for i in range(len(valid_f1))]

    ax_b.scatter(gf_f1, f1, c=colors_f1, s=120, zorder=5, edgecolors="white",
                 linewidth=1.5)
    for m, x, y in zip(valid_f1, gf_f1, f1):
        ax_b.annotate(m, (x, y), textcoords="offset points",
                      xytext=(5, 5), fontsize=8, fontweight="bold")

    z2 = np.polyfit(gf_f1, f1, 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(min(gf_f1) * 0.9, max(gf_f1) * 1.1, 100)
    ax_b.plot(x_line2, p2(x_line2), "--", color="#666", alpha=0.7)

    rho2 = correlations.get("GF Score vs kNN micro-F1", {}).get("spearman_rho", 0)
    p_val2 = correlations.get("GF Score vs kNN micro-F1", {}).get("spearman_p", 1)
    ax_b.text(0.05, 0.95, f"rho = {rho2:.3f}\nP = {p_val2:.3f}\nn = {len(valid_f1)}",
              transform=ax_b.transAxes, va="top", fontsize=10,
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax_b.set_xlabel("GF Score", fontsize=12)
    ax_b.set_ylabel("kNN micro-F1", fontsize=12)
    ax_b.set_title("B: GF Score vs kNN micro-F1", fontsize=13,
                   fontweight="bold")
    ax_b.grid(True, alpha=0.3)

    # --- Panel C: Rank discrepancy heatmap ---
    ax_c = fig.add_subplot(gs[1, 0])
    disc_methods = sorted(discrepancies.keys(),
                          key=lambda m: discrepancies[m]["max_disc"],
                          reverse=True)
    rank_data = np.array([
        [discrepancies[m]["gf_rank"],
         discrepancies[m]["auc_rank"],
         discrepancies[m]["f1_rank"]]
        for m in disc_methods
    ])

    im = ax_c.imshow(rank_data, aspect="auto", cmap="YlOrRd")
    ax_c.set_xticks([0, 1, 2])
    ax_c.set_xticklabels(["GF Score", "Link Pred\nAUROC", "kNN\nmicro-F1"],
                          fontsize=10)
    ax_c.set_yticks(range(len(disc_methods)))
    ax_c.set_yticklabels(disc_methods, fontsize=10)

    for i in range(len(disc_methods)):
        for j in range(3):
            ax_c.text(j, i, str(rank_data[i, j]),
                      ha="center", va="center", fontsize=9,
                      color="white" if rank_data[i, j] > 5 else "black")

    plt.colorbar(im, ax=ax_c, shrink=0.8, label="Rank (higher = better)")
    ax_c.set_title("C: Method Rank Comparison", fontsize=13, fontweight="bold")

    # --- Panel D: Unique variance Venn-style summary ---
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    if unique_var:
        rho_gf_auc = unique_var["rho_gf_auc"]
        rho_gf_f1 = unique_var["rho_gf_f1"]
        rho_auc_f1 = unique_var["rho_auc_f1"]

        summary_text = (
            f"Metric Correlation Matrix\n"
            f"{'':20s} {'GF Score':>12s} {'LP-AUROC':>12s} {'kNN-F1':>12s}\n"
            f"{'GF Score':20s} {'1.000':>12s} {rho_gf_auc:.3f}{'':>6s} "
            f"{rho_gf_f1:.3f}\n"
            f"{'LP-AUROC':20s} {rho_gf_auc:.3f}{'':>6s} {'1.000':>12s} "
            f"{rho_auc_f1:.3f}\n"
            f"{'kNN-F1':20s} {rho_gf_f1:.3f}{'':>7s} {rho_auc_f1:.3f}{'':>7s} "
            f"{'1.000':>12s}\n\n"
            f"Unique Variance (not shared with any other metric):\n"
            f"  GF Score:     {unique_var['unique_variance_gf']:.1%}\n"
            f"  LP-AUROC:     {unique_var['unique_variance_auc']:.1%}\n\n"
            f"n = {unique_var['n_methods']} methods (curated network)"
        )

        ax_d.text(0.1, 0.5, summary_text, transform=ax_d.transAxes,
                  fontsize=11, family="monospace", va="center",
                  bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow",
                            alpha=0.9))

    ax_d.set_title("D: Metric Independence Analysis", fontsize=13,
                   fontweight="bold")

    fig.suptitle("Phase 16: GF Score vs Traditional Metrics",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(FIGURES / "Fig78_metric_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig78_metric_comparison.png")


def plot_fig79_permutation_test(perm_result):
    """Fig79: Permutation test histogram."""
    if perm_result is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    perm_rhos = perm_result["perm_rhos"]
    obs_rho = perm_result["observed_rho"]

    ax.hist(perm_rhos, bins=30, alpha=0.7, color="#3182bd",
            edgecolor="white", label="Permuted correlations")
    ax.axvline(obs_rho, color="#e6550d", linewidth=3, linestyle="--",
               label=f"Observed rho = {obs_rho:.3f}")
    ax.axvline(-obs_rho, color="#e6550d", linewidth=1.5, linestyle=":",
               alpha=0.5)

    ax.set_xlabel("Spearman rho", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Permutation Test: GF Score vs Full-Network MRR",
                 fontsize=14, fontweight="bold")

    # Annotation
    text = (f"n = {perm_result['n_methods']} methods\n"
            f"Observed rho = {obs_rho:.3f}\n"
            f"Parametric P = {perm_result['parametric_p']:.4f}\n"
            f"Permutation P = {perm_result['permutation_p']:.4f}\n"
            f"({perm_result.get('n_permutations', 10000)} permutations)")
    ax.text(0.97, 0.95, text, transform=ax.transAxes,
            va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.savefig(FIGURES / "Fig79_permutation_test.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig79_permutation_test.png")


# ============================================================
# Main
# ============================================================

def run():
    print(BANNER)
    print("Phase 16: Metric Comparison & Statistical Rigor")
    print(BANNER)

    np.random.seed(SEED)

    # Load data
    print("\n[1/5] Loading metrics...")
    data, methods = load_all_metrics()
    print(f"  {len(methods)} methods loaded")
    for m in methods:
        d = data[m]
        print(f"  {m:12s}: GF={d.get('gf_score', '-')}, "
              f"AUC={d.get('link_pred_auc', '-')}, "
              f"F1={d.get('knn_f1', '-')}, "
              f"MRR={d.get('full_mrr', '-')}")

    # Pairwise correlations
    print("\n[2/5] Pairwise correlations with bootstrap CIs...")
    correlations = pairwise_correlations(data, methods)

    # Discordance analysis
    print("\n[3/5] Discordance analysis...")
    discrepancies = discordance_analysis(data, methods)

    # Permutation test
    print("\n[4/5] Permutation test...")
    perm_result = permutation_test_gf_mrr(data, methods)

    # Unique variance
    print("\n[5/5] Unique variance analysis...")
    unique_var = unique_variance_analysis(data, methods)

    # Figures
    print("\n  Generating figures...")
    plot_fig78_metric_comparison(data, methods, correlations,
                                  discrepancies, unique_var)
    plot_fig79_permutation_test(perm_result)

    # Save results
    output = {
        "description": "Phase 16: Metric Comparison & Statistical Rigor",
        "pairwise_correlations": correlations,
        "rank_discrepancies": {
            m: d for m, d in discrepancies.items()
        },
        "permutation_test": perm_result,
        "unique_variance": unique_var,
    }

    result_file = RESULTS / "metric_comparison_extended.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {result_file.name}")

    # Summary
    print(f"\n{BANNER}")
    print("PHASE 16 SUMMARY")
    print(BANNER)

    print(f"\n  GF Score correlations with traditional metrics:")
    for key, corr in correlations.items():
        print(f"    {key}: rho={corr['spearman_rho']:.3f}, "
              f"p={corr['spearman_p']:.4f}, "
              f"95%CI={corr['ci_95']}")

    if unique_var:
        print(f"\n  Unique variance:")
        print(f"    GF Score: {unique_var['unique_variance_gf']:.1%}")
        print(f"    Link Pred AUROC: {unique_var['unique_variance_auc']:.1%}")

    if perm_result:
        print(f"\n  Permutation test:")
        print(f"    Observed rho: {perm_result['observed_rho']:.4f}")
        print(f"    Permutation p: {perm_result['permutation_p']:.4f}")

    print(f"\n{BANNER}")
    print("Phase 16 complete.")
    print(BANNER)


if __name__ == "__main__":
    run()
