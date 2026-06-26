#!/usr/bin/env python3
"""
Bootstrap 95% confidence intervals for key Spearman correlations
in the GF-consistency-framework project.

Computes bootstrap CIs (10,000 resamples, with replacement) for:
  1. H1 max persistence vs G-F Score (yeast, n=11)
  2. G-F Score vs Link Prediction AUC (n=11)
  3. G-F Score vs k-NN F1 (n=11)
  4. Topological consistency vs G-F Score (yeast, n=11)
"""
from __future__ import annotations

import json
import os
import numpy as np
from scipy import stats

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Load data files
with open(os.path.join(RESULTS_DIR, "topological_correlation_analysis.json"), encoding="utf-8") as f:
    topo_data = json.load(f)

with open(os.path.join(RESULTS_DIR, "metric_comparison.json"), encoding="utf-8") as f:
    metric_data = json.load(f)

# ---------------------------------------------------------------------------
# Extract per-method paired data
# ---------------------------------------------------------------------------

# Source 1: topological_correlation_analysis.json has all 11 methods with
# full-precision gf_score, h1_max_persistence, and topo_consistency.
topo_table = topo_data["correlation_table"]
methods_topo = [row["method"] for row in topo_table]
gf_scores_topo = np.array([row["gf_score"] for row in topo_table])
h1_max_persist = np.array([row["h1_max_persistence"] for row in topo_table])
topo_consistency = np.array([row["topo_consistency"] for row in topo_table])

# Source 2: metric_comparison.json has gf_score, link_pred_auc, knn_micro_f1
# Use the full-precision gf_score from topo_table (same methods, same order).
metric_per_method = metric_data["per_method"]
methods_metric = metric_data["methods"]  # preserves canonical order

gf_scores_metric = np.array([
    # Prefer full-precision from topo_data when available
    next(row["gf_score"] for row in topo_table if row["method"] == m)
    if any(row["method"] == m for row in topo_table)
    else metric_per_method[m]["gf_score"]
    for m in methods_metric
])
link_pred_auc = np.array([metric_per_method[m]["link_pred_auc"] for m in methods_metric])
knn_micro_f1 = np.array([metric_per_method[m]["knn_micro_f1"] for m in methods_metric])

# ---------------------------------------------------------------------------
# Bootstrap function
# ---------------------------------------------------------------------------
SEED = 42
N_BOOTSTRAP = 10_000


def bootstrap_spearman_ci(x, y, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Compute Spearman rho, p-value, and bootstrap 95% CI.

    Parameters
    ----------
    x, y : array-like, shape (n,)
    n_bootstrap : int
    seed : int

    Returns
    -------
    dict with rho, p_value, ci_lower, ci_upper, bootstrap_rhos
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    # Point estimate
    rho, p_value = stats.spearmanr(x, y)

    # Bootstrap resampling
    bootstrap_rhos = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Handle degenerate resamples (all identical ranks) gracefully
        try:
            r, _ = stats.spearmanr(x[idx], y[idx])
            if np.isnan(r):
                r = 0.0
        except Exception as e:
            r = 0.0
        bootstrap_rhos[i] = r

    ci_lower = float(np.percentile(bootstrap_rhos, 2.5))
    ci_upper = float(np.percentile(bootstrap_rhos, 97.5))

    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
        "n_bootstrap": n_bootstrap,
        "bootstrap_rhos": bootstrap_rhos.tolist(),
    }


# ---------------------------------------------------------------------------
# Run the four analyses
# ---------------------------------------------------------------------------
analyses = {}

# 1. H1 max persistence vs G-F Score
res1 = bootstrap_spearman_ci(h1_max_persist, gf_scores_topo)
analyses["h1_max_persistence_vs_gf_score"] = {
    "description": "H1 Max Persistence vs G-F Score (yeast, n=11 methods)",
    "methods": methods_topo,
    "rho": res1["rho"],
    "p_value": res1["p_value"],
    "ci_95": [res1["ci_lower"], res1["ci_upper"]],
    "n": res1["n"],
    "n_bootstrap": res1["n_bootstrap"],
}

# 2. G-F Score vs Link Prediction AUC
res2 = bootstrap_spearman_ci(gf_scores_metric, link_pred_auc)
analyses["gf_score_vs_link_pred_auc"] = {
    "description": "G-F Score vs Link Prediction AUC (n=11 methods)",
    "methods": methods_metric,
    "rho": res2["rho"],
    "p_value": res2["p_value"],
    "ci_95": [res2["ci_lower"], res2["ci_upper"]],
    "n": res2["n"],
    "n_bootstrap": res2["n_bootstrap"],
}

# 3. G-F Score vs k-NN F1
res3 = bootstrap_spearman_ci(gf_scores_metric, knn_micro_f1)
analyses["gf_score_vs_knn_f1"] = {
    "description": "G-F Score vs k-NN micro-F1 (n=11 methods)",
    "methods": methods_metric,
    "rho": res3["rho"],
    "p_value": res3["p_value"],
    "ci_95": [res3["ci_lower"], res3["ci_upper"]],
    "n": res3["n"],
    "n_bootstrap": res3["n_bootstrap"],
}

# 4. Topological consistency vs G-F Score
res4 = bootstrap_spearman_ci(topo_consistency, gf_scores_topo)
analyses["topo_consistency_vs_gf_score"] = {
    "description": "Topological Consistency vs G-F Score (yeast, n=11 methods)",
    "methods": methods_topo,
    "rho": res4["rho"],
    "p_value": res4["p_value"],
    "ci_95": [res4["ci_lower"], res4["ci_upper"]],
    "n": res4["n"],
    "n_bootstrap": res4["n_bootstrap"],
}

# ---------------------------------------------------------------------------
# Save results (exclude bulky bootstrap_rhos from JSON output)
# ---------------------------------------------------------------------------
output = {}
for key, val in analyses.items():
    output[key] = {k: v for k, v in val.items()}

output_path = os.path.join(RESULTS_DIR, "bootstrap_correlations.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

# ---------------------------------------------------------------------------
# Pretty-print for manuscript
# ---------------------------------------------------------------------------
print("=" * 80)
print("  BOOTSTRAP 95% CONFIDENCE INTERVALS FOR SPEARMAN CORRELATIONS")
print("  (10,000 resamples, with replacement, seed=42)")
print("=" * 80)
print()

for key, val in analyses.items():
    print(f"  {val['description']}")
    print(f"    Spearman rho = {val['rho']:.4f}")
    print(f"    P-value      = {val['p_value']:.4f}")
    print(f"    95% CI       = [{val['ci_95'][0]:.4f}, {val['ci_95'][1]:.4f}]")
    sig = "Yes" if val["p_value"] < 0.05 else "No"
    print(f"    Significant (P<0.05): {sig}")
    print()

print(f"Results saved to: {output_path}")
print("=" * 80)
