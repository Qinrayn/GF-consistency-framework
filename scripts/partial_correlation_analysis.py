#!/usr/bin/env python3
"""
partial_correlation_analysis.py -- Partial Correlation: GF Score vs Downstream Tasks
====================================================================================

Addresses reviewer concern (3.4): GF Score correlations with downstream
tasks may be confounded by shared dependence on effective rank or GNN
architecture.  Computes partial Spearman correlations controlling for:

  (1) Effective rank (continuous confounder)
  (2) GNN vs non-GNN indicator (binary confounder)
  (3) Both simultaneously

Partial Spearman is computed via rank-based partial correlation:
rank-transform all variables, then compute partial correlation of the
ranks using OLS residuals.

Outputs:
  - results/partial_correlation_analysis.json
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, rankdata

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from utils import get_results_dir

RES = get_results_dir()


def partial_spearman(x, y, Z):
    """
    Partial Spearman correlation between x and y controlling for Z.

    Parameters
    ----------
    x : array of length n
    y : array of length n
    Z : array of shape (n, k) -- k confounders

    Returns
    -------
    rho : partial Spearman correlation
    p_value : approximate p-value (t-distribution)
    """
    n = len(x)
    # Rank-transform all variables
    rx = rankdata(x)
    ry = rankdata(y)
    rZ = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])])

    # Add intercept column
    ones = np.ones((n, 1))
    rZ_aug = np.hstack([rZ, ones])

    # Residualise rx on rZ
    beta_x = np.linalg.lstsq(rZ_aug, rx, rcond=None)[0]
    res_x = rx - rZ_aug @ beta_x

    # Residualise ry on rZ
    beta_y = np.linalg.lstsq(rZ_aug, ry, rcond=None)[0]
    res_y = ry - rZ_aug @ beta_y

    # Spearman of residuals = partial Spearman
    rho, p_raw = spearmanr(res_x, res_y)

    # Adjusted p-value using t-distribution with n-k-2 df
    k = Z.shape[1]
    df = n - k - 2
    if abs(rho) >= 1.0:
        t_stat = np.inf if rho > 0 else -np.inf
    else:
        t_stat = rho * np.sqrt(df / (1 - rho**2))
    from scipy.stats import t as t_dist
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))

    return float(rho), float(p_value)


def main():
    print("=" * 70)
    print("Partial Correlation Analysis: GF Score vs Downstream Tasks")
    print("=" * 70)

    # Load data
    with open(RES / "metric_comparison.json", encoding="utf-8") as f:
        mc = json.load(f)

    with open(RES / "gat_collapse_formal_proof.json", encoding="utf-8") as f:
        gcf = json.load(f)

    with open(RES / "function_prediction_full.json", encoding="utf-8") as f:
        fp = json.load(f)

    methods_all = ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec",
                   "VGAE", "VGAE-feat", "PCA", "GraphSAGE", "GAT", "GIN"]

    # Extract per-method data
    data = {}
    for m in methods_all:
        pm = mc.get("per_method", {}).get(m, {})
        t2 = gcf.get("theorems", {}).get("T2_effective_rank_bound",
                                          {}).get("method_results", {}).get(m, {})
        mrr = fp.get("mean_reciprocal_rank", {}).get(m, None)

        gf = pm.get("gf_score")
        auc = pm.get("link_pred_auc")
        f1 = pm.get("knn_micro_f1")
        eff_rank = t2.get("effective_rank")

        if all(v is not None for v in [gf, auc, f1, eff_rank]):
            data[m] = {
                "gf_score": gf,
                "link_pred_auc": auc,
                "knn_f1": f1,
                "effective_rank": eff_rank,
                "mrr": mrr if mrr is not None else 0.0,
                "is_gnn": 1.0 if m in ["GraphSAGE", "GAT", "GIN", "VGAE", "VGAE-feat"] else 0.0,
            }

    methods = sorted(data.keys())
    n = len(methods)
    print(f"\n  Methods with complete data: {n}")
    for m in methods:
        d = data[m]
        print(f"    {m:12s}: GF={d['gf_score']:.4f}, AUC={d['link_pred_auc']:.4f}, "
              f"F1={d['knn_f1']:.4f}, eff_rank={d['effective_rank']:.4f}, "
              f"MRR={d['mrr']:.4f}, GNN={int(d['is_gnn'])}")

    # Arrays
    gf = np.array([data[m]["gf_score"] for m in methods])
    auc = np.array([data[m]["link_pred_auc"] for m in methods])
    f1 = np.array([data[m]["knn_f1"] for m in methods])
    eff_rank = np.array([data[m]["effective_rank"] for m in methods])
    is_gnn = np.array([data[m]["is_gnn"] for m in methods])
    mrr = np.array([data[m]["mrr"] for m in methods])

    # ---- Pairwise Spearman (baseline, no confounders) ----
    print("\n--- Pairwise Spearman (no confounders) ---")
    rho_gf_auc, p_gf_auc = spearmanr(gf, auc)
    rho_gf_f1, p_gf_f1 = spearmanr(gf, f1)
    rho_gf_mrr, p_gf_mrr = spearmanr(gf, mrr)
    rho_gf_er, p_gf_er = spearmanr(gf, eff_rank)
    rho_auc_er, p_auc_er = spearmanr(auc, eff_rank)
    rho_f1_er, p_f1_er = spearmanr(f1, eff_rank)
    rho_mrr_er, p_mrr_er = spearmanr(mrr, eff_rank)

    print(f"  GF vs AUC:      rho={rho_gf_auc:+.3f} (p={p_gf_auc:.4f})")
    print(f"  GF vs kNN F1:   rho={rho_gf_f1:+.3f} (p={p_gf_f1:.4f})")
    print(f"  GF vs MRR:      rho={rho_gf_mrr:+.3f} (p={p_gf_mrr:.4f})")
    print(f"  GF vs eff_rank: rho={rho_gf_er:+.3f} (p={p_gf_er:.4f})")
    print(f"  AUC vs eff_rank: rho={rho_auc_er:+.3f} (p={p_auc_er:.4f})")
    print(f"  F1 vs eff_rank:  rho={rho_f1_er:+.3f} (p={p_f1_er:.4f})")
    print(f"  MRR vs eff_rank: rho={rho_mrr_er:+.3f} (p={p_mrr_er:.4f})")

    # ---- Partial Spearman controlling for effective rank ----
    print("\n--- Partial Spearman (controlling for effective rank) ---")
    Z_er = eff_rank.reshape(-1, 1)

    pr_gf_auc_er, pp_gf_auc_er = partial_spearman(gf, auc, Z_er)
    pr_gf_f1_er, pp_gf_f1_er = partial_spearman(gf, f1, Z_er)
    pr_gf_mrr_er, pp_gf_mrr_er = partial_spearman(gf, mrr, Z_er)

    print(f"  GF vs AUC | eff_rank:      rho={pr_gf_auc_er:+.3f} (p={pp_gf_auc_er:.4f})")
    print(f"  GF vs kNN F1 | eff_rank:   rho={pr_gf_f1_er:+.3f} (p={pp_gf_f1_er:.4f})")
    print(f"  GF vs MRR | eff_rank:      rho={pr_gf_mrr_er:+.3f} (p={pp_gf_mrr_er:.4f})")

    # ---- Partial Spearman controlling for GNN indicator ----
    print("\n--- Partial Spearman (controlling for GNN indicator) ---")
    Z_gnn = is_gnn.reshape(-1, 1)

    pr_gf_auc_gnn, pp_gf_auc_gnn = partial_spearman(gf, auc, Z_gnn)
    pr_gf_f1_gnn, pp_gf_f1_gnn = partial_spearman(gf, f1, Z_gnn)
    pr_gf_mrr_gnn, pp_gf_mrr_gnn = partial_spearman(gf, mrr, Z_gnn)

    print(f"  GF vs AUC | GNN:      rho={pr_gf_auc_gnn:+.3f} (p={pp_gf_auc_gnn:.4f})")
    print(f"  GF vs kNN F1 | GNN:   rho={pr_gf_f1_gnn:+.3f} (p={pp_gf_f1_gnn:.4f})")
    print(f"  GF vs MRR | GNN:      rho={pr_gf_mrr_gnn:+.3f} (p={pp_gf_mrr_gnn:.4f})")

    # ---- Partial Spearman controlling for both ----
    print("\n--- Partial Spearman (controlling for eff_rank + GNN) ---")
    Z_both = np.column_stack([eff_rank, is_gnn])

    pr_gf_auc_both, pp_gf_auc_both = partial_spearman(gf, auc, Z_both)
    pr_gf_f1_both, pp_gf_f1_both = partial_spearman(gf, f1, Z_both)
    pr_gf_mrr_both, pp_gf_mrr_both = partial_spearman(gf, mrr, Z_both)

    print(f"  GF vs AUC | eff_rank+GNN:      rho={pr_gf_auc_both:+.3f} (p={pp_gf_auc_both:.4f})")
    print(f"  GF vs kNN F1 | eff_rank+GNN:   rho={pr_gf_f1_both:+.3f} (p={pp_gf_f1_both:.4f})")
    print(f"  GF vs MRR | eff_rank+GNN:      rho={pr_gf_mrr_both:+.3f} (p={pp_gf_mrr_both:.4f})")

    # ---- Interpretation ----
    print("\n--- Interpretation ---")
    for name, raw_rho, raw_p, pr_er, pp_er, pr_both, pp_both in [
        ("Link Pred AUC", rho_gf_auc, p_gf_auc, pr_gf_auc_er, pp_gf_auc_er,
         pr_gf_auc_both, pp_gf_auc_both),
        ("kNN F1", rho_gf_f1, p_gf_f1, pr_gf_f1_er, pp_gf_f1_er,
         pr_gf_f1_both, pp_gf_f1_both),
        ("MRR", rho_gf_mrr, p_gf_mrr, pr_gf_mrr_er, pp_gf_mrr_er,
         pr_gf_mrr_both, pp_gf_mrr_both),
    ]:
        attenuation_er = 1 - abs(pr_er) / max(abs(raw_rho), 1e-10)
        attenuation_both = 1 - abs(pr_both) / max(abs(raw_rho), 1e-10)
        print(f"  {name}: raw={raw_rho:+.3f}, |eff_rank={pr_er:+.3f} "
              f"(attenuation {attenuation_er:.0%}), "
              f"|both={pr_both:+.3f} (attenuation {attenuation_both:.0%})")

    # Save results
    output = {
        "analysis": "Partial Correlation: GF Score vs Downstream Tasks",
        "n_methods": n,
        "methods": methods,
        "per_method": {m: data[m] for m in methods},
        "pairwise_spearman": {
            "gf_vs_link_pred_auc": {"rho": rho_gf_auc, "p": p_gf_auc},
            "gf_vs_knn_f1": {"rho": rho_gf_f1, "p": p_gf_f1},
            "gf_vs_mrr": {"rho": rho_gf_mrr, "p": p_gf_mrr},
            "gf_vs_effective_rank": {"rho": rho_gf_er, "p": p_gf_er},
            "auc_vs_effective_rank": {"rho": rho_auc_er, "p": p_auc_er},
            "f1_vs_effective_rank": {"rho": rho_f1_er, "p": p_f1_er},
            "mrr_vs_effective_rank": {"rho": rho_mrr_er, "p": p_mrr_er},
        },
        "partial_correlations": {
            "controlling_effective_rank": {
                "gf_vs_link_pred_auc": {"rho": pr_gf_auc_er, "p": pp_gf_auc_er},
                "gf_vs_knn_f1": {"rho": pr_gf_f1_er, "p": pp_gf_f1_er},
                "gf_vs_mrr": {"rho": pr_gf_mrr_er, "p": pp_gf_mrr_er},
            },
            "controlling_gnn_indicator": {
                "gf_vs_link_pred_auc": {"rho": pr_gf_auc_gnn, "p": pp_gf_auc_gnn},
                "gf_vs_knn_f1": {"rho": pr_gf_f1_gnn, "p": pp_gf_f1_gnn},
                "gf_vs_mrr": {"rho": pr_gf_mrr_gnn, "p": pp_gf_mrr_gnn},
            },
            "controlling_eff_rank_and_gnn": {
                "gf_vs_link_pred_auc": {"rho": pr_gf_auc_both, "p": pp_gf_auc_both},
                "gf_vs_knn_f1": {"rho": pr_gf_f1_both, "p": pp_gf_f1_both},
                "gf_vs_mrr": {"rho": pr_gf_mrr_both, "p": pp_gf_mrr_both},
            },
        },
        "interpretation": {
            "note": (
                "Partial correlations control for effective rank and GNN "
                "architecture as potential confounders. All correlations are "
                "exploratory (n=11 methods); results should be interpreted "
                "with caution given the small sample size."
            ),
        },
    }

    output_path = RES / "partial_correlation_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved {output_path}")

    print("\n" + "=" * 70)
    print("Partial correlation analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
