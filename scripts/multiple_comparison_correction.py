#!/usr/bin/env python3
"""
Multiple Comparison Correction for All Spearman Correlation Tests
=================================================================
Addresses P0-1: The project reports 14+ Spearman correlation tests on
n=11 methods without multiple comparison correction.  This script
collects every reported Spearman p-value, applies Benjamini-Hochberg
FDR correction and Bonferroni correction, and outputs a unified report.

Output: results/multiple_comparison_correction.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array-like of float
        Raw p-values from m simultaneous tests.

    Returns
    -------
    dict with keys:
        - 'p_corrected': list of BH-adjusted p-values (same order as input)
        - 'rejected': list of bool, whether each test is rejected at alpha=0.05
        - 'n_tests': number of tests
        - 'alpha': significance level used
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    # Sort p-values ascending, keep track of original order
    order = np.argsort(p)
    ranked = p[order]

    # BH adjusted p: p_(k) * m / k, then enforce monotonicity from largest
    adjusted = np.empty(m)
    adjusted[-1] = ranked[-1] * m / m  # k=m -> p*m/m = p
    for k in range(m - 2, -1, -1):
        rank = k + 1  # 1-based rank
        val = ranked[k] * m / rank
        adjusted[k] = min(val, adjusted[k + 1])  # enforce monotonicity

    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Un-sort to match original order
    result = np.empty(m)
    result[order] = adjusted

    alpha = 0.05
    rejected = result < alpha

    return {
        "p_corrected": result.tolist(),
        "rejected": rejected.tolist(),
        "n_tests": m,
        "alpha": alpha,
    }


def bonferroni_correction(p_values):
    """Apply Bonferroni correction.

    Returns dict with 'p_corrected' (min(p*m, 1)) and 'rejected'.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    corrected = np.clip(p * m, 0.0, 1.0)
    alpha = 0.05
    rejected = corrected < alpha
    return {
        "p_corrected": corrected.tolist(),
        "rejected": rejected.tolist(),
        "n_tests": m,
        "alpha": alpha,
    }


def collect_all_correlations():
    """Collect all Spearman correlation p-values from result files.

    Returns a list of dicts, each with:
        - 'test_name': human-readable identifier
        - 'rho': Spearman rho
        - 'p_value': raw p-value
        - 'n': sample size
        - 'source_file': which results JSON it came from
        - 'family': which correction family it belongs to
    """
    tests = []

    # ------------------------------------------------------------------
    # Family 1: GF Score vs downstream metrics (n=11, primary)
    # Source: bootstrap_correlations.json + metric_comparison.json +
    #         function_prediction_full.json + partial_correlation_analysis.json
    # ------------------------------------------------------------------

    # 1. H1 max persistence vs GF Score
    bc_file = RESULTS_DIR / "bootstrap_correlations.json"
    if bc_file.exists():
        with open(bc_file, encoding="utf-8") as f:
            bc = json.load(f)
        entry = bc.get("h1_max_persistence_vs_gf_score", {})
        if entry:
            tests.append({
                "test_name": "H1 max persistence vs GF Score",
                "rho": entry.get("rho"),
                "p_value": entry.get("p_value"),
                "n": entry.get("n"),
                "source_file": "bootstrap_correlations.json",
                "family": "primary",
            })
        entry = bc.get("gf_score_vs_link_pred_auc", {})
        if entry:
            tests.append({
                "test_name": "GF Score vs Link Pred AUC",
                "rho": entry.get("rho"),
                "p_value": entry.get("p_value"),
                "n": entry.get("n"),
                "source_file": "bootstrap_correlations.json",
                "family": "primary",
            })
        entry = bc.get("gf_score_vs_knn_f1", {})
        if entry:
            tests.append({
                "test_name": "GF Score vs k-NN F1",
                "rho": entry.get("rho"),
                "p_value": entry.get("p_value"),
                "n": entry.get("n"),
                "source_file": "bootstrap_correlations.json",
                "family": "primary",
            })
        entry = bc.get("topo_consistency_vs_gf_score", {})
        if entry:
            tests.append({
                "test_name": "Topo consistency vs GF Score",
                "rho": entry.get("rho"),
                "p_value": entry.get("p_value"),
                "n": entry.get("n"),
                "source_file": "bootstrap_correlations.json",
                "family": "primary",
            })

    # 2. GF Score vs MRR (full 11-method)
    fp_file = RESULTS_DIR / "function_prediction_full.json"
    if fp_file.exists():
        with open(fp_file, encoding="utf-8") as f:
            fp = json.load(f)
        gf_mrr = fp.get("gf_correlation", fp.get("gf_mrr_correlation", {}))
        if not gf_mrr:
            # Try alternative key structure
            gf_mrr = fp.get("correlation", {})
        if gf_mrr:
            tests.append({
                "test_name": "GF Score vs Function Prediction MRR (11 methods)",
                "rho": gf_mrr.get("spearman_rho", gf_mrr.get("rho")),
                "p_value": gf_mrr.get("spearman_p",
                                       gf_mrr.get("p_value",
                                                  gf_mrr.get("p"))),
                "n": gf_mrr.get("n", 11),
                "permutation_p": gf_mrr.get("permutation_p"),
                "source_file": "function_prediction_full.json",
                "family": "primary",
            })

    # 3. Cross-species consistency (yeast vs human)
    cs_file = RESULTS_DIR / "cross_species_consistency.json"
    if cs_file.exists():
        with open(cs_file, encoding="utf-8") as f:
            cs = json.load(f)
        sp = cs.get("spearman_correlation", {})
        if sp:
            tests.append({
                "test_name": "Cross-species rank consistency (yeast vs human)",
                "rho": sp.get("rho"),
                "p_value": sp.get("p_value"),
                "n": sp.get("n", 11),
                "permutation_p": cs.get("permutation_test", {}).get("p_perm"),
                "source_file": "cross_species_consistency.json",
                "family": "primary",
            })

    # 4. Standard vs IC-weighted (human)
    ic_file = RESULTS_DIR / "human_ic_weighted_gf.json"
    if ic_file.exists():
        with open(ic_file, encoding="utf-8") as f:
            ic = json.load(f)
        # Search for the correlation entry
        corr = ic.get("spearman_correlation", ic.get("correlation", {}))
        if corr:
            tests.append({
                "test_name": "Standard vs IC-weighted GF (human)",
                "rho": corr.get("rho"),
                "p_value": corr.get("p_value", corr.get("p")),
                "n": corr.get("n", 11),
                "source_file": "human_ic_weighted_gf.json",
                "family": "primary",
            })

    # 5. Degree-embedding correlation (cross-method)
    de_file = RESULTS_DIR / "degree_embedding_correlation.json"
    if de_file.exists():
        with open(de_file, encoding="utf-8") as f:
            de = json.load(f)
        cm = de.get("cross_method_correlation", {})
        for key, val in cm.items():
            label = key.replace("_", " ").replace("z vs", "DP-null z vs")
            tests.append({
                "test_name": f"DP-null z-score vs {label.split('vs')[-1].strip()}",
                "rho": val.get("rho"),
                "p_value": val.get("p"),
                "n": 11,
                "source_file": "degree_embedding_correlation.json",
                "family": "primary",
            })

    # 6. Pairwise Spearman from partial_correlation_analysis.json
    pa_file = RESULTS_DIR / "partial_correlation_analysis.json"
    if pa_file.exists():
        with open(pa_file, encoding="utf-8") as f:
            pa = json.load(f)
        pairwise = pa.get("pairwise_spearman", {})
        for key, val in pairwise.items():
            # Skip GF vs AUC/F1/MRR — already collected above
            if key in ("gf_vs_link_pred_auc", "gf_vs_knn_f1", "gf_vs_mrr"):
                continue
            label = key.replace("_", " ").replace("gf", "GF").replace("auc", "AUC")
            tests.append({
                "test_name": f"Pairwise: {label}",
                "rho": val.get("rho"),
                "p_value": val.get("p"),
                "n": 11,
                "source_file": "partial_correlation_analysis.json",
                "family": "primary",
            })

    # 7. Extended 15-method comparisons
    mc15_file = RESULTS_DIR / "metric_comparison_extended_15.json"
    if mc15_file.exists():
        with open(mc15_file, encoding="utf-8") as f:
            mc15 = json.load(f)
        corr = mc15.get("correlations", {})
        for key, val in corr.items():
            if isinstance(val, dict) and "p_value" in val:
                tests.append({
                    "test_name": f"Extended 15-method: {key}",
                    "rho": val.get("spearman_rho", val.get("rho")),
                    "p_value": val.get("p_value"),
                    "n": val.get("n", 15),
                    "source_file": "metric_comparison_extended_15.json",
                    "family": "extended",
                })

    return tests


def main():
    print("=" * 72)
    print("  Multiple Comparison Correction for All Spearman Correlations")
    print("=" * 72)

    tests = collect_all_correlations()

    if not tests:
        print("  No correlation tests found. Check results/ directory.")
        return

    # Extract raw p-values for the primary family (n=11 methods)
    primary_tests = [t for t in tests if t["family"] == "primary"]
    extended_tests = [t for t in tests if t["family"] == "extended"]

    print(f"\n  Collected {len(primary_tests)} primary tests (n=11 methods)")
    print(f"  Collected {len(extended_tests)} extended tests (n=15 methods)")

    # --- Apply corrections to primary family ---
    raw_p = [t["p_value"] for t in primary_tests if t["p_value"] is not None]
    valid_tests = [t for t in primary_tests if t["p_value"] is not None]

    bh = benjamini_hochberg(raw_p)
    bonf = bonferroni_correction(raw_p)

    # --- Build output ---
    corrected_tests = []
    for i, t in enumerate(valid_tests):
        entry = dict(t)
        entry["p_bh"] = bh["p_corrected"][i]
        entry["significant_bh"] = bh["rejected"][i]
        entry["p_bonferroni"] = bonf["p_corrected"][i]
        entry["significant_bonferroni"] = bonf["rejected"][i]
        entry["significant_raw"] = t["p_value"] < 0.05
        corrected_tests.append(entry)

    # Sort by raw p-value for readability
    corrected_tests.sort(key=lambda x: x["p_value"])

    # --- Print table ---
    print(f"\n  {'Test':<50s} {'rho':>7s} {'p_raw':>10s} "
          f"{'p_BH':>10s} {'p_Bonf':>10s} {'sig_BH':>7s}")
    print("  " + "-" * 100)
    for t in corrected_tests:
        sig_marker = "***" if t["significant_bh"] else "ns"
        print(f"  {t['test_name']:<50s} {t['rho']:+7.3f} "
              f"{t['p_value']:10.4e} {t['p_bh']:10.4e} "
              f"{t['p_bonferroni']:10.4e} {sig_marker:>7s}")

    print(f"\n  Family size (primary): {len(valid_tests)} tests")
    print(f"  Significant at FDR<0.05: {sum(bh['rejected'])}/{len(valid_tests)}")
    print(f"  Significant at Bonferroni<0.05: "
          f"{sum(bonf['rejected'])}/{len(valid_tests)}")

    # --- Interpretation ---
    print("\n  --- Interpretation ---")
    n_sig_raw = sum(1 for t in corrected_tests if t["significant_raw"])
    n_sig_bh = sum(1 for t in corrected_tests if t["significant_bh"])
    n_sig_bonf = sum(1 for t in corrected_tests if t["significant_bonferroni"])

    print(f"  Raw P<0.05:          {n_sig_raw}/{len(corrected_tests)} tests significant")
    print(f"  BH FDR<0.05:         {n_sig_bh}/{len(corrected_tests)} tests significant")
    print(f"  Bonferroni<0.05:     {n_sig_bonf}/{len(corrected_tests)} tests significant")

    # Key findings
    key_tests = {
        "GF Score vs Link Pred AUC": "GF vs AUC",
        "GF Score vs k-NN F1": "GF vs F1",
        "GF Score vs Function Prediction MRR": "GF vs MRR",
        "H1 max persistence vs GF Score": "H1 vs GF",
    }
    print("\n  --- Key conclusions after correction ---")
    for t in corrected_tests:
        for pattern, short in key_tests.items():
            if pattern in t["test_name"]:
                status = "SIGNIFICANT" if t["significant_bh"] else "NOT significant"
                print(f"  {short}: rho={t['rho']:+.3f}, "
                      f"p_raw={t['p_value']:.4f}, "
                      f"p_BH={t['p_bh']:.4f} -> {status} (FDR<0.05)")
                break

    # --- Save JSON ---
    output = {
        "description": "Multiple comparison correction for all Spearman "
                       "correlation tests in the G-F consistency framework",
        "correction_methods": ["Benjamini-Hochberg FDR", "Bonferroni"],
        "alpha": 0.05,
        "primary_family": {
            "n_tests": len(valid_tests),
            "n_significant_raw": n_sig_raw,
            "n_significant_bh": n_sig_bh,
            "n_significant_bonferroni": n_sig_bonf,
        },
        "tests": corrected_tests,
    }

    output_path = RESULTS_DIR / "multiple_comparison_correction.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
