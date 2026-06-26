#!/usr/bin/env python3
"""
Phase 8C: Leave-One-Out Sensitivity Analysis
=============================================
Tests how each correlation changes when one method is excluded.
Identifies whether Spectral (or other methods) are driving the
Phase 8B results.

Depends on: results/human_tda_full.json
Generates:  results/human_loo_sensitivity.json
            figures/Fig49_loo_sensitivity.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import get_results_dir, get_figures_dir

RESULTS = get_results_dir()
FIGURES = get_figures_dir()
BANNER = "=" * 70


def load_data():
    """Load Phase 8B feature matrix."""
    fpath = RESULTS / "human_tda_full.json"
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    methods = data["methods"]
    pm = data["per_method"]
    gf = np.array([pm[m]["gf_score"] for m in methods])
    sa = np.array([pm[m]["spectral_alignment"] for m in methods])
    er = np.array([pm[m]["effective_rank"] for m in methods])
    h1 = np.array([pm[m]["h1_max_persistence"] for m in methods])
    return methods, gf, sa, er, h1


def compute_correlations(gf, sa, er, h1):
    """Compute all single/multi-factor correlations."""
    n = len(gf)
    results = {}

    # Single factors
    for name, vals in [("spectral_alignment", sa),
                       ("effective_rank", er),
                       ("h1_max_persistence", h1)]:
        rho, p = spearmanr(vals, gf)
        results[name] = {"rho": round(float(rho), 3), "p": round(float(p), 4)}

    # Two-factor
    sa_r = rankdata(sa)
    er_r = rankdata(er)
    two_f = 0.5 * sa_r + 0.5 * er_r
    rho, p = spearmanr(two_f, gf)
    results["two_factor"] = {"rho": round(float(rho), 3), "p": round(float(p), 4)}

    # Three-factor
    h1_r = rankdata(h1)
    three_f = (1/3) * sa_r + (1/3) * er_r + (1/3) * h1_r
    rho, p = spearmanr(three_f, gf)
    results["three_factor"] = {"rho": round(float(rho), 3), "p": round(float(p), 4)}

    return results


def run_loo():
    print(BANNER)
    print("Phase 8C: Leave-One-Out Sensitivity Analysis")
    print(BANNER)

    methods, gf, sa, er, h1 = load_data()
    n = len(methods)
    print(f"\n  Full dataset: n={n} methods")

    # Full correlations
    full = compute_correlations(gf, sa, er, h1)
    print(f"\n  Full correlations (baseline):")
    for k, v in full.items():
        print(f"    {k:<30}: rho={v['rho']:+.3f} (p={v['p']:.4f})")

    # LOO analysis
    loo_results = {}
    print(f"\n  Leave-one-out results:")
    print(f"    {'Excluded':<12}", end="")
    for k in full:
        short = k[:8]
        print(f" {short:>10}", end="")
    print()
    print(f"    {'-'*12}", end="")
    for _ in full:
        print(f" {'-'*10}", end="")
    print()

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        excluded = methods[i]

        gf_sub = gf[mask]
        sa_sub = sa[mask]
        er_sub = er[mask]
        h1_sub = h1[mask]

        corr = compute_correlations(gf_sub, sa_sub, er_sub, h1_sub)
        loo_results[excluded] = corr

        print(f"    {excluded:<12}", end="")
        for k in full:
            delta = corr[k]["rho"] - full[k]["rho"]
            marker = ""
            if abs(delta) > 0.15:
                marker = " *"
            if abs(delta) > 0.30:
                marker = " **"
            print(f" {corr[k]['rho']:>+7.3f}{marker:<3}", end="")
        print()

    # Summary: which method has the biggest impact on each correlation?
    print(f"\n  Impact summary (max |delta_rho| per predictor):")
    summary = {}
    for k in full:
        deltas = {}
        for excluded, corr in loo_results.items():
            deltas[excluded] = corr[k]["rho"] - full[k]["rho"]
        max_method = max(deltas, key=lambda m: abs(deltas[m]))
        max_delta = deltas[max_method]
        loo_rho = loo_results[max_method][k]["rho"]
        summary[k] = {
            "most_influential": max_method,
            "delta_rho": round(float(max_delta), 3),
            "loo_rho": round(float(loo_rho), 3),
            "full_rho": full[k]["rho"],
        }
        print(f"    {k:<30}: excluding {max_method:<12} -> rho={loo_rho:+.3f} (delta={max_delta:+.3f})")

    # Specific check: excluding Spectral
    print(f"\n  Key check: excluding Spectral:")
    spec_loo = loo_results.get("Spectral", {})
    for k in full:
        if k in spec_loo:
            delta = spec_loo[k]["rho"] - full[k]["rho"]
            print(f"    {k:<30}: {full[k]['rho']:+.3f} -> {spec_loo[k]['rho']:+.3f} (delta={delta:+.3f})")

    # Also check yeast LOO for comparison
    print(f"\n  Yeast LOO comparison (from Phase 7 data):")
    yeast_check()

    # Save
    output = {
        "analysis": "Phase 8C: Leave-One-Out Sensitivity",
        "species": "human",
        "n_methods": n,
        "full_correlations": full,
        "loo_results": loo_results,
        "impact_summary": summary,
    }
    out_path = RESULTS / "human_loo_sensitivity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {out_path}")

    # Generate figure
    generate_figure(methods, full, loo_results, summary)

    print(f"\n{BANNER}")
    print("Phase 8C complete.")
    print(BANNER)

    return output


def yeast_check():
    """Quick LOO check on yeast Phase 7 data for comparison."""
    yeast_path = RESULTS / "tda_geometry_bridge.json"
    if not yeast_path.exists():
        print("    (yeast data not available for comparison)")
        return
    with open(yeast_path, encoding="utf-8") as f:
        ydata = json.load(f)

    # Build yeast feature vectors from per_method data
    pm = ydata.get("per_method", {})
    if not pm:
        print("    (yeast per_method data not available)")
        return
    methods = sorted(pm.keys())
    gf = np.array([pm[m].get("gf_score", 0) for m in methods])
    sa = np.array([pm[m].get("spectral_alignment", 0) for m in methods])
    er = np.array([pm[m].get("effective_rank", 0) for m in methods])
    h1 = np.array([pm[m].get("h1_max_persistence", 0) for m in methods])
    n = len(methods)

    full = compute_correlations(gf, sa, er, h1)
    # LOO for Spectral only
    if "Spectral" in methods:
        idx = methods.index("Spectral")
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        loo = compute_correlations(gf[mask], sa[mask], er[mask], h1[mask])
        print(f"    Yeast full (n={n}): h1_max rho={full['h1_max_persistence']['rho']:+.3f}")
        print(f"    Yeast excl Spectral: h1_max rho={loo['h1_max_persistence']['rho']:+.3f} "
              f"(delta={loo['h1_max_persistence']['rho'] - full['h1_max_persistence']['rho']:+.3f})")


def generate_figure(methods, full, loo_results, summary):
    """Generate Fig49: LOO sensitivity (3 panels)."""
    print("  Generating Fig49...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    predictors = list(full.keys())

    # Panel A: LOO delta heatmap-style bar chart
    ax = axes[0]
    n = len(methods)
    n_pred = len(predictors)
    delta_matrix = np.zeros((n, n_pred))
    for i, m in enumerate(sorted(methods, key=lambda x: loo_results[x].get("h1_max_persistence", {}).get("rho", 0))):
        for j, k in enumerate(predictors):
            delta_matrix[i, j] = loo_results[m][k]["rho"] - full[k]["rho"]

    x_pos = np.arange(n_pred)
    width = 0.7 / n
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, n))
    for i in range(n):
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x_pos + offset, delta_matrix[i], width * 0.9,
                      label=sorted(methods)[i] if i < 5 else None,
                      color=colors[i], edgecolor="k", linewidth=0.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace("_", "\n") for p in predictors], fontsize=7)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.set_ylabel("Delta rho (LOO - Full)")
    ax.set_title("A. Leave-One-Out Impact on Correlations\n(delta rho when excluding each method)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: LOO rho values for H1 max persistence (main focus)
    ax = axes[1]
    h1_full_rho = full["h1_max_persistence"]["rho"]
    loo_rhos = []
    loo_methods = []
    for m in sorted(methods, key=lambda x: loo_results[x]["h1_max_persistence"]["rho"], reverse=True):
        loo_rhos.append(loo_results[m]["h1_max_persistence"]["rho"])
        loo_methods.append(m)

    colors_bar = ["#E6550D" if m == "Spectral" else "#3182BD" for m in loo_methods]
    bars = ax.barh(range(len(loo_methods)), loo_rhos, color=colors_bar, edgecolor="k", linewidth=0.5)
    ax.axvline(x=h1_full_rho, color="gray", linestyle="--", linewidth=1.5,
               label=f"Full (rho={h1_full_rho:.3f})")
    ax.set_yticks(range(len(loo_methods)))
    ax.set_yticklabels(loo_methods, fontsize=8)
    ax.set_xlabel("Spearman rho (H1 max persistence vs G-F Score)")
    ax.set_title("B. H1 Max Persistence: LOO Sensitivity\n(rho when excluding each method)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    # Annotate
    for i, (m, rho) in enumerate(zip(loo_methods, loo_rhos)):
        delta = rho - h1_full_rho
        ax.annotate(f"rho={rho:+.3f} ({delta:+.3f})",
                    xy=(rho, i), xytext=(5, 0), textcoords="offset points",
                    fontsize=7, va="center")

    # Panel C: Scatter — H1 max persistence vs G-F Score (full + excl-Spectral overlay)
    ax = axes[2]
    gf = [loo_results[m]["h1_max_persistence"].get("_gf", 0) for m in methods]
    # Reload for scatter
    fpath = RESULTS / "human_tda_full.json"
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    pm = data["per_method"]
    gf_vals = np.array([pm[m]["gf_score"] for m in methods])
    h1_vals = np.array([pm[m]["h1_max_persistence"] for m in methods])

    # All points
    ax.scatter(h1_vals, gf_vals, s=70, c="#BDBDBD", edgecolors="k", linewidth=0.5,
               zorder=2, label="All (rho={:.3f})".format(full["h1_max_persistence"]["rho"]))
    # Spectral highlighted
    spec_idx = list(methods).index("Spectral")
    ax.scatter(h1_vals[spec_idx], gf_vals[spec_idx], s=120, c="#E6550D",
               edgecolors="k", linewidth=1.5, zorder=4, label="Spectral (outlier)")
    ax.annotate("Spectral", (h1_vals[spec_idx], gf_vals[spec_idx]),
                fontsize=8, fontweight="bold", ha="left", va="bottom",
                xytext=(5, 5), textcoords="offset points")

    # Excluding Spectral: trend line
    mask = np.ones(len(methods), dtype=bool)
    mask[spec_idx] = False
    x_no = h1_vals[mask]
    y_no = gf_vals[mask]
    rho_no, p_no = spearmanr(x_no, y_no)
    z = np.polyfit(x_no, y_no, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x_no.min(), x_no.max(), 100)
    ax.plot(x_line, p_line(x_line), "b--", alpha=0.5, lw=1.5,
            label=f"Excl Spectral (rho={rho_no:+.3f})")

    # Full trend line
    z_full = np.polyfit(h1_vals, gf_vals, 1)
    p_full = np.poly1d(z_full)
    ax.plot(x_line, p_full(x_line), "k--", alpha=0.3, lw=1)

    ax.set_xlabel("H1 Max Persistence")
    ax.set_ylabel("G-F Score")
    ax.set_title("C. H1 vs G-F: Spectral Outlier Effect\n(gray=all, blue=excl. Spectral)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Label all points
    for m, h, g in zip(methods, h1_vals, gf_vals):
        if m != "Spectral":
            ax.annotate(m, (h, g), fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    fig_path = FIGURES / "Fig49_loo_sensitivity.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_loo()
