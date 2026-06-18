#!/usr/bin/env python3
"""
GF-MRR Bridge Theorem (Step 73 / Phase 22)
============================================

Bridge the gap between GF Score (geometric measure) and MRR
(downstream task performance).

Empirical: Fit MRR as a function of GF Score, dimension, and
network size using data from:
  - Yeast: 3 ontologies x 3 Spectral dimensions = 9 data points
  - Plus all 11 methods at d=2 = 11 more data points
  - Existing species: yeast, human, mouse GF scores

Theorem: Given GF Score g on network G with embedding dimension d,
         MRR >= f(g, d, n).

Output
------
- results/gf_mrr_bridge.json
- figures/Fig81_gf_mrr_bridge.png
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_results_dir, get_figures_dir

# ============================================================
# Constants
# ============================================================

RESULTS = get_results_dir()
FIGURES = get_figures_dir()
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

BANNER = "=" * 64


# ============================================================
# Collect Data Points
# ============================================================

def collect_data_points():
    """Collect (GF_score, MRR, dimension, ontology, species) tuples."""

    data_points = []

    # --- Yeast atlas (multi-ontology, multi-dimension) ---
    atlas_file = RESULTS / "function_prediction_atlas.json"
    if atlas_file.exists():
        with open(atlas_file, encoding="utf-8") as f:
            atlas = json.load(f)

        # Yeast GF scores (from gf_scores.json or known values)
        gf_file = RESULTS / "gf_scores.json"
        yeast_gf = {}
        if gf_file.exists():
            with open(gf_file, encoding="utf-8") as f:
                gf_data = json.load(f)
            for method, score in gf_data.items():
                if isinstance(score, (int, float)):
                    yeast_gf[method] = float(score)

        # Default Spectral GF from curated results
        spectral_gf = yeast_gf.get("Spectral", 0.163)

        for aspect, aspect_data in atlas.get("ontologies", {}).items():
            # Extract PPI baseline for this ontology
            ppi_mrr = None
            for m, r in aspect_data.get("methods", {}).items():
                if "PPI_MRR" in r:
                    ppi_mrr = r["PPI_MRR"]
                    break

            for method, method_data in aspect_data.get("methods", {}).items():
                mrr = method_data.get("MRR", 0)
                if mrr > 0:
                    # Parse dimension from method name
                    if "Spectral-d" in method:
                        dim = int(method.split("d")[1])
                        gf = spectral_gf
                    else:
                        dim = 2
                        gf = yeast_gf.get(method.replace("-d2", ""), 0.0)

                    # Absolute MRR
                    data_points.append({
                        "species": "yeast",
                        "ontology": aspect,
                        "method": method,
                        "dimension": dim,
                        "gf_score": gf,
                        "mrr": mrr,
                    })

                    # Relative MRR (vs PPI baseline) for Spectral methods
                    if ppi_mrr and ppi_mrr > 0 and "Spectral" in method:
                        data_points.append({
                            "species": "yeast",
                            "ontology": aspect,
                            "method": method + "_relative",
                            "dimension": dim,
                            "gf_score": gf,
                            "mrr": mrr / ppi_mrr,
                        })

    # --- Dimension sweep (BP only, more dimensions) ---
    dim_file = RESULTS / "dimension_sweep_512.json"
    if dim_file.exists():
        with open(dim_file, encoding="utf-8") as f:
            dim_data = json.load(f)

        spectral_gf = 0.163
        for dim_str, mrr in dim_data.get("mrr_by_dimension", {}).items():
            dim = int(dim_str)
            data_points.append({
                "species": "yeast",
                "ontology": "BP",
                "method": f"Spectral-d{dim}",
                "dimension": dim,
                "gf_score": spectral_gf,
                "mrr": mrr,
            })

    # --- Cross-species (if available) ---
    cross_file = RESULTS / "cross_species_atlas.json"
    if cross_file.exists():
        with open(cross_file, encoding="utf-8") as f:
            cross_data = json.load(f)

        # GF scores for other species (from existing results)
        species_gf = {
            "human": 0.852,
            "mouse": 0.309,
        }

        for sp_key, sp_data in cross_data.get("species", {}).items():
            gf = species_gf.get(sp_key, 0)
            for aspect, methods in sp_data.get("ontologies", {}).items():
                for method, method_data in methods.items():
                    mrr = method_data.get("MRR", 0)
                    if mrr > 0 and gf > 0:
                        dim = int(method.split("d")[1]) if "d" in method else 2
                        data_points.append({
                            "species": sp_key,
                            "ontology": aspect,
                            "method": method,
                            "dimension": dim,
                            "gf_score": gf,
                            "mrr": mrr,
                        })

    return data_points


# ============================================================
# Analysis
# ============================================================

def analyze_bridge(data_points):
    """Fit and analyze the GF-MRR bridge."""

    if len(data_points) < 5:
        print("  WARNING: too few data points for analysis")
        return {}

    gf_scores = np.array([d["gf_score"] for d in data_points])
    mrr_values = np.array([d["mrr"] for d in data_points])
    dimensions = np.array([d["dimension"] for d in data_points])
    log_dims = np.log2(dimensions + 1)

    # --- Model 1: MRR ~ GF alone ---
    rho_gf, p_gf = spearmanr(gf_scores, mrr_values)
    slope_gf, intercept_gf, r_gf, _, _ = linregress(gf_scores, mrr_values)

    # --- Model 2: MRR ~ GF + log(d) ---
    X = np.column_stack([gf_scores, log_dims])
    X_aug = np.column_stack([np.ones(len(X)), X])
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X_aug, mrr_values, rcond=None)
        mrr_pred = X_aug @ beta
        ss_res = np.sum((mrr_values - mrr_pred) ** 2)
        ss_tot = np.sum((mrr_values - mrr_values.mean()) ** 2)
        r2_combined = 1 - ss_res / ss_tot
    except Exception:
        beta = np.array([0, 0, 0, 0])
        r2_combined = 0

    # --- Model 3: MRR ~ GF * log(d) (interaction) ---
    interaction = gf_scores * log_dims
    X_int = np.column_stack([np.ones(len(interaction)), gf_scores, log_dims, interaction])
    try:
        beta_int, _, _, _ = np.linalg.lstsq(X_int, mrr_values, rcond=None)
        mrr_pred_int = X_int @ beta_int
        ss_res_int = np.sum((mrr_values - mrr_pred_int) ** 2)
        r2_interaction = 1 - ss_res_int / ss_tot
    except Exception:
        beta_int = np.array([0, 0, 0, 0])
        r2_interaction = 0

    # --- d=2 method comparison: GF Score vs MRR at fixed dimension ---
    d2_points = [d for d in data_points if d["dimension"] == 2
                 and not d["method"].endswith("_relative")]
    if len(d2_points) >= 5:
        d2_gf = np.array([d["gf_score"] for d in d2_points])
        d2_mrr = np.array([d["mrr"] for d in d2_points])
        rho_d2, p_d2 = spearmanr(d2_gf, d2_mrr)
        slope_d2, intercept_d2, r_d2, _, _ = linregress(d2_gf, d2_mrr)
    else:
        rho_d2, p_d2, r_d2 = 0, 1, 0
        slope_d2, intercept_d2 = 0, 0

    # --- Model 4: Dimension-only (MRR ~ log2(d)) ---
    slope_dim, intercept_dim, r_dim, p_dim, _ = linregress(log_dims, mrr_values)
    rho_dim, p_rho_dim = spearmanr(log_dims, mrr_values)

    # --- Model 5: Universal relative improvement per dimension doubling ---
    # For Spectral methods only, compute MRR(d)/MRR(d_prev) across ontologies
    spectral_points = [d for d in data_points if "Spectral" in d["method"]
                       and not d["method"].endswith("_relative")]
    spectral_by_ont = defaultdict(list)
    for dp in spectral_points:
        spectral_by_ont[dp["ontology"]].append(dp)

    relative_improvements = []
    for ont, points in spectral_by_ont.items():
        points.sort(key=lambda x: x["dimension"])
        for i in range(1, len(points)):
            ratio = points[i]["mrr"] / max(points[i-1]["mrr"], 1e-10)
            dim_ratio = points[i]["dimension"] / max(points[i-1]["dimension"], 1)
            relative_improvements.append({
                "ontology": ont,
                "from_dim": points[i-1]["dimension"],
                "to_dim": points[i]["dimension"],
                "mrr_ratio": ratio,
                "dim_ratio": dim_ratio,
            })
    gf_bins = np.linspace(0, 1, 11)
    lower_bound_points = []
    for i in range(len(gf_bins) - 1):
        mask = (gf_scores >= gf_bins[i]) & (gf_scores < gf_bins[i+1])
        if mask.sum() >= 2:
            min_mrr = float(np.min(mrr_values[mask]))
            mean_gf = float(np.mean(gf_scores[mask]))
            lower_bound_points.append((mean_gf, min_mrr))

    results = {
        "n_data_points": len(data_points),
        "n_d2_points": len(d2_points),
        "model_gf_only": {
            "rho": float(rho_gf),
            "p": float(p_gf),
            "R_squared": float(r_gf ** 2),
            "slope": float(slope_gf),
            "intercept": float(intercept_gf),
        },
        "model_d2_only": {
            "rho": float(rho_d2),
            "p": float(p_d2),
            "R_squared": float(r_d2 ** 2),
            "slope": float(slope_d2),
            "intercept": float(intercept_d2),
        },
        "model_dimension_only": {
            "rho": float(rho_dim),
            "p": float(p_rho_dim),
            "R_squared": float(r_dim ** 2),
            "slope": float(slope_dim),
            "intercept": float(intercept_dim),
            "interpretation": "MRR per log2(d) doubling",
        },
        "model_gf_plus_logd": {
            "R_squared": float(r2_combined),
            "coefficients": {
                "intercept": float(beta[0]),
                "gf_score": float(beta[1]),
                "log2_dimension": float(beta[2]),
            },
        },
        "model_interaction": {
            "R_squared": float(r2_interaction),
            "coefficients": {
                "intercept": float(beta_int[0]),
                "gf_score": float(beta_int[1]),
                "log2_dimension": float(beta_int[2]),
                "gf_x_logd": float(beta_int[3]),
            },
        },
        "universal_improvement_ratios": relative_improvements,
        "lower_bound_points": [(float(g), float(m)) for g, m in lower_bound_points],
    }

    return results


# ============================================================
# Main
# ============================================================

def run():
    t_start = time.time()
    print(BANNER)
    print("  GF-MRR Bridge Theorem")
    print(BANNER)

    # Collect data
    print(f"\n[1/4] Collecting data points ...")
    data_points = collect_data_points()
    print(f"  {len(data_points)} data points collected")

    # Deduplicate (same method/dim/ontology)
    seen = set()
    unique_points = []
    for dp in data_points:
        key = (dp["species"], dp["ontology"], dp["method"])
        if key not in seen:
            seen.add(key)
            unique_points.append(dp)
    data_points = unique_points
    print(f"  {len(data_points)} unique data points after dedup")

    # Analyze
    print(f"\n[2/4] Fitting models ...")
    results = analyze_bridge(data_points)

    if not results:
        print("  Analysis failed!")
        return

    # Print results
    print(f"\n  Model 1: MRR ~ GF Score alone (all data)")
    m1 = results["model_gf_only"]
    print(f"    rho = {m1['rho']:.4f}, p = {m1['p']:.4e}")
    print(f"    R^2 = {m1['R_squared']:.4f}")

    print(f"\n  Model 1b: MRR ~ GF Score (d=2 methods only)")
    m1b = results["model_d2_only"]
    print(f"    rho = {m1b['rho']:.4f}, p = {m1b['p']:.4e}")
    print(f"    R^2 = {m1b['R_squared']:.4f}")
    print(f"    MRR = {m1b['slope']:.4f} * GF + {m1b['intercept']:.4f}")

    print(f"\n  Model 2: MRR ~ GF + log2(d)")
    m2 = results["model_gf_plus_logd"]
    print(f"    R^2 = {m2['R_squared']:.4f}")
    print(f"    MRR = {m2['coefficients']['gf_score']:.4f} * GF "
          f"+ {m2['coefficients']['log2_dimension']:.4f} * log2(d) "
          f"+ {m2['coefficients']['intercept']:.4f}")

    print(f"\n  Model 3: MRR ~ GF * log2(d) (interaction)")
    m3 = results["model_interaction"]
    print(f"    R^2 = {m3['R_squared']:.4f}")

    print(f"\n  Model 4: MRR ~ log2(d) (dimension only)")
    m4 = results["model_dimension_only"]
    print(f"    rho = {m4['rho']:.4f}, p = {m4['p']:.4e}")
    print(f"    R^2 = {m4['R_squared']:.4f}")
    print(f"    MRR gain per log2(d) doubling: {m4['slope']:.4f}")

    print(f"\n  Model 5: Universal improvement ratios (Spectral)")
    for r in results.get("universal_improvement_ratios", []):
        print(f"    {r['ontology']}: d={r['from_dim']}->{r['to_dim']}: "
              f"MRR ratio = {r['mrr_ratio']:.3f}x")

    # Save results
    print(f"\n[3/4] Saving results ...")
    output = {
        "description": "GF-MRR Bridge: Linking Geometric Quality to Function Prediction",
        "data_points": data_points,
        "analysis": results,
    }

    out_file = RESULTS / "gf_mrr_bridge.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {out_file}")

    # Figure
    print(f"\n[4/4] Generating figure ...")
    plot_bridge(data_points, results)

    elapsed = time.time() - t_start
    print(f"\nGF-MRR bridge completed in {elapsed:.1f}s")
    return results


# ============================================================
# Figure
# ============================================================

def plot_bridge(data_points, results):
    """GF-MRR bridge figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    gf_scores = np.array([d["gf_score"] for d in data_points])
    mrr_values = np.array([d["mrr"] for d in data_points])
    dimensions = np.array([d["dimension"] for d in data_points])

    species_colors = {
        "yeast": "#2ca02c", "human": "#d62728", "mouse": "#3182bd",
    }
    ontology_markers = {"BP": "o", "MF": "s", "CC": "^"}

    # --- Left: GF vs MRR scatter ---
    ax = axes[0]
    for dp in data_points:
        color = species_colors.get(dp["species"], "#999")
        marker = ontology_markers.get(dp["ontology"], "o")
        size = 20 + 5 * np.log2(dp["dimension"] + 1)
        ax.scatter(dp["gf_score"], dp["mrr"], c=color, marker=marker,
                  s=size, alpha=0.7, edgecolors="white", linewidth=0.5)

    # Fit line
    if "model_gf_only" in results:
        m1 = results["model_gf_only"]
        x_line = np.linspace(0, 1, 50)
        y_line = m1["slope"] * x_line + m1["intercept"]
        ax.plot(x_line, y_line, "--", color="grey", linewidth=1.5, alpha=0.5,
               label=f"R$^2$={m1['R_squared']:.3f}, rho={m1['rho']:.3f}")

    # Lower bound
    lb = results.get("lower_bound_points", [])
    if lb:
        lb_gf, lb_mrr = zip(*lb)
        ax.plot(lb_gf, lb_mrr, "k-", linewidth=2, alpha=0.3,
               label="Empirical lower bound")

    ax.set_xlabel("GF Score", fontsize=12)
    ax.set_ylabel("MRR", fontsize=12)
    ax.set_title("GF Score vs Function Prediction Performance",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Legend for species/ontology
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
              markersize=8, label="Yeast"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
              markersize=8, label="Human"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3182bd",
              markersize=8, label="Mouse"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
              markersize=8, label="BP"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
              markersize=8, label="MF"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
              markersize=8, label="CC"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right",
             ncol=2)

    # --- Right: Residuals by dimension ---
    ax = axes[1]
    m2 = results.get("model_gf_plus_logd", {})
    if m2.get("coefficients"):
        c = m2["coefficients"]
        predicted = c["intercept"] + c["gf_score"] * gf_scores + c["log2_dimension"] * np.log2(dimensions + 1)
        residuals = mrr_values - predicted

        for dp, res in zip(data_points, residuals):
            color = species_colors.get(dp["species"], "#999")
            marker = ontology_markers.get(dp["ontology"], "o")
            ax.scatter(dp["dimension"], res, c=color, marker=marker,
                      s=40, alpha=0.7, edgecolors="white", linewidth=0.5)

        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Embedding Dimension (d)", fontsize=12)
        ax.set_ylabel("Residual (MRR - predicted)", fontsize=12)
        ax.set_title(f"Model 2 Residuals (R$^2$={m2.get('R_squared', 0):.3f})",
                    fontsize=13, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle("GF-MRR Bridge: Geometry Predicts Function",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = FIGURES / "Fig81_gf_mrr_bridge.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


if __name__ == "__main__":
    run()
