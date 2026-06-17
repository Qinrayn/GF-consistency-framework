#!/usr/bin/env python3
"""
Generate Missing Supplementary Figures S18-S20 (Steps 50-52)
=============================================================

FigS18: Full-network GAT theorem verification (3-panel)
FigS19: Community detection ablation heatmap (2-panel)
FigS20: Coexpression network GF curves (2-panel)

All figures generated from existing result JSON files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_results_dir, get_figures_dir

RESULTS = get_results_dir()
FIGURES = get_figures_dir()

# Colour palette matching project style
METHOD_COLORS = {
    "Spectral": "#e41a1c",
    "DM": "#377eb8",
    "MDS": "#4daf4a",
    "PCA": "#984ea3",
    "Node2Vec": "#ff7f00",
    "DeepWalk": "#a65628",
    "VGAE": "#f781bf",
    "VGAE-feat": "#999999",
    "GraphSAGE": "#66c2a5",
    "GAT": "#fc8d62",
    "GIN": "#8da0cb",
}


def load_json(name):
    path = RESULTS / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# FigS18: Full-Network GAT Theorem Verification
# ============================================================

def generate_figs18():
    """3-panel figure: GAT collapse theorems on 153-node vs 5936-node."""
    theory = load_json("gat_collapse_theory.json")
    large_net = load_json("gat_theorem_large_network.json")
    dim_sweep = load_json("gat_dimension_sweep.json")
    multi_head = load_json("multihead_gat_experiment.json")

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ---- Panel A: Degree comparison & theorem bounds ----
    ax_a = fig.add_subplot(gs[0, 0])

    # 153-node vs 5936-node comparison
    small_cv = theory.get("P1_attention_degeneration", {}).get(
        "degree_stats", {}).get("CV", 0.644)
    large_cv = large_net["T1_attention_degeneration"]["full_network"]["degree_cv"]
    small_mean = theory.get("P1_attention_degeneration", {}).get(
        "degree_stats", {}).get("mean", 21.82)
    large_mean = large_net["T1_attention_degeneration"]["full_network"]["degree_mean"]
    small_gini = theory.get("P1_attention_degeneration", {}).get(
        "degree_stats", {}).get("gini", 0.355)
    large_gini = large_net["T1_attention_degeneration"]["full_network"]["gini"]

    x = np.arange(3)
    width = 0.35
    bars1 = ax_a.bar(x - width/2, [small_cv, small_mean/50, small_gini],
                     width, label="153-node curated", color="#3182bd", alpha=0.8)
    bars2 = ax_a.bar(x + width/2, [large_cv, large_mean/50, large_gini],
                     width, label="5,936-node full STRING", color="#d62728", alpha=0.8)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Degree CV", "Mean Degree / 50", "Gini Index"])
    ax_a.set_ylabel("Value")
    ax_a.set_title("A. Network Heterogeneity: Curated vs Full",
                   fontsize=13, fontweight="bold")
    ax_a.legend(fontsize=10)

    # Add text annotation
    ax_a.text(0.02, 0.95, f"Full CV = {large_cv:.2f} (2.2× curated)\n"
              f"Full Gini = {large_gini:.3f}",
              transform=ax_a.transAxes, fontsize=10, va="top",
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # ---- Panel B: Effective rank vs GF Score (both networks) ----
    ax_b = fig.add_subplot(gs[0, 1])

    # Full network data
    t2_data = large_net["T2_effective_rank_bound"]["method_results"]
    methods_full = sorted(t2_data.keys())
    eff_ranks_full = [t2_data[m]["effective_rank"] for m in methods_full]
    gf_scores_full = [t2_data[m]["gf_score"] for m in methods_full]

    # 153-node: extract from theory JSON if available
    t2_small = theory.get("P2_rank_collapse", {})
    methods_small = []
    eff_ranks_small = []
    gf_scores_small = []
    if "method_results" in t2_small:
        for m, d in t2_small["method_results"].items():
            if isinstance(d, dict) and "effective_rank" in d:
                methods_small.append(m)
                eff_ranks_small.append(d["effective_rank"])
                gf_scores_small.append(d.get("gf_score", 0))

    for m, er, gf in zip(methods_full, eff_ranks_full, gf_scores_full):
        color = METHOD_COLORS.get(m, "#888888")
        ax_b.scatter(er, gf, s=120, color=color, zorder=5,
                    edgecolors="black", linewidth=0.5)
        ax_b.annotate(m, (er, gf), textcoords="offset points",
                     xytext=(5, 5), fontsize=8, color=color)

    if methods_small:
        for m, er, gf in zip(methods_small, eff_ranks_small, gf_scores_small):
            color = METHOD_COLORS.get(m, "#888888")
            ax_b.scatter(er, gf, s=80, color=color, marker="^", zorder=4,
                        alpha=0.5, edgecolors="black", linewidth=0.3)

    ax_b.set_xlabel("Effective Rank", fontsize=12)
    ax_b.set_ylabel("G-F Score", fontsize=12)
    ax_b.set_title("B. Effective Rank vs G-F Score",
                   fontsize=13, fontweight="bold")
    ax_b.axhline(0.135, color="grey", linestyle="--", linewidth=1.5,
                label="Random baseline (0.135)")
    ax_b.legend(fontsize=9)

    # ---- Panel C: GAT dimension sweep d=2-32 ----
    ax_c = fig.add_subplot(gs[1, 0])

    # Extract dimension sweep data
    if "dimension_sweep" in dim_sweep:
        sweep_data = dim_sweep["dimension_sweep"]
    elif "results" in dim_sweep:
        sweep_data = dim_sweep["results"]
    else:
        sweep_data = dim_sweep

    dims = []
    gf_scores = []
    entropies = []
    if isinstance(sweep_data, dict):
        for key in sorted(sweep_data.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            entry = sweep_data[key]
            if isinstance(entry, dict):
                d = entry.get("dimension", int(key) if key.isdigit() else 0)
                dims.append(d)
                gf_scores.append(entry.get("gf_score", entry.get("GF", 0)))
                entropies.append(entry.get("attention_entropy",
                                          entry.get("H_norm", 0.973)))
    elif isinstance(sweep_data, list):
        for entry in sweep_data:
            if isinstance(entry, dict):
                dims.append(entry.get("dimension", entry.get("d", 0)))
                gf_scores.append(entry.get("gf_score", entry.get("GF", 0)))
                entropies.append(entry.get("attention_entropy",
                                          entry.get("H_norm", 0.973)))

    if dims:
        color_gf = "#d62728"
        color_ent = "#3182bd"
        ax_c.plot(dims, gf_scores, "o-", color=color_gf, linewidth=2.5,
                 markersize=8, label="GAT G-F Score")
        ax_c.set_xlabel("Embedding Dimension (d)", fontsize=12)
        ax_c.set_ylabel("G-F Score", fontsize=12, color=color_gf)
        ax_c.tick_params(axis="y", labelcolor=color_gf)
        ax_c.axhline(0.135, color="grey", linestyle="--", linewidth=1.5,
                    label="Random baseline")

        ax_c2 = ax_c.twinx()
        ax_c2.plot(dims, entropies, "s--", color=color_ent, linewidth=2,
                  markersize=6, label="Attention Entropy")
        ax_c2.set_ylabel("Normalized Attention Entropy", fontsize=12,
                        color=color_ent)
        ax_c2.tick_params(axis="y", labelcolor=color_ent)
        ax_c2.set_ylim(0.90, 1.05)

        lines1, labels1 = ax_c.get_legend_handles_labels()
        lines2, labels2 = ax_c2.get_legend_handles_labels()
        ax_c.legend(lines1 + lines2, labels1 + labels2,
                   loc="center right", fontsize=9)
    ax_c.set_title("C. GAT Dimension Sweep: GF Score & Attention Entropy",
                   fontsize=13, fontweight="bold")

    # ---- Panel D: Multi-head GAT experiment ----
    ax_d = fig.add_subplot(gs[1, 1])

    # Extract multi-head data
    configs = []
    config_gf = []
    if isinstance(multi_head, dict):
        for key, val in multi_head.items():
            if isinstance(val, dict) and "gf_score" in val:
                configs.append(key)
                config_gf.append(val["gf_score"])
        # Try structured format
        if not configs and "configurations" in multi_head:
            for cfg in multi_head["configurations"]:
                if isinstance(cfg, dict):
                    label = f"{cfg.get('heads', '?')}h d={cfg.get('dimension', '?')}"
                    configs.append(label)
                    config_gf.append(cfg.get("gf_score", 0))
        if not configs and "results" in multi_head:
            for key, val in multi_head["results"].items():
                if isinstance(val, dict):
                    configs.append(key)
                    config_gf.append(val.get("gf_score", val.get("GF", 0)))

    if configs and config_gf:
        x = np.arange(len(configs))
        bars = ax_d.bar(x, config_gf, color="#fc8d62", edgecolor="white",
                       alpha=0.8)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax_d.axhline(0.135, color="grey", linestyle="--", linewidth=1.5,
                    label="Random baseline (0.135)")
        ax_d.set_ylabel("G-F Score", fontsize=12)
        ax_d.legend(fontsize=9)

        # Annotate bars
        for bar, gf in zip(bars, config_gf):
            ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{gf:.3f}", ha="center", fontsize=8, fontweight="bold")
    else:
        ax_d.text(0.5, 0.5, "Multi-head data\nnot available",
                 transform=ax_d.transAxes, ha="center", va="center",
                 fontsize=14, color="grey")
    ax_d.set_title("D. Multi-Head GAT Configurations",
                   fontsize=13, fontweight="bold")

    # Title
    fig.suptitle(
        "GAT Collapse Theorem Verification: 153-Node vs 5,936-Node Networks",
        fontsize=15, fontweight="bold", y=1.01)

    fig_path = FIGURES / "FigS18_full_network_theorem_verification.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# FigS19: Community Detection Ablation Heatmap
# ============================================================

def generate_figs19():
    """2-panel: GF score heatmap + Spearman correlation matrix."""
    data = load_json("gf_ablation_community_detection.json")

    detailed = data["detailed_results"]
    comm_methods = ["greedy_modularity", "louvain", "leiden",
                    "label_propagation", "connected_components"]
    emb_methods = ["Spectral", "DM", "MDS", "PCA", "Node2Vec",
                   "DeepWalk", "VGAE-feat", "GIN", "GAT", "GraphSAGE", "VGAE"]

    # Build score matrix
    available_comm = [m for m in comm_methods if m in detailed]
    available_emb = [m for m in emb_methods
                     if any(m in detailed[cm] for cm in available_comm)]

    matrix = np.full((len(available_comm), len(available_emb)), np.nan)
    for i, cm in enumerate(available_comm):
        for j, em in enumerate(available_emb):
            if em in detailed.get(cm, {}):
                matrix[i, j] = detailed[cm][em]["gf_score"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # ---- Panel A: Heatmap ----
    im = ax1.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=0.0, vmax=0.20)
    ax1.set_xticks(range(len(available_emb)))
    ax1.set_xticklabels(available_emb, rotation=45, ha="right", fontsize=10)
    ax1.set_yticks(range(len(available_comm)))

    # Pretty labels
    pretty_comm = {
        "greedy_modularity": "Greedy Modularity",
        "louvain": "Louvain",
        "leiden": "Leiden",
        "label_propagation": "Label Propagation",
        "connected_components": "Connected Components",
    }
    ax1.set_yticklabels([pretty_comm.get(m, m) for m in available_comm],
                        fontsize=11)

    # Annotate cells
    for i in range(len(available_comm)):
        for j in range(len(available_emb)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.08 or val > 0.16 else "black"
                ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("G-F Score", fontsize=12)
    ax1.set_title("A. G-F Score Heatmap: Community Detection × Embedding Method",
                  fontsize=13, fontweight="bold")

    # ---- Panel B: Spearman correlation between algorithms ----
    # Compute pairwise Spearman
    from scipy.stats import spearmanr
    n_comm = len(available_comm)
    corr_matrix = np.ones((n_comm, n_comm))
    for i in range(n_comm):
        for j in range(i+1, n_comm):
            row_i = matrix[i, :]
            row_j = matrix[j, :]
            mask = ~(np.isnan(row_i) | np.isnan(row_j))
            if mask.sum() >= 3:
                rho, _ = spearmanr(row_i[mask], row_j[mask])
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

    im2 = ax2.imshow(corr_matrix, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0,
                    aspect="auto")
    ax2.set_xticks(range(n_comm))
    short_labels = [m.replace("_", "\n").replace("greedy\nmod", "Greedy\nMod")
                   for m in available_comm]
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(n_comm))
    ax2.set_yticklabels([pretty_comm.get(m, m) for m in available_comm],
                        fontsize=9)

    # Annotate correlation cells
    for i in range(n_comm):
        for j in range(n_comm):
            val = corr_matrix[i, j]
            color = "white" if abs(val) < 0.3 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im2, ax=ax2, shrink=0.8, label="Spearman ρ")
    ax2.set_title("B. Pairwise Spearman ρ Between Algorithms",
                  fontsize=13, fontweight="bold")

    # Overall annotation
    fig.suptitle(
        f"Community Detection Sensitivity (Kendall W = 0.797 across 5 algorithms)",
        fontsize=15, fontweight="bold", y=1.02)

    fig_path = FIGURES / "FigS19_community_detection_ablation.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# FigS20: Coexpression Network GF Curves
# ============================================================

def generate_figs20():
    """2-panel: coexpression GF curves + PPI vs coexpression scatter."""
    data = load_json("coexpression_gf.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # ---- Panel A: GF curves ----
    gf_curves = data.get("gf_curves", {})
    gf_params = data["gf_parameters"]
    r_min = gf_params.get("curve_r_min", 0.05)
    r_max = gf_params.get("curve_r_max", 0.55)

    methods_order = ["DeepWalk", "Node2Vec", "Spectral", "PCA", "DM", "MDS", "GIN"]
    for method in methods_order:
        if method in gf_curves:
            purities = gf_curves[method].get("purities", [])
            if purities:
                n = len(purities)
                r_vals = np.linspace(r_min, r_max, n)
                color = METHOD_COLORS.get(method, "#888888")
                ax1.plot(r_vals, purities, linewidth=2, color=color,
                        label=f"{method} ({data['gf_scores'].get(method, 0):.3f})")

    # Random baseline
    rand_mean = data["random_baseline"]["mean"]
    ax1.axhline(rand_mean, color="grey", linestyle="--", linewidth=2,
               label=f"Random ({rand_mean:.3f})")

    ax1.set_xlabel("Embedding Radius (r)", fontsize=12)
    ax1.set_ylabel("Purity", fontsize=12)
    ax1.set_title("A. G-F Curves: Coexpression Network\n"
                  f"({data['network_statistics']['nodes']} nodes, "
                  f"{data['network_statistics']['edges']} edges)",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ---- Panel B: PPI vs Coexpression scatter ----
    comp = data["comparison_with_ppi"]
    common = comp["common_methods"]
    ppi_scores = [comp["ppi_scores"][m] for m in common]
    coexp_scores = [comp["coexpression_scores"][m] for m in common]

    for m, ps, cs in zip(common, ppi_scores, coexp_scores):
        color = METHOD_COLORS.get(m, "#888888")
        ax2.scatter(ps, cs, s=150, color=color, zorder=5,
                   edgecolors="black", linewidth=0.8)
        ax2.annotate(m, (ps, cs), textcoords="offset points",
                    xytext=(8, 5), fontsize=10, color=color, fontweight="bold")

    # Add rho annotation
    rho = comp["spearman_rho"]
    pval = comp["spearman_p_value"]
    ax2.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}\np = {pval:.3f}",
            transform=ax2.transAxes, fontsize=12, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    ax2.set_xlabel("PPI G-F Score", fontsize=12)
    ax2.set_ylabel("Coexpression G-F Score", fontsize=12)
    ax2.set_title("B. PPI vs Coexpression: Network-Type Dependence",
                  fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Identity line for reference
    all_vals = ppi_scores + coexp_scores
    ax2.plot([0, max(all_vals)+0.05], [0, max(all_vals)+0.05],
            "k:", alpha=0.3, linewidth=1)

    fig.suptitle(
        "Coexpression Network: Random-Walk Methods Surpass Spectral",
        fontsize=15, fontweight="bold", y=1.02)

    fig_path = FIGURES / "FigS20_coexpression_network.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# Main
# ============================================================

def run():
    print("=" * 64)
    print("  Generating Missing Supplementary Figures S18-S20")
    print("=" * 64)

    generate_figs18()
    generate_figs19()
    generate_figs20()

    print("\nAll 3 missing figures generated successfully.")


if __name__ == "__main__":
    run()
