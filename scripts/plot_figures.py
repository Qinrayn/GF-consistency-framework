#!/usr/bin/env python3
"""
plot_figures.py
Generate all paper figures from result JSON files.
Figures 1-6 + supplementary figures.
"""

import argparse
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_results_dir, get_figures_dir


# ------------------------------------------------------------------
# Figure 1: G-F curves for six embedding strategies (purity + modularity)
# ------------------------------------------------------------------
def plot_figure1_gf_curves(results_dir, figures_dir):
    """Figure 1: G-F curves for six embedding strategies."""
    # Try pkl first (contains all methods), then json
    pkl_file = results_dir / "gf_curves_200pts.pkl"
    json_file = results_dir / "gf_curves_200pts.json"

    data = None
    if pkl_file.exists():
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
    elif json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
    else:
        print("Figure 1: no gf_curves data found, skipping")
        return

    r = np.array(data["r"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    methods = ['DM', 'MDS', 'Spectral', 'DeepWalk', 'Node2Vec', 'VGAE']
    colors = ['#4E79A7', '#E15759', '#59A14F', '#F28E2B', '#B07AA1', '#76B7B2']
    markers = ['o', 's', '^', 'v', 'D', 'x']
    # Plot every Nth marker to avoid clutter
    marker_every = max(1, len(r) // 15)

    for method, color, marker in zip(methods, colors, markers):
        purity_key = f"{method}_purity"
        mod_key = f"{method}_modularity"
        if purity_key not in data:
            print(f"  {purity_key} not found, skipping")
            continue
        # Line + sparse markers
        ax1.plot(r, data[purity_key], '-', color=color, linewidth=1.5, label=method)
        ax1.plot(r[::marker_every], np.array(data[purity_key])[::marker_every],
                 marker, color=color, markersize=4, linewidth=0)
        ax2.plot(r, data[mod_key], '-', color=color, linewidth=1.5, label=method)
        ax2.plot(r[::marker_every], np.array(data[mod_key])[::marker_every],
                 marker, color=color, markersize=4, linewidth=0)

    # Add random baseline if available
    if "random_baseline_purity" in data:
        ax1.axhline(y=np.mean(data["random_baseline_purity"]),
                     color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    ax1.set_xlabel('Distance threshold $r$', fontsize=11)
    ax1.set_ylabel('Functional purity', fontsize=11)
    ax1.set_title('(A) G-F Curves: Purity', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='best', framealpha=0.9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Distance threshold $r$', fontsize=11)
    ax2.set_ylabel('Modularity $Q$', fontsize=11)
    ax2.set_title('(B) G-F Curves: Modularity', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best', framealpha=0.9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig1_GF_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved")


# ------------------------------------------------------------------
# Figure 2: PCA control (DM vs PCA)
# ------------------------------------------------------------------
def plot_figure2_pca_control(results_dir, figures_dir):
    """Figure 2: PCA control experiment (DM vs PCA)."""
    # Try loading from the unified G-F curves file (contains PCA data)
    data = None
    pkl_file = results_dir / "gf_curves_200pts.pkl"
    json_file = results_dir / "gf_curves_200pts.json"

    if pkl_file.exists():
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
    elif json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
    else:
        print("Figure 2: no G-F curves data found, skipping")
        return

    if "DM_purity" not in data or "PCA_purity" not in data:
        print("Figure 2: DM or PCA purity data missing, skipping")
        return

    r = np.array(data["r"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r, data["DM_purity"], '-', color='#4E79A7', label='DM (Diffusion Map)', linewidth=2)
    ax.plot(r[::15], np.array(data["DM_purity"])[::15], 'o', color='#4E79A7', markersize=5)
    ax.plot(r, data["PCA_purity"], '--', color='#E15759', label='PCA (control)', linewidth=2)
    ax.plot(r[::15], np.array(data["PCA_purity"])[::15], 's', color='#E15759', markersize=5)

    # Add G-F integration interval shading
    ax.axvspan(0.05, 0.422, alpha=0.08, color='gray', label='G-F interval [0.05, 0.422]')

    ax.set_xlabel('Distance threshold r')
    ax.set_ylabel('Functional purity')
    ax.set_title('PCA Control: DM vs PCA')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig2_PCA_control.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved")


# ------------------------------------------------------------------
# Figure 3: Robustness across 10 random subsets
# ------------------------------------------------------------------
def plot_figure3_robustness(results_dir, figures_dir):
    """Figure 3: Robustness across 10 random subsets."""
    robust_file = results_dir / "subset_summary.json"
    if not robust_file.exists():
        print("Figure 3: subset_summary.json not found, skipping")
        return

    with open(robust_file) as f:
        data = json.load(f)

    r = np.array(data["r_values"])
    dm_mean = np.array(data["mean_purity_dm"])
    dm_std = np.array(data["std_purity_dm"])
    mds_mean = np.array(data["mean_purity_mds"])
    mds_std = np.array(data["std_purity_mds"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(r, dm_mean, 'b-', label='DM', linewidth=2)
    ax.fill_between(r, dm_mean - dm_std, dm_mean + dm_std,
                    alpha=0.2, color='blue', label='DM \u00b11 SD')

    ax.plot(r, mds_mean, 'r-', label='MDS', linewidth=2)
    ax.fill_between(r, mds_mean - mds_std, mds_mean + mds_std,
                    alpha=0.2, color='red', label='MDS \u00b11 SD')

    ax.set_xlabel('Distance threshold r')
    ax.set_ylabel('Functional purity')
    ax.set_title('Robustness Across 10 Random Subsets')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig3_subset_robustness.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved")


# ------------------------------------------------------------------
# Figure 4: Full network validation
# ------------------------------------------------------------------
def plot_figure4_full_network(results_dir, figures_dir):
    """Figure 4: Full network validation."""
    full_file = results_dir / "full_network_validation.json"
    if not full_file.exists():
        print("Figure 4: full_network_validation.json not found, skipping")
        return

    with open(full_file) as f:
        data = json.load(f)

    r = np.array(data["r"])

    fig, ax = plt.subplots(figsize=(8, 5))

    color_map = {"DM": "steelblue", "Node2Vec": "brown",
                 "MDS": "coral", "VGAE": "pink"}

    for key in data:
        if key.endswith("_purity"):
            method = key.replace("_purity", "")
            color = color_map.get(method, "gray")
            ax.plot(r, data[key], 'o-', color=color, label=method, markersize=4)

    n_full = data.get("n_nodes_full", "?")
    n_eval = data.get("n_nodes_evaluated", "?")
    ax.set_xlabel('Distance threshold r')
    ax.set_ylabel('Functional purity')
    ax.set_title(f'Full Network Validation ({n_full} nodes, {n_eval} evaluated)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig4_full_network.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 4 saved")


# ------------------------------------------------------------------
# Figure 5: G-F Score ranking bar chart
# ------------------------------------------------------------------
def plot_figure5_gf_scores(results_dir, figures_dir):
    """Figure 5: G-F Score ranking."""
    scores_file = results_dir / "gf_scores.json"
    if not scores_file.exists():
        print("Figure 5: gf_scores.json not found, skipping")
        return

    with open(scores_file) as f:
        data = json.load(f)

    # Handle both key naming conventions
    if "scores" in data:
        scores = data["scores"]
    elif "scores_paper_interval" in data:
        scores = data["scores_paper_interval"]
    else:
        print("Figure 5: no scores key found in gf_scores.json")
        return

    methods = list(scores.keys())
    values = list(scores.values())

    # Sort by score descending
    sorted_idx = np.argsort(values)[::-1]
    methods = [methods[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use a distinct color palette — top method highlighted, rest graded
    palette = ['#4E79A7', '#76B7B2', '#59A14F', '#F28E2B', '#E15759', '#B07AA1', '#EDC948', '#FF9DA7']
    colors = [palette[i] if i < len(palette) else 'lightgray' for i in range(len(methods))]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)

    # Get interval for title
    if "unified_interval" in data:
        interval = data["unified_interval"]
    elif "unified_interval_paper" in data:
        interval = data["unified_interval_paper"]
    else:
        interval = [0.05, 0.422]

    ax.set_ylabel('G-F Score')
    ax.set_title(f'G-F Score Ranking (Interval: [{interval[0]:.3f}, {interval[1]:.3f}])')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig5_GF_scores.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 5 saved")


# ------------------------------------------------------------------
# Figure 6: Human interactome cross-species validation
# ------------------------------------------------------------------
def plot_figure6_human(results_dir, figures_dir):
    """Figure 6: Human interactome validation."""
    # Look in multiple possible locations
    human_file = results_dir / "human_ppi_results.json"
    if not human_file.exists():
        human_file = results_dir / "human_validation_results.json"
    if not human_file.exists():
        # Check human_validation directory
        hv_dir = results_dir.parent / "human_validation"
        for name in ["human_ppi_results.json", "human_validation_results.json",
                      "human_gf_results.json"]:
            if (hv_dir / name).exists():
                human_file = hv_dir / name
                break

    if not human_file.exists():
        print("Figure 6: Human validation results not found, skipping")
        return

    with open(human_file) as f:
        data = json.load(f)

    r = np.array(data["r"])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Support both key naming conventions from human_validation scripts
    n2v_key = None
    for candidate in ("Node2Vec_purity", "Node2Vec_cleaned_purity"):
        if candidate in data:
            n2v_key = candidate
            break

    if "DM_purity" in data:
        ax.plot(r, data["DM_purity"], 'o-', color='steelblue',
                label='DM', markersize=5)
    if n2v_key is not None:
        ax.plot(r, data[n2v_key], 's-', color='brown',
                label='Node2Vec (cleaned)', markersize=5)

    n_nodes = data.get("n_nodes", "?")
    ax.set_xlabel('Distance threshold r')
    ax.set_ylabel('Functional purity')
    ax.set_title(f'Cross-Species Validation: Human ({n_nodes} nodes)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig6_human_validation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 6 saved")


# ------------------------------------------------------------------
# Supplementary: 30 vs 200 point sampling comparison
# ------------------------------------------------------------------
def plot_supplementary_30vs200(results_dir, figures_dir):
    """Supplementary: 30 vs 200 point sampling comparison."""
    density_file = results_dir / "sampling_density_comparison.json"
    if not density_file.exists():
        print("Supplementary: sampling_density_comparison.json not found, skipping")
        return

    with open(density_file) as f:
        data = json.load(f)

    methods = list(data.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        mdata = data[method]
        for grid_name in ["30", "200"]:
            if grid_name in mdata:
                ax.bar(grid_name, mdata[grid_name]["plateau_width"],
                       color='steelblue' if grid_name == "200" else 'coral')
        ax.set_title(method)
        ax.set_ylabel('Plateau width W')

    plt.suptitle('Plateau Width: 30-point vs 200-point Grid')
    plt.tight_layout()
    plt.savefig(figures_dir / "comparison_30vs200_points.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Supplementary 30vs200 comparison saved")


# ------------------------------------------------------------------
# Figure 7: Plateau width comparison
# ------------------------------------------------------------------
def plot_figure7_plateau_width(results_dir, figures_dir):
    """Figure 7: Plateau width comparison across methods."""
    plateau_file = results_dir / "plateau_width_v3_200pts.csv"
    if not plateau_file.exists():
        print("Figure 7: plateau_width_v3_200pts.csv not found, skipping")
        return

    import csv
    methods, widths = [], []
    with open(plateau_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            methods.append(row.get("Method", row.get("method", "")))
            widths.append(float(row.get("W", 0)))

    if not methods:
        print("Figure 7: no plateau data found, skipping")
        return

    # Sort by width descending
    sorted_idx = np.argsort(widths)[::-1]
    methods = [methods[i] for i in sorted_idx]
    widths = [widths[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    # Color gradient based on width value
    cmap = plt.cm.YlGnBu
    max_w = max(widths) if max(widths) > 0 else 1.0
    colors = [cmap(0.3 + 0.7 * w / max_w) if w > 0 else (0.85, 0.85, 0.85, 1.0) for w in widths]
    bars = ax.bar(methods, widths, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Plateau Width W')
    ax.set_title('Plateau Width Across Embedding Methods (threshold = 0.5)')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')

    for bar, val in zip(bars, widths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig7_plateau_width.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 7 saved")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", type=str, help="Results directory")
    parser.add_argument("--figures-dir", type=str, help="Figures directory")
    parser.add_argument("--figures", nargs="+",
                        default=["1", "2", "3", "4", "5", "6", "7", "S"],
                        help="Figures to generate (1-6, S for supplementary)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else get_results_dir()
    figures_dir = Path(args.figures_dir) if args.figures_dir else get_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory: {results_dir}")
    print(f"Figures directory: {figures_dir}\n")

    figure_map = {
        "1": plot_figure1_gf_curves,
        "2": plot_figure2_pca_control,
        "3": plot_figure3_robustness,
        "4": plot_figure4_full_network,
        "5": plot_figure5_gf_scores,
        "6": plot_figure6_human,
        "7": plot_figure7_plateau_width,
        "S": plot_supplementary_30vs200,
    }

    for fig_num in args.figures:
        if fig_num in figure_map:
            try:
                figure_map[fig_num](results_dir, figures_dir)
            except Exception as e:
                print(f"Error generating Figure {fig_num}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Unknown figure: {fig_num}")

    print("\nAll requested figures generated")


if __name__ == "__main__":
    main()
