#!/usr/bin/env python3
"""
cross_species_consistency.py
Step 30: Cross-Species Rank Consistency Analysis.

Compares G-F Score method rankings between yeast (curated 153) and human
(~14,679-node STRING network) to test whether embedding quality is a
method-intrinsic property that transcends species boundaries.

Analyses:
  1. Spearman rank correlation between yeast and human G-F Scores
     for the 6 shared methods (DM, MDS, Spectral, DeepWalk, Node2Vec, VGAE).
  2. Kendall's W (coefficient of concordance) for rank stability.
  3. Permutation test for rank concordance significance.
  4. Rank shift analysis: which methods improve or degrade cross-species.

Output:
  - results/cross_species_consistency.json
  - figures/Fig17_cross_species_rank_consistency.png
"""
from __future__ import annotations

import sys
import json
import os
import numpy as np
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import SEED, get_results_dir, get_project_root

# Shared methods between yeast and human analyses
SHARED_METHODS = ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE",
                  "PCA", "VGAE-feat", "GraphSAGE", "GAT", "GIN"]


def load_yeast_scores(results_dir):
    """Load yeast G-F Scores from gf_scores.json + gnn_gf_scores.json."""
    scores = {}

    # Classical + PCA/VGAE-feat
    gf_file = results_dir / "gf_scores.json"
    if gf_file.exists():
        with open(gf_file, encoding="utf-8") as f:
            data = json.load(f)
        scores.update(data.get("scores_paper_interval", data.get("scores", {})))

    # GNN methods
    gnn_file = results_dir / "gnn_gf_scores.json"
    if gnn_file.exists():
        with open(gnn_file, encoding="utf-8") as f:
            gnn_data = json.load(f)
        if "gf_scores" in gnn_data:
            scores.update(gnn_data["gf_scores"])

    return scores


def load_human_scores(results_dir):
    """Load human G-F Scores from human_gf_scores_extended.json (11 methods)
    or fall back to human_gf_scores.json (6 methods)."""
    scores = {}
    # Prefer extended (11-method) results
    ext_file = results_dir / "human_gf_scores_extended.json"
    if ext_file.exists():
        with open(ext_file, encoding="utf-8") as f:
            data = json.load(f)
        scores.update(data.get("scores", {}))
        return scores
    # Fallback to original 6-method results
    human_file = results_dir / "human_gf_scores.json"
    if human_file.exists():
        with open(human_file, encoding="utf-8") as f:
            data = json.load(f)
        scores.update(data.get("scores", {}))
    return scores


def spearman_correlation(x, y):
    """Compute Spearman rank correlation between two arrays."""
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def kendalls_w(rankings):
    """Compute Kendall's W (coefficient of concordance).

    Parameters
    ----------
    rankings : list of lists
        Each inner list is a ranking (1 = best) for one species/condition.

    Returns
    -------
    W : float
        Kendall's W in [0, 1].  1 = perfect agreement.
    """
    k = len(rankings)        # number of rankings (species)
    n = len(rankings[0])     # number of items ranked

    # Sum of ranks for each item
    rank_sums = np.sum(rankings, axis=0)
    mean_rank_sum = np.mean(rank_sums)
    S = np.sum((rank_sums - mean_rank_sum) ** 2)

    W = (12.0 * S) / (k ** 2 * n * (n ** 2 - 1))
    return float(W)


def ranks_from_scores(score_dict, methods, ascending=False):
    """Convert score dict to rank array.

    Parameters
    ----------
    ascending : bool
        If False (default), higher score = rank 1.
    """
    scores = [score_dict.get(m, np.nan) for m in methods]
    arr = np.array(scores)
    # Rank: argsort of argsort gives ranks (0-based), +1 for 1-based
    if ascending:
        ranks = arr.argsort().argsort() + 1
    else:
        ranks = (-arr).argsort().argsort() + 1
    return ranks.tolist(), scores


def permutation_test_concordance(ranks_species1, ranks_species2, n_perm=10000, rng=None):
    """Permutation test for rank concordance between two species.

    H0: rankings are independent.
    Test statistic: Spearman rho.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    observed_rho, _ = spearman_correlation(ranks_species1, ranks_species2)

    n = len(ranks_species1)
    perm_rhos = np.zeros(n_perm)
    for i in range(n_perm):
        perm_ranks = rng.permutation(ranks_species2)
        perm_rhos[i], _ = spearman_correlation(ranks_species1, perm_ranks)

    p_perm = np.mean(np.abs(perm_rhos) >= abs(observed_rho))
    return float(observed_rho), float(p_perm), perm_rhos


def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = get_project_root() / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load scores ----
    print("Loading G-F Scores...")
    yeast_scores = load_yeast_scores(results_dir)
    human_scores = load_human_scores(results_dir)

    print(f"  Yeast methods: {list(yeast_scores.keys())}")
    print(f"  Human methods: {list(human_scores.keys())}")

    # ---- Identify shared methods ----
    shared = [m for m in SHARED_METHODS if m in yeast_scores and m in human_scores]
    print(f"  Shared methods ({len(shared)}): {shared}")

    if len(shared) < 3:
        print("ERROR: Too few shared methods for meaningful analysis.")
        return

    # ---- Compute ranks and correlation ----
    yeast_ranks, yeast_vals = ranks_from_scores(yeast_scores, shared, ascending=False)
    human_ranks, human_vals = ranks_from_scores(human_scores, shared, ascending=False)

    rho, p_spearman = spearman_correlation(yeast_vals, human_vals)
    rho_obs, p_perm, perm_rhos = permutation_test_concordance(
        yeast_ranks, human_ranks, n_perm=10000, rng=rng
    )

    print(f"\nCross-species Spearman rho = {rho:.4f} (P = {p_spearman:.4f})")
    print(f"Permutation test: rho = {rho_obs:.4f} (P_perm = {p_perm:.4f})")

    # ---- Kendall's W ----
    W = kendalls_w([yeast_ranks, human_ranks])
    print(f"Kendall's W = {W:.4f}")

    # ---- Rank shift analysis ----
    rank_shifts = {}
    for i, m in enumerate(shared):
        shift = human_ranks[i] - yeast_ranks[i]
        rank_shifts[m] = {
            "yeast_rank": yeast_ranks[i],
            "human_rank": human_ranks[i],
            "shift": shift,
            "yeast_score": float(yeast_vals[i]),
            "human_score": float(human_vals[i]),
        }
        direction = "^" if shift < 0 else ("v" if shift > 0 else "->")
        print(f"  {m:12s}  yeast#{yeast_ranks[i]} -> human#{human_ranks[i]}  "
              f"({direction}{abs(shift)})")

    # ---- Normalised score comparison ----
    # Z-score normalise within each species for fair comparison
    yeast_arr = np.array(yeast_vals)
    human_arr = np.array(human_vals)
    yeast_z = (yeast_arr - yeast_arr.mean()) / (yeast_arr.std() + 1e-10)
    human_z = (human_arr - human_arr.mean()) / (human_arr.std() + 1e-10)
    z_correlation, z_p = spearman_correlation(yeast_z, human_z)

    # ---- Save results ----
    output = {
        "shared_methods": shared,
        "n_shared": len(shared),
        "yeast_gf_scores": {m: float(yeast_scores[m]) for m in shared},
        "human_gf_scores": {m: float(human_scores[m]) for m in shared},
        "yeast_ranks": {m: yeast_ranks[i] for i, m in enumerate(shared)},
        "human_ranks": {m: human_ranks[i] for i, m in enumerate(shared)},
        "spearman_correlation": {
            "rho": rho,
            "p_value": p_spearman,
            "n": len(shared),
        },
        "permutation_test": {
            "rho": rho_obs,
            "p_perm": p_perm,
            "n_permutations": 10000,
        },
        "kendalls_w": W,
        "z_score_spearman": {
            "rho": z_correlation,
            "p_value": z_p,
        },
        "rank_shifts": rank_shifts,
        "interpretation": _interpret(rho, W, p_spearman),
    }

    output_file = results_dir / "cross_species_consistency.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_file}")

    # ---- Generate figure ----
    _generate_figure(
        shared, yeast_ranks, human_ranks, yeast_vals, human_vals,
        rho, p_spearman, W, rank_shifts, figures_dir
    )


def _interpret(rho, W, p):
    """Generate a plain-language interpretation."""
    strength = "strong" if abs(rho) > 0.7 else ("moderate" if abs(rho) > 0.4 else "weak")
    sig = "significant" if p < 0.05 else "not significant"
    concordance = "high" if W > 0.7 else ("moderate" if W > 0.4 else "low")
    return (
        f"Cross-species rank consistency is {strength} (ρ={rho:.3f}, {sig} at α=0.05) "
        f"with {concordance} concordance (W={W:.3f}). "
        f"This {'supports' if abs(rho) > 0.5 else 'partially supports'} the hypothesis "
        f"that embedding quality is a method-intrinsic property."
    )


def _generate_figure(methods, yeast_ranks, human_ranks, yeast_vals, human_vals,
                     rho, p_spearman, W, rank_shifts, figures_dir):
    """Generate Fig17: Cross-species rank consistency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1.2, 1], wspace=0.35)

    # (A) Paired rank comparison (bump chart)
    ax1 = fig.add_subplot(gs[0, 0])
    n = len(methods)
    x_positions = [1, 2]
    colours = plt.cm.Set2(np.linspace(0, 1, n))

    for i, m in enumerate(methods):
        ax1.plot(x_positions, [yeast_ranks[i], human_ranks[i]],
                 'o-', color=colours[i], linewidth=2, markersize=8, label=m)

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(["Yeast\n(153 nodes)", "Human\n(~14,679 nodes)"])
    ax1.set_ylabel("Method Rank (1 = best)")
    ax1.set_ylim(n + 0.5, 0.5)  # Invert: rank 1 at top
    ax1.set_title(f"(A) Rank Trajectory  (W={W:.3f})")
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(True, alpha=0.3)

    # (B) Scatter plot: yeast vs human G-F Scores
    ax2 = fig.add_subplot(gs[0, 1])
    yeast_arr = np.array(yeast_vals)
    human_arr = np.array(human_vals)

    # Normalise for visual comparison
    yeast_norm = (yeast_arr - yeast_arr.min()) / (yeast_arr.max() - yeast_arr.min() + 1e-10)
    human_norm = (human_arr - human_arr.min()) / (human_arr.max() - human_arr.min() + 1e-10)

    ax2.scatter(yeast_norm, human_norm, s=100, c=colours, edgecolors="black",
                linewidths=0.5, zorder=5)
    for i, m in enumerate(methods):
        ax2.annotate(m, (yeast_norm[i], human_norm[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Fit line
    z = np.polyfit(yeast_norm, human_norm, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax2.plot(x_line, p_line(x_line), "--", color="gray", alpha=0.6, linewidth=1)

    p_str = f"P = {p_spearman:.3f}" if p_spearman >= 0.001 else "P < 0.001"
    ax2.set_xlabel("Yeast G-F Score (normalised)")
    ax2.set_ylabel("Human G-F Score (normalised)")
    ax2.set_title(f"(B) Score Correlation  (ρ={rho:.3f}, {p_str})")
    ax2.grid(True, alpha=0.3)

    # (C) Rank shift waterfall
    ax3 = fig.add_subplot(gs[0, 2])
    shift_values = [rank_shifts[m]["shift"] for m in methods]
    bar_colors = ["#e74c3c" if s > 0 else "#2ecc71" if s < 0 else "#95a5a6"
                  for s in shift_values]
    bars = ax3.barh(methods, shift_values, color=bar_colors, edgecolor="black",
                    linewidth=0.5)
    ax3.set_xlabel("Rank Shift (Human − Yeast)")
    ax3.set_title("(C) Cross-Species Rank Shift")
    ax3.axvline(x=0, color="black", linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis="x")

    for bar, s in zip(bars, shift_values):
        label = f"+{s}" if s > 0 else str(s)
        ax3.text(bar.get_width() + 0.05 * np.sign(s) if s != 0 else 0.05,
                 bar.get_y() + bar.get_height() / 2,
                 label, va="center", ha="left" if s >= 0 else "right", fontsize=8)

    fig.suptitle("Fig 17 — Cross-Species Rank Consistency (Yeast vs Human)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig_path = figures_dir / "Fig17_cross_species_rank_consistency.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
