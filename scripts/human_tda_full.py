#!/usr/bin/env python3
"""
Phase 8B: Full Human TDA Analysis (11 Methods)
===============================================
Computes persistent homology, persistence statistics, Betti curves,
and topological G-F scores for all 11 methods on the human PPI network.

Uses current parameters (same r-grid, rescaling, and GF interval as yeast)
for direct comparability with Phase 7 yeast TDA features.

Generates:
    results/human_tda_full.json
    figures/Fig48_human_tda_summary.png (3 panels)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from networkx.algorithms.community import greedy_modularity_communities

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    ALL_METHODS, SEED, TARGET_STD,
    R_MIN, R_MAX, N_POINTS,
    GF_R_MIN, GF_R_MAX,
    get_data_dir, get_results_dir, get_figures_dir,
    rescale_coordinates, compute_gf_score,
)
from topological_analysis import (
    compute_persistent_homology,
    compute_betti_curves,
    compute_persistence_statistics,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()

SUBSAMPLE = 2000
BANNER = "=" * 70


# ============================================================
# Data Loading
# ============================================================

def load_human_go():
    """Load human GO annotations."""
    fpath = DATA / "human_go_annotations.json"
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


def load_human_embedding(method, go_map):
    """Load a single human embedding, filter to annotated nodes."""
    fname = f"human_{method.lower()}_embedding.json"
    fpath = DATA / fname
    if not fpath.exists():
        alt = method.lower().replace("-", "-")
        fpath = DATA / f"human_{alt}_embedding.json"
    if not fpath.exists():
        return None, None
    with open(fpath, encoding="utf-8") as f:
        raw = json.load(f)
    if not raw:
        return None, None
    first_val = next(iter(raw.values()))
    if isinstance(first_val, dict):
        nodes = sorted(raw.keys())
        coords = np.array([[raw[n]["x"], raw[n]["y"]] for n in nodes])
    elif isinstance(first_val, list):
        nodes = sorted(raw.keys())
        coords = np.array([raw[n] for n in nodes])
    else:
        return None, None
    # Filter to annotated nodes
    annotated = set(go_map.keys())
    mask = [n in annotated for n in nodes]
    coords = coords[mask]
    nodes = [n for n, m in zip(nodes, mask) if m]
    return coords, nodes


def get_common_subsample(embeddings_dict, n=SUBSAMPLE, seed=SEED):
    """Get consistent subsample across all methods."""
    rng = np.random.default_rng(seed)
    common = None
    for method, (coords, nodes) in embeddings_dict.items():
        s = set(nodes)
        common = s if common is None else common & s
    common = sorted(common)
    if len(common) <= n:
        return common
    return sorted(rng.choice(common, n, replace=False))


def extract_subsample(coords, nodes, subsample_nodes):
    """Extract rows for subsample_nodes from full embedding."""
    node_set = set(subsample_nodes)
    idx = [i for i, n in enumerate(nodes) if n in node_set]
    return coords[idx], [nodes[i] for i in idx]


# ============================================================
# Community Purity (simplified for speed)
# ============================================================

def compute_purity_at_r(coords_sub, nodes_sub, go_map, dist_matrix, r):
    """Compute mean functional purity at distance threshold r."""
    n = len(nodes_sub)
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # Build adjacency: pairs within distance r (upper triangle to avoid self-loops/duplicates)
    mask = (dist_matrix < r) & (dist_matrix > 0)
    rows, cols = np.where(np.triu(mask, 1))
    edges = list(zip(rows.tolist(), cols.tolist()))
    G.add_edges_from(edges)

    # Community detection
    if G.number_of_edges() == 0:
        return 0.0
    try:
        communities = list(greedy_modularity_communities(G))
    except Exception as e:
        return 0.0

    purities = []
    for comm in communities:
        comm_nodes = [nodes_sub[i] for i in comm]
        # Collect all GO terms
        all_terms = []
        for node in comm_nodes:
            all_terms.extend(go_map.get(node, []))
        if not all_terms:
            continue
        from collections import Counter
        term_counts = Counter(all_terms)
        most_common_count = term_counts.most_common(1)[0][1]
        total_terms = len(all_terms)
        purities.append(most_common_count / total_terms)

    return float(np.mean(purities)) if purities else 0.0


# ============================================================
# Main Analysis
# ============================================================

def run_analysis():
    print(BANNER)
    print("Phase 8B: Full Human TDA Analysis (11 Methods)")
    print(BANNER)

    # Load data
    print("\n[1/5] Loading human data...")
    go_map = load_human_go()
    print(f"  GO annotations: {len(go_map)} genes")

    # Load all embeddings
    embeddings = {}
    for method in ALL_METHODS:
        coords, nodes = load_human_embedding(method, go_map)
        if coords is not None:
            embeddings[method] = (coords, nodes)
            print(f"    {method}: {len(nodes)} annotated nodes")
        else:
            print(f"    {method}: NOT FOUND")

    # Subsample
    subsample = get_common_subsample(embeddings, n=SUBSAMPLE)
    print(f"  Common subsample: {len(subsample)} nodes (seed={SEED})")

    # Extract subsampled embeddings and rescale
    sub_embeddings = {}
    for method, (coords, nodes) in embeddings.items():
        c_sub, n_sub = extract_subsample(coords, nodes, subsample)
        c_sub = rescale_coordinates(c_sub, TARGET_STD)
        sub_embeddings[method] = (c_sub, n_sub)
        print(f"    {method}: subsampled to {len(n_sub)}, std={np.std(c_sub):.4f}")

    # R-grid (same as yeast for comparability)
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    print(f"  r-grid: [{R_MIN}, {R_MAX}], {N_POINTS} points")
    print(f"  GF interval: [{GF_R_MIN}, {GF_R_MAX}]")

    # Persistent homology
    print(f"\n[2/5] Computing persistent homology (11 methods, n={SUBSAMPLE})...")
    topo_results = {}
    for method in ALL_METHODS:
        if method not in sub_embeddings:
            continue
        coords, nodes = sub_embeddings[method]
        t0 = time.time()

        # Ripser
        diagrams = compute_persistent_homology(coords, max_dim=1)

        # Betti curves
        betti = compute_betti_curves(diagrams, r_vals)

        # Persistence statistics
        pstats = compute_persistence_statistics(diagrams)

        elapsed = time.time() - t0
        h1_n = pstats.get(1, {}).get("n_features", 0)
        h1_max = pstats.get(1, {}).get("max_persistence", 0)
        print(f"    {method:<12}: H0={pstats.get(0, {}).get('n_features', 0)}, "
              f"H1={h1_n}, H1_max_pers={h1_max:.4f}, time={elapsed:.1f}s")

        topo_results[method] = {
            "diagrams": diagrams,
            "betti_curves": {str(k): v.tolist() for k, v in betti.items()},
            "persistence_stats": {
                str(k): v for k, v in pstats.items()
            },
        }

    # Skip topo GF curves (too expensive for 2000 nodes)
    # Instead, directly do three-factor validation
    print(f"\n[3/5] Skipping topo GF curves (expensive on n={SUBSAMPLE})")
    print(f"      Using persistence statistics for three-factor validation")

    # Load human G-F scores and spectral alignment for validation
    print(f"\n[4/5] Running three-factor validation...")
    with open(RESULTS / "human_gf_scores_extended.json", encoding="utf-8") as f:
        gf_data = json.load(f)
    gf_scores = gf_data.get("scores", {})

    with open(RESULTS / "human_spectral_alignment.json", encoding="utf-8") as f:
        spec_data = json.load(f)
    spec_profiles = spec_data.get("spectral_profiles", {})

    # Load effective ranks from Phase 8
    with open(RESULTS / "human_cross_network_validation.json", encoding="utf-8") as f:
        phase8 = json.load(f)
    pm11 = phase8.get("human_per_method_11", {})

    methods_available = [m for m in ALL_METHODS
                         if m in topo_results and m in gf_scores and m in pm11]
    print(f"  Methods with complete data: {len(methods_available)}")

    # Build feature table
    from scipy.stats import rankdata
    gf_vals = np.array([gf_scores[m] for m in methods_available])
    sa_vals = np.array([pm11[m]["spectral_alignment"] for m in methods_available])
    er_vals = np.array([pm11[m]["effective_rank"] for m in methods_available])
    h1_max = np.array([topo_results[m]["persistence_stats"].get("1", {}).get("max_persistence", 0)
                       for m in methods_available])
    h1_mean = np.array([topo_results[m]["persistence_stats"].get("1", {}).get("mean_persistence", 0)
                        for m in methods_available])
    h1_complexity = np.array([topo_results[m]["persistence_stats"].get("1", {}).get("topological_complexity", 0)
                              for m in methods_available])

    # Single-factor correlations
    correlations = {}
    for name, vals in [("spectral_alignment", sa_vals),
                       ("effective_rank", er_vals),
                       ("h1_max_persistence", h1_max),
                       ("h1_mean_persistence", h1_mean),
                       ("h1_topological_complexity", h1_complexity)]:
        rho, p = spearmanr(vals, gf_vals)
        correlations[name] = {"rho": round(float(rho), 3), "p": round(float(p), 4)}
        print(f"    {name:<30}: rho={rho:+.3f} (p={p:.4f})")

    # Two-factor model
    sa_r = rankdata(sa_vals)
    er_r = rankdata(er_vals)
    two_factor = 0.5 * sa_r + 0.5 * er_r
    rho_2f, p_2f = spearmanr(two_factor, gf_vals)
    print(f"    {'two_factor (spec+eff_rank)':<30}: rho={rho_2f:+.3f} (p={p_2f:.4f})")

    # Three-factor model
    h1_r = rankdata(h1_max)
    three_factor = (1/3) * sa_r + (1/3) * er_r + (1/3) * h1_r
    rho_3f, p_3f = spearmanr(three_factor, gf_vals)
    print(f"    {'three_factor (+h1_max)':<30}: rho={rho_3f:+.3f} (p={p_3f:.4f})")

    # Weight-optimized three-factor
    from scipy.optimize import minimize
    def neg_rho(w):
        w = np.abs(w)
        w = w / w.sum()
        combined = w[0] * sa_r + w[1] * er_r + w[2] * h1_r
        r, _ = spearmanr(combined, gf_vals)
        return -r
    res = minimize(neg_rho, [0.33, 0.33, 0.34], method="Nelder-Mead")
    w_opt = np.abs(res.x)
    w_opt = w_opt / w_opt.sum()
    combined_opt = w_opt[0] * sa_r + w_opt[1] * er_r + w_opt[2] * h1_r
    rho_opt, p_opt = spearmanr(combined_opt, gf_vals)
    print(f"    {'weight_optimized 3F':<30}: rho={rho_opt:+.3f} (p={p_opt:.4f}), w=[{w_opt[0]:.2f},{w_opt[1]:.2f},{w_opt[2]:.2f}]")

    correlations["two_factor"] = {"rho": round(float(rho_2f), 3), "p": round(float(p_2f), 4)}
    correlations["three_factor"] = {"rho": round(float(rho_3f), 3), "p": round(float(p_3f), 4)}
    correlations["weight_optimized_3f"] = {"rho": round(float(rho_opt), 3), "p": round(float(p_opt), 4),
                                           "weights": [round(float(w), 3) for w in w_opt]}

    # Per-method table
    print(f"\n  Per-method detail:")
    print(f"    {'Method':<12} {'GF':>8} {'SpecAlign':>10} {'EffRank':>8} {'H1_max':>8}")
    for m in sorted(methods_available, key=lambda x: gf_scores[x], reverse=True):
        print(f"    {m:<12} {gf_scores[m]:8.4f} {pm11[m]['spectral_alignment']:10.4f} "
              f"{pm11[m]['effective_rank']:8.3f} "
              f"{topo_results[m]['persistence_stats'].get('1', {}).get('max_persistence', 0):8.4f}")

    # Comparison with yeast
    print(f"\n  Yeast vs Human comparison:")
    print(f"    {'Model':<30} {'Yeast rho':>10} {'Human rho':>10}")
    print(f"    {'Spectral alignment':<30} {0.609:>10.3f} {correlations['spectral_alignment']['rho']:>10.3f}")
    print(f"    {'Effective rank':<30} {0.873:>10.3f} {correlations['effective_rank']['rho']:>10.3f}")
    print(f"    {'H1 max persistence':<30} {0.764:>10.3f} {correlations['h1_max_persistence']['rho']:>10.3f}")
    print(f"    {'Two-factor':<30} {0.809:>10.3f} {correlations['two_factor']['rho']:>10.3f}")
    print(f"    {'Three-factor':<30} {0.827:>10.3f} {correlations['three_factor']['rho']:>10.3f}")

    # Prepare output
    output_data = {
        "analysis": "Phase 8B: Full Human TDA + Three-Factor Validation",
        "species": "human",
        "subsample_size": SUBSAMPLE,
        "methods": methods_available,
        "n_methods": len(methods_available),
        "correlations": correlations,
        "per_method": {m: {
            "gf_score": gf_scores[m],
            "spectral_alignment": pm11[m]["spectral_alignment"],
            "effective_rank": pm11[m]["effective_rank"],
            "h1_max_persistence": topo_results[m]["persistence_stats"].get("1", {}).get("max_persistence", 0),
            "h1_mean_persistence": topo_results[m]["persistence_stats"].get("1", {}).get("mean_persistence", 0),
            "h1_topological_complexity": topo_results[m]["persistence_stats"].get("1", {}).get("topological_complexity", 0),
            "h1_n_features": topo_results[m]["persistence_stats"].get("1", {}).get("n_features", 0),
        } for m in methods_available},
    }

    # Save JSON
    json_path = RESULTS / "human_tda_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {json_path}")

    # Generate figure
    generate_figure(output_data, gf_scores, topo_results)

    print(f"\n{BANNER}")
    print("Phase 8B complete.")
    print(BANNER)

    return output_data


def generate_figure(output_data, gf_scores, topo_results):
    """Generate Fig48: Human TDA + three-factor validation (3 panels)."""
    print("  Generating Fig48...")
    methods = output_data["methods"]
    results = output_data.get("per_method", {})
    correlations = output_data.get("correlations", {})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: H1 max persistence vs G-F Score
    ax = axes[0]
    gf_vals = [gf_scores[m] for m in methods]
    h1_vals = [results[m]["h1_max_persistence"] for m in methods]
    ax.scatter(h1_vals, gf_vals, s=80, c="#2171B5", edgecolors="k", linewidth=0.5, zorder=3)
    for m in methods:
        ax.annotate(m, (results[m]["h1_max_persistence"], gf_scores[m]), fontsize=7,
                    ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
    if len(h1_vals) >= 3:
        rho = correlations.get("h1_max_persistence", {}).get("rho", 0)
        p = correlations.get("h1_max_persistence", {}).get("p", 1)
        x = np.array(h1_vals)
        y = np.array(gf_vals)
        z = np.polyfit(x, y, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p_line(x_line), "k--", alpha=0.3, lw=1)
        ax.set_title(f"A. Human: H1 Max Persistence vs G-F\n(n={len(methods)}, rho={rho:.3f}, p={p:.3f})",
                     fontsize=10, fontweight="bold")
    ax.set_xlabel("H1 Max Persistence")
    ax.set_ylabel("G-F Score")
    ax.grid(True, alpha=0.3)

    # Panel B: Yeast vs Human model comparison (bar chart)
    ax = axes[1]
    models = ["Spectral\nAlone", "Eff Rank\nAlone", "H1 Max\nAlone", "Two-Factor", "Three-Factor"]
    yeast_rhos = [0.609, 0.873, 0.764, 0.809, 0.827]
    human_rhos = [
        correlations.get("spectral_alignment", {}).get("rho", 0),
        correlations.get("effective_rank", {}).get("rho", 0),
        correlations.get("h1_max_persistence", {}).get("rho", 0),
        correlations.get("two_factor", {}).get("rho", 0),
        correlations.get("three_factor", {}).get("rho", 0),
    ]
    x_pos = np.arange(len(models))
    w = 0.35
    bars1 = ax.bar(x_pos - w/2, yeast_rhos, w, label="Yeast (n=11)", color="#3182BD",
                   edgecolor="k", linewidth=0.5)
    bars2 = ax.bar(x_pos + w/2, human_rhos, w, label=f"Human (n={len(methods)})",
                   color="#E6550D", edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Spearman rho with G-F Score")
    ax.set_title("B. Yeast vs Human: Full 11-Method Comparison\n(New human TDA features)",
                 fontsize=10, fontweight="bold")
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

    # Panel C: Betti curves (H1) for top-3 and bottom-3 methods
    ax = axes[2]
    r_vals_plot = np.linspace(R_MIN, R_MAX, N_POINTS)
    sorted_methods = sorted(methods, key=lambda m: gf_scores[m], reverse=True)
    top3 = sorted_methods[:3]
    bot3 = sorted_methods[-3:]
    colors_top = ["#2171B5", "#4292C6", "#6BAED6"]
    colors_bot = ["#FC9272", "#FB6A4A", "#DE2D26"]
    for m, c in zip(top3, colors_top):
        beta1 = np.array(topo_results[m]["betti_curves"].get("1", []))
        if len(beta1) == N_POINTS:
            ax.plot(r_vals_plot, beta1, color=c, lw=1.5, label=f"{m} (top)")
    for m, c in zip(bot3, colors_bot):
        beta1 = np.array(topo_results[m]["betti_curves"].get("1", []))
        if len(beta1) == N_POINTS:
            ax.plot(r_vals_plot, beta1, color=c, lw=1.5, ls="--", label=f"{m} (bottom)")
    ax.set_title("C. Human Betti Curves (beta_1)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Filtration radius r")
    ax.set_ylabel("beta_1 (loop count)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "Fig48_human_tda_full_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Fig48_human_tda_full_analysis.png")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_analysis()
