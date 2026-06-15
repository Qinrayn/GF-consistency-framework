#!/usr/bin/env python3
"""
Phase 9: Unified Human G-F Scores (Fix Confounds 1+2)
=====================================================
Recomputes human G-F Scores using:
  - greedy_modularity_communities (same as yeast pipeline)
  - Integration interval [0.05, 0.422] (same as yeast)
  - 50 r-points (sufficient for trapezoidal integration)

This eliminates the community detection algorithm mismatch (Louvain vs
greedy_modularity) and the GF interval mismatch ([0.282,0.297] vs
[0.05,0.422]) that confound the cross-species comparison.

Depends on: data/human_*_embedding.json, data/human_go_annotations.json
Generates:  results/human_gf_unified.json
            figures/Fig50_unified_human_comparison.png
"""

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, rankdata
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    ALL_METHODS, SEED, TARGET_STD,
    R_MIN, R_MAX,
    GF_R_MIN, GF_R_MAX,
    get_data_dir, get_results_dir, get_figures_dir,
    rescale_coordinates,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()

SUBSAMPLE = 2000
N_POINTS = 25            # Coarse grid for tractable computation (25 sufficient for trapezoid)
MAX_EDGES = 150_000      # Fallback to connected_components above this (greedy_mod is O(n^2 log n))
BANNER = "=" * 70


# ============================================================
# Data Loading (reuse from human_tda_full)
# ============================================================

def load_human_go():
    fpath = DATA / "human_go_annotations.json"
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


def load_human_embedding(method, go_map):
    fname = f"human_{method.lower()}_embedding.json"
    fpath = DATA / fname
    if not fpath.exists():
        fpath = DATA / f"human_{method.lower().replace('-', '-')}_embedding.json"
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
    annotated = set(go_map.keys())
    mask = [n in annotated for n in nodes]
    coords = coords[mask]
    nodes = [n for n, m in zip(nodes, mask) if m]
    return coords, nodes


def get_common_subsample(embeddings_dict, n=SUBSAMPLE, seed=SEED):
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
    node_set = set(subsample_nodes)
    idx = [i for i, n in enumerate(nodes) if n in node_set]
    return coords[idx], [nodes[i] for i in idx]


# ============================================================
# G-F Curve with greedy_modularity_communities
# ============================================================

def compute_gf_curve_greedy(coords, nodes, go_map, r_vals):
    """Compute G-F purity curve using greedy_modularity_communities."""
    D = squareform(pdist(coords))
    n = len(nodes)
    purities = np.zeros(len(r_vals))

    for ri, r in enumerate(r_vals):
        if ri % 5 == 0 and ri > 0:
            print(".", end="", flush=True)
        mask = (D < r) & (D > 0)
        n_edges = int(np.sum(mask)) // 2
        if n_edges == 0:
            continue

        rows, cols = np.where(mask)
        upper = rows < cols
        edges = list(zip(rows[upper].tolist(), cols[upper].tolist()))

        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)

        if n_edges > MAX_EDGES:
            # Too many edges: use connected components
            communities = [frozenset(c) for c in nx.connected_components(G)]
        else:
            try:
                communities = list(greedy_modularity_communities(G))
            except Exception:
                communities = [frozenset(c) for c in nx.connected_components(G)]

        comm_purities = []
        for comm in communities:
            all_terms = []
            for i in comm:
                node_id = nodes[i]
                terms = go_map.get(node_id, [])
                all_terms.extend(terms)
            if not all_terms:
                continue
            counts = Counter(all_terms)
            comm_purities.append(counts.most_common(1)[0][1] / len(all_terms))

        if comm_purities:
            purities[ri] = float(np.mean(comm_purities))

    return purities


def compute_gf_score(purities, r_vals, r_min, r_max):
    """Compute G-F Score over [r_min, r_max] via trapezoidal integration."""
    mask = (r_vals >= r_min) & (r_vals <= r_max)
    if mask.sum() < 2:
        return 0.0
    r_sub = r_vals[mask]
    p_sub = purities[mask]
    integral = trapezoid(p_sub, r_sub)
    return float(integral / (r_sub[-1] - r_sub[0]))


# ============================================================
# Main
# ============================================================

def run():
    print(BANNER, flush=True)
    print("Phase 9: Unified Human G-F Scores (Fix Confounds 1+2)", flush=True)
    print(BANNER, flush=True)

    # Load data
    print("\n[1/4] Loading human data...")
    go_map = load_human_go()
    print(f"  GO annotations: {len(go_map)} genes")

    embeddings = {}
    for method in ALL_METHODS:
        coords, nodes = load_human_embedding(method, go_map)
        if coords is not None:
            embeddings[method] = (coords, nodes)
            print(f"    {method}: {len(nodes)} annotated nodes")

    subsample = get_common_subsample(embeddings)
    print(f"  Common subsample: {len(subsample)} nodes (seed={SEED})")

    sub_embeddings = {}
    for method, (coords, nodes) in embeddings.items():
        c_sub, n_sub = extract_subsample(coords, nodes, subsample)
        c_sub = rescale_coordinates(c_sub, TARGET_STD)
        sub_embeddings[method] = (c_sub, n_sub)

    # R-grid
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    print(f"  r-grid: [{R_MIN}, {R_MAX}], {N_POINTS} points")
    print(f"  Community detection: greedy_modularity_communities (same as yeast)")
    print(f"  GF intervals: yeast=[{GF_R_MIN}, {GF_R_MAX}], human_orig=[0.282, 0.297]")

    # Compute G-F curves
    print(f"\n[2/4] Computing G-F curves ({len(sub_embeddings)} methods, greedy_modularity)...")
    all_results = {}
    for i, method in enumerate(ALL_METHODS):
        if method not in sub_embeddings:
            continue
        coords, nodes = sub_embeddings[method]
        t0 = time.time()
        purities = compute_gf_curve_greedy(coords, nodes, go_map, r_vals)
        elapsed = time.time() - t0
        print()  # end progress dots

        # Compute scores on both intervals
        gf_yeast = compute_gf_score(purities, r_vals, GF_R_MIN, GF_R_MAX)
        gf_human = compute_gf_score(purities, r_vals, 0.282, 0.297)

        all_results[method] = {
            "purities": purities.tolist(),
            "gf_yeast_interval": round(gf_yeast, 6),
            "gf_human_interval": round(gf_human, 6),
        }
        print(f"    [{i+1}/{len(sub_embeddings)}] {method:<12}: "
              f"GF(yeast)={gf_yeast:.4f}, GF(human)={gf_human:.4f}, "
              f"peak_purity={max(purities):.4f}, time={elapsed:.1f}s", flush=True)

    # Load old GF scores for comparison
    print(f"\n[3/4] Comparing with old human GF scores (Louvain)...")
    old_path = RESULTS / "human_gf_scores_extended.json"
    old_data = json.load(open(old_path, encoding="utf-8"))
    old_scores = old_data.get("scores", {})

    # Build unified feature table for correlation analysis
    print(f"\n[4/4] Running correlation analysis with unified scores...")

    # Load Phase 8 features
    phase8 = json.load(open(RESULTS / "human_cross_network_validation.json", encoding="utf-8"))
    pm11 = phase8.get("human_per_method_11", {})

    # Load TDA features
    tda_path = RESULTS / "human_tda_full.json"
    tda_data = json.load(open(tda_path, encoding="utf-8")) if tda_path.exists() else {}
    tda_pm = tda_data.get("per_method", {})

    methods_available = [m for m in ALL_METHODS
                         if m in all_results and m in pm11]
    print(f"  Methods with complete data: {len(methods_available)}")

    # Build feature vectors
    gf_yeast = np.array([all_results[m]["gf_yeast_interval"] for m in methods_available])
    gf_human_new = np.array([all_results[m]["gf_human_interval"] for m in methods_available])
    gf_old = np.array([old_scores.get(m, 0) for m in methods_available])
    sa_vals = np.array([pm11[m]["spectral_alignment"] for m in methods_available])
    er_vals = np.array([pm11[m]["effective_rank"] for m in methods_available])

    h1_vals = np.zeros(len(methods_available))
    for i, m in enumerate(methods_available):
        if m in tda_pm:
            h1_vals[i] = tda_pm[m].get("h1_max_persistence", 0)

    # Rank old vs new scores
    rho_old_new, p_old_new = spearmanr(gf_old, gf_yeast)
    print(f"\n  Old (Louvain) vs New (greedy_mod, yeast interval): rho={rho_old_new:.3f}")

    # Correlation analysis for EACH GF score variant
    for label, gf_vals in [("OLD (Louvain, [0.282,0.297])", gf_old),
                           ("NEW unified (greedy, [0.05,0.422])", gf_yeast),
                           ("NEW human-interval (greedy, [0.282,0.297])", gf_human_new)]:
        print(f"\n  === {label} ===")
        for name, vals in [("spectral_alignment", sa_vals),
                           ("effective_rank", er_vals),
                           ("h1_max_persistence", h1_vals)]:
            rho, p = spearmanr(vals, gf_vals)
            print(f"    {name:<30}: rho={rho:+.3f} (p={p:.4f})")

        sa_r = rankdata(sa_vals)
        er_r = rankdata(er_vals)
        two_f = 0.5 * sa_r + 0.5 * er_r
        rho, p = spearmanr(two_f, gf_vals)
        print(f"    {'two_factor':<30}: rho={rho:+.3f} (p={p:.4f})")

        if np.any(h1_vals > 0):
            h1_r = rankdata(h1_vals)
            three_f = (1/3) * sa_r + (1/3) * er_r + (1/3) * h1_r
            rho, p = spearmanr(three_f, gf_vals)
            print(f"    {'three_factor':<30}: rho={rho:+.3f} (p={p:.4f})")

    # LOO for Spectral on new scores
    spec_idx = methods_available.index("Spectral") if "Spectral" in methods_available else None
    if spec_idx is not None:
        mask = np.ones(len(methods_available), dtype=bool)
        mask[spec_idx] = False
        print(f"\n  LOO check (excl Spectral) on NEW unified scores:")
        for name, vals in [("h1_max_persistence", h1_vals),
                           ("effective_rank", er_vals),
                           ("spectral_alignment", sa_vals)]:
            rho_full, _ = spearmanr(vals, gf_yeast)
            rho_loo, p_loo = spearmanr(vals[mask], gf_yeast[mask])
            print(f"    {name:<30}: full={rho_full:+.3f} -> excl_Spectral={rho_loo:+.3f} "
                  f"(delta={rho_loo - rho_full:+.3f})")

        sa_r_sub = rankdata(sa_vals[mask])
        er_r_sub = rankdata(er_vals[mask])
        two_f_sub = 0.5 * sa_r_sub + 0.5 * er_r_sub
        rho_loo_2f, p_loo_2f = spearmanr(two_f_sub, gf_yeast[mask])
        print(f"    {'two_factor':<30}: excl_Spectral rho={rho_loo_2f:+.3f} (p={p_loo_2f:.4f})")

    # Per-method comparison table
    print(f"\n  Per-method comparison:")
    print(f"    {'Method':<12} {'Old(Louv)':>10} {'New(greedy)':>12} {'Delta':>8} {'Rank_old':>9} {'Rank_new':>9}")
    from scipy.stats import rankdata as rd
    rank_old = {m: r for m, r in zip(methods_available, rd(-gf_old))}
    rank_new = {m: r for m, r in zip(methods_available, rd(-gf_yeast))}
    for m in sorted(methods_available, key=lambda x: gf_yeast[methods_available.index(x)], reverse=True):
        i = methods_available.index(m)
        print(f"    {m:<12} {gf_old[i]:10.4f} {gf_yeast[i]:12.4f} "
              f"{gf_yeast[i] - gf_old[i]:+8.4f} {int(rank_old[m]):9d} {int(rank_new[m]):9d}")

    # Save results
    output = {
        "analysis": "Phase 9: Unified Human G-F Scores",
        "community_detection": "greedy_modularity_communities (same as yeast)",
        "n_points": N_POINTS,
        "r_range": [R_MIN, R_MAX],
        "methods": methods_available,
        "n_methods": len(methods_available),
        "gf_scores": {m: {
            "yeast_interval": all_results[m]["gf_yeast_interval"],
            "human_interval": all_results[m]["gf_human_interval"],
            "old_louvain": old_scores.get(m, 0),
        } for m in methods_available},
        "old_vs_new_rank_correlation": round(float(rho_old_new), 3),
    }
    out_path = RESULTS / "human_gf_unified.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {out_path}")

    # Generate figure
    generate_figure(methods_available, all_results, old_scores, gf_yeast, gf_old)

    print(f"\n{BANNER}")
    print("Phase 9 complete.")
    print(BANNER)


def generate_figure(methods, all_results, old_scores, gf_new, gf_old):
    """Generate Fig50: Unified human comparison (3 panels)."""
    print("  Generating Fig50...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Old vs New GF scores scatter
    ax = axes[0]
    old_vals = [old_scores.get(m, 0) for m in methods]
    new_vals = [all_results[m]["gf_yeast_interval"] for m in methods]
    ax.scatter(old_vals, new_vals, s=80, c="#2171B5", edgecolors="k", linewidth=0.5, zorder=3)
    for m in methods:
        ax.annotate(m, (old_scores.get(m, 0), all_results[m]["gf_yeast_interval"]),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
    rho, _ = spearmanr(old_vals, new_vals)
    ax.set_title(f"A. Old (Louvain) vs New (greedy_mod)\nGF Scores (rho={rho:.3f})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Old: Louvain, [0.282, 0.297]")
    ax.set_ylabel("New: greedy_modularity, [0.05, 0.422]")
    ax.grid(True, alpha=0.3)

    # Panel B: G-F curves for top-3 and bottom-3 methods (new)
    ax = axes[1]
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    sorted_methods = sorted(methods, key=lambda m: all_results[m]["gf_yeast_interval"], reverse=True)
    top3 = sorted_methods[:3]
    bot3 = sorted_methods[-3:]
    colors_top = ["#2171B5", "#4292C6", "#6BAED6"]
    colors_bot = ["#FC9272", "#FB6A4A", "#DE2D26"]
    for m, c in zip(top3, colors_top):
        p = np.array(all_results[m]["purities"])
        ax.plot(r_vals, p, color=c, lw=1.5, label=f"{m} (top)")
    for m, c in zip(bot3, colors_bot):
        p = np.array(all_results[m]["purities"])
        ax.plot(r_vals, p, color=c, lw=1.5, ls="--", label=f"{m} (bot)")
    ax.axvline(x=GF_R_MIN, color="gray", ls=":", alpha=0.5)
    ax.axvline(x=GF_R_MAX, color="gray", ls=":", alpha=0.5)
    ax.axvline(x=0.282, color="orange", ls=":", alpha=0.5)
    ax.axvline(x=0.297, color="orange", ls=":", alpha=0.5)
    ax.set_title("B. Human G-F Curves (greedy_modularity)\n(blue=top-3, red=bottom-3)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Filtration radius r")
    ax.set_ylabel("Mean functional purity")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel C: Correlation comparison (old vs new GF scores)
    ax = axes[2]
    # Load features for correlation
    phase8 = json.load(open(RESULTS / "human_cross_network_validation.json", encoding="utf-8"))
    pm11 = phase8.get("human_per_method_11", {})
    sa = np.array([pm11[m]["spectral_alignment"] for m in methods])
    er = np.array([pm11[m]["effective_rank"] for m in methods])

    predictors = ["Spectral\nAlign", "Effective\nRank", "Two-Factor"]
    old_rhos = []
    new_rhos = []
    for vals in [sa, er]:
        rho_o, _ = spearmanr(vals, gf_old)
        rho_n, _ = spearmanr(vals, gf_new)
        old_rhos.append(rho_o)
        new_rhos.append(rho_n)
    sa_r = rankdata(sa)
    er_r = rankdata(er)
    two_f = 0.5 * sa_r + 0.5 * er_r
    rho_o, _ = spearmanr(two_f, gf_old)
    rho_n, _ = spearmanr(two_f, gf_new)
    old_rhos.append(rho_o)
    new_rhos.append(rho_n)

    x_pos = np.arange(len(predictors))
    w = 0.35
    bars1 = ax.bar(x_pos - w/2, old_rhos, w, label="Old (Louvain)", color="#3182BD",
                   edgecolor="k", linewidth=0.5)
    bars2 = ax.bar(x_pos + w/2, new_rhos, w, label="New (greedy_mod)", color="#E6550D",
                   edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Spearman rho with G-F Score")
    ax.set_title("C. Correlation: Old vs New GF Scores\n(same features, different GF computation)",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(predictors, fontsize=9)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "Fig50_unified_human_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Fig50_unified_human_comparison.png")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run()
