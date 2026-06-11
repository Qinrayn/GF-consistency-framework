#!/usr/bin/env python3
"""
semantic_similarity_analysis.py — Step 30: Semantic Purity Robustness Check.

Computes G-F curves and scores using three purity variants for all 11
embedding methods:

1. **Standard** (count-based): baseline, same as the main pipeline.
2. **IC-weighted**: down-weights general GO terms by Information Content,
   addressing DAG expansion inflation.
3. **Semantic** (Resnik MICA): captures functional coherence via GO DAG
   topology — communities with semantically related terms score high even
   without exact term matches.

Outputs
-------
- ``results/semantic_purity_analysis.json`` — full results (G-F scores,
  correlations, DAG diagnostics).
- ``figures/Fig16_semantic_purity_comparison.png`` — three-panel figure:
  (A) G-F curves comparison, (B) score scatter, (C) DAG inflation bar chart.

Usage::

    python scripts/semantic_similarity_analysis.py
"""

from __future__ import annotations

import sys
import json
import time
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from scipy.integrate import trapezoid

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    load_curated_network,
    rescale_coordinates,
    compute_gf_score,
    SEED, GF_R_MIN, GF_R_MAX, R_MIN, R_MAX, N_POINTS, TARGET_STD,
)
from go_propagation import parse_go_obo, build_go_dag, propagate_annotations
from semantic_purity import (
    compute_term_frequencies,
    compute_ic,
    build_similarity_index,
    compute_gf_curves_all_variants,
    community_purity_standard,
    community_purity_ic_weighted,
    diagnose_dag_inflation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHODS = [
    "DM", "MDS", "Spectral", "DeepWalk", "Node2Vec",
    "VGAE", "VGAE-feat", "PCA",
    "GraphSAGE", "GAT", "GIN",
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"


# ===================================================================
# Helper: load embedding and align nodes
# ===================================================================

def _load_embedding(method: str, common_nodes: list):
    """Load embedding .npy + _nodes.json, align to common_nodes order."""
    # try {method}_153 first, then {method}
    for suffix in ("_153", ""):
        npy_path = EMBEDDINGS_DIR / f"{method}{suffix}.npy"
        nodes_path = EMBEDDINGS_DIR / f"{method}{suffix}_nodes.json"
        if npy_path.exists() and nodes_path.exists():
            break
    else:
        return None
    coords = np.load(npy_path)
    with open(nodes_path) as f:
        emb_nodes = json.load(f)
    node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
    indices = [node_to_idx[n] for n in common_nodes if n in node_to_idx]
    if len(indices) != len(common_nodes):
        return None
    return coords[indices]


# ===================================================================
# Main analysis
# ===================================================================

def main():
    t0 = time.time()
    np.random.seed(SEED)

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # ---- 1. Load curated network + GO map ----
    logger.info("Loading curated network and GO annotations ...")
    G, nodes, go_map = load_curated_network(DATA_DIR)
    logger.info("  %d nodes, %d annotated", len(nodes), sum(1 for v in go_map.values() if v))

    # ---- 2. Load GO DAG ----
    logger.info("Loading GO DAG from go.obo ...")
    obo_path = DATA_DIR / "go.obo"
    if not obo_path.exists() or obo_path.stat().st_size < 1000:
        logger.error("data/go.obo not found or too small. Run Step 19 first.")
        sys.exit(1)

    go_data = parse_go_obo(obo_path)
    child_to_parents, parent_to_children, bp_terms = build_go_dag(go_data)
    logger.info("  %d BP terms, DAG loaded", len(bp_terms))

    # ---- 3. Build propagated GO map for DAG diagnostics ----
    logger.info("Propagating GO annotations (True Path Rule) ...")
    propagated_map = propagate_annotations(go_map, child_to_parents, bp_terms)
    n_orig = np.mean([len(v) for v in go_map.values() if v])
    n_prop = np.mean([len(v) for v in propagated_map.values() if v])
    logger.info("  Terms/gene: %.1f → %.1f  (×%.1f)", n_orig, n_prop, n_prop / max(n_orig, 0.1))

    # ---- 4. Compute IC using full network for better statistics ----
    logger.info("Computing Information Content ...")
    # Use propagated annotations for IC (gives better IC distribution)
    term_freq, n_genes = compute_term_frequencies(propagated_map)
    ic = compute_ic(term_freq, n_genes)
    max_ic = max(ic.values()) if ic else 1.0
    logger.info("  %d terms with IC, max_ic=%.3f", len(ic), max_ic)

    # ---- 5. Build Resnik similarity index ----
    # Only compute pairwise similarities for terms in curated annotations
    curated_terms = set()
    for terms in go_map.values():
        curated_terms.update(terms)
    for terms in propagated_map.values():
        curated_terms.update(terms)
    curated_terms &= bp_terms  # restrict to valid BP terms
    logger.info("Building Resnik similarity index (%d curated terms) ...", len(curated_terms))
    t_sim = time.time()
    sim_cache, max_ic_val = build_similarity_index(
        child_to_parents, bp_terms, ic, subset_terms=curated_terms,
    )
    logger.info("  Similarity index ready (%.1fs, %d entries)", time.time() - t_sim, len(sim_cache))

    # ---- 6. DAG inflation diagnostics ----
    logger.info("Running DAG inflation diagnostics ...")
    inflation = diagnose_dag_inflation(go_map, propagated_map, ic)
    logger.info("  Expansion: %.1f → %.1f terms/gene (×%.1f)",
                inflation["original_mean_terms_per_gene"],
                inflation["propagated_mean_terms_per_gene"],
                inflation["expansion_factor"])
    logger.info("  Standard purity inflation: %.4f → %.4f (ratio %.2f)",
                inflation["standard_purity_original"],
                inflation["standard_purity_propagated"],
                inflation["inflation_ratio_std"])
    logger.info("  IC-weighted purity: %.4f → %.4f (ratio %.2f)",
                inflation["ic_weighted_purity_original"],
                inflation["ic_weighted_purity_propagated"],
                inflation["inflation_ratio_ic"])

    # ---- 7. Compute GF curves for all methods × 3 variants ----
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    results = {}

    for method in METHODS:
        logger.info("Processing %s ...", method)
        coords = _load_embedding(method, nodes)
        if coords is None:
            logger.warning("  %s: embedding not found, skipping", method)
            continue

        coords_r = rescale_coordinates(coords, TARGET_STD)

        std_p, ic_p, sem_p = compute_gf_curves_all_variants(
            coords_r, nodes, go_map, r_vals, ic, sim_cache,
        )

        gf_std = compute_gf_score(r_vals, std_p, GF_R_MIN, GF_R_MAX)
        gf_ic = compute_gf_score(r_vals, ic_p, GF_R_MIN, GF_R_MAX)
        gf_sem = compute_gf_score(r_vals, sem_p, GF_R_MIN, GF_R_MAX)

        results[method] = {
            "gf_standard": round(gf_std, 4),
            "gf_ic_weighted": round(gf_ic, 4),
            "gf_semantic": round(gf_sem, 4),
        }
        logger.info("  GF: std=%.4f  IC=%.4f  sem=%.4f", gf_std, gf_ic, gf_sem)

    # ---- 8. Also compute with propagated annotations for comparison ----
    logger.info("Computing GF scores with propagated annotations ...")
    results_propagated = {}
    for method in METHODS:
        coords = _load_embedding(method, nodes)
        if coords is None:
            continue
        coords_r = rescale_coordinates(coords, TARGET_STD)

        std_p, ic_p, sem_p = compute_gf_curves_all_variants(
            coords_r, nodes, propagated_map, r_vals, ic, sim_cache,
        )

        gf_std = compute_gf_score(r_vals, std_p, GF_R_MIN, GF_R_MAX)
        gf_ic = compute_gf_score(r_vals, ic_p, GF_R_MIN, GF_R_MAX)
        gf_sem = compute_gf_score(r_vals, sem_p, GF_R_MIN, GF_R_MAX)

        results_propagated[method] = {
            "gf_standard": round(gf_std, 4),
            "gf_ic_weighted": round(gf_ic, 4),
            "gf_semantic": round(gf_sem, 4),
        }

    # ---- 9. Rank correlations between variants ----
    logger.info("Computing rank correlations ...")
    common_methods = [m for m in METHODS if m in results]
    gf_std_arr = np.array([results[m]["gf_standard"] for m in common_methods])
    gf_ic_arr = np.array([results[m]["gf_ic_weighted"] for m in common_methods])
    gf_sem_arr = np.array([results[m]["gf_semantic"] for m in common_methods])

    correlations = {}
    pairs = [
        ("standard_vs_ic_weighted", gf_std_arr, gf_ic_arr),
        ("standard_vs_semantic", gf_std_arr, gf_sem_arr),
        ("ic_weighted_vs_semantic", gf_ic_arr, gf_sem_arr),
    ]
    for name, a, b in pairs:
        if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0:
            rho, p = spearmanr(a, b)
            correlations[name] = {
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(p), 4),
                "n": len(a),
            }
            logger.info("  %s: ρ=%.4f, P=%.4f", name, rho, p)
        else:
            correlations[name] = {"spearman_rho": None, "p_value": None, "n": len(a)}

    # Propagated correlations
    if results_propagated:
        prop_methods = [m for m in METHODS if m in results_propagated]
        p_std = np.array([results_propagated[m]["gf_standard"] for m in prop_methods])
        p_ic = np.array([results_propagated[m]["gf_ic_weighted"] for m in prop_methods])
        p_sem = np.array([results_propagated[m]["gf_semantic"] for m in prop_methods])

        for name, a, b in [
            ("propagated_standard_vs_ic", p_std, p_ic),
            ("propagated_standard_vs_semantic", p_std, p_sem),
        ]:
            if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0:
                rho, p = spearmanr(a, b)
                correlations[name] = {
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(p), 4),
                    "n": len(a),
                }

    # ---- 10. Save results ----
    output = {
        "gf_scores": {m: results[m] for m in common_methods},
        "gf_scores_propagated": {m: results_propagated[m] for m in results_propagated},
        "correlations": correlations,
        "dag_inflation": inflation,
        "ic_stats": {
            "n_terms_with_ic": len(ic),
            "max_ic": round(max_ic_val, 4),
            "n_bp_terms": len(bp_terms),
            "sim_cache_size": len(sim_cache),
        },
    }

    json_path = RESULTS_DIR / "semantic_purity_analysis.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", json_path)

    # ---- 11. Generate figure ----
    logger.info("Generating figure ...")
    try:
        _generate_figure(results, results_propagated, inflation, correlations, common_methods)
    except Exception as e:
        logger.warning("Figure generation failed: %s", e)

    elapsed = time.time() - t0
    logger.info("Semantic similarity analysis complete (%.1fs)", elapsed)
    print(f"\nStep 30 complete: {len(common_methods)} methods × 3 purity variants")
    print(f"  Spearman(standard, IC-weighted): {correlations.get('standard_vs_ic_weighted', {}).get('spearman_rho', 'N/A')}")
    print(f"  Spearman(standard, semantic):    {correlations.get('standard_vs_semantic', {}).get('spearman_rho', 'N/A')}")
    print(f"  DAG inflation ratio (std):       {inflation['inflation_ratio_std']}")
    print(f"  DAG inflation ratio (IC):        {inflation['inflation_ratio_ic']}")


# ===================================================================
# Figure generation
# ===================================================================

def _generate_figure(results, results_propagated, inflation, correlations, methods):
    """Generate Fig16: three-panel semantic purity comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 5.5), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)

    # --- Panel A: G-F Score bar comparison ---
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))
    width = 0.25
    gf_std = [results[m]["gf_standard"] for m in methods]
    gf_ic = [results[m]["gf_ic_weighted"] for m in methods]
    gf_sem = [results[m]["gf_semantic"] for m in methods]

    ax_a.barh(x - width, gf_std, width, label="Standard", color="#4477AA", alpha=0.85)
    ax_a.barh(x, gf_ic, width, label="IC-weighted", color="#EE6677", alpha=0.85)
    ax_a.barh(x + width, gf_sem, width, label="Semantic (Resnik)", color="#228833", alpha=0.85)
    ax_a.set_yticks(x)
    ax_a.set_yticklabels(methods, fontsize=8)
    ax_a.set_xlabel("G-F Score")
    ax_a.set_title("(A) G-F Score: Three Purity Variants", fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=7, loc="lower right")
    ax_a.invert_yaxis()

    # --- Panel B: Scatter standard vs IC-weighted ---
    ax_b = fig.add_subplot(gs[0, 1])
    std_arr = np.array(gf_std)
    ic_arr = np.array(gf_ic)
    ax_b.scatter(std_arr, ic_arr, s=50, c="#4477AA", edgecolors="white", linewidth=0.5, zorder=3)
    for i, m in enumerate(methods):
        ax_b.annotate(m, (std_arr[i], ic_arr[i]), fontsize=6, ha="left", va="bottom",
                      xytext=(3, 3), textcoords="offset points")
    # diagonal
    lims = [0, max(std_arr.max(), ic_arr.max()) * 1.1]
    ax_b.plot(lims, lims, "--", color="grey", linewidth=0.8, alpha=0.6)
    ax_b.set_xlabel("Standard G-F Score", fontsize=9)
    ax_b.set_ylabel("IC-weighted G-F Score", fontsize=9)
    rho_str = f"ρ = {correlations.get('standard_vs_ic_weighted', {}).get('spearman_rho', 'N/A')}"
    ax_b.set_title(f"(B) Standard vs IC-weighted\n{rho_str}", fontsize=10, fontweight="bold")
    ax_b.set_xlim(lims)
    ax_b.set_ylim(lims)

    # --- Panel C: DAG inflation bar chart ---
    ax_c = fig.add_subplot(gs[0, 2])
    labels = ["Standard\n(original)", "Standard\n(propagated)", "IC-weighted\n(original)", "IC-weighted\n(propagated)"]
    values = [
        inflation["standard_purity_original"],
        inflation["standard_purity_propagated"],
        inflation["ic_weighted_purity_original"],
        inflation["ic_weighted_purity_propagated"],
    ]
    colors = ["#4477AA", "#4477AA", "#EE6677", "#EE6677"]
    bars = ax_c.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    # mark propagated bars with hatching
    bars[1].set_hatch("///")
    bars[3].set_hatch("///")
    ax_c.set_ylabel("Purity (all-genes community)")
    ax_c.set_title("(C) DAG Inflation Diagnosis", fontsize=10, fontweight="bold")
    ax_c.set_ylim(0, 1.15)
    for bar, val in zip(bars, values):
        ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f"{val:.3f}", ha="center", fontsize=8)

    fig.suptitle("Semantic Purity Robustness Analysis", fontsize=12, fontweight="bold", y=1.02)
    fig_path = FIGURES_DIR / "Fig16_semantic_purity_comparison.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    main()
