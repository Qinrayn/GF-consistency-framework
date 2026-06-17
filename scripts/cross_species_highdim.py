#!/usr/bin/env python3
"""
Step 53: Cross-species functional conservation in high-dimensional embedding space.

Re-runs the cross-species dark matter conservation analysis using d=64
spectral embeddings (instead of 2D). Directly compares 2D vs 64D to
demonstrate the dimensionality effect on conservation signal strength.

Key question: does higher-dimensional embedding space reveal stronger
functional-geometric conservation across species?

Outputs:
    results/cross_species_highdim.json
    figures/Fig74_cross_species_highdim.png
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_results_dir, get_data_dir, get_figures_dir, get_embeddings_dir, SEED

RESULTS = get_results_dir()
DATA = get_data_dir()
FIGURES = get_figures_dir()
EMBEDDINGS = get_embeddings_dir()

BANNER = "=" * 64
N_PERMUTATIONS = 5000
MAX_SAMPLE = 50  # max proteins to sample per GO category

CONSERVED_TERMS = {
    "GO:0055085": "transmembrane transport",
    "GO:0036503": "ERAD pathway",
    "GO:0034599": "cellular response to oxidative stress",
    "GO:0006457": "protein folding",
    "GO:0016126": "sterol biosynthetic process",
    "GO:0006631": "fatty acid metabolic process",
    "GO:0006879": "intracellular iron ion homeostasis",
}


# ================================================================
# Data loading
# ================================================================

def load_highdim_embedding(species):
    """Load d=64 spectral embedding from .npy + nodes JSON."""
    npy_path = EMBEDDINGS / f"{species}_spectral_d64.npy"
    nodes_path = EMBEDDINGS / f"{species}_spectral_d64_nodes.json"
    coords = np.load(str(npy_path))
    with open(nodes_path, encoding="utf-8") as f:
        nodes = json.load(f)
    id_to_idx = {pid: i for i, pid in enumerate(nodes)}
    return coords, nodes, id_to_idx


def load_2d_embedding(species):
    """Load 2D spectral embedding from JSON for comparison."""
    path = DATA / f"{species}_spectral_embedding.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    ids = list(raw.keys())
    coords = np.array([[v["x"], v["y"]] for v in raw.values()])
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    return coords, ids, id_to_idx


def load_species_go(species):
    """Load GO annotations for human or mouse."""
    path = DATA / f"{species}_go_annotations.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ================================================================
# Distance and statistics
# ================================================================

def mean_pairwise_distance(coords, indices):
    """Mean pairwise Euclidean distance for a group of proteins."""
    if len(indices) < 2:
        return float("inf")
    pts = coords[list(indices)]
    n = len(pts)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(pts[i] - pts[j])
            count += 1
    return total / count if count > 0 else float("inf")


def permutation_test(observed_mean, coords, all_indices, group_size,
                     n_perm=N_PERMUTATIONS, seed=SEED):
    """Permutation test: is the observed mean distance significantly small?"""
    rng = np.random.RandomState(seed)
    n_below = 0
    all_idx = np.array(list(all_indices))
    for _ in range(n_perm):
        sample = rng.choice(all_idx, size=group_size, replace=False)
        perm_mean = mean_pairwise_distance(coords, sample)
        if perm_mean <= observed_mean:
            n_below += 1
    return (n_below + 1) / (n_perm + 1)


def compute_background_mean(coords, all_indices, group_size, n_samples=500,
                            seed=SEED):
    """Compute mean pairwise distance of random samples (background)."""
    rng = np.random.RandomState(seed)
    all_idx = np.array(list(all_indices))
    bg_means = []
    for _ in range(n_samples):
        sample = rng.choice(all_idx, size=group_size, replace=False)
        bg_means.append(mean_pairwise_distance(coords, sample))
    return float(np.mean(bg_means))


# ================================================================
# Core analysis
# ================================================================

def run_clustering_analysis(species, coords, ids, id_to_idx, go_annotations):
    """Run within-category clustering test for all conserved GO terms."""
    results = {}
    all_indices = list(range(len(ids)))

    for term, name in sorted(CONSERVED_TERMS.items(), key=lambda x: x[1]):
        # Find proteins with this GO term that exist in the embedding
        term_ids = [pid for pid, terms in go_annotations.items()
                    if term in terms and pid in id_to_idx]
        if len(term_ids) < 5:
            continue

        term_indices = [id_to_idx[pid] for pid in term_ids]
        sample_size = min(len(term_indices), MAX_SAMPLE)

        # Observed mean pairwise distance
        if sample_size < len(term_indices):
            rng = np.random.RandomState(SEED)
            sampled = rng.choice(term_indices, size=sample_size, replace=False)
            obs_mean = mean_pairwise_distance(coords, sampled)
        else:
            obs_mean = mean_pairwise_distance(coords, term_indices)

        # Permutation test
        p_val = permutation_test(obs_mean, coords, all_indices, sample_size)

        # Background
        bg_mean = compute_background_mean(coords, all_indices, sample_size)
        enrichment = bg_mean / obs_mean if obs_mean > 0 else float("inf")

        results[term] = {
            "name": name,
            "n_proteins": len(term_ids),
            "n_sampled": sample_size,
            "observed_mean_dist": round(float(obs_mean), 6),
            "background_mean_dist": round(float(bg_mean), 6),
            "enrichment_ratio": round(float(enrichment), 3),
            "p_value": round(float(p_val), 6),
        }

    return results


# ================================================================
# Main
# ================================================================

def run():
    t_start = time.time()
    print(BANNER)
    print("  Phase 22: High-Dimensional Cross-Species Conservation")
    print("  d=64 Spectral Embedding vs 2D Comparison")
    print(BANNER)

    random.seed(SEED)
    np.random.seed(SEED)

    # ---- Load data ----
    print("\n[1/5] Loading high-dimensional embeddings ...")

    print("  Human d=64:")
    human_64_coords, human_64_ids, human_64_idx = load_highdim_embedding("human")
    print(f"    {human_64_coords.shape[0]} nodes, {human_64_coords.shape[1]}D")

    print("  Mouse d=64:")
    mouse_64_coords, mouse_64_ids, mouse_64_idx = load_highdim_embedding("mouse")
    print(f"    {mouse_64_coords.shape[0]} nodes, {mouse_64_coords.shape[1]}D")

    print("  Human 2D (for comparison):")
    human_2d_coords, human_2d_ids, human_2d_idx = load_2d_embedding("human")
    print(f"    {human_2d_coords.shape[0]} nodes, {human_2d_coords.shape[1]}D")

    print("  Mouse 2D (for comparison):")
    mouse_2d_coords, mouse_2d_ids, mouse_2d_idx = load_2d_embedding("mouse")
    print(f"    {mouse_2d_coords.shape[0]} nodes, {mouse_2d_coords.shape[1]}D")

    print("\n[2/5] Loading GO annotations ...")
    human_go = load_species_go("human")
    mouse_go = load_species_go("mouse")
    print(f"  Human: {len(human_go)} proteins with GO")
    print(f"  Mouse: {len(mouse_go)} proteins with GO")

    # ---- Run 64D analysis ----
    print("\n[3/5] Running 64D clustering analysis ...")

    print("\n  Human (d=64):")
    human_64_res = run_clustering_analysis(
        "human", human_64_coords, human_64_ids, human_64_idx, human_go)
    for term, r in sorted(human_64_res.items(), key=lambda x: x[1]["name"]):
        mark = "*" if r["p_value"] < 0.05 else " "
        print(f"    {mark} {r['name']:45s}: n={r['n_proteins']:3d}, "
              f"enrich={r['enrichment_ratio']:.2f}x, p={r['p_value']:.4f}")

    print("\n  Mouse (d=64):")
    mouse_64_res = run_clustering_analysis(
        "mouse", mouse_64_coords, mouse_64_ids, mouse_64_idx, mouse_go)
    for term, r in sorted(mouse_64_res.items(), key=lambda x: x[1]["name"]):
        mark = "*" if r["p_value"] < 0.05 else " "
        print(f"    {mark} {r['name']:45s}: n={r['n_proteins']:3d}, "
              f"enrich={r['enrichment_ratio']:.2f}x, p={r['p_value']:.4f}")

    # ---- Run 2D analysis (for comparison) ----
    print("\n[4/5] Running 2D clustering analysis (comparison) ...")

    print("\n  Human (d=2):")
    human_2d_res = run_clustering_analysis(
        "human", human_2d_coords, human_2d_ids, human_2d_idx, human_go)
    for term, r in sorted(human_2d_res.items(), key=lambda x: x[1]["name"]):
        mark = "*" if r["p_value"] < 0.05 else " "
        print(f"    {mark} {r['name']:45s}: n={r['n_proteins']:3d}, "
              f"enrich={r['enrichment_ratio']:.2f}x, p={r['p_value']:.4f}")

    print("\n  Mouse (d=2):")
    mouse_2d_res = run_clustering_analysis(
        "mouse", mouse_2d_coords, mouse_2d_ids, mouse_2d_idx, mouse_go)
    for term, r in sorted(mouse_2d_res.items(), key=lambda x: x[1]["name"]):
        mark = "*" if r["p_value"] < 0.05 else " "
        print(f"    {mark} {r['name']:45s}: n={r['n_proteins']:3d}, "
              f"enrich={r['enrichment_ratio']:.2f}x, p={r['p_value']:.4f}")

    # ---- Load original 2D results from Step 49 for reference ----
    orig_path = RESULTS / "cross_species_dark_matter.json"
    orig_data = {}
    if orig_path.exists():
        with open(orig_path, encoding="utf-8") as f:
            orig_data = json.load(f)

    # ---- Comparison summary ----
    print("\n  === Dimension comparison ===")
    comparison = {}
    for term, name in sorted(CONSERVED_TERMS.items(), key=lambda x: x[1]):
        entry = {"name": name}
        for species_label, res_2d, res_64 in [
            ("human", human_2d_res, human_64_res),
            ("mouse", mouse_2d_res, mouse_64_res),
        ]:
            r2 = res_2d.get(term, {})
            r64 = res_64.get(term, {})
            entry[f"{species_label}_2d_p"] = r2.get("p_value", None)
            entry[f"{species_label}_2d_enrich"] = r2.get("enrichment_ratio", None)
            entry[f"{species_label}_64d_p"] = r64.get("p_value", None)
            entry[f"{species_label}_64d_enrich"] = r64.get("enrichment_ratio", None)

        # Count significant results
        n_sig_2d = sum(1 for k in ["human_2d_p", "mouse_2d_p"]
                       if entry.get(k) is not None and entry[k] < 0.05)
        n_sig_64 = sum(1 for k in ["human_64d_p", "mouse_64d_p"]
                       if entry.get(k) is not None and entry[k] < 0.05)
        entry["n_sig_2d"] = n_sig_2d
        entry["n_sig_64d"] = n_sig_64
        comparison[term] = entry

        delta = "UP" if n_sig_64 > n_sig_2d else ("SAME" if n_sig_64 == n_sig_2d else "DOWN")
        print(f"    {name:45s}: 2D={n_sig_2d} sig, 64D={n_sig_64} sig [{delta}]")

    # Conservation counts
    def count_conserved(human_res, mouse_res):
        n = 0
        for term in CONSERVED_TERMS:
            h_sig = human_res.get(term, {}).get("p_value", 1.0) < 0.05
            m_sig = mouse_res.get(term, {}).get("p_value", 1.0) < 0.05
            if h_sig or m_sig:
                n += 1
        return n

    conserved_2d = count_conserved(human_2d_res, mouse_2d_res)
    conserved_64d = count_conserved(human_64_res, mouse_64_res)

    # ---- Pooled analysis (Fisher's method for combining p-values) ----
    # Combine human + mouse p-values per term using Fisher's method
    from scipy.stats import fisher_exact
    pooled_results = {}
    for term, name in sorted(CONSERVED_TERMS.items(), key=lambda x: x[1]):
        # 64D pooled
        p_h64 = human_64_res.get(term, {}).get("p_value", None)
        p_m64 = mouse_64_res.get(term, {}).get("p_value", None)
        pvals_64 = [p for p in [p_h64, p_m64] if p is not None]

        if len(pvals_64) >= 1:
            # Fisher's combined probability test
            chi2_stat = -2 * sum(math.log(max(p, 1e-10)) for p in pvals_64)
            k = len(pvals_64)
            # chi-squared with 2k df
            from scipy.stats import chi2 as chi2_dist
            fisher_p = 1 - chi2_dist.cdf(chi2_stat, df=2*k)
            pooled_results[term] = {
                "name": name,
                "n_species": k,
                "fisher_p_64d": round(float(fisher_p), 6),
                "individual_pvals_64d": pvals_64,
            }

    n_pooled_sig = sum(1 for v in pooled_results.values()
                       if v["fisher_p_64d"] < 0.05)

    # ---- Save results ----
    print("\n[5/5] Generating outputs ...")

    output = {
        "description": "Cross-Species Functional Conservation: 2D vs 64D Comparison",
        "version": "2.0.0",
        "method": (
            "Within-category spatial clustering test comparing 2D (Fiedler pair) "
            "vs 64D spectral embeddings. Permutation test (5000 iterations) for "
            "each species x dimension combination. Fisher's method for cross-species "
            "p-value combination."
        ),
        "embedding_dimensions": {"2D": 2, "64D": 64},
        "conserved_categories_2d": int(conserved_2d),
        "conserved_categories_64d": int(conserved_64d),
        "fisher_pooled_sig_64d": int(n_pooled_sig),
        "human_64d_clustering": human_64_res,
        "mouse_64d_clustering": mouse_64_res,
        "human_2d_clustering": human_2d_res,
        "mouse_2d_clustering": mouse_2d_res,
        "dimension_comparison": comparison,
        "fisher_pooled_64d": pooled_results,
        "conclusion_64d": (
            f"With d=64 spectral embeddings, {conserved_64d}/{len(CONSERVED_TERMS)} "
            f"GO categories show significant spatial clustering (p<0.05) in at least "
            f"one mammalian species, compared to {conserved_2d}/{len(CONSERVED_TERMS)} "
            f"with 2D embeddings. Fisher's combined test yields {n_pooled_sig}/"
            f"{len(pooled_results)} significant categories."
        ),
    }

    out_file = RESULTS / "cross_species_highdim.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_file}")

    # ---- Generate figure ----
    generate_figure(output, human_2d_res, human_64_res,
                    mouse_2d_res, mouse_64_res, comparison)

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  2D conserved: {conserved_2d}/{len(CONSERVED_TERMS)}")
    print(f"  64D conserved: {conserved_64d}/{len(CONSERVED_TERMS)}")
    print(f"  Fisher pooled (64D): {n_pooled_sig}/{len(pooled_results)} sig")


# ================================================================
# Figure generation
# ================================================================

def generate_figure(output, human_2d, human_64, mouse_2d, mouse_64, comparison):
    """Generate 3-panel figure: 2D vs 64D comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    common_terms = sorted(
        [t for t in CONSERVED_TERMS if t in comparison],
        key=lambda t: CONSERVED_TERMS[t]
    )
    names = [CONSERVED_TERMS[t][:30] for t in common_terms]
    x = np.arange(len(common_terms))

    # ---- Panel A: Enrichment ratio comparison (grouped bars) ----
    ax_a = fig.add_subplot(gs[0, :2])
    width = 0.2

    h2_enrich = [human_2d.get(t, {}).get("enrichment_ratio", 0) or 0
                 for t in common_terms]
    h64_enrich = [human_64.get(t, {}).get("enrichment_ratio", 0) or 0
                  for t in common_terms]
    m2_enrich = [mouse_2d.get(t, {}).get("enrichment_ratio", 0) or 0
                 for t in common_terms]
    m64_enrich = [mouse_64.get(t, {}).get("enrichment_ratio", 0) or 0
                  for t in common_terms]

    ax_a.bar(x - 1.5*width, h2_enrich, width, label="Human 2D",
             color="#FF8A65", alpha=0.85, edgecolor="white")
    ax_a.bar(x - 0.5*width, h64_enrich, width, label="Human 64D",
             color="#FF5722", alpha=0.85, edgecolor="white")
    ax_a.bar(x + 0.5*width, m2_enrich, width, label="Mouse 2D",
             color="#81C784", alpha=0.85, edgecolor="white")
    ax_a.bar(x + 1.5*width, m64_enrich, width, label="Mouse 64D",
             color="#4CAF50", alpha=0.85, edgecolor="white")

    # Significance markers
    for i, t in enumerate(common_terms):
        for j, (res, offset, color) in enumerate([
            (human_2d, -1.5*width, "#FF8A65"),
            (human_64, -0.5*width, "#FF5722"),
            (mouse_2d, 0.5*width, "#81C784"),
            (mouse_64, 1.5*width, "#4CAF50"),
        ]):
            p = res.get(t, {}).get("p_value", 1.0)
            enrich = res.get(t, {}).get("enrichment_ratio", 0) or 0
            if p < 0.05:
                ax_a.text(x[i] + offset, enrich + 0.05, "*",
                          ha="center", va="bottom", fontsize=12,
                          fontweight="bold", color=color)

    ax_a.set_xlabel("GO Biological Process Term", fontsize=11)
    ax_a.set_ylabel("Clustering Enrichment Ratio", fontsize=11)
    ax_a.set_title("(A) Functional Clustering: 2D vs 64D Spectral Embeddings",
                    fontsize=13, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax_a.legend(fontsize=8, loc="upper right", ncol=2)
    ax_a.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ---- Panel B: -log10(p) heatmap (4 columns: H2D, H64D, M2D, M64D) ----
    ax_b = fig.add_subplot(gs[0, 2])

    matrix = []
    labels_y = []
    for t in common_terms:
        row = []
        for res in [human_2d, human_64, mouse_2d, mouse_64]:
            p = res.get(t, {}).get("p_value", None)
            if p is not None:
                row.append(-math.log10(max(p, 1e-6)))
            else:
                row.append(0)
        matrix.append(row)
        labels_y.append(CONSERVED_TERMS[t][:25])

    matrix = np.array(matrix)
    vmax = max(3.0, np.max(matrix) if matrix.size > 0 else 3.0)
    im = ax_b.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax_b.set_xticks([0, 1, 2, 3])
    ax_b.set_xticklabels(["H 2D", "H 64D", "M 2D", "M 64D"], fontsize=8,
                          rotation=45)
    ax_b.set_yticks(range(len(labels_y)))
    ax_b.set_yticklabels(labels_y, fontsize=7)
    ax_b.set_title("(B) -log10(p) by Dimension", fontsize=13, fontweight="bold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 1.5 else "black"
            ax_b.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                      fontsize=6, color=color)

    fig.colorbar(im, ax=ax_b, shrink=0.8, label="-log10(p)")

    # ---- Panel C: P-value improvement scatter (64D vs 2D) ----
    ax_c = fig.add_subplot(gs[1, 0])

    # For each species x term, plot 2D p vs 64D p
    for species_label, res_2d, res_64, color, marker in [
        ("Human", human_2d, human_64, "#FF5722", "o"),
        ("Mouse", mouse_2d, mouse_64, "#4CAF50", "s"),
    ]:
        p2d_list, p64_list = [], []
        for t in common_terms:
            p2 = res_2d.get(t, {}).get("p_value", None)
            p64 = res_64.get(t, {}).get("p_value", None)
            if p2 is not None and p64 is not None:
                p2d_list.append(max(p2, 1e-4))
                p64_list.append(max(p64, 1e-4))

        if p2d_list:
            ax_c.scatter(p2d_list, p64_list, c=color, marker=marker,
                        s=60, alpha=0.7, edgecolors="white",
                        label=f"{species_label}", zorder=3)

    ax_c.plot([1e-4, 1.0], [1e-4, 1.0], "k--", alpha=0.3, label="y=x")
    ax_c.axhline(y=0.05, color="red", linestyle=":", alpha=0.4,
                 label="p=0.05 threshold")
    ax_c.axvline(x=0.05, color="red", linestyle=":", alpha=0.4)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("2D p-value", fontsize=11)
    ax_c.set_ylabel("64D p-value", fontsize=11)
    ax_c.set_title("(C) P-value: 2D vs 64D", fontsize=13, fontweight="bold")
    ax_c.legend(fontsize=8, loc="upper left")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ---- Panel D: Fisher pooled significance ----
    ax_d = fig.add_subplot(gs[1, 1])

    fisher_data = output.get("fisher_pooled_64d", {})
    fisher_terms = sorted(fisher_data.keys(),
                          key=lambda t: fisher_data[t]["fisher_p_64d"])
    fisher_names = [fisher_data[t]["name"][:25] for t in fisher_terms]
    fisher_pvals = [fisher_data[t]["fisher_p_64d"] for t in fisher_terms]
    fisher_neg_logp = [-math.log10(max(p, 1e-6)) for p in fisher_pvals]

    colors = ["#4CAF50" if p < 0.05 else "#BDBDBD" for p in fisher_pvals]
    ax_d.barh(range(len(fisher_terms)), fisher_neg_logp, color=colors,
              alpha=0.85, edgecolor="white")
    ax_d.set_yticks(range(len(fisher_terms)))
    ax_d.set_yticklabels(fisher_names, fontsize=8)
    ax_d.axvline(x=-math.log10(0.05), color="red", linestyle="--", alpha=0.5,
                 label="p=0.05")
    ax_d.set_xlabel("-log10(Fisher combined p)", fontsize=11)
    ax_d.set_title("(D) Fisher Pooled (64D)", fontsize=13, fontweight="bold")
    ax_d.legend(fontsize=8)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    # ---- Panel E: Summary statistics ----
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis("off")

    summary_text = (
        f"Cross-Species Conservation Summary\n"
        f"{'='*40}\n\n"
        f"Embedding dimensions compared:\n"
        f"  2D (Fiedler pair) vs 64D (eigvecs 1-64)\n\n"
        f"Conserved categories (p<0.05):\n"
        f"  2D:  {output['conserved_categories_2d']}/{len(CONSERVED_TERMS)}\n"
        f"  64D: {output['conserved_categories_64d']}/{len(CONSERVED_TERMS)}\n\n"
        f"Fisher pooled (64D):\n"
        f"  {output['fisher_pooled_sig_64d']}/{len(fisher_data)} significant\n\n"
        f"Improvement: "
        f"{'+' if output['conserved_categories_64d'] > output['conserved_categories_2d'] else ''}"
        f"{output['conserved_categories_64d'] - output['conserved_categories_2d']} categories\n"
    )
    ax_e.text(0.05, 0.95, summary_text, transform=ax_e.transAxes,
              fontsize=10, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9",
                        edgecolor="#4CAF50", alpha=0.8))

    # ---- Save ----
    fig_path = FIGURES / "Fig74_cross_species_highdim.png"
    fig.savefig(str(fig_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Figure saved to {fig_path}")


if __name__ == "__main__":
    run()
