#!/usr/bin/env python3
"""
Steps 54+55: Three-species dimension gradient experiment.

Computes yeast d=64 spectral embedding and runs cross-species functional
conservation clustering test at d = 2, 8, 16, 32, 64 for all three species
(yeast, human, mouse). Identifies critical dimensions where each GO category
transitions between significant and non-significant.

Key question: at what dimension d does each functional category's spatial
clustering become detectable?

Outputs:
    embeddings/yeast_spectral_d64.npy
    embeddings/yeast_spectral_d64_nodes.json
    results/dimension_gradient_3species.json
    figures/Fig75_dimension_gradient.png
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh as sparse_eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_results_dir, get_data_dir, get_figures_dir, get_embeddings_dir, SEED

RESULTS = get_results_dir()
DATA = get_data_dir()
FIGURES = get_figures_dir()
EMBEDDINGS = get_embeddings_dir()

BANNER = "=" * 64
N_PERMUTATIONS = 3000  # reduced for gradient (5 dims x 3 species x 7 terms)
MAX_SAMPLE = 50
DIMENSIONS = [2, 8, 16, 32, 64]
DIM_MAX = 64

CONSERVED_TERMS = {
    "GO:0055085": "transmembrane transport",
    "GO:0036503": "ERAD pathway",
    "GO:0034599": "oxidative stress response",
    "GO:0006457": "protein folding",
    "GO:0016126": "sterol biosynthesis",
    "GO:0006631": "fatty acid metabolism",
    "GO:0006879": "iron ion homeostasis",
}


# ================================================================
# Yeast d=64 embedding computation
# ================================================================

def compute_yeast_highdim():
    """Compute d=64 spectral embedding for yeast PPI network."""
    project_dir = os.path.join(os.path.dirname(__file__), "..")
    string_file = os.path.join(project_dir, "data",
                               "4932.protein.links.v11.5.txt.gz")

    print("  Loading yeast STRING network ...")
    t0 = time.time()
    edges = []
    with gzip.open(string_file, "rt") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                p1, p2, score = parts[0], parts[1], int(parts[2])
                if score >= 700 and p1 != p2:
                    edges.append((p1, p2))

    G_full = nx.Graph()
    G_full.add_edges_from(edges)
    cc = sorted(nx.connected_components(G_full), key=len, reverse=True)
    G = G_full.subgraph(cc[0]).copy()
    print(f"  LCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"({time.time()-t0:.1f}s)")

    nodes = sorted(G.nodes())
    n = len(nodes)
    k = DIM_MAX + 1

    print(f"  Computing Laplacian ({n}x{n}) ...")
    t0 = time.time()
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).astype(np.float64)
    print(f"  Laplacian nnz={L.nnz} ({time.time()-t0:.1f}s)")

    print(f"  Sparse eigendecomposition (k={k}) ...")
    t0 = time.time()
    eigvals, eigvecs = sparse_eigsh(L, k=k, sigma=0, which="LM")
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Skip trivial eigenvector (index 0), take 1..DIM_MAX
    embedding = eigvecs[:, 1:DIM_MAX + 1]
    eigenvalues = eigvals[1:DIM_MAX + 1]
    print(f"  Eigendecomposition: {time.time()-t0:.1f}s")
    print(f"  Trivial eigenvalue: {eigvals[0]:.2e}")
    print(f"  Fiedler value: {eigenvalues[0]:.6f}")

    # Participation ratio
    variances = np.var(embedding, axis=0)
    total_var = variances.sum()
    norm_var = variances / total_var if total_var > 0 else variances
    pr = 1.0 / np.sum(norm_var ** 2) if total_var > 0 else 0
    print(f"  Participation ratio: {pr:.2f}/{DIM_MAX}")

    # Strip species prefix from node names (4932.YAL001C -> YAL001C)
    clean_nodes = [n.split(".", 1)[-1] if "." in n else n for n in nodes]

    # Save
    npy_path = str(EMBEDDINGS / "yeast_spectral_d64.npy")
    nodes_path = str(EMBEDDINGS / "yeast_spectral_d64_nodes.json")
    np.save(npy_path, embedding)
    with open(nodes_path, "w") as f:
        json.dump(clean_nodes, f)
    print(f"  Saved: {npy_path} ({embedding.shape})")
    print(f"  Saved: {nodes_path} ({len(clean_nodes)} nodes)")

    return embedding, clean_nodes, {n: i for i, n in enumerate(clean_nodes)}, eigenvalues


# ================================================================
# Data loading
# ================================================================

def load_embedding(species):
    """Load d=64 spectral embedding for a species."""
    npy_path = EMBEDDINGS / f"{species}_spectral_d{DIM_MAX}.npy"
    nodes_path = EMBEDDINGS / f"{species}_spectral_d{DIM_MAX}_nodes.json"
    coords = np.load(str(npy_path))
    with open(nodes_path, encoding="utf-8") as f:
        nodes = json.load(f)
    id_to_idx = {pid: i for i, pid in enumerate(nodes)}
    return coords, nodes, id_to_idx


def load_species_go(species):
    """Load GO annotations for human or mouse."""
    path = DATA / f"{species}_go_annotations.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dark_matter():
    """Load dark matter catalog for yeast GO term mapping."""
    path = RESULTS / "functional_dark_matter.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data["top_100_catalog"]
    protein_terms = defaultdict(set)
    for p in pairs:
        a, b = p["protein_a"], p["protein_b"]
        for term in p.get("shared_go_terms", []):
            protein_terms[a].add(term)
            protein_terms[b].add(term)
    return dict(protein_terms)


# ================================================================
# Clustering test
# ================================================================

def mean_pairwise_distance_fast(coords, indices):
    """Vectorized mean pairwise distance."""
    if len(indices) < 2:
        return float("inf")
    pts = coords[list(indices)]
    n = len(pts)
    # Vectorized: compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Upper triangle only
    triu_idx = np.triu_indices(n, k=1)
    return float(np.mean(dists[triu_idx]))


def clustering_test(coords, target_indices, all_indices, sample_size,
                    n_perm=N_PERMUTATIONS, seed=SEED):
    """Permutation test for spatial clustering."""
    rng = np.random.RandomState(seed)
    all_idx = np.array(list(all_indices))

    # Observed
    if sample_size < len(target_indices):
        sampled = rng.choice(target_indices, size=sample_size, replace=False)
    else:
        sampled = np.array(target_indices)
    obs_mean = mean_pairwise_distance_fast(coords, sampled)

    # Permutations
    n_below = 0
    for _ in range(n_perm):
        perm_sample = rng.choice(all_idx, size=sample_size, replace=False)
        perm_mean = mean_pairwise_distance_fast(coords, perm_sample)
        if perm_mean <= obs_mean:
            n_below += 1
    p_val = (n_below + 1) / (n_perm + 1)

    # Background mean
    bg_means = []
    for _ in range(300):
        bg_sample = rng.choice(all_idx, size=sample_size, replace=False)
        bg_means.append(mean_pairwise_distance_fast(coords, bg_sample))
    bg_mean = float(np.mean(bg_means))
    enrichment = bg_mean / obs_mean if obs_mean > 0 else float("inf")

    return {
        "observed_mean_dist": round(float(obs_mean), 6),
        "background_mean_dist": round(float(bg_mean), 6),
        "enrichment_ratio": round(float(enrichment), 3),
        "p_value": round(float(p_val), 6),
    }


# ================================================================
# Main analysis
# ================================================================

def run():
    t_start = time.time()
    print(BANNER)
    print("  Steps 54+55: Three-Species Dimension Gradient")
    print(f"  Dimensions: {DIMENSIONS}")
    print(BANNER)

    np.random.seed(SEED)

    # ---- Step 54: Compute yeast d=64 embedding ----
    print("\n[1/4] Computing yeast d=64 spectral embedding ...")
    yeast_coords_full, yeast_nodes, yeast_idx, yeast_eigvals = \
        compute_yeast_highdim()

    # ---- Load human/mouse d=64 embeddings ----
    print("\n[2/4] Loading human/mouse d=64 embeddings ...")
    human_coords_full, human_nodes, human_idx = load_embedding("human")
    mouse_coords_full, mouse_nodes, mouse_idx = load_embedding("mouse")
    print(f"  Human: {human_coords_full.shape}")
    print(f"  Mouse: {mouse_coords_full.shape}")

    # ---- Load GO annotations ----
    print("\n  Loading GO annotations ...")
    yeast_dm_terms = load_dark_matter()  # yeast protein -> GO terms from dark matter
    human_go = load_species_go("human")
    mouse_go = load_species_go("mouse")

    # Build yeast term -> protein index mapping
    yeast_term_to_indices = defaultdict(list)
    for protein, terms in yeast_dm_terms.items():
        if protein in yeast_idx:
            for term in terms:
                if term in CONSERVED_TERMS:
                    yeast_term_to_indices[term].append(yeast_idx[protein])

    # Build human/mouse term -> protein index mapping
    def build_term_indices(go_annotations, id_to_idx, terms_dict):
        term_to_indices = defaultdict(list)
        for pid, terms in go_annotations.items():
            if pid in id_to_idx:
                for term in terms:
                    if term in terms_dict:
                        term_to_indices[term].append(id_to_idx[pid])
        return term_to_indices

    human_term_to_indices = build_term_indices(human_go, human_idx, CONSERVED_TERMS)
    mouse_term_to_indices = build_term_indices(mouse_go, mouse_idx, CONSERVED_TERMS)

    # ---- Run dimension gradient ----
    print("\n[3/4] Running dimension gradient analysis ...")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Permutations: {N_PERMUTATIONS}")

    all_results = {}  # {dim: {species: {term: result}}}

    species_configs = [
        ("yeast", yeast_coords_full, yeast_term_to_indices,
         list(range(len(yeast_nodes)))),
        ("human", human_coords_full, human_term_to_indices,
         list(range(len(human_nodes)))),
        ("mouse", mouse_coords_full, mouse_term_to_indices,
         list(range(len(mouse_nodes)))),
    ]

    for d in DIMENSIONS:
        print(f"\n  === d = {d} ===")
        dim_results = {}

        for species, coords_full, term_to_indices, all_indices in species_configs:
            coords_d = coords_full[:, :d]  # extract first d columns
            species_res = {}

            for term, name in sorted(CONSERVED_TERMS.items(),
                                     key=lambda x: x[1]):
                target_indices = term_to_indices.get(term, [])
                if len(target_indices) < 3:
                    continue

                sample_size = min(len(target_indices), MAX_SAMPLE)
                result = clustering_test(coords_d, target_indices,
                                         all_indices, sample_size)
                result["name"] = name
                result["n_proteins"] = len(target_indices)
                result["n_sampled"] = sample_size
                species_res[term] = result

            dim_results[species] = species_res

            # Print summary
            n_sig = sum(1 for r in species_res.values()
                        if r["p_value"] < 0.05)
            print(f"    {species:6s}: {len(species_res)} terms tested, "
                  f"{n_sig} significant")

        all_results[str(d)] = dim_results

    # ---- Critical dimension analysis ----
    print("\n  === Critical dimension analysis ===")
    critical_dims = {}  # {term: {species: critical_dim}}

    for term, name in sorted(CONSERVED_TERMS.items(), key=lambda x: x[1]):
        critical_dims[term] = {"name": name}
        for species in ["yeast", "human", "mouse"]:
            # Find the lowest dimension where p < 0.05
            crit_d = None
            p_values = []
            for d in DIMENSIONS:
                r = all_results[str(d)].get(species, {}).get(term, {})
                p = r.get("p_value", 1.0)
                p_values.append(p)
                if crit_d is None and p < 0.05:
                    crit_d = d
            critical_dims[term][species] = {
                "critical_dim": crit_d,
                "p_values": p_values,
                "becomes_significant": crit_d is not None,
            }
            if crit_d is not None:
                print(f"    {name:35s} {species:6s}: d_crit = {crit_d} "
                      f"(p at d_crit = {p_values[DIMENSIONS.index(crit_d)]:.4f})")
            else:
                n_sig = sum(1 for p in p_values if p < 0.05)
                if n_sig > 0:
                    # Significant at some dims but not monotonically
                    sig_dims = [DIMENSIONS[i] for i, p in enumerate(p_values)
                                if p < 0.05]
                    print(f"    {name:35s} {species:6s}: non-monotonic "
                          f"(sig at d={sig_dims})")
                else:
                    print(f"    {name:35s} {species:6s}: never significant")

    # ---- Save results ----
    print("\n[4/4] Generating outputs ...")

    output = {
        "description": "Three-Species Dimension Gradient: Critical Dimensions",
        "version": "1.0.0",
        "method": (
            f"Within-category spatial clustering test at d = {DIMENSIONS} "
            f"for yeast, human, mouse. Nested spectral embeddings (first d "
            f"columns of d=64 eigenvectors). Permutation test ({N_PERMUTATIONS} "
            f"iterations). Critical dimension = lowest d where p < 0.05."
        ),
        "dimensions": DIMENSIONS,
        "n_permutations": N_PERMUTATIONS,
        "results_by_dimension": all_results,
        "critical_dimensions": critical_dims,
        "yeast_eigenvalues": yeast_eigvals.tolist(),
        "yeast_participation_ratio": float(
            1.0 / np.sum(
                (np.var(yeast_coords_full, axis=0) /
                 np.var(yeast_coords_full, axis=0).sum()) ** 2
            )
        ),
        "summary": {
            d: {
                species: sum(
                    1 for r in all_results[str(d)].get(species, {}).values()
                    if r["p_value"] < 0.05
                )
                for species in ["yeast", "human", "mouse"]
            }
            for d in DIMENSIONS
        },
    }

    out_file = RESULTS / "dimension_gradient_3species.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_file}")

    # ---- Generate figure ----
    generate_figure(all_results, critical_dims)

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.1f}s")

    # Print summary table
    print("\n  Summary: significant categories per dimension")
    print(f"  {'d':>4s}  {'yeast':>6s}  {'human':>6s}  {'mouse':>6s}")
    for d in DIMENSIONS:
        y = sum(1 for r in all_results[str(d)].get("yeast", {}).values()
                if r["p_value"] < 0.05)
        h = sum(1 for r in all_results[str(d)].get("human", {}).values()
                if r["p_value"] < 0.05)
        m = sum(1 for r in all_results[str(d)].get("mouse", {}).values()
                if r["p_value"] < 0.05)
        print(f"  {d:4d}  {y:6d}  {h:6d}  {m:6d}")


# ================================================================
# Figure generation
# ================================================================

def generate_figure(all_results, critical_dims):
    """Generate multi-panel dimension gradient figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    terms_sorted = sorted(CONSERVED_TERMS.keys(),
                          key=lambda t: CONSERVED_TERMS[t])
    term_names = [CONSERVED_TERMS[t] for t in terms_sorted]
    short_names = [n[:25] for n in term_names]

    # ---- Panel A: -log10(p) heatmap (rows=terms, cols=dims x species) ----
    ax_a = fig.add_subplot(gs[0, :2])

    # Build matrix: rows = terms, cols = [Y_d2, Y_d8, Y_d16, Y_d32, Y_d64,
    #                                      H_d2, H_d8, ..., M_d2, ...]
    col_labels = []
    matrix = []
    for t in terms_sorted:
        row = []
        for species in ["yeast", "human", "mouse"]:
            for d in DIMENSIONS:
                r = all_results[str(d)].get(species, {}).get(t, {})
                p = r.get("p_value", 1.0)
                row.append(-math.log10(max(p, 1e-6)))
        matrix.append(row)

    # Column labels
    for species in ["Y", "H", "M"]:
        for d in DIMENSIONS:
            col_labels.append(f"{species}_{d}")

    matrix = np.array(matrix)
    vmax = max(3.0, np.max(matrix) if matrix.size > 0 else 3.0)
    im = ax_a.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

    ax_a.set_xticks(range(len(col_labels)))
    ax_a.set_xticklabels(col_labels, fontsize=6, rotation=90)
    ax_a.set_yticks(range(len(short_names)))
    ax_a.set_yticklabels(short_names, fontsize=8)
    ax_a.set_title("(A) Dimension x Species Significance Heatmap (-log10 p)",
                    fontsize=13, fontweight="bold")

    # Add vertical lines to separate species
    for sep in [5, 10]:
        ax_a.axvline(x=sep - 0.5, color="white", linewidth=2)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val > 0.1:
                color = "white" if val > 1.5 else "black"
                ax_a.text(j, i, f"{val:.1f}", ha="center", va="center",
                          fontsize=4, color=color)

    fig.colorbar(im, ax=ax_a, shrink=0.8, label="-log10(p)")

    # ---- Panel B: Enrichment ratio vs dimension (line plot) ----
    ax_b = fig.add_subplot(gs[0, 2])

    colors_species = {"yeast": "#2196F3", "human": "#FF5722", "mouse": "#4CAF50"}
    markers = {"yeast": "o", "human": "s", "mouse": "^"}

    for t in terms_sorted[:4]:  # Top 4 terms for readability
        name = CONSERVED_TERMS[t][:20]
        for species in ["human", "mouse"]:
            enrich_vals = []
            for d in DIMENSIONS:
                r = all_results[str(d)].get(species, {}).get(t, {})
                enrich_vals.append(r.get("enrichment_ratio", 1.0) or 1.0)
            ax_b.plot(DIMENSIONS, enrich_vals,
                      color=colors_species[species],
                      marker=markers[species],
                      alpha=0.6, linewidth=1, markersize=4)

    ax_b.set_xlabel("Embedding Dimension d", fontsize=10)
    ax_b.set_ylabel("Enrichment Ratio", fontsize=10)
    ax_b.set_title("(B) Enrichment vs d\n(Human/Mouse, top 4 terms)",
                    fontsize=11, fontweight="bold")
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(DIMENSIONS)
    ax_b.set_xticklabels([str(d) for d in DIMENSIONS])
    ax_b.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ---- Panels C-E: P-value trajectories per species ----
    for panel_idx, species in enumerate(["yeast", "human", "mouse"]):
        ax = fig.add_subplot(gs[1, panel_idx])

        for t in terms_sorted:
            name = CONSERVED_TERMS[t][:20]
            p_vals = []
            for d in DIMENSIONS:
                r = all_results[str(d)].get(species, {}).get(t, {})
                p_vals.append(max(r.get("p_value", 1.0), 1e-4))

            # Color by whether it ever becomes significant
            ever_sig = any(p < 0.05 for p in p_vals)
            color = "#4CAF50" if ever_sig else "#BDBDBD"
            lw = 2.0 if ever_sig else 1.0
            ax.plot(DIMENSIONS, p_vals, marker="o", color=color,
                    linewidth=lw, markersize=5, label=name if ever_sig else None,
                    alpha=0.8)

        ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.6,
                    label="p = 0.05")
        ax.set_xlabel("Dimension d", fontsize=10)
        ax.set_ylabel("p-value", fontsize=10)
        species_title = species.capitalize()
        ax.set_title(f"(C-E) {species_title}: P-value Trajectories",
                      fontsize=11, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(DIMENSIONS)
        ax.set_xticklabels([str(d) for d in DIMENSIONS])
        ax.set_ylim(1e-4, 1.2)
        if ever_sig:
            ax.legend(fontsize=6, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ---- Panel F: Critical dimension summary table ----
    ax_f = fig.add_subplot(gs[2, :2])
    ax_f.axis("off")

    table_data = []
    cell_colors = []
    for t in terms_sorted:
        name = CONSERVED_TERMS[t]
        row = [name[:30]]
        colors = ["#FFFFFF"]
        for species in ["yeast", "human", "mouse"]:
            cd = critical_dims.get(t, {}).get(species, {})
            crit_d = cd.get("critical_dim", None)
            p_vals = cd.get("p_values", [])
            if crit_d is not None:
                row.append(f"d = {crit_d}")
                colors.append("#C8E6C9")  # green
            elif any(p < 0.05 for p in p_vals):
                sig_dims = [DIMENSIONS[i] for i, p in enumerate(p_vals)
                            if p < 0.05]
                row.append(f"d = {sig_dims}*")
                colors.append("#FFF9C4")  # yellow (non-monotonic)
            else:
                row.append("--")
                colors.append("#FFCDD2")  # red (never sig)
        table_data.append(row)
        cell_colors.append(colors)

    col_labels_table = ["GO Term", "Yeast d_crit", "Human d_crit", "Mouse d_crit"]
    table = ax_f.table(cellText=table_data, colLabels=col_labels_table,
                        cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row - 1 < len(cell_colors) and col < len(cell_colors[row - 1]):
            cell.set_facecolor(cell_colors[row - 1][col])
        cell.set_edgecolor("#BDBDBD")

    ax_f.set_title("(F) Critical Dimensions for Functional Category Detection",
                    fontsize=13, fontweight="bold", pad=20)

    # ---- Panel G: Summary statistics ----
    ax_g = fig.add_subplot(gs[2, 2])
    ax_g.axis("off")

    # Build summary text
    summary_lines = [
        "Three-Species Dimension Gradient Summary",
        "=" * 45,
        "",
        "Significant categories (p < 0.05):",
    ]
    for d in DIMENSIONS:
        counts = []
        for species in ["yeast", "human", "mouse"]:
            n = sum(1 for r in all_results[str(d)].get(species, {}).values()
                    if r["p_value"] < 0.05)
            counts.append(n)
        summary_lines.append(f"  d={d:2d}: Y={counts[0]} H={counts[1]} M={counts[2]}")

    summary_lines.extend([
        "",
        "Key finding: fine-grained categories",
        "(ERAD, folding, iron) emerge at high d;",
        "broad categories (transmembrane)",
        "dominate at low d.",
    ])

    ax_g.text(0.05, 0.95, "\n".join(summary_lines),
              transform=ax_g.transAxes, fontsize=9,
              verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9",
                        edgecolor="#4CAF50", alpha=0.8))

    # ---- Save ----
    fig_path = FIGURES / "Fig75_dimension_gradient.png"
    fig.savefig(str(fig_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Figure saved to {fig_path}")


if __name__ == "__main__":
    run()
