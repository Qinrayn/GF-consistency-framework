#!/usr/bin/env python3
"""
Cross-Species Functional Conservation of Dark Matter Categories (Step 49)
========================================================================

Tests whether the GO biological process categories enriched among yeast
dark matter proteins also show spatial clustering in human/mouse Spectral
embedding space.

Rather than requiring 1:1 ortholog mapping (which fails for many
yeast-specific proteins), we test conservation at the *functional
category* level:

  For each enriched GO BP term:
    1. In yeast: are dark matter proteins with this term closer in the
       high-D Spectral embedding than random annotated proteins?
    2. In human: are proteins annotated with the SAME GO term closer
       in the human Spectral embedding than random?
    3. In mouse: same test.

  If both species show significant clustering for the same term,
  the functional-geometric relationship is evolutionarily conserved.

Output:
  - results/cross_species_dark_matter.json
  - figures/Fig73_cross_species_dark_matter.png
"""

from __future__ import annotations

import gzip
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

# GO terms enriched in dark matter that have cross-species conservation
# (yeast-specific terms like ascospore formation excluded)
CONSERVED_TERMS = {
    "GO:0055085": "transmembrane transport",
    "GO:0036503": "ERAD pathway",
    "GO:0034599": "cellular response to oxidative stress",
    "GO:0006457": "protein folding",
    "GO:0016126": "sterol biosynthetic process",
    "GO:0006631": "fatty acid metabolic process",
    "GO:0006879": "intracellular iron ion homeostasis",
}

# Yeast-specific terms (no human ortholog process)
YEAST_SPECIFIC_TERMS = {
    "GO:0045944": "positive regulation of transcription by RNA Pol II",
    "GO:0030437": "ascospore formation",
    "GO:0007124": "pseudohyphal growth",
    "GO:0031505": "fungal-type cell wall organization",
    "GO:0030476": "ascospore wall assembly",
    "GO:0042147": "retrograde transport, endosome to Golgi",
    "GO:0001403": "invasive growth in response to glucose limitation",
    "GO:0007117": "budding cell bud growth",
    "GO:0006355": "regulation of DNA-templated transcription",
    "GO:0007059": "chromosome segregation",
    "GO:0006357": "regulation of transcription by RNA Pol II",
    "GO:0000122": "negative regulation of transcription by RNA Pol II",
}


# ================================================================
# Data loading
# ================================================================

def load_dark_matter():
    """Load dark matter pairs and extract unique proteins + GO annotations."""
    path = RESULTS / "functional_dark_matter.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pairs = data["top_100_catalog"]
    proteins = set()
    # Map: protein -> set of GO terms it appears with in dark matter pairs
    protein_terms = defaultdict(set)
    for p in pairs:
        a, b = p["protein_a"], p["protein_b"]
        proteins.add(a)
        proteins.add(b)
        for term in p.get("shared_go_terms", []):
            protein_terms[a].add(term)
            protein_terms[b].add(term)

    return pairs, sorted(proteins), dict(protein_terms), data["characterisation"]["go_enrichment"]


def load_yeast_embedding():
    """Load full-network yeast Spectral embedding (multi-D)."""
    emb_file = EMBEDDINGS / "Spectral_full.npy"
    nodes_file = EMBEDDINGS / "Spectral_full_nodes.json"
    coords = np.load(str(emb_file))
    with open(nodes_file, encoding="utf-8") as f:
        nodes = json.load(f)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    return coords, nodes, node_to_idx


def load_species_embedding(species):
    """Load human or mouse 2D Spectral embedding from JSON."""
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


def build_alias_map():
    """Build ORF -> preferred gene name mapping from STRING aliases."""
    alias_path = DATA / "4932.protein.aliases.v11.5.txt.gz"
    orf_to_gene = {}
    orf_to_aliases = defaultdict(list)

    with gzip.open(str(alias_path), "rt", encoding="utf-8") as f:
        header = next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            string_id, alias, source = parts[0], parts[1], parts[2]
            # Extract ORF from STRING ID (e.g., 4932.YAL001C -> YAL001C)
            orf = string_id.split(".", 1)[-1] if "." in string_id else string_id
            orf_to_aliases[orf].append((alias, source))

    # For each ORF, pick the best gene name
    # Priority: Ensembl_SGD_GENE (standard gene name) > SGD_ID > first synonym
    for orf, aliases in orf_to_aliases.items():
        gene_name = None
        for alias, source in aliases:
            if source == "Ensembl_SGD_GENE" and alias != orf:
                gene_name = alias
                break
        if gene_name is None:
            for alias, source in aliases:
                if source == "BLAST_UniProt_GN_OrderedLocusNames" and alias != orf:
                    gene_name = alias
                    break
        if gene_name is None:
            # Use the ORF name itself if no gene name found
            gene_name = orf
        orf_to_gene[orf] = gene_name

    return orf_to_gene


# ================================================================
# Distance computation
# ================================================================

def pairwise_distances(coords, indices):
    """Compute all pairwise Euclidean distances for a set of indices."""
    if len(indices) < 2:
        return []
    pts = coords[list(indices)]
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            dists.append(float(d))
    return dists


def mean_pairwise_distance(coords, indices):
    """Mean pairwise Euclidean distance for a group of proteins."""
    dists = pairwise_distances(coords, indices)
    return float(np.mean(dists)) if dists else float("inf")


def permutation_test(observed_mean, coords, all_indices, group_size, n_perm=N_PERMUTATIONS):
    """Permutation test: is the observed mean distance significantly small?"""
    rng = np.random.RandomState(SEED)
    n_below = 0
    all_idx = np.array(list(all_indices))
    for _ in range(n_perm):
        sample = rng.choice(all_idx, size=group_size, replace=False)
        perm_dists = pairwise_distances(coords, sample)
        perm_mean = float(np.mean(perm_dists)) if perm_dists else float("inf")
        if perm_mean <= observed_mean:
            n_below += 1
    p_value = (n_below + 1) / (n_perm + 1)
    return p_value


# ================================================================
# Analysis
# ================================================================

def run():
    t_start = time.time()
    print(BANNER)
    print("  Phase 21: Cross-Species Functional Conservation")
    print("  of Dark Matter Categories")
    print(BANNER)

    random.seed(SEED)
    np.random.seed(SEED)

    # ---- Load data ----
    print("\n[1/6] Loading dark matter catalog ...")
    pairs, dm_proteins, protein_terms, go_enrichment = load_dark_matter()
    print(f"  {len(pairs)} pairs, {len(dm_proteins)} unique proteins")

    print("[2/6] Loading yeast Spectral embedding ...")
    yeast_coords, yeast_nodes, yeast_idx = load_yeast_embedding()
    print(f"  {yeast_coords.shape[0]} nodes, {yeast_coords.shape[1]}D")

    print("[3/6] Loading human/mouse embeddings + GO annotations ...")
    human_coords, human_ids, human_idx = load_species_embedding("human")
    mouse_coords, mouse_ids, mouse_idx = load_species_embedding("mouse")
    human_go = load_species_go("human")
    mouse_go = load_species_go("mouse")
    print(f"  Human: {len(human_ids)} proteins, {len(human_go)} with GO")
    print(f"  Mouse: {len(mouse_ids)} proteins, {len(mouse_go)} with GO")

    print("[4/6] Resolving ORF names to gene names ...")
    alias_map = build_alias_map()
    dm_gene_names = {}
    for p in dm_proteins:
        dm_gene_names[p] = alias_map.get(p, p)
    named = sum(1 for g in dm_gene_names.values() if g != list(dm_gene_names.keys())[list(dm_gene_names.values()).index(g)])
    resolved = sum(1 for orf, gene in dm_gene_names.items() if gene != orf)
    print(f"  {resolved}/{len(dm_proteins)} proteins resolved to gene names")

    # ---- Build GO term -> dark matter protein index ----
    # Map each GO term to indices of DM proteins annotated with it
    dm_protein_set = set(dm_proteins)
    term_to_dm_indices = defaultdict(list)
    for i, p in enumerate(dm_proteins):
        for term in protein_terms.get(p, []):
            term_to_dm_indices[term].append(i)

    # ---- Compute yeast within-term clustering ----
    print("\n[5/6] Computing within-category clustering ...")
    dm_yeast_indices = [yeast_idx[p] for p in dm_proteins if p in yeast_idx]
    all_yeast_indices = list(range(len(yeast_nodes)))

    print("  Yeast dark matter proteins:")
    yeast_results = {}
    for term, name in sorted(CONSERVED_TERMS.items(), key=lambda x: x[1]):
        term_dm = term_to_dm_indices.get(term, [])
        if len(term_dm) < 2:
            continue
        # Get yeast embedding indices for DM proteins with this term
        yeast_term_idx = [yeast_idx[dm_proteins[i]] for i in term_dm
                          if dm_proteins[i] in yeast_idx]
        if len(yeast_term_idx) < 2:
            continue

        obs_mean = mean_pairwise_distance(yeast_coords, yeast_term_idx)
        p_val = permutation_test(obs_mean, yeast_coords, all_yeast_indices,
                                 len(yeast_term_idx))

        # Background: mean distance of random DM-sized samples
        rng = np.random.RandomState(SEED)
        bg_means = []
        for _ in range(500):
            sample = rng.choice(all_yeast_indices, size=len(yeast_term_idx),
                                replace=False)
            bg_means.append(mean_pairwise_distance(yeast_coords, sample))
        bg_mean = float(np.mean(bg_means))
        enrichment = bg_mean / obs_mean if obs_mean > 0 else float("inf")

        yeast_results[term] = {
            "name": name,
            "n_dm_proteins": len(yeast_term_idx),
            "observed_mean_dist": round(obs_mean, 6),
            "background_mean_dist": round(bg_mean, 6),
            "enrichment_ratio": round(enrichment, 3),
            "p_value": round(p_val, 6),
        }
        mark = "*" if p_val < 0.05 else " "
        print(f"  {mark} {name:45s}: n={len(yeast_term_idx):2d}, "
              f"enrich={enrichment:.1f}x, p={p_val:.4f}")

    # ---- Compute human/mouse within-term clustering ----
    print("\n  Human proteins:")
    human_results = {}
    for term, name in sorted(CONSERVED_TERMS.items(), key=lambda x: x[1]):
        # Find human proteins with this GO term
        human_term_ids = [pid for pid, terms in human_go.items()
                          if term in terms and pid in human_idx]
        if len(human_term_ids) < 5:
            continue

        human_term_indices = [human_idx[pid] for pid in human_term_ids]
        all_human_indices = list(range(len(human_ids)))

        obs_mean = mean_pairwise_distance(human_coords, human_term_indices)
        p_val = permutation_test(obs_mean, human_coords, all_human_indices,
                                 min(len(human_term_indices), 50))

        # Background
        rng = np.random.RandomState(SEED)
        sample_size = min(len(human_term_indices), 50)
        bg_means = []
        for _ in range(500):
            sample = rng.choice(all_human_indices, size=sample_size,
                                replace=False)
            bg_means.append(mean_pairwise_distance(human_coords, sample))
        bg_mean = float(np.mean(bg_means))
        enrichment = bg_mean / obs_mean if obs_mean > 0 else float("inf")

        human_results[term] = {
            "name": name,
            "n_proteins": len(human_term_ids),
            "n_sampled": sample_size,
            "observed_mean_dist": round(obs_mean, 6),
            "background_mean_dist": round(bg_mean, 6),
            "enrichment_ratio": round(enrichment, 3),
            "p_value": round(p_val, 6),
        }
        mark = "*" if p_val < 0.05 else " "
        print(f"  {mark} {name:45s}: n={len(human_term_ids):3d}, "
              f"enrich={enrichment:.2f}x, p={p_val:.4f}")

    print("\n  Mouse proteins:")
    mouse_results = {}
    for term, name in sorted(CONSERVED_TERMS.items(), key=lambda x: x[1]):
        mouse_term_ids = [pid for pid, terms in mouse_go.items()
                          if term in terms and pid in mouse_idx]
        if len(mouse_term_ids) < 5:
            continue

        mouse_term_indices = [mouse_idx[pid] for pid in mouse_term_ids]
        all_mouse_indices = list(range(len(mouse_ids)))

        obs_mean = mean_pairwise_distance(mouse_coords, mouse_term_indices)
        p_val = permutation_test(obs_mean, mouse_coords, all_mouse_indices,
                                 min(len(mouse_term_indices), 50))

        rng = np.random.RandomState(SEED)
        sample_size = min(len(mouse_term_indices), 50)
        bg_means = []
        for _ in range(500):
            sample = rng.choice(all_mouse_indices, size=sample_size,
                                replace=False)
            bg_means.append(mean_pairwise_distance(mouse_coords, sample))
        bg_mean = float(np.mean(bg_means))
        enrichment = bg_mean / obs_mean if obs_mean > 0 else float("inf")

        mouse_results[term] = {
            "name": name,
            "n_proteins": len(mouse_term_ids),
            "n_sampled": sample_size,
            "observed_mean_dist": round(obs_mean, 6),
            "background_mean_dist": round(bg_mean, 6),
            "enrichment_ratio": round(enrichment, 3),
            "p_value": round(p_val, 6),
        }
        mark = "*" if p_val < 0.05 else " "
        print(f"  {mark} {name:45s}: n={len(mouse_term_ids):3d}, "
              f"enrich={enrichment:.2f}x, p={p_val:.4f}")

    # ---- Conservation summary ----
    conserved_count = 0
    for term in CONSERVED_TERMS:
        y_sig = yeast_results.get(term, {}).get("p_value", 1.0) < 0.05
        h_sig = human_results.get(term, {}).get("p_value", 1.0) < 0.05
        m_sig = mouse_results.get(term, {}).get("p_value", 1.0) < 0.05
        if y_sig and (h_sig or m_sig):
            conserved_count += 1

    # ---- Save results ----
    print("\n[6/6] Generating outputs ...")

    # Build protein name table
    protein_table = []
    for p in dm_proteins:
        protein_table.append({
            "orf": p,
            "gene_name": dm_gene_names.get(p, p),
            "go_terms": sorted(protein_terms.get(p, [])),
        })

    output = {
        "description": "Cross-Species Functional Conservation of Dark Matter Categories",
        "version": "1.0.0",
        "method": (
            "Category-level functional conservation test: for each GO BP term "
            "enriched in yeast dark matter, test whether proteins annotated with "
            "the same term in human/mouse are spatially clustered in the Spectral "
            "embedding. Permutation test (5000 iterations) against random proteins."
        ),
        "n_dark_matter_pairs": len(pairs),
        "n_dark_matter_proteins": len(dm_proteins),
        "n_conserved_categories": conserved_count,
        "n_total_conserved_terms": len(CONSERVED_TERMS),
        "n_yeast_specific_terms": len(YEAST_SPECIFIC_TERMS),
        "yeast_clustering": yeast_results,
        "human_clustering": human_results,
        "mouse_clustering": mouse_results,
        "protein_name_table": protein_table,
        "conserved_terms": CONSERVED_TERMS,
        "yeast_specific_terms": YEAST_SPECIFIC_TERMS,
        "conclusion": (
            f"Of {len(CONSERVED_TERMS)} cross-species GO BP categories enriched in "
            f"dark matter, {conserved_count} show significant spatial clustering "
            f"(p<0.05) in both yeast and at least one mammalian species. "
            f"This suggests {'conserved' if conserved_count > 0 else 'limited'} "
            f"functional-geometric organization across eukaryotes."
        ),
    }

    out_file = RESULTS / "cross_species_dark_matter.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_file}")

    # ---- Generate figure ----
    generate_figure(output, yeast_results, human_results, mouse_results)

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Conserved categories: {conserved_count}/{len(CONSERVED_TERMS)}")
    print(f"  Conclusion: {output['conclusion']}")


# ================================================================
# Figure generation
# ================================================================

def generate_figure(output, yeast_res, human_res, mouse_res):
    """Generate 3-panel cross-species conservation figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Common terms that have results in all three species
    common_terms = sorted(
        [t for t in CONSERVED_TERMS
         if t in yeast_res and t in human_res and t in mouse_res],
        key=lambda t: CONSERVED_TERMS[t]
    )
    if not common_terms:
        # Fall back to any terms available
        common_terms = sorted(
            [t for t in CONSERVED_TERMS if t in yeast_res],
            key=lambda t: CONSERVED_TERMS[t]
        )

    names = [CONSERVED_TERMS[t][:35] for t in common_terms]

    # ---- Panel A: Enrichment ratio comparison ----
    ax_a = fig.add_subplot(gs[0, :2])
    x = np.arange(len(common_terms))
    width = 0.25

    yeast_enrich = [yeast_res.get(t, {}).get("enrichment_ratio", 0)
                    for t in common_terms]
    human_enrich = [human_res.get(t, {}).get("enrichment_ratio", 0)
                    for t in common_terms]
    mouse_enrich = [mouse_res.get(t, {}).get("enrichment_ratio", 0)
                    for t in common_terms]

    bars_y = ax_a.bar(x - width, yeast_enrich, width, label="Yeast (Spectral)",
                       color="#2196F3", alpha=0.85, edgecolor="white")
    bars_h = ax_a.bar(x, human_enrich, width, label="Human (Spectral)",
                       color="#FF5722", alpha=0.85, edgecolor="white")
    bars_m = ax_a.bar(x + width, mouse_enrich, width, label="Mouse (Spectral)",
                       color="#4CAF50", alpha=0.85, edgecolor="white")

    # Add significance markers
    for i, t in enumerate(common_terms):
        y_p = yeast_res.get(t, {}).get("p_value", 1.0)
        h_p = human_res.get(t, {}).get("p_value", 1.0)
        m_p = mouse_res.get(t, {}).get("p_value", 1.0)
        if y_p < 0.05:
            ax_a.text(x[i] - width, yeast_enrich[i] + 0.05, "*",
                      ha="center", va="bottom", fontsize=14, fontweight="bold",
                      color="#2196F3")
        if h_p < 0.05:
            ax_a.text(x[i], human_enrich[i] + 0.05, "*",
                      ha="center", va="bottom", fontsize=14, fontweight="bold",
                      color="#FF5722")
        if m_p < 0.05:
            ax_a.text(x[i] + width, mouse_enrich[i] + 0.05, "*",
                      ha="center", va="bottom", fontsize=14, fontweight="bold",
                      color="#4CAF50")

    ax_a.set_xlabel("GO Biological Process Term", fontsize=11)
    ax_a.set_ylabel("Clustering Enrichment (observed / background)", fontsize=11)
    ax_a.set_title("(A) Functional Category Clustering Across Species", fontsize=13,
                    fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax_a.legend(fontsize=9, loc="upper right")
    ax_a.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ---- Panel B: Conservation summary (heatmap-style) ----
    ax_b = fig.add_subplot(gs[0, 2])

    # Build a matrix: rows=terms, cols=[yeast, human, mouse]
    all_terms_sorted = sorted(
        [t for t in CONSERVED_TERMS if t in yeast_res],
        key=lambda t: CONSERVED_TERMS[t]
    )
    matrix = []
    labels_y = []
    for t in all_terms_sorted:
        row = []
        for species_res in [yeast_res, human_res, mouse_res]:
            if t in species_res:
                p = species_res[t]["p_value"]
                # -log10(p) for visualization
                val = -math.log10(max(p, 1e-6))
                row.append(val)
            else:
                row.append(0)
        matrix.append(row)
        labels_y.append(CONSERVED_TERMS[t][:30])

    matrix = np.array(matrix)
    im = ax_b.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0,
                     vmax=max(3.0, np.max(matrix) if matrix.size > 0 else 3.0))
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(["Yeast", "Human", "Mouse"], fontsize=9)
    ax_b.set_yticks(range(len(labels_y)))
    ax_b.set_yticklabels(labels_y, fontsize=7)
    ax_b.set_title("(B) -log10(p) Significance Heatmap", fontsize=13,
                    fontweight="bold")

    # Annotate cells with p-values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            p_val = None
            species_list = [yeast_res, human_res, mouse_res]
            t = all_terms_sorted[i]
            if t in species_list[j]:
                p_val = species_list[j][t]["p_value"]
            text = f"{p_val:.3f}" if p_val is not None else "-"
            color = "white" if matrix[i, j] > 1.5 else "black"
            ax_b.text(j, i, text, ha="center", va="center", fontsize=6,
                      color=color)

    fig.colorbar(im, ax=ax_b, shrink=0.8, label="-log10(p)")

    # ---- Panel C: Protein count + term classification ----
    ax_c = fig.add_subplot(gs[1, :])

    # Table-style visualization of all GO terms
    conserved = sorted(CONSERVED_TERMS.items(), key=lambda x: x[1])
    yeast_spec = sorted(YEAST_SPECIFIC_TERMS.items(), key=lambda x: x[1])

    # Build table data
    table_data = []
    cell_colors = []
    for term, name in conserved:
        n_yeast = yeast_res.get(term, {}).get("n_dm_proteins", 0)
        n_human = human_res.get(term, {}).get("n_proteins", 0)
        n_mouse = mouse_res.get(term, {}).get("n_proteins", 0)
        y_p = yeast_res.get(term, {}).get("p_value", 1.0)
        h_p = human_res.get(term, {}).get("p_value", 1.0)
        m_p = mouse_res.get(term, {}).get("p_value", 1.0)
        conserved_flag = (y_p < 0.05) and (h_p < 0.05 or m_p < 0.05)
        table_data.append([
            name[:40], "Conserved", str(n_yeast), str(n_human), str(n_mouse),
            f"{y_p:.4f}", f"{h_p:.4f}", f"{m_p:.4f}",
            "Yes" if conserved_flag else "No"
        ])
        cell_colors.append("#E8F5E9" if conserved_flag else "#FFF3E0")

    for term, name in yeast_spec:
        # Count DM proteins for yeast-specific terms
        n_dm = 0
        dm_data = output.get("protein_name_table", [])
        for entry in dm_data:
            if term in entry.get("go_terms", []):
                n_dm += 1
        table_data.append([
            name[:40], "Yeast-specific", str(n_dm), "-", "-",
            "-", "-", "-", "-"
        ])
        cell_colors.append("#E3F2FD")

    col_labels = ["GO Term", "Category", "Yeast n", "Human n", "Mouse n",
                  "Yeast p", "Human p", "Mouse p", "Conserved?"]

    ax_c.axis("off")
    table = ax_c.table(cellText=table_data, colLabels=col_labels,
                       cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)

    # Color cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row - 1 < len(cell_colors):
            cell.set_facecolor(cell_colors[row - 1])
        cell.set_edgecolor("#BDBDBD")

    ax_c.set_title("(C) Dark Matter GO Term Classification & Cross-Species Coverage",
                    fontsize=13, fontweight="bold", pad=20)

    # ---- Save ----
    fig_path = FIGURES / "Fig73_cross_species_dark_matter.png"
    fig.savefig(str(fig_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Figure saved to {fig_path}")


if __name__ == "__main__":
    run()
