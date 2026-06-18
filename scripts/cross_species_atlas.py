#!/usr/bin/env python3
"""
Cross-Species Function Prediction Atlas (Step 70 / Phase 21)
=============================================================

Replicate the yeast multi-ontology atlas on Human and Mouse.

For each species:
  1. Compute Spectral embeddings at d = 2, 64, 256
  2. Run LOTO-CV (BP, MF, CC) at each dimension
  3. Compare with PPI-Neighbors baseline

Key question: Does d=256 exceed PPI topology for human/mouse too?

Output
------
- results/cross_species_atlas.json
- figures/Fig80_cross_species_comparison.png
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
    TARGET_STD,
)
from function_prediction import (
    ppi_neighbor_predict,
    twohop_diffusion_predict,
    knn_predict_fast,
    compute_mean_reciprocal_rank,
    K_MAX,
)

# ============================================================
# Constants
# ============================================================

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()
EMB = get_embeddings_dir()

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

BANNER = "=" * 64

SPECTRAL_DIMENSIONS = [2, 64, 256]

ASPECTS = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}

# Species configurations
SPECIES = {
    "human": {
        "name": "Human",
        "network_loader": "load_human_network",
        "annotation_file": DATA / "human_go_annotations.json",
        "color": "#d62728",
    },
    "mouse": {
        "name": "Mouse",
        "network_loader": "load_mouse_network",
        "annotation_file": DATA / "mouse_go_annotations.json",
        "color": "#3182bd",
    },
    "yeast": {
        "name": "Yeast",
        "network_loader": "load_yeast_network",
        "annotation_file": None,  # uses GAF parsing
        "color": "#2ca02c",
    },
}


# ============================================================
# Network Loaders
# ============================================================

def load_yeast_network():
    """Load yeast PPI network."""
    net_file = DATA / "yeast_ppi_5936.edgelist"
    G = nx.read_edgelist(str(net_file))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def load_human_network():
    """Load human PPI network from STRING v12.0."""
    import os
    links_file = Path(__file__).resolve().parent.parent / "human_validation" / "9606.protein.links.v12.0.txt.gz"
    if not links_file.exists():
        raise FileNotFoundError(f"Human STRING file not found: {links_file}")

    G = nx.Graph()
    with gzip.open(str(links_file), "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("protein1"):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                score = int(parts[2])
                if score >= 700:
                    G.add_edge(parts[0], parts[1])

    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def load_mouse_network():
    """Load mouse PPI network from edgelist."""
    net_file = DATA / "mouse_ppi.edgelist"
    if net_file.exists():
        G = nx.read_edgelist(str(net_file))
    else:
        # Fallback: load from STRING
        links_file = DATA / "10090.protein.links.v11.5.txt.gz"
        G = nx.Graph()
        with gzip.open(str(links_file), "rt") as f:
            for line in f:
                if line.startswith("#") or line.startswith("protein1"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    score = int(parts[2])
                    if score >= 700:
                        G.add_edge(parts[0], parts[1])

    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


# ============================================================
# GO Aspect Mapping
# ============================================================

def load_go_aspect_map():
    """Load GO term -> aspect mapping from go.obo."""
    go_aspect = {}
    obo_file = DATA / "go.obo"
    if not obo_file.exists():
        print("  WARNING: go.obo not found, cannot split by aspect")
        return go_aspect

    current_id = None
    current_namespace = None
    with open(obo_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("namespace:"):
                current_namespace = line.split("namespace: ")[1]
                if current_id:
                    go_aspect[current_id] = current_namespace
                    current_id = None

    return go_aspect


# ============================================================
# Annotation Loading (by aspect)
# ============================================================

def load_annotations_by_aspect(species_key, go_aspect_map, network_nodes):
    """Load annotations split by GO aspect.

    Returns dict: {aspect_short: {protein_id: set(go_terms)}}
    """
    aspect_annotations = {short: defaultdict(set) for short in ASPECTS.values()}

    if species_key == "yeast":
        # Parse from SGD GAF
        from function_prediction import build_alias_mapping, EXPERIMENTAL_CODES
        sgd_map, orf_map, _ = build_alias_mapping()
        gaf_file = DATA / "gene_association.sgd.gaf.gz"
        with gzip.open(str(gaf_file), "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("!") or line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 10:
                    continue
                evidence = cols[6]
                aspect = cols[8]
                if evidence not in EXPERIMENTAL_CODES or aspect not in ("P", "F", "C"):
                    continue
                go_term = cols[4]
                if not go_term.startswith("GO:"):
                    continue

                sgd_id = cols[1]
                gene_sym = cols[2]
                orf_name = cols[10] if len(cols) > 10 else ""

                string_id = None
                if sgd_id in sgd_map:
                    c = sgd_map[sgd_id]
                    if c in network_nodes:
                        string_id = c
                if string_id is None and orf_name and orf_name in orf_map:
                    c = orf_map[orf_name]
                    if c in network_nodes:
                        string_id = c
                if string_id is None and gene_sym in orf_map:
                    c = orf_map[gene_sym]
                    if c in network_nodes:
                        string_id = c

                if string_id:
                    short = {"P": "BP", "F": "MF", "C": "CC"}[aspect]
                    aspect_annotations[short][string_id].add(go_term)
    else:
        # Load from JSON annotation file
        ann_file = SPECIES[species_key]["annotation_file"]
        with open(ann_file, encoding="utf-8") as f:
            raw_annotations = json.load(f)

        for pid, terms in raw_annotations.items():
            if pid not in network_nodes:
                continue
            for term in terms:
                aspect_full = go_aspect_map.get(term)
                if aspect_full and aspect_full in ASPECTS:
                    short = ASPECTS[aspect_full]
                    aspect_annotations[short][pid].add(term)

    # Convert to regular dict
    result = {}
    for short in ASPECTS.values():
        result[short] = {k: v for k, v in aspect_annotations[short].items()
                        if len(v) > 0}

    return result


# ============================================================
# Spectral Embedding
# ============================================================

def compute_spectral_embedding(graph, dim):
    """Compute Spectral embedding at given dimension."""
    nodes = sorted(graph.nodes())
    n = len(nodes)

    adj = nx.adjacency_matrix(graph, nodelist=nodes, weight=None).astype(float)
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(degrees)

    D_inv_sqrt = diags(d_inv_sqrt)
    L_norm = D_inv_sqrt @ (diags(degrees) - adj) @ D_inv_sqrt

    n_eigs = min(dim + 1, n - 2)
    eigenvalues, eigenvectors = eigsh(L_norm, k=n_eigs, which="SM", tol=1e-6)

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    coords = eigenvectors[:, 1:dim + 1]

    for j in range(coords.shape[1]):
        col_std = coords[:, j].std()
        if col_std > 1e-10:
            coords[:, j] = coords[:, j] / col_std * TARGET_STD

    return coords, nodes, eigenvalues[:dim + 1]


# ============================================================
# LOTO-CV Engine
# ============================================================

def run_loto(coords, nodes, graph, annotations, term_freq, method_name,
             compute_ppi=False):
    """Run LOTO-CV for a single method/dimension/ontology."""
    node_set = set(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in node_set and pid in graph
    }

    n_trials = sum(len(terms) for terms in query_proteins.values())
    if n_trials == 0:
        return 0.0, {}

    n_neighbors = min(K_MAX + 1, len(coords))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(coords)

    random_ranking = term_freq.most_common()

    rank_results = []
    completed = 0
    t0 = time.time()

    for pid, terms in sorted(query_proteins.items()):
        for hidden_term in sorted(terms):
            trial_rank = {}

            if pid in node_to_idx:
                query_idx = node_to_idx[pid]
                preds = knn_predict_fast(
                    query_idx, nn_model, coords, nodes,
                    annotations, k=K_MAX, hidden_term=hidden_term,
                )
                pred_terms = [t for t, _ in preds]
                try:
                    trial_rank[method_name] = pred_terms.index(hidden_term) + 1
                except ValueError:
                    trial_rank[method_name] = 0

            if compute_ppi:
                ppi_preds = ppi_neighbor_predict(pid, graph, annotations, hidden_term)
                ppi_list = [t for t, _ in ppi_preds]
                try:
                    trial_rank["PPI-Neighbors"] = ppi_list.index(hidden_term) + 1
                except ValueError:
                    trial_rank["PPI-Neighbors"] = 0

                hop_preds = twohop_diffusion_predict(pid, graph, annotations, hidden_term)
                hop_list = [t for t, _ in hop_preds]
                try:
                    trial_rank["2-Hop Diffusion"] = hop_list.index(hidden_term) + 1
                except ValueError:
                    trial_rank["2-Hop Diffusion"] = 0

                rand_list = [t for t, _ in random_ranking]
                try:
                    trial_rank["Random"] = rand_list.index(hidden_term) + 1
                except ValueError:
                    trial_rank["Random"] = 0

            rank_results.append(trial_rank)
            completed += 1

            if completed % 5000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_trials - completed) / rate if rate > 0 else 0
                print(f"        {completed}/{n_trials} "
                      f"({100*completed/n_trials:.0f}%) "
                      f"-- {rate:.0f}/s -- ETA {eta:.0f}s")

    elapsed = time.time() - t0
    mrr_dict = compute_mean_reciprocal_rank(rank_results)
    mrr = float(mrr_dict.get(method_name, 0.0))
    print(f"        {completed} trials in {elapsed:.1f}s "
          f"({completed/max(elapsed,0.1):.0f}/s) -> MRR={mrr:.4f}")

    return mrr, mrr_dict


# ============================================================
# Main
# ============================================================

def run(species_list=None):
    """Run cross-species atlas."""
    t_start = time.time()
    print(BANNER)
    print("  Cross-Species Function Prediction Atlas")
    print(BANNER)

    np.random.seed(SEED)

    if species_list is None:
        species_list = ["human", "mouse"]

    # Load GO aspect map
    print(f"\n[1/4] Loading GO aspect map ...")
    go_aspect_map = load_go_aspect_map()
    print(f"  {len(go_aspect_map)} GO terms with aspect info")

    all_species_results = {}

    for sp_idx, sp_key in enumerate(species_list):
        sp_info = SPECIES[sp_key]
        print(f"\n{'='*64}")
        print(f"  Species: {sp_info['name']} ({sp_idx+1}/{len(species_list)})")
        print(f"{'='*64}")

        # ---- Load network ----
        print(f"\n  [1] Loading network ...")
        loader = globals()[sp_info["network_loader"]]
        G = loader()
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # ---- Load annotations ----
        print(f"\n  [2] Loading annotations by aspect ...")
        aspect_annotations = load_annotations_by_aspect(
            sp_key, go_aspect_map, set(G.nodes()))

        for short, ann in aspect_annotations.items():
            n_terms = sum(len(v) for v in ann.values())
            print(f"    {short}: {len(ann)} proteins, {n_terms} annotations")

        # ---- Compute Spectral embeddings ----
        print(f"\n  [3] Computing Spectral embeddings at d = {SPECTRAL_DIMENSIONS} ...")
        spectral_embeddings = {}
        for dim in SPECTRAL_DIMENSIONS:
            n_nodes = G.number_of_nodes()
            if dim + 1 >= n_nodes - 1:
                print(f"    SKIP d={dim}: too few nodes")
                continue
            t_emb = time.time()
            coords, nodes, eigenvalues = compute_spectral_embedding(G, dim)
            emb_time = time.time() - t_emb
            print(f"    d={dim}: {coords.shape} in {emb_time:.1f}s")

            # Save
            emb_path = EMB / f"{sp_key}_spectral_d{dim}.npy"
            nodes_path = EMB / f"{sp_key}_spectral_d{dim}_nodes.json"
            np.save(str(emb_path), coords)
            with open(nodes_path, "w", encoding="utf-8") as f:
                json.dump(nodes, f)

            spectral_embeddings[dim] = (coords, nodes)

        # ---- Run LOTO-CV per ontology ----
        print(f"\n  [4] Running LOTO-CV per ontology ...")
        species_results = {"name": sp_info["name"], "network_size": G.number_of_nodes(),
                          "ontologies": {}}

        for aspect_short, annotations in aspect_annotations.items():
            if len(annotations) < 10:
                print(f"\n    SKIP {aspect_short}: too few annotations")
                continue

            n_proteins = len(annotations)
            n_trials = sum(len(v) for v in annotations.values())
            print(f"\n    === {aspect_short} ({n_proteins} proteins, {n_trials} trials) ===")

            # Build term frequency
            term_freq = Counter()
            for terms in annotations.values():
                term_freq.update(terms)

            ontology_results = {}

            # Spectral at multiple dimensions
            for dim in SPECTRAL_DIMENSIONS:
                if dim not in spectral_embeddings:
                    continue
                coords, nodes = spectral_embeddings[dim]
                method_name = f"Spectral-d{dim}"
                print(f"\n      [{method_name}]")
                compute_ppi = (dim == min(SPECTRAL_DIMENSIONS))
                mrr, mrr_dict = run_loto(coords, nodes, G, annotations, term_freq,
                                        method_name, compute_ppi=compute_ppi)

                result = {"MRR": float(mrr), "n_trials": int(n_trials)}
                if compute_ppi and mrr_dict:
                    if "PPI-Neighbors" in mrr_dict:
                        result["PPI_MRR"] = float(mrr_dict["PPI-Neighbors"])
                    if "2-Hop Diffusion" in mrr_dict:
                        result["2Hop_MRR"] = float(mrr_dict["2-Hop Diffusion"])
                    if "Random" in mrr_dict:
                        result["Random_MRR"] = float(mrr_dict["Random"])

                ontology_results[method_name] = result

            species_results["ontologies"][aspect_short] = ontology_results

        all_species_results[sp_key] = species_results

        # Save intermediate results
        interim_file = RESULTS / f"cross_species_atlas_{sp_key}.json"
        with open(interim_file, "w", encoding="utf-8") as f:
            json.dump(species_results, f, indent=2)
        print(f"\n  Saved interim: {interim_file}")

    # ---- Save combined results ----
    print(f"\n[3/4] Saving combined results ...")
    output = {
        "description": "Cross-Species Function Prediction Atlas",
        "spectral_dimensions": SPECTRAL_DIMENSIONS,
        "species": all_species_results,
    }

    out_file = RESULTS / "cross_species_atlas.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {out_file}")

    # ---- Print summary ----
    print(f"\n[4/4] Summary")
    print(f"{'='*64}")
    for sp_key, sp_data in all_species_results.items():
        print(f"\n  {sp_data['name']} (n={sp_data['network_size']})")
        for aspect, methods in sp_data["ontologies"].items():
            ppi = None
            for m, r in methods.items():
                if "PPI_MRR" in r:
                    ppi = r["PPI_MRR"]
                    break

            spectral_best = max(
                [(m, r["MRR"]) for m, r in methods.items() if m.startswith("Spectral")],
                key=lambda x: x[1], default=(None, 0))

            if ppi and spectral_best[1] > ppi:
                status = f"EXCEEDS by {100*(spectral_best[1]/ppi - 1):.1f}%"
            elif ppi:
                status = f"below by {100*(1 - spectral_best[1]/ppi):.1f}%"
            else:
                status = "no PPI baseline"

            ppi_val = ppi if ppi else 0.0
            print(f"    {aspect}: PPI={ppi_val:.4f}, "
                  f"best Spectral={spectral_best[0]} MRR={spectral_best[1]:.4f} ({status})")

    # ---- Figure ----
    plot_cross_species(all_species_results)

    elapsed = time.time() - t_start
    print(f"\nCross-species atlas completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return output


# ============================================================
# Figure
# ============================================================

def plot_cross_species(all_results):
    """Cross-species comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)

    aspect_names = {"BP": "Biological Process", "MF": "Molecular Function",
                    "CC": "Cellular Component"}
    species_colors = {"human": "#d62728", "mouse": "#3182bd", "yeast": "#2ca02c"}

    for ax_idx, aspect in enumerate(["BP", "MF", "CC"]):
        ax = axes[ax_idx]

        for sp_key, sp_data in all_results.items():
            if aspect not in sp_data["ontologies"]:
                continue
            methods = sp_data["ontologies"][aspect]

            # Get spectral points
            dims, mrrs = [], []
            for m, r in sorted(methods.items()):
                if m.startswith("Spectral-d"):
                    dim = int(m.split("d")[1])
                    dims.append(dim)
                    mrrs.append(r["MRR"])

            if dims:
                ax.plot(dims, mrrs, "o-",
                       color=species_colors.get(sp_key, "#999"),
                       linewidth=2.5, markersize=10,
                       label=f"{sp_data['name']}")

                for d, mrr in zip(dims, mrrs):
                    ax.annotate(f"{mrr:.3f}", (d, mrr),
                              textcoords="offset points",
                              xytext=(0, 10), ha="center", fontsize=8,
                              fontweight="bold")

            # PPI baseline
            for m, r in methods.items():
                if "PPI_MRR" in r:
                    ax.axhline(r["PPI_MRR"],
                             color=species_colors.get(sp_key, "#999"),
                             linestyle="--", linewidth=1.5, alpha=0.5)
                    break

        ax.set_xlabel("Embedding Dimension (d)", fontsize=12)
        ax.set_title(f"{aspect_names.get(aspect, aspect)}", fontsize=13,
                    fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 64, 256])
        ax.set_xticklabels(["2", "64", "256"])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=12)
    axes[0].legend(fontsize=10, loc="lower right")

    plt.suptitle("Cross-Species: Spectral Embedding vs PPI Topology",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = FIGURES / "Fig80_cross_species_comparison.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", nargs="+", default=["human", "mouse"],
                       choices=["human", "mouse", "yeast"])
    args = parser.parse_args()
    run(species_list=args.species)
