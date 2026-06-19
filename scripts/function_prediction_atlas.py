#!/usr/bin/env python3
"""
Function Prediction Atlas: Multi-Ontology, Multi-Dimension (Step 67 / Phase 20)
================================================================================

Scale up function prediction from BP-only at d=2 to a multi-ontology atlas:
  - 3 GO ontologies: BP (P), MF (F), CC (C)
  - Multiple dimensions: d = 2, 64, 256
  - 11 embedding methods at d=2
  - PPI-Neighbors and 2-Hop Diffusion baselines

Key questions:
  1. Does d=256 exceed PPI topology for MF and CC too?
  2. Which ontology shows the strongest dimension scaling?
  3. Is Spectral consistently the best embedding method across ontologies?

Output
------
- results/function_prediction_atlas.json
- figures/Fig77_atlas_ontology_comparison.png
- figures/Fig78_atlas_dimension_curves.png
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
    TARGET_STD, load_embedding, rescale_coordinates,
)
from function_prediction import (
    build_alias_mapping,
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

NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"
GAF_FILE = DATA / "gene_association.sgd.gaf.gz"

EXPERIMENTAL_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}

ASPECTS = {
    "P": "Biological Process",
    "F": "Molecular Function",
    "C": "Cellular Component",
}

# Dimensions for Spectral sweep
SPECTRAL_DIMENSIONS = [2, 64, 256]

# Methods for d=2 comparison (all have full-network 2D embeddings)
ALL_METHODS_D2 = [
    "DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE",
    "PCA", "VGAE-feat", "GIN", "GAT", "GraphSAGE",
]

BANNER = "=" * 64


# ============================================================
# Multi-Aspect GAF Parser
# ============================================================

def parse_gaf_by_aspect(sgd_to_string, orf_to_string, network_nodes, aspect="P"):
    """Parse SGD GAF, keeping only experimental annotations for one aspect.

    Parameters
    ----------
    aspect : str
        'P' for BP, 'F' for MF, 'C' for CC.
    """
    annotations = defaultdict(set)
    total_lines = 0
    filtered = 0
    unmapped = 0

    with gzip.open(str(GAF_FILE), "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("!") or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            total_lines += 1

            evidence = cols[6]
            asp = cols[8]
            if evidence not in EXPERIMENTAL_CODES or asp != aspect:
                continue

            go_term = cols[4]
            if not go_term.startswith("GO:"):
                continue

            filtered += 1

            # Map gene ID to STRING node ID
            sgd_id = cols[1]
            gene_sym = cols[2]
            orf_name = cols[10] if len(cols) > 10 else ""

            string_id = None

            if sgd_id in sgd_to_string:
                candidate = sgd_to_string[sgd_id]
                if candidate in network_nodes:
                    string_id = candidate

            if string_id is None and orf_name and orf_name in orf_to_string:
                candidate = orf_to_string[orf_name]
                if candidate in network_nodes:
                    string_id = candidate

            if string_id is None and gene_sym in orf_to_string:
                candidate = orf_to_string[gene_sym]
                if candidate in network_nodes:
                    string_id = candidate

            if string_id is None:
                unmapped += 1
                continue

            annotations[string_id].add(go_term)

    stats = {
        "aspect": aspect,
        "total_gaf_lines": total_lines,
        "experimental_annotations": filtered,
        "mapped_proteins": len(annotations),
        "unmapped": unmapped,
        "unique_go_terms": len({t for s in annotations.values() for t in s}),
    }

    return dict(annotations), stats


# ============================================================
# Spectral Embedding at Arbitrary Dimension
# ============================================================

def compute_spectral_embedding(graph, dim):
    """Compute Spectral embedding at given dimension via normalized Laplacian."""
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

    return coords, nodes


# ============================================================
# LOTO Prediction Engine
# ============================================================

def run_loto(coords, nodes, graph, annotations, term_freq, method_name,
             compute_ppi=False):
    """Run LOTO-CV for a single method/dimension."""
    node_set = set(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in node_set and pid in graph
    }

    n_trials = sum(len(terms) for terms in query_proteins.values())

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

    return mrr, mrr_dict, rank_results


# ============================================================
# Main
# ============================================================

def run():
    """Run the multi-ontology function prediction atlas."""
    t_start = time.time()
    print(BANNER)
    print("  Function Prediction Atlas: 3 Ontologies x Multi-Dimension")
    print(BANNER)

    np.random.seed(SEED)

    # ---- Load network ----
    print(f"\n[1/5] Loading network ...")
    G = nx.read_edgelist(str(NETWORK_FILE))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---- Build alias mapping ----
    print(f"\n[2/5] Building alias mapping ...")
    sgd_map, orf_map, net_nodes = build_alias_mapping()

    # ---- Parse all three aspects ----
    print(f"\n[3/5] Parsing GO annotations by aspect ...")
    aspect_annotations = {}
    aspect_stats = {}
    for code, name in ASPECTS.items():
        ann, stats = parse_gaf_by_aspect(sgd_map, orf_map, net_nodes, code)
        aspect_annotations[code] = ann
        aspect_stats[code] = stats
        print(f"  {name} ({code}): {stats['mapped_proteins']} proteins, "
              f"{stats['unique_go_terms']} GO terms, "
              f"{stats['experimental_annotations']} annotations")

    # ---- Compute Spectral embeddings at target dimensions ----
    print(f"\n[4/5] Computing Spectral embeddings at d = {SPECTRAL_DIMENSIONS} ...")
    spectral_embeddings = {}
    for dim in SPECTRAL_DIMENSIONS:
        if dim == 2:
            # Load existing 2D embedding
            try:
                raw_coords, emb_nodes = load_embedding("Spectral", "full")
                coords = rescale_coordinates(raw_coords, target_std=TARGET_STD)
                spectral_embeddings[dim] = (coords, emb_nodes)
                print(f"  d={dim}: loaded existing ({coords.shape})")
                continue
            except Exception:
                pass

        coords, nodes = compute_spectral_embedding(G, dim)
        spectral_embeddings[dim] = (coords, nodes)
        # Save
        np.save(str(EMB / f"Spectral_d{dim}_full.npy"), coords)
        with open(EMB / f"Spectral_d{dim}_full_nodes.json", "w") as f:
            json.dump(nodes, f)
        print(f"  d={dim}: computed ({coords.shape})")

    # ---- Run predictions ----
    print(f"\n[5/5] Running LOTO-CV predictions ...")

    all_results = {}

    for code, name in ASPECTS.items():
        print(f"\n  === {name} ({code}) ===")
        annotations = aspect_annotations[code]

        if not annotations:
            print(f"  SKIP: no annotations for {name}")
            continue

        # Build term frequency
        term_freq = Counter()
        for terms in annotations.values():
            term_freq.update(terms)

        n_proteins = len(annotations)
        n_trials = sum(len(t) for t in annotations.values())
        print(f"  {n_proteins} proteins, {n_trials} LOTO trials")

        if n_proteins < 10 or n_trials < 20:
            print(f"  SKIP: too few annotations")
            continue

        ontology_results = {}

        # --- Spectral at multiple dimensions ---
        for dim in SPECTRAL_DIMENSIONS:
            coords, nodes = spectral_embeddings[dim]
            method_name = f"Spectral-d{dim}"
            print(f"\n    [{method_name}]")
            compute_ppi = (dim == min(SPECTRAL_DIMENSIONS))
            mrr, mrr_dict, ranks = run_loto(coords, nodes, G, annotations, term_freq,
                                  method_name, compute_ppi=compute_ppi)

            result = {"MRR": float(mrr), "n_trials": int(len(ranks))}

            # Extract baselines from mrr_dict (computed over ALL trials)
            if compute_ppi:
                if "PPI-Neighbors" in mrr_dict:
                    result["PPI_MRR"] = float(mrr_dict["PPI-Neighbors"])
                if "2-Hop Diffusion" in mrr_dict:
                    result["2Hop_MRR"] = float(mrr_dict["2-Hop Diffusion"])
                if "Random" in mrr_dict:
                    result["Random_MRR"] = float(mrr_dict["Random"])

            ontology_results[method_name] = result

        # --- Other methods at d=2 ---
        coords_2d, nodes_2d = spectral_embeddings[2]
        for method in ALL_METHODS_D2:
            if method == "Spectral":
                continue  # Already computed above
            try:
                emb, emb_nodes = load_embedding(method, "full")
                emb = rescale_coordinates(emb, target_std=TARGET_STD)
                # Reorder to match spectral nodes
                node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
                reorder = [node_to_idx[n] for n in nodes_2d]
                emb = emb[reorder]

                method_name = f"{method}-d2"
                print(f"\n    [{method_name}]")
                mrr, _, ranks = run_loto(emb, nodes_2d, G, annotations, term_freq,
                                      method_name)
                ontology_results[method_name] = {"MRR": float(mrr)}
            except Exception as e:
                print(f"\n    [{method}-d2] FAILED: {e}")

        all_results[code] = {
            "name": name,
            "stats": aspect_stats[code],
            "methods": ontology_results,
        }

    # ---- Summary ----
    print(f"\n{'='*64}")
    print("ATLAS SUMMARY")
    print(f"{'='*64}")

    for code, data in all_results.items():
        print(f"\n  {data['name']} ({code}):")
        methods = data["methods"]
        # Sort by MRR
        ranked = sorted(methods.items(), key=lambda x: -x[1]["MRR"])
        for method, result in ranked[:5]:
            print(f"    {method}: MRR={result['MRR']:.4f}")

    # Check PPI crossing per ontology
    print(f"\n  PPI-topology crossing by ontology:")
    for code, data in all_results.items():
        methods = data["methods"]
        ppi_mrr = None
        for m, r in methods.items():
            if "PPI_MRR" in r:
                ppi_mrr = r["PPI_MRR"]
                break
        if ppi_mrr is None:
            print(f"    {code}: PPI baseline not computed")
            continue

        spectral_mrrs = [(m, r["MRR"]) for m, r in methods.items()
                        if m.startswith("Spectral")]
        crossed = [(m, mrr) for m, mrr in spectral_mrrs if mrr > ppi_mrr]
        if crossed:
            first = min(crossed, key=lambda x: int(x[0].split("d")[1]))
            print(f"    {code}: PPI={ppi_mrr:.4f}, "
                  f"crossed at {first[0]} (MRR={first[1]:.4f})")
        else:
            best = max(spectral_mrrs, key=lambda x: x[1])
            print(f"    {code}: PPI={ppi_mrr:.4f}, "
                  f"best Spectral={best[0]} MRR={best[1]:.4f} "
                  f"(gap={ppi_mrr - best[1]:.4f})")

    # ---- Save ----
    output = {
        "description": "Function Prediction Atlas: 3 Ontologies x Multi-Dimension",
        "ontologies": all_results,
        "spectral_dimensions": SPECTRAL_DIMENSIONS,
        "methods_d2": ALL_METHODS_D2,
    }

    out_file = RESULTS / "function_prediction_atlas.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_file}")

    # ---- Figures ----
    plot_ontology_comparison(all_results)
    plot_dimension_curves(all_results)

    elapsed = time.time() - t_start
    print(f"\nAtlas completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return output


# ============================================================
# Figures
# ============================================================

def plot_ontology_comparison(all_results):
    """Bar chart comparing methods across ontologies at d=2."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(18, 7), sharey=True)
    if len(all_results) == 1:
        axes = [axes]

    colors = {
        "Spectral": "#d62728", "MDS": "#3182bd", "DM": "#08306b",
        "PCA": "#2ca02c", "DeepWalk": "#9467bd", "Node2Vec": "#fb6a4a",
        "VGAE": "#67000d", "VGAE-feat": "#8c564b", "GIN": "#e377c2",
        "GAT": "#7f7f7f", "GraphSAGE": "#bcbd22",
    }

    for ax, (code, data) in zip(axes, all_results.items()):
        methods = data["methods"]
        d2_methods = [(m, r["MRR"]) for m, r in methods.items()
                     if m.endswith("-d2")]
        d2_methods.sort(key=lambda x: -x[1])

        names = [m.replace("-d2", "") for m, _ in d2_methods]
        mrrs = [v for _, v in d2_methods]
        bar_colors = [colors.get(n, "#999999") for n in names]

        bars = ax.barh(range(len(names)), mrrs, color=bar_colors, edgecolor="white")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("MRR", fontsize=11)
        ax.set_title(f"{data['name']} ({code})", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # Add PPI baseline
        for m, r in methods.items():
            if "PPI_MRR" in r:
                ax.axvline(r["PPI_MRR"], color="grey", linestyle="--",
                          linewidth=1.5, alpha=0.7)
                ax.text(r["PPI_MRR"], len(names) - 0.5, "PPI",
                       fontsize=8, color="grey", ha="center")

    plt.suptitle("Function Prediction: Method Comparison by Ontology (d=2)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = FIGURES / "Fig77_atlas_ontology_comparison.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_dimension_curves(all_results):
    """Line chart showing MRR vs dimension for Spectral across ontologies."""
    fig, ax = plt.subplots(figsize=(10, 7))

    markers = {"P": "o", "F": "s", "C": "^"}
    colors = {"P": "#d62728", "F": "#3182bd", "C": "#2ca02c"}

    for code, data in all_results.items():
        methods = data["methods"]
        spectral_points = []
        for m, r in methods.items():
            if m.startswith("Spectral-d"):
                dim = int(m.split("d")[1])
                spectral_points.append((dim, r["MRR"]))

        if spectral_points:
            spectral_points.sort()
            dims, mrrs = zip(*spectral_points)
            ax.plot(dims, mrrs, f"{markers.get(code, 'o')}-",
                   color=colors.get(code, "#999"), linewidth=2.5,
                   markersize=10, label=f"{data['name']} ({code})")

            # Annotate
            for d, mrr in zip(dims, mrrs):
                ax.annotate(f"{mrr:.3f}", (d, mrr), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=9,
                           fontweight="bold")

        # PPI baseline
        for m, r in methods.items():
            if "PPI_MRR" in r:
                ax.axhline(r["PPI_MRR"], color=colors.get(code, "#999"),
                          linestyle="--", linewidth=1.5, alpha=0.5)
                break

    ax.set_xlabel("Embedding Dimension (d)", fontsize=13)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=13)
    ax.set_title("Spectral Embedding: Function Prediction by Dimension and Ontology",
                 fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256"])
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES / "Fig78_atlas_dimension_curves.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run()
