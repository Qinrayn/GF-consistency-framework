#!/usr/bin/env python3
"""
Cross-Species Function Prediction Atlas — Optimized (Step 70b / Phase 21)
==========================================================================

Same as cross_species_atlas.py but with optimized LOTO-CV:
  - Precompute kNN neighbors once per protein (not per trial)
  - Reuse neighbors across all hidden terms for the same protein
  - ~6x speedup on average (proteins have ~6 terms each)

Usage: python scripts/cross_species_atlas_optimized.py --species human mouse
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
    compute_mean_reciprocal_rank,
    K_MAX,
    EXPERIMENTAL_CODES,
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

SPECIES = {
    "human": {
        "name": "Human",
        "annotation_file": DATA / "human_go_annotations.json",
        "color": "#d62728",
    },
    "mouse": {
        "name": "Mouse",
        "annotation_file": DATA / "mouse_go_annotations.json",
        "color": "#3182bd",
    },
    "yeast": {
        "name": "Yeast",
        "annotation_file": None,
        "color": "#2ca02c",
    },
}


# ============================================================
# Network Loaders
# ============================================================

def load_yeast_network():
    net_file = DATA / "yeast_ppi_5936.edgelist"
    G = nx.read_edgelist(str(net_file))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def load_human_network():
    links_file = Path(__file__).resolve().parent.parent / "human_validation" / "9606.protein.links.v12.0.txt.gz"
    if not links_file.exists():
        raise FileNotFoundError(f"Human STRING not found: {links_file}")
    G = nx.Graph()
    with gzip.open(str(links_file), "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("protein1"):
                continue
            parts = line.strip().split()
            if len(parts) >= 3 and int(parts[2]) >= 700:
                G.add_edge(parts[0], parts[1])
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def load_mouse_network():
    net_file = DATA / "mouse_ppi.edgelist"
    if net_file.exists():
        G = nx.read_edgelist(str(net_file))
    else:
        links_file = DATA / "10090.protein.links.v11.5.txt.gz"
        G = nx.Graph()
        with gzip.open(str(links_file), "rt") as f:
            for line in f:
                if line.startswith("#") or line.startswith("protein1"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3 and int(parts[2]) >= 700:
                    G.add_edge(parts[0], parts[1])
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


# ============================================================
# GO Aspect Mapping
# ============================================================

def load_go_aspect_map():
    go_aspect = {}
    obo_file = DATA / "go.obo"
    if not obo_file.exists():
        return go_aspect
    current_id = None
    with open(obo_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("namespace:") and current_id:
                go_aspect[current_id] = line.split("namespace: ")[1]
                current_id = None
    return go_aspect


# ============================================================
# Annotation Loading
# ============================================================

def load_annotations_by_aspect(species_key, go_aspect_map, network_nodes):
    aspect_annotations = {short: defaultdict(set) for short in ASPECTS.values()}

    if species_key == "yeast":
        from function_prediction import build_alias_mapping
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

    return {short: {k: v for k, v in ann.items() if len(v) > 0}
            for short, ann in aspect_annotations.items()}


# ============================================================
# Spectral Embedding
# ============================================================

def compute_spectral_embedding(graph, dim):
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
# Optimized LOTO-CV (Precomputed Neighbors)
# ============================================================

def run_loto_optimized(coords, nodes, graph, annotations, term_freq,
                       method_name, compute_ppi=False):
    """Optimized LOTO-CV with precomputed kNN neighbors.

    Instead of calling knn_predict_fast for every trial (redundantly
    re-querying kNN for the same protein), precompute neighbors once
    per query protein and reuse across all hidden terms.

    This gives ~Nx speedup where N = average terms per protein.
    """
    node_set = set(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in node_set and pid in graph
    }

    n_trials = sum(len(terms) for terms in query_proteins.values())
    if n_trials == 0:
        return 0.0, {}

    # Precompute kNN for ALL query proteins at once
    k = K_MAX
    n_neighbors = min(k + 1, len(coords))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(coords)

    print(f"        Precomputing kNN for {len(query_proteins)} proteins...")
    t_knn = time.time()

    query_indices = []
    query_pids = []
    for pid in sorted(query_proteins.keys()):
        if pid in node_to_idx:
            query_indices.append(node_to_idx[pid])
            query_pids.append(pid)

    # Batch kNN query
    query_coords = coords[query_indices]
    distances, indices = nn_model.kneighbors(query_coords, n_neighbors=n_neighbors)

    # Cache: pid -> [(neighbor_id, weight), ...]
    neighbor_cache = {}
    for i, pid in enumerate(query_pids):
        query_idx = query_indices[i]
        neighbors = []
        for dist, idx in zip(distances[i], indices[i]):
            if idx == query_idx:
                continue
            neighbor_id = nodes[idx]
            weight = 1.0 / (dist + 1e-10)
            neighbors.append((neighbor_id, weight))
        neighbor_cache[pid] = neighbors

    knn_time = time.time() - t_knn
    print(f"        kNN precomputed in {knn_time:.1f}s")

    # Now run LOTO trials using cached neighbors
    random_ranking = term_freq.most_common()
    rank_results = []
    completed = 0
    t0 = time.time()

    for pid in sorted(query_proteins.keys()):
        terms = query_proteins[pid]
        if pid not in neighbor_cache:
            continue

        neighbors = neighbor_cache[pid]

        for hidden_term in sorted(terms):
            trial_rank = {}

            # Embedding KNN prediction (using cached neighbors)
            # Note: neighbors are OTHER proteins, so all their terms
            # (including the hidden term) contribute to prediction scores.
            # The hidden term is only excluded from the query protein itself.
            term_scores = Counter()
            for neighbor_id, weight in neighbors:
                neighbor_terms = annotations.get(neighbor_id, set())
                for term in neighbor_terms:
                    term_scores[term] += weight

            pred_terms = [t for t, _ in term_scores.most_common()]
            try:
                trial_rank[method_name] = pred_terms.index(hidden_term) + 1
            except ValueError:
                trial_rank[method_name] = 0

            # PPI baselines (not cached — these depend on network topology)
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
    t_start = time.time()
    print(BANNER)
    print("  Cross-Species Atlas (Optimized LOTO-CV)")
    print(BANNER)

    np.random.seed(SEED)

    if species_list is None:
        species_list = ["human", "mouse"]

    go_aspect_map = load_go_aspect_map()
    print(f"  GO aspect map: {len(go_aspect_map)} terms")

    all_species_results = {}

    for sp_idx, sp_key in enumerate(species_list):
        sp_info = SPECIES[sp_key]
        print(f"\n{'='*64}")
        print(f"  {sp_info['name']} ({sp_idx+1}/{len(species_list)})")
        print(f"{'='*64}")

        # Load network
        print(f"\n  [1] Loading network ...")
        loader = globals()[f"load_{sp_key}_network"]
        G = loader()
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Load annotations
        print(f"\n  [2] Loading annotations ...")
        aspect_annotations = load_annotations_by_aspect(
            sp_key, go_aspect_map, set(G.nodes()))
        for short, ann in aspect_annotations.items():
            n_trials = sum(len(v) for v in ann.values())
            print(f"    {short}: {len(ann)} proteins, {n_trials} trials")

        # Compute embeddings
        print(f"\n  [3] Computing Spectral embeddings ...")
        spectral_embeddings = {}
        for dim in SPECTRAL_DIMENSIONS:
            if dim + 1 >= G.number_of_nodes() - 1:
                continue
            t_emb = time.time()
            coords, nodes, eigenvalues = compute_spectral_embedding(G, dim)
            print(f"    d={dim}: {coords.shape} in {time.time()-t_emb:.1f}s")

            emb_path = EMB / f"{sp_key}_spectral_d{dim}.npy"
            nodes_path = EMB / f"{sp_key}_spectral_d{dim}_nodes.json"
            np.save(str(emb_path), coords)
            with open(nodes_path, "w", encoding="utf-8") as f:
                json.dump(nodes, f)

            spectral_embeddings[dim] = (coords, nodes)

        # LOTO-CV
        print(f"\n  [4] LOTO-CV per ontology ...")
        species_results = {"name": sp_info["name"],
                          "network_size": G.number_of_nodes(),
                          "ontologies": {}}

        for aspect_short, annotations in aspect_annotations.items():
            if len(annotations) < 10:
                continue

            n_trials = sum(len(v) for v in annotations.values())
            print(f"\n    === {aspect_short} ({len(annotations)} proteins, "
                  f"{n_trials} trials) ===")

            term_freq = Counter()
            for terms in annotations.values():
                term_freq.update(terms)

            ontology_results = {}

            for dim in SPECTRAL_DIMENSIONS:
                if dim not in spectral_embeddings:
                    continue
                coords, nodes = spectral_embeddings[dim]
                method_name = f"Spectral-d{dim}"
                print(f"\n      [{method_name}]")
                compute_ppi = (dim == min(SPECTRAL_DIMENSIONS))

                mrr, mrr_dict = run_loto_optimized(
                    coords, nodes, G, annotations, term_freq,
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

        # Save interim
        interim_file = RESULTS / f"cross_species_atlas_{sp_key}.json"
        with open(interim_file, "w", encoding="utf-8") as f:
            json.dump(species_results, f, indent=2)
        print(f"\n  Saved interim: {interim_file}")

    # Save combined
    output = {
        "description": "Cross-Species Function Prediction Atlas (Optimized)",
        "spectral_dimensions": SPECTRAL_DIMENSIONS,
        "species": all_species_results,
    }

    out_file = RESULTS / "cross_species_atlas.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {out_file}")

    # Summary
    print(f"\n{'='*64}")
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
                  f"best={spectral_best[0]} MRR={spectral_best[1]:.4f} ({status})")

    # Figure
    plot_cross_species(all_species_results)

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return output


# ============================================================
# Figure
# ============================================================

def plot_cross_species(all_results):
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
                              textcoords="offset points", xytext=(0, 10),
                              ha="center", fontsize=8, fontweight="bold")
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", nargs="+", default=["human", "mouse"],
                       choices=["human", "mouse", "yeast"])
    args = parser.parse_args()
    run(species_list=args.species)
