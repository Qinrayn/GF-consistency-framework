#!/usr/bin/env python3
"""
Human Function Prediction Atlas (Step 70c / Phase 21)
======================================================
Focused script for human LOTO-CV using pre-computed embeddings.
Avoids recomputing spectral embeddings — loads them directly.

Usage: python scripts/human_atlas.py
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
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
)
from function_prediction import (
    ppi_neighbor_predict, twohop_diffusion_predict,
    compute_mean_reciprocal_rank, K_MAX,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
EMB = get_embeddings_dir()

SPECTRAL_DIMS = [2, 64, 256]

ASPECT_MAP = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}


def load_human_network():
    """Load human STRING v12.0 network (score >= 700)."""
    links_file = Path(__file__).resolve().parent.parent / "human_validation" / "9606.protein.links.v12.0.txt.gz"
    print(f"  Loading human STRING network from {links_file.name} ...")
    G = nx.Graph()
    with gzip.open(str(links_file), "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or line.startswith("protein1"):
                continue
            parts = line.strip().split()
            if len(parts) >= 3 and int(parts[2]) >= 700:
                G.add_edge(parts[0], parts[1])
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_annotations(network_nodes):
    """Load human GO annotations, split by aspect using go.obo."""
    # Load GO aspect map
    go_aspect = {}
    obo_file = DATA / "go.obo"
    current_id = None
    with open(obo_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("namespace:") and current_id:
                go_aspect[current_id] = line.split("namespace: ")[1]
                current_id = None

    # Load annotations
    ann_file = DATA / "human_go_annotations.json"
    with open(ann_file, encoding="utf-8") as f:
        raw = json.load(f)

    aspect_ann = {"BP": defaultdict(set), "MF": defaultdict(set),
                  "CC": defaultdict(set)}
    for pid, terms in raw.items():
        if pid not in network_nodes:
            continue
        for term in terms:
            aspect_full = go_aspect.get(term)
            if aspect_full and aspect_full in ASPECT_MAP:
                short = ASPECT_MAP[aspect_full]
                aspect_ann[short][pid].add(term)

    result = {}
    for aspect, ann in aspect_ann.items():
        filtered = {k: v for k, v in ann.items() if len(v) > 0}
        if filtered:
            result[aspect] = filtered
    return result


def run_loto_optimized(coords, nodes, graph, annotations, term_freq,
                       method_name, compute_ppi=False):
    """Optimized LOTO-CV with precomputed kNN neighbors."""
    node_set = set(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in node_set and pid in graph
    }

    n_trials = sum(len(terms) for terms in query_proteins.values())
    if n_trials == 0:
        return 0.0, {}, []

    # Precompute kNN for ALL query proteins at once
    k = K_MAX
    n_neighbors = min(k + 1, len(coords))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(coords)

    print(f"      Precomputing kNN for {len(query_proteins)} proteins ...")
    t_knn = time.time()

    query_indices = []
    query_pids = []
    for pid in sorted(query_proteins.keys()):
        if pid in node_to_idx:
            query_indices.append(node_to_idx[pid])
            query_pids.append(pid)

    query_coords = coords[query_indices]
    distances, indices = nn_model.kneighbors(query_coords, n_neighbors=n_neighbors)

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

    print(f"      kNN done in {time.time()-t_knn:.1f}s")

    # Run LOTO trials
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

            # Embedding KNN prediction (neighbors are OTHER proteins)
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

            # PPI baselines
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

            if completed % 10000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_trials - completed) / rate if rate > 0 else 0
                print(f"      {completed}/{n_trials} "
                      f"({100*completed/n_trials:.0f}%) "
                      f"-- {rate:.0f}/s -- ETA {eta:.0f}s")

    elapsed = time.time() - t0
    mrr_dict = compute_mean_reciprocal_rank(rank_results)
    mrr = float(mrr_dict.get(method_name, 0.0))
    print(f"      {completed} trials in {elapsed:.1f}s "
          f"({completed/max(elapsed,0.1):.0f}/s) -> MRR={mrr:.4f}")

    return mrr, mrr_dict, rank_results


def main():
    t_start = time.time()
    print("=" * 64)
    print("  Human Function Prediction Atlas")
    print("=" * 64)
    np.random.seed(SEED)

    # Load network (for PPI baselines)
    G = load_human_network()
    nodes_set = set(G.nodes())

    # Load annotations
    print("\n  Loading annotations ...")
    aspect_annotations = load_annotations(nodes_set)
    for aspect, ann in aspect_annotations.items():
        n_trials = sum(len(v) for v in ann.values())
        print(f"    {aspect}: {len(ann)} proteins, {n_trials} trials")

    # Load pre-computed embeddings
    print("\n  Loading pre-computed Spectral embeddings ...")
    embeddings = {}
    for dim in SPECTRAL_DIMS:
        npy = EMB / f"human_spectral_d{dim}.npy"
        nodes_json = EMB / f"human_spectral_d{dim}_nodes.json"
        if npy.exists() and nodes_json.exists():
            coords = np.load(str(npy))
            with open(nodes_json) as f:
                nodes = json.load(f)
            embeddings[dim] = (coords, nodes)
            print(f"    d={dim}: {coords.shape}")
        else:
            print(f"    d={dim}: NOT FOUND")

    # Run LOTO-CV
    print("\n  Running LOTO-CV per ontology ...")
    species_results = {
        "name": "Human",
        "network_size": G.number_of_nodes(),
        "ontologies": {}
    }

    for aspect, annotations in aspect_annotations.items():
        n_trials = sum(len(v) for v in annotations.values())
        print(f"\n  === {aspect} ({len(annotations)} proteins, {n_trials} trials) ===")

        term_freq = Counter()
        for terms in annotations.values():
            term_freq.update(terms)

        ontology_results = {}

        for dim in SPECTRAL_DIMS:
            if dim not in embeddings:
                continue
            coords, nodes = embeddings[dim]
            method_name = f"Spectral-d{dim}"
            print(f"\n    [{method_name}]")
            compute_ppi = (dim == min(SPECTRAL_DIMS))

            mrr, mrr_dict, ranks = run_loto_optimized(
                coords, nodes, G, annotations, term_freq,
                method_name, compute_ppi=compute_ppi)

            result = {"MRR": float(mrr), "n_trials": int(len(ranks))}
            if compute_ppi and mrr_dict:
                if "PPI-Neighbors" in mrr_dict:
                    result["PPI_MRR"] = float(mrr_dict["PPI-Neighbors"])
                if "2-Hop Diffusion" in mrr_dict:
                    result["2Hop_MRR"] = float(mrr_dict["2-Hop Diffusion"])
                if "Random" in mrr_dict:
                    result["Random_MRR"] = float(mrr_dict["Random"])

            ontology_results[method_name] = result

        species_results["ontologies"][aspect] = ontology_results

    # Save
    out_file = RESULTS / "cross_species_atlas_human.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(species_results, f, indent=2)
    print(f"\n  Saved {out_file}")

    # Summary
    print("\n" + "=" * 64)
    print(f"  Human (n={species_results['network_size']})")
    for aspect, methods in species_results["ontologies"].items():
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

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return species_results


if __name__ == "__main__":
    main()
