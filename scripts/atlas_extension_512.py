#!/usr/bin/env python3
"""
Atlas Extension: MF and CC at d=512, d=1024 (Step 67b / Phase 1.2)
==================================================================

Extend the multi-ontology atlas to d=512 and d=1024 for MF and CC.
BP already has these from dimension_sweep_512.json.

Uses pre-computed embeddings — no eigendecomposition needed.

Output: results/atlas_extension_512.json
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
from utils import SEED, get_data_dir, get_results_dir, get_embeddings_dir
from function_prediction import (
    build_alias_mapping,
    knn_predict_fast,
    compute_mean_reciprocal_rank,
    K_MAX,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
EMB = get_embeddings_dir()
RESULTS.mkdir(parents=True, exist_ok=True)

NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"
GAF_FILE = DATA / "gene_association.sgd.gaf.gz"
EXPERIMENTAL_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}

NEW_DIMENSIONS = [512, 1024]
ASPECTS = {"F": "MF", "C": "CC"}

BANNER = "=" * 64


def parse_gaf_aspect(aspect_code, sgd_to_string, orf_to_string, network_nodes):
    """Parse GAF for a single aspect."""
    annotations = defaultdict(set)
    with gzip.open(str(GAF_FILE), "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("!") or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            evidence = cols[6]
            asp = cols[8]
            if evidence not in EXPERIMENTAL_CODES or asp != aspect_code:
                continue
            qualifier = cols[3].strip().lower()
            if "not" in qualifier:
                continue
            go_term = cols[4]
            if not go_term.startswith("GO:"):
                continue
            sgd_id = cols[1]
            orf_name = cols[10] if len(cols) > 10 else ""
            gene_sym = cols[2]
            string_id = None
            if sgd_id in sgd_to_string:
                c = sgd_to_string[sgd_id]
                if c in network_nodes:
                    string_id = c
            if string_id is None and orf_name and orf_name in orf_to_string:
                c = orf_to_string[orf_name]
                if c in network_nodes:
                    string_id = c
            if string_id is None and gene_sym in orf_to_string:
                c = orf_to_string[gene_sym]
                if c in network_nodes:
                    string_id = c
            if string_id is None and orf_name and orf_name in network_nodes:
                string_id = orf_name
            if string_id is None:
                continue
            annotations[string_id].add(go_term)
    return dict(annotations)


def run_loto_quick(coords, nodes, graph, annotations, dim, aspect):
    """Quick LOTO-CV for a single dimension x aspect."""
    node_set = set(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in node_set and pid in graph
    }
    if not query_proteins:
        return 0.0, 0

    n_trials = sum(len(terms) for terms in query_proteins.values())
    n_neighbors = min(K_MAX + 1, len(coords))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(coords)

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
                    trial_rank[f"Spectral-d{dim}"] = pred_terms.index(hidden_term) + 1
                except ValueError:
                    trial_rank[f"Spectral-d{dim}"] = 0
            rank_results.append(trial_rank)
            completed += 1
            if completed % 3000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_trials - completed) / rate if rate > 0 else 0
                print(f"    [{aspect}] d={dim}: {completed}/{n_trials} "
                      f"({100*completed/n_trials:.0f}%) {rate:.0f}/s ETA {eta:.0f}s")

    elapsed = time.time() - t0
    mrr_dict = compute_mean_reciprocal_rank(rank_results)
    spectral_mrr = float(mrr_dict.get(f"Spectral-d{dim}", 0.0))
    print(f"    [{aspect}] d={dim}: {completed} trials in {elapsed:.1f}s "
          f"-> MRR={spectral_mrr:.4f}")
    return spectral_mrr, completed


def run():
    t_start = time.time()
    print(BANNER)
    print("  Atlas Extension: MF/CC at d=512, d=1024")
    print(BANNER)
    np.random.seed(SEED)

    # Load network
    print(f"\n[1/4] Loading network ...")
    G = nx.read_edgelist(str(NETWORK_FILE))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build alias mapping and parse annotations
    print(f"\n[2/4] Parsing annotations ...")
    sgd_map, orf_map, net_nodes = build_alias_mapping()

    annotations_by_aspect = {}
    for code, label in ASPECTS.items():
        ann = parse_gaf_aspect(code, sgd_map, orf_map, net_nodes)
        annotations_by_aspect[label] = ann
        n_prot = len(ann)
        n_terms = len(set(t for ts in ann.values() for t in ts))
        print(f"  {label}: {n_prot} proteins, {n_terms} terms")

    # Load embeddings
    print(f"\n[3/4] Loading embeddings ...")
    embeddings = {}
    for dim in NEW_DIMENSIONS:
        emb_path = EMB / f"Spectral_d{dim}_full.npy"
        nodes_path = EMB / f"Spectral_d{dim}_full_nodes.json"
        if emb_path.exists():
            coords = np.load(str(emb_path))
            with open(nodes_path, encoding="utf-8") as f:
                nodes = json.load(f)
            embeddings[dim] = (coords, nodes)
            print(f"  d={dim}: {coords.shape}")
        else:
            print(f"  d={dim}: MISSING, skipping")

    # Run LOTO
    print(f"\n[4/4] Running LOTO-CV ...")
    results = {}
    for aspect in ["MF", "CC"]:
        ann = annotations_by_aspect[aspect]
        results[aspect] = {}
        for dim in sorted(embeddings.keys()):
            coords, nodes = embeddings[dim]
            mrr, n = run_loto_quick(coords, nodes, G, ann, dim, aspect)
            results[aspect][str(dim)] = {"Spectral_MRR": mrr, "n_trials": n}

    # Load BP results from dimension sweep for comparison
    bp_results = {}
    bp_file = RESULTS / "dimension_sweep_512.json"
    if bp_file.exists():
        bp_data = json.load(open(bp_file, encoding="utf-8"))
        for dim in NEW_DIMENSIONS:
            mrr = bp_data["mrr_by_dimension"].get(str(dim), 0.0)
            bp_results[str(dim)] = {"Spectral_MRR": float(mrr)}

    # Load existing atlas results for context
    atlas_file = RESULTS / "function_prediction_atlas.json"
    ppi_baselines = {}
    if atlas_file.exists():
        atlas_data = json.load(open(atlas_file, encoding="utf-8"))
        for code, label in ASPECTS.items():
            ont_data = atlas_data.get("ontologies", {}).get(code, {})
            for m, r in ont_data.get("methods", {}).items():
                if "PPI_MRR" in r:
                    ppi_baselines[label] = r["PPI_MRR"]
                    break

    output = {
        "description": "Atlas Extension: MF/CC at d=512, d=1024",
        "results": results,
        "bp_from_sweep": bp_results,
        "ppi_baselines": ppi_baselines,
    }

    out_file = RESULTS / "atlas_extension_512.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Summary
    print(f"\n{'='*64}")
    print("  SUMMARY: Full Atlas (BP + MF + CC) at d=512 and d=1024")
    print(f"{'='*64}")
    for aspect in ["BP", "MF", "CC"]:
        ppi_val = ppi_baselines.get(aspect, 0.0)
        if aspect == "BP":
            data = bp_results
        else:
            data = results.get(aspect, {})
        for dim_str in sorted(data.keys(), key=int):
            mrr = data[dim_str].get("Spectral_MRR", data[dim_str].get("Spectral_MRR", 0.0))
            status = "EXCEEDS" if mrr > ppi_val > 0 else "below"
            delta = mrr - ppi_val if ppi_val > 0 else 0
            print(f"  {aspect} d={dim_str}: MRR={mrr:.4f} vs PPI={ppi_val:.4f} "
                  f"({delta:+.4f}) [{status}]")

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return output


if __name__ == "__main__":
    run()
