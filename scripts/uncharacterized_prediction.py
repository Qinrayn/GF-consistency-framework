#!/usr/bin/env python3
"""
Uncharacterized Protein Mining (Step 68 / Phase 20)
====================================================

Identify yeast proteins with NO experimental GO annotation and predict
their functions using the d=256 Spectral embedding.

For each uncharacterized protein:
  1. Find k=10 nearest neighbors in d=256 Spectral embedding
  2. Predict GO terms via weighted voting (cosine similarity weights)
  3. Compute confidence metrics (neighbor consensus, functional coherence)

Output
------
- results/uncharacterized_predictions.tsv
- results/uncharacterized_summary.json
- figures/Fig79_uncharacterized_network.png
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
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
    TARGET_STD,
)
from function_prediction import (
    build_alias_mapping,
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

GAF_FILE = DATA / "gene_association.sgd.gaf.gz"
NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"

# Use d=256 Spectral embedding (best performing)
EMBEDDING_DIM = 256
K_NEIGHBORS = 10
CONSENSUS_THRESHOLD = 0.5  # >50% of neighbors agree on a term
MIN_NEIGHBORS_AGREED = 3   # at least 3 neighbors share the term

# GO aspect codes
ASPECTS = {"P": "Biological Process", "F": "Molecular Function", "C": "Cellular Component"}


# ============================================================
# Parse ALL experimental annotations (all 3 aspects)
# ============================================================

def parse_gaf_all_aspects(sgd_to_string, orf_to_string, network_nodes):
    """Parse GAF for all 3 GO aspects, experimental evidence only."""
    annotations = defaultdict(lambda: defaultdict(set))
    total_lines = 0
    experimental = 0
    mapped = 0
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
            aspect = cols[8]
            if evidence not in EXPERIMENTAL_CODES:
                continue
            if aspect not in ASPECTS:
                continue

            experimental += 1
            go_term = cols[4]
            if not go_term.startswith("GO:"):
                continue

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

            if string_id is not None:
                annotations[string_id][aspect].add(go_term)
                mapped += 1
            else:
                unmapped += 1

    stats = {
        "total_gaf_lines": total_lines,
        "experimental_annotations": experimental,
        "mapped": mapped,
        "unmapped": unmapped,
    }
    return annotations, stats


# ============================================================
# Load GO term names from OBO
# ============================================================

def load_go_names():
    """Load GO term names from the OBO file or GAF comments."""
    go_names = {}
    obo_file = DATA / "go.obo"
    if obo_file.exists():
        with open(obo_file, encoding="utf-8") as f:
            current_id = None
            for line in f:
                line = line.strip()
                if line.startswith("id: GO:"):
                    current_id = line.split("id: ")[1]
                elif line.startswith("name:") and current_id:
                    go_names[current_id] = line.split("name: ")[1]
                    current_id = None
    return go_names


# ============================================================
# Prediction Engine
# ============================================================

def predict_uncharacterized(coords, nodes, annotations, k=K_NEIGHBORS):
    """Predict functions for uncharacterized proteins.

    Parameters
    ----------
    coords : ndarray (n, d)
        Embedding coordinates.
    nodes : list
        Node IDs in same order as coords rows.
    annotations : dict
        {node_id: {aspect: set(go_terms)}}
    k : int
        Number of nearest neighbors.

    Returns
    -------
    predictions : list of dict
        Sorted by confidence, each with:
        - protein_id, aspect, go_term, go_name,
        - score, n_neighbors_agreed, n_neighbors_total,
        - neighbor_ids, consensus_fraction
    """
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Identify annotated vs uncharacterized proteins
    annotated = set()
    for pid in nodes:
        if pid in annotations:
            total_terms = sum(len(v) for v in annotations[pid].values())
            if total_terms > 0:
                annotated.add(pid)

    uncharacterized = [n for n in nodes if n not in annotated]
    print(f"  Annotated: {len(annotated)}, Uncharacterized: {len(uncharacterized)}")

    if not uncharacterized:
        print("  No uncharacterized proteins found!")
        return []

    # Build kNN model on ALL proteins
    n_neighbors = min(k + 1, len(coords))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(coords)

    # For each uncharacterized protein, find neighbors and predict
    predictions = []
    for i, pid in enumerate(uncharacterized):
        if pid not in node_to_idx:
            continue

        query_idx = node_to_idx[pid]
        query_vec = coords[query_idx]

        # Find k nearest neighbors (excluding self)
        distances, indices = nn_model.kneighbors([query_vec], n_neighbors=k + 1)
        distances = distances[0]
        indices = indices[0]

        # Remove self from neighbors
        neighbor_mask = indices != query_idx
        neighbor_indices = indices[neighbor_mask][:k]
        neighbor_distances = distances[neighbor_mask][:k]

        # Compute cosine similarity weights (convert distance to similarity)
        neighbor_ids = [nodes[j] for j in neighbor_indices]
        # Use inverse distance as weight
        weights = 1.0 / np.maximum(neighbor_distances, 1e-10)
        weights = weights / weights.sum()  # normalize

        # Collect weighted votes per aspect per term
        for aspect in ASPECTS:
            term_scores = defaultdict(float)
            term_neighbors = defaultdict(list)

            for nid, w in zip(neighbor_ids, weights):
                if nid in annotations and aspect in annotations[nid]:
                    for term in annotations[nid][aspect]:
                        term_scores[term] += w
                        term_neighbors[term].append(nid)

            # Filter by consensus
            n_annotated_neighbors = sum(
                1 for nid in neighbor_ids
                if nid in annotations and aspect in annotations[nid]
            )

            for term, score in term_scores.items():
                n_agreed = len(term_neighbors[term])
                consensus = n_agreed / max(n_annotated_neighbors, 1)

                if consensus >= CONSENSUS_THRESHOLD and n_agreed >= MIN_NEIGHBORS_AGREED:
                    predictions.append({
                        "protein_id": pid,
                        "aspect": aspect,
                        "go_term": term,
                        "score": float(score),
                        "n_neighbors_agreed": n_agreed,
                        "n_neighbors_total": n_annotated_neighbors,
                        "consensus_fraction": float(consensus),
                        "neighbor_ids": ",".join(term_neighbors[term]),
                    })

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(uncharacterized)} "
                  f"uncharacterized proteins...")

    # Sort by score descending
    predictions.sort(key=lambda x: -x["score"])
    return predictions, uncharacterized


# ============================================================
# Main
# ============================================================

def run():
    """Run uncharacterized protein mining."""
    t_start = time.time()
    print(BANNER)
    print("  Uncharacterized Protein Mining (d=256 Spectral)")
    print(BANNER)

    np.random.seed(SEED)

    # ---- Load network ----
    print(f"\n[1/6] Loading network ...")
    G = nx.read_edgelist(str(NETWORK_FILE))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---- Load embedding ----
    print(f"\n[2/6] Loading Spectral d={EMBEDDING_DIM} embedding ...")
    emb_file = EMB / f"Spectral_d{EMBEDDING_DIM}_full.npy"
    nodes_file = EMB / f"Spectral_d{EMBEDDING_DIM}_full_nodes.json"
    coords = np.load(str(emb_file))
    with open(nodes_file, encoding="utf-8") as f:
        nodes = json.load(f)
    print(f"  Embedding: {coords.shape}")

    # ---- Parse annotations ----
    print(f"\n[3/6] Parsing GO annotations (all aspects, experimental only) ...")
    sgd_map, orf_map, net_nodes = build_alias_mapping()
    annotations, ann_stats = parse_gaf_all_aspects(sgd_map, orf_map, net_nodes)

    total_annotated = sum(
        1 for pid, aspects in annotations.items()
        if sum(len(v) for v in aspects.values()) > 0
    )
    print(f"  Annotations: {ann_stats}")
    print(f"  Proteins with annotations: {total_annotated}")

    # Per-aspect counts
    for code, name in ASPECTS.items():
        n = sum(1 for a in annotations.values() if code in a and len(a[code]) > 0)
        print(f"    {name} ({code}): {n} proteins")

    # ---- Load GO term names ----
    print(f"\n[4/6] Loading GO term names ...")
    go_names = load_go_names()
    print(f"  Loaded {len(go_names)} GO term names")

    # ---- Run predictions ----
    print(f"\n[5/6] Predicting functions for uncharacterized proteins ...")
    result = predict_uncharacterized(coords, nodes, annotations)

    if not result or not result[0]:
        print("  No predictions generated!")
        return

    predictions, uncharacterized_list = result
    n_proteins_predicted = len(set(p["protein_id"] for p in predictions))
    print(f"\n  Total predictions: {len(predictions)}")
    print(f"  Proteins with >= 1 prediction: {n_proteins_predicted}")
    print(f"  Uncharacterized proteins: {len(uncharacterized_list)}")

    # Per-aspect breakdown
    for code, name in ASPECTS.items():
        aspect_preds = [p for p in predictions if p["aspect"] == code]
        aspect_proteins = len(set(p["protein_id"] for p in aspect_preds))
        print(f"    {name} ({code}): {len(aspect_preds)} predictions "
              f"for {aspect_proteins} proteins")

    # ---- Save predictions ----
    print(f"\n[6/6] Saving results ...")

    # Add GO term names
    for p in predictions:
        p["go_name"] = go_names.get(p["go_term"], "unknown")

    # TSV output
    tsv_file = RESULTS / "uncharacterized_predictions.tsv"
    with open(tsv_file, "w", encoding="utf-8") as f:
        f.write("protein_id\taspect\tgo_term\tgo_name\tscore\t"
                "n_neighbors_agreed\tn_neighbors_total\t"
                "consensus_fraction\tneighbor_ids\n")
        for p in predictions:
            f.write(f"{p['protein_id']}\t{p['aspect']}\t{p['go_term']}\t"
                    f"{p['go_name']}\t{p['score']:.4f}\t"
                    f"{p['n_neighbors_agreed']}\t{p['n_neighbors_total']}\t"
                    f"{p['consensus_fraction']:.3f}\t{p['neighbor_ids']}\n")
    print(f"  Saved {tsv_file}")

    # Summary JSON
    summary = {
        "description": "Uncharacterized Protein Function Predictions from d=256 Spectral Embedding",
        "embedding_dimension": EMBEDDING_DIM,
        "k_neighbors": K_NEIGHBORS,
        "consensus_threshold": CONSENSUS_THRESHOLD,
        "min_neighbors_agreed": MIN_NEIGHBORS_AGREED,
        "network_size": G.number_of_nodes(),
        "annotated_proteins": total_annotated,
        "uncharacterized_proteins": len(uncharacterized_list),
        "total_predictions": len(predictions),
        "proteins_predicted": n_proteins_predicted,
        "annotation_stats": ann_stats,
        "per_aspect": {},
        "top_20": [],
    }

    for code, name in ASPECTS.items():
        aspect_preds = [p for p in predictions if p["aspect"] == code]
        aspect_proteins = len(set(p["protein_id"] for p in aspect_preds))
        summary["per_aspect"][code] = {
            "name": name,
            "predictions": len(aspect_preds),
            "proteins": aspect_proteins,
        }

    # Top 20 highest-confidence predictions
    for p in predictions[:20]:
        summary["top_20"].append({
            "protein_id": p["protein_id"],
            "aspect": p["aspect"],
            "go_term": p["go_term"],
            "go_name": p["go_name"],
            "score": p["score"],
            "consensus": p["consensus_fraction"],
            "n_agreed": p["n_neighbors_agreed"],
        })

    json_file = RESULTS / "uncharacterized_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved {json_file}")

    # ---- Figure ----
    plot_uncharacterized_summary(predictions, uncharacterized_list, G, nodes, coords)

    elapsed = time.time() - t_start
    print(f"\nUncharacterized mining completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Print top-10
    print(f"\n{'='*64}")
    print("  TOP 10 HIGHEST-CONFIDENCE PREDICTIONS")
    print(f"{'='*64}")
    for i, p in enumerate(predictions[:10], 1):
        print(f"  {i:2d}. {p['protein_id'][:20]:20s} | {ASPECTS[p['aspect']]:22s} | "
              f"{p['go_name'][:35]:35s} | consensus={p['consensus_fraction']:.0%} "
              f"({p['n_neighbors_agreed']}/{p['n_neighbors_total']})")

    return summary


# ============================================================
# Figure
# ============================================================

def plot_uncharacterized_summary(predictions, uncharacterized, G, nodes, coords):
    """Visualize uncharacterized protein predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: Prediction counts by aspect ---
    ax = axes[0]
    aspect_counts = Counter(p["aspect"] for p in predictions)
    aspect_proteins = {
        code: len(set(p["protein_id"] for p in predictions if p["aspect"] == code))
        for code in ASPECTS
    }

    x = np.arange(len(ASPECTS))
    width = 0.35

    bars1 = ax.bar(x - width/2, [aspect_counts.get(c, 0) for c in ASPECTS],
                   width, label="Predictions", color=["#d62728", "#3182bd", "#2ca02c"],
                   edgecolor="white")
    bars2 = ax.bar(x + width/2, [aspect_proteins.get(c, 0) for c in ASPECTS],
                   width, label="Proteins", color=["#ff9896", "#9ecae1", "#98df8a"],
                   edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ASPECTS[c]}\n({c})" for c in ASPECTS], fontsize=10)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Predictions by GO Ontology", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 5, str(int(h)),
                   ha="center", fontsize=10, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 5, str(int(h)),
                   ha="center", fontsize=10)

    # --- Right: Consensus distribution ---
    ax = axes[1]
    consensus_values = [p["consensus_fraction"] for p in predictions]
    if consensus_values:
        for code, (color, label) in zip(ASPECTS,
            [("#d62728", "BP"), ("#3182bd", "MF"), ("#2ca02c", "CC")]):
            vals = [p["consensus_fraction"] for p in predictions if p["aspect"] == code]
            if vals:
                ax.hist(vals, bins=20, alpha=0.6, color=color, label=f"{label} (n={len(vals)})")

    ax.set_xlabel("Consensus Fraction", fontsize=12)
    ax.set_ylabel("Number of Predictions", fontsize=12)
    ax.set_title("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axvline(CONSENSUS_THRESHOLD, color="grey", linestyle="--", alpha=0.5,
              label=f"Threshold ({CONSENSUS_THRESHOLD:.0%})")

    plt.suptitle(f"Uncharacterized Protein Mining (d={EMBEDDING_DIM}, k={K_NEIGHBORS})\n"
                 f"{len(predictions)} predictions for {len(set(p['protein_id'] for p in predictions))} "
                 f"of {len(uncharacterized)} uncharacterized proteins",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = FIGURES / "Fig79_uncharacterized_summary.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run()
