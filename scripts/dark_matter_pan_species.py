#!/usr/bin/env python3
"""
Pan-Species Dark Matter Mining (Step 72 / Phase 3.1)
=====================================================

Mine functional dark matter pairs across all 5 species using uniform criteria:
  1. Network distance >= 5 hops (or disconnected)
  2. Top-50 nearest neighbors in Spectral d=64 embedding
  3. STRING score < 700 (or no edge)
  4. Shared experimental GO BP annotation

Species: Yeast, Human, Mouse, Fly, E. coli

Output
------
- results/dark_matter_pan_species.json
- figures/Fig88_pan_species_dark_matter.png
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
    SEED, TARGET_STD, STRING_MIN_SCORE,
    get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
)
from function_prediction import EXPERIMENTAL_CODES

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()
EMB = get_embeddings_dir()

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

KNN_SEARCH = 50
MIN_NETWORK_DIST = 5

BANNER = "=" * 64

# Species registry
SPECIES = {
    "yeast": {
        "name": "Yeast (S. cerevisiae)",
        "network_file": DATA / "yeast_ppi_5936.edgelist",
        "links_file": DATA / "4932.protein.links.v11.5.txt.gz",
        "gaf_file": DATA / "gene_association.sgd.gaf.gz",
        "aliases_file": DATA / "4932.protein.aliases.v11.5.txt.gz",
        "taxon": "4932",
        "embedding_dim": 64,
    },
    "human": {
        "name": "Human (H. sapiens)",
        "network_file": DATA / "human_validation" / "9606.protein.links.v12.0.txt.gz",
        "links_file": None,  # use network_file directly
        "gaf_file": DATA / "human_validation" / "goa_human.gaf.gz",
        "aliases_file": DATA / "human_validation" / "9606.protein.aliases.v12.0.txt.gz",
        "taxon": "9606",
        "embedding_dim": 64,
    },
    "mouse": {
        "name": "Mouse (M. musculus)",
        "network_file": DATA / "mouse_ppi.edgelist",
        "links_file": DATA / "10090.protein.links.v11.5.txt.gz",
        "gaf_file": DATA / "mgi.gaf.gz",
        "aliases_file": DATA / "10090.protein.aliases.v11.5.txt.gz",
        "taxon": "10090",
        "embedding_dim": 64,
    },
    "fly": {
        "name": "Fly (D. melanogaster)",
        "network_file": DATA / "fly" / "7227.protein.links.v11.5.txt.gz",
        "links_file": DATA / "fly" / "7227.protein.links.v11.5.txt.gz",
        "gaf_file": DATA / "fly" / "fb.gaf.gz",
        "aliases_file": DATA / "fly" / "7227.protein.aliases.v11.5.txt.gz",
        "taxon": "7227",
        "embedding_dim": 64,
    },
    "ecoli": {
        "name": "E. coli K-12",
        "network_file": DATA / "511145.protein.links.v11.5.txt.gz",
        "links_file": DATA / "511145.protein.links.v11.5.txt.gz",
        "gaf_file": DATA / "gene_association.ecocyc.gaf.gz",
        "aliases_file": DATA / "511145.protein.aliases.v11.5.txt.gz",
        "taxon": "511145",
        "embedding_dim": 64,
    },
}


# ============================================================
# Network Loading
# ============================================================

def load_network(species_key, config):
    """Load PPI network for a species."""
    net_file = config["network_file"]

    if species_key == "human":
        # Human uses STRING v12.0 gzipped links file
        G = nx.Graph()
        with gzip.open(str(net_file), "rt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    n1, n2 = parts[0], parts[1]
                    try:
                        score = int(parts[2])
                    except ValueError:
                        continue
                    if score >= STRING_MIN_SCORE:
                        G.add_edge(n1, n2)
        # Take LCC
        if G.number_of_nodes() > 0:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        return G

    if species_key in ("fly", "ecoli"):
        # Fly and E. coli use gzipped STRING links
        G = nx.Graph()
        with gzip.open(str(net_file), "rt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    n1, n2 = parts[0], parts[1]
                    try:
                        score = int(parts[2])
                    except ValueError:
                        continue
                    if score >= STRING_MIN_SCORE:
                        G.add_edge(n1, n2)
        if G.number_of_nodes() > 0:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        return G

    if species_key == "mouse":
        # Mouse uses pre-processed edgelist
        G = nx.read_edgelist(str(net_file))
        if G.number_of_nodes() > 0:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        return G

    # Yeast
    G = nx.read_edgelist(str(net_file))
    if G.number_of_nodes() > 0:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


# ============================================================
# Spectral Embedding (compute or load)
# ============================================================

def get_spectral_embedding(species_key, graph, dim):
    """Get Spectral embedding — load from disk if available, else compute."""
    # Check for pre-computed d=64 embeddings
    for pattern in [
        f"{species_key}_spectral_d{dim}.npy",
        f"Spectral_d{dim}_full.npy",
    ]:
        emb_path = EMB / pattern
        nodes_path = EMB / pattern.replace(".npy", "_nodes.json")
        if emb_path.exists() and nodes_path.exists():
            coords = np.load(str(emb_path))
            with open(nodes_path) as f:
                nodes = json.load(f)
            # Filter to graph nodes
            graph_nodes = set(graph.nodes())
            valid = [i for i, n in enumerate(nodes) if n in graph_nodes]
            if len(valid) > len(graph_nodes) * 0.8:
                return coords, nodes

    # Compute fresh
    nodes = sorted(graph.nodes())
    n = len(nodes)
    if n < dim + 2:
        print(f"    Network too small ({n} nodes) for d={dim}")
        return None, None

    adj = nx.adjacency_matrix(graph, nodelist=nodes, weight=None).astype(float)
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    D_inv_sqrt = diags(d_inv_sqrt)
    L_norm = D_inv_sqrt @ (diags(degrees) - adj) @ D_inv_sqrt

    n_eigs = min(dim + 1, n - 2)
    eigenvalues, eigenvectors = eigsh(L_norm, k=n_eigs, which="SM", tol=1e-6)
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    coords = eigenvectors[:, 1:dim + 1]

    for j in range(coords.shape[1]):
        col_std = coords[:, j].std()
        if col_std > 1e-10:
            coords[:, j] = coords[:, j] / col_std * TARGET_STD

    # Save
    emb_path = EMB / f"{species_key}_spectral_d{dim}.npy"
    nodes_path = EMB / f"{species_key}_spectral_d{dim}_nodes.json"
    np.save(str(emb_path), coords)
    with open(nodes_path, "w") as f:
        json.dump(nodes, f)

    return coords, nodes


# ============================================================
# GO Annotations
# ============================================================

def load_annotations(species_key, config):
    """Load experimental GO BP annotations for a species."""
    gaf_file = config["gaf_file"]
    annotations = defaultdict(set)

    if species_key in ("human", "mouse"):
        # Pre-processed JSON available
        json_file = DATA / f"{species_key}_go_annotations.json"
        if json_file.exists():
            with open(json_file) as f:
                raw = json.load(f)
            for pid, terms in raw.items():
                annotations[pid] = set(terms)
            return dict(annotations)

    # Parse GAF file
    with gzip.open(str(gaf_file), "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!") or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            evidence = cols[6]
            if evidence not in EXPERIMENTAL_CODES:
                continue
            aspect = cols[8]
            if aspect != "P":  # BP only
                continue
            go_term = cols[4]
            if not go_term.startswith("GO:"):
                continue
            # Use gene ID directly (species-specific format)
            gene_id = cols[1]
            annotations[gene_id].add(go_term)

    # Try to map to STRING IDs via aliases
    aliases_file = config["aliases_file"]
    if aliases_file.exists():
        alias_map = {}
        taxon = config["taxon"]
        opener = gzip.open if str(aliases_file).endswith(".gz") else open
        with opener(str(aliases_file), "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                string_id = parts[0]
                if "." in string_id:
                    string_id = string_id.split(".", 1)[1]
                alias_map[parts[1]] = string_id

        mapped = defaultdict(set)
        for gene_id, terms in annotations.items():
            string_id = alias_map.get(gene_id)
            if string_id:
                mapped[string_id].update(terms)
        if mapped:
            return dict(mapped)

    return dict(annotations)


# ============================================================
# Dark Matter Mining
# ============================================================

def mine_dark_matter(graph, coords, nodes, annotations):
    """Mine dark matter pairs for one species."""
    node_set = set(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Build kNN index
    nn_model = NearestNeighbors(n_neighbors=min(KNN_SEARCH + 1, len(coords)),
                                 metric="euclidean")
    nn_model.fit(coords)

    # Precompute shortest path lengths (sample for large networks)
    n_nodes = graph.number_of_nodes()

    pairs = []
    checked = set()

    for i, pid in enumerate(nodes):
        if pid not in annotations or pid not in graph:
            continue
        pid_terms = annotations[pid]
        if not pid_terms:
            continue

        # Find embedding neighbors
        if pid not in node_to_idx:
            continue
        query_idx = node_to_idx[pid]
        distances, indices = nn_model.kneighbors(
            coords[query_idx:query_idx + 1],
            n_neighbors=min(KNN_SEARCH + 1, len(coords))
        )

        for dist, idx in zip(distances[0], indices[0]):
            if idx == query_idx:
                continue
            neighbor_id = nodes[idx]
            if neighbor_id not in annotations:
                continue

            # Avoid duplicate pairs
            pair_key = tuple(sorted([pid, neighbor_id]))
            if pair_key in checked:
                continue
            checked.add(pair_key)

            # Check shared GO terms
            shared_terms = pid_terms & annotations.get(neighbor_id, set())
            if not shared_terms:
                continue

            # Check network distance
            try:
                net_dist = nx.shortest_path_length(graph, pid, neighbor_id)
            except nx.NetworkXNoPath:
                net_dist = -1  # disconnected

            if net_dist >= 0 and net_dist < MIN_NETWORK_DIST:
                continue

            pairs.append({
                "protein_a": pair_key[0],
                "protein_b": pair_key[1],
                "emb_dist": round(float(dist), 4),
                "network_dist": net_dist if net_dist >= 0 else -1,
                "disconnected": net_dist < 0,
                "shared_go_terms": sorted(shared_terms),
                "n_shared": len(shared_terms),
            })

        if (i + 1) % 500 == 0:
            print(f"    Checked {i+1}/{len(nodes)} proteins, "
                  f"{len(pairs)} DM pairs found so far")

    return pairs


# ============================================================
# Main
# ============================================================

def run():
    t_start = time.time()
    print(BANNER)
    print("  Pan-Species Dark Matter Mining (Phase 3.1)")
    print(BANNER)
    np.random.seed(SEED)

    all_results = {}

    for species_key in ["yeast", "human", "mouse", "fly", "ecoli"]:
        config = SPECIES[species_key]
        print(f"\n{'='*50}")
        print(f"  {config['name']}")
        print(f"{'='*50}")

        # Load network
        print(f"  Loading network ...")
        try:
            G = load_network(species_key, config)
        except Exception as e:
            print(f"  FAILED to load network: {e}")
            all_results[species_key] = {"error": str(e)}
            continue
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if G.number_of_nodes() < 100:
            print(f"  SKIP: network too small")
            all_results[species_key] = {"error": "network too small"}
            continue

        # Get embedding
        print(f"  Computing/loading Spectral d={config['embedding_dim']} embedding ...")
        try:
            coords, emb_nodes = get_spectral_embedding(species_key, G, config["embedding_dim"])
        except Exception as e:
            print(f"  FAILED to compute embedding: {e}")
            all_results[species_key] = {"error": str(e)}
            continue
        if coords is None:
            all_results[species_key] = {"error": "embedding computation failed"}
            continue
        print(f"  Embedding: {coords.shape}")

        # Load annotations
        print(f"  Loading GO BP annotations ...")
        try:
            annotations = load_annotations(species_key, config)
        except Exception as e:
            print(f"  FAILED to load annotations: {e}")
            all_results[species_key] = {"error": str(e)}
            continue

        # Filter to network nodes
        graph_nodes = set(G.nodes())
        ann_filtered = {k: v for k, v in annotations.items() if k in graph_nodes}
        n_annotated = len(ann_filtered)
        n_terms = len(set(t for ts in ann_filtered.values() for t in ts))
        print(f"  {n_annotated} annotated proteins, {n_terms} unique BP terms")

        if n_annotated < 50:
            print(f"  SKIP: too few annotated proteins")
            all_results[species_key] = {"error": "too few annotations"}
            continue

        # Mine dark matter
        print(f"  Mining dark matter pairs ...")
        t_mine = time.time()
        try:
            dm_pairs = mine_dark_matter(G, coords, emb_nodes, ann_filtered)
        except Exception as e:
            print(f"  FAILED during mining: {e}")
            all_results[species_key] = {"error": str(e)}
            continue
        mine_time = time.time() - t_mine

        n_pairs = len(dm_pairs)
        n_disconnected = sum(1 for p in dm_pairs if p["disconnected"])
        n_multi = sum(1 for p in dm_pairs if p["n_shared"] >= 2)
        n_unique_proteins = len(set(
            p["protein_a"] for p in dm_pairs
        ) | set(
            p["protein_b"] for p in dm_pairs
        ))

        print(f"  Found {n_pairs} dark matter pairs in {mine_time:.1f}s")
        print(f"    Disconnected: {n_disconnected}")
        print(f"    Multi-term: {n_multi}")
        print(f"    Unique proteins: {n_unique_proteins}")

        # Sort by embedding distance
        dm_pairs.sort(key=lambda p: p["emb_dist"])

        all_results[species_key] = {
            "name": config["name"],
            "network_nodes": G.number_of_nodes(),
            "network_edges": G.number_of_edges(),
            "annotated_proteins": n_annotated,
            "embedding_dim": config["embedding_dim"],
            "total_dm_pairs": n_pairs,
            "disconnected_pairs": n_disconnected,
            "multi_term_pairs": n_multi,
            "unique_dm_proteins": n_unique_proteins,
            "top_20_pairs": dm_pairs[:20],
        }

    # Save results
    output = {
        "description": "Pan-Species Dark Matter Mining (5 species, uniform criteria)",
        "version": "2.11.0",
        "criteria": {
            "min_network_distance": MIN_NETWORK_DIST,
            "knn_search_radius": KNN_SEARCH,
            "string_min_score": STRING_MIN_SCORE,
            "go_aspect": "Biological Process (experimental evidence only)",
            "embedding": "Spectral d=64",
        },
        "species_results": all_results,
    }

    out_file = RESULTS / "dark_matter_pan_species.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\nSaved to {out_file}")

    # Summary
    print(f"\n{'='*64}")
    print("  PAN-SPECIES DARK MATTER SUMMARY")
    print(f"{'='*64}")
    for sp_key in ["yeast", "human", "mouse", "fly", "ecoli"]:
        r = all_results.get(sp_key, {})
        if "error" in r:
            print(f"  {sp_key}: ERROR ({r['error']})")
        else:
            print(f"  {r['name']}: {r['total_dm_pairs']} pairs "
                  f"({r['disconnected_pairs']} disconnected, "
                  f"{r['unique_dm_proteins']} proteins)")

    # Figure
    plot_pan_species_summary(all_results)

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return output


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ============================================================
# Figure
# ============================================================

def plot_pan_species_summary(all_results):
    """Bar chart of dark matter pairs per species."""
    fig, ax = plt.subplots(figsize=(12, 6))

    species_names = []
    dm_counts = []
    disc_counts = []
    colors = ["#d62728", "#3182bd", "#2ca02c", "#ff7f0e", "#9467bd"]

    for sp_key, color in zip(["yeast", "human", "mouse", "fly", "ecoli"], colors):
        r = all_results.get(sp_key, {})
        if "error" in r:
            continue
        species_names.append(r["name"].split(" (")[0])
        dm_counts.append(r["total_dm_pairs"])
        disc_counts.append(r["disconnected_pairs"])

    if not species_names:
        plt.close(fig)
        return

    x = np.arange(len(species_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, dm_counts, width, label="Total DM pairs",
                   color=colors[:len(species_names)], edgecolor="white", alpha=0.8)
    bars2 = ax.bar(x + width/2, disc_counts, width, label="Disconnected",
                   color=colors[:len(species_names)], edgecolor="white", alpha=0.4,
                   hatch="//")

    for bar, count in zip(bars1, dm_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha="center", fontsize=11, fontweight="bold")
    for bar, count in zip(bars2, disc_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(species_names, fontsize=11)
    ax.set_ylabel("Number of Dark Matter Pairs", fontsize=12)
    ax.set_title("Pan-Species Functional Dark Matter Mining\n"
                 "(network dist ≥ 5, embedding KNN ≤ 50, shared GO BP)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES / "Fig88_pan_species_dark_matter.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


if __name__ == "__main__":
    run()
