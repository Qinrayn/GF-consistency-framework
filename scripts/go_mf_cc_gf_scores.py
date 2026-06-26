#!/usr/bin/env python3
"""
go_mf_cc_gf_scores.py
=======================
Compute G-F Scores using GO Molecular Function (MF) and Cellular Component
(CC) annotations in addition to the default Biological Process (BP).

This script demonstrates that the G-F consistency framework generalises
across all three GO ontology aspects.  For each aspect it:

1. Parses the SGD GAF file (experimental evidence codes only) to build a
   gene-to-GO-term map restricted to the curated 153-node subnetwork.
2. Computes a Spectral embedding (2-D, normalised Laplacian) of the
   curated PPI subnetwork.
3. Computes the G-F Score (mean functional purity over the unified
   integration interval [0.05, 0.422]) using greedy modularity communities.

Results are saved as JSON and printed as a summary table.

Usage
-----
    python scripts/go_mf_cc_gf_scores.py
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
from scipy.integrate import trapezoid
from scipy.sparse import csgraph
from networkx.algorithms.community import greedy_modularity_communities, modularity

# ---------------------------------------------------------------------------
# Path setup -- allow running from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    TARGET_STD,
    get_data_dir,
    get_results_dir,
    load_curated_network,
    rescale_coordinates,
    precompute_distance_matrix,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA = get_data_dir()
RESULTS = get_results_dir()
# RESULTS.mkdir(parents=True, exist_ok=True)  # deferred to run() — P1-4b

GAF_FILE = DATA / "gene_association.sgd.gaf.gz"
ALIAS_FILE = DATA / "4932.protein.aliases.v11.5.txt.gz"

# Experimental evidence codes (same as function_prediction.py)
EXPERIMENTAL_CODES = {
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
    "HTP", "HDA", "HMP", "HGI", "HEP",
}

# GO ontology aspects: GAF column 8 -> label
ASPECT_MAP = {
    "P": "BP",   # Biological Process
    "F": "MF",   # Molecular Function
    "C": "CC",   # Cellular Component
}

# G-F Score integration parameters (same as compute_gf.py / utils.py)
GF_R_MIN = 0.05
GF_R_MAX = 0.422
N_POINTS = 25

BANNER = "=" * 64


# ---------------------------------------------------------------------------
# 1. Alias mapping (SGD / ORF -> network node ID)
# ---------------------------------------------------------------------------

def build_alias_mapping():
    """Build ORF-name -> ORF-name identity mapping for the curated network.

    The curated 153-node network already uses ORF names (e.g. YBR160W) as
    node identifiers, so we simply collect the set of valid ORF names from
    the curated node list.  We also build a mapping from SGD systematic IDs
    and gene symbols to ORF names via the aliases file for GAF parsing.

    Returns
    -------
    sgd_to_orf : dict[str, str]
        SGD systematic ID -> ORF name.
    symbol_to_orf : dict[str, str]
        Gene symbol -> ORF name.
    curated_nodes : set[str]
        All ORF names in the curated 153-node network.
    """
    # Load curated node list
    nodes_file = DATA / "curated_153_nodes.txt"
    curated_nodes = set()
    if nodes_file.exists():
        with open(nodes_file, encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    curated_nodes.add(name)

    # If nodes file missing, fall back to loading the edgelist
    if not curated_nodes:
        edgelist_file = DATA / "curated_153_ppi.edgelist"
        with open(edgelist_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    curated_nodes.add(parts[0])
                    curated_nodes.add(parts[1])

    # Build SGD_ID -> ORF and gene-symbol -> ORF from aliases file
    sgd_to_orfs = defaultdict(set)
    symbol_to_orfs = defaultdict(set)

    with gzip.open(str(ALIAS_FILE), "rt", encoding="utf-8") as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            raw_string_id = parts[0]
            alias = parts[1]
            source = parts[2]

            # Extract node ID (strip "4932." prefix)
            string_id = raw_string_id.split(".", 1)[1] if "." in raw_string_id else raw_string_id

            # We only care about mappings to curated ORF names
            # The STRING IDs themselves might be ORF names
            target_orf = None
            if string_id in curated_nodes:
                target_orf = string_id

            if target_orf is None:
                continue

            if source == "SGD_ID":
                sgd_to_orfs[alias].add(target_orf)
            elif source in ("Ensembl_SGD_GENE", "SGD_SYNONYM",
                            "Ensembl_SGD_TRANSCRIPT"):
                if alias and len(alias) >= 3 and alias[0] == "Y":
                    symbol_to_orfs[alias].add(target_orf)

    sgd_to_orf = {k: min(v, key=len) for k, v in sgd_to_orfs.items() if v}
    symbol_to_orf = {k: min(v, key=len) for k, v in symbol_to_orfs.items() if v}

    print(f"  Alias mapping: {len(sgd_to_orf)} SGD IDs, "
          f"{len(symbol_to_orf)} symbols -> ORF names")
    print(f"  Curated network nodes: {len(curated_nodes)}")

    return sgd_to_orf, symbol_to_orf, curated_nodes


# ---------------------------------------------------------------------------
# 2. GAF parsing -- all three aspects
# ---------------------------------------------------------------------------

def parse_gaf_all_aspects(sgd_to_orf, symbol_to_orf, curated_nodes):
    """Parse SGD GAF for experimental annotations across BP, MF, CC.

    Parameters
    ----------
    sgd_to_orf : dict
        SGD systematic ID -> ORF name.
    symbol_to_orf : dict
        Gene symbol -> ORF name.
    curated_nodes : set
        Valid ORF names in the curated network.

    Returns
    -------
    annotations : dict[str, dict[str, set[str]]]
        {aspect_label: {orf_name: set(GO terms)}}
        aspect_label is one of "BP", "MF", "CC".
    stats : dict
        Parsing statistics per aspect.
    """
    annotations = {asp: defaultdict(set) for asp in ASPECT_MAP.values()}
    counts = {asp: 0 for asp in ASPECT_MAP.values()}
    total_lines = 0
    experimental_lines = 0

    with gzip.open(str(GAF_FILE), "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("!") or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            total_lines += 1

            evidence = cols[6]
            aspect_code = cols[8]

            # Filter: experimental evidence only
            if evidence not in EXPERIMENTAL_CODES:
                continue

            # Map aspect code to label
            if aspect_code not in ASPECT_MAP:
                continue

            aspect_label = ASPECT_MAP[aspect_code]
            experimental_lines += 1

            go_term = cols[4]
            if not go_term.startswith("GO:"):
                continue

            # Map gene ID to ORF name
            sgd_id = cols[1]
            gene_sym = cols[2]
            orf_name = cols[10] if len(cols) > 10 else ""

            target_orf = None

            # Try direct ORF name match
            if orf_name and orf_name in curated_nodes:
                target_orf = orf_name

            # Try SGD systematic ID
            if target_orf is None and sgd_id in sgd_to_orf:
                candidate = sgd_to_orf[sgd_id]
                if candidate in curated_nodes:
                    target_orf = candidate

            # Try gene symbol as ORF
            if target_orf is None and orf_name and orf_name in symbol_to_orf:
                candidate = symbol_to_orf[orf_name]
                if candidate in curated_nodes:
                    target_orf = candidate

            # Try gene_sym in symbol_to_orf
            if target_orf is None and gene_sym in symbol_to_orf:
                candidate = symbol_to_orf[gene_sym]
                if candidate in curated_nodes:
                    target_orf = candidate

            if target_orf is None:
                continue

            annotations[aspect_label][target_orf].add(go_term)
            counts[aspect_label] += 1

    # Convert defaultdicts to regular dicts with list values
    result = {}
    stats = {}
    for asp in ["BP", "MF", "CC"]:
        go_map = {k: sorted(v) for k, v in annotations[asp].items() if v}
        result[asp] = go_map
        unique_terms = set(t for ts in go_map.values() for t in ts)
        stats[asp] = {
            "annotation_lines": counts[asp],
            "proteins_annotated": len(go_map),
            "unique_terms": len(unique_terms),
        }

    stats["total_gaf_lines"] = total_lines
    stats["total_experimental_lines"] = experimental_lines

    print(f"  GAF: {total_lines} total lines, {experimental_lines} experimental")
    for asp in ["BP", "MF", "CC"]:
        s = stats[asp]
        print(f"    {asp}: {s['annotation_lines']} annotations, "
              f"{s['proteins_annotated']} proteins, "
              f"{s['unique_terms']} unique terms")

    return result, stats


# ---------------------------------------------------------------------------
# 3. Spectral embedding from curated network
# ---------------------------------------------------------------------------

def compute_spectral_embedding(G, nodes):
    """Compute 2-D Spectral embedding from normalised Laplacian.

    Parameters
    ----------
    G : nx.Graph
        The curated subnetwork.
    nodes : list[str]
        Ordered node labels.

    Returns
    -------
    coords : np.ndarray, shape (n, 2)
        Spectral embedding coordinates (rescaled to target_std).
    """
    n = len(nodes)
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes)

    # Use sparse eigendecomposition for efficiency
    from scipy.sparse.linalg import eigsh
    eigvals, eigvecs = eigsh(L.astype(float), k=min(3, n - 1), which="SM",
                             tol=1e-6)

    # Sort by eigenvalue ascending; skip the trivial eigenvector (index 0)
    sort_idx = np.argsort(eigvals)
    # Take eigenvectors 1 and 2 (non-trivial)
    coords = eigvecs[:, sort_idx[1:3]]

    # Rescale to target standard deviation
    coords = rescale_coordinates(coords, target_std=TARGET_STD)

    return coords


# ---------------------------------------------------------------------------
# 4. G-F curve and score computation (standalone, no precomputed embeddings)
# ---------------------------------------------------------------------------

def compute_gf_curve_standalone(coords, nodes, go_map, r_vals):
    """Compute G-F purity curve using greedy modularity communities.

    This is a standalone version that does not depend on precomputed
    embeddings.  It follows the same algorithm as utils.compute_gf_curve.

    Parameters
    ----------
    coords : np.ndarray (n, d)
    nodes : list[str]
    go_map : dict  {node: [go_terms]}
    r_vals : np.ndarray

    Returns
    -------
    purities : list[float]
    """
    dist_matrix = precompute_distance_matrix(coords)
    n = dist_matrix.shape[0]

    # Pre-sort edges by distance
    iu = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[iu]
    sort_idx = np.argsort(edge_dists)
    sorted_rows = iu[0][sort_idx]
    sorted_cols = iu[1][sort_idx]
    sorted_d = edge_dists[sort_idx]

    r_order = np.argsort(r_vals)
    purities_out = [0.0] * len(r_vals)

    G_r = nx.Graph()
    G_r.add_nodes_from(range(n))
    edge_ptr = 0
    n_edges_total = len(sorted_d)

    _cache = {}

    for rank, orig_idx in enumerate(r_order):
        r = float(r_vals[orig_idx])

        while edge_ptr < n_edges_total and sorted_d[edge_ptr] < r:
            G_r.add_edge(int(sorted_rows[edge_ptr]),
                         int(sorted_cols[edge_ptr]))
            edge_ptr += 1

        ne = G_r.number_of_edges()
        if ne == 0:
            continue

        if ne in _cache:
            communities = _cache[ne]
        else:
            communities = list(greedy_modularity_communities(G_r))
            _cache[ne] = communities

        # Compute functional purity
        purities = []
        for comm in communities:
            if not comm:
                continue
            comm_names = [nodes[idx] for idx in comm]
            go_terms = []
            for name in comm_names:
                if name in go_map:
                    go_terms.extend(go_map[name])
            if not go_terms:
                continue
            term_counts = Counter(go_terms)
            most_common_count = term_counts.most_common(1)[0][1]
            purity = most_common_count / len(go_terms)
            purities.append(purity)

        purities_out[orig_idx] = float(np.mean(purities)) if purities else 0.0

    return purities_out


def compute_gf_score_from_curve(r_vals, purity_vals,
                                r_min=GF_R_MIN, r_max=GF_R_MAX):
    """Compute G-F Score as mean purity over [r_min, r_max]."""
    r = np.asarray(r_vals)
    p = np.asarray(purity_vals)
    mask = (r >= r_min) & (r <= r_max)
    r_sub = r[mask]
    p_sub = p[mask]
    if len(r_sub) < 2:
        return 0.0
    return float(trapezoid(p_sub, r_sub) / (r_max - r_min))


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def main():
    np.random.seed(SEED)

    RESULTS.mkdir(parents=True, exist_ok=True)
    print(BANNER)
    print("G-F Scores across GO Ontology Aspects (BP, MF, CC)")
    print(BANNER)

    # Step 1: Build alias mapping
    print("\n[1/5] Building alias mapping ...")
    sgd_to_orf, symbol_to_orf, curated_nodes = build_alias_mapping()

    # Step 2: Parse GAF for all three aspects
    print("\n[2/5] Parsing GAF file (BP, MF, CC) ...")
    annotations, ann_stats = parse_gaf_all_aspects(
        sgd_to_orf, symbol_to_orf, curated_nodes,
    )

    # Step 3: Load curated 153-node network
    print("\n[3/5] Loading curated 153-node network ...")
    G, nodes, _ = load_curated_network(DATA)
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Filter annotations to nodes actually in the network
    go_maps = {}
    for asp in ["BP", "MF", "CC"]:
        filtered = {n: annotations[asp][n]
                    for n in nodes if n in annotations[asp]}
        go_maps[asp] = filtered
        n_annotated = len(filtered)
        print(f"  {asp}: {n_annotated}/{len(nodes)} nodes annotated "
              f"in curated network")

    # Step 4: Compute Spectral embedding
    print("\n[4/5] Computing Spectral embedding (2-D) ...")
    t0 = time.time()
    coords = compute_spectral_embedding(G, nodes)
    print(f"  Spectral embedding computed in {time.time() - t0:.1f}s")
    print(f"  Coordinate range: x=[{coords[:,0].min():.3f}, {coords[:,0].max():.3f}], "
          f"y=[{coords[:,1].min():.3f}, {coords[:,1].max():.3f}]")

    # Step 5: Compute G-F Scores for each ontology aspect
    print("\n[5/5] Computing G-F Scores ...")
    r_vals = np.linspace(GF_R_MIN, GF_R_MAX, N_POINTS)

    results = {}
    for asp in ["BP", "MF", "CC"]:
        go_map = go_maps[asp]
        if not go_map:
            print(f"  {asp}: No annotations available, skipping")
            results[asp] = {"Spectral": 0.0}
            continue

        # Filter to annotated nodes only for curve computation
        annotated_mask = [i for i, n in enumerate(nodes) if n in go_map]
        ann_nodes = [nodes[i] for i in annotated_mask]
        ann_coords = coords[annotated_mask]

        purities = compute_gf_curve_standalone(
            ann_coords, ann_nodes, go_map, r_vals,
        )
        score = compute_gf_score_from_curve(r_vals, purities)
        results[asp] = {"Spectral": round(score, 4)}

        print(f"  {asp}: GF Score = {score:.4f}  "
              f"(purity range [{min(purities):.3f}, {max(purities):.3f}])")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "description": (
            "G-F Scores computed with Spectral embedding on the curated "
            "153-node yeast PPI subnetwork, using each GO ontology aspect "
            "(BP, MF, CC) independently."
        ),
        "integration_interval": [GF_R_MIN, GF_R_MAX],
        "n_points": N_POINTS,
        "embedding": "Spectral (2-D, normalised Laplacian)",
        "network": "curated_153_ppi",
        "annotation_stats": ann_stats,
        "gf_scores": results,
    }

    out_file = RESULTS / "gf_scores_go_aspects.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {out_file}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print(f"\n{BANNER}")
    print("Summary: G-F Scores by GO Ontology Aspect")
    print(BANNER)
    print(f"  {'Aspect':<8s} {'GF Score':>10s} {'Annotated':>10s} {'Terms':>8s}")
    print(f"  {'-'*38}")
    for asp in ["BP", "MF", "CC"]:
        score = results[asp].get("Spectral", 0.0)
        n_ann = ann_stats[asp]["proteins_annotated"]
        n_terms = ann_stats[asp]["unique_terms"]
        print(f"  {asp:<8s} {score:>10.4f} {n_ann:>10d} {n_terms:>8d}")
    print(f"\n  Integration interval: [{GF_R_MIN}, {GF_R_MAX}]")
    print(f"  Sampling points: {N_POINTS}")
    print(f"  Network: curated 153-node yeast PPI")
    print(f"  Embedding: Spectral (2-D, normalised Laplacian)")
    print(BANNER)


if __name__ == "__main__":
    main()
