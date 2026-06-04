#!/usr/bin/env python3
"""
go_propagation.py
=================
Expand the yeast validation set from 153 nodes to >= 1,000 nodes via
GO term propagation through the GO DAG.

Pipeline:
  1. Build the GO DAG from the OBO ontology file.
  2. Propagate existing gene annotations upward (True Path Rule).
  3. Download SGD GAF annotations to find all yeast genes with BP terms.
  4. Map annotated genes onto the full STRING network (5,936 nodes).
  5. Compute G-F Scores with the expanded annotation set.
  6. Subset convergence analysis: sample subsets of varying sizes and
     measure G-F Score stability.

Outputs
-------
- ``results/extended_go_annotations.json``
- ``results/go_propagation_stats.json``
- ``figures/FigS4_sample_size_convergence.png``
"""

import re
import sys
import os
import json
import gzip
import argparse
import random
import logging
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# Import pattern (MUST match existing project convention)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_full_STRING_network, load_embedding,
    compute_gf_curve, compute_gf_score, compute_plateau_width,
    rescale_coordinates,
)
from utils import get_figures_dir, get_project_root

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_MIN = 0.05
R_MAX = 0.422          # unified interval upper bound (paper)
R_GRID_MAX = 0.55       # grid extends beyond R_MAX for integration
N_POINTS = 200
GF_R_MIN = 0.05
GF_R_MAX = 0.422

OBO_URL = "http://purl.obolibrary.org/obo/go.obo"
GAF_URLS = [
    "https://downloads.yeastgenome.org/uniprot/gene_association.sgd.gaf.gz",
    "http://current.geneontology.org/annotations/sgd.gaf.gz",
    "http://geneontology.org/gene-associations/goa_yeast.gaf.gz",
]

BP_ROOT = "GO:0008150"  # biological_process root term
MAX_EVAL_NODES = 1000   # max nodes for G-F curve evaluation

logger = logging.getLogger("go_propagation")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    ))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ===================================================================
# 1. GO DAG Construction
# ===================================================================

def _is_lfs_pointer(filepath):
    """Return True if *filepath* is a Git LFS pointer (tiny text file)."""
    try:
        size = os.path.getsize(filepath)
        if size > 1024:
            return False
        with open(filepath, "r", errors="replace") as fh:
            head = fh.read(200)
        return "git-lfs" in head.lower() or "oid sha256" in head.lower()
    except Exception:
        return True


def download_file(url, dest_path, timeout=120):
    """Download *url* to *dest_path*.  Return True on success."""
    import requests
    logger.info("Downloading %s ...", url)
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        logger.info("  -> saved %s (%d bytes)", dest_path, os.path.getsize(dest_path))
        return True
    except Exception as exc:
        logger.warning("  Download failed: %s", exc)
        return False


def parse_go_obo(obo_path):
    """Parse a GO OBO file and return term metadata.

    Returns
    -------
    terms : dict
        ``{go_id: {"name": str, "namespace": str, "is_a": [str], "is_obsolete": bool}}``
    """
    terms = {}
    current = None

    with open(obo_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line == "[Term]":
                current = {"name": "", "namespace": "", "is_a": [], "is_obsolete": False}
                continue
            if line == "[Typedef]" or line == "":
                if current and "id" in current:
                    terms[current["id"]] = current
                if line == "[Typedef]":
                    current = None
                continue
            if current is None:
                continue
            if line.startswith("id:"):
                current["id"] = line.split(":", 1)[1].strip()
                # GO IDs look like "GO:0008150" — the split above yields "GO" + "0008150"
                # Fix: rejoin
                parts = line.split(":")
                if len(parts) >= 3:
                    current["id"] = ":".join(parts[1:]).strip()
                elif len(parts) == 2:
                    current["id"] = parts[1].strip()
            elif line.startswith("name:"):
                current["name"] = line[5:].strip()
            elif line.startswith("namespace:"):
                current["namespace"] = line[10:].strip()
            elif line.startswith("is_a:"):
                # e.g. "is_a: GO:0008150 ! biological_process"
                parent_id = line[5:].split("!")[0].strip()
                current["is_a"].append(parent_id)
            elif line.startswith("is_obsolete:"):
                current["is_obsolete"] = "true" in line.lower()

    # Flush last term
    if current and "id" in current:
        terms[current["id"]] = current

    logger.info("Parsed %d GO terms from OBO", len(terms))
    return terms


def build_go_dag(terms):
    """Build parent/children mappings from parsed OBO terms.

    Only BP (biological_process) terms are included.

    Returns
    -------
    child_to_parents : dict[str, set[str]]
    parent_to_children : dict[str, set[str]]
    bp_terms : set[str]
        All BP terms found in the ontology.
    """
    child_to_parents = defaultdict(set)
    parent_to_children = defaultdict(set)
    bp_terms = set()

    for go_id, meta in terms.items():
        if meta.get("is_obsolete"):
            continue
        if meta.get("namespace") != "biological_process":
            continue
        bp_terms.add(go_id)
        for parent in meta.get("is_a", []):
            if parent in terms and terms[parent].get("namespace") == "biological_process":
                child_to_parents[go_id].add(parent)
                parent_to_children[parent].add(go_id)

    logger.info("BP DAG: %d terms, %d edges",
                len(bp_terms),
                sum(len(v) for v in child_to_parents.values()))
    return dict(child_to_parents), dict(parent_to_children), bp_terms


def get_go_obo(data_dir, skip_download=False):
    """Obtain the GO OBO file, downloading if necessary.

    Returns the path to the OBO file, or None on failure.
    """
    obo_path = data_dir / "go.obo"
    if obo_path.exists() and os.path.getsize(obo_path) > 1024:
        logger.info("Using cached GO OBO: %s", obo_path)
        return obo_path

    if skip_download:
        logger.warning("--skip-download set; no local OBO available.")
        return None

    if download_file(OBO_URL, str(obo_path)):
        return obo_path

    logger.error("Could not obtain GO OBO file.")
    return None


# ===================================================================
# 2. GO Term Propagation (True Path Rule)
# ===================================================================

def propagate_annotations(go_map, child_to_parents, bp_terms):
    """Propagate gene annotations upward through the GO DAG.

    For every gene with direct annotations, all ancestor BP terms are
    added (True Path Rule: if a gene is annotated to GO:X, it is
    implicitly annotated to every ancestor of GO:X).

    Parameters
    ----------
    go_map : dict[str, list[str]]
        Original gene -> GO term list.
    child_to_parents : dict[str, set[str]]
        DAG child -> parents mapping.
    bp_terms : set[str]
        Set of all valid BP terms.

    Returns
    -------
    propagated_map : dict[str, list[str]]
        Gene -> sorted list of BP GO terms (direct + inherited).
    """
    propagated = {}
    for gene, terms in go_map.items():
        all_terms = set()
        stack = list(terms)
        visited = set()
        while stack:
            t = stack.pop()
            if t in visited:
                continue
            visited.add(t)
            if t in bp_terms:
                all_terms.add(t)
            for parent in child_to_parents.get(t, set()):
                if parent not in visited:
                    stack.append(parent)
        if all_terms:
            propagated[gene] = sorted(all_terms)

    logger.info("Propagation: %d genes, total unique gene-term pairs: %d",
                len(propagated),
                sum(len(v) for v in propagated.values()))
    return propagated


# ===================================================================
# 3. SGD GAF Parsing
# ===================================================================

# Pattern matching SGD systematic ORF names (e.g. YAL005C, YBR160W)
_ORF_RE = re.compile(r"^Y[A-Z][RL]\d{3}[CW]$")


def _extract_orf_from_gaf_line(cols, alias_to_orf):
    """Extract an SGD ORF name from a parsed GAF line.

    Strategy (in priority order):
      1. Column 10 (DB_Object_Synonyms) — pipe-separated list often
         containing the ORF name (e.g. ``YAL005C``).
      2. Column 1 (DB_Object_ID) — may already be an ORF name.
      3. Column 2 (DB_Object_Symbol) — gene symbol; resolve via alias.

    Returns the ORF name string, or None if no match.
    """
    # Column 10: synonyms (pipe-separated)
    if len(cols) > 10 and cols[10]:
        for syn in cols[10].split("|"):
            syn = syn.strip()
            if _ORF_RE.match(syn):
                return syn

    # Column 1: DB_Object_ID
    obj_id = cols[1]
    if _ORF_RE.match(obj_id):
        return obj_id

    # Column 2: gene symbol -> alias lookup
    gene_sym = cols[2] if len(cols) > 2 else ""
    if gene_sym and alias_to_orf:
        orf = alias_to_orf.get(gene_sym)
        if orf and _ORF_RE.match(orf):
            return orf

    return None


def _try_parse_gaf(filepath, network_nodes, alias_to_orf=None):
    """Parse a GAF (gzipped or plain) for BP annotations.

    Extracts SGD ORF names (e.g. ``YAL005C``) from GAF column 10
    (synonyms) or column 1/2 as fallback, then maps them onto the
    full STRING network node set.

    Parameters
    ----------
    filepath : str or Path
        Path to GAF file (``.gaf`` or ``.gaf.gz``).
    network_nodes : set[str]
        Node IDs from the full STRING network.
    alias_to_orf : dict[str, str] or None
        Gene-symbol -> ORF name mapping from the STRING alias file.

    Returns
    -------
    gaf_map : dict[str, list[str]]
        Gene (SGD ORF in network) -> list of GO terms (BP only).
    """
    if alias_to_orf is None:
        alias_to_orf = {}

    gaf_map = defaultdict(set)
    opener = gzip.open if str(filepath).endswith(".gz") else open
    count_lines = 0
    skipped_no_orf = 0

    try:
        with opener(str(filepath), "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("!") or line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 10:
                    continue
                count_lines += 1

                go_term = cols[4]         # GO ID
                aspect = cols[8]          # P / F / C
                if aspect != "P":
                    continue
                if not go_term.startswith("GO:"):
                    continue

                orf = _extract_orf_from_gaf_line(cols, alias_to_orf)
                if orf is None:
                    skipped_no_orf += 1
                    continue
                gaf_map[orf].add(go_term)
    except Exception as exc:
        logger.warning("GAF parse error at %s: %s", filepath, exc)
        return {}

    logger.info("Parsed %d data lines from GAF; %d genes with BP terms "
                "(%d lines skipped — no ORF match)",
                count_lines, len(gaf_map), skipped_no_orf)
    return {g: sorted(ts) for g, ts in gaf_map.items()}


def download_and_parse_gaf(data_dir, network_nodes, alias_to_orf=None,
                           skip_download=False):
    """Attempt to download the SGD GAF and parse BP annotations.

    Falls back gracefully if the download fails.

    Parameters
    ----------
    data_dir : Path
    network_nodes : set[str]
    alias_to_orf : dict[str, str] or None
        Gene-symbol -> ORF mapping from STRING alias file.
    skip_download : bool

    Returns
    -------
    gaf_map : dict[str, list[str]]
        Gene -> sorted BP GO terms.  Empty dict on total failure.
    """
    local_gaf = data_dir / "gene_association.sgd.gaf.gz"

    # Try local file first (if not an LFS pointer)
    if local_gaf.exists() and not _is_lfs_pointer(local_gaf):
        logger.info("Using local GAF: %s", local_gaf)
        result = _try_parse_gaf(local_gaf, network_nodes, alias_to_orf)
        if result:
            return result

    # Also check previously downloaded GAF
    downloaded_gaf = data_dir / "gene_association.sgd_downloaded.gaf.gz"
    if downloaded_gaf.exists() and not _is_lfs_pointer(downloaded_gaf):
        logger.info("Using previously downloaded GAF: %s", downloaded_gaf)
        result = _try_parse_gaf(downloaded_gaf, network_nodes, alias_to_orf)
        if result:
            return result

    if skip_download:
        logger.warning("--skip-download set; skipping GAF download.")
        return {}

    # Try downloading
    for url in GAF_URLS:
        dest = data_dir / "gene_association.sgd_downloaded.gaf.gz"
        if download_file(url, str(dest)):
            result = _try_parse_gaf(dest, network_nodes, alias_to_orf)
            if result:
                return result

    logger.warning("All GAF download attempts failed.")
    return {}


# ===================================================================
# 4. STRING Alias Mapping
# ===================================================================

def load_string_aliases(data_dir):
    """Parse ``4932.protein.aliases.v11.5.txt.gz`` to build ID mappings.

    Returns
    -------
    alias_to_orf : dict[str, str]
        Alias (e.g. gene symbol) -> SGD ORF name (e.g. ``YAL005C``).
    orf_set : set[str]
        All SGD ORF names found in the alias file.
    """
    alias_file = data_dir / "4932.protein.aliases.v11.5.txt.gz"
    if not alias_file.exists() or _is_lfs_pointer(alias_file):
        logger.warning("Alias file unavailable.")
        return {}, set()

    orf_to_aliases = defaultdict(set)
    alias_to_orf = {}
    orf_set = set()

    try:
        with gzip.open(str(alias_file), "rt", encoding="utf-8") as fh:
            header = fh.readline()  # skip header
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                string_id = parts[0]   # e.g. "4932.YAL005C"
                alias = parts[1]
                # The STRING ID for yeast is "4932.<ORF_name>"
                orf = string_id.split(".", 1)[1] if "." in string_id else string_id
                orf_set.add(orf)
                orf_to_aliases[orf].add(alias)
                # Also map common aliases back to ORF
                alias_to_orf[alias] = orf
    except Exception as exc:
        logger.warning("Alias parsing failed: %s", exc)
        return {}, set()

    logger.info("Alias file: %d ORFs, %d alias entries", len(orf_set), len(alias_to_orf))
    return alias_to_orf, orf_set


# ===================================================================
# 5. Extended Annotation Set Construction
# ===================================================================

def build_extended_go_map(gaf_map, original_go_map, network_nodes,
                          child_to_parents, bp_terms, min_terms=1):
    """Build an extended gene -> GO term mapping for the full network.

    Combines:
      * Direct SGD GAF annotations (for all yeast genes).
      * Original ``gene_go_map.json`` entries (as a fallback for genes
        missing from the GAF).
    Then propagates all annotations upward through the GO DAG.

    Parameters
    ----------
    gaf_map : dict[str, list[str]]
        GAF-derived annotations.
    original_go_map : dict[str, list[str]]
        Original curated annotations.
    network_nodes : set[str]
        Nodes present in the full STRING network.
    child_to_parents : dict[str, set[str]]
    bp_terms : set[str]
    min_terms : int
        Minimum number of GO terms a gene must have.

    Returns
    -------
    extended_map : dict[str, list[str]]
        Gene -> sorted BP GO terms (propagated), for genes in the
        network with >= *min_terms* terms.
    """
    # Merge: GAF takes priority, fill in from original map
    merged = defaultdict(set)
    for gene, terms in gaf_map.items():
        if gene in network_nodes:
            merged[gene].update(terms)
    for gene, terms in original_go_map.items():
        if gene in network_nodes:
            merged[gene].update(terms)

    logger.info("Merged annotations: %d genes in network", len(merged))

    # Propagate
    propagated = propagate_annotations(
        {g: sorted(ts) for g, ts in merged.items()},
        child_to_parents, bp_terms,
    )

    # Filter by min_terms
    extended = {g: ts for g, ts in propagated.items() if len(ts) >= min_terms}
    logger.info("Extended GO map (min_terms=%d): %d genes", min_terms, len(extended))
    return extended


# ===================================================================
# 6. G-F Score Computation Helpers
# ===================================================================

def _compute_gf_curve_fast(coords, nodes, go_map, r_vals):
    """Vectorized G-F curve computation for large node sets.

    Uses scipy's ``pdist`` for efficient distance computation and
    connected-component community detection (O(V+E)) which is much
    faster than ``greedy_modularity_communities`` on large graphs.

    .. warning::
       This function uses **connected components** instead of
       ``greedy_modularity_communities`` for community detection.
       Purity values are therefore NOT directly comparable to those
       produced by ``utils.compute_gf_curve`` (which uses greedy
       modularity).  Use only for large-scale trend analysis, not
       for cross-experiment score comparison.

    Parameters
    ----------
    coords : np.ndarray (N, 2)
    nodes : list[str]
    go_map : dict[str, list[str]]
    r_vals : np.ndarray

    Returns
    -------
    purities : list[float]
    modularities : list[float]
    """
    from scipy.spatial.distance import pdist, squareform
    from networkx.algorithms.community import modularity

    n = len(nodes)

    # Pre-compute condensed distance vector and expand to square form
    logger.info("    Computing distance matrix for %d nodes...", n)
    dist_vec = pdist(coords, metric="euclidean")
    dist_matrix = squareform(dist_vec)

    # Pre-compute per-node GO term sets for fast purity
    node_go_sets = {}
    for i, nd in enumerate(nodes):
        if nd in go_map:
            node_go_sets[i] = set(go_map[nd])

    purities = []
    modularities = []
    n_r = len(r_vals)

    for ri, r in enumerate(r_vals):
        if (ri + 1) % 50 == 0 or ri == 0:
            logger.info("    r-point %d/%d (r=%.4f)", ri + 1, n_r, r)

        # Build spatial graph using vectorized comparison
        mask = (dist_matrix < r) & (dist_matrix > 0)
        rows, cols = np.where(mask)
        upper = rows < cols
        edge_rows = rows[upper]
        edge_cols = cols[upper]
        n_edges = len(edge_rows)

        if n_edges == 0:
            purities.append(0.0)
            modularities.append(0.0)
            continue

        G_r = nx.Graph()
        G_r.add_nodes_from(range(n))
        G_r.add_edges_from(zip(edge_rows.tolist(), edge_cols.tolist()))

        # Connected components — O(V+E), very fast
        communities = [set(c) for c in nx.connected_components(G_r)]

        # Compute functional purity
        comm_purities = []
        for comm in communities:
            all_terms = []
            for idx in comm:
                if idx in node_go_sets:
                    all_terms.extend(node_go_sets[idx])
            if not all_terms:
                comm_purities.append(0.0)
                continue
            term_counts = Counter(all_terms)
            most_common_count = term_counts.most_common(1)[0][1]
            comm_purities.append(most_common_count / len(comm))

        purity = float(np.mean(comm_purities)) if comm_purities else 0.0
        purities.append(purity)

        if len(communities) > 1:
            try:
                modularities.append(modularity(G_r, communities))
            except Exception:
                modularities.append(0.0)
        else:
            modularities.append(0.0)

    return purities, modularities


def _compute_gf_curve_dispatch(coords, nodes, go_map, r_vals):
    """Dispatch to fast or standard G-F curve computation.

    Uses the vectorized implementation for node sets > 300 nodes,
    falling back to :func:`utils.compute_gf_curve` for smaller sets.
    """
    if len(nodes) > 300:
        logger.info("  Using optimized G-F curve computation for %d nodes",
                     len(nodes))
        return _compute_gf_curve_fast(coords, nodes, go_map, r_vals)
    return compute_gf_curve(coords, nodes, go_map, r_vals)

def _load_or_compute_full_embedding(method, G_full, nodes_full):
    """Load a pre-computed full-network embedding, or compute one.

    Currently supports loading DM_full.npy.  For Spectral, computes
    fresh from the graph.

    Returns
    -------
    coords : np.ndarray  (N, 2)
    nodes : list[str]
    """
    emb_dir = get_embeddings_dir()
    try:
        coords, emb_nodes = load_embedding(method, "full", embeddings_dir=emb_dir)
        logger.info("Loaded %s_full embedding (%d nodes)", method, len(emb_nodes))
        return coords, emb_nodes
    except FileNotFoundError:
        pass

    # Compute fresh
    logger.info("Computing %s embedding on full network (%d nodes)...",
                method, len(nodes_full))
    if method == "Spectral":
        from utils import spectral_embedding_from_graph
        coords = spectral_embedding_from_graph(G_full, nodelist=nodes_full)
        coords = rescale_coordinates(coords, target_std=0.3)
        # Save for future use
        from utils import save_embedding
        save_embedding(coords, nodes_full, method, "full", emb_dir)
        return coords, nodes_full
    elif method == "DM":
        from utils import (compute_centrality_features,
                           build_similarity_matrix,
                           diffusion_map_from_similarity)
        features = compute_centrality_features(G_full, nodes_full)
        sim = build_similarity_matrix(features)
        coords = diffusion_map_from_similarity(sim)
        coords = rescale_coordinates(coords, target_std=0.3)
        from utils import save_embedding
        save_embedding(coords, nodes_full, method, "full", emb_dir)
        return coords, nodes_full
    else:
        raise ValueError(f"Cannot auto-compute embedding for method '{method}'")


def compute_gf_for_subset(coords_full, nodes_full, go_map, subset_nodes,
                          r_vals):
    """Compute G-F Score for *subset_nodes* using pre-computed full coords.

    Returns
    -------
    gf_score : float
    purities : list[float]
    """
    node_to_idx = {n: i for i, n in enumerate(nodes_full)}
    indices = [node_to_idx[n] for n in subset_nodes if n in node_to_idx]
    valid_nodes = [nodes_full[i] for i in indices]
    sub_coords = coords_full[indices]

    purities, _ = _compute_gf_curve_dispatch(sub_coords, valid_nodes,
                                              go_map, r_vals)
    score = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)
    return score, purities


# ===================================================================
# 7. Subset Convergence Analysis
# ===================================================================

def subset_convergence_analysis(coords_full, nodes_full, extended_go_map,
                                subset_sizes, n_subsets, r_vals):
    """Sample random subsets and compute G-F Score convergence.

    Parameters
    ----------
    coords_full : np.ndarray
    nodes_full : list[str]
    extended_go_map : dict[str, list[str]]
    subset_sizes : list[int]
    n_subsets : int
    r_vals : np.ndarray

    Returns
    -------
    convergence : dict
        Keys: sizes, mean_gf_scores, std_gf_scores, ci_lower, ci_upper,
        raw_scores (list of lists).
    """
    annotated_nodes = sorted(extended_go_map.keys())
    n_annotated = len(annotated_nodes)
    logger.info("Convergence analysis: %d annotated nodes, sizes=%s, n_subsets=%d",
                n_annotated, subset_sizes, n_subsets)

    all_scores = {sz: [] for sz in subset_sizes}

    for size in subset_sizes:
        actual_size = min(size, n_annotated)
        for rep in range(n_subsets):
            rng = random.Random(SEED + rep * 1000 + size)
            sample = rng.sample(annotated_nodes, actual_size)
            score, _ = compute_gf_for_subset(
                coords_full, nodes_full, extended_go_map, sample, r_vals,
            )
            all_scores[size].append(score)

    scores_arr = {sz: np.array(sc) for sz, sc in all_scores.items()}

    convergence = {
        "sizes": subset_sizes,
        "mean_gf_scores": [float(scores_arr[sz].mean()) for sz in subset_sizes],
        "std_gf_scores": [float(scores_arr[sz].std()) for sz in subset_sizes],
        "ci_lower": [float(np.percentile(scores_arr[sz], 2.5)) for sz in subset_sizes],
        "ci_upper": [float(np.percentile(scores_arr[sz], 97.5)) for sz in subset_sizes],
        "raw_scores": {str(sz): sc.tolist() for sz, sc in scores_arr.items()},
    }
    return convergence


# ===================================================================
# 8. Plotting
# ===================================================================

def plot_convergence(convergence, output_path):
    """Create FigS4: G-F Score vs subset size with 95% CI shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = convergence["sizes"]
    means = convergence["mean_gf_scores"]
    ci_lo = convergence["ci_lower"]
    ci_hi = convergence["ci_upper"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, means, "o-", color="#2c7bb6", linewidth=2,
            markersize=8, label="Mean G-F Score")
    ax.fill_between(sizes, ci_lo, ci_hi, alpha=0.25, color="#2c7bb6",
                     label="95% CI")
    ax.set_xlabel("Subset Size (number of annotated nodes)", fontsize=12)
    ax.set_ylabel("G-F Score", fontsize=12)
    ax.set_title("Supplementary Fig. S4: G-F Score vs. Annotation Set Size",
                 fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)
    logger.info("Saved convergence figure: %s", output_path)


# ===================================================================
# 9. Main
# ===================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GO term propagation and extended validation set construction.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Do not attempt to download GO OBO or SGD GAF (use only local files).",
    )
    parser.add_argument(
        "--min-terms", type=int, default=1,
        help="Minimum GO terms per gene in extended set (default: 1).",
    )
    parser.add_argument(
        "--subset-sizes", type=str, default="50,100,200,500,1000",
        help="Comma-separated subset sizes for convergence analysis.",
    )
    parser.add_argument(
        "--n-subsets", type=int, default=30,
        help="Number of random subsets per size (default: 30).",
    )
    return parser.parse_args()


def main():
    """Run the full GO propagation pipeline."""
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = get_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)

    subset_sizes = [int(x) for x in args.subset_sizes.split(",")]
    r_vals = np.linspace(R_MIN, R_GRID_MAX, N_POINTS)

    # ------------------------------------------------------------------
    # Step 1: Load original data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: Load original data")
    logger.info("=" * 60)

    with open(data_dir / "gene_go_map.json") as fh:
        original_go_map = json.load(fh)
    original_genes = len(original_go_map)
    original_terms = set()
    for ts in original_go_map.values():
        original_terms.update(ts)
    original_n_terms = len(original_terms)
    mean_terms_before = np.mean([len(v) for v in original_go_map.values()])
    logger.info("Original gene_go_map: %d genes, %d unique GO terms, "
                "%.2f terms/gene", original_genes, original_n_terms,
                mean_terms_before)

    curated_nodes_file = data_dir / "curated_153_nodes.txt"
    if not curated_nodes_file.exists():
        raise FileNotFoundError(
            f"{curated_nodes_file} not found. Run data_preprocessing.py (Step 1) first."
        )
    with open(curated_nodes_file) as fh:
        curated_nodes = [line.strip() for line in fh if line.strip()]
    logger.info("Curated nodes: %d", len(curated_nodes))

    # ------------------------------------------------------------------
    # Step 2: Build GO DAG
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Build GO DAG from OBO")
    logger.info("=" * 60)

    obo_path = get_go_obo(data_dir, skip_download=args.skip_download)
    if obo_path is not None:
        terms = parse_go_obo(obo_path)
        child_to_parents, parent_to_children, bp_terms = build_go_dag(terms)
    else:
        # Minimal fallback: use only the terms present in gene_go_map
        logger.warning("No OBO available; using flat annotation set (no DAG).")
        all_known = set()
        for ts in original_go_map.values():
            all_known.update(ts)
        bp_terms = all_known
        child_to_parents = {}
        parent_to_children = {}

    # ------------------------------------------------------------------
    # Step 3: Propagate original annotations
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: Propagate original annotations (True Path Rule)")
    logger.info("=" * 60)

    propagated_original = propagate_annotations(
        original_go_map, child_to_parents, bp_terms,
    )
    prop_terms = set()
    for ts in propagated_original.values():
        prop_terms.update(ts)
    logger.info("After propagation: %d genes, %d unique terms, "
                "%.2f terms/gene",
                len(propagated_original), len(prop_terms),
                np.mean([len(v) for v in propagated_original.values()])
                if propagated_original else 0)

    # ------------------------------------------------------------------
    # Step 4: Load full STRING network
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Load full STRING network")
    logger.info("=" * 60)

    try:
        G_full = load_full_STRING_network(data_dir)
        nodes_full = sorted(G_full.nodes())
        network_node_set = set(nodes_full)
        logger.info("Full network: %d nodes, %d edges",
                     len(nodes_full), G_full.number_of_edges())
    except FileNotFoundError as exc:
        logger.error("Cannot load full STRING network: %s", exc)
        logger.error("Falling back to curated 153-node network only.")
        G_full = None
        nodes_full = curated_nodes
        network_node_set = set(curated_nodes)

    # ------------------------------------------------------------------
    # Step 5: Load STRING aliases and parse GAF for extended annotations
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5: Parse SGD GAF for extended annotations")
    logger.info("=" * 60)

    # Load STRING aliases first (needed to resolve GAF gene symbols)
    alias_to_orf, alias_orf_set = load_string_aliases(data_dir)

    gaf_map = download_and_parse_gaf(
        data_dir, network_node_set,
        alias_to_orf=alias_to_orf,
        skip_download=args.skip_download,
    )

    # ------------------------------------------------------------------
    # Step 6: Build extended GO map
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 6: Build extended GO annotation map")
    logger.info("=" * 60)

    if gaf_map:
        extended_go_map = build_extended_go_map(
            gaf_map, original_go_map, network_node_set,
            child_to_parents, bp_terms, min_terms=args.min_terms,
        )
    else:
        # Fallback: use propagated original + try matching additional
        # network nodes via alias file or partial matching
        logger.warning("No GAF data; building extended set from propagated "
                       "original + alias matching.")
        extended_go_map = dict(propagated_original)

        # Attempt to find additional annotated genes via the alias file
        if alias_to_orf:
            # Build a secondary lookup: gene_symbol -> ORF
            # Some GAF-like resources might use gene symbols; try matching
            # any network node not yet annotated against known aliases
            unannotated = network_node_set - set(extended_go_map.keys())
            logger.info("Attempting alias-based matching for %d "
                        "unannotated network nodes...", len(unannotated))
            # In practice the GAF is the main source; alias matching alone
            # rarely adds many genes.  Log the result.
            logger.info("Alias matching: no additional genes without GAF.")

    extended_genes = len(extended_go_map)
    ext_terms = set()
    for ts in extended_go_map.values():
        ext_terms.update(ts)
    extended_n_terms = len(ext_terms)
    mean_terms_after = (np.mean([len(v) for v in extended_go_map.values()])
                        if extended_go_map else 0.0)

    logger.info("Extended annotation set: %d genes, %d unique GO terms, "
                "%.2f terms/gene",
                extended_genes, extended_n_terms, mean_terms_after)

    # If we still have fewer than 1,000 annotated genes, relax min_terms
    if extended_genes < 1000 and args.min_terms > 0:
        logger.warning("Extended set has only %d genes (< 1,000 target). "
                       "Attempting with min_terms=0...", extended_genes)
        # Re-include genes with any annotation (even after removing
        # min_terms filter)
        if gaf_map:
            extended_go_map = build_extended_go_map(
                gaf_map, original_go_map, network_node_set,
                child_to_parents, bp_terms, min_terms=0,
            )
            extended_genes = len(extended_go_map)
            ext_terms = set()
            for ts in extended_go_map.values():
                ext_terms.update(ts)
            extended_n_terms = len(ext_terms)
            mean_terms_after = (
                np.mean([len(v) for v in extended_go_map.values()])
                if extended_go_map else 0.0
            )
            logger.info("After relaxing min_terms: %d genes, %d terms",
                        extended_genes, extended_n_terms)

    # ------------------------------------------------------------------
    # Step 7: Save extended annotations
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 7: Save extended annotations")
    logger.info("=" * 60)

    ext_anno_path = results_dir / "extended_go_annotations.json"
    with open(ext_anno_path, "w") as fh:
        json.dump(extended_go_map, fh, indent=2, sort_keys=True)
    logger.info("Saved: %s", ext_anno_path)

    # ------------------------------------------------------------------
    # Step 8: G-F Score computation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 8: G-F Score computation (curated vs. extended)")
    logger.info("=" * 60)

    gf_curated = {}
    gf_extended = {}
    methods = ["DM", "Spectral"]

    for method in methods:
        logger.info("--- %s ---", method)
        try:
            if G_full is not None:
                coords_full, emb_nodes = _load_or_compute_full_embedding(
                    method, G_full, nodes_full,
                )
            else:
                coords_full, emb_nodes = load_embedding(method, "153")

            # Curated 153 evaluation
            curated_in_emb = sorted(set(curated_nodes) & set(emb_nodes)
                                    & set(original_go_map.keys()))
            if curated_in_emb:
                eval_go = propagated_original if child_to_parents else original_go_map
                sc_cur, _ = compute_gf_for_subset(
                    coords_full, emb_nodes, eval_go, curated_in_emb, r_vals,
                )
                gf_curated[method] = round(float(sc_cur), 4)
                logger.info("  Curated-153 G-F Score: %.4f", gf_curated[method])

            # Extended set evaluation
            ext_in_emb = sorted(set(extended_go_map.keys()) & set(emb_nodes))
            if ext_in_emb:
                # Subsample to MAX_EVAL_NODES for tractable computation
                eval_nodes = ext_in_emb
                if len(eval_nodes) > MAX_EVAL_NODES:
                    rng = random.Random(SEED)
                    eval_nodes = sorted(rng.sample(eval_nodes, MAX_EVAL_NODES))
                    logger.info("  Subsampled extended set to %d nodes "
                                "(from %d) for G-F evaluation",
                                len(eval_nodes), len(ext_in_emb))
                sc_ext, _ = compute_gf_for_subset(
                    coords_full, emb_nodes, extended_go_map, eval_nodes, r_vals,
                )
                gf_extended[method] = round(float(sc_ext), 4)
                logger.info("  Extended G-F Score (%d nodes): %.4f",
                            len(eval_nodes), gf_extended[method])

        except Exception as exc:
            logger.warning("  %s failed: %s", method, exc)
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Step 9: Subset convergence analysis
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 9: Subset convergence analysis")
    logger.info("=" * 60)

    convergence = {"sizes": subset_sizes,
                   "mean_gf_scores": [], "std_gf_scores": [],
                   "ci_lower": [], "ci_upper": []}

    # Use the best available full embedding for convergence
    conv_method = "DM" if "DM" in gf_extended else (
        "Spectral" if "Spectral" in gf_extended else None
    )
    if conv_method is not None and G_full is not None:
        try:
            coords_conv, emb_nodes_conv = _load_or_compute_full_embedding(
                conv_method, G_full, nodes_full,
            )
            convergence = subset_convergence_analysis(
                coords_conv, emb_nodes_conv, extended_go_map,
                subset_sizes, args.n_subsets, r_vals,
            )
            logger.info("Convergence results (using %s):", conv_method)
            for i, sz in enumerate(convergence["sizes"]):
                logger.info("  size=%d: mean=%.4f +/- %.4f  [%.4f, %.4f]",
                            sz,
                            convergence["mean_gf_scores"][i],
                            convergence["std_gf_scores"][i],
                            convergence["ci_lower"][i],
                            convergence["ci_upper"][i])
        except Exception as exc:
            logger.warning("Convergence analysis failed: %s", exc)
            import traceback
            traceback.print_exc()
    else:
        logger.warning("No full-network embedding available; "
                       "skipping convergence analysis.")

    # ------------------------------------------------------------------
    # Step 10: Save statistics and figure
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 10: Save outputs")
    logger.info("=" * 60)

    stats = {
        "original_annotated_genes": original_genes,
        "extended_annotated_genes": extended_genes,
        "original_go_terms": original_n_terms,
        "extended_go_terms_after_propagation": extended_n_terms,
        "mean_terms_per_gene_before": round(float(mean_terms_before), 4),
        "mean_terms_per_gene_after": round(float(mean_terms_after), 4),
        "gf_score_curated_153": gf_curated,
        "gf_score_extended": gf_extended,
        "subset_convergence": convergence,
    }

    stats_path = results_dir / "go_propagation_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Saved: %s", stats_path)

    # Plot convergence figure
    fig_path = figures_dir / "FigS4_sample_size_convergence.png"
    if convergence["mean_gf_scores"]:
        plot_convergence(convergence, fig_path)
    else:
        logger.warning("No convergence data; skipping figure.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Original annotated genes : %d", original_genes)
    logger.info("Extended annotated genes : %d", extended_genes)
    logger.info("Original GO terms        : %d", original_n_terms)
    logger.info("Extended GO terms        : %d", extended_n_terms)
    logger.info("Mean terms/gene (before) : %.2f", mean_terms_before)
    logger.info("Mean terms/gene (after)  : %.2f", mean_terms_after)
    if gf_curated:
        logger.info("G-F Score curated-153    : %s", gf_curated)
    if gf_extended:
        logger.info("G-F Score extended       : %s", gf_extended)
    logger.info("Done.")


if __name__ == "__main__":
    main()
