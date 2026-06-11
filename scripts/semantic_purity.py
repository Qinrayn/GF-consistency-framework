#!/usr/bin/env python3
"""
semantic_purity.py — IC-weighted and semantic-similarity-based purity variants.

Addresses two known limitations of the count-based G-F purity metric:

1. **DAG expansion inflation**: True Path Rule propagation adds all ancestor
   GO terms (~3.8 → ~28.9 terms/gene), causing purity → 1.0 because general
   ancestor terms dilute the denominator.  IC-weighted purity down-weights
   non-specific terms, restoring discriminative power after propagation.

2. **Exact-match only**: Standard purity treats all GO terms as independent
   tokens.  Semantic purity uses Resnik's Most Informative Common Ancestor
   (MICA) to capture functional coherence based on GO DAG topology —
   communities with *semantically related* (but non-identical) terms still
   receive high purity.

Three purity variants:

  * **Standard** (baseline):
      purity(C) = max_t count(t, C) / |T(C)|

  * **IC-weighted**:
      purity_IC(C) = max_t [count(t,C) · IC(t)] / Σ_t [count(t,C) · IC(t)]

  * **Semantic** (Resnik-based):
      purity_sem(C) = 2/(n(n−1)) · Σ_{i<j} sim_norm(t_i, t_j)
    where sim_norm is Resnik MICA similarity normalised to [0, 1].

Dependencies: ``scripts.go_propagation`` for GO OBO parsing and DAG
construction.  Requires ``data/go.obo`` (OBO format 1.2) for DAG-based
computations; gracefully falls back to corpus-only IC if OBO is missing.

Author: Yuhan Zhang (Qinrayn)
"""

from __future__ import annotations

import sys
import json
import math
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, Set, Tuple, Optional, List

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BP_ROOT = "GO:0008150"  # biological_process root
EPS = 1e-12            # numerical guard for log / division


# ===================================================================
# 1.  Information Content (IC)
# ===================================================================

def compute_term_frequencies(
    go_map: Dict[str, List[str]],
) -> Tuple[Counter, int]:
    """Count how many genes each GO term annotates.

    Parameters
    ----------
    go_map : dict
        gene_id → list[go_term_id]  (direct *or* propagated annotations).

    Returns
    -------
    (term_gene_count, n_genes)
        Counter mapping go_term → number of annotated genes,
        and total number of genes in the corpus.
    """
    term_gene_count: Counter = Counter()
    for _gene, terms in go_map.items():
        seen = set(terms)           # guard against duplicate terms per gene
        for t in seen:
            term_gene_count[t] += 1
    return term_gene_count, len(go_map)


def compute_ic(
    term_gene_count: Counter,
    n_genes: int,
) -> Dict[str, float]:
    """Corpus-based Information Content for every observed GO term.

    .. math::

        IC(t) = -\\log\\!\\left(\\frac{\\text{genes}(t)}{N}\\right)

    Parameters
    ----------
    term_gene_count : Counter
        go_term → gene count  (from :func:`compute_term_frequencies`).
    n_genes : int
        Total number of genes in the annotation corpus.

    Returns
    -------
    dict mapping go_term → IC value (float ≥ 0).
    Terms annotating all genes have IC ≈ 0; rare terms have high IC.
    """
    ic: Dict[str, float] = {}
    for term, count in term_gene_count.items():
        p = count / max(n_genes, 1)
        ic[term] = -math.log(p + EPS)
    return ic


# ===================================================================
# 2.  Ancestor sets & Resnik MICA similarity
# ===================================================================

def _precompute_ancestors(
    child_to_parents: Dict[str, Set[str]],
    bp_terms: Set[str],
    subset_terms: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """Compute the full ancestor set for BP terms (including self).

    Traverses *child → parents* edges upward until the root.
    If *subset_terms* is given, only compute ancestors for those terms
    (their ancestors may still include any reachable BP term).

    Uses an iterative stack to avoid recursion-depth issues on deep DAGs.

    Returns
    -------
    dict mapping go_term → set of ancestors (term itself included).
    """
    ancestors: Dict[str, Set[str]] = {}
    targets = subset_terms if subset_terms is not None else bp_terms

    for start in targets:
        if start in ancestors:
            continue
        # iterative DFS upward
        stack = [start]
        order: list = []
        visited: Set[str] = set()
        while stack:
            term = stack.pop()
            if term in visited:
                continue
            visited.add(term)
            order.append(term)
            for parent in child_to_parents.get(term, set()):
                if parent in bp_terms and parent not in visited:
                    stack.append(parent)

        # build ancestor sets bottom-up (reverse topological order)
        for term in reversed(order):
            if term in ancestors:
                continue
            anc = {term}
            for parent in child_to_parents.get(term, set()):
                if parent in bp_terms and parent in ancestors:
                    anc |= ancestors[parent]
            ancestors[term] = anc

    return ancestors


def build_similarity_index(
    child_to_parents: Dict[str, Set[str]],
    bp_terms: Set[str],
    ic: Dict[str, float],
    subset_terms: Optional[Set[str]] = None,
) -> Tuple[Dict[Tuple[str, str], float], float]:
    """Pre-compute normalised Resnik similarity for GO term pairs.

    Resnik similarity between terms *a* and *b*:

    .. math::

        \\text{sim}(a, b) = \\max_{c \\in \\text{Anc}(a) \\cap \\text{Anc}(b)} IC(c)

    Normalised to [0, 1] by dividing by ``max_ic`` (the highest IC in the
    ontology).

    Parameters
    ----------
    child_to_parents : dict
        go_term → set of parent terms  (from ``build_go_dag``).
    bp_terms : set
        All valid biological_process term IDs.
    ic : dict
        Information content per term (from :func:`compute_ic`).
    subset_terms : set, optional
        If given, only compute pairwise similarities for these terms.
        Ancestor traversal still covers the full DAG (needed for MICA).
        **Strongly recommended** when bp_terms is large (>1 000).

    Returns
    -------
    (sim_cache, max_ic)
        sim_cache maps (term_a, term_b) → normalised similarity ∈ [0, 1].
        max_ic is the maximum IC across all BP terms (for denormalisation
        if needed).
    """
    # only compute ancestors for subset terms (their ancestor sets still
    # include all reachable BP terms, so MICA is exact)
    ancestors = _precompute_ancestors(child_to_parents, bp_terms, subset_terms)

    max_ic = max((ic.get(t, 0.0) for t in bp_terms), default=1.0)
    if max_ic < EPS:
        max_ic = 1.0  # safety guard

    sim_cache: Dict[Tuple[str, str], float] = {}
    pair_terms = sorted(subset_terms) if subset_terms else sorted(bp_terms)
    n = len(pair_terms)

    for i in range(n):
        ti = pair_terms[i]
        anc_i = ancestors.get(ti, {ti})
        ic_i = ic.get(ti, 0.0)
        # self-similarity
        sim_cache[(ti, ti)] = min(ic_i / max_ic, 1.0)
        for j in range(i + 1, n):
            tj = pair_terms[j]
            anc_j = ancestors.get(tj, {tj})
            # MICA: max IC among common ancestors
            common = anc_i & anc_j
            mica_ic = max((ic.get(c, 0.0) for c in common), default=0.0)
            norm_sim = min(mica_ic / max_ic, 1.0)
            sim_cache[(ti, tj)] = norm_sim
            sim_cache[(tj, ti)] = norm_sim

    logger.info(
        "Similarity index built: %d terms, %d pairs, max_ic=%.3f",
        n, len(sim_cache), max_ic,
    )
    return sim_cache, max_ic


# ===================================================================
# 3.  Community purity variants
# ===================================================================

def community_purity_standard(
    comm_nodes: List[str],
    go_map: Dict[str, List[str]],
) -> float:
    """Standard count-based purity (same as ``utils._community_purity``).

    purity(C) = max_t count(t, C) / |T(C)|
    """
    all_terms: list = []
    for node in comm_nodes:
        all_terms.extend(go_map.get(node, []))
    if not all_terms:
        return 0.0
    counts = Counter(all_terms)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(all_terms)


def community_purity_ic_weighted(
    comm_nodes: List[str],
    go_map: Dict[str, List[str]],
    ic: Dict[str, float],
) -> float:
    """IC-weighted purity.

    purity_IC(C) = max_t [count(t) · IC(t)] / Σ_t [count(t) · IC(t)]

    Down-weights general ancestor terms (low IC) so that DAG-propagated
    annotations do not inflate purity.
    """
    all_terms: list = []
    for node in comm_nodes:
        all_terms.extend(go_map.get(node, []))
    if not all_terms:
        return 0.0

    counts = Counter(all_terms)
    weighted_sum = 0.0
    max_weighted = 0.0
    for term, cnt in counts.items():
        w = cnt * ic.get(term, 0.0)
        weighted_sum += w
        if w > max_weighted:
            max_weighted = w

    if weighted_sum < EPS:
        return 0.0
    return max_weighted / weighted_sum


def community_purity_semantic(
    comm_nodes: List[str],
    go_map: Dict[str, List[str]],
    sim_cache: Dict[Tuple[str, str], float],
) -> float:
    """Semantic purity based on average pairwise Resnik similarity.

    purity_sem(C) = (2 / (n(n−1))) · Σ_{i<j} sim_norm(t_i, t_j)

    where *n* is the total number of GO terms in the community (including
    duplicates).  Self-similarity is 1.0 (normalised).

    For a single-term community, returns 1.0 (trivially coherent).
    For a community with no GO terms, returns 0.0.
    """
    all_terms: list = []
    for node in comm_nodes:
        all_terms.extend(go_map.get(node, []))
    n = len(all_terms)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    total_sim = 0.0
    n_pairs = 0
    for i in range(n):
        ti = all_terms[i]
        for j in range(i + 1, n):
            tj = all_terms[j]
            total_sim += sim_cache.get((ti, tj), 0.0)
            n_pairs += 1

    if n_pairs == 0:
        return 0.0
    return total_sim / n_pairs


# ===================================================================
# 4.  Mean purity wrappers (for use with GF curve computation)
# ===================================================================

def mean_purity_standard(
    communities,
    go_map: Dict[str, List[str]],
    nodes: Optional[List[str]] = None,
) -> float:
    """Mean standard purity across all communities."""
    purities = []
    for comm in communities:
        if nodes is not None:
            names = [nodes[idx] for idx in comm]
        else:
            names = list(comm)
        purities.append(community_purity_standard(names, go_map))
    return float(np.mean(purities)) if purities else 0.0


def mean_purity_ic_weighted(
    communities,
    go_map: Dict[str, List[str]],
    ic: Dict[str, float],
    nodes: Optional[List[str]] = None,
) -> float:
    """Mean IC-weighted purity across all communities."""
    purities = []
    for comm in communities:
        if nodes is not None:
            names = [nodes[idx] for idx in comm]
        else:
            names = list(comm)
        purities.append(community_purity_ic_weighted(names, go_map, ic))
    return float(np.mean(purities)) if purities else 0.0


def mean_purity_semantic(
    communities,
    go_map: Dict[str, List[str]],
    sim_cache: Dict[Tuple[str, str], float],
    nodes: Optional[List[str]] = None,
) -> float:
    """Mean semantic purity across all communities."""
    purities = []
    for comm in communities:
        if nodes is not None:
            names = [nodes[idx] for idx in comm]
        else:
            names = list(comm)
        purities.append(community_purity_semantic(names, go_map, sim_cache))
    return float(np.mean(purities)) if purities else 0.0


# ===================================================================
# 5.  GF curve computation (three variants in one pass)
# ===================================================================

def compute_gf_curves_all_variants(
    coords: np.ndarray,
    nodes: List[str],
    go_map: Dict[str, List[str]],
    r_vals: np.ndarray,
    ic: Dict[str, float],
    sim_cache: Dict[Tuple[str, str], float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute G-F curves for all three purity variants simultaneously.

    Since all variants share the same community structure at each *r*,
    community detection is performed only once per threshold.

    Parameters
    ----------
    coords : (N, d) array of embedding coordinates.
    nodes : list of N node names.
    go_map : gene → GO terms mapping.
    r_vals : array of distance thresholds.
    ic : information content per GO term.
    sim_cache : pre-computed normalised Resnik similarity.

    Returns
    -------
    (std_purities, ic_purities, sem_purities) — each shape (len(r_vals),).
    """
    from scipy.spatial.distance import pdist, squareform
    from networkx import Graph
    from networkx.algorithms.community import greedy_modularity_communities

    dist_matrix = squareform(pdist(coords, metric="euclidean"))
    idx_i, idx_j = np.triu_indices_from(dist_matrix, k=1)
    edge_order = np.argsort(dist_matrix[idx_i, idx_j])
    sorted_i = idx_i[edge_order]
    sorted_j = idx_j[edge_order]
    sorted_d = dist_matrix[sorted_i, sorted_j]

    n = len(r_vals)
    std_p = np.zeros(n)
    ic_p = np.zeros(n)
    sem_p = np.zeros(n)

    edge_ptr = 0
    n_edges = len(sorted_d)

    for k, r in enumerate(r_vals):
        # advance edges
        while edge_ptr < n_edges and sorted_d[edge_ptr] < r:
            edge_ptr += 1

        G = Graph()
        G.add_nodes_from(range(len(nodes)))
        if edge_ptr > 0:
            edges = list(zip(sorted_i[:edge_ptr], sorted_j[:edge_ptr]))
            G.add_edges_from(
                (u, v) for u, v in edges
                if 0 < dist_matrix[u, v] < r
            )

        if G.number_of_edges() == 0:
            continue

        comms = list(greedy_modularity_communities(G))

        # compute all three variants
        sp_list, ip_list, ep_list = [], [], []
        for comm in comms:
            names = [nodes[idx] for idx in comm]
            sp_list.append(community_purity_standard(names, go_map))
            ip_list.append(community_purity_ic_weighted(names, go_map, ic))
            ep_list.append(community_purity_semantic(names, go_map, sim_cache))

        std_p[k] = float(np.mean(sp_list)) if sp_list else 0.0
        ic_p[k] = float(np.mean(ip_list)) if ip_list else 0.0
        sem_p[k] = float(np.mean(ep_list)) if ep_list else 0.0

    return std_p, ic_p, sem_p


# ===================================================================
# 6.  DAG inflation diagnostics
# ===================================================================

def diagnose_dag_inflation(
    go_map_original: Dict[str, List[str]],
    go_map_propagated: Dict[str, List[str]],
    ic: Dict[str, float],
) -> dict:
    """Quantify the purity inflation caused by True Path Rule expansion.

    Computes standard and IC-weighted purity for a representative community
    under both original and propagated annotations, returning diagnostic
    statistics.

    Returns
    -------
    dict with keys:
        original_mean_terms_per_gene, propagated_mean_terms_per_gene,
        expansion_factor, inflation_ratio (std_purity propagated / original).
    """
    def _mean_terms(gm):
        counts = [len(v) for v in gm.values() if v]
        return float(np.mean(counts)) if counts else 0.0

    orig_mean = _mean_terms(go_map_original)
    prop_mean = _mean_terms(go_map_propagated)

    expansion = prop_mean / max(orig_mean, EPS)

    # Compute purity on a synthetic "all genes" community to show inflation
    all_genes = sorted(go_map_original.keys())
    std_orig = community_purity_standard(all_genes, go_map_original)
    std_prop = community_purity_standard(all_genes, go_map_propagated)
    ic_orig = community_purity_ic_weighted(all_genes, go_map_original, ic)
    ic_prop = community_purity_ic_weighted(all_genes, go_map_propagated, ic)

    return {
        "original_mean_terms_per_gene": round(orig_mean, 2),
        "propagated_mean_terms_per_gene": round(prop_mean, 2),
        "expansion_factor": round(expansion, 2),
        "standard_purity_original": round(std_orig, 4),
        "standard_purity_propagated": round(std_prop, 4),
        "ic_weighted_purity_original": round(ic_orig, 4),
        "ic_weighted_purity_propagated": round(ic_prop, 4),
        "inflation_ratio_std": round(std_prop / max(std_orig, EPS), 4),
        "inflation_ratio_ic": round(ic_prop / max(ic_orig, EPS), 4),
    }
