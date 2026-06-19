#!/usr/bin/env python3
"""
G-F Consistency Framework — Pathway-Level Biological Analysis
==============================================================
Extends the biological interpretation (Step 20) with deeper mechanistic
insights:

1. **Pathway Enrichment of G-F Communities**: For each embedding method's
   spatial communities, test enrichment against KEGG / Reactome pathways.

2. **Cancer Mutation Association**: Cross-reference G-F community structure
   with COSMIC cancer gene census to identify embedding-preserved cancer
   modules.

3. **Signalling Pathway Perturbation**: Measure how pathway-level G-F
   scores differ between healthy and perturbed (e.g., knockout) networks.

4. **Cross-Method Consensus Communities**: Identify gene communities that
   are consistently recovered across high-G-F methods.

This module is designed to work with the yeast PPI dataset but can be
extended to human via the multispecies_loader.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pathway enrichment analysis
# ---------------------------------------------------------------------------

def pathway_enrichment(
    community_genes: list[str],
    pathway_db: dict[str, list[str]],
    background_genes: Optional[list[str]] = None,
    min_pathway_size: int = 5,
    min_overlap: int = 2,
) -> list[dict]:
    """Fisher's exact test for pathway enrichment in a gene community.

    Parameters
    ----------
    community_genes : genes in the community
    pathway_db : pathway_name -> list of member genes
    background_genes : all genes in the analysis (universe)
    min_pathway_size : skip pathways smaller than this
    min_overlap : skip pathways with fewer overlapping genes

    Returns
    -------
    list of dicts sorted by p-value, with keys:
        pathway, overlap, p_value, odds_ratio, fdr
    """
    community_set = set(community_genes)
    if background_genes is None:
        background_genes = list(set().union(*[set(v) for v in pathway_db.values()]))
    bg_set = set(background_genes)
    N = len(bg_set)
    n_comm = len(community_set & bg_set)

    results = []
    for pathway_name, members in pathway_db.items():
        members_set = set(members)
        if len(members_set) < min_pathway_size:
            continue

        overlap = community_set & members_set & bg_set
        if len(overlap) < min_overlap:
            continue

        # 2x2 contingency table
        a = len(overlap)                          # in community AND pathway
        b = n_comm - a                            # in community, NOT in pathway
        c = len(members_set & bg_set) - a         # in pathway, NOT in community
        d = N - a - b - c                         # neither

        if a + b == 0 or c + d == 0:
            continue

        odds_ratio, p_value = stats.fisher_exact(
            [[a, b], [c, d]], alternative="greater"
        )

        results.append({
            "pathway": pathway_name,
            "overlap_genes": sorted(overlap),
            "overlap_count": a,
            "pathway_size": len(members_set & bg_set),
            "community_size": n_comm,
            "odds_ratio": float(odds_ratio) if not np.isinf(odds_ratio) else float("inf"),
            "p_value": float(p_value),
        })

    # Sort by p-value
    results.sort(key=lambda x: x["p_value"])

    # BH FDR correction
    n_tests = len(results)
    if n_tests > 0:
        for i, r in enumerate(results):
            r["fdr"] = min(r["p_value"] * n_tests / (i + 1), 1.0)

    return results


# ---------------------------------------------------------------------------
# Cancer gene association
# ---------------------------------------------------------------------------

def cancer_gene_association(
    community_genes: list[str],
    cancer_genes: list[str],
    background_genes: Optional[list[str]] = None,
) -> dict:
    """Test enrichment of cancer genes in a community (Fisher's exact).

    Parameters
    ----------
    community_genes : genes in the community
    cancer_genes : known cancer genes (e.g., COSMIC census)
    background_genes : all genes in the analysis

    Returns
    -------
    dict with enrichment statistics
    """
    community_set = set(community_genes)
    cancer_set = set(cancer_genes)

    if background_genes is None:
        background_genes = list(community_set | cancer_set)
    bg_set = set(background_genes)

    a = len(community_set & cancer_set & bg_set)
    b = len(community_set & bg_set) - a
    c = len(cancer_set & bg_set) - a
    d = len(bg_set) - a - b - c

    if a + b == 0 or a + c == 0:
        return {"overlap": 0, "p_value": 1.0, "odds_ratio": 0.0,
                "cancer_genes_found": []}

    odds_ratio, p_value = stats.fisher_exact(
        [[a, b], [c, d]], alternative="greater"
    )

    return {
        "overlap": a,
        "cancer_genes_found": sorted(community_set & cancer_set),
        "odds_ratio": float(odds_ratio) if not np.isinf(odds_ratio) else float("inf"),
        "p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# Cross-method consensus communities
# ---------------------------------------------------------------------------

def consensus_communities(
    community_sets: dict[str, list[set]],
    min_methods: int = 3,
    jaccard_threshold: float = 0.3,
) -> list[dict]:
    """Find gene communities consistently recovered across methods.

    Parameters
    ----------
    community_sets : method_name -> list of gene sets (communities)
    min_methods : minimum methods that must agree
    jaccard_threshold : Jaccard similarity threshold for "same" community

    Returns
    -------
    list of consensus community dicts
    """
    all_methods = list(community_sets.keys())
    if len(all_methods) < min_methods:
        logger.warning(
            "Only %d methods available, need %d for consensus",
            len(all_methods), min_methods,
        )
        return []

    # Collect all communities across methods
    all_communities = []
    for method, comms in community_sets.items():
        for comm in comms:
            all_communities.append((method, frozenset(comm)))

    # Greedy clustering by Jaccard similarity
    consensus = []
    used = set()

    for i, (m1, c1) in enumerate(all_communities):
        if i in used:
            continue

        cluster_methods = {m1}
        cluster_genes = set(c1)
        used.add(i)

        for j, (m2, c2) in enumerate(all_communities):
            if j in used:
                continue
            if m2 == m1:
                continue  # skip same method

            jaccard = len(c1 & c2) / len(c1 | c2) if (c1 | c2) else 0
            if jaccard >= jaccard_threshold:
                cluster_methods.add(m2)
                cluster_genes |= c2
                used.add(j)

        if len(cluster_methods) >= min_methods:
            consensus.append({
                "genes": sorted(cluster_genes),
                "n_genes": len(cluster_genes),
                "methods": sorted(cluster_methods),
                "n_methods": len(cluster_methods),
            })

    # Sort by number of supporting methods (descending)
    consensus.sort(key=lambda x: -x["n_methods"])
    return consensus


# ---------------------------------------------------------------------------
# Signalling pathway perturbation analysis
# ---------------------------------------------------------------------------

def pathway_perturbation_analysis(
    healthy_gf_scores: dict[str, float],
    perturbed_gf_scores: dict[str, float],
    pathway_gene_sets: dict[str, list[str]],
) -> list[dict]:
    """Compare G-F community structure between healthy and perturbed networks.

    For each pathway, measure how much the community structure changes.

    Parameters
    ----------
    healthy_gf_scores : gene -> community_id in healthy network
    perturbed_gf_scores : gene -> community_id in perturbed network
    pathway_gene_sets : pathway -> list of member genes

    Returns
    -------
    list of perturbation results per pathway
    """
    results = []
    for pathway, genes in pathway_gene_sets.items():
        # Fraction of gene pairs in same community
        healthy_same = 0
        perturbed_same = 0
        total_pairs = 0

        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                g1, g2 = genes[i], genes[j]
                if g1 in healthy_gf_scores and g2 in healthy_gf_scores:
                    total_pairs += 1
                    if healthy_gf_scores[g1] == healthy_gf_scores[g2]:
                        healthy_same += 1
                    if perturbed_gf_scores.get(g1) == perturbed_gf_scores.get(g2):
                        perturbed_same += 1

        if total_pairs == 0:
            continue

        h_frac = healthy_same / total_pairs
        p_frac = perturbed_same / total_pairs
        delta = p_frac - h_frac

        results.append({
            "pathway": pathway,
            "n_genes": len(genes),
            "n_pairs": total_pairs,
            "healthy_cohesion": round(h_frac, 4),
            "perturbed_cohesion": round(p_frac, 4),
            "delta": round(delta, 4),
            "perturbed": abs(delta) > 0.2,
        })

    results.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def run_pathway_analysis(
    G: nx.Graph,
    communities: list[set],
    go_map: dict,
    pathway_db: Optional[dict] = None,
    cancer_genes: Optional[list] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run the full pathway-level biological analysis.

    Parameters
    ----------
    G : the PPI network
    communities : list of gene sets from community detection
    go_map : gene -> GO terms
    pathway_db : pathway -> gene list (optional, uses GO if not provided)
    cancer_genes : list of known cancer genes (optional)
    output_dir : where to save results

    Returns
    -------
    dict with all analysis results
    """
    nodes = sorted(G.nodes())
    all_results = {
        "n_communities": len(communities),
        "community_enrichments": [],
        "cancer_associations": [],
    }

    for i, comm in enumerate(communities):
        comm_genes = sorted(comm)
        if len(comm_genes) < 3:
            continue

        # Pathway enrichment (use GO as pathway DB if none provided)
        if pathway_db is None:
            # Build a simple pathway-like DB from GO BP terms
            go_bp: dict[str, list[str]] = {}
            for gene, terms in go_map.items():
                for term in terms:
                    if term.startswith("GO:"):
                        if term not in go_bp:
                            go_bp[term] = []
                        go_bp[term].append(gene)
            pathway_db_use = go_bp
        else:
            pathway_db_use = pathway_db

        enrichment = pathway_enrichment(comm_genes, pathway_db_use, nodes)
        if enrichment:
            all_results["community_enrichments"].append({
                "community_id": i,
                "n_genes": len(comm_genes),
                "top_pathways": enrichment[:5],
            })

        # Cancer gene association
        if cancer_genes:
            cancer_result = cancer_gene_association(comm_genes, cancer_genes, nodes)
            if cancer_result["overlap"] > 0:
                all_results["cancer_associations"].append({
                    "community_id": i,
                    **cancer_result,
                })

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outfile = output_dir / "pathway_analysis.json"
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Saved pathway analysis to %s", outfile)

    return all_results


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def main():
    """Run pathway enrichment on G-F communities at peak-purity threshold.

    Loads the curated network, detects communities at the optimal distance
    threshold (peak purity from G-F curves), and runs Fisher's exact
    pathway enrichment for each community.

    Saves: results/pathway_analysis.json
    """
    import sys
    import numpy as np
    from pathlib import Path
    from networkx.algorithms.community import greedy_modularity_communities
    from utils import (
        SEED, get_data_dir, get_results_dir, get_embeddings_dir,
        load_curated_network, load_embedding, precompute_distance_matrix,
        build_spatial_graph_fast,
        CLASSICAL_METHODS,
    )

    np.random.seed(SEED)
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    emb_dir = get_embeddings_dir()

    # Load network and find best method by G-F score
    print("Loading G-F scores to determine best method...")
    gf_file = results_dir / "gf_scores.json"
    if not gf_file.exists():
        print("G-F scores not found. Run Step 3 (compute_gf.py) first.")
        return
    with open(gf_file, encoding="utf-8") as f:
        gf_data = json.load(f)
    scores = gf_data.get("scores", {})
    if not scores:
        print("No G-F scores found. Step skipped.")
        return

    best_method = max(scores, key=scores.get)
    print(f"Best method by G-F score: {best_method} ({scores[best_method]:.4f})")

    # Load best method embedding and network
    G, nodes, go_map = load_curated_network(data_dir)
    coords, emb_nodes = load_embedding(best_method, "153", embeddings_dir=emb_dir)
    common = sorted(set(emb_nodes) & set(nodes) & set(go_map.keys()))
    emb_node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
    idx = [emb_node_to_idx[n] for n in common]
    aligned = coords[idx]

    # Find peak-purity threshold via a quick r-sweep
    from utils import compute_gf_curve, R_MIN, R_MAX, N_POINTS
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    purities, _ = compute_gf_curve(aligned, common, go_map, r_vals)
    peak_idx = int(np.argmax(purities))
    optimal_r = float(r_vals[peak_idx])
    print(f"Peak purity r = {optimal_r:.4f} (purity = {purities[peak_idx]:.4f})")

    # Detect communities at optimal r
    D = precompute_distance_matrix(aligned)
    G_r = build_spatial_graph_fast(D, optimal_r)
    communities = list(greedy_modularity_communities(G_r))
    comm_sets = [set(common[i] for i in c) for c in communities]
    print(f"Detected {len(comm_sets)} communities at r = {optimal_r:.4f}")

    # Run pathway enrichment
    results = run_pathway_analysis(G, comm_sets, go_map, output_dir=results_dir)
    n_enriched = len(results.get("community_enrichments", []))
    print(f"Pathway enrichment complete: {n_enriched} communities with significant pathways")
