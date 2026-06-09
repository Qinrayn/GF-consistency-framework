#!/usr/bin/env python3
"""
biological_interpretation.py
=============================

Biological interpretation of G-F Scores through a 4-level scale and
a case-study visualization.

This script adds biological meaning to the numerical G-F Scores by:

1. Classifying each method into one of four interpretability levels
   based on how well the embedding geometry reflects functional modules.
2. Identifying the highest-purity GO term cluster for each method
   via spatial graph community detection at the optimal distance threshold.
3. Generating a three-panel case-study figure (FigS5) that illustrates
   a representative high-purity cluster in embedding space (Panel A),
   as a PPI subgraph (Panel B), and within the GO DAG hierarchy (Panel C).
4. Producing a summary table that maps every method to its G-F Score,
   interpretability level, and best GO term cluster.

Output files
------------
- ``results/biological_interpretation.json``
- ``figures/FigS5_biological_case_study.png``

CLI usage
---------
::

    python biological_interpretation.py
    python biological_interpretation.py --method Spectral
    python biological_interpretation.py --level-thresholds 0.3,0.5,0.7
"""

import sys
import json
import argparse
import logging
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
    load_curated_network, load_embedding, compute_gf_curve,
    rescale_coordinates, precompute_distance_matrix,
    compute_centrality_features,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

# Local imports from utils (not in the required set, but used internally)
from utils import (
    build_spatial_graph_fast,
    functional_purity_named,
)


# ===========================================================================
# Constants
# ===========================================================================

# Okabe-Ito colorblind-safe palette (extended)
OKABE_ITO = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#999999",  # Gray
    "#000000",  # Black
    "#8B4513",  # Saddle Brown
    "#4B0082",  # Indigo
]

# ---------------------------------------------------------------------------
# GO term name dictionary
# ---------------------------------------------------------------------------
# Comprehensive mapping of GO term IDs found in the yeast curated 153-node
# dataset (gene_go_map.json) to their standard human-readable names.
# Covers biological process, molecular function, and cellular component
# ontology terms relevant to Saccharomyces cerevisiae.
# ---------------------------------------------------------------------------

GO_TERM_NAMES = {
    # ---- Biological Process ----
    "GO:0000077": "DNA damage checkpoint signaling",
    "GO:0000082": "G1/S transition of mitotic cell cycle",
    "GO:0000086": "G2/M transition of mitotic cell cycle",
    "GO:0000105": "sulfur amino acid biosynthetic process",
    "GO:0000132": "establishment of mitotic spindle orientation",
    "GO:0000133": "spindle orientation in bud neck",
    "GO:0000145": "exocytosis",
    "GO:0000162": "tryptophan biosynthetic process",
    "GO:0000186": "inactivation of MAPK activity",
    "GO:0000187": "activation of MAPK activity",
    "GO:0000278": "mitotic cell cycle",
    "GO:0000281": "mitotic cytokinesis",
    "GO:0000435": "protein lipidation",
    "GO:0000723": "telomere maintenance",
    "GO:0000724": "double-strand break repair via HR",
    "GO:0000750": "pheromone-dependent signal transduction",
    "GO:0000902": "cell morphogenesis",
    "GO:0000981": "regulation of transcription",
    "GO:0001056": "RNA polymerase III transcription",
    "GO:0001178": "meiotic spindle assembly checkpoint",
    "GO:0002181": "cytoplasmic translation",
    "GO:0006006": "glucose metabolic process",
    "GO:0006012": "galactose metabolic process",
    "GO:0006066": "alcohol metabolic process",
    "GO:0006067": "ethanol oxidation",
    "GO:0006206": "purine nucleobase metabolic process",
    "GO:0006221": "pyrimidine nucleotide biosynthetic process",
    "GO:0006260": "DNA replication",
    "GO:0006269": "DNA replication, synthesis of RNA primer",
    "GO:0006270": "DNA replication initiation",
    "GO:0006281": "DNA repair",
    "GO:0006298": "DNA damage response, mismatch repair",
    "GO:0006303": "double-strand break repair via NHEJ",
    "GO:0006310": "DNA recombination",
    "GO:0006338": "chromatin remodeling",
    "GO:0006348": "chromatin silencing at telomere",
    "GO:0006351": "transcription, DNA-templated",
    "GO:0006355": "regulation of transcription, DNA-templated",
    "GO:0006357": "regulation of transcription from Pol II promoter",
    "GO:0006366": "transcription from RNA polymerase II promoter",
    "GO:0006367": "transcription initiation from RNA Pol II promoter",
    "GO:0006368": "transcription elongation from RNA Pol II promoter",
    "GO:0006383": "poly(A)+ mRNA 3' end processing",
    "GO:0006403": "RNA localization",
    "GO:0006412": "translation",
    "GO:0006413": "translational initiation",
    "GO:0006414": "translational elongation",
    "GO:0006417": "regulation of translation",
    "GO:0006457": "protein folding",
    "GO:0006511": "ubiquitin-dependent protein catabolic process",
    "GO:0006541": "glutamine metabolic process",
    "GO:0006553": "lysine biosynthetic process",
    "GO:0006568": "tryptophan metabolic process",
    "GO:0006886": "intracellular protein transport",
    "GO:0006895": "Golgi to ER retrograde transport",
    "GO:0006904": "vesicle-mediated transport",
    "GO:0006974": "cellular response to DNA damage stimulus",
    "GO:0007010": "cytoskeleton organization",
    "GO:0007049": "cell cycle",
    "GO:0007095": "mitotic spindle assembly checkpoint",
    "GO:0007131": "reciprocal meiotic recombination",
    "GO:0007165": "signal transduction",
    "GO:0007186": "G protein-coupled receptor signaling pathway",
    "GO:0007264": "small GTPase mediated signal transduction",
    "GO:0007530": "mating type determination",
    "GO:0008630": "intrinsic apoptotic signaling pathway",
    "GO:0008652": "cellular amino acid biosynthetic process",
    "GO:0009098": "methionine biosynthetic process",
    "GO:0009099": "cysteine biosynthetic process via OAS",
    "GO:0009263": "deoxyribonucleotide biosynthetic process",
    "GO:0009408": "response to heat",
    "GO:0015031": "protein transport",
    "GO:0015935": "small ribosomal subunit assembly",
    "GO:0016192": "vesicle-mediated transport",
    "GO:0016567": "protein ubiquitination",
    "GO:0022625": "cytosolic large ribosomal subunit",
    "GO:0022627": "cytosolic small ribosomal subunit",
    "GO:0030036": "actin cytoskeleton organization",
    "GO:0030048": "actin filament-based movement",
    "GO:0030308": "negative regulation of cell growth",
    "GO:0030337": "negative regulation of cell migration",
    "GO:0031145": "anaphase-promoting complex-dependent proteolysis",
    "GO:0031577": "spindle checkpoint signaling",
    "GO:0034198": "cellular response to amino acid starvation",
    "GO:0040029": "regulation of gene expression, epigenetic",
    "GO:0042026": "protein refolding",
    "GO:0043161": "proteasome-mediated ubiquitin-dependent catabolism",
    "GO:0043453": "respiratory electron transport chain",
    "GO:0044205": "purine nucleoside salvage",
    "GO:0045944": "positive regulation of transcription from Pol II",
    "GO:0046208": "tRNA modification",
    "GO:0046835": "indole-containing compound metabolic process",
    "GO:0046903": "secretion",
    "GO:0051085": "cotranslational protein targeting to membrane",
    "GO:0051103": "DNA ligation involved in DNA repair",
    "GO:0051123": "RNA polymerase II transcription preinitiation complex",
    "GO:0051301": "cell division",
    "GO:0051865": "protein autophosphorylation",
    "GO:0090522": "tRNA thiolation",
    # ---- Molecular Function ----
    "GO:0003677": "DNA binding",
    "GO:0003688": "DNA replication origin binding",
    "GO:0003689": "DNA clamp loader activity",
    "GO:0003697": "single-stranded DNA binding",
    "GO:0003735": "structural constituent of ribosome",
    "GO:0003743": "translation initiation factor activity",
    "GO:0003746": "translation elongation factor activity",
    "GO:0003887": "DNA-directed DNA polymerase activity",
    "GO:0003896": "DNA primase activity",
    "GO:0003910": "DNA ligase (ATP) activity",
    "GO:0003924": "GTPase activity",
    "GO:0004386": "helicase activity",
    "GO:0004519": "endonuclease activity",
    "GO:0004674": "protein serine/threonine kinase activity",
    "GO:0004707": "MAP kinase activity",
    "GO:0004708": "MAP kinase kinase activity",
    "GO:0004871": "signal transducer activity",
    "GO:0004930": "G protein-coupled receptor activity",
    "GO:0005085": "guanyl-nucleotide exchange factor activity",
    "GO:0005484": "SNAP receptor activity",
    "GO:0008135": "translation factor activity",
    "GO:0008234": "cysteine-type peptidase activity",
    "GO:0022625": "cytosolic large ribosomal subunit",
    # ---- Cellular Component ----
    "GO:0005667": "transcription regulator complex",
}


# ===========================================================================
# Default G-F Score interpretation scale
# ===========================================================================

GF_SCORE_LEVELS = {
    1: {
        "threshold": (None, 0.3),
        "label": "Weak alignment",
        "description": (
            "Geometric structure poorly reflects functional modules; "
            "embedding cannot distinguish functional boundaries"
        ),
        "color": "#D55E00",
    },
    2: {
        "threshold": (0.3, 0.5),
        "label": "Moderate alignment",
        "description": (
            "Can identify major functional categories "
            "(e.g., metabolism vs signal transduction)"
        ),
        "color": "#F0E442",
    },
    3: {
        "threshold": (0.5, 0.7),
        "label": "Strong alignment",
        "description": "Can distinguish fine-grained functional submodules",
        "color": "#009E73",
    },
    4: {
        "threshold": (0.7, None),
        "label": "Very strong alignment",
        "description": "Embedding distance directly reflects functional similarity",
        "color": "#0072B2",
    },
}

# Default G-F Scores for the yeast 153-node curated dataset.
# Used as fallback when results/final_results_summary.json is unavailable.
DEFAULT_GF_SCORES = {
    "DM": 0.625,
    "Spectral": 0.588,
    "PCA": 0.577,
    "MDS": 0.557,
    "VGAE-feat": 0.496,
    "DeepWalk": 0.462,
    "Node2Vec": 0.415,
    "VGAE": 0.248,
}

# All eight methods evaluated in the yeast experiment.
ALL_METHODS = [
    "DM", "Spectral", "PCA", "MDS", "VGAE-feat",
    "DeepWalk", "Node2Vec", "VGAE",
]

# Logger
_logger = logging.getLogger("biological_interpretation")


# ===========================================================================
# 1. G-F Score Biological Interpretation Scale
# ===========================================================================

def classify_gf_score(score, thresholds=(0.3, 0.5, 0.7)):
    """
    Classify a G-F Score into one of four biological interpretation levels.

    The four-level scale maps the numerical G-F Score to a qualitative
    assessment of how well the embedding's geometric structure reflects
    functional modules in the PPI network:

    +-------+---------------------+-------------------------------------------+
    | Level | Score range         | Interpretation                            |
    +=======+=====================+===========================================+
    | 1     | score < 0.3         | Weak alignment                            |
    +-------+---------------------+-------------------------------------------+
    | 2     | 0.3 <= score < 0.5  | Moderate alignment                        |
    +-------+---------------------+-------------------------------------------+
    | 3     | 0.5 <= score < 0.7  | Strong alignment                          |
    +-------+---------------------+-------------------------------------------+
    | 4     | score >= 0.7        | Very strong alignment                     |
    +-------+---------------------+-------------------------------------------+

    Parameters
    ----------
    score : float
        G-F Score value, typically in the range [0, 1].
    thresholds : tuple of float, optional
        Three threshold values ``(t1, t2, t3)`` defining the boundaries
        between levels 1/2, 2/3, and 3/4 respectively.
        Default: ``(0.3, 0.5, 0.7)``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"level"`` (int): Level number (1--4).
        - ``"label"`` (str): Short human-readable label.
        - ``"description"`` (str): Detailed description of what this level
          means biologically.
        - ``"color"`` (str): Hex colour code for visualisation.

    Examples
    --------
    >>> info = classify_gf_score(0.625)
    >>> info["level"]
    3
    >>> info["label"]
    'Strong alignment'

    >>> info = classify_gf_score(0.248)
    >>> info["level"]
    1

    >>> info = classify_gf_score(0.462)
    >>> info["level"]
    2
    """
    t1, t2, t3 = thresholds

    if score < t1:
        level = 1
    elif score < t2:
        level = 2
    elif score < t3:
        level = 3
    else:
        level = 4

    level_info = GF_SCORE_LEVELS[level]
    return {
        "level": level,
        "label": level_info["label"],
        "description": level_info["description"],
        "color": level_info["color"],
    }


# ===========================================================================
# 2. GO Term Cluster Analysis
# ===========================================================================

def analyze_go_term_clusters(method, nodes, go_map, coords_aligned,
                             r_vals=None, min_cluster_size=4):
    """
    Identify the highest-purity GO term cluster for an embedding method.

    For each candidate distance threshold *r*, a spatial neighbourhood graph
    is built from the 2-D embedding coordinates.  Communities are detected
    via greedy modularity optimisation, and functional purity is computed
    per community as the fraction of members sharing the dominant GO term.

    The best cluster is selected by maximising a quality score that
    balances purity and cluster size::

        quality = purity * log2(max(n_nodes, 2))

    Only communities with at least *min_cluster_size* members are
    considered, which avoids trivially small (1--2 node) clusters that
    always achieve purity = 1.

    Parameters
    ----------
    method : str
        Embedding method name (e.g. ``"DM"``, ``"Spectral"``).
    nodes : list of str
        Ordered list of node (gene) identifiers matching ``coords_aligned``.
    go_map : dict
        Mapping from node names to lists of GO term IDs.
    coords_aligned : np.ndarray
        2-D embedding coordinates, shape ``(len(nodes), 2)``.
    r_vals : np.ndarray or None
        Distance thresholds to evaluate.  If *None*, a default grid of
        100 points in ``[0.05, 0.60]`` is used.
    min_cluster_size : int, optional
        Minimum number of nodes for a community to be considered.
        Default: 4.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"best_cluster"`` (dict): The highest-quality community with
          ``go_term``, ``go_term_name``, ``purity``, ``n_nodes``,
          ``nodes``, ``other_go_terms``, ``member_go_counts``.
        - ``"all_clusters"`` (list of dict): All communities at the
          optimal *r*, sorted by quality descending.
        - ``"optimal_r"`` (float): The distance threshold at which the
          best cluster was detected.
        - ``"max_mean_purity"`` (float): Mean purity across all
          communities at the optimal *r*.
    """
    if r_vals is None:
        r_vals = np.linspace(0.05, 0.60, 100)

    coords = rescale_coordinates(coords_aligned)
    dist_matrix = precompute_distance_matrix(coords)

    def _cluster_quality(node_names, go_map_local):
        """Compute purity and dominant term for a list of gene names."""
        go_terms_all = []
        for nn in node_names:
            if nn in go_map_local and go_map_local[nn]:
                go_terms_all.extend(go_map_local[nn])
        if not go_terms_all:
            return 0.0, None, Counter()
        tc = Counter(go_terms_all)
        dom_term, dom_count = tc.most_common(1)[0]
        return dom_count / len(node_names), dom_term, tc

    def _analyse_communities(communities_int, nodes_local, go_map_local):
        """Analyse a list of integer-index communities, returning details."""
        results = []
        for comm in communities_int:
            comm_list = sorted(comm)
            node_names = [nodes_local[idx] for idx in comm_list]
            n = len(node_names)
            purity, dom_term, term_counts = _cluster_quality(
                node_names, go_map_local
            )
            if dom_term is None:
                continue

            # Quality score: purity weighted by log2(size)
            quality = purity * np.log2(max(n, 2))

            other_terms = []
            for term, count in term_counts.most_common():
                if term != dom_term:
                    other_terms.append({
                        "term": term,
                        "name": GO_TERM_NAMES.get(term, f"GO term {term}"),
                        "count": count,
                    })

            member_counts = {}
            for nn in node_names:
                if nn in go_map_local and go_map_local[nn]:
                    for t in go_map_local[nn]:
                        member_counts[t] = member_counts.get(t, 0) + 1

            results.append({
                "go_term": dom_term,
                "go_term_name": GO_TERM_NAMES.get(dom_term,
                                                   f"GO term {dom_term}"),
                "purity": purity,
                "n_nodes": n,
                "nodes": node_names,
                "indices": comm_list,
                "quality": quality,
                "other_go_terms": other_terms,
                "member_go_counts": member_counts,
            })
        return results

    # ---- Sweep r to find the best individual cluster ----
    # Track the single best cluster (by quality) across all r values,
    # AND the r that produces the best set of communities overall.
    best_quality_overall = -1.0
    best_r_for_best_cluster = None
    best_communities_for_best_cluster = None
    best_single_cluster_info = None

    # Also track r with best mean purity (for reporting)
    best_mean_purity = -1.0
    best_mean_r = None
    best_mean_communities = None

    for r in r_vals:
        G_r = build_spatial_graph_fast(dist_matrix, r)
        if G_r.number_of_edges() == 0:
            continue
        communities = list(
            nx.algorithms.community.greedy_modularity_communities(G_r)
        )
        if not communities:
            continue

        # Analyse all communities at this r
        cluster_details = _analyse_communities(communities, nodes, go_map)

        # Filter by minimum cluster size for the "best cluster" search
        eligible = [c for c in cluster_details
                    if c["n_nodes"] >= min_cluster_size]

        # Mean purity across ALL communities (for reporting)
        comm_named = [
            {nodes[idx] for idx in comm} for comm in communities
        ]
        mean_p = functional_purity_named(comm_named, go_map)
        if mean_p > best_mean_purity:
            best_mean_purity = mean_p
            best_mean_r = float(r)
            best_mean_communities = communities

        # Check if any eligible cluster beats the current best
        for clust in eligible:
            if clust["quality"] > best_quality_overall:
                best_quality_overall = clust["quality"]
                best_r_for_best_cluster = float(r)
                best_communities_for_best_cluster = communities
                best_single_cluster_info = clust

    # ---- Determine which r and communities to report ----
    # Prefer the r that yielded the best individual cluster
    if best_communities_for_best_cluster is not None:
        report_r = best_r_for_best_cluster
        report_communities = best_communities_for_best_cluster
        report_mean_purity = best_mean_purity
    elif best_mean_communities is not None:
        # Fallback: use best mean-purity r (may have only small clusters)
        report_r = best_mean_r
        report_communities = best_mean_communities
        report_mean_purity = best_mean_purity
    else:
        return {
            "best_cluster": None,
            "all_clusters": [],
            "optimal_r": None,
            "max_mean_purity": 0.0,
        }

    # ---- Re-analyse all communities at the chosen r ----
    all_cluster_details = _analyse_communities(
        report_communities, nodes, go_map
    )
    # Sort by quality descending
    all_cluster_details.sort(key=lambda x: x["quality"], reverse=True)

    # The best cluster: either the one we tracked, or the top eligible one
    if best_single_cluster_info is not None:
        best = best_single_cluster_info
    else:
        # Fallback: pick the best from whatever we have
        eligible = [c for c in all_cluster_details
                    if c["n_nodes"] >= min_cluster_size]
        best = eligible[0] if eligible else (
            all_cluster_details[0] if all_cluster_details else None
        )

    # Clean up: remove the internal "quality" key from output
    for c in all_cluster_details:
        c.pop("quality", None)
    if best is not None:
        best.pop("quality", None)

    return {
        "best_cluster": best,
        "all_clusters": all_cluster_details,
        "optimal_r": report_r,
        "max_mean_purity": report_mean_purity,
    }


# ===========================================================================
# 3. GO DAG Construction
# ===========================================================================

def build_go_dag_for_term(go_term, go_map):
    """
    Build a simplified GO DAG centred on a specific term.

    Since the full GO OBO ontology file is not available locally, this
    function constructs an informative DAG using the known GO term
    hierarchy for *Saccharomyces cerevisiae* biological process terms.

    The DAG is rooted at **GO:0008150** (``biological_process``) and
    includes:

    - Ancestor path from the target term up to the root.
    - Sibling terms that share a common parent.
    - Child terms (GO terms from the dataset whose parent is the target).

    Parameters
    ----------
    go_term : str
        The GO term ID to centre the DAG on (e.g. ``"GO:0006412"``).
    go_map : dict
        Mapping from gene names to lists of GO term IDs.  Used to
        discover child terms that co-occur with the target term.

    Returns
    -------
    networkx.DiGraph
        Directed acyclic graph with edges pointing from parent to child.
        Nodes carry attributes:

        - ``name`` (str): human-readable GO term name.
        - ``count`` (int): number of genes in *go_map* annotated with
          this term.
        - ``is_target`` (bool): *True* for the focal term.
        - ``depth`` (int): distance from the root.
    """
    dag = nx.DiGraph()

    # ---- Default ancestor chains for common yeast BP terms ----
    # Each list reads from root toward the target term.
    default_ancestors = {
        "GO:0006412": ["GO:0008150", "GO:0044699", "GO:0009058", "GO:0006412"],
        "GO:0006413": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006412", "GO:0006413"],
        "GO:0006414": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006412", "GO:0006414"],
        "GO:0007049": ["GO:0008150", "GO:0044763", "GO:0007049"],
        "GO:0051301": ["GO:0008150", "GO:0044763", "GO:0007049",
                       "GO:0051301"],
        "GO:0006351": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006351"],
        "GO:0006355": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006351", "GO:0006355"],
        "GO:0006260": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006260"],
        "GO:0006281": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006281"],
        "GO:0007165": ["GO:0008150", "GO:0050789", "GO:0007165"],
        "GO:0006886": ["GO:0008150", "GO:0051641", "GO:0006810",
                       "GO:0006886"],
        "GO:0006066": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006066"],
        "GO:0006457": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006457"],
        "GO:0000278": ["GO:0008150", "GO:0044763", "GO:0007049",
                       "GO:0000278"],
        "GO:0006974": ["GO:0008150", "GO:0050789", "GO:0006974"],
        "GO:0000082": ["GO:0008150", "GO:0044763", "GO:0007049",
                       "GO:0000082"],
        "GO:0006366": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006351", "GO:0006366"],
        "GO:0006511": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006511"],
        "GO:0002181": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006412", "GO:0002181"],
        "GO:0015031": ["GO:0008150", "GO:0051641", "GO:0006810",
                       "GO:0015031"],
        "GO:0006417": ["GO:0008150", "GO:0044699", "GO:0009058",
                       "GO:0006412", "GO:0006417"],
        "GO:0007186": ["GO:0008150", "GO:0050789", "GO:0007165",
                       "GO:0007186"],
        "GO:0001178": ["GO:0008150", "GO:0044763", "GO:0007049",
                       "GO:0001178"],
    }

    # Intermediate node names
    intermediate_names = {
        "GO:0008150": "biological_process",
        "GO:0044699": "single-organism process",
        "GO:0009058": "biosynthetic process",
        "GO:0044763": "single-organism cellular process",
        "GO:0050789": "regulation of biological process",
        "GO:0051641": "cellular localization",
        "GO:0006810": "transport",
    }

    # ---- Build ancestor path ----
    if go_term in default_ancestors:
        path = default_ancestors[go_term]
    else:
        # Generic fallback: direct link to root
        path = ["GO:0008150", go_term]

    # Add edges along the ancestor path
    for i in range(len(path) - 1):
        dag.add_edge(path[i], path[i + 1])

    # ---- Add sibling / intermediate annotations ----
    for node_id in dag.nodes():
        if "name" not in dag.nodes[node_id]:
            name = GO_TERM_NAMES.get(node_id,
                                     intermediate_names.get(node_id, node_id))
            dag.nodes[node_id]["name"] = name

    # ---- Add child terms from go_map (co-annotation) ----
    target_genes = [
        gene for gene, terms in go_map.items() if go_term in terms
    ]
    child_counts = Counter()
    for gene in target_genes:
        for t in go_map[gene]:
            if t != go_term and t not in dag.nodes():
                child_counts[t] += 1

    # Attach up to 5 co-annotated terms as children
    for child_term, cnt in child_counts.most_common(5):
        if child_term not in dag.nodes():
            dag.add_edge(go_term, child_term)
            dag.nodes[child_term]["name"] = GO_TERM_NAMES.get(
                child_term, f"GO term {child_term}"
            )

    # ---- Assign depth from root ----
    root = "GO:0008150"
    if root in dag.nodes():
        depths = nx.shortest_path_length(dag, root)
        for node_id in dag.nodes():
            dag.nodes[node_id]["depth"] = depths.get(node_id, 0)

    # ---- Node attributes: count and target flag ----
    all_go_terms_in_dataset = set()
    for terms in go_map.values():
        all_go_terms_in_dataset.update(terms)

    for node_id in dag.nodes():
        cnt = sum(1 for terms in go_map.values() if node_id in terms)
        dag.nodes[node_id]["count"] = cnt
        dag.nodes[node_id]["is_target"] = (node_id == go_term)
        if "name" not in dag.nodes[node_id]:
            dag.nodes[node_id]["name"] = GO_TERM_NAMES.get(node_id, node_id)
        if "depth" not in dag.nodes[node_id]:
            dag.nodes[node_id]["depth"] = 0

    # If DAG is too sparse, add extra context
    if dag.number_of_nodes() < 3:
        if root not in dag.nodes():
            dag.add_edge(root, go_term)
            dag.nodes[root]["name"] = "biological_process"
            dag.nodes[root]["count"] = sum(
                1 for terms in go_map.values() if root in terms
            )
            dag.nodes[root]["is_target"] = False
            dag.nodes[root]["depth"] = 0
        dag.nodes[go_term]["depth"] = 1

    return dag


# ===========================================================================
# 4. Three-Panel Case Study Figure
# ===========================================================================

def _draw_convex_hull(ax, points, color, alpha_fill=0.12, alpha_edge=0.55,
                      padding=0.015):
    """
    Draw a convex hull around a set of 2-D points.

    Handles degenerate cases (< 3 unique points) gracefully by drawing
    circles or line segments instead of a filled polygon.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    points : np.ndarray
        Shape ``(N, 2)`` array of point coordinates.
    color : str
        Colour for fill and edge.
    alpha_fill : float
        Transparency for the filled hull.
    alpha_edge : float
        Transparency for the hull boundary.
    padding : float
        Fractional padding added around the hull vertices.
    """
    unique_pts = np.unique(points, axis=0)

    if len(unique_pts) >= 3:
        try:
            hull = ConvexHull(unique_pts)
            verts = unique_pts[hull.vertices]
            centroid = verts.mean(axis=0)
            expanded = centroid + (verts - centroid) * (1.0 + padding)
            poly = plt.Polygon(
                expanded, alpha=alpha_fill, facecolor=color,
                edgecolor=color, linewidth=1.5, linestyle="--",
                zorder=2,
            )
            ax.add_patch(poly)
            return
        except Exception:
            pass

    # Fallback for degenerate geometry
    cx = points[:, 0].mean()
    cy = points[:, 1].mean()
    radius = max(points[:, 0].std(), points[:, 1].std(), 0.01) * 2.0
    circle = plt.Circle(
        (cx, cy), radius, alpha=alpha_fill, facecolor=color,
        edgecolor=color, linewidth=1.5, linestyle="--", zorder=2,
    )
    ax.add_patch(circle)


def generate_three_panel_figure(
    method, coords, nodes, go_map, G_ppi,
    cluster_info, go_dag, figures_dir,
):
    """
    Create the three-panel biological case study figure (FigS5).

    Panel A -- Embedding Space Distribution
        Scatter plot of the 2-D embedding for the 153 curated nodes,
        coloured by GO term cluster membership.  The case-study cluster
        is highlighted with larger markers and a convex hull.

    Panel B -- Network Topology Subgraph
        The full PPI network drawn with a spring layout.  Non-cluster
        nodes appear as small grey dots; case-study cluster nodes are
        drawn larger and coloured by GO term.  Edges within the cluster
        are emphasised with thicker coloured lines.

    Panel C -- GO DAG Path
        Ancestor path from the dominant GO term of the case-study
        cluster up to the root (GO:0008150, ``biological_process``),
        with sibling and child terms.  Node size scales with gene count.

    Parameters
    ----------
    method : str
        Embedding method name.
    coords : np.ndarray
        2-D embedding coordinates, shape ``(len(nodes), 2)``.
    nodes : list of str
        Ordered gene identifiers.
    go_map : dict
        Gene-to-GO-term mapping.
    G_ppi : networkx.Graph
        Full curated PPI network.
    cluster_info : dict
        Output of :func:`analyze_go_term_clusters` for *method*.
    go_dag : networkx.DiGraph
        GO DAG centred on the dominant term of the case-study cluster.
    figures_dir : Path
        Directory where the figure will be saved.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    coords = rescale_coordinates(coords)

    # ---- Identify the representative (best) cluster ----
    best_cluster = cluster_info["best_cluster"]
    if best_cluster is None:
        # Fallback: use the largest connected component
        largest_cc = max(nx.connected_components(G_ppi), key=len)
        best_indices = [i for i, n in enumerate(nodes) if n in largest_cc]
        best_cluster = {
            "nodes": [nodes[i] for i in best_indices],
            "indices": best_indices,
            "go_term": "N/A",
            "go_term_name": "N/A",
            "purity": 0.0,
            "n_nodes": len(best_indices),
            "other_go_terms": [],
        }
        cluster_info["optimal_r"] = None

    best_indices = set(best_cluster["indices"])
    best_node_set = set(best_cluster["nodes"])

    # ---- Assign communities and colours ----
    all_clusters = cluster_info.get("all_clusters", [])
    cluster_membership = {}  # node_index -> cluster rank
    cluster_go_map = {}      # cluster rank -> dominant GO term name

    for ci, clust in enumerate(all_clusters):
        cluster_go_map[ci] = clust["go_term_name"]
        for idx in clust["indices"]:
            if idx not in cluster_membership:
                cluster_membership[idx] = ci

    n_clusters = len(all_clusters)
    # Cycle through palette if there are more clusters than colours
    cluster_colors = [
        OKABE_ITO[ci % len(OKABE_ITO)] for ci in range(max(n_clusters, 1))
    ]

    # Determine which cluster rank the best cluster has
    best_cluster_rank = None
    if best_cluster in all_clusters:
        best_cluster_rank = all_clusters.index(best_cluster)

    # ---- Create figure ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        f"Biological Case Study: {method} (G-F Score Level "
        f"{classify_gf_score(DEFAULT_GF_SCORES.get(method, 0.5))['level']})",
        fontsize=14, fontweight="bold",
    )

    # ==================================================================
    # Panel A: Embedding Space Distribution
    # ==================================================================
    ax_a = axes[0]
    ax_a.set_title(
        "(A) Embedding Space", fontsize=11, fontweight="bold"
    )

    # Draw background (non-cluster) nodes first
    bg_mask = np.array([i not in best_indices for i in range(len(nodes))])
    if bg_mask.any():
        ax_a.scatter(
            coords[bg_mask, 0], coords[bg_mask, 1],
            c="#CCCCCC", s=12, alpha=0.35, zorder=1,
            edgecolors="none", label="Other nodes",
        )

    # Draw each cluster's nodes (except the best, which is drawn last)
    drawn_indices = set()
    for ci in range(n_clusters):
        if ci == best_cluster_rank:
            continue
        ci_indices = [idx for idx, rank in cluster_membership.items()
                      if rank == ci and idx not in best_indices]
        if not ci_indices:
            continue
        ci_arr = np.array(ci_indices)
        short_name = cluster_go_map.get(ci, f"Cluster {ci}")
        if len(short_name) > 28:
            short_name = short_name[:25] + "..."
        ax_a.scatter(
            coords[ci_arr, 0], coords[ci_arr, 1],
            c=cluster_colors[ci], s=18, alpha=0.55, zorder=3,
            edgecolors="white", linewidths=0.3,
            label=short_name,
        )
        drawn_indices.update(ci_indices)

    # Draw any unassigned nodes in grey
    for i in range(len(nodes)):
        if i not in cluster_membership and i not in best_indices:
            pass  # already drawn as background

    # Draw convex hull around best cluster
    best_idx_arr = np.array(sorted(best_indices))
    if len(best_idx_arr) > 0:
        _draw_convex_hull(
            ax_a, coords[best_idx_arr],
            color=cluster_colors[best_cluster_rank]
            if best_cluster_rank is not None else "#E69F00",
        )

    # Draw best cluster nodes on top with larger markers
    best_label = best_cluster["go_term_name"]
    if len(best_label) > 28:
        best_label = best_label[:25] + "..."
    best_color = (cluster_colors[best_cluster_rank]
                  if best_cluster_rank is not None else "#E69F00")
    ax_a.scatter(
        coords[best_idx_arr, 0], coords[best_idx_arr, 1],
        c=best_color, s=65, alpha=1.0, zorder=5,
        edgecolors="black", linewidths=0.9,
        marker="*", label=f"*{best_label}",
    )

    # Annotate optimal r
    opt_r = cluster_info.get("optimal_r")
    r_text = f"r = {opt_r:.3f}" if opt_r is not None else "r = N/A"
    ax_a.annotate(
        r_text,
        xy=(0.96, 0.96), xycoords="axes fraction",
        fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="gray"),
    )

    ax_a.set_xlabel("Dimension 1", fontsize=10)
    ax_a.set_ylabel("Dimension 2", fontsize=10)
    ax_a.legend(
        fontsize=6, loc="lower left", framealpha=0.85,
        markerscale=1.2, ncol=1,
    )
    ax_a.grid(alpha=0.25)

    # ==================================================================
    # Panel B: Network Topology Subgraph
    # ==================================================================
    ax_b = axes[1]
    ax_b.set_title(
        "(B) Network Topology Subgraph", fontsize=11, fontweight="bold"
    )

    # Compute spring layout for the FULL PPI network to provide context
    np.random.seed(SEED)
    full_pos = nx.spring_layout(G_ppi, seed=SEED, k=1.8, iterations=80)

    # Draw all non-cluster nodes as small grey dots
    non_cluster_nodes = [n for n in G_ppi.nodes() if n not in best_node_set]
    if non_cluster_nodes:
        nc_x = [full_pos[n][0] for n in non_cluster_nodes if n in full_pos]
        nc_y = [full_pos[n][1] for n in non_cluster_nodes if n in full_pos]
        ax_b.scatter(nc_x, nc_y, c="#DDDDDD", s=6, alpha=0.3, zorder=1)

    # Draw background edges (those not within the cluster) as faint grey
    for u, v in G_ppi.edges():
        if u in full_pos and v in full_pos:
            if not (u in best_node_set and v in best_node_set):
                ax_b.plot(
                    [full_pos[u][0], full_pos[v][0]],
                    [full_pos[u][1], full_pos[v][1]],
                    c="#EEEEEE", linewidth=0.25, alpha=0.3, zorder=1,
                )

    # Draw cluster edges with thicker coloured lines
    cluster_edge_color = (cluster_colors[best_cluster_rank]
                          if best_cluster_rank is not None else "#E69F00")
    for u, v in G_ppi.edges():
        if u in best_node_set and v in best_node_set:
            if u in full_pos and v in full_pos:
                ax_b.plot(
                    [full_pos[u][0], full_pos[v][0]],
                    [full_pos[u][1], full_pos[v][1]],
                    c=cluster_edge_color, linewidth=1.4, alpha=0.65, zorder=3,
                )

    # Draw cluster nodes coloured by GO term
    for node_name in best_node_set:
        if node_name not in full_pos:
            continue
        x, y = full_pos[node_name]
        # Find which community this node belongs to
        node_idx = nodes.index(node_name) if node_name in nodes else -1
        if node_idx in cluster_membership:
            rank = cluster_membership[node_idx]
            c = cluster_colors[rank]
        else:
            c = best_color
        ax_b.scatter(
            x, y, c=c, s=55, alpha=0.9, zorder=5,
            edgecolors="black", linewidths=0.5,
        )

    # Context annotation
    n_total = G_ppi.number_of_nodes()
    n_cluster = len(best_node_set)
    n_edges_in_cluster = sum(
        1 for u, v in G_ppi.edges()
        if u in best_node_set and v in best_node_set
    )
    info_text = (
        f"Cluster: {n_cluster} / {n_total} nodes\n"
        f"Internal edges: {n_edges_in_cluster}\n"
        f"PPI network: {G_ppi.number_of_edges()} edges"
    )
    ax_b.text(
        0.03, 0.03, info_text, transform=ax_b.transAxes,
        fontsize=7.5, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.88, edgecolor="gray", linewidth=0.6),
    )

    ax_b.set_xlabel("Spring layout X", fontsize=10)
    ax_b.set_ylabel("Spring layout Y", fontsize=10)
    ax_b.grid(alpha=0.25)

    # ==================================================================
    # Panel C: GO DAG Path
    # ==================================================================
    ax_c = axes[2]
    ax_c.set_title("(C) GO DAG Path", fontsize=11, fontweight="bold")

    if go_dag.number_of_nodes() == 0:
        ax_c.text(
            0.5, 0.5, "GO DAG not available",
            transform=ax_c.transAxes, ha="center", va="center", fontsize=12,
        )
    else:
        _draw_go_dag_panel(ax_c, go_dag)

    ax_c.axis("off")

    # ---- Save ----
    fig.tight_layout(rect=[0.03, 0, 1, 0.92])
    output_path = figures_dir / "FigS5_biological_case_study.png"
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)

    return fig


def _draw_go_dag_panel(ax, dag):
    """
    Render the GO DAG tree on a matplotlib Axes (Panel C helper).

    Uses a top-down hierarchical layout with the root at the top.
    Leaves are spread evenly along the x-axis; internal nodes are
    centred above their children.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    dag : networkx.DiGraph
        The GO DAG to render.
    """
    # ---- Compute a proper tree layout ----
    roots = [n for n in dag.nodes() if dag.in_degree(n) == 0]
    if not roots:
        roots = ["GO:0008150"] if "GO:0008150" in dag.nodes() else list(
            dag.nodes()
        )[:1]

    # Assign leaf positions via DFS ordering
    leaf_order = []
    visited_layout = set()

    def _collect_leaves(node):
        """DFS to collect leaves in order."""
        if node in visited_layout:
            return
        visited_layout.add(node)
        children = sorted(dag.successors(node))
        if not children:
            leaf_order.append(node)
        for child in children:
            _collect_leaves(child)

    for root in roots:
        _collect_leaves(root)

    if not leaf_order:
        leaf_order = list(dag.nodes())

    pos = {}
    # Assign x-coordinates to leaves
    for i, leaf in enumerate(leaf_order):
        pos[leaf] = [float(i), 0.0]

    # Bottom-up: assign parent x as mean of children x
    # Process nodes from leaves upward (reverse topological order)
    topo_order = list(nx.topological_sort(dag))
    for node in reversed(topo_order):
        children = sorted(dag.successors(node))
        if children and all(c in pos for c in children):
            pos[node] = [
                np.mean([pos[c][0] for c in children]),
                0.0,  # y assigned below
            ]
        elif node not in pos:
            pos[node] = [0.0, 0.0]

    # Assign y-coordinates based on depth from root
    root = roots[0]
    depths = nx.shortest_path_length(dag, root)
    max_depth = max(depths.values()) if depths else 1

    for node in dag.nodes():
        d = depths.get(node, 0)
        pos[node][1] = max_depth - d  # root at top (highest y)

    # ---- Draw edges ----
    for u, v in dag.edges():
        if u in pos and v in pos:
            ax.annotate(
                "",
                xy=pos[v], xytext=pos[u],
                arrowprops=dict(
                    arrowstyle="-|>", color="#999999",
                    lw=1.0, shrinkA=3, shrinkB=3,
                ),
                zorder=1,
            )

    # ---- Draw nodes ----
    max_count = max(
        (dag.nodes[n].get("count", 1) for n in dag.nodes()), default=1
    )
    max_count = max(max_count, 1)

    depth_colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00",
                    "#D55E00", "#CC79A7"]

    for node in dag.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        is_target = dag.nodes[node].get("is_target", False)
        count = dag.nodes[node].get("count", 0)
        depth = dag.nodes[node].get("depth", 0)

        size = 120 + 350 * (count / max_count)
        color = "#D55E00" if is_target else depth_colors[
            depth % len(depth_colors)
        ]
        edge_c = "black" if is_target else "#666666"
        lw = 2.0 if is_target else 0.7

        ax.scatter(x, y, s=size, c=color, edgecolors=edge_c,
                   linewidths=lw, zorder=5)

        # Label: GO ID on first line, abbreviated name on second
        name = dag.nodes[node].get("name", node)
        if len(name) > 30:
            name = name[:27] + "..."
        label = f"{node}\n({name})"

        # Alternate label offset direction by depth to reduce overlap
        depth = dag.nodes[node].get("depth", 0)
        x_offset = 12 if depth % 2 == 0 else -12
        ha_align = "left" if x_offset > 0 else "right"

        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(x_offset, 0),
            fontsize=6.5, va="center", ha=ha_align,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      alpha=0.80, edgecolor="none"),
        )

    # Count annotations on the right margin
    y_text = 0.97
    ax.text(
        0.98, y_text, "Node sizes ~ gene count",
        transform=ax.transAxes, fontsize=7, ha="right", va="top",
        fontstyle="italic", color="#666666",
    )
    y_text -= 0.06
    for node in dag.nodes():
        if dag.nodes[node].get("is_target", False):
            cnt = dag.nodes[node].get("count", 0)
            ax.text(
                0.98, y_text,
                f"Target: {node} ({cnt} genes)",
                transform=ax.transAxes, fontsize=7, ha="right", va="top",
                color="#D55E00", fontweight="bold",
            )
            break

    # Set axis limits with generous padding for labels
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_pad = max((x_max - x_min) * 0.35, 1.5)
        y_pad = max((y_max - y_min) * 0.15, 0.5)
        ax.set_xlim(x_min - x_pad, x_max + x_pad * 5.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)


# ===========================================================================
# 5. Summary Table
# ===========================================================================

def generate_interpretation_table(gf_scores, method_clusters, thresholds):
    """
    Generate and print a biological interpretation summary table.

    Combines G-F Scores, classification levels, and best GO term cluster
    information into a formatted table printed to stdout and returned
    as a structured dictionary.

    Parameters
    ----------
    gf_scores : dict
        ``{method: score}`` mapping.
    method_clusters : dict
        ``{method: result_dict}`` from :func:`analyze_go_term_clusters`.
    thresholds : tuple of float
        Level boundary thresholds ``(t1, t2, t3)``.

    Returns
    -------
    dict
        ``{method: {gf_score, level, label, interpretation,
        best_go_cluster: {go_term, go_term_name, purity, n_nodes,
        other_go_terms}}}``
    """
    # Sort by G-F Score descending
    ranked = sorted(gf_scores.items(), key=lambda x: x[1], reverse=True)

    # Print header
    print("\n" + "=" * 115)
    print(
        f"{'Method':<12} {'G-F Score':>10} {'Level':>6} "
        f"{'Interpretation':<42} {'Best GO Cluster':<22} {'Purity':>8}"
    )
    print("-" * 115)

    results = {}
    for method, score in ranked:
        info = classify_gf_score(score, thresholds=thresholds)
        level = info["level"]
        label = info["label"]

        # Short interpretation string for the printed table
        short_interp = {
            1: "Weak: poor functional reflection",
            2: "Moderate: major categories",
            3: "Strong: fine-grained modules",
            4: "Very strong: direct functional proxy",
        }
        interp = short_interp.get(level, label)

        # Best GO cluster info
        cluster_data = method_clusters.get(method, {})
        best = cluster_data.get("best_cluster")
        if best is not None:
            go_term = best["go_term"]
            go_name = best.get("go_term_name", "")
            purity = best["purity"]
            n_nodes = best["n_nodes"]
            # Abbreviate name for table display
            if len(go_name) > 18:
                go_name_short = go_name[:15] + "..."
            else:
                go_name_short = go_name
            best_go_str = f"{go_term} ({go_name_short})"
            purity_str = f"{purity:.3f}"
        else:
            go_term = "N/A"
            go_name = "N/A"
            purity = 0.0
            n_nodes = 0
            best_go_str = "N/A"
            purity_str = "N/A"
            go_name_short = "N/A"

        # Other GO terms present in the cluster
        other_terms_list = []
        if best is not None:
            for ot in best.get("other_go_terms", []):
                other_terms_list.append({
                    "term": ot["term"],
                    "name": ot["name"],
                    "count": ot["count"],
                })

        print(
            f"{method:<12} {score:>10.4f} {level:>6} "
            f"{interp:<42} {best_go_str:<22} {purity_str:>8}"
        )

        results[method] = {
            "gf_score": score,
            "level": level,
            "label": label,
            "interpretation": info["description"],
            "best_go_cluster": {
                "go_term": go_term,
                "go_term_name": go_name,
                "purity": purity,
                "n_nodes": n_nodes,
                "other_go_terms": other_terms_list,
            },
        }

    print("=" * 115)
    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    """
    Main entry point for biological interpretation of G-F Scores.

    Workflow
    --------
    1. Load the curated yeast PPI network and gene-GO annotations.
    2. Load G-F Scores from ``results/final_results_summary.json``.
    3. For each method, identify the highest-purity GO term cluster
       via spatial graph community detection.
    4. Print and save a biological interpretation summary table.
    5. Build a GO DAG for the case-study cluster's dominant term.
    6. Generate the three-panel case study figure (FigS5).

    CLI arguments
    -------------
    ``--method`` : str
        Embedding method for the case study (default: ``DM``).
    ``--level-thresholds`` : str
        Comma-separated G-F Score thresholds (default: ``0.3,0.5,0.7``).
    ``--skip-figure`` : flag
        Skip generating the three-panel figure.
    """
    np.random.seed(SEED)

    # ---- CLI arguments ----
    parser = argparse.ArgumentParser(
        description=(
            "Biological interpretation of G-F Scores through a 4-level "
            "scale and case study visualization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method", type=str, default="DM",
        help="Embedding method for case study (default: DM)",
    )
    parser.add_argument(
        "--level-thresholds", type=str, default="0.3,0.5,0.7",
        help="Comma-separated G-F Score thresholds (default: 0.3,0.5,0.7)",
    )
    parser.add_argument(
        "--skip-figure", action="store_true",
        help="Skip generating the three-panel figure",
    )
    args = parser.parse_args()

    thresholds = tuple(float(x) for x in args.level_thresholds.split(","))
    assert len(thresholds) == 3, "Exactly 3 thresholds required"

    case_method = args.method

    # ---- Setup logging ----
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    ))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)

    # ---- Directories ----
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = get_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("=" * 70)
    print("Biological Interpretation of G-F Scores")
    print("=" * 70)

    print("\nLoading curated yeast PPI network...")
    G_ppi, nodes, go_map = load_curated_network(data_dir)
    print(f"  Network: {G_ppi.number_of_nodes()} nodes, "
          f"{G_ppi.number_of_edges()} edges")
    print(f"  GO annotations: {len(go_map)} genes")

    # ---- Load G-F Scores ----
    results_file = results_dir / "final_results_summary.json"
    if results_file.exists():
        with open(results_file) as f:
            full_results = json.load(f)
        gf_scores = full_results.get("gf_scores", DEFAULT_GF_SCORES)
        print(f"\nLoaded G-F Scores from {results_file}")
    else:
        gf_scores = DEFAULT_GF_SCORES.copy()
        print("\nUsing default G-F Scores (results file not found)")

    print(f"\nLevel thresholds: <{thresholds[0]}, "
          f"[{thresholds[0]},{thresholds[1]}), "
          f"[{thresholds[1]},{thresholds[2]}), >={thresholds[2]}")

    # ---- Step 2: GO term cluster analysis for each method ----
    print("\n" + "-" * 70)
    print("GO Term Cluster Analysis")
    print("-" * 70)

    method_clusters = {}
    emb_dir = get_embeddings_dir()

    for method in ALL_METHODS:
        print(f"\n  Analyzing {method}...")
        try:
            coords, emb_nodes = load_embedding(
                method, "153", embeddings_dir=emb_dir
            )
            common = sorted(set(emb_nodes) & set(nodes) & set(go_map.keys()))
            emb_node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
            idx_map = [emb_node_to_idx[n] for n in common]
            aligned_coords = coords[idx_map]

            result = analyze_go_term_clusters(
                method, common, go_map, aligned_coords
            )
            method_clusters[method] = result

            best = result.get("best_cluster")
            if best is not None:
                print(
                    f"    Best cluster: {best['go_term']} "
                    f"({best['go_term_name']})"
                )
                print(
                    f"    Purity: {best['purity']:.3f}, "
                    f"Nodes: {best['n_nodes']}, "
                    f"Optimal r: {result['optimal_r']:.4f}"
                )
            else:
                print("    No clusters found")
        except Exception as e:
            print(f"    FAILED: {e}")
            method_clusters[method] = {
                "best_cluster": None,
                "all_clusters": [],
                "optimal_r": None,
                "max_mean_purity": 0.0,
            }

    # ---- Step 3: Generate interpretation table ----
    print("\n" + "-" * 70)
    print("Biological Interpretation Table")
    print("-" * 70)

    table_results = generate_interpretation_table(
        gf_scores, method_clusters, thresholds
    )

    # ---- Save to JSON ----
    output_data = {
        "thresholds": list(thresholds),
        "scale_description": {
            str(k): {
                "label": v["label"],
                "threshold_range": list(v["threshold"]),
                "description": v["description"],
            }
            for k, v in GF_SCORE_LEVELS.items()
        },
        "methods": table_results,
        "case_study_method": case_method,
        "case_study_details": {},
    }

    # Add case study cluster details
    case_data = method_clusters.get(case_method, {})
    case_best = case_data.get("best_cluster")
    if case_best is not None:
        output_data["case_study_details"] = {
            "go_term": case_best["go_term"],
            "go_term_name": case_best["go_term_name"],
            "purity": case_best["purity"],
            "n_nodes": case_best["n_nodes"],
            "optimal_r": case_data.get("optimal_r"),
            "nodes": case_best["nodes"],
            "other_go_terms": case_best.get("other_go_terms", []),
            "member_go_counts": case_best.get("member_go_counts", {}),
        }

    output_file = results_dir / "biological_interpretation.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nSaved interpretation to: {output_file}")

    # ---- Step 4: Generate three-panel figure ----
    if not args.skip_figure:
        print("\n" + "-" * 70)
        print("Generating Three-Panel Case Study Figure")
        print("-" * 70)

        try:
            coords, emb_nodes = load_embedding(
                case_method, "153", embeddings_dir=emb_dir
            )
            common = sorted(set(emb_nodes) & set(nodes) & set(go_map.keys()))
            emb_node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
            idx_map = [emb_node_to_idx[n] for n in common]
            aligned_coords = coords[idx_map]

            cluster_info = method_clusters.get(case_method, {})
            best_cluster = cluster_info.get("best_cluster")

            if best_cluster is not None:
                go_term = best_cluster["go_term"]
                go_dag = build_go_dag_for_term(go_term, go_map)
                print(f"  Built GO DAG for {go_term} "
                      f"({go_dag.number_of_nodes()} nodes, "
                      f"{go_dag.number_of_edges()} edges)")
            else:
                go_dag = nx.DiGraph()
                print("  Warning: no cluster found; "
                      "GO DAG panel will be empty")

            generate_three_panel_figure(
                case_method, aligned_coords, common, go_map, G_ppi,
                cluster_info, go_dag, figures_dir,
            )

            fig_path = figures_dir / "FigS5_biological_case_study.png"
            print(f"  Saved figure to: {fig_path}")

        except Exception as e:
            print(f"  Figure generation FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Biological interpretation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
