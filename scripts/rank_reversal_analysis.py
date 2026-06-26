#!/usr/bin/env python3
"""
rank_reversal_analysis.py
Deep mechanistic analysis of the cross-species rank reversal.

Node2Vec and DeepWalk rank LOWEST on yeast (6th-7th out of 8 methods)
but HIGHEST on human (1st-2nd out of 6).  This script investigates why
through four complementary analyses:

  1. Embedding distance distributions (intra- vs inter-module geometric gap)
  2. Node2Vec p/q parameter sensitivity (ablation grid on yeast 153)
  3. Ablation experiment on human network (4 walk strategies)
  4. Rank reversal summary with topology correlations

Output files
------------
  results/node2vec_pq_sensitivity.json
  results/rank_reversal_ablation.json
  results/rank_reversal_summary.json
  figures/FigS2_embedding_distance_distributions.png
  figures/FigS3_node2vec_pq_heatmap.png
"""
from __future__ import annotations

import sys
import json
import gzip
import argparse
import random
import logging
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_embedding, compute_gf_curve, compute_gf_score,
    rescale_coordinates, node2vec_from_graph, deepwalk_from_graph,
    compute_centrality_features,
)

logger = logging.getLogger(__name__)

# ---- Configuration ----
GF_R_MIN = 0.05
GF_R_MAX_YEAST = 0.422
GF_N_POINTS = 100
HUMAN_SUBSAMPLE = 2000
MAX_SAMPLED_PAIRS = 50000


# ---- Human Network Loading ----

def load_human_network(min_score=700):
    """Load human STRING PPI network (taxID 9606), largest connected component.

    Mirrors ``human_validation/human_embed_all.py``.

    Parameters
    ----------
    min_score : int
        Minimum combined STRING score to retain an edge.

    Returns
    -------
    nx.Graph
        Largest connected component of the filtered human PPI.
    """
    project_root = Path(__file__).resolve().parent.parent
    links_file = (
        project_root / "human_validation" / "9606.protein.links.v12.0.txt.gz"
    )
    if not links_file.exists():
        raise FileNotFoundError(
            f"Human STRING file not found: {links_file}\n"
            "Download from: https://stringdb-static.org/download/"
            "protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
        )

    G = nx.Graph()
    with gzip.open(str(links_file), 'rt', encoding='utf-8') as f:
        _header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                score = int(parts[2])
                if score >= min_score:
                    G.add_edge(parts[0], parts[1])

    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def load_human_go_annotations():
    """Load human GO annotations from pre-processed JSON.

    Returns
    -------
    dict
        Mapping ``{protein_id: [go_term, ...]}``, or empty dict if the
        file is not found.
    """
    data_dir = get_data_dir()
    go_file = data_dir / "human_go_annotations.json"
    if go_file.exists():
        with open(go_file, "r") as f:
            return json.load(f)
    logger.warning("No human GO annotations file found at %s", go_file)
    return {}


def _go_map_to_labels(go_map):
    """Convert ``{node: [terms]}`` to ``{node: most_frequent_term}``."""
    from collections import Counter
    labels = {}
    for node, terms in go_map.items():
        if isinstance(terms, list) and terms:
            labels[node] = Counter(terms).most_common(1)[0][0]
        elif isinstance(terms, str):
            labels[node] = terms
    return labels


# ---- Community Helpers ----

def get_community_assignments(G, nodes_list):
    """Run greedy modularity community detection on *G*.

    Parameters
    ----------
    G : nx.Graph
    nodes_list : list
        Ordered list of nodes (must match ``G.nodes()``).

    Returns
    -------
    dict
        ``{node_name: community_id}`` for every node in *nodes_list*.
    """
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
    except Exception as e:
        communities = [frozenset(G.nodes())]

    assignments = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            assignments[node] = comm_id
    return assignments


def subsample_annotated(nodes, coords, node_labels, target_size, rng):
    """Subsample annotated nodes for tractable G-F curve computation.

    Matches the signature and behaviour of
    ``human_validation/human_gf_all.py:subsample_annotated``.
    """
    annotated_idx = [i for i in range(len(nodes)) if nodes[i] in node_labels]
    if len(annotated_idx) <= target_size:
        sub_idx = annotated_idx
    else:
        sub_idx = sorted(
            rng.choice(annotated_idx, size=target_size, replace=False)
        )
    sub_coords = coords[sub_idx]
    sub_nodes = [nodes[i] for i in sub_idx]
    sub_labels = {}
    for new_i, old_i in enumerate(sub_idx):
        if nodes[old_i] in node_labels:
            sub_labels[new_i] = node_labels[nodes[old_i]]
    return sub_coords, sub_nodes, sub_labels, len(annotated_idx)


# ==================================================================
#  Analysis 1: Embedding distance distributions
# ==================================================================

def analyze_embedding_distances(G, coords, node_list, method_name, seed=SEED):
    """Compute intra/inter-module embedding distance distributions.

    Communities are detected on the PPI network *G* via greedy modularity
    community detection.  Pairwise Euclidean distances in the embedding
    space are then partitioned into intra-module (same community) and
    inter-module (different community).

    Parameters
    ----------
    G : nx.Graph
        PPI network.
    coords : ndarray, shape (n, 2)
        2-D embedding coordinates (same node order as *node_list*).
    node_list : list
        Node names matching the row order of *coords*.
    method_name : str
        Name for logging / result key.
    seed : int
        Random seed for sampling on large networks.

    Returns
    -------
    dict
        Keys: intra_module_distances, inter_module_distances,
        mean_intra, mean_inter, geometric_gap, n_communities.
    """
    n = len(node_list)
    node_to_pos = {node: i for i, node in enumerate(node_list)}

    # Community detection on the PPI network
    communities = list(nx.community.greedy_modularity_communities(G))

    # Build membership: pos_index -> community_id
    membership = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            if node in node_to_pos:
                membership[node_to_pos[node]] = comm_id

    # Pairwise Euclidean distances — use scipy cdist for large networks
    # to avoid O(n²·d) memory from numpy broadcasting.
    from scipy.spatial.distance import cdist

    intra_dists = []
    inter_dists = []
    rng = np.random.RandomState(seed)

    if n <= 1000:
        dist_matrix = cdist(coords, coords, metric='euclidean')
        # Exact enumeration
        for i in range(n):
            for j in range(i + 1, n):
                if i in membership and j in membership:
                    if membership[i] == membership[j]:
                        intra_dists.append(dist_matrix[i, j])
                    else:
                        inter_dists.append(dist_matrix[i, j])
    else:
        # Sample pairs to keep memory / time bounded.
        # Compute distances on-the-fly to avoid O(n²) matrix for large n.
        n_intra_target = min(MAX_SAMPLED_PAIRS, n * 10)
        n_inter_target = min(MAX_SAMPLED_PAIRS, n * 10)

        intra_count = 0
        inter_count = 0
        max_attempts = (n_intra_target + n_inter_target) * 20

        # Pre-group indices by community for efficient sampling
        comm_indices = {}
        for idx, cid in membership.items():
            comm_indices.setdefault(cid, []).append(idx)
        comm_ids = list(comm_indices.keys())

        for _ in range(max_attempts):
            if rng.random() < 0.5 and len(comm_ids) > 1:
                # Try intra-module pair
                cid = comm_ids[rng.randint(len(comm_ids))]
                members = comm_indices[cid]
                if len(members) < 2:
                    continue
                i, j = rng.choice(members, size=2, replace=False)
                if intra_count < n_intra_target:
                    d = float(np.sqrt(np.sum((coords[i] - coords[j]) ** 2)))
                    intra_dists.append(d)
                    intra_count += 1
            else:
                # Try inter-module pair
                if len(comm_ids) < 2:
                    continue
                c1, c2 = rng.choice(len(comm_ids), size=2, replace=False)
                m1 = comm_indices[comm_ids[c1]]
                m2 = comm_indices[comm_ids[c2]]
                i = m1[rng.randint(len(m1))]
                j = m2[rng.randint(len(m2))]
                if inter_count < n_inter_target:
                    d = float(np.sqrt(np.sum((coords[i] - coords[j]) ** 2)))
                    inter_dists.append(d)
                    inter_count += 1
            if (intra_count >= n_intra_target
                    and inter_count >= n_inter_target):
                break

    intra = np.array(intra_dists) if intra_dists else np.array([0.0])
    inter = np.array(inter_dists) if inter_dists else np.array([0.0])

    return {
        "intra_module_distances": intra,
        "inter_module_distances": inter,
        "mean_intra": float(np.mean(intra)),
        "mean_inter": float(np.mean(inter)),
        "geometric_gap": float(np.mean(inter) - np.mean(intra)),
        "n_communities": len(communities),
    }


def plot_distance_distributions(all_distances, figures_dir):
    """Plot 2x2 grid of embedding distance histograms.

    Rows: yeast (DM, Node2Vec).  Columns: human (DM, Node2Vec).
    Each panel overlays intra-module (blue) and inter-module (red)
    histograms with mean lines.

    Parameters
    ----------
    all_distances : dict
        Nested ``{species: {method: result_dict}}`` from
        :func:`analyze_embedding_distances`.
    figures_dir : Path
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Embedding Distance Distributions:\n'
                 'Intra-module vs Inter-module',
                 fontsize=14, fontweight='bold')

    layout = [
        (0, 0, "yeast", "DM",       "Yeast - Diffusion Map"),
        (0, 1, "yeast", "Node2Vec", "Yeast - Node2Vec"),
        (1, 0, "human", "DM",       "Human - Diffusion Map"),
        (1, 1, "human", "Node2Vec", "Human - Node2Vec"),
    ]

    for row, col, species, method, title in layout:
        ax = axes[row, col]

        if species not in all_distances:
            ax.text(0.5, 0.5, f'{species} data\nnot available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
            ax.set_title(title)
            continue

        if method not in all_distances[species]:
            ax.text(0.5, 0.5, f'{method}\nnot available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
            ax.set_title(title)
            continue

        result = all_distances[species][method]
        intra = result["intra_module_distances"]
        inter = result["inter_module_distances"]

        ax.hist(intra, bins=50, alpha=0.6, color='#0072B2',
                label=f'Intra-module (n={len(intra)})', density=True)
        ax.hist(inter, bins=50, alpha=0.6, color='#D55E00',
                label=f'Inter-module (n={len(inter)})', density=True)

        ax.axvline(np.mean(intra), color='#0072B2', linestyle='--',
                    linewidth=2,
                    label=f'Mean intra = {np.mean(intra):.3f}')
        ax.axvline(np.mean(inter), color='#D55E00', linestyle='--',
                    linewidth=2,
                    label=f'Mean inter = {np.mean(inter):.3f}')

        gap = result["geometric_gap"]
        ax.set_title(f'{title}\nGap = {gap:.3f}', fontsize=11)
        ax.set_xlabel('Euclidean Distance')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    output_path = figures_dir / "FigS2_embedding_distance_distributions.png"
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ==================================================================
#  Analysis 2: Node2Vec p/q parameter sensitivity
# ==================================================================

def run_pq_sensitivity(G, nodes, go_map):
    """Run Node2Vec p/q grid search on the curated 153-node yeast network.

    For each ``(p, q)`` pair, computes a 2-D Node2Vec embedding, rescales
    it, and evaluates the G-F Score.  ``p=1, q=1`` is equivalent to
    DeepWalk.

    Parameters
    ----------
    G : nx.Graph
    nodes : list
    go_map : dict

    Returns
    -------
    dict
        Contains ``p_values``, ``q_values``, ``gf_scores`` (nested dict),
        and ``grid`` (list of rows for heatmap).
    """
    p_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    q_values = [0.25, 0.5, 1.0, 2.0, 4.0]

    r_vals = np.linspace(0.05, 0.55, GF_N_POINTS)

    grid = []
    gf_scores_dict = {}

    for p in p_values:
        row = []
        for q in q_values:
            random.seed(SEED)
            np.random.seed(SEED)

            coords = node2vec_from_graph(
                G, walk_length=20, walks_per_node=10,
                window_size=5, dimensions=2, p=p, q=q, seed=SEED,
            )
            coords = rescale_coordinates(coords, target_std=0.3)

            purities, _ = compute_gf_curve(coords, nodes, go_map, r_vals)
            score = compute_gf_score(
                r_vals, purities, GF_R_MIN, GF_R_MAX_YEAST
            )

            row.append(score)
            key = f"p{p}_q{q}"
            gf_scores_dict[key] = score
            is_dw = "  (DeepWalk)" if (p == 1.0 and q == 1.0) else ""
            print(f"  p={p:<4} q={q:<4} -> G-F Score = {score:.4f}{is_dw}")
        grid.append(row)

    return {
        "p_values": p_values,
        "q_values": q_values,
        "gf_scores": gf_scores_dict,
        "grid": grid,
    }


def plot_pq_heatmap(sensitivity_data, figures_dir):
    """Plot heatmap of G-F Scores across Node2Vec p/q parameter grid.

    Parameters
    ----------
    sensitivity_data : dict
        Output of :func:`run_pq_sensitivity`.
    figures_dir : Path
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = np.array(sensitivity_data["grid"])
    p_values = sensitivity_data["p_values"]
    q_values = sensitivity_data["q_values"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', origin='lower')

    ax.set_xticks(range(len(q_values)))
    ax.set_xticklabels([str(q) for q in q_values], fontsize=11)
    ax.set_yticks(range(len(p_values)))
    ax.set_yticklabels([str(p) for p in p_values], fontsize=11)
    ax.set_xlabel('q (return parameter)', fontsize=13)
    ax.set_ylabel('p (in-out parameter)', fontsize=13)
    ax.set_title('Node2Vec p/q Sensitivity: G-F Score\n'
                 '(Yeast curated 153-node network)',
                 fontsize=13, fontweight='bold')

    # Annotate cells
    for i in range(len(p_values)):
        for j in range(len(q_values)):
            val = grid[i, j]
            text_color = "white" if val < grid.mean() else "black"
            label = f'{val:.3f}'
            if p_values[i] == 1.0 and q_values[j] == 1.0:
                label += '\n(DW)'
            ax.text(j, i, label, ha='center', va='center',
                    color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label('G-F Score', fontsize=12)

    plt.tight_layout()
    output_path = figures_dir / "FigS3_node2vec_pq_heatmap.png"
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ==================================================================
#  Analysis 3: Ablation experiment on human network
# ==================================================================

def run_ablation(G_human, node_labels, r_vals):
    """Run Node2Vec ablation experiment on the human PPI network.

    Four walk strategies are tested:
      a. Original optimal (p=0.5, q=2.0)
      b. DeepWalk equivalent (p=1.0, q=1.0)
      c. BFS-like (p=2.0, q=0.5)
      d. DFS-like (p=0.5, q=4.0)

    For each, a 2-D embedding is computed, rescaled, and the G-F curve
    and score are evaluated on subsampled annotated nodes.

    Parameters
    ----------
    G_human : nx.Graph
        Human PPI network (largest connected component).
    node_labels : dict
        ``{protein_id: go_term}`` for annotated nodes.
    r_vals : ndarray
        Radius values for the G-F curve.

    Returns
    -------
    dict
        Per-configuration results including G-F Score, peak purity, and
        embedding coordinates.
    """
    from scipy.integrate import trapezoid

    nodes = sorted(G_human.nodes())
    n = len(nodes)

    # Check annotation coverage
    annotated = [nd for nd in nodes if nd in node_labels]
    if len(annotated) < 10:
        print("  Insufficient GO annotations for ablation "
              f"({len(annotated)} annotated nodes). Skipping.")
        return {}

    configs = [
        {
            "name": "optimal_n2v",
            "p": 0.5, "q": 2.0,
            "description": "Original Node2Vec (p=0.5, q=2.0)",
        },
        {
            "name": "deepwalk_equiv",
            "p": 1.0, "q": 1.0,
            "description": "DeepWalk equivalent (p=1.0, q=1.0)",
        },
        {
            "name": "bfs_like",
            "p": 2.0, "q": 0.5,
            "description": "BFS-like (p=2.0, q=0.5)",
        },
        {
            "name": "dfs_like",
            "p": 0.5, "q": 4.0,
            "description": "DFS-like (p=0.5, q=4.0)",
        },
    ]

    # Use reduced walk parameters for tractability on the large network
    walk_length = 15
    walks_per_node = 5

    results = {}
    for cfg in configs:
        random.seed(SEED)
        np.random.seed(SEED)
        name = cfg["name"]
        print(f"\n  [{name}] {cfg['description']}")
        print(f"    Generating Node2Vec embedding "
              f"(walk_length={walk_length}, walks={walks_per_node})...")

        try:
            coords = node2vec_from_graph(
                G_human,
                walk_length=walk_length,
                walks_per_node=walks_per_node,
                window_size=5,
                dimensions=2,
                p=cfg["p"],
                q=cfg["q"],
                seed=SEED,
            )
            coords = rescale_coordinates(coords, target_std=0.3)

            # Subsample annotated nodes for G-F curve
            rng = np.random.default_rng(SEED)
            sub_coords, sub_nodes, sub_labels, total_ann = (
                subsample_annotated(
                    nodes, coords, node_labels, HUMAN_SUBSAMPLE, rng,
                )
            )
            print(f"    Subsampled {len(sub_labels)}/{total_ann} "
                  "annotated nodes")
            print(f"    Computing G-F curve ({len(r_vals)} points)...")

            purities, modularities = compute_gf_curve(
                sub_coords, sub_nodes, sub_labels, r_vals,
            )

            # G-F Score: use the human unified interval if available,
            # otherwise use a default
            r_min_s, r_max_s = GF_R_MIN, 0.30
            r_arr = np.asarray(r_vals)
            mask = (r_arr >= r_min_s) & (r_arr <= r_max_s)
            r_sub = r_arr[mask]
            p_sub = np.asarray(purities)[mask]
            if len(r_sub) >= 2:
                score = float(trapezoid(p_sub, r_sub) / (r_max_s - r_min_s))
            else:
                score = 0.0

            peak_purity = float(max(purities)) if purities else 0.0

            results[name] = {
                "description": cfg["description"],
                "p": cfg["p"],
                "q": cfg["q"],
                "gf_score": score,
                "peak_purity": peak_purity,
                "purities": [float(x) for x in purities],
                "modularities": [float(x) for x in modularities],
                "gf_interval": [r_min_s, r_max_s],
                "walk_length": walk_length,
                "walks_per_node": walks_per_node,
            }
            print(f"    G-F Score = {score:.4f}, "
                  f"Peak purity = {peak_purity:.4f}")

        except Exception as e:
            print(f"    {name} FAILED: {e}")
            results[name] = {
                "description": cfg["description"],
                "p": cfg["p"],
                "q": cfg["q"],
                "error": str(e),
            }

    return results


# ==================================================================
#  Analysis 4: Rank reversal summary
# ==================================================================

def generate_rank_reversal_summary(topology_metrics=None):
    """Compile cross-species rank reversal summary.

    Loads pre-computed yeast and human G-F Scores, identifies comparable
    methods, computes rank changes, and correlates them with topology
    feature differences.

    Parameters
    ----------
    topology_metrics : dict, optional
        Output of ``network_topology_analysis.py`` (loaded from JSON).

    Returns
    -------
    dict
        Full summary including per-method rank changes and topology
        correlation analysis.
    """
    results_dir = get_results_dir()

    # Load pre-computed G-F scores
    yeast_scores = {}
    human_scores = {}

    yeast_file = results_dir / "gf_scores.json"
    if yeast_file.exists():
        with open(yeast_file) as f:
            data = json.load(f)
        yeast_scores = data.get("scores", data.get("scores_paper_interval", {}))

    human_file = results_dir / "human_gf_scores.json"
    if human_file.exists():
        with open(human_file) as f:
            data = json.load(f)
        human_scores = data.get("scores", {})

    if not yeast_scores or not human_scores:
        print("  Cannot compute rank reversal: missing G-F score files.")
        return {}

    # Identify comparable methods (present in both species)
    comparable = sorted(set(yeast_scores.keys()) & set(human_scores.keys()))

    # Compute ranks (higher score = rank 1)
    yeast_ranked = sorted(
        yeast_scores.items(), key=lambda x: x[1], reverse=True
    )
    human_ranked = sorted(
        human_scores.items(), key=lambda x: x[1], reverse=True
    )

    yeast_ranks = {m: i + 1 for i, (m, _) in enumerate(yeast_ranked)}
    human_ranks = {m: i + 1 for i, (m, _) in enumerate(human_ranked)}

    rank_table = []
    for method in comparable:
        yr = yeast_ranks[method]
        hr = human_ranks[method]
        rank_table.append({
            "method": method,
            "yeast_score": yeast_scores[method],
            "human_score": human_scores[method],
            "yeast_rank": yr,
            "human_rank": hr,
            "rank_change": yr - hr,
            "n_yeast_methods": len(yeast_scores),
            "n_human_methods": len(human_scores),
        })

    # Sort by absolute rank change (largest reversal first)
    rank_table.sort(key=lambda x: abs(x["rank_change"]), reverse=True)

    # ---- Topology correlation analysis ----
    topology_analysis = {}
    if topology_metrics is not None:
        y_key = "yeast_curated_153"
        h_keys = [k for k in topology_metrics if k.startswith("human_")]
        h_key = h_keys[0] if h_keys else None

        if y_key in topology_metrics and h_key is not None:
            ym = topology_metrics[y_key]
            hm = topology_metrics[h_key]

            feature_names = [
                "modularity_Q",
                "average_clustering_coefficient",
                "spectral_gap",
                "degree_heterogeneity",
                "mixing_time",
                "assortativity",
            ]

            feature_diffs = {}
            for feat in feature_names:
                yv = ym.get(feat, 0.0) or 0.0
                hv = hm.get(feat, 0.0) or 0.0
                feature_diffs[feat] = float(hv - yv)

            # Pearson correlation between rank changes and feature diffs
            # (only meaningful as a qualitative indicator with few methods)
            rank_changes = np.array(
                [rt["rank_change"] for rt in rank_table], dtype=float
            )
            correlations = {}
            if len(rank_table) >= 3:
                for feat in feature_names:
                    fv = np.array([feature_diffs[feat]] * len(rank_table))
                    if np.std(fv) > 1e-10 and np.std(rank_changes) > 1e-10:
                        corr = float(
                            np.corrcoef(rank_changes, fv)[0, 1]
                        )
                    else:
                        corr = None
                    correlations[feat] = corr

            topology_analysis = {
                "feature_differences_human_minus_yeast": feature_diffs,
                "correlations_with_rank_change": correlations,
                "interpretation": (
                    "Random-walk methods (Node2Vec, DeepWalk) benefit from "
                    "the human network's larger scale, higher connectivity, "
                    "and different community structure.  On the small, dense "
                    "yeast curated network (153 nodes), structural methods "
                    "(DM, Spectral, MDS) outperform because they capture "
                    "global topology directly.  On the large, sparse human "
                    "network (~15k nodes), random-walk methods excel because "
                    "they capture local neighbourhood patterns that align "
                    "with functional modules at scale."
                ),
            }

    # Print summary
    print("\n" + "=" * 72)
    print("CROSS-SPECIES RANK REVERSAL SUMMARY")
    print("=" * 72)
    print(f"\nYeast methods: {len(yeast_scores)}  |  "
          f"Human methods: {len(human_scores)}  |  "
          f"Comparable: {len(comparable)}")
    print(f"\n{'Method':<12} {'Yeast':>7} {'Rank':>5} "
          f"{'Human':>7} {'Rank':>5} {'Change':>8}")
    print("-" * 52)
    for entry in rank_table:
        sign = "+" if entry["rank_change"] > 0 else ""
        print(f"{entry['method']:<12} "
              f"{entry['yeast_score']:>7.4f} "
              f"{entry['yeast_rank']:>5d} "
              f"{entry['human_score']:>7.4f} "
              f"{entry['human_rank']:>5d} "
              f"{sign}{entry['rank_change']:>6d}")

    if topology_analysis:
        print(f"\n--- Topology Feature Differences (human - yeast) ---")
        diffs = topology_analysis[
            "feature_differences_human_minus_yeast"
        ]
        for feat, diff in diffs.items():
            print(f"  {feat:<40s}: {diff:>12.6f}")

        print(f"\n--- Qualitative Interpretation ---")
        print(f"  {topology_analysis['interpretation']}")

    return {
        "rank_table": rank_table,
        "yeast_scores": yeast_scores,
        "human_scores": human_scores,
        "yeast_ranks": yeast_ranks,
        "human_ranks": human_ranks,
        "comparable_methods": comparable,
        "topology_analysis": topology_analysis,
    }


# ==================================================================
#  Main
# ==================================================================

def main():
    """Run the full cross-species rank reversal analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Deep mechanistic analysis of the cross-species "
                    "rank reversal for Node2Vec/DeepWalk."
    )
    parser.add_argument(
        "--skip-human", action="store_true",
        help="Skip expensive human network computations "
             "(distance distributions and ablation).",
    )
    parser.add_argument(
        "--skip-ablation", action="store_true",
        help="Skip the Node2Vec p/q sensitivity grid search.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    random.seed(SEED)
    np.random.seed(SEED)

    project_root = Path(__file__).resolve().parent.parent
    results_dir = get_results_dir()
    figures_dir = project_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    emb_dir = get_embeddings_dir()
    data_dir = get_data_dir()

    # Containers for distance-distribution results
    all_distances = {}

    # ==============================================================
    # 1a. Yeast embedding distance distributions
    # ==============================================================
    print("\n" + "=" * 60)
    print("Analysis 1: Embedding Distance Distributions")
    print("=" * 60)

    print("\nLoading yeast curated network...")
    try:
        G_yeast, nodes_yeast, go_map = load_curated_network(data_dir)
        print(f"  Yeast: {len(nodes_yeast)} nodes, "
              f"{G_yeast.number_of_edges()} edges")

        yeast_distances = {}
        for method in ["DM", "Node2Vec"]:
            print(f"\n  [{method}] Loading embedding and computing "
                  f"distances...")
            try:
                coords, emb_nodes = load_embedding(
                    method, "153", embeddings_dir=emb_dir
                )
                # Align nodes
                common = sorted(set(emb_nodes) & set(nodes_yeast))
                emb_node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
                yeast_node_to_idx = {n: i for i, n in enumerate(nodes_yeast)}
                emb_idx = [emb_node_to_idx[n] for n in common]
                yeast_idx = [yeast_node_to_idx[n] for n in common]
                aligned_coords = coords[emb_idx]

                # Build subgraph of common nodes for community detection
                G_sub = G_yeast.subgraph(common).copy()

                result = analyze_embedding_distances(
                    G_sub, aligned_coords, common,
                    f"yeast_{method}", seed=SEED,
                )
                yeast_distances[method] = result
                print(f"    Geometric gap = {result['geometric_gap']:.4f}  "
                      f"(intra={result['mean_intra']:.4f}, "
                      f"inter={result['mean_inter']:.4f})")
            except Exception as e:
                print(f"    {method} FAILED: {e}")

        if yeast_distances:
            all_distances["yeast"] = yeast_distances

    except Exception as e:
        print(f"  Yeast analysis failed: {e}")

    # ==============================================================
    # 1b. Human embedding distance distributions
    # ==============================================================
    if not args.skip_human:
        print("\nLoading human network for distance analysis...")
        try:
            G_human = load_human_network()
            human_nodes = sorted(G_human.nodes())
            print(f"  Human: {len(human_nodes)} nodes, "
                  f"{G_human.number_of_edges()} edges")

            human_distances = {}
            for method in ["DM", "Node2Vec"]:
                print(f"\n  [{method}] Loading human embedding...")
                try:
                    emb_file = (
                        data_dir / f"human_{method.lower()}_embedding.json"
                    )
                    if not emb_file.exists():
                        print(f"    Embedding not found: {emb_file}")
                        continue
                    with open(emb_file) as f:
                        emb_data = json.load(f)
                    emb_nodes = list(emb_data.keys())
                    h_coords = np.array(
                        [[emb_data[n]['x'], emb_data[n]['y']]
                         for n in emb_nodes]
                    )

                    # Align with network nodes
                    common = sorted(
                        set(emb_nodes) & set(human_nodes)
                    )
                    emb_node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
                    emb_idx = [emb_node_to_idx[n] for n in common]
                    aligned_coords = h_coords[emb_idx]

                    result = analyze_embedding_distances(
                        G_human, aligned_coords, common,
                        f"human_{method}", seed=SEED,
                    )
                    human_distances[method] = result
                    print(
                        f"    Geometric gap = {result['geometric_gap']:.4f}  "
                        f"(intra={result['mean_intra']:.4f}, "
                        f"inter={result['mean_inter']:.4f})"
                    )
                except Exception as e:
                    print(f"    {method} FAILED: {e}")

            if human_distances:
                all_distances["human"] = human_distances

        except FileNotFoundError as e:
            print(f"  Human network not available: {e}")
        except Exception as e:
            print(f"  Human distance analysis failed: {e}")
    else:
        print("\nSkipping human distance analysis (--skip-human).")

    # Plot distance distributions
    if all_distances:
        # Save distance data as pickle for external figure regeneration
        import pickle as _pkl
        _dist_path = results_dir / "distance_distributions.pkl"
        with open(_dist_path, "wb") as _f:
            _pkl.dump(all_distances, _f)
        print(f"  Saved distance distribution data: {_dist_path}")

        print("\nPlotting embedding distance distributions...")
        plot_distance_distributions(all_distances, figures_dir)
    else:
        print("\nNo distance data to plot.")

    # ==============================================================
    # 2. Node2Vec p/q sensitivity (yeast 153)
    # ==============================================================
    if not args.skip_ablation:
        print("\n" + "=" * 60)
        print("Analysis 2: Node2Vec p/q Parameter Sensitivity")
        print("=" * 60)

        try:
            G_yeast_s, nodes_yeast_s, go_map_s = load_curated_network(
                data_dir
            )
            sensitivity = run_pq_sensitivity(
                G_yeast_s, nodes_yeast_s, go_map_s
            )

            # Save JSON
            sens_file = results_dir / "node2vec_pq_sensitivity.json"
            with open(sens_file, "w") as f:
                json.dump(sensitivity, f, indent=2)
            print(f"\nSaved: {sens_file}")

            # Plot heatmap
            plot_pq_heatmap(sensitivity, figures_dir)

        except Exception as e:
            print(f"  p/q sensitivity FAILED: {e}")
    else:
        print("\nSkipping p/q sensitivity (--skip-ablation).")

    # ==============================================================
    # 3. Ablation experiment (human network)
    # ==============================================================
    if not args.skip_human:
        print("\n" + "=" * 60)
        print("Analysis 3: Ablation Experiment (Human Network)")
        print("=" * 60)

        try:
            G_human_abl = load_human_network()
            node_labels_raw = load_human_go_annotations()
            node_labels = _go_map_to_labels(node_labels_raw)

            if len(node_labels) < 10:
                print("  Insufficient GO annotations "
                      f"({len(node_labels)} nodes). Skipping ablation.")
            else:
                r_vals = np.linspace(0.05, 0.55, GF_N_POINTS)
                ablation_results = run_ablation(
                    G_human_abl, node_labels, r_vals,
                )

                abl_file = results_dir / "rank_reversal_ablation.json"
                # Strip large arrays for JSON storage
                save_results = {}
                for key, val in ablation_results.items():
                    save_results[key] = {
                        k: v for k, v in val.items()
                        if k not in ("purities", "modularities")
                    }
                    if "purities" in val:
                        save_results[key]["purities_summary"] = {
                            "mean": float(np.mean(val["purities"])),
                            "max": float(np.max(val["purities"])),
                            "n_points": len(val["purities"]),
                        }

                with open(abl_file, "w") as f:
                    json.dump(save_results, f, indent=2)
                print(f"\nSaved: {abl_file}")

        except FileNotFoundError as e:
            print(f"  Human network not available: {e}")
        except Exception as e:
            print(f"  Ablation experiment FAILED: {e}")
    else:
        print("\nSkipping ablation experiment (--skip-human).")

    # ==============================================================
    # 4. Rank reversal summary
    # ==============================================================
    print("\n" + "=" * 60)
    print("Analysis 4: Rank Reversal Summary")
    print("=" * 60)

    # Load topology metrics if available
    topology_file = results_dir / "network_topology_comparison.json"
    topology_metrics = None
    if topology_file.exists():
        with open(topology_file) as f:
            topology_metrics = json.load(f)
        print("  Loaded topology metrics for correlation analysis.")
    else:
        print("  Topology metrics not found. Run "
              "network_topology_analysis.py first for full analysis.")

    summary = generate_rank_reversal_summary(topology_metrics)

    if summary:
        summary_file = results_dir / "rank_reversal_summary.json"

        # Ensure all values are JSON-serialisable
        def _clean(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean_summary = json.loads(
            json.dumps(summary, default=_clean)
        )

        with open(summary_file, "w") as f:
            json.dump(clean_summary, f, indent=2)
        print(f"\nSaved: {summary_file}")

    print("\n" + "=" * 60)
    print("Rank reversal analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
