#!/usr/bin/env python3
"""
network_topology_analysis.py
Comparative network topology analysis of yeast and human PPI networks.

Computes basic topology, degree distribution, modularity, random walk
properties, and community structure for:
  - Yeast curated 153-node network
  - Yeast full STRING network (~5,936 nodes)
  - Human STRING network (~15,882 nodes before CC extraction; 14,679 in largest CC)

Output:
  - results/network_topology_comparison.json
  - figures/FigS1_topology_radar.png
"""

import sys
import json
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
import gzip
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_full_STRING_network,
)

logger = logging.getLogger(__name__)


# ---- Human Network Loading ----

def load_human_network(min_score=700):
    """Load human STRING PPI network (taxID 9606), largest connected component.

    Mirrors the loading logic in ``human_validation/human_embed_all.py``.

    Parameters
    ----------
    min_score : int
        Minimum combined STRING score to retain an edge.

    Returns
    -------
    nx.Graph
        Largest connected component of the filtered network.
    """
    project_root = Path(__file__).resolve().parent.parent
    links_file = project_root / "human_validation" / "9606.protein.links.v12.0.txt.gz"
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


# ---- Topology Metric Functions ----

def compute_basic_topology(G):
    """Compute basic network topology metrics.

    Returns
    -------
    dict
        Keys: n_nodes, n_edges, density, average_degree,
        average_clustering_coefficient, diameter, assortativity.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees))
    avg_clustering = nx.average_clustering(G)

    # Diameter: exact for small networks, BFS-approximated for large
    if n <= 2000:
        try:
            diameter = nx.diameter(G)
        except nx.NetworkXError:
            diameter = -1
    else:
        try:
            eccentricities = []
            sample_size = min(100, n)
            rng = np.random.RandomState(SEED)
            sample_nodes = rng.choice(list(G.nodes()), size=sample_size,
                                       replace=False)
            for node in sample_nodes:
                ecc = nx.eccentricity(G, v=node)
                eccentricities.append(ecc)
            diameter = int(max(eccentricities))
        except nx.NetworkXError:
            diameter = -1

    # Degree assortativity
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except (ValueError, ZeroDivisionError):
        assortativity = 0.0

    return {
        "n_nodes": n,
        "n_edges": m,
        "density": float(density),
        "average_degree": avg_degree,
        "average_clustering_coefficient": float(avg_clustering),
        "diameter": diameter,
        "assortativity": float(assortativity),
    }


def compute_degree_distribution(G):
    """Analyse the degree distribution.

    Returns
    -------
    dict
        Keys: power_law_exponent, degree_heterogeneity (std/mean),
        max_degree, min_degree.
    """
    degrees = np.array([d for _, d in G.degree()])
    k_min = max(int(degrees.min()), 1)

    # Power-law exponent via MLE: alpha = 1 + n / sum(ln(k_i / (k_min - 0.5)))
    valid = degrees[degrees >= k_min]
    if len(valid) > 1 and k_min > 0:
        alpha = 1.0 + len(valid) / np.sum(np.log(valid / (k_min - 0.5)))
    else:
        alpha = float('nan')

    # Degree heterogeneity = std / mean
    mean_k = float(np.mean(degrees))
    std_k = float(np.std(degrees))
    heterogeneity = std_k / mean_k if mean_k > 1e-10 else 0.0

    return {
        "power_law_exponent": float(alpha) if np.isfinite(alpha) else None,
        "degree_heterogeneity": heterogeneity,
        "max_degree": int(degrees.max()),
        "min_degree": int(degrees.min()),
    }


def compute_modularity_metrics(G):
    """Compute network modularity using greedy modularity communities.

    Falls back gracefully if community detection fails.

    Returns
    -------
    dict
        Keys: modularity_Q, n_communities (greedy); modularity_Q_leiden,
        n_communities_leiden if leidenalg is available.
    """
    result = {
        "modularity_Q": 0.0,
        "n_communities": 0,
    }

    # Greedy modularity communities (NetworkX built-in)
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        if communities:
            Q = nx.community.modularity(G, communities)
            result["modularity_Q"] = float(Q)
            result["n_communities"] = len(communities)
    except Exception as e:
        logger.warning("Greedy modularity detection failed: %s", e)

    # Leiden via igraph (optional, faster for large networks)
    try:
        import igraph as ig
        import leidenalg

        node_list = list(G.nodes())
        node_map = {i: node_list[i] for i in range(len(node_list))}
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        edges_idx = [(node_to_idx[u], node_to_idx[v])
                     for u, v in G.edges()]
        ig_graph = ig.Graph(n=len(node_list), edges=edges_idx,
                            directed=False)
        partition = leidenalg.find_partition(
            ig_graph, leidenalg.ModularityVertexPartition,
            seed=SEED,
        )
        # Map igraph vertex indices back to NetworkX node names
        communities_leiden = [
            frozenset(node_map[v] for v in comm) for comm in partition
        ]
        Q_leiden = nx.community.modularity(G, communities_leiden)
        result["modularity_Q_leiden"] = float(Q_leiden)
        result["n_communities_leiden"] = len(communities_leiden)
    except ImportError:
        logger.info("leidenalg not available; using greedy modularity only.")
    except Exception as e:
        logger.warning("Leiden community detection failed: %s", e)

    return result


def compute_random_walk_properties(G):
    """Compute spectral and random-walk properties of the network.

    Uses the normalised Laplacian to derive the spectral gap and an
    estimate of the mixing time.  Return probabilities are computed via
    explicit matrix powers for small networks and via the stationary
    distribution approximation for large ones.

    Returns
    -------
    dict
        Keys: spectral_gap, mixing_time, return_prob_k{2,4,6,8,10},
        return_prob_avg.
    """
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())

    # Normalised Laplacian eigenvalues
    L_norm = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray()
    eigvals = np.sort(np.linalg.eigvalsh(L_norm))

    # Spectral gap = lambda_2 - lambda_1  (lambda_1 ~ 0 for connected graph)
    spectral_gap = float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else 0.0

    # Mixing time ~ 1 / spectral_gap
    mixing_time = float(1.0 / spectral_gap) if spectral_gap > 1e-10 else float('inf')

    result = {
        "spectral_gap": spectral_gap,
        "mixing_time": mixing_time,
    }

    # Return probabilities via transition matrix powers
    k_values = [2, 4, 6, 8, 10]

    if n <= 2000:
        # Exact computation: P = D^{-1} A, return_prob(i,k) = (P^k)_{ii}
        A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(float)
        degrees = A.sum(axis=1)
        D_inv = np.diag(1.0 / np.maximum(degrees, 1e-10))
        P = D_inv @ A

        return_probs = {}
        for k in k_values:
            P_k = np.linalg.matrix_power(P, k)
            avg_return = float(np.mean(np.diag(P_k)))
            return_probs[f"return_prob_k{k}"] = avg_return
        result.update(return_probs)
    else:
        # For large networks use stationary distribution approximation:
        # pi_i = d_i / (2m), return_prob(i, k) -> pi_i for large k.
        degrees = np.array([G.degree(u) for u in nodes], dtype=float)
        two_m = degrees.sum()
        stationary = degrees / two_m
        avg_stationary_return = float(np.mean(stationary))
        for k in k_values:
            result[f"return_prob_k{k}"] = avg_stationary_return
        logger.info("Return probabilities approximated via stationary "
                     "distribution (n=%d > 2000).", n)

    prob_values = [result.get(f"return_prob_k{k}", 0.0) for k in k_values]
    result["return_prob_avg"] = float(np.mean(prob_values))
    return result


def compute_community_structure(G):
    """Compute community structure using greedy modularity communities.

    Returns
    -------
    dict
        Keys: n_communities, avg_community_size, community_size_std,
        community_sizes.
    """
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
    except Exception:
        communities = []

    if not communities:
        return {
            "n_communities": 0,
            "avg_community_size": 0.0,
            "community_size_std": 0.0,
            "community_sizes": [],
        }

    sizes = [len(c) for c in communities]
    return {
        "n_communities": len(communities),
        "avg_community_size": float(np.mean(sizes)),
        "community_size_std": float(np.std(sizes)),
        "community_sizes": sizes,
    }


# ---- Orchestrator ----

def compute_all_metrics(G, label="network"):
    """Run all topology analyses on a single network.

    Parameters
    ----------
    G : nx.Graph
        The input network (must be connected).
    label : str
        Human-readable label used in log output.

    Returns
    -------
    dict
        Nested dictionary of all computed metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"Computing topology for: {label}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"{'=' * 60}")

    print("  Basic topology...")
    basic = compute_basic_topology(G)

    print("  Degree distribution...")
    degree_dist = compute_degree_distribution(G)

    print("  Modularity...")
    modularity = compute_modularity_metrics(G)

    print("  Random walk properties...")
    rw = compute_random_walk_properties(G)

    print("  Community structure...")
    community = compute_community_structure(G)

    metrics = {}
    metrics.update(basic)
    metrics.update(degree_dist)
    metrics.update(modularity)
    metrics.update(rw)
    metrics.update(community)

    # Log key results
    print(f"  density            = {basic['density']:.6f}")
    print(f"  avg_clustering     = {basic['average_clustering_coefficient']:.4f}")
    print(f"  assortativity      = {basic['assortativity']:.4f}")
    print(f"  power_law_exponent = {degree_dist['power_law_exponent']}")
    print(f"  degree_heterogen.  = {degree_dist['degree_heterogeneity']:.4f}")
    print(f"  modularity_Q       = {modularity['modularity_Q']:.4f}")
    print(f"  spectral_gap       = {rw['spectral_gap']:.6f}")
    print(f"  mixing_time        = {rw['mixing_time']:.2f}")
    print(f"  n_communities      = {community['n_communities']}")

    return metrics


# ---- Visualisation ----

def generate_radar_chart(all_metrics, figures_dir, human_key="human_15882"):
    """Generate a radar chart comparing normalised topology metrics.

    Two polygons are drawn: yeast curated (blue) and human (orange),
    across six axes (modularity, clustering, spectral gap, degree
    heterogeneity, mixing time, assortativity).

    Parameters
    ----------
    all_metrics : dict
        Must contain keys ``"yeast_curated_153"`` and the key specified
        by *human_key*.
    figures_dir : Path
        Directory where the figure will be saved.
    human_key : str
        Key in *all_metrics* for the human network.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    categories = [
        'Modularity', 'Clustering', 'Spectral\nGap',
        'Degree\nHeterogeneity', 'Mixing\nTime', 'Assortativity',
    ]
    metric_keys = [
        'modularity_Q', 'average_clustering_coefficient', 'spectral_gap',
        'degree_heterogeneity', 'mixing_time', 'assortativity',
    ]

    yeast_key = "yeast_curated_153"

    if yeast_key not in all_metrics or human_key not in all_metrics:
        print("  Skipping radar chart: missing yeast or human metrics.")
        return

    yeast_metrics = all_metrics[yeast_key]
    human_metrics = all_metrics[human_key]

    # Gather raw values and normalise to [0, 1] across the two species
    yeast_vals = []
    human_vals = []
    for key in metric_keys:
        yv = yeast_metrics.get(key, 0.0) or 0.0
        hv = human_metrics.get(key, 0.0) or 0.0
        yeast_vals.append(yv)
        human_vals.append(hv)

    all_vals = np.array([yeast_vals, human_vals])
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-10] = 1.0

    yeast_norm = (np.array(yeast_vals) - mins) / ranges
    human_norm = (np.array(human_vals) - mins) / ranges

    # Polar plot setup
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    yeast_plot = np.append(yeast_norm, yeast_norm[0])
    human_plot = np.append(human_norm, human_norm[0])

    # Colorblind-safe palette (Okabe-Ito)
    yeast_color = '#0072B2'   # blue
    human_color = '#E69F00'   # orange

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.88)

    ax.plot(angles, yeast_plot, 'o-', linewidth=2.5, color=yeast_color,
            label='Yeast (curated 153)', markersize=7)
    ax.fill(angles, yeast_plot, alpha=0.15, color=yeast_color)

    human_n = all_metrics[human_key].get('n_nodes', '')
    ax.plot(angles, human_plot, 's-', linewidth=2.5, color=human_color,
            label=f'Human ({human_n})', markersize=7)
    ax.fill(angles, human_plot, alpha=0.15, color=human_color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9,
                        color='grey')

    ax.set_title('Network Topology Comparison\n(normalised metrics)',
                 fontsize=14, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)

    fig.tight_layout()
    output_path = figures_dir / "FigS1_topology_radar.png"
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nRadar chart saved to: {output_path}")


# ---- Main ----

def main():
    """Run the full comparative topology analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute and compare network topology metrics "
                    "between yeast and human PPI networks."
    )
    parser.add_argument(
        "--skip-human", action="store_true",
        help="Skip expensive human network computations.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    np.random.seed(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    # ---- 1. Yeast curated 153-node network ----
    print("\nLoading yeast curated 153-node network...")
    G_yeast, nodes_yeast, go_map = load_curated_network(data_dir)
    print(f"  Yeast curated: {len(nodes_yeast)} nodes, "
          f"{G_yeast.number_of_edges()} edges")
    all_metrics["yeast_curated_153"] = compute_all_metrics(
        G_yeast, "Yeast Curated (153)"
    )

    # ---- 2. Yeast full STRING network ----
    print("\nLoading yeast full STRING network...")
    try:
        G_yeast_full = load_full_STRING_network(data_dir)
        print(f"  Yeast full: {G_yeast_full.number_of_nodes()} nodes, "
              f"{G_yeast_full.number_of_edges()} edges")
        all_metrics["yeast_full_5936"] = compute_all_metrics(
            G_yeast_full,
            f"Yeast Full ({G_yeast_full.number_of_nodes()})"
        )
    except FileNotFoundError as e:
        print(f"  Yeast full network not available: {e}")

    # ---- 3. Human STRING network ----
    if not args.skip_human:
        print("\nLoading human STRING network...")
        try:
            G_human = load_human_network()
            n_human = G_human.number_of_nodes()
            print(f"  Human: {n_human} nodes, {G_human.number_of_edges()} edges")
            all_metrics[f"human_{n_human}"] = compute_all_metrics(
                G_human, f"Human ({n_human})"
            )
        except FileNotFoundError as e:
            print(f"  Human network not available: {e}")
    else:
        print("\nSkipping human network (--skip-human).")

    # ---- Comparison summary ----
    yeast_key = "yeast_curated_153"
    human_keys = [k for k in all_metrics if k.startswith("human_")]
    human_key = human_keys[0] if human_keys else None

    comparison = {}
    if yeast_key in all_metrics and human_key is not None:
        ym = all_metrics[yeast_key]
        hm = all_metrics[human_key]

        def _safe_ratio(a, b):
            if b is None or a is None:
                return None
            return float(a / b) if abs(b) > 1e-10 else None

        comparison = {
            "yeast_vs_human_modularity_ratio": _safe_ratio(
                ym.get("modularity_Q"), hm.get("modularity_Q")),
            "yeast_vs_human_mixing_time_ratio": _safe_ratio(
                ym.get("mixing_time"), hm.get("mixing_time")),
            "yeast_vs_human_spectral_gap_ratio": _safe_ratio(
                ym.get("spectral_gap"), hm.get("spectral_gap")),
            "yeast_vs_human_degree_heterogeneity_ratio": _safe_ratio(
                ym.get("degree_heterogeneity"), hm.get("degree_heterogeneity")),
        }
        all_metrics["comparison_summary"] = comparison
        print("\n=== Comparison Summary (yeast / human) ===")
        for k, v in comparison.items():
            print(f"  {k}: {v}")

    # ---- Radar chart ----
    if yeast_key in all_metrics and human_key is not None:
        # Use the canonical key for the radar chart
        chart_metrics = dict(all_metrics)
        chart_metrics[human_key] = all_metrics[human_key]
        print("\nGenerating radar chart...")
        generate_radar_chart(chart_metrics, figures_dir, human_key=human_key)
    else:
        print("\nSkipping radar chart (need both yeast and human metrics).")

    # ---- Save JSON ----
    output_file = results_dir / "network_topology_comparison.json"

    # Convert community_sizes to plain lists for JSON serialisation
    serialisable = {}
    for key, metrics in all_metrics.items():
        if isinstance(metrics, dict):
            clean = {}
            for k, v in metrics.items():
                if isinstance(v, np.integer):
                    clean[k] = int(v)
                elif isinstance(v, np.floating):
                    clean[k] = float(v)
                else:
                    clean[k] = v
            serialisable[key] = clean
        else:
            serialisable[key] = metrics

    with open(output_file, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
