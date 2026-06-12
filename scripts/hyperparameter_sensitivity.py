#!/usr/bin/env python3
"""
hyperparameter_sensitivity.py
Step 35: Hyperparameter Sensitivity Analysis for G-F Score.

Systematically varies key pipeline hyperparameters and measures their
impact on G-F Scores to answer: "How robust is the G-F Score to
reasonable choices in embedding and evaluation parameters?"

Parameters tested
-----------------
  1. **Integration interval r-points** (N_POINTS): 30, 50, 100, 200, 500
  2. **Community detection resolution** (Louvain γ): 0.5, 0.8, 1.0, 1.5, 2.0
  3. **Embedding dimension** (latent_dim): 2, 3, 4, 8
  4. **Random walk length** (DeepWalk/Node2Vec): 10, 20, 40, 80
  5. **Random walk window size**: 3, 5, 10
  6. **Node2Vec bias (p, q)**: grid of (0.25,0.5,1,2,4) × (0.25,0.5,1,2,4)

Uses the curated 153-node yeast network for speed (all runs < 5 s each).
Four representative methods: Spectral, DeepWalk, Node2Vec, PCA.

Output
------
  results/hyperparameter_sensitivity.json
  figures/Fig21_hyperparameter_sensitivity.png
"""

import sys
import os
import json
import time
import itertools
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir,
    load_curated_network,
    spectral_embedding_from_graph, deepwalk_from_graph,
    node2vec_from_graph, rescale_coordinates,
    compute_gf_curve, compute_gf_score, compute_centrality_features,
    GF_R_MIN, GF_R_MAX, TARGET_STD,
)

# ---- Configuration ----
R_MIN = 0.05
R_MAX = 0.55
METHODS = ["Spectral", "DeepWalk", "Node2Vec", "PCA"]


def _embed_method(G, nodes, method, **kwargs):
    """Compute embedding for a single method with optional kwargs."""
    if method == "Spectral":
        coords = spectral_embedding_from_graph(G, nodelist=nodes)
    elif method == "DeepWalk":
        coords = deepwalk_from_graph(
            G, walk_length=kwargs.get("walk_length", 20),
            walks_per_node=kwargs.get("walks_per_node", 10),
            window_size=kwargs.get("window_size", 5),
            dimensions=kwargs.get("dimensions", 2),
            seed=SEED,
        )
    elif method == "Node2Vec":
        coords = node2vec_from_graph(
            G, walk_length=kwargs.get("walk_length", 20),
            walks_per_node=kwargs.get("walks_per_node", 10),
            window_size=kwargs.get("window_size", 5),
            dimensions=kwargs.get("dimensions", 2),
            p=kwargs.get("p", 0.5), q=kwargs.get("q", 2.0),
            seed=SEED,
        )
    elif method == "PCA":
        features = compute_centrality_features(G, nodes)
        features_c = features - features.mean(axis=0)
        cov = features_c.T @ features_c / (len(nodes) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        dims = min(kwargs.get("dimensions", 2), features.shape[1])
        coords = features_c @ eigvecs[:, -dims:]
    else:
        raise ValueError(f"Unknown method: {method}")

    return rescale_coordinates(coords, TARGET_STD)


def _compute_gf_custom(coords, nodes, go_map, n_points=100, resolution=1.0):
    """Compute GF score with custom n_points and Louvain resolution."""
    import community as community_louvain
    from scipy.spatial.distance import pdist, squareform
    from scipy.integrate import trapezoid

    r_values = np.linspace(R_MIN, R_MAX, n_points)
    dist_matrix = squareform(pdist(coords))
    n = dist_matrix.shape[0]
    purities = np.zeros(len(r_values))

    for ri, r in enumerate(r_values):
        mask = (dist_matrix < r) & (dist_matrix > 0)
        n_edges = np.sum(mask) // 2
        if n_edges == 0:
            continue

        rows, cols = np.where(mask)
        upper = rows < cols
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(zip(rows[upper].tolist(), cols[upper].tolist()))

        if n_edges > 200_000:
            communities = [frozenset(c) for c in nx.connected_components(G)]
        else:
            try:
                partition = community_louvain.best_partition(
                    G, resolution=resolution, random_state=SEED
                )
                groups = defaultdict(set)
                for node, comm in partition.items():
                    groups[comm].add(node)
                communities = [frozenset(g) for g in groups.values()]
            except Exception:
                communities = [frozenset(c) for c in nx.connected_components(G)]

        comm_purities = []
        for comm in communities:
            all_terms = []
            for i in comm:
                if i < len(nodes) and nodes[i] in go_map:
                    terms = go_map[nodes[i]]
                    if isinstance(terms, list):
                        all_terms.extend(terms)
                    elif terms:
                        all_terms.append(terms)
            if not all_terms:
                continue
            counts = Counter(all_terms)
            comm_purities.append(counts.most_common(1)[0][1] / len(all_terms))

        if comm_purities:
            purities[ri] = np.mean(comm_purities)

    # Compute GF score over default interval
    score_mask = (r_values >= GF_R_MIN) & (r_values <= GF_R_MAX)
    if not np.any(score_mask):
        return 0.0
    r_sub, p_sub = r_values[score_mask], purities[score_mask]
    return float(trapezoid(p_sub, r_sub) / (GF_R_MAX - GF_R_MIN))


# ============================================================
# Sensitivity experiments
# ============================================================

def experiment_n_points(G, nodes, go_map):
    """Vary the number of r-points in GF curve computation."""
    n_points_list = [30, 50, 100, 200, 500]
    results = {}
    for np_ in n_points_list:
        print(f"  N_POINTS={np_}")
        scores = {}
        for method in METHODS:
            coords = _embed_method(G, nodes, method)
            score = _compute_gf_custom(coords, nodes, go_map, n_points=np_)
            scores[method] = score
        results[np_] = scores
    return {"n_points": results, "values_tested": n_points_list}


def experiment_resolution(G, nodes, go_map):
    """Vary Louvain community detection resolution."""
    resolutions = [0.5, 0.8, 1.0, 1.5, 2.0]
    results = {}
    for res in resolutions:
        print(f"  resolution={res}")
        scores = {}
        for method in METHODS:
            coords = _embed_method(G, nodes, method)
            score = _compute_gf_custom(coords, nodes, go_map, resolution=res)
            scores[method] = score
        results[res] = scores
    return {"resolution": results, "values_tested": resolutions}


def experiment_dimensions(G, nodes, go_map):
    """Vary embedding dimensionality."""
    dims_list = [2, 3, 4, 8]
    results = {}
    for dim in dims_list:
        print(f"  dimensions={dim}")
        scores = {}
        for method in METHODS:
            try:
                coords = _embed_method(G, nodes, method, dimensions=dim)
                # For GF computation, use only first 2 dims if dim > 2
                coords_2d = coords[:, :2] if dim > 2 else coords
                coords_2d = rescale_coordinates(coords_2d, TARGET_STD)
                score = _compute_gf_custom(coords_2d, nodes, go_map)
                scores[method] = score
            except Exception as e:
                print(f"    [{method}] Error at dim={dim}: {e}")
                scores[method] = None
        results[dim] = scores
    return {"dimensions": results, "values_tested": dims_list}


def experiment_walk_length(G, nodes, go_map):
    """Vary random walk length for DeepWalk and Node2Vec."""
    walk_lengths = [10, 20, 40, 80]
    results = {}
    for wl in walk_lengths:
        print(f"  walk_length={wl}")
        scores = {}
        for method in ["DeepWalk", "Node2Vec"]:
            coords = _embed_method(G, nodes, method, walk_length=wl)
            score = _compute_gf_custom(coords, nodes, go_map)
            scores[method] = score
        results[wl] = scores
    return {"walk_length": results, "values_tested": walk_lengths}


def experiment_window_size(G, nodes, go_map):
    """Vary random walk window size for DeepWalk and Node2Vec."""
    windows = [3, 5, 10]
    results = {}
    for ws in windows:
        print(f"  window_size={ws}")
        scores = {}
        for method in ["DeepWalk", "Node2Vec"]:
            coords = _embed_method(G, nodes, method, window_size=ws)
            score = _compute_gf_custom(coords, nodes, go_map)
            scores[method] = score
        results[ws] = scores
    return {"window_size": results, "values_tested": windows}


def experiment_n2v_pq(G, nodes, go_map):
    """Grid search over Node2Vec (p, q) hyperparameters."""
    p_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    q_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    results = {}
    for p, q in itertools.product(p_values, q_values):
        coords = _embed_method(G, nodes, "Node2Vec", p=p, q=q)
        score = _compute_gf_custom(coords, nodes, go_map)
        results[f"p={p},q={q}"] = score
        print(f"  p={p}, q={q}: {score:.4f}")
    return {
        "n2v_pq": results,
        "p_values": p_values,
        "q_values": q_values,
    }


def compute_cv(d):
    """Compute coefficient of variation from a dict of values."""
    vals = [v for v in d.values() if v is not None and isinstance(v, (int, float))]
    if len(vals) < 2:
        return None
    return float(np.std(vals) / max(np.mean(vals), 1e-10))


# ============================================================
# Figure generation
# ============================================================

def generate_figure(all_results, figures_dir):
    """Generate multi-panel sensitivity figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel A: N_POINTS sensitivity
    ax = axes[0, 0]
    data = all_results["n_points"]["n_points"]
    x_vals = all_results["n_points"]["values_tested"]
    for method in METHODS:
        y = [data[str(xv)].get(method, 0) if str(xv) in data
             else data[xv].get(method, 0) for xv in x_vals]
        ax.plot(x_vals, y, "o-", label=method, markersize=5)
    ax.set_xlabel("Number of r-points")
    ax.set_ylabel("G-F Score")
    ax.set_title("(A) Sensitivity to r-point Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Resolution sensitivity
    ax = axes[0, 1]
    data = all_results["resolution"]["resolution"]
    x_vals = all_results["resolution"]["values_tested"]
    for method in METHODS:
        y = [data.get(xv, {}).get(method, 0) for xv in x_vals]
        ax.plot(x_vals, y, "o-", label=method, markersize=5)
    ax.set_xlabel("Louvain Resolution (γ)")
    ax.set_ylabel("G-F Score")
    ax.set_title("(B) Sensitivity to Community Detection Resolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Embedding dimensions
    ax = axes[0, 2]
    data = all_results["dimensions"]["dimensions"]
    x_vals = all_results["dimensions"]["values_tested"]
    for method in METHODS:
        y = [data.get(xv, {}).get(method, 0) or 0 for xv in x_vals]
        ax.plot(x_vals, y, "o-", label=method, markersize=5)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("G-F Score (first 2 dims)")
    ax.set_title("(C) Sensitivity to Embedding Dimensionality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Walk length
    ax = axes[1, 0]
    data = all_results["walk_length"]["walk_length"]
    x_vals = all_results["walk_length"]["values_tested"]
    for method in ["DeepWalk", "Node2Vec"]:
        y = [data.get(xv, {}).get(method, 0) for xv in x_vals]
        ax.plot(x_vals, y, "o-", label=method, markersize=5)
    ax.set_xlabel("Walk Length")
    ax.set_ylabel("G-F Score")
    ax.set_title("(D) Sensitivity to Random Walk Length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel E: Window size
    ax = axes[1, 1]
    data = all_results["window_size"]["window_size"]
    x_vals = all_results["window_size"]["values_tested"]
    for method in ["DeepWalk", "Node2Vec"]:
        y = [data.get(xv, {}).get(method, 0) for xv in x_vals]
        ax.plot(x_vals, y, "o-", label=method, markersize=5)
    ax.set_xlabel("Window Size")
    ax.set_ylabel("G-F Score")
    ax.set_title("(E) Sensitivity to Sliding Window Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel F: Node2Vec p-q heatmap
    ax = axes[1, 2]
    n2v_data = all_results["n2v_pq"]["n2v_pq"]
    p_vals = all_results["n2v_pq"]["p_values"]
    q_vals = all_results["n2v_pq"]["q_values"]
    heatmap = np.zeros((len(p_vals), len(q_vals)))
    for pi, p in enumerate(p_vals):
        for qi, q in enumerate(q_vals):
            heatmap[pi, qi] = n2v_data.get(f"p={p},q={q}", 0)
    im = ax.imshow(heatmap, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(q_vals)))
    ax.set_xticklabels([str(q) for q in q_vals])
    ax.set_yticks(range(len(p_vals)))
    ax.set_yticklabels([str(p) for p in p_vals])
    ax.set_xlabel("q (return parameter)")
    ax.set_ylabel("p (in-out parameter)")
    ax.set_title("(F) Node2Vec (p, q) Grid Search")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out_file = figures_dir / "Fig21_hyperparameter_sensitivity.png"
    plt.savefig(str(out_file), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_file}")


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(SEED)
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    figures_dir = get_figures_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load network
    print("Loading curated 153-node network...")
    G, nodes, go_map = load_curated_network(data_dir)
    print(f"  {len(nodes)} nodes, {G.number_of_edges()} edges, "
          f"{len(go_map)} annotated")

    all_results = {}

    # ---- Experiment 1: N_POINTS ----
    print(f"\n{'=' * 60}")
    print("Experiment 1: r-point count sensitivity")
    t0 = time.time()
    all_results["n_points"] = experiment_n_points(G, nodes, go_map)
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- Experiment 2: Resolution ----
    print(f"\n{'=' * 60}")
    print("Experiment 2: Louvain resolution sensitivity")
    t0 = time.time()
    all_results["resolution"] = experiment_resolution(G, nodes, go_map)
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- Experiment 3: Embedding dimensions ----
    print(f"\n{'=' * 60}")
    print("Experiment 3: Embedding dimensionality sensitivity")
    t0 = time.time()
    all_results["dimensions"] = experiment_dimensions(G, nodes, go_map)
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- Experiment 4: Walk length ----
    print(f"\n{'=' * 60}")
    print("Experiment 4: Random walk length sensitivity")
    t0 = time.time()
    all_results["walk_length"] = experiment_walk_length(G, nodes, go_map)
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- Experiment 5: Window size ----
    print(f"\n{'=' * 60}")
    print("Experiment 5: Window size sensitivity")
    t0 = time.time()
    all_results["window_size"] = experiment_window_size(G, nodes, go_map)
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- Experiment 6: Node2Vec p-q grid ----
    print(f"\n{'=' * 60}")
    print("Experiment 6: Node2Vec (p, q) grid search")
    t0 = time.time()
    all_results["n2v_pq"] = experiment_n2v_pq(G, nodes, go_map)
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ---- Compute summary statistics ----
    print(f"\n{'=' * 60}")
    print("Computing sensitivity summary...")

    summary = {}
    # CV for each experiment
    for exp_name in ["n_points", "resolution", "dimensions", "walk_length", "window_size"]:
        exp_data = all_results[exp_name][exp_name]
        cvs = {}
        for method in METHODS:
            method_scores = []
            for param_val, scores in exp_data.items():
                if isinstance(scores, dict) and method in scores:
                    v = scores[method]
                    if v is not None:
                        method_scores.append(v)
            if method_scores:
                cvs[method] = float(np.std(method_scores) / max(np.mean(method_scores), 1e-10))
        summary[exp_name] = cvs

    # N2V p-q: best and worst
    n2v_scores = list(all_results["n2v_pq"]["n2v_pq"].values())
    summary["n2v_pq"] = {
        "best": max(n2v_scores),
        "worst": min(n2v_scores),
        "range": max(n2v_scores) - min(n2v_scores),
        "cv": float(np.std(n2v_scores) / max(np.mean(n2v_scores), 1e-10)),
    }

    # ---- Save results ----
    output = {
        "experiments": {},
        "summary": summary,
        "network": {
            "nodes": len(nodes),
            "edges": G.number_of_edges(),
            "annotated": len(go_map),
        },
    }
    # Convert numpy keys to strings for JSON serialization
    for exp_name, exp_data in all_results.items():
        clean_data = {}
        for key, val in exp_data.items():
            if key == exp_name:  # The results dict
                clean_results = {}
                for k, v in val.items():
                    clean_results[str(k)] = v
                clean_data[key] = clean_results
            else:
                clean_data[key] = val
        output["experiments"][exp_name] = clean_data

    out_file = results_dir / "hyperparameter_sensitivity.json"
    with open(str(out_file), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_file}")

    # ---- Generate figure ----
    # Fix the key types for figure generation (they're already correct in all_results)
    generate_figure(all_results, figures_dir)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("HYPERPARAMETER SENSITIVITY SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Experiment':<25} {'Method CVs':>40}")
    print("-" * 70)
    for exp_name in ["n_points", "resolution", "dimensions", "walk_length", "window_size"]:
        cvs = summary.get(exp_name, {})
        cv_str = ", ".join(f"{m}:{v:.3f}" for m, v in cvs.items())
        print(f"  {exp_name:<23} {cv_str:>40}")
    n2v = summary.get("n2v_pq", {})
    print(f"  {'n2v_pq':<23} range={n2v.get('range',0):.4f}, "
          f"CV={n2v.get('cv',0):.3f}")

    print("\nInterpretation:")
    max_cv = max(
        max(v for v in summary.get(exp, {}).values())
        for exp in ["n_points", "resolution", "dimensions", "walk_length", "window_size"]
        if summary.get(exp)
    )
    if max_cv < 0.10:
        print("  G-F Scores are HIGHLY ROBUST to hyperparameter variation (max CV < 10%)")
    elif max_cv < 0.25:
        print("  G-F Scores are MODERATELY ROBUST to hyperparameter variation (max CV < 25%)")
    else:
        print("  G-F Scores show SENSITIVITY to some hyperparameters (max CV >= 25%)")


if __name__ == "__main__":
    main()
