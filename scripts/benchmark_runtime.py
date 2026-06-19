#!/usr/bin/env python3
"""
benchmark_runtime.py
Benchmark the 72-step G-F consistency analysis pipeline and perform
theoretical complexity analysis.

This script:
  - Times each pipeline step individually on the curated 153-node network.
  - Optionally benchmarks key steps on the full 5,936-node network.
  - Measures G-F curve scalability across different sampling-point counts.
  - Reports Big-O theoretical complexity alongside empirical timings.
  - Produces two supplementary figures (Figure S6 and Figure S7).

Usage:
    python benchmark_runtime.py                              # Default (fast)
    python benchmark_runtime.py --skip-full                  # Skip full network
    python benchmark_runtime.py --n-repeat 5                 # 5 repetitions
    python benchmark_runtime.py --sampling-points 50,100,200 # Custom grid
"""

import sys
import json
import time
import random
import platform
import argparse
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup (must match existing scripts)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
# NOTE: This script must be run as a standalone process (via subprocess from
# run_all_analysis.py), not imported as a module.  The second sys.path.insert
# below enables bare imports like `from leiden_baseline import main`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
    load_curated_network, load_full_STRING_network, load_embedding,
    compute_gf_curve, compute_gf_score, compute_plateau_width,
    rescale_coordinates, precompute_distance_matrix,
    compute_centrality_features, build_similarity_matrix,
    diffusion_map_from_similarity, classical_mds_from_distances,
    spectral_embedding_from_graph, deepwalk_from_graph, node2vec_from_graph,
    vgae_from_graph,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_MIN = 0.05
R_MAX = 0.55
N_POINTS_DEFAULT = 200
GF_R_MIN = 0.05
GF_R_MAX = 0.422

METHODS = ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE"]
ALL_METHODS = ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE",
               "VGAE-feat", "PCA"]

# Okabe-Ito color palette (colorblind-safe)
OKABE_ITO = {
    "orange":  "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

# Ordered palette for bar stacking
STEP_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
    "#888888",  # gray
    "#4E79A7",  # Tableau blue
    "#E15759",  # Tableau red
    "#59A14F",  # Tableau green
    "#F28E2B",  # Tableau orange
    "#B07AA1",  # Tableau purple
]

# ---------------------------------------------------------------------------
# Theoretical complexity table
# ---------------------------------------------------------------------------
THEORETICAL_COMPLEXITY = {
    "embedding": {
        "DM": "O(n^3)",
        "MDS": "O(n^3)",
        "Spectral": "O(n^3)",
        "DeepWalk": "O(n*w*l)",
        "Node2Vec": "O(n*w*l)",
        "VGAE": "O(E*epochs)",
        "VGAE-feat": "O(E*epochs)",
        "PCA": "O(n*d^2)",
    },
    "distance_matrix": "O(n^2)",
    "gf_curve": "O(K * (n^2 + Leiden))",
    "gf_curve_detail": "O(K * n^2 * log(n))",
    "leiden_community_detection": "O(n*log(n))",
    "gf_score": "O(K)",
    "total_pipeline": "O(n^3 + K*n^2*log(n))",
}

# Step-level complexity annotations
STEP_COMPLEXITY = {
    "step_1_data_preprocessing": "O(V+E)",
    "step_2_embeddings": {
        "DM": "O(n^3)",
        "MDS": "O(n^3)",
        "Spectral": "O(n^3)",
        "DeepWalk": "O(n*w*l)",
        "Node2Vec": "O(n*w*l)",
        "VGAE": "O(E*epochs)",
        "VGAE-feat": "O(E*epochs)",
        "PCA": "O(n*d^2)",
    },
    "step_3_gf_curves": "O(K*n^2*log(n))",
    "step_4_leiden_baseline": "O(n*log(n))",
    "step_5_subset_robustness": "O(S*(n^3 + K*n^2*log(n)))",
    "step_6_full_network": "O(n^3 + K*n^2*log(n))",
    "step_7_geometric_analysis": "O(n^2)",
    "step_8_link_prediction": "O(F*(n+E))",
    "step_9_downstream_knn": "O(F*n*log(n))",
    "step_10_randomization_control": "O(S*K*n^2*log(n))",
    "step_11_sampling_density": "O(K_30*n^2*log(n) + K_200*n^2*log(n))",
    "step_12_sensitivity_analysis": "O(K)",
    "step_13_human_validation": "O(n^3 + K*n^2*log(n))",
    "step_14_figure_generation": "O(n_steps)",
}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def timed(fn, n_repeat=1):
    """Run *fn* ``n_repeat`` times and return (mean_seconds, std_seconds).

    Uses ``time.perf_counter()`` for high-resolution monotonic timing.
    """
    times = []
    result = None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times)), float(np.std(times)), result


def timed_once(fn):
    """Run *fn* once and return (seconds, result)."""
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    return t1 - t0, result


# ---------------------------------------------------------------------------
# Embedding functions (duplicated from embed_all.py for per-method timing)
# ---------------------------------------------------------------------------

def embed_diffusion_map(G, nodes):
    """Diffusion Map: centrality features -> similarity -> eigendecomposition."""
    features = compute_centrality_features(G, nodes)
    sim = build_similarity_matrix(features)
    coords = diffusion_map_from_similarity(sim)
    return rescale_coordinates(coords, target_std=0.3)


def embed_mds(G, nodes):
    """Classical MDS on shortest-path distances."""
    import networkx as nx
    n = len(nodes)
    lengths = dict(nx.shortest_path_length(G))
    D = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if j >= i:
                d = lengths[u].get(v, n)
                D[i, j] = d
                D[j, i] = d
    coords = classical_mds_from_distances(D)
    return rescale_coordinates(coords, target_std=0.3)


def embed_spectral(G, nodes):
    """Spectral embedding from normalised Laplacian."""
    coords = spectral_embedding_from_graph(G, nodelist=nodes)
    return rescale_coordinates(coords, target_std=0.3)


def embed_deepwalk(G, nodes):
    """DeepWalk: uniform random walks + co-occurrence + SVD."""
    coords = deepwalk_from_graph(G, walk_length=20, walks_per_node=10,
                                 window_size=5, seed=SEED)
    return rescale_coordinates(coords, target_std=0.3)


def embed_node2vec(G, nodes):
    """Node2Vec: biased random walks + co-occurrence + SVD."""
    coords = node2vec_from_graph(G, walk_length=20, walks_per_node=10,
                                 window_size=5, p=0.5, q=2.0, seed=SEED)
    return rescale_coordinates(coords, target_std=0.3)


def embed_vgae(G, nodes, features=None):
    """VGAE: Variational Graph Autoencoder (2-layer GCN encoder)."""
    coords = vgae_from_graph(G, hidden_dim=4, latent_dim=2,
                             epochs=300, lr=0.01, features=features, seed=SEED)
    return rescale_coordinates(coords, target_std=0.3)


def embed_pca(G, nodes):
    """PCA control: PCA on centrality features."""
    features = compute_centrality_features(G, nodes)
    features_centered = features - features.mean(axis=0)
    cov = features_centered.T @ features_centered / (len(nodes) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    coords = features_centered @ eigvecs[:, -2:]
    return rescale_coordinates(coords, target_std=0.3)


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def get_system_info():
    """Collect CPU, Python, and library version information."""
    info = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
    }
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    try:
        import networkx
        info["networkx_version"] = networkx.__version__
    except ImportError:
        pass
    try:
        import matplotlib
        info["matplotlib_version"] = matplotlib.__version__
    except ImportError:
        pass
    try:
        import torch
        info["torch_version"] = torch.__version__
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------------------
# Step benchmarking functions
# ---------------------------------------------------------------------------

def bench_step1_preprocessing(data_dir, n_repeat):
    """Benchmark Step 1: Data preprocessing (load curated network)."""
    def _run():
        G, nodes, go_map = load_curated_network(data_dir)
        return G, nodes, go_map

    mean_t, std_t, result = timed(_run, n_repeat)
    return mean_t, std_t, result


def bench_step2_embeddings(G, nodes, n_repeat):
    """Benchmark Step 2: Embedding computation (each of 8 methods)."""
    features = compute_centrality_features(G, nodes)

    embed_fns = {
        "DM": lambda: embed_diffusion_map(G, nodes),
        "MDS": lambda: embed_mds(G, nodes),
        "Spectral": lambda: embed_spectral(G, nodes),
        "DeepWalk": lambda: embed_deepwalk(G, nodes),
        "Node2Vec": lambda: embed_node2vec(G, nodes),
        "VGAE": lambda: embed_vgae(G, nodes, features=None),
        "VGAE-feat": lambda: embed_vgae(G, nodes, features=features),
        "PCA": lambda: embed_pca(G, nodes),
    }

    results = {}
    for method, fn in embed_fns.items():
        print(f"    Timing {method} ...", end=" ", flush=True)
        random.seed(SEED)
        np.random.seed(SEED)
        try:
            mean_t, std_t, coords = timed(fn, n_repeat)
            results[method] = {
                "time_s": round(mean_t, 4),
                "time_std_s": round(std_t, 4),
                "complexity": THEORETICAL_COMPLEXITY["embedding"][method],
            }
            print(f"{mean_t:.3f}s (+/- {std_t:.3f}s)")
        except Exception as e:
            results[method] = {"time_s": None, "error": str(e)}
            print(f"FAILED ({e})")

    total = sum(v["time_s"] for v in results.values()
                if isinstance(v.get("time_s"), (int, float)))
    return results, total


def bench_step3_gf_curves(coords_dict, nodes_dict, go_map, n_points, n_repeat):
    """Benchmark Step 3: G-F curve computation (200-point grid, all methods)."""
    r_vals = np.linspace(R_MIN, R_MAX, n_points)
    times = {}
    for method in METHODS:
        if method not in coords_dict:
            continue
        c = coords_dict[method]
        n = nodes_dict[method]

        def _run(c=c, n=n, r=r_vals):
            return compute_gf_curve(c, n, go_map, r)

        print(f"    Timing GF curve for {method} ...", end=" ", flush=True)
        mean_t, std_t, _ = timed(_run, n_repeat)
        times[method] = {"time_s": round(mean_t, 4), "time_std_s": round(std_t, 4)}
        print(f"{mean_t:.3f}s (+/- {std_t:.3f}s)")

    total = sum(v["time_s"] for v in times.values())
    return times, total


def bench_step4_leiden(data_dir, n_repeat):
    """Benchmark Step 4: Leiden baseline community detection."""
    from leiden_baseline import main as leiden_main
    mean_t, std_t, _ = timed(leiden_main, n_repeat)
    return mean_t, std_t


def bench_step5_robustness(n_repeat):
    """Benchmark Step 5: Subset robustness (10 subsets)."""
    from robustness import main as robustness_main
    mean_t, std_t, _ = timed(robustness_main, n_repeat)
    return mean_t, std_t


def bench_step6_full_network(n_repeat):
    """Benchmark Step 6: Full network validation (5,936 nodes)."""
    from full_network import main as full_network_main
    mean_t, std_t, _ = timed(full_network_main, n_repeat)
    return mean_t, std_t


def bench_step7_geometric(n_repeat):
    """Benchmark Step 7: Geometric analysis (d_intra / d_inter)."""
    from geometric_analysis import main as geometric_main
    mean_t, std_t, _ = timed(geometric_main, n_repeat)
    return mean_t, std_t


def bench_step8_link_prediction(n_repeat):
    """Benchmark Step 8: Link prediction (5-fold CV)."""
    from link_prediction import main as link_pred_main
    mean_t, std_t, _ = timed(link_pred_main, n_repeat)
    return mean_t, std_t


def bench_step9_downstream_knn(n_repeat):
    """Benchmark Step 9: Downstream k-NN evaluation."""
    from downstream_knn import main as knn_main
    mean_t, std_t, _ = timed(knn_main, n_repeat)
    return mean_t, std_t


def bench_step10_randomization(n_repeat):
    """Benchmark Step 10: Randomization control."""
    from randomization_control import main as rand_main
    mean_t, std_t, _ = timed(rand_main, n_repeat)
    return mean_t, std_t


def bench_step11_sampling_density(n_repeat):
    """Benchmark Step 11: Sampling density verification."""
    from sampling_density import main as density_main
    mean_t, std_t, _ = timed(density_main, n_repeat)
    return mean_t, std_t


def bench_step12_sensitivity(n_repeat):
    """Benchmark Step 12: G-F score sensitivity analysis."""
    from gf_score_sensitivity import main as sensitivity_main
    # sensitivity_main has argparse; call with empty argv to avoid conflicts
    old_argv = sys.argv
    sys.argv = ["gf_score_sensitivity"]
    try:
        mean_t, std_t, _ = timed(sensitivity_main, n_repeat)
    finally:
        sys.argv = old_argv
    return mean_t, std_t


def bench_step13_human(n_repeat):
    """Benchmark Step 13: Human cross-species validation."""
    from human_validation import main as human_main
    mean_t, std_t, _ = timed(human_main, n_repeat)
    return mean_t, std_t


def bench_step14_figures(n_repeat):
    """Benchmark Step 14: Figure generation."""
    from plot_figures import main as plot_main
    old_argv = sys.argv
    sys.argv = ["plot_figures"]
    try:
        mean_t, std_t, _ = timed(plot_main, n_repeat)
    finally:
        sys.argv = old_argv
    return mean_t, std_t


# ---------------------------------------------------------------------------
# Scalability experiment
# ---------------------------------------------------------------------------

def scalability_experiment(coords, nodes, go_map, sampling_points):
    """Measure G-F curve runtime at varying sampling-point counts.

    Uses the DM embedding on the 153-node curated network as the test case.

    Parameters
    ----------
    coords : np.ndarray
        Embedding coordinates for DM method.
    nodes : list
        Node labels aligned with *coords*.
    go_map : dict
        Gene-to-GO-term mapping.
    sampling_points : list of int
        Grid sizes to test (e.g. [50, 100, 200, 500]).

    Returns
    -------
    dict
        Keys: ``n_points``, ``times_s``, ``time_per_point_ms``.
    """
    results = {"n_points": [], "times_s": [], "time_per_point_ms": []}
    for n_pts in sorted(sampling_points):
        r_vals = np.linspace(R_MIN, R_MAX, n_pts)

        def _run():
            return compute_gf_curve(coords, nodes, go_map, r_vals)

        # Average over 3 runs for stability
        mean_t, std_t, _ = timed(_run, n_repeat=3)
        per_point_ms = (mean_t / n_pts) * 1000.0

        results["n_points"].append(n_pts)
        results["times_s"].append(round(mean_t, 4))
        results["time_per_point_ms"].append(round(per_point_ms, 4))

        print(f"    {n_pts:>4d} points: {mean_t:.3f}s "
              f"({per_point_ms:.2f} ms/point)")

    return results


# ---------------------------------------------------------------------------
# Full-network key step benchmarking
# ---------------------------------------------------------------------------

def bench_full_network_key_steps(n_repeat):
    """Benchmark selected steps on the full 5,936-node STRING network.

    Only DM embedding and a single G-F curve are timed (not the full step 6
    pipeline) to keep runtime manageable.
    """
    print("\n  Loading full STRING network ...")
    t0 = time.perf_counter()
    data_dir = get_data_dir()
    G_full = load_full_STRING_network(data_dir)
    t_load = time.perf_counter() - t0
    nodes_full = sorted(G_full.nodes())
    n_full = len(nodes_full)
    print(f"    {n_full} nodes, {G_full.number_of_edges()} edges "
          f"(loaded in {t_load:.1f}s)")

    results = {
        "n_nodes": n_full,
        "n_edges": G_full.number_of_edges(),
        "load_time_s": round(t_load, 4),
    }

    # DM embedding on full network
    print("    Timing DM embedding on full network ...")
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        mean_t, std_t, _ = timed(
            lambda: embed_diffusion_map(G_full, nodes_full), n_repeat
        )
        results["DM_embedding"] = {
            "time_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "complexity": "O(n^3)",
        }
        print(f"      {mean_t:.3f}s (+/- {std_t:.3f}s)")
    except Exception as e:
        results["DM_embedding"] = {"error": str(e)}
        print(f"      FAILED: {e}")

    # G-F curve on annotated subset (153 nodes) using existing DM embedding
    try:
        with open(data_dir / "gene_go_map.json") as f:
            go_map = json.load(f)
        annotated = sorted(set(go_map.keys()) & set(nodes_full))
        if len(annotated) > 0:
            features = compute_centrality_features(G_full, nodes_full)
            sim = build_similarity_matrix(features)
            coords_full = diffusion_map_from_similarity(sim)
            coords_full = rescale_coordinates(coords_full, target_std=0.3)
            nodes_full_to_idx = {n: i for i, n in enumerate(nodes_full)}
            ann_idx = [nodes_full_to_idx[n] for n in annotated]
            sub_coords = coords_full[ann_idx]

            r_vals = np.linspace(R_MIN, R_MAX, N_POINTS_DEFAULT)

            def _gf_run():
                return compute_gf_curve(sub_coords, annotated, go_map, r_vals)

            mean_t, std_t, _ = timed(_gf_run, n_repeat)
            results["gf_curve_annotated"] = {
                "time_s": round(mean_t, 4),
                "time_std_s": round(std_t, 4),
                "n_annotated": len(annotated),
                "complexity": "O(K*n_ann^2*log(n_ann))",
            }
            print(f"    GF curve on {len(annotated)} annotated nodes: "
                  f"{mean_t:.3f}s (+/- {std_t:.3f}s)")
    except Exception as e:
        results["gf_curve_annotated"] = {"error": str(e)}
        print(f"    GF curve on annotated subset FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_fig_s6_runtime_breakdown(step_times, figures_dir):
    """Generate Figure S6: stacked bar chart of pipeline step runtimes.

    Parameters
    ----------
    step_times : dict
        Mapping of step label to time in seconds.
    figures_dir : Path
        Directory in which to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(step_times.keys())
    times = [step_times[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Single horizontal stacked bar
    left = 0.0
    for i, (label, t) in enumerate(zip(labels, times)):
        color = STEP_COLORS[i % len(STEP_COLORS)]
        ax.barh(0, t, left=left, height=0.6, color=color, edgecolor="white",
                linewidth=0.5, label=label)
        # Add time label inside bar if wide enough
        if t > 0.5:
            ax.text(left + t / 2, 0, f"{t:.1f}s",
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color="white")
        left += t

    total = sum(times)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title(f"Pipeline Runtime Breakdown (Total: {total:.1f}s / "
                 f"{total / 60:.1f} min)", fontsize=12, fontweight="bold")
    ax.set_yticks([])

    # Legend below
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=7, framealpha=0.9)

    plt.tight_layout()
    out = figures_dir / "FigS6_runtime_breakdown.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_fig_s7_time_accuracy_tradeoff(scalability_data, figures_dir):
    """Generate Figure S7: time vs sampling-point count trade-off.

    Parameters
    ----------
    scalability_data : dict
        Output of :func:`scalability_experiment`.
    figures_dir : Path
        Directory in which to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pts = scalability_data["n_points"]
    times = scalability_data["times_s"]
    per_point = scalability_data["time_per_point_ms"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: total time vs n_points
    ax1.plot(n_pts, times, "o-", color=OKABE_ITO["blue"], linewidth=2,
             markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    ax1.fill_between(n_pts, times, alpha=0.15, color=OKABE_ITO["blue"])
    ax1.set_xlabel("Number of sampling points (K)", fontsize=11)
    ax1.set_ylabel("G-F curve computation time (s)", fontsize=11)
    ax1.set_title("(A) Total Time vs Sampling Points", fontsize=12,
                  fontweight="bold")
    ax1.grid(alpha=0.3)

    # Annotate each point
    for x, y in zip(n_pts, times):
        ax1.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9,
                     color=OKABE_ITO["blue"])

    # Right: time per point
    ax2.bar([str(n) for n in n_pts], per_point,
            color=[OKABE_ITO["blue"], OKABE_ITO["sky_blue"],
                   OKABE_ITO["bluish_green"], OKABE_ITO["orange"]],
            edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Number of sampling points (K)", fontsize=11)
    ax2.set_ylabel("Time per point (ms)", fontsize=11)
    ax2.set_title("(B) Time per Sampling Point", fontsize=12,
                  fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    for i, (x, y) in enumerate(zip(n_pts, per_point)):
        ax2.text(i, y + max(per_point) * 0.02, f"{y:.2f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = figures_dir / "FigS7_time_accuracy_tradeoff.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the complete runtime benchmark pipeline."""
    parser = argparse.ArgumentParser(
        description="Benchmark the G-F consistency analysis pipeline."
    )
    parser.add_argument(
        "--skip-full", action="store_true", default=True,
        help="Skip full-network (5,936-node) benchmarks (default: skip)."
    )
    parser.add_argument(
        "--run-full", action="store_true", default=False,
        help="Include full-network benchmarks (overrides --skip-full)."
    )
    parser.add_argument(
        "--skip-human", action="store_true", default=True,
        help="Skip human validation benchmark (default: skip)."
    )
    parser.add_argument(
        "--run-human", action="store_true", default=False,
        help="Include human validation benchmark (overrides --skip-human)."
    )
    parser.add_argument(
        "--n-repeat", type=int, default=3,
        help="Number of timing repetitions for averaging (default: 3)."
    )
    parser.add_argument(
        "--sampling-points", type=str, default="50,100,200,500",
        help="Comma-separated sampling point counts (default: '50,100,200,500')."
    )
    args = parser.parse_args()

    # Clear sys.argv so that imported main() functions with their own
    # argparse do not choke on the benchmark's CLI flags.
    sys.argv = [sys.argv[0]]

    run_full = args.run_full
    run_human = args.run_human
    n_repeat = args.n_repeat
    sampling_points = [int(x.strip()) for x in args.sampling_points.split(",")]

    # Seed everything
    random.seed(SEED)
    np.random.seed(SEED)

    # Directories
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = get_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  G-F Consistency Framework - Runtime Benchmark")
    print("=" * 70)
    print(f"  System:      {platform.platform()}")
    print(f"  Python:      {platform.python_version()}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  Repetitions: {n_repeat}")
    print(f"  Full network: {'ENABLED' if run_full else 'SKIPPED'}")
    print(f"  Human validation: {'ENABLED' if run_human else 'SKIPPED'}")
    print(f"  Sampling points: {sampling_points}")
    print()

    pipeline_t0 = time.perf_counter()
    step_times = {}       # label -> seconds (for bar chart)
    json_output = {       # for benchmark_runtime.json
        "system_info": get_system_info(),
        "step_times_153": {},
        "theoretical_complexity": THEORETICAL_COMPLEXITY,
    }

    # ==================================================================
    # STEP 1: Data Preprocessing
    # ==================================================================
    print("-" * 60)
    print("Step 1: Data Preprocessing")
    print("-" * 60)
    mean_t, std_t, (G, nodes, go_map) = bench_step1_preprocessing(
        data_dir, n_repeat
    )
    step_times["1. Preprocessing"] = mean_t
    json_output["step_times_153"]["step_1_data_preprocessing"] = {
        "time_s": round(mean_t, 4),
        "time_std_s": round(std_t, 4),
        "complexity": STEP_COMPLEXITY["step_1_data_preprocessing"],
    }
    print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")

    # ==================================================================
    # STEP 2: Embedding Computation
    # ==================================================================
    print("-" * 60)
    print("Step 2: Embedding Computation (8 methods)")
    print("-" * 60)
    emb_results, emb_total = bench_step2_embeddings(G, nodes, n_repeat)
    step_times["2. Embeddings"] = emb_total
    json_output["step_times_153"]["step_2_embeddings"] = emb_results
    json_output["step_times_153"]["step_2_embeddings"]["_total_s"] = round(
        emb_total, 4
    )
    print(f"  Total embedding time: {emb_total:.3f}s\n")

    # Collect DM coords for scalability experiment and GF curve benchmark
    print("  Pre-computing DM coords for later benchmarks ...")
    random.seed(SEED)
    np.random.seed(SEED)
    dm_coords = embed_diffusion_map(G, nodes)
    dm_nodes = list(nodes)
    print(f"  DM coords shape: {dm_coords.shape}\n")

    # ==================================================================
    # STEP 3: G-F Curve Computation
    # ==================================================================
    print("-" * 60)
    print("Step 3: G-F Curve Computation (200-point grid)")
    print("-" * 60)
    # We need coords for all methods; compute them once (not timed here)
    coords_dict = {}
    nodes_dict = {}
    features = compute_centrality_features(G, nodes)

    _embed_map = {
        "DM": lambda: embed_diffusion_map(G, nodes),
        "MDS": lambda: embed_mds(G, nodes),
        "Spectral": lambda: embed_spectral(G, nodes),
        "DeepWalk": lambda: embed_deepwalk(G, nodes),
        "Node2Vec": lambda: embed_node2vec(G, nodes),
        "VGAE": lambda: embed_vgae(G, nodes, features=None),
    }

    for method, fn in _embed_map.items():
        random.seed(SEED)
        np.random.seed(SEED)
        try:
            c = fn()
            coords_dict[method] = c
            nodes_dict[method] = list(nodes)
        except Exception as e:
            print(f"    Could not compute {method}: {e}")

    gf_times, gf_total = bench_step3_gf_curves(
        coords_dict, nodes_dict, go_map, N_POINTS_DEFAULT, n_repeat
    )
    step_times["3. GF Curves"] = gf_total
    json_output["step_times_153"]["step_3_gf_curves"] = {
        "per_method": gf_times,
        "time_s": round(gf_total, 4),
        "complexity": STEP_COMPLEXITY["step_3_gf_curves"],
    }
    print(f"  Total GF curve time: {gf_total:.3f}s\n")

    # ==================================================================
    # STEP 4: Leiden Baseline
    # ==================================================================
    print("-" * 60)
    print("Step 4: Leiden Baseline")
    print("-" * 60)
    try:
        mean_t, std_t = bench_step4_leiden(data_dir, n_repeat)
        step_times["4. Leiden"] = mean_t
        json_output["step_times_153"]["step_4_leiden_baseline"] = {
            "time_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "complexity": STEP_COMPLEXITY["step_4_leiden_baseline"],
        }
        print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        step_times["4. Leiden"] = 0.0

    # ==================================================================
    # STEP 5: Subset Robustness
    # ==================================================================
    print("-" * 60)
    print("Step 5: Subset Robustness (10 subsets)")
    print("-" * 60)
    try:
        mean_t, std_t = bench_step5_robustness(n_repeat)
        step_times["5. Robustness"] = mean_t
        json_output["step_times_153"]["step_5_subset_robustness"] = {
            "time_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "complexity": STEP_COMPLEXITY["step_5_subset_robustness"],
        }
        print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        step_times["5. Robustness"] = 0.0

    # ==================================================================
    # STEP 6: Full Network Validation
    # ==================================================================
    print("-" * 60)
    print("Step 6: Full Network Validation (5,936 nodes)")
    print("-" * 60)
    if run_full:
        try:
            full_results = bench_full_network_key_steps(n_repeat)
            json_output["step_times_5936"] = full_results
            t_full = full_results.get("DM_embedding", {}).get("time_s", 0.0)
            step_times["6. Full Network"] = t_full
            print(f"  Full network benchmark complete.\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")
            step_times["6. Full Network"] = 0.0
    else:
        print("  SKIPPED (use --run-full to enable)\n")
        step_times["6. Full Network"] = 0.0

    # ==================================================================
    # STEP 7: Geometric Analysis
    # ==================================================================
    print("-" * 60)
    print("Step 7: Geometric Analysis")
    print("-" * 60)
    try:
        mean_t, std_t = bench_step7_geometric(n_repeat)
        step_times["7. Geometric"] = mean_t
        json_output["step_times_153"]["step_7_geometric_analysis"] = {
            "time_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "complexity": STEP_COMPLEXITY["step_7_geometric_analysis"],
        }
        print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        step_times["7. Geometric"] = 0.0

    # ==================================================================
    # STEP 8: Link Prediction
    # ==================================================================
    print("-" * 60)
    print("Step 8: Link Prediction (5-fold CV)")
    print("-" * 60)
    try:
        mean_t, std_t = bench_step8_link_prediction(n_repeat)
        step_times["8. Link Pred"] = mean_t
        json_output["step_times_153"]["step_8_link_prediction"] = {
            "time_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "complexity": STEP_COMPLEXITY["step_8_link_prediction"],
        }
        print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        step_times["8. Link Pred"] = 0.0

    # ==================================================================
    # STEP 9: Downstream k-NN
    # ==================================================================
    print("-" * 60)
    print("Step 9: Downstream k-NN Evaluation")
    print("-" * 60)
    try:
        mean_t, std_t = bench_step9_downstream_knn(n_repeat)
        step_times["9. k-NN"] = mean_t
        json_output["step_times_153"]["step_9_downstream_knn"] = {
            "time_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "complexity": STEP_COMPLEXITY["step_9_downstream_knn"],
        }
        print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        step_times["9. k-NN"] = 0.0

    # ==================================================================
    # STEP 10: Randomization Control
    # ==================================================================
    print("-" * 60)
    print("Step 10: Randomization Control")
    print("-" * 60)
    try:
        mean_t, std_t = bench_step10_randomization(n_repeat)
        step_times["10. Randomization"] = mean_t
        json_output["step_times_153"]["step_10_randomization_control"] = {
            "time_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "complexity": STEP_COMPLEXITY["step_10_randomization_control"],
        }
        print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        step_times["10. Randomization"] = 0.0

    # ==================================================================
    # STEP 11: Sampling Density
    # ==================================================================
    print("-" * 60)
    print("Step 11: Sampling Density Verification")
    print("-" * 60)
    try:
        mean_t, std_t = bench_step11_sampling_density(n_repeat)
        step_times["11. Sampling"] = mean_t
        json_output["step_times_153"]["step_11_sampling_density"] = {
            "time_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "complexity": STEP_COMPLEXITY["step_11_sampling_density"],
        }
        print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        step_times["11. Sampling"] = 0.0

    # ==================================================================
    # STEP 12: Sensitivity Analysis
    # ==================================================================
    print("-" * 60)
    print("Step 12: G-F Score Sensitivity Analysis")
    print("-" * 60)
    try:
        mean_t, std_t = bench_step12_sensitivity(n_repeat)
        step_times["12. Sensitivity"] = mean_t
        json_output["step_times_153"]["step_12_sensitivity_analysis"] = {
            "time_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "complexity": STEP_COMPLEXITY["step_12_sensitivity_analysis"],
        }
        print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        step_times["12. Sensitivity"] = 0.0

    # ==================================================================
    # STEP 13: Human Validation
    # ==================================================================
    print("-" * 60)
    print("Step 13: Human Cross-Species Validation")
    print("-" * 60)
    if run_human:
        try:
            mean_t, std_t = bench_step13_human(n_repeat)
            step_times["13. Human Val"] = mean_t
            json_output["step_times_153"]["step_13_human_validation"] = {
                "time_s": round(mean_t, 4),
                "time_std_s": round(std_t, 4),
                "complexity": STEP_COMPLEXITY["step_13_human_validation"],
            }
            print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")
            step_times["13. Human Val"] = 0.0
    else:
        print("  SKIPPED (use --run-human to enable)\n")
        step_times["13. Human Val"] = 0.0

    # ==================================================================
    # STEP 14: Figure Generation
    # ==================================================================
    print("-" * 60)
    print("Step 14: Figure Generation")
    print("-" * 60)
    try:
        mean_t, std_t = bench_step14_figures(n_repeat)
        step_times["14. Figures"] = mean_t
        json_output["step_times_153"]["step_14_figure_generation"] = {
            "time_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "complexity": STEP_COMPLEXITY["step_14_figure_generation"],
        }
        print(f"  Mean: {mean_t:.4f}s (+/- {std_t:.4f}s)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        step_times["14. Figures"] = 0.0

    # ==================================================================
    # Scalability Experiment
    # ==================================================================
    print("-" * 60)
    print("Scalability Experiment: G-F curve vs sampling points")
    print("-" * 60)
    # Use DM coords on 153-node network
    common_dm = sorted(set(dm_nodes) & set(nodes) & set(go_map.keys()))
    dm_node_to_idx = {n: i for i, n in enumerate(dm_nodes)}
    dm_idx = [dm_node_to_idx[n] for n in common_dm]
    aligned_dm = dm_coords[dm_idx]

    scal_data = scalability_experiment(
        aligned_dm, common_dm, go_map, sampling_points
    )
    json_output["scalability_gf_curves"] = scal_data
    print()

    # ==================================================================
    # Pipeline total
    # ==================================================================
    pipeline_elapsed = time.perf_counter() - pipeline_t0
    json_output["total_pipeline_time_s"] = round(pipeline_elapsed, 4)

    # ==================================================================
    # Build summary table: theoretical + empirical per method x step
    # ==================================================================
    method_step_table = {}
    for method in ALL_METHODS:
        entry = {
            "embedding_complexity": THEORETICAL_COMPLEXITY["embedding"].get(
                method, "N/A"
            ),
            "embedding_time_s": emb_results.get(method, {}).get("time_s"),
            "gf_curve_time_s": gf_times.get(method, {}).get("time_s"),
        }
        method_step_table[method] = entry
    json_output["method_step_table"] = method_step_table

    # ==================================================================
    # Save JSON
    # ==================================================================
    json_path = results_dir / "benchmark_runtime.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Saved benchmark results to: {json_path}")

    # ==================================================================
    # Generate Figures S6 and S7
    # ==================================================================
    print("\nGenerating supplementary figures ...")
    # Remove zero-time steps from bar chart for cleanliness
    chart_times = {k: v for k, v in step_times.items() if v > 0}
    if chart_times:
        plot_fig_s6_runtime_breakdown(chart_times, figures_dir)
    if scal_data["n_points"]:
        plot_fig_s7_time_accuracy_tradeoff(scal_data, figures_dir)

    # ==================================================================
    # Print summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n{'Step':<30s} {'Time (s)':>10s} {'Complexity':<30s}")
    print("-" * 70)
    for label, t in step_times.items():
        complexity = ""
        # Map short labels back to complexity keys
        for key in STEP_COMPLEXITY:
            if label.split(". ")[0] in key:
                c = STEP_COMPLEXITY[key]
                complexity = c if isinstance(c, str) else str(c)
                break
        print(f"  {label:<28s} {t:>10.3f}   {complexity:<30s}")
    print("-" * 70)
    total_tracked = sum(step_times.values())
    print(f"  {'Total (tracked steps)':<28s} {total_tracked:>10.3f}")
    print(f"  {'Total (wall-clock)':<28s} {pipeline_elapsed:>10.3f}")
    print(f"\n  Scalability (DM, 153-node):")
    for n_pts, t_s, ms_pt in zip(
        scal_data["n_points"], scal_data["times_s"],
        scal_data["time_per_point_ms"]
    ):
        print(f"    K={n_pts:>4d}: {t_s:.3f}s  ({ms_pt:.2f} ms/point)")

    print("\n  Theoretical Complexity (dominant terms):")
    print(f"    Embedding (DM/MDS/Spectral): O(n^3)")
    print(f"    Embedding (DeepWalk/Node2Vec): O(n*w*l)")
    print(f"    Embedding (VGAE): O(E*epochs)")
    print(f"    G-F Curve (K points): O(K * n^2 * log(n))")
    print(f"    G-F Score: O(K)")
    print(f"    Leiden: O(n*log(n))")
    print(f"    Total pipeline: O(n^3 + K*n^2*log(n))")

    print(f"\n  Pipeline complete in {pipeline_elapsed / 60:.1f} min.")
    print(f"  Results: {json_path}")
    print(f"  Figures: {figures_dir / 'FigS6_runtime_breakdown.png'}")
    print(f"           {figures_dir / 'FigS7_time_accuracy_tradeoff.png'}")
    print()


if __name__ == "__main__":
    main()
