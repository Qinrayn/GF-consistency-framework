#!/usr/bin/env python3
"""
G-F Consistency Framework — Hyperbolic Space Embeddings
========================================================
Implements Poincare Ball model embeddings for hierarchical PPI networks.

The Poincare Ball is the Riemannian manifold of negative curvature, ideally
suited for representing tree-like / hierarchical structures common in
biological networks (e.g., protein complexes within pathways within
functional modules).

Methods
-------
- ``poincare_ball_embedding``: Direct optimisation on the Poincare Ball
  using Riemannian SGD (requires ``geoopt`` if available, falls back to
  projected gradient descent otherwise).

- ``hyperbolic_distance``: Compute pairwise geodesic distances on the
  Poincare Ball for downstream G-F curve analysis.

References
----------
Nickel & Kiela (2017). Poincare Embeddings for Learning Hierarchical
Representations. NeurIPS.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Poincare Ball geometry
# ---------------------------------------------------------------------------

# Curvature (c < 0; we use c = -1 so radius = 1)
_CURVATURE: float = -1.0
_EPS: float = 1e-5


def _project_to_ball(x: np.ndarray, c: float = _CURVATURE) -> np.ndarray:
    """Project points onto the open Poincare Ball (||x|| < 1/sqrt(|c|))."""
    max_norm = 1.0 / np.sqrt(abs(c)) - _EPS
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    cond = norms > max_norm
    return np.where(cond, x / norms * max_norm, x)


def _mobius_add(x: np.ndarray, y: np.ndarray, c: float = _CURVATURE) -> np.ndarray:
    """Mobius addition on the Poincare Ball."""
    c_abs = abs(c)
    x2 = np.sum(x ** 2, axis=-1, keepdims=True)
    y2 = np.sum(y ** 2, axis=-1, keepdims=True)
    xy = np.sum(x * y, axis=-1, keepdims=True)
    num = (1 + 2 * c_abs * xy + c_abs * y2) * x + (1 - c_abs * x2) * y
    denom = 1 + 2 * c_abs * xy + c_abs ** 2 * x2 * y2
    return _project_to_ball(num / (denom + _EPS))


def _exp_map(x: np.ndarray, v: np.ndarray, c: float = _CURVATURE) -> np.ndarray:
    """Exponential map: move from x along tangent vector v."""
    c_abs = abs(c)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + _EPS
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True) + _EPS
    lambda_x = 2.0 / (1.0 - c_abs * x_norm ** 2 + _EPS)
    y = np.tanh(c_abs ** 0.5 * lambda_x * v_norm / 2.0) * v / (v_norm * c_abs ** 0.5 + _EPS)
    return _mobius_add(x, y, c)


def _log_map(x: np.ndarray, y: np.ndarray, c: float = _CURVATURE) -> np.ndarray:
    """Logarithmic map: tangent vector at x pointing to y."""
    c_abs = abs(c)
    xy = _mobius_add(-x, y, c)
    xy_norm = np.linalg.norm(xy, axis=-1, keepdims=True) + _EPS
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True) + _EPS
    lambda_x = 2.0 / (1.0 - c_abs * x_norm ** 2 + _EPS)
    return xy / xy_norm * np.arctanh(np.clip(c_abs ** 0.5 * xy_norm, -1 + _EPS, 1 - _EPS)) / (c_abs ** 0.5 + _EPS)


def hyperbolic_distance(x: np.ndarray, y: np.ndarray,
                        c: float = _CURVATURE) -> np.ndarray:
    """Pairwise geodesic distance on the Poincare Ball.

    Parameters
    ----------
    x, y : (n, d) and (m, d) arrays
    c : curvature (default -1)

    Returns
    -------
    (n, m) distance matrix
    """
    c_abs = abs(c)
    diff = _mobius_add(-x[:, np.newaxis, :], y[np.newaxis, :, :], c)
    diff_norm = np.linalg.norm(diff, axis=-1)
    clipped = np.clip(c_abs ** 0.5 * diff_norm, 0, 1 - _EPS)
    return 2.0 * np.arctanh(clipped) / c_abs ** 0.5


def poincare_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute the full pairwise hyperbolic distance matrix."""
    n = coords.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hyperbolic_distance(
                coords[i:i+1], coords[j:j+1]
            )[0, 0]
            D[i, j] = d
            D[j, i] = d
    return D


# ---------------------------------------------------------------------------
# Embedding via Riemannian optimisation
# ---------------------------------------------------------------------------

def poincare_ball_embedding(
    G: nx.Graph,
    dim: int = 2,
    epochs: int = 300,
    lr: float = 0.01,
    n_neg: int = 10,
    seed: int = 42,
    nodelist: Optional[list] = None,
) -> np.ndarray:
    """Embed a graph onto the Poincare Ball via Riemannian SGD.

    Minimises the negative sampling loss:
        L = -log(exp(-d(u,v)) / sum exp(-d(u,v')))
    where (u,v) are edges and v' are negative samples.

    Parameters
    ----------
    G : networkx Graph
    dim : embedding dimension (2 for visualisation)
    epochs : training iterations
    lr : learning rate
    n_neg : negative samples per positive edge
    seed : random seed
    nodelist : optional node ordering

    Returns
    -------
    (n, dim) array of Poincare Ball coordinates
    """
    np.random.seed(seed)
    current_lr = lr

    if nodelist is None:
        nodelist = list(G.nodes())
    n = len(nodelist)
    node_to_idx = {u: i for i, u in enumerate(nodelist)}

    # Build adjacency
    edges = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            edges.append((node_to_idx[u], node_to_idx[v]))
    edges = np.array(edges)

    # Initialise near origin (small random)
    coords = np.random.randn(n, dim) * 0.01
    coords = _project_to_ball(coords)

    # Riemannian SGD (simplified — no geoopt dependency)
    for epoch in range(epochs):
        # Sample positive edges
        batch_edges = edges[np.random.choice(len(edges), min(64, len(edges)), replace=True)]

        for u, v in batch_edges:
            # Positive pair
            z_u = coords[u]
            z_v = coords[v]
            d_pos = hyperbolic_distance(z_u[np.newaxis], z_v[np.newaxis])[0, 0]

            # Negative samples
            neg_idx = np.random.choice(n, size=n_neg, replace=True)
            z_neg = coords[neg_idx]
            d_neg = hyperbolic_distance(z_u[np.newaxis], z_neg)[0]

            # Gradient approximation (numerical)
            eps_grad = 1e-4
            grad_u = np.zeros(dim)
            for d in range(dim):
                z_u_plus = z_u.copy()
                z_u_plus[d] += eps_grad
                z_u_plus = _project_to_ball(z_u_plus[np.newaxis])[0]

                d_pos_plus = hyperbolic_distance(z_u_plus[np.newaxis], z_v[np.newaxis])[0, 0]
                d_neg_plus = hyperbolic_distance(z_u_plus[np.newaxis], z_neg)[0]

                loss_base = d_pos + np.log(np.sum(np.exp(-d_neg)) + _EPS)
                loss_plus = d_pos_plus + np.log(np.sum(np.exp(-d_neg_plus)) + _EPS)
                grad_u[d] = (loss_plus - loss_base) / eps_grad

            # Riemannian gradient (scale by conformal factor)
            c_abs = abs(_CURVATURE)
            u_norm = np.linalg.norm(z_u)
            lambda_u = 2.0 / (1.0 - c_abs * u_norm ** 2 + _EPS)
            riem_grad = grad_u / (lambda_u ** 2 + _EPS)

            # Update with retraction
            coords[u] = _exp_map(z_u[np.newaxis], -current_lr * riem_grad[np.newaxis])[0]
            coords[u] = _project_to_ball(coords[u][np.newaxis])[0]

        # Decay learning rate
        if (epoch + 1) % 100 == 0:
            current_lr *= 0.8

    return coords


# ---------------------------------------------------------------------------
# Integration with the G-F pipeline
# ---------------------------------------------------------------------------

def compute_gf_curve_hyperbolic(
    coords: np.ndarray,
    nodes: list,
    go_map: dict,
    n_points: int = 200,
    r_max_frac: float = 0.5,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Compute G-F curves using hyperbolic distances.

    Parameters
    ----------
    coords : (n, d) Poincare Ball coordinates
    nodes : node labels
    go_map : gene -> GO terms mapping
    n_points : number of r-values
    r_max_frac : fraction of max pairwise distance to use as r_max

    Returns
    -------
    (r_vals, purities, modularities)
    """
    from scripts.utils import (
        build_spatial_graph_fast,
        functional_purity,
    )
    from networkx.algorithms.community import greedy_modularity_communities, modularity

    # Compute hyperbolic distance matrix
    D = poincare_distance_matrix(coords)

    r_max = float(D.max()) * r_max_frac
    r_vals = np.linspace(0.001, r_max, n_points)

    purities = []
    modularities = []
    for r in r_vals:
        G_r = build_spatial_graph_fast(D, r)
        if G_r.number_of_edges() == 0:
            purities.append(0.0)
            modularities.append(0.0)
            continue
        communities = list(greedy_modularity_communities(G_r))
        purities.append(functional_purity(communities, go_map, nodes))
        if len(communities) > 1:
            modularities.append(modularity(G_r, communities))
        else:
            modularities.append(0.0)

    return r_vals, purities, modularities


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def main():
    """Compute Poincare Ball embedding and hyperbolic G-F curve.

    Saves:
      - embeddings/hyperbolic_153.npy + _nodes.json
      - results/hyperbolic_gf_curves.json
    """
    import json
    from pathlib import Path
    from scripts.utils import (
        SEED, TARGET_STD, get_data_dir, get_embeddings_dir, get_results_dir,
        load_curated_network, load_embedding, rescale_coordinates,
        coords_to_dict,
    )

    np.random.seed(SEED)

    data_dir = get_data_dir()
    emb_dir = get_embeddings_dir()
    results_dir = get_results_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load network
    print("Loading curated network...")
    G, nodes, go_map = load_curated_network(data_dir)

    # Compute Poincare Ball embedding
    print("Computing Poincare Ball embedding (dim=2, epochs=300)...")
    coords = poincare_ball_embedding(G, dim=2, epochs=300, lr=0.01, nodelist=nodes)

    # Save embedding
    emb_file = emb_dir / "hyperbolic_153.npy"
    nodes_file = emb_dir / "hyperbolic_153_nodes.json"
    np.save(emb_file, coords)
    with open(nodes_file, "w") as f:
        json.dump(nodes, f)
    print(f"  Saved embedding to {emb_file}")

    # Compute hyperbolic G-F curve
    print("Computing hyperbolic G-F curve...")
    common_nodes = sorted(set(nodes) & set(go_map.keys()))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    node_indices = [node_to_idx[n] for n in common_nodes]
    aligned_coords = coords[node_indices]

    r_vals, purities, modularities = compute_gf_curve_hyperbolic(
        aligned_coords, common_nodes, go_map,
    )

    # Save results
    output = {
        "method": "hyperbolic",
        "r_values": r_vals.tolist(),
        "purity": purities,
        "modularity": modularities,
        "n_nodes": len(common_nodes),
    }
    out_file = results_dir / "hyperbolic_gf_curves.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved G-F curves to {out_file}")
    print(f"  Purity range: [{min(purities):.3f}, {max(purities):.3f}]")
