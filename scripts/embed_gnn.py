#!/usr/bin/env python3
"""
embed_gnn.py
Compute 3 GNN embedding methods (GraphSAGE, GAT, GIN) on the curated
153-node yeast PPI network and optionally on the full STRING network.

After saving embeddings, computes G-F curves, G-F Scores, plateau widths,
link prediction AUROC (5-fold CV), and k-NN micro-F1 for the new methods.
Results are saved to ``results/gnn_gf_scores.json``.

Methods: GraphSAGE, GAT, GIN
"""

from __future__ import annotations

import sys
import json
import random
import argparse
from typing import Optional

import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_full_STRING_network, load_embedding,
    compute_centrality_features, rescale_coordinates, save_embedding,
    compute_gf_curve, compute_gf_score, compute_plateau_width,
    precompute_distance_matrix,
    check_embedding_collapse, align_embedding_to_nodes,
    CLASSICAL_METHODS, ALL_CURATED_METHODS, GNN_METHODS, ALL_METHODS,
    GF_R_MIN, GF_R_MAX, R_MIN, R_MAX, N_POINTS, TARGET_STD,
    CV_FOLDS, K_NEIGHBORS, MIN_LABEL_COUNT, PLATEAU_RELATIVE_THRESHOLD,
)

# Seeds are set inside main() to avoid side-effects on import.


# ============================================================
# GNN Encoder Builders (with BatchNorm1d)
# ============================================================

def _build_sage_encoder(in_dim: int, hidden_dim: int, latent_dim: int,
                        use_bn: bool = True):
    """Build a 2-layer GraphSAGE encoder with optional BatchNorm.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality.
    hidden_dim : int
        Hidden layer dimensionality.
    latent_dim : int
        Output embedding dimensionality.
    use_bn : bool
        Whether to apply BatchNorm1d after each convolution layer.

    Returns
    -------
    torch.nn.Module
        A SAGEEncoder instance.
    """
    import torch.nn as nn
    import torch.nn.functional as F_func
    from torch_geometric.nn import SAGEConv

    class SAGEEncoder(nn.Module):
        """2-layer GraphSAGE encoder with mean aggregation."""

        def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int,
                     use_bn: bool = True):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
            self.conv2 = SAGEConv(hidden_dim, latent_dim, aggr="mean")
            self.use_bn = use_bn
            if use_bn:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(latent_dim)

        def forward(self, x, edge_index):
            h = self.conv1(x, edge_index)
            if self.use_bn:
                h = self.bn1(h)
            h = F_func.relu(h)
            z = self.conv2(h, edge_index)
            if self.use_bn:
                z = self.bn2(z)
            return z

    return SAGEEncoder(in_dim, hidden_dim, latent_dim, use_bn)


def _build_gat_encoder(in_dim: int, hidden_dim: int, latent_dim: int,
                       use_bn: bool = True):
    """Build a 2-layer GAT encoder with optional BatchNorm.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality.
    hidden_dim : int
        Hidden layer dimensionality.
    latent_dim : int
        Output embedding dimensionality.
    use_bn : bool
        Whether to apply BatchNorm1d after each convolution layer.

    Returns
    -------
    torch.nn.Module
        A GATEncoder instance.
    """
    import torch.nn as nn
    import torch.nn.functional as F_func
    from torch_geometric.nn import GATConv

    class GATEncoder(nn.Module):
        """2-layer GAT encoder with single attention head."""

        def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int,
                     use_bn: bool = True):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden_dim, heads=1, concat=False)
            self.conv2 = GATConv(hidden_dim, latent_dim, heads=1, concat=False)
            self.use_bn = use_bn
            if use_bn:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(latent_dim)

        def forward(self, x, edge_index):
            h = self.conv1(x, edge_index)
            if self.use_bn:
                h = self.bn1(h)
            h = F_func.relu(h)
            z = self.conv2(h, edge_index)
            if self.use_bn:
                z = self.bn2(z)
            return z

    return GATEncoder(in_dim, hidden_dim, latent_dim, use_bn)


def _build_gin_encoder(in_dim: int, hidden_dim: int, latent_dim: int,
                       use_bn: bool = True):
    """Build a 2-layer GIN encoder with optional BatchNorm.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality.
    hidden_dim : int
        Hidden layer dimensionality.
    latent_dim : int
        Output embedding dimensionality.
    use_bn : bool
        Whether to apply BatchNorm1d after each convolution layer.

    Returns
    -------
    torch.nn.Module
        A GINEncoder instance.
    """
    import torch.nn as nn
    import torch.nn.functional as F_func
    from torch_geometric.nn import GINConv

    class GINEncoder(nn.Module):
        """2-layer GIN encoder with learnable MLPs."""

        def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int,
                     use_bn: bool = True):
            super().__init__()
            mlp1_layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ]
            if use_bn:
                mlp1_layers.insert(1, nn.BatchNorm1d(hidden_dim))
            mlp1 = nn.Sequential(*mlp1_layers)

            mlp2_layers = [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            ]
            if use_bn:
                mlp2_layers.insert(1, nn.BatchNorm1d(hidden_dim))
            mlp2 = nn.Sequential(*mlp2_layers)

            self.conv1 = GINConv(mlp1)
            self.conv2 = GINConv(mlp2)
            self.use_bn = use_bn
            if use_bn:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(latent_dim)

        def forward(self, x, edge_index):
            h = self.conv1(x, edge_index)
            if self.use_bn:
                h = self.bn1(h)
            h = F_func.relu(h)
            z = self.conv2(h, edge_index)
            if self.use_bn:
                z = self.bn2(z)
            return z

    return GINEncoder(in_dim, hidden_dim, latent_dim, use_bn)


# ============================================================
# Generic GNN Training Loop
# ============================================================

def _train_gnn_encoder(
    G: nx.Graph,
    encoder_builder,
    hidden_dim: int = 16,
    latent_dim: int = 2,
    epochs: int = 300,
    lr: float = 0.01,
    features: Optional[np.ndarray] = None,
    seed: int = SEED,
) -> np.ndarray:
    """Generic 2-layer GNN encoder training with BCE reconstruction loss.

    Trains the encoder produced by *encoder_builder* using an inner-product
    decoder and binary cross-entropy loss against the graph adjacency matrix.

    Parameters
    ----------
    G : networkx.Graph
        Input protein-protein interaction graph.
    encoder_builder : callable
        Factory ``(in_dim, hidden_dim, latent_dim, use_bn) -> nn.Module``.
    hidden_dim : int
        Hidden layer dimensionality (default 16).
    latent_dim : int
        Output embedding dimensionality (default 2).
    epochs : int
        Number of training epochs (default 300).
    lr : float
        Adam learning rate (default 0.01).
    features : np.ndarray or None
        Node feature matrix of shape ``(n, d)``.  If *None*, one-hot
        identity features are used.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Embedding coordinates of shape ``(n, latent_dim)``.
    """
    import torch
    import torch.nn.functional as F
    from torch_geometric.utils import from_networkx

    torch.manual_seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    n = len(nodes)
    data = from_networkx(G)

    if features is not None:
        data.x = torch.tensor(features, dtype=torch.float32)
        in_dim = features.shape[1]
    else:
        data.x = torch.eye(n)
        in_dim = n

    model = encoder_builder(in_dim, hidden_dim, latent_dim, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Precompute adjacency target (matches VGAE pattern in utils.py)
    adj_target = torch.zeros(n, n)
    ei = data.edge_index
    adj_target[ei[0], ei[1]] = 1.0

    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        adj_recon = torch.sigmoid(z @ z.T)
        recon_loss = F.binary_cross_entropy(
            adj_recon, adj_target, reduction="sum"
        )
        loss = recon_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        coords = z.numpy()
    return coords


# ============================================================
# GNN Embedding Functions
# ============================================================

def graphsage_from_graph(
    G: nx.Graph,
    hidden_dim: int = 16,
    latent_dim: int = 2,
    epochs: int = 300,
    lr: float = 0.01,
    features: Optional[np.ndarray] = None,
    seed: int = SEED,
) -> np.ndarray:
    """GraphSAGE embedding (2D) with 2-layer mean-aggregation SAGE.

    Trains a 2-layer GraphSAGE encoder with BCE reconstruction loss
    (inner-product decoder) and returns 2-D node embeddings.

    Parameters
    ----------
    G : networkx.Graph
        Input protein-protein interaction graph.
    hidden_dim : int
        Hidden layer dimensionality (default 16).
    latent_dim : int
        Output embedding dimensionality (default 2).
    epochs : int
        Number of training epochs (default 300).
    lr : float
        Adam learning rate (default 0.01).
    features : np.ndarray or None
        Node feature matrix of shape ``(n, d)``.  If *None*, one-hot
        identity features are used.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Embedding coordinates of shape ``(n, latent_dim)``.
    """
    return _train_gnn_encoder(
        G, _build_sage_encoder, hidden_dim=hidden_dim,
        latent_dim=latent_dim, epochs=epochs, lr=lr,
        features=features, seed=seed,
    )


def gat_from_graph(
    G: nx.Graph,
    hidden_dim: int = 16,
    latent_dim: int = 2,
    epochs: int = 300,
    lr: float = 0.01,
    features: Optional[np.ndarray] = None,
    seed: int = SEED,
) -> np.ndarray:
    """GAT embedding (2D) with 2-layer graph attention.

    Trains a 2-layer GAT encoder (single attention head per layer) with
    BCE reconstruction loss and returns 2-D node embeddings.

    Parameters
    ----------
    G : networkx.Graph
        Input protein-protein interaction graph.
    hidden_dim : int
        Hidden layer dimensionality (default 16).
    latent_dim : int
        Output embedding dimensionality (default 2).
    epochs : int
        Number of training epochs (default 300).
    lr : float
        Adam learning rate (default 0.01).
    features : np.ndarray or None
        Node feature matrix of shape ``(n, d)``.  If *None*, one-hot
        identity features are used.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Embedding coordinates of shape ``(n, latent_dim)``.
    """
    return _train_gnn_encoder(
        G, _build_gat_encoder, hidden_dim=hidden_dim,
        latent_dim=latent_dim, epochs=epochs, lr=lr,
        features=features, seed=seed,
    )


def gin_from_graph(
    G: nx.Graph,
    hidden_dim: int = 16,
    latent_dim: int = 2,
    epochs: int = 300,
    lr: float = 0.01,
    features: Optional[np.ndarray] = None,
    seed: int = SEED,
) -> np.ndarray:
    """GIN embedding (2D) with 2-layer graph isomorphism network.

    Trains a 2-layer GIN encoder (each layer wraps a 2-layer MLP) with
    BCE reconstruction loss and returns 2-D node embeddings.

    Parameters
    ----------
    G : networkx.Graph
        Input protein-protein interaction graph.
    hidden_dim : int
        Hidden layer dimensionality (default 16).
    latent_dim : int
        Output embedding dimensionality (default 2).
    epochs : int
        Number of training epochs (default 300).
    lr : float
        Adam learning rate (default 0.01).
    features : np.ndarray or None
        Node feature matrix of shape ``(n, d)``.  If *None*, one-hot
        identity features are used.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Embedding coordinates of shape ``(n, latent_dim)``.
    """
    return _train_gnn_encoder(
        G, _build_gin_encoder, hidden_dim=hidden_dim,
        latent_dim=latent_dim, epochs=epochs, lr=lr,
        features=features, seed=seed,
    )


# ============================================================
# High-level wrappers (match embed_all.py pattern)
# ============================================================

def embed_graphsage(
    G: nx.Graph,
    nodes: list,
    features: Optional[np.ndarray] = None,
    hidden_dim: int = 16,
    latent_dim: int = 2,
    epochs: int = 300,
    lr: float = 0.01,
) -> np.ndarray:
    """GraphSAGE: 2-layer mean-aggregation SAGE -> 2D embedding."""
    coords = graphsage_from_graph(
        G, hidden_dim=hidden_dim, latent_dim=latent_dim,
        epochs=epochs, lr=lr, features=features, seed=SEED,
    )
    collapse_info = check_embedding_collapse(coords, "GraphSAGE")
    if collapse_info["collapsed"]:
        print(f"  WARNING: GraphSAGE embedding collapsed: "
              f"{collapse_info['reasons']}")
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_gat(
    G: nx.Graph,
    nodes: list,
    features: Optional[np.ndarray] = None,
    hidden_dim: int = 16,
    latent_dim: int = 2,
    epochs: int = 300,
    lr: float = 0.01,
) -> np.ndarray:
    """GAT: 2-layer graph attention -> 2D embedding."""
    coords = gat_from_graph(
        G, hidden_dim=hidden_dim, latent_dim=latent_dim,
        epochs=epochs, lr=lr, features=features, seed=SEED,
    )
    collapse_info = check_embedding_collapse(coords, "GAT")
    if collapse_info["collapsed"]:
        print(f"  WARNING: GAT embedding collapsed: "
              f"{collapse_info['reasons']}")
    return rescale_coordinates(coords, target_std=TARGET_STD)


def embed_gin(
    G: nx.Graph,
    nodes: list,
    features: Optional[np.ndarray] = None,
    hidden_dim: int = 16,
    latent_dim: int = 2,
    epochs: int = 300,
    lr: float = 0.01,
) -> np.ndarray:
    """GIN: 2-layer graph isomorphism network -> 2D embedding."""
    coords = gin_from_graph(
        G, hidden_dim=hidden_dim, latent_dim=latent_dim,
        epochs=epochs, lr=lr, features=features, seed=SEED,
    )
    collapse_info = check_embedding_collapse(coords, "GIN")
    if collapse_info["collapsed"]:
        print(f"  WARNING: GIN embedding collapsed: "
              f"{collapse_info['reasons']}")
    return rescale_coordinates(coords, target_std=TARGET_STD)


# ============================================================
# Evaluation Helpers
# ============================================================

def compute_gf_curves_and_scores(
    method_coords: dict[str, np.ndarray],
    nodes: list[str],
    go_map: dict[str, list[str]],
    r_vals: np.ndarray,
) -> tuple[dict[str, list[float]], dict[str, float], dict[str, dict]]:
    """Compute G-F purity curves and G-F Scores for embedding methods.

    Parameters
    ----------
    method_coords : dict[str, np.ndarray]
        Mapping from method name to coordinate array of shape ``(n, 2)``.
    nodes : list[str]
        Ordered node labels corresponding to rows in each coordinate array.
    go_map : dict[str, list[str]]
        Gene-to-GO-term annotation map.
    r_vals : np.ndarray
        Radius grid for G-F curve evaluation.

    Returns
    -------
    all_purities : dict[str, list[float]]
    all_gf_scores : dict[str, float]
    all_plateau_widths : dict[str, dict]
    """
    all_purities = {}
    all_gf_scores = {}
    all_plateau_widths = {}

    for method, coords in method_coords.items():
        print(f"  Computing G-F curve for {method}...")
        common_nodes = sorted(set(nodes) & set(go_map.keys()))
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        node_indices = [node_to_idx[n] for n in common_nodes]
        aligned_coords = coords[node_indices]

        purities, _ = compute_gf_curve(
            aligned_coords, common_nodes, go_map, r_vals
        )
        all_purities[method] = purities

        score = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)
        all_gf_scores[method] = score
        print(f"    G-F Score: {score:.4f}")

        pw = compute_plateau_width(
            r_vals, purities, relative_threshold=PLATEAU_RELATIVE_THRESHOLD
        )
        all_plateau_widths[method] = {
            "W": round(pw["W"], 4),
            "r_min": round(pw["r_min"], 4),
            "r_max": round(pw["r_max"], 4),
            "peak_purity": round(pw["peak_purity"], 4),
        }
        print(f"    Plateau width W: {pw['W']:.4f} "
              f"(peak purity: {pw['peak_purity']:.4f})")

    return all_purities, all_gf_scores, all_plateau_widths


def evaluate_link_prediction(
    method_coords: dict[str, np.ndarray],
    nodes: list[str],
    G: nx.Graph,
) -> dict[str, dict]:
    """Link prediction AUROC with 5-fold CV logistic regression.

    Follows the same protocol as ``link_prediction.py``: Hadamard product
    features, balanced positive/negative edges, stratified 5-fold CV.

    Parameters
    ----------
    method_coords : dict[str, np.ndarray]
        Mapping from method name to coordinate array.
    nodes : list[str]
        Ordered node labels.
    G : networkx.Graph
        Source graph (positive edges).

    Returns
    -------
    dict[str, dict]
        ``{method: {"auroc_mean": float, "auroc_std": float}}``
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    all_edges = list(G.edges())
    nodes_list = list(G.nodes())
    n_nodes = len(nodes_list)
    edge_set = set(frozenset([u, v]) for u, v in all_edges)

    # Generate negative samples (same count as positive edges)
    np.random.seed(SEED)
    all_non_edges: list[tuple[str, str]] = []
    max_neg = len(all_edges)
    neg_count = 0
    attempts = 0
    while neg_count < max_neg and attempts < max_neg * 20:
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        if i != j and frozenset([nodes_list[i], nodes_list[j]]) not in edge_set:
            all_non_edges.append((nodes_list[i], nodes_list[j]))
            edge_set.add(frozenset([nodes_list[i], nodes_list[j]]))
            neg_count += 1
        attempts += 1

    print(f"  Positive edges: {len(all_edges)}, "
          f"Negative edges: {len(all_non_edges)}")

    auroc_results: dict[str, dict] = {}

    for method, coords in method_coords.items():
        print(f"  Evaluating {method} link prediction...")
        emb_dict = {nodes[i]: coords[i] for i in range(len(nodes))}

        pos_edges = [(u, v) for u, v in all_edges
                     if u in emb_dict and v in emb_dict]
        neg_edges = [(u, v) for u, v in all_non_edges
                     if u in emb_dict and v in emb_dict]
        min_len = min(len(pos_edges), len(neg_edges))
        pos_edges = pos_edges[:min_len]
        neg_edges = neg_edges[:min_len]

        features_list = []
        labels = []
        for u, v in pos_edges:
            features_list.append(emb_dict[u] * emb_dict[v])
            labels.append(1)
        for u, v in neg_edges:
            features_list.append(emb_dict[u] * emb_dict[v])
            labels.append(0)

        X = np.array(features_list)
        y = np.array(labels)

        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                              random_state=SEED)
        aurocs = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf = LogisticRegression(random_state=SEED, max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            aurocs.append(roc_auc_score(y_test, y_pred))

        auroc_mean = float(np.mean(aurocs))
        auroc_std = float(np.std(aurocs))
        auroc_results[method] = {
            "auroc_mean": round(auroc_mean, 4),
            "auroc_std": round(auroc_std, 4),
        }
        print(f"    AUROC: {auroc_mean:.4f} +/- {auroc_std:.4f}")

    return auroc_results


def evaluate_downstream_knn(
    method_coords: dict[str, np.ndarray],
    nodes: list[str],
    go_map: dict[str, list[str]],
) -> dict[str, dict]:
    """k-NN GO term prediction micro-F1 (5-fold CV, k=5).

    Follows the same protocol as ``downstream_knn.py``: most-frequent GO
    term per protein, labels appearing >= 3 times, 5-NN classifier.

    Parameters
    ----------
    method_coords : dict[str, np.ndarray]
        Mapping from method name to coordinate array.
    nodes : list[str]
        Ordered node labels.
    go_map : dict[str, list[str]]
        Gene-to-GO-term annotation map.

    Returns
    -------
    dict[str, dict]
        ``{method: {"micro_f1_mean": float, "micro_f1_std": float}}``
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from collections import Counter

    # Get most frequent GO term per protein
    labels: dict[str, str] = {}
    for node, terms in go_map.items():
        if terms:
            term_counts = Counter(terms)
            labels[node] = term_counts.most_common(1)[0][0]

    # Filter: labels appearing >= MIN_LABEL_COUNT times
    label_counts = Counter(labels.values())
    valid_labels = {lbl for lbl, cnt in label_counts.items()
                    if cnt >= MIN_LABEL_COUNT}
    valid_nodes = sorted(
        [n for n in nodes if n in labels and labels[n] in valid_labels]
    )

    print(f"  k-NN valid nodes: {len(valid_nodes)}, "
          f"categories: {len(valid_labels)}")

    y_raw = [labels[n] for n in valid_nodes]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    knn_results: dict[str, dict] = {}

    for method, coords in method_coords.items():
        print(f"  Evaluating {method} k-NN...")
        emb_dict = {nodes[i]: coords[i] for i in range(len(nodes))}

        mask = [n in emb_dict for n in valid_nodes]
        X = np.array([emb_dict[n] for n, m in zip(valid_nodes, mask) if m])
        y_filtered = np.array([yi for yi, m in zip(y, mask) if m])

        try:
            knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
            scores = cross_val_score(
                knn, X, y_filtered, cv=CV_FOLDS, scoring="f1_micro"
            )
            f1_mean = float(np.mean(scores))
            f1_std = float(np.std(scores))
            knn_results[method] = {
                "micro_f1_mean": round(f1_mean, 4),
                "micro_f1_std": round(f1_std, 4),
            }
            print(f"    micro-F1: {f1_mean:.3f} +/- {f1_std:.3f}")
        except Exception as e:
            print(f"    {method} k-NN FAILED: {e}")

    return knn_results


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run GNN embeddings and G-F evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute GNN embeddings (GraphSAGE, GAT, GIN) "
                    "and evaluate with G-F metrics.",
    )
    parser.add_argument(
        "--full-network",
        action="store_true",
        help="Also run on the full 5,936-node STRING network.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam learning rate (default: 0.01).",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)

    # ---- Determine network and subset label ----
    if args.full_network:
        print("Loading full STRING network...")
        G = load_full_STRING_network(data_dir)
        nodes = sorted(G.nodes())
        subset = "full"
        # Load GO map for annotated subset evaluation
        with open(data_dir / "gene_go_map.json") as f:
            go_map = json.load(f)
        annotated = sorted(set(go_map.keys()) & set(nodes))
        print(f"Full network: {len(nodes)} nodes, "
              f"{G.number_of_edges()} edges")
        print(f"Annotated subset: {len(annotated)} nodes")
    else:
        print("Loading curated 153-node network...")
        G, nodes, go_map = load_curated_network(data_dir)
        subset = "153"
        print(f"Network: {len(nodes)} nodes, "
              f"{G.number_of_edges()} edges")

    # Compute centrality features once (used as node features for GNNs)
    print("Computing centrality features...")
    features = compute_centrality_features(G, nodes)

    # ---- GNN embedding methods ----
    methods: dict[str, callable] = {
        "GraphSAGE": lambda: embed_graphsage(
            G, nodes, features=features,
            epochs=args.epochs, lr=args.lr,
        ),
        "GAT": lambda: embed_gat(
            G, nodes, features=features,
            epochs=args.epochs, lr=args.lr,
        ),
        "GIN": lambda: embed_gin(
            G, nodes, features=features,
            epochs=args.epochs, lr=args.lr,
        ),
    }

    computed_coords: dict[str, np.ndarray] = {}

    for method_name, embed_fn in methods.items():
        print(f"\nComputing {method_name}...")
        random.seed(SEED)
        np.random.seed(SEED)
        try:
            coords = embed_fn()

            # Collapse detection via pairwise distance statistics
            from scipy.spatial.distance import pdist
            dists = pdist(coords)
            if len(dists) > 0:
                median_dist = float(np.median(dists))
                cv = float(np.std(dists) / (np.mean(dists) + 1e-12))
                if cv < 0.05:
                    print(f"  WARNING: {method_name} embedding appears collapsed "
                          f"(median dist = {median_dist:.6f}, CV = {cv:.4f})")

            save_embedding(coords, nodes, method_name, subset, emb_dir)
            computed_coords[method_name] = coords
            print(f"  {method_name}: std={np.std(coords):.4f}, "
                  f"shape={coords.shape}")
        except Exception as e:
            print(f"  {method_name} FAILED: {e}")

    # ---- Post-embedding collapse detection ----
    print("\n--- Embedding Collapse Check ---")
    for method_name, coords in computed_coords.items():
        collapse_info = check_embedding_collapse(coords, method_name)
        if collapse_info["collapsed"]:
            print(f"  ALERT: {method_name} embedding COLLAPSED!")
            for reason in collapse_info["reasons"]:
                print(f"    - {reason}")
        else:
            print(f"  {method_name}: OK "
                  f"(median_dist={collapse_info['median_dist']:.4f}, "
                  f"CV={collapse_info['cv']:.4f})")

    # ---- Print results table ----
    print("\n" + "=" * 60)
    print(f"{'Method':<15} {'Shape':<15} {'Std':<10}")
    print("-" * 60)
    for method_name, coords in computed_coords.items():
        print(f"{method_name:<15} {str(coords.shape):<15} "
              f"{np.std(coords):.4f}")
    print("=" * 60)

    if not computed_coords:
        print("\nNo embeddings were computed successfully.")
        return

    # ============================================================
    # G-F Metrics Evaluation
    # ============================================================
    print("\n" + "=" * 60)
    print("Computing G-F metrics for GNN methods...")
    print("=" * 60)

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # 1. G-F Curves and Scores
    print("\n--- G-F Curves & Scores ---")
    all_purities, gf_scores, plateau_widths = compute_gf_curves_and_scores(
        computed_coords, nodes, go_map, r_vals,
    )

    # 2. Link Prediction AUROC (only for curated 153-node network)
    link_pred_results: dict[str, dict] = {}
    knn_results: dict[str, dict] = {}

    if subset == "153":
        print("\n--- Link Prediction (5-fold CV) ---")
        link_pred_results = evaluate_link_prediction(
            computed_coords, nodes, G,
        )

        # 3. k-NN Downstream Task
        print("\n--- k-NN GO Term Prediction (5-fold CV) ---")
        knn_results = evaluate_downstream_knn(
            computed_coords, nodes, go_map,
        )

    # 4. Spearman correlation (AUROC vs G-F Score)
    rho = 0.0
    if link_pred_results and len(link_pred_results) >= 2:
        from scipy import stats as sp_stats

        auroc_list = []
        gf_list = []
        for m in link_pred_results:
            if m in gf_scores:
                auroc_list.append(link_pred_results[m]["auroc_mean"])
                gf_list.append(gf_scores[m])
        if len(auroc_list) >= 3:
            rho, _ = sp_stats.spearmanr(auroc_list, gf_list)
            rho = float(rho)
    print(f"\nSpearman rho (AUROC vs G-F Score): {rho:.4f}")

    # ---- Assemble and save results ----
    results = {
        "methods": list(computed_coords.keys()),
        "gf_scores": {m: round(s, 4) for m, s in gf_scores.items()},
        "plateau_widths": plateau_widths,
        "link_prediction": link_pred_results,
        "downstream_knn": knn_results,
        "spearman_rho_auroc_gf": round(rho, 4),
        "unified_interval": [GF_R_MIN, GF_R_MAX],
    }

    output_file = results_dir / "gnn_gf_scores.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved GNN G-F scores to: {output_file}")

    # ---- Print final summary ----
    print("\n=== G-F Score Ranking (GNN Methods) ===")
    ranked = sorted(gf_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (method, score) in enumerate(ranked, 1):
        print(f"  {i}. {method}: {score:.4f}")

    if link_pred_results:
        print("\n=== Link Prediction AUROC ===")
        ranked_lp = sorted(
            link_pred_results.items(),
            key=lambda x: x[1]["auroc_mean"],
            reverse=True,
        )
        for i, (method, data) in enumerate(ranked_lp, 1):
            print(f"  {i}. {method}: "
                  f"{data['auroc_mean']:.4f} +/- {data['auroc_std']:.4f}")

    if knn_results:
        print("\n=== k-NN micro-F1 ===")
        ranked_knn = sorted(
            knn_results.items(),
            key=lambda x: x[1]["micro_f1_mean"],
            reverse=True,
        )
        for i, (method, data) in enumerate(ranked_knn, 1):
            print(f"  {i}. {method}: "
                  f"{data['micro_f1_mean']:.3f} +/- "
                  f"{data['micro_f1_std']:.3f}")

    print("\nAll GNN embeddings and evaluations complete!")



def embed_gnn_method_by_name(
    G, nodes, method_name, features=None, latent_dim=2, hidden_dim=16,
    epochs=300, lr=0.01, seed=SEED
):
    """Factory to compute GNN embeddings at arbitrary dimensionality.

    Parameters
    ----------
    G : nx.Graph
    nodes : list
    method_name : str
        One of GraphSAGE, GAT, GIN.
    features : np.ndarray, optional
        Node features; if None, one-hot identity is used.
    latent_dim : int, default 2
        Output embedding dimensionality.
    hidden_dim : int, default 16
        Hidden layer dimensionality.
    epochs : int, default 300
    lr : float, default 0.01
    seed : int

    Returns
    -------
    np.ndarray
        Embedding coordinates (n_nodes, latent_dim).
    """
    method_name = method_name.strip()
    if method_name == "GraphSAGE":
        return graphsage_from_graph(
            G, hidden_dim=hidden_dim, latent_dim=latent_dim,
            epochs=epochs, lr=lr, features=features, seed=seed
        )
    elif method_name == "GAT":
        return gat_from_graph(
            G, hidden_dim=hidden_dim, latent_dim=latent_dim,
            epochs=epochs, lr=lr, features=features, seed=seed
        )
    elif method_name == "GIN":
        return gin_from_graph(
            G, hidden_dim=hidden_dim, latent_dim=latent_dim,
            epochs=epochs, lr=lr, features=features, seed=seed
        )
    else:
        raise ValueError(f"Unknown GNN method: {method_name}")


if __name__ == "__main__":
    main()
