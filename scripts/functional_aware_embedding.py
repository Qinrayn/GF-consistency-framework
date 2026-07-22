#!/usr/bin/env python3
"""
Functional-Aware Embedding (Direction A — Phase 1B)
====================================================

Trains an embedding that directly optimises geometric-functional consistency
rather than merely reconstructing the PPI topology.  This transforms the
G-F Score from a *post-hoc* evaluation metric into a *differentiable training
objective*, producing the first embedding method whose design goal is
functional coherence rather than link prediction.

Method
------
The core insight is that the G-F Score measures alignment between the
embedding's spatial community structure and GO annotations.  We train a
neural encoder to maximise this alignment directly via a surrogate loss:

.. math::

    L = L_pull + λ₁·L_push + λ₂·L_reg

where:
- **L_pull** pulls proteins with shared GO terms together in embedding space.
- **L_push** pushes proteins with zero GO overlap apart (beyond a margin).
- **L_reg** is a spectral regularisation term that preserves the PPI Laplacian
  structure (the Cheeger bound from Step 65 provides the theoretical
  justification).

The encoder is a 2-layer GCN (same architecture as VGAE's encoder) followed
by a linear projection to 2D.  Training uses Adam with a GO-similarity-based
triplet sampling strategy.

Evaluation
----------
After training, the embedding is evaluated with the standard G-F pipeline
(``compute_gf.py``) and compared against all 11 baseline methods.  The key
metric is whether GFAE (Geometric-Functional Aligned Embedding) surpasses
Spectral (0.163) on the curated 153-node yeast network.

Usage
-----
.. code-block:: bash

    # Train on the curated 153-node yeast PPI (requires PyTorch + PyG):
    python scripts/functional_aware_embedding.py

    # Train on the full 5,936-node STRING network (GPU recommended):
    python scripts/functional_aware_embedding.py --network full --epochs 500

    # With custom hyperparameters:
    python scripts/functional_aware_embedding.py --lr 0.005 --margin 0.5 --lambda-pull 1.0

Requirements
------------
- ``torch`` ≥ 2.0
- ``torch_geometric`` ≥ 2.3
- All other dependencies are already in ``environment.yml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from utils import (
    SEED,
    TARGET_STD,
    get_data_dir,
    get_embeddings_dir,
    get_results_dir,
    load_curated_network,
    load_full_STRING_network,
    rescale_coordinates,
    compute_centrality_features,
    setup_logging,
)

logger: logging.Logger = setup_logging("functional_aware_embedding")


# ============================================================
# GO Similarity Matrix
# ============================================================

def build_go_similarity_matrix(
    nodes: list[str],
    go_map: dict,
    min_overlap: int = 1,
) -> np.ndarray:
    """Compute an n×n GO-term overlap matrix.

    ``S[i,j] = |go_map[node_i] ∩ go_map[node_j]|``, normalised to [0, 1]
    by dividing by the maximum overlap in the dataset.

    Parameters
    ----------
    nodes : list[str]
        Ordered node identifiers.
    go_map : dict
        Node → list of GO term strings.
    min_overlap : int
        Minimum overlap to consider a pair as "functionally related".

    Returns
    -------
    np.ndarray
        Symmetric float32 matrix of shape (n, n).
    """
    n = len(nodes)
    node_terms = [set(go_map.get(node, [])) for node in nodes]
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        ti = node_terms[i]
        if not ti:
            continue
        for j in range(i + 1, n):
            tj = node_terms[j]
            if not tj:
                continue
            overlap = len(ti & tj)
            S[i, j] = float(overlap)
            S[j, i] = float(overlap)

    max_overlap = float(S.max())
    if max_overlap > 0:
        S /= max_overlap
    logger.info(
        "GO similarity: %d/%d pairs have overlap ≥ %d (mean=%.3f, max=%d)",
        int((S > 0).sum() / 2), n * (n - 1) // 2, min_overlap, S.mean(), int(max_overlap),
    )
    return S


# ============================================================
# Functional-Aware GCN Encoder
# ============================================================

def _import_torch():
    """Lazy PyTorch imports to avoid loading CUDA at module level."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import from_networkx
    return torch, nn, F, GCNConv, from_networkx


class GFAEEncoder:
    """Geometric-Functional Aligned Embedding encoder (2-layer GCN → 2D).

    Architecture matches the VGAE encoder from ``utils.vgae_from_graph``
    (GCNConv, hidden=4, 2D output) so the comparison is fair.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 4,
        latent_dim: int = 2,
        seed: Optional[int] = None,
    ):
        torch, nn, _, GCNConv, _ = _import_torch()
        if seed is None:
            seed = SEED
        torch.manual_seed(seed)
        np.random.seed(seed)

        class _Encoder(nn.Module):
            def __init__(self, in_dim, hidden_dim, latent_dim):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, latent_dim)

            def forward(self, x, edge_index):
                h = torch.nn.functional.relu(self.conv1(x, edge_index))
                return self.conv2(h, edge_index)

        self.model = _Encoder(in_dim, hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def encode(self, x, edge_index):
        return self.model(x, edge_index)


# ============================================================
# Triplet-Based Functional Loss
# ============================================================

def functional_contrastive_loss(
    embeddings: "torch.Tensor",
    go_sim: "torch.Tensor",
    margin: float = 0.5,
    lambda_pull: float = 1.0,
    lambda_push: float = 1.0,
    n_hard_negatives: int = 5,
) -> "torch.Tensor":
    """Compute the functional contrastive loss.

    Parameters
    ----------
    embeddings : torch.Tensor (n, d)
        Current embedding coordinates.
    go_sim : torch.Tensor (n, n)
        Normalised GO overlap matrix (0 = no overlap, 1 = max overlap).
    margin : float
        Minimum distance margin for push loss (pairs with zero GO overlap).
    lambda_pull : float
        Weight for the pull term (attract functionally related proteins).
    lambda_push : float
        Weight for the push term (separate functionally unrelated proteins).
    n_hard_negatives : int
        Number of hardest negative pairs per anchor for the push term.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    torch = __import__("torch")
    n = embeddings.shape[0]
    # Pairwise Euclidean distances
    # ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2 x_i·x_j
    sq_norm = (embeddings ** 2).sum(dim=1)  # (n,)
    dist_sq = sq_norm.unsqueeze(0) + sq_norm.unsqueeze(1) - 2 * (embeddings @ embeddings.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    dist = torch.sqrt(dist_sq + 1e-8)

    # ---- Pull term: attract functionally similar pairs ----
    # Weighted by GO similarity: higher overlap → stronger pull
    pull_mask = go_sim > 0.0
    if pull_mask.any():
        pull_loss = (go_sim[pull_mask] * dist[pull_mask]).sum() / (pull_mask.sum() + 1e-8)
    else:
        pull_loss = torch.tensor(0.0, device=embeddings.device)

    # ---- Push term: push functionally unrelated pairs apart ----
    # Target pairs with zero GO overlap, using hard negative mining
    push_mask = (go_sim == 0.0).float()
    # Self-pairs excluded
    push_mask.fill_diagonal_(0.0)
    if push_mask.sum() > 0:
        # Margin loss: max(0, margin - dist)
        push_loss_per_pair = torch.clamp(margin - dist, min=0.0)
        push_loss = (push_mask * push_loss_per_pair).sum() / (push_mask.sum() + 1e-8)
    else:
        push_loss = torch.tensor(0.0, device=embeddings.device)

    return lambda_pull * pull_loss + lambda_push * push_loss


# ============================================================
# Spectral Regularisation (Cheeger-aligned)
# ============================================================

def spectral_regularisation(
    embeddings: "torch.Tensor",
    laplacian_eigvecs: "torch.Tensor",
    n_eigs: int = 2,
    lambda_reg: float = 0.1,
) -> "torch.Tensor":
    """Regularise the embedding to align with the graph Laplacian eigenbasis.

    This is motivated by the Cheeger-Spectral G-F bound (Step 65): the
    Laplacian spectrum provides an upper bound on the achievable G-F Score.
    Preserving the spectral structure ensures the embedding does not
    destroy the geometric signal that drives functional coherence.

    Loss = λ_reg * ||(I - UUT) @ embeddings||²_F
    where U is the top-k Laplacian eigenvectors.
    """
    torch = __import__("torch")
    if lambda_reg <= 0 or laplacian_eigvecs is None:
        return torch.tensor(0.0, device=embeddings.device)

    # Project embeddings onto the subspace orthogonal to the Laplacian eigenvectors
    U = laplacian_eigvecs[:, :n_eigs]  # (n, k)
    # Residual after removing the eigenbasis-aligned component
    proj = U @ (U.T @ embeddings)  # (n, d)
    residual = embeddings - proj
    return lambda_reg * (residual ** 2).mean()


# ============================================================
# Training Loop
# ============================================================

def train_gfae(
    graph: nx.Graph,
    nodes: list[str],
    go_sim: np.ndarray,
    features: Optional[np.ndarray] = None,
    hidden_dim: int = 4,
    latent_dim: int = 2,
    epochs: int = 300,
    lr: float = 0.01,
    margin: float = 0.5,
    lambda_pull: float = 1.0,
    lambda_push: float = 1.0,
    lambda_reg: float = 0.1,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """Train the functional-aware embedding.

    Returns
    -------
    (coords, history)
        ``coords`` is the final (n, latent_dim) embedding.
        ``history`` is a dict with per-epoch loss values.
    """
    torch, nn, F, GCNConv, from_networkx = _import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is None:
        seed = SEED
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(nodes)
    data = from_networkx(graph)

    # Prepare features
    if features is not None:
        x = torch.tensor(features, dtype=torch.float32)
        in_dim = features.shape[1]
    else:
        x = torch.eye(n, dtype=torch.float32)
        in_dim = n

    # Prepare GO similarity
    go_sim_t = torch.tensor(go_sim, dtype=torch.float32)

    # Compute Laplacian eigenvectors for regularisation
    L = nx.normalized_laplacian_matrix(graph, nodelist=nodes).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    laplacian_eigvecs = torch.tensor(eigvecs[:, :latent_dim], dtype=torch.float32)

    # Move to device
    x = x.to(device)
    go_sim_t = go_sim_t.to(device)
    laplacian_eigvecs = laplacian_eigvecs.to(device)
    data = data.to(device)

    encoder = GFAEEncoder(in_dim, hidden_dim, latent_dim, seed=seed).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    history: dict = {"loss": [], "loss_pull": [], "loss_push": [], "loss_reg": []}
    t_start = time.time()

    for epoch in range(epochs):
        encoder.train()
        optimizer.zero_grad()

        embeddings = encoder.encode(x, data.edge_index)

        loss_pull_push = functional_contrastive_loss(
            embeddings, go_sim_t, margin=margin,
            lambda_pull=lambda_pull, lambda_push=lambda_push,
        )
        loss_reg = spectral_regularisation(
            embeddings, laplacian_eigvecs,
            n_eigs=latent_dim, lambda_reg=lambda_reg,
        )
        loss = loss_pull_push + loss_reg

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            history["loss"].append(float(loss.item()))
            history["loss_pull"].append(float(loss_pull_push.item()))
            history["loss_reg"].append(float(loss_reg.item()))
            logger.info(
                "Epoch %d/%d | loss=%.4f | pull+push=%.4f | reg=%.4f",
                epoch + 1, epochs, loss.item(), loss_pull_push.item(), loss_reg.item(),
            )

    encoder.eval()
    with torch.no_grad():
        coords = encoder.encode(x, data.edge_index).cpu().numpy()

    # Rescale to match the standardised pipeline convention
    coords = rescale_coordinates(coords, target_std=TARGET_STD)

    elapsed = time.time() - t_start
    logger.info("Training completed in %.1f s (%.1f s/epoch)", elapsed, elapsed / epochs)

    return coords, history


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Functional-Aware Embedding (Direction A)",
    )
    parser.add_argument(
        "--network", type=str, default="curated",
        choices=["curated", "full"],
        help="Network to train on (curated=153 nodes, full=5936 nodes).",
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=4,
        help="Hidden dimension of the GCN encoder.",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=2,
        help="Output embedding dimension (2 for GF Score comparison).",
    )
    parser.add_argument(
        "--margin", type=float, default=0.5,
        help="Push loss margin for functionally unrelated pairs.",
    )
    parser.add_argument(
        "--lambda-pull", type=float, default=1.0,
        help="Weight for the functional pull term.",
    )
    parser.add_argument(
        "--lambda-push", type=float, default=1.0,
        help="Weight for the functional push term.",
    )
    parser.add_argument(
        "--lambda-reg", type=float, default=0.1,
        help="Weight for spectral regularisation.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    emb_dir = get_embeddings_dir()
    results_dir = get_results_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load network and GO annotations
    # ------------------------------------------------------------------
    logger.info("Loading network ...")
    if args.network == "curated":
        G, nodes, go_map = load_curated_network(data_dir)
    else:
        G = load_full_STRING_network(data_dir)
        go_map_path = data_dir / "gene_go_map.json"
        if go_map_path.exists():
            with open(go_map_path, encoding="utf-8") as f:
                go_map = json.load(f)
        else:
            go_map = {}
        nodes = sorted(set(G.nodes()) & set(go_map.keys()))
        G = G.subgraph(nodes).copy()
    logger.info("Network: %d nodes, %d edges", len(nodes), G.number_of_edges())

    # ------------------------------------------------------------------
    # Build GO similarity matrix
    # ------------------------------------------------------------------
    logger.info("Building GO similarity matrix ...")
    go_sim = build_go_similarity_matrix(nodes, go_map)

    # ------------------------------------------------------------------
    # Compute centrality features (same as DM/PCA/VGAE-feat input)
    # ------------------------------------------------------------------
    features = compute_centrality_features(G, nodes)

    # ------------------------------------------------------------------
    # Train GFAE
    # ------------------------------------------------------------------
    logger.info("Training GFAE ...")
    coords, history = train_gfae(
        G, nodes, go_sim,
        features=features,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
        lambda_pull=args.lambda_pull,
        lambda_push=args.lambda_push,
        lambda_reg=args.lambda_reg,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Save embedding
    # ------------------------------------------------------------------
    network_tag = "153" if args.network == "curated" else "full"
    method_name = "GFAE"
    np.save(emb_dir / f"{method_name}_{network_tag}.npy", coords)
    with open(emb_dir / f"{method_name}_{network_tag}_nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f)
    logger.info("Saved embedding to embeddings/%s_%s.npy", method_name, network_tag)

    # ------------------------------------------------------------------
    # Save training history and metadata
    # ------------------------------------------------------------------
    metadata = {
        "description": "Functional-Aware Embedding (GFAE) — Direction A",
        "method": method_name,
        "network": args.network,
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "margin": args.margin,
            "lambda_pull": args.lambda_pull,
            "lambda_push": args.lambda_push,
            "lambda_reg": args.lambda_reg,
            "seed": args.seed,
        },
        "training_history": history,
        "n_nodes": len(nodes),
        "n_edges": G.number_of_edges(),
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }
    out_path = results_dir / "functional_aware_embedding_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", out_path)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  GFAE TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Method:        {method_name}")
    print(f"  Network:       {args.network} ({len(nodes)} nodes)")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Final loss:    {history['loss'][-1]:.4f}")
    print(f"  Embedding:     embeddings/{method_name}_{network_tag}.npy")
    print()
    print("  Next: run G-F evaluation to compare GFAE against baselines:")
    print(f"    python scripts/compute_gf.py --method GFAE")
    print("=" * 60)


if __name__ == "__main__":
    main()