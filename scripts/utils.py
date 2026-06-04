#!/usr/bin/env python3
# ============================================================
# G-F Consistency Framework - Utility Functions
# ============================================================
# Shared utilities for data loading, embedding, G-F curve
# computation, and I/O.  All scripts in ``scripts/`` and
# ``human_validation/`` import from this module.
# ============================================================

import json
import gzip
import random
import logging
import numpy as np
import networkx as nx
from collections import Counter
from pathlib import Path

SEED = 42

def get_project_root():
    return Path(__file__).resolve().parent.parent

def get_data_dir():
    return get_project_root() / "data"

def get_results_dir():
    return get_project_root() / "results"

def get_figures_dir():
    return get_project_root() / "figures"

def get_embeddings_dir():
    return get_project_root() / "embeddings"

# ---- Data Loading ----

def load_curated_network(data_dir=None):
    if data_dir is None:
        data_dir = get_data_dir()
    data_dir = Path(data_dir)

    G = nx.Graph()
    edgelist_file = data_dir / "curated_153_ppi.edgelist"
    if not edgelist_file.exists():
        edgelist_file = data_dir / "yeast_ppi_final_clean.edgelist"
    with open(edgelist_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])

    with open(data_dir / "gene_go_map.json") as f:
        go_map = json.load(f)

    valid = sorted(set(G.nodes()) & set(go_map.keys()))
    G = G.subgraph(valid).copy()
    nodes = sorted(G.nodes())
    return G, nodes, go_map


def load_full_STRING_network(data_dir=None, min_score=700):
    if data_dir is None:
        data_dir = get_data_dir()
    data_dir = Path(data_dir)
    string_file = data_dir / "4932.protein.links.v11.5.txt.gz"
    if not string_file.exists():
        raise FileNotFoundError(f"STRING file not found: {string_file}")

    G = nx.Graph()
    with gzip.open(str(string_file), 'rt', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p1, p2, score = parts
            if int(score) >= min_score:
                p1_clean = p1.split('.')[1]
                p2_clean = p2.split('.')[1]
                G.add_edge(p1_clean, p2_clean)

    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def load_embedding(method, subset="153", embeddings_dir=None):
    if embeddings_dir is None:
        embeddings_dir = get_embeddings_dir()
    embeddings_dir = Path(embeddings_dir)

    npy_file = embeddings_dir / f"{method}_{subset}.npy"
    nodes_file = embeddings_dir / f"{method}_{subset}_nodes.json"

    if npy_file.exists():
        coords = np.load(npy_file)
        if nodes_file.exists():
            with open(nodes_file) as f:
                nodes = json.load(f)
        else:
            nodes = [str(i) for i in range(len(coords))]
        return coords, nodes

    # Fallback: JSON format in data dir
    data_dir = get_data_dir()
    json_file = data_dir / f"embeddings_{method.lower()}.json"
    if json_file.exists():
        with open(json_file) as f:
            pos = json.load(f)
        nodes = sorted(pos.keys())
        coords = np.array([pos[n] for n in nodes])
        return coords, nodes

    raise FileNotFoundError(f"No embedding found for {method}_{subset}")


def save_embedding(coords, nodes, method, subset="153", embeddings_dir=None):
    if embeddings_dir is None:
        embeddings_dir = get_embeddings_dir()
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / f"{method}_{subset}.npy", coords)
    with open(embeddings_dir / f"{method}_{subset}_nodes.json", "w") as f:
        json.dump(nodes, f)

# ---- Spatial Graph Construction ----

def precompute_distance_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def build_spatial_graph_fast(dist_matrix, r):
    n = dist_matrix.shape[0]
    G_r = nx.Graph()
    G_r.add_nodes_from(range(n))
    mask = (dist_matrix < r) & (dist_matrix > 0)
    rows, cols = np.where(mask)
    for i, j in zip(rows, cols):
        if i < j:
            G_r.add_edge(i, j)
    return G_r

# ---- Functional Purity ----

def functional_purity(communities, go_map, nodes):
    purities = []
    for comm in communities:
        if not comm:
            continue
        comm_list = list(comm)
        go_terms = []
        for idx in comm_list:
            node_name = nodes[idx]
            if node_name in go_map:
                go_terms.extend(go_map[node_name])
        if not go_terms:
            purities.append(0.0)
            continue
        term_counts = Counter(go_terms)
        most_common_count = term_counts.most_common(1)[0][1]
        purities.append(most_common_count / len(comm_list))
    return float(np.mean(purities)) if purities else 0.0


def functional_purity_named(communities, go_map):
    purities = []
    for comm in communities:
        if not comm:
            continue
        comm_list = list(comm)
        go_terms = []
        for node_name in comm_list:
            if node_name in go_map:
                go_terms.extend(go_map[node_name])
        if not go_terms:
            purities.append(0.0)
            continue
        term_counts = Counter(go_terms)
        most_common_count = term_counts.most_common(1)[0][1]
        purities.append(most_common_count / len(comm_list))
    return float(np.mean(purities)) if purities else 0.0

# ---- G-F Curve Computation ----

def compute_gf_curve(coords, nodes, go_map, r_vals):
    from networkx.algorithms.community import greedy_modularity_communities, modularity
    dist_matrix = precompute_distance_matrix(coords)
    purities = []
    modularities = []
    for r in r_vals:
        G_r = build_spatial_graph_fast(dist_matrix, r)
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
    return purities, modularities

# ---- G-F Score ----

def compute_gf_score(r_vals, purity_vals, r_min=0.05, r_max=0.422):
    from scipy.integrate import trapezoid
    r = np.asarray(r_vals)
    p = np.asarray(purity_vals)
    mask = (r >= r_min) & (r <= r_max)
    r_sub = r[mask]
    p_sub = p[mask]
    if len(r_sub) < 2:
        return 0.0
    return trapezoid(p_sub, r_sub) / (r_max - r_min)

# ---- Centrality Features ----

def compute_centrality_features(G, nodes=None):
    if nodes is None:
        nodes = list(G.nodes())
    n = len(nodes)
    deg = nx.degree_centrality(G)
    eig = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-5)
    pr = nx.pagerank(G)
    clust = nx.clustering(G)
    avg_deg = nx.average_neighbor_degree(G)
    kcore = nx.core_number(G)

    features = np.zeros((n, 6))
    for i, u in enumerate(nodes):
        features[i, 0] = deg[u]
        features[i, 1] = eig[u]
        features[i, 2] = pr[u]
        features[i, 3] = clust[u]
        features[i, 4] = avg_deg[u]
        features[i, 5] = kcore[u]

    norms = np.linalg.norm(features, axis=0)
    norms[norms < 1e-10] = 1.0
    features = features / norms
    return features

# ---- Coordinate Rescaling ----

def rescale_coordinates(coords, target_std=0.3):
    current_std = np.std(coords)
    if current_std < 1e-10:
        return coords
    return coords / current_std * target_std

def coords_to_dict(coords, nodes):
    return {nodes[i]: coords[i].tolist() for i in range(len(nodes))}

# ---- Plateau Width ----

def compute_plateau_width(r_vals, purity_vals, threshold=0.5):
    r = np.asarray(r_vals)
    p = np.asarray(purity_vals)
    mask = p >= threshold
    if not mask.any():
        return 0.0
    r_plateau = r[mask]
    return float(r_plateau[-1] - r_plateau[0])


# ---- Logging ----

def setup_logging(name, level=logging.INFO):
    """Return a configured logger with consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---- Reusable Embedding Functions ----
# These wrap the per-method logic so that ``embed_all.py``,
# ``full_network.py``, ``robustness.py`` and ``human_embed_all.py``
# can share a single implementation.

def build_similarity_matrix(features):
    """Inner-product similarity from normalised feature matrix."""
    return features @ features.T


def diffusion_map_from_similarity(sim):
    """Diffusion Map coordinates (2-D) from a similarity matrix."""
    row_sums = sim.sum(axis=1, keepdims=True)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(row_sums.flatten()) + 1e-10))
    norm_sim = D_inv_sqrt @ sim @ D_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(norm_sim)
    idx = np.argsort(eigvals)
    coords = np.column_stack([eigvecs[:, idx[-2]], eigvecs[:, idx[-3]]])
    return coords


def classical_mds_from_distances(D):
    """Classical MDS (2-D) from a square distance matrix."""
    n = D.shape[0]
    D_sq = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D_sq @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    coords = eigvecs[:, idx[:2]] * np.sqrt(np.abs(eigvals[idx[:2]]))
    return coords


def spectral_embedding_from_graph(G, nodelist=None):
    """Spectral embedding (2-D) from normalised Laplacian."""
    L = nx.normalized_laplacian_matrix(G, nodelist=nodelist).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvecs[:, 1:3]


def deepwalk_from_graph(G, walk_length=20, walks_per_node=10,
                         window_size=5, dimensions=2, seed=SEED):
    """DeepWalk embedding via uniform random walks + SVD."""
    random.seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    adj = {i: [] for i in range(n)}
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i].append(j)
        adj[j].append(i)

    walks = []
    for start in range(n):
        for _ in range(walks_per_node):
            walk = [start]
            for _ in range(walk_length - 1):
                cur = walk[-1]
                if not adj[cur]:
                    break
                walk.append(np.random.choice(adj[cur]))
            if len(walk) >= 2:
                walks.append(walk)

    cooc = np.zeros((n, n))
    for walk in walks:
        for i, ni in enumerate(walk):
            for j in range(max(0, i - window_size),
                           min(len(walk), i + window_size + 1)):
                if i != j:
                    cooc[ni, walk[j]] += 1

    U, S, _ = np.linalg.svd(cooc, full_matrices=False)
    return U[:, :dimensions] * np.sqrt(S[:dimensions])


def node2vec_from_graph(G, walk_length=20, walks_per_node=10,
                        window_size=5, dimensions=2, p=0.5, q=2.0,
                        seed=SEED):
    """Node2Vec embedding via biased random walks + SVD."""
    random.seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    adj = {i: [] for i in range(n)}
    adj_sets = {i: set() for i in range(n)}
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i].append(j)
        adj[j].append(i)
        adj_sets[i].add(j)
        adj_sets[j].add(i)

    walks = []
    for start in range(n):
        for _ in range(walks_per_node):
            walk = [start]
            for _ in range(walk_length - 1):
                cur = walk[-1]
                if not adj[cur]:
                    break
                if len(walk) == 1:
                    walk.append(np.random.choice(adj[cur]))
                else:
                    prev = walk[-2]
                    probs = []
                    for nbr in adj[cur]:
                        if nbr == prev:
                            probs.append(1.0 / p)
                        elif nbr in adj_sets[prev]:
                            probs.append(1.0)
                        else:
                            probs.append(1.0 / q)
                    probs = np.array(probs)
                    probs /= probs.sum()
                    walk.append(np.random.choice(adj[cur], p=probs))
            if len(walk) >= 2:
                walks.append(walk)

    # Use dict for large graphs to save memory
    if n > 1000:
        cooc = {}
        for walk in walks:
            for i, ni in enumerate(walk):
                for j in range(max(0, i - window_size),
                               min(len(walk), i + window_size + 1)):
                    if i != j:
                        key = (ni, walk[j])
                        cooc[key] = cooc.get(key, 0) + 1
        cooc_matrix = np.zeros((n, n))
        for (i, j), val in cooc.items():
            cooc_matrix[i, j] = val
    else:
        cooc_matrix = np.zeros((n, n))
        for walk in walks:
            for i, ni in enumerate(walk):
                for j in range(max(0, i - window_size),
                               min(len(walk), i + window_size + 1)):
                    if i != j:
                        cooc_matrix[ni, walk[j]] += 1

    U, S, _ = np.linalg.svd(cooc_matrix, full_matrices=False)
    return U[:, :dimensions] * np.sqrt(S[:dimensions])


def vgae_from_graph(G, hidden_dim=4, latent_dim=2, epochs=300,
                    lr=0.01, features=None, seed=SEED):
    """VGAE embedding (2-D latent) with 2-layer GCN encoder."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
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

    class Encoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, latent_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv_mu = GCNConv(hidden_dim, latent_dim)
            self.conv_logvar = GCNConv(hidden_dim, latent_dim)

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            return self.conv_mu(h, edge_index), self.conv_logvar(h, edge_index)

    model = Encoder(in_dim, hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Precompute adjacency target once (avoids redundant work per epoch)
    adj_target = torch.zeros(n, n)
    ei = data.edge_index
    adj_target[ei[0], ei[1]] = 1.0

    for epoch in range(epochs):
        optimizer.zero_grad()
        mu, logvar = model(data.x, data.edge_index)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        adj_recon = torch.sigmoid(z @ z.T)
        recon_loss = F.binary_cross_entropy(adj_recon, adj_target, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        mu, _ = model(data.x, data.edge_index)
        coords = mu.numpy()
    return coords


def standardize_coordinates(coords, target_std=0.3):
    """Alias for :func:`rescale_coordinates`."""
    return rescale_coordinates(coords, target_std=target_std)
