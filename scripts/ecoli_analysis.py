#!/usr/bin/env python3
"""
Cross-Species G-F Score Analysis: E. coli (4th Species)
========================================================
Computes G-F Scores for all 11 embedding methods on the E. coli K-12
STRING PPI network, then updates the cross-species Kendall's W from
3 species (yeast, human, mouse) to 4 species.

Pipeline (mirrors human/mouse exactly):
  1. Load STRING v11.5 PPI (combined_score >= 700), largest CC
  2. Map GO annotations via STRING protein aliases
  3. Compute 11 embeddings (scalable approximations for large network):
     - Landmark MDS (500 landmarks + Nystrom)
     - Sparse eigendecomposition for DM, Spectral
     - Sparse negative-sampling for VGAE, GNN methods
  4. Subsample to common annotated nodes
  5. Compute GF Scores using greedy_modularity_communities
  6. Report method ranking + update Kendall's W across 4 species

Output:
  results/ecoli_gf_scores.json
"""

import json
import sys
import time
import gzip
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import networkx as nx
from scipy.integrate import trapezoid
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    ALL_METHODS, SEED, TARGET_STD,
    GF_R_MIN, GF_R_MAX, R_MIN, R_MAX,
    get_data_dir, get_results_dir,
    compute_centrality_features,
    rescale_coordinates,
    deepwalk_from_graph,
    node2vec_from_graph,
)

from human_embed_extended import (
    _train_gnn_sparse,
    _build_sage_encoder,
    _build_gat_encoder,
    _build_gin_encoder,
)

DATA_DIR = get_data_dir()
RESULTS_DIR = get_results_dir()

SCORE_THRESHOLD = 700
OUTLIER_THRESHOLD = 100
LANDMARK_COUNT = 500
SUBSAMPLE_SIZE = 2000
N_POINTS = 25           # Coarse grid for GF curve (same as human_gf_unified)
MAX_EDGES_GF = 150_000  # Fallback to CC above this
BANNER = "=" * 70


# ============================================================
# 1. Network Loading
# ============================================================

def load_ecoli_network():
    """Load E. coli STRING network, score >= 700, largest CC."""
    string_file = DATA_DIR / "511145.protein.links.v11.5.txt.gz"
    G = nx.Graph()
    with gzip.open(str(string_file), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and int(parts[2]) >= SCORE_THRESHOLD:
                p1 = parts[0].split(".", 1)[-1]
                p2 = parts[1].split(".", 1)[-1]
                G.add_edge(p1, p2)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    return G


# ============================================================
# 2. GO Annotation Loading + ID Mapping via STRING Aliases
# ============================================================

def build_alias_mapping():
    """Build STRING protein ID -> set of aliases (lowercase) mapping."""
    aliases_file = DATA_DIR / "511145.protein.aliases.v11.5.txt.gz"
    string_to_aliases = {}
    with gzip.open(str(aliases_file), "rt", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                string_id = parts[0].split(".", 1)[-1]
                alias = parts[1].lower()
                if string_id not in string_to_aliases:
                    string_to_aliases[string_id] = set()
                string_to_aliases[string_id].add(alias)
    return string_to_aliases


def load_ecoli_go_annotations():
    """Load EcoCyc GAF and return gene_name -> [GO terms] mapping."""
    gaf_file = DATA_DIR / "gene_association.ecocyc.gaf.gz"
    go_map = {}
    with gzip.open(str(gaf_file), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            gene_id = parts[2].lower()
            go_term = parts[4]
            if gene_id not in go_map:
                go_map[gene_id] = []
            if go_term not in go_map[gene_id]:
                go_map[gene_id].append(go_term)
    return go_map


def map_go_to_string(string_to_aliases, go_map):
    """Map GO annotations to STRING protein IDs via alias matching.

    Returns dict: STRING_protein_id -> [GO terms]
    """
    # Build reverse map: alias (lowercase) -> gene_name
    alias_to_gene = {}
    for gene_name in go_map:
        alias_to_gene[gene_name.lower()] = gene_name

    string_go = {}
    for string_id, aliases in string_to_aliases.items():
        go_terms = []
        for alias in aliases:
            if alias in alias_to_gene:
                gene = alias_to_gene[alias]
                go_terms.extend(go_map[gene])
        if go_terms:
            # Deduplicate
            string_go[string_id] = list(set(go_terms))

    return string_go


# ============================================================
# 3. Embedding Methods (Scalable for ~4K nodes)
# ============================================================

def detect_and_remove_outliers(coords, node_list, method_name):
    """Remove nodes with extreme coordinates (>100 std from mean)."""
    x_std = np.std(coords[:, 0])
    y_std = np.std(coords[:, 1])
    x_mean = np.mean(coords[:, 0])
    y_mean = np.mean(coords[:, 1])
    clean_mask = np.ones(len(node_list), dtype=bool)
    n_out = 0
    for i in range(len(node_list)):
        xd = abs(coords[i, 0] - x_mean) / max(x_std, 1e-10)
        yd = abs(coords[i, 1] - y_mean) / max(y_std, 1e-10)
        if xd > OUTLIER_THRESHOLD or yd > OUTLIER_THRESHOLD:
            clean_mask[i] = False
            n_out += 1
    if n_out > 0:
        print(f"    [{method_name}] Removed {n_out} outlier(s)")
    return coords[clean_mask], [n for n, m in zip(node_list, clean_mask) if m]


def landmark_mds(G, node_list, n_landmarks=500, d=2, seed=42):
    """Landmark MDS: BFS from landmarks + Nystrom extension."""
    rng = np.random.RandomState(seed)
    n = len(node_list)
    lm_indices = rng.choice(n, size=min(n_landmarks, n), replace=False)

    node_to_idx = {node: i for i, node in enumerate(node_list)}
    lm_nodes = [node_list[i] for i in lm_indices]
    n_lm = len(lm_nodes)

    print(f"    BFS from {n_lm} landmarks...", flush=True)
    t0 = time.time()
    D_lm = np.zeros((n_lm, n), dtype=np.float64)
    for li, src in enumerate(lm_nodes):
        lengths = nx.single_source_shortest_path_length(G, src)
        for node, length in lengths.items():
            j = node_to_idx.get(node)
            if j is not None:
                D_lm[li, j] = length
        if (li + 1) % 100 == 0:
            print(f"      {li+1}/{n_lm} ({time.time()-t0:.0f}s)", flush=True)

    max_finite = np.max(D_lm[D_lm > 0]) if np.any(D_lm > 0) else 1
    D_lm[D_lm == 0] = max_finite + 1
    for li, idx in enumerate(lm_indices):
        D_lm[li, idx] = 0.0

    print(f"    BFS done ({time.time()-t0:.0f}s). Computing landmark MDS...", flush=True)

    D_ll = D_lm[:, lm_indices]
    D_ll_sq = D_ll ** 2
    row_mean = D_ll_sq.mean(axis=1, keepdims=True)
    col_mean = D_ll_sq.mean(axis=0, keepdims=True)
    grand_mean = D_ll_sq.mean()
    B_ll = -0.5 * (D_ll_sq - row_mean - col_mean + grand_mean)

    eigvals, eigvecs = np.linalg.eigh(B_ll)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pos_mask = eigvals > 1e-10
    n_pos = min(np.sum(pos_mask), d)
    if n_pos < d:
        n_pos = max(n_pos, 1)

    Lambda = eigvals[:n_pos]
    V = eigvecs[:, :n_pos]
    delta = D_ll_sq.mean(axis=0)
    D_all_sq = D_lm ** 2
    delta_mat = delta[:, np.newaxis]
    diff = delta_mat - D_all_sq
    W = V / Lambda[np.newaxis, :]
    coords = (0.5 * W.T @ diff).T

    if coords.shape[1] < d:
        pad = np.zeros((n, d - coords.shape[1]))
        coords = np.hstack([coords, pad])

    return coords


def sparse_vgae(G, hidden_dim=4, latent_dim=2, epochs=300, lr=0.01,
                features=None, seed=42, neg_ratio=1):
    """VGAE with sparse negative-sampling BCE loss."""
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
        data.x = torch.eye(n, dtype=torch.float32)
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

    edge_index = data.edge_index
    n_pos = edge_index.shape[1]

    rng_np = np.random.RandomState(seed)
    edge_set = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        edge_set.add((min(u, v), max(u, v)))

    n_neg = n_pos * neg_ratio
    neg_pairs = []
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 10:
        u = rng_np.randint(0, n)
        v = rng_np.randint(0, n)
        if u != v and (min(u, v), max(u, v)) not in edge_set:
            neg_pairs.append((u, v))
        attempts += 1

    neg_pairs = np.array(neg_pairs, dtype=np.int64).reshape(-1, 2)
    if len(neg_pairs) == 0:
        neg_pairs = np.array([[0, 1]], dtype=np.int64)

    pos_src = edge_index[0]
    pos_dst = edge_index[1]
    neg_src = torch.tensor(neg_pairs[:, 0], dtype=torch.long)
    neg_dst = torch.tensor(neg_pairs[:, 1], dtype=torch.long)

    pos_labels = torch.ones(n_pos)
    neg_labels = torch.zeros(len(neg_pairs))
    all_labels = torch.cat([pos_labels, neg_labels])

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        mu, logvar = model(data.x, edge_index)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        pos_scores = torch.sigmoid(torch.sum(z[pos_src] * z[pos_dst], dim=1))
        neg_scores = torch.sigmoid(torch.sum(z[neg_src] * z[neg_dst], dim=1))
        all_scores = torch.cat([pos_scores, neg_scores])

        recon_loss = F.binary_cross_entropy(all_scores, all_labels, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"      Epoch {epoch+1}/{epochs}, loss={loss.item():.2f}", flush=True)

    model.eval()
    with torch.no_grad():
        mu, _ = model(data.x, edge_index)
    return mu.numpy()


# ============================================================
# 4. GF Curve + Score (greedy_modularity, same as human)
# ============================================================

def compute_gf_curve_greedy(coords, nodes, go_map, r_vals):
    """Compute G-F purity curve using greedy_modularity_communities."""
    from networkx.algorithms.community import greedy_modularity_communities

    D = squareform(pdist(coords))
    n = len(nodes)
    purities = np.zeros(len(r_vals))

    for ri, r in enumerate(r_vals):
        if ri % 5 == 0 and ri > 0:
            print(".", end="", flush=True)
        mask = (D < r) & (D > 0)
        n_edges = int(np.sum(mask)) // 2
        if n_edges == 0:
            continue

        rows, cols = np.where(mask)
        upper = rows < cols
        edges = list(zip(rows[upper].tolist(), cols[upper].tolist()))

        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)

        if n_edges > MAX_EDGES_GF:
            communities = [frozenset(c) for c in nx.connected_components(G)]
        else:
            try:
                communities = list(greedy_modularity_communities(G))
            except Exception:
                communities = [frozenset(c) for c in nx.connected_components(G)]

        comm_purities = []
        for comm in communities:
            all_terms = []
            for i in comm:
                node_id = nodes[i]
                terms = go_map.get(node_id, [])
                all_terms.extend(terms)
            if not all_terms:
                continue
            counts = Counter(all_terms)
            comm_purities.append(counts.most_common(1)[0][1] / len(all_terms))

        if comm_purities:
            purities[ri] = float(np.mean(comm_purities))

    return purities


def compute_gf_score(purities, r_vals, r_min, r_max):
    """Compute G-F Score over [r_min, r_max] via trapezoidal integration."""
    mask = (r_vals >= r_min) & (r_vals <= r_max)
    if mask.sum() < 2:
        return 0.0
    r_sub = r_vals[mask]
    p_sub = purities[mask]
    integral = trapezoid(p_sub, r_sub)
    return float(integral / (r_max - r_min))


# ============================================================
# 5. Cross-Species Kendall's W
# ============================================================

def kendalls_w(rankings):
    """Compute Kendall's W (coefficient of concordance).

    Parameters
    ----------
    rankings : list of lists
        Each inner list is a ranking (1 = best) for one species/condition.
    """
    k = len(rankings)
    n = len(rankings[0])
    rank_sums = np.sum(rankings, axis=0)
    mean_rank_sum = np.mean(rank_sums)
    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = (12.0 * S) / (k ** 2 * n * (n ** 2 - 1))
    return float(W)


# ============================================================
# Main Pipeline
# ============================================================

def run():
    print(BANNER)
    print("Cross-Species G-F Score Analysis: E. coli (4th Species)")
    print(BANNER)

    # ----- Step 1: Load network -----
    print("\n[1/6] Loading E. coli STRING network...")
    t0 = time.time()
    G = load_ecoli_network()
    node_list = list(G.nodes())
    n = len(node_list)
    print(f"  Network: {n} nodes, {G.number_of_edges()} edges (score >= {SCORE_THRESHOLD})")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ----- Step 2: Load GO annotations + map -----
    print("\n[2/6] Loading GO annotations and mapping to STRING IDs...")
    t0 = time.time()
    string_to_aliases = build_alias_mapping()
    go_map_raw = load_ecoli_go_annotations()
    go_map = map_go_to_string(string_to_aliases, go_map_raw)
    print(f"  GAF: {len(go_map_raw)} gene names with GO terms")
    print(f"  Mapped: {len(go_map)} STRING proteins with GO terms")

    # Intersect network nodes with GO annotations
    annotated_nodes = sorted(set(G.nodes()) & set(go_map.keys()))
    G_annotated = G.subgraph(annotated_nodes).copy()
    # Re-take largest CC after annotation filtering
    if len(annotated_nodes) < n:
        largest_cc = max(nx.connected_components(G_annotated), key=len)
        G_annotated = G_annotated.subgraph(largest_cc).copy()
        annotated_nodes = sorted(G_annotated.nodes())
    print(f"  Final network: {len(annotated_nodes)} nodes, {G_annotated.number_of_edges()} edges")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ----- Step 3: Compute embeddings -----
    print(f"\n[3/6] Computing 11 embedding methods...")
    node_list = list(G_annotated.nodes())
    n = len(node_list)

    # Centrality features
    print("  Computing centrality features...", flush=True)
    features = compute_centrality_features(G_annotated, node_list)
    print(f"  Features shape: {features.shape}")

    embeddings = {}  # method_name -> (coords, clean_nodes)

    # ---- 1. DM (sparse eigendecomposition) ----
    print("\n  [1/11] Diffusion Map (sparse eigsh)...", flush=True)
    t1 = time.time()
    try:
        from scipy.sparse.linalg import eigsh
        feat_norm = normalize(features, norm="l2", axis=0)
        sim = feat_norm @ feat_norm.T
        row_sums = sim.sum(axis=1)
        d_inv_sqrt = 1.0 / (np.sqrt(row_sums) + 1e-10)
        sim *= d_inv_sqrt[:, np.newaxis]
        sim *= d_inv_sqrt[np.newaxis, :]
        eigvals, eigvecs = eigsh(sim, k=3, which="LM")
        idx = np.argsort(eigvals)
        coords = eigvecs[:, idx[-2::-1][:2]]
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "DM")
        coords = rescale_coordinates(coords, TARGET_STD)
        embeddings["DM"] = (coords, clean_nodes)
        print(f"    {time.time()-t1:.1f}s, {len(clean_nodes)} nodes")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 2. MDS (landmark) ----
    print(f"\n  [2/11] Landmark MDS ({LANDMARK_COUNT} landmarks)...", flush=True)
    t1 = time.time()
    try:
        coords = landmark_mds(G_annotated, node_list, n_landmarks=LANDMARK_COUNT,
                              d=2, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "MDS")
        coords = rescale_coordinates(coords, TARGET_STD)
        embeddings["MDS"] = (coords, clean_nodes)
        print(f"    {time.time()-t1:.1f}s, {len(clean_nodes)} nodes")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 3. Spectral (sparse Laplacian + eigsh) ----
    print("\n  [3/11] Spectral Embedding (sparse eigsh)...", flush=True)
    t1 = time.time()
    try:
        from scipy.sparse.linalg import eigsh as sparse_eigsh
        L = nx.normalized_laplacian_matrix(G_annotated).astype(np.float64)
        eigvals, eigvecs = sparse_eigsh(L, k=3, sigma=0, which="LM")
        idx = np.argsort(eigvals)
        coords = eigvecs[:, idx[1:3]]
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "Spectral")
        coords = rescale_coordinates(coords, TARGET_STD)
        embeddings["Spectral"] = (coords, clean_nodes)
        print(f"    {time.time()-t1:.1f}s, {len(clean_nodes)} nodes")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 4. DeepWalk ----
    print("\n  [4/11] DeepWalk...", flush=True)
    t1 = time.time()
    try:
        coords = deepwalk_from_graph(G_annotated, walk_length=20, walks_per_node=10,
                                      window_size=5, dimensions=2, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "DeepWalk")
        coords = rescale_coordinates(coords, TARGET_STD)
        embeddings["DeepWalk"] = (coords, clean_nodes)
        print(f"    {time.time()-t1:.1f}s, {len(clean_nodes)} nodes")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 5. Node2Vec ----
    print("\n  [5/11] Node2Vec...", flush=True)
    t1 = time.time()
    try:
        coords = node2vec_from_graph(G_annotated, walk_length=20, walks_per_node=10,
                                      window_size=5, dimensions=2,
                                      p=0.5, q=2.0, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "Node2Vec")
        coords = rescale_coordinates(coords, TARGET_STD)
        embeddings["Node2Vec"] = (coords, clean_nodes)
        print(f"    {time.time()-t1:.1f}s, {len(clean_nodes)} nodes")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 6. VGAE (sparse, one-hot) ----
    print("\n  [6/11] VGAE (sparse neg-sampling, one-hot)...", flush=True)
    t1 = time.time()
    try:
        coords = sparse_vgae(G_annotated, hidden_dim=4, latent_dim=2, epochs=300,
                             lr=0.01, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "VGAE")
        coords = rescale_coordinates(coords, TARGET_STD)
        embeddings["VGAE"] = (coords, clean_nodes)
        print(f"    {time.time()-t1:.1f}s, {len(clean_nodes)} nodes")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 7. PCA ----
    print("\n  [7/11] PCA...", flush=True)
    t1 = time.time()
    try:
        fc = features - features.mean(axis=0)
        cov = fc.T @ fc / (n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        coords = fc @ eigvecs[:, -2:]
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "PCA")
        coords = rescale_coordinates(coords, TARGET_STD)
        embeddings["PCA"] = (coords, clean_nodes)
        print(f"    {time.time()-t1:.1f}s, {len(clean_nodes)} nodes")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 8. VGAE-feat (sparse, centrality features) ----
    print("\n  [8/11] VGAE-feat (sparse neg-sampling, centrality features)...", flush=True)
    t1 = time.time()
    try:
        coords = sparse_vgae(G_annotated, hidden_dim=4, latent_dim=2, epochs=300,
                             lr=0.01, features=features, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        coords, clean_nodes = detect_and_remove_outliers(coords, node_list, "VGAE-feat")
        coords = rescale_coordinates(coords, TARGET_STD)
        embeddings["VGAE-feat"] = (coords, clean_nodes)
        print(f"    {time.time()-t1:.1f}s, {len(clean_nodes)} nodes")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ---- 9-11. GNN methods (sparse neg-sampling) ----
    for idx_gnn, (name, builder) in enumerate([
        ("GraphSAGE", _build_sage_encoder),
        ("GAT", _build_gat_encoder),
        ("GIN", _build_gin_encoder),
    ], start=9):
        print(f"\n  [{idx_gnn}/11] {name} (sparse neg-sampling)...", flush=True)
        t1 = time.time()
        try:
            coords = _train_gnn_sparse(
                G_annotated, builder,
                hidden_dim=16, latent_dim=2,
                epochs=200, lr=0.01,
                features=features, seed=SEED, neg_ratio=1,
            )
            coords = rescale_coordinates(coords, TARGET_STD)
            coords, clean_nodes = detect_and_remove_outliers(coords, node_list, name)
            coords = rescale_coordinates(coords, TARGET_STD)
            embeddings[name] = (coords, clean_nodes)
            print(f"    {time.time()-t1:.1f}s, {len(clean_nodes)} nodes")
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\n  Embeddings complete: {len(embeddings)}/11 methods")

    # ----- Step 4: Subsample to common annotated nodes -----
    print(f"\n[4/6] Subsampling to common nodes...")
    rng = np.random.default_rng(SEED)

    # Find common nodes across all embeddings
    common = None
    for method, (coords, nodes) in embeddings.items():
        s = set(nodes)
        common = s if common is None else common & s
    common = sorted(common)
    print(f"  Common nodes across all methods: {len(common)}")

    if len(common) > SUBSAMPLE_SIZE:
        subsample_nodes = sorted(rng.choice(common, SUBSAMPLE_SIZE, replace=False))
    else:
        subsample_nodes = common
    print(f"  Subsample size: {len(subsample_nodes)}")

    sub_embeddings = {}
    for method, (coords, nodes) in embeddings.items():
        node_set = set(subsample_nodes)
        idx = [i for i, n in enumerate(nodes) if n in node_set]
        sub_coords = coords[idx]
        sub_nodes = [nodes[i] for i in idx]
        sub_coords = rescale_coordinates(sub_coords, TARGET_STD)
        sub_embeddings[method] = (sub_coords, sub_nodes)

    # ----- Step 5: Compute GF curves and scores -----
    print(f"\n[5/6] Computing G-F curves ({len(sub_embeddings)} methods, greedy_modularity)...")
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    print(f"  r-grid: [{R_MIN}, {R_MAX}], {N_POINTS} points")
    print(f"  Community detection: greedy_modularity_communities")
    print(f"  GF interval: [{GF_R_MIN}, {GF_R_MAX}]")

    all_results = {}
    for i, method in enumerate(ALL_METHODS):
        if method not in sub_embeddings:
            print(f"    [{i+1}/11] {method}: SKIPPED (no embedding)")
            continue
        coords, nodes = sub_embeddings[method]
        t1 = time.time()
        purities = compute_gf_curve_greedy(coords, nodes, go_map, r_vals)
        elapsed = time.time() - t1
        print()  # end progress dots

        gf_score = compute_gf_score(purities, r_vals, GF_R_MIN, GF_R_MAX)
        peak_purity = float(np.max(purities))

        all_results[method] = {
            "gf_score": round(gf_score, 6),
            "peak_purity": round(peak_purity, 4),
            "n_nodes": len(nodes),
        }
        print(f"    [{i+1}/11] {method:<12}: GF={gf_score:.4f}, "
              f"peak_purity={peak_purity:.4f}, time={elapsed:.1f}s", flush=True)

    # ----- Step 6: Ranking + Cross-species Kendall's W -----
    print(f"\n[6/6] Method ranking + cross-species Kendall's W...")

    # Rank methods by GF score (highest = rank 1)
    methods_ranked = sorted(all_results.keys(),
                            key=lambda m: all_results[m]["gf_score"], reverse=True)
    ecoli_ranks = {}
    for rank, method in enumerate(methods_ranked, 1):
        ecoli_ranks[method] = rank

    print(f"\n  E. coli Method Ranking (by G-F Score):")
    for rank, method in enumerate(methods_ranked, 1):
        gf = all_results[method]["gf_score"]
        print(f"    {rank:2d}. {method:<12}: {gf:.4f}")

    # Load existing 3-species results
    print(f"\n  Loading existing cross-species data...")
    three_way_file = RESULTS_DIR / "cross_species_three_way.json"
    with open(three_way_file, encoding="utf-8") as f:
        three_way = json.load(f)

    common_methods = [m for m in ALL_METHODS if m in ecoli_ranks]
    yeast_ranks = three_way["yeast_ranks"]
    human_ranks = three_way["human_ranks"]
    mouse_ranks = three_way["mouse_ranks"]

    # Ensure all 11 methods are present in all species
    methods_4sp = [m for m in common_methods
                   if m in yeast_ranks and m in human_ranks and m in mouse_ranks]
    print(f"  Common methods across 4 species: {len(methods_4sp)}")

    # Build rank arrays
    rank_arrays = []
    for sp_name, sp_ranks in [("yeast", yeast_ranks), ("human", human_ranks),
                               ("mouse", mouse_ranks), ("ecoli", ecoli_ranks)]:
        ranks = [sp_ranks[m] for m in methods_4sp]
        rank_arrays.append(ranks)

    # Kendall's W (3 species - original)
    W_3sp = kendalls_w(rank_arrays[:3])
    # Kendall's W (4 species - updated)
    W_4sp = kendalls_w(rank_arrays)

    print(f"\n  Kendall's W (3 species: yeast+human+mouse): {W_3sp:.4f}")
    print(f"  Kendall's W (4 species: +E. coli):          {W_4sp:.4f}")

    # Check: does Spectral > GNN pattern hold for E. coli?
    spectral_rank = ecoli_ranks.get("Spectral", -1)
    gnn_ranks = [ecoli_ranks.get(m, 99) for m in ["GraphSAGE", "GAT", "GIN"]]
    spectral_beats_gnn = all(spectral_rank < r for r in gnn_ranks if r < 99)
    spectral_beats_all_gnn = spectral_rank < min(gnn_ranks) if gnn_ranks else False

    print(f"\n  Spectral rank: #{spectral_rank}")
    print(f"  GNN ranks: GraphSAGE=#{ecoli_ranks.get('GraphSAGE', 'N/A')}, "
          f"GAT=#{ecoli_ranks.get('GAT', 'N/A')}, "
          f"GIN=#{ecoli_ranks.get('GIN', 'N/A')}")
    print(f"  Spectral > all GNN methods: {spectral_beats_all_gnn}")

    # SQI: Spectral Quality Index = (mean GNN rank - Spectral rank) / (n_methods - 1)
    n_methods = len(methods_4sp)
    if spectral_rank > 0 and gnn_ranks:
        mean_gnn_rank = np.mean([r for r in gnn_ranks if r < 99])
        sqi = (mean_gnn_rank - spectral_rank) / (n_methods - 1)
    else:
        sqi = 0.0
    print(f"  SQI (Spectral Quality Index): {sqi:.4f}")

    # ----- Save results -----
    output = {
        "analysis": "Cross-Species G-F Score Analysis: E. coli (4th Species)",
        "species": "Escherichia coli K-12 MG1655 (taxon 511145)",
        "string_version": "v11.5",
        "score_threshold": SCORE_THRESHOLD,
        "community_detection": "greedy_modularity_communities",
        "gf_interval": [GF_R_MIN, GF_R_MAX],
        "n_points": N_POINTS,
        "r_range": [R_MIN, R_MAX],
        "subsample_size": len(subsample_nodes),
        "network_stats": {
            "n_nodes": len(annotated_nodes),
            "n_edges": G_annotated.number_of_edges(),
            "n_annotated_with_go": len(subsample_nodes),
        },
        "gf_scores": {m: all_results[m] for m in methods_ranked},
        "ranking": {m: ecoli_ranks[m] for m in methods_ranked},
        "cross_species": {
            "methods": methods_4sp,
            "yeast_ranks": {m: yeast_ranks[m] for m in methods_4sp},
            "human_ranks": {m: human_ranks[m] for m in methods_4sp},
            "mouse_ranks": {m: mouse_ranks[m] for m in methods_4sp},
            "ecoli_ranks": {m: ecoli_ranks[m] for m in methods_4sp},
            "kendall_W_3_species": round(W_3sp, 4),
            "kendall_W_4_species": round(W_4sp, 4),
            "W_change": round(W_4sp - W_3sp, 4),
        },
        "spectral_quality": {
            "spectral_rank": spectral_rank,
            "gnn_ranks": {m: ecoli_ranks.get(m) for m in ["GraphSAGE", "GAT", "GIN"]},
            "spectral_beats_all_gnn": spectral_beats_all_gnn,
            "sqi": round(sqi, 4),
        },
        "interpretation": (
            f"E. coli G-F analysis ({len(annotated_nodes)} nodes, {len(subsample_nodes)} subsampled). "
            f"Spectral rank #{spectral_rank} of {n_methods}. "
            f"Kendall's W across 4 species: {W_4sp:.4f} "
            f"({'increased' if W_4sp > W_3sp else 'decreased'} from {W_3sp:.4f} with 3 species). "
            f"Spectral > GNN pattern {'HOLDS' if spectral_beats_all_gnn else 'does NOT hold'} for E. coli."
        ),
    }

    out_file = RESULTS_DIR / "ecoli_gf_scores.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_file}")

    print(f"\n{BANNER}")
    print("E. coli cross-species analysis complete.")
    print(BANNER)

    return output


if __name__ == "__main__":
    run()
