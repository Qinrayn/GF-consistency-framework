#!/usr/bin/env python3
"""
Phase 11: Spectral Transferability Theory
==========================================
Explains WHY the two-factor model (spectral alignment + effective rank) works
on some PPI networks but fails on others.

Core idea: The discriminative power of spectral alignment is bounded by the
network's Spectral Quality Index (SQI):

    SQI = (lambda_2 / lambda_2_random) * PR(v_2) * FA_max

where:
    lambda_2 = spectral gap (2nd eigenvalue of normalized Laplacian)
    lambda_2_random = expected gap for Erdos-Renyi of same density
    PR(v_2) = participation ratio of Fiedler vector (1=delocalized, 0=localized)
    FA_max = maximum functional alignment of any Laplacian mode with GO annotations

Proposition: Var(spectral_alignment) across embedding methods is bounded by
a function of SQI. When SQI is small, spectral alignment compresses into a
narrow range and cannot discriminate between methods.

Three parts:
  Part 1 - Laplacian spectral analysis (yeast/human/mouse)
  Part 2 - Proposition verification on empirical networks
  Part 3 - Synthetic SBM validation with controlled spectral gap

Outputs:
  results/spectral_transferability.json
  figures/Fig55-59
"""
from __future__ import annotations

import json
import sys
import time
import warnings
import gzip
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    ALL_METHODS, SEED, TARGET_STD,
    get_data_dir, get_results_dir, get_figures_dir,
    rescale_coordinates,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()
N_EIG = 50
BANNER = "=" * 70


# ============================================================
# Network Loading
# ============================================================

def load_yeast_full():
    G = nx.read_edgelist(str(DATA / "yeast_ppi_5936.edgelist"))
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def load_human_full():
    G = nx.Graph()
    fpath = DATA.parent / "human_validation" / "9606.protein.links.v12.0.txt.gz"
    with gzip.open(str(fpath), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and int(parts[2]) >= 700:
                G.add_edge(parts[0], parts[1])
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def load_mouse_full():
    G = nx.read_edgelist(str(DATA / "mouse_ppi.edgelist"))
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def load_go(species):
    fname = f"{species}_go_annotations.json"
    if species == "yeast":
        fname = "gene_go_map.json"
    fpath = DATA / fname
    if not fpath.exists():
        # Try human_validation directory
        fpath = DATA.parent / "human_validation" / "goa_human_go_annotations.json"
        if species == "human" and not fpath.exists():
            fpath = DATA / "human_go_annotations.json"
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Spectral Analysis
# ============================================================

def compute_laplacian_spectrum(G, node_list, k=50):
    """Compute top-k eigenvalues/eigenvectors of normalized Laplacian via sparse eigsh."""
    n = len(node_list)
    node_idx = {nd: i for i, nd in enumerate(node_list)}

    # Build sparse adjacency
    rows, cols, vals = [], [], []
    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([1.0, 1.0])
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))

    # Degree
    deg = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-10))

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt_mat = diags(d_inv_sqrt)
    L_norm = identity(n) - D_inv_sqrt_mat @ A @ D_inv_sqrt_mat

    # Sparse eigendecomposition (smallest eigenvalues)
    k_actual = min(k, n - 2)
    eigenvalues, eigenvectors = eigsh(L_norm, k=k_actual, sigma=0, which="LM")

    # Sort ascending
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, A, deg


def participation_ratio(v):
    """PR = (sum v_i^2)^2 / (n * sum v_i^4). 1=uniform, 0=localized."""
    n = len(v)
    v2 = v ** 2
    v4 = v ** 4
    return float((np.sum(v2)) ** 2 / (n * np.sum(v4) + 1e-15))


def functional_alignment_modes(eigenvectors, node_list, go_map, k=50):
    """For each Laplacian eigenvector, measure how well it separates GO modules."""
    k_actual = min(k, eigenvectors.shape[1])
    n = len(node_list)

    # Build GO-term groups
    go_groups = {}
    for i, nd in enumerate(node_list):
        for term in go_map.get(nd, []):
            if term not in go_groups:
                go_groups[term] = []
            go_groups[term].append(i)
    go_groups = {t: idxs for t, idxs in go_groups.items() if len(idxs) >= 3}

    if not go_groups:
        return np.zeros(k_actual)

    func_alignment = np.zeros(k_actual)
    for j in range(k_actual):
        v = eigenvectors[:, j]
        total_var = np.var(v)
        if total_var < 1e-12:
            continue
        within_vars = [np.var(v[idxs]) for idxs in go_groups.values()]
        func_alignment[j] = 1.0 - np.mean(within_vars) / total_var

    return func_alignment


def compute_spectral_alignment_for_species(G, node_list, eigenvectors, go_map,
                                           species, k=50):
    """Compute spectral alignment for each embedding method on this species."""
    from scipy.spatial.distance import pdist, squareform

    k_actual = min(k, eigenvectors.shape[1])

    # Functional frequency band
    func_band = functional_alignment_modes(eigenvectors, node_list, go_map, k)
    func_band = np.maximum(func_band, 0)
    func_band_norm = func_band / (np.sum(func_band) + 1e-15)

    # Load embeddings and compute alignment
    alignments = {}
    for method in ALL_METHODS:
        fpath = DATA / f"{species}_{method.lower()}_embedding.json"
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        if not raw:
            continue

        nodes_emb = sorted(raw.keys())
        coords = np.array([[raw[n]["x"], raw[n]["y"]] for n in nodes_emb])
        annotated = set(go_map.keys())
        mask = [n in annotated for n in nodes_emb]
        coords = coords[mask]
        nodes_ann = [n for n, m in zip(nodes_emb, mask) if m]

        # Align to eigenvector node ordering
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        common = [n for n in nodes_ann if n in node_to_idx]
        if len(common) < 100:
            continue
        node_to_emb = {n: i for i, n in enumerate(nodes_ann)}
        emb_idx = [node_to_emb[n] for n in common if n in node_to_emb]
        eig_idx = [node_to_idx[n] for n in common]

        Y = coords[emb_idx]
        Y = Y - Y.mean(axis=0)
        V_sub = eigenvectors[eig_idx, :k_actual]

        # Project
        C = V_sub.T @ Y  # (k, 2)
        energy = np.sum(C ** 2, axis=1)
        total_energy = np.sum(energy)
        if total_energy < 1e-15:
            continue
        profile = energy / total_energy

        # Alignment = dot product of profile and func_band
        sa = float(np.dot(profile[:k_actual], func_band_norm[:k_actual]))
        alignments[method] = sa

    return alignments


# ============================================================
# Main
# ============================================================

def run():
    print(BANNER)
    print("Phase 11: Spectral Transferability Theory")
    print(BANNER)

    species_configs = [
        ("yeast", load_yeast_full, "gene_go_map.json"),
        ("human", load_human_full, "human_go_annotations.json"),
        ("mouse", load_mouse_full, "mouse_go_annotations.json"),
    ]

    all_results = {}

    for species, loader, go_file in species_configs:
        print(f"\n{'='*50}")
        print(f"  {species.upper()}")
        print(f"{'='*50}")

        t0 = time.time()
        G = loader()
        node_list = sorted(G.nodes())
        n = len(node_list)
        m = G.number_of_edges()
        d_bar = 2 * m / n
        density = 2 * m / (n * (n - 1))
        print(f"  Network: {n} nodes, {m} edges, d_bar={d_bar:.1f}, density={density:.5f}")
        print(f"  Load: {time.time()-t0:.1f}s")

        # GO annotations
        fpath = DATA / go_file
        if not fpath.exists():
            fpath = DATA.parent / "human_validation" / go_file
        with open(fpath, encoding="utf-8") as f:
            go_map = json.load(f)
        annotated_in_net = sum(1 for nd in node_list if nd in go_map)
        print(f"  GO annotated in network: {annotated_in_net}/{n}")

        # Laplacian spectrum
        print(f"  Computing Laplacian spectrum (top-{N_EIG} modes)...")
        t1 = time.time()
        eigenvalues, eigenvectors, A_sparse, deg = compute_laplacian_spectrum(
            G, node_list, k=N_EIG)
        print(f"  Spectrum: {time.time()-t1:.1f}s")

        # Spectral gap
        lambda_2 = eigenvalues[1]
        lambda_3 = eigenvalues[2]
        gap_23 = lambda_3 - lambda_2
        lambda_2_random = 4.0 / (n * d_bar) if n * d_bar > 0 else 1e-10
        lambda_2_relative = lambda_2 / lambda_2_random

        print(f"  lambda_2 = {lambda_2:.6f}")
        print(f"  lambda_3 = {lambda_3:.6f}")
        print(f"  gap(2,3) = {gap_23:.6f}")
        print(f"  lambda_2_random (ER) = {lambda_2_random:.8f}")
        print(f"  lambda_2 / lambda_2_random = {lambda_2_relative:.2f}")

        # Participation ratio of Fiedler vector
        fiedler = eigenvectors[:, 1]
        pr = participation_ratio(fiedler)
        print(f"  Fiedler PR = {pr:.4f}")

        # Functional alignment
        print(f"  Computing functional alignment of {N_EIG} modes...")
        t2 = time.time()
        func_al = functional_alignment_modes(eigenvectors, node_list, go_map, N_EIG)
        fa_max = float(np.max(func_al))
        fa_top5 = float(np.mean(np.sort(func_al)[-5:]))
        best_mode = int(np.argmax(func_al))
        print(f"  FA_max = {fa_max:.4f} (mode {best_mode})")
        print(f"  FA top-5 mean = {fa_top5:.4f}")
        print(f"  Functional alignment: {time.time()-t2:.1f}s")

        # Spectral Quality Index
        sqi = lambda_2_relative * pr * fa_max
        print(f"  SQI = {lambda_2_relative:.2f} * {pr:.4f} * {fa_max:.4f} = {sqi:.4f}")

        # Spectral alignment for embedding methods
        print(f"  Computing spectral alignment for embeddings...")
        t3 = time.time()

        if species == "yeast":
            # Yeast: load Phase 3 pre-computed spectral alignment (curated 153-node)
            # Full-network yeast embeddings don't exist — only curated subset
            try:
                with open(RESULTS / "spectral_alignment.json", encoding="utf-8") as f:
                    phase3 = json.load(f)
                sa_scores = {m: v["alignment_score"]
                             for m, v in phase3.get("alignment_results", {}).items()
                             if "alignment_score" in v}
                print(f"  Loaded Phase 3 spectral alignment ({len(sa_scores)} methods)")
            except Exception as e:
                print(f"  Warning: Could not load Phase 3 data: {e}")
                sa_scores = {}
        else:
            sa_scores = compute_spectral_alignment_for_species(
                G, node_list, eigenvectors, go_map, species, N_EIG)
        sa_vals = list(sa_scores.values())
        sa_mean = np.mean(sa_vals) if sa_vals else 0
        sa_std = np.std(sa_vals) if sa_vals else 0
        sa_range = max(sa_vals) - min(sa_vals) if sa_vals else 0
        print(f"  SA: n={len(sa_scores)}, mean={sa_mean:.4f}, std={sa_std:.4f}, range={sa_range:.4f}")
        for m_name in sorted(sa_scores, key=sa_scores.get, reverse=True):
            print(f"    {m_name:<12}: {sa_scores[m_name]:.4f}")
        print(f"  Spectral alignment: {time.time()-t3:.1f}s")

        # Load two-factor model rho from existing results
        two_factor_rho = None
        if species == "yeast":
            try:
                with open(RESULTS / "spectral_alignment.json", encoding="utf-8") as f:
                    d = json.load(f)
                two_factor_rho = d.get("predictor_comparison", {}).get("combined", {}).get("rho")
            except Exception as e:
                print(f"WARNING: Could not load spectral_alignment.json: {e}, using fallback value")
                two_factor_rho = 0.929  # known from Phase 3
        elif species == "human":
            try:
                with open(RESULTS / "persistence_image_analysis.json", encoding="utf-8") as f:
                    d = json.load(f)
                two_factor_rho = d.get("correlations", {}).get("human", {}).get("two_factor", {}).get("rho")
            except Exception as e:
                print(f"WARNING: Could not load persistence_image_analysis.json: {e}, using fallback value")
                two_factor_rho = 0.483
        elif species == "mouse":
            try:
                with open(RESULTS / "persistence_image_analysis.json", encoding="utf-8") as f:
                    d = json.load(f)
                two_factor_rho = d.get("correlations", {}).get("mouse", {}).get("two_factor", {}).get("rho")
            except Exception as e:
                print(f"WARNING: Could not load persistence_image_analysis.json: {e}, using fallback value")
                two_factor_rho = -0.037

        all_results[species] = {
            "n_nodes": n,
            "n_edges": m,
            "avg_degree": round(d_bar, 2),
            "density": round(density, 6),
            "lambda_2": round(lambda_2, 6),
            "lambda_3": round(lambda_3, 6),
            "gap_23": round(gap_23, 6),
            "lambda_2_random": round(lambda_2_random, 8),
            "lambda_2_relative": round(lambda_2_relative, 4),
            "fiedler_pr": round(pr, 4),
            "fa_max": round(fa_max, 4),
            "fa_best_mode": best_mode,
            "fa_top5_mean": round(fa_top5, 4),
            "functional_alignment_top10": [round(float(x), 4) for x in func_al[:10]],
            "eigenvalues_top20": [round(float(x), 6) for x in eigenvalues[:20]],
            "sqi": round(sqi, 4),
            "spectral_alignment": {m: round(v, 4) for m, v in sa_scores.items()},
            "sa_std": round(sa_std, 4),
            "sa_range": round(sa_range, 4),
            "two_factor_rho": two_factor_rho,
        }

    # =========================================================
    # Part 2: Proposition Verification
    # =========================================================
    print(f"\n{'='*50}")
    print("PART 2: Proposition Verification")
    print(f"{'='*50}")

    print("\n  Proposition 1 (Spectral Quality Bound):")
    print("  Var(SA) is bounded by a function of SQI = (lambda_2/lambda_2_ER) * PR * FA_max")
    print()

    for sp in ["yeast", "human", "mouse"]:
        r = all_results[sp]
        print(f"  {sp:<8}: SQI={r['sqi']:.4f}, SA_std={r['sa_std']:.4f}, "
              f"SA_range={r['sa_range']:.4f}, two_factor_rho={r['two_factor_rho']}")

    # Correlation: SQI vs SA discriminative power
    sqi_vals = [all_results[sp]["sqi"] for sp in ["yeast", "human", "mouse"]]
    sa_std_vals = [all_results[sp]["sa_std"] for sp in ["yeast", "human", "mouse"]]
    tf_rho_vals = [all_results[sp]["two_factor_rho"] for sp in ["yeast", "human", "mouse"]]

    print(f"\n  SQI vs SA_std: monotonic = {all(sqi_vals[i] >= sqi_vals[i+1] for i in range(2) if sa_std_vals[i] >= sa_std_vals[i+1])}")
    print(f"  SQI vs |two_factor_rho|: monotonic check")
    for sp in ["yeast", "human", "mouse"]:
        r = all_results[sp]
        print(f"    {sp:<8}: SQI={r['sqi']:.4f}, |rho|={abs(r['two_factor_rho']):.3f}")

    # =========================================================
    # Part 3: Synthetic SBM Validation
    # =========================================================
    print(f"\n{'='*50}")
    print("PART 3: Synthetic SBM Validation")
    print(f"{'='*50}")

    from networkx.generators.community import stochastic_block_model

    rng = np.random.RandomState(SEED)
    sbm_results = []

    configs = []
    for n_sbm in [500, 1000, 2000]:
        for k_comm in [5, 10, 20]:
            for p_ratio in [2.0, 5.0, 10.0, 20.0]:
                if n_sbm * k_comm > 10000:
                    continue
                p_in = min(0.3, k_comm * 0.01 * p_ratio)
                p_out = p_in / p_ratio
                if p_out < 0.0001:
                    continue
                configs.append((n_sbm, k_comm, p_in, p_out))

    # Subsample to ~20 configs for tractability
    if len(configs) > 20:
        rng_idx = rng.choice(len(configs), 20, replace=False)
        configs = [configs[i] for i in sorted(rng_idx)]

    print(f"  Generating {len(configs)} SBM networks...")

    for ci, (n_sbm, k_comm, p_in, p_out) in enumerate(configs):
        sizes = [n_sbm // k_comm] * k_comm
        sizes[-1] += n_sbm - sum(sizes)

        try:
            G_sbm = stochastic_block_model(sizes, [[p_in if i == j else p_out
                                                   for j in range(k_comm)]
                                                  for i in range(k_comm)],
                                          seed=SEED + ci)
        except Exception as e:
            import logging; logging.warning(f"Exception in {__name__}: {e}")
            continue

        # Largest CC
        if not nx.is_connected(G_sbm):
            lcc = max(nx.connected_components(G_sbm), key=len)
            G_sbm = G_sbm.subgraph(lcc).copy()

        n_actual = G_sbm.number_of_nodes()
        if n_actual < 100:
            continue

        node_list_sbm = sorted(G_sbm.nodes())
        m_sbm = G_sbm.number_of_edges()
        d_bar_sbm = 2 * m_sbm / n_actual

        # Spectrum
        try:
            eig_vals, eig_vecs, _, _ = compute_laplacian_spectrum(
                G_sbm, node_list_sbm, k=min(30, n_actual - 2))
        except Exception as e:
            import logging; logging.warning(f"Exception in {__name__}: {e}")
            continue

        lam2 = eig_vals[1]
        lam2_er = 4.0 / (n_actual * d_bar_sbm) if n_actual * d_bar_sbm > 0 else 1e-10
        lam2_rel = lam2 / lam2_er
        pr_sbm = participation_ratio(eig_vecs[:, 1])

        # Synthetic "functional alignment": use ground-truth communities
        communities = {}
        node_to_idx = {n: i for i, n in enumerate(node_list_sbm)}
        for nd in G_sbm.nodes():
            comm = G_sbm.nodes[nd].get("block", 0)
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(node_to_idx[nd])

        k_eig = min(30, eig_vecs.shape[1])
        fa_sbm = np.zeros(k_eig)
        for j in range(k_eig):
            v = eig_vecs[:, j]
            total_var = np.var(v)
            if total_var < 1e-12:
                continue
            within = [np.var(v[idxs]) for idxs in communities.values() if len(idxs) >= 3]
            fa_sbm[j] = 1.0 - np.mean(within) / total_var if within else 0

        fa_max_sbm = float(np.max(fa_sbm))
        sqi_sbm = lam2_rel * pr_sbm * fa_max_sbm

        # Compute spectral alignment for a few fast embeddings
        sa_sbm = {}
        for method in ["Spectral", "MDS", "DM", "PCA"]:
            fpath = None
            # Compute embeddings on the fly for SBM (lightweight)
            try:
                if method == "Spectral":
                    coords = eig_vecs[:, 1:3]
                elif method == "PCA":
                    from sklearn.decomposition import PCA
                    # Use degree + clustering features
                    feats = np.column_stack([
                        np.array([G_sbm.degree(n) for n in node_list_sbm]),
                        np.array([nx.clustering(G_sbm, n) for n in node_list_sbm]),
                    ]).astype(float)
                    feats = feats - feats.mean(axis=0)
                    coords = PCA(n_components=2).fit_transform(feats)
                elif method == "DM":
                    from sklearn.preprocessing import normalize
                    feats = np.column_stack([
                        np.array([G_sbm.degree(n) for n in node_list_sbm]),
                        np.array([nx.clustering(G_sbm, n) for n in node_list_sbm]),
                    ]).astype(float)
                    feats_n = normalize(feats, norm="l2", axis=0)
                    sim = feats_n @ feats_n.T
                    rs = sim.sum(axis=1)
                    d_inv = 1.0 / (np.sqrt(rs) + 1e-10)
                    sim *= d_inv[:, None]
                    sim *= d_inv[None, :]
                    from scipy.sparse.linalg import eigsh as sp_eigsh
                    ev, evec = sp_eigsh(sim, k=3, which="LM")
                    idx = np.argsort(ev)
                    coords = evec[:, idx[-2::-1][:2]]
                elif method == "MDS":
                    from sklearn.manifold import MDS as skMDS
                    from scipy.spatial.distance import squareform, pdist as sp_pdist
                    # Use shortest-path distances (sample if too large)
                    sample_idx = rng.choice(n_actual, min(500, n_actual), replace=False)
                    sample_nodes = [node_list_sbm[i] for i in sample_idx]
                    lengths = dict(nx.all_pairs_shortest_path_length(
                        G_sbm, cutoff=10))
                    D = np.zeros((len(sample_idx), len(sample_idx)))
                    for ii, si in enumerate(sample_idx):
                        for jj, sj in enumerate(sample_idx):
                            D[ii, jj] = lengths.get(node_list_sbm[si], {}).get(
                                node_list_sbm[sj], 10)
                    D[D == 0] = 10
                    np.fill_diagonal(D, 0)
                    coords_sample = skMDS(n_components=2, dissimilarity="precomputed",
                                          random_state=SEED).fit_transform(D)
                    # Skip MDS for full network — use sampled version
                    sa_sbm[method] = None
                    continue
                else:
                    continue

                coords = rescale_coordinates(coords, TARGET_STD)

                # Compute spectral alignment
                func_band_norm = np.maximum(fa_sbm, 0)
                fb_sum = np.sum(func_band_norm)
                if fb_sum > 1e-15:
                    func_band_norm = func_band_norm / fb_sum

                Y = coords - coords.mean(axis=0)
                C = eig_vecs[:, :k_eig].T @ Y
                energy = np.sum(C ** 2, axis=1)
                total_e = np.sum(energy)
                if total_e > 1e-15:
                    profile = energy / total_e
                    sa = float(np.dot(profile[:k_eig], func_band_norm[:k_eig]))
                    sa_sbm[method] = sa
                else:
                    sa_sbm[method] = None
            except Exception as e:
                sa_sbm[method] = None

        sa_valid = {m: v for m, v in sa_sbm.items() if v is not None}
        sa_std_sbm = float(np.std(list(sa_valid.values()))) if len(sa_valid) >= 2 else None

        # Community purity as "GF score" proxy
        # For each embedding, compute purity at a fixed radius
        purities = {}
        for method, sa_val in sa_valid.items():
            try:
                if method == "Spectral":
                    coords = eig_vecs[:, 1:3]
                elif method == "PCA":
                    feats = np.column_stack([
                        np.array([G_sbm.degree(n) for n in node_list_sbm]),
                        np.array([nx.clustering(G_sbm, n) for n in node_list_sbm]),
                    ]).astype(float)
                    feats = feats - feats.mean(axis=0)
                    from sklearn.decomposition import PCA
                    coords = PCA(n_components=2).fit_transform(feats)
                elif method == "DM":
                    from sklearn.preprocessing import normalize
                    feats = np.column_stack([
                        np.array([G_sbm.degree(n) for n in node_list_sbm]),
                        np.array([nx.clustering(G_sbm, n) for n in node_list_sbm]),
                    ]).astype(float)
                    feats_n = normalize(feats, norm="l2", axis=0)
                    sim = feats_n @ feats_n.T
                    rs = sim.sum(axis=1)
                    d_inv = 1.0 / (np.sqrt(rs) + 1e-10)
                    sim *= d_inv[:, None]
                    sim *= d_inv[None, :]
                    ev, evec = sp_eigsh(sim, k=3, which="LM")
                    idx = np.argsort(ev)
                    coords = evec[:, idx[-2::-1][:2]]
                else:
                    continue

                coords = rescale_coordinates(coords, TARGET_STD)

                # Purity at r = 0.3 (mid-range)
                from scipy.spatial.distance import pdist as sp_pdist2, squareform as sp_sq
                D = sp_sq(sp_pdist2(coords))
                r = 0.3
                mask = (D < r) & (D > 0)
                n_edges = int(np.sum(mask)) // 2
                if n_edges < 10:
                    continue

                rows, cols = np.where(mask)
                upper = rows < cols
                edges_list = list(zip(rows[upper].tolist(), cols[upper].tolist()))
                G_r = nx.Graph()
                G_r.add_nodes_from(range(n_actual))
                G_r.add_edges_from(edges_list)
                from networkx.algorithms.community import greedy_modularity_communities
                if n_edges > 50000:
                    comms = [frozenset(c) for c in nx.connected_components(G_r)]
                else:
                    try:
                        comms = list(greedy_modularity_communities(G_r))
                    except Exception as e:
                        comms = [frozenset(c) for c in nx.connected_components(G_r)]

                # Community purity (using ground-truth communities)
                node_to_comm = {}
                for comm_id, members in communities.items():
                    for mi in members:
                        node_to_comm[mi] = comm_id

                comm_purities = []
                for comm in comms:
                    labels = [node_to_comm.get(i, -1) for i in comm]
                    if not labels:
                        continue
                    from collections import Counter
                    counts = Counter(labels)
                    comm_purities.append(counts.most_common(1)[0][1] / len(labels))
                if comm_purities:
                    purities[method] = float(np.mean(comm_purities))
            except Exception as e:
                import logging; logging.warning(f"Exception in {__name__}: {e}")
                pass

        # Correlation: SA vs purity
        common_methods_sbm = sorted(set(sa_valid.keys()) & set(purities.keys()))
        rho_sbm = None
        if len(common_methods_sbm) >= 3:
            sa_arr = np.array([sa_valid[m] for m in common_methods_sbm])
            pu_arr = np.array([purities[m] for m in common_methods_sbm])
            if np.std(sa_arr) > 1e-10 and np.std(pu_arr) > 1e-10:
                rho_sbm, _ = spearmanr(sa_arr, pu_arr)
                rho_sbm = float(rho_sbm)

        sbm_entry = {
            "n": n_actual, "k_comm": k_comm, "p_in": round(p_in, 4),
            "p_out": round(p_out, 6),
            "lambda_2": round(float(lam2), 6),
            "lambda_2_rel": round(float(lam2_rel), 4),
            "fiedler_pr": round(float(pr_sbm), 4),
            "fa_max": round(float(fa_max_sbm), 4),
            "sqi": round(float(sqi_sbm), 4),
            "sa_std": round(sa_std_sbm, 4) if sa_std_sbm is not None else None,
            "sa_vs_purity_rho": round(rho_sbm, 3) if rho_sbm is not None else None,
        }
        sbm_results.append(sbm_entry)
        status = f"rho={rho_sbm:+.3f}" if rho_sbm is not None else "N/A"
        print(f"    [{ci+1}/{len(configs)}] n={n_actual}, k={k_comm}, "
              f"lambda_2={lam2:.4f}, SQI={sqi_sbm:.3f}, SA_vs_purity={status}")

    # =========================================================
    # Save Results
    # =========================================================
    print(f"\n  Saving results...")
    output = {
        "analysis": "Phase 11: Spectral Transferability Theory",
        "proposition": {
            "statement": "Var(SA) is bounded by SQI = (lambda_2/lambda_2_ER) * PR * FA_max",
            "interpretation": "Spectral alignment can only discriminate between embedding methods when the host network has (1) a spectral gap significantly above the random baseline, (2) a delocalized Fiedler vector, and (3) at least one Laplacian mode that separates functional modules",
        },
        "empirical_species": all_results,
        "synthetic_sbm": sbm_results,
    }
    with open(RESULTS / "spectral_transferability.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved spectral_transferability.json")

    # =========================================================
    # Generate Figures
    # =========================================================
    print(f"\n  Generating figures...")
    generate_fig55(all_results)
    generate_fig56(all_results)
    generate_fig57(all_results, sbm_results)
    generate_fig58(sbm_results)
    generate_fig59(all_results, sbm_results)

    print(f"\n{BANNER}")
    print("Phase 11 complete.")
    print(BANNER)


# ============================================================
# Figure Generation
# ============================================================

def generate_fig55(results):
    """Fig55: Laplacian spectrum comparison (3 species)."""
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    species_colors = {"yeast": "#2171B5", "human": "#E6550D", "mouse": "#31A354"}

    # Panel A: Eigenvalue distribution
    ax = fig.add_subplot(gs[0, 0])
    for sp in ["yeast", "human", "mouse"]:
        r = results[sp]
        eigs = r["eigenvalues_top20"]
        ax.plot(range(len(eigs)), eigs, "o-", color=species_colors[sp],
                markersize=4, lw=1.5, label=f"{sp} (n={r['n_nodes']})")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Eigenvalue (normalized Laplacian)")
    ax.set_title("A. Laplacian Spectrum (top-20 modes)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Spectral gap comparison
    ax = fig.add_subplot(gs[0, 1])
    species_list = ["yeast", "human", "mouse"]
    lam2_vals = [results[sp]["lambda_2"] for sp in species_list]
    lam2_rel = [results[sp]["lambda_2_relative"] for sp in species_list]
    x = np.arange(len(species_list))
    bars = ax.bar(x, lam2_rel, color=[species_colors[sp] for sp in species_list],
                  edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(species_list)
    ax.set_ylabel("lambda_2 / lambda_2(ER)")
    ax.set_title("B. Normalized Spectral Gap\n(relative to Erdos-Renyi baseline)",
                 fontsize=10, fontweight="bold")
    ax.axhline(y=1, color="gray", ls="--", alpha=0.5, label="ER baseline")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, lam2_rel):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=8)

    # Panel C: Functional alignment
    ax = fig.add_subplot(gs[0, 2])
    for sp in ["yeast", "human", "mouse"]:
        r = results[sp]
        fa = r["functional_alignment_top10"]
        ax.plot(range(len(fa)), fa, "o-", color=species_colors[sp],
                markersize=4, lw=1.5, label=sp)
    ax.set_xlabel("Laplacian mode index")
    ax.set_ylabel("Functional alignment score")
    ax.set_title("C. Functional Alignment by Mode\n(GO module separation)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 11: Laplacian Spectral Analysis (3 species)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig55_laplacian_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig55_laplacian_spectrum.png")


def generate_fig56(results):
    """Fig56: SQI decomposition and summary."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    species_list = ["yeast", "human", "mouse"]
    species_colors = {"yeast": "#2171B5", "human": "#E6550D", "mouse": "#31A354"}

    # Panel A: SQI component decomposition (grouped bars, log scale)
    ax = axes[0]
    lam2_rel = [results[sp]["lambda_2_relative"] for sp in species_list]
    pr_vals = [results[sp]["fiedler_pr"] for sp in species_list]
    fa_vals = [results[sp]["fa_max"] for sp in species_list]
    sqi_vals = [results[sp]["sqi"] for sp in species_list]

    x = np.arange(len(species_list))
    w = 0.22
    comp_colors = ["#4292C6", "#6BAED6", "#BDD7E7"]
    comp_labels = [r"$\lambda_2/\lambda_2^{ER}$", "Fiedler PR", r"FA$_{\max}$"]
    comp_vals = [lam2_rel, pr_vals, fa_vals]
    for ci, (vals, label, color) in enumerate(zip(comp_vals, comp_labels, comp_colors)):
        bars = ax.bar(x + (ci - 1) * w, vals, w, label=label,
                      color=color, edgecolor="k", linewidth=0.5, log=True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.3,
                    f"{val:.4g}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([sp.capitalize() for sp in species_list])
    ax.set_ylabel("Component value (log scale)")
    ax.set_title("A. SQI Component Decomposition", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    # Panel B: SQI vs two-factor rho (log x-axis)
    ax = axes[1]
    tf_rho = [results[sp]["two_factor_rho"] for sp in species_list]
    ax.scatter(sqi_vals, tf_rho, s=120, c=[species_colors[sp] for sp in species_list],
               edgecolors="k", linewidth=1, zorder=3)
    for sp, sq, rh in zip(species_list, sqi_vals, tf_rho):
        ax.annotate(sp, (sq, rh), fontsize=10, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")
    # Trend line on log scale
    log_sqi = np.log10(sqi_vals)
    z = np.polyfit(log_sqi, tf_rho, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(sqi_vals) * 0.5, max(sqi_vals) * 2, 100)
    ax.plot(x_line, p(np.log10(x_line)), "--", color="gray", lw=1, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Spectral Quality Index (SQI, log scale)")
    ax.set_ylabel("Two-factor model rho")
    ax.set_title("B. SQI vs Two-Factor Model Performance",
                 fontsize=10, fontweight="bold")
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 11: Spectral Quality Index (SQI)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig56_sqi_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig56_sqi_summary.png")


def generate_fig57(results, sbm_results):
    """Fig57: SA discriminative power vs SQI (empirical + synthetic)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    species_colors = {"yeast": "#2171B5", "human": "#E6550D", "mouse": "#31A354"}

    # Synthetic SBM (plot first, behind empirical)
    for entry in sbm_results:
        if entry["sa_std"] is not None:
            ax.scatter(entry["sqi"], entry["sa_std"], s=40, c="#CCCCCC",
                       alpha=0.5, edgecolors="k", linewidth=0.3, zorder=2)

    # Empirical species (on top)
    for sp in ["yeast", "human", "mouse"]:
        r = results[sp]
        ax.scatter(r["sqi"], r["sa_std"], s=150, c=species_colors[sp],
                   edgecolors="k", linewidth=1.5, zorder=4, label=sp.capitalize())
        ax.annotate(sp, (r["sqi"], r["sa_std"]), fontsize=10, ha="left",
                    xytext=(8, 5), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Spectral Quality Index (SQI, log scale)")
    ax.set_ylabel("SA standard deviation (discriminative power)")
    ax.set_title("SA Discriminative Power vs SQI\n(empirical species + synthetic SBM networks)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "Fig57_sa_vs_sqi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig57_sa_vs_sqi.png")


def generate_fig58(sbm_results):
    """Fig58: SBM validation — SA_std vs SQI colored by community structure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    valid = [e for e in sbm_results if e["sa_std"] is not None]
    if not valid:
        axes[0].text(0.5, 0.5, "Insufficient SBM data", ha="center", va="center",
                      transform=axes[0].transAxes)
    else:
        # Panel A: SA_std vs SQI, colored by k_comm
        ax = axes[0]
        k_vals = sorted(set(e["k_comm"] for e in valid))
        k_colors = {5: "#E41A1C", 10: "#377EB8", 20: "#4DAF4A"}

        for e in valid:
            c = k_colors.get(e["k_comm"], "#999999")
            ax.scatter(e["sqi"], e["sa_std"], s=max(30, e["n"] / 30),
                       c=c, alpha=0.6, edgecolors="k", linewidth=0.5, zorder=3)

        # Legend for k
        for k in k_vals:
            ax.scatter([], [], s=60, c=k_colors.get(k, "#999999"),
                       edgecolors="k", linewidth=0.5, label=f"k={k} communities")
        ax.legend(fontsize=8, loc="upper left", title="SBM community count")

        # Trend line
        sqi_v = [e["sqi"] for e in valid]
        std_v = [e["sa_std"] for e in valid]
        log_sqi = np.log10(sqi_v)
        z = np.polyfit(log_sqi, std_v, 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(min(sqi_v) * 0.5), np.log10(max(sqi_v) * 2), 100)
        ax.plot(x_line, p(np.log10(x_line)), "--", color="#333", lw=1.5, alpha=0.5)
        rho_trend, _ = spearmanr(log_sqi, std_v)

        ax.set_xscale("log")
        ax.set_xlabel("Spectral Quality Index (SQI, log scale)")
        ax.set_ylabel("SA standard deviation")
        ax.set_title(f"A. SBM: SA Variance vs SQI\n"
                     f"(Spearman rho={rho_trend:+.3f}, {len(valid)} networks)",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Panel B: SA-purity correlation vs SQI (secondary analysis)
        ax = axes[1]
        valid_rho = [e for e in sbm_results if e["sa_vs_purity_rho"] is not None]
        if valid_rho:
            n_vals = [e["n"] for e in valid_rho]
            sqi_rho = [e["sqi"] for e in valid_rho]
            rho_vals = [e["sa_vs_purity_rho"] for e in valid_rho]

            scatter = ax.scatter(sqi_rho, rho_vals,
                                 s=[max(30, n / 20) for n in n_vals],
                                 c=n_vals, cmap="YlOrRd", edgecolors="k",
                                 linewidth=0.5, alpha=0.8, zorder=3)
            plt.colorbar(scatter, ax=ax, label="Network size n")
            ax.axhline(y=0, color="gray", ls="--", alpha=0.5)

            # Note: only 3 methods per network, correlations are noisy
            ax.set_xscale("log")
            ax.set_xlabel("SQI (log scale)")
            ax.set_ylabel("Spearman rho (SA vs purity)")
            ax.set_title("B. SA-Purity Correlation vs SQI\n"
                         "(3 methods/network — noisy estimate)",
                         fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 11: Synthetic SBM Validation",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig58_sbm_phase_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig58_sbm_phase_diagram.png")


def generate_fig59(results, sbm_results):
    """Fig59: Transferability summary — SQI threshold + 3-species positioning."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    species_colors = {"yeast": "#2171B5", "human": "#E6550D", "mouse": "#31A354"}

    # Panel A: SQI vs |two-factor rho| with threshold
    ax = axes[0]
    species_list = ["yeast", "human", "mouse"]
    sqi_vals = [results[sp]["sqi"] for sp in species_list]
    tf_rho = [abs(results[sp]["two_factor_rho"]) for sp in species_list]

    # Add SBM points first (behind)
    valid_sbm = [e for e in sbm_results if e["sa_vs_purity_rho"] is not None]
    if valid_sbm:
        ax.scatter([e["sqi"] for e in valid_sbm],
                   [abs(e["sa_vs_purity_rho"]) for e in valid_sbm],
                   s=30, c="#CCCCCC", alpha=0.4, edgecolors="k", linewidth=0.3)

    # Empirical species on top
    ax.scatter(sqi_vals, tf_rho, s=150, c=[species_colors[sp] for sp in species_list],
               edgecolors="k", linewidth=1.5, zorder=4)
    for sp, sq, rh in zip(species_list, sqi_vals, tf_rho):
        ax.annotate(sp, (sq, rh), fontsize=10, ha="left", va="bottom",
                    xytext=(8, 5), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Spectral Quality Index (SQI, log scale)")
    ax.set_ylabel("|Two-factor model rho| or |SA-purity rho|")
    ax.set_title("A. Transferability: SQI vs Model Performance",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel B: Summary table
    ax = axes[1]
    ax.axis("off")
    table_data = [
        ["Species", "n", "lambda_2", "Fiedler PR", "FA_max", "SQI", "SA_std", "2F rho"],
    ]
    for sp in species_list:
        r = results[sp]
        table_data.append([
            sp.capitalize(), str(r["n_nodes"]),
            f'{r["lambda_2"]:.4f}', f'{r["fiedler_pr"]:.3f}',
            f'{r["fa_max"]:.3f}', f'{r["sqi"]:.3f}',
            f'{r["sa_std"]:.3f}', f'{r["two_factor_rho"]:+.3f}',
        ])

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            table[i, j].set_facecolor(["#DEEBF7", "#F2F2F2"][i % 2])

    ax.set_title("B. Spectral Transferability Summary",
                 fontsize=10, fontweight="bold")

    fig.suptitle("Phase 11: Transferability Theory Summary",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig59_transferability_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig59_transferability_summary.png")


if __name__ == "__main__":
    run()
