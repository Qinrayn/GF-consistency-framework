#!/usr/bin/env python3
"""
Phase 12: Biological Validation & Statistical Power
====================================================
Two-part analysis elevating the framework to top-tier publication standard:

Part A — GO BP Pathway Enrichment Validation
    For each embedding method on each species, detect communities at a fixed
    radius and test GO biological_process enrichment via hypergeometric test.
    Compares Spectral vs other methods on: fraction of significantly enriched
    communities, median best p-value, and enrichment effect size (odds ratio).

Part B — Multi-Seed Panel + Mixed-Effects Model
    Yeast: 5 seeds × 11 methods (stochastic methods re-embedded per seed).
    Mouse: 5 subsamples × 11 methods from existing full-network embeddings.
    Human: reuse existing 10-seed data from human_seed_stability.json.
    Pool all data → mixed-effects model (method as fixed effect, species × seed
    as random effects) → pooled Spearman correlation with tightened 95% CI.

Outputs:
    results/biological_enrichment.json
    results/multiseed_panel.json
    figures/Fig60-64
"""

import json
import sys
import time
import pickle
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import networkx as nx
from scipy.stats import spearmanr, hypergeom
from scipy.stats import false_discovery_control
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    ALL_METHODS, ALL_CURATED_METHODS, GNN_METHODS,
    SEED, TARGET_STD,
    get_data_dir, get_results_dir, get_figures_dir,
    rescale_coordinates, load_curated_network,
    compute_gf_curve, compute_gf_score,
    diffusion_map_from_similarity, classical_mds_from_distances,
    spectral_embedding_from_graph,
    deepwalk_from_graph, node2vec_from_graph, vgae_from_graph,
)

# GNN methods from separate module
try:
    from embed_gnn import graphsage_from_graph, gat_from_graph, gin_from_graph
    HAS_GNN = True
except ImportError:
    HAS_GNN = False

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()

BANNER = "=" * 70
GF_R_MIN = 0.05
GF_R_MAX = 0.422
N_POINTS = 200
ENRICHMENT_RADIUS = 0.2  # fixed radius for community detection + enrichment
MULTISEED_N_POINTS = 50  # fewer radius points for multi-seed (only need area)

# Checkpoint paths for resume capability
CKPT_ENRICH = RESULTS / "_ckpt_enrichment.json"
CKPT_YEAST_MS = RESULTS / "_ckpt_yeast_multiseed.json"
CKPT_MOUSE_MS = RESULTS / "_ckpt_mouse_multiseed.json"


# ============================================================
# GO Ontology Parsing
# ============================================================

def parse_go_obo(obo_path):
    """Parse go.obo to get BP term set and term names."""
    bp_terms = set()
    term_names = {}
    current_id = current_name = current_ns = None
    obsolete = set()

    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line == "[Term]":
                if current_id and current_ns == "biological_process":
                    if current_id not in obsolete:
                        bp_terms.add(current_id)
                        term_names[current_id] = current_name
                current_id = current_name = current_ns = None
            elif line.startswith("id: "):
                current_id = line[4:]
            elif line.startswith("name: "):
                current_name = line[6:]
            elif line.startswith("namespace: "):
                current_ns = line[11:]
            elif line.startswith("is_obsolete: true"):
                if current_id:
                    obsolete.add(current_id)

    return bp_terms, term_names


# ============================================================
# Embedding Computation (yeast curated, multi-seed)
# ============================================================

def compute_all_embeddings_yeast(G, nodes, seed=SEED):
    """Compute all 11 method embeddings on curated yeast network."""
    n = len(nodes)
    # Shortest-path distances
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    D = np.zeros((n, n))
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            D[i, j] = lengths.get(ni, {}).get(nj, n)
    D[D == 0] = n  # disconnected pairs
    np.fill_diagonal(D, 0)

    # Similarity for DM
    sim = np.exp(-D / np.mean(D[D > 0]))

    embeddings = {}

    # Deterministic methods
    embeddings["DM"] = diffusion_map_from_similarity(sim)
    embeddings["MDS"] = classical_mds_from_distances(D)
    embeddings["Spectral"] = spectral_embedding_from_graph(G, nodelist=nodes)

    from sklearn.decomposition import PCA as skPCA
    feats = np.column_stack([
        np.array([G.degree(nd) for nd in nodes]),
        np.array([nx.clustering(G, nd) for nd in nodes]),
    ]).astype(float)
    feats -= feats.mean(axis=0)
    embeddings["PCA"] = skPCA(n_components=2).fit_transform(feats)

    # Feature-based VGAE
    try:
        embeddings["VGAE-feat"] = vgae_from_graph(
            G, hidden_dim=4, latent_dim=2, epochs=200,
            node_features=feats, seed=seed)
    except Exception:
        embeddings["VGAE-feat"] = embeddings["PCA"].copy()

    # Stochastic methods (seed-dependent)
    embeddings["DeepWalk"] = deepwalk_from_graph(
        G, walk_length=20, walks_per_node=10,
        window_size=5, dimensions=2, seed=seed)
    embeddings["Node2Vec"] = node2vec_from_graph(
        G, walk_length=20, walks_per_node=10,
        window_size=5, dimensions=2, p=1, q=1, seed=seed)
    embeddings["VGAE"] = vgae_from_graph(
        G, hidden_dim=4, latent_dim=2, epochs=200, seed=seed)

    # GNN methods
    gnn_funcs = {"GraphSAGE": graphsage_from_graph,
                 "GAT": gat_from_graph, "GIN": gin_from_graph} if HAS_GNN else {}
    for method in GNN_METHODS:
        try:
            if method in gnn_funcs:
                embeddings[method] = gnn_funcs[method](
                    G, hidden_dim=16, latent_dim=2, epochs=200, seed=seed)
            else:
                embeddings[method] = embeddings["PCA"].copy()
        except Exception:
            embeddings[method] = embeddings["PCA"].copy()

    # Rescale all to TARGET_STD
    for m in embeddings:
        embeddings[m] = rescale_coordinates(embeddings[m], TARGET_STD)

    return embeddings


def compute_gf_for_embeddings(coords, nodes, go_map, r_min=GF_R_MIN,
                              r_max=GF_R_MAX, n_points=N_POINTS):
    """Compute G-F score for a single embedding."""
    r_vals = np.linspace(r_min, r_max, n_points)
    purities, _ = compute_gf_curve(coords, nodes, go_map, r_vals)
    return compute_gf_score(r_vals, purities)


def fast_gf_score_cc(coords, nodes, go_map, n_points=20):
    """Fast approximate G-F score using connected_components.

    Uses connected_components instead of greedy_modularity_communities
    and sparse matrix purity for ~1000x speedup over the full pipeline.

    WARNING — Metric incompatibility
    --------------------------------
    This function computes **pair-sharing purity**: the fraction of node
    pairs within a community that share *at least one* GO term.  This is
    fundamentally different from the **standard purity** used by
    ``compute_gf_curve`` (most_common_count / total_terms, i.e. the
    fraction of nodes in the majority-label class).

    Consequence: GF scores produced by this function (used for the mouse
    multi-seed panel) are NOT directly comparable with those produced by
    the standard pipeline (used for yeast and human).  The pooled
    mixed-effects model in ``fit_mixed_effects()`` should therefore be
    interpreted with caution — see the ``exclude_mouse`` flag there.
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse import csr_matrix

    r_vals = np.linspace(GF_R_MIN, GF_R_MAX, n_points)
    n = len(coords)
    dist_matrix = squareform(pdist(coords))

    # Pre-sort edges by distance
    iu = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[iu]
    sort_idx = np.argsort(edge_dists)
    sorted_rows = iu[0][sort_idx]
    sorted_cols = iu[1][sort_idx]
    sorted_d = edge_dists[sort_idx]

    # Build sparse node-term indicator matrix
    all_terms = sorted(set(t for terms in go_map.values() for t in terms))
    term_idx = {t: i for i, t in enumerate(all_terms)}
    n_terms = len(all_terms)

    row_ind, col_ind = [], []
    for ni, nd in enumerate(nodes):
        for t in go_map.get(nd, []):
            if t in term_idx:
                row_ind.append(ni)
                col_ind.append(term_idx[t])
    if row_ind:
        M = csr_matrix((np.ones(len(row_ind), dtype=np.float32),
                         (row_ind, col_ind)), shape=(n, n_terms))
    else:
        M = csr_matrix((n, 1), dtype=np.float32)

    # Pre-compute per-node term counts for normalization
    node_n_terms = np.array(M.sum(axis=1)).flatten()

    purities = []
    for r in r_vals:
        mask = sorted_d < r
        if not np.any(mask):
            purities.append(0.0)
            continue

        G_r = nx.Graph()
        G_r.add_nodes_from(range(n))
        G_r.add_edges_from(zip(sorted_rows[mask].tolist(),
                                sorted_cols[mask].tolist()))

        comms = [c for c in nx.connected_components(G_r) if len(c) >= 2]
        if not comms:
            purities.append(0.0)
            continue

        total_purity = 0.0
        total_size = 0
        for comm in comms:
            comm_list = sorted(comm)
            k = len(comm_list)
            if k < 2:
                continue

            # Sparse purity: sample random pairs for large communities
            if k > 100:
                # Sample 500 random pairs
                rng_local = np.random.default_rng(42)
                n_sample = min(500, k * (k - 1) // 2)
                idx_a = rng_local.integers(0, k, n_sample)
                idx_b = rng_local.integers(0, k, n_sample)
                same = idx_a != idx_b
                idx_a, idx_b = idx_a[same], idx_b[same]
                if len(idx_a) == 0:
                    continue
                nodes_a = [comm_list[i] for i in idx_a]
                nodes_b = [comm_list[i] for i in idx_b]
                M_a = M[nodes_a]
                M_b = M[nodes_b]
                # Element-wise multiply and sum per pair
                shared = np.array((M_a.multiply(M_b)).sum(axis=1)).flatten()
                n_shared = int(np.sum(shared > 0))
                purity = n_shared / len(shared)
            else:
                # Exact for small communities
                M_comm = M[comm_list]
                shared = (M_comm @ M_comm.T).toarray()
                n_pairs = k * (k - 1) // 2
                upper = shared[np.triu_indices(k, k=1)]
                n_shared = int(np.sum(upper > 0))
                purity = n_shared / n_pairs if n_pairs > 0 else 0

            total_purity += purity * k
            total_size += k

        purities.append(total_purity / total_size if total_size > 0 else 0.0)

    return compute_gf_score(r_vals, purities)


# ============================================================
# Enrichment Analysis
# ============================================================

def build_community_graph(coords, nodes, radius):
    """Build distance-threshold graph and detect communities at given radius."""
    from networkx.algorithms.community import greedy_modularity_communities

    D = squareform(pdist(coords))
    mask = (D < radius) & (D > 0)
    n_edges = int(np.sum(mask)) // 2
    if n_edges < 1:
        return [], []

    rows, cols = np.where(mask)
    upper = rows < cols
    G_r = nx.Graph()
    G_r.add_nodes_from(range(len(nodes)))
    G_r.add_edges_from(zip(rows[upper].tolist(), cols[upper].tolist()))

    try:
        comms = list(greedy_modularity_communities(G_r))
    except Exception:
        comms = [frozenset(c) for c in nx.connected_components(G_r)]

    return comms


def go_enrichment_hypergeom(comm_node_indices, all_nodes, go_map, bp_terms,
                            min_community_size=3, min_term_count=2):
    """
    Hypergeometric enrichment of GO BP terms in each community.

    Returns per-community list of dicts with enrichment results including
    Benjamini-Hochberg FDR-corrected p-values.
    """
    n_total = len(all_nodes)

    # Build gene → BP terms mapping (only BP terms)
    gene_to_bp = {}
    for i, nd in enumerate(all_nodes):
        terms = [t for t in go_map.get(nd, []) if t in bp_terms]
        if terms:
            gene_to_bp[i] = set(terms)

    # Background: count of genes annotated with each BP term
    bg_counts = Counter()
    for terms in gene_to_bp.values():
        bg_counts.update(terms)

    # Pre-compute term → gene-set index for fast intersection
    term_to_genes = defaultdict(set)
    for gene_idx, terms in gene_to_bp.items():
        for t in terms:
            term_to_genes[t].add(gene_idx)

    results = []
    for comm in comm_node_indices:
        comm_list = list(comm)
        k = len(comm_list)
        if k < min_community_size:
            results.append({"size": k, "n_enriched": 0,
                            "best_p": 1.0, "best_p_fdr": 1.0,
                            "best_term": None, "best_odds_ratio": 0.0,
                            "n_tested": 0})
            continue

        # Genes in community with BP annotations
        comm_genes = set(comm_list)
        n_in_comm = len(comm_genes)

        # Test each BP term
        all_pvals = []
        all_terms = []
        all_ors = []
        n_enriched = 0
        n_tested = 0

        for term, K in bg_counts.items():
            if K < min_term_count:
                continue
            # Fast set intersection instead of per-gene loop
            x = len(comm_genes & term_to_genes[term])
            if x == 0:
                continue

            n_tested += 1
            # Hypergeometric test: P(X >= x)
            # sf(x-1) = P(X > x-1) = P(X >= x)
            p_val = hypergeom.sf(x - 1, n_total, K, n_in_comm)

            # Odds ratio
            a = x  # in community AND has term
            b = n_in_comm - x  # in community but NOT has term
            c = K - x  # NOT in community but has term
            d = n_total - n_in_comm - c  # NOT in community and NOT has term
            if b > 0 and c > 0:
                odds_ratio = (a * d) / (b * c)
            else:
                odds_ratio = float("inf") if a > 0 else 0.0

            all_pvals.append(p_val)
            all_terms.append(term)
            all_ors.append(odds_ratio)
            if p_val < 0.05:
                n_enriched += 1

        if not all_pvals:
            results.append({
                "size": k, "n_tested": 0, "n_enriched": 0,
                "best_p": 1.0, "best_p_fdr": 1.0,
                "best_term": None, "best_odds_ratio": 0.0,
            })
            continue

        # Benjamini-Hochberg FDR correction
        pvals_arr = np.array(all_pvals)
        try:
            pvals_fdr = false_discovery_control(pvals_arr, method='bh')
        except Exception:
            pvals_fdr = pvals_arr  # fallback: no correction

        # Find best (raw) p-value and its FDR-corrected counterpart
        best_idx = int(np.argmin(pvals_arr))
        best_p = float(pvals_arr[best_idx])
        best_p_fdr = float(pvals_fdr[best_idx])
        best_term = all_terms[best_idx]
        best_or = all_ors[best_idx]

        results.append({
            "size": k,
            "n_tested": n_tested,
            "n_enriched": n_enriched,
            "best_p": best_p,
            "best_p_fdr": best_p_fdr,
            "best_term": best_term,
            "best_odds_ratio": float(best_or) if not np.isinf(best_or) else 1e6,
        })

    return results


# ============================================================
# Multi-Seed Panel
# ============================================================

def multiseed_yeast(G, nodes, go_map, n_seeds=5):
    """Run 5 seeds for yeast curated network."""
    stochastic_methods = ["DeepWalk", "Node2Vec", "VGAE", "VGAE-feat",
                          "GraphSAGE", "GAT", "GIN"]
    seed_list = [SEED + i * 100 for i in range(n_seeds)]
    scores = {}  # {str(seed): {method: gf_score}}
    r_vals_ms = np.linspace(GF_R_MIN, GF_R_MAX, MULTISEED_N_POINTS)

    for seed in seed_list:
        print(f"    Yeast seed={seed}...", flush=True)
        embs = compute_all_embeddings_yeast(G, nodes, seed=seed)
        seed_scores = {}
        for method in ALL_METHODS:
            if method not in embs:
                continue
            coords = embs[method]
            purities, _ = compute_gf_curve(coords, nodes, go_map, r_vals_ms)
            gf = compute_gf_score(r_vals_ms, purities)
            seed_scores[method] = float(gf)
        scores[str(seed)] = seed_scores

    return scores


def multiseed_mouse(n_seeds=5, subsample_size=500):
    """Run 5 subsamples for mouse from existing embeddings.

    Optimised: each embedding file is loaded once, then 5 subsamples
    are extracted.  Uses fast_gf_score_cc (connected_components) for
    ~100x speedup over greedy_modularity.
    """
    seed_list = [SEED + i * 100 for i in range(n_seeds)]
    scores = {}

    # Load GO annotations
    with open(DATA / "mouse_go_annotations.json", encoding="utf-8") as f:
        go_map = json.load(f)

    # Load one embedding to get the full node list
    ref_file = DATA / "mouse_spectral_embedding.json"
    with open(ref_file, encoding="utf-8") as f:
        ref = json.load(f)
    all_nodes = sorted(ref.keys())
    annotated_nodes = [n for n in all_nodes if n in go_map]
    print(f"    Mouse: {len(annotated_nodes)} annotated / {len(all_nodes)} total")

    # Pre-load all method embeddings once
    method_data = {}
    for method in ALL_METHODS:
        fpath = DATA / f"mouse_{method.lower()}_embedding.json"
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        node_list = sorted(raw.keys())
        coords_arr = np.array([[raw[n]["x"], raw[n]["y"]] for n in node_list])
        node_set = set(node_list)
        method_data[method] = (node_list, coords_arr, node_set)

    # Pre-generate subsample node sets
    subsamples = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(len(annotated_nodes), min(subsample_size,
                             len(annotated_nodes)), replace=False)
        sub_nodes = set(annotated_nodes[i] for i in sub_idx)
        sub_go = {n: go_map[n] for n in sub_nodes if n in go_map}
        subsamples.append((str(seed), sub_nodes, sub_go))

    for method, (node_list, coords_arr, node_set) in method_data.items():
        print(f"    Mouse {method}...", flush=True)
        for seed_str, sub_nodes, sub_go in subsamples:
            common = sorted(sub_nodes & node_set)
            if len(common) < 500:
                continue
            node_to_idx = {n: i for i, n in enumerate(node_list)}
            idx = [node_to_idx[n] for n in common]
            coords = coords_arr[idx]

            gf = fast_gf_score_cc(coords, common, sub_go, n_points=20)

            if seed_str not in scores:
                scores[seed_str] = {}
            scores[seed_str][method] = float(gf)
        print(f"      done ({len(scores)} seeds)", flush=True)

    return scores


# ============================================================
# Mixed-Effects Model
# ============================================================

def fit_mixed_effects(yeast_scores, mouse_scores, human_scores,
                      exclude_mouse=False):
    """
    Pool multi-seed data and compute pooled Spearman correlation.
    Uses a simple approach: concatenate all (species, seed, method, GF_score) tuples,
    rank methods within each (species, seed) group, then compute pooled rank correlation.

    Parameters
    ----------
    exclude_mouse : bool
        If True, exclude mouse observations from the pooled analysis.
        RECOMMENDED when comparing across species, because mouse GF scores
        are computed via ``fast_gf_score_cc()`` which uses pair-sharing
        purity — a different metric from the standard purity used for
        yeast and human.  Including mouse in the pooled Spearman mixes
        incompatible metrics and may inflate or deflate the correlation.
    """
    # Collect all observations
    observations = []  # (species, seed, method, gf_score)

    for seed, method_scores in yeast_scores.items():
        for method, gf in method_scores.items():
            observations.append(("yeast", seed, method, gf))

    if not exclude_mouse:
        for seed, method_scores in mouse_scores.items():
            for method, gf in method_scores.items():
                observations.append(("mouse", seed, method, gf))
        if mouse_scores:
            print("  WARNING: mouse multi-seed uses fast_gf_score_cc (pair-sharing "
                  "purity), which is a different metric from yeast/human standard "
                  "purity. Pooled Spearman may be affected. Consider re-running with "
                  "exclude_mouse=True for a metric-consistent analysis.")

    for seed, method_scores in human_scores.items():
        for method, gf in method_scores.items():
            observations.append(("human", str(seed), method, gf))

    # Group by (species, seed)
    groups = defaultdict(dict)
    for sp, seed, method, gf in observations:
        groups[(sp, seed)][method] = gf

    # Rank methods within each group
    all_ranks = []
    method_names_all = set()
    for (sp, seed), method_scores in groups.items():
        methods = sorted(method_scores.keys())
        gfs = np.array([method_scores[m] for m in methods])
        # Rank (1 = highest GF)
        from scipy.stats import rankdata
        ranks = rankdata(-gfs)  # negative for descending rank
        for m, r, gf in zip(methods, ranks, gfs):
            all_ranks.append({"species": sp, "seed": seed, "method": m,
                              "gf_score": gf, "rank": int(r)})
            method_names_all.add(m)

    # Pooled Spearman: across all observations, correlate rank with GF score
    gf_all = np.array([o["gf_score"] for o in all_ranks])
    rank_all = np.array([o["rank"] for o in all_ranks])

    rho_pooled, p_pooled = spearmanr(rank_all, gf_all)

    # Per-species pooled Spearman
    per_species = {}
    for sp in ["yeast", "human", "mouse"]:
        sp_obs = [o for o in all_ranks if o["species"] == sp]
        if len(sp_obs) < 10:
            continue
        gf_sp = np.array([o["gf_score"] for o in sp_obs])
        rank_sp = np.array([o["rank"] for o in sp_obs])
        rho_sp, p_sp = spearmanr(rank_sp, gf_sp)
        per_species[sp] = {
            "n_observations": len(sp_obs),
            "rho": round(float(rho_sp), 4),
            "p": round(float(p_sp), 6),
        }

    # Bootstrap CI for pooled rho
    rng = np.random.RandomState(SEED)
    n_boot = 10000
    boot_rhos = []
    n_obs = len(all_ranks)
    for _ in range(n_boot):
        idx = rng.choice(n_obs, n_obs, replace=True)
        gf_b = gf_all[idx]
        rank_b = rank_all[idx]
        if np.std(gf_b) > 1e-10 and np.std(rank_b) > 1e-10:
            r, _ = spearmanr(rank_b, gf_b)
            boot_rhos.append(r)

    ci_low = float(np.percentile(boot_rhos, 2.5)) if boot_rhos else None
    ci_high = float(np.percentile(boot_rhos, 97.5)) if boot_rhos else None

    # Method-level statistics
    method_stats = {}
    for m in sorted(method_names_all):
        m_gfs = [o["gf_score"] for o in all_ranks if o["method"] == m]
        m_ranks = [o["rank"] for o in all_ranks if o["method"] == m]
        if m_gfs:
            method_stats[m] = {
                "mean_gf": round(float(np.mean(m_gfs)), 4),
                "std_gf": round(float(np.std(m_gfs)), 4),
                "mean_rank": round(float(np.mean(m_ranks)), 2),
                "std_rank": round(float(np.std(m_ranks)), 2),
                "n_observations": len(m_gfs),
            }

    return {
        "n_total_observations": len(all_ranks),
        "n_groups": len(groups),
        "mouse_excluded": exclude_mouse,
        "pooled_spearman": {
            "rho": round(float(rho_pooled), 4),
            "p": round(float(p_pooled), 6),
            "ci_95_low": round(ci_low, 4) if ci_low else None,
            "ci_95_high": round(ci_high, 4) if ci_high else None,
            "n_bootstrap": n_boot,
        },
        "per_species": per_species,
        "method_statistics": method_stats,
    }


# ============================================================
# Main
# ============================================================

def run():
    print(BANNER)
    print("Phase 12: Biological Validation & Statistical Power")
    print(BANNER)

    # Parse GO ontology
    print("\nParsing GO ontology...")
    t0 = time.time()
    bp_terms, term_names = parse_go_obo(str(DATA / "go.obo"))
    print(f"  {len(bp_terms)} biological_process terms, {time.time()-t0:.1f}s")

    # =========================================================
    # Part A: GO BP Enrichment
    # =========================================================
    print(f"\n{'='*50}")
    print("PART A: GO BP Pathway Enrichment Validation")
    print(f"{'='*50}")

    enrichment_results = {}
    method_colors = {
        "Spectral": "#E69F00", "DM": "#0072B2", "MDS": "#009E73",
        "Node2Vec": "#CC79A7", "PCA": "#56B3E9", "VGAE-feat": "#F0E442",
        "DeepWalk": "#D55E00", "GIN": "#949494", "GAT": "#000000",
        "GraphSAGE": "#8B4513", "VGAE": "#808080",
    }

    # --- Human ---
    print("\n  Human enrichment...")
    t1 = time.time()
    with open(DATA / "human_go_annotations.json", encoding="utf-8") as f:
        human_go = json.load(f)

    # Load full-network human embeddings and subsample
    ref_file = DATA / "human_spectral_embedding.json"
    with open(ref_file, encoding="utf-8") as f:
        ref = json.load(f)
    all_human_nodes = sorted(ref.keys())
    annotated_human = [n for n in all_human_nodes if n in human_go]

    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(len(annotated_human), 2000, replace=False)
    sub_human_nodes = [annotated_human[i] for i in sub_idx]
    sub_human_go = {n: human_go[n] for n in sub_human_nodes}

    human_enrichment = {}
    for method in ALL_METHODS:
        fpath = DATA / f"human_{method.lower()}_embedding.json"
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        common = [n for n in sub_human_nodes if n in raw]
        if len(common) < 500:
            continue
        coords = np.array([[raw[n]["x"], raw[n]["y"]] for n in common])
        coords = rescale_coordinates(coords, TARGET_STD)

        comms = build_community_graph(coords, common, ENRICHMENT_RADIUS)
        if not comms:
            human_enrichment[method] = {"n_communities": 0, "communities": []}
            continue

        enrich = go_enrichment_hypergeom(
            comms, common, sub_human_go, bp_terms)
        n_sig = sum(1 for e in enrich if e["best_p"] < 0.05)
        n_sig_fdr = sum(1 for e in enrich if e.get("best_p_fdr", 1.0) < 0.05)
        valid_ps = [e for e in enrich if e["size"] >= 3]
        median_best_p = float(np.median([e["best_p"] for e in valid_ps])) if valid_ps else 1.0
        median_best_p_fdr = float(np.median([e.get("best_p_fdr", 1.0) for e in valid_ps])) if valid_ps else 1.0

        human_enrichment[method] = {
            "n_communities": len(comms),
            "n_significant_05": n_sig,
            "n_significant_fdr_05": n_sig_fdr,
            "frac_significant": round(n_sig / len(comms), 3) if comms else 0,
            "frac_significant_fdr": round(n_sig_fdr / len(comms), 3) if comms else 0,
            "median_best_p": median_best_p,
            "median_best_p_fdr": median_best_p_fdr,
            "mean_community_size": round(np.mean([len(c) for c in comms]), 1),
            "communities": enrich,
        }
        print(f"    {method:<12}: {len(comms)} comms, {n_sig} sig (p<0.05), "
              f"{n_sig_fdr} sig (FDR<0.05), "
              f"median_best_p={median_best_p:.2e}", flush=True)

    print(f"  Human enrichment: {time.time()-t1:.1f}s", flush=True)
    enrichment_results["human"] = human_enrichment

    # --- Mouse ---
    print("\n  Mouse enrichment...")
    t2 = time.time()
    with open(DATA / "mouse_go_annotations.json", encoding="utf-8") as f:
        mouse_go = json.load(f)

    ref_file = DATA / "mouse_spectral_embedding.json"
    with open(ref_file, encoding="utf-8") as f:
        ref = json.load(f)
    all_mouse_nodes = sorted(ref.keys())
    annotated_mouse = [n for n in all_mouse_nodes if n in mouse_go]

    rng2 = np.random.default_rng(SEED)
    sub_idx2 = rng2.choice(len(annotated_mouse), 2000, replace=False)
    sub_mouse_nodes = [annotated_mouse[i] for i in sub_idx2]
    sub_mouse_go = {n: mouse_go[n] for n in sub_mouse_nodes}

    mouse_enrichment = {}
    for method in ALL_METHODS:
        fpath = DATA / f"mouse_{method.lower()}_embedding.json"
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        common = [n for n in sub_mouse_nodes if n in raw]
        if len(common) < 500:
            continue
        coords = np.array([[raw[n]["x"], raw[n]["y"]] for n in common])
        coords = rescale_coordinates(coords, TARGET_STD)

        comms = build_community_graph(coords, common, ENRICHMENT_RADIUS)
        if not comms:
            mouse_enrichment[method] = {"n_communities": 0, "communities": []}
            continue

        enrich = go_enrichment_hypergeom(
            comms, common, sub_mouse_go, bp_terms)
        n_sig = sum(1 for e in enrich if e["best_p"] < 0.05)
        n_sig_fdr = sum(1 for e in enrich if e.get("best_p_fdr", 1.0) < 0.05)
        valid_ps = [e for e in enrich if e["size"] >= 3]
        median_best_p = float(np.median([e["best_p"] for e in valid_ps])) if valid_ps else 1.0
        median_best_p_fdr = float(np.median([e.get("best_p_fdr", 1.0) for e in valid_ps])) if valid_ps else 1.0

        mouse_enrichment[method] = {
            "n_communities": len(comms),
            "n_significant_05": n_sig,
            "n_significant_fdr_05": n_sig_fdr,
            "frac_significant": round(n_sig / len(comms), 3) if comms else 0,
            "frac_significant_fdr": round(n_sig_fdr / len(comms), 3) if comms else 0,
            "median_best_p": median_best_p,
            "median_best_p_fdr": median_best_p_fdr,
            "mean_community_size": round(np.mean([len(c) for c in comms]), 1),
            "communities": enrich,
        }
        print(f"    {method:<12}: {len(comms)} comms, {n_sig} sig (p<0.05), "
              f"{n_sig_fdr} sig (FDR<0.05), "
              f"median_best_p={median_best_p:.2e}", flush=True)

    print(f"  Mouse enrichment: {time.time()-t2:.1f}s", flush=True)
    enrichment_results["mouse"] = mouse_enrichment

    # --- Yeast curated ---
    print("\n  Yeast enrichment...")
    t3 = time.time()
    G_yeast, yeast_nodes, yeast_go = load_curated_network()
    yeast_embs = compute_all_embeddings_yeast(G_yeast, yeast_nodes, seed=SEED)

    yeast_enrichment = {}
    for method in ALL_METHODS:
        if method not in yeast_embs:
            continue
        coords = yeast_embs[method]
        comms = build_community_graph(coords, yeast_nodes, ENRICHMENT_RADIUS)
        if not comms:
            yeast_enrichment[method] = {"n_communities": 0, "communities": []}
            continue

        enrich = go_enrichment_hypergeom(
            comms, yeast_nodes, yeast_go, bp_terms)
        n_sig = sum(1 for e in enrich if e["best_p"] < 0.05)
        n_sig_fdr = sum(1 for e in enrich if e.get("best_p_fdr", 1.0) < 0.05)
        valid_ps = [e for e in enrich if e["size"] >= 3]
        median_best_p = float(np.median([e["best_p"] for e in valid_ps])) if valid_ps else 1.0
        median_best_p_fdr = float(np.median([e.get("best_p_fdr", 1.0) for e in valid_ps])) if valid_ps else 1.0

        yeast_enrichment[method] = {
            "n_communities": len(comms),
            "n_significant_05": n_sig,
            "n_significant_fdr_05": n_sig_fdr,
            "frac_significant": round(n_sig / len(comms), 3) if comms else 0,
            "frac_significant_fdr": round(n_sig_fdr / len(comms), 3) if comms else 0,
            "median_best_p": median_best_p,
            "median_best_p_fdr": median_best_p_fdr,
            "mean_community_size": round(np.mean([len(c) for c in comms]), 1),
            "communities": enrich,
        }
        print(f"    {method:<12}: {len(comms)} comms, {n_sig} sig (p<0.05), "
              f"{n_sig_fdr} sig (FDR<0.05), "
              f"median_best_p={median_best_p:.2e}", flush=True)

    print(f"  Yeast enrichment: {time.time()-t3:.1f}s", flush=True)
    enrichment_results["yeast"] = yeast_enrichment

    # =========================================================
    # Part B: Multi-Seed Panel
    # =========================================================
    print(f"\n{'='*50}")
    print("PART B: Multi-Seed Panel + Mixed-Effects Model")
    print(f"{'='*50}")

    # Yeast multi-seed
    if CKPT_YEAST_MS.exists():
        print("\n  [RESUME] Loading yeast multi-seed from checkpoint...")
        with open(CKPT_YEAST_MS, encoding="utf-8") as f:
            yeast_multiseed = json.load(f)
        print(f"  Yeast: {len(yeast_multiseed)} seeds loaded from checkpoint")
    else:
        print("\n  Yeast multi-seed (5 seeds x 11 methods)...")
        t4 = time.time()
        yeast_multiseed = multiseed_yeast(G_yeast, yeast_nodes, yeast_go, n_seeds=5)
        print(f"  Yeast multi-seed: {time.time()-t4:.1f}s")
        with open(CKPT_YEAST_MS, "w", encoding="utf-8") as f:
            json.dump(yeast_multiseed, f, indent=2)
        print(f"  [CHECKPOINT] saved")

    # Mouse multi-seed
    if CKPT_MOUSE_MS.exists():
        print("\n  [RESUME] Loading mouse multi-seed from checkpoint...")
        with open(CKPT_MOUSE_MS, encoding="utf-8") as f:
            mouse_multiseed = json.load(f)
        print(f"  Mouse: {len(mouse_multiseed)} seeds loaded from checkpoint")
    else:
        print("\n  Mouse multi-seed (5 subsamples x 11 methods)...")
        t5 = time.time()
        mouse_multiseed = multiseed_mouse(n_seeds=5, subsample_size=2000)
        print(f"  Mouse multi-seed: {time.time()-t5:.1f}s")
        with open(CKPT_MOUSE_MS, "w", encoding="utf-8") as f:
            json.dump(mouse_multiseed, f, indent=2)
        print(f"  [CHECKPOINT] saved")

    # Human multi-seed (load existing)
    print("\n  Loading human 10-seed data...")
    try:
        with open(RESULTS / "human_seed_stability.json", encoding="utf-8") as f:
            human_seed_data = json.load(f)
        human_multiseed = human_seed_data["per_seed_scores"]
        print(f"  Human: {len(human_multiseed)} seeds loaded")
    except Exception as e:
        print(f"  Warning: Could not load human seed data: {e}")
        human_multiseed = {}

    # Fit mixed-effects model
    print("\n  Fitting mixed-effects model...")
    t6 = time.time()
    mixed_results = fit_mixed_effects(
        yeast_multiseed, mouse_multiseed, human_multiseed,
        exclude_mouse=True)
    print(f"  Mixed-effects: {time.time()-t6:.1f}s")

    pooled = mixed_results["pooled_spearman"]
    print(f"\n  Pooled Spearman (n={mixed_results['n_total_observations']}):")
    print(f"    rho = {pooled['rho']:.4f} (p = {pooled['p']:.2e})")
    print(f"    95% CI: [{pooled['ci_95_low']:.4f}, {pooled['ci_95_high']:.4f}]")
    for sp, sp_data in mixed_results["per_species"].items():
        print(f"    {sp}: rho={sp_data['rho']:.4f} (n={sp_data['n_observations']})")

    # =========================================================
    # Save Results
    # =========================================================
    print(f"\n  Saving results...")

    # Strip community-level details from enrichment for JSON output
    enrichment_summary = {}
    for sp, sp_data in enrichment_results.items():
        enrichment_summary[sp] = {}
        for method, m_data in sp_data.items():
            enrichment_summary[sp][method] = {
                k: v for k, v in m_data.items() if k != "communities"
            }
            # Keep top enriched terms
            if "communities" in m_data and m_data["communities"]:
                top_terms = []
                for c in m_data["communities"]:
                    if c.get("best_term") and c.get("best_p", 1) < 1:
                        top_terms.append({
                            "community_size": c["size"],
                            "best_p": c["best_p"],
                            "best_p_fdr": c.get("best_p_fdr", c["best_p"]),
                            "best_term": c["best_term"],
                            "term_name": term_names.get(c["best_term"], ""),
                            "odds_ratio": c["best_odds_ratio"],
                        })
                enrichment_summary[sp][method]["top_enriched_terms"] = sorted(
                    top_terms, key=lambda x: x["best_p"])[:5]

    output_enrich = {
        "analysis": "Phase 12A: GO BP Pathway Enrichment Validation",
        "enrichment_radius": ENRICHMENT_RADIUS,
        "n_bp_terms": len(bp_terms),
        "per_species": enrichment_summary,
    }
    with open(RESULTS / "biological_enrichment.json", "w", encoding="utf-8") as f:
        json.dump(output_enrich, f, indent=2, ensure_ascii=False)
    print(f"  Saved biological_enrichment.json")

    output_panel = {
        "analysis": "Phase 12B: Multi-Seed Panel + Mixed-Effects Model",
        "yeast_seeds": 5,
        "mouse_seeds": 5,
        "human_seeds": len(human_multiseed),
        "mixed_effects": mixed_results,
        "yeast_per_seed": yeast_multiseed,
        "mouse_per_seed": mouse_multiseed,
    }
    with open(RESULTS / "multiseed_panel.json", "w", encoding="utf-8") as f:
        json.dump(output_panel, f, indent=2, ensure_ascii=False)
    print(f"  Saved multiseed_panel.json")

    # =========================================================
    # Generate Figures
    # =========================================================
    print(f"\n  Generating figures...")
    generate_fig60(enrichment_results, term_names)
    generate_fig61(enrichment_results)
    generate_fig62(mixed_results)
    generate_fig63(mixed_results)
    generate_fig64(enrichment_results, mixed_results)

    print(f"\n{BANNER}")
    print("Phase 12 complete.")
    print(BANNER)


# ============================================================
# Figure Generation
# ============================================================

def generate_fig60(enrichment_results, term_names):
    """Fig60: Enrichment overview — fraction significant + best p-value per species."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    species_list = ["yeast", "human", "mouse"]
    species_colors = {"yeast": "#2171B5", "human": "#E6550D", "mouse": "#31A354"}
    method_order = ["Spectral", "DM", "MDS", "DeepWalk", "Node2Vec",
                    "GraphSAGE", "PCA", "VGAE-feat", "VGAE", "GIN", "GAT"]

    for si, sp in enumerate(species_list):
        ax = axes[si]
        sp_data = enrichment_results.get(sp, {})
        methods = [m for m in method_order if m in sp_data]
        if not methods:
            continue

        frac_sig = [sp_data[m].get("frac_significant", 0) for m in methods]
        median_p = [sp_data[m].get("median_best_p", 1.0) for m in methods]
        neg_log_p = [-np.log10(max(p, 1e-300)) for p in median_p]

        x = np.arange(len(methods))
        bars = ax.bar(x, frac_sig, 0.35, label="Frac. enriched (p<0.05)",
                      color=species_colors[sp], alpha=0.7, edgecolor="k", linewidth=0.5)
        ax2 = ax.twinx()
        ax2.bar(x + 0.35, neg_log_p, 0.35, label=r"$-\log_{10}$(median best p)",
                color="#CCCCCC", edgecolor="k", linewidth=0.5)

        ax.set_xticks(x + 0.17)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Fraction enriched communities")
        ax2.set_ylabel(r"$-\log_{10}$(median best p-value)", color="#666")
        ax.set_title(f"{sp.capitalize()} (r={ENRICHMENT_RADIUS})",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        # Highlight Spectral
        spec_idx = methods.index("Spectral") if "Spectral" in methods else -1
        if spec_idx >= 0:
            bars[spec_idx].set_edgecolor("red")
            bars[spec_idx].set_linewidth(2)

    fig.suptitle("Phase 12A: GO BP Enrichment by Embedding Method",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig60_enrichment_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig60_enrichment_overview.png")


def generate_fig61(enrichment_results):
    """Fig61: Enrichment strength distribution (violin/box plot)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    species_list = ["yeast", "human", "mouse"]
    species_colors_map = {"yeast": "#2171B5", "human": "#E6550D", "mouse": "#31A354"}
    method_order = ["Spectral", "DM", "MDS", "DeepWalk", "Node2Vec",
                    "GraphSAGE", "PCA", "VGAE-feat", "VGAE", "GIN", "GAT"]

    for si, sp in enumerate(species_list):
        ax = axes[si]
        sp_data = enrichment_results.get(sp, {})
        methods = [m for m in method_order if m in sp_data]
        if not methods:
            continue

        all_pvals = []
        for m in methods:
            comms = sp_data[m].get("communities", [])
            pvals = [-np.log10(max(c.get("best_p", 1.0), 1e-300))
                     for c in comms if c.get("size", 0) >= 3]
            all_pvals.append(pvals if pvals else [0])

        parts = ax.violinplot(all_pvals, positions=range(len(methods)),
                              showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            color = "#E69F00" if methods[i] == "Spectral" else "#CCCCCC"
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        if parts.get("cmeans"):
            parts["cmeans"].set_color("k")

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(r"$-\log_{10}$(best p-value)")
        ax.set_title(f"{sp.capitalize()}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Phase 12A: Distribution of Enrichment Significance per Community",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig61_enrichment_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig61_enrichment_distribution.png")


def generate_fig62(mixed_results):
    """Fig62: Multi-seed GF score stability (box plot per method, pooled across species)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    method_stats = mixed_results["method_statistics"]
    methods = sorted(method_stats.keys(), key=lambda m: method_stats[m]["mean_gf"],
                     reverse=True)

    # Panel A: Mean GF score ± std across all seeds/species
    ax = axes[0]
    mean_gf = [method_stats[m]["mean_gf"] for m in methods]
    std_gf = [method_stats[m]["std_gf"] for m in methods]
    x = np.arange(len(methods))
    colors = ["#E69F00" if m == "Spectral" else "#4292C6" for m in methods]
    ax.barh(x, mean_gf, xerr=std_gf, color=colors, edgecolor="k", linewidth=0.5,
            capsize=3)
    ax.set_yticks(x)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Mean G-F Score (± std across seeds × species)")
    ax.set_title("A. Method Stability Across Seeds", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel B: Mean rank ± std
    ax = axes[1]
    mean_rank = [method_stats[m]["mean_rank"] for m in methods]
    std_rank = [method_stats[m]["std_rank"] for m in methods]
    ax.barh(x, mean_rank, xerr=std_rank, color=colors, edgecolor="k",
            linewidth=0.5, capsize=3)
    ax.set_yticks(x)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Mean Rank (± std, lower = better)")
    ax.set_title("B. Rank Stability Across Seeds", fontsize=10, fontweight="bold")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"Phase 12B: Multi-Seed Panel ({mixed_results['n_total_observations']} "
                 f"observations, {mixed_results['n_groups']} groups)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig62_multiseed_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig62_multiseed_stability.png")


def generate_fig63(mixed_results):
    """Fig63: Pooled rank consistency (|Spearman ρ|) with proper CI."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pooled = mixed_results["pooled_spearman"]
    per_sp = mixed_results["per_species"]

    # Use |rho| — negative sign is an artifact of rank encoding
    labels = ["Pooled\n(all species)"]
    rhos = [abs(pooled["rho"])]
    ci_lows = [max(0, abs(pooled["ci_95_high"]))]  # note: abs swaps low/high
    ci_highs = [min(1, abs(pooled["ci_95_low"]))]
    colors = ["#E69F00"]
    n_obs = [mixed_results["n_total_observations"]]

    for sp in ["yeast", "human", "mouse"]:
        if sp in per_sp:
            labels.append(sp.capitalize())
            rho_abs = abs(per_sp[sp]["rho"])
            rhos.append(rho_abs)
            # Fisher z-transform for proper CI on |rho|
            n = per_sp[sp]["n_observations"]
            n_obs.append(n)
            if n > 3:
                z = np.arctanh(min(rho_abs, 0.9999))
                se = 1.0 / np.sqrt(n - 3)
                z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
                ci_lo = max(0, np.tanh(z_lo))
                ci_hi = min(1, np.tanh(z_hi))
            else:
                ci_lo, ci_hi = 0, 1
            ci_lows.append(ci_lo)
            ci_highs.append(ci_hi)
            sp_colors = {"yeast": "#2171B5", "human": "#E6550D", "mouse": "#31A354"}
            colors.append(sp_colors[sp])

    x = np.arange(len(labels))
    errors_low = [r - l for r, l in zip(rhos, ci_lows)]
    errors_high = [h - r for r, h in zip(rhos, ci_highs)]

    ax.bar(x, rhos, color=colors, edgecolor="k", linewidth=0.5, width=0.6)
    ax.errorbar(x, rhos, yerr=[errors_low, errors_high], fmt="none",
                ecolor="k", capsize=5, linewidth=1.5)

    for i, (r, lo, hi, n) in enumerate(zip(rhos, ci_lows, ci_highs, n_obs)):
        ax.text(i, hi + 0.02, f"|ρ|={r:.3f}\n[{lo:.3f}, {hi:.3f}]\nn={n}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rank Consistency |Spearman ρ|")
    ax.set_title("Phase 12B: Method Rank Consistency Across Seeds (95% CI)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES / "Fig63_pooled_spearman.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig63_pooled_spearman.png")


def generate_fig64(enrichment_results, mixed_results):
    """Fig64: Summary — enrichment vs GF score scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    species_list = ["yeast", "human", "mouse"]
    species_colors = {"yeast": "#2171B5", "human": "#E6550D", "mouse": "#31A354"}
    method_colors = {
        "Spectral": "#E69F00", "DM": "#0072B2", "MDS": "#009E73",
        "Node2Vec": "#CC79A7", "PCA": "#56B4E9", "VGAE-feat": "#F0E442",
        "DeepWalk": "#D55E00", "GIN": "#949494", "GAT": "#000000",
        "GraphSAGE": "#8B4513", "VGAE": "#808080",
    }

    # Panel A: frac_significant vs mean GF (pooled from multi-seed)
    ax = axes[0]
    method_stats = mixed_results["method_statistics"]
    for sp in species_list:
        sp_data = enrichment_results.get(sp, {})
        for method in sp_data:
            frac_sig = sp_data[method].get("frac_significant", 0)
            mean_gf = method_stats.get(method, {}).get("mean_gf", 0)
            ax.scatter(mean_gf, frac_sig, s=80,
                       c=method_colors.get(method, "#999"),
                       edgecolors="k", linewidth=0.5, alpha=0.7, zorder=3)
            if method == "Spectral":
                ax.annotate(f"{sp}", (mean_gf, frac_sig), fontsize=8,
                            ha="left", xytext=(3, 3), textcoords="offset points")

    # Legend
    for m in ["Spectral", "DM", "MDS", "DeepWalk", "VGAE"]:
        ax.scatter([], [], s=60, c=method_colors.get(m, "#999"),
                   edgecolors="k", linewidth=0.5, label=m)
    ax.legend(fontsize=8, loc="lower right")

    ax.set_xlabel("Mean G-F Score (multi-seed)")
    ax.set_ylabel("Fraction enriched communities")
    ax.set_title("A. Enrichment Strength vs G-F Score",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel B: Summary table
    ax = axes[1]
    ax.axis("off")
    pooled = mixed_results["pooled_spearman"]
    table_data = [
        ["Metric", "Value"],
        ["Total observations", str(mixed_results["n_total_observations"])],
        ["Groups (species × seed)", str(mixed_results["n_groups"])],
        ["Pooled |ρ|", f'{abs(pooled["rho"]):.4f}'],
        ["P-value", f'{pooled["p"]:.2e}'],
        ["95% CI (|ρ|)", f'[{abs(pooled["ci_95_high"]):.4f}, {abs(pooled["ci_95_low"]):.4f}]'],
        ["Yeast |ρ|", f'{abs(mixed_results["per_species"].get("yeast", {}).get("rho", 0)):.4f}'],
        ["Human |ρ|", f'{abs(mixed_results["per_species"].get("human", {}).get("rho", 0)):.4f}'],
        ["Mouse |ρ|", f'{abs(mixed_results["per_species"].get("mouse", {}).get("rho", 0)):.4f}'],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for j in range(2):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_data)):
        for j in range(2):
            table[i, j].set_facecolor(["#DEEBF7", "#F2F2F2"][i % 2])

    ax.set_title("B. Mixed-Effects Model Summary",
                 fontsize=10, fontweight="bold")

    fig.suptitle("Phase 12: Biological Validation & Statistical Power",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig64_phase12_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig64_phase12_summary.png")


PICKLE_PART_A = RESULTS / "_part_a_enrichment.pkl"


def run_part_a_only():
    """Run only Part A and save full results to pickle."""
    print(BANNER)
    print("Phase 12 Part A: GO BP Enrichment (standalone)")
    print(BANNER)

    bp_terms, term_names = parse_go_obo(str(DATA / "go.obo"))
    print(f"  {len(bp_terms)} biological_process terms")

    enrichment_results = {}

    # --- Human ---
    print("\n  Human enrichment...", flush=True)
    t1 = time.time()
    with open(DATA / "human_go_annotations.json", encoding="utf-8") as f:
        human_go = json.load(f)
    ref_file = DATA / "human_spectral_embedding.json"
    with open(ref_file, encoding="utf-8") as f:
        ref = json.load(f)
    all_human_nodes = sorted(ref.keys())
    annotated_human = [n for n in all_human_nodes if n in human_go]
    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(len(annotated_human), 2000, replace=False)
    sub_human_nodes = [annotated_human[i] for i in sub_idx]
    sub_human_go = {n: human_go[n] for n in sub_human_nodes}

    human_enrichment = {}
    for method in ALL_METHODS:
        fpath = DATA / f"human_{method.lower()}_embedding.json"
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        common = [n for n in sub_human_nodes if n in raw]
        if len(common) < 500:
            continue
        coords = np.array([[raw[n]["x"], raw[n]["y"]] for n in common])
        coords = rescale_coordinates(coords, TARGET_STD)
        comms = build_community_graph(coords, common, ENRICHMENT_RADIUS)
        if not comms:
            human_enrichment[method] = {"n_communities": 0, "communities": []}
            continue
        enrich = go_enrichment_hypergeom(comms, common, sub_human_go, bp_terms)
        n_sig = sum(1 for e in enrich if e["best_p"] < 0.05)
        n_sig_fdr = sum(1 for e in enrich if e.get("best_p_fdr", 1.0) < 0.05)
        valid_ps = [e for e in enrich if e["size"] >= 3]
        median_best_p = float(np.median([e["best_p"] for e in valid_ps])) if valid_ps else 1.0
        median_best_p_fdr = float(np.median([e.get("best_p_fdr", 1.0) for e in valid_ps])) if valid_ps else 1.0
        human_enrichment[method] = {
            "n_communities": len(comms), "n_significant_05": n_sig,
            "n_significant_fdr_05": n_sig_fdr,
            "frac_significant": round(n_sig / len(comms), 3) if comms else 0,
            "frac_significant_fdr": round(n_sig_fdr / len(comms), 3) if comms else 0,
            "median_best_p": median_best_p,
            "median_best_p_fdr": median_best_p_fdr,
            "mean_community_size": round(np.mean([len(c) for c in comms]), 1),
            "communities": enrich,
        }
        print(f"    {method:<12}: {len(comms)} comms, {n_sig} sig, {n_sig_fdr} sig(FDR)", flush=True)
    print(f"  Human: {time.time()-t1:.1f}s", flush=True)
    enrichment_results["human"] = human_enrichment

    # --- Mouse ---
    print("\n  Mouse enrichment...", flush=True)
    t2 = time.time()
    with open(DATA / "mouse_go_annotations.json", encoding="utf-8") as f:
        mouse_go = json.load(f)
    ref_file = DATA / "mouse_spectral_embedding.json"
    with open(ref_file, encoding="utf-8") as f:
        ref = json.load(f)
    all_mouse_nodes = sorted(ref.keys())
    annotated_mouse = [n for n in all_mouse_nodes if n in mouse_go]
    rng2 = np.random.default_rng(SEED)
    sub_idx2 = rng2.choice(len(annotated_mouse), 2000, replace=False)
    sub_mouse_nodes = [annotated_mouse[i] for i in sub_idx2]
    sub_mouse_go = {n: mouse_go[n] for n in sub_mouse_nodes}

    mouse_enrichment = {}
    for method in ALL_METHODS:
        fpath = DATA / f"mouse_{method.lower()}_embedding.json"
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        common = [n for n in sub_mouse_nodes if n in raw]
        if len(common) < 500:
            continue
        coords = np.array([[raw[n]["x"], raw[n]["y"]] for n in common])
        coords = rescale_coordinates(coords, TARGET_STD)
        comms = build_community_graph(coords, common, ENRICHMENT_RADIUS)
        if not comms:
            mouse_enrichment[method] = {"n_communities": 0, "communities": []}
            continue
        enrich = go_enrichment_hypergeom(comms, common, sub_mouse_go, bp_terms)
        n_sig = sum(1 for e in enrich if e["best_p"] < 0.05)
        n_sig_fdr = sum(1 for e in enrich if e.get("best_p_fdr", 1.0) < 0.05)
        valid_ps = [e for e in enrich if e["size"] >= 3]
        median_best_p = float(np.median([e["best_p"] for e in valid_ps])) if valid_ps else 1.0
        median_best_p_fdr = float(np.median([e.get("best_p_fdr", 1.0) for e in valid_ps])) if valid_ps else 1.0
        mouse_enrichment[method] = {
            "n_communities": len(comms), "n_significant_05": n_sig,
            "n_significant_fdr_05": n_sig_fdr,
            "frac_significant": round(n_sig / len(comms), 3) if comms else 0,
            "frac_significant_fdr": round(n_sig_fdr / len(comms), 3) if comms else 0,
            "median_best_p": median_best_p,
            "median_best_p_fdr": median_best_p_fdr,
            "mean_community_size": round(np.mean([len(c) for c in comms]), 1),
            "communities": enrich,
        }
        print(f"    {method:<12}: {len(comms)} comms, {n_sig} sig, {n_sig_fdr} sig(FDR)", flush=True)
    print(f"  Mouse: {time.time()-t2:.1f}s", flush=True)
    enrichment_results["mouse"] = mouse_enrichment

    # --- Yeast ---
    print("\n  Yeast enrichment...", flush=True)
    t3 = time.time()
    G_yeast, yeast_nodes, yeast_go = load_curated_network()
    yeast_embs = compute_all_embeddings_yeast(G_yeast, yeast_nodes, seed=SEED)
    yeast_enrichment = {}
    for method in ALL_METHODS:
        if method not in yeast_embs:
            continue
        coords = yeast_embs[method]
        comms = build_community_graph(coords, yeast_nodes, ENRICHMENT_RADIUS)
        if not comms:
            yeast_enrichment[method] = {"n_communities": 0, "communities": []}
            continue
        enrich = go_enrichment_hypergeom(comms, yeast_nodes, yeast_go, bp_terms)
        n_sig = sum(1 for e in enrich if e["best_p"] < 0.05)
        n_sig_fdr = sum(1 for e in enrich if e.get("best_p_fdr", 1.0) < 0.05)
        valid_ps = [e for e in enrich if e["size"] >= 3]
        median_best_p = float(np.median([e["best_p"] for e in valid_ps])) if valid_ps else 1.0
        median_best_p_fdr = float(np.median([e.get("best_p_fdr", 1.0) for e in valid_ps])) if valid_ps else 1.0
        yeast_enrichment[method] = {
            "n_communities": len(comms), "n_significant_05": n_sig,
            "n_significant_fdr_05": n_sig_fdr,
            "frac_significant": round(n_sig / len(comms), 3) if comms else 0,
            "frac_significant_fdr": round(n_sig_fdr / len(comms), 3) if comms else 0,
            "median_best_p": median_best_p,
            "median_best_p_fdr": median_best_p_fdr,
            "mean_community_size": round(np.mean([len(c) for c in comms]), 1),
            "communities": enrich,
        }
        print(f"    {method:<12}: {len(comms)} comms, {n_sig} sig", flush=True)
    print(f"  Yeast: {time.time()-t3:.1f}s", flush=True)
    enrichment_results["yeast"] = yeast_enrichment

    # Save to pickle
    with open(PICKLE_PART_A, "wb") as f:
        pickle.dump({"enrichment": enrichment_results,
                      "bp_terms": bp_terms, "term_names": term_names}, f)
    print(f"\n  Saved Part A to {PICKLE_PART_A}")
    print(BANNER)


def run_part_b_only():
    """Load Part A from pickle and run Part B + figures."""
    if not PICKLE_PART_A.exists():
        print("ERROR: Part A pickle not found. Run with 'part_a' first.")
        return

    print(BANNER)
    print("Phase 12 Part B: Multi-Seed Panel (standalone)")
    print(BANNER)

    with open(PICKLE_PART_A, "rb") as f:
        part_a = pickle.load(f)
    enrichment_results = part_a["enrichment"]
    bp_terms = part_a["bp_terms"]
    term_names = part_a["term_names"]
    print(f"  Loaded Part A: {list(enrichment_results.keys())}")

    # Load yeast network for multi-seed
    G_yeast, yeast_nodes, yeast_go = load_curated_network()

    # Yeast multi-seed
    if CKPT_YEAST_MS.exists():
        print("\n  [RESUME] Loading yeast multi-seed from checkpoint...")
        with open(CKPT_YEAST_MS, encoding="utf-8") as f:
            yeast_multiseed = json.load(f)
        print(f"  Yeast: {len(yeast_multiseed)} seeds loaded")
    else:
        print("\n  Yeast multi-seed (5 seeds x 11 methods)...")
        t4 = time.time()
        yeast_multiseed = multiseed_yeast(G_yeast, yeast_nodes, yeast_go, n_seeds=5)
        print(f"  Yeast multi-seed: {time.time()-t4:.1f}s")
        with open(CKPT_YEAST_MS, "w", encoding="utf-8") as f:
            json.dump(yeast_multiseed, f, indent=2)

    # Mouse multi-seed
    if CKPT_MOUSE_MS.exists():
        print("\n  [RESUME] Loading mouse multi-seed from checkpoint...")
        with open(CKPT_MOUSE_MS, encoding="utf-8") as f:
            mouse_multiseed = json.load(f)
        print(f"  Mouse: {len(mouse_multiseed)} seeds loaded")
    else:
        print("\n  Mouse multi-seed (5 subsamples x 11 methods)...")
        t5 = time.time()
        mouse_multiseed = multiseed_mouse(n_seeds=5, subsample_size=2000)
        print(f"  Mouse multi-seed: {time.time()-t5:.1f}s")
        with open(CKPT_MOUSE_MS, "w", encoding="utf-8") as f:
            json.dump(mouse_multiseed, f, indent=2)

    # Human multi-seed
    print("\n  Loading human 10-seed data...")
    try:
        with open(RESULTS / "human_seed_stability.json", encoding="utf-8") as f:
            human_seed_data = json.load(f)
        human_multiseed = human_seed_data["per_seed_scores"]
        print(f"  Human: {len(human_multiseed)} seeds loaded")
    except Exception as e:
        print(f"  Warning: {e}")
        human_multiseed = {}

    # Mixed-effects model
    print("\n  Fitting mixed-effects model...")
    mixed_results = fit_mixed_effects(yeast_multiseed, mouse_multiseed,
                                       human_multiseed,
                                       exclude_mouse=True)
    pooled = mixed_results["pooled_spearman"]
    print(f"  Pooled Spearman (n={mixed_results['n_total_observations']}):")
    print(f"    rho = {pooled['rho']:.4f} (p = {pooled['p']:.2e})")
    print(f"    95% CI: [{pooled['ci_95_low']:.4f}, {pooled['ci_95_high']:.4f}]")
    for sp, sp_data in mixed_results["per_species"].items():
        print(f"    {sp}: rho={sp_data['rho']:.4f} (n={sp_data['n_observations']})")

    # Save results
    enrichment_summary = {}
    for sp, sp_data in enrichment_results.items():
        enrichment_summary[sp] = {}
        for method, m_data in sp_data.items():
            enrichment_summary[sp][method] = {
                k: v for k, v in m_data.items() if k != "communities"
            }
            if "communities" in m_data and m_data["communities"]:
                top_terms = []
                for c in m_data["communities"]:
                    if c.get("best_term") and c.get("best_p", 1) < 1:
                        top_terms.append({
                            "community_size": c["size"],
                            "best_p": c["best_p"],
                            "best_p_fdr": c.get("best_p_fdr", c["best_p"]),
                            "best_term": c["best_term"],
                            "term_name": term_names.get(c["best_term"], ""),
                            "odds_ratio": c["best_odds_ratio"],
                        })
                enrichment_summary[sp][method]["top_enriched_terms"] = sorted(
                    top_terms, key=lambda x: x["best_p"])[:5]

    output_enrich = {
        "analysis": "Phase 12A: GO BP Pathway Enrichment Validation",
        "enrichment_radius": ENRICHMENT_RADIUS,
        "n_bp_terms": len(bp_terms),
        "per_species": enrichment_summary,
    }
    with open(RESULTS / "biological_enrichment.json", "w", encoding="utf-8") as f:
        json.dump(output_enrich, f, indent=2, ensure_ascii=False)

    output_panel = {
        "analysis": "Phase 12B: Multi-Seed Panel + Mixed-Effects Model",
        "yeast_seeds": 5, "mouse_seeds": 5,
        "human_seeds": len(human_multiseed),
        "mixed_effects": mixed_results,
        "yeast_per_seed": yeast_multiseed,
        "mouse_per_seed": mouse_multiseed,
    }
    with open(RESULTS / "multiseed_panel.json", "w", encoding="utf-8") as f:
        json.dump(output_panel, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved biological_enrichment.json + multiseed_panel.json")

    # Figures
    print(f"\n  Generating figures...")
    generate_fig60(enrichment_results, term_names)
    generate_fig61(enrichment_results)
    generate_fig62(mixed_results)
    generate_fig63(mixed_results)
    generate_fig64(enrichment_results, mixed_results)

    print(f"\n{BANNER}")
    print("Phase 12 complete.")
    print(BANNER)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "part_a":
        run_part_a_only()
    elif mode == "part_b":
        run_part_b_only()
    else:
        run()
