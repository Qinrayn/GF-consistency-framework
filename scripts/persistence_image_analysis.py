#!/usr/bin/env python3
"""
Phase 10C: Persistence Image TDA + Cross-Species Analysis
============================================================
Three-species (yeast/human/mouse) G-F consistency analysis with
persistence image TDA features.

Part 1 — Mouse G-F Analysis:
  Compute G-F scores for all 11 methods on mouse PPI using unified
  parameters (greedy_modularity, yeast interval [0.05, 0.422]).

Part 2 — Persistence Images:
  Recompute H1 persistence diagrams for human + mouse embeddings.
  Generate persistence images and extract features (total_energy,
  max_density, spread, entropy). Test as GF predictors.

Part 3 — Cross-Species Comparison:
  Two-factor model (spectral alignment + effective rank) on mouse.
  Three-species rank concordance (Kendall's W).
  Spectral H1 anomaly check at mouse scale.

Depends on: data/mouse_{method}_embedding.json
            data/mouse_go_annotations.json
            data/human_{method}_embedding.json
            data/human_go_annotations.json
Generates:  results/mouse_gf_analysis.json
            results/persistence_image_analysis.json
            results/cross_species_three_way.json
            results/phase10_report.md
            figures/Fig51-54
"""

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, rankdata
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from ripser import ripser
from persim import PersistenceImager

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    ALL_METHODS, SEED, TARGET_STD,
    R_MIN, R_MAX, GF_R_MIN, GF_R_MAX,
    get_data_dir, get_results_dir, get_figures_dir,
    rescale_coordinates,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()

SUBSAMPLE = 2000
N_POINTS = 25
MAX_EDGES = 150_000
BANNER = "=" * 70


# ============================================================
# Data Loading (unified for human + mouse)
# ============================================================

def load_go(species):
    fname = f"{species}_go_annotations.json"
    with open(DATA / fname, encoding="utf-8") as f:
        return json.load(f)


def load_embedding(species, method, go_map):
    fname = f"{species}_{method.lower()}_embedding.json"
    fpath = DATA / fname
    if not fpath.exists():
        return None, None
    with open(fpath, encoding="utf-8") as f:
        raw = json.load(f)
    if not raw:
        return None, None
    first_val = next(iter(raw.values()))
    if isinstance(first_val, dict):
        nodes = sorted(raw.keys())
        coords = np.array([[raw[n]["x"], raw[n]["y"]] for n in nodes])
    elif isinstance(first_val, list):
        nodes = sorted(raw.keys())
        coords = np.array([raw[n] for n in nodes])
    else:
        return None, None
    annotated = set(go_map.keys())
    mask = [n in annotated for n in nodes]
    coords = coords[mask]
    nodes = [n for n, m in zip(nodes, mask) if m]
    return coords, nodes


def get_common_subsample(embeddings_dict, n=SUBSAMPLE, seed=SEED):
    rng = np.random.default_rng(seed)
    common = None
    for method, (coords, nodes) in embeddings_dict.items():
        s = set(nodes)
        common = s if common is None else common & s
    common = sorted(common)
    if len(common) <= n:
        return common
    return sorted(rng.choice(common, n, replace=False))


def extract_subsample(coords, nodes, subsample_nodes):
    node_set = set(subsample_nodes)
    idx = [i for i, n in enumerate(nodes) if n in node_set]
    return coords[idx], [nodes[i] for i in idx]


# ============================================================
# G-F Curve Computation (greedy_modularity)
# ============================================================

def compute_gf_curve(coords, nodes, go_map, r_vals):
    """Compute G-F purity curve using greedy_modularity_communities."""
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

        if n_edges > MAX_EDGES:
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
                terms = go_map.get(nodes[i], [])
                all_terms.extend(terms)
            if not all_terms:
                continue
            counts = Counter(all_terms)
            comm_purities.append(counts.most_common(1)[0][1] / len(all_terms))

        if comm_purities:
            purities[ri] = float(np.mean(comm_purities))

    return purities


def compute_gf_score(purities, r_vals, r_min, r_max):
    mask = (r_vals >= r_min) & (r_vals <= r_max)
    if mask.sum() < 2:
        return 0.0
    r_sub = r_vals[mask]
    p_sub = purities[mask]
    return float(trapezoid(p_sub, r_sub) / (r_sub[-1] - r_sub[0]))


# ============================================================
# Persistence Diagrams + Images
# ============================================================

def compute_persistence_diagrams(coords):
    """Compute H0 and H1 persistence diagrams via ripser."""
    result = ripser(coords, maxdim=1, do_cocycles=False)
    diagrams = {}
    for dim in range(2):
        dgm = result["dgms"][dim]
        # Clip infinite deaths
        max_val = np.max(np.ptp(coords, axis=0)) * 1.5
        dgm[:, 1] = np.clip(dgm[:, 1], 0, max_val)
        # Filter zero-persistence features
        pers = dgm[:, 1] - dgm[:, 0]
        mask = pers > 1e-10
        diagrams[dim] = dgm[mask]
    return diagrams


def compute_persistence_image(diagram, birth_range=(0, 0.3),
                              pers_range=(0, 0.3), pixel_size=0.01):
    """Convert persistence diagram to persistence image."""
    if len(diagram) == 0:
        n_birth = int((birth_range[1] - birth_range[0]) / pixel_size) + 1
        n_pers = int((pers_range[1] - pers_range[0]) / pixel_size) + 1
        return np.zeros((n_pers, n_birth))

    pi = PersistenceImager(
        birth_range=birth_range,
        pers_range=pers_range,
        pixel_size=pixel_size,
    )
    return pi.fit_transform(diagram)


def extract_pi_features(img):
    """Extract scalar features from persistence image."""
    total_energy = float(np.sum(img))
    max_density = float(np.max(img)) if img.size > 0 else 0.0

    # Spread: spatial std of density-weighted coordinates
    if total_energy > 1e-12:
        rows, cols = np.indices(img.shape)
        row_center = float(np.sum(rows * img) / total_energy)
        col_center = float(np.sum(cols * img) / total_energy)
        row_var = float(np.sum(((rows - row_center) ** 2) * img) / total_energy)
        col_var = float(np.sum(((cols - col_center) ** 2) * img) / total_energy)
        spread = float(np.sqrt(row_var + col_var))
    else:
        spread = 0.0

    # Entropy: Shannon entropy of normalized image
    img_flat = img.flatten()
    img_flat = img_flat[img_flat > 0]
    if len(img_flat) > 1:
        p = img_flat / img_flat.sum()
        entropy = float(-np.sum(p * np.log(p)))
    else:
        entropy = 0.0

    return {
        "total_energy": total_energy,
        "max_density": max_density,
        "spread": spread,
        "entropy": entropy,
    }


# ============================================================
# Spectral Alignment + Effective Rank
# ============================================================

def compute_spectral_alignment(coords, nodes):
    """Compute spectral alignment: correlation of embedding with Laplacian eigenvectors."""
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix

    D = squareform(pdist(coords))
    n = len(nodes)

    # Build k-NN graph (k=10)
    k = min(10, n - 1)
    adj = np.zeros((n, n))
    for i in range(n):
        dists = D[i].copy()
        dists[i] = np.inf
        nn_idx = np.argsort(dists)[:k]
        for j in nn_idx:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # Normalized Laplacian
    degree = np.sum(adj, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
    L_norm = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt

    # Compute bottom eigenvectors
    try:
        n_eig = min(20, n - 2)
        eigvals, eigvecs = eigsh(csr_matrix(L_norm), k=n_eig, which="SM")
    except Exception:
        eigvals, eigvecs = np.linalg.eigh(L_norm)
        eigvals = eigvals[:20]
        eigvecs = eigvecs[:, :20]

    # Project embedding coordinates onto eigenvectors
    proj_x = eigvecs.T @ coords[:, 0]
    proj_y = eigvecs.T @ coords[:, 1]
    alignment = np.sqrt(proj_x ** 2 + proj_y ** 2)
    alignment /= max(np.sum(alignment), 1e-10)

    return float(np.sum(alignment[:5]))  # top-5 eigenvector alignment


def compute_effective_rank(coords):
    """Compute SVD-based effective rank of embedding coordinates."""
    centered = coords - coords.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    s = s[s > 1e-10]
    if len(s) < 2:
        return 1.0
    p = s ** 2 / np.sum(s ** 2)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


# ============================================================
# Main
# ============================================================

def run():
    print(BANNER)
    print("Phase 10C: Persistence Image TDA + Cross-Species Analysis")
    print(BANNER)

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # =========================================================
    # PART 1: Mouse G-F Scores
    # =========================================================
    print("\n" + "=" * 50)
    print("PART 1: Mouse G-F Scores (greedy_modularity)")
    print("=" * 50)

    mouse_go = load_go("mouse")
    print(f"  Mouse GO: {len(mouse_go)} genes")

    mouse_emb = {}
    for m in ALL_METHODS:
        c, n = load_embedding("mouse", m, mouse_go)
        if c is not None:
            mouse_emb[m] = (c, n)
            print(f"    {m}: {len(n)} annotated nodes")

    mouse_sub = get_common_subsample(mouse_emb)
    print(f"  Common subsample: {len(mouse_sub)} nodes")

    mouse_sub_emb = {}
    for m, (c, n) in mouse_emb.items():
        c_s, n_s = extract_subsample(c, n, mouse_sub)
        c_s = rescale_coordinates(c_s, TARGET_STD)
        mouse_sub_emb[m] = (c_s, n_s)

    mouse_results = {}
    print(f"\n  Computing G-F curves ({len(mouse_sub_emb)} methods)...")
    for i, m in enumerate(ALL_METHODS):
        if m not in mouse_sub_emb:
            continue
        coords, nodes = mouse_sub_emb[m]
        t0 = time.time()
        purities = compute_gf_curve(coords, nodes, mouse_go, r_vals)
        print()
        gf = compute_gf_score(purities, r_vals, GF_R_MIN, GF_R_MAX)
        mouse_results[m] = {
            "purities": purities.tolist(),
            "gf_score": round(gf, 6),
            "peak_purity": round(float(max(purities)), 4),
        }
        print(f"    [{i+1}/{len(mouse_sub_emb)}] {m:<12}: GF={gf:.4f}, "
              f"peak={max(purities):.4f}, time={time.time()-t0:.1f}s", flush=True)

    # =========================================================
    # PART 2: Persistence Diagrams + Images (Human + Mouse)
    # =========================================================
    print("\n" + "=" * 50)
    print("PART 2: Persistence Diagrams + Images")
    print("=" * 50)

    # Load human embeddings (reuse Phase 9 subsample)
    human_go = load_go("human")
    human_emb = {}
    for m in ALL_METHODS:
        c, n = load_embedding("human", m, human_go)
        if c is not None:
            human_emb[m] = (c, n)

    human_sub = get_common_subsample(human_emb)
    human_sub_emb = {}
    for m, (c, n) in human_emb.items():
        c_s, n_s = extract_subsample(c, n, human_sub)
        c_s = rescale_coordinates(c_s, TARGET_STD)
        human_sub_emb[m] = (c_s, n_s)

    # Compute PH for both species
    species_data = {"human": human_sub_emb, "mouse": mouse_sub_emb}
    ph_results = {}  # {(species, method): {diagrams, pi_features, h1_stats}}

    for species, sub_emb in species_data.items():
        print(f"\n  Computing persistence diagrams for {species}...")
        for m in ALL_METHODS:
            if m not in sub_emb:
                continue
            coords, nodes = sub_emb[m]
            diagrams = compute_persistence_diagrams(coords)

            # H1 persistence stats
            h1_dgm = diagrams.get(1, np.empty((0, 2)))
            h1_pers = h1_dgm[:, 1] - h1_dgm[:, 0] if len(h1_dgm) > 0 else np.array([0])
            h1_stats = {
                "n_features": int(len(h1_dgm)),
                "max_persistence": float(np.max(h1_pers)) if len(h1_pers) > 0 else 0.0,
                "mean_persistence": float(np.mean(h1_pers)) if len(h1_pers) > 0 else 0.0,
                "total_persistence": float(np.sum(h1_pers)) if len(h1_pers) > 0 else 0.0,
            }

            # Persistence image
            # Determine ranges from data
            if len(h1_dgm) > 0:
                max_birth = max(np.max(h1_dgm[:, 0]) * 1.2, 0.3)
                max_pers = max(np.max(h1_pers) * 1.2, 0.3)
            else:
                max_birth, max_pers = 0.3, 0.3

            img = compute_persistence_image(
                h1_dgm,
                birth_range=(0, max_birth),
                pers_range=(0, max_pers),
                pixel_size=0.01,
            )
            pi_feats = extract_pi_features(img)

            ph_results[(species, m)] = {
                "h1_stats": h1_stats,
                "pi_features": pi_feats,
                "pi_image": img,  # keep in memory for figure generation
            }
            print(f"    {species}/{m:<12}: H1 n={h1_stats['n_features']}, "
                  f"max_pers={h1_stats['max_persistence']:.4f}, "
                  f"PI_energy={pi_feats['total_energy']:.6f}")

    # =========================================================
    # PART 3: Correlation Analysis + Cross-Species
    # =========================================================
    print("\n" + "=" * 50)
    print("PART 3: Correlation Analysis + Cross-Species")
    print("=" * 50)

    # Compute spectral alignment + effective rank for mouse
    print("\n  Computing mouse spectral alignment + effective rank...")
    mouse_features = {}
    for m in ALL_METHODS:
        if m not in mouse_sub_emb:
            continue
        coords, nodes = mouse_sub_emb[m]
        sa = compute_spectral_alignment(coords, nodes)
        er = compute_effective_rank(coords)
        mouse_features[m] = {"spectral_alignment": sa, "effective_rank": er}
        print(f"    {m:<12}: SA={sa:.4f}, ER={er:.4f}")

    # Load human Phase 8 features for comparison
    with open(RESULTS / "human_cross_network_validation.json", encoding="utf-8") as f:
        phase8 = json.load(f)
    human_pm = phase8.get("human_per_method_11", {})

    # Load human GF scores (Phase 9 unified)
    with open(RESULTS / "human_gf_unified.json", encoding="utf-8") as f:
        human_gf_data = json.load(f)

    # Build feature tables
    analysis_results = {}
    for species in ["human", "mouse"]:
        methods_avail = [m for m in ALL_METHODS
                         if m in (mouse_results if species == "mouse" else human_gf_data.get("gf_scores", {}))
                         and m in (mouse_features if species == "mouse" else human_pm)
                         and (species, m) in ph_results]

        if not methods_avail:
            continue

        if species == "human":
            gf_vals = np.array([human_gf_data["gf_scores"][m]["yeast_interval"] for m in methods_avail])
            sa_vals = np.array([human_pm[m]["spectral_alignment"] for m in methods_avail])
            er_vals = np.array([human_pm[m]["effective_rank"] for m in methods_avail])
        else:
            gf_vals = np.array([mouse_results[m]["gf_score"] for m in methods_avail])
            sa_vals = np.array([mouse_features[m]["spectral_alignment"] for m in methods_avail])
            er_vals = np.array([mouse_features[m]["effective_rank"] for m in methods_avail])

        h1_max = np.array([ph_results[(species, m)]["h1_stats"]["max_persistence"] for m in methods_avail])
        pi_energy = np.array([ph_results[(species, m)]["pi_features"]["total_energy"] for m in methods_avail])
        pi_max = np.array([ph_results[(species, m)]["pi_features"]["max_density"] for m in methods_avail])
        pi_spread = np.array([ph_results[(species, m)]["pi_features"]["spread"] for m in methods_avail])
        pi_entropy = np.array([ph_results[(species, m)]["pi_features"]["entropy"] for m in methods_avail])

        print(f"\n  === {species.upper()} ({len(methods_avail)} methods) ===")
        print(f"  GF Score correlations:")
        corr_data = {}
        for name, vals in [
            ("spectral_alignment", sa_vals),
            ("effective_rank", er_vals),
            ("h1_max_persistence", h1_max),
            ("pi_total_energy", pi_energy),
            ("pi_max_density", pi_max),
            ("pi_spread", pi_spread),
            ("pi_entropy", pi_entropy),
        ]:
            if np.std(vals) > 1e-10:
                rho, p = spearmanr(vals, gf_vals)
            else:
                rho, p = float("nan"), float("nan")
            corr_data[name] = {"rho": round(float(rho), 3) if not np.isnan(rho) else None,
                               "p": round(float(p), 4) if not np.isnan(p) else None}
            print(f"    {name:<30}: rho={rho:+.3f} (p={p:.4f})")

        # Two-factor model
        sa_r = rankdata(sa_vals)
        er_r = rankdata(er_vals)
        two_f = 0.5 * sa_r + 0.5 * er_r
        rho_2f, p_2f = spearmanr(two_f, gf_vals)
        corr_data["two_factor"] = {"rho": round(float(rho_2f), 3), "p": round(float(p_2f), 4)}
        print(f"    {'two_factor':<30}: rho={rho_2f:+.3f} (p={p_2f:.4f})")

        # Three-factor with best PI feature
        best_pi_name = "pi_total_energy"
        best_pi_vals = pi_energy
        for nm, vl in [("pi_max_density", pi_max), ("pi_spread", pi_spread), ("pi_entropy", pi_entropy)]:
            if np.std(vl) > 1e-10:
                r, _ = spearmanr(vl, gf_vals)
                if not np.isnan(r) and abs(r) > abs(spearmanr(best_pi_vals, gf_vals)[0]):
                    best_pi_name = nm
                    best_pi_vals = vl

        if np.std(best_pi_vals) > 1e-10:
            pi_r = rankdata(best_pi_vals)
            three_f = (1/3) * sa_r + (1/3) * er_r + (1/3) * pi_r
            rho_3f, p_3f = spearmanr(three_f, gf_vals)
            corr_data["three_factor_pi"] = {
                "rho": round(float(rho_3f), 3),
                "p": round(float(p_3f), 4),
                "best_pi_feature": best_pi_name,
            }
            print(f"    {'three_factor (+'+best_pi_name+')':<30}: rho={rho_3f:+.3f} (p={p_3f:.4f})")

        analysis_results[species] = {
            "methods": methods_avail,
            "correlations": corr_data,
            "gf_scores": {m: float(gf_vals[i]) for i, m in enumerate(methods_avail)},
        }

    # Cross-species rank concordance
    print("\n  Cross-species rank concordance (yeast/human/mouse):")
    # Load yeast GF scores
    yeast_gf = {}
    yeast_file = RESULTS / "final_results_summary.json"
    if yeast_file.exists():
        with open(yeast_file, encoding="utf-8") as f:
            yd = json.load(f)
        gs = yd.get("gf_scores", {})
        for m in ALL_METHODS:
            if m in gs:
                yeast_gf[m] = gs[m]

    common_methods = sorted(
        set(yeast_gf.keys()) &
        set(analysis_results.get("human", {}).get("gf_scores", {}).keys()) &
        set(analysis_results.get("mouse", {}).get("gf_scores", {}).keys())
    )

    if len(common_methods) >= 3:
        from scipy.stats import kendalltau
        y_ranks = {m: i+1 for i, m in enumerate(sorted(common_methods, key=lambda x: yeast_gf[x], reverse=True))}
        h_ranks = {m: i+1 for i, m in enumerate(sorted(common_methods, key=lambda x: analysis_results["human"]["gf_scores"][x], reverse=True))}
        m_ranks = {m: i+1 for i, m in enumerate(sorted(common_methods, key=lambda x: analysis_results["mouse"]["gf_scores"][x], reverse=True))}

        # Pairwise Spearman
        for sp1, r1, sp2, r2 in [
            ("yeast", y_ranks, "human", h_ranks),
            ("yeast", y_ranks, "mouse", m_ranks),
            ("human", h_ranks, "mouse", m_ranks),
        ]:
            v1 = np.array([r1[m] for m in common_methods])
            v2 = np.array([r2[m] for m in common_methods])
            rho, p = spearmanr(v1, v2)
            print(f"    {sp1} vs {sp2}: rho={rho:+.3f} (p={p:.4f})")

        # Kendall's W
        rank_matrix = np.array([[y_ranks[m], h_ranks[m], m_ranks[m]] for m in common_methods])
        k = rank_matrix.shape[1]
        n_m = rank_matrix.shape[0]
        rank_sums = rank_matrix.sum(axis=1)
        S = np.sum((rank_sums - rank_sums.mean()) ** 2)
        W = 12 * S / (k ** 2 * n_m * (n_m ** 2 - 1))
        print(f"    Kendall W (3 species): {W:.3f}")

        analysis_results["cross_species"] = {
            "common_methods": common_methods,
            "yeast_ranks": {m: y_ranks[m] for m in common_methods},
            "human_ranks": {m: h_ranks[m] for m in common_methods},
            "mouse_ranks": {m: m_ranks[m] for m in common_methods},
            "kendall_W": round(float(W), 3),
        }

    # =========================================================
    # Save Results
    # =========================================================
    print("\n  Saving results...")

    # Mouse GF analysis
    mouse_out = {
        "analysis": "Phase 10: Mouse G-F Analysis",
        "community_detection": "greedy_modularity_communities",
        "n_points": N_POINTS,
        "r_range": [R_MIN, R_MAX],
        "methods": list(mouse_results.keys()),
        "gf_scores": {m: {
            "gf_score": mouse_results[m]["gf_score"],
            "peak_purity": mouse_results[m]["peak_purity"],
        } for m in mouse_results},
        "features": {m: mouse_features.get(m, {}) for m in mouse_results},
    }
    with open(RESULTS / "mouse_gf_analysis.json", "w", encoding="utf-8") as f:
        json.dump(mouse_out, f, indent=2, ensure_ascii=False)

    # Persistence image analysis
    pi_out = {
        "analysis": "Phase 10: Persistence Image TDA Analysis",
        "species": ["human", "mouse"],
        "per_method": {},
    }
    for (species, m), data in ph_results.items():
        if species not in pi_out["per_method"]:
            pi_out["per_method"][species] = {}
        pi_out["per_method"][species][m] = {
            "h1_stats": data["h1_stats"],
            "pi_features": data["pi_features"],
        }
    pi_out["correlations"] = {s: d["correlations"] for s, d in analysis_results.items() if s != "cross_species"}
    with open(RESULTS / "persistence_image_analysis.json", "w", encoding="utf-8") as f:
        json.dump(pi_out, f, indent=2, ensure_ascii=False)

    # Cross-species
    cs_out = analysis_results.get("cross_species", {})
    cs_out["analysis"] = "Phase 10: Three-Species Cross-Species Comparison"
    with open(RESULTS / "cross_species_three_way.json", "w", encoding="utf-8") as f:
        json.dump(cs_out, f, indent=2, ensure_ascii=False)

    # =========================================================
    # Generate Figures
    # =========================================================
    print("\n  Generating figures...")
    generate_fig51(mouse_results, r_vals)
    generate_fig52(ph_results)
    generate_fig53(analysis_results, yeast_gf, common_methods)
    generate_fig54(ph_results, analysis_results)

    print(f"\n{BANNER}")
    print("Phase 10C complete.")
    print(BANNER)


# ============================================================
# Figure Generation
# ============================================================

def generate_fig51(mouse_results, r_vals):
    """Fig51: Mouse G-F curves (top-3 + bottom-3)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: G-F curves
    ax = axes[0]
    sorted_methods = sorted(mouse_results.keys(),
                           key=lambda m: mouse_results[m]["gf_score"], reverse=True)
    top3 = sorted_methods[:3]
    bot3 = sorted_methods[-3:]
    colors_top = ["#2171B5", "#4292C6", "#6BAED6"]
    colors_bot = ["#FC9272", "#FB6A4A", "#DE2D26"]
    for m, c in zip(top3, colors_top):
        p = np.array(mouse_results[m]["purities"])
        ax.plot(r_vals, p, color=c, lw=1.5, label=f"{m} (GF={mouse_results[m]['gf_score']:.3f})")
    for m, c in zip(bot3, colors_bot):
        p = np.array(mouse_results[m]["purities"])
        ax.plot(r_vals, p, color=c, lw=1.5, ls="--", label=f"{m} (GF={mouse_results[m]['gf_score']:.3f})")
    ax.axvline(x=GF_R_MIN, color="gray", ls=":", alpha=0.5)
    ax.axvline(x=GF_R_MAX, color="gray", ls=":", alpha=0.5)
    ax.set_title("A. Mouse G-F Curves (greedy_modularity)\n(blue=top-3, red=bottom-3)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Filtration radius r")
    ax.set_ylabel("Mean functional purity")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel B: GF Score ranking bar chart
    ax = axes[1]
    methods = sorted(mouse_results.keys(), key=lambda m: mouse_results[m]["gf_score"], reverse=True)
    scores = [mouse_results[m]["gf_score"] for m in methods]
    colors = ["#2171B5" if s > np.median(scores) else "#FC9272" for s in scores]
    ax.barh(range(len(methods)), scores, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("G-F Score")
    ax.set_title("B. Mouse G-F Score Ranking\n(11 methods, unified parameters)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(FIGURES / "Fig51_mouse_gf_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig51_mouse_gf_curves.png")


def generate_fig52(ph_results):
    """Fig52: Persistence images gallery (human vs mouse)."""
    methods = sorted(set(m for (_, m) in ph_results.keys()))
    species_list = ["human", "mouse"]

    # Select 6 methods: top-3 + bottom-3 by mean H1 max persistence across species
    h1_mean = {}
    for m in methods:
        vals = []
        for sp in species_list:
            if (sp, m) in ph_results:
                vals.append(ph_results[(sp, m)]["h1_stats"]["max_persistence"])
        h1_mean[m] = np.mean(vals) if vals else 0.0
    ranked = sorted(methods, key=lambda x: h1_mean[x], reverse=True)
    show_methods = ranked[:3] + ranked[-3:]

    n_cols = len(show_methods)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.5 * n_cols, 5.5))
    for row, species in enumerate(species_list):
        for col, m in enumerate(show_methods):
            ax = axes[row][col]
            key = (species, m)
            if key in ph_results:
                img = ph_results[key]["pi_image"]
                img_2d = np.atleast_2d(img)
                if img_2d.size > 0 and np.max(img_2d) > 1e-12:
                    ax.imshow(img_2d, cmap="hot_r", aspect="auto", origin="lower")
                else:
                    ax.text(0.5, 0.5, "No H1\nfeatures", ha="center", va="center",
                            fontsize=8, transform=ax.transAxes)
                h1_max = ph_results[key]["h1_stats"]["max_persistence"]
                ax.set_title(f"{m}\nH1_max={h1_max:.4f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(species.capitalize(), fontsize=10, fontweight="bold")

    fig.suptitle("Persistence Images: Human vs Mouse (H1)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig52_persistence_images.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig52_persistence_images.png")


def generate_fig53(analysis_results, yeast_gf, common_methods):
    """Fig53: Three-species comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if "cross_species" not in analysis_results or len(common_methods) < 3:
        fig.text(0.5, 0.5, "Insufficient cross-species data", ha="center", fontsize=14)
        fig.savefig(FIGURES / "Fig53_cross_species.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    cs = analysis_results["cross_species"]

    # Panel A: Yeast vs Human scatter
    ax = axes[0]
    y_vals = [yeast_gf[m] for m in common_methods]
    h_vals = [analysis_results["human"]["gf_scores"][m] for m in common_methods]
    ax.scatter(y_vals, h_vals, s=60, c="#2171B5", edgecolors="k", linewidth=0.5)
    for m in common_methods:
        ax.annotate(m, (yeast_gf[m], analysis_results["human"]["gf_scores"][m]),
                    fontsize=6, ha="left", va="bottom", xytext=(2, 2), textcoords="offset points")
    rho, _ = spearmanr(y_vals, h_vals)
    ax.set_title(f"A. Yeast vs Human\nrho={rho:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Yeast G-F Score")
    ax.set_ylabel("Human G-F Score")
    ax.grid(True, alpha=0.3)

    # Panel B: Yeast vs Mouse scatter
    ax = axes[1]
    m_vals = [analysis_results["mouse"]["gf_scores"][m] for m in common_methods]
    ax.scatter(y_vals, m_vals, s=60, c="#E6550D", edgecolors="k", linewidth=0.5)
    for m in common_methods:
        ax.annotate(m, (yeast_gf[m], analysis_results["mouse"]["gf_scores"][m]),
                    fontsize=6, ha="left", va="bottom", xytext=(2, 2), textcoords="offset points")
    rho, _ = spearmanr(y_vals, m_vals)
    ax.set_title(f"B. Yeast vs Mouse\nrho={rho:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Yeast G-F Score")
    ax.set_ylabel("Mouse G-F Score")
    ax.grid(True, alpha=0.3)

    # Panel C: Human vs Mouse scatter
    ax = axes[2]
    ax.scatter(h_vals, m_vals, s=60, c="#31A354", edgecolors="k", linewidth=0.5)
    for m in common_methods:
        ax.annotate(m, (analysis_results["human"]["gf_scores"][m], analysis_results["mouse"]["gf_scores"][m]),
                    fontsize=6, ha="left", va="bottom", xytext=(2, 2), textcoords="offset points")
    rho, _ = spearmanr(h_vals, m_vals)
    ax.set_title(f"C. Human vs Mouse\nrho={rho:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Human G-F Score")
    ax.set_ylabel("Mouse G-F Score")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Three-Species G-F Score Comparison (Kendall W={cs.get('kendall_W', 'N/A')})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig53_cross_species.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig53_cross_species.png")


def generate_fig54(ph_results, analysis_results):
    """Fig54: TDA feature comparison (max_persistence vs PI features)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for idx, species in enumerate(["human", "mouse"]):
        ax = axes[idx]
        if species not in analysis_results:
            continue

        methods = analysis_results[species]["methods"]
        gf = np.array([analysis_results[species]["gf_scores"][m] for m in methods])

        h1_max = np.array([ph_results[(species, m)]["h1_stats"]["max_persistence"] for m in methods])
        pi_energy = np.array([ph_results[(species, m)]["pi_features"]["total_energy"] for m in methods])

        # Scatter: GF vs H1 max persistence
        ax.scatter(h1_max, gf, s=60, c="#3182BD", edgecolors="k", linewidth=0.5,
                   label="H1 max persistence", zorder=3)
        # Scatter: GF vs PI energy (on secondary axis)
        ax2 = ax.twinx()
        ax2.scatter(pi_energy, gf, s=60, c="#E6550D", marker="^", edgecolors="k",
                    linewidth=0.5, label="PI total energy", zorder=3)

        rho_h1, _ = spearmanr(h1_max, gf) if np.std(h1_max) > 1e-10 else (float("nan"), float("nan"))
        rho_pi, _ = spearmanr(pi_energy, gf) if np.std(pi_energy) > 1e-10 else (float("nan"), float("nan"))

        ax.set_title(f"{'Human' if species == 'human' else 'Mouse'}: TDA Features vs G-F Score\n"
                     f"H1_max rho={rho_h1:+.3f}, PI_energy rho={rho_pi:+.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Feature value")
        ax.set_ylabel("G-F Score (blue)")
        ax2.set_ylabel("G-F Score (orange)")
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle("TDA Feature Comparison: Scalar vs Persistence Image", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig54_tda_feature_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig54_tda_feature_comparison.png")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run()
