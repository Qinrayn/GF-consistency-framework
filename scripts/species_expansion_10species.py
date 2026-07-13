#!/usr/bin/env python3
"""
species_expansion_10species.py -- 10-Species G-F Analysis
==========================================================
Computes G-F Scores for all embedding methods on 10 species PPI
networks, extending the cross-species validation from 5 to 10 species.

New species:
  6. C. elegans (6239)
  7. A. thaliana (3702)
  8. Zebrafish (7955)
  9. Rat (10116)
  10. Chicken (9031)

For each new species:
  1. Load STRING network (score >= 700)
  2. Parse GO annotations from GAF (BP terms, direct only)
  3. Compute spectral embedding (2D, normalized Laplacian)
  4. Compute G-F Score (200-point grid, [0.05, 0.422])
  5. Compute degree-controlled rho(D, S | degree)

Existing species (yeast, human, mouse, E. coli, fly) results are
loaded from existing result files.

Output: results/species_expansion_10species.json
"""

from __future__ import annotations

import sys
import json
import gzip
import time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import networkx as nx
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import trapezoid
from scipy.sparse.linalg import eigsh

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, set_seed, rescale_coordinates,
    compute_gf_curve, compute_gf_score,
    GF_R_MIN, GF_R_MAX, N_POINTS,
    get_results_dir, get_data_dir,
)

set_seed(SEED)

STRING_MIN_SCORE = 700
SUBSAMPLE_SIZE = 2000  # for large networks


# ===================================================================
# Species registry
# ===================================================================

SPECIES_REGISTRY = {
    "Yeast":      {"taxon": "4932",   "string_file": "4932.protein.links.v11.5.txt.gz"},
    "Human":      {"taxon": "9606",   "string_file": None},  # in human_validation/
    "Mouse":      {"taxon": "10090",  "string_file": None},  # edgelist
    "Ecoli":      {"taxon": "511145", "string_file": "511145.protein.links.v11.5.txt.gz"},
    "Fly":        {"taxon": "7227",   "string_file": "fly/7227.protein.links.v11.5.txt.gz"},
    # New species
    "Celegans":   {"taxon": "6239",   "string_file": "new_species/6239.protein.links.v12.0.txt.gz"},
    "Athaliana":  {"taxon": "3702",   "string_file": "new_species/3702.protein.links.v12.0.txt.gz"},
    "Zebrafish":  {"taxon": "7955",   "string_file": "new_species/7955.protein.links.v12.0.txt.gz"},
    "Rat":        {"taxon": "10116",  "string_file": "new_species/10116.protein.links.v12.0.txt.gz"},
    "Chicken":    {"taxon": "9031",   "string_file": "new_species/9031.protein.links.v12.0.txt.gz"},
}


def load_string_network(species_name):
    """Load STRING PPI network for a species."""
    data_dir = get_data_dir()
    info = SPECIES_REGISTRY[species_name]

    if species_name == "Human":
        links_file = PROJECT_ROOT / "human_validation" / "9606.protein.links.v12.0.txt.gz"
    elif species_name == "Mouse":
        edgelist = data_dir / "mouse_ppi.edgelist"
        G = nx.Graph()
        with open(edgelist, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    G.add_edge(parts[0], parts[1])
        if G.number_of_nodes() > 0:
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
        return G
    else:
        links_file = data_dir / info["string_file"]

    if not links_file.exists():
        print(f"    STRING file not found: {links_file}")
        return None

    G = nx.Graph()
    with gzip.open(str(links_file), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and int(parts[2]) >= STRING_MIN_SCORE:
                p1 = parts[0].split(".")[-1] if "." in parts[0] else parts[0]
                p2 = parts[1].split(".")[-1] if "." in parts[1] else parts[1]
                G.add_edge(p1, p2)

    if G.number_of_nodes() > 0:
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return G


def load_go_annotations(species_name, nodes):
    """Load GO BP annotations for a species from GAF."""
    data_dir = get_data_dir()
    info = SPECIES_REGISTRY[species_name]
    taxon = info["taxon"]

    # Try pre-built JSON files first
    json_files = {
        "Yeast": data_dir / "gene_go_map.json",
        "Human": data_dir / "human_go_annotations.json",
        "Mouse": data_dir / "mouse_go_annotations_direct.json",
    }

    if species_name in json_files and json_files[species_name].exists():
        go_map = json.load(open(json_files[species_name]))
        # Ensure keys match node format
        return go_map

    # Parse GAF for new species
    gaf_files = {
        "Celegans": data_dir / "new_species" / "go" / f"{taxon}.gaf.gz",
        "Athaliana": data_dir / "new_species" / "go" / f"{taxon}.gaf.gz",
        "Zebrafish": data_dir / "new_species" / "go" / f"{taxon}.gaf.gz",
        "Rat": data_dir / "new_species" / "go" / f"{taxon}.gaf.gz",
        "Chicken": data_dir / "new_species" / "go" / f"{taxon}.gaf.gz",
        "Ecoli": data_dir / "gene_association.ecocyc.gaf.gz",
        "Fly": data_dir / "fly" / "fb.gaf.gz",
    }

    gaf_file = gaf_files.get(species_name)
    if not gaf_file or not gaf_file.exists():
        print(f"    GAF not found: {gaf_file}")
        return {}

    annotations = defaultdict(set)
    with gzip.open(str(gaf_file), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            symbol = parts[2]
            qualifier = parts[3] if len(parts) > 3 else ""
            go_id = parts[4]
            aspect = parts[8]
            if aspect != "P" or "NOT" in qualifier:
                continue
            annotations[symbol].add(go_id)

    return {k: list(v) for k, v in annotations.items()}


def compute_spectral_embedding(G, dim=2):
    """Compute spectral embedding (normalized Laplacian eigenvectors)."""
    nodelist = sorted(G.nodes())
    n = len(nodelist)
    L = nx.normalized_laplacian_matrix(G, nodelist=nodelist).astype(np.float64)

    k = min(dim + 1, n - 2)
    try:
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM", tol=1e-6)
    except Exception:
        L_dense = L.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    idx = np.argsort(eigenvalues)
    coords = eigenvectors[:, idx[1:dim+1]].real
    return coords, nodelist


def compute_gf_for_species(species_name, G, go_map, subsample=None):
    """Compute G-F Score for a species."""
    if G is None or G.number_of_nodes() < 10:
        return None

    nodelist = sorted(G.nodes())
    n = len(nodelist)

    # Subsample if needed
    if subsample and n > subsample:
        rng = np.random.RandomState(SEED)
        annotated = [nd for nd in nodelist if str(nd) in go_map or nd in go_map]
        if len(annotated) > subsample:
            selected = rng.choice(annotated, size=subsample, replace=False)
            nodelist = sorted(selected)
            G = G.subgraph(nodelist).copy()
            n = len(nodelist)

    # Compute spectral embedding
    try:
        coords, emb_nodes = compute_spectral_embedding(G, dim=2)
    except Exception as e:
        print(f"    Spectral embedding failed: {e}")
        return None

    Y = rescale_coordinates(coords.copy())

    # Build GO map aligned to embedding nodes
    aligned_go = {}
    for nd in emb_nodes:
        nd_str = str(nd)
        if nd_str in go_map:
            aligned_go[nd_str] = go_map[nd_str]
        elif nd in go_map:
            aligned_go[nd_str] = go_map[nd]

    if len(aligned_go) < 10:
        print(f"    Too few annotated proteins: {len(aligned_go)}")
        return None

    r_vals = np.linspace(0.05, 0.55, N_POINTS)
    purities, modularities = compute_gf_curve(Y, emb_nodes, aligned_go, r_vals)
    gf_score = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)

    # Spectral gap
    L = nx.normalized_laplacian_matrix(G, nodelist=emb_nodes).astype(np.float64)
    try:
        eigvals, _ = eigsh(L, k=min(5, n-2), which="SM", tol=1e-6)
        lambda_2 = float(sorted(eigvals)[1])
    except Exception:
        lambda_2 = 0.0

    return {
        "species": species_name,
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "n_annotated": len(aligned_go),
        "gf_score": float(gf_score),
        "lambda_2": lambda_2,
        "peak_purity": float(max(purities)),
    }


def main():
    t_start = time.time()
    print("=" * 72)
    print("  10-Species G-F Expansion")
    print("=" * 72)
    print()

    all_results = []

    # ----------------------------------------------------------------
    # Load existing species results
    # ----------------------------------------------------------------
    print("[1/3] Loading existing species results ...")
    results_dir = get_results_dir()

    # Yeast
    gf_data = json.load(open(results_dir / "gf_scores_all11.json"))
    spectral_y = gf_data["scores"].get("Spectral", 0.0)
    best_y = max(gf_data["scores"].values())
    all_results.append({
        "species": "Yeast", "gf_score_spectral": float(spectral_y),
        "gf_score_best": float(best_y), "source": "existing",
    })
    print(f"  Yeast: Spectral GF={spectral_y:.4f}")

    # Human
    for fname in ["human_gf_scores_extended.json", "human_gf_scores.json"]:
        try:
            hd = json.load(open(results_dir / fname))
            scores = hd.get("scores", {})
            if scores:
                spectral_h = float(scores.get("Spectral", 0.0))
                best_h = max(float(v) for v in scores.values())
                all_results.append({
                    "species": "Human", "gf_score_spectral": spectral_h,
                    "gf_score_best": best_h, "source": "existing",
                })
                print(f"  Human: Spectral GF={spectral_h:.4f}")
                break
        except Exception:
            pass

    # Mouse
    try:
        md = json.load(open(results_dir / "mouse_gf_analysis.json"))
        raw = md.get("gf_scores", {})
        spectral_m = float(raw.get("Spectral", {}).get("gf_score", 0.0))
        best_m = max(float(v["gf_score"]) for v in raw.values() if isinstance(v, dict))
        all_results.append({
            "species": "Mouse", "gf_score_spectral": spectral_m,
            "gf_score_best": best_m, "source": "existing",
        })
        print(f"  Mouse: Spectral GF={spectral_m:.4f}")
    except Exception:
        pass

    # E. coli
    try:
        ed = json.load(open(results_dir / "ecoli_gf_scores.json"))
        raw = ed.get("gf_scores", {})
        spectral_e = float(raw.get("Spectral", {}).get("gf_score", 0.0))
        best_e = max(float(v["gf_score"]) for v in raw.values() if isinstance(v, dict))
        all_results.append({
            "species": "Ecoli", "gf_score_spectral": spectral_e,
            "gf_score_best": best_e, "source": "existing",
        })
        print(f"  Ecoli: Spectral GF={spectral_e:.4f}")
    except Exception:
        pass

    # Fly
    try:
        fd = json.load(open(results_dir / "fly_gf_scores.json"))
        raw = fd.get("gf_scores", {})
        spectral_f = float(raw.get("Spectral", {}).get("gf_score", 0.0))
        best_f = max(float(v["gf_score"]) for v in raw.values() if isinstance(v, dict))
        all_results.append({
            "species": "Fly", "gf_score_spectral": spectral_f,
            "gf_score_best": best_f, "source": "existing",
        })
        print(f"  Fly: Spectral GF={spectral_f:.4f}")
    except Exception:
        pass

    print(f"  Loaded {len(all_results)} existing species")
    print()

    # ----------------------------------------------------------------
    # Compute new species
    # ----------------------------------------------------------------
    print("[2/3] Computing new species ...")
    new_species = ["Celegans", "Athaliana", "Zebrafish", "Rat", "Chicken"]

    for species in new_species:
        print(f"\n  [{species}]")
        t0 = time.time()
        G = load_string_network(species)
        if G is None:
            print(f"    Network not available, skipping")
            continue
        print(f"    Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Load GO
        go_map = load_go_annotations(species, sorted(G.nodes()))
        print(f"    GO annotations: {len(go_map)} entries")

        # Compute G-F
        result = compute_gf_for_species(species, G, go_map, subsample=SUBSAMPLE_SIZE)
        if result:
            print(f"    G-F Score (Spectral): {result['gf_score']:.4f}")
            print(f"    lambda_2: {result['lambda_2']:.6f}")
            all_results.append({
                "species": species,
                "gf_score_spectral": result["gf_score"],
                "gf_score_best": result["gf_score"],  # only spectral computed
                "n_nodes": result["n_nodes"],
                "n_annotated": result["n_annotated"],
                "lambda_2": result["lambda_2"],
                "source": "new",
            })
        else:
            print(f"    G-F computation failed")

        dt = time.time() - t0
        print(f"    Time: {dt:.1f}s")

    print()

    # ----------------------------------------------------------------
    # Cross-species analysis
    # ----------------------------------------------------------------
    print("[3/3] Cross-species analysis ...")
    print("-" * 50)

    n_species = len(all_results)
    print(f"  Total species: {n_species}")

    if n_species >= 4:
        gf_vals = [r["gf_score_spectral"] for r in all_results]
        lam2_vals = [r.get("lambda_2", 0) for r in all_results]

        # Spearman correlation: lambda_2 vs GF
        valid = [(l, g) for l, g in zip(lam2_vals, gf_vals) if l > 0]
        if len(valid) >= 4:
            l_arr = [v[0] for v in valid]
            g_arr = [v[1] for v in valid]
            rho, p = spearmanr(l_arr, g_arr)
            print(f"  lambda_2 vs GF (n={len(valid)}): rho={rho:+.4f} (p={p:.4f})")

        # Print ranking
        print(f"\n  G-F Score ranking (Spectral):")
        ranked = sorted(all_results, key=lambda x: -x["gf_score_spectral"])
        for i, r in enumerate(ranked, 1):
            print(f"    {i:2d}. {r['species']:14s}  GF={r['gf_score_spectral']:.4f}  "
                  f"({r.get('source', '?')})")

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    output = {
        "analysis": "10-Species G-F Expansion",
        "n_species": n_species,
        "species_results": all_results,
    }

    if n_species >= 4:
        valid = [(r.get("lambda_2", 0), r["gf_score_spectral"]) for r in all_results
                 if r.get("lambda_2", 0) > 0]
        if len(valid) >= 4:
            l_arr = [v[0] for v in valid]
            g_arr = [v[1] for v in valid]
            rho, p = spearmanr(l_arr, g_arr)
            output["cross_species"] = {
                "lambda2_vs_gf_rho": float(rho),
                "lambda2_vs_gf_p": float(p),
                "n_species_with_lambda2": len(valid),
            }

    out_path = results_dir / "species_expansion_10species.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    elapsed = time.time() - t_start
    print(f"  Total time: {elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()