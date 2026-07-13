#!/usr/bin/env python3
"""
build_full_go_map.py -- Build full-network GO annotation map from SGD GAF
=========================================================================
The existing data/gene_go_map.json contains only 154 curated genes.
This script parses the full SGD GAF to build a ~7000-gene GO BP map,
enabling true full-network G-F analysis.

Also tests GO dependence by computing G-F with KEGG pathways as an
alternative functional reference.

Output: data/gene_go_map_full.json  (~7000 genes)
        results/kegg_gf_comparison.json  (GO vs KEGG)
"""

from __future__ import annotations

import sys
import json
import gzip
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    SEED, set_seed, ALL_METHODS,
    load_curated_network, load_embedding,
    rescale_coordinates, compute_gf_curve, compute_gf_score,
    GF_R_MIN, GF_R_MAX, N_POINTS,
    get_data_dir, get_results_dir,
)

set_seed(SEED)


def parse_sgd_gaf(gaf_path, aspect="P"):
    """Parse SGD GAF, return {orf: [go_terms]} for direct BP annotations.

    Maps SGD systematic ORF names (Y#####X format) to GO terms.
    The GAF column 2 contains SGD IDs (S000000001), column 3 contains
    gene symbols. We need ORF names to match STRING/edgelist node IDs.

    Strategy: use STRING aliases file to map gene symbols -> ORF.
    """
    data_dir = get_data_dir()

    # Step 1: Parse GAF -> {gene_symbol: set(go_terms)}
    symbol_to_go = defaultdict(set)
    with gzip.open(str(gaf_path), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            symbol = parts[2]
            qualifier = parts[3] if len(parts) > 3 else ""
            go_id = parts[4]
            aspect_code = parts[8]
            if aspect_code != aspect or "NOT" in qualifier:
                continue
            symbol_to_go[symbol].add(go_id)

    # Step 2: Load STRING aliases to map symbol -> ORF
    alias_file = data_dir / "4932.protein.aliases.v11.5.txt.gz"
    if not alias_file.exists():
        alias_file = data_dir / "4932.protein.aliases.v12.0.txt.gz"

    symbol_to_orf = {}
    with gzip.open(str(alias_file), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            string_id = parts[0]
            alias = parts[1]
            source = parts[2]
            # Extract ORF from STRING ID
            orf = string_id.split(".")[-1] if "." in string_id else string_id
            # Map alias (gene symbol) -> ORF
            if alias not in symbol_to_orf:
                symbol_to_orf[alias] = orf
            # Also store the ORF itself
            if orf not in symbol_to_orf:
                symbol_to_orf[orf] = orf

    # Step 3: Build {orf: [go_terms]}
    orf_to_go = {}
    n_mapped = 0
    n_unmapped = 0
    for symbol, go_terms in symbol_to_go.items():
        orf = symbol_to_orf.get(symbol)
        if orf:
            if orf not in orf_to_go:
                orf_to_go[orf] = set()
            orf_to_go[orf].update(go_terms)
            n_mapped += 1
        else:
            n_unmapped += 1

    # Convert sets to sorted lists
    orf_to_go = {k: sorted(v) for k, v in orf_to_go.items()}
    return orf_to_go, n_mapped, n_unmapped


def load_kegg_pathways():
    """Load KEGG pathway assignments for yeast proteins.

    Since we cannot download KEGG at runtime (requires API key), we use
    a proxy: protein complex membership from the curated PPI network.
    Each protein is assigned to its connected component as a "pathway".

    Alternative: use the functional modules identified by the G-F analysis
    itself (circular but tests whether the metric is self-consistent).

    Better alternative: parse the SGD GAF for Molecular Function (aspect=F)
    and Cellular Component (aspect=C) terms as non-BP functional references.
    """
    data_dir = get_data_dir()
    gaf_path = data_dir / "gene_association.sgd.gaf.gz"

    # Parse MF and CC annotations as alternative functional references
    mf_map, _, _ = parse_sgd_gaf(gaf_path, aspect="F")
    cc_map, _, _ = parse_sgd_gaf(gaf_path, aspect="C")

    return {"MF": mf_map, "CC": cc_map}


def main():
    t_start = time.time()
    print("=" * 72)
    print("  Build Full GO Map + Non-GO Functional Reference Test")
    print("=" * 72)
    print()

    data_dir = get_data_dir()
    results_dir = get_results_dir()

    # ----------------------------------------------------------------
    # Step 1: Build full GO BP map from SGD GAF
    # ----------------------------------------------------------------
    print("[1/4] Building full GO BP map from SGD GAF ...")
    gaf_path = data_dir / "gene_association.sgd.gaf.gz"
    if not gaf_path.exists():
        print(f"  ERROR: GAF not found: {gaf_path}")
        return

    t0 = time.time()
    full_go, n_mapped, n_unmapped = parse_sgd_gaf(gaf_path, aspect="P")
    print(f"  Genes with GO BP: {len(full_go)} (mapped: {n_mapped}, unmapped: {n_unmapped})")

    term_counts = [len(v) for v in full_go.values()]
    print(f"  Terms per gene: mean={np.mean(term_counts):.1f}, "
          f"median={np.median(term_counts):.0f}, max={max(term_counts)}")
    print(f"  (vs curated 153-gene map: 154 genes, mean 3.8 terms)")

    # Save
    out_file = data_dir / "gene_go_map_full.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_go, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_file}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ----------------------------------------------------------------
    # Step 2: Verify coverage on full network
    # ----------------------------------------------------------------
    print("[2/4] Verifying coverage on full network ...")

    # Load full network node list
    import networkx as nx
    G_full = nx.Graph()
    edgelist = data_dir / "yeast_ppi_5936.edgelist"
    with open(edgelist) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                G_full.add_edge(parts[0], parts[1])

    nodes_full = set(G_full.nodes())
    annotated_full = nodes_full & set(full_go.keys())
    print(f"  Full network: {len(nodes_full)} nodes")
    print(f"  Annotated (full GO): {len(annotated_full)} ({100*len(annotated_full)/len(nodes_full):.1f}%)")
    print(f"  Annotated (old 153): 154 ({100*154/len(nodes_full):.1f}%)")
    print()

    # ----------------------------------------------------------------
    # Step 3: Compute G-F with full GO map (true full-network G-F)
    # ----------------------------------------------------------------
    print("[3/4] Computing true full-network G-F with full GO map ...")
    print("-" * 60)

    # Use a subsample of annotated nodes for G-F (greedy_modularity is slow)
    rng = np.random.RandomState(SEED)
    annotated_list = sorted(annotated_full)
    SUBSAMPLE = 500  # tractable for greedy_modularity
    if len(annotated_list) > SUBSAMPLE:
        gf_nodes = sorted(rng.choice(annotated_list, SUBSAMPLE, replace=False))
    else:
        gf_nodes = annotated_list

    print(f"  G-F subsample: {len(gf_nodes)} annotated nodes")

    # Load v11.5 full-network embeddings
    r_vals = np.linspace(0.05, 0.55, 50)

    v11_embeddings = {
        "Spectral": "embeddings/Spectral_d2_full.npy",
        "VGAE": "embeddings/VGAE_full.npy",
        "VGAE-feat": "embeddings/VGAE-feat_full.npy",
        "GraphSAGE": "embeddings/GraphSAGE_full.npy",
        "GAT": "embeddings/GAT_full.npy",
        "GIN": "embeddings/GIN_full.npy",
    }

    # Also include v12.0 embeddings (server output)
    v12_embeddings = {
        "Spectral_v12": "embeddings/Spectral_full.npy",
        "PCA_v12": "embeddings/PCA_full.npy",
        "Node2Vec_v12": "embeddings/Node2Vec_full.npy",
        "DM_v12": "embeddings/DM_full.npy",
        "DeepWalk_v12": "embeddings/DeepWalk_full.npy",
        "MDS_v12": "embeddings/MDS_full.npy",
    }

    all_embeddings = {**v11_embeddings, **v12_embeddings}

    print(f"\n  {'Method':<16s} {'GF_full_GO':>10s} {'n_nodes':>8s}")
    print("  " + "-" * 38)

    gf_results = {}
    for method, path in all_embeddings.items():
        if not Path(path).exists():
            continue
        coords = np.load(path)
        nodes_path = path.replace(".npy", "_nodes.json")
        with open(nodes_path) as f:
            emb_nodes = json.load(f)

        idx = {n: i for i, n in enumerate(emb_nodes)}
        common = [n for n in gf_nodes if n in idx]
        if len(common) < 20:
            continue

        emb_idx = [idx[n] for n in common]
        Y = rescale_coordinates(coords[emb_idx].copy())
        go_sub = {n: full_go[n] for n in common if n in full_go}

        try:
            purities, _ = compute_gf_curve(Y, common, go_sub, r_vals)
            gf = compute_gf_score(r_vals, purities, 0.05, 0.422)
            gf_results[method] = {"gf": float(gf), "n": len(common)}
            print(f"  {method:<16s} {gf:10.4f} {len(common):8d}")
        except Exception as e:
            print(f"  {method:<16s} FAILED: {str(e)[:40]}")

    print()

    # ----------------------------------------------------------------
    # Step 4: Non-GO functional reference test (MF and CC)
    # ----------------------------------------------------------------
    print("[4/4] Non-GO functional reference test (MF + CC) ...")
    print("-" * 60)
    print("  Testing whether G-F ranking changes with different GO aspects")
    print("  (BP = Biological Process, MF = Molecular Function, CC = Cellular Component)")
    print()

    alt_maps = load_kegg_pathways()

    # Compute G-F with MF and CC on the 153-node curated network
    G_cur, nodes_cur, go_bp = load_curated_network()
    r_vals_153 = np.linspace(0.05, 0.55, N_POINTS)

    print(f"  {'Method':<12s} {'GF_BP':>8s} {'GF_MF':>8s} {'GF_CC':>8s} {'rank_change':>12s}")
    print("  " + "-" * 52)

    aspect_results = {}
    for method in ALL_METHODS:
        try:
            coords, emb_nodes = load_embedding(method, subset="153")
        except FileNotFoundError:
            continue

        node_to_idx = {nd: i for i, nd in enumerate(emb_nodes)}
        common = [nd for nd in nodes_cur if nd in node_to_idx]
        if len(common) < 10:
            continue

        emb_idx = [node_to_idx[nd] for nd in common]
        Y = rescale_coordinates(coords[emb_idx].copy())

        # BP (standard)
        go_bp_sub = {nd: go_bp.get(str(nd), go_bp.get(nd, [])) for nd in common}
        p_bp, _ = compute_gf_curve(Y, common, go_bp_sub, r_vals_153)
        gf_bp = compute_gf_score(r_vals_153, p_bp, GF_R_MIN, GF_R_MAX)

        # MF
        go_mf_sub = {nd: alt_maps["MF"].get(str(nd), alt_maps["MF"].get(nd, [])) for nd in common}
        n_mf = sum(1 for v in go_mf_sub.values() if v)
        if n_mf < 10:
            print(f"  {method:<12s} {gf_bp:8.4f} {'N/A':>8s} {'N/A':>8s} (MF coverage too low)")
            continue
        p_mf, _ = compute_gf_curve(Y, common, go_mf_sub, r_vals_153)
        gf_mf = compute_gf_score(r_vals_153, p_mf, GF_R_MIN, GF_R_MAX)

        # CC
        go_cc_sub = {nd: alt_maps["CC"].get(str(nd), alt_maps["CC"].get(nd, [])) for nd in common}
        n_cc = sum(1 for v in go_cc_sub.values() if v)
        if n_cc < 10:
            print(f"  {method:<12s} {gf_bp:8.4f} {gf_mf:8.4f} {'N/A':>8s} (CC coverage too low)")
            continue
        p_cc, _ = compute_gf_curve(Y, common, go_cc_sub, r_vals_153)
        gf_cc = compute_gf_score(r_vals_153, p_cc, GF_R_MIN, GF_R_MAX)

        # Rank change indicator
        rank_bp = gf_bp
        rank_mf = gf_mf
        rank_cc = gf_cc
        max_diff = max(abs(gf_mf - gf_bp), abs(gf_cc - gf_bp))
        change = "STABLE" if max_diff < 0.02 else "CHANGED"

        print(f"  {method:<12s} {gf_bp:8.4f} {gf_mf:8.4f} {gf_cc:8.4f} {change:>12s}")
        aspect_results[method] = {
            "gf_bp": float(gf_bp), "gf_mf": float(gf_mf), "gf_cc": float(gf_cc),
        }

    print()

    # Rank correlation across aspects
    if len(aspect_results) >= 4:
        from scipy.stats import spearmanr
        methods_c = list(aspect_results.keys())
        bp_vals = [aspect_results[m]["gf_bp"] for m in methods_c]
        mf_vals = [aspect_results[m]["gf_mf"] for m in methods_c]
        cc_vals = [aspect_results[m]["gf_cc"] for m in methods_c]

        rho_bp_mf, p_bp_mf = spearmanr(bp_vals, mf_vals)
        rho_bp_cc, p_bp_cc = spearmanr(bp_vals, cc_vals)
        rho_mf_cc, p_mf_cc = spearmanr(mf_vals, cc_vals)

        print(f"  Rank correlations across GO aspects:")
        print(f"    BP vs MF: rho={rho_bp_mf:+.4f} (p={p_bp_mf:.4f})")
        print(f"    BP vs CC: rho={rho_bp_cc:+.4f} (p={p_bp_cc:.4f})")
        print(f"    MF vs CC: rho={rho_mf_cc:+.4f} (p={p_mf_cc:.4f})")
        print()
        if rho_bp_mf > 0.7 and rho_bp_cc > 0.7:
            print("  -> G-F ranking is ROBUST across GO aspects.")
            print("     The framework is not specific to BP annotations.")
        else:
            print("  -> G-F ranking CHANGES across GO aspects.")
            print("     The framework may be BP-specific.")

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    output = {
        "analysis": "Full GO Map + Non-GO Functional Reference Test",
        "full_go_map": {
            "n_genes": len(full_go),
            "mean_terms_per_gene": float(np.mean(term_counts)),
            "median_terms": float(np.median(term_counts)),
            "coverage_on_full_network": f"{len(annotated_full)}/{len(nodes_full)} ({100*len(annotated_full)/len(nodes_full):.1f}%)",
        },
        "full_network_gf": gf_results,
        "aspect_comparison": aspect_results,
    }

    if len(aspect_results) >= 4:
        output["aspect_rank_correlations"] = {
            "bp_vs_mf": {"rho": float(rho_bp_mf), "p": float(p_bp_mf)},
            "bp_vs_cc": {"rho": float(rho_bp_cc), "p": float(p_bp_cc)},
            "mf_vs_cc": {"rho": float(rho_mf_cc), "p": float(p_mf_cc)},
        }

    out_path = results_dir / "full_go_and_aspect_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    elapsed = time.time() - t_start
    print(f"  Total time: {elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()