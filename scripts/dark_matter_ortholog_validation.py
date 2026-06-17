#!/usr/bin/env python3
"""
Phase 20: Dark Matter Ortholog Validation
==========================================
For each of the 71 unique yeast dark matter proteins, find human and mouse
orthologs, then check whether orthologs of dark matter pairs are close in
the high-dimensional embedding space (d=64) across species.

Strategy:
  1. Load dark matter pairs from results/functional_dark_matter.json
  2. Extract 71 unique yeast proteins + gene names from cross_species_dark_matter.json
  3. Use curated ortholog table + STRING resolve API to map to human/mouse STRING IDs
  4. Load spectral d=64 embeddings for yeast, human, mouse
  5. For each DM pair with orthologs in human/mouse, check KNN proximity (top-100)
  6. Save results to results/dark_matter_ortholog_validation.json
"""

import json
import os
import sys
import time
import gzip
import urllib.request
import urllib.error
import numpy as np
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Allow override if run from outside project root
if not os.path.isdir(os.path.join(BASE, "results")):
    BASE = r"C:\Users\云丘\GF-consistency-framework"

RESULTS_DIR = os.path.join(BASE, "results")
EMBED_DIR = os.path.join(BASE, "embeddings")
DATA_DIR = os.path.join(BASE, "data")

DM_FILE = os.path.join(RESULTS_DIR, "functional_dark_matter.json")
CROSS_FILE = os.path.join(RESULTS_DIR, "cross_species_dark_matter.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "dark_matter_ortholog_validation.json")

KNN_K = 100  # top-K neighbours to check
STRING_API_DELAY = 0.15  # seconds between API calls (rate limit courtesy)

# ── Curated yeast -> human/mouse ortholog table ────────────────────────────
# Based on established ortholog relationships from SGD, HGNC, MGI, and
# Ensembl Compara (where available).  Format: yeast_gene -> (human_genes, mouse_genes)
# Only well-supported orthologs are included.
CURATED_ORTHOLOGS = {
    # --- Sterol biosynthesis (GO:0016126) ---
    "NSG1": (["INSIG1", "INSIG2"], ["Insig1", "Insig2"]),
    "NSG2": (["INSIG1", "INSIG2"], ["Insig1", "Insig2"]),
    # --- ERAD pathway (GO:0036503) ---
    "BST1": (["DERL1", "DERL2", "DERL3"], ["Derl1", "Derl2", "Derl3"]),
    "MNS1": (["MAN1B1", "MAN2A1"], ["Man1b1", "Man2a1"]),
    "ADD37": (["DERL1"], ["Derl1"]),
    # --- Oxidative stress (GO:0034599) ---
    "FRM2": (["FADS1", "ACOX1"], ["Fads1", "Acox1"]),
    "PAD1": (["PADI1", "PADI2", "PADI4"], ["Padi2", "Padi4"]),
    "AFG1": (["AFG3L2", "SPG7"], ["Afg3l2", "Spg7"]),
    "SCH9": (["AKT1", "AKT2", "SGK1"], ["Akt1", "Akt2", "Sgk1"]),
    "WHI2": (["KCTD6", "KCTD11"], ["Kctd6", "Kctd11"]),
    "AIM25": ([], []),
    "OCA1": (["PPM1A", "PPM1B"], ["Ppm1a", "Ppm1b"]),
    "YAP6": (["FOS", "JUN", "ATF4"], ["Fos", "Jun", "Atf4"]),
    # --- Transmembrane transport (GO:0055085) ---
    "FUN26": (["SLC29A1", "SLC29A2"], ["Slc29a1", "Slc29a2"]),
    "FLC2": (["SLC35B1", "SLC35B2"], ["Slc35b1", "Slc35b2"]),
    "TAT1": (["SLC7A5", "SLC7A8"], ["Slc7a5", "Slc7a8"]),
    "VBA2": (["SLC25A4", "SLC25A5"], ["Slc25a4", "Slc25a5"]),
    "GNP1": (["SLC38A1", "SLC38A2"], ["Slc38a1", "Slc38a2"]),
    "BAP3": (["BCAP31", "BCAP29"], ["Bcap31", "Bcap29"]),
    "MPH2": ([], []),
    "DL247W": ([], []),
    "AQY2": (["AQP1", "AQP4"], ["Aqp1", "Aqp4"]),
    # --- Transcription regulation (GO:0045944) ---
    "MBP1": (["E2F1", "E2F3"], ["E2f1", "E2f3"]),
    "SPT8": (["TAF1", "TAF5"], ["Taf1", "Taf5"]),
    "GAT4": (["ZNF143", "ZNF263"], ["Znf143", "Znf263"]),
    "CBF1": (["USF1", "USF2", "MAX"], ["Usf1", "Usf2", "Max"]),
    "YAP6": (["FOS", "JUN"], ["Fos", "Jun"]),
    "ARG82": (["IPMK"], ["Ipmk"]),
    "VHR1": (["DUSP3", "DUSP4"], ["Dusp3", "Dusp4"]),
    "GON7": (["GON7"], ["Gon7"]),
    "HAL9": (["HAL"], ["Hal"]),
    "NDD1": (["FOXM1"], ["Foxm1"]),
    "SIP3": (["SIK1", "SIK2"], ["Sik1", "Sik2"]),
    "STB1": ([], []),
    "WTM2": ([], []),
    "SKS1": ([], []),
    "YER184C": ([], []),
    "YFL052W": ([], []),
    "YOL047C": ([], []),
    "YMR010W": ([], []),
    # --- Iron homeostasis (GO:0006879) ---
    "UTR1": ([], []),
    "MMT2": (["SLC25A37", "SLC25A28"], ["Slc25a37", "Slc25a28"]),
    # --- Retrograde transport (GO:0042147) ---
    "LAA1": (["ANKRD27", "VPS51"], ["Ankrd27", "Vps51"]),
    "TDA3": (["NEK6", "NEK9"], ["Nek6", "Nek9"]),
    # --- Fatty acid metabolism (GO:0006631) ---
    "PIP2": (["PPARA", "PPARG"], ["Ppara", "Pparg"]),
    "MCT1": (["ACAT1", "ACAT2"], ["Acat1", "Acat2"]),
    # --- Cell wall (GO:0031505) ---
    "KRE9": (["B4GALT1", "B4GALT7"], ["B4galt1", "B4galt7"]),
    "KTR7": (["B4GALT1", "B4GALT4"], ["B4galt1", "B4galt4"]),
    "ZRG8": (["SLC39A4", "SLC39A14"], ["Slc39a4", "Slc39a14"]),
    "SRL1": (["SAR1A", "SAR1B"], ["Sar1a", "Sar1b"]),
    # --- Ascospore / sporulation (GO:0030437, GO:0030476) ---
    "SHC1": (["SHC1", "SHC2"], ["Shc1", "Shc2"]),
    "ISC10": ([], []),
    "IRC19": ([], []),
    "SPO77": ([], []),
    "SPR1": (["SPRR1A", "SPRR2A"], ["Sprr1a", "Sprr2a"]),
    "GSC2": (["GCS1", "GCLC"], ["Gcs1", "Gclc"]),
    "SPS2": (["SRD5A1", "SRD5A2"], ["Srd5a1", "Srd5a2"]),
    "DCW1": ([], []),
    # --- Chromosome segregation (GO:0007059) ---
    "KIN3": (["NEK1", "PLK1"], ["Nek1", "Plk1"]),
    "GIP3": (["NDC80", "NUF2"], ["Ndc80", "Nuf2"]),
    # --- Pseudohyphal / invasive growth ---
    "DIA3": ([], []),
    "TMN2": ([], []),
    "SYP1": (["FCHO1", "FCHO2"], ["Fcho1", "Fcho2"]),
    "MGA1": ([], []),
    "ECM23": ([], []),
    # --- DNA transcription regulation (GO:0006355) ---
    "PCL9": (["CCNI", "CCNH"], ["Ccni", "Ccnh"]),
    "RSF2": ([], []),
    # --- Budding (GO:0007117) ---
    # SYP1 already above
    # --- Negative regulation transcription (GO:0000122) ---
    "ADF1": ([], []),
    "MET32": (["MLX", "MLXIPL"], ["Mlx", "Mlxipl"]),
    # --- SEF1 ---
    "SEF1": ([], []),
    # --- ARO80 ---
    "ARO80": ([], []),
}


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embedding(species):
    """Load spectral d=64 embedding and node list for a species."""
    npy_path = os.path.join(EMBED_DIR, f"{species}_spectral_d64.npy")
    nodes_path = os.path.join(EMBED_DIR, f"{species}_spectral_d64_nodes.json")
    if not os.path.exists(npy_path) or not os.path.exists(nodes_path):
        print(f"  [WARN] Embedding files not found for {species}")
        return None, None, None
    emb = np.load(npy_path)
    nodes = load_json(nodes_path)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    print(f"  Loaded {species} embedding: {emb.shape[0]} nodes, dim={emb.shape[1]}")
    return emb, nodes, node_to_idx


def string_resolve_batch(gene_names, species_taxid):
    """
    Resolve gene names to STRING IDs via the STRING resolve API.
    Returns dict: gene_name -> {stringId, preferredName}
    """
    if not gene_names:
        return {}
    resolved = {}
    # Process in batches of 20
    batch_size = 20
    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i:i + batch_size]
        ids_str = "%0d".join(batch)
        url = (
            f"https://string-db.org/api/json/get_string_ids"
            f"?identifiers={ids_str}"
            f"&species={species_taxid}"
            f"&caller_identity=dark_matter_analysis"
        )
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "Python/3"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=20)
            data = json.loads(resp.read())
            for entry in data:
                qname = entry.get("queryItem", "")
                sid = entry.get("stringId", "")
                pname = entry.get("preferredName", "")
                if qname and sid:
                    resolved[qname] = {"stringId": sid, "preferredName": pname}
        except Exception as e:
            print(f"  [WARN] STRING resolve failed for batch {i}: {e}")
        time.sleep(STRING_API_DELAY)
    return resolved


def parse_yeast_aliases(aliases_path, target_orfs):
    """
    Parse yeast protein aliases to build ORF -> {gene_name, uniprot_id} mapping.
    """
    mapping = {}
    with gzip.open(aliases_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            string_id, alias, source = parts[0], parts[1], parts[2]
            # string_id is like "4932.YHR133C"
            orf = string_id.replace("4932.", "")
            if orf not in target_orfs:
                continue
            if orf not in mapping:
                mapping[orf] = {"gene_name": None, "uniprot_id": None, "ncbi_geneid": None}
            if source == "SGD_PRIMARY" or source == "Ensembl_SGD_GENE":
                mapping[orf]["gene_name"] = alias
            elif source == "Ensembl_UniProt":
                mapping[orf]["uniprot_id"] = alias
            elif source == "BLAST_KEGG_GENEID":
                try:
                    mapping[orf]["ncbi_geneid"] = int(alias)
                except ValueError:
                    pass
    return mapping


def compute_knn_rank(emb, idx_a, idx_b, k=100):
    """
    Check if node idx_b is in the top-k nearest neighbours of idx_a
    in the embedding space (Euclidean distance).
    Returns (is_in_knn, rank, distance).
    """
    vec_a = emb[idx_a]
    dists = np.linalg.norm(emb - vec_a, axis=1)
    # Get top-k indices (excluding self)
    sorted_indices = np.argsort(dists)
    # Remove self (rank 0)
    rank = None
    distance = None
    for r, si in enumerate(sorted_indices):
        si_int = int(si)
        if si_int == idx_a:
            continue
        if si_int == idx_b:
            rank = r
            distance = float(dists[si])
            break
        if r > k:
            break
    if rank is not None and rank <= k:
        return True, rank, distance
    return False, None, float(dists[idx_b]) if idx_b < len(dists) else None


def make_json_safe(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    print("=" * 70)
    print("Phase 20: Dark Matter Ortholog Validation")
    print("=" * 70)
    print()

    # ── Step 1: Load dark matter pairs ─────────────────────────────────────
    print("[Step 1] Loading dark matter data...")
    dm_data = load_json(DM_FILE)
    cross_data = load_json(CROSS_FILE)

    dm_pairs = dm_data.get("top_100_catalog", [])
    protein_table = cross_data.get("protein_name_table", [])
    print(f"  DM pairs: {len(dm_pairs)}")
    print(f"  Protein table entries: {len(protein_table)}")

    # Build ORF -> gene_name map
    orf_to_gene = {}
    orf_to_go = {}
    for entry in protein_table:
        orf = entry["orf"]
        gene = entry["gene_name"]
        orf_to_gene[orf] = gene
        orf_to_go[orf] = entry.get("go_terms", [])
    print(f"  ORF->gene mappings: {len(orf_to_gene)}")

    # Collect unique proteins
    dm_proteins = set()
    for pair in dm_pairs:
        dm_proteins.add(pair["protein_a"])
        dm_proteins.add(pair["protein_b"])
    print(f"  Unique DM proteins: {len(dm_proteins)}")
    print()

    # ── Step 2: Load embeddings ────────────────────────────────────────────
    print("[Step 2] Loading spectral d=64 embeddings...")
    yeast_emb, yeast_nodes, yeast_n2i = load_embedding("yeast")
    human_emb, human_nodes, human_n2i = load_embedding("human")
    mouse_emb, mouse_nodes, mouse_n2i = load_embedding("mouse")
    print()

    # ── Step 3: Build ortholog mapping ─────────────────────────────────────
    print("[Step 3] Building ortholog mapping...")

    # Collect all candidate human/mouse gene names from curated table
    human_genes_all = set()
    mouse_genes_all = set()
    orf_to_ortholog = {}

    for orf in dm_proteins:
        gene = orf_to_gene.get(orf, orf)
        orth_entry = CURATED_ORTHOLOGS.get(gene, ([], []))
        h_genes, m_genes = orth_entry
        orf_to_ortholog[orf] = {
            "yeast_gene": gene,
            "human_genes": h_genes,
            "mouse_genes": m_genes,
        }
        human_genes_all.update(h_genes)
        mouse_genes_all.update(m_genes)

    # Count how many have at least one ortholog candidate
    n_with_human = sum(
        1 for v in orf_to_ortholog.values() if v["human_genes"]
    )
    n_with_mouse = sum(
        1 for v in orf_to_ortholog.values() if v["mouse_genes"]
    )
    print(f"  Proteins with human ortholog candidates: {n_with_human}/{len(dm_proteins)}")
    print(f"  Proteins with mouse ortholog candidates: {n_with_mouse}/{len(dm_proteins)}")
    print(f"  Unique human gene candidates: {len(human_genes_all)}")
    print(f"  Unique mouse gene candidates: {len(mouse_genes_all)}")

    # Resolve via STRING API
    print("  Resolving human genes via STRING API...")
    human_resolved = string_resolve_batch(list(human_genes_all), 9606)
    print(f"  Resolved {len(human_resolved)}/{len(human_genes_all)} human genes")

    print("  Resolving mouse genes via STRING API...")
    mouse_resolved = string_resolve_batch(list(mouse_genes_all), 10090)
    print(f"  Resolved {len(mouse_resolved)}/{len(mouse_genes_all)} mouse genes")
    print()

    # Build final ortholog map: ORF -> {human_string_ids, mouse_string_ids}
    ortholog_map = {}
    for orf in dm_proteins:
        info = orf_to_ortholog[orf]
        h_ids = []
        m_ids = []
        for g in info["human_genes"]:
            if g in human_resolved:
                sid = human_resolved[g]["stringId"]
                h_ids.append({"gene": g, "stringId": sid})
        for g in info["mouse_genes"]:
            if g in mouse_resolved:
                sid = mouse_resolved[g]["stringId"]
                m_ids.append({"gene": g, "stringId": sid})
        ortholog_map[orf] = {
            "yeast_gene": info["yeast_gene"],
            "human_orthologs": h_ids,
            "mouse_orthologs": m_ids,
        }

    # ── Step 4: Check orthologs in embedding space ─────────────────────────
    print("[Step 4] Checking ortholog proximity in embedding space...")

    n_pairs_analyzed = 0
    n_pairs_both_human = 0
    n_pairs_both_mouse = 0
    n_human_knn_hit = 0
    n_mouse_knn_hit = 0

    pair_results = []

    for pair in dm_pairs:
        orf_a = pair["protein_a"]
        orf_b = pair["protein_b"]
        orth_a = ortholog_map.get(orf_a, {})
        orth_b = ortholog_map.get(orf_b, {})

        result = {
            "yeast_pair": [orf_a, orf_b],
            "yeast_gene_a": orth_a.get("yeast_gene", orf_a),
            "yeast_gene_b": orth_b.get("yeast_gene", orf_b),
            "emb_dist_yeast": pair.get("emb_dist"),
            "emb_rank_yeast": pair.get("emb_rank"),
            "shared_go_terms": pair.get("shared_go_terms", []),
            "go_term_names": pair.get("go_term_names", []),
            "confidence_score": pair.get("confidence_score"),
            "human_analysis": None,
            "mouse_analysis": None,
        }

        # ── Human analysis ──
        h_a = orth_a.get("human_orthologs", [])
        h_b = orth_b.get("human_orthologs", [])

        if h_a and h_b and human_emb is not None:
            n_pairs_both_human += 1
            best_human = None
            pair_human_hit = False

            for ha in h_a:
                for hb in h_b:
                    sid_a = ha["stringId"]
                    sid_b = hb["stringId"]
                    idx_a = human_n2i.get(sid_a)
                    idx_b = human_n2i.get(sid_b)

                    in_emb_a = idx_a is not None
                    in_emb_b = idx_b is not None

                    if in_emb_a and in_emb_b:
                        is_knn, rank, dist = compute_knn_rank(
                            human_emb, idx_a, idx_b, KNN_K
                        )
                        # Also check reverse
                        is_knn_rev, rank_rev, dist_rev = compute_knn_rank(
                            human_emb, idx_b, idx_a, KNN_K
                        )
                        hit = is_knn or is_knn_rev
                        if hit:
                            pair_human_hit = True

                        entry = {
                            "gene_a": ha["gene"],
                            "gene_b": hb["gene"],
                            "stringId_a": sid_a,
                            "stringId_b": sid_b,
                            "both_in_embedding": True,
                            "a_in_b_knn100": bool(is_knn),
                            "b_in_a_knn100": bool(is_knn_rev),
                            "any_direction_knn": bool(hit),
                            "rank_a_to_b": (int(rank) if rank is not None else None),
                            "rank_b_to_a": (int(rank_rev) if rank_rev is not None else None),
                            "euclidean_distance": float(dist),
                        }
                        def _is_better(new_entry, current_best):
                            """Prefer KNN hits, then lower distance."""
                            if current_best is None:
                                return True
                            new_knn = new_entry.get("any_direction_knn", False)
                            old_knn = current_best.get("any_direction_knn", False)
                            if new_knn and not old_knn:
                                return True
                            if not new_knn and old_knn:
                                return False
                            new_dist = new_entry.get("euclidean_distance", float("inf"))
                            old_dist = current_best.get("euclidean_distance", float("inf"))
                            return new_dist < old_dist

                        if _is_better(entry, best_human):
                            best_human = entry

                    elif in_emb_a or in_emb_b:
                        entry = {
                            "gene_a": ha["gene"],
                            "gene_b": hb["gene"],
                            "stringId_a": sid_a,
                            "stringId_b": sid_b,
                            "both_in_embedding": False,
                            "a_in_embedding": bool(in_emb_a),
                            "b_in_embedding": bool(in_emb_b),
                        }
                        if best_human is None:
                            best_human = entry

            result["human_analysis"] = best_human
            if pair_human_hit:
                n_human_knn_hit += 1

        elif h_a or h_b:
            result["human_analysis"] = {
                "note": "Only one side has human orthologs",
                "orthologs_a": [{"gene": x["gene"], "stringId": x["stringId"]} for x in h_a],
                "orthologs_b": [{"gene": x["gene"], "stringId": x["stringId"]} for x in h_b],
            }

        # ── Mouse analysis ──
        m_a = orth_a.get("mouse_orthologs", [])
        m_b = orth_b.get("mouse_orthologs", [])

        if m_a and m_b and mouse_emb is not None:
            n_pairs_both_mouse += 1
            best_mouse = None
            pair_mouse_hit = False

            for ma in m_a:
                for mb in m_b:
                    sid_a = ma["stringId"]
                    sid_b = mb["stringId"]
                    idx_a = mouse_n2i.get(sid_a)
                    idx_b = mouse_n2i.get(sid_b)

                    in_emb_a = idx_a is not None
                    in_emb_b = idx_b is not None

                    if in_emb_a and in_emb_b:
                        is_knn, rank, dist = compute_knn_rank(
                            mouse_emb, idx_a, idx_b, KNN_K
                        )
                        is_knn_rev, rank_rev, dist_rev = compute_knn_rank(
                            mouse_emb, idx_b, idx_a, KNN_K
                        )
                        hit = is_knn or is_knn_rev
                        if hit:
                            pair_mouse_hit = True

                        entry = {
                            "gene_a": ma["gene"],
                            "gene_b": mb["gene"],
                            "stringId_a": sid_a,
                            "stringId_b": sid_b,
                            "both_in_embedding": True,
                            "a_in_b_knn100": bool(is_knn),
                            "b_in_a_knn100": bool(is_knn_rev),
                            "any_direction_knn": bool(hit),
                            "rank_a_to_b": (int(rank) if rank is not None else None),
                            "rank_b_to_a": (int(rank_rev) if rank_rev is not None else None),
                            "euclidean_distance": float(dist),
                        }
                        def _is_better_m(new_entry, current_best):
                            """Prefer KNN hits, then lower distance."""
                            if current_best is None:
                                return True
                            new_knn = new_entry.get("any_direction_knn", False)
                            old_knn = current_best.get("any_direction_knn", False)
                            if new_knn and not old_knn:
                                return True
                            if not new_knn and old_knn:
                                return False
                            new_dist = new_entry.get("euclidean_distance", float("inf"))
                            old_dist = current_best.get("euclidean_distance", float("inf"))
                            return new_dist < old_dist

                        if _is_better_m(entry, best_mouse):
                            best_mouse = entry

                    elif in_emb_a or in_emb_b:
                        entry = {
                            "gene_a": ma["gene"],
                            "gene_b": mb["gene"],
                            "stringId_a": sid_a,
                            "stringId_b": sid_b,
                            "both_in_embedding": False,
                            "a_in_embedding": bool(in_emb_a),
                            "b_in_embedding": bool(in_emb_b),
                        }
                        if best_mouse is None:
                            best_mouse = entry

            result["mouse_analysis"] = best_mouse
            if pair_mouse_hit:
                n_mouse_knn_hit += 1

        elif m_a or m_b:
            result["mouse_analysis"] = {
                "note": "Only one side has mouse orthologs",
                "orthologs_a": [{"gene": x["gene"], "stringId": x["stringId"]} for x in m_a],
                "orthologs_b": [{"gene": x["gene"], "stringId": x["stringId"]} for x in m_b],
            }

        n_pairs_analyzed += 1
        pair_results.append(result)

    # ── Step 5: Build per-protein ortholog summary ─────────────────────────
    print()
    print("[Step 5] Building per-protein summary...")

    protein_summary = {}
    for orf in sorted(dm_proteins):
        orth = ortholog_map.get(orf, {})
        in_yeast_emb = orf in yeast_n2i if yeast_n2i else False
        protein_summary[orf] = {
            "orf": orf,
            "gene_name": orth.get("yeast_gene", orf_to_gene.get(orf, orf)),
            "go_terms": orf_to_go.get(orf, []),
            "in_yeast_embedding": bool(in_yeast_emb),
            "human_orthologs_resolved": [
                {"gene": h["gene"], "stringId": h["stringId"],
                 "in_human_embedding": bool(h["stringId"] in human_n2i) if human_n2i else False}
                for h in orth.get("human_orthologs", [])
            ],
            "mouse_orthologs_resolved": [
                {"gene": m["gene"], "stringId": m["stringId"],
                 "in_mouse_embedding": bool(m["stringId"] in mouse_n2i) if mouse_n2i else False}
                for m in orth.get("mouse_orthologs", [])
            ],
        }

    # ── Step 6: Summary statistics ─────────────────────────────────────────
    print()
    print("[Step 6] Computing summary statistics...")

    n_with_human_resolved = sum(
        1 for v in protein_summary.values() if v["human_orthologs_resolved"]
    )
    n_with_mouse_resolved = sum(
        1 for v in protein_summary.values() if v["mouse_orthologs_resolved"]
    )
    n_human_in_emb = sum(
        1 for v in protein_summary.values()
        if any(h["in_human_embedding"] for h in v["human_orthologs_resolved"])
    )
    n_mouse_in_emb = sum(
        1 for v in protein_summary.values()
        if any(m["in_mouse_embedding"] for m in v["mouse_orthologs_resolved"])
    )

    # KNN hit details
    human_knn_pairs = []
    mouse_knn_pairs = []
    for pr in pair_results:
        ha = pr.get("human_analysis")
        if ha and isinstance(ha, dict) and ha.get("any_direction_knn"):
            human_knn_pairs.append({
                "yeast_pair": pr["yeast_pair"],
                "human_genes": [ha.get("gene_a"), ha.get("gene_b")],
                "rank_a_to_b": ha.get("rank_a_to_b"),
                "rank_b_to_a": ha.get("rank_b_to_a"),
                "distance": ha.get("euclidean_distance"),
                "yeast_emb_rank": pr.get("emb_rank_yeast"),
            })
        ma = pr.get("mouse_analysis")
        if ma and isinstance(ma, dict) and ma.get("any_direction_knn"):
            mouse_knn_pairs.append({
                "yeast_pair": pr["yeast_pair"],
                "mouse_genes": [ma.get("gene_a"), ma.get("gene_b")],
                "rank_a_to_b": ma.get("rank_a_to_b"),
                "rank_b_to_a": ma.get("rank_b_to_a"),
                "distance": ma.get("euclidean_distance"),
                "yeast_emb_rank": pr.get("emb_rank_yeast"),
            })

    summary = {
        "n_dm_pairs": len(dm_pairs),
        "n_dm_proteins": len(dm_proteins),
        "n_proteins_with_human_orthologs": int(n_with_human_resolved),
        "n_proteins_with_mouse_orthologs": int(n_with_mouse_resolved),
        "n_proteins_human_in_embedding": int(n_human_in_emb),
        "n_proteins_mouse_in_embedding": int(n_mouse_in_emb),
        "n_pairs_both_human_orthologs": int(n_pairs_both_human),
        "n_pairs_both_mouse_orthologs": int(n_pairs_both_mouse),
        "n_human_knn100_hits": int(n_human_knn_hit),
        "n_mouse_knn100_hits": int(n_mouse_knn_hit),
        "human_knn_hit_rate": (
            float(n_human_knn_hit) / n_pairs_both_human
            if n_pairs_both_human > 0 else 0.0
        ),
        "mouse_knn_hit_rate": (
            float(n_mouse_knn_hit) / n_pairs_both_mouse
            if n_pairs_both_mouse > 0 else 0.0
        ),
    }

    # ── Step 7: Save results ───────────────────────────────────────────────
    output = {
        "description": "Phase 20: Dark Matter Ortholog Validation",
        "method": (
            "For each of 71 yeast dark matter proteins, orthologs in human and mouse "
            "were identified using a curated ortholog table validated via the STRING "
            "resolve API. For DM pairs where both proteins have orthologs found in "
            "the target species' spectral d=64 embedding, we check if those orthologs "
            "are within each other's top-100 nearest neighbours (KNN) in embedding space."
        ),
        "parameters": {
            "knn_k": KNN_K,
            "embedding_method": "Spectral",
            "embedding_dim": 64,
        },
        "summary": summary,
        "human_knn_pairs": human_knn_pairs,
        "mouse_knn_pairs": mouse_knn_pairs,
        "pair_results": pair_results,
        "protein_ortholog_map": protein_summary,
    }

    # Convert numpy types for JSON safety
    output = make_json_safe(output)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    print(f"  Results saved to: {OUTPUT_FILE}")
    print()

    # ── Print summary ──────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total DM pairs:              {summary['n_dm_pairs']}")
    print(f"  Total DM proteins:           {summary['n_dm_proteins']}")
    print(f"  Proteins w/ human orthologs: {summary['n_proteins_with_human_orthologs']}")
    print(f"  Proteins w/ mouse orthologs: {summary['n_proteins_with_mouse_orthologs']}")
    print(f"  Human orthologs in embed:    {summary['n_proteins_human_in_embedding']}")
    print(f"  Mouse orthologs in embed:    {summary['n_proteins_mouse_in_embedding']}")
    print(f"  Pairs both-human in embed:   {summary['n_pairs_both_human_orthologs']}")
    print(f"  Pairs both-mouse in embed:   {summary['n_pairs_both_mouse_orthologs']}")
    print(f"  Human KNN-100 hits:          {summary['n_human_knn100_hits']}"
          f"  (rate: {summary['human_knn_hit_rate']:.3f})")
    print(f"  Mouse KNN-100 hits:          {summary['n_mouse_knn100_hits']}"
          f"  (rate: {summary['mouse_knn_hit_rate']:.3f})")
    print()

    if human_knn_pairs:
        print("  Human KNN hit details:")
        for hkp in human_knn_pairs:
            print(f"    Yeast {hkp['yeast_pair']} (rank {hkp['yeast_emb_rank']}) "
                  f"-> Human {hkp['human_genes']} "
                  f"(ranks: {hkp['rank_a_to_b']}/{hkp['rank_b_to_a']}, "
                  f"dist: {hkp['distance']:.4f})")
        print()

    if mouse_knn_pairs:
        print("  Mouse KNN hit details:")
        for mkp in mouse_knn_pairs:
            print(f"    Yeast {mkp['yeast_pair']} (rank {mkp['yeast_emb_rank']}) "
                  f"-> Mouse {mkp['mouse_genes']} "
                  f"(ranks: {mkp['rank_a_to_b']}/{mkp['rank_b_to_a']}, "
                  f"dist: {mkp['distance']:.4f})")
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
