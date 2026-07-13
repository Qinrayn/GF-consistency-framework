#!/usr/bin/env python3
"""
dark_matter_disease_enrichment.py -- Local Disease Enrichment of Dark Matter
============================================================================
Maps the 44 yeast functional dark matter protein pairs to human functional
orthologs using GO term overlap (no external API required), then tests
whether the human orthologs are enriched for disease-associated genes.

Pipeline:
  1. Map yeast ORFs to gene symbols via STRING aliases
  2. Parse SGD GAF to get GO BP terms for all 71 dark matter proteins
  3. Find human proteins sharing >= 2 GO BP terms (functional orthologs)
  4. Test enrichment: are these human proteins more likely to be in
     disease-associated GO terms than random human proteins?
  5. Pairwise test: do dark matter ortholog pairs share more GO terms
     than random human protein pairs?

Output: results/dark_matter_disease_enrichment.json
"""

from __future__ import annotations

import sys
import json
import gzip
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import get_results_dir, get_data_dir


def load_orf_to_symbol():
    """Map yeast ORF IDs to gene symbols via STRING aliases."""
    data_dir = get_data_dir()
    alias_file = data_dir / "4932.protein.aliases.v12.0.txt.gz"
    if not alias_file.exists():
        alias_file = data_dir / "4932.protein.aliases.v11.5.txt.gz"
    orf_to_symbol = {}
    with gzip.open(str(alias_file), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            string_id = parts[0]
            alias = parts[1]
            source = parts[2]
            orf = string_id.split(".")[-1] if "." in string_id else string_id
            if "SGD" in source or orf not in orf_to_symbol:
                orf_to_symbol[orf] = alias
    return orf_to_symbol


def parse_gaf(gaf_path, aspect="P"):
    """Parse GAF, return {gene_symbol: set(go_ids)} for direct annotations."""
    annotations = defaultdict(set)
    with gzip.open(str(gaf_path), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            symbol = parts[2]  # Gene symbol (column 3)
            qualifier = parts[3] if len(parts) > 3 else ""
            go_id = parts[4]
            aspect_code = parts[8]
            if aspect_code != aspect:
                continue
            if "NOT" in qualifier:
                continue
            annotations[symbol].add(go_id)
    return dict(annotations)


def main():
    t_start = time.time()
    print("=" * 72)
    print("  Dark Matter Disease Enrichment (Local, No API)")
    print("  GO-based functional orthology + disease enrichment")
    print("=" * 72)
    print()

    data_dir = get_data_dir()
    results_dir = get_results_dir()

    # ----------------------------------------------------------------
    # Step 1: Load dark matter pairs
    # ----------------------------------------------------------------
    print("[1/6] Loading dark matter pairs ...")
    dm_data = json.load(open(results_dir / "functional_dark_matter.json"))
    dm_pairs = dm_data["top_100_catalog"]
    yeast_proteins = set()
    for pair in dm_pairs:
        yeast_proteins.add(pair["protein_a"])
        yeast_proteins.add(pair["protein_b"])
    print(f"  {len(dm_pairs)} pairs, {len(yeast_proteins)} unique proteins")
    print()

    # ----------------------------------------------------------------
    # Step 2: Map ORFs to gene symbols
    # ----------------------------------------------------------------
    print("[2/6] Mapping ORFs to gene symbols ...")
    orf_to_sym = load_orf_to_symbol()
    dm_symbols = {}
    for orf in yeast_proteins:
        sym = orf_to_sym.get(orf, orf)
        dm_symbols[orf] = sym
    n_mapped = sum(1 for o in yeast_proteins if o in orf_to_sym)
    print(f"  Mapped: {n_mapped}/{len(yeast_proteins)}")
    print()

    # ----------------------------------------------------------------
    # Step 3: Parse SGD GAF for yeast GO annotations
    # ----------------------------------------------------------------
    print("[3/6] Parsing SGD GAF for yeast GO BP annotations ...")
    yeast_go = parse_gaf(data_dir / "gene_association.sgd.gaf.gz", aspect="P")
    print(f"  {len(yeast_go)} yeast genes with BP annotations")

    # Get GO terms for dark matter proteins
    dm_go = {}
    for orf, sym in dm_symbols.items():
        if sym in yeast_go:
            dm_go[orf] = yeast_go[sym]
    print(f"  Dark matter proteins with GO: {len(dm_go)}/{len(yeast_proteins)}")

    all_dm_terms = set()
    for terms in dm_go.values():
        all_dm_terms.update(terms)
    print(f"  Unique GO BP terms: {len(all_dm_terms)}")
    print()

    # ----------------------------------------------------------------
    # Step 4: Find human functional orthologs
    # ----------------------------------------------------------------
    print("[4/6] Finding human functional orthologs ...")
    human_go = json.load(open(data_dir / "human_go_annotations.json"))
    print(f"  Human GO annotations: {len(human_go)} proteins")

    # Build GO term -> human proteins index
    go_to_human = defaultdict(list)
    for hprot, hterms in human_go.items():
        for t in hterms:
            go_to_human[t].append(hprot)

    # For each dark matter protein, find human proteins with GO overlap
    human_orthologs = {}  # yeast_orf -> best human protein
    all_human_candidates = set()

    for orf, dm_terms in dm_go.items():
        dm_term_set = set(dm_terms)
        overlap_counts = defaultdict(int)
        for t in dm_terms:
            for hprot in go_to_human.get(t, []):
                overlap_counts[hprot] += 1

        # Keep best match (most shared GO terms)
        if overlap_counts:
            best_human = max(overlap_counts, key=overlap_counts.get)
            best_count = overlap_counts[best_human]
            if best_count >= 2:
                human_orthologs[orf] = {
                    "human_id": best_human,
                    "shared_go": best_count,
                    "yeast_go_terms": len(dm_terms),
                }
                all_human_candidates.add(best_human)

    print(f"  Human functional orthologs: {len(human_orthologs)}/{len(dm_go)}")
    print(f"  Unique human ortholog candidates: {len(all_human_candidates)}")
    print()

    # ----------------------------------------------------------------
    # Step 5: Disease enrichment test
    # ----------------------------------------------------------------
    print("[5/6] Disease enrichment analysis ...")
    print("  (Testing if dark matter orthologs are enriched for")
    print("   disease-associated GO terms vs random human proteins)")
    print()

    # Define "disease-associated" GO terms (manually curated subset)
    # These are GO BP terms commonly associated with human disease
    disease_go_terms = {
        "GO:0006915",   # apoptotic process
        "GO:0008283",   # cell population proliferation
        "GO:0007165",   # signal transduction
        "GO:0006955",   # immune response
        "GO:0007267",   # cell-cell signaling
        "GO:0001501",   # skeletal system development
        "GO:0007399",   # nervous system development
        "GO:0001568",   # blood vessel development
        "GO:0048513",   # animal organ development
        "GO:0007275",   # multicellular organism development
        "GO:0007610",   # behavior
        "GO:0006355",   # regulation of DNA-templated transcription
        "GO:0006936",   # muscle contraction
        "GO:0005975",   # carbohydrate metabolic process
        "GO:0006629",   # lipid metabolic process
    }

    # Count disease-associated proteins in dark matter orthologs vs background
    dm_human_disease = 0
    for hprot in all_human_candidates:
        hterms = set(human_go.get(hprot, []))
        if hterms & disease_go_terms:
            dm_human_disease += 1

    # Background: all human proteins with GO annotations
    bg_human_disease = 0
    bg_total = len(human_go)
    for hprot, hterms in human_go.items():
        if set(hterms) & disease_go_terms:
            bg_human_disease += 1

    # Fisher exact test
    table = [[dm_human_disease, len(all_human_candidates) - dm_human_disease],
             [bg_human_disease, bg_total - bg_human_disease]]
    odds, p_val = fisher_exact(table, alternative="greater")

    print(f"  Dark matter orthologs: {dm_human_disease}/{len(all_human_candidates)} "
          f"disease-associated ({100*dm_human_disease/max(len(all_human_candidates),1):.1f}%)")
    print(f"  Background (all human): {bg_human_disease}/{bg_total} "
          f"({100*bg_human_disease/bg_total:.1f}%)")
    print(f"  Fisher exact: odds={odds:.3f}, p={p_val:.4f}")
    print()

    # ----------------------------------------------------------------
    # Step 6: Pairwise GO overlap test
    # ----------------------------------------------------------------
    print("[6/6] Pairwise GO overlap test ...")
    print("  (Do dark matter ortholog pairs share more GO terms")
    print("   than random human protein pairs?)")
    print()

    # Build human dark matter pairs
    human_dm_pairs = []
    for pair in dm_pairs:
        ya, yb = pair["protein_a"], pair["protein_b"]
        if ya in human_orthologs and yb in human_orthologs:
            ha = human_orthologs[ya]["human_id"]
            hb = human_orthologs[yb]["human_id"]
            if ha != hb:
                ha_terms = set(human_go.get(ha, []))
                hb_terms = set(human_go.get(hb, []))
                overlap = len(ha_terms & hb_terms)
                union = len(ha_terms | hb_terms)
                jaccard = overlap / union if union > 0 else 0
                human_dm_pairs.append({
                    "yeast_a": ya, "yeast_b": yb,
                    "human_a": ha, "human_b": hb,
                    "shared_go": overlap,
                    "jaccard": jaccard,
                })

    print(f"  Human dark matter pairs: {len(human_dm_pairs)}")

    if len(human_dm_pairs) >= 3:
        dm_jaccards = [p["jaccard"] for p in human_dm_pairs]

        # Random pairs: sample 10000 random human protein pairs
        all_human = list(human_go.keys())
        rng = np.random.RandomState(42)
        random_jaccards = []
        for _ in range(min(10000, len(all_human) * 2)):
            i, j = rng.choice(len(all_human), 2, replace=False)
            ti = set(human_go[all_human[i]])
            tj = set(human_go[all_human[j]])
            u = ti | tj
            if len(u) > 0:
                random_jaccards.append(len(ti & tj) / len(u))

        # Mann-Whitney U test
        u_stat, u_p = mannwhitneyu(dm_jaccards, random_jaccards, alternative="greater")

        print(f"  DM pair Jaccard: mean={np.mean(dm_jaccards):.4f}, "
              f"median={np.median(dm_jaccards):.4f}")
        print(f"  Random pair Jaccard: mean={np.mean(random_jaccards):.4f}, "
              f"median={np.median(random_jaccards):.4f}")
        print(f"  Mann-Whitney U: {u_stat:.0f}, p={u_p:.4f}")
        print(f"  Enrichment ratio: {np.mean(dm_jaccards)/np.mean(random_jaccards):.2f}x")
    else:
        u_stat, u_p = 0, 1.0
        dm_jaccards = []
        random_jaccards = []
    print()

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    output = {
        "analysis": "Dark Matter Disease Enrichment (Local)",
        "description": (
            "Maps 44 yeast dark matter pairs to human functional orthologs "
            "via GO BP term overlap (no external API), then tests disease "
            "enrichment and pairwise GO sharing."
        ),
        "n_yeast_proteins": len(yeast_proteins),
        "n_yeast_with_go": len(dm_go),
        "n_human_orthologs": len(human_orthologs),
        "n_human_dm_pairs": len(human_dm_pairs),
        "ortholog_mapping": dm_symbols,
        "human_orthologs": human_orthologs,
        "human_dm_pairs": human_dm_pairs,
        "disease_enrichment": {
            "n_disease_go_terms": len(disease_go_terms),
            "dm_disease_count": dm_human_disease,
            "dm_total": len(all_human_candidates),
            "bg_disease_count": bg_human_disease,
            "bg_total": bg_total,
            "fisher_odds": float(odds),
            "fisher_p": float(p_val),
        },
        "pairwise_overlap": {
            "dm_jaccard_mean": float(np.mean(dm_jaccards)) if dm_jaccards else 0,
            "dm_jaccard_median": float(np.median(dm_jaccards)) if dm_jaccards else 0,
            "random_jaccard_mean": float(np.mean(random_jaccards)) if random_jaccards else 0,
            "random_jaccard_median": float(np.median(random_jaccards)) if random_jaccards else 0,
            "mannwhitney_u": float(u_stat),
            "mannwhitney_p": float(u_p),
            "enrichment_ratio": float(np.mean(dm_jaccards) / np.mean(random_jaccards))
                if random_jaccards and np.mean(random_jaccards) > 0 else 0,
        },
    }

    out_path = results_dir / "dark_matter_disease_enrichment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()