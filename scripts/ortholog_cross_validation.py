#!/usr/bin/env python3
"""
Ortholog Cross-Validation (Step 71 / Phase 2.3)
================================================
Systematic ortholog mapping via STRING homology API to cross-validate
function predictions across species.

For each of the top N well-annotated yeast proteins:
  1. Look up human/mouse orthologs via STRING homology API
  2. Check annotation overlap between yeast and ortholog
  3. Report concordance rate and Jaccard similarity

Usage:
  python scripts/ortholog_cross_validation.py --max-proteins 100
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
)
from function_prediction import EXPERIMENTAL_CODES, build_alias_mapping

DATA = get_data_dir()
RESULTS = get_results_dir()

STRING_API_DELAY = 0.35  # seconds between API calls


def load_yeast_annotations():
    """Parse SGD GAF for all experimental annotations (all aspects)."""
    sgd_map, orf_map, _ = build_alias_mapping()
    gaf_file = DATA / "gene_association.sgd.gaf.gz"
    annotations = defaultdict(set)
    orf_to_string = {}

    with gzip.open(str(gaf_file), "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("!") or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            evidence = cols[6]
            aspect = cols[8]
            if evidence not in EXPERIMENTAL_CODES or aspect not in ("P", "F", "C"):
                continue
            go_term = cols[4]
            if not go_term.startswith("GO:"):
                continue

            sgd_id = cols[1]
            gene_sym = cols[2]
            orf_name = cols[10] if len(cols) > 10 else ""

            string_id = None
            if sgd_id in sgd_map:
                string_id = sgd_map[sgd_id]
            if string_id is None and orf_name and orf_name in orf_map:
                string_id = orf_map[orf_name]
            if string_id is None and gene_sym in orf_map:
                string_id = orf_map[gene_sym]

            if string_id:
                annotations[string_id].add(go_term)
                orf_to_string[orf_name] = string_id

    return dict(annotations), orf_to_string


def load_species_annotations(species_key):
    """Load human/mouse annotations from JSON."""
    go_aspect = {}
    obo_file = DATA / "go.obo"
    current_id = None
    with open(obo_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("namespace:") and current_id:
                go_aspect[current_id] = line.split("namespace: ")[1]
                current_id = None

    ann_file = DATA / f"{species_key}_go_annotations.json"
    if not ann_file.exists():
        return {}

    with open(ann_file, encoding="utf-8") as f:
        raw = json.load(f)

    # Merge all aspects
    merged = {}
    for pid, terms in raw.items():
        merged[pid] = set(terms)
    return merged


def string_get_homologs(yeast_string_ids, target_species_id=9606):
    """Look up orthologs via STRING homology API."""
    homologs = {}
    total = len(yeast_string_ids)

    for i, yid in enumerate(sorted(yeast_string_ids)):
        url = (
            f"https://string-db.org/api/json/homology"
            f"?identifier={yid}"
            f"&species=4932"
            f"&required_species={target_species_id}"
            f"&caller_identity=gf_framework"
        )
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json",
                         "User-Agent": "Python/3"},
            )
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())

            orthologs = []
            for entry in data:
                target_id = entry.get("stringId_B", "")
                score = entry.get("score", 0)
                if target_id and score > 0:
                    orthologs.append({
                        "stringId": target_id,
                        "preferredName": entry.get("preferredName_B", ""),
                        "score": float(score),
                    })
            homologs[yid] = orthologs

        except Exception as e:
            homologs[yid] = []

        if (i + 1) % 10 == 0 or i == total - 1:
            n_found = sum(1 for v in homologs.values() if v)
            print(f"    [{i+1}/{total}] orthologs: {n_found}/{i+1}")

        time.sleep(STRING_API_DELAY)

    return homologs


def main():
    t_start = time.time()
    print("=" * 64)
    print("  Ortholog Cross-Validation (Step 71)")
    print("=" * 64)
    np.random.seed(SEED)

    # Load yeast annotations
    print("\n  Loading yeast annotations ...")
    yeast_ann, _ = load_yeast_annotations()
    print(f"  Yeast: {len(yeast_ann)} annotated proteins")

    # Select top proteins by annotation count
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-proteins", type=int, default=100)
    args = parser.parse_args()

    protein_ranking = sorted(
        [(pid, len(terms)) for pid, terms in yeast_ann.items()],
        key=lambda x: -x[1]
    )
    selected = protein_ranking[:args.max_proteins]
    selected_ids = [pid for pid, _ in selected]
    print(f"  Selected top {len(selected_ids)} proteins "
          f"(annotations: {selected[-1][1]} to {selected[0][1]})")

    # Look up orthologs
    print(f"\n  Looking up human orthologs via STRING API ...")
    human_homologs = string_get_homologs(selected_ids, target_species_id=9606)
    n_human = sum(1 for v in human_homologs.values() if v)
    print(f"  Human orthologs found: {n_human}/{len(selected_ids)}")

    print(f"\n  Looking up mouse orthologs via STRING API ...")
    mouse_homologs = string_get_homologs(selected_ids, target_species_id=10090)
    n_mouse = sum(1 for v in mouse_homologs.values() if v)
    print(f"  Mouse orthologs found: {n_mouse}/{len(selected_ids)}")

    # Load target species annotations
    print(f"\n  Loading human/mouse annotations ...")
    human_ann = load_species_annotations("human")
    mouse_ann = load_species_annotations("mouse")
    print(f"  Human: {len(human_ann)} proteins, Mouse: {len(mouse_ann)} proteins")

    # Load GO aspect map for analysis
    go_aspect = {}
    obo_file = DATA / "go.obo"
    current_id = None
    with open(obo_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("namespace:") and current_id:
                go_aspect[current_id] = line.split("namespace: ")[1]
                current_id = None

    # Concordance analysis
    print(f"\n  Computing concordance ...")
    results = []
    stats = {
        "human": {"total": 0, "concordant": 0, "jaccard": []},
        "mouse": {"total": 0, "concordant": 0, "jaccard": []},
    }

    for yeast_id in selected_ids:
        yeast_terms = yeast_ann.get(yeast_id, set())
        if not yeast_terms:
            continue

        entry = {
            "yeast_id": yeast_id,
            "n_yeast_terms": len(yeast_terms),
        }

        for species, homologs, ann in [
            ("human", human_homologs, human_ann),
            ("mouse", mouse_homologs, mouse_ann),
        ]:
            orthologs = homologs.get(yeast_id, [])
            if not orthologs:
                continue

            best = orthologs[0]  # highest score
            orth_id = best["stringId"]
            orth_terms = ann.get(orth_id, set())

            if not orth_terms:
                continue

            overlap = yeast_terms & orth_terms
            union = yeast_terms | orth_terms
            jaccard = len(overlap) / len(union) if union else 0

            stats[species]["total"] += 1
            if overlap:
                stats[species]["concordant"] += 1
            stats[species]["jaccard"].append(float(jaccard))

            entry[f"{species}_ortholog"] = {
                "stringId": orth_id,
                "gene": best.get("preferredName", ""),
                "score": best.get("score", 0),
                "n_orth_terms": len(orth_terms),
                "n_overlap": len(overlap),
                "jaccard": float(jaccard),
                "overlapping_terms": sorted(overlap),
            }

        results.append(entry)

    # Summary
    print(f"\n  === Concordance Summary ===")
    summary = {}
    for species in ["human", "mouse"]:
        s = stats[species]
        if s["total"] > 0:
            rate = s["concordant"] / s["total"]
            mean_j = np.mean(s["jaccard"])
            median_j = np.median(s["jaccard"])
            print(f"  {species}: {s['concordant']}/{s['total']} concordant "
                  f"({100*rate:.1f}%), Jaccard mean={mean_j:.3f} median={median_j:.3f}")
            summary[species] = {
                "n_orthologs_with_annotations": s["total"],
                "n_concordant": s["concordant"],
                "concordance_rate": float(rate),
                "mean_jaccard": float(mean_j),
                "median_jaccard": float(median_j),
            }
        else:
            summary[species] = {"n_orthologs_with_annotations": 0}

    # Save
    output = {
        "description": "Ortholog Cross-Validation (Step 71)",
        "n_yeast_proteins": len(selected_ids),
        "n_human_orthologs_found": n_human,
        "n_mouse_orthologs_found": n_mouse,
        "summary": summary,
        "results": results,
    }

    out_file = RESULTS / "ortholog_concordance.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {out_file}")

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return output


if __name__ == "__main__":
    main()
