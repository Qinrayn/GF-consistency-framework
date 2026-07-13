#!/usr/bin/env python3
"""
fix_mouse_go_annotations.py -- Rebuild mouse GO annotations from raw GAF
=========================================================================
The existing data/mouse_go_annotations.json contains DAG-propagated
annotations (mean 17.1 terms/protein), which inflates GO Jaccard and
confounds cross-species comparison. This script rebuilds the annotation
map from the raw MGI GAF file, keeping only direct (non-propagated)
annotations.

GAF format (tab-separated, version 2.2):
  Column 1: DB              (e.g. MGI)
  Column 2: DB_Object_ID    (e.g. MGI:1916787)
  Column 5: GO_ID           (e.g. GO:0005737)
  Column 9: Aspect          (F=MF, P=BP, C=CC)
  Column 4: Qualifier       (skip NOT, contributes_to, colocalizes_with)
  Column 7: Evidence_Code   (keep all; we can filter later)

We also need to map MGI IDs to Ensembl protein IDs (ENSMUSP) to match
the embedding node names. The existing mouse_go_annotations.json uses
ENSMUSP keys, so we need the alias mapping from STRING.

Output: data/mouse_go_annotations_direct.json
"""

from __future__ import annotations

import sys
import json
import gzip
import time
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import get_data_dir


def parse_gaf(gaf_path, aspect="P"):
    """Parse a GAF file, returning {db_object_id: set(go_ids)}.

    Only direct annotations (no DAG propagation).
    Filters to the specified aspect (P=BP, F=MF, C=CC).
    Skips NOT qualifiers.
    """
    annotations = defaultdict(set)

    with gzip.open(str(gaf_path), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            db = parts[0]
            db_object_id = parts[1]
            qualifier = parts[3] if len(parts) > 3 else ""
            go_id = parts[4]
            aspect_code = parts[8]

            # Filter by aspect
            if aspect_code != aspect:
                continue

            # Skip negative annotations
            if "NOT" in qualifier:
                continue

            annotations[db_object_id].add(go_id)

    return dict(annotations)


def load_string_aliases(taxon_id="10090"):
    """Load STRING alias file to map MGI/UniProt IDs to STRING protein IDs.

    STRING alias file format (tab-separated):
      column 1: STRING protein ID (e.g. 10090.ENSMUSP00000000001)
      column 2: alias
      column 3: source
    """
    data_dir = get_data_dir()
    alias_file = data_dir / f"{taxon_id}.protein.aliases.v11.5.txt.gz"

    # Build mapping: alias -> string_protein_id
    alias_to_string = {}

    with gzip.open(str(alias_file), "rt", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            string_id = parts[0]
            alias = parts[1]
            source = parts[2] if len(parts) > 2 else ""

            # Store all aliases; prefer Ensembl protein IDs
            if alias not in alias_to_string:
                alias_to_string[alias] = string_id

    return alias_to_string


def build_ensembl_to_go(annotations, alias_to_string):
    """Map GAF annotations (MGI/UniProt keys) to Ensembl protein IDs.

    The STRING alias file maps ENSMUSP IDs to MGI symbols and UniProt IDs.
    We reverse this: MGI_symbol -> ENSMUSP, then MGI_symbol's GO terms
    -> ENSMUSP's GO terms.
    """
    # Build reverse mapping: MGI symbol/UniProt -> ENSMUSP
    mgi_to_ensembl = {}
    for alias, string_id in alias_to_string.items():
        if "ENSMUSP" in string_id:
            # alias could be MGI:xxx, UniProt ID, or gene symbol
            mgi_to_ensembl[alias] = string_id.split(".")[-1]

    # Map GO annotations to Ensembl protein IDs
    ensembl_go = {}
    n_mapped = 0
    n_unmapped = 0

    for db_object_id, go_terms in annotations.items():
        # Try direct mapping
        ensembl_id = mgi_to_ensembl.get(db_object_id)

        if ensembl_id is None:
            # Try MGI: prefix variations
            for prefix in [db_object_id, f"MGI:{db_object_id}",
                          db_object_id.replace("MGI:", "")]:
                if prefix in mgi_to_ensembl:
                    ensembl_id = mgi_to_ensembl[prefix]
                    break

        if ensembl_id is None:
            # Try as gene symbol (MGI GAF uses symbols in column 2)
            if db_object_id in mgi_to_ensembl:
                ensembl_id = mgi_to_ensembl[db_object_id]

        if ensembl_id:
            if ensembl_id not in ensembl_go:
                ensembl_go[ensembl_id] = set()
            ensembl_go[ensembl_id].update(go_terms)
            n_mapped += 1
        else:
            n_unmapped += 1

    # Convert sets to lists
    ensembl_go = {k: sorted(v) for k, v in ensembl_go.items()}

    return ensembl_go, n_mapped, n_unmapped


def main():
    t_start = time.time()
    print("=" * 72)
    print("  Fix Mouse GO Annotations: Direct (non-propagated) from GAF")
    print("=" * 72)
    print()

    data_dir = get_data_dir()
    gaf_file = data_dir / "mgi.gaf.gz"

    if not gaf_file.exists():
        print(f"ERROR: GAF file not found: {gaf_file}")
        return

    # ----------------------------------------------------------------
    # Step 1: Parse raw GAF (BP terms only, matching yeast pipeline)
    # ----------------------------------------------------------------
    print("[1/4] Parsing MGI GAF (BP terms, direct annotations only) ...")
    t0 = time.time()
    annotations = parse_gaf(gaf_file, aspect="P")
    print(f"  Parsed {len(annotations)} unique gene IDs with BP annotations")
    term_counts = [len(v) for v in annotations.values()]
    import numpy as np
    print(f"  Terms per gene: mean={np.mean(term_counts):.1f}, "
          f"median={np.median(term_counts):.0f}, max={max(term_counts)}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ----------------------------------------------------------------
    # Step 2: Load STRING aliases for ID mapping
    # ----------------------------------------------------------------
    print("[2/4] Loading STRING aliases (mouse) ...")
    t0 = time.time()
    alias_to_string = load_string_aliases("10090")
    print(f"  Loaded {len(alias_to_string):,} alias mappings")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ----------------------------------------------------------------
    # Step 3: Map to Ensembl protein IDs
    # ----------------------------------------------------------------
    print("[3/4] Mapping GAF annotations to Ensembl protein IDs ...")
    t0 = time.time()
    ensembl_go, n_mapped, n_unmapped = build_ensembl_to_go(
        annotations, alias_to_string
    )
    print(f"  Mapped: {n_mapped}, Unmapped: {n_unmapped} "
          f"({n_mapped/(n_mapped+n_unmapped)*100:.1f}% success)")
    print(f"  Ensembl proteins with GO: {len(ensembl_go)}")

    if ensembl_go:
        term_counts_new = [len(v) for v in ensembl_go.values()]
        print(f"  Terms per protein (direct): mean={np.mean(term_counts_new):.1f}, "
              f"median={np.median(term_counts_new):.0f}, max={max(term_counts_new)}")
        print(f"  (vs propagated: mean=17.1, median=12)")

    # Add 10090. prefix to match embedding node names
    ensembl_go_prefixed = {f"10090.{k}": v for k, v in ensembl_go.items()}
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ----------------------------------------------------------------
    # Step 4: Save
    # ----------------------------------------------------------------
    print("[4/4] Saving ...")
    out_file = data_dir / "mouse_go_annotations_direct.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ensembl_go_prefixed, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_file}")
    print(f"  Entries: {len(ensembl_go_prefixed)}")

    # Compare with existing propagated annotations
    old_file = data_dir / "mouse_go_annotations.json"
    if old_file.exists():
        old_go = json.load(open(old_file))
        old_counts = [len(v) for v in old_go.values()]
        new_counts = [len(v) for v in ensembl_go_prefixed.values()]
        print()
        print("  Comparison (propagated vs direct):")
        print(f"    Proteins:      {len(old_go):,} -> {len(ensembl_go_prefixed):,}")
        print(f"    Mean terms:    {np.mean(old_counts):.1f} -> {np.mean(new_counts):.1f}")
        print(f"    Median terms:  {np.median(old_counts):.0f} -> {np.median(new_counts):.0f}")
        print(f"    Max terms:     {max(old_counts)} -> {max(new_counts)}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()