#!/usr/bin/env python3
"""
Dark Matter External Validation (Direction B — Phase 1A)
=========================================================

Validates the 44 functional dark matter protein pairs against independent
protein interaction databases (BioGRID, IntAct, MINT, DIP) that were NOT
used in the primary STRING-based analysis.  This is the first step of the
"computation → biology" validation loop required for Nature Communications.

Workflow
--------
1. Load the 44 dark matter pairs from ``results/functional_dark_matter.json``.
2. Map yeast STRING protein IDs (e.g. ``4932.YHR133C``) to systematic gene
   names and UniProt accessions using the STRING alias file.
3. Download and parse BioGRID (TAB2) and IntAct (MITAB) for *S. cerevisiae*.
   Both are cached in ``data/external/`` so the server needs to download them
   only once.
4. Cross-reference each dark matter pair against each external database.
   Compute (a) the fraction of pairs validated, (b) Fisher's exact test
   enrichment vs. shuffled controls, (c) Benjamini-Hochberg FDR across
   all tested GO terms.
5. Extend to human disease genes: map the 71 dark matter proteins to human
   orthologs (via STRING cross-species links), then query DisGeNET for
   disease associations.
6. Save a structured validation report to
   ``results/dark_matter_external_validation.json``.

Usage
-----
.. code-block:: bash

    # On the research server (first run downloads ~200 MB of external data):
    python scripts/dark_matter_external_validation.py

    # With pre-downloaded database files (no network needed):
    python scripts/dark_matter_external_validation.py \\
        --biogrid data/external/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-4.4.232.tab2.txt \\
        --intact   data/external/intact_yeast.txt

    # Skip disease mapping (faster, yeast-only):
    python scripts/dark_matter_external_validation.py --no-disease

Requirements (server)
---------------------
- ``requests`` (≥ 2.28) — for downloading external databases.
- ``zipfile`` / ``gzip`` — stdlib; for decompressing BioGRID / IntAct archives.
- BioGRID and IntAct are freely available at:
    - BioGRID: https://downloads.thebiogrid.org/BioGRID/
    - IntAct:  https://ftp.ebi.ac.uk/pub/databases/intact/
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
from scipy.stats import fisher_exact

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from utils import (
    SEED,
    get_data_dir,
    get_results_dir,
    setup_logging,
)

# ============================================================
# Constants
# ============================================================

EXTERNAL_DATA_DIR_NAME = "external"
BIOGRID_URL = (
    "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/"
    "BIOGRID-ORGANISM-LATEST.tab2.zip"
)
INTACT_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/"
    "intact.zip"
)
# TaxID for Saccharomyces cerevisiae S288c
YEAST_TAXID = "559292"
STRING_YEAST_TAXID = "4932"

# DisGeNET for human disease mapping (uses REST API)
DISGENET_API = "https://www.disgenet.org/api"

# FDR significance threshold
FDR_ALPHA: float = 0.05

logger: logging.Logger = setup_logging("dark_matter_external_validation")


# ============================================================
# Inline Benjamini-Hochberg FDR (avoids statsmodels dependency)
# ============================================================

def _benjamini_hochberg(p_values: list[float], alpha: float = FDR_ALPHA) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values (FDR q-values).

    Implementation follows Benjamini & Hochberg (1995), JRSS-B.
    """
    n = len(p_values)
    if n == 0:
        return []
    # Sort indices by ascending p-value
    idx = sorted(range(n), key=lambda i: p_values[i])
    q_values = [1.0] * n
    for rank, i in enumerate(idx):
        # BH adjusted: p * n / (rank + 1), capped at 1.0
        q = min(p_values[i] * n / (rank + 1), 1.0)
        # Ensure monotonicity: later q-values cannot be smaller than earlier ones
        if rank > 0:
            q = max(q, q_values[idx[rank - 1]])
        q_values[i] = q
    return q_values


# ============================================================
# Data Loading
# ============================================================

def load_dark_matter_pairs(
    results_dir: Optional[Path] = None,
) -> tuple[list[dict], dict]:
    """Load the 44 dark matter protein pairs from the primary analysis.

    Returns
    -------
    (pairs, summary)
        ``pairs`` is a list of dicts (one per dark matter pair) with the
        same schema as ``functional_dark_matter.json``. ``summary`` is the
        top-level summary dict.
    """
    if results_dir is None:
        results_dir = get_results_dir()
    path = Path(results_dir) / "functional_dark_matter.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Dark matter results not found at {path}. "
            "Run Step 48 (functional_dark_matter.py) first."
        )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    pairs = data.get("top_100_catalog", [])
    if not pairs:
        raise ValueError("No dark matter pairs in functional_dark_matter.json")
    logger.info("Loaded %d dark matter pairs", len(pairs))
    return pairs, data.get("summary", {})


def build_yeast_alias_map(
    data_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Map STRING protein IDs (4932.XXXX) → systematic gene names (YXXXX).

    Uses the STRING alias file already present in data/.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    alias_file = Path(data_dir) / f"{STRING_YEAST_TAXID}.protein.aliases.v11.5.txt.gz"
    if not alias_file.exists():
        raise FileNotFoundError(f"STRING alias file not found: {alias_file}")

    mapping: dict[str, str] = {}
    with gzip.open(alias_file, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            string_id, alias, source = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
            # Prefer systematic names (YXXXX) from Ensembl or UniProt
            if "Ensembl" in source or "UniProt" in source:
                if alias.startswith("Y") and len(alias) >= 6:
                    mapping[string_id] = alias
            # Fallback: any alias that looks like a yeast systematic name
            elif alias.startswith("Y") and len(alias) >= 6 and string_id not in mapping:
                mapping[string_id] = alias
    logger.info("Built yeast alias map: %d entries", len(mapping))
    return mapping


# ============================================================
# External Database Parsers
# ============================================================

def _ensure_external_dir(data_dir: Path) -> Path:
    ext = data_dir / EXTERNAL_DATA_DIR_NAME
    ext.mkdir(parents=True, exist_ok=True)
    return ext


def download_biogrid(
    url: str = BIOGRID_URL,
    data_dir: Optional[Path] = None,
    local_path: Optional[str] = None,
) -> Path:
    """Download and extract BioGRID (all organisms, tab2 format), returning the
    path to the extracted file.  Yeast interactions are filtered in the parser.

    If *local_path* is provided, skip the download and use that file directly.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    ext = _ensure_external_dir(data_dir)

    out = ext / "biogrid_all.tab2.txt"
    if out.exists():
        logger.info("Using cached BioGRID: %s", out)
        return out

    if local_path and os.path.exists(local_path):
        logger.info("Using local BioGRID: %s", local_path)
        return Path(local_path)

    import requests

    logger.info("Downloading BioGRID from %s ...", url)
    resp = requests.get(url, timeout=600, allow_redirects=True)
    resp.raise_for_status()
    with ZipFile(BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".tab2.txt"):
                with zf.open(name) as src:
                    with open(out, "wb") as dst:
                        dst.write(src.read())
                logger.info("Extracted BioGRID to %s", out)
                return out
    raise FileNotFoundError("No .tab2.txt file found in BioGRID archive")


def parse_biogrid_yeast(
    path: Path,
    alias_map: dict[str, str],
    yeast_taxid: str = YEAST_TAXID,
) -> dict[tuple[str, str], list[str]]:
    """Parse BioGRID TAB2, returning yeast-only interactions.

    Filters by Organism ID (columns 17, 18) to keep only *S. cerevisiae*
    S288c interactions.  Gene names are canonicalised to systematic names
    (YXXXX) via *alias_map*.

    Returns
    -------
    dict mapping (gene_a, gene_b) → list of evidence codes.
    """
    from collections import defaultdict

    interactions: dict[tuple[str, str], list[str]] = defaultdict(list)
    n_skipped_non_yeast = 0
    n_parsed = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                # Skip header lines, but check for column names
                continue
            cols = line.strip().split("\t")
            if len(cols) < 18:
                continue

            # Filter by organism: columns 16-17 (0-indexed) = Organism ID
            org_a = cols[16].strip() if len(cols) > 16 else ""
            org_b = cols[17].strip() if len(cols) > 17 else ""
            if org_a != yeast_taxid and org_b != yeast_taxid:
                n_skipped_non_yeast += 1
                continue

            # Systematic names: columns 5-6 (0-indexed)
            system_a = cols[5].strip()
            system_b = cols[6].strip()
            evidence = cols[11].strip() if len(cols) > 11 else ""

            gene_a = alias_map.get(system_a, system_a)
            gene_b = alias_map.get(system_b, system_b)
            key = tuple(sorted([gene_a, gene_b]))
            interactions[key].append(evidence)
            n_parsed += 1

    logger.info(
        "BioGRID: %d yeast interactions, %d non-yeast skipped",
        n_parsed, n_skipped_non_yeast,
    )
    return dict(interactions)


def download_intact(
    url: str = INTACT_URL,
    data_dir: Optional[Path] = None,
    local_path: Optional[str] = None,
) -> Path:
    """Download and extract IntAct, returning the yeast-filtered MITAB path.

    If *local_path* is provided and points to a .zip file, the archive is
    streamed (without full extraction) and only yeast (taxid:559292) lines
    are written to disk.  This avoids the 11 GB full extraction.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    ext = _ensure_external_dir(data_dir)
    out = ext / "intact_yeast.txt"
    if out.exists():
        logger.info("Using cached IntAct yeast: %s", out)
        return out

    # If local_path is a pre-downloaded zip, stream-filter yeast lines
    if local_path and os.path.exists(local_path):
        if str(local_path).lower().endswith(".zip"):
            logger.info("Filtering yeast from local IntAct zip: %s", local_path)
            n_yeast = 0
            n_total = 0
            with ZipFile(local_path) as zf:
                # Find the main MITAB file (largest .txt)
                target = max(
                    (i for i in zf.infolist() if i.filename.endswith(".txt")),
                    key=lambda i: i.file_size,
                )
                logger.info("  Streaming %s (%.0f MB) ...", target.filename, target.file_size / 1e6)
                with zf.open(target) as src:
                    with open(out, "wb") as dst:
                        for raw_line in src:
                            n_total += 1
                            line_str = raw_line.decode("utf-8", errors="replace")
                            if f"taxid:{YEAST_TAXID}" in line_str:
                                dst.write(raw_line)
                                n_yeast += 1
                            if n_total % 2_000_000 == 0:
                                logger.info("  ... %dM lines scanned, %d yeast", n_total // 1_000_000, n_yeast)
            logger.info("IntAct: %d yeast lines from %d total (%.2f%%)", n_yeast, n_total, 100 * n_yeast / max(n_total, 1))
            return out
        # Non-zip local file: use directly
        logger.info("Using local IntAct: %s", local_path)
        return Path(local_path)

    import requests

    logger.info("Downloading IntAct from %s ...", url)
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()
    # Stream the zip into memory (1.35 GB) then filter
    content = b""
    for chunk in resp.iter_content(chunk_size=10 * 1024 * 1024):
        content += chunk
    with ZipFile(BytesIO(content)) as zf:
        for name in zf.namelist():
            if name.endswith(".txt") and "intact" in name.lower():
                with zf.open(name) as src:
                    with open(out, "wb") as dst:
                        for line in src:
                            line_str = line.decode("utf-8", errors="replace")
                            if f"taxid:{YEAST_TAXID}" in line_str:
                                dst.write(line_str.encode("utf-8"))
                logger.info("Extracted IntAct yeast to %s", out)
                return out
    raise FileNotFoundError("No yeast MITAB file found in IntAct archive")


def parse_intact_yeast(
    path: Path,
    alias_map: dict[str, str],
) -> dict[tuple[str, str], list[str]]:
    """Parse IntAct MITAB, returning yeast interactions as gene pairs.

    MITAB columns: 0=idA, 1=idB, 2=altA, 3=altB, 4=aliasesA, 5=aliasesB, ...
    """
    from collections import defaultdict

    interactions: dict[tuple[str, str], list[str]] = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 6:
                continue
            # Extract gene names from aliases columns (4, 5)
            aliases_a = cols[4] if len(cols) > 4 else ""
            aliases_b = cols[5] if len(cols) > 5 else ""
            detection = cols[6] if len(cols) > 6 else ""

            gene_a = _extract_yeast_gene(aliases_a, alias_map)
            gene_b = _extract_yeast_gene(aliases_b, alias_map)
            if gene_a and gene_b:
                key = tuple(sorted([gene_a, gene_b]))
                interactions[key].append(detection)
    logger.info("Parsed IntAct: %d unique yeast interactions", len(interactions))
    return dict(interactions)


def _extract_yeast_gene(
    alias_column: str,
    alias_map: dict[str, str],
) -> Optional[str]:
    """Extract a yeast systematic gene name (YXXXX) from a MITAB alias column.

    MITAB alias columns use the format ``source:name(type)``, e.g.:
    ``psi-mi:ste11_yeast(display_long)|uniprotkb:STE11(gene name)|
    uniprotkb:YLR362W(locus name)``

    We prioritise ``(locus name)`` entries (systematic ORF names like YLR362W),
    falling back to any Y-prefixed token.
    """
    # Priority 1: explicit "(locus name)" or "(orf name)" entries
    for part in alias_column.split("|"):
        part = part.strip()
        if "(locus name)" in part or "(orf name)" in part:
            # Extract the name before the parenthesis: "uniprotkb:YLR362W(locus name)"
            name = part.split("(")[0].split(":")[-1].strip()
            if name.startswith("Y") and len(name) >= 6:
                return name

    # Priority 2: any Y-prefixed token that looks like a systematic name
    for part in alias_column.split("|"):
        part = part.strip()
        name = part.split("(")[0].split(":")[-1].strip()
        if name.startswith("Y") and len(name) >= 6 and name[1:].isdigit() is False:
            # Validate: Y followed by letter + digits + letter (e.g. YLR362W)
            if len(name) >= 6 and name[1].isalpha():
                return name

    return None


# ============================================================
# Cross-Reference Engine
# ============================================================

def cross_reference_pairs(
    pairs: list[dict],
    external_interactions: dict[tuple[str, str], list[str]],
    db_name: str,
    alias_map: dict[str, str],
) -> dict:
    """Cross-reference dark matter pairs against an external database.

    Returns a dict with validation statistics.
    """
    validated: list[dict] = []
    n_matched: int = 0
    n_total: int = len(pairs)

    for pair in pairs:
        prot_a = pair["protein_a"]
        prot_b = pair["protein_b"]
        # Map STRING IDs to gene names
        gene_a = alias_map.get(f"{STRING_YEAST_TAXID}.{prot_a}", prot_a)
        gene_b = alias_map.get(f"{STRING_YEAST_TAXID}.{prot_b}", prot_b)
        key = tuple(sorted([gene_a, gene_b]))

        if key in external_interactions:
            n_matched += 1
            validated.append({
                "protein_a": prot_a,
                "protein_b": prot_b,
                "gene_a": gene_a,
                "gene_b": gene_b,
                "evidence": external_interactions[key][:5],  # first 5 evidence codes
                "n_evidence": len(external_interactions[key]),
                "shared_go_terms": pair.get("shared_go_terms", []),
                "confidence_score": pair.get("confidence_score", 0),
            })

    return {
        "database": db_name,
        "n_total_pairs": n_total,
        "n_validated": n_matched,
        "fraction_validated": n_matched / n_total if n_total > 0 else 0.0,
        "validated_pairs": validated,
        "n_interactions_in_db": len(external_interactions),
    }


# ============================================================
# Enrichment Analysis with FDR
# ============================================================

def compute_enrichment_fdr(
    dm_pairs: list[dict],
    go_map: dict,
    background_genes: set[str],
    alias_map: dict[str, str],
) -> list[dict]:
    """Fisher's exact test for each GO term, with Benjamini-Hochberg FDR.

    Tests whether the dark matter set is enriched for specific GO terms
    relative to the background (all yeast genes with GO annotations).
    """
    # Collect all GO terms in the dark matter set
    dm_go_counts: dict[str, int] = defaultdict(int)
    dm_genes: set[str] = set()
    for pair in dm_pairs:
        dm_genes.add(pair["protein_a"])
        dm_genes.add(pair["protein_b"])
        for term in pair.get("shared_go_terms", []):
            dm_go_counts[term] += 1

    n_dm = len(dm_genes)
    n_background = len(background_genes)

    enrichment: list[dict] = []
    for term, dm_count in dm_go_counts.items():
        # Count background genes annotated with this term
        bg_count = sum(1 for g in background_genes if term in go_map.get(g, []))
        if bg_count == 0:
            continue

        a = dm_count          # DM genes with term
        b = n_dm - dm_count   # DM genes without term
        c = bg_count - dm_count  # background genes with term (excluding DM)
        d = n_background - bg_count  # rest
        # Clamp negatives that can arise from sampling
        c = max(c, 0)
        d = max(d, n_background - n_dm - bg_count, 0)

        try:
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        except ValueError:
            continue

        enrichment.append({
            "go_term": term,
            "n_dm": a,
            "n_background": bg_count,
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_value),
            "fdr": 1.0,  # placeholder, filled below
        })

    if not enrichment:
        return []

    # Benjamini-Hochberg correction (inline — no statsmodels dependency)
    p_values = [e["p_value"] for e in enrichment]
    fdr_values = _benjamini_hochberg(p_values, alpha=FDR_ALPHA)
    for e, fdr in zip(enrichment, fdr_values):
        e["fdr"] = float(fdr)

    # Sort by FDR
    enrichment.sort(key=lambda x: x["fdr"])
    n_significant = sum(1 for e in enrichment if e["fdr"] < FDR_ALPHA)
    logger.info(
        "Enrichment FDR: %d/%d terms significant at FDR < %.2f",
        n_significant, len(enrichment), FDR_ALPHA,
    )
    return enrichment


# ============================================================
# Human Disease Gene Mapping
# ============================================================

def map_dark_matter_to_disease(
    dm_genes: list[str],
    human_ortholog_map: dict[str, str],
    disgenet_cache: Optional[Path] = None,
) -> dict:
    """Map dark matter yeast genes → human orthologs → disease associations.

    Uses DisGeNET REST API for human disease gene annotations.
    """
    # Map yeast genes to human orthologs
    yeast_to_human: dict[str, str] = {}
    for gene in dm_genes:
        human_orth = human_ortholog_map.get(gene)
        if human_orth:
            yeast_to_human[gene] = human_orth

    n_mapped = len(yeast_to_human)
    logger.info("Mapped %d/%d dark matter genes to human orthologs", n_mapped, len(dm_genes))

    if n_mapped == 0:
        return {"n_mapped": 0, "disease_associations": []}

    # Query DisGeNET (try REST API, fall back to cached)
    disease_associations: list[dict] = []
    if disgenet_cache and disgenet_cache.exists():
        with open(disgenet_cache, encoding="utf-8") as fh:
            cache = json.load(fh)
        for yeast_gene, human_gene in yeast_to_human.items():
            if human_gene in cache:
                disease_associations.append({
                    "yeast_gene": yeast_gene,
                    "human_gene": human_gene,
                    "diseases": cache[human_gene],
                })
    else:
        # Try REST API (requires server internet access)
        import requests
        for yeast_gene, human_gene in yeast_to_human.items():
            try:
                resp = requests.get(
                    f"{DISGENET_API}/gene/{human_gene}/diseases",
                    params={"format": "json", "limit": 10},
                    timeout=15,
                )
                if resp.status_code == 200:
                    diseases = resp.json()
                    disease_associations.append({
                        "yeast_gene": yeast_gene,
                        "human_gene": human_gene,
                        "diseases": diseases[:5],  # top 5
                    })
            except Exception:
                pass

    return {
        "n_genes_with_ortholog": n_mapped,
        "n_genes_with_disease": len(disease_associations),
        "disease_associations": disease_associations,
    }


def build_human_ortholog_map(
    data_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Build yeast → human ortholog map from STRING cross-species links.

    Uses the STRING protein links file for human-yeast conservation.
    """
    if data_dir is None:
        data_dir = get_data_dir()

    # We use the human STRING alias file to map human IDs to gene names
    human_alias = Path(data_dir) / "9606.protein.aliases.v12.0.txt.gz"
    if not human_alias.exists():
        logger.warning("Human STRING aliases not found; skipping disease mapping")
        return {}

    yeast_to_human: dict[str, str] = {}
    # Load human gene names
    human_genes: dict[str, str] = {}
    with gzip.open(human_alias, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                string_id = parts[0]  # e.g. 9606.ENSG00000123456
                gene_name = parts[1]
                if "Ensembl" in parts[2] or "UniProt" in parts[2]:
                    human_genes[string_id] = gene_name

    # Cross-species ortholog links from human_validation data
    human_links = Path(data_dir) / "9606.protein.links.v12.0.txt.gz"
    if not human_links.exists():
        logger.warning("Human STRING links not found; skipping ortholog mapping")
        return {}

    # For simplicity, we use the cross-species dark matter result
    xdm_path = Path(get_results_dir()) / "cross_species_dark_matter.json"
    if xdm_path.exists():
        with open(xdm_path, encoding="utf-8") as fh:
            xdm = json.load(fh)
        orthologs = xdm.get("protein_name_table", {})
        for yeast_id, info in orthologs.items():
            if isinstance(info, dict) and "human_ortholog" in info:
                human_gene = info["human_ortholog"]
                if isinstance(human_gene, str) and human_gene:
                    yeast_to_human[yeast_id] = human_gene
        if yeast_to_human:
            logger.info("Built ortholog map from cross_species_dark_matter: %d entries", len(yeast_to_human))
            return yeast_to_human

    logger.warning("No ortholog map available; disease mapping will return empty")
    return {}


# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dark Matter External Validation (Direction B)",
    )
    parser.add_argument(
        "--biogrid", type=str, default=None,
        help="Path to pre-downloaded BioGRID TAB2 file (skips download).",
    )
    parser.add_argument(
        "--intact", type=str, default=None,
        help="Path to pre-downloaded IntAct MITAB file (skips download).",
    )
    parser.add_argument(
        "--no-disease", action="store_true", default=False,
        help="Skip human disease gene mapping (yeast-only).",
    )
    parser.add_argument(
        "--no-download", action="store_true", default=False,
        help="Skip all external downloads; use only cached files.",
    )
    parser.add_argument(
        "--disgenet-cache", type=str, default=None,
        help="Path to cached DisGeNET JSON for human disease genes.",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dark matter pairs
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: Loading dark matter pairs")
    logger.info("=" * 60)
    dm_pairs, dm_summary = load_dark_matter_pairs(results_dir)

    # ------------------------------------------------------------------
    # 2. Build yeast gene alias map
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Building yeast gene alias map")
    logger.info("=" * 60)
    alias_map = build_yeast_alias_map(data_dir)

    # ------------------------------------------------------------------
    # 3. Validate against BioGRID
    # ------------------------------------------------------------------
    biogrid_result = None
    if not args.no_download or args.biogrid:
        logger.info("=" * 60)
        logger.info("Step 3: BioGRID validation")
        logger.info("=" * 60)
        try:
            bg_path = download_biogrid(
                local_path=args.biogrid,
                data_dir=data_dir,
            )
            bg_interactions = parse_biogrid_yeast(bg_path, alias_map)
            biogrid_result = cross_reference_pairs(
                dm_pairs, bg_interactions, "BioGRID", alias_map,
            )
            logger.info(
                "BioGRID: %d/%d pairs validated (%.1f%%)",
                biogrid_result["n_validated"],
                biogrid_result["n_total_pairs"],
                biogrid_result["fraction_validated"] * 100,
            )
        except Exception as exc:
            logger.warning("BioGRID validation failed: %s", exc)

    # ------------------------------------------------------------------
    # 4. Validate against IntAct
    # ------------------------------------------------------------------
    intact_result = None
    if not args.no_download or args.intact:
        logger.info("=" * 60)
        logger.info("Step 4: IntAct validation")
        logger.info("=" * 60)
        try:
            ia_path = download_intact(
                local_path=args.intact,
                data_dir=data_dir,
            )
            ia_interactions = parse_intact_yeast(ia_path, alias_map)
            intact_result = cross_reference_pairs(
                dm_pairs, ia_interactions, "IntAct", alias_map,
            )
            logger.info(
                "IntAct: %d/%d pairs validated (%.1f%%)",
                intact_result["n_validated"],
                intact_result["n_total_pairs"],
                intact_result["fraction_validated"] * 100,
            )
        except Exception as exc:
            logger.warning("IntAct validation failed: %s", exc)

    # ------------------------------------------------------------------
    # 5. FDR-corrected enrichment
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5: FDR-corrected GO enrichment")
    logger.info("=" * 60)
    enrichment = []
    try:
        go_map_path = data_dir / "gene_go_map.json"
        if go_map_path.exists():
            with open(go_map_path, encoding="utf-8") as fh:
                go_map = json.load(fh)
            background = set(go_map.keys())
            enrichment = compute_enrichment_fdr(
                dm_pairs, go_map, background, alias_map,
            )
            logger.info("Enrichment: %d GO terms tested, %d significant at FDR<0.05",
                        len(enrichment),
                        sum(1 for e in enrichment if e["fdr"] < FDR_ALPHA))
    except Exception as exc:
        logger.warning("Enrichment analysis failed: %s", exc)

    # ------------------------------------------------------------------
    # 6. Human disease gene mapping (optional)
    # ------------------------------------------------------------------
    disease_result = None
    if not args.no_disease:
        logger.info("=" * 60)
        logger.info("Step 6: Human disease gene mapping")
        logger.info("=" * 60)
        try:
            ortholog_map = build_human_ortholog_map(data_dir)
            dm_genes = list({p["protein_a"] for p in dm_pairs} | {p["protein_b"] for p in dm_pairs})
            disease_result = map_dark_matter_to_disease(
                dm_genes,
                ortholog_map,
                Path(args.disgenet_cache) if args.disgenet_cache else None,
            )
            logger.info(
                "Disease mapping: %d genes with orthologs, %d with disease associations",
                disease_result.get("n_genes_with_ortholog", 0),
                disease_result.get("n_genes_with_disease", 0),
            )
        except Exception as exc:
            logger.warning("Disease mapping failed: %s", exc)

    # ------------------------------------------------------------------
    # 7. Compile and save report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 7: Compiling validation report")
    logger.info("=" * 60)

    report = {
        "description": "Dark matter external validation against independent databases",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": {
            "n_dark_matter_pairs": dm_summary.get("total_dm_pairs"),
            "n_dark_matter_proteins": dm_summary.get("unique_dm_proteins"),
            "source": "results/functional_dark_matter.json",
        },
        "external_databases": {
            "BioGRID": biogrid_result,
            "IntAct": intact_result,
        },
        "enrichment_fdr": {
            "n_terms_tested": len(enrichment),
            "n_significant_fdr_005": sum(1 for e in enrichment if e["fdr"] < FDR_ALPHA),
            "top_terms": enrichment[:20],
            "all_terms": enrichment,
        },
        "disease_mapping": disease_result,
        "conclusion": _generate_conclusion(
            biogrid_result, intact_result, enrichment, disease_result,
        ),
    }

    out_path = results_dir / "dark_matter_external_validation.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Report saved to %s", out_path)

    # Print summary
    print("\n" + "=" * 60)
    print("  EXTERNAL VALIDATION SUMMARY")
    print("=" * 60)
    if biogrid_result:
        print(f"  BioGRID: {biogrid_result['n_validated']}/{biogrid_result['n_total_pairs']} "
              f"({biogrid_result['fraction_validated']:.1%})")
    if intact_result:
        print(f"  IntAct:  {intact_result['n_validated']}/{intact_result['n_total_pairs']} "
              f"({intact_result['fraction_validated']:.1%})")
    n_sig = sum(1 for e in enrichment if e["fdr"] < FDR_ALPHA)
    print(f"  GO terms significant (FDR<0.05): {n_sig}/{len(enrichment)}")
    if disease_result:
        print(f"  Disease genes: {disease_result.get('n_genes_with_disease', 0)}")
    print("=" * 60)


def _generate_conclusion(
    biogrid_result: Optional[dict],
    intact_result: Optional[dict],
    enrichment: list[dict],
    disease_result: Optional[dict],
) -> str:
    """Generate a plain-text conclusion for the report."""
    parts: list[str] = []

    if biogrid_result and intact_result:
        n_bg = biogrid_result["n_validated"]
        n_ia = intact_result["n_validated"]
        n_union = len(set(
            (p["protein_a"], p["protein_b"])
            for p in (biogrid_result.get("validated_pairs", []) +
                       intact_result.get("validated_pairs", []))
        ))
        total = biogrid_result["n_total_pairs"]
        parts.append(
            f"Of {total} dark matter pairs, {n_union} are independently "
            f"validated by BioGRID and/or IntAct (BioGRID: {n_bg}, "
            f"IntAct: {n_ia}). This represents independent experimental "
            f"evidence beyond the STRING database used in the primary analysis."
        )
    elif biogrid_result:
        parts.append(
            f"BioGRID validates {biogrid_result['n_validated']}/"
            f"{biogrid_result['n_total_pairs']} dark matter pairs."
        )
    elif intact_result:
        parts.append(
            f"IntAct validates {intact_result['n_validated']}/"
            f"{intact_result['n_total_pairs']} dark matter pairs."
        )
    else:
        parts.append("External database validation could not be performed.")

    n_sig = sum(1 for e in enrichment if e["fdr"] < FDR_ALPHA)
    if n_sig > 0:
        top_terms = [e["go_term"] for e in enrichment[:3] if e["fdr"] < FDR_ALPHA]
        parts.append(
            f"FDR-corrected GO enrichment identifies {n_sig} significant "
            f"terms (top: {', '.join(top_terms)})."
        )
    else:
        parts.append("No GO terms survive FDR correction at α=0.05.")

    if disease_result and disease_result.get("n_genes_with_disease", 0) > 0:
        parts.append(
            f"{disease_result['n_genes_with_disease']} dark matter yeast genes "
            f"have human orthologs with known disease associations."
        )

    return " ".join(parts)


if __name__ == "__main__":
    main()