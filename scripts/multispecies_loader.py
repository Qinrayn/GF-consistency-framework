#!/usr/bin/env python3
"""
G-F Consistency Framework — Multi-Species Dataset Loader
=========================================================
Extends the pipeline beyond yeast to support arbitrary species with
STRING v11.5 PPI data and GO annotations.

Supported species (built-in):
- ``yeast`` (Saccharomyces cerevisiae, taxon 4932)
- ``human`` (Homo sapiens, taxon 9606)
- ``ecoli`` (Escherichia coli K-12, taxon 511145)
- ``mouse`` (Mus musculus, taxon 10090)
- ``fly`` (Drosophila melanogaster, taxon 7227)

Usage
-----
    from scripts.multispecies_loader import load_species_dataset

    G, nodes, go_map = load_species_dataset("human", data_dir="data/human")
"""

from __future__ import annotations

import json
import gzip
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Species registry
# ---------------------------------------------------------------------------

SPECIES_REGISTRY: dict[str, dict] = {
    "yeast": {
        "taxon_id": "4932",
        "string_prefix": "4932",
        "name": "Saccharomyces cerevisiae",
        "default_curated_nodes": 153,
        "go_db": "gene_association.sgd.gaf.gz",
        "default_min_score": 700,
    },
    "human": {
        "taxon_id": "9606",
        "string_prefix": "9606",
        "name": "Homo sapiens",
        "default_curated_nodes": None,
        "go_db": "gene_association.goa_human.gaf.gz",
        "default_min_score": 700,
    },
    "ecoli": {
        "taxon_id": "511145",
        "string_prefix": "511145",
        "name": "Escherichia coli K-12",
        "default_curated_nodes": None,
        "go_db": "gene_association.ecocyc.gaf.gz",
        "default_min_score": 400,
    },
    "mouse": {
        "taxon_id": "10090",
        "string_prefix": "10090",
        "name": "Mus musculus",
        "default_curated_nodes": None,
        "go_db": "gene_association.mgi.gaf.gz",
        "default_min_score": 700,
    },
    "fly": {
        "taxon_id": "7227",
        "string_prefix": "7227",
        "name": "Drosophila melanogaster",
        "default_curated_nodes": None,
        "go_db": "fb.gaf.gz",
        "default_min_score": 700,
    },
}


def register_species(species_key: str, config: dict) -> None:
    """Register a new species for the pipeline.

    Parameters
    ----------
    species_key : short identifier (e.g., "fly")
    config : dict with keys: taxon_id, string_prefix, name,
             default_curated_nodes, go_db, default_min_score
    """
    required = {"taxon_id", "string_prefix", "name"}
    missing = required - set(config.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    SPECIES_REGISTRY[species_key] = {
        "default_curated_nodes": None,
        "default_min_score": 700,
        **config,
    }
    logger.info("Registered species '%s' (%s)", species_key, config.get("name"))


def list_species() -> list[dict]:
    """Return a list of registered species with metadata."""
    result = []
    for key, meta in SPECIES_REGISTRY.items():
        result.append({
            "key": key,
            "name": meta["name"],
            "taxon_id": meta["taxon_id"],
            "has_curated": meta.get("default_curated_nodes") is not None,
        })
    return result


# ---------------------------------------------------------------------------
# STRING network loader (species-aware)
# ---------------------------------------------------------------------------

def load_string_network(
    species: str = "yeast",
    data_dir: Optional[Path] = None,
    min_score: Optional[int] = None,
    string_file: Optional[str] = None,
) -> nx.Graph:
    """Load a STRING PPI network for the given species.

    Parameters
    ----------
    species : species key (must be in registry)
    data_dir : directory containing STRING files
    min_score : minimum combined score (default from registry)
    string_file : explicit filename override
    """
    if species not in SPECIES_REGISTRY:
        raise ValueError(
            f"Unknown species '{species}'. Available: {list(SPECIES_REGISTRY.keys())}"
        )

    meta = SPECIES_REGISTRY[species]
    if min_score is None:
        min_score = meta.get("default_min_score", 700)

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir = Path(data_dir)

    if string_file:
        filepath = data_dir / string_file
    else:
        filepath = data_dir / f"{meta['string_prefix']}.protein.links.v11.5.txt.gz"

    if not filepath.exists():
        raise FileNotFoundError(
            f"STRING file not found: {filepath}\n"
            f"Download from: https://string-db.org/cgi/download?speciesId={meta['taxon_id']}"
        )

    G = nx.Graph()
    with gzip.open(str(filepath), "rt", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p1, p2, score = parts
            if int(score) >= min_score:
                # Strip species prefix
                p1_clean = p1.split(".", 1)[-1] if "." in p1 else p1
                p2_clean = p2.split(".", 1)[-1] if "." in p2 else p2
                G.add_edge(p1_clean, p2_clean)

    if G.number_of_nodes() == 0:
        logger.warning("No edges passed the score filter (%d)", min_score)
        return G

    # Largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    logger.info(
        "Loaded %s STRING network: %d nodes, %d edges (score >= %d)",
        meta["name"], G.number_of_nodes(), G.number_of_edges(), min_score,
    )
    return G


# ---------------------------------------------------------------------------
# GO annotation loader (GAF format)
# ---------------------------------------------------------------------------

def load_go_annotations_gaf(
    gaf_file: Path,
    gene_id_field: int = 2,
    go_field: int = 4,
    aspect_field: int = 8,
    aspects: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """Parse a GAF (Gene Annotation Format) file into a gene->GO map.

    Parameters
    ----------
    gaf_file : path to .gaf or .gaf.gz
    gene_id_field : column index for gene identifier
    go_field : column index for GO term
    aspect_field : column index for ontology aspect (P/F/C)
    aspects : filter by aspects (default: all)

    Returns
    -------
    dict mapping gene_id -> list of GO term IDs
    """
    gaf_file = Path(gaf_file)
    go_map: dict[str, list[str]] = {}

    opener = gzip.open if str(gaf_file).endswith(".gz") else open
    with opener(str(gaf_file), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < max(gene_id_field, go_field, aspect_field) + 1:
                continue

            aspect = parts[aspect_field]
            if aspects and aspect not in aspects:
                continue

            gene_id = parts[gene_id_field]
            go_term = parts[go_field]

            if gene_id not in go_map:
                go_map[gene_id] = []
            if go_term not in go_map[gene_id]:
                go_map[gene_id].append(go_term)

    logger.info("Loaded GO annotations for %d genes from %s", len(go_map), gaf_file.name)
    return go_map


def load_go_annotations_json(json_file: Path) -> dict[str, list[str]]:
    """Load pre-processed GO annotations from a JSON file."""
    with open(json_file, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Unified species dataset loader
# ---------------------------------------------------------------------------

def load_species_dataset(
    species: str = "yeast",
    data_dir: Optional[Path] = None,
    min_score: Optional[int] = None,
    go_file: Optional[str] = None,
    go_aspects: Optional[list[str]] = None,
) -> tuple[nx.Graph, list[str], dict]:
    """Load a complete dataset (network + GO) for a given species.

    Returns
    -------
    (graph, sorted_nodes, gene_go_map)
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir = Path(data_dir)

    # Load network
    G = load_string_network(species, data_dir, min_score)

    # Load GO annotations
    meta = SPECIES_REGISTRY[species]
    if go_file:
        go_path = data_dir / go_file
    else:
        go_path = data_dir / meta["go_db"]

    if str(go_path).endswith(".json"):
        go_map = load_go_annotations_json(go_path)
    elif go_path.exists():
        go_map = load_go_annotations_gaf(go_path, aspects=go_aspects)
    else:
        raise FileNotFoundError(
            f"GO annotation file not found: {go_path}\n"
            f"Provide a GAF file or pre-processed JSON."
        )

    # Intersect network nodes with GO annotations
    valid_nodes = sorted(set(G.nodes()) & set(go_map.keys()))
    G = G.subgraph(valid_nodes).copy()

    logger.info(
        "%s dataset: %d nodes with GO annotations, %d edges",
        meta["name"], len(valid_nodes), G.number_of_edges(),
    )
    return G, valid_nodes, go_map


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def main():
    """Validate multispecies loader by listing registered species.

    For each registered species, checks whether the required data files
    exist and reports network/annotation availability.
    """
    import json
    from pathlib import Path
    from utils import get_data_dir

    data_dir = get_data_dir()
    print("Registered species and data availability:")
    print("-" * 60)

    for name, meta in SPECIES_REGISTRY.items():
        species_dir = data_dir / name
        has_data = species_dir.exists()
        string_file = species_dir / f"{meta['string_prefix']}.protein.links.v11.5.txt.gz" \
            if has_data else None
        has_string = string_file is not None and string_file.exists()

        status_parts = []
        if has_data:
            status_parts.append(f"data dir: {species_dir}")
        else:
            status_parts.append("data dir: NOT FOUND")
        status_parts.append(f"STRING file: {'YES' if has_string else 'NO'}")

        status = " | ".join(status_parts)
        tag = " [available]" if has_string else ""
        print(f"  {name:10s} ({meta['name']:30s}) {status}{tag}")

    print()
    print("To run the pipeline on a different species:")
    print("  python run_all_analysis.py --species human --run-human")
    print()
    print("Required data files should be placed in data/{species}/")
    print("See README for download instructions.")
