#!/usr/bin/env python3
"""
Phase 10A: Mouse Data Preparation
===================================
Downloads and preprocesses Mus musculus PPI + GO annotations for
cross-species G-F consistency analysis.

Data sources:
  - STRING v11.5 PPI (taxon 10090): ~81 MB compressed
  - MGI GO annotations (GAF format): ~13 MB compressed
  - STRING aliases (gene symbol mapping): ~13 MB compressed

ID mapping strategy:
  STRING protein IDs (e.g. 10090.ENSMUSP00000000001) are mapped to gene
  symbols via the Ensembl_MGI alias source, which provides the bridge
  to MGI GAF gene symbols (column 2). The resulting GO annotation JSON
  is keyed by STRING protein IDs (with species prefix).

Outputs:
  - data/10090.protein.links.v11.5.txt.gz  (raw STRING)
  - data/mouse_go_annotations.json         (STRING_ID -> [GO terms])
  - data/mouse_network_info.json           (metadata)
"""

import gzip
import json
import sys
from pathlib import Path

import networkx as nx

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

DATA_DIR = SCRIPT_DIR.parent / "data"

STRING_URL = "https://stringdb-static.org/download/protein.links.v11.5/10090.protein.links.v11.5.txt.gz"
MGI_GAF_URL = "https://current.geneontology.org/annotations/mgi.gaf.gz"
ALIAS_URL = "https://stringdb-static.org/download/protein.aliases.v11.5/10090.protein.aliases.v11.5.txt.gz"

MIN_SCORE = 700
BANNER = "=" * 70


def download_file(url, dest, desc=""):
    """Download a file with progress reporting. Skip if already exists."""
    dest = Path(dest)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True

    import requests

    print(f"  Downloading {desc or dest.name}...")
    print(f"    URL: {url}")

    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  [error] Download failed: {e}")
        return False

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 256

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = 100 * downloaded / total
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                print(f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    print()
    print(f"  [done] Saved {dest.name} ({downloaded / (1024*1024):.1f} MB)")
    return True


def load_string_network(string_file, min_score=MIN_SCORE):
    """Load mouse STRING network, keeping species prefix on protein IDs."""
    G = nx.Graph()
    with gzip.open(str(string_file), "rt", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p1, p2, score = parts
            if int(score) >= min_score:
                G.add_edge(p1, p2)

    if G.number_of_nodes() == 0:
        print("  [warning] No edges passed score filter")
        return G

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    return G


def build_alias_map(alias_file):
    """Build gene_symbol -> STRING_protein_ID mapping from Ensembl_MGI source.

    The Ensembl_MGI alias source maps STRING protein IDs to MGI gene symbols
    (e.g., 10090.ENSMUSP00000000001 -> Gnai3). These gene symbols match the
    MGI GAF DB_Object_Symbol field (column 2).
    """
    gene_to_string = {}
    n_entries = 0

    with gzip.open(str(alias_file), "rt", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            string_id = parts[0]
            alias = parts[1]
            source = parts[2]

            if source == "Ensembl_MGI":
                n_entries += 1
                # alias is the gene symbol (e.g., "Gnai3")
                if alias not in gene_to_string:
                    gene_to_string[alias] = string_id

    return gene_to_string, n_entries


def parse_mgi_gaf(gaf_file):
    """Parse MGI GAF to build gene_symbol -> [GO terms] mapping.

    Uses column 2 (DB_Object_Symbol) as the gene identifier,
    which matches the Ensembl_MGI alias gene symbols.
    """
    go_map = {}
    n_lines = 0

    with gzip.open(str(gaf_file), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 10:
                continue

            db = parts[0]
            if db != "MGI":
                continue

            gene_symbol = parts[2]  # e.g., "Cfl1"
            go_term = parts[4]
            n_lines += 1

            if gene_symbol not in go_map:
                go_map[gene_symbol] = []
            if go_term not in go_map[gene_symbol]:
                go_map[gene_symbol].append(go_term)

    return go_map, n_lines


def run():
    """Main data preparation pipeline for mouse."""
    print(BANNER)
    print("Phase 10A: Mouse Data Preparation")
    print(BANNER)

    DATA_DIR.mkdir(exist_ok=True)

    # Step 1: Download STRING PPI
    string_file = DATA_DIR / "10090.protein.links.v11.5.txt.gz"
    print("\n[1/5] Mouse STRING PPI network")
    ok = download_file(STRING_URL, string_file, "mouse STRING v11.5")
    if not ok:
        print("  FATAL: Cannot download STRING network. Aborting.")
        sys.exit(1)

    # Step 2: Download MGI GAF
    gaf_file = DATA_DIR / "mgi.gaf.gz"
    print("\n[2/5] MGI GO annotations")
    ok = download_file(MGI_GAF_URL, gaf_file, "MGI GAF")
    if not ok:
        print("  FATAL: Cannot download GO annotations. Aborting.")
        sys.exit(1)

    # Step 3: Download STRING aliases
    alias_file = DATA_DIR / "10090.protein.aliases.v11.5.txt.gz"
    print("\n[3/5] STRING protein aliases")
    download_file(ALIAS_URL, alias_file, "mouse STRING aliases")

    # Step 4: Load network + build ID mapping
    print("\n[4/5] Loading network and building ID mapping...")

    G = load_string_network(string_file)
    print(f"  STRING network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build gene_symbol -> STRING_ID mapping from aliases
    gene_to_string, n_alias_entries = build_alias_map(alias_file)
    print(f"  Alias map: {len(gene_to_string)} gene symbols -> STRING IDs "
          f"(from {n_alias_entries} Ensembl_MGI entries)")

    # Step 5: Parse GAF and bridge to STRING IDs
    print("\n[5/5] Bridging GO annotations to STRING IDs...")

    go_by_symbol, n_gaf_lines = parse_mgi_gaf(gaf_file)
    print(f"  MGI GAF: {len(go_by_symbol)} gene symbols, {n_gaf_lines} annotation lines")

    # Map: gene_symbol -> STRING_ID -> GO terms
    go_map = {}
    unmapped_symbols = []
    for gene_symbol, go_terms in go_by_symbol.items():
        if gene_symbol in gene_to_string:
            string_id = gene_to_string[gene_symbol]
            go_map[string_id] = go_terms
        else:
            unmapped_symbols.append(gene_symbol)

    print(f"  Mapped: {len(go_map)} STRING proteins with GO annotations")
    print(f"  Unmapped gene symbols: {len(unmapped_symbols)}")

    # Intersect with network
    network_nodes = set(G.nodes())
    annotated_in_network = sorted(set(go_map.keys()) & network_nodes)
    print(f"  Annotated nodes in network: {len(annotated_in_network)}")

    # Save outputs
    go_out = DATA_DIR / "mouse_go_annotations.json"
    with open(go_out, "w", encoding="utf-8") as f:
        json.dump(go_map, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {go_out.name} ({len(go_map)} genes)")

    info = {
        "species": "Mus musculus",
        "taxon_id": "10090",
        "string_version": "v11.5",
        "min_score": MIN_SCORE,
        "network_nodes": G.number_of_nodes(),
        "network_edges": G.number_of_edges(),
        "annotated_nodes_in_network": len(annotated_in_network),
        "go_genes_total": len(go_map),
        "gaf_gene_symbols": len(go_by_symbol),
        "gaf_lines": n_gaf_lines,
        "alias_gene_symbols": len(gene_to_string),
        "unmapped_symbols": len(unmapped_symbols),
    }
    info_out = DATA_DIR / "mouse_network_info.json"
    with open(info_out, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  Saved {info_out.name}")

    edgelist_out = DATA_DIR / "mouse_ppi.edgelist"
    with open(edgelist_out, "w", encoding="utf-8") as f:
        for u, v in G.edges():
            f.write(f"{u}\t{v}\n")
    print(f"  Saved {edgelist_out.name} ({G.number_of_edges()} edges)")

    print(f"\n{BANNER}")
    print(f"Phase 10A complete.")
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Annotated: {len(annotated_in_network)} nodes in network")
    print(f"{BANNER}")


if __name__ == "__main__":
    run()
