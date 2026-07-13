#!/bin/bash
# ============================================================
# GF-consistency server tasks: all-in-one runner
# ============================================================
# Usage: bash run_all.sh
# Working directory: the folder containing this script
#
# Prerequisites:
#   - Python 3.11+
#   - pip install numpy scipy networkx scikit-learn
#   - 16GB+ RAM, ~2GB disk
#   - Internet access (for STRING/GO downloads)
#
# Tasks:
#   1. Download STRING networks for 10 species
#   2. Download GO annotations for new species
#   3. Fix mouse GO annotations (direct, non-propagated)
#   4. 10-species G-F analysis
#   5. Full-network 11-method G-F (yeast 5936)
#   6. Dark matter disease enrichment
# ============================================================
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  GF-Consistency Server Tasks"
echo "  Time: $(date)"
echo "  Working directory: $(pwd)"
echo "============================================================"

# Ensure output directories exist
mkdir -p data/new_species/go embeddings results

# ------------------------------------------------------------
# Task 1: Download STRING networks
# ------------------------------------------------------------
echo ""
echo "[Task 1] Downloading STRING networks ..."

# Existing species (download if not already present)
declare -A SPECIES_URLS=(
    ["4932"]="https://stringdb-downloads.org/download/protein.links.v12.0/4932.protein.links.v12.0.txt.gz"
    ["9606"]="https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    ["511145"]="https://stringdb-downloads.org/download/protein.links.v12.0/511145.protein.links.v12.0.txt.gz"
    ["7227"]="https://stringdb-downloads.org/download/protein.links.v12.0/7227.protein.links.v12.0.txt.gz"
    ["6239"]="https://stringdb-downloads.org/download/protein.links.v12.0/6239.protein.links.v12.0.txt.gz"
    ["3702"]="https://stringdb-downloads.org/download/protein.links.v12.0/3702.protein.links.v12.0.txt.gz"
    ["7955"]="https://stringdb-downloads.org/download/protein.links.v12.0/7955.protein.links.v12.0.txt.gz"
    ["10116"]="https://stringdb-downloads.org/download/protein.links.v12.0/10116.protein.links.v12.0.txt.gz"
    ["9031"]="https://stringdb-downloads.org/download/protein.links.v12.0/9031.protein.links.v12.0.txt.gz"
)

for taxon in 4932 9606 511145 7227 6239 3702 7955 10116 9031; do
    outfile="data/${taxon}.protein.links.v12.0.txt.gz"
    if [ ! -f "$outfile" ]; then
        echo "  Downloading $taxon ..."
        wget -q "${SPECIES_URLS[$taxon]}" -O "$outfile" || echo "    FAILED: $taxon"
    else
        echo "  $taxon: already exists"
    fi
done

# Also download yeast aliases (needed for ORF -> symbol mapping)
if [ ! -f "data/4932.protein.aliases.v12.0.txt.gz" ]; then
    wget -q "https://stringdb-downloads.org/download/protein.aliases.v12.0/4932.protein.aliases.v12.0.txt.gz" \
        -O "data/4932.protein.aliases.v12.0.txt.gz" || echo "    FAILED: yeast aliases"
fi
echo "[Task 1] Done. Time: $(date)"

# ------------------------------------------------------------
# Task 2: Download GO annotations for new species
# ------------------------------------------------------------
echo ""
echo "[Task 2] Downloading GO annotations ..."

declare -A GAF_URLS=(
    ["6239"]="https://current.geneontology.org/annotations/wb.gaf.gz"
    ["3702"]="https://current.geneontology.org/annotations/tair.gaf.gz"
    ["7955"]="https://current.geneontology.org/annotations/zfin.gaf.gz"
    ["10116"]="https://current.geneontology.org/annotations/rgd.gaf.gz"
    ["9031"]="https://current.geneontology.org/annotations/goa_chicken.gaf.gz"
    ["511145"]="https://current.geneontology.org/annotations/ecocyc.gaf.gz"
    ["7227"]="https://current.geneontology.org/annotations/fb.gaf.gz"
)

for taxon in 6239 3702 7955 10116 9031 511145 7227; do
    outfile="data/new_species/go/${taxon}.gaf.gz"
    if [ ! -f "$outfile" ]; then
        echo "  Downloading GAF for $taxon ..."
        wget -q "${GAF_URLS[$taxon]}" -O "$outfile" || echo "    FAILED: $taxon GAF"
    else
        echo "  $taxon GAF: already exists"
    fi
done
echo "[Task 2] Done. Time: $(date)"

# ------------------------------------------------------------
# Task 2b: Build full GO map from SGD GAF (5909 genes, 94.3% coverage)
# ------------------------------------------------------------
echo ""
echo "[Task 2b] Building full GO map from SGD GAF ..."
python scripts/build_full_go_map.py || echo "  Task 2b failed (non-critical)"
echo "[Task 2b] Done. Time: $(date)"

# ------------------------------------------------------------
# Task 3: Fix mouse GO annotations
# ------------------------------------------------------------
echo ""
echo "[Task 3] Fixing mouse GO annotations (direct, non-propagated) ..."
python scripts/fix_mouse_go_annotations.py || echo "  Task 3 failed (non-critical)"
echo "[Task 3] Done. Time: $(date)"

# ------------------------------------------------------------
# Task 4: 10-species G-F analysis
# ------------------------------------------------------------
echo ""
echo "[Task 4] Running 10-species G-F analysis ..."
python scripts/species_expansion_10species.py || echo "  Task 4 failed (check script)"
echo "[Task 4] Done. Time: $(date)"

# ------------------------------------------------------------
# Task 5: Full-network 11-method G-F (yeast 5936)
# ------------------------------------------------------------
echo ""
echo "[Task 5] Full-network 11-method G-F (yeast 5936 nodes) ..."
python scripts/full_network_all11.py || echo "  Task 5 failed (check script)"
echo "[Task 5] Done. Time: $(date)"

# ------------------------------------------------------------
# Task 6: Dark matter disease enrichment
# ------------------------------------------------------------
echo ""
echo "[Task 6] Dark matter disease enrichment ..."
python scripts/dark_matter_disease_enrichment.py || echo "  Task 6 failed (non-critical)"
echo "[Task 6] Done. Time: $(date)"

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "  All tasks complete!"
echo "  Time: $(date)"
echo "============================================================"
echo ""
echo "Result files:"
ls -lh results/ 2>/dev/null
echo ""
echo "Embedding files:"
ls -lh embeddings/ 2>/dev/null | head -20
echo ""
echo "Data files:"
ls -lh data/mouse_go_annotations_direct.json 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. Copy results/*.json to project results/ directory"
echo "  2. Copy embeddings/*.npy + *_nodes.json to project embeddings/"
echo "  3. Copy data/mouse_go_annotations_direct.json to project data/"
echo "  4. Run degree_controlled_gf_multispecies.py locally with new data"