#!/bin/bash
# Run ONLY the 10-species expansion (skip other tasks)
# Usage: bash run_species_only.sh
set -e
cd "$(dirname "$0")"
mkdir -p data/new_species/go embeddings results

echo "============================================================"
echo "  10-Species G-F Expansion (rerun)"
echo "  Time: $(date)"
echo "============================================================"

# Download STRING v12.0 for all species
echo "[1/3] Downloading STRING networks ..."
for taxon in 4932 9606 511145 7227 6239 3702 7955 10116 9031; do
    outfile="data/${taxon}.protein.links.v12.0.txt.gz"
    if [ ! -f "$outfile" ]; then
        echo "  Downloading $taxon ..."
        wget -q "https://stringdb-downloads.org/download/protein.links.v12.0/${taxon}.protein.links.v12.0.txt.gz" \
            -O "$outfile" || echo "    FAILED: $taxon"
    else
        echo "  $taxon: exists"
    fi
done

# Download GO annotations
echo ""
echo "[2/3] Downloading GO annotations ..."
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
        echo "  Downloading GAF $taxon ..."
        wget -q "${GAF_URLS[$taxon]}" -O "$outfile" || echo "    FAILED: $taxon"
    fi
done

# Run species expansion
echo ""
echo "[3/3] Running 10-species G-F analysis ..."
python scripts/species_expansion_10species.py || echo "  FAILED - check errors above"
echo ""
echo "Done. Time: $(date)"
echo "Copy results/species_expansion_10species.json back to project."
