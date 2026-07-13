# GF-Consistency Server Tasks

Self-contained package for server-side computation of species expansion
and full-network analysis. Copy this entire `server-tasks/` directory
to the server and run `bash run_all.sh`.

## Contents

```
server-tasks/
├── run_all.sh                          # Main entry point
├── scripts/
│   ├── utils.py                        # Shared utilities (from project)
│   ├── species_expansion_10species.py  # 10-species G-F analysis
│   ├── full_network_all11.py           # Full 5936-node 11-method G-F
│   ├── degree_controlled_gf_multispecies.py  # Cross-species degree control
│   ├── fix_mouse_go_annotations.py     # Rebuild mouse GO from GAF
│   └── dark_matter_disease_enrichment.py     # Dark matter -> disease GO
├── data/                               # Essential data files
│   ├── curated_153_ppi.edgelist        # 28 KB
│   ├── gene_go_map.json                # 14 KB
│   ├── yeast_ppi_5936.edgelist         # 2.1 MB
│   ├── mouse_ppi.edgelist              # 12 MB
│   ├── mouse_go_annotations.json       # 6.4 MB
│   ├── human_go_annotations.json       # 2.7 MB
│   ├── mgi.gaf.gz                      # 14 MB
│   ├── gene_association.sgd.gaf.gz     # 2.6 MB
│   └── 4932.protein.aliases.v11.5.txt.gz  # 2.1 MB
├── embeddings/                         # Output: generated embeddings
└── results/                            # Output: JSON results
```

Total size: ~40 MB (STRING networks downloaded on server).

## Usage

```bash
# 1. Copy to server
scp -r server-tasks/ user@server:~/

# 2. Install dependencies (if not already available)
pip install numpy scipy networkx scikit-learn python-igraph python-louvain

# 3. Run all tasks
cd ~/server-tasks
bash run_all.sh

# 4. Copy results back
scp -r user@server:~/server-tasks/results/*.json ./results/
scp -r user@server:~/server-tasks/embeddings/*.npy ./embeddings/
```

## Tasks

### Task 1: Download STRING networks (5 existing + 5 new species)

Downloads STRING v12.0 PPI networks for:
- Existing: yeast (4932), human (9606), E. coli (511145), fly (7227)
- New: C. elegans (6239), A. thaliana (3702), zebrafish (7955),
  rat (10116), chicken (9031)

Mouse (10090) is already included as edgelist.

### Task 2: Download GO annotations

Downloads GAF files from geneontology.org for all new species.

### Task 3: 10-species G-F analysis

Computes spectral embedding + G-F Score for all 10 species.
Existing results (yeast, human, mouse, E. coli, fly) are loaded
from result files; new species are computed fresh.

Expected output: `results/species_expansion_10species.json`

### Task 4: Full-network 11-method G-F (yeast 5936)

Computes all 11 embedding methods on the full yeast STRING network
(5936 nodes), not just the curated 153-node subset. Missing embeddings
are computed and saved.

Expected output: `results/full_network_all11.json`

### Task 5: Fix mouse GO annotations

Rebuilds mouse GO from raw MGI GAF (direct, non-propagated).

Expected output: `data/mouse_go_annotations_direct.json`

### Task 6: Dark matter disease enrichment

Maps 44 yeast dark matter pairs to human functional orthologs and
tests disease GO enrichment.

Expected output: `results/dark_matter_disease_enrichment.json`

## After completion

Copy these back to the main project:
```
results/species_expansion_10species.json      -> results/
results/full_network_all11.json               -> results/
results/dark_matter_disease_enrichment.json   -> results/
data/mouse_go_annotations_direct.json         -> data/
embeddings/*_full.npy                          -> embeddings/
embeddings/*_full_nodes.json                   -> embeddings/
```

Then run locally:
```bash
python scripts/degree_controlled_gf_multispecies.py  # with 10 species
```

## Requirements

- Python >= 3.11
- 16 GB RAM (for 16K-node networks)
- ~2 GB disk (for STRING downloads)
- No GPU required
- Estimated runtime: 2-4 hours
