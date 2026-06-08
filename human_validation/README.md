# Human Cross-Species Validation

This directory contains scripts for cross-species validation on the human PPI network (largest connected component), comparing all six embedding methods: Diffusion Map (DM), MDS, Spectral, DeepWalk, Node2Vec, and VGAE.

## Important Note

**This analysis requires significant computational resources and is NOT included in the main reproduction pipeline (`run_all_analysis.py`).**

The human interactome analysis (Paper Sections 2.10, 3.6, Figure 6) was performed on a high-performance server with:
- ~16GB RAM for embedding generation
- ~30 minutes runtime for all six methods
- ~2GB disk space for STRING data downloads

## Directory Structure

```
human_validation/
|---- README.md                           # This file
|---- human_embed_all.py                  # Generate all 6 embeddings (DM, MDS, Spectral, DW, N2V, VGAE)
|---- human_gf_all.py                     # Compute G-F curves, scores, and plateau widths for all methods
|---- 12_human_ppi_validation.py        # Legacy: DM + Node2Vec with Louvain (quick validation)
|---- 12d_human_n2v_cleaned_scan.py     # Legacy: Node2Vec outlier detection and cleaning
|---- 12e_human_dm_quick.py              # Legacy: DM quick scan with Leiden
|---- [Data files - generated on first run]
    |---- 9606.protein.links.v12.0.txt.gz    # STRING PPI network
    |---- goa_human.gaf.gz                   # GO annotations
    |---- 9606.protein.aliases.v12.0.txt.gz  # ID mapping
    |---- human_*_embedding.json             # Generated embeddings
    |---- outlier_report.txt                 # Outlier detection report
```

## Quick Start

### Option 1: Full 6-Method Analysis (Recommended)

```bash
# Step 1: Generate all 6 embeddings
cd human_validation
python human_embed_all.py

# Step 2: Compute G-F curves and scores
python human_gf_all.py
```

### Option 2: Quick DM + Node2Vec Only

```bash
cd human_validation
python 12_human_ppi_validation.py
```

## Method Comparison

| Method | Embedding Type | Key Parameters | Approx. Runtime |
|--------|---------------|----------------|-----------------|
| **DM** | Nonlinear manifold | 6 centrality features | ~2 min |
| **MDS** | Distance-based | Shortest-path distances | ~5 min |
| **Spectral** | Laplacian | Normalized Laplacian | ~1 min |
| **DeepWalk** | Random walk | Walk length 20, 10 walks/node | ~10 min |
| **Node2Vec** | Biased walk | p=0.5, q=2.0 | ~15 min |
| **VGAE** | Graph neural network | 2-layer GCN, 300 epochs | ~20 min |

## Key Findings (from Paper Section 3.6)

1. **No Stable Plateau**: Neither DM nor Node2Vec maintained a stable purity plateau on the human network, unlike yeast.

2. **Dimensional Collapse Detected**: Node2Vec exhibited partial dimensional collapse due to a single extreme outlier (ENSP00000334051, x=-40.75, >100σ from mean). The G-F framework diagnosed this pathology through an anomalous curve shape.

3. **Scale Compression**: After outlier removal, Node2Vec's x-axis standard deviation collapsed to 0.08, indicating quasi-one-dimensional embedding.

4. **G-F Score Rankings** (unified interval):
   - DM: highest plateau width despite fluctuations
   - Node2Vec: high peak purity but sensitive to scale
   - Other methods: intermediate performance

## Output Files

| File | Description |
|------|-------------|
| `data/human_*_embedding.json` | 2D coordinates for each method |
| `results/human_gf_curves_200pts.pkl` | G-F curve data (purity & modularity) |
| `results/human_gf_scores.json` | G-F Scores and rankings |
| `results/human_plateau_widths.json` | Plateau width measurements |
| `human_validation/outlier_report.txt` | Outlier detection details |

## Data Sources

- **PPI Network**: STRING v12.0, taxID 9606, combined score ≥ 700
  - URL: https://stringdb-static.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
  - Size: ~200 MB compressed

- **GO Annotations**: GOA Human (Gene Ontology Annotation)
  - URL: https://current.geneontology.org/annotations/goa_human.gaf.gz
  - Size: ~50 MB compressed

- **ID Mapping**: STRING aliases file
  - URL: https://stringdb-static.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz

## Memory Requirements

| Step | Peak RAM | Notes |
|------|----------|-------|
| Network loading | ~2 GB | Sparse adjacency matrix |
| Shortest-path (MDS) | ~8 GB | Distance matrix computation |
| Random walks (DW/N2V) | ~4 GB | Co-occurrence matrix |
| VGAE training | ~6 GB | PyTorch GPU/CPU |
| G-F curve computation | ~4 GB | Distance matrix per method |

## Troubleshooting

### Out of Memory
- Use `human_embed_all.py` with reduced walk counts: edit `walks_per_node` parameter
- For MDS, consider using landmark MDS or skip if memory constrained

### Slow Node2Vec
- Reduce `walks_per_node` from 10 to 5
- Reduce `walk_length` from 20 to 15

### Missing GO Annotations
The scripts will attempt to download GO annotations automatically. If this fails:
1. Manually download from: https://current.geneontology.org/annotations/goa_human.gaf.gz
2. Place in `human_validation/` directory
3. Re-run scripts

## Citation

When using the human validation results, please cite:

```bibtex
@article{zhang2026gf,
  title={A Geometric-Functional Consistency Framework for Evaluating 
         Protein Interaction Network Embeddings},
  author={Zhang, Yuhan},
  year={2026},
  note={Human validation: ~15,882 nodes (14,679 in largest CC), 6 methods}
}
```

## Contact

For questions about the human validation analysis, please open an issue at:
https://github.com/Qinrayn/GF-consistency-framework/issues
