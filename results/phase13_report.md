# Phase 13: Protein Function Prediction

## Methodology

Leave-one-term-out (LOTO) cross-validation on the full yeast STRING network (4709 proteins, 2486 unique GO BP terms). For each (protein, GO term) pair, the term is hidden and the protein's function is predicted via KNN in embedding space (k = 3 to 30). Predictions are validated against three network-topology baselines: PPI direct neighbours, 2-hop neighbourhood diffusion (decay=0.5), and random annotation-frequency sampling.

### Embedding Methods

| Method | Full-network nodes | GF Score (curated) |
|--------|-------------------|-------------------|
| DM | 3133 | 0.1552 |
| MDS | 3133 | 0.1522 |
| Spectral | 3133 | 0.1633 |
| Node2Vec | 3133 | 0.1512 |
| VGAE | 3133 | 0.0657 |

## Data Summary

- GAF total lines: 138996
- Experimental BP annotations: 22568
- Mapped to network: 21854
- Proteins with annotations: 4709
- Mean terms per protein: 3.0
- Query proteins (≥2 terms): 3133
- Total LOTO trials: 12690

## Results: Precision@k

| Method | P@3 | P@5 | P@7 | P@10 | P@15 | P@20 | P@30 |
|--------|---|---|---|---|---|---|---|
| DM | 0.037 | 0.055 | 0.068 | 0.084 | 0.106 | 0.124 | 0.149 |
| MDS | 0.057 | 0.088 | 0.111 | 0.135 | 0.173 | 0.203 | 0.253 |
| Spectral | 0.067 | 0.098 | 0.122 | 0.147 | 0.181 | 0.209 | 0.251 |
| Node2Vec | 0.009 | 0.013 | 0.017 | 0.022 | 0.031 | 0.040 | 0.055 |
| VGAE | 0.009 | 0.013 | 0.017 | 0.022 | 0.031 | 0.038 | 0.050 |
| PPI-Neighbors | 0.247 | 0.331 | 0.384 | 0.443 | 0.506 | 0.548 | 0.606 |
| 2-Hop Diffusion | 0.106 | 0.146 | 0.180 | 0.222 | 0.275 | 0.317 | 0.383 |
| Random | 0.037 | 0.049 | 0.059 | 0.073 | 0.095 | 0.118 | 0.156 |

## Results: Mean Reciprocal Rank

| Method | MRR |
|--------|-----|
| PPI-Neighbors | 0.2188 |
| 2-Hop Diffusion | 0.1046 |
| Spectral | 0.0656 |
| MDS | 0.0601 |
| Random | 0.0411 |
| DM | 0.0368 |
| Node2Vec | 0.0111 |
| VGAE | 0.0109 |

## GF Score Correlation (Closing Loop)

- Spearman ρ = 0.9
- P-value = 0.037386
- 95% CI: [0.1111, 1.0]
- Pearson r = 0.6174
- n = 5

### Leave-One-Out Sensitivity

| Removed Method | ρ (remaining) |
|---------------|---------------|
| DM | 1.0000 |
| MDS | 1.0000 |
| Node2Vec | 0.8000 |
| Spectral | 0.8000 |
| VGAE | 0.8000 |

## Interpretation

The correlation between curated-network GF Score and full-network prediction accuracy tests whether the framework's structural quality metric transfers across network scales. A positive Spearman ρ indicates that embedding methods with higher geometric-functional consistency on small networks also produce better function predictions on large networks.

## Limitations

- Leave-one-term-out (not temporal holdout) due to single GAF version
- 5 methods (full-network embeddings available)
- Exact term matching only (no semantic similarity)
- BP aspect only (MF and CC not tested)

*Generated: 2026-06-19 11:27*