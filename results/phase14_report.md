# Phase 14: Long-Range Functional Link Discovery

## Core Hypothesis

Graph embeddings encode long-range functional topology that direct network neighbors cannot see. While embeddings cannot replace topology for local function prediction (Phase 13), they capture complementary information about distant functional relationships.

## Methodology

Full yeast STRING network: 5,936 nodes, 120,357 edges. 4,709 proteins with experimental BP annotations, 2,486 unique GO terms. BFS from 4,664 annotated proteins for complete shortest-path distance matrix.

### Part 1: Distance-Stratified Recovery

176,914 protein pairs sharing at least one GO BP term, stratified by shortest-path network distance. Recovery rate = fraction found in top-30 nearest neighbors (embedding) or direct neighbors (PPI).

| Stratum | Pairs | PPI Recovery | Spectral Recovery | Gap |
|---------|-------|-------------|-------------------|-----|
| 1-3 hops | 154,952 (87.6%) | 0.254 | 0.039 | -0.214 |
| 4-6 hops | 21,937 (12.4%) | 0.000 | 0.007 | **+0.007** |
| 7+ hops | 25 (0.01%) | 0.000 | 0.000 | 0.000 |

Key finding: at 4-6 hops, PPI recovery is exactly zero (direct neighbors cannot reach), but Spectral embedding KNN recovers 0.7% of functional associations. This is the unique signal that embeddings capture.

### Part 2a: Score-Weighted Hybrid (Failed)

Naive score combination (PPI votes + embedding 1/distance weights) fails catastrophically: embedding distance weights are numerically much larger than PPI vote counts, overwhelming topology signal even at w=0.1. Lesson: score-based combination requires careful normalization.

### Part 2b: Rank-Based Fallback Hybrid (Success)

Two-stage prediction: PPI neighbors predict first, embedding KNN fills gaps for terms PPI cannot reach.

| Method | Fallback MRR | PPI MRR | Gain |
|--------|-------------|---------|------|
| MDS | 0.2192 | 0.2187 | **+0.0005 (+0.2%)** |
| VGAE | 0.2191 | 0.2187 | +0.0004 (+0.2%) |
| DM | 0.2190 | 0.2187 | +0.0003 (+0.2%) |
| Spectral | 0.2190 | 0.2187 | +0.0003 (+0.1%) |
| Node2Vec | 0.2189 | 0.2187 | +0.0002 (+0.1%) |

ALL five methods improve over pure PPI via rank fallback.

### Part 2c: Rank Aggregation (Borda Count)

Borda-count aggregation at various weights shows pure PPI (w=0) remains optimal — any embedding weight dilutes the dominant PPI signal. The improvement from fallback comes entirely from gap-filling, not rank reordering.

### Part 3: Long-Range Functional Link Discovery

Protein pairs >= 4 hops apart in the PPI network but within top-30 KNN in embedding space that share experimental GO BP annotations:

| Method | Discoveries |
|--------|-------------|
| Spectral | 256 |
| DM | 251 |
| VGAE | 234 |
| Node2Vec | 212 |
| MDS | 145 |

### Part 4: Embedding-Rescued Trials

Categorisation of 12,690 LOTO trials (MDS, k=30):

| Category | Count | Percentage |
|----------|-------|-----------|
| PPI-only (PPI finds, embedding misses) | 5,370 | 42.3% |
| Both (both methods find) | 4,483 | 35.3% |
| **Emb-rescue (embedding finds, PPI misses)** | **258** | **2.0%** |
| Miss (neither method finds) | 2,579 | 20.3% |

The 258 embedding-rescued trials represent cases where direct network topology completely fails but the embedding geometric structure recovers the functional association.

## Dimension Sweep (Phase 13b)

Spectral embedding at d = {2, 4, 8, 16, 32, 64}:

| Dimension | MRR | vs PPI (0.219) | Eigenvalue Info Captured |
|-----------|-----|----------------|-------------------------|
| 2 | 0.066 | -0.154 | 0.7% |
| 4 | 0.087 | -0.132 | 1.9% |
| 8 | 0.128 | -0.091 | 5.3% |
| 16 | 0.160 | -0.059 | 14.3% |
| 32 | 0.190 | -0.029 | 38.3% |
| 64 | 0.205 | -0.014 | 100% |

2D embeddings capture only 0.7% of spectral information. MRR improves 213% from d=2 to d=64, but no dimension exceeds the PPI baseline. The MRR curve is logarithmic — each dimension doubling adds diminishing returns.

## Interpretation

1. **Embeddings are complementary, not competitive**: Direct PPI topology is the dominant signal for function prediction (87.6% of functional pairs are within 1-3 hops). Embeddings provide a small but genuine improvement by recovering long-range associations.

2. **258 genuine discoveries**: In 2.0% of LOTO trials, the embedding finds a functional association that the entire PPI neighborhood cannot. These are proteins in the same biological process but separated by 4+ hops in the interaction network.

3. **2D is fundamentally lossy**: The dimension sweep proves that 2D visualizations discard 99.3% of the spectral information. The framework's GF Score remains valid as a quality metric, but 2D embedding-based prediction should not be used as the primary tool.

4. **Practical recommendation**: Use PPI topology as the primary predictor; use embedding KNN as a supplementary tool for proteins with sparse network connectivity or for long-range functional link discovery.

## Figures

- Fig 69: Dimension sweep MRR curve + Laplacian eigenvalue spectrum
- Fig 70: Marginal MRR per dimension + cumulative eigenvalue information
- Fig 71: Distance-stratified recovery bar chart + embedding-PPI gap
- Fig 72: Score-weighted hybrid sweep (failed approach)
- Fig 73: Long-range discoveries count + network distance distribution
- Fig 74: Phase 14 four-panel summary dashboard
- Fig 75: Fixed hybrid comparison (rank fallback + Borda aggregation + trial outcomes)

*Generated: 2026-06-15*
