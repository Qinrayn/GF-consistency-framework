## G-F Consistency Framework - Key Results Summary (v1.1.0)

### Framework Overview
The G-F Score quantifies alignment between geometric structure (embeddings) and functional structure (GO annotations) in protein-protein interaction networks:

G-F Score = (1 / (r_max - r_min)) * integral from r_min to r_max of purity(r) dr

Unified interval: [0.05, 0.422] (stable region across all methods)
Plateau Width W: r-interval where purity >= 80% of each method's peak (relative threshold)

### Yeast PPI Network (STRING v11.5, 5,936 nodes, 153 curated)

| Method | G-F Score | Plateau Width W | Peak Purity | Link Pred. AUROC | k-NN micro-F1 |
|--------|:---------:|:---------------:|:-----------:|:-----------------:|:-------------:|
| Spectral | 0.163 | 0.103 | 0.256 | 0.819 +/- 0.008 | 0.673 +/- 0.060 |
| DM | 0.155 | 0.083 | 0.247 | 0.738 +/- 0.009 | 0.513 +/- 0.055 |
| MDS | 0.152 | 0.146 | 0.245 | 0.804 +/- 0.013 | 0.606 +/- 0.074 |
| Node2Vec | 0.151 | 0.073 | 0.221 | 0.525 +/- 0.025 | 0.143 +/- 0.068 |
| PCA | 0.138 | 0.055 | 0.231 | N/A | N/A |
| VGAE-feat | 0.124 | 0.500 | 0.144 | N/A | N/A |
| DeepWalk | 0.123 | 0.091 | 0.196 | 0.518 +/- 0.024 | 0.186 +/- 0.079 |
| GIN | 0.122 | 0.063 | 0.215 | 0.488 +/- 0.024 | 0.203 +/- 0.099 |
| GAT | 0.069 | 0.048 | 0.128 | 0.591 +/- 0.021 | 0.159 +/- 0.030 |
| GraphSAGE | 0.069 | 0.098 | 0.107 | 0.642 +/- 0.026 | 0.185 +/- 0.051 |
| VGAE | 0.066 | 0.085 | 0.083 | 0.472 +/- 0.008 | 0.211 +/- 0.076 |

Leiden baseline purity: 0.180 (same formula as G-F curve: most-common GO term / total GO terms)
Leiden baseline ≈ best G-F Score (0.180 vs 0.163), indicating spatial embeddings capture functional structure at a level comparable to graph-based community detection

### Statistical Validation
- Spearman rho (AUROC vs G-F Score, classical methods): 0.943 (p = 0.005)
- Leave-one-out sensitivity: rho range [0.900, 1.000], mean 0.933
- Bonferroni correction (30 subsets, size 150): 9/30 significant after correction
- Randomization control: original max 0.247 > shuffled 0.230 +/- 0.007 (Z = 6.95)
- Adaptive interval: 4/8 genuine, 4/8 fallback, consensus [0.05, 0.2284]

### Human PPI Validation (STRING v12.0, 15,882 nodes)

| Method | G-F Score | Plateau Width W |
|--------|:---------:|:---------------:|
| Node2Vec | 0.852 | 0.450 |
| DeepWalk | 0.840 | 0.450 |
| Spectral | 0.811 | 0.480 |
| DM | 0.515 | 0.202 |
| MDS | 0.415 | 0.202 |
| VGAE | 0.270 | 0.364 |

Cross-species unified interval: [0.05, 0.297]
Key finding: Node2Vec/DeepWalk rank lowest on yeast but highest on human, demonstrating G-F scores are network-specific.

### GO DAG Propagation (Step 19)
- Pre-propagation: 154 genes, avg 3.8 terms/gene
- Post-propagation (True Path Rule): 5,429 genes, avg 28.9 terms/gene
- Note: Extended G-F scores approach ~1.0 due to annotation inflation from hierarchical expansion. Main results use pre-propagation annotations.

### Runtime Performance
- Total pipeline: ~1,058 seconds (~17.6 minutes)
- Top bottlenecks: Robustness (486.6s), Randomization (200.1s), Sampling (147.0s)
- Environment: Windows 11, CPython 3.13, NumPy 2.4, SciPy 1.17

### Biological Interpretation
All 11 methods classified as Level 1 (Weak: poor functional reflection) on the 4-level G-F interpretation scale. GO DAG propagation expanded annotations from 154 to 5,429 genes via True Path Rule.

### Software & Engineering (v1.1.0)
- **Package**: `pip install gf-consistency` (PyPI-ready, pyproject.toml)
- **CLI**: `gf-consistency --config pipeline_config.yaml`
- **Configuration**: YAML-based, 200+ configurable parameters
- **Testing**: 51 unit tests (pytest), all passing
- **Input validation**: Pre-flight checks for networks, embeddings, GO annotations
- **Extension modules**: Hyperbolic embeddings (Poincare Ball), multi-species loader (4 species), temporal network framework, pathway-level analysis
- **Supported species**: Yeast, Human, E. coli, Mouse (extensible registry)
- **Dependencies**: 14 core + 3 dev (numpy, scipy, networkx, torch, torch_geometric, ...)
