# G-F Consistency Framework

**A Geometric-Functional Consistency Framework for Evaluating Protein Interaction Network Embeddings**

Complete reproduction of the experimental pipeline: 11 embedding methods · 21-step validation workflow · 200-point G-F curve sampling · publication-quality figures · `random_seed = 42`

<details>
<summary><strong>Key Results at a Glance</strong></summary>

| Method | G-F Score | Link Pred. AUROC | k-NN micro-F1 |
|--------|:---------:|:-----------------:|:-------------:|
| **Spectral** | **0.163** | 0.819 ± 0.008 | 0.673 ± 0.060 |
| DM | 0.155 | 0.738 ± 0.009 | 0.513 ± 0.055 |
| MDS | 0.152 | 0.804 ± 0.013 | 0.606 ± 0.074 |
| Node2Vec | 0.151 | 0.525 ± 0.025 | 0.143 ± 0.068 |
| PCA | 0.138 | — | — |
| VGAE-feat | 0.124 | — | — |
| DeepWalk | 0.123 | 0.518 ± 0.024 | 0.186 ± 0.079 |
| GIN | 0.122 | 0.488 ± 0.024 | 0.203 ± 0.099 |
| GAT | 0.069 | 0.591 ± 0.021 | 0.159 ± 0.030 |
| GraphSAGE | 0.069 | 0.642 ± 0.026 | 0.185 ± 0.051 |
| VGAE | 0.066 | 0.472 ± 0.008 | 0.211 ± 0.076 |

- Leiden baseline purity: **0.689** (matches paper)
- Bonferroni (30 subsets, size 150): **9/30** significant after correction
- Randomization: original max 0.247 > shuffled 0.230 ± 0.002 (10 permutations, Z = 6.95)
- Spearman rho (AUROC vs G-F Score): **0.943** (*p* = 0.005)
- Unified interval: **[0.05, 0.422]**

All metrics → [`results/final_results_summary.json`](results/final_results_summary.json)

</details>

---

## Quick Start

```bash
# 1. Install
conda env create -f environment.yml
conda activate gf-consistency

# Or install as a package
pip install .

# 2. Run full pipeline (≈ 60 min on standard laptop)
python run_all_analysis.py

# Or use the CLI entry point after pip install
gf-consistency

# Configuration: edit pipeline_config.yaml or pass via CLI
python run_all_analysis.py --config my_config.yaml

# Options:
python run_all_analysis.py --run-human          # Include human validation
python run_all_analysis.py --start-from 3       # Resume from step 3
python run_all_analysis.py --skip-plots         # Skip figure generation
python run_all_analysis.py --skip-gnn           # Skip GNN embeddings (GraphSAGE/GAT/GIN)
python run_all_analysis.py --skip-extended      # Skip extended analyses (Steps 17-21)
python run_all_analysis.py --seed 123           # Override random seed
python run_all_analysis.py --species human      # Target a different species
```

Individual steps work standalone:

```bash
python scripts/compute_gf.py          # G-F curves only
python scripts/link_prediction.py     # Link prediction only
python scripts/plot_figures.py        # Regenerate figures from existing results
```

---

## Configuration

All pipeline parameters can be customised via `pipeline_config.yaml` without editing any script:

```yaml
pipeline:
  seed: 42
  species: yeast          # yeast | human | ecoli | mouse
  start_from: 1

gf_score:
  r_min: 0.05
  r_max: 0.55
  n_points: 200
  gf_r_min: 0.05
  gf_r_max: 0.422
  plateau_relative_threshold: 0.80

embeddings:
  classical_methods: [DM, MDS, Spectral, DeepWalk, Node2Vec, VGAE]
  gnn_methods: [GraphSAGE, GAT, GIN]
  node2vec:
    p: 0.5
    q: 2.0
```

CLI flags (`--seed`, `--species`, `--start-from`, etc.) take precedence over config file values. Run `gf-consistency --help` for all options.

---

## G-F Framework Theory

The **G-F Score** quantifies alignment between geometric structure (embeddings) and functional structure (GO annotations):

$$\text{G-F Score} = \frac{1}{r_{\max} - r_{\min}} \int_{r_{\min}}^{r_{\max}} \text{purity}(r) \, dr$$

| Concept | Definition |
|---------|-----------|
| **purity(r)** | Mean functional purity of communities at distance threshold *r* |
| **Unified interval** | [0.05, 0.422] — stable region across all methods |
| **Plateau width W** | Range of *r* where purity remains stable → well-separated clusters |
| **Geometric gap** | d<sub>inter</sub> − d<sub>intra</sub>: margin between intra- and inter-module distances |

Mathematical proofs (Propositions 1–2, Theorems 1–4) → [`Supplementary_Materials.pdf`](Supplementary_Materials.pdf) · Supplementary data tables → [`Supplementary_Materials.txt`](Supplementary_Materials.txt)

---

## Pipeline Overview

```
Step 1  ─ Data Preprocessing ────────── Yeast PPI + GO annotations
Step 2  ─ Compute Embeddings ────────── 8 methods → embeddings/*.npy
Step 3  ─ G-F Curves & Scores ───────── 200-point grid → gf_scores.json
Step 4  ─ Leiden Baseline ────────────── Community detection baseline
Step 5  ─ Subset Robustness ──────────── 30 subsets × 5 sizes + Bonferroni correction
Step 6  ─ Full Network ───────────────── 5,936-node STRING validation
Step 7  ─ Geometric Analysis ─────────── d_intra / d_inter margin
Step 8  ─ Link Prediction ────────────── 5-fold CV AUROC
Step 9  ─ Downstream k-NN ────────────── GO term prediction
Step 10 ─ Randomization Control ──────── Shuffled-label baseline
Step 11 ─ Sampling Density ──────────── 30 vs 200-point verification
Step 12 ─ Sensitivity Analysis ───────── Interval robustness
Step 13 ─ Human Validation ───────────── Cross-species (15,882 nodes)
Step 14 ─ Figure Generation ──────────── figures/*.png
Step 15 ─ GNN Embeddings ─────────────── GraphSAGE, GAT, GIN → embeddings/*.npy
Step 16 ─ Adaptive Unified Interval ──── Data-driven consensus [r_min, r_max]
Step 17 ─ Network Topology Analysis ──── Cross-species topology comparison
Step 18 ─ Rank Reversal Analysis ──────── p/q sensitivity + geometric gap mechanism
Step 19 ─ GO Propagation ─────────────── True Path Rule DAG expansion (154 → 5,429 genes)
Step 20 ─ Biological Interpretation ──── 4-level G-F scale + GO cluster case study
Step 21 ─ Runtime Benchmark ──────────── Step-wise profiling + complexity analysis
Step 22 ─ Persistent Homology ─────────── Betti curves, persistence diagrams (Ripser)
Step 23 ─ Topological Statistics ──────── Topo-GF correlations, consistency scores
Step 24 ─ Hyperbolic Embedding ────────── Poincare Ball (Riemannian SGD)
Step 25 ─ Pathway Enrichment ──────────── Fisher's exact test on G-F communities
Step 26 ─ Statistical Summary ─────────── Spearman, Wilcoxon, bootstrap CIs
         └─ Summary ─────────────────── final_results_summary.json
```

---

## Embedding Methods

| Method | Input | Strategy | 2D Projection |
|--------|-------|----------|---------------|
| Diffusion Map (DM) | 6 centrality features → Markov matrix | Eigendecomposition | Top-2 eigenvectors |
| Classical MDS | Shortest-path distances | Double-centering + eigendecomposition | Top-2 eigenvectors |
| Spectral | Normalized Laplacian | Eigendecomposition | Top-2 eigenvectors |
| DeepWalk | Random walks → co-occurrence | Truncated SVD | Top-2 singular vectors |
| Node2Vec | Biased random walks → co-occurrence | Truncated SVD | Top-2 singular vectors |
| VGAE | 2-layer GCN encoder | Variational autoencoder (300 epochs, Adam lr=0.01) | Latent space |
| VGAE-feat | GCN + 6 centrality features | Variational autoencoder | Latent space |
| PCA | 6 centrality features | PCA | Top-2 components |
| GraphSAGE | SAGEConv mean aggregation | 2-layer GNN + BCE reconstruction (300 epochs) | Latent space |
| GAT | GATConv attention (heads=1) | 2-layer GNN + BCE reconstruction (300 epochs) | Latent space |
| GIN | GINConv + MLP | 2-layer GNN + BCE reconstruction (300 epochs) | Latent space |

All embeddings standardized to **σ = 0.3** before G-F analysis.

---

## Project Structure

```
GF-consistency-framework/
├── scripts/                    # 21-step analysis pipeline + extensions
│   ├── data_preprocessing.py   # Load PPI + GO data
│   ├── embed_all.py            # Compute 8 classical/NN embeddings
│   ├── compute_gf.py           # G-F curves + scores
│   ├── leiden_baseline.py      # Leiden community baseline
│   ├── robustness.py           # Subset analysis + Bonferroni
│   ├── full_network.py         # Full 5,936-node network
│   ├── geometric_analysis.py   # d_intra / d_inter margins
│   ├── link_prediction.py      # 5-fold CV link prediction
│   ├── downstream_knn.py       # k-NN GO term prediction
│   ├── randomization_control.py# Shuffled-label control
│   ├── sampling_density.py     # 30 vs 200 pt comparison
│   ├── gf_score_sensitivity.py # Interval sensitivity
│   ├── plot_figures.py         # Generate all figures
│   ├── human_validation.py     # Cross-species validation
│   ├── embed_gnn.py            # GraphSAGE, GAT, GIN embeddings (Step 15)
│   ├── adaptive_interval.py    # Data-driven consensus interval (Step 16)
│   ├── network_topology_analysis.py  # Cross-species topology (Step 17)
│   ├── rank_reversal_analysis.py     # p/q sensitivity + mechanism (Step 18)
│   ├── go_propagation.py       # GO DAG True Path Rule expansion (Step 19)
│   ├── biological_interpretation.py  # 4-level G-F scale + case study (Step 20)
│   ├── benchmark_runtime.py    # Pipeline profiling + complexity (Step 21)
│   ├── statistical_analysis.py # Spearman, Wilcoxon, bootstrap, permutation
│   ├── robustness_analysis.py  # Extended 30-subset convergence
│   ├── visualization_helpers.py# Okabe-Ito colorblind-safe plotting
│   ├── config_loader.py        # YAML configuration loader + validator
│   ├── input_validator.py      # Pre-flight input validation
│   ├── embed_hyperbolic.py     # Poincare Ball hyperbolic embeddings
│   ├── multispecies_loader.py  # Multi-species dataset loader
│   ├── temporal_network.py     # Dynamic/temporal PPI framework
│   ├── pathway_analysis.py     # Pathway enrichment + cancer gene analysis
│   └── utils.py                # Shared utilities
│
├── tests/                      # pytest test suite (51 tests)
│   ├── conftest.py             # Shared fixtures
│   ├── test_utils.py           # Core computation tests
│   └── test_config_loader.py   # Config loader tests
│
├── data/                       # Input data (STRING v11.5, GO)
│   ├── *.edgelist              # PPI networks (curated 153, full, subsets)
│   ├── gene_go_map.json        # Gene → GO term mapping
│   └── *.gz                    # STRING/GAF files (Git LFS)
│
├── embeddings/                 # Generated .npy + _nodes.json per method
├── results/                    # JSON/CSV result files
│   └── final_results_summary.json  ← Master summary
│
├── figures/                    # Publication figures (PNG, 300 dpi)
│   ├── Fig1–7                  # Main numbered figures (incl. Fig6: human validation)
│   ├── Fig8–14                 # Topological analysis figures (Betti curves, persistence, etc.)
│   ├── FigS1–S7                # Supplementary figures
│   └── comparison_30vs200      # Sampling density check
│
├── human_validation/           # Cross-species (optional, STRING v12.0)
│
├── run_all_analysis.py         # One-command Python pipeline (21 steps)
├── pipeline_config.yaml        # YAML configuration (all parameters)
├── pyproject.toml              # Python package metadata
├── environment.yml             # Conda environment
├── requirements.txt            # pip dependencies
├── Supplementary_Materials.pdf # Mathematical proofs (Propositions 1-2, Theorems 1-4)
└── LICENSE                     # MIT License
```

---

## Data Sources

| Dataset | Source | Nodes | Notes |
|---------|--------|:-----:|-------|
| Yeast PPI | STRING v11.5 | 5,936 | Score ≥ 700, curated |
| Yeast GO | SGD GAF | 153 annotated (5,429 after GO DAG propagation) | BP terms, level ≥ 3 |
| Human PPI | STRING v12.0 | 15,882 | Cross-species (largest CC) |
| Human GO | GOA Human GAF | 16,818 annotated | BP terms |

Large files (`*.txt.gz`, `*.gaf.gz`) tracked via **Git LFS** — run `git lfs install` before cloning.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.11 | Runtime |
| numpy | ≥ 1.24 | Numerical computation |
| scipy | ≥ 1.10 | Integration, statistics |
| scikit-learn | ≥ 1.3 | k-NN, logistic regression |
| matplotlib | ≥ 3.7 | Figure generation |
| pandas | ≥ 2.0 | Data handling |
| networkx | ≥ 3.0 | Graph operations |
| python-igraph | ≥ 0.10 | Leiden community detection |
| python-louvain | ≥ 0.16 | Louvain community detection (human validation) |
| torch | ≥ 2.0 | VGAE encoder |
| torch_geometric | ≥ 2.3 | GCN layers |
| requests | ≥ 2.28 | STRING API fallback |
| seaborn | ≥ 0.12 | Statistical visualization |
| pyyaml | ≥ 6.0 | YAML configuration parsing |

Full spec → [`requirements.txt`](requirements.txt) · [`environment.yml`](environment.yml)

---

## Extension Modules (v1.1+)

Beyond the core 21-step pipeline, the framework provides extensible modules for advanced analyses. Modules marked with * are integrated into `run_all_analysis.py` (Steps 22–26).

| Module | Description |
|--------|-------------|
| `embed_hyperbolic.py` * | Poincare Ball embeddings via Riemannian SGD — suited for hierarchical PPI structures |
| `multispecies_loader.py` | Species registry (yeast, human, *E. coli*, mouse) with STRING network + GAF parsing |
| `temporal_network.py` | `TemporalNetwork` container for time-resolved PPI analysis (requires temporal PPI data) |
| `pathway_analysis.py` * | Fisher's exact pathway enrichment, cancer gene association, consensus communities |
| `topological_analysis.py` * | Persistent homology computation (Betti curves, persistence diagrams) via Ripser |
| `topological_stats.py` * | Topological feature extraction and statistical summaries |
| `robustness_analysis.py` * | Convergence analysis, randomization null test, power curve estimation |
| `statistical_analysis.py` * | Spearman correlations, Wilcoxon tests, bootstrap CIs, cross-species comparison |
| `input_validator.py` | Pre-flight validation for networks, embeddings, GO annotations |
| `config_loader.py` | YAML configuration loader with deep merge, validation, CLI overrides |

New species can be registered at runtime:

```python
from scripts.multispecies_loader import register_species, load_species_dataset

register_species("fly", {
    "taxon_id": "7227",
    "string_prefix": "7227",
    "name": "Drosophila melanogaster",
    "go_db": "gene_association.fb.gaf.gz",
})
G, nodes, go_map = load_species_dataset("fly", data_dir="data/fly")
```

---

## Human PPI Validation

Cross-species validation on STRING v12.0 human interactome (15,882 nodes in largest CC, 236,712 edges; 14,562 with BP GO annotations). G-F curves computed on 2,000-node subsample using Louvain community detection (100 r-points in [0.05, 0.55]):

| Method | G-F Score | Plateau Width W | Peak Purity |
|--------|:---------:|:---------------:|:-----------:|
| **Node2Vec** | **0.852** | 0.450 | 0.901 |
| DeepWalk | 0.840 | 0.450 | 0.884 |
| Spectral | 0.811 | 0.480 | 0.839 |
| DM | 0.515 | 0.202 | 0.765 |
| MDS | 0.415 | 0.202 | 0.683 |
| VGAE | 0.270 | 0.364 | 0.427 |

- Human unified interval: **[0.05, 0.297]**
- Cross-species rank reversal: Node2Vec/DeepWalk rank lowest on yeast but highest on human, demonstrating that G-F scores are **network-specific** rather than universally biased
- DM drops from 1st (yeast) to 4th (human), confirming the framework discriminates context-dependent embedding quality

```bash
cd human_validation
python run_all_human.py
```

Details → [`human_validation/README.md`](human_validation/README.md) · Figure → [`figures/Fig6_human_validation.png`](figures/Fig6_human_validation.png)

---

## Topological Analysis Figures

Persistent homology analysis (Vietoris–Rips complexes via Ripser) generates additional figures characterizing the topological structure of PPI embeddings:

| Figure | File | Description |
|--------|------|-------------|
| Fig8 | `Fig8_betti_curves.png` | Betti number curves (β₀, β₁) across filtration scales for all embedding methods |
| Fig9 | `Fig9_topo_vs_standard_purity.png` | Comparison of topological purity vs. standard G-F purity profiles |
| Fig10 | `Fig10_persistence_diagrams.png` | Persistence diagrams (H₀, H₁) for each embedding method |
| Fig11 | `Fig11_topo_consistency_vs_gf_score.png` | Scatter plot of topological consistency score vs. G-F Score |
| Fig12 | `Fig12_topo_metric_correlations.png` | Correlation matrix of topological metrics (Betti numbers, persistence, entropy) |
| Fig13 | `Fig13_topo_vs_standard_gf_bars.png` | Bar comparison of standard vs. topologically-weighted G-F scores |
| Fig14 | `Fig14_human_topo_scatter.png` | Human PPI topological feature scatter (cross-species validation) |

Generated by `scripts/topological_analysis.py` and `scripts/topological_stats.py`. See `results/topological_analysis.json` for raw numerical data.

---

## Limitations

- **Network scale**: The primary ranking is based on a curated 153-node yeast subnetwork. Full-network (5,936 nodes) and cross-species (15,882 nodes) validations confirm general trends, but fine-grained method ordering may vary with network size.
- **GO annotation bias**: G-F Score depends on GO annotation quality and coverage. Well-studied genes have richer annotations, potentially inflating purity for communities dominated by such genes.
- **GO DAG propagation artifact**: True Path Rule expansion (Step 19) increases annotations from ~3.8 to ~28.9 terms/gene, causing community purity to approach 1.0 (G-F Score ≈ 0.9996). This is a known artifact of hierarchical expansion; the main results use pre-propagation annotations.
- **Community detection**: Only greedy modularity optimization is used for distance-threshold communities. Alternative algorithms (Leiden, Louvain) may produce different purity profiles.
- **2D projection**: All embeddings are standardized to σ = 0.3 and projected to 2D before G-F analysis. High-dimensional geometric properties may not be fully captured in 2D.
- **Plateau width**: Defined as the r-interval where purity ≥ 80% of each method's peak (relative threshold). Methods with very flat purity curves may yield wide plateaus despite low absolute purity.
- **Spearman correlation**: rho = 0.943 is based on n = 6 method-level observations (classical methods with both AUROC and G-F Score). Leave-one-out sensitivity analysis is provided in `results/link_prediction.json`.

---

## Author

**Yuhan Zhang (张宇涵)**  
Department of Chemical Engineering and Pharmacy, Guangling College, Yangzhou University  
Correspondence: qinray@hotmail.com  
ORCID: [0009-0000-2769-467X](https://orcid.org/0009-0000-2769-467X)

---

## Citation

> Zhang, Y. "A Geometric-Functional Consistency Framework for Evaluating Protein Interaction Network Embeddings." (2026)
>
> ```bibtex
> @article{zhang2026gf,
>   title   = {A Geometric-Functional Consistency Framework for Evaluating
>              Protein Interaction Network Embeddings},
>   author  = {Zhang, Yuhan},
>   year    = {2026},
>   note    = {Reproducible pipeline: 11 methods, 21-step validation.},
> }
> ```

---

## License

[MIT](LICENSE) — free for academic and commercial use.