# G-F Consistency Framework

**A Geometric-Functional Consistency Framework for Evaluating Protein Interaction Network Embeddings**

Complete reproduction of the experimental pipeline: 11 embedding methods · 39-step validation workflow · 200-point G-F curve sampling · publication-quality figures · `random_seed = 42`

<details>
<summary><strong>Key Results at a Glance</strong></summary>

| Method | G-F Score | Link Pred. AUC | k-NN micro-F1 |
|--------|:---------:|:--------------:|:-------------:|
| **Spectral** | **0.163** | 0.885 | 0.672 |
| DM | 0.155 | 0.726 | 0.513 |
| MDS | 0.152 | 0.904 | 0.639 |
| Node2Vec | 0.151 | 0.491 | 0.185 |
| PCA | 0.138 | 0.678 | 0.521 |
| VGAE-feat | 0.124 | 0.466 | 0.168 |
| DeepWalk | 0.123 | 0.489 | 0.193 |
| GIN | 0.122 | 0.483 | 0.210 |
| GAT | 0.069 | 0.534 | 0.160 |
| GraphSAGE | 0.069 | 0.531 | 0.177 |
| VGAE | 0.066 | 0.491 | 0.227 |

- Leiden baseline purity: **0.180** (same formula as G-F curve: most-common GO term / total GO terms)
- Leiden baseline ≈ best G-F Score (0.180 vs 0.163), indicating spatial embeddings capture functional structure at a level comparable to graph-based community detection
- Bonferroni (30 subsets, size 150): **9/30** significant after correction
- Randomization: original max 0.247 > shuffled 0.230 ± 0.002 (10 permutations, Z = 6.95)
- Spearman rho (G-F Score vs Link Pred AUC): **+0.591** (*P* = 0.056, 95% CI [−0.09, 0.88], n = 11)
- Spearman rho (G-F Score vs k-NN F1): **+0.609** (*P* = 0.047, 95% CI [−0.05, 0.92], n = 11)
- Spearman rho (standard vs IC-weighted purity): **ρ = +0.964** (*P* < 0.001) — IC-weighting preserves rankings while suppressing DAG inflation
- Spearman rho (standard vs semantic purity): **ρ = +0.491** (*P* = 0.125) — semantic (Resnik) captures complementary coherence signal
- H1 max persistence vs G-F Score: **ρ = +0.764** (*P* = 0.006, 95% CI [0.27, 0.97])
- Cross-species rank consistency (yeast vs human, 11 methods): **ρ = +0.500** (*P* = 0.117), **Kendall W = 0.750** — strong concordance; Spectral #1 and MDS #2 in both species
- Scale gradient Kendall W (500-4000 nodes, 4 methods): **W = 0.700** — rank stability across scales; PCA consistently #1 at all scales
- Density-corrected GF (STRING threshold gradient, 4 methods): **W<sub>raw</sub> = 0.178 → W<sub>corrected</sub> = 0.70** (ΔW = 0.522) — density correction dramatically improves rank concordance
- Human seed stability (10 seeds, 11 methods): **Kendall W = 0.675** — strong rank stability; Spectral mean rank 1.1 (std 0.3), consistently #1
- Human IC-weighted GF (11 methods): **Spearman ρ = +0.991** (*P* < 0.001) — IC-weighting preserves rankings while correcting for DAG inflation
- GAT collapse diagnosis (5 variants): all variants show persistent collapse — architectural, not optimization-related
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

# 2. Pull large data files (tracked via Git LFS)
git lfs pull

# 3. Run full pipeline (≈ 60 min on standard laptop)
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
python run_all_analysis.py --skip-extended      # Skip extended analyses (Steps 16-21, 24-39)
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
| **purity(r)** | Mean over communities of (dominant GO term count / total GO terms in community) at distance threshold *r* |
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
Step 27 ─ Metric Comparison ────────────── G-F Score vs link prediction AUC + k-NN F1
Step 28 ─ Bootstrap Correlations ───────── 95% CI for key Spearman correlations (10k resamples)
Step 29 ─ Semantic Purity Analysis ─────── IC-weighted + Resnik semantic purity + DAG inflation diagnosis
Step 30 ─ Cross-Species Consistency ───── Yeast vs human rank concordance (Spearman + Kendall W)
Step 31 ─ Scale Gradient Analysis ──────── Scale-dependent topology coupling (500-4000 nodes)
Step 32 ─ Bootstrap Stability ─────────── 30-resample CI for G-F Score rankings (80% sampling)
Step 33 ─ Human Extended Embeddings ───── 11-method human network embeddings (PCA, VGAE-feat, GNNs)
Step 34 ─ Multi-Modal Anchoring ───────── STRING threshold gradient + channel-specific GF analysis
Step 35 ─ Hyperparameter Sensitivity ──── r-points, resolution, dimensions, walk parameters
Step 36 ─ Density-Corrected GF ─────────── STRING threshold gradient with random baseline correction (ΔW = 0.522)
Step 37 ─ Human Seed Stability ──────────── 10-seed subsampling stability (Kendall W = 0.675)
Step 38 ─ Human IC-Weighted GF ─────────── IC-weighted purity on human network (ρ = 0.991)
Step 39 ─ GAT Collapse Diagnosis ────────── 5-variant ablation: clip, warmup, multi-head analysis
         └─ Summary ─────────────────── final_results_summary.json
```

---

## Embedding Methods

| Method | Input | Strategy | 2D Output |
|--------|-------|----------|-----------|
| Diffusion Map (DM) | 6 centrality features → Markov matrix | Eigendecomposition | 2nd and 3rd eigenvectors |
| Classical MDS | Shortest-path distances | Double-centering + eigendecomposition | Top-2 eigenvectors |
| Spectral | Normalized Laplacian | Eigendecomposition | Fiedler vectors (eig 1,2) |
| DeepWalk | Random walks → co-occurrence | Truncated SVD | Top-2 singular vectors |
| Node2Vec | Biased random walks → co-occurrence | Truncated SVD | Top-2 singular vectors |
| VGAE | One-hot identity matrix | 2-layer GCN VAE (hidden=4, 300 epochs, Adam lr=0.01) | Latent mean μ |
| VGAE-feat | 6 centrality features | 2-layer GCN VAE (hidden=4, 300 epochs) | Latent mean μ |
| PCA | 6 centrality features | PCA via covariance eigendecomposition | Top-2 components |
| GraphSAGE | 6 centrality features | 2-layer SAGEConv mean agg (hidden=16, 300 epochs) | Latent space |
| GAT | 6 centrality features | 2-layer GATConv 1-head (hidden=16, 300 epochs) | Latent space |
| GIN | 6 centrality features | 2-layer GINConv + MLP (hidden=16, 300 epochs) | Latent space |

All embeddings standardized to **σ = 0.3** before G-F analysis.

---

## Project Structure

```
GF-consistency-framework/
├── scripts/                    # 39-step analysis pipeline + extensions
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
│   ├── metric_comparison.py    # G-F Score vs link prediction AUC + k-NN F1 (Step 27)
│   ├── bootstrap_correlations.py # Bootstrap 95% CI for Spearman correlations (Step 28)
│   ├── semantic_purity.py      # IC-weighted + Resnik semantic purity core library (Step 29)
│   ├── semantic_similarity_analysis.py  # 3-variant purity comparison + DAG diagnostics (Step 29)
│   ├── cross_species_consistency.py     # Yeast vs human rank concordance analysis (Step 30)
│   ├── scale_gradient.py               # Scale-dependent topology coupling (Step 31)
│   ├── bootstrap_stability.py          # Bootstrap CI for G-F Score rankings (Step 32)
│   ├── human_embed_extended.py         # Extended human embeddings (Step 33a)
│   ├── human_gf_extended.py            # Human GF analysis all 11 methods (Step 33b)
│   ├── multimodal_functional_anchoring.py # STRING threshold + channel analysis (Step 34)
│   ├── hyperparameter_sensitivity.py   # Parameter sensitivity analysis (Step 35)
│   ├── density_corrected_gf.py         # Density-corrected GF with random baseline (Step 36)
│   ├── human_seed_stability.py         # Human 10-seed stability analysis (Step 37)
│   ├── human_ic_weighted_gf.py         # Human IC-weighted GF analysis (Step 38)
│   ├── gat_collapse_diagnosis.py       # GAT variant ablation study (Step 39)
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
│   ├── Fig1–14                 # Main figures (G-F curves, topology, human validation)
│   ├── Fig15                   # Metric comparison scatter (G-F vs link pred vs k-NN)
│   ├── Fig16                   # Semantic purity comparison (standard, IC-weighted, Resnik)
│   ├── Fig17                   # Cross-species rank consistency (yeast vs human)
│   ├── Fig18                   # Scale-dependent topology coupling (500-4000 nodes)
│   ├── Fig19                   # Bootstrap stability of G-F Score rankings
│   ├── Fig20                   # Multi-modal functional anchoring (threshold + channel)
│   ├── Fig21                   # Hyperparameter sensitivity analysis
│   ├── FigS1–S7                # Supplementary figures
│   └── FigS8                   # Sampling density comparison
│
├── human_validation/           # Cross-species (optional, STRING v12.0)
│
├── run_all_analysis.py         # One-command Python pipeline (39 steps)
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
| Human PPI | STRING v12.0 | 15,882 (14,679 in largest CC) | Cross-species validation |
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

Beyond the core 39-step pipeline, the framework provides extensible modules for advanced analyses. Modules marked with * are integrated into `run_all_analysis.py` (Steps 22–39).

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
| `metric_comparison.py` * | G-F Score vs link prediction AUC and k-NN node classification F1 across all 11 methods |
| `bootstrap_correlations.py` * | Bootstrap 95% CI for key Spearman correlations (10,000 resamples) |
| `semantic_purity.py` * | IC-weighted purity and Resnik MICA semantic similarity for GO DAG–aware community evaluation |
| `semantic_similarity_analysis.py` * | Robustness check: 3-variant G-F score comparison + DAG inflation diagnostics (Fig 16) |
| `cross_species_consistency.py` * | Cross-species rank concordance: yeast vs human Spearman + Kendall W (Fig 17) |
| `scale_gradient.py` * | Scale-dependent topology coupling: 500-4000 node gradient analysis (Fig 18) |
| `bootstrap_stability.py` * | Bootstrap stability: 30-resample 95% CI for G-F Score rankings (Fig 19) |
| `human_embed_extended.py` * | Extended human embeddings: PCA, VGAE-feat, GraphSAGE, GAT, GIN (Step 33a) |
| `human_gf_extended.py` * | Human GF analysis for all 11 methods (Step 33b) |
| `multimodal_functional_anchoring.py` * | STRING threshold gradient + channel-specific GF analysis (Fig 20, Step 34) |
| `hyperparameter_sensitivity.py` * | r-points, resolution, dimension, walk parameter sensitivity (Fig 21, Step 35) |
| `density_corrected_gf.py` * | Density-corrected GF with random baseline subtraction (Step 36) |
| `human_seed_stability.py` * | Human 10-seed subsampling stability analysis (Step 37) |
| `human_ic_weighted_gf.py` * | Human IC-weighted GF analysis across 11 methods (Step 38) |
| `gat_collapse_diagnosis.py` * | GAT variant ablation: clip, warmup, multi-head analysis (Step 39) |
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

Cross-species validation on STRING v12.0 human interactome (~15,882 nodes after score ≥ 700 filtering; 14,679 in largest CC, 236,712 edges; 14,562 with BP GO annotations). G-F curves computed on 2,000-node subsample using Louvain community detection (200 r-points, standard purity formula):

| Method | G-F Score | Plateau Width W | Peak Purity |
|--------|:---------:|:---------------:|:-----------:|
| **Spectral** | **0.402** | 0.480 | 0.478 |
| MDS | 0.367 | 0.217 | 0.671 |
| Node2Vec | 0.166 | 0.495 | 0.206 |
| GraphSAGE | 0.133 | 0.495 | 0.256 |
| DeepWalk | 0.101 | 0.495 | 0.204 |
| GIN | 0.089 | 0.379 | 0.184 |
| VGAE-feat | 0.089 | 0.495 | 0.212 |
| PCA | 0.086 | 0.495 | 0.264 |
| DM | 0.060 | 0.495 | 0.165 |
| VGAE | 0.014 | 0.500 | 0.018 |
| GAT | 0.011 | 0.500 | 0.011 |

- Human unified interval: **[0.282, 0.297]** (11 methods, standard purity)
- Spectral ranks #1 in both yeast and human, demonstrating strong cross-species consistency
- DM drops from 2nd (yeast) to 9th (human), confirming the framework discriminates context-dependent embedding quality

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
- **2D output**: All embeddings directly produce 2-dimensional coordinate spaces (not projected from higher dimensions). While 2D is sufficient for G-F curve analysis, higher-dimensional embeddings may capture additional geometric properties.
- **Plateau width**: Defined as the r-interval where purity ≥ 80% of each method's peak (relative threshold). Methods with very flat purity curves may yield wide plateaus despite low absolute purity.
- **Spearman correlations**: G-F Score vs link prediction AUC (ρ = 0.591, P = 0.056) and vs k-NN F1 (ρ = 0.609, P = 0.047) are based on n = 11 methods. Bootstrap 95% CIs indicate moderate precision; see `results/bootstrap_correlations.json` for full details.
- **All embeddings are 2-dimensional**: All 11 methods produce 2D coordinate spaces (standardized to σ = 0.3). Higher-dimensional geometric properties are not captured in this analysis.
- **GAT embedding collapse**: GAT exhibits persistent embedding collapse on the human network (Step 39), with near-zero coordinate variance. Five variants tested (gradient clipping, warmup, multi-head attention) did not resolve the issue; the collapse appears to be architectural rather than optimization-related.
- **Density-dependent rankings**: Method rankings are network-density-dependent (W<sub>raw</sub> = 0.178 across STRING thresholds 400-900), though density correction substantially improves concordance (W<sub>corrected</sub> = 0.70, Step 36).

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
>   note    = {Reproducible pipeline: 11 methods, 39-step validation.},
> }
> ```

---

## License

[MIT](LICENSE) — free for academic and commercial use.