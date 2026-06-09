## Submission Checklist — Bioinformatics (Oxford)
### G-F Consistency Framework v1.1.1
### Author: Yuhan Zhang (张宇涵) | ORCID: 0009-0000-2769-467X

---

### A. Required Manuscript Files

- [x] Main manuscript (.docx) — `GF-Consistency Framework.docx`
- [x] Supplementary Materials (.pdf) — `supplementary/Supplementary_Materials.pdf`
- [x] Supplementary Materials (.txt source) — `supplementary/Supplementary_Materials.txt`
- [x] Cover Letter — `Cover_Letter.md`
- [x] CHANGELOG.md — version history
- [x] Key Results Summary — `Key_Results_Summary.md`

---

### B. Bioinformatics Compliance

| Requirement | Status | Notes |
|---|---|---|
| Article type | Original Article | |
| Abstract (≤250 words) | 220 words | Verified |
| Keywords (6) | Included | PPI networks, network embedding, random geometric graphs, functional modules, diffusion maps, evaluation framework |
| Data Availability Statement | Included | github.com/Qinrayn/GF-consistency-framework |
| Software Availability | Included | pip installable via pyproject.toml |
| Funding statement | Included | No specific funding |
| Author contributions (CRediT) | Included | Y.Z. all roles |
| Competing Interests | Included | None declared |
| Ethics approval | N/A | Computational study using public data |
| References | 21 refs, numbered | Bioinformatics Vancouver style |
| Supplementary: single PDF | Yes | |

---

### C. Main Figures (14 figures, 300 DPI, colorblind-safe Okabe-Ito palette)

- [x] `Fig1_GF_curves.png` — G-F purity + modularity curves (8 methods)
- [x] `Fig2_PCA_control.png` — PCA control analysis
- [x] `Fig3_subset_robustness.png` — Subset robustness (DM vs MDS, Bonferroni)
- [x] `Fig4_full_network.png` — Full 5,936-node network validation
- [x] `Fig5_GF_scores.png` — G-F Score ranking (11 methods, fixed labels)
- [x] `Fig6_human_validation.png` — Cross-species human validation
- [x] `Fig7_plateau_width.png` — Plateau width analysis (relative threshold)
- [x] `Fig8_betti_curves.png` — Betti curves β₀ and β₁ for all methods
- [x] `Fig9_topo_vs_standard_purity.png` — Topological vs standard purity comparison
- [x] `Fig10_persistence_diagrams.png` — Persistence diagrams (H0, H1) for all methods
- [x] `Fig11_topo_consistency_vs_gf_score.png` — Topological consistency vs G-F Score
- [x] `Fig12_topo_metric_correlations.png` — Multi-panel topological metric correlations
- [x] `Fig13_topo_vs_standard_gf_bars.png` — Standard vs topological G-F Score bars
- [x] `Fig14_human_topo_scatter.png` — Human PPI topological validation (3-panel)

### D. Supplementary Figures (8 figures)

- [x] `FigS1_topology_radar.png` — Network topology radar chart
- [x] `FigS2_embedding_distance_distributions.png` — Pairwise distance distributions
- [x] `FigS3_node2vec_pq_heatmap.png` — Node2Vec p/q sensitivity heatmap
- [x] `FigS4_sample_size_convergence.png` — Sample size convergence analysis
- [x] `FigS5_biological_case_study.png` — GO biological case study
- [x] `FigS6_runtime_breakdown.png` — Pipeline runtime breakdown
- [x] `FigS7_time_accuracy_tradeoff.png` — Time-accuracy tradeoff
- [x] `comparison_30vs200_points.png` — Sampling density comparison

---

### E. Key Results (Verified v1.1.1)

| Metric | Value |
|---|---|
| Best method (G-F Score) | Spectral: 0.163 |
| Spearman rho (AUROC vs G-F) | 0.943 (p = 0.005) |
| Spearman LOO range | [0.900, 1.000], mean 0.933 |
| Bonferroni significant | 9/30 subsets |
| Randomization Z-score | 6.95 |
| Unified interval | [0.05, 0.422] |
| Leiden baseline purity | 0.180 (consistent formula) |
| Embedding methods | 11 (6 classical + 2 curated + 3 GNN) |
| Pipeline steps | 21 |
| Species validated | Yeast + Human |
| Plateau width method | Relative threshold (80% peak) |
| H1 max persistence vs G-F (yeast) | rho = 0.764, P = 0.006 |
| Topo consistency vs G-F (human) | rho = -0.143, P = 0.787 |
| Human subsample size | 2,000 nodes |
| Unit tests | 51 passed |

---

### F. Software & Reproducibility (v1.1.1)

- [x] `pyproject.toml` — pip-installable package (`gf-consistency`)
- [x] `pipeline_config.yaml` — YAML configuration system
- [x] `tests/` — 51 unit tests (pytest)
- [x] `scripts/config_loader.py` — config validation
- [x] `scripts/input_validator.py` — pre-flight input checks
- [x] `environment.yml` — conda environment specification
- [x] `requirements.txt` — pip dependencies (lower bounds)
- [x] `requirements.lock.txt` — exact version lock (pip freeze)
- [x] GitHub: github.com/Qinrayn/GF-consistency-framework
- [x] Git tag: v1.1.1

### G. Extension Modules (v1.1.0)

- [x] `scripts/embed_hyperbolic.py` — Poincare Ball embeddings
- [x] `scripts/multispecies_loader.py` — 4 species (yeast, human, E. coli, mouse)
- [x] `scripts/temporal_network.py` — Dynamic PPI framework
- [x] `scripts/pathway_analysis.py` — Pathway enrichment + cancer genes

---

### H. Pre-Submission Verification (v1.1.1)

1. [x] Abstract word count: 220 words (≤250 limit)
2. [x] Funding Statement added to manuscript
3. [x] CRediT Author Contributions added to manuscript
4. [x] Competing Interests declaration added (None)
5. [x] Data Availability statement added
6. [x] Supplementary Information note added
7. [x] Potential reviewers filled in Cover Letter (3 suggestions)
8. [x] All citations converted to numbered format [1]-[21]
9. [x] 21 references verified (no orphans, no fabrications)
10. [x] GNN methods properly cited (GraphSAGE, GAT, GIN)
11. [x] Embedding hyperparameters detailed in Methods 2.2
12. [x] Topological consistency score mathematically defined in Methods 2.7
13. [x] Non-significant results properly qualified
14. [x] G-F interval sensitivity analysis referenced
15. [x] Human subsample corrected from 500 to 2,000 nodes
16. [x] Figures at 300 DPI (all 22 verified)
17. [x] Final read-through for consistency with v1.1.1 results
