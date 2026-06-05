# Changelog

All notable changes to the G-F Consistency Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-06-05

### Added
- Complete 21-step reproducible analysis pipeline for G-F consistency evaluation
- 11 embedding methods: DM, MDS, Spectral, DeepWalk, Node2Vec, VGAE, VGAE-feat, PCA, GraphSAGE, GAT, GIN
- G-F Score metric with relative plateau-width detection (80% of peak purity)
- Adaptive unified interval algorithm with transparency reporting (fallback ratio)
- Cross-species validation: yeast (STRING v11.5, 5,936 nodes) and human (STRING v12.0, 15,882 nodes)
- Bonferroni-corrected subset robustness analysis (30 subsets, 5 size levels)
- Randomization control with shuffled-label baseline (Z-score significance)
- Spearman rank correlation with leave-one-out sensitivity analysis
- GO DAG True Path Rule propagation (154 -> 5,429 annotated genes)
- 4-level biological interpretation framework with GO cluster case study
- Step-wise runtime benchmarking with complexity analysis
- Publication-quality figures: Okabe-Ito colorblind-safe palette, 300 DPI
- `pyproject.toml` for pip-installable package with `gf-consistency` CLI entry point
- Limitations section documenting known constraints and artifacts

### Fixed
- GIN embedding collapse: increased hidden_dim from 4 to 16, added BatchNorm1d
- Functional purity denominator: corrected to `total_terms` (not `max_count`)
- Adaptive interval diagnostics: added `r_min`/`r_max` keys for consensus computation
- Benchmark `sys.argv` leak: cleared CLI flags before importing sub-script `main()` functions
- Figure 6 guard: graceful skip when human validation data missing `r` key
- Figure 5 label truncation: increased figure width and adjusted rotation for 11 methods
- Figure S5 title clipping: fixed `tight_layout` rect and added padding
- Corrupted GAF file (14 bytes): replaced with valid 2.6 MB data

### Changed
- Plateau width: switched from absolute threshold (0.5) to relative threshold (80% of peak)
- Unified interval: [0.05, 0.422] for yeast, [0.05, 0.297] for human
- Spearman rho (classical): 0.943 (p = 0.005), leave-one-out range [0.900, 1.000]
