# Changelog

All notable changes to the G-F Consistency Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-06-05

### Added
- **YAML Configuration System** (`pipeline_config.yaml` + `scripts/config_loader.py`): users can now customise all pipeline parameters (species, methods, intervals, thresholds, paths) via a single YAML file without editing script source code. Supports deep merge, validation, and CLI overrides.
- **pytest Test Suite** (`tests/`): 51 unit tests covering core computation functions (`compute_gf_curve`, `compute_gf_score`, `_community_purity`, `compute_plateau_width`), config loader, deep merge, and validation logic.
- **Input Validation** (`scripts/input_validator.py`): pre-flight checks for edgelists, embeddings, GO annotations, STRING files, and pipeline parameters. Runs automatically before the pipeline starts.
- **Hyperbolic Space Embeddings** (`scripts/embed_hyperbolic.py`): Poincare Ball model with Riemannian SGD optimisation, geodesic distance computation, and Mobius arithmetic. Includes G-F curve computation on hyperbolic distances.
- **Multi-Species Dataset Loader** (`scripts/multispecies_loader.py`): species registry (yeast, human, E. coli, mouse) with STRING network loading, GAF parser, and extensible `register_species()` API.
- **Temporal / Dynamic Network Framework** (`scripts/temporal_network.py`): `TemporalNetwork` container, time-stamped edgelist parser, per-snapshot G-F analysis, and temporal consistency scoring.
- **Pathway-Level Biological Analysis** (`scripts/pathway_analysis.py`): Fisher's exact test for pathway enrichment, cancer gene association, cross-method consensus communities, and signalling pathway perturbation analysis.
- `--config` CLI flag to specify a custom configuration file
- `--seed` and `--species` CLI flags for quick overrides
- `pytest-cov` to dev dependencies for coverage reporting
- Pre-flight validation step at pipeline startup

### Changed
- `run_all_analysis.py` now loads configuration from `pipeline_config.yaml` (with CLI overrides taking precedence)
- Pipeline banner now shows version, species, and seed
- `scripts/__init__.py` updated with extension module documentation

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
