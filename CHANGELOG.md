# Changelog

All notable changes to the G-F Consistency Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `metric_comparison.py`: G-F Score vs link prediction AUC and k-NN node classification F1 across all 11 embedding methods (Step 27)
- `bootstrap_correlations.py`: bootstrap 95% CI for key Spearman correlations with 10,000 resamples (Step 28)
- `semantic_purity.py`: IC-weighted purity and Resnik MICA semantic similarity — addresses DAG expansion inflation and provides GO DAG–aware community evaluation (Step 29)
- `semantic_similarity_analysis.py`: robustness check comparing 3 purity variants (standard, IC-weighted, Resnik semantic) across all 11 methods with DAG inflation diagnostics (Step 29)
- `semantic_purity_analysis.json` and `Fig16_semantic_purity_comparison.png` in results/ and figures/
- Steps 27-29 integrated into `run_all_analysis.py` pipeline
- `generate_final_summary()` now merges metric comparison, bootstrap correlation, and semantic purity results

### Changed
- `Supplementary_Materials.txt`: added correct Table S3 (embedding hyperparameters matching actual code, all 2D output), synced with submission version
- `requirements.lock.txt`: converted from UTF-16 LE to UTF-8 encoding
- Renamed `comparison_30vs200_points.png` to `FigS8_sampling_density_comparison.png` for naming consistency
- `final_results_summary.json`: corrected stale Leiden baseline purity (0.689 → 0.180)
- Version unified to 1.2.0 across `pyproject.toml`, `scripts/__init__.py`, `run_all_analysis.py`
- README: updated Key Results table with all 11 methods' link prediction AUC and k-NN F1, updated Spearman correlations to n=11 with bootstrap CIs, corrected Embedding Methods table (all 2D output, accurate hyperparameters), added Steps 27-28 to Pipeline Overview, added new scripts to Project Structure and Extension Modules

### Fixed
- `run_all_analysis.py`: integrated Steps 27-29 (metric_comparison, bootstrap_correlations, semantic_purity); updated `--start-from` range (1-29) and `--skip-extended` scope (Steps 16-21, 24-29)
- `Supplementary_Materials.txt` Algorithm S1 step 2c: corrected purity formula from `(max GO term count / community size)` to `(max GO term count / total GO terms in community)`
- `data_preprocessing.py`: corrected filename from `yeast_ppi_5966.edgelist` to `yeast_ppi_5936.edgelist` (docstring + code)
- README: corrected all "21-step" references to "29-step" to match actual pipeline scope
- `scripts/__init__.py`: added Steps 22-29 module section; removed duplicate entries from Extension modules

## [1.2.0] — 2026-06-11

### Fixed
- `leiden_baseline.py`: unified purity formula with G-F curve (`most_common / total_GO_terms` instead of `most_common / cluster_size`); baseline purity updated from 0.689 to 0.180
- `compute_gf.py`: replaced `hash(method)` with `zlib.crc32(method.encode())` for cross-process reproducibility (2 occurrences)
- `robustness_analysis.py`: replaced `hash(tuple(q_idx))` with `zlib.crc32(str(q_idx).encode())` for deterministic seeding
- Replaced 14 `list.index()` calls with dict lookups across 10 files for O(1) node alignment: `randomization_control.py`, `sampling_density.py`, `embed_hyperbolic.py`, `biological_interpretation.py` (3 sites), `embed_gnn.py`, `benchmark_runtime.py` (2 sites), `pathway_analysis.py`, `rank_reversal_analysis.py` (3 sites), `robustness_analysis.py`, `network_topology_analysis.py`
- `biological_interpretation.py`: added `node_to_idx` dict in `generate_three_panel_figure()` to replace `nodes.index()` call
- `leiden_baseline.json`: regenerated with corrected purity formula
- README, Key_Results_Summary, Submission_Checklist: Leiden baseline value updated from 0.689 to 0.180; conclusion revised — Leiden baseline now comparable to best G-F Score, indicating spatial embeddings capture functional structure at a level similar to graph-based community detection
- `robustness_analysis.py` main(): corrected function call signatures for `convergence_analysis()` and `randomization_null_test()` to match current API
- `human_gf_all.py`: replaced hardcoded fallback node count (15882) and fragile `dir()` check with `len(nodes)` from the actual embedding
- `network_topology_analysis.py`: parameterised `human_key` in `generate_radar_chart()` instead of hardcoding `"human_15882"`
- Human PPI node count references unified across all files (~15,882 after score filtering; 14,679 in largest CC)
- Replaced "placeholder" wording with "fallback"/"simplified" in human validation scripts
- Manuscript (`format_manuscript.py`): node count consistency, small-sample caveat added

### Added
- README Quick Start: `git lfs pull` step for large data files

## [1.1.1] — 2026-06-08

### Added
- Mathematical proofs (Propositions 1-2, Theorems 1-4) in Supplementary_Materials.txt
- Topological analysis integration (Steps 22-23): persistent homology via Ripser, Betti curves, topological statistics
- Extension module pipeline integration (Steps 24-26): hyperbolic embedding, pathway enrichment, statistical summary
- `--skip-topological` CLI flag

### Fixed
- `human_embed_all.py` import error: `standardize_coordinates` → `rescale_coordinates`
- README: corrected Supplementary_Materials description (PDF contains figures, not proofs)
- Renamed `yeast_ppi_5966.edgelist` → `yeast_ppi_5936.edgelist` (correct node count)
- Randomization test standard deviation in README (0.007 → 0.002)

### Changed
- Extension modules now integrated into `run_all_analysis.py` (Steps 22-26)
- Pipeline expanded from 21 steps to 26 steps

## [1.1.0] — 2026-06-02

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

## [1.0.0] — 2026-05-20

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
