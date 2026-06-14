# Changelog

All notable changes to the G-F Consistency Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (Phase 6: Formal Proofs of GAT Collapse)
- `gat_collapse_formal_proof.py`: Formalises Phase 4 empirical theory into 3 theorems with proofs and numerical verification — T1 (attention degeneration bound), T2 (effective rank bound for mean-aggregation GNN), T3 (G-F Score upper bound for rank-1 embeddings), plus combined corollary validated by Phase 5B dimension sweep (Phase 6, Fig 42-43)
- `phase6_formal_proof_report.md`: Comprehensive proof report with theorem statements, proof sketches, numerical verification, and practical implications
- `gat_collapse_formal_proof.json` and Figs 42-43

### Added (Phase 5: Self-Validation and Causal Disentanglement)
- `human_spectral_alignment.py`: Cross-network two-factor model transfer test — replicates Phase 3 spectral alignment on human STRING v12.0 (1,310-node largest CC), tests whether spectral alignment + effective dimensionality predict human G-F Score (Phase 5A, Fig 39-40)
- `gat_dimension_sweep.py`: GAT latent dimension sweep d={2,4,8,16,32} with GraphSAGE control — measures G-F Score, attention entropy, effective dimensionality, matrix rank at each dimension; proves attention degeneration is dimension-independent (Phase 5B, Fig 41)
- `phase5_deep_analysis_report.md`: Comprehensive Phase 5 report with 8 findings, revised GAT causal chain, negative result interpretation
- `human_spectral_alignment.json`, `gat_dimension_sweep.json` and Figs 39-41

### Added (Deep Analysis Modules)
- `deep_geometric_analysis.py`: Multi-scale geometric fingerprint decomposition — distance-function correspondence (DFC), geometric feature extraction (6 features), G-F curve shape decomposition across 11 methods (Phase 1, Fig 26-29)
- `geometric_predictor.py`: Cross-species geometric predictability model — builds yeast-trained geometric predictor and validates on human PPI; includes network Laplacian spectrum analysis, collapse diagnostics, method clustering (Phase 2, Fig 30-33)
- `spectral_alignment.py`: Network-aware spectral alignment — decomposes embeddings in Laplacian eigenbasis, identifies functional frequency band, computes alignment scores predicting G-F Score (Phase 3, Fig 34-35)
- `gat_collapse_theory.py`: Mathematical theory of GAT collapse — 4-pillar analysis: attention degeneration (entropy=0.973), rank collapse (eff_rank=1.019), per-node density-collapse, architectural impossibility across 5 variants (Phase 4, Fig 36-38)
- `deep_geometric_analysis.json`, `geometric_predictor.json`, `spectral_alignment.json`, `gat_collapse_theory.json` and Figs 26-38

### Added
- `density_corrected_gf.py`: Density-corrected G-F Score analysis — computes random baseline via GO-label permutation at each STRING threshold (400-900), normalizes GF scores as (GF_method - GF_random)/(1 - GF_random), and recalculates Kendall's W (Step 36, Fig 22)
- `human_seed_stability.py`: Seed stability analysis — 10 random seeds × 2000-node subsamples × 11 methods on human PPI, measuring Kendall's W and per-method CV across seeds (Step 37, Fig 23)
- `human_ic_weighted_gf.py`: IC-weighted G-F Score on human PPI — computes corpus-based IC for all human GO terms and produces parallel standard vs IC-weighted GF rankings for 11 methods (Step 38, Fig 24)
- `gat_collapse_diagnosis.py`: GAT embedding collapse root-cause analysis — tests 5 variants (baseline, gradient clipping, LR warmup, combined, multi-head attention) and measures attention entropy, embedding collapse diagnostics, and GF Scores (Step 39, Fig 25)
- `density_corrected_gf.json`, `human_seed_stability.json`, `human_ic_weighted_gf.json`, `gat_collapse_diagnosis.json` and Figs 22-25
- `metric_comparison.py`: G-F Score vs link prediction AUC and k-NN node classification F1 across all 11 embedding methods (Step 27)
- `bootstrap_correlations.py`: bootstrap 95% CI for key Spearman correlations with 10,000 resamples (Step 28)
- `semantic_purity.py`: IC-weighted purity and Resnik MICA semantic similarity — addresses DAG expansion inflation and provides GO DAG–aware community evaluation (Step 29)
- `semantic_similarity_analysis.py`: robustness check comparing 3 purity variants (standard, IC-weighted, Resnik semantic) across all 11 methods with DAG inflation diagnostics (Step 29)
- `semantic_purity_analysis.json` and `Fig16_semantic_purity_comparison.png` in results/ and figures/
- `cross_species_consistency.py`: cross-species rank concordance analysis — yeast vs human Spearman correlation + Kendall W + rank shift analysis (Step 30)
- `scale_gradient.py`: scale-dependent topology coupling — subsamples yeast STRING network at 500/1000/2000/4000 nodes and measures G-F Score stability across scales (Step 31)
- `bootstrap_stability.py`: bootstrap stability analysis — 30 resamples with 80% sampling to compute 95% CI, CV, and pairwise rank stability for all 11 methods (Step 32)
- `cross_species_consistency.json`, `scale_gradient.json`, `bootstrap_stability.json` and Figs 17-19 in results/ and figures/
- `human_embed_extended.py`: extended human embeddings — computes PCA, VGAE-feat, GraphSAGE, GAT, GIN on 15,882-node human STRING network using sparse negative-sampling GNN training (Step 33a)
- `human_gf_extended.py`: G-F analysis for all 11 methods on the human network, producing unified 11-method ranking for cross-species comparison (Step 33b)
- `multimodal_functional_anchoring.py`: STRING threshold gradient (400-900) + channel-specific network analysis — evaluates G-F Score robustness across evidence modalities (Step 34, Fig 20)
- `hyperparameter_sensitivity.py`: systematic sensitivity analysis of r-points, Louvain resolution, embedding dimensions, walk length, window size, and Node2Vec (p,q) grid (Step 35, Fig 21)
- `human_gf_scores_extended.json`, `multimodal_anchoring.json`, `hyperparameter_sensitivity.json` and Figs 20-21 in results/ and figures/
- Steps 27-35 integrated into `run_all_analysis.py` pipeline
- `generate_final_summary()` now merges metric comparison, bootstrap correlation, semantic purity, cross-species, scale gradient, bootstrap stability, human extended, multi-modal, and hyperparameter sensitivity results

### Changed
- README: added Phase 1-6 deep analysis modules, Fig 26-43, key geometric, collapse, self-validation, and formal proof findings
- `Supplementary_Materials.txt`: added correct Table S3 (embedding hyperparameters matching actual code, all 2D output), synced with submission version
- `requirements.lock.txt`: converted from UTF-16 LE to UTF-8 encoding
- Renamed `comparison_30vs200_points.png` to `FigS8_sampling_density_comparison.png` for naming consistency
- `final_results_summary.json`: corrected stale Leiden baseline purity (0.689 → 0.180)
- Version unified to 1.5.0 across `scripts/__init__.py`; added Steps 36-39 documentation
- `__init__.py`: registered Steps 36-39 modules in package docstring
- Version unified to 1.4.0 across `pyproject.toml`, `scripts/__init__.py`, `run_all_analysis.py`
- README: updated to "35-step" pipeline; added Steps 33-35 to Pipeline Overview, Project Structure, and Extension Modules table; updated `--skip-extended` scope
- `pipeline_config.yaml`: updated `start_from` range (1-35) and `skip_extended` scope (Steps 16-21, 24-35)

### Fixed
- **v6 audit — Purity formula standardization across all pipeline steps:**
  - `bootstrap_stability.py`: collect all GO terms per community instead of `terms[0]`; divide by `total_terms` instead of `len(labels)` (P0)
  - `hyperparameter_sensitivity.py`: same purity formula fix — all GO terms + `most_common / total_terms` (P1)
  - `human_gf_extended.py` + `human_gf_all.py`: store all GO terms per node (not just `terms[0]`); purity uses `most_common / total_terms` (P1)
  - `biological_interpretation.py`: `_cluster_quality` purity fix — `dom_count / len(go_terms_all)` instead of `dom_count / len(node_names)` (P0, missed in earlier fix)
  - `biological_interpretation.py`: `DEFAULT_GF_SCORES` fallback updated to current values
  - Cross-species consistency improved: ρ = 0.500, W = 0.750 (was ρ = 0.355, W = 0.677 with non-standard purity)
- **v6 audit — encoding and style cleanup:**
  - Added `encoding="utf-8"` to all `open()` calls in `cross_species_consistency.py`, `bootstrap_stability.py`, `hyperparameter_sensitivity.py`, `scale_gradient.py`, `multimodal_functional_anchoring.py` (P1)
  - Removed `from __future__ import annotations` from `semantic_purity.py` and `semantic_similarity_analysis.py` (P2)
  - Removed `>>>` docstring examples from `biological_interpretation.py` and `config_loader.py` (P2)
  - `leiden_baseline.py`: docstring expected purity updated from 0.6886 to 0.180
  - `visualization_helpers.py`: demo convergence data updated to match current bootstrap values; human interval updated to [0.282, 0.297]
- `run_all_analysis.py`: integrated Steps 27-35; updated `--start-from` range (1-35) and `--skip-extended` scope (Steps 16-21, 24-35)
- `Supplementary_Materials.txt` Algorithm S1 step 2c: corrected purity formula from `(max GO term count / community size)` to `(max GO term count / total GO terms in community)`
- `data_preprocessing.py`: corrected filename from `yeast_ppi_5966.edgelist` to `yeast_ppi_5936.edgelist` (docstring + code)
- README: corrected all step count references to "35-step" to match actual pipeline scope
- README: updated Human PPI Validation table to 11 methods with standard purity results (Spectral #1: 0.402); cross-species values updated to ρ=0.500, W=0.750; human interval updated to [0.282, 0.297]
- `scripts/__init__.py`: added Steps 22-35 module sections; version bumped to 1.4.0

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
