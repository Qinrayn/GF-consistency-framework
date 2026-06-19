# Verification Log (June 2026)

Scripts and results were developed locally since February 2026. Throughout June, each batch was reviewed for correctness, output validity, and code quality before being published to this repository. This log records what was audited each day and what issues were found.

---

## 2026-06-04

**Scope:** Core framework review (Steps 1-15) — initial push of locally developed codebase.

- `data_preprocessing.py`, `embed_all.py`, `compute_gf.py`: reviewed 200-point G-F curve sampling, unified interval [0.05, 0.422]
- `embed_gnn.py`: reviewed GraphSAGE/GAT/GIN training (hidden=16, 300 epochs), confirmed GIN collapse behaviour
- `leiden_baseline.py`: reviewed community detection baseline
- `link_prediction.py`, `downstream_knn.py`: reviewed 5-fold CV AUROC and k-NN GO term prediction
- `human_validation.py`: reviewed cross-species validation (15,882 nodes)
- Performance fix: sparse SVD for Node2Vec on large networks (memory reduction), `scipy.cdist` for distance analysis
- NaN guard added to several embedding scripts

## 2026-06-05

**Scope:** Extension modules (Steps 16-21) + v1.0.0/v1.1.0 preparation.

- `adaptive_interval.py` (Step 16): reviewed data-driven consensus interval computation
- `network_topology_analysis.py` (Step 17): reviewed cross-species topology comparison
- `rank_reversal_analysis.py` (Step 18): reviewed p/q sensitivity + geometric gap mechanism
- `go_propagation.py` (Step 19): reviewed True Path Rule DAG expansion (154 → 5,429 genes)
- `biological_interpretation.py` (Step 20): reviewed 4-level G-F scale + GO cluster case study
- `benchmark_runtime.py` (Step 21): reviewed step-wise profiling
- Found and fixed: GIN collapse correction, adaptive interval key mismatch, `benchmark_runtime.py` argv leak
- `config_loader.py`, `input_validator.py`: reviewed YAML config system and pre-flight validation
- v1.1.0: 51 pytest unit tests verified passing
- Found and regenerated 4 broken figures from stale plotting scripts

## 2026-06-06

**Scope:** Topological analysis modules (Steps 22-23).

- `topological_analysis.py` (Step 22): reviewed persistent homology computation via Ripser, Betti curves
- `topological_stats.py` (Step 23): reviewed topological feature extraction
- Found and corrected: human PPI topological validation methodology — initial approach used wrong persistence dimension

## 2026-06-08

**Scope:** v1.1.1 — mathematical proofs + extension module integration.

- Reviewed Propositions 1-2, Theorems 1-4 in `Supplementary_Materials.txt`
- `embed_hyperbolic.py`, `pathway_analysis.py`: reviewed extension module pipeline integration (Steps 24-26)
- Verified `--skip-topological` CLI flag behaviour

## 2026-06-09

**Scope:** v1.2.0 — cross-file consistency audit.

- Found and replaced `hash(method)` with `zlib.crc32(method.encode())` in `compute_gf.py` for cross-process reproducibility
- Found and replaced `hash(tuple(q_idx))` with `zlib.crc32()` in `robustness_analysis.py` for deterministic seeding
- Found and replaced 14 `list.index()` calls with dict lookups across 10 files for O(1) node alignment
- Found and corrected: `leiden_baseline.py` purity formula (`most_common / cluster_size` → `most_common / total_GO_terms`), baseline purity revised from 0.689 to 0.180
- Found and corrected: `robustness_analysis.py` function call signatures for `convergence_analysis()` and `randomization_null_test()`
- Found and corrected: `human_gf_all.py` hardcoded fallback node count (15882) replaced with `len(nodes)`
- Removed submission-specific documents from repository

## 2026-06-10

**Scope:** Purity formula unification + deprecation cleanup.

- Verified purity formula consistency across all pipeline steps after v1.2.0 fix
- Removed deprecated human validation scripts
- Added `requirements.lock.txt` for exact version pinning (88 dependencies)

## 2026-06-11

**Scope:** Steps 27-32 — metric comparison, bootstrap, semantic purity, cross-species.

- `metric_comparison.py` (Step 27): reviewed G-F Score vs link prediction AUC + k-NN F1 comparison
- `bootstrap_correlations.py` (Step 28): reviewed 95% CI for Spearman correlations (10k resamples)
- `semantic_purity.py` (Step 29): reviewed IC-weighted + Resnik semantic purity + DAG inflation diagnosis
- `cross_species_consistency.py` (Step 30): reviewed yeast vs human rank concordance (Spearman + Kendall W)
- `scale_gradient.py` (Step 31): reviewed scale-dependent topology coupling (500-4000 nodes)
- `bootstrap_stability.py` (Step 32): reviewed 30-resample CI for G-F Score rankings
- Found and corrected: stale step counts in `pipeline_config.yaml` and `__init__.py`
- Found and corrected: "50 resamples" → "30 resamples" in Step 32 documentation

## 2026-06-12

**Scope:** Steps 33-35 — human extended analysis modules.

- `human_embed_extended.py` (Step 33a): reviewed 11-method human network embeddings
- `human_gf_extended.py` (Step 33b): reviewed human GF analysis for all 11 methods
- `multimodal_functional_anchoring.py` (Step 34): reviewed STRING threshold gradient + channel-specific GF
- `hyperparameter_sensitivity.py` (Step 35): reviewed r-points, resolution, dimension, walk parameter sensitivity
- Verified purity formula standardisation across all v1.4.0 steps
- v6 audit: `biological_interpretation.py` purity formula confirmed correct

## 2026-06-13

**Scope:** Steps 36-39 — density correction, stability, GAT diagnosis + deep analysis modules.

- `density_corrected_gf.py` (Step 36): reviewed random baseline subtraction (ΔW = 0.522)
- `human_seed_stability.py` (Step 37): reviewed 10-seed subsampling stability (Kendall W = 0.675)
- `human_ic_weighted_gf.py` (Step 38): reviewed IC-weighted purity (ρ = 0.991)
- `gat_collapse_diagnosis.py` (Step 39): reviewed 5-variant ablation (clip, warmup, multi-head)
- Found and corrected: Step 37/38/39 numbering misalignment with CHANGELOG
- Deep analysis modules reviewed:
  - `human_spectral_alignment.py`: reviewed cross-network two-factor model transfer test
  - `gat_dimension_sweep.py`: reviewed GAT latent dimension sweep (d=2-32)
  - `gat_collapse_theory.py`: reviewed attention degeneration + rank analysis
  - `spectral_alignment.py`: reviewed Laplacian eigenbasis alignment
- Updated `__init__.py` pipeline description from 35-step to 39-step

## 2026-06-14

**Scope:** Phases 5-8C — theoretical proofs + cross-network validation.

- v8 audit: found and fixed hardcoded paths, dead code, unused imports, remaining `list.index()` calls
- `human_spectral_alignment.py` (Phase 5A): reviewed two-factor model non-transfer to human PPI
- `gat_dimension_sweep.py` (Phase 5B): reviewed dimension-independent attention degeneration
- `gat_collapse_formal_proof.py` (Phase 6): reviewed 3 theorems (T1 attention degeneration, T2 effective rank, T3 GF upper bound)
- `tda_geometry_bridge.py` (Phase 7): reviewed persistent homology + partial correlations (partial rho=0.845)
- `human_cross_network_validation.py` (Phase 8): reviewed bootstrap CIs (10k resamples)
- `human_tda_full.py` (Phase 8B): reviewed persistent homology for all 11 methods on human PPI
- `human_loo_sensitivity.py` (Phase 8C): reviewed Spectral as topological outlier (H1 80x lower)

## 2026-06-15

**Scope:** Phases 9-13 — human unification, mouse validation, transferability theory, function prediction.

- `human_gf_unified.py` (Phase 9): reviewed community-detection confound elimination (Louvain→greedy_modularity)
- `mouse_data_prep.py` (Phase 10A): reviewed STRING PPI + MGI GAF download, Ensembl_MGI alias mapping
- `mouse_embeddings_full.py` (Phase 10B): reviewed full-network mouse embeddings (11 methods, ~16K nodes)
- `persistence_image_analysis.py` (Phase 10C): reviewed persistence images + three-species Kendall's W
- `spectral_transferability.py` (Phase 11): reviewed SQI derivation (λ₂/λ₂_ER × PR × FA_max) + SBM validation
- `biological_validation.py` (Phase 12): reviewed GO BP enrichment + multi-seed panel (n=220)
- `function_prediction.py` (Phase 13): reviewed LOTO-CV (5,936 nodes, 12,690 trials)
- `longrange_functional_links.py`: reviewed distance-stratified functional recovery + hybrid predictor
- `dimension_sweep.py`: reviewed d=2-64 MRR gradient (213% improvement, no dimension exceeds PPI)
- `rescue_protein_analysis.py`: reviewed 235 rescue proteins characterisation
- `metric_comparison_extended.py`: reviewed GF Score vs link prediction + k-NN F1 with permutation test
- v2.1.0 released after all Phase 9-13 results verified

## 2026-06-16

**Scope:** Steps 40-48 — submission-tier experiments + breakthrough results.

- Code quality pass: fixed Windows GBK encoding crashes, hardcoded absolute paths, Unicode in `print()`, random seed propagation
- `multihead_gat_experiment.py`: reviewed multi-head GAT sweep (1/4/8 heads × d=2-32)
- `ecoli_analysis.py` (Step 40): reviewed E. coli K-12 as 4th species (SQI=0.7, MDS #1 GF=0.245)
- `coexpression_gf.py` (Step 41): reviewed network-type dependence (DeepWalk #1 on coexpression)
- `degree_preserving_null.py` (Step 42): reviewed 50 double-edge-swap randomizations
- `gat_theorem_large_network.py` (Step 43): reviewed full 5936-node theorem verification
- `gf_ablation_community_detection.py` (Step 44): reviewed 5 community detection algorithms
- `function_prediction_full.py` (Step 45): reviewed expanded 11-method LOTO-CV
- `gf_phase_transition.py` (Step 46): reviewed critical radii + Betti coincidence
- `dimension_sweep_extended.py` (Step 47): reviewed d=128/256 extension
- `functional_dark_matter.py` (Step 48): reviewed 74 dark matter pairs, GO enrichment (ERAD 521x)

## 2026-06-17

**Scope:** Steps 49-59 — cross-species, high-dim, UMAP, GO generality, GATv2.

- `cross_species_dark_matter.py` (Step 49): reviewed human/mouse ortholog mapping
- `generate_missing_figures.py` (Step 50): reviewed FigS18-S20 generation
- `string_v12_revalidation.py` (Step 51): confirmed all 44 dark matter pairs absent in STRING v12.0
- `highdim_spectral_embeddings.py` (Step 52): reviewed d=64 for human (15,882) + mouse (16,180)
- `cross_species_highdim.py` (Step 53): reviewed 2D vs 64D conservation (3/7 → 4/7 conserved)
- `dimension_gradient_3species.py` (Steps 54-55): reviewed critical dimensions per GO category
- `umap_tsne_gf.py` (Step 56): reviewed adjacency UMAP GF=0.177, shortest-path collapse (GF=0.068)
- `go_mf_cc_gf_scores.py` (Step 57): reviewed MF (GF=0.348) + CC (GF=0.191) in addition to BP
- `string_threshold_sensitivity.py` (Step 58): reviewed 600/700/800 threshold gradient + regime shift
- `gatv2_experiment.py` (Step 59): verified attention entropy (0.903 vs 0.927), GF Score near-random

## 2026-06-18

**Scope:** Steps 60-72 — 5th species, theoretical foundations, function prediction atlas + full pipeline run.

- `fly_analysis.py` (Step 60): verified Drosophila network stats (6,909 nodes, 89,685 edges), Spectral GF=0.619
- `prone_harp_gf.py` (Step 61): confirmed ProNE GF=0.087, HARP GF=0.114 both below random baseline (0.135)
- `function_prediction_cosine.py` (Step 62): cosine voting improves MRR for all 5 methods, GF-MRR rho=0.90
- `heat_kernel_multiscale.py` (Step 63): verified Spectral = t→0 limit (rho=1.0 across all time scales)
- `position_encoding_comparison.py` (Step 64): Laplacian PE GF=0.163 matches Spectral; sign-flip std=9.4e-5
- `cheeger_gf_bound.py` (Step 65): 4-component bound valid for all 6 networks; tightness values cross-checked
- `dimension_sweep_512.py` (Step 66): peak MRR at d=512 (+11.4% above PPI), log-linear R²=0.961
- `function_prediction_atlas.py` (Step 67): Spectral d=256 exceeds PPI in 9/9 ontology-species cases
- `uncharacterized_prediction.py` (Step 68): 511 predictions for 285 proteins verified
- `cross_species_atlas.py` (Step 69): found and fixed invalid f-string format bug in output logging
- `atlas_extension_512.py` (Step 70): confirms d=512 as universal optimum across all ontology-species combinations
- `ortholog_cross_validation.py` (Step 71): cross-species geometric conservation rho=0.618 verified
- `dark_matter_pan_species.py` (Step 72): found and fixed STRING v12.0 header crash (try/except for malformed lines)
- Rebuilt `cross_species_atlas.json` — Human BP results were all zeros (LOTO-CV bug)
- Full pipeline run: all 72 steps complete, all figures regenerated

## 2026-06-19

**Scope:** Final engineering audit — documentation consistency, tag/release alignment, module table completeness.

Audit 1 — Step count and method count:
- Found 5 remaining "65-step" references in README → all corrected to "72-step"
- Found "11 methods" in README header → corrected to "18 methods" (7 additional: UMAP, UMAP-adj, t-SNE, t-SNE-sp, ProNE, HARP, GATv2)
- `scripts/__init__.py` docstring: "65-step" → "72-step"
- `scripts/benchmark_runtime.py` docstring: "14-step" → "72-step"

Audit 2 — Project structure completeness:
- README directory tree missing Steps 63-72 → added all 10 scripts with descriptions
- Steps 60-62 ordering wrong in 3 of 4 README sections (Key Results, Step list, module table had prone=60, cosine=61, fly=62) → corrected to fly=60, prone=61, cosine=62 per `run_all_analysis.py` authoritative ordering
- `run_all_analysis.py` docstring: Steps 69-70 descriptions were swapped → corrected

Audit 3 — Orphan script review:
- `cross_species_atlas_optimized.py` (591 lines): zero references anywhere, no output artifacts → removed
- `gf_mrr_bridge.py`: regression analysis over `function_prediction_atlas.json` data, unique interaction model (R²=0.282) but fully reconstructable from existing JSON → removed (data preserved in `results/gf_mrr_bridge.json`)
- `human_atlas.py`: fully duplicated by `cross_species_atlas.py` (identical MRR values in output) → removed

Audit 4 — Module name / filename mismatches:
- `scripts/__init__.py`: `dimension_sweep_512_1024` → `dimension_sweep_512` (actual filename)
- `scripts/__init__.py`: `uncharacterized_protein_mining` → `uncharacterized_prediction` (actual filename)
- `scripts/__init__.py`: `atlas_extension` → `atlas_extension_512` (actual filename)

Audit 5 — Script count:
- "113 scripts" in BibTeX was pre-deletion count → corrected to 110 (verified: `find scripts/ -name "*.py" | wc -l` = 110)

Audit 6 — Tag/release alignment:
- Deleted orphan tags v1.1.1 and v1.2.0 (had tags but no releases)
- Final state: 3 tags = 3 releases (v1.0.0, v1.1.0, v2.11.0-paper-submission)

Audit 7 — Pipeline Overview:
- Step 50 was "Rescue Protein Analysis" but `run_all_analysis.py` defines Step 50 as `generate_missing_figures.py` → corrected
- `rescue_protein_analysis.py` is a supplementary module, not a pipeline step → removed step number from module table
- Module table missing `generate_missing_figures.py` (Step 50) → added
- Module table missing Steps 63-72 (10 entries) → added all

Audit 8 — Source code patterns:
- Verified all 100+ `json.dump` calls use protective patterns: explicit `float()`/`int()` casts, `default=str` fallback, or custom `_json_default` handler — no unprotected numpy types
- Verified zero Unicode characters in `print()` statements — all statistical symbols only in matplotlib labels
- Verified no stale imports referencing deleted files
- Verified all 48 `from scripts.` imports in `run_all_analysis.py` target existing modules
