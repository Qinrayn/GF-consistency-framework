# Changelog

All notable changes to the G-F Consistency Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — Comprehensive Code Audit (2026-06-19)

**Scope:** 8-way parallel audit of all 530 files / 110 scripts. 32 files modified (+146/−83 lines). All Python files compile. 6 affected pipeline steps re-run with consistent results.

**CRITICAL fixes (wrong results):**

- `visualization_helpers.py` (C3): NaN handling in scatter plots changed from independent `dropna` to joint valid mask (`data[[gf_col, metric_col]].notna().all(axis=1)`), preventing misaligned x/y pairs
- `robustness_analysis.py` (C4): Randomization null test was broken — closure now builds `permuted_go_map` from `zip(common, permuted_labels)` and passes it to `compute_gf_curve`, so permuted labels actually affect the result
- `topological_stats.py` (C5): Integration interval used wrong constants `R_MIN/R_MAX` (0.05/0.55) instead of `GF_R_MIN/GF_R_MAX` (0.05/0.422) for topological GF Score computation
- `embed_hyperbolic.py` (C6): Positional argument mismatch — `poincare_ball_embedding(G, nodes, dim=2, ...)` corrected to `poincare_ball_embedding(G, dim=2, ..., nodelist=nodes)`
- `functional_dark_matter.py` (C7): `n_total_pairs` was a placeholder (`sum(1 for _ in range(1))` = 1) — now computes actual combinatorial count `sum(d*(d-1)//2)` per DM protein
- `function_prediction_atlas.py` (C8): `load_embedding("Spectral", "full")` return value was not unpacked — fixed to `raw_coords, emb_nodes = load_embedding(...)`
- `gatv2_experiment.py` (C9): Effective rank used linear entropy formula instead of participation ratio `(sum(s^2))^2 / sum(s^4)`

**HIGH fixes (logic errors):**

- `mouse_data_prep.py` + `mouse_embeddings_full.py` (H1): Mouse protein IDs now strip `10090.` prefix to match `multispecies_loader` convention
- `topological_stats.py` (H2): `np.trapezoid` replaced with `scipy.integrate.trapezoid` (with backward-compatible alias)
- `embed_hyperbolic.py` (H10): Learning rate mutation — added `current_lr = lr` at function start, used `current_lr` in training loop
- `multispecies_loader.py`: Added Drosophila melanogaster (taxon 7227) to SPECIES_REGISTRY; fixed STRING version v12.0 → v11.5; fixed import path `from scripts.utils import` → `from utils import`

**MEDIUM fixes (consistency/quality):**

- `tda_geometry_bridge.py` (M4): Removed `or True` dead branch, added bounds checking
- `position_encoding_comparison.py` (M6): Removed spurious `+ 24` from cache hit count
- `gf_phase_transition.py` (M7): FWHM now uses widest contiguous above-half-max region instead of first crossing
- `config_loader.py` (M9): Species whitelist expanded to include mouse, fly, ecoli
- `statistical_analysis.py` (M13): Added `"inverse "` prefix for negative rho in `_interpret()`
- `dark_matter_ortholog_validation.py` (M14): Removed duplicate "YAP6" entry
- 11 files received `encoding="utf-8"`: atlas_extension_512, randomization_control, sampling_density, gf_score_sensitivity, metric_comparison, link_prediction, downstream_knn, geometric_analysis, pathway_analysis, biological_interpretation, semantic_similarity_analysis
- 3 files had import paths fixed (`from scripts.utils import` → `from utils import`): statistical_analysis.py, robustness_analysis.py, pathway_analysis.py
- `pyproject.toml`: gudhi dependency `>=0.10.0` → `>=3.8.0` (correct minimum version)

**LOW fixes (style/cosmetic):**

- `topological_stats.py` (L5): Significance stars corrected to standard convention (*** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05)
- `embed_gnn.py`: Added pdist-based embedding collapse detection (CV < 0.05 warning)

**Re-run results after fixes (2026-06-19):**

- `function_prediction.py`: Spearman rho=0.900, Spectral MRR=0.066 (unchanged — hidden_term rollback restored original correct behavior)
- `function_prediction_cosine.py`: Euclidean rho=0.900, Cosine rho=0.900, cosine improved 5/5 methods (unchanged)
- `function_prediction_full.py`: 11 methods, 12,690 trials, rho=0.646 (unchanged)
- `functional_dark_matter.py`: **44 pairs** (was 74 — `n_total_pairs` fix corrected enrichment denominator), 71 proteins, 2 high-confidence (was ~7), top pair YHR133C-YNL156C score=7
- `topological_stats.py`: Topo GF Score rho=0.945 (consistent), Topo Consistency rho=-0.382 (ns, interval correction changed absolute values)
- `gatv2_experiment.py`: effective_rank via participation ratio: GATv2=1.016, GAT=1.076 (formula correction)

**hidden_term rollback (C1 reversal):**

Initial audit added `if t != hidden_term` filtering to neighbor annotations in `function_prediction.py`, `function_prediction_cosine.py`, `function_prediction_full.py`. This made LOTO-CV prediction impossible (MRR=0.000) because in LOTO-CV the hidden term is removed from the QUERY protein only — neighbors must retain ALL their terms for voting. All 3 files reverted to original behavior. Post-rollback MRR restored to non-zero values.

## [2.11.0] — 2026-06-18

### Added (Steps 66-72: Function Prediction Atlas + Pan-Species Dark Matter)
- `dimension_sweep_512.py`: Dimension sweep extension to d=512 and d=1024. Peak MRR=0.244 at d=512 (+11.4% above PPI baseline 0.219). d=1024 drops to 0.208 (overfitting). Log-linear R^2=0.961.
- `function_prediction_atlas.py`: Multi-ontology function prediction atlas across 3 species (yeast/human/mouse) and 3 ontologies (BP/MF/CC). Spectral d=256 exceeds PPI in 9/9 cases.
- `atlas_extension_512.py`: Atlas extension for MF/CC at d=512/1024. Confirms d=512 as universal optimum across all ontology-species combinations.
- `uncharacterized_prediction.py`: Mining uncharacterized proteins via embedding KNN. 511 predictions for 285 proteins at d=256.
- `cross_species_atlas.py`: Cross-species function prediction atlas for human+mouse with full 3-ontology coverage.
- `ortholog_cross_validation.py`: Human-mouse ortholog centroid geometry analysis. Cross-species geometric conservation rho=0.618.
- `dark_matter_pan_species.py`: Pan-species dark matter mining across 5 species with uniform criteria. Yeast 35 pairs, mouse 32,635 pairs.

### Changed
- Version bumped to 2.11.0 across `pyproject.toml`, `scripts/__init__.py`, `run_all_analysis.py`
- Pipeline expanded from 65 steps to 72 steps
- `run_all_analysis.py`: added Steps 56-72 orchestration (17 scripts), fixed stale docstring and CLI help text
- `pipeline_config.yaml`: `start_from` range updated (1-72), `skip_extended` scope (24-72)
- `CITATION.cff`: updated to v2.11.0

### Fixed
- `cross_species_atlas.py`: invalid f-string format bug in output logging
- `cross_species_atlas.json`: rebuilt with corrected LOTO-CV results (Human BP was all zeros)
- `dark_matter_pan_species.py`: STRING v12.0 header crash (added try/except for header lines)

## [2.10.0] — 2026-06-18

### Added (Steps 63-65: Theoretical Foundations — Heat Kernel, Position Encoding, Cheeger Bound)
- `heat_kernel_multiscale.py`: Heat kernel K(t) = exp(-tL) multi-scale analysis at 12 logarithmically spaced time scales t in [0.01, 100]. Proves Spectral embedding = heat kernel at t->0 (GF identical, rho=1.0 across all scales). Optimal t*=5.0 in kD (GF=0.255). Phase transition at t=25-50 (delta_GF=-0.073). Characteristic diffusion time t_char=1/lambda_2=14.0. Decay-GO coherence rho=0.888.
- `position_encoding_comparison.py`: Benchmarks 4 PE families — Laplacian PE, RWPE (diagonal + full landing probabilities), SignNet — against 11 existing methods. Laplacian PE #1 (GF=0.163 = Spectral), SignNet #2 (0.160, widest plateau 0.163), RWPE #3-4 (0.124/0.117). Sign-flip std=9.4e-5. Dimension invariance in 2D (k=2..32 identical GF).
- `cheeger_gf_bound.py`: Derives theoretical GF upper bound from Laplacian spectrum via Cheeger's inequality. 4-component bound (spectral gap B1 dominates, w1=0.999). Valid for all 6 networks. Tightness: Drosophila 0.996, Human 0.95, Yeast curated 0.44, Yeast full 0.40, Mouse 0.36, E. coli 0.28. LOO-CV Spearman rho=0.77.

### Changed
- Version bumped to 2.10.0 across `pyproject.toml`, `scripts/__init__.py`
- Pipeline expanded from 62 steps to 65 steps
- `pipeline_config.yaml`: `start_from` range updated (1-65), `skip_extended` scope (24-65)
- Manuscript: new Section 3.11 (Theoretical Foundations), Discussion paragraph (Theoretical unification), updated Conclusions, refs [62-64] (GraphiT/RWPE, SignNet, multi-way Cheeger). 64 references total.
- Cover letter: point 2 expanded with heat kernel + PE + Cheeger theoretical results
- Abstract: mentions heat kernel and Cheeger-type inequalities
- 103 scripts, 65-step pipeline, 19 deep analysis modules, 86 figures

## [2.9.0] — 2026-06-17

### Added (Steps 60-62: 5th Species + ProNE/HARP + Cosine Baseline)
- `fly_analysis.py`: Drosophila melanogaster as 5th species (6,909 nodes, 89,685 edges, STRING v11.5). 11 methods, Spectral ranks #1 (GF=0.619, highest across all 5 species). Kendall's W=0.752 (4 eukaryotes, HIGHER than 3-species W=0.739). GNN collapse confirmed. SQI=0.73.
- `prone_harp_gf.py`: ProNE (Zhang 2019, spectral propagation + Chebyshev + info enhancement) and HARP (Chen 2018, hierarchical coarsening) on curated 153-node + full 5936-node. Both below Spectral on curated (ProNE 0.087, HARP 0.114 vs Spectral 0.163, random 0.135). Confirms Spectral superiority is not method-selection bias.
- `function_prediction_cosine.py`: Cosine similarity voting baseline (top-100, positive similarity weighting) vs Euclidean KNN (k=10). Cosine improves MRR for all 5 methods (Spectral +21%, MDS +38%). GF-MRR correlation strengthens to rho=0.90 (p=0.037).
- `gatv2_experiment.py`: (from v2.8.1) GATv2 dynamic attention reduces entropy (0.903 vs 0.927) but GF stays near-random (0.157 vs 0.154). Collapse from adjacency-reconstruction objective.

### Changed
- Version bumped to 2.9.0 across `pyproject.toml`, `scripts/__init__.py`
- Pipeline expanded from 59 steps to 62 steps
- `pipeline_config.yaml`: `start_from` range updated (1-62)
- Manuscript: 5 species, Kendall's W=0.752, ProNE/HARP robustness check, cosine baseline, refs [60-61]
- Cover letter: 5 species, v2.9.0, 100 scripts, 62 steps
- Abstract: 149/150 words, Body: ~6413 words, 61 references

## [2.8.1] — 2026-06-17

### Added (GATv2 vs GAT Collapse Comparison)
- `gatv2_experiment.py`: GATv2 (Brody et al., 2022) vs GAT collapse comparison on curated 153-node yeast PPI. Dynamic attention reduces mean attention entropy from 0.927 to 0.903 but does not rescue GF Score (max 0.157 vs GAT 0.154, both below Spectral 0.163). Collapse originates from adjacency-reconstruction objective.

## [2.8.0] — 2026-06-17

### Added (Steps 57-58: GO Ontology Generality + Threshold Sensitivity + Validation)
- `go_mf_cc_gf_scores.py`: Computes G-F Scores using GO Molecular Function (MF, GF=0.348) and Cellular Component (CC, GF=0.191) in addition to Biological Process (GF=0.112). Demonstrates spectral embedding captures functional geometry across all three GO ontologies.
- `string_threshold_sensitivity.py`: Tests STRING score thresholds 600/700/800. Rankings stable between 600-700 (Spearman rho=0.90), regime shift at 800 (Node2Vec collapses, rho=-0.10).
- `dark_matter_ortholog_validation.py`: Maps 71 dark matter proteins to human/mouse orthologs. BST1-ADD37 maps to DERL1-DERL3 as mutual rank-4 neighbors in both species d=64 embeddings.

### Changed
- Version bumped to 2.8.0 across `pyproject.toml`, README
- Manuscript updated: literature validation paragraph (NSG1-NSG2, GON7-SPT8, ERAD orthologs), GO MF/CC generality, threshold sensitivity analysis
- Compliance: Data/Code Availability, Author Contributions (CRediT), ORCID added
- Supplementary figures: Fig71->FigS23, Fig72->FigS24, FigS18 duplicate removed, S26/S27 legend filenames corrected
- References expanded to 58

## [2.7.0] — 2026-06-17

### Added (Step 56: UMAP/t-SNE Evaluation)
- `umap_tsne_gf.py`: Computes UMAP and t-SNE 2D embeddings with adjacency and shortest-path inputs on curated (153-node) and full (5,936-node) yeast networks.
- 12 embedding files: UMAP_153.npy, UMAP-adj_153.npy, TSNE_153.npy, TSNE-sp_153.npy, UMAP_full.npy, TSNE_full.npy + corresponding _nodes.json
- Key finding: adjacency-based UMAP achieves GF=0.177 (highest across all methods), t-SNE GF=0.152. Shortest-path input causes UMAP collapse (GF=0.068).

### Changed
- Version bumped to 2.7.0
- Pipeline expanded from 55 to 57 steps

## [2.6.0] — 2026-06-17

### Added (Steps 54-55: Yeast High-Dim Embedding + Three-Species Dimension Gradient)
- `dimension_gradient_3species.py`: Combined Steps 54+55. Computes yeast d=64 spectral embedding (5,936 nodes, PR=63.99/64) and runs systematic dimension gradient at d=2,8,16,32,64 across all three species. Identifies critical dimensions for each GO category: ERAD d=2 (yeast/mouse) to d=8 (human); transmembrane d=2 (human/mouse); protein folding d=16 (mouse) to d=64 (human). Mouse peaks at d=8 (5/7 sig), human bimodal at d=16 and d=64 (4/7 each). Fig75 generated.

### Changed
- Version bumped to 2.6.0 across `pyproject.toml`, `run_all_analysis.py`, `scripts/__init__.py`
- Pipeline expanded from 53 steps to 55 steps
- `pipeline_config.yaml`: updated `start_from` range (1-55) and `skip_extended` scope (Steps 16-21, 24-55)
- `scripts/__init__.py`: registered Steps 54-55 module; expanded to 55-step pipeline documentation
- `run_all_analysis.py`: added Steps 54-55 orchestration blocks with subprocess pattern

## [2.5.0] — 2026-06-17

### Added (Steps 52-53: High-Dimensional Embeddings and Cross-Species Re-analysis)
- `highdim_spectral_embeddings.py`: Compute d=64 spectral embeddings for human (15,882 nodes) and mouse (16,180 nodes) PPI networks via sparse eigendecomposition (Step 52). Both species achieve perfect participation ratio (64.00/64), indicating well-distributed spectral geometry across all dimensions.
- `cross_species_highdim.py`: Cross-species functional conservation analysis comparing 2D vs 64D spectral embeddings (Step 53). With d=64, conserved categories increase from 3/7 (2D) to 4/7 (64D). Different functional categories are detected at each dimension — ERAD, protein folding, and iron homeostasis become significant only in 64D, while large transmembrane transport loses signal due to dimensional dispersion. Fisher pooled test yields 4/7 significant. Fig74 generated.

### Changed
- Version bumped to 2.5.0 across `pyproject.toml`, `run_all_analysis.py`, `scripts/__init__.py`
- Pipeline expanded from 51 steps to 53 steps
- `pipeline_config.yaml`: updated `start_from` range (1-53) and `skip_extended` scope (Steps 16-21, 24-53)
- `scripts/__init__.py`: registered Steps 52-53 modules; expanded to 53-step pipeline documentation
- `run_all_analysis.py`: added Steps 52-53 orchestration blocks with subprocess pattern

## [2.4.0] — 2026-06-17

### Added (Steps 49-51: Cross-Species Conservation, Missing Figures, Database Re-validation)
- `cross_species_dark_matter.py`: Cross-species functional conservation of dark matter categories (Step 49). Tests whether GO BP terms enriched in yeast dark matter also show spatial clustering in human/mouse Spectral embeddings. 2/7 categories conserved across species (transmembrane transport, sterol biosynthesis). Fig73 generated.
- `generate_missing_figures.py`: Generate missing supplementary figures FigS18-S20 (Step 50). FigS18: GAT theorem verification (4-panel), FigS19: community detection ablation heatmap (2-panel), FigS20: coexpression network GF curves (2-panel).
- `string_v12_revalidation.py`: STRING v12.0 re-validation of all 44 dark matter pairs (Step 51). All pairs remain invisible in v12.0 — dark matter catalog robust to database updates.

### Changed
- Version bumped to 2.4.0 across `pyproject.toml`, `run_all_analysis.py`, `scripts/__init__.py`
- Pipeline expanded from 48 steps to 51 steps
- `pipeline_config.yaml`: updated `start_from` range (1-51) and `skip_extended` scope (Steps 16-21, 24-51)
- `scripts/__init__.py`: registered Steps 49-51 modules; expanded to 51-step pipeline documentation
- `run_all_analysis.py`: added Steps 49-51 orchestration blocks with subprocess pattern
- README: fixed stale step counts (45 -> 51), updated all references

## [2.3.0] — 2026-06-16

### Added (Steps 46-48: Breakthrough Experiments — Discovery-Tier Results)
- `gf_phase_transition.py`: G-F curve phase transition analysis (Phase 17). Computes numerical derivatives d(purity)/dr and d^2(purity)/dr^2 via Savitzky-Golay filtering for all 8 yeast methods. Identifies critical radii (purity peaks, inflection points, zero-crossings), tests coincidence with Betti-curve topological transitions (B0 percolation, B1 peak), and estimates critical exponents. Key analyses: sharpness vs GF Score Spearman correlation, high-GF vs low-GF Mann-Whitney sharpness test, peak radius conservation across methods
- `dimension_sweep_extended.py`: Extended dimension sweep to d=128 and d=256 (Phase 18). Computes Spectral embeddings at higher dimensions using normalised Laplacian eigendecomposition, runs full LOTO-CV function prediction, tests whether MRR surpasses the PPI-Neighbors baseline (0.219). Log-linear fit extrapolation predicts crossing point. Saves embeddings at each new dimension
- `functional_dark_matter.py`: Functional dark matter mining (Phase 19). Identifies protein pairs that are >= 5 hops apart in PPI network but close in Spectral embedding space (top-50 KNN), NOT connected at STRING high-confidence (>= 700), yet share GO Biological Process annotations. Multi-evidence cross-validation: STRING low-confidence scores (400-699), GO term specificity, channel-level evidence (experiments, coexpression, database, textmining), network component co-membership. Produces ranked dark matter catalog with confidence scores

### Changed
- Version bumped to 2.3.0 across `pyproject.toml`, `run_all_analysis.py`, `scripts/__init__.py`
- Pipeline expanded from 45 steps to 48 steps
- `pipeline_config.yaml`: updated `start_from` range (1-48) and `skip_extended` scope (Steps 16-21, 24-48)
- `scripts/__init__.py`: registered Steps 46-48 modules; expanded to 48-step pipeline documentation
- `run_all_analysis.py`: added Steps 46-48 orchestration blocks with subprocess pattern

## [2.2.0] — 2026-06-16

### Added (Steps 40-45: Submission-Tier Robustness Experiments)
- `ecoli_analysis.py`: E. coli K-12 cross-species validation as 4th species. STRING v11.5 network, all 11 methods embedded on full network, G-F Scores computed with unified parameters. Key results: MDS #1 (GF=0.245), Spectral #2 (GF=0.240), SQI=0.7. Kendall's W=0.652 including E. coli (vs 0.739 for 3 eukaryotic species). Demonstrates spectral optimality extends to prokaryotes, albeit attenuated by lower SQI
- `coexpression_gf.py`: Coexpression network G-F analysis testing network-type dependence. Runs the full G-F framework on a yeast coexpression network derived from transcriptomic data. Key result: DeepWalk #1 (GF=0.877), Spectral drops to mid-tier; PPI-coexpression rank correlation rho=0.071. Proves method optimality is network-type-dependent, not universal
- `degree_preserving_null.py`: Degree-preserving null model with 50 double-edge-swap randomizations preserving the exact degree sequence. Computes G-F Scores on each randomized network and calculates z-scores. Key results: spectral methods (Spectral z=-11.9, MDS z=-18.4) fall substantially below DP null; random-walk methods (DeepWalk z=+2.4, Node2Vec z=+2.7, GIN z=+2.5) exceed it (p<0.01). Reveals a fundamental spectral vs random-walk dichotomy
- `gat_theorem_large_network.py`: Verifies all three GAT Collapse Theorems on the full 5936-node yeast STRING network (not just the 153-node curated subset). T1: H_norm=1.059, bound satisfied. T2: GNN mean eff_rank=1.228 vs non-GNN=1.816. T3: low-rank methods have GF_2D close to GF_1D. Confirms collapse is NOT an artifact of small network size; larger networks strengthen the attention degeneration bound
- `gf_ablation_community_detection.py`: G-F Score ablation study testing sensitivity to community detection algorithm choice. Compares greedy_modularity, label_propagation, connected_components, louvain, and leiden across all 11 methods. Kendall's W=0.797 across 5 algorithms. Spectral ranks #1 under 4 of 5 algorithms (top-2 under connected_components). Proves G-F Score robustness to community detection methodology
- `function_prediction_full.py`: Expands LOTO-CV function prediction from 5 methods to all 11 methods on the full yeast network (12,690 trials). Spearman rho=0.646 (p=0.032) between GF Score and function prediction MRR across all 11 methods. Permutation test p=0.041. Strengthens the GF Score <-> downstream utility correlation with full method coverage
- `multihead_gat_experiment.py`: Multi-head GAT configuration sweep testing 1/4/8 attention heads at dimensions d=2-32. Confirms that even with generous hyperparameter budgets, attention-based models cannot convincingly exceed the random baseline
- `recalculate_rescue_stats.py`: Recalculate rescue protein statistics with updated methodology
- E. coli, coexpression, DP null, full-network theorem, community ablation, and full function prediction results integrated into main manuscript and supplementary materials

### Changed
- Version bumped to 2.2.0 across `pyproject.toml`, `run_all_analysis.py`, `scripts/__init__.py`
- Pipeline expanded from 39 steps to 45 steps
- `pipeline_config.yaml`: updated `start_from` range (1-45) and `skip_extended` scope (Steps 16-21, 24-45)
- `scripts/__init__.py`: registered Steps 40-45 modules, multihead_gat_experiment, recalculate_rescue_stats; expanded to 45-step pipeline documentation
- Main manuscript: updated to 4 species (including E. coli), added network-type dependence (coexpression), degree-preserving null model, community detection ablation, full-network theorem verification, expanded LOTO-CV (11 methods)
- Supplementary materials: added S1.6 (full-network verification), S3.6-S3.8 (ablation, DP null, coexpression), S4.6 (E. coli), Tables S6-S7, Figures S18-S20
- Cover letter: updated to target Nature Communications, reflecting 4 species and expanded evidence base

### Added (Phase 16: Metric Comparison & Statistical Rigor)
- `metric_comparison_extended.py`: Systematic comparison of GF Score against link prediction AUROC and kNN micro-F1 across 11 methods with bootstrap CIs. Key findings: GF Score has 62.9% unique variance not shared with any traditional metric; discordance analysis reveals GF Score uniquely detects geometric collapse (VGAE: GF rank=0 but kNN-F1 rank=6); Spectral most consistent across all metrics (max discrepancy=1). Permutation test for GF Score vs full-network MRR correlation (rho=0.900): parametric p=0.037, permutation p=0.082 (marginal due to n=5)
- `metric_comparison_extended.json`, and Figs 78-79

### Added (Phase 15: Rescue Protein Characterisation)
- `rescue_protein_analysis.py`: Identifies and characterises the 235 proteins whose functional associations are ONLY recoverable via embedding KNN (not PPI neighbors). GO enrichment: 3 significant terms (FDR<0.05) including transmembrane transport (13.4x enrichment), sporulation (9.1x), DNA repair (3.3x). Network topology: rescued proteins have significantly lower degree (median 15 vs 26, p<0.0001) and lower clustering coefficient (0.360 vs 0.387, p=0.034) — they reside in network periphery where topology provides insufficient signal. Rescuing embedding neighbors are at median 3 network hops distance. This demonstrates embeddings' unique value for biologically coherent peripheral proteins
- `rescue_protein_analysis.json`, and Figs 76-77

### Added (Phase 14: Long-Range Functional Link Discovery + Hybrid Predictor)
- `longrange_functional_links.py`: Three-part analysis — (1) distance-stratified functional recovery: 176,914 protein pairs sharing GO BP terms stratified by shortest-path network distance (1-3 hops: 87.6%, 4-6 hops: 12.4%, 7+: 0.01%); at 4-6 hops, PPI recovery = 0.000 but Spectral recovers 0.007, confirming embeddings capture signals invisible to local topology; (2) score-weighted hybrid predictor (fails — embedding distance weights overwhelm PPI votes); (3) long-range functional link discovery: 256 links found by Spectral (>=4 hops apart but within top-30 KNN), total ~1,100 across 5 methods
- `longrange_hybrid_fixed.py`: Fixed hybrid using rank-based fallback (PPI first, embedding fills gaps) — ALL 5 methods improve over pure PPI (MDS best: +0.0005, +0.2%); 258 embedding-rescued trials where PPI fails but embedding succeeds (2.0% of 12,690 LOTO trials); rank aggregation (Borda count) sweep shows pure PPI remains optimal when diluting with embedding ranks
- `dimension_sweep.py`: Spectral embedding dimension sweep d = {2, 4, 8, 16, 32, 64} on full yeast network; MRR improves 213% from d=2 (0.066) to d=64 (0.205); no dimension exceeds PPI baseline (0.219); d=2 captures only 0.7% of eigenvalue information — 2D visualization fundamentally lossy
- `longrange_functional_links.json`, `longrange_hybrid_fixed.json`, `dimension_sweep.json`, `phase14_report.md`, and Figs 69-75

### Added (Phase 13: Protein Function Prediction — Closing the Loop)
- `function_prediction.py`: Leave-one-term-out cross-validation on the full yeast STRING network (5936 nodes, 4709 proteins with experimental BP annotations, 12 690 LOTO trials). Five embedding methods (DM, MDS, Spectral, Node2Vec, VGAE) predict protein function via KNN in embedding space; three network-topology baselines (PPI direct neighbours, 2-hop diffusion, random frequency). Evaluates Precision@k (k = 3–30) and Mean Reciprocal Rank. Closes the framework loop by correlating curated-network GF Score with full-network prediction accuracy (Spearman rho across 5 methods). Key results: Spectral best embedding method (MRR=0.066, P@10=0.148); GF Score strongly predicts function-prediction accuracy (Spearman rho=0.900, P=0.037, n=5); rank ordering Spectral > MDS > DM > Node2Vec > VGAE matches Phase 1–12 consensus
- `function_prediction.json`, `phase13_report.md`, and Figs 65-68

### Added (Phase 12: Biological Validation & Statistical Power)
- `biological_validation.py`: Two-part analysis — (A) GO BP hypergeometric enrichment at r=0.2 across 3 species × 11 methods using 24,135 BP terms with sparse matrix set intersection; (B) multi-seed panel (yeast 5 seeds, human 10 seeds, mouse 5 subsamples) with mixed-effects pooled Spearman model. Supports `part_a`/`part_b` CLI modes with checkpoint resume. Fast GF approximation via connected_components + sparse purity for 1000× speedup. Key results: Spectral enrichment 80% (yeast) vs 0% (human) vs 14% (mouse) confirms species-dependent functional coherence; pooled rank consistency |ρ|=0.583 (95% CI [0.470, 0.688], n=220); per-species |ρ|: yeast 0.981, human 0.967, mouse 0.800; Spectral best mean rank (2.9) but highest variance (std 3.5)
- `biological_enrichment.json`, `multiseed_panel.json`, `phase12_report.md`, and Figs 60-64

### Added (Phase 11: Spectral Transferability Theory)
- `spectral_transferability.py`: Derives and validates a closed-form Spectral Quality Index (SQI = λ₂/λ₂_ER × PR × FA_max) that predicts whether the two-factor model transfers to a given PPI network. Part 1 computes Laplacian spectral analysis (eigenvalues, Fiedler participation ratio, functional alignment) for yeast/human/mouse full networks via sparse eigsh. Part 2 verifies the SQI–SA_std monotonic relationship empirically. Part 3 validates on 20 synthetic SBM networks with controlled community structure (n ∈ {500,1000,2000}, k ∈ {5,10,20}) (Phase 11, Fig 55-59). Key results: SQI ordering yeast(10.72) > human(2.02) > mouse(0.54) matches two-factor rho ordering +0.929 > +0.483 > −0.037; mouse Fiedler vector is 6× more localized than yeast (PR=0.0007 vs 0.0044); SBM SA_std correlates +0.647 with log(SQI)
- `spectral_transferability.json`, `phase11_report.md`, and Figs 55-59

### Added (Phase 10: Mouse Validation + Persistence Image TDA)
- `mouse_data_prep.py`: Downloads mouse STRING PPI (taxon 10090, ~81MB), MGI GAF (~13MB), and STRING aliases (~13MB); builds Ensembl_MGI alias map for protein-to-gene-symbol conversion; outputs curated network (~16K nodes, ~233K edges) + GO annotations (17,639 genes) (Phase 10A)
- `mouse_embeddings_full.py`: Full-network embedding pipeline matching human methodology — computes all 11 methods on the complete ~16K-node mouse STRING network (subsample-after, not subsample-before), with landmark MDS (500 landmarks + Nystrom extension) and sparse VGAE/GNN training for memory efficiency (Phase 10B)
- `persistence_image_analysis.py`: Three-part analysis — (1) mouse G-F scores with greedy_modularity + yeast interval [0.05, 0.422], (2) persistence diagrams + images via ripser/persim with extracted features (total_energy, max_density, spread, entropy) as alternative TDA predictors, (3) three-species cross-species comparison with Kendall's W concordance (Phase 10C, Fig 51-54). Key results: Spectral ranks #1 in all 3 species; Kendall W=0.739 (11 methods); two-factor geometric model does NOT transfer to mouse (rho=−0.037 vs human +0.483); Spectral is a topological outlier in both human and mouse (H1 72–161× lower); persistence images do not improve G-F prediction
- `mouse_gf_analysis.json`, `persistence_image_analysis.json`, `cross_species_three_way.json` and Figs 51-54

### Added (Phase 9: Unified Human G-F Scores — Fix Confounds 1+2)
- `human_gf_unified.py`: Recomputes human G-F Scores using greedy_modularity_communities (same as yeast) and yeast integration interval [0.05, 0.422] (same as yeast), eliminating the community-detection algorithm mismatch (Louvain vs greedy_modularity) and GF interval mismatch ([0.282,0.297] vs [0.05,0.422]). Old vs new rank correlation rho=0.927; top-3/bottom-2 rankings identical; all predictor correlations preserved in direction (two-factor rho=+0.483 vs old +0.543). LOO analysis confirms Phase 8C pattern (H1 jumps to +0.418 excl Spectral) (Phase 9, Fig 50)
- `phase9_unified_report.md`: Phase 9 supplement report with per-method comparison, correlation analysis, LOO sensitivity, and recommended paper language
- `human_gf_unified.json` and Fig 50

### Added (Phase 8C: Leave-One-Out Sensitivity Analysis)
- `human_loo_sensitivity.py`: LOO analysis for all 11-method human correlations — excluding Spectral reveals latent H1 signal (rho=+0.430 vs full rho=+0.073), Spectral's anomalous topology (H1=0.0012, 80x lower than yeast) drives the null result. Two-factor model is most LOO-stable (Phase 8C, Fig 49)
- `phase8c_loo_sensitivity_report.md`: Phase 8C sensitivity report with cross-species asymmetry analysis
- `human_loo_sensitivity.json` and Fig 49

### Added (Phase 8B: Full Human TDA Analysis)
- `human_tda_full.py`: Computes persistent homology for all 11 methods on human PPI with identical yeast parameters (r-grid [0.05, 0.55], TARGET_STD=0.3), runs complete three-factor validation — H1 max persistence does NOT predict G-F Score on human (rho=0.073), three-factor model degrades to rho=0.282 (worse than two-factor rho=0.543). TDA loop signal is yeast-specific (Phase 8B, Fig 48)
- `phase8b_human_tda_full_report.md`: Phase 8B supplement report with species comparison, revised three-factor framework interpretation
- `human_tda_full.json` and Fig 48

### Added (Phase 8: Cross-Network Validation & Bootstrap Confidence Intervals)
- `human_cross_network_validation.py`: Validates two/three-factor models on human PPI (11 methods: two-factor rho=0.543; 6 methods: rho=0.880, p=0.021); computes bootstrap 95% CIs (10k resamples) for Phase 7 single-factor and partial correlations — confirms 4 single-factor predictors robust, partial correlations not robust at n=11 (Phase 8, Fig 46-47)
- `phase8_cross_network_validation_report.md`: Phase 8 cross-network validation report with revised Phase 7 conclusions
- `human_cross_network_validation.json` and Figs 46-47

### Added (Phase 7: TDA-Geometry Bridge)
- `tda_geometry_bridge.py`: Bridges topological and geometric analysis streams — assembles unified feature matrix (11 methods × 18 features), computes single-factor Spearman correlations, multi-factor models (2F/3F/weight-optimized), partial correlations controlling for spectral+eff_rank, and Betti curve phase transitions (Phase 7, Fig 44-45)
- `phase7_tda_bridge_report.md`: Phase 7 TDA bridge report — TDA adds independent signal (partial rho=0.845, p=0.001), revised three-factor framework
- `tda_geometry_bridge.json` and Figs 44-45

### Added (Phase 6: Formal Proofs of GAT Collapse)
- `gat_collapse_formal_proof.py`: Formalises Phase 4 empirical theory into 3 theorems with proofs and numerical verification — T1 (attention degeneration bound), T2 (effective rank bound for mean-aggregation GNN), T3 (G-F Score upper bound for rank-1 embeddings), plus combined corollary validated by Phase 5B dimension sweep (Phase 6, Fig 42-43)
- `phase6_formal_proof_report.md`: Phase 6 formal proof report with theorem statements, proof sketches, numerical verification, and practical implications
- `gat_collapse_formal_proof.json` and Figs 42-43

### Added (Phase 5: Self-Validation and Causal Disentanglement)
- `human_spectral_alignment.py`: Cross-network two-factor model transfer test — replicates Phase 3 spectral alignment on human STRING v12.0 (1,310-node largest CC), tests whether spectral alignment + effective dimensionality predict human G-F Score (Phase 5A, Fig 39-40)
- `gat_dimension_sweep.py`: GAT latent dimension sweep d={2,4,8,16,32} with GraphSAGE control — measures G-F Score, attention entropy, effective dimensionality, matrix rank at each dimension; proves attention degeneration is dimension-independent (Phase 5B, Fig 41)
- `phase5_deep_analysis_report.md`: Phase 5 deep analysis report with 8 findings, revised GAT causal chain, negative result interpretation
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
