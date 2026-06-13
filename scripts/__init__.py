"""
scripts/ — G-F Consistency Framework analysis pipeline.

This package contains the 35-step reproducible analysis pipeline for
evaluating protein interaction network embeddings via geometric-functional
consistency.

Modules (Steps 1-14: core pipeline)
------------------------------------
utils                         Shared utilities (data loading, G-F computation)
data_preprocessing            Step 1: Prepare data from raw STRING files
embed_all                     Step 2: Compute 8 classical/NN embedding methods
compute_gf                    Step 3: G-F curves (200-point grid) and scores
leiden_baseline               Step 4: Leiden community detection baseline
robustness                    Step 5: Subset robustness (30 x 5 sizes)
full_network                  Step 6: Full 5,936-node STRING validation
geometric_analysis            Step 7: d_intra / d_inter geometric margins
link_prediction               Step 8: 5-fold CV link prediction (AUROC)
downstream_knn                Step 9: k-NN GO term prediction (micro-F1)
randomization_control         Step 10: Shuffled-label control experiment
sampling_density              Step 11: 30 vs 200-point grid comparison
gf_score_sensitivity          Step 12: Integration interval sensitivity
plot_figures                  Step 13: Publication-quality figure generation
human_validation              Step 14 (optional): Cross-species human PPI

Modules (Steps 15-21: extended analyses)
-----------------------------------------
embed_gnn                     Step 15: GraphSAGE, GAT, GIN embeddings
adaptive_interval             Step 16: Data-driven consensus interval
network_topology_analysis     Step 17: Cross-species topology comparison
rank_reversal_analysis        Step 18: p/q sensitivity + geometric gap
go_propagation                Step 19: GO DAG True Path Rule expansion
biological_interpretation     Step 20: 4-level G-F scale + case study
benchmark_runtime             Step 21: Pipeline profiling + complexity

Modules (Steps 22-29: topological, pathway, statistical, and semantic analyses)
--------------------------------------------------------------------------------
topological_analysis          Step 22: Persistent homology (Betti curves, Ripser)
topological_stats             Step 23: Topological feature extraction + statistics
embed_hyperbolic              Step 24: Poincare Ball hyperbolic space embeddings
pathway_analysis              Step 25: Pathway enrichment and cancer gene association
statistical_analysis          Step 26: Spearman, Wilcoxon, bootstrap, permutation tests
metric_comparison             Step 27: G-F Score vs link prediction AUC + k-NN F1
bootstrap_correlations        Step 28: Bootstrap 95% CI for key Spearman correlations
semantic_purity               Step 29: IC-weighted + Resnik semantic purity (core library)
semantic_similarity_analysis  Step 29: Robustness check — 3-variant GF comparison + DAG diagnostics

Modules (Steps 30-32: cross-species, scale, and stability analyses)
--------------------------------------------------------------------
cross_species_consistency     Step 30: Yeast vs human rank concordance (Spearman, Kendall W)
scale_gradient                Step 31: Scale-dependent topology coupling (500-4000 nodes)
bootstrap_stability           Step 32: Bootstrap CI for G-F Score rankings (30 resamples)

Modules (Steps 33-35: human extension, multi-modal, and sensitivity analyses)
------------------------------------------------------------------------------
human_embed_extended          Step 33a: Extended human embeddings (PCA, VGAE-feat, GNNs)
human_gf_extended             Step 33b: Human GF analysis for all 11 methods
multimodal_functional_anchoring Step 34: STRING threshold gradient + channel networks
hyperparameter_sensitivity    Step 35: r-points, resolution, dimensions, walk parameters

Modules (Steps 36-39: density correction, seed stability, IC-weighted, GAT diagnosis)
--------------------------------------------------------------------------------------
density_corrected_gf          Step 36: Density-corrected G-F Score (random baseline normalization)
human_seed_stability          Step 37: 10-seed subsample stability test (human 11 methods)
human_ic_weighted_gf          Step 38: IC-weighted G-F Score on human PPI (11 methods)
gat_collapse_diagnosis        Step 39: GAT embedding collapse root-cause analysis

Extension modules (v1.1+)
--------------------------
config_loader                 YAML configuration loader and validator
input_validator               Pre-flight input validation and error handling
multispecies_loader           Multi-species dataset loader (yeast, human, ...)
temporal_network              Dynamic/temporal PPI network framework

Support modules
---------------
robustness_analysis           Extended 30-subset convergence analysis
visualization_helpers         Okabe-Ito colorblind-safe plotting utilities
"""

__version__ = "1.5.0"
