"""
scripts/ — G-F Consistency Framework analysis pipeline.

This package contains the 21-step reproducible analysis pipeline for
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

Extension modules (v1.1+)
--------------------------
config_loader                 YAML configuration loader and validator
input_validator               Pre-flight input validation and error handling
embed_hyperbolic              Poincare Ball hyperbolic space embeddings
multispecies_loader           Multi-species dataset loader (yeast, human, ...)
temporal_network              Dynamic/temporal PPI network framework
pathway_analysis              Pathway enrichment and cancer gene association

Support modules
---------------
statistical_analysis          Spearman, Wilcoxon, bootstrap, permutation tests
robustness_analysis           Extended 30-subset convergence analysis
visualization_helpers         Okabe-Ito colorblind-safe plotting utilities
"""

__version__ = "1.1.0"
