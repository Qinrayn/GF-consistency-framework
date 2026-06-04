"""
scripts/ — G-F Consistency Framework analysis pipeline.

This package contains the 14-step reproducible analysis pipeline for
evaluating protein interaction network embeddings via geometric-functional
consistency.

Modules
-------
utils                   Shared utilities (data loading, G-F curve computation)
data_preprocessing      Step 1: Prepare data from raw STRING files
embed_all               Step 2: Compute 8 embedding methods
compute_gf              Step 3: G-F curves (200-point grid) and scores
leiden_baseline         Step 4: Leiden community detection baseline
robustness              Step 5: Subset robustness + Bonferroni correction
full_network            Step 6: Full 5,936-node STRING validation
geometric_analysis      Step 7: d_intra / d_inter geometric margins
link_prediction         Step 8: 5-fold CV link prediction (AUROC)
downstream_knn          Step 9: k-NN GO term prediction (micro-F1)
randomization_control   Step 10: Shuffled-label control experiment
sampling_density        Step 11: 30 vs 200-point grid comparison
gf_score_sensitivity    Step 12: Integration interval sensitivity
plot_figures            Step 13: Publication-quality figure generation
human_validation        Step 14 (optional): Cross-species human PPI validation
"""
