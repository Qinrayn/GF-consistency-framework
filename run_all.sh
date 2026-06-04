#!/bin/bash
# G-F Consistency Framework - Complete Analysis Pipeline
# Run this script to reproduce all experiments from scratch.
# Usage: bash run_all.sh [--run-human]
#
# NOTE: This script covers Steps 1–14 (classical/NN embeddings, G-F curves,
# robustness, geometric analysis, link prediction, k-NN, figures).
# Steps 15–21 (GNN embeddings, adaptive interval, topology, rank reversal,
# GO propagation, biological interpretation, benchmarking) are handled by
# run_all_analysis.py. Use `python run_all_analysis.py` for the full 21-step pipeline.

set -e

echo "=========================================="
echo "  G-F Consistency Framework"
echo "  Complete Analysis Pipeline"
echo "=========================================="

RUN_HUMAN=false
if [ "$1" == "--run-human" ]; then
    RUN_HUMAN=true
fi

echo ""
echo "Step 1: Data Preprocessing"
echo "---------------------------"
python scripts/data_preprocessing.py

echo ""
echo "Step 2: Compute All Embeddings (DM, MDS, Spectral, DeepWalk, Node2Vec, VGAE, VGAE-feat, PCA)"
echo "---------------------------------------------------------------------------------------------"
python scripts/embed_all.py

echo ""
echo "Step 3: Compute G-F Curves (200-point grid) and G-F Scores"
echo "----------------------------------------------------------"
python scripts/compute_gf.py

echo ""
echo "Step 4: Leiden Baseline on Original Network"
echo "--------------------------------------------"
python scripts/leiden_baseline.py

echo ""
echo "Step 5: Robustness Analysis (30 subsets × 5 sizes + Bonferroni)"
echo "-----------------------------------------------------------------"
python scripts/robustness.py

echo ""
echo "Step 6: Full Network Validation (5,936 nodes)"
echo "----------------------------------------------"
python scripts/full_network.py

echo ""
echo "Step 7: Geometric Analysis (d_intra / d_inter)"
echo "-----------------------------------------------"
python scripts/geometric_analysis.py

echo ""
echo "Step 8: Link Prediction (5-fold CV)"
echo "------------------------------------"
python scripts/link_prediction.py

echo ""
echo "Step 9: Downstream k-NN Evaluation"
echo "------------------------------------"
python scripts/downstream_knn.py

echo ""
echo "Step 10: Randomization Control"
echo "------------------------------"
python scripts/randomization_control.py

echo ""
echo "Step 11: Sampling Density Verification (30 vs 200 pts)"
echo "------------------------------------------------------"
python scripts/sampling_density.py

echo ""
echo "Step 12: G-F Score Sensitivity Analysis"
echo "----------------------------------------"
python scripts/gf_score_sensitivity.py

if [ "$RUN_HUMAN" = true ]; then
    echo ""
    echo "Step 13: Human Cross-Species Validation (14,679 nodes)"
    echo "------------------------------------------------------"
    echo "  WARNING: This step is resource-intensive!"
    python human_validation/run_human_validation.py
else
    echo ""
    echo "Step 13: Human Cross-Species Validation - SKIPPED"
    echo "  Use --run-human to include, or run separately:"
    echo "  python human_validation/run_human_validation.py"
fi

echo ""
echo "Step 14: Generate All Figures"
echo "------------------------------"
python scripts/plot_figures.py

echo ""
echo "Final: Generate Results Summary"
echo "----------------------------------------"
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from run_all_analysis import generate_final_summary
generate_final_summary(Path('results'))
"

echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved in: results/"
echo "Figures saved in: figures/"
echo "Embeddings saved in: embeddings/"
echo "Data saved in: data/"
echo ""
echo "Final summary: results/final_results_summary.json"
echo ""
if [ "$RUN_HUMAN" = true ]; then
    HUMAN_STATUS="INCLUDED"
else
    HUMAN_STATUS="SKIPPED"
fi
echo "NOTE: Human cross-species validation was ${HUMAN_STATUS}."
echo "      See human_validation/README.md for details."
echo "=========================================="