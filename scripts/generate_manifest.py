#!/usr/bin/env python3
"""
Generate results/MANIFEST.md — traceability metadata for all result files.

P1-5: Result file traceability enhancement.
Scans results/ directory, matches files to pipeline steps, and generates
a manifest with file name, size, source script, step number, and description.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Mapping: result file pattern -> (step number, source script, description)
FILE_MAP = [
    ("gf_scores.json", "Step 3", "compute_gf.py", "G-F curves and scores (11 methods)"),
    ("gnn_gf_scores.json", "Step 15", "embed_gnn.py", "GNN G-F scores"),
    ("leiden_baseline.json", "Step 4", "leiden_baseline.py", "Community detection baseline (4 algorithms)"),
    ("bonferroni_results.json", "Step 5", "robustness.py", "Bonferroni-corrected subset robustness"),
    ("geometric_analysis.json", "Step 7", "geometric_analysis.py", "d_intra/d_inter geometric margins"),
    ("link_prediction.json", "Step 8", "link_prediction.py", "Link prediction AUC (6 methods, classifier CV)"),
    ("downstream_knn.json", "Step 9", "downstream_knn.py", "k-NN GO term prediction F1"),
    ("randomization_control.json", "Step 10", "randomization_control.py", "Shuffled-label control"),
    ("plateau_width_v3_200pts.csv", "Step 3", "compute_gf.py", "Plateau widths (200-point grid)"),
    ("adaptive_interval.json", "Step 16", "adaptive_interval.py", "Data-driven consensus interval"),
    ("go_propagation.json", "Step 19", "go_propagation.py", "GO DAG True Path Rule expansion"),
    ("bonferroni_results.json", "Step 5", "robustness.py", "Bonferroni correction results"),
    ("topological_analysis.json", "Step 22", "topological_analysis.py", "Persistent homology / Betti curves"),
    ("topological_correlation_analysis.json", "Step 23", "topological_stats.py", "Topological feature correlations"),
    ("topological_stats.json", "Step 23", "topological_stats.py", "Topological statistics"),
    ("metric_comparison.json", "Step 27", "metric_comparison.py", "GF Score vs AUC + F1 (11 methods)"),
    ("metric_comparison_extended_15.json", "Step 27+", "metric_comparison_extended.py", "Extended 15-method comparison"),
    ("bootstrap_correlations.json", "Step 28", "bootstrap_correlations.py", "Bootstrap 95% CI for Spearman"),
    ("semantic_purity_analysis.json", "Step 29", "semantic_similarity_analysis.py", "IC-weighted + Resnik purity"),
    ("cross_species_consistency.json", "Step 30", "cross_species_consistency.py", "Yeast vs human rank concordance"),
    ("scale_gradient.json", "Step 31", "scale_gradient.py", "Scale-dependent topology coupling"),
    ("bootstrap_stability.json", "Step 32", "bootstrap_stability.py", "Bootstrap CI for GF rankings"),
    ("human_gf_scores_extended.json", "Step 33", "human_gf_extended.py", "Human 11-method GF scores"),
    ("multimodal_anchoring.json", "Step 34", "multimodal_functional_anchoring.py", "STRING threshold + channel GF"),
    ("hyperparameter_sensitivity.json", "Step 35", "hyperparameter_sensitivity.py", "Parameter sensitivity"),
    ("density_corrected_gf.json", "Step 36", "density_corrected_gf.py", "Density-corrected GF"),
    ("human_seed_stability.json", "Step 37", "human_seed_stability.py", "Human 10-seed stability"),
    ("human_ic_weighted_gf.json", "Step 38", "human_ic_weighted_gf.py", "Human IC-weighted GF"),
    ("gat_collapse_diagnosis.json", "Step 39", "gat_collapse_diagnosis.py", "GAT 5-variant ablation"),
    ("cross_species_three_way.json", "Step 39b", "persistence_image_analysis.py", "Three-species persistence images"),
    ("ecoli_gf_scores.json", "Step 40", "ecoli_analysis.py", "E. coli cross-species validation"),
    ("coexpression_gf.json", "Step 41", "coexpression_gf.py", "Coexpression network GF"),
    ("degree_preserving_null.json", "Step 42", "degree_preserving_null.py", "Degree-preserving null model"),
    ("degree_embedding_correlation.json", "Step 42b", "degree_preserving_null.py", "Degree-embedding correlation"),
    ("gat_theorem_full_network.json", "Step 43", "gat_theorem_large_network.py", "GAT theorem on 5936-node network"),
    ("gf_ablation_community_detection.json", "Step 44", "gf_ablation_community_detection.py", "Community detection ablation"),
    ("function_prediction_full.json", "Step 45", "function_prediction_full.py", "11-method LOTO-CV function prediction"),
    ("gf_phase_transition.json", "Step 46", "gf_phase_transition.py", "G-F curve phase transition"),
    ("dimension_sweep_extended.json", "Step 47", "dimension_sweep_extended.py", "Extended dimension sweep d=128/256"),
    ("functional_dark_matter.json", "Step 48", "functional_dark_matter.py", "Functional dark matter (2D, 44 pairs)"),
    ("functional_dark_matter_d64.json", "Step 48", "functional_dark_matter.py --dim 64", "Functional dark matter (64D, 35 pairs)"),
    ("cross_species_dark_matter.json", "Step 49", "cross_species_dark_matter.py", "Cross-species dark matter"),
    ("string_v12_revalidation.json", "Step 51", "string_v12_revalidation.py", "STRING v12.0 re-validation"),
    ("cross_species_highdim.json", "Step 53", "cross_species_highdim.py", "2D vs 64D cross-species conservation"),
    ("highdim_gf_comparison.json", "Steps 54-55", "dimension_gradient_3species.py", "High-dim GF comparison d=2-64"),
    ("umap_tsne_gf.json", "Step 56", "umap_tsne_gf.py", "UMAP/t-SNE GF evaluation"),
    ("go_mf_cc_gf_scores.json", "Step 57", "go_mf_cc_gf_scores.py", "GO MF/CC/BP GF scores"),
    ("string_threshold_sensitivity.json", "Step 58", "string_threshold_sensitivity.py", "STRING threshold 600-800"),
    ("gatv2_experiment.json", "Step 59", "gatv2_experiment.py", "GATv2 vs GAT comparison"),
    ("fly_gf_scores.json", "Step 60", "fly_analysis.py", "Drosophila 5th species validation"),
    ("prone_harp_gf.json", "Step 61", "prone_harp_gf.py", "ProNE + HARP GF scores"),
    ("function_prediction_cosine.json", "Step 62", "function_prediction_cosine.py", "Cosine voting function prediction"),
    ("heat_kernel_multiscale.json", "Step 63", "heat_kernel_multiscale.py", "Heat kernel multi-scale analysis"),
    ("position_encoding_comparison.json", "Step 64", "position_encoding_comparison.py", "PE benchmark: Laplacian, RWPE, SignNet"),
    ("cheeger_gf_bound.json", "Step 65", "cheeger_gf_bound.py", "Cheeger-Spectral GF upper bound"),
    ("dimension_sweep_512.json", "Step 66", "dimension_sweep_512.py", "Dimension sweep d=512/1024"),
    ("function_prediction_atlas.json", "Step 67", "function_prediction_atlas.py", "Multi-ontology function prediction atlas"),
    ("cross_species_atlas.json", "Step 69", "cross_species_atlas.py", "Cross-species atlas (human + mouse)"),
    ("ortholog_cross_validation.json", "Step 71", "ortholog_cross_validation.py", "Ortholog cross-validation"),
    ("dark_matter_pan_species.json", "Step 72", "dark_matter_pan_species.py", "Pan-species dark matter"),
    ("final_results_summary.json", "Summary", "run_all_analysis.py", "Master summary (auto-generated)"),
    ("multiple_comparison_correction.json", "P0-1", "multiple_comparison_correction.py", "FDR correction for 13 Spearman tests"),
    ("outlier_sensitivity_analysis.json", "P1-1", "outlier_sensitivity_analysis.py", "Human embedding outlier analysis"),
    ("interval_sensitivity_analysis.json", "P1-2", "interval_sensitivity_analysis.py", "Interval sensitivity (7 intervals)"),
    ("partial_correlation_analysis.json", "Suppl", "partial_correlation_analysis.py", "Partial Spearman correlations"),
    ("multiseed_panel.json", "Phase 12", "biological_validation.py", "Multi-seed panel (n=220)"),
]


def get_file_info(filepath):
    """Get file size, modification time, and JSON description if available."""
    stat = filepath.stat()
    size_kb = stat.st_size / 1024
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")

    description = ""
    if filepath.suffix == ".json":
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                description = data.get("description", data.get("desc", ""))[:80]
        except Exception as e:
            import logging; logging.warning(f"Exception in {__name__}: {e}")
            pass

    return size_kb, mtime, description


def match_file(filename):
    """Match a filename to the FILE_MAP entries."""
    for pattern, step, script, desc in FILE_MAP:
        if pattern in filename:
            return step, script, desc
    return "?", "?", ""


def main():
    print("=" * 64)
    print("  P1-5: Generating results/MANIFEST.md")
    print("=" * 64)

    files = sorted(RESULTS_DIR.glob("*"))
    json_files = [f for f in files if f.suffix in (".json", ".csv")]
    other_files = [f for f in files if f.suffix not in (".json", ".csv") and f.is_file()]

    print(f"  Found {len(json_files)} JSON/CSV files, {len(other_files)} other files")

    lines = [
        "# Results Manifest\n",
        f"\nAuto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"Total files: {len(json_files)} JSON/CSV + {len(other_files)} other\n\n",
        "| File | Step | Source Script | Size (KB) | Modified | Description |\n",
        "|------|------|---------------|-----------|----------|-------------|\n",
    ]

    for f in json_files:
        size_kb, mtime, json_desc = get_file_info(f)
        step, script, mapped_desc = match_file(f.name)
        desc = mapped_desc or json_desc or ""
        lines.append(f"| `{f.name}` | {step} | `{script}` | {size_kb:.1f} | {mtime} | {desc} |\n")

    lines.append(f"\n## Other files ({len(other_files)})\n\n")
    lines.append("| File | Size (KB) | Modified |\n")
    lines.append("|------|-----------|----------|\n")
    for f in other_files:
        size_kb, mtime, _ = get_file_info(f)
        lines.append(f"| `{f.name}` | {size_kb:.1f} | {mtime} |\n")

    manifest_path = RESULTS_DIR / "MANIFEST.md"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n  Manifest written to: {manifest_path}")
    print(f"  {len(json_files)} files catalogued")
    print("=" * 64)


if __name__ == "__main__":
    main()
