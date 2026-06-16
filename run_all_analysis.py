#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Master script to run all analyses for the G-F consistency framework.

Version: 2.1.0

This script orchestrates the complete analysis pipeline:
1.  Data preprocessing
2.  Embedding computation (8 classical + 3 GNN methods)
3.  G-F curve computation (200-point grid)
4.  Leiden baseline
5.  Robustness analysis (30 subsets x 5 size levels)
6.  Full network validation (5,936 nodes)
7.  Geometric analysis
8.  Link prediction (5-fold CV)
9.  Downstream k-NN evaluation
10. Randomization control
11. Sampling density verification
12. G-F score sensitivity analysis
13. Human cross-species validation (optional)
14. Figure generation
15. GNN embeddings (GraphSAGE, GAT, GIN)
16. Adaptive unified interval
17. Network topology analysis (cross-species)
18. Rank reversal analysis (p/q sensitivity + geometric gap)
19. GO propagation (True Path Rule DAG expansion)
20. Biological interpretation (4-level G-F scale + case study)
21. Runtime benchmark (step-wise profiling + complexity)
22. Persistent homology / topological analysis
23. Topological statistics & correlation
24. Hyperbolic embedding (Poincare Ball)
25. Pathway enrichment analysis
26. Statistical analysis summary
27. Metric comparison (G-F Score vs link pred AUC + k-NN F1)
28. Bootstrap correlations (95% CI for key Spearman correlations)
29. Semantic purity & similarity analysis (IC-weighted + Resnik + DAG diagnostics)
30. Cross-species rank consistency (yeast vs human Spearman + Kendall W)
31. Scale-dependent topology coupling (500-4000 node gradient)
32. Bootstrap stability analysis (30 resamples, 80% sampling)
33. Human extended embeddings (11 methods: PCA, VGAE-feat, GraphSAGE, GAT, GIN)
34. Multi-modal functional anchoring (STRING threshold gradient + channel networks)
35. Hyperparameter sensitivity analysis (r-points, resolution, dimensions, walk params)
36. Density-corrected G-F Score (random baseline normalization)
37. 10-seed subsample stability test (human 11 methods)
38. IC-weighted G-F Score on human PPI (11 methods)
39. GAT embedding collapse root-cause analysis
40. E. coli K-12 cross-species validation (4th species)
41. Coexpression network G-F analysis (network-type dependence)
42. Degree-preserving null model (50 randomizations)
43. GAT collapse theorem verification on full 5936-node network
44. G-F Score ablation: community detection sensitivity (5 algorithms)
45. Full 11-method LOTO-CV function prediction
46. G-F curve phase transition analysis (critical radii, Betti coincidence)
47. Extended dimension sweep d=128/256 (test PPI baseline crossing)
48. Functional dark matter mining (embedding-only functional associations)

Usage:
    python run_all_analysis.py                         # Skip human validation
    python run_all_analysis.py --run-human              # Include human validation
    python run_all_analysis.py --skip-plots             # Skip figure generation
    python run_all_analysis.py --skip-gnn               # Skip GNN embeddings
    python run_all_analysis.py --skip-extended          # Skip Steps 16-21, 24-48
    python run_all_analysis.py --skip-topological       # Skip Steps 22-23
    python run_all_analysis.py --start-from 3           # Start from step 3
    python run_all_analysis.py --config my_config.yaml  # Custom config file
    gf-consistency --config pipeline_config.yaml        # Via pip entry point
"""

__version__ = "2.3.0"

import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def cli_main():
    """Entry point for the ``gf-consistency`` console script."""
    main()


def print_header(step_name):
    """Print a formatted header for each analysis step."""
    print("\n" + "=" * 70)
    print(f"  {step_name}")
    print("=" * 70 + "\n")


def run_step(step_fn, step_name):
    """Run a pipeline step with timing and error handling.

    Returns True on success, False on failure.
    """
    t0 = time.time()
    try:
        step_fn()
        elapsed = time.time() - t0
        print(f"  {step_name} completed ({elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {step_name} FAILED after {elapsed:.1f}s: {e}")
        return False


def generate_final_summary(results_dir):
    """Generate final_results_summary.json from all result files."""
    results_dir = Path(results_dir)
    summary = {}

    # 1. G-F Scores
    gf_file = results_dir / "gf_scores.json"
    if gf_file.exists():
        with open(gf_file, encoding="utf-8") as f:
            data = json.load(f)
        summary["gf_scores"] = data.get("scores", data.get("scores_paper_interval", {}))
        summary["unified_interval"] = data.get("unified_interval",
                                                 data.get("unified_interval_paper", [0.05, 0.422]))
        summary["random_baseline"] = data.get("random_baseline",
                                               data.get("random_baseline_mean", None))

    # 1b. Merge GNN method results
    gnn_file = results_dir / "gnn_gf_scores.json"
    if gnn_file.exists():
        with open(gnn_file, encoding="utf-8") as f:
            gnn_data = json.load(f)
        if "gf_scores" in summary and "gf_scores" in gnn_data:
            summary["gf_scores"].update(gnn_data["gf_scores"])
        if "link_prediction" not in summary:
            summary["link_prediction"] = {}
        if "link_prediction" in gnn_data:
            summary["link_prediction"].update(gnn_data["link_prediction"])
        if "downstream_knn" not in summary:
            summary["downstream_knn"] = {}
        if "downstream_knn" in gnn_data:
            summary["downstream_knn"].update(gnn_data["downstream_knn"])
        if "plateau_width_200pts" not in summary:
            summary["plateau_width_200pts"] = {}
        if "plateau_widths" in gnn_data:
            summary["plateau_width_200pts"].update(gnn_data["plateau_widths"])

    # 2. Leiden baseline
    leiden_file = results_dir / "leiden_baseline.json"
    if leiden_file.exists():
        with open(leiden_file, encoding="utf-8") as f:
            data = json.load(f)
        summary["leiden_baseline_purity"] = data.get("leiden_baseline_purity", None)

    # 3. Geometric analysis
    geom_file = results_dir / "geometric_analysis.json"
    if geom_file.exists():
        with open(geom_file, encoding="utf-8") as f:
            data = json.load(f)
        summary["geometric_margins"] = {}
        for method, vals in data.items():
            if isinstance(vals, dict):
                summary["geometric_margins"][method] = {
                    "d_intra": vals.get("d_intra"),
                    "d_inter": vals.get("d_inter"),
                    "gap": vals.get("gap", vals.get("margin")),
                }

    # 4. Link prediction
    lp_file = results_dir / "link_prediction.json"
    if not lp_file.exists():
        lp_file = results_dir / "link_prediction_yeast.json"
    if lp_file.exists():
        with open(lp_file, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Old format: list of method results
            summary["link_prediction"] = {}
            for item in data:
                method = item.get("method")
                if method:
                    summary["link_prediction"][method] = {
                        "auroc_mean": item.get("auroc_mean"),
                        "auroc_std": item.get("auroc_std"),
                    }
        elif isinstance(data, dict):
            summary["link_prediction"] = data.get("auroc_results", {})
            summary["spearman_rho_auroc_gf"] = data.get("spearman_rho_auroc_gf")

    # 5. Downstream k-NN
    knn_file = results_dir / "downstream_knn.json"
    if knn_file.exists():
        with open(knn_file, encoding="utf-8") as f:
            data = json.load(f)
        summary["downstream_knn"] = data.get("results", {})
        summary["knn_n_nodes"] = data.get("n_nodes")
        summary["knn_n_categories"] = data.get("n_categories")

    # 6. Plateau widths
    plateau_file = results_dir / "plateau_width_v3_200pts.csv"
    if plateau_file.exists():
        import csv
        summary["plateau_width_200pts"] = {}
        with open(plateau_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = row.get("Method", row.get("method", ""))
                summary["plateau_width_200pts"][method] = {
                    "W": float(row.get("W", 0)),
                    "r_min": float(row.get("r_min", 0)),
                    "r_max": float(row.get("r_max", 0)),
                    "peak_purity": float(row.get("peak_purity", 0)),
                    "effective_threshold": float(row.get("effective_threshold", 0)),
                }

    # 7. Bonferroni results (with cross-validation)
    bonf_file = results_dir / "bonferroni_results.json"
    if bonf_file.exists():
        with open(bonf_file, encoding="utf-8") as f:
            data = json.load(f)

        # Cross-validate n_significant_corrected against the boolean array
        n_sig_corrected = data.get("n_significant_corrected")
        sig_corrected_array = data.get("significant_corrected")
        if sig_corrected_array is not None:
            array_count = sum(sig_corrected_array)
            if n_sig_corrected != array_count:
                print(f"WARNING: bonferroni_results.json n_significant_corrected={n_sig_corrected} "
                      f"but sum(significant_corrected)={array_count}. Using array count.")
                n_sig_corrected = array_count
        summary["bonferroni_n_significant"] = n_sig_corrected

        # Cross-validate n_significant_in_plateau against available array data
        n_sig_plateau = data.get("n_significant_in_plateau")
        sig_in_plateau_array = data.get("significant_in_plateau")
        if sig_in_plateau_array is not None:
            plateau_count = sum(sig_in_plateau_array)
            if n_sig_plateau != plateau_count:
                print(f"WARNING: bonferroni_results.json n_significant_in_plateau={n_sig_plateau} "
                      f"but sum(significant_in_plateau)={plateau_count}. Using array count.")
                n_sig_plateau = plateau_count
        summary["bonferroni_n_significant_in_plateau"] = n_sig_plateau

    # 8. Randomization control
    rand_file = results_dir / "randomization_control.json"
    if rand_file.exists():
        with open(rand_file, encoding="utf-8") as f:
            data = json.load(f)
        summary["randomization_original_max_purity"] = data.get("original_max_purity")
        summary["randomization_shuffled_max_purity"] = data.get("shuffled_max_purity")

    # 9. Metric comparison (Step 27)
    mc_file = results_dir / "metric_comparison.json"
    if mc_file.exists():
        with open(mc_file, encoding="utf-8") as f:
            data = json.load(f)
        if "correlations" in data:
            summary["metric_comparison"] = data["correlations"]

    # 10. Bootstrap correlations (Step 28)
    bc_file = results_dir / "bootstrap_correlations.json"
    if bc_file.exists():
        with open(bc_file, encoding="utf-8") as f:
            data = json.load(f)
        if "bootstrap_correlations" in data:
            summary["bootstrap_correlations"] = data["bootstrap_correlations"]

    # 11. Semantic purity analysis (Step 29)
    sp_file = results_dir / "semantic_purity_analysis.json"
    if sp_file.exists():
        with open(sp_file, encoding="utf-8") as f:
            data = json.load(f)
        if "gf_scores" in data:
            summary["semantic_purity_scores"] = data["gf_scores"]
        if "correlations" in data:
            summary["semantic_purity_correlations"] = data["correlations"]
        if "dag_inflation" in data:
            summary["dag_inflation"] = data["dag_inflation"]

    # 12. Cross-species rank consistency (Step 30)
    cs_file = results_dir / "cross_species_consistency.json"
    if cs_file.exists():
        with open(cs_file, encoding="utf-8") as f:
            data = json.load(f)
        if "spearman_correlation" in data:
            summary["cross_species_consistency"] = {
                "spearman": data["spearman_correlation"],
                "kendalls_w": data.get("kendalls_w"),
                "n_shared_methods": data.get("n_shared"),
            }

    # 13. Scale gradient analysis (Step 31)
    sg_file = results_dir / "scale_gradient.json"
    if sg_file.exists():
        with open(sg_file, encoding="utf-8") as f:
            data = json.load(f)
        if "kendalls_w" in data:
            summary["scale_gradient"] = {
                "kendalls_w": data["kendalls_w"],
                "scales": data.get("scale_sizes"),
                "methods": data.get("methods"),
            }

    # 14. Bootstrap stability analysis (Step 32)
    bs_file = results_dir / "bootstrap_stability.json"
    if bs_file.exists():
        with open(bs_file, encoding="utf-8") as f:
            data = json.load(f)
        if "bootstrap_stats" in data:
            summary["bootstrap_stability"] = {
                m: {"ci_95": s["ci_95"], "cv": s["cv"]}
                for m, s in data["bootstrap_stats"].items()
            }
        if "adjacent_significance" in data:
            summary["bootstrap_adjacent_significance"] = data["adjacent_significance"]

    # 15. Human extended GF scores (Step 33)
    hex_file = results_dir / "human_gf_scores_extended.json"
    if hex_file.exists():
        with open(hex_file, encoding="utf-8") as f:
            data = json.load(f)
        summary["human_gf_scores_extended"] = {
            "scores": data.get("scores", {}),
            "ranking": data.get("ranking", []),
            "n_methods": data.get("n_methods"),
            "unified_interval": data.get("unified_interval"),
        }

    # 16. Multi-modal functional anchoring (Step 34)
    mm_file = results_dir / "multimodal_anchoring.json"
    if mm_file.exists():
        with open(mm_file, encoding="utf-8") as f:
            data = json.load(f)
        summary["multimodal_anchoring"] = {
            "kendalls_w_threshold": data.get("kendalls_w_threshold"),
            "kendalls_w_channel": data.get("kendalls_w_channel"),
            "thresholds_tested": data.get("thresholds_tested"),
            "n_valid_thresholds": len(data.get("valid_thresholds", [])),
        }

    # 17. Hyperparameter sensitivity (Step 35)
    hp_file = results_dir / "hyperparameter_sensitivity.json"
    if hp_file.exists():
        with open(hp_file, encoding="utf-8") as f:
            data = json.load(f)
        if "summary" in data:
            summary["hyperparameter_sensitivity"] = data["summary"]

    # Save
    output_file = results_dir / "final_results_summary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved final results summary to: {output_file}")
    return summary


def main():
    """Run all analysis steps in sequence."""
    parser = argparse.ArgumentParser(
        description="Run the complete G-F consistency analysis pipeline.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (default: pipeline_config.yaml)")
    parser.add_argument("--run-human", action="store_true",
                        help="Include human cross-species validation (resource-intensive)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--start-from", type=int, default=None,
                        help="Start from a specific step (1-48)")
    parser.add_argument("--skip-gnn", action="store_true",
                        help="Skip GNN embedding computation (Step 15)")
    parser.add_argument("--skip-extended", action="store_true",
                        help="Skip extended analysis steps (16-21, 24-48)")
    parser.add_argument("--skip-topological", action="store_true",
                        help="Skip topological analysis steps (22-23)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--species", type=str, default=None,
                        help="Target species (yeast or human)")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    # ---- Load configuration ----
    try:
        from scripts.config_loader import load_config
        cfg = load_config(
            config_path=args.config,
            project_root=project_root,
            merge_cli=True,
        )
    except Exception as e:
        print(f"Warning: config loading failed ({e}), using defaults")
        cfg = {"pipeline": {}, "paths": {}}

    pipeline_cfg = cfg.get("pipeline", {})

    # CLI flags take final precedence
    run_human = args.run_human or pipeline_cfg.get("run_human", False)
    skip_plots = args.skip_plots or pipeline_cfg.get("skip_plots", False)
    skip_gnn = args.skip_gnn or pipeline_cfg.get("skip_gnn", False)
    skip_extended = args.skip_extended or pipeline_cfg.get("skip_extended", False)
    skip_topological = args.skip_topological or pipeline_cfg.get("skip_topological", False)
    start_from = args.start_from if args.start_from is not None else pipeline_cfg.get("start_from", 1)
    seed = args.seed if args.seed is not None else pipeline_cfg.get("seed", 42)
    species = args.species or pipeline_cfg.get("species", "yeast")

    pipeline_start = time.time()
    completed, failed = 0, 0

    print("\n" + "#" * 70)
    print(f"  G-F Consistency Framework v{__version__}")
    print(f"  Species: {species}  |  Seed: {seed}")
    print("#" * 70)
    if run_human:
        print("  Human validation: ENABLED (may require significant resources)")
    if skip_plots:
        print("  Figure generation: SKIPPED")
    if skip_gnn:
        print("  GNN embeddings: SKIPPED")
    if skip_extended:
        print("  Extended analysis (Steps 16-21, 24-48): SKIPPED")
    if skip_topological:
        print("  Topological analysis (Steps 22-23): SKIPPED")
    print()

    # ---- Pre-flight validation ----
    try:
        from scripts.input_validator import preflight_check
        check = preflight_check(cfg)
        if check.errors:
            print("Pre-flight validation ERRORS:")
            for e in check.errors:
                print(f"  [ERROR] {e}")
        if check.warnings:
            print("Pre-flight validation warnings:")
            for w in check.warnings:
                print(f"  [WARN]  {w}")
        if not check.valid:
            print("\nPre-flight check FAILED. Fix errors above or use --start-from to skip.")
            print("Continuing anyway (best effort)...\n")
        else:
            print(f"Pre-flight check passed: {check.summary()}\n")
    except Exception as e:
        print(f"Pre-flight validation skipped ({e})\n")

    # Step 1: Data Preprocessing
    if start_from <= 1:
        print_header("Step 1: Data Preprocessing")
        from scripts.data_preprocessing import main as preprocess_main
        if run_step(preprocess_main, "Data preprocessing"):
            completed += 1
        else:
            failed += 1

    # Step 2: Compute all embeddings
    if start_from <= 2:
        print_header("Step 2: Computing Embeddings (8 methods)")
        from scripts.embed_all import main as embed_main
        if run_step(embed_main, "Embedding computation"):
            completed += 1
        else:
            failed += 1

    # Step 3: Compute G-F curves
    if start_from <= 3:
        print_header("Step 3: Computing G-F Curves (200-point grid)")
        from scripts.compute_gf import main as gf_main
        # Guard against sys.argv contamination from the master script's flags
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(gf_main, "G-F curve computation"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 4: Leiden baseline
    if start_from <= 4:
        print_header("Step 4: Leiden Baseline on Original Network")
        from scripts.leiden_baseline import main as leiden_main
        if run_step(leiden_main, "Leiden baseline"):
            completed += 1
        else:
            failed += 1

    # Step 5: Robustness analysis
    if start_from <= 5:
        print_header("Step 5: Robustness Analysis (30 subsets x 5 sizes)")
        from scripts.robustness import main as robustness_main
        # Guard against sys.argv contamination from the master script's flags
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(robustness_main, "Robustness analysis"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 6: Full network validation
    if start_from <= 6:
        print_header("Step 6: Full Network Validation (5,936 nodes)")
        from scripts.full_network import main as full_network_main
        if run_step(full_network_main, "Full network validation"):
            completed += 1
        else:
            failed += 1

    # Step 7: Geometric analysis
    if start_from <= 7:
        print_header("Step 7: Geometric Analysis (d_intra / d_inter)")
        from scripts.geometric_analysis import main as geometric_main
        if run_step(geometric_main, "Geometric analysis"):
            completed += 1
        else:
            failed += 1

    # Step 8: Link prediction
    if start_from <= 8:
        print_header("Step 8: Link Prediction (5-fold CV)")
        from scripts.link_prediction import main as link_pred_main
        if run_step(link_pred_main, "Link prediction"):
            completed += 1
        else:
            failed += 1

    # Step 9: Downstream k-NN evaluation
    if start_from <= 9:
        print_header("Step 9: Downstream k-NN Evaluation")
        from scripts.downstream_knn import main as knn_main
        if run_step(knn_main, "Downstream evaluation"):
            completed += 1
        else:
            failed += 1

    # Step 10: Randomization control
    if start_from <= 10:
        print_header("Step 10: Randomization Control")
        from scripts.randomization_control import main as rand_main
        if run_step(rand_main, "Randomization control"):
            completed += 1
        else:
            failed += 1

    # Step 11: Sampling density verification
    if start_from <= 11:
        print_header("Step 11: Sampling Density Verification (30 vs 200 pts)")
        from scripts.sampling_density import main as density_main
        if run_step(density_main, "Sampling density verification"):
            completed += 1
        else:
            failed += 1

    # Step 12: G-F score sensitivity analysis
    if start_from <= 12:
        print_header("Step 12: G-F Score Sensitivity Analysis")
        from scripts.gf_score_sensitivity import main as sensitivity_main
        # Guard against sys.argv contamination from the master script's flags
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(sensitivity_main, "G-F score sensitivity"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 13: Human cross-species validation (RESOURCE-INTENSIVE)
    if start_from <= 13:
        print_header("Step 13: Human Cross-Species Validation")
        if run_human:
            from scripts.human_validation import main as human_main
            if run_step(human_main, "Human validation"):
                completed += 1
            else:
                failed += 1
        else:
            print("  Skipped. Use --run-human to enable, or run separately:")
            print("    python human_validation/run_human_validation.py")

    # Step 14: Generate figures
    if start_from <= 14 and not skip_plots:
        print_header("Step 14: Generating Figures")
        from scripts.plot_figures import main as plot_main
        # Guard against sys.argv contamination from the master script's flags
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(plot_main, "Figure generation"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # ---- Extended Analysis Steps (P0/P1/P2) ----

    # Step 15: GNN Embeddings (GraphSAGE, GAT, GIN)
    if start_from <= 15 and not skip_gnn:
        print_header("Step 15: GNN Embeddings (GraphSAGE, GAT, GIN)")
        from scripts.embed_gnn import main as gnn_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(gnn_main, "GNN embedding computation"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 16: Adaptive Unified Interval
    if start_from <= 16 and not skip_extended:
        print_header("Step 16: Adaptive Unified Interval")
        from scripts.adaptive_interval import main as adaptive_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(adaptive_main, "Adaptive interval"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 17: Network Topology Analysis
    if start_from <= 17 and not skip_extended:
        print_header("Step 17: Network Topology Analysis")
        import subprocess
        topo_cmd = [sys.executable, str(Path(__file__).parent / "scripts" / "network_topology_analysis.py")]
        if not run_human:
            topo_cmd.append("--skip-human")
        def run_topology():
            subprocess.run(topo_cmd, check=True)
        if run_step(run_topology, "Network topology analysis"):
            completed += 1
        else:
            failed += 1

    # Step 18: Rank Reversal Analysis
    if start_from <= 18 and not skip_extended:
        print_header("Step 18: Rank Reversal Analysis")
        rr_cmd = [sys.executable, str(Path(__file__).parent / "scripts" / "rank_reversal_analysis.py")]
        if not run_human:
            rr_cmd.append("--skip-human")
        def run_rank_reversal():
            subprocess.run(rr_cmd, check=True)
        if run_step(run_rank_reversal, "Rank reversal analysis"):
            completed += 1
        else:
            failed += 1

    # Step 19: GO Propagation & Validation Set Expansion
    if start_from <= 19 and not skip_extended:
        print_header("Step 19: GO Propagation & Validation Set Expansion")
        from scripts.go_propagation import main as go_prop_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(go_prop_main, "GO propagation"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 20: Biological Interpretation
    if start_from <= 20 and not skip_extended:
        print_header("Step 20: Biological Interpretation")
        from scripts.biological_interpretation import main as bio_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(bio_main, "Biological interpretation"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 21: Runtime Benchmark
    if start_from <= 21 and not skip_extended:
        print_header("Step 21: Runtime Benchmark")
        bench_cmd = [sys.executable, str(Path(__file__).parent / "scripts" / "benchmark_runtime.py"),
                     "--n-repeat", "1", "--sampling-points", "50,100,200"]
        def run_benchmark():
            subprocess.run(bench_cmd, check=True)
        if run_step(run_benchmark, "Runtime benchmark"):
            completed += 1
        else:
            failed += 1

    # Step 22: Persistent Homology / Topological Analysis
    if start_from <= 22 and not skip_topological:
        print_header("Step 22: Persistent Homology (Topological Analysis)")
        from scripts.topological_analysis import main as topo_analysis_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(topo_analysis_main, "Persistent homology analysis"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 23: Topological Statistics & Correlation
    if start_from <= 23 and not skip_topological:
        print_header("Step 23: Topological Statistics & Correlation")
        from scripts.topological_stats import main as topo_stats_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(topo_stats_main, "Topological statistics"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 24: Hyperbolic Embedding (Poincare Ball)
    if start_from <= 24 and not skip_extended:
        print_header("Step 24: Hyperbolic Embedding (Poincare Ball)")
        from scripts.embed_hyperbolic import main as hyperbolic_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(hyperbolic_main, "Hyperbolic embedding"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 25: Pathway Enrichment Analysis
    if start_from <= 25 and not skip_extended:
        print_header("Step 25: Pathway Enrichment Analysis")
        from scripts.pathway_analysis import main as pathway_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(pathway_main, "Pathway enrichment"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 26: Statistical Analysis Summary
    if start_from <= 26 and not skip_extended:
        print_header("Step 26: Statistical Analysis Summary")
        from scripts.statistical_analysis import main as stat_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(stat_main, "Statistical analysis"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 27: Metric Comparison (G-F Score vs Link Pred AUC + k-NN F1)
    if start_from <= 27 and not skip_extended:
        print_header("Step 27: Metric Comparison")
        from scripts.metric_comparison import main as metric_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(metric_main, "Metric comparison"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 28: Bootstrap Correlations (95% CI for key Spearman correlations)
    if start_from <= 28 and not skip_extended:
        print_header("Step 28: Bootstrap Correlations")
        from scripts.bootstrap_correlations import main as bootstrap_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(bootstrap_main, "Bootstrap correlations"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 29: Semantic Purity & Similarity Analysis (IC-weighted, Resnik, DAG diagnostics)
    if start_from <= 29 and not skip_extended:
        print_header("Step 29: Semantic Purity & Similarity Analysis")
        semantic_cmd = [sys.executable,
                        str(Path(__file__).parent / "scripts" / "semantic_similarity_analysis.py")]
        def run_semantic():
            import subprocess
            subprocess.run(semantic_cmd, check=True)
        if run_step(run_semantic, "Semantic purity analysis"):
            completed += 1
        else:
            failed += 1

    # Step 30: Cross-Species Rank Consistency (yeast vs human)
    if start_from <= 30 and not skip_extended:
        print_header("Step 30: Cross-Species Rank Consistency")
        cross_cmd = [sys.executable,
                     str(Path(__file__).parent / "scripts" / "cross_species_consistency.py")]
        def run_cross_species():
            import subprocess
            subprocess.run(cross_cmd, check=True)
        if run_step(run_cross_species, "Cross-species consistency"):
            completed += 1
        else:
            failed += 1

    # Step 31: Scale-Dependent Topology Coupling (500-4000 nodes)
    if start_from <= 31 and not skip_extended:
        print_header("Step 31: Scale-Dependent Topology Coupling")
        scale_cmd = [sys.executable,
                     str(Path(__file__).parent / "scripts" / "scale_gradient.py")]
        def run_scale_gradient():
            import subprocess
            subprocess.run(scale_cmd, check=True)
        if run_step(run_scale_gradient, "Scale gradient analysis"):
            completed += 1
        else:
            failed += 1

    # Step 32: Bootstrap Stability Analysis (30 resamples, 80% sampling)
    if start_from <= 32 and not skip_extended:
        print_header("Step 32: Bootstrap Stability Analysis")
        boot_cmd = [sys.executable,
                    str(Path(__file__).parent / "scripts" / "bootstrap_stability.py")]
        def run_bootstrap_stability():
            import subprocess
            subprocess.run(boot_cmd, check=True)
        if run_step(run_bootstrap_stability, "Bootstrap stability"):
            completed += 1
        else:
            failed += 1

    # Step 33: Human Extended Embeddings (11 methods)
    if start_from <= 33 and not skip_extended:
        print_header("Step 33: Human Extended Embeddings (11 methods)")
        he_cmd = [sys.executable,
                  str(Path(__file__).parent / "scripts" / "human_embed_extended.py")]
        hgf_cmd = [sys.executable,
                   str(Path(__file__).parent / "scripts" / "human_gf_extended.py")]
        def run_human_extended():
            import subprocess
            subprocess.run(he_cmd, check=True)
            subprocess.run(hgf_cmd, check=True)
        if run_step(run_human_extended, "Human extended embeddings + GF analysis"):
            completed += 1
        else:
            failed += 1

    # Step 34: Multi-Modal Functional Anchoring
    if start_from <= 34 and not skip_extended:
        print_header("Step 34: Multi-Modal Functional Anchoring")
        mm_cmd = [sys.executable,
                  str(Path(__file__).parent / "scripts" / "multimodal_functional_anchoring.py")]
        def run_multimodal():
            import subprocess
            subprocess.run(mm_cmd, check=True)
        if run_step(run_multimodal, "Multi-modal functional anchoring"):
            completed += 1
        else:
            failed += 1

    # Step 35: Hyperparameter Sensitivity Analysis
    if start_from <= 35 and not skip_extended:
        print_header("Step 35: Hyperparameter Sensitivity Analysis")
        hp_cmd = [sys.executable,
                  str(Path(__file__).parent / "scripts" / "hyperparameter_sensitivity.py")]
        def run_hyperparam():
            import subprocess
            subprocess.run(hp_cmd, check=True)
        if run_step(run_hyperparam, "Hyperparameter sensitivity"):
            completed += 1
        else:
            failed += 1

    # Step 36: Density-Corrected G-F Score
    if start_from <= 36 and not skip_extended:
        print_header("Step 36: Density-Corrected G-F Score")
        from scripts.density_corrected_gf import main as density_corrected_main
        _saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            if run_step(density_corrected_main, "Density-corrected GF"):
                completed += 1
            else:
                failed += 1
        finally:
            sys.argv = _saved_argv

    # Step 37: Human Seed Stability (10-seed subsample)
    if start_from <= 37 and not skip_extended:
        print_header("Step 37: Human Seed Stability (10-seed)")
        hs_cmd = [sys.executable,
                  str(Path(__file__).parent / "scripts" / "human_seed_stability.py")]
        def run_seed_stability():
            import subprocess
            subprocess.run(hs_cmd, check=True)
        if run_step(run_seed_stability, "Human seed stability"):
            completed += 1
        else:
            failed += 1

    # Step 38: IC-Weighted G-F Score (Human)
    if start_from <= 38 and not skip_extended:
        print_header("Step 38: IC-Weighted G-F Score (Human)")
        ic_cmd = [sys.executable,
                  str(Path(__file__).parent / "scripts" / "human_ic_weighted_gf.py")]
        def run_ic_weighted():
            import subprocess
            subprocess.run(ic_cmd, check=True)
        if run_step(run_ic_weighted, "IC-weighted GF score"):
            completed += 1
        else:
            failed += 1

    # Step 39: GAT Collapse Diagnosis
    if start_from <= 39 and not skip_extended:
        print_header("Step 39: GAT Collapse Diagnosis")
        gat_cmd = [sys.executable,
                   str(Path(__file__).parent / "scripts" / "gat_collapse_diagnosis.py")]
        def run_gat_diagnosis():
            import subprocess
            subprocess.run(gat_cmd, check=True)
        if run_step(run_gat_diagnosis, "GAT collapse diagnosis"):
            completed += 1
        else:
            failed += 1

    # Step 40: E. coli K-12 Cross-Species Validation (4th species)
    if start_from <= 40 and not skip_extended:
        print_header("Step 40: E. coli K-12 Cross-Species Validation")
        ecoli_cmd = [sys.executable,
                     str(Path(__file__).parent / "scripts" / "ecoli_analysis.py")]
        def run_ecoli():
            import subprocess
            subprocess.run(ecoli_cmd, check=True)
        if run_step(run_ecoli, "E. coli cross-species validation"):
            completed += 1
        else:
            failed += 1

    # Step 41: Coexpression Network G-F Analysis
    if start_from <= 41 and not skip_extended:
        print_header("Step 41: Coexpression Network G-F Analysis")
        coex_cmd = [sys.executable,
                    str(Path(__file__).parent / "scripts" / "coexpression_gf.py")]
        def run_coexpression():
            import subprocess
            subprocess.run(coex_cmd, check=True)
        if run_step(run_coexpression, "Coexpression network analysis"):
            completed += 1
        else:
            failed += 1

    # Step 42: Degree-Preserving Null Model
    if start_from <= 42 and not skip_extended:
        print_header("Step 42: Degree-Preserving Null Model (50 randomizations)")
        dp_cmd = [sys.executable,
                  str(Path(__file__).parent / "scripts" / "degree_preserving_null.py")]
        def run_dp_null():
            import subprocess
            subprocess.run(dp_cmd, check=True)
        if run_step(run_dp_null, "Degree-preserving null model"):
            completed += 1
        else:
            failed += 1

    # Step 43: GAT Collapse Theorem Verification on Full Network
    if start_from <= 43 and not skip_extended:
        print_header("Step 43: GAT Theorem Verification (Full 5936-Node Network)")
        gat_full_cmd = [sys.executable,
                        str(Path(__file__).parent / "scripts" / "gat_theorem_large_network.py")]
        def run_gat_full():
            import subprocess
            subprocess.run(gat_full_cmd, check=True)
        if run_step(run_gat_full, "GAT theorem full-network verification"):
            completed += 1
        else:
            failed += 1

    # Step 44: G-F Score Ablation — Community Detection Sensitivity
    if start_from <= 44 and not skip_extended:
        print_header("Step 44: G-F Score Ablation (Community Detection Sensitivity)")
        ablation_cmd = [sys.executable,
                        str(Path(__file__).parent / "scripts" / "gf_ablation_community_detection.py")]
        def run_ablation():
            import subprocess
            subprocess.run(ablation_cmd, check=True)
        if run_step(run_ablation, "Community detection ablation"):
            completed += 1
        else:
            failed += 1

    # Step 45: Full 11-Method LOTO-CV Function Prediction
    if start_from <= 45 and not skip_extended:
        print_header("Step 45: Full 11-Method LOTO-CV Function Prediction")
        func_full_cmd = [sys.executable,
                         str(Path(__file__).parent / "scripts" / "function_prediction_full.py")]
        def run_func_full():
            import subprocess
            subprocess.run(func_full_cmd, check=True)
        if run_step(run_func_full, "Full 11-method function prediction"):
            completed += 1
        else:
            failed += 1

    # Step 46: G-F Curve Phase Transition Analysis
    if start_from <= 46 and not skip_extended:
        print_header("Step 46: G-F Curve Phase Transition Analysis")
        phase_cmd = [sys.executable,
                     str(Path(__file__).parent / "scripts" / "gf_phase_transition.py")]
        def run_phase():
            import subprocess
            subprocess.run(phase_cmd, check=True)
        if run_step(run_phase, "G-F curve phase transition analysis"):
            completed += 1
        else:
            failed += 1

    # Step 47: Extended Dimension Sweep (d = 128, 256)
    if start_from <= 47 and not skip_extended:
        print_header("Step 47: Extended Dimension Sweep (d = 128, 256)")
        dimsweep_cmd = [sys.executable,
                        str(Path(__file__).parent / "scripts" / "dimension_sweep_extended.py")]
        def run_dimsweep():
            import subprocess
            subprocess.run(dimsweep_cmd, check=True)
        if run_step(run_dimsweep, "Extended dimension sweep"):
            completed += 1
        else:
            failed += 1

    # Step 48: Functional Dark Matter Mining
    if start_from <= 48 and not skip_extended:
        print_header("Step 48: Functional Dark Matter Mining")
        darkmatter_cmd = [sys.executable,
                          str(Path(__file__).parent / "scripts" / "functional_dark_matter.py")]
        def run_darkmatter():
            import subprocess
            subprocess.run(darkmatter_cmd, check=True)
        if run_step(run_darkmatter, "Functional dark matter mining"):
            completed += 1
        else:
            failed += 1

    # Generate final summary
    results_dir = Path(__file__).parent / "results"
    print_header("Final: Generating Results Summary")
    try:
        summary = generate_final_summary(results_dir)
        print("\n=== Key Results ===")
        if "gf_scores" in summary:
            print("\nG-F Scores:")
            for method, score in sorted(summary["gf_scores"].items(),
                                         key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
                print(f"  {method}: {score:.4f}" if isinstance(score, float) else f"  {method}: {score}")
        if "leiden_baseline_purity" in summary:
            print(f"\nLeiden baseline purity: {summary['leiden_baseline_purity']:.4f}")
    except Exception as e:
        print(f"  Summary generation failed: {e}")

    # Pipeline summary
    total_elapsed = time.time() - pipeline_start
    print("\n" + "#" * 70)
    print(f"  Analysis Pipeline Complete!  ({total_elapsed / 60:.1f} min)")
    print(f"  Steps: {completed} succeeded, {failed} failed")
    print("#" * 70)
    print("\nResults are saved in:")
    print("  - results/          : G-F curves, scores, and statistics")
    print("  - figures/          : Publication-ready figures")
    print("  - embeddings/       : Embedding coordinates (.npy)")
    print("  - data/             : Processed data files")
    print()


if __name__ == "__main__":
    main()
