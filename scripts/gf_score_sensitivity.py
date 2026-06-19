#!/usr/bin/env python3
"""
gf_score_sensitivity.py
Step 12: Sensitivity analysis of G-F Score to integration interval choice.
Paper Section: 3.7
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_results_dir, compute_gf_score


def find_interval(r_vals, purity_vals, modularity_vals, random_baseline=0.30):
    """Find structurally informative interval for a single method."""
    mask = (
        (np.array(modularity_vals) > 0.3)
        & (np.array(purity_vals) > random_baseline + 0.1)
    )
    r_filtered = np.array(r_vals)[mask]
    if len(r_filtered) == 0:
        return None
    return float(r_filtered[0]), float(r_filtered[-1])


def main():
    parser = argparse.ArgumentParser(
        description="G-F Score sensitivity analysis"
    )
    parser.add_argument("--results-dir", type=str, help="Results directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else get_results_dir()
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load G-F curves from compute_gf.py output
    curves_file = results_dir / "gf_curves_200pts.json"
    if not curves_file.exists():
        # Fallback: try pickle format
        pkl_file = results_dir / "gf_curves_200pts.pkl"
        if pkl_file.exists():
            import pickle
            with open(pkl_file, "rb") as f:
                gf_curves = pickle.load(f)
        else:
            print(f"Error: {curves_file} not found. Run compute_gf.py first.")
            sys.exit(1)
    else:
        with open(curves_file, encoding="utf-8") as f:
            gf_curves = json.load(f)

    r_vals = np.array(gf_curves["r"])
    methods = ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE"]

    # Compute per-method intervals
    intervals = {}
    for method in methods:
        purity_key = f"{method}_purity"
        mod_key = f"{method}_modularity"
        if purity_key not in gf_curves:
            continue
        interval = find_interval(
            r_vals, gf_curves[purity_key], gf_curves[mod_key]
        )
        intervals[method] = interval
        print(f"{method} interval: {interval}")

    # Compute unified interval
    all_r = set()
    for method, iv in intervals.items():
        if iv is not None:
            mask = (r_vals >= iv[0]) & (r_vals <= iv[1])
            all_r.update(r_vals[mask].tolist())
    r_min_unified = min(all_r) if all_r else 0.05
    r_max_unified = max(all_r) if all_r else 0.422
    print(f"\nUnified interval: [{r_min_unified:.3f}, {r_max_unified:.3f}]")

    # Compute scores with different intervals
    reference_methods = ["DM", "MDS", "Node2Vec"]
    all_scores = {}

    for ref in reference_methods:
        if ref not in intervals or intervals[ref] is None:
            continue
        r_min_ref, r_max_ref = intervals[ref]
        scores = {}
        for method in methods:
            purity_key = f"{method}_purity"
            if purity_key not in gf_curves:
                continue
            scores[method] = round(
                compute_gf_score(r_vals, gf_curves[purity_key], r_min_ref, r_max_ref),
                4,
            )
        all_scores[f"{ref}_interval"] = {
            "interval": [round(r_min_ref, 4), round(r_max_ref, 4)],
            "scores": scores,
        }
        print(f"\nScores with {ref} interval [{r_min_ref:.3f}, {r_max_ref:.3f}]:")
        for m, s in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {m}: {s}")

    # Spearman correlation between rankings
    if "DM_interval" in all_scores and "MDS_interval" in all_scores:
        dm_scores = all_scores["DM_interval"]["scores"]
        mds_scores = all_scores["MDS_interval"]["scores"]
        common = sorted(set(dm_scores.keys()) & set(mds_scores.keys()))

        dm_ranks = sorted(common, key=lambda m: -dm_scores[m])
        mds_ranks = sorted(common, key=lambda m: -mds_scores[m])
        dm_pos = {m: i for i, m in enumerate(dm_ranks)}
        rank_nums = [dm_pos[m] for m in mds_ranks]
        rho, p_val = spearmanr(rank_nums, list(range(len(rank_nums))))
        print(f"\nSpearman rho (DM vs MDS rankings): {rho:.2f}, p = {p_val:.2f}")
    else:
        rho, p_val = None, None

    # Save
    output = {
        "unified_interval": [round(r_min_unified, 4), round(r_max_unified, 4)],
        "individual_intervals": {
            k: list(v) if v else None for k, v in intervals.items()
        },
        "sensitivity_analysis": all_scores,
        "spearman_rho": round(rho, 2) if rho is not None else None,
        "spearman_p": round(p_val, 2) if p_val is not None else None,
    }

    output_file = output_dir / "gf_score_sensitivity.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
