#!/usr/bin/env python3
"""
sampling_density.py
Step 11: Sampling density verification - compare 30-point vs 200-point grid.
Generate Supplementary Table S3.
DM plateau width W should be stable; MDS pseudo-plateau should disappear at 200 pts.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, R_MIN, R_MAX, GF_R_MIN, GF_R_MAX, CLASSICAL_METHODS,
    PLATEAU_RELATIVE_THRESHOLD, get_data_dir, get_results_dir,
    get_embeddings_dir, load_curated_network, load_embedding,
    compute_gf_curve, compute_plateau_width, compute_gf_score,
)


def main():
    np.random.seed(SEED)
    
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()
    
    G, nodes, go_map = load_curated_network(data_dir)
    
    methods = CLASSICAL_METHODS
    grids = {"30": 30, "200": 200}
    
    results = {}
    
    for method in methods:
        print(f"\nEvaluating {method}...")
        try:
            coords, emb_nodes = load_embedding(method, "153", embeddings_dir=emb_dir)
            common = sorted(set(emb_nodes) & set(nodes))
            emb_node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
            idx_map = [emb_node_to_idx[n] for n in common]
            aligned_coords = coords[idx_map]
            
            results[method] = {}
            for grid_name, n_pts in grids.items():
                r_vals = np.linspace(R_MIN, R_MAX, n_pts)
                purities, modularities = compute_gf_curve(aligned_coords, common, go_map, r_vals)
                
                gf_score = compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)
                pw = compute_plateau_width(r_vals, purities,
                                           relative_threshold=PLATEAU_RELATIVE_THRESHOLD)
                
                results[method][grid_name] = {
                    "n_points": n_pts,
                    "gf_score": gf_score,
                    "plateau_width": pw["W"],
                    "max_purity": pw["peak_purity"],
                }
                print(f"  {grid_name}-pt: GF={gf_score:.4f}, W={pw['W']:.4f}, "
                      f"max_pur={pw['peak_purity']:.4f}")
        except Exception as e:
            print(f"  {method} FAILED: {e}")
    
    # Save as Supplementary Table S3
    output_file = results_dir / "sampling_density_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
