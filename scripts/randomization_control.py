#!/usr/bin/env python3
"""
randomization_control.py
Step 10: Randomization control - shuffle DM embedding coordinates multiple times,
verify that the G-F purity plateau is significantly reduced relative to the original.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_embeddings_dir,
    load_curated_network, load_embedding, compute_gf_curve,
)

R_MIN = 0.05
R_MAX = 0.55
N_POINTS = 200
N_SHUFFLES = 10


def main():
    np.random.seed(SEED)
    
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()
    
    G, nodes, go_map = load_curated_network(data_dir)
    
    # Load DM embedding
    print("Loading DM embedding...")
    dm_coords, dm_nodes = load_embedding("DM", "153", embeddings_dir=emb_dir)
    common = sorted(set(dm_nodes) & set(nodes))
    idx_map = [dm_nodes.index(n) for n in common]
    aligned_coords = dm_coords[idx_map]
    
    # Compute original G-F curve
    print("Computing original DM G-F curve...")
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    orig_purities, orig_modularities = compute_gf_curve(aligned_coords, common, go_map, r_vals)
    
    # Multiple shuffles for robust null-model comparison
    print(f"Computing {N_SHUFFLES} shuffled DM G-F curves...")
    all_shuffled_purities = []
    all_shuffled_modularities = []
    shuffled_max_purities = []
    
    for i in range(N_SHUFFLES):
        rng = np.random.RandomState(SEED + 9999 + i)
        perm = rng.permutation(len(common))
        shuffled_coords = aligned_coords[perm]
        shuf_purities, shuf_modularities = compute_gf_curve(
            shuffled_coords, common, go_map, r_vals
        )
        all_shuffled_purities.append(shuf_purities)
        all_shuffled_modularities.append(shuf_modularities)
        shuffled_max_purities.append(max(shuf_purities))
        print(f"  Shuffle {i+1}/{N_SHUFFLES}: max purity = {max(shuf_purities):.4f}")
    
    # Average shuffled curves
    mean_shuf_purities = list(np.mean(all_shuffled_purities, axis=0))
    mean_shuf_modularities = list(np.mean(all_shuffled_modularities, axis=0))
    mean_shuf_max = float(np.mean(shuffled_max_purities))
    std_shuf_max = float(np.std(shuffled_max_purities))
    
    # Compare
    orig_max_pur = max(orig_purities)
    print(f"\nOriginal DM max purity: {orig_max_pur:.4f}")
    print(f"Shuffled max purity (mean ± std): {mean_shuf_max:.4f} ± {std_shuf_max:.4f}")
    print(f"Drop: {orig_max_pur - mean_shuf_max:.4f} ({(orig_max_pur - mean_shuf_max) / orig_max_pur * 100:.1f}%)")
    
    # Statistical test: is original significantly above shuffled?
    z_score = (orig_max_pur - mean_shuf_max) / (std_shuf_max + 1e-10)
    print(f"Z-score (original vs shuffled): {z_score:.2f}")
    
    # Save results (use first shuffle for backward compatibility)
    result = {
        "r": r_vals.tolist(),
        "DM_original_purity": orig_purities,
        "DM_original_modularity": orig_modularities,
        "DM_shuffled_purity": mean_shuf_purities,
        "DM_shuffled_modularity": mean_shuf_modularities,
        "original_max_purity": orig_max_pur,
        "shuffled_max_purity": mean_shuf_max,
        "shuffled_max_purity_std": std_shuf_max,
        "n_shuffles": N_SHUFFLES,
        "z_score": round(z_score, 2),
    }
    
    output_file = results_dir / "randomization_control.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
