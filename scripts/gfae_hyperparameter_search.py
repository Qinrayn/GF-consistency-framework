#!/usr/bin/env python3
"""
GFAE Hyperparameter Search (Direction A - Phase 2)
====================================================

Systematic hyperparameter sweep for the Functional-Aware Embedding (GFAE).
Trains 20 configurations on the curated 153-node yeast PPI network and
evaluates each with the G-F Score, identifying the settings that maximise
geometric-functional consistency.

Search Space
------------
The sweep covers four axes motivated by the first GFAE run (0.124, rank 6/12):
- **lambda_push**: the push loss was likely too aggressive (1.0); try 0.1-0.5.
- **lambda_reg**: spectral regularisation weight; try 0.05-0.3.
- **hidden_dim**: encoder capacity; try 4, 8, 16.
- **margin**: push loss margin; try 0.3, 0.5, 0.8.

Each configuration is trained for 500 epochs (longer than the initial 300
to ensure convergence) and evaluated with the full 200-point G-F pipeline.

Output
------
- ``results/gfae_hyperparameter_search.json``: all 20 configurations with
  G-F Scores, loss histories, and ranking.
- ``embeddings/GFAE_best_153.npy``: the best-performing embedding.

Usage
-----
.. code-block:: bash

    # Full search (20 configs, ~15 min on GPU, ~30 min on CPU):
    python scripts/gfae_hyperparameter_search.py

    # Quick search (8 configs):
    python scripts/gfae_hyperparameter_search.py --quick

    # Custom number of epochs:
    python scripts/gfae_hyperparameter_search.py --epochs 800
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from utils import (
    SEED,
    TARGET_STD,
    GF_R_MIN,
    GF_R_MAX,
    R_MIN,
    R_MAX,
    N_POINTS,
    get_data_dir,
    get_embeddings_dir,
    get_results_dir,
    load_curated_network,
    compute_centrality_features,
    compute_gf_curve,
    compute_gf_score,
    setup_logging,
)
from functional_aware_embedding import (
    build_go_similarity_matrix,
    train_gfae,
)

logger: logging.Logger = setup_logging("gfae_hyperparameter_search")


# ============================================================
# Hyperparameter Configurations
# ============================================================

def build_search_configs(quick: bool = False) -> list[dict]:
    """Return a list of hyperparameter configurations to evaluate.

    The search is guided by the initial GFAE result (GF=0.124, rank 6/12):
    - The push loss (lambda_push=1.0) was too aggressive; we explore lower values.
    - Spectral regularisation (lambda_reg=0.1) showed promise; we explore a range.
    - Encoder capacity (hidden_dim=4) may be limiting; we try larger.
    - The margin (0.5) controls how far unrelated proteins are pushed apart.
    """
    if quick:
        # 8 configurations for a fast test
        configs = []
        for lambda_push in [0.1, 0.3]:
            for lambda_reg in [0.05, 0.2]:
                for hidden_dim in [4, 8]:
                    configs.append({
                        "name": f"lp{lambda_push}_lr{lambda_reg}_hd{hidden_dim}",
                        "lambda_push": lambda_push,
                        "lambda_reg": lambda_reg,
                        "hidden_dim": hidden_dim,
                        "margin": 0.5,
                        "lambda_pull": 1.0,
                        "lr": 0.01,
                        "epochs": 500,
                    })
        return configs

    # Full 20-configuration search
    configs = []

    # Group 1: Vary lambda_push (the suspected bottleneck)
    for lp in [0.05, 0.1, 0.2, 0.3, 0.5]:
        configs.append({
            "name": f"push_{lp}",
            "lambda_push": lp,
            "lambda_reg": 0.1,
            "hidden_dim": 4,
            "margin": 0.5,
            "lambda_pull": 1.0,
            "lr": 0.01,
            "epochs": 500,
        })

    # Group 2: Vary lambda_reg (spectral alignment strength)
    for lr_reg in [0.0, 0.05, 0.2, 0.3]:
        configs.append({
            "name": f"reg_{lr_reg}",
            "lambda_push": 0.2,
            "lambda_reg": lr_reg,
            "hidden_dim": 4,
            "margin": 0.5,
            "lambda_pull": 1.0,
            "lr": 0.01,
            "epochs": 500,
        })

    # Group 3: Vary hidden_dim (encoder capacity)
    for hd in [8, 16]:
        configs.append({
            "name": f"hd_{hd}",
            "lambda_push": 0.2,
            "lambda_reg": 0.1,
            "hidden_dim": hd,
            "margin": 0.5,
            "lambda_pull": 1.0,
            "lr": 0.01,
            "epochs": 500,
        })

    # Group 4: Vary margin (push distance threshold)
    for m in [0.3, 0.8]:
        configs.append({
            "name": f"margin_{m}",
            "lambda_push": 0.2,
            "lambda_reg": 0.1,
            "hidden_dim": 4,
            "margin": m,
            "lambda_pull": 1.0,
            "lr": 0.01,
            "epochs": 500,
        })

    # Group 5: Combined promising directions
    configs.append({
        "name": "combined_1",
        "lambda_push": 0.1,
        "lambda_reg": 0.2,
        "hidden_dim": 8,
        "margin": 0.5,
        "lambda_pull": 1.0,
        "lr": 0.01,
        "epochs": 500,
    })
    configs.append({
        "name": "combined_2",
        "lambda_push": 0.3,
        "lambda_reg": 0.05,
        "hidden_dim": 8,
        "margin": 0.8,
        "lambda_pull": 1.0,
        "lr": 0.01,
        "epochs": 500,
    })
    configs.append({
        "name": "combined_3",
        "lambda_push": 0.05,
        "lambda_reg": 0.3,
        "hidden_dim": 16,
        "margin": 0.5,
        "lambda_pull": 1.0,
        "lr": 0.01,
        "epochs": 500,
    })

    return configs


# ============================================================
# Evaluation
# ============================================================

def evaluate_gf_score(
    coords: np.ndarray,
    emb_nodes: list[str],
    graph_nodes: list[str],
    go_map: dict,
    r_vals: np.ndarray,
) -> float:
    """Evaluate the G-F Score for a trained embedding.

    Aligns the embedding to the network nodes, then computes the full
    200-point G-F purity curve and integrates over the unified interval.
    """
    node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
    common = sorted(set(node_to_idx) & set(graph_nodes) & set(go_map))
    indices = [node_to_idx[n] for n in common]
    aligned = coords[indices]
    purities, _ = compute_gf_curve(aligned, common, go_map, r_vals)
    return compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GFAE Hyperparameter Search (Direction A - Phase 2)",
    )
    parser.add_argument(
        "--quick", action="store_true", default=False,
        help="Quick search: 8 configs instead of 20.",
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Training epochs per config (default: 500).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Base random seed.",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    emb_dir = get_embeddings_dir()
    results_dir = get_results_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load network and precompute shared data
    # ------------------------------------------------------------------
    logger.info("Loading network ...")
    G, nodes, go_map = load_curated_network(data_dir)
    logger.info("Network: %d nodes, %d edges", len(nodes), G.number_of_edges())

    logger.info("Building GO similarity matrix ...")
    go_sim = build_go_similarity_matrix(nodes, go_map)

    logger.info("Computing centrality features ...")
    features = compute_centrality_features(G, nodes)

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # Load baseline scores for comparison
    baseline_path = results_dir / "gf_scores_all11.json"
    baselines = {}
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as fh:
            baselines = json.load(fh).get("scores", {})
    spectral_score = baselines.get("Spectral", 0.1633)
    random_baseline = 0.1348  # from gf_scores.json

    # ------------------------------------------------------------------
    # Run hyperparameter search
    # ------------------------------------------------------------------
    configs = build_search_configs(quick=args.quick)
    # Override epochs if specified
    for cfg in configs:
        cfg["epochs"] = args.epochs

    logger.info("=" * 60)
    logger.info("Starting hyperparameter search: %d configs", len(configs))
    logger.info("=" * 60)

    results: list[dict] = []
    best_score = -1.0
    best_coords = None
    best_nodes = None
    best_config_name = ""

    t_total = time.time()

    for i, cfg in enumerate(configs, 1):
        cfg_name = cfg["name"]
        logger.info("")
        logger.info("[%d/%d] Config: %s", i, len(configs), cfg_name)
        logger.info(
            "  lambda_push=%.2f  lambda_reg=%.2f  hidden_dim=%d  margin=%.2f  epochs=%d",
            cfg["lambda_push"], cfg["lambda_reg"],
            cfg["hidden_dim"], cfg["margin"], cfg["epochs"],
        )

        t_start = time.time()
        seed = args.seed + i  # distinct seed per config

        try:
            coords, history = train_gfae(
                G, nodes, go_sim,
                features=features,
                hidden_dim=cfg["hidden_dim"],
                latent_dim=2,
                epochs=cfg["epochs"],
                lr=cfg["lr"],
                margin=cfg["margin"],
                lambda_pull=cfg["lambda_pull"],
                lambda_push=cfg["lambda_push"],
                lambda_reg=cfg["lambda_reg"],
                seed=seed,
            )

            gf_score = evaluate_gf_score(
                coords, nodes, nodes, go_map, r_vals,
            )

            elapsed = time.time() - t_start
            final_loss = history["loss"][-1] if history["loss"] else 0.0

            logger.info(
                "  -> GF Score: %.4f  (loss=%.4f, %.1fs)  %s",
                gf_score, final_loss, elapsed,
                "★ BEST" if gf_score > best_score else "",
            )

            result = {
                "config_name": cfg_name,
                "hyperparameters": {k: v for k, v in cfg.items() if k != "name"},
                "gf_score": round(gf_score, 6),
                "final_loss": round(final_loss, 6),
                "training_time_s": round(elapsed, 1),
                "seed": seed,
                "loss_history": history["loss"],
            }
            results.append(result)

            if gf_score > best_score:
                best_score = gf_score
                best_coords = coords
                best_nodes = nodes
                best_config_name = cfg_name

        except Exception as exc:
            logger.warning("  Config %s FAILED: %s", cfg_name, exc)
            results.append({
                "config_name": cfg_name,
                "hyperparameters": {k: v for k, v in cfg.items() if k != "name"},
                "gf_score": 0.0,
                "error": str(exc),
                "seed": seed,
            })

    total_elapsed = time.time() - t_total
    logger.info("")
    logger.info("=" * 60)
    logger.info("Search complete: %d configs in %.1f min", len(configs), total_elapsed / 60)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Sort and rank results
    # ------------------------------------------------------------------
    results.sort(key=lambda x: x.get("gf_score", 0.0), reverse=True)

    logger.info("")
    logger.info("Ranking (by GF Score):")
    logger.info("  %-4s %-20s %8s %8s  %s", "#", "Config", "GF", "Loss", "Notes")
    logger.info("  %s", "-" * 60)
    for rank, r in enumerate(results, 1):
        gf = r.get("gf_score", 0.0)
        loss = r.get("final_loss", 0.0)
        notes = []
        if gf > spectral_score:
            notes.append(">> Spectral!")
        if gf > random_baseline:
            notes.append("> random")
        note_str = ", ".join(notes)
        marker = " ★" if r["config_name"] == best_config_name else ""
        logger.info(
            "  %-4d %-20s %8.4f %8.4f  %s%s",
            rank, r["config_name"], gf, loss, note_str, marker,
        )
    logger.info("  %s", "-" * 60)
    logger.info("  Spectral baseline:  %.4f", spectral_score)
    logger.info("  Random baseline:    %.4f", random_baseline)
    logger.info("  Greedy-mod baseline: 0.1278")

    # ------------------------------------------------------------------
    # Save best embedding
    # ------------------------------------------------------------------
    if best_coords is not None:
        np.save(emb_dir / "GFAE_best_153.npy", best_coords)
        with open(emb_dir / "GFAE_best_153_nodes.json", "w", encoding="utf-8") as fh:
            json.dump(best_nodes, fh)
        logger.info("")
        logger.info("Best config: %s (GF=%.4f)", best_config_name, best_score)
        logger.info("Saved best embedding to embeddings/GFAE_best_153.npy")

    # ------------------------------------------------------------------
    # Save full results
    # ------------------------------------------------------------------
    output = {
        "description": "GFAE Hyperparameter Search - Direction A Phase 2",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_configs": len(configs),
        "total_time_min": round(total_elapsed / 60, 1),
        "baselines": {
            "Spectral": spectral_score,
            "random": random_baseline,
            "greedy_modularity": 0.1278,
        },
        "best_config": best_config_name,
        "best_gf_score": round(best_score, 6),
        "best_beats_spectral": best_score > spectral_score,
        "best_beats_random": best_score > random_baseline,
        "results": results,
    }

    out_path = results_dir / "gfae_hyperparameter_search.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    logger.info("Saved search results to %s", out_path)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  GFAE HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 60)
    print(f"  Configs tested:   {len(configs)}")
    print(f"  Total time:       {total_elapsed / 60:.1f} min")
    print(f"  Best config:      {best_config_name}")
    print(f"  Best GF Score:    {best_score:.4f}")
    print(f"  Spectral:         {spectral_score:.4f}")
    print(f"  Beats Spectral:   {'YES ★' if best_score > spectral_score else 'no'}")
    print(f"  Beats random:     {'YES' if best_score > random_baseline else 'no'}")
    print(f"  Best embedding:   embeddings/GFAE_best_153.npy")
    print("=" * 60)


if __name__ == "__main__":
    main()