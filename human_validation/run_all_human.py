#!/usr/bin/env python
"""
Run complete human interactome validation pipeline.

Convenience wrapper that calls the modern pipeline scripts in sequence:
1. human_embed_all.py  — generate all 6 embeddings
2. human_gf_all.py     — compute G-F curves, scores, plateau widths

Usage:
    python run_all_human.py

Output:
    - data/human_*_embedding.json       (embeddings)
    - results/human_gf_curves_200pts.pkl (G-F curves)
    - results/human_gf_scores.json      (scores and rankings)
    - results/human_plateau_widths.json  (plateau widths)
"""

import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a subprocess with timing and error reporting."""
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"  Command: {cmd}")
    print(f"{'=' * 70}")

    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: Command failed after {elapsed:.1f}s")
        return False

    print(f"  Completed in {elapsed:.1f}s")
    return True


def main():
    print("=" * 70)
    print("Human Interactome G-F Consistency Analysis")
    print("=" * 70)

    script_dir = Path(__file__).parent
    python = sys.executable
    total_start = time.time()

    # Step 1: Compute all embeddings
    if not run_command(f"{python} {script_dir / 'human_embed_all.py'}",
                       "Step 1/2: Computing embeddings for all 6 methods"):
        return 1

    # Step 2: Compute G-F curves, scores, and plateau widths
    if not run_command(f"{python} {script_dir / 'human_gf_all.py'}",
                       "Step 2/2: Computing G-F curves and scores"):
        return 1

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Human validation completed ({total_elapsed / 60:.1f} min)")
    print("=" * 70)
    print("\nOutput files:")
    print("  - data/human_*_embedding.json")
    print("  - results/human_gf_curves_200pts.pkl")
    print("  - results/human_gf_scores.json")
    print("  - results/human_plateau_widths.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
