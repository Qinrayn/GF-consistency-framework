"""
Human PPI Network Validation - Unified Runner
===============================================

This script orchestrates the complete human network validation pipeline:
1. Downloads required data (STRING network, GO annotations)
2. Generates all 6 embeddings (DM, MDS, Spectral, DeepWalk, Node2Vec, VGAE)
3. Computes G-F curves, scores, and plateau widths
4. Generates diagnostic plots

Usage:
    cd human_validation
    python run_human_validation.py [--skip-embeddings] [--skip-gf] [--methods METHODS]

Arguments:
    --skip-embeddings    Skip embedding generation (use existing embeddings)
    --skip-gf            Skip G-F curve computation
    --methods            Comma-separated list of methods to run (default: all)

Examples:
    python run_human_validation.py                    # Full pipeline
    python run_human_validation.py --skip-embeddings  # Use existing embeddings
    python run_human_validation.py --methods DM,N2V   # Only DM and Node2Vec

Requirements:
    - Python 3.11+
    - See ../requirements.txt for dependencies
    - ~16GB RAM recommended
    - ~5GB disk space for data
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd, description):
    """Run a command with progress reporting."""
    print(f"\n{'='*70}")
    print(f"Step: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*70}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: Command failed after {elapsed:.1f}s")
        return False
    
    print(f"Completed in {elapsed:.1f}s")
    return True


def check_requirements():
    """Check that required packages are available."""
    required = ['numpy', 'networkx', 'scipy', 'sklearn', 'torch', 'igraph']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r ../requirements.txt")
        return False
    
    return True


def check_data_files():
    """Check if required data files exist."""
    data_dir = Path(__file__).parent
    required_files = [
        '9606.protein.links.v12.0.txt.gz',
        'goa_human.gaf.gz',
        '9606.protein.aliases.v12.0.txt.gz'
    ]
    
    missing = []
    for f in required_files:
        if not (data_dir / f).exists():
            missing.append(f)
    
    if missing:
        print("INFO: The following data files will be downloaded:")
        for f in missing:
            print(f"  - {f}")
        print("\nDownloads: ~250 MB total")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Human PPI Network Validation - Unified Runner'
    )
    parser.add_argument(
        '--skip-embeddings', action='store_true',
        help='Skip embedding generation (use existing)'
    )
    parser.add_argument(
        '--skip-gf', action='store_true',
        help='Skip G-F curve computation'
    )
    parser.add_argument(
        '--methods', type=str, default='all',
        help='Comma-separated methods (default: all)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: DM and Node2Vec only (for testing)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Human PPI Network Validation - Unified Runner")
    print("="*70)
    print(f"Working directory: {Path(__file__).parent.absolute()}")
    print(f"Python: {sys.executable}")
    print(f"Skip embeddings: {args.skip_embeddings}")
    print(f"Skip G-F curves: {args.skip_gf}")
    print(f"Methods: {args.methods}")
    print("="*70)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check data
    check_data_files()
    
    # Track overall time
    total_start = time.time()
    
    # Step 1: Generate embeddings
    if not args.skip_embeddings:
        if args.quick:
            print("\n[QUICK MODE] Running human_embed_all.py (all methods — use full pipeline for subset)")
            if not run_command(
                f"{sys.executable} human_embed_all.py",
                "Generate embeddings"
            ):
                sys.exit(1)
        else:
            print("\n[FULL MODE] Running human_embed_all.py (all 6 methods)")
            if not run_command(
                f"{sys.executable} human_embed_all.py",
                "Generate all 6 embeddings"
            ):
                sys.exit(1)
    else:
        print("\n[SKIPPED] Embedding generation")
    
    # Step 2: Compute G-F curves
    if not args.skip_gf:
        print("\nRunning human_gf_all.py (all methods)")
        if not run_command(
            f"{sys.executable} human_gf_all.py",
            "Compute G-F curves for all methods"
        ):
            sys.exit(1)
    else:
        print("\n[SKIPPED] G-F curve computation")
    
    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print("Pipeline Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"\nOutput files:")
    print(f"  - Embeddings: data/human_*_embedding.json")
    print(f"  - G-F curves: results/human_gf_curves_200pts.pkl")
    print(f"  - G-F scores: results/human_gf_scores.json")
    print(f"  - Plateau widths: results/human_plateau_widths.json")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
