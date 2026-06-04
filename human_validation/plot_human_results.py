#!/usr/bin/env python
"""
Generate comparative figures for human PPI network validation.

This script creates:
1. G-F curve comparison across all 6 methods
2. Outlier diagnosis visualization (before/after cleanup)
3. G-F Score bar chart
4. Coordinate distribution plots

Usage:
    python plot_human_results.py
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

def load_gf_curves():
    """Load G-F curve results for all methods."""
    # Try the main results directory first, then human_validation/results
    results_dir = project_root / "results"
    alt_results_dir = project_root / "human_validation" / "results"

    methods = ["dm", "mds", "spectral", "deepwalk", "node2vec", "vgae"]
    method_names = {
        "dm": "Diffusion Map",
        "mds": "MDS",
        "spectral": "Spectral",
        "deepwalk": "DeepWalk",
        "node2vec": "Node2Vec",
        "vgae": "VGAE"
    }
    
    curves = {}
    for method in methods:
        # Search in both possible result directories
        for rdir in (results_dir, alt_results_dir):
            filepath = rdir / f"human_gf_curves_{method}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    curves[method] = {
                        'r': data.get('r_values', []),
                        'purity': data.get('purity', []),
                        'modularity': data.get('modularity', []),
                        'gf_score': data.get('gf_score', None)
                    }
                break
    
    return curves, method_names

def plot_gf_comparison(curves, method_names, save_path=None):
    """Plot G-F purity curves for all methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        "dm": "#1f77b4",
        "mds": "#ff7f0e",
        "spectral": "#2ca02c",
        "deepwalk": "#9467bd",
        "node2vec": "#d62728",
        "vgae": "#e377c2"
    }
    
    # Panel A: Purity curves
    ax = axes[0]
    for method, data in curves.items():
        if data['r'] and data['purity']:
            ax.plot(data['r'], data['purity'], 
                   label=method_names[method], 
                   color=colors.get(method, 'gray'),
                   linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Distance Threshold (r)', fontsize=12)
    ax.set_ylabel('Functional Purity', fontsize=12)
    ax.set_title('A. G-F Purity Curves - Human PPI Network', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0.2, 1.0)
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    
    # Panel B: Modularity curves
    ax = axes[1]
    for method, data in curves.items():
        if data['r'] and data.get('modularity'):
            ax.plot(data['r'], data['modularity'], 
                   label=method_names[method], 
                   color=colors.get(method, 'gray'),
                   linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Distance Threshold (r)', fontsize=12)
    ax.set_ylabel('Modularity (Q)', fontsize=12)
    ax.set_title('B. Modularity Curves - Human PPI Network', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.1, 0.8)
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Q > 0.3 threshold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_gf_score_bar(curves, method_names, save_path=None):
    """Plot G-F Score bar chart."""
    scores = []
    labels = []
    
    for method, data in curves.items():
        if data['gf_score'] is not None:
            scores.append(data['gf_score'])
            labels.append(method_names[method])
    
    # Sort by score
    sorted_idx = np.argsort(scores)[::-1]
    scores = [scores[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(scores))]
    bars = ax.bar(range(len(scores)), scores, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Embedding Method', fontsize=12)
    ax.set_ylabel('G-F Score', fontsize=12)
    ax.set_title('G-F Score Comparison - Human PPI Network', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, max(scores) * 1.2)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_outlier_diagnosis(save_path=None):
    """Plot outlier diagnosis visualization (placeholder)."""
    # This would require loading coordinate data
    # For now, create a placeholder figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.text(0.5, 0.5, 
            'Outlier diagnosis requires\ncoordinate data loading.\n\nSee Section 3.6 and\nSupplementary Figure S2\nfor details.',
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Outlier Diagnosis Visualization', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Main function to generate all figures."""
    print("Generating human validation figures...")
    
    figures_dir = project_root / "human_validation" / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Load data
    curves, method_names = load_gf_curves()
    
    if not curves:
        print(" No G-F curve data found. Run human_gf_all.py first.")
        return 1
    
    print(f" Loaded G-F curves for {len(curves)} methods")
    
    # Generate figures
    plot_gf_comparison(
        curves, method_names,
        save_path=figures_dir / "human_gf_comparison.png"
    )
    
    plot_gf_score_bar(
        curves, method_names,
        save_path=figures_dir / "human_gf_score_bar.png"
    )
    
    plot_outlier_diagnosis(
        save_path=figures_dir / "human_outlier_diagnosis.png"
    )
    
    print(f"\n All figures saved to: {figures_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
