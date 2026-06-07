#!/usr/bin/env python3
"""
topological_stats.py
====================

Statistical analysis: Spearman correlations between topological
metrics and standard G-F Scores, plus the key scatter plot showing the
topology-function duality.

Outputs
-------
- ``figures/Fig11_topo_consistency_vs_gf_score.png``  (the "money plot")
- ``figures/Fig12_topo_metric_correlations.png``      (multi-panel correlations)
- ``results/topological_correlation_analysis.json``

Usage
-----
::

    python scripts/topological_stats.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, R_MIN, R_MAX, N_POINTS,
    ALL_METHODS,
    get_results_dir, get_figures_dir,
)

# ── colour palette (consistent with rest of project) ──────────────────
METHOD_COLORS = {
    'DM': '#1f77b4', 'MDS': '#ff7f0e', 'Spectral': '#2ca02c',
    'DeepWalk': '#d62728', 'Node2Vec': '#9467bd', 'VGAE': '#8c564b',
    'PCA': '#e377c2', 'VGAE-feat': '#7f7f7f',
    'GraphSAGE': '#bcbd22', 'GAT': '#17becf', 'GIN': '#aec7e8',
}

METHOD_FAMILY = {
    'DM': 'Geometric', 'MDS': 'Geometric', 'Spectral': 'Geometric',
    'PCA': 'Geometric',
    'DeepWalk': 'Random-walk', 'Node2Vec': 'Random-walk',
    'VGAE': 'Deep Learning', 'VGAE-feat': 'Deep Learning',
    'GraphSAGE': 'GNN', 'GAT': 'GNN', 'GIN': 'GNN',
}

FAMILY_SHAPES = {
    'Geometric': 'o', 'Random-walk': 's',
    'Deep Learning': '^', 'GNN': 'D',
}

# ── helpers ────────────────────────────────────────────────────────────

def load_all_data():
    """Load and merge all relevant data sources."""
    results_dir = get_results_dir()

    # 1. Standard GF scores
    gf_data = json.load(open(results_dir / 'gf_scores.json'))
    gf_scores = dict(gf_data['scores'])

    # 2. GNN GF scores
    gnn_data = json.load(open(results_dir / 'gnn_gf_scores.json'))
    gf_scores.update(gnn_data['gf_scores'])

    # 3. Topological analysis
    topo_data = json.load(open(results_dir / 'topological_analysis.json'))

    return gf_scores, topo_data


def compute_topo_gf_scores(topo_data):
    """Compute topological G-F score = integral of topo_purity over [R_MIN, R_MAX]."""
    r_vals = np.array(topo_data['r_vals'])
    topo_gf = {}
    for method in topo_data['methods']:
        curve = topo_data['topo_gf_curves'][method]
        topo_purities = np.array(curve['topo_purities'])
        std_purities = np.array(curve['standard_purities'])

        # Restrict to [R_MIN, R_MAX]
        mask = (r_vals >= R_MIN) & (r_vals <= R_MAX)
        r_sub = r_vals[mask]
        tp_sub = topo_purities[mask]
        sp_sub = std_purities[mask]

        if len(r_sub) > 1:
            topo_gf[method] = float(np.trapezoid(tp_sub, r_sub))
            std_gf_recomputed = float(np.trapezoid(sp_sub, r_sub))
        else:
            topo_gf[method] = 0.0
            std_gf_recomputed = 0.0

        # Also store the recomputed standard GF for cross-check
        topo_gf.setdefault('_std_recomputed', {})[method] = std_gf_recomputed

    return topo_gf


def build_correlation_table(gf_scores, topo_data, topo_gf):
    """Build a flat table of all metrics for each method."""
    methods = topo_data['methods']
    cons = topo_data['consistency_scores']
    pers = topo_data['persistence_statistics']

    rows = []
    for m in methods:
        row = {
            'method': m,
            'family': METHOD_FAMILY[m],
            'gf_score': gf_scores.get(m, None),
            'topo_gf_score': topo_gf.get(m, None),
            'topo_consistency': cons.get(m, None),
            'h1_features': pers[m]['1']['n_features'],
            'h1_max_persistence': pers[m]['1']['max_persistence'],
            'h1_mean_persistence': pers[m]['1']['mean_persistence'],
            'h1_entropy': pers[m]['1']['persistence_entropy'],
            'h1_complexity': pers[m]['1']['topological_complexity'],
            'h0_features': pers[m]['0']['n_features'],
            'h0_max_persistence': pers[m]['0']['max_persistence'],
        }
        rows.append(row)

    return rows


def spearman_analysis(table):
    """Compute Spearman correlations between GF score and all topo metrics."""
    # Filter rows with valid GF scores
    valid = [r for r in table if r['gf_score'] is not None]
    if len(valid) < 4:
        print("WARNING: Too few methods with valid GF scores for correlation analysis")
        return {}

    gf = np.array([r['gf_score'] for r in valid])
    metrics = [
        ('topo_consistency', 'Topological Consistency'),
        ('topo_gf_score', 'Topological GF Score'),
        ('h1_features', 'H1 Features (count)'),
        ('h1_max_persistence', 'H1 Max Persistence'),
        ('h1_mean_persistence', 'H1 Mean Persistence'),
        ('h1_entropy', 'H1 Persistence Entropy'),
        ('h1_complexity', 'H1 Topological Complexity'),
        ('h0_features', 'H0 Features (count)'),
        ('h0_max_persistence', 'H0 Max Persistence'),
    ]

    results = {}
    for key, label in metrics:
        vals = np.array([r[key] for r in valid])
        # Skip if all same value
        if np.std(vals) < 1e-12:
            continue
        rho, pval = stats.spearmanr(gf, vals)
        results[key] = {
            'label': label,
            'spearman_rho': float(rho),
            'p_value': float(pval),
            'significant_005': bool(pval < 0.05),
            'n': len(valid),
        }
        print(f"  {label:35s}: rho = {rho:+.4f}  (p = {pval:.4f})  "
              f"{'***' if pval < 0.01 else '**' if pval < 0.05 else ''}")

    return results


# ── plotting ───────────────────────────────────────────────────────────

def plot_money_plot(table, save_path):
    """
    Fig11: Topological Consistency vs G-F Score scatter plot.
    The key figure showing the inverse correlation / topology-function duality.
    """
    valid = [r for r in table if r['gf_score'] is not None]
    gf = np.array([r['gf_score'] for r in valid])
    tc = np.array([r['topo_consistency'] for r in valid])

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Regression line
    slope, intercept, r_val, p_val, std_err = stats.linregress(gf, tc)
    x_line = np.linspace(gf.min() * 0.9, gf.max() * 1.1, 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.4, linewidth=1.5,
            label=f'Linear fit (R={r_val:.3f}, p={p_val:.3f})')

    # Spearman
    rho, sp_p = stats.spearmanr(gf, tc)

    # Scatter by family
    families_plotted = set()
    for r in valid:
        fam = r['family']
        marker = FAMILY_SHAPES.get(fam, 'o')
        color = METHOD_COLORS.get(r['method'], '#333333')
        label_fam = fam if fam not in families_plotted else None
        families_plotted.add(fam)
        ax.scatter(r['gf_score'], r['topo_consistency'],
                   c=color, marker=marker, s=120, edgecolors='black',
                   linewidths=0.8, zorder=5, label=label_fam)

    # Annotate each point with method name
    for r in valid:
        ax.annotate(r['method'],
                    (r['gf_score'], r['topo_consistency']),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=8.5, fontweight='medium',
                    color=METHOD_COLORS.get(r['method'], '#333333'))

    ax.set_xlabel('G-F Score (standard)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Topological Consistency Score', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Topological Consistency vs G-F Score\n'
        f'Spearman $\\rho$ = {rho:.3f}  (p = {sp_p:.3f})',
        fontsize=14, fontweight='bold', pad=12
    )

    # Legend for families
    from matplotlib.lines import Line2D
    legend_elements = []
    for fam, mk in FAMILY_SHAPES.items():
        legend_elements.append(
            Line2D([0], [0], marker=mk, color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=9, label=fam)
        )
    # Also add the regression line to legend
    legend_elements.append(
        Line2D([0], [0], color='k', linestyle='--', alpha=0.4, label=f'Linear fit (R={r_val:.3f})')
    )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              framealpha=0.9, edgecolor='gray')

    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_multi_correlation(table, corr_results, save_path):
    """
    Fig12: Multi-panel scatter showing correlations of GF Score with
    several topological metrics.
    """
    valid = [r for r in table if r['gf_score'] is not None]
    gf = np.array([r['gf_score'] for r in valid])

    panels = [
        ('h1_features', 'H1 Features (count)'),
        ('h1_max_persistence', 'H1 Max Persistence'),
        ('h1_entropy', 'H1 Persistence Entropy'),
        ('topo_gf_score', 'Topological GF Score'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(panels):
        ax = axes[idx]
        vals = np.array([r[key] for r in valid])

        # Colour by family
        for r in valid:
            color = METHOD_COLORS.get(r['method'], '#333333')
            ax.scatter(r[key], r['gf_score'], c=color, s=80,
                       edgecolors='black', linewidths=0.6, zorder=5)

        # Annotate
        for r in valid:
            ax.annotate(r['method'],
                        (r[key], r['gf_score']),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7.5,
                        color=METHOD_COLORS.get(r['method'], '#333333'))

        # Stats
        rho, pval = stats.spearmanr(gf, vals)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('G-F Score', fontsize=11)
        ax.set_title(f'{label}\n$\\rho$={rho:.3f}, p={pval:.3f}', fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Trend line
        slope, intercept, r_val, _, _ = stats.linregress(vals, gf)
        x_line = np.linspace(vals.min(), vals.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.3, linewidth=1.2)

    plt.suptitle('Correlations Between Topological Metrics and G-F Score',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_topo_gf_comparison(table, save_path):
    """
    Bonus figure: Grouped bar chart comparing standard GF vs topological GF
    for all 11 methods.
    """
    valid = sorted(
        [r for r in table if r['gf_score'] is not None],
        key=lambda x: x['gf_score'], reverse=True
    )

    methods = [r['method'] for r in valid]
    gf_std = [r['gf_score'] for r in valid]
    gf_topo = [r['topo_gf_score'] for r in valid]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width / 2, gf_std, width, label='Standard GF Score',
                   color='steelblue', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, gf_topo, width, label='Topological GF Score',
                   color='coral', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Embedding Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Standard vs Topological G-F Score Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── JSON helpers ───────────────────────────────────────────────────────

def _json_default(obj):
    """Handle numpy types during JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Topological-Functional Correlation Analysis")
    print("=" * 65)

    # Load data
    gf_scores, topo_data = load_all_data()
    print(f"\nLoaded GF scores for {len(gf_scores)} methods")
    print(f"Loaded topological data for {len(topo_data['methods'])} methods")

    # Compute topological GF scores
    topo_gf = compute_topo_gf_scores(topo_data)
    print(f"Computed topological GF scores for {len([k for k in topo_gf if k != '_std_recomputed'])} methods")

    # Build correlation table
    table = build_correlation_table(gf_scores, topo_data, topo_gf)

    # Print summary table
    print(f"\n{'Method':12s} {'GF Score':>10s} {'Topo GF':>10s} {'Topo Cons':>10s} "
          f"{'H1 feat':>8s} {'H1 maxP':>10s} {'Family':>15s}")
    print("-" * 80)
    for r in sorted(table, key=lambda x: x['gf_score'] or 0, reverse=True):
        print(f"{r['method']:12s} {(r['gf_score'] or 0):10.4f} "
              f"{(r['topo_gf_score'] or 0):10.4f} "
              f"{(r['topo_consistency'] or 0):10.4f} "
              f"{r['h1_features']:8d} "
              f"{r['h1_max_persistence']:10.4f} "
              f"{r['family']:>15s}")

    # Spearman analysis
    print(f"\n--- Spearman Correlation Analysis (n={len([r for r in table if r['gf_score'] is not None])} methods) ---")
    corr_results = spearman_analysis(table)

    # Save results
    results_dir = get_results_dir()
    output = {
        'methods': [r['method'] for r in table],
        'correlation_table': table,
        'spearman_results': corr_results,
        'note': 'Topological correlation analysis: topology-function duality in PPI embeddings',
    }
    out_path = results_dir / 'topological_correlation_analysis.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\nSaved: {out_path}")

    # Generate figures
    figures_dir = get_figures_dir()

    print("\n--- Generating Figures ---")
    plot_money_plot(
        table,
        figures_dir / 'Fig11_topo_consistency_vs_gf_score.png'
    )
    plot_multi_correlation(
        table, corr_results,
        figures_dir / 'Fig12_topo_metric_correlations.png'
    )
    plot_topo_gf_comparison(
        table,
        figures_dir / 'Fig13_topo_vs_standard_gf_bars.png'
    )

    # Key findings summary
    print("\n" + "=" * 65)
    print("  KEY FINDINGS")
    print("=" * 65)
    if 'topo_consistency' in corr_results:
        tc = corr_results['topo_consistency']
        direction = "negative" if tc['spearman_rho'] < 0 else "positive"
        sig = "SIGNIFICANT" if tc['significant_005'] else "not significant"
        print(f"  Topological Consistency vs GF Score: {direction} correlation")
        print(f"    Spearman rho = {tc['spearman_rho']:.4f}, p = {tc['p_value']:.4f} ({sig})")
        if tc['spearman_rho'] < -0.5:
            print("    -> Strong inverse relationship: methods with high functional")
            print("       purity tend to have LOW topological consistency, suggesting")
            print("       a topology-function DUALITY in PPI embeddings.")
        elif tc['spearman_rho'] < 0:
            print("    -> Moderate inverse relationship between topological")
            print("       consistency and functional purity.")

    print(f"\n  All figures saved to: {figures_dir}")
    print("  Done!")


if __name__ == '__main__':
    main()
