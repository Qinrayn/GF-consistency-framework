#!/usr/bin/env python3
"""
human_topological_analysis.py
==============================

Cross-species validation: Validate the topology-function duality on the **human** PPI
network.  Loads existing human embeddings (DM, MDS, Spectral, DeepWalk,
Node2Vec, VGAE), subsamples annotated nodes, computes persistent homology
via Ripser, and correlates topological metrics with human G-F Scores.

Outputs
-------
- ``results/human_topological_analysis.json``
- ``figures/Fig14_human_topo_scatter.png``

Usage
-----
::

    python human_validation/human_topological_analysis.py
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Ripser for persistent homology
try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("WARNING: ripser not installed. Install with: pip install ripser")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.utils import SEED

# ---- Configuration ----
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

METHODS = ['DM', 'MDS', 'Spectral', 'DeepWalk', 'Node2Vec', 'VGAE']
SUBSAMPLE_SIZE = 2000  # Match human GF Score subsample size
R_MIN_TOPO = 0.05
R_MAX_TOPO = 0.30      # Match human GF unified interval (~0.297)
N_R_POINTS = 100

METHOD_COLORS = {
    'DM': '#1f77b4', 'MDS': '#ff7f0e', 'Spectral': '#2ca02c',
    'DeepWalk': '#d62728', 'Node2Vec': '#9467bd', 'VGAE': '#8c564b',
}


# ---- Data loading ----

def load_human_go_annotations():
    go_file = os.path.join(DATA_DIR, 'human_go_annotations.json')
    if os.path.exists(go_file):
        with open(go_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_embedding(method):
    filepath = os.path.join(DATA_DIR, f'human_{method.lower()}_embedding.json')
    if not os.path.exists(filepath):
        print(f"  Warning: {filepath} not found, skipping {method}")
        return None, None
    with open(filepath, 'r', encoding='utf-8') as f:
        emb = json.load(f)
    nodes = list(emb.keys())
    coords = np.array([[emb[n]['x'], emb[n]['y']] for n in nodes])
    return nodes, coords


def subsample_annotated(nodes, coords, go_map, target_size, rng):
    """Select annotated nodes and subsample for tractable computation."""
    annotated_idx = [i for i, n in enumerate(nodes) if n in go_map and len(go_map[n]) > 0]
    if len(annotated_idx) <= target_size:
        sub_idx = annotated_idx
    else:
        sub_idx = sorted(rng.choice(annotated_idx, size=target_size, replace=False).tolist())

    sub_coords = coords[sub_idx]
    sub_nodes = [nodes[i] for i in sub_idx]
    return sub_coords, sub_nodes, len(annotated_idx)


# ---- Persistent homology ----

def compute_persistent_homology(coords, max_dim=1):
    """Compute persistent homology using Ripser."""
    if not HAS_RIPSER:
        return None
    result = ripser(coords, maxdim=max_dim)
    return result['dgms']


def compute_betti_curves(diagrams, r_vals):
    """Extract Betti numbers at each r from persistence diagrams."""
    betti_0 = np.zeros(len(r_vals), dtype=int)
    betti_1 = np.zeros(len(r_vals), dtype=int)

    # H0: connected components
    for birth, death in diagrams[0]:
        for j, r in enumerate(r_vals):
            if birth <= r < death:
                betti_0[j] += 1

    # H1: loops
    if len(diagrams) > 1:
        for birth, death in diagrams[1]:
            for j, r in enumerate(r_vals):
                if birth <= r < death:
                    betti_1[j] += 1

    return betti_0, betti_1


def compute_persistence_statistics(diagrams):
    """Compute summary statistics from persistence diagrams."""
    result = {}
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            result[f'H{dim}'] = {
                'n_features': 0,
                'mean_persistence': 0.0,
                'max_persistence': 0.0,
            }
            continue

        # Filter out infinite features
        finite = [(b, d) for b, d in dgm if np.isfinite(d)]
        if not finite:
            result[f'H{dim}'] = {
                'n_features': len(dgm),
                'mean_persistence': 0.0,
                'max_persistence': 0.0,
            }
            continue

        persistences = [d - b for b, d in finite]
        result[f'H{dim}'] = {
            'n_features': len(dgm),
            'n_finite': len(finite),
            'mean_persistence': float(np.mean(persistences)),
            'max_persistence': float(np.max(persistences)),
        }

    return result


# ---- Topological consistency ----

def compute_purity_at_r(coords, nodes, go_map, dist_matrix, r):
    """Compute functional purity at distance threshold r using Louvain."""
    import networkx as nx
    from scipy.spatial import cKDTree

    n = len(nodes)
    # Build spatial graph using KDTree (O(n log n) vs O(n^2))
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r)
    edges = list(pairs)

    if not edges:
        return 0.0

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    # Louvain community detection (no edge cap)
    try:
        partition = nx.community.louvain_communities(G, seed=SEED)
    except Exception:
        return 0.0

    # Compute purity (consistent with utils._community_purity)
    purities = []
    for community in partition:
        comm_list = list(community)
        if len(comm_list) < 2:
            continue

        # Count GO term frequencies
        all_terms = []
        for idx in comm_list:
            node = nodes[idx]
            if node in go_map and go_map[node]:
                all_terms.extend(go_map[node])

        if all_terms:
            term_counts = {}
            for term in all_terms:
                term_counts[term] = term_counts.get(term, 0) + 1
            max_count = max(term_counts.values())
            purities.append(max_count / len(all_terms))

    return float(sum(purities) / len(purities)) if purities else 0.0


def topological_consistency_score(diagrams, purities, r_vals):
    """
    Compute topological consistency: alignment between topological
    transitions (birth/death events) and functional purity gradient.

    Returns both combined (H0+H1) and H1-only scores.
    H1-only is more meaningful since H0 events are trivial merges.
    """
    dp_dr = np.gradient(purities, r_vals)

    def _score_from_events(event_r_values):
        if len(event_r_values) == 0:
            return 0.0
        total = sum(abs(dp_dr[np.argmin(np.abs(r_vals - ev_r))])
                    for ev_r in event_r_values)
        return float(total / len(event_r_values))

    # H0 events (connected components — trivial for any point cloud)
    h0_events = []
    for birth, death in diagrams[0]:
        if np.isfinite(death):
            h0_events.extend([birth, death])

    # H1 events (loops — topologically meaningful)
    h1_events = []
    if len(diagrams) > 1:
        for birth, death in diagrams[1]:
            if np.isfinite(death):
                h1_events.extend([birth, death])

    combined = _score_from_events(h0_events + h1_events)
    h1_only = _score_from_events(h1_events)
    return combined, h1_only


# ---- Main analysis ----

def analyze_method(method, coords, nodes, go_map, r_vals, rng):
    """Run full topological analysis for one embedding method."""
    print(f"\n  [{method}] Computing persistent homology...")
    t0 = time.time()
    diagrams = compute_persistent_homology(coords, max_dim=1)
    t_ph = time.time() - t0
    print(f"    PH computed in {t_ph:.1f}s")

    if diagrams is None:
        return None

    # Betti curves
    betti_0, betti_1 = compute_betti_curves(diagrams, r_vals)
    print(f"    Max b0={betti_0.max()}, Max b1={betti_1.max()}")

    # Persistence statistics
    pstats = compute_persistence_statistics(diagrams)

    # Purity curve (subsampled: ~25 r points for speed)
    print(f"  [{method}] Computing purity curve ({len(nodes)} nodes)...")
    t1 = time.time()

    step = max(1, len(r_vals) // 25)
    r_sub = r_vals[::step]
    purities = []
    for r in r_sub:
        p = compute_purity_at_r(coords, nodes, go_map, None, r)
        purities.append(p)
    purities = np.array(purities)
    t_purity = time.time() - t1
    print(f"    Purity computed in {t_purity:.1f}s ({len(r_sub)} r-values)")

    # Interpolate to full r_vals
    from scipy.interpolate import interp1d
    if len(r_sub) > 1 and np.std(purities) > 0:
        f_interp = interp1d(r_sub, purities, kind='linear',
                            fill_value='extrapolate')
        purities_full = f_interp(r_vals)
    else:
        purities_full = np.full_like(r_vals, purities.mean() if len(purities) > 0 else 0.0)

    # Topological consistency score (combined + H1-only)
    topo_cons_all, topo_cons_h1 = topological_consistency_score(
        diagrams, purities_full, r_vals
    )
    print(f"    Topological consistency (all) = {topo_cons_all:.4f}")
    print(f"    Topological consistency (H1)  = {topo_cons_h1:.4f}")
    print(f"    Purity range: [{purities.min():.4f}, {purities.max():.4f}], std={purities.std():.4f}")

    return {
        'diagrams': diagrams,
        'betti_0': betti_0.tolist(),
        'betti_1': betti_1.tolist(),
        'persistence_stats': pstats,
        'purities': purities_full.tolist(),
        'topo_consistency': topo_cons_all,
        'topo_consistency_h1': topo_cons_h1,
        'r_vals': r_vals.tolist(),
        'ph_time_seconds': t_ph,
    }


def main():
    print("=" * 65)
    print("  Human PPI Topological Validation")
    print("=" * 65)

    if not HAS_RIPSER:
        print("ERROR: ripser not installed. Run: pip install ripser")
        sys.exit(1)

    rng = np.random.default_rng(SEED)
    r_vals = np.linspace(R_MIN_TOPO, R_MAX_TOPO, N_R_POINTS)

    # Load GO annotations
    go_map = load_human_go_annotations()
    n_annotated = sum(1 for v in go_map.values() if len(v) > 0)
    print(f"\nLoaded GO annotations: {len(go_map)} proteins, {n_annotated} annotated")

    # Load human GF scores for comparison
    gf_path = os.path.join(RESULTS_DIR, 'human_gf_scores.json')
    human_gf = {}
    if os.path.exists(gf_path):
        with open(gf_path, 'r', encoding='utf-8') as f:
            human_gf = json.load(f).get('scores', {})

    # Analyze each method
    all_results = {}
    for method in METHODS:
        nodes, coords = load_embedding(method)
        if nodes is None:
            continue

        print(f"\n{'='*50}")
        print(f"  Method: {method}  ({len(nodes)} total nodes)")
        print(f"{'='*50}")

        # Subsample annotated nodes
        sub_coords, sub_nodes, n_annot = subsample_annotated(
            nodes, coords, go_map, SUBSAMPLE_SIZE, rng
        )
        print(f"  Subsampled {len(sub_nodes)} annotated nodes (from {n_annot} total)")

        if len(sub_nodes) < 50:
            print(f"  WARNING: Too few annotated nodes for {method}, skipping")
            continue

        # Rescale coordinates to [0, 1]
        mins = sub_coords.min(axis=0)
        maxs = sub_coords.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        sub_coords = (sub_coords - mins) / ranges

        result = analyze_method(method, sub_coords, sub_nodes, go_map, r_vals, rng)
        if result is not None:
            result['n_nodes'] = len(sub_nodes)
            result['gf_score'] = human_gf.get(method, None)
            all_results[method] = result

    # ---- Summary ----
    print("\n" + "=" * 65)
    print("  SUMMARY: Human PPI Topological Analysis (v2)")
    print("=" * 65)
    print(f"{'Method':12s} {'GF Score':>10s} {'Cons(all)':>10s} {'Cons(H1)':>10s} "
          f"{'H1 feat':>8s} {'H1 maxP':>10s} {'Max b1':>8s} {'Nodes':>6s}")
    print("-" * 78)

    methods_list = []
    gf_list = []
    tc_all_list = []
    tc_h1_list = []

    for method in METHODS:
        if method not in all_results:
            continue
        r = all_results[method]
        gf = r.get('gf_score', None)
        h1 = r['persistence_stats'].get('H1', {})
        b1_arr = np.array(r['betti_1'])
        print(f"{method:12s} {(gf or 0):10.4f} {r['topo_consistency']:10.4f} "
              f"{r['topo_consistency_h1']:10.4f} "
              f"{h1.get('n_features', 0):8d} {h1.get('max_persistence', 0):10.4f} "
              f"{b1_arr.max():8d} {r['n_nodes']:6d}")

        if gf is not None:
            methods_list.append(method)
            gf_list.append(gf)
            tc_all_list.append(r['topo_consistency'])
            tc_h1_list.append(r['topo_consistency_h1'])

    # Spearman correlations
    if len(gf_list) >= 4:
        rho_all, p_all = stats.spearmanr(gf_list, tc_all_list)
        rho_h1c, p_h1c = stats.spearmanr(gf_list, tc_h1_list)
        print(f"\n  Spearman rho(GF, TopoCons_all) = {rho_all:+.4f}  (p = {p_all:.4f})")
        print(f"  Spearman rho(GF, TopoCons_H1)  = {rho_h1c:+.4f}  (p = {p_h1c:.4f})")

        h1_maxp = [all_results[m]['persistence_stats'].get('H1', {}).get('max_persistence', 0)
                    for m in methods_list]
        if np.std(h1_maxp) > 1e-12:
            rho_h1, p_h1 = stats.spearmanr(gf_list, h1_maxp)
            print(f"  Spearman rho(GF, H1 maxP)      = {rho_h1:+.4f}  (p = {p_h1:.4f})")
        else:
            rho_h1, p_h1 = 0.0, 1.0
    else:
        rho_all, p_all = 0.0, 1.0
        rho_h1c, p_h1c = 0.0, 1.0
        rho_h1, p_h1 = 0.0, 1.0

    # ---- Save results ----
    output = {
        'methods': list(all_results.keys()),
        'subsample_size': SUBSAMPLE_SIZE,
        'r_range': [R_MIN_TOPO, R_MAX_TOPO],
        'spearman_gf_topo_cons_all': {'rho': float(rho_all), 'p': float(p_all)},
        'spearman_gf_topo_cons_h1': {'rho': float(rho_h1c), 'p': float(p_h1c)},
        'spearman_gf_h1_maxp': {'rho': float(rho_h1), 'p': float(p_h1)},
        'results': {}
    }

    for method, r in all_results.items():
        dgm_serializable = []
        for dgm in r['diagrams']:
            dgm_serializable.append(
                [[float(b), float(d)] for b, d in dgm if np.isfinite(d)]
            )
        output['results'][method] = {
            'n_nodes': r['n_nodes'],
            'gf_score': r['gf_score'],
            'topo_consistency': r['topo_consistency'],
            'topo_consistency_h1': r['topo_consistency_h1'],
            'persistence_stats': r['persistence_stats'],
            'betti_0': r['betti_0'],
            'betti_1': r['betti_1'],
            'purities': r['purities'],
            'diagrams': dgm_serializable,
            'ph_time_seconds': r['ph_time_seconds'],
        }

    out_path = os.path.join(RESULTS_DIR, 'human_topological_analysis.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ---- Generate figure ----
    generate_human_topo_figure(all_results, methods_list, gf_list,
                               tc_all_list, tc_h1_list,
                               rho_all, p_all, rho_h1c, p_h1c,
                               rho_h1, p_h1)

    print("\nDone!")


def generate_human_topo_figure(all_results, methods_list, gf_list,
                                tc_all_list, tc_h1_list,
                                rho_all, p_all, rho_h1c, p_h1c,
                                rho_h1, p_h1):
    """Generate Fig14: Human network topological validation scatter (3 panels)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Topo Consistency (all) vs GF Score
    ax = axes[0]
    for m, gf, tc in zip(methods_list, gf_list, tc_all_list):
        color = METHOD_COLORS.get(m, '#333333')
        ax.scatter(gf, tc, c=color, s=120, edgecolors='black',
                   linewidths=0.8, zorder=5)
        ax.annotate(m, (gf, tc), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, fontweight='medium', color=color)
    if len(gf_list) >= 3:
        slope, intercept, _, _, _ = stats.linregress(gf_list, tc_all_list)
        x_line = np.linspace(min(gf_list) * 0.9, max(gf_list) * 1.1, 50)
        ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.3, linewidth=1.5)
    ax.set_xlabel('Human G-F Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Topological Consistency (all)', fontsize=12, fontweight='bold')
    ax.set_title(f'(A) Topo Consistency (all) vs GF\n'
                 f'rho={rho_all:.3f}, p={p_all:.3f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Topo Consistency (H1-only) vs GF Score
    ax = axes[1]
    for m, gf, tc in zip(methods_list, gf_list, tc_h1_list):
        color = METHOD_COLORS.get(m, '#333333')
        ax.scatter(gf, tc, c=color, s=120, edgecolors='black',
                   linewidths=0.8, zorder=5)
        ax.annotate(m, (gf, tc), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, fontweight='medium', color=color)
    if len(gf_list) >= 3 and np.std(tc_h1_list) > 1e-12:
        slope, intercept, _, _, _ = stats.linregress(gf_list, tc_h1_list)
        x_line = np.linspace(min(gf_list) * 0.9, max(gf_list) * 1.1, 50)
        ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.3, linewidth=1.5)
    ax.set_xlabel('Human G-F Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Topological Consistency (H1 only)', fontsize=12, fontweight='bold')
    ax.set_title(f'(B) Topo Consistency (H1) vs GF\n'
                 f'rho={rho_h1c:.3f}, p={p_h1c:.3f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: H1 Max Persistence vs GF Score
    ax = axes[2]
    h1_maxp = [all_results[m]['persistence_stats'].get('H1', {}).get('max_persistence', 0)
               for m in methods_list]
    if len(gf_list) >= 3 and np.std(h1_maxp) > 1e-12:
        for m, gf, mp in zip(methods_list, gf_list, h1_maxp):
            color = METHOD_COLORS.get(m, '#333333')
            ax.scatter(mp, gf, c=color, s=120, edgecolors='black',
                       linewidths=0.8, zorder=5)
            ax.annotate(m, (mp, gf), textcoords="offset points",
                        xytext=(8, 5), fontsize=9, fontweight='medium',
                        color=color)

        slope, intercept, r_val, _, _ = stats.linregress(h1_maxp, gf_list)
        x_line = np.linspace(min(h1_maxp) * 0.9, max(h1_maxp) * 1.1, 50)
        ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.3, linewidth=1.5)

    ax.set_xlabel('H1 Max Persistence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Human G-F Score', fontsize=12, fontweight='bold')
    ax.set_title(f'(C) H1 Max Persistence vs GF\n'
                 f'rho={rho_h1:.3f}, p={p_h1:.3f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle(f'Human PPI Network: Topological Validation ({SUBSAMPLE_SIZE} nodes)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'Fig14_human_topo_scatter.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"\nSaved: {fig_path}")


if __name__ == '__main__':
    main()
