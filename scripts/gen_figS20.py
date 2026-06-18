"""
Generate FigS20: Coexpression Network G-F Curves
Purity vs radius r for all methods on coexpression network.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import os

# ── Okabe-Ito colorblind-safe palette ──────────────────────────────────────
OI_COLORS = [
    '#0072B2',  # blue
    '#E69F00',  # orange
    '#009E73',  # green
    '#D55E00',  # vermilion
    '#CC79A7',  # pink
    '#56B4E9',  # skyblue
    '#F0E442',  # yellow
    '#999999',  # gray
    '#984EA3',  # purple
    '#E41A1C',  # red
    '#000000',  # black
]

# ── Load data ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

data_path = os.path.join(RESULTS_DIR, 'coexpression_gf.json')
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

r_values = data['r_values']
gf_curves = data['gf_curves']
gf_scores = data['gf_scores']
random_baseline = data['random_baseline']['mean']
random_std = data['random_baseline']['std']

# ── Method ordering by GF score (descending) ──────────────────────────────
method_order = sorted(gf_scores.keys(), key=lambda m: gf_scores[m], reverse=True)
top_method = method_order[0]  # DeepWalk

# ── Create figure ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

# ── Plot each method's curve ──────────────────────────────────────────────
for idx, method in enumerate(method_order):
    if method not in gf_curves:
        continue
    purities = gf_curves[method]['purities']
    # Truncate to match r_values length
    n_pts = min(len(purities), len(r_values))
    r = r_values[:n_pts]
    p = purities[:n_pts]

    color = OI_COLORS[idx % len(OI_COLORS)]
    lw = 2.5 if method == top_method else 1.3
    alpha = 1.0 if method == top_method else 0.65
    ls = '-' if method == top_method else '-'
    zorder = 10 if method == top_method else 5

    gf_val = gf_scores[method]
    label = f'{method} (GF={gf_val:.3f})'

    ax.plot(r, p, color=color, linewidth=lw, alpha=alpha, linestyle=ls,
            label=label, zorder=zorder)

# ── Random baseline ───────────────────────────────────────────────────────
ax.axhline(y=random_baseline, color=OI_COLORS[-1], linestyle='--', linewidth=1.2,
           alpha=0.7, label=f'Random baseline (mean={random_baseline:.3f})')
ax.axhspan(random_baseline - random_std, random_baseline + random_std,
           color=OI_COLORS[-1], alpha=0.08, zorder=1)

# ── Highlight top performer ───────────────────────────────────────────────
# Add annotation arrow pointing to DeepWalk curve peak
if top_method in gf_curves:
    top_purities = gf_curves[top_method]['purities']
    peak_idx = np.argmax(top_purities[:len(r_values)])
    peak_r = r_values[peak_idx]
    peak_p = top_purities[peak_idx]
    ax.annotate(
        f'{top_method}\n(top performer)',
        xy=(peak_r, peak_p),
        xytext=(peak_r + 0.08, peak_p - 0.08),
        fontsize=9,
        fontweight='bold',
        color=OI_COLORS[0],
        arrowprops=dict(arrowstyle='->', color=OI_COLORS[0], lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor=OI_COLORS[0], alpha=0.9),
        zorder=20,
    )

# ── Formatting ────────────────────────────────────────────────────────────
ax.set_xlabel('Radius r', fontsize=12)
ax.set_ylabel('Purity P(r)', fontsize=12)
ax.set_title('FigS20: Coexpression Network G-F Curves', fontsize=13, fontweight='bold')
ax.set_xlim(r_values[0], r_values[-1])
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=8, loc='lower left', framealpha=0.9, ncol=1)

# Grid
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add network info annotation
ax.text(0.98, 0.98, f'Coexpression network\nn={data["network_statistics"]["nodes"]} nodes, '
        f'm={data["network_statistics"]["edges"]} edges',
        transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.8))

# ── Save ──────────────────────────────────────────────────────────────────
outpath = os.path.join(FIGURES_DIR, 'FigS20_coexpression_gf_curves.png')
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'[OK] Saved: {outpath}')
