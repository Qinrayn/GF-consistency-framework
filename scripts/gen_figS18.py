"""
Generate FigS18: GAT Collapse Theorem Verification on Full 5936-Node Network
3-panel figure with Okabe-Ito palette, 300 DPI, ASCII-safe text.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np
import os

# ── Okabe-Ito colorblind-safe palette ──────────────────────────────────────
OI = {
    'black':   '#000000',
    'orange':  '#E69F00',
    'skyblue': '#56B4E9',
    'green':   '#009E73',
    'yellow':  '#F0E442',
    'blue':    '#0072B2',
    'vermilion': '#D55E00',
    'pink':    '#CC79A7',
    'gray':    '#999999',
    'red':     '#E41A1C',
    'purple':  '#984EA3',
}

# ── Load data ──────────────────────────────────────────────────────────────
RESULTS_DIR = r'C:\Users\云丘\GF-consistency-framework\results'
FIGURES_DIR = r'C:\Users\云丘\GF-consistency-framework\figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

data_path = os.path.join(RESULTS_DIR, 'gat_theorem_large_network.json')
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ── Extract Panel A data (T1: Attention Degeneration Bound) ────────────────
h_norm_153 = data['comparison_153_vs_full']['t1_153']['H_norm_mean']
h_norm_full = data['T1_attention_degeneration']['empirical']['H_norm_mean']
h_norm_std_full = data['T1_attention_degeneration']['empirical']['H_norm_std']

# ── Extract Panel B data (T2: Effective Rank Bound) ────────────────────────
method_results = data['T2_effective_rank_bound']['method_results']
gnn_methods = ['VGAE', 'VGAE-feat', 'GraphSAGE', 'GAT', 'GIN']
non_gnn_methods = ['DM', 'MDS', 'Spectral', 'DeepWalk', 'Node2Vec', 'PCA']

gnn_eff_ranks = [method_results[m]['effective_rank'] for m in gnn_methods]
non_gnn_eff_ranks = [method_results[m]['effective_rank'] for m in non_gnn_methods]
gnn_mean = data['T2_effective_rank_bound']['verification']['gnn_mean_eff_rank']
non_gnn_mean = data['T2_effective_rank_bound']['verification']['non_gnn_mean_eff_rank']

# ── Extract Panel C data (T3: GF Score Upper Bound) ───────────────────────
t3_results = data['T3_gf_upper_bound']['method_results']
all_methods_t3 = list(t3_results.keys())
gf_ratios = [t3_results[m]['gf_ratio'] for m in all_methods_t3]
eff_ranks_t3 = [t3_results[m]['effective_rank'] for m in all_methods_t3]
rho_t3 = data['T3_gf_upper_bound']['verification']['rho_gf_ratio_vs_eff_rank']
p_t3 = data['T3_gf_upper_bound']['verification']['p_gf_ratio_vs_eff_rank']

# ── Create figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.subplots_adjust(wspace=0.35)

# ── Panel A: Theorem 1 - Attention Degeneration Bound ─────────────────────
ax = axes[0]
bars = ax.bar(
    ['153 nodes', '5936 nodes'],
    [h_norm_153, h_norm_full],
    color=[OI['skyblue'], OI['blue']],
    edgecolor='black',
    linewidth=0.8,
    width=0.5,
)
# Error bar on full network
ax.errorbar(1, h_norm_full, yerr=h_norm_std_full, fmt='none',
            ecolor='black', capsize=5, capthick=1.5, linewidth=1.5)
# Reference line at H_norm = 1.0
ax.axhline(y=1.0, color=OI['vermilion'], linestyle='--', linewidth=1.2, label='Bound (H_norm >= 1.0)')
ax.set_ylabel('Normalized Attention Entropy (H_norm)', fontsize=10)
ax.set_title('(A) Theorem 1: Attention Degeneration Bound', fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.8)
ax.legend(fontsize=8, loc='upper right')
# Value labels
for bar, val in zip(bars, [h_norm_153, h_norm_full]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Panel B: Theorem 2 - Effective Rank Bound ─────────────────────────────
ax = axes[1]
all_methods_b = non_gnn_methods + gnn_methods
all_eff_ranks = non_gnn_eff_ranks + gnn_eff_ranks
colors_b = [OI['orange']] * len(non_gnn_methods) + [OI['green']] * len(gnn_methods)

x_pos = np.arange(len(all_methods_b))
bars = ax.bar(x_pos, all_eff_ranks, color=colors_b, edgecolor='black', linewidth=0.5, width=0.7)

# Mean lines
ax.axhline(y=gnn_mean, color=OI['green'], linestyle='--', linewidth=1.2, alpha=0.8)
ax.axhline(y=non_gnn_mean, color=OI['orange'], linestyle='--', linewidth=1.2, alpha=0.8)

# Mean labels
ax.text(len(all_methods_b) - 0.5, non_gnn_mean + 0.04,
        f'non-GNN mean={non_gnn_mean:.3f}', fontsize=7, color=OI['orange'], ha='right')
ax.text(len(all_methods_b) - 0.5, gnn_mean + 0.04,
        f'GNN mean={gnn_mean:.3f}', fontsize=7, color=OI['green'], ha='right')

ax.set_xticks(x_pos)
ax.set_xticklabels(all_methods_b, rotation=45, ha='right', fontsize=7.5)
ax.set_ylabel('Effective Rank', fontsize=10)
ax.set_title('(B) Theorem 2: Effective Rank Bound', fontsize=11, fontweight='bold')
ax.set_ylim(0, 2.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend patches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=OI['orange'], edgecolor='black', label='Non-GNN'),
                   Patch(facecolor=OI['green'], edgecolor='black', label='GNN')]
ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

# ── Panel C: Theorem 3 - GF Score Upper Bound ─────────────────────────────
ax = axes[2]

# Color points by whether GNN or not
gnn_set = set(gnn_methods)
for i, m in enumerate(all_methods_t3):
    c = OI['green'] if m in gnn_set else OI['orange']
    marker = 's' if m in gnn_set else 'o'
    ax.scatter(eff_ranks_t3[i], gf_ratios[i], c=c, marker=marker, s=70,
               edgecolors='black', linewidths=0.5, zorder=3)
    # Label
    offset_x = 0.03
    offset_y = 0.03
    ax.annotate(m, (eff_ranks_t3[i], gf_ratios[i]),
                xytext=(eff_ranks_t3[i] + offset_x, gf_ratios[i] + offset_y),
                fontsize=7, ha='left')

# Reference line at ratio = 1
ax.axhline(y=1.0, color=OI['gray'], linestyle=':', linewidth=1.0, alpha=0.7, label='Ratio = 1 (2D = 1D)')

ax.set_xlabel('Effective Rank', fontsize=10)
ax.set_ylabel('GF_2D / GF_1D Ratio', fontsize=10)
ax.set_title('(C) Theorem 3: GF Score Upper Bound', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotation
ax.text(0.05, 0.95, f'rho={rho_t3:.3f} (p={p_t3:.3f})',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

# ── Save ───────────────────────────────────────────────────────────────────
outpath = os.path.join(FIGURES_DIR, 'FigS18_gat_theorem_full_network.png')
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'[OK] Saved: {outpath}')
