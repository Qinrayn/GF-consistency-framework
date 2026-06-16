"""
Generate FigS19: Community Detection Ablation Heatmap
11 methods x 5 community algorithms, with rank annotations and Kendall's W.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
}

# ── Load data ──────────────────────────────────────────────────────────────
RESULTS_DIR = r'C:\Users\云丘\GF-consistency-framework\results'
FIGURES_DIR = r'C:\Users\云丘\GF-consistency-framework\figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

data_path = os.path.join(RESULTS_DIR, 'gf_ablation_community_detection.json')
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ── Build heatmap matrix ──────────────────────────────────────────────────
community_methods = ['greedy_modularity', 'label_propagation', 'connected_components', 'louvain', 'leiden']
method_order = ['Spectral', 'DM', 'Node2Vec', 'MDS', 'PCA', 'DeepWalk', 'VGAE-feat', 'GIN', 'GAT', 'GraphSAGE', 'VGAE']

gf_matrix = np.zeros((len(method_order), len(community_methods)))
rank_matrix = np.zeros((len(method_order), len(community_methods)), dtype=int)

gf_by_community = data['gf_scores_by_community']
rankings = data['rank_consistency']['rankings']

for j, comm in enumerate(community_methods):
    for i, method in enumerate(method_order):
        gf_matrix[i, j] = gf_by_community[comm][method]
        rank_matrix[i, j] = rankings[comm][method]

kendall_w = data['rank_consistency']['kendall_W']

# ── Create figure ──────────────────────────────────────────────────────────
fig_width = 10
fig_height = 7
fig = plt.figure(figsize=(fig_width, fig_height))

# Main heatmap axes + sidebar for Kendall's W
gs = fig.add_gridspec(1, 2, width_ratios=[5, 0.8], wspace=0.05)
ax_heat = fig.add_subplot(gs[0, 0])
ax_side = fig.add_subplot(gs[0, 1])

# ── Plot heatmap ──────────────────────────────────────────────────────────
# Use a perceptually uniform colormap
im = ax_heat.imshow(gf_matrix, cmap='YlOrRd', aspect='auto', vmin=0.04, vmax=0.20)

# Add rank annotations in each cell
for i in range(len(method_order)):
    for j in range(len(community_methods)):
        val = gf_matrix[i, j]
        rank = rank_matrix[i, j]
        # Choose text color based on background brightness
        text_color = 'white' if val > 0.13 else 'black'
        # GF score (main value)
        ax_heat.text(j, i - 0.15, f'{val:.3f}', ha='center', va='center',
                     fontsize=7, color=text_color, fontweight='bold')
        # Rank annotation
        ax_heat.text(j, i + 0.2, f'#{rank}', ha='center', va='center',
                     fontsize=6.5, color=text_color, alpha=0.75, style='italic')

# Labels
community_labels = ['Greedy\nModularity', 'Label\nPropagation', 'Connected\nComponents', 'Louvain', 'Leiden']
ax_heat.set_xticks(range(len(community_methods)))
ax_heat.set_xticklabels(community_labels, fontsize=9)
ax_heat.set_yticks(range(len(method_order)))
ax_heat.set_yticklabels(method_order, fontsize=9)
ax_heat.set_title('(A) G-F Score by Community Detection Algorithm', fontsize=12, fontweight='bold', pad=12)

# Colorbar below
cbar = fig.colorbar(im, ax=ax_heat, orientation='horizontal', pad=0.12, shrink=0.9)
cbar.set_label('G-F Score', fontsize=10)
cbar.ax.tick_params(labelsize=8)

# ── Sidebar: Kendall's W ──────────────────────────────────────────────────
ax_side.set_xlim(0, 1)
ax_side.set_ylim(0, 1)
ax_side.axis('off')

# Draw Kendall's W as a vertical gauge
bar_x = 0.5
bar_width = 0.35
# Background bar
ax_side.barh(0.5, 1.0, height=bar_width, left=0, color='#EEEEEE', edgecolor='gray', linewidth=0.5)
# Filled portion
ax_side.barh(0.5, kendall_w, height=bar_width, left=0, color=OI['blue'], edgecolor='black', linewidth=0.8)
# Value label
ax_side.text(bar_x, 0.5, f'{kendall_w:.3f}', ha='center', va='center',
             fontsize=14, fontweight='bold', color='white')
ax_side.text(bar_x, 0.75, "Kendall's W", ha='center', va='center', fontsize=9, fontweight='bold')
ax_side.text(bar_x, 0.28, '(Rank agreement)', ha='center', va='center', fontsize=7.5, color=OI['gray'])

# Interpretation
if kendall_w >= 0.7:
    interp = 'Strong\nagreement'
elif kendall_w >= 0.5:
    interp = 'Moderate\nagreement'
else:
    interp = 'Weak\nagreement'
ax_side.text(bar_x, 0.10, interp, ha='center', va='center', fontsize=8,
             color=OI['blue'], fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

# ── Save ───────────────────────────────────────────────────────────────────
outpath = os.path.join(FIGURES_DIR, 'FigS19_community_detection_ablation.png')
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'[OK] Saved: {outpath}')
