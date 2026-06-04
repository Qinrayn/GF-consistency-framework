"""
12d_human_n2v_cleaned_scan.py  [DEPRECATED — use human_embed_all.py + human_gf_all.py]

Legacy script kept for reference only.  Paths are outdated and will not
work without manual adjustment.
"""
import warnings as _warnings
_warnings.warn(
    "12d_human_n2v_cleaned_scan.py is deprecated. "
    "Use human_embed_all.py + human_gf_all.py instead.",
    DeprecationWarning, stacklevel=2,
)
import json, os, numpy as np, networkx as nx
from collections import Counter
import time
from scipy.spatial import cKDTree
import igraph as ig
import gc

data_dir = "./yeast_ppi_data/human"

# ====== 1. 加载原始嵌入和 GO 注释 ======
print("加载原始 Node2Vec 嵌入...")
with open(os.path.join(data_dir, "human_n2v_emb.json")) as f:
    raw_coords = json.load(f)

print("加载人类网络和 GO 注释...")
import gzip
from collections import defaultdict

ppi_file = "./yeast_ppi_data/human/9606.protein.links.v12.0.txt.gz"
goa_file = "./yeast_ppi_data/human/goa_human.gaf.gz"
alias_file = "./yeast_ppi_data/human/9606.protein.aliases.v12.0.txt.gz"

G_human = nx.Graph()
with gzip.open(ppi_file, 'rt') as f:
    f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3: continue
        p1, p2, sc = parts[0], parts[1], int(parts[2])
        if sc >= 700:
            id1 = p1.split('.')[1] if '.' in p1 else p1
            id2 = p2.split('.')[1] if '.' in p2 else p2
            G_human.add_edge(id1, id2, weight=sc)

uniprot_go = defaultdict(set)
with gzip.open(goa_file, 'rt', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('!'): continue
        parts = line.strip().split('\t')
        if len(parts) < 13: continue
        if parts[8] == 'P':
            uniprot_go[parts[2]].add(parts[4])

string_to_uniprot = {}
with gzip.open(alias_file, 'rt', encoding='utf-8', errors='ignore') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2: continue
        sid = parts[0].split('.')[1] if '.' in parts[0] else parts[0]
        aliases = parts[1].split()
        for alias in aliases:
            if alias in uniprot_go:
                string_to_uniprot[sid] = alias
                break

annotated = {n for n in G_human.nodes() if n in string_to_uniprot}
G_anno = G_human.subgraph(annotated).copy()
largest_cc = max(nx.connected_components(G_anno), key=len)
G_final = G_anno.subgraph(largest_cc).copy()
nodes_list = list(G_final.nodes())
print(f"注释网络: {len(nodes_list)} 节点")

human_go_map = {node: list(uniprot_go[string_to_uniprot[node]]) for node in nodes_list}

# ====== 2. 剔除离群点并重新标准化 ======
# 将所有节点坐标转换为数组
all_nodes = list(raw_coords.keys())
arr = np.array([raw_coords[n] for n in all_nodes])

# 识别离群点：x坐标小于 -1.0 的点（我们的离群点是 -40，这个阈值足够安全）
outlier_mask = arr[:, 0] < -1.0
outlier_nodes = [all_nodes[i] for i in np.where(outlier_mask)[0]]
print(f"剔除离群点: {outlier_nodes}")

# 保留正常点
clean_mask = ~outlier_mask
clean_nodes = [all_nodes[i] for i in np.where(clean_mask)[0]]
arr_clean = arr[clean_mask]

# 重新缩放到目标标准差 0.3
target_std = 0.3
current_std = arr_clean.std(axis=0)
arr_scaled = (arr_clean - arr_clean.mean(axis=0)) / current_std * target_std

# 构建新的坐标字典
pos_n2v_cleaned = {clean_nodes[i]: arr_scaled[i].tolist() for i in range(len(clean_nodes))}

# 同时确保我们只评估在人类网络中有 GO 注释且在清洁列表中的节点
valid_nodes = sorted(set(pos_n2v_cleaned.keys()) & set(human_go_map.keys()))
coords_n2v = np.array([pos_n2v_cleaned[u] for u in valid_nodes])
print(f"最终评估节点数: {len(valid_nodes)}")

# ====== 3. G-F 扫描 (cKDTree + Leiden) ======
r_vals = np.linspace(0.05, 0.30, 12)
tree = cKDTree(coords_n2v)

def leiden_purity_fast(pairs, nodes_subset, go_map):
    G_r = nx.Graph()
    G_r.add_nodes_from(nodes_subset)
    if len(pairs) > 0:
        edges = [(nodes_subset[i], nodes_subset[j]) for i, j in pairs]
        G_r.add_edges_from(edges)
    if G_r.number_of_edges() == 0:
        return 0.0

    mapping = {u: i for i, u in enumerate(G_r.nodes())}
    rev = {i: u for u, i in mapping.items()}
    g = ig.Graph()
    g.add_vertices(len(mapping))
    g.add_edges([(mapping[u], mapping[v]) for u, v in G_r.edges()])
    partition = g.community_leiden(objective_function='modularity')

    purities = []
    for cluster in partition:
        cluster_nodes = [rev[i] for i in cluster]
        if not cluster_nodes: continue
        cnt = Counter()
        for node in cluster_nodes:
            cnt.update(go_map.get(node, []))
        if cnt:
            purities.append(cnt.most_common(1)[0][1] / len(cluster_nodes))
    return np.mean(purities) if purities else 0.0

print("\nG-F 扫描 Node2Vec (清洗后) ...")
purities_n2v = []
for r_val in r_vals:
    t0 = time.time()
    pairs = tree.query_pairs(r=r_val, output_type='ndarray')
    pur = leiden_purity_fast(pairs, valid_nodes, human_go_map)
    purities_n2v.append(pur)
    print(f"  r={r_val:.3f}, purity={pur:.4f}, edges={len(pairs)}, time={time.time()-t0:.1f}s")

# ====== 4. 加载 DM 人类网络结果并对比 ======
with open("./yeast_ppi_data/human/human_ppi_results_serial.json") as f:
    human_dm = json.load(f)
dm_purities = human_dm["DM_purity_main"][:len(r_vals)]

print("\n人类网络 DM vs Node2Vec (清洗后) 对比:")
print("r\tDM_purity\tN2V_cleaned_purity")
for i, r_val in enumerate(r_vals):
    print(f"{r_val:.3f}\t{dm_purities[i]:.4f}\t{purities_n2v[i]:.4f}")

# 保存结果
result = {
    "r": r_vals.tolist(),
    "Node2Vec_purity_cleaned": purities_n2v,
    "DM_purity": dm_purities,
    "n_nodes": len(valid_nodes),
    "removed_outliers": outlier_nodes
}
with open(os.path.join(data_dir, "human_n2v_cleaned_results.json"), 'w') as f:
    json.dump(result, f)

print("\n清洗后的结果已保存。")