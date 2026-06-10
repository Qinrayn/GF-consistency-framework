"""
12_human_ppi_validation.py  [DEPRECATED — use human_embed_all.py + human_gf_all.py]

Legacy script kept for reference only.  Paths are outdated
(./yeast_ppi_data/human/) and will not work without manual adjustment.

For reproducible results, use the modern pipeline:
    python human_embed_all.py
    python human_gf_all.py

Original description (legacy):
人类（Homo sapiens）PPI 网络 G-F 曲线验证（DM, Node2Vec）
使用 cKDTree 秒级建图 + Louvain 社区检测 + 内存友好
"""
import warnings as _warnings
_warnings.warn(
    "12_human_ppi_validation.py is deprecated. "
    "Use human_embed_all.py + human_gf_all.py instead.",
    DeprecationWarning, stacklevel=2,
)
import json, os, gzip, numpy as np, networkx as nx
from collections import Counter, defaultdict
import requests
import gc
import time
import random

# 检查必需库
try:
    import community as community_louvain
except ImportError:
    raise ImportError("请先安装 python-louvain: pip install python-louvain")

from scipy.spatial import cKDTree

data_dir = "./yeast_ppi_data"
human_dir = os.path.join(data_dir, "human")
os.makedirs(human_dir, exist_ok=True)

# ========== 1. 下载/加载人类 STRING 网络 ==========
print("=== 人类 STRING 网络 ===")
ppi_file = os.path.join(human_dir, "9606.protein.links.v12.0.txt.gz")
if not os.path.exists(ppi_file):
    print("下载 STRING 网络（约 200MB）...")
    url = "https://stringdb-static.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    r = requests.get(url, stream=True)
    with open(ppi_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    print("下载完成。")

print("解析人类 PPI 网络...")
G_human = nx.Graph()
with gzip.open(ppi_file, 'rt') as f:
    f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        p1, p2, sc = parts[0], parts[1], int(parts[2])
        if sc >= 700:
            id1 = p1.split('.')[1] if '.' in p1 else p1
            id2 = p2.split('.')[1] if '.' in p2 else p2
            G_human.add_edge(id1, id2, weight=sc)

print(f"人类高置信度网络: {G_human.number_of_nodes()} 节点, {G_human.number_of_edges()} 边")

# ========== 2. 下载/加载人类 GO 注释 ==========
print("\n=== 人类 GO 注释 ===")
goa_file = os.path.join(human_dir, "goa_human.gaf.gz")
if not os.path.exists(goa_file):
    print("下载 GOA 文件（约 50MB）...")
    url = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
    r = requests.get(url, stream=True)
    with open(goa_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    print("下载完成。")

print("解析 GOA 文件...")
uniprot_go = defaultdict(set)
with gzip.open(goa_file, 'rt', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('!'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 13:
            continue
        uniprot = parts[2]
        go_id = parts[4]
        aspect = parts[8]
        if aspect == 'P':
            uniprot_go[uniprot].add(go_id)
print(f"有 GO 生物过程注释的 UniProt 数: {len(uniprot_go)}")

# ========== 3. 建立 STRING ID → UniProt ID 映射 ==========
print("\n=== STRING ID → UniProt 映射 ===")
alias_file = os.path.join(human_dir, "9606.protein.aliases.v12.0.txt.gz")
if not os.path.exists(alias_file):
    print("下载别名文件...")
    url = "https://stringdb-static.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz"
    r = requests.get(url, stream=True)
    with open(alias_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    print("下载完成。")

print("解析别名文件...")
string_to_uniprot = {}
with gzip.open(alias_file, 'rt', encoding='utf-8', errors='ignore') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        sid = parts[0].split('.')[1] if '.' in parts[0] else parts[0]
        aliases = parts[1].split()
        for alias in aliases:
            if alias in uniprot_go:
                string_to_uniprot[sid] = alias
                break
print(f"映射到 UniProt 的 STRING ID 数: {len(string_to_uniprot)}")

# ========== 4. 构建有 GO 注释的子网络 ==========
print("\n=== 构建注释子网络 ===")
annotated = {n for n in G_human.nodes() if n in string_to_uniprot}
G_anno = G_human.subgraph(annotated).copy()
comps = list(nx.connected_components(G_anno))
if not comps:
    raise ValueError("无连通注释节点")
largest_cc = max(comps, key=len)
G_final = G_anno.subgraph(largest_cc).copy()
nodes_list = list(G_final.nodes())
n = len(nodes_list)
print(f"人类注释网络: {n} 节点, {G_final.number_of_edges()} 边")

human_go_map = {node: list(uniprot_go[string_to_uniprot[node]]) for node in nodes_list}

# ========== 5. DM 嵌入 ==========
print("\n=== 计算 DM 嵌入 ===")
deg = nx.degree_centrality(G_final)
eig = nx.eigenvector_centrality(G_final, max_iter=2000, tol=1e-4)
pr = nx.pagerank(G_final, max_iter=200)
clust = nx.clustering(G_final)
avg_deg = nx.average_neighbor_degree(G_final)
kcore = nx.core_number(G_final)

feat = np.zeros((n, 6))
for i, u in enumerate(nodes_list):
    feat[i, 0] = deg[u]
    feat[i, 1] = eig[u]
    feat[i, 2] = pr[u]
    feat[i, 3] = clust[u]
    feat[i, 4] = avg_deg[u]
    feat[i, 5] = kcore[u]

feat = feat / (np.linalg.norm(feat, axis=0) + 1e-10)
sim = feat @ feat.T
deg_sim = sim.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / (np.sqrt(deg_sim) + 1e-10))
norm_sim = D_inv_sqrt @ sim @ D_inv_sqrt

eigvals, eigvecs = np.linalg.eigh(norm_sim)
idx = np.argsort(eigvals)
dm_coords = np.column_stack([eigvecs[:, idx[-2]], eigvecs[:, idx[-3]]])
dm_coords = dm_coords / np.std(dm_coords) * 0.3
pos_dm = {nodes_list[i]: dm_coords[i].tolist() for i in range(n)}

del sim, norm_sim, D_inv_sqrt, feat
gc.collect()
print("DM 嵌入完成。")

# ========== 6. Node2Vec 嵌入 ==========
print("\n=== 计算 Node2Vec 嵌入 ===")
walks_per_node = 5
walk_length = 15
p, q = 0.5, 2.0
window = 5

np.random.seed(42)
node_to_idx = {u: i for i, u in enumerate(nodes_list)}


def node2vec_walk(G, start, length, p, q):
    walk = [start]
    for _ in range(length - 1):
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))
        if not neighbors:
            break
        if len(walk) == 1:
            nxt = np.random.choice(neighbors)
        else:
            prev = walk[-2]
            probs = []
            for nbr in neighbors:
                if nbr == prev:
                    probs.append(1.0 / p)
                elif G.has_edge(nbr, prev):
                    probs.append(1.0)
                else:
                    probs.append(1.0 / q)
            probs = np.array(probs)
            probs /= probs.sum()
            nxt = np.random.choice(neighbors, p=probs)
        walk.append(nxt)
    return walk


walks = []
for i, node in enumerate(nodes_list):
    for _ in range(walks_per_node):
        walk_ids = node2vec_walk(G_final, node, walk_length, p, q)
        walks.append([node_to_idx[n] for n in walk_ids])
    if (i + 1) % 5000 == 0:
        print(f"  已处理 {i + 1}/{n} 个节点")

print("构建共现矩阵...")
cooc = np.zeros((n, n), dtype=np.float32)
for w in walks:
    for i, node_i in enumerate(w):
        start = max(0, i - window)
        end = min(len(w), i + window + 1)
        for j in range(start, end):
            if i != j:
                cooc[node_i, w[j]] += 1

print("SVD 分解...")
U, S, Vt = np.linalg.svd(cooc, full_matrices=False)
del cooc, walks
gc.collect()
n2v_emb = U[:, :2] @ np.diag(np.sqrt(S[:2]))
n2v_emb = n2v_emb / np.std(n2v_emb) * 0.3
pos_n2v = {nodes_list[i]: n2v_emb[i].tolist() for i in range(n)}
print("Node2Vec 嵌入完成。")

# ========== 7. 极速 G-F 扫描（cKDTree + Louvain） ==========
print("\n=== 极速 G-F 扫描 ===")


def fast_spatial_graph(r, coord_dict, nodes_subset):
    """使用 cKDTree 内存友好地构建空间图"""
    coords = np.array([coord_dict[u] for u in nodes_subset])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=r, output_type='ndarray')
    G_r = nx.Graph()
    G_r.add_nodes_from(nodes_subset)
    if len(pairs) > 0:
        edges = [(nodes_subset[i], nodes_subset[j]) for i, j in pairs]
        G_r.add_edges_from(edges)
    return G_r


def fast_purity(G_comm, go_map):
    """使用 Louvain 算法快速计算功能纯度"""
    if G_comm.number_of_edges() == 0:
        return 0.0
    
    # 固定 Louvain 随机性
    random.seed(42)
    partition = community_louvain.best_partition(G_comm, resolution=1.0)
    
    comm_dict = defaultdict(list)
    for node, cid in partition.items():
        comm_dict[cid].append(node)
    
    purities = []
    for nodes in comm_dict.values():
        if not nodes:
            continue
        all_terms = []
        for node in nodes:
            all_terms.extend(go_map.get(node, []))
        if not all_terms:
            continue
        most_common = Counter(all_terms).most_common(1)[0][1]
        # NOTE: Uses old purity formula (most_common / cluster_size).
        # See utils._community_purity for the current standard (most_common / total_GO_terms).
        purities.append(most_common / len(nodes))
    
    return np.mean(purities) if purities else 0.0


# 关键优化：只扫描有意义的 r 范围 [0.05, 0.20]
# 人类网络在 r > 0.2 后图过于稠密，Louvain 极慢且纯度趋于随机基线
r_vals = np.linspace(0.05, 0.20, 20)

print("扫描 DM ...")
purities_dm = []
for r in r_vals:
    t0 = time.time()
    Gr = fast_spatial_graph(r, pos_dm, nodes_list)
    pur = fast_purity(Gr, human_go_map)
    purities_dm.append(pur)
    print(f"  r={r:.3f}, purity={pur:.4f}, edges={Gr.number_of_edges()}, time={time.time()-t0:.1f}s")

print("扫描 Node2Vec ...")
purities_n2v = []
for r in r_vals:
    t0 = time.time()
    Gr = fast_spatial_graph(r, pos_n2v, nodes_list)
    pur = fast_purity(Gr, human_go_map)
    purities_n2v.append(pur)
    print(f"  r={r:.3f}, purity={pur:.4f}, edges={Gr.number_of_edges()}, time={time.time()-t0:.1f}s")

# ========== 8. 保存结果 ==========
human_result = {
    "n_nodes": n,
    "n_edges": G_final.number_of_edges(),
    "r": r_vals.tolist(),
    "DM_purity": purities_dm,
    "Node2Vec_purity": purities_n2v
}
with open(os.path.join(human_dir, "human_ppi_results.json"), 'w') as f:
    json.dump(human_result, f)

print(f"\n{'='*50}")
print(f"人类网络验证完成！")
print(f"网络规模: {n} 节点, {G_final.number_of_edges()} 边")
print(f"DM 最高纯度: {max(purities_dm):.4f} (r={r_vals[np.argmax(purities_dm)]:.3f})")
print(f"Node2Vec 最高纯度: {max(purities_n2v):.4f} (r={r_vals[np.argmax(purities_n2v)]:.3f})")
print(f"{'='*50}")  