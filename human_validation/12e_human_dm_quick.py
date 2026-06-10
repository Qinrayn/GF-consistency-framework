"""
12e_human_dm_quick.py  [DEPRECATED — use human_embed_all.py + human_gf_all.py]

Legacy script kept for reference only.  Paths are outdated and will not
work without manual adjustment.
"""
import warnings as _warnings
_warnings.warn(
    "12e_human_dm_quick.py is deprecated. "
    "Use human_embed_all.py + human_gf_all.py instead.",
    DeprecationWarning, stacklevel=2,
)
import json, os, gzip, numpy as np, networkx as nx
from collections import Counter, defaultdict
import time
from scipy.spatial import cKDTree
import igraph as ig

data_dir = "./yeast_ppi_data/human"

# 加载网络和注释（与之前一致）
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
n = len(nodes_list)
print(f"注释网络: {n} 节点")

human_go_map = {node: list(uniprot_go[string_to_uniprot[node]]) for node in nodes_list}

# DM嵌入（与酵母相同的6个中心性）
print("计算 DM 嵌入...")
deg = nx.degree_centrality(G_final)
eig = nx.eigenvector_centrality(G_final, max_iter=2000, tol=1e-4)
pr = nx.pagerank(G_final, max_iter=200)
clust = nx.clustering(G_final)
avg_deg = nx.average_neighbor_degree(G_final)
kcore = nx.core_number(G_final)

feat = np.zeros((n, 6))
for i, u in enumerate(nodes_list):
    feat[i, 0] = deg[u]; feat[i, 1] = eig[u]; feat[i, 2] = pr[u]
    feat[i, 3] = clust[u]; feat[i, 4] = avg_deg[u]; feat[i, 5] = kcore[u]
feat = feat / (np.linalg.norm(feat, axis=0) + 1e-10)
sim = feat @ feat.T
deg_sim = sim.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / (np.sqrt(deg_sim) + 1e-10))
norm_sim = D_inv_sqrt @ sim @ D_inv_sqrt
eigvals, eigvecs = np.linalg.eigh(norm_sim)
idx = np.argsort(eigvals)
v2 = eigvecs[:, idx[-2]]; v3 = eigvecs[:, idx[-3]]
dm_coords = np.column_stack([v2, v3])
dm_coords = dm_coords / np.std(dm_coords) * 0.3
pos_dm = {nodes_list[i]: dm_coords[i].tolist() for i in range(n)}

# 扫描 (cKDTree + Leiden)
valid_nodes = nodes_list
coords_dm = np.array([pos_dm[u] for u in valid_nodes])
r_vals = np.linspace(0.05, 0.30, 12)
tree = cKDTree(coords_dm)

def leiden_purity(pairs, nodes_subset, go_map):
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
            # NOTE: Uses old purity formula (most_common / cluster_size).
            # See utils._community_purity for the current standard (most_common / total_GO_terms).
            purities.append(cnt.most_common(1)[0][1] / len(cluster_nodes))
    return np.mean(purities) if purities else 0.0

print("G-F扫描 DM...")
purities_dm = []
for r_val in r_vals:
    t0 = time.time()
    pairs = tree.query_pairs(r=r_val, output_type='ndarray')
    pur = leiden_purity(pairs, valid_nodes, human_go_map)
    purities_dm.append(pur)
    print(f"  r={r_val:.3f}, purity={pur:.4f}, edges={len(pairs)}, time={time.time()-t0:.1f}s")

# 保存
result = {"r": r_vals.tolist(), "DM_purity": purities_dm, "n_nodes": n}
with open(os.path.join(data_dir, "human_dm_quick_results.json"), 'w') as f:
    json.dump(result, f)
print("DM快速扫描完成，结果已保存。")