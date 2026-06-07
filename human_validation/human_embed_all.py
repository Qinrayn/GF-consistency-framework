"""
Human PPI Network: Generate all six embeddings (DM, MDS, Spectral, DeepWalk, Node2Vec, VGAE)
for the human STRING network (largest connected component).

This script extends the existing human_validation scripts to cover all methods,
matching the output format and standards used in scripts/embed_all.py.
"""

import os
import sys
import json
import gzip
import numpy as np
import networkx as nx
from scipy.spatial.distance import squareform, pdist
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.utils import (
    build_similarity_matrix, diffusion_map_from_similarity,
    classical_mds_from_distances, spectral_embedding_from_graph,
    deepwalk_from_graph, node2vec_from_graph, vgae_from_graph,
    rescale_coordinates, SEED
)

# ---- Configuration ----
HUMAN_DATA_DIR = os.path.dirname(__file__)
LINKS_FILE = os.path.join(HUMAN_DATA_DIR, '9606.protein.links.v12.0.txt.gz')
ALIASES_FILE = os.path.join(HUMAN_DATA_DIR, '9606.protein.aliases.v12.0.txt.gz')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTLIER_REPORT = os.path.join(HUMAN_DATA_DIR, 'outlier_report.txt')
SCORE_THRESHOLD = 700
TARGET_STD = 0.3
OUTLIER_STD_THRESHOLD = 100


def load_human_network():
    """Load human STRING network, filter by score, take largest connected component."""
    print("Loading human STRING network...")
    
    # Load edges
    edges = []
    with gzip.open(LINKS_FILE, 'rt', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                score = int(parts[2])
                if score >= SCORE_THRESHOLD:
                    edges.append((parts[0], parts[1], score))
    
    print(f"  Loaded {len(edges)} edges with score >= {SCORE_THRESHOLD}")
    
    # Build graph
    G = nx.Graph()
    for p1, p2, score in edges:
        G.add_edge(p1, p2, weight=score)
    
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Take largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"  Largest CC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def load_go_annotations():
    """
    Load GO biological process annotations for human proteins.
    Uses STRING alias file to map to UniProt, then loads GOA annotations.
    For this experiment, we use a simplified approach: load annotations from
    the data directory if available, otherwise use a placeholder.
    """
    # Try to load from existing processed data
    go_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'human_go_annotations.json')
    if os.path.exists(go_file):
        with open(go_file, 'r') as f:
            return json.load(f)
    
    # Placeholder: return empty dict (GF computation will use network structure only)
    print("  Warning: No GO annotations file found. Using empty annotations.")
    return {}


def detect_and_remove_outliers(coords, node_list, method_name):
    """
    Detect outliers: nodes with coordinate std > OUTLIER_STD_THRESHOLD.
    Returns cleaned coordinates, cleaned node list, and outlier info.
    """
    x_std = np.std(coords[:, 0])
    y_std = np.std(coords[:, 1])
    
    outliers = []
    clean_mask = np.ones(len(node_list), dtype=bool)
    
    for i, node in enumerate(node_list):
        x_dev = abs(coords[i, 0] - np.mean(coords[:, 0])) / max(x_std, 1e-10)
        y_dev = abs(coords[i, 1] - np.mean(coords[:, 1])) / max(y_std, 1e-10)
        if x_dev > OUTLIER_STD_THRESHOLD or y_dev > OUTLIER_STD_THRESHOLD:
            outliers.append({
                'node': node,
                'x': float(coords[i, 0]),
                'y': float(coords[i, 1]),
                'x_dev_sigma': float(x_dev),
                'y_dev_sigma': float(y_dev)
            })
            clean_mask[i] = False
    
    if outliers:
        print(f"  [{method_name}] Found {len(outliers)} outlier(s)")
        clean_coords = coords[clean_mask]
        clean_nodes = [n for n, m in zip(node_list, clean_mask) if m]
    else:
        print(f"  [{method_name}] No outliers detected")
        clean_coords = coords
        clean_nodes = list(node_list)
    
    return clean_coords, clean_nodes, outliers


def save_embedding(coords, node_list, method_name):
    """Save embedding as JSON in the standard format."""
    embedding = {}
    for i, node in enumerate(node_list):
        embedding[node] = {
            'x': float(coords[i, 0]),
            'y': float(coords[i, 1])
        }
    
    output_file = os.path.join(OUTPUT_DIR, f'human_{method_name.lower()}_embedding.json')
    with open(output_file, 'w') as f:
        json.dump(embedding, f, indent=2)
    print(f"  Saved: {output_file} ({len(embedding)} nodes)")
    return output_file


def compute_all_embeddings(G):
    """Compute all six embeddings for the human network."""
    node_list = list(G.nodes())
    n = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    print(f"\nComputing embeddings for {n} nodes...")
    
    all_outliers = {}
    embeddings = {}
    
    # ---- 1. Diffusion Map ----
    print("\n[1/6] Diffusion Map (DM)...")
    try:
        # Compute centrality features
        degree_centrality = nx.degree_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        pagerank = nx.pagerank(G)
        clustering_coeff = nx.clustering(G)
        avg_neighbor_degree = nx.average_neighbor_degree(G)
        core_number = nx.core_number(G)
        
        features = np.zeros((n, 6))
        for i, node in enumerate(node_list):
            features[i, 0] = degree_centrality[node]
            features[i, 1] = eigenvector_centrality[node]
            features[i, 2] = pagerank[node]
            features[i, 3] = clustering_coeff[node]
            features[i, 4] = avg_neighbor_degree[node]
            features[i, 5] = core_number[node]
        
        # L2 normalize columns
        features = normalize(features, norm='l2', axis=0)
        
        # Build similarity matrix and compute diffusion map
        sim_matrix = build_similarity_matrix(features)
        coords = diffusion_map_from_similarity(sim_matrix)
        coords = rescale_coordinates(coords, TARGET_STD)
        
        # Outlier detection
        coords, clean_nodes, outliers = detect_and_remove_outliers(coords, node_list, 'DM')
        all_outliers['DM'] = outliers
        
        # Re-standardize after outlier removal
        coords = rescale_coordinates(coords, TARGET_STD)
        
        embeddings['DM'] = (coords, clean_nodes)
        save_embedding(coords, clean_nodes, 'DM')
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # ---- 2. Classical MDS ----
    print("\n[2/6] Classical MDS...")
    try:
        # For large networks, compute shortest paths on a subset or use approximation
        # Using BFS-based shortest paths
        print("  Computing shortest-path distance matrix (this may take a while)...")
        dist_matrix = np.zeros((n, n))
        for i, source in enumerate(node_list):
            lengths = nx.single_source_shortest_path_length(G, source)
            for j, target in enumerate(node_list):
                dist_matrix[i, j] = lengths.get(target, float('inf'))
        
        # Replace inf with max finite distance + 1
        max_dist = np.max(dist_matrix[dist_matrix < float('inf')])
        dist_matrix[dist_matrix == float('inf')] = max_dist + 1
        
        coords = classical_mds_from_distances(dist_matrix)
        coords = rescale_coordinates(coords, TARGET_STD)
        
        coords, clean_nodes, outliers = detect_and_remove_outliers(coords, node_list, 'MDS')
        all_outliers['MDS'] = outliers
        coords = rescale_coordinates(coords, TARGET_STD)
        
        embeddings['MDS'] = (coords, clean_nodes)
        save_embedding(coords, clean_nodes, 'MDS')
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # ---- 3. Spectral Embedding ----
    print("\n[3/6] Spectral Embedding...")
    try:
        coords = spectral_embedding_from_graph(G)
        coords = rescale_coordinates(coords, TARGET_STD)
        
        coords, clean_nodes, outliers = detect_and_remove_outliers(coords, node_list, 'Spectral')
        all_outliers['Spectral'] = outliers
        coords = rescale_coordinates(coords, TARGET_STD)
        
        embeddings['Spectral'] = (coords, clean_nodes)
        save_embedding(coords, clean_nodes, 'Spectral')
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # ---- 4. DeepWalk ----
    print("\n[4/6] DeepWalk...")
    try:
        coords = deepwalk_from_graph(G, walk_length=20, walks_per_node=10, 
                                      window_size=5, dimensions=2, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        
        coords, clean_nodes, outliers = detect_and_remove_outliers(coords, node_list, 'DeepWalk')
        all_outliers['DeepWalk'] = outliers
        coords = rescale_coordinates(coords, TARGET_STD)
        
        embeddings['DeepWalk'] = (coords, clean_nodes)
        save_embedding(coords, clean_nodes, 'DeepWalk')
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # ---- 5. Node2Vec ----
    print("\n[5/6] Node2Vec...")
    try:
        coords = node2vec_from_graph(G, walk_length=20, walks_per_node=10,
                                      window_size=5, dimensions=2, p=0.5, q=2.0, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        
        coords, clean_nodes, outliers = detect_and_remove_outliers(coords, node_list, 'Node2Vec')
        all_outliers['Node2Vec'] = outliers
        coords = rescale_coordinates(coords, TARGET_STD)
        
        embeddings['Node2Vec'] = (coords, clean_nodes)
        save_embedding(coords, clean_nodes, 'Node2Vec')
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # ---- 6. VGAE ----
    print("\n[6/6] VGAE...")
    try:
        coords = vgae_from_graph(G, hidden_dim=4, latent_dim=2, epochs=300, 
                                  lr=0.01, seed=SEED)
        coords = rescale_coordinates(coords, TARGET_STD)
        
        coords, clean_nodes, outliers = detect_and_remove_outliers(coords, node_list, 'VGAE')
        all_outliers['VGAE'] = outliers
        coords = rescale_coordinates(coords, TARGET_STD)
        
        embeddings['VGAE'] = (coords, clean_nodes)
        save_embedding(coords, clean_nodes, 'VGAE')
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # ---- Write outlier report ----
    print(f"\nWriting outlier report to {OUTLIER_REPORT}...")
    with open(OUTLIER_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("HUMAN PPI NETWORK - OUTLIER DETECTION REPORT\n")
        f.write(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
        f.write(f"Outlier threshold: > {OUTLIER_STD_THRESHOLD} standard deviations\n")
        f.write("=" * 70 + "\n\n")
        
        for method, outliers in all_outliers.items():
            f.write(f"\n--- {method} ---\n")
            if outliers:
                f.write(f"Outliers found: {len(outliers)}\n")
                for o in outliers:
                    f.write(f"  Node: {o['node']}\n")
                    f.write(f"    Coordinates: ({o['x']:.4f}, {o['y']:.4f})\n")
                    f.write(f"    Deviation: x={o['x_dev_sigma']:.1f}σ, y={o['y_dev_sigma']:.1f}σ\n")
            else:
                f.write("No outliers detected.\n")
    
    print(f"\nEmbedding generation complete. {len(embeddings)} methods succeeded.")
    return embeddings


def main():
    np.random.seed(SEED)
    
    # Load network
    G = load_human_network()
    
    # Load GO annotations (for reference)
    go_annotations = load_go_annotations()
    
    # Compute all embeddings
    embeddings = compute_all_embeddings(G)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for method, (coords, nodes) in embeddings.items():
        print(f"  {method}: {len(nodes)} nodes, shape={coords.shape}")
        print(f"    x: mean={np.mean(coords[:, 0]):.4f}, std={np.std(coords[:, 0]):.4f}")
        print(f"    y: mean={np.mean(coords[:, 1]):.4f}, std={np.std(coords[:, 1]):.4f}")


if __name__ == '__main__':
    main()
