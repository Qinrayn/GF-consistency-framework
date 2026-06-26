#!/usr/bin/env python3
"""
human_validation.py
Step 13: Human PPI cross-species validation.
- Load STRING v12.0 human PPI (score >= 700)
- Compute DM and Node2Vec on full human PPI network (14,679 nodes in largest CC; ~15,882 before CC extraction)
- Evaluate G-F curves on GO-annotated nodes
- Detect and remove Node2Vec outlier (ENSP00000334051, x ≈ -40.75)
- Compare original vs cleaned curves
"""
from __future__ import annotations

import sys
import json
import random
import gzip
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_project_root, get_data_dir, get_results_dir, get_embeddings_dir,
    compute_centrality_features, rescale_coordinates,
    compute_gf_curve, save_embedding,
    build_similarity_matrix, diffusion_map_from_similarity,
    node2vec_from_graph,
)

R_MIN = 0.05
R_MAX = 0.55
N_POINTS = 200


def load_human_ppi(data_dir=None, min_score=700):
    """Load human PPI from STRING v12.0 file."""
    if data_dir is None:
        # Human STRING data lives in the human_validation/ subdirectory
        data_dir = get_project_root() / "human_validation"
    data_dir = Path(data_dir)
    
    string_file = data_dir / "9606.protein.links.v12.0.txt.gz"
    if not string_file.exists():
        raise FileNotFoundError(
            f"Human STRING file not found: {string_file}\n"
            f"Download from https://string-db.org/cgi/download?species_text=9606"
        )
    
    G = nx.Graph()
    with gzip.open(str(string_file), 'rt', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p1, p2, score = parts
            if int(score) >= min_score:
                p1_clean = p1.split('.')[1]
                p2_clean = p2.split('.')[1]
                G.add_edge(p1_clean, p2_clean)
    
    # Largest connected component
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    return G


def load_human_go_annotations(data_dir=None):
    """Load human GO annotations mapped to STRING protein IDs.
    
    Returns dict {protein_id: [GO_terms]}
    """
    if data_dir is None:
        data_dir = get_project_root() / "human_validation"
    data_dir = Path(data_dir)
    
    goa_file = data_dir / "human_go_annotations.json"
    if goa_file.exists():
        with open(goa_file) as f:
            return json.load(f)
    
    # Fallback: try the project-level data directory
    alt_goa = get_data_dir() / "human_go_annotations.json"
    if alt_goa.exists():
        with open(alt_goa) as f:
            return json.load(f)
    
    # Try to build from GOA GAF file
    gaf_file = data_dir / "goa_human.gaf"
    if not gaf_file.exists():
        # Try gzipped version
        gaf_gz = data_dir / "goa_human.gaf.gz"
        if gaf_gz.exists():
            gaf_file = gaf_gz
        else:
            print("WARNING: No human GO annotation file found.")
            print("Download from https://www.ebi.ac.uk/GOA/goaHuman")
            return {}
    
    go_map = {}
    opener_kwargs = {'encoding': 'utf-8'} if str(gaf_file).endswith('.gz') else {'encoding': 'utf-8'}
    with (gzip.open if str(gaf_file).endswith('.gz') else open)(str(gaf_file), 'rt', **opener_kwargs) as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
            db_id = parts[1]  # UniProt ID
            go_term = parts[4]  # GO term
            
            # Map UniProt to STRING ID format (ENSP...)
            # This requires a mapping file; simplified version:
            if db_id not in go_map:
                go_map[db_id] = []
            go_map[db_id].append(go_term)
    
    return go_map


def embed_dm_human(G, nodes):
    """DM on human network."""
    print("  Computing centrality features on human network...")
    features = compute_centrality_features(G, nodes)
    sim = build_similarity_matrix(features)
    coords = diffusion_map_from_similarity(sim)
    return rescale_coordinates(coords, target_std=0.3)


def embed_node2vec_human(G, nodes, walk_length=20, walks_per_node=10, window=5, p=0.5, q=2.0):
    """Node2Vec on human network."""
    print(f"  Generating Node2Vec embedding for {len(nodes)} human nodes...")
    coords = node2vec_from_graph(G, walk_length=walk_length,
                                  walks_per_node=walks_per_node,
                                  window_size=window, p=p, q=q, seed=SEED)
    return rescale_coordinates(coords, target_std=0.3)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = get_embeddings_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    # Load human PPI (data files reside in human_validation/)
    print("Loading human PPI network...")
    try:
        G_human = load_human_ppi()
    except FileNotFoundError as e:
        print(f"SKIPPED: {e}")
        return
    
    nodes_human = sorted(G_human.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_human)}
    print(f"Human network: {len(nodes_human)} nodes, {G_human.number_of_edges()} edges")
    
    # Load GO annotations
    print("Loading human GO annotations...")
    go_map_human = load_human_go_annotations()
    if not go_map_human:
        print("SKIPPED: No GO annotations available")
        return
    
    # Find annotated nodes
    annotated_nodes = sorted(set(go_map_human.keys()) & set(nodes_human))
    print(f"Annotated nodes: {len(annotated_nodes)}")
    
    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)
    results = {"r": r_vals.tolist(), "n_nodes": len(nodes_human),
               "n_annotated": len(annotated_nodes)}
    
    # ---- DM ----
    print("\nComputing DM on human network...")
    try:
        dm_coords = embed_dm_human(G_human, nodes_human)
        save_embedding(dm_coords, nodes_human, "DM_human", "full", emb_dir)
        
        ann_idx = [node_to_idx[n] for n in annotated_nodes]
        dm_subset = dm_coords[ann_idx]
        dm_pur, dm_mod = compute_gf_curve(dm_subset, annotated_nodes, go_map_human, r_vals)
        results["DM_purity"] = dm_pur
        results["DM_modularity"] = dm_mod
        print(f"  DM purity range: [{min(dm_pur):.3f}, {max(dm_pur):.3f}]")
    except Exception as e:
        print(f"  DM FAILED: {e}")
    
    # ---- Node2Vec (with outlier detection) ----
    print("\nComputing Node2Vec on human network...")
    try:
        n2v_coords = embed_node2vec_human(G_human, nodes_human)
        save_embedding(n2v_coords, nodes_human, "Node2Vec_human", "full", emb_dir)
        
        # Detect outliers
        x_vals = n2v_coords[:, 0]
        x_mean, x_std = np.mean(x_vals), np.std(x_vals)
        outliers = np.abs(x_vals - x_mean) > 5 * x_std
        outlier_nodes = [nodes_human[i] for i in range(len(nodes_human)) if outliers[i]]
        print(f"  Outlier nodes: {outlier_nodes}")
        
        # Original evaluation (with outliers removed)
        ann_idx = [node_to_idx[n] for n in annotated_nodes if not outliers[node_to_idx[n]]]
        ann_nodes_clean = [n for n in annotated_nodes if not outliers[node_to_idx[n]]]
        n2v_subset = n2v_coords[ann_idx]
        
        # Re-normalize after outlier removal
        n2v_subset = rescale_coordinates(n2v_subset, target_std=0.3)
        n2v_pur, n2v_mod = compute_gf_curve(n2v_subset, ann_nodes_clean, go_map_human, r_vals)
        results["Node2Vec_cleaned_purity"] = n2v_pur
        results["Node2Vec_cleaned_modularity"] = n2v_mod
        
        # Also evaluate with outliers
        ann_idx_all = [node_to_idx[n] for n in annotated_nodes]
        n2v_subset_all = n2v_coords[ann_idx_all]
        n2v_pur_raw, n2v_mod_raw = compute_gf_curve(n2v_subset_all, annotated_nodes, go_map_human, r_vals)
        results["Node2Vec_raw_purity"] = n2v_pur_raw
        results["Node2Vec_raw_modularity"] = n2v_mod_raw
        results["outlier_nodes"] = outlier_nodes
        
        print(f"  Node2Vec cleaned purity range: [{min(n2v_pur):.3f}, {max(n2v_pur):.3f}]")
        print(f"  Node2Vec raw purity range: [{min(n2v_pur_raw):.3f}, {max(n2v_pur_raw):.3f}]")
    except Exception as e:
        print(f"  Node2Vec FAILED: {e}")
    
    # Save results
    output_file = results_dir / "human_ppi_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
