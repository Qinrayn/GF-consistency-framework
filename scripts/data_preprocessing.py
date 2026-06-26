#!/usr/bin/env python3
"""
data_preprocessing.py
Step 1: Prepare all data files from raw STRING data.
- Generate yeast_ppi_5936.edgelist (full STRING network, score >= 700)
- Generate curated_153_ppi.edgelist (curated subgraph)
- Generate 10 random 150-node subset edgelists
"""
from __future__ import annotations

import sys
import json
import gzip
import random
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import SEED, get_data_dir, load_full_STRING_network

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    data_dir = get_data_dir()
    
    # ---- Step 1: Full STRING network ----
    print("Loading full STRING network...")
    G_full = load_full_STRING_network(data_dir)
    print(f"Full network: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
    
    # Save full edgelist
    full_edgelist = data_dir / "yeast_ppi_5936.edgelist"
    with open(full_edgelist, "w", encoding="utf-8") as f:
        for u, v in G_full.edges():
            f.write(f"{u}\t{v}\n")
    print(f"Saved {full_edgelist.name}: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
    
    # ---- Step 2: Curated 153-node subgraph ----
    print("\nBuilding curated 153-node subgraph...")
    go_map_file = data_dir / "gene_go_map.json"
    with open(go_map_file, encoding="utf-8") as f:
        go_map = json.load(f)
    
    curated_nodes_file = data_dir / "curated_153_nodes.txt"
    if curated_nodes_file.exists():
        with open(curated_nodes_file, encoding="utf-8") as f:
            curated_nodes = [line.strip() for line in f if line.strip()]
    else:
        # Intersect GO-annotated nodes with full network
        curated_nodes = sorted(set(go_map.keys()) & set(G_full.nodes()))
        with open(curated_nodes_file, "w", encoding="utf-8") as f:
            for n in curated_nodes:
                f.write(n + "\n")
    
    G_curated = G_full.subgraph(curated_nodes).copy()
    if not nx.is_connected(G_curated):
        comp = max(nx.connected_components(G_curated), key=len)
        G_curated = G_curated.subgraph(comp).copy()
        curated_nodes = sorted(G_curated.nodes())
        # Update node list
        with open(curated_nodes_file, "w", encoding="utf-8") as f:
            for n in curated_nodes:
                f.write(n + "\n")
    
    print(f"Curated subgraph: {G_curated.number_of_nodes()} nodes, {G_curated.number_of_edges()} edges")
    
    curated_edgelist = data_dir / "curated_153_ppi.edgelist"
    with open(curated_edgelist, "w", encoding="utf-8") as f:
        for u, v in G_curated.edges():
            f.write(f"{u}\t{v}\n")
    print(f"Saved {curated_edgelist.name}")
    
    # ---- Step 3: Generate 10 random 150-node subsets ----
    print("\nGenerating 10 random 150-node subsets...")
    all_nodes = curated_nodes
    
    for i in range(10):
        random.seed(SEED + i)
        subset_nodes = random.sample(all_nodes, 150)
        G_sub = G_curated.subgraph(subset_nodes).copy()
        
        if not nx.is_connected(G_sub):
            comp = max(nx.connected_components(G_sub), key=len)
            G_sub = G_sub.subgraph(comp).copy()
        
        subset_file = data_dir / f"subset_150_{i+1}.edgelist"
        with open(subset_file, "w", encoding="utf-8") as f:
            for u, v in G_sub.edges():
                f.write(f"{u}\t{v}\n")
        print(f"  Subset {i+1}: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges -> {subset_file.name}")
    
    print("\nData preprocessing complete!")

if __name__ == "__main__":
    main()
