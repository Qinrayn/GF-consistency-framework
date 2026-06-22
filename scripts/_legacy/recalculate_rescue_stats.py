#!/usr/bin/env python3
"""
Recalculate Mann-Whitney U test statistics for rescued protein analysis.

Fixes two bugs in the original rescue_protein_analysis.py:
1. Effect size r was computed as U/(n1*n2) instead of Z/sqrt(N)
2. Non-rescue group was subsampled to 500 instead of using all proteins
"""

import sys
import time
import json
from pathlib import Path
from collections import Counter

import numpy as np
import networkx as nx
from scipy import stats
from sklearn.neighbors import NearestNeighbors

# Set up paths
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from utils import SEED, load_embedding
from function_prediction import (
    build_alias_mapping, parse_gaf_experimental,
    ppi_neighbor_predict, K_MAX,
)

DATA = PROJECT / "data"
EMB = PROJECT / "embeddings"
NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"

np.random.seed(SEED)

# ============================================================
# Step 1: Load data
# ============================================================
print("[1/6] Loading data...")
sgd_to_string, orf_to_string, network_nodes = build_alias_mapping()
annotations, ann_stats = parse_gaf_experimental(
    sgd_to_string, orf_to_string, network_nodes
)

G = nx.Graph()
with open(str(NETWORK_FILE), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            G.add_edge(parts[0], parts[1])
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()
print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ============================================================
# Step 2: Load MDS embedding
# ============================================================
print("\n[2/6] Loading MDS embedding...")
coords, emb_nodes = load_embedding("MDS", "full", embeddings_dir=EMB)
node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
common = [n for n in emb_nodes if n in set(G.nodes())]
indices = [node_to_idx[n] for n in common]
filtered_coords = coords[indices]
node_to_idx_common = {n: i for i, n in enumerate(common)}

nn_model = NearestNeighbors(
    n_neighbors=min(K_MAX + 1, len(filtered_coords)), metric="euclidean"
)
nn_model.fit(filtered_coords)
print(f"  MDS: {len(common)} nodes")

# ============================================================
# Step 3: Identify rescue proteins (same logic as original)
# ============================================================
print("\n[3/6] Identifying rescue proteins (LOTO with MDS)...")

query_proteins = {
    pid: terms for pid, terms in annotations.items()
    if len(terms) >= 2 and pid in G and pid in node_to_idx_common
}

rescued_proteins = set()
rescued_trials = 0
ppi_only_trials = 0
both_trials = 0
miss_trials = 0
completed = 0
t0 = time.time()

total_trials = sum(len(terms) for terms in query_proteins.values())
print(f"  Query proteins: {len(query_proteins)}, Total trials: {total_trials}")

for pid, terms in sorted(query_proteins.items()):
    for hidden_term in sorted(terms):
        # PPI prediction
        ppi_preds = ppi_neighbor_predict(pid, G, annotations, hidden_term)
        ppi_terms = set(t for t, _ in ppi_preds)
        ppi_found = hidden_term in ppi_terms

        # Embedding prediction
        query_idx = node_to_idx_common[pid]
        n_neighbors = min(K_MAX + 1, len(filtered_coords))
        dists, idxs = nn_model.kneighbors(
            filtered_coords[query_idx:query_idx + 1], n_neighbors=n_neighbors
        )
        ts = Counter()
        for d_val, idx in zip(dists[0], idxs[0]):
            if idx == query_idx:
                continue
            nid = common[idx]
            w = 1.0 / (d_val + 1e-10)
            for term in annotations.get(nid, set()):
                ts[term] += w
        emb_found = hidden_term in ts

        if ppi_found and emb_found:
            both_trials += 1
        elif ppi_found:
            ppi_only_trials += 1
        elif emb_found:
            rescued_proteins.add(pid)
            rescued_trials += 1
        else:
            miss_trials += 1

        completed += 1
        if completed % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    {completed}/{total_trials} ({100*completed/total_trials:.0f}%) "
                  f"-- {elapsed:.1f}s")

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")
print(f"  Rescue trials: {rescued_trials}, Proteins: {len(rescued_proteins)}")
print(f"  PPI-only: {ppi_only_trials}, Both: {both_trials}, Miss: {miss_trials}")

rescue_proteins_sorted = sorted(rescued_proteins)

# ============================================================
# Step 4: Network topology analysis — ALL non-rescue proteins
# ============================================================
print("\n[4/6] Network topology analysis (ALL non-rescue proteins)...")

all_annotated_in_network = set(annotations.keys()) & set(G.nodes())
non_rescue = all_annotated_in_network - rescued_proteins

n1 = len(rescued_proteins & set(G.nodes()))
n2 = len(non_rescue)
N = n1 + n2
print(f"  Rescued proteins in network: n1 = {n1}")
print(f"  Non-rescued proteins: n2 = {n2}")
print(f"  Total: N = {N}")

# Compute degrees for ALL proteins
print("  Computing degrees...")
rescue_degrees = []
nonrescue_degrees = []

for pid in rescue_proteins_sorted:
    if pid in G:
        rescue_degrees.append(G.degree(pid))

for pid in sorted(non_rescue):
    if pid in G:
        nonrescue_degrees.append(G.degree(pid))

print(f"  Rescue degrees: {len(rescue_degrees)}, Non-rescue degrees: {len(nonrescue_degrees)}")

# Compute clustering coefficients for ALL proteins
print("  Computing clustering coefficients...")
rescue_clustering = []
nonrescue_clustering = []

for pid in rescue_proteins_sorted:
    if pid in G:
        rescue_clustering.append(nx.clustering(G, pid))

for pid in sorted(non_rescue):
    if pid in G:
        nonrescue_clustering.append(nx.clustering(G, pid))

# Compute betweenness centrality (using k=200 for approximation, same as original)
print("  Computing betweenness centrality (k=200, approximate)...")
bc = nx.betweenness_centrality(G, k=min(200, G.number_of_nodes()))

rescue_betweenness = []
nonrescue_betweenness = []

for pid in rescue_proteins_sorted:
    if pid in G:
        rescue_betweenness.append(bc.get(pid, 0))

for pid in sorted(non_rescue):
    if pid in G:
        nonrescue_betweenness.append(bc.get(pid, 0))

# ============================================================
# Step 5: Statistical tests with CORRECT formulas
# ============================================================
print("\n[5/6] Statistical tests...")

def correct_effect_size_r(u_stat, n1, n2):
    """Compute the correct effect size r = Z / sqrt(N).
    
    Uses the normal approximation of the Mann-Whitney U distribution.
    """
    N = n1 + n2
    # Expected U under H0
    mu = n1 * n2 / 2
    # Standard deviation of U under H0 (no tie correction for simplicity)
    sigma = np.sqrt(n1 * n2 * (N + 1) / 12)
    # Z-score
    z = (u_stat - mu) / sigma
    # Effect size
    r = z / np.sqrt(N)
    return r, z

results = {}

for metric_name, r_vals, nr_vals in [
    ("degree", rescue_degrees, nonrescue_degrees),
    ("clustering", rescue_clustering, nonrescue_clustering),
    ("betweenness", rescue_betweenness, nonrescue_betweenness),
]:
    n1_actual = len(r_vals)
    n2_actual = len(nr_vals)
    
    u_stat, u_p = stats.mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
    r_correct, z_score = correct_effect_size_r(u_stat, n1_actual, n2_actual)
    
    # Also compute what the ORIGINAL (buggy) code would have reported
    r_buggy = u_stat / (n1_actual * n2_actual)
    
    r_med = float(np.median(r_vals))
    nr_med = float(np.median(nr_vals))
    r_mean = float(np.mean(r_vals))
    nr_mean = float(np.mean(nr_vals))
    
    results[metric_name] = {
        "n1": n1_actual,
        "n2": n2_actual,
        "rescue_median": r_med,
        "nonrescue_median": nr_med,
        "rescue_mean": r_mean,
        "nonrescue_mean": nr_mean,
        "U": float(u_stat),
        "p_value": float(u_p),
        "Z": float(z_score),
        "r_correct": float(r_correct),
        "r_buggy": float(r_buggy),
    }

# ============================================================
# Step 6: Print results
# ============================================================
print("\n" + "=" * 72)
print("CORRECTED MANN-WHITNEY U TEST RESULTS")
print("=" * 72)

for metric, res in results.items():
    print(f"\n--- {metric.upper()} ---")
    print(f"  n1 (rescued)     = {res['n1']}")
    print(f"  n2 (non-rescued) = {res['n2']}")
    print(f"  Rescue median    = {res['rescue_median']}")
    print(f"  Non-rescue median= {res['nonrescue_median']}")
    print(f"  Rescue mean      = {res['rescue_mean']:.6f}")
    print(f"  Non-rescue mean  = {res['nonrescue_mean']:.6f}")
    print(f"  U statistic      = {res['U']:.1f}")
    print(f"  p-value (2-sided)= {res['p_value']:.6e}")
    print(f"  Z-score          = {res['Z']:.4f}")
    print(f"  r (CORRECT)      = {res['r_correct']:.4f}  [r = Z/sqrt(N)]")
    print(f"  r (BUGGY)        = {res['r_buggy']:.4f}  [r = U/(n1*n2) -- WRONG]")

print("\n" + "=" * 72)
print("ORIGINAL (BUGGY) RESULTS FROM JSON (n2=500 subsampled)")
print("=" * 72)

# Load original results for comparison
with open(PROJECT / "results" / "rescue_protein_analysis.json") as f:
    original = json.load(f)

for metric in ["degree", "clustering", "betweenness"]:
    orig = original["topology_comparison"][metric]
    print(f"\n--- {metric.upper()} (original, n2=500) ---")
    print(f"  U = {orig['mann_whitney_u']:.1f}")
    print(f"  p = {orig['p_value']:.6e}")
    print(f"  r (buggy) = {orig['effect_size_r']:.4f}")

print("\n" + "=" * 72)
print("CORRECTED MANUSCRIPT TEXT")
print("=" * 72)

deg = results["degree"]
bet = results["betweenness"]
clu = results["clustering"]

# Format p-values nicely
def fmt_p(p):
    if p < 1e-10:
        return f"{p:.1e}"
    elif p < 0.001:
        return f"{p:.1e}"
    else:
        return f"{p:.3f}"

def fmt_u(u):
    return f"{u:,.0f}"

print(f"\nDegree: median {deg['rescue_median']:.0f} vs {deg['nonrescue_median']:.0f}")
print(f"  U = {fmt_u(deg['U'])}, p = {fmt_p(deg['p_value'])}, r = {abs(deg['r_correct']):.2f}")
print(f"  n1={deg['n1']}, n2={deg['n2']}, N={deg['n1']+deg['n2']}")

print(f"\nBetweenness: median {bet['rescue_median']:.2e} vs {bet['nonrescue_median']:.2e}")
print(f"  U = {fmt_u(bet['U'])}, p = {fmt_p(bet['p_value'])}, r = {abs(bet['r_correct']):.2f}")

print(f"\nClustering: median {clu['rescue_median']:.3f} vs {clu['nonrescue_median']:.3f}")
print(f"  U = {fmt_u(clu['U'])}, p = {fmt_p(clu['p_value'])}, r = {abs(clu['r_correct']):.2f}")

# Verify internal consistency
print("\n" + "=" * 72)
print("INTERNAL CONSISTENCY CHECK")
print("=" * 72)

for metric, res in results.items():
    n1, n2 = res['n1'], res['n2']
    N = n1 + n2
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (N + 1) / 12)
    z_from_u = (res['U'] - mu) / sigma
    # scipy's p-value should match 2 * (1 - Phi(|z|))
    p_from_z = 2 * (1 - stats.norm.cdf(abs(z_from_u)))
    r_from_z = z_from_u / np.sqrt(N)
    
    print(f"\n{metric.upper()}:")
    print(f"  Z from U: {z_from_u:.4f}")
    print(f"  p from Z: {p_from_z:.6e}")
    print(f"  p from scipy: {res['p_value']:.6e}")
    print(f"  r from Z: {r_from_z:.4f}")
    print(f"  Match: {'YES' if abs(p_from_z - res['p_value']) / max(res['p_value'], 1e-300) < 0.01 else 'NO (tie correction?)'}")

print("\n\nDone.")
