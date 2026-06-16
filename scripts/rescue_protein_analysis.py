#!/usr/bin/env python3
"""
Phase 15: Rescue Protein Characterisation
==========================================

Who are the proteins that ONLY embeddings can find functional associations
for? What makes them biologically special?

Three analyses:
  1. Identify the 258 embedding-rescued trials and extract proteins + terms
  2. GO enrichment of rescued proteins (hypergeometric test)
  3. Network topology characterisation (degree, betweenness, clustering)
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import hypergeom
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED, get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
)
from function_prediction import (
    build_alias_mapping, parse_gaf_experimental,
    build_knn_index, ppi_neighbor_predict,
    K_MAX,
)

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()
EMB = get_embeddings_dir()
NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"
BANNER = "=" * 64

FULL_METHODS = ["DM", "MDS", "Spectral", "Node2Vec", "VGAE"]
METHOD_COLORS = {
    "DM": "#08306b", "MDS": "#08519c", "Spectral": "#3182bd",
    "Node2Vec": "#fb6a4a", "VGAE": "#67000d",
}


def identify_rescue_trials(embeddings, graph, annotations, best_method="MDS"):
    """Re-run LOTO to identify specific rescued trials.

    Returns list of dicts with rescue details.
    """
    emb = embeddings[best_method]
    node_to_idx = emb["node_to_idx"]
    coords = emb["coords"]
    nodes = emb["nodes"]

    nn_model = build_knn_index(coords)

    query_proteins = {
        pid: terms for pid, terms in annotations.items()
        if len(terms) >= 2 and pid in graph and pid in node_to_idx
    }

    rescued = []
    ppi_only = []
    both_found = []
    completed = 0
    t0 = time.time()

    for pid, terms in sorted(query_proteins.items()):
        for hidden_term in sorted(terms):
            # PPI prediction
            ppi_preds = ppi_neighbor_predict(pid, graph, annotations, hidden_term)
            ppi_terms = set(t for t, _ in ppi_preds)
            ppi_found = hidden_term in ppi_terms

            # Embedding prediction
            query_idx = node_to_idx[pid]
            n_neighbors = min(K_MAX + 1, len(coords))
            dists, idxs = nn_model.kneighbors(
                coords[query_idx:query_idx + 1], n_neighbors=n_neighbors
            )
            ts = Counter()
            for d_val, idx in zip(dists[0], idxs[0]):
                if idx == query_idx:
                    continue
                nid = nodes[idx]
                w = 1.0 / (d_val + 1e-10)
                for term in annotations.get(nid, set()):
                    ts[term] += w
            emb_found = hidden_term in ts

            if ppi_found and emb_found:
                both_found.append({"protein": pid, "term": hidden_term})
            elif ppi_found:
                ppi_only.append({"protein": pid, "term": hidden_term})
            elif emb_found:
                # Find which embedding neighbor contributed
                rescue_neighbors = []
                for d_val, idx in zip(dists[0], idxs[0]):
                    if idx == query_idx:
                        continue
                    nid = nodes[idx]
                    if hidden_term in annotations.get(nid, set()):
                        net_dist = None
                        try:
                            net_dist = nx.shortest_path_length(graph, pid, nid)
                        except nx.NetworkXNoPath:
                            net_dist = -1
                        rescue_neighbors.append({
                            "neighbor_id": nid,
                            "embedding_dist": float(d_val),
                            "network_dist": net_dist,
                        })

                rescued.append({
                    "protein": pid,
                    "term": hidden_term,
                    "protein_terms": sorted(annotations[pid]),
                    "embedding_neighbors": rescue_neighbors,
                })

            completed += 1
            if completed % 5000 == 0:
                elapsed = time.time() - t0
                print(f"    {completed}/{sum(len(t) for t in query_proteins.values())} "
                      f"({100*completed/sum(len(t) for t in query_proteins.values()):.0f}%)")

    print(f"    Rescue: {len(rescued)}, PPI-only: {len(ppi_only)}, "
          f"Both: {len(both_found)}")
    return rescued, ppi_only, both_found


def go_enrichment_analysis(rescue_proteins, all_annotations, all_terms_set):
    """Hypergeometric enrichment of GO terms among rescued proteins.

    For each GO term, test if rescued proteins are enriched compared to
    the background (all annotated proteins).
    """
    rescue_set = set(rescue_proteins)
    N = len(all_annotations)  # total annotated proteins
    n = len(rescue_set)  # rescued proteins

    # Count term frequencies in background and rescue set
    term_bg_count = Counter()
    term_rescue_count = Counter()

    for pid, terms in all_annotations.items():
        for t in terms:
            term_bg_count[t] += 1
            if pid in rescue_set:
                term_rescue_count[t] += 1

    # Hypergeometric test for each term
    enrichment = []
    for term in all_terms_set:
        K = term_bg_count.get(term, 0)  # term count in background
        k = term_rescue_count.get(term, 0)  # term count in rescue set
        if K == 0 or k == 0:
            continue

        # P(X >= k) = 1 - P(X < k) = 1 - CDF(k-1)
        # hypergeom: M=N (pop), n=K (success in pop), N=n (sample)
        p_val = hypergeom.sf(k - 1, N, K, n)

        enrichment.append({
            "term": term,
            "k_rescue": k,
            "K_background": K,
            "n_rescue_total": n,
            "N_total": N,
            "fold_enrichment": (k / n) / (K / N) if K > 0 and n > 0 else 0,
            "p_value": p_val,
        })

    # Sort by p-value
    enrichment.sort(key=lambda x: x["p_value"])

    # Bonferroni correction
    n_tests = len(enrichment)
    for e in enrichment:
        e["p_bonferroni"] = min(e["p_value"] * n_tests, 1.0)

    # FDR (Benjamini-Hochberg)
    for i, e in enumerate(enrichment):
        e["p_fdr"] = min(e["p_value"] * n_tests / (i + 1), 1.0)

    n_sig = sum(1 for e in enrichment if e["p_fdr"] < 0.05)
    n_sig_bonf = sum(1 for e in enrichment if e["p_bonferroni"] < 0.05)

    print(f"\n  Enrichment: {n_tests} terms tested")
    print(f"  Significant (FDR<0.05): {n_sig}")
    print(f"  Significant (Bonferroni<0.05): {n_sig_bonf}")

    if enrichment:
        print(f"\n  Top 15 enriched GO BP terms:")
        print(f"  {'Term':12s} {'k':>4s} {'K':>6s} {'Fold':>6s} {'p_FDR':>10s} {'Description'}")
        for e in enrichment[:15]:
            print(f"  {e['term']:12s} {e['k_rescue']:>4d} {e['K_background']:>6d} "
                  f"{e['fold_enrichment']:>6.1f}x {e['p_fdr']:>10.2e}")

    return enrichment


def network_topology_analysis(rescue_proteins, graph, all_annotations):
    """Compare network topology of rescued vs non-rescued proteins."""
    rescue_set = set(rescue_proteins)
    non_rescue = set(all_annotations.keys()) & set(graph.nodes()) - rescue_set

    metrics = {"degree": [], "clustering": [], "betweenness": [],
               "neighbor_mean_degree": []}
    metrics_nr = {"degree": [], "clustering": [], "betweenness": [],
                  "neighbor_mean_degree": []}

    # Use ALL non-rescue proteins (no subsampling)
    nr_all = sorted(non_rescue)

    # Betweenness centrality (approximate with k samples)
    print("  Computing betweenness centrality...")
    bc = nx.betweenness_centrality(graph, k=min(200, graph.number_of_nodes()))

    for pid in rescue_proteins:
        if pid not in graph:
            continue
        deg = graph.degree(pid)
        clust = nx.clustering(graph, pid)
        bet = bc.get(pid, 0)
        # Mean neighbor degree
        neighbors = list(graph.neighbors(pid))
        mean_nd = np.mean([graph.degree(n) for n in neighbors]) if neighbors else 0

        metrics["degree"].append(deg)
        metrics["clustering"].append(clust)
        metrics["betweenness"].append(bet)
        metrics["neighbor_mean_degree"].append(mean_nd)

    for pid in nr_all:
        if pid not in graph:
            continue
        deg = graph.degree(pid)
        clust = nx.clustering(graph, pid)
        bet = bc.get(pid, 0)
        neighbors = list(graph.neighbors(pid))
        mean_nd = np.mean([graph.degree(n) for n in neighbors]) if neighbors else 0

        metrics_nr["degree"].append(deg)
        metrics_nr["clustering"].append(clust)
        metrics_nr["betweenness"].append(bet)
        metrics_nr["neighbor_mean_degree"].append(mean_nd)

    # Statistical comparison
    comparison = {}
    for key in metrics:
        r_vals = metrics[key]
        nr_vals = metrics_nr[key]
        if r_vals and nr_vals:
            n1 = len(r_vals)
            n2 = len(nr_vals)
            N = n1 + n2
            u_stat, u_p = stats.mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
            # Correct effect size: r = Z / sqrt(N)
            mu_u = n1 * n2 / 2.0
            sigma_u = np.sqrt(n1 * n2 * (N + 1) / 12.0)
            z_score = (u_stat - mu_u) / sigma_u
            effect_r = z_score / np.sqrt(N)
            comparison[key] = {
                "rescue_median": float(np.median(r_vals)),
                "nonrescue_median": float(np.median(nr_vals)),
                "rescue_mean": float(np.mean(r_vals)),
                "nonrescue_mean": float(np.mean(nr_vals)),
                "mann_whitney_u": float(u_stat),
                "p_value": float(u_p),
                "effect_size_r": float(effect_r),
                "n1": n1,
                "n2": n2,
            }

    print(f"\n  Network topology comparison (rescued vs non-rescued):")
    print(f"  {'Metric':25s} {'Rescue':>12s} {'Non-rescue':>12s} {'p-value':>10s}")
    for key, comp in comparison.items():
        print(f"  {key:25s} {comp['rescue_median']:>12.3f} "
              f"{comp['nonrescue_median']:>12.3f} {comp['p_value']:>10.4f}")

    return metrics, metrics_nr, comparison


def plot_fig76_rescue_characterisation(enrichment, topo_comp, metrics,
                                        metrics_nr, rescued, n_categories):
    """Fig76: Rescue protein characterisation dashboard."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Panel A: GO enrichment volcano-style ---
    ax_a = fig.add_subplot(gs[0, 0])
    if enrichment:
        # Plot top enriched terms
        top_n = min(20, len(enrichment))
        top = enrichment[:top_n]

        y_pos = range(top_n)
        fold = [e["fold_enrichment"] for e in top]
        p_fdr = [e["p_fdr"] for e in top]

        # Color by significance
        colors = ["#2ca25f" if p < 0.05 else "#cccccc" for p in p_fdr]

        bars = ax_a.barh(y_pos, fold, color=colors, edgecolor="white")
        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels([e["term"] for e in top], fontsize=7)
        ax_a.set_xlabel("Fold Enrichment", fontsize=11)
        ax_a.set_title("A: Top Enriched GO BP Terms in Rescued Proteins",
                       fontsize=12, fontweight="bold")
        ax_a.grid(True, axis="x", alpha=0.3)
        ax_a.invert_yaxis()

        for i, (f, p, e) in enumerate(zip(fold, p_fdr, top)):
            sig = "*" if p < 0.05 else ""
            ax_a.text(f + 0.1, i, f"{f:.1f}x{sig} (k={e['k_rescue']})",
                      va="center", fontsize=7)

    # --- Panel B: Degree distribution ---
    ax_b = fig.add_subplot(gs[0, 1])
    r_deg = metrics["degree"]
    nr_deg = metrics_nr["degree"]

    ax_b.hist(nr_deg, bins=50, alpha=0.5, color="#cccccc",
              label=f"Non-rescued (n={len(nr_deg)})", density=True)
    ax_b.hist(r_deg, bins=30, alpha=0.7, color="#3182bd",
              label=f"Rescued (n={len(r_deg)})", density=True)
    ax_b.set_xlabel("Node Degree", fontsize=11)
    ax_b.set_ylabel("Density", fontsize=11)
    ax_b.set_title("B: Degree Distribution", fontsize=12, fontweight="bold")
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3)

    p_deg = topo_comp.get("degree", {}).get("p_value", 1)
    ax_b.text(0.95, 0.95, f"Mann-Whitney p = {p_deg:.4f}",
              transform=ax_b.transAxes, ha="right", va="top",
              fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat",
                                      alpha=0.8))

    # --- Panel C: Clustering coefficient ---
    ax_c = fig.add_subplot(gs[1, 0])
    r_clust = metrics["clustering"]
    nr_clust = metrics_nr["clustering"]

    ax_c.hist(nr_clust, bins=30, alpha=0.5, color="#cccccc",
              label=f"Non-rescued (n={len(nr_clust)})", density=True)
    ax_c.hist(r_clust, bins=20, alpha=0.7, color="#3182bd",
              label=f"Rescued (n={len(r_clust)})", density=True)
    ax_c.set_xlabel("Clustering Coefficient", fontsize=11)
    ax_c.set_ylabel("Density", fontsize=11)
    ax_c.set_title("C: Local Clustering Coefficient", fontsize=12,
                   fontweight="bold")
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)

    p_clust = topo_comp.get("clustering", {}).get("p_value", 1)
    ax_c.text(0.95, 0.95, f"p = {p_clust:.4f}",
              transform=ax_c.transAxes, ha="right", va="top",
              fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat",
                                      alpha=0.8))

    # --- Panel D: Summary statistics table ---
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    n_sig = sum(1 for e in enrichment if e.get("p_fdr", 1) < 0.05)
    table_data = [
        ["Rescued proteins", str(len(set(r["protein"] for r in rescued)))],
        ["Rescued trials", str(len(rescued))],
        ["PPI-only trials", str(n_categories.get("PPI-only", "-"))],
        ["Both found", str(n_categories.get("Both", "-"))],
        ["Miss (neither)", str(n_categories.get("Miss", "-"))],
        ["", ""],
        ["GO enrichment (FDR<0.05)", str(n_sig)],
        ["", ""],
        ["Median degree (rescued)",
         f"{topo_comp.get('degree', {}).get('rescue_median', '-'):.1f}"],
        ["Median degree (non-rescued)",
         f"{topo_comp.get('degree', {}).get('nonrescue_median', '-'):.1f}"],
        ["Degree p-value",
         f"{topo_comp.get('degree', {}).get('p_value', '-'):.4f}"],
        ["", ""],
        ["Median clustering (rescued)",
         f"{topo_comp.get('clustering', {}).get('rescue_median', '-'):.3f}"],
        ["Median clustering (non-rescued)",
         f"{topo_comp.get('clustering', {}).get('nonrescue_median', '-'):.3f}"],
        ["Clustering p-value",
         f"{topo_comp.get('clustering', {}).get('p_value', '-'):.4f}"],
    ]

    table = ax_d.table(cellText=table_data, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#e0e0e0")
        cell.set_edgecolor("white")
    ax_d.set_title("D: Summary Statistics", fontsize=12, fontweight="bold",
                   pad=20)

    fig.suptitle("Phase 15: Embedding-Rescued Protein Characterisation",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(FIGURES / "Fig76_rescue_characterisation.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig76_rescue_characterisation.png")


def plot_fig77_network_distance_analysis(rescued):
    """Fig77: Network distance of embedding-rescuing neighbors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect all network distances of rescuing neighbors
    net_dists = []
    emb_dists = []
    for r in rescued:
        for n in r["embedding_neighbors"]:
            if n["network_dist"] > 0:
                net_dists.append(n["network_dist"])
            emb_dists.append(n["embedding_dist"])

    # Panel A: Network distance histogram
    ax = axes[0]
    if net_dists:
        ax.hist(net_dists, bins=range(1, max(net_dists) + 2), alpha=0.7,
                color="#3182bd", edgecolor="white", align="left")
    ax.set_xlabel("Network Distance (hops)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("A: Network Distance of Rescuing Neighbors", fontsize=13,
                 fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Median line
    if net_dists:
        med = np.median(net_dists)
        ax.axvline(med, color="#e6550d", linewidth=2, linestyle="--",
                   label=f"Median = {med:.0f} hops")
        ax.legend(fontsize=10)

    # Panel B: Embedding distance vs network distance scatter
    ax2 = axes[1]
    if net_dists and emb_dists and len(net_dists) == len(emb_dists):
        ax2.scatter(net_dists, emb_dists, alpha=0.3, s=10, color="#3182bd")
        # Trend line
        if len(net_dists) > 10:
            z = np.polyfit(net_dists, emb_dists, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(net_dists), max(net_dists), 100)
            ax2.plot(x_line, p(x_line), "--", color="#e6550d", linewidth=2,
                     label=f"Trend (slope={z[0]:.3f})")
            rho, _ = stats.spearmanr(net_dists, emb_dists)
            ax2.text(0.95, 0.95, f"Spearman rho = {rho:.3f}",
                     transform=ax2.transAxes, ha="right", va="top",
                     fontsize=10, bbox=dict(boxstyle="round",
                                             facecolor="wheat", alpha=0.8))
            ax2.legend(fontsize=10)

    ax2.set_xlabel("Network Distance (hops)", fontsize=12)
    ax2.set_ylabel("Embedding Distance", fontsize=12)
    ax2.set_title("B: Embedding vs Network Distance", fontsize=13,
                  fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Phase 15: Rescuing Neighbor Distance Analysis",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig77_rescue_distance.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Fig77_rescue_distance.png")


def run():
    print(BANNER)
    print("Phase 15: Rescue Protein Characterisation")
    print("Who Are the Proteins Only Embeddings Can Find?")
    print(BANNER)

    np.random.seed(SEED)

    # Load data
    print("\n[1/6] Loading data...")
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

    # Load best method embedding (MDS had best fallback)
    print("\n[2/6] Loading MDS embedding (best fallback method)...")
    from utils import load_embedding
    coords, emb_nodes = load_embedding("MDS", "full", embeddings_dir=EMB)
    node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
    common = [n for n in emb_nodes if n in set(G.nodes())]
    indices = [node_to_idx[n] for n in common]
    filtered_coords = coords[indices]
    embeddings = {
        "MDS": {
            "coords": filtered_coords,
            "nodes": common,
            "node_to_idx": {n: i for i, n in enumerate(common)},
        }
    }
    print(f"  MDS: {len(common)} nodes")

    # Identify rescue trials
    print("\n[3/6] Identifying embedding-rescued trials...")
    rescued, ppi_only, both_found = identify_rescue_trials(
        embeddings, G, annotations, best_method="MDS"
    )

    rescue_proteins = sorted(set(r["protein"] for r in rescued))
    rescue_terms = Counter(r["term"] for r in rescued)
    print(f"\n  Rescued proteins: {len(rescue_proteins)}")
    print(f"  Rescued trials: {len(rescued)}")
    print(f"  Unique rescued terms: {len(rescue_terms)}")
    print(f"  Terms/protein: {len(rescued)/len(rescue_proteins):.1f}")

    n_categories = {
        "PPI-only": len(ppi_only),
        "Both": len(both_found),
        "Emb-rescue": len(rescued),
        "Miss": 12690 - len(ppi_only) - len(both_found) - len(rescued),
    }

    # GO enrichment
    print("\n[4/6] GO enrichment analysis...")
    all_terms_set = set(t for ts in annotations.values() for t in ts)
    enrichment = go_enrichment_analysis(
        rescue_proteins, annotations, all_terms_set
    )

    # Network topology
    print("\n[5/6] Network topology analysis...")
    metrics, metrics_nr, topo_comp = network_topology_analysis(
        rescue_proteins, G, annotations
    )

    # Summary
    print(f"\n{BANNER}")
    print("PHASE 15 RESULTS")
    print(BANNER)
    print(f"\n  Rescued proteins: {len(rescue_proteins)}")
    print(f"  Enriched GO terms (FDR<0.05): "
          f"{sum(1 for e in enrichment if e.get('p_fdr', 1) < 0.05)}")

    for key, comp in topo_comp.items():
        sig = "***" if comp["p_value"] < 0.001 else \
              "**" if comp["p_value"] < 0.01 else \
              "*" if comp["p_value"] < 0.05 else "ns"
        print(f"  {key}: rescue={comp['rescue_median']:.3f}, "
              f"non-rescue={comp['nonrescue_median']:.3f}, "
              f"p={comp['p_value']:.4f} {sig}")

    # Save results
    output = {
        "description": "Phase 15: Rescue Protein Characterisation",
        "n_rescue_proteins": len(rescue_proteins),
        "n_rescue_trials": len(rescued),
        "n_categories": n_categories,
        "rescue_proteins": rescue_proteins[:100],  # save top 100
        "enrichment_top50": enrichment[:50],
        "topology_comparison": topo_comp,
        "rescue_network_dists": {
            "median": float(np.median([n["network_dist"] for r in rescued
                                       for n in r["embedding_neighbors"]
                                       if n["network_dist"] > 0]))
            if rescued else 0,
            "mean": float(np.mean([n["network_dist"] for r in rescued
                                   for n in r["embedding_neighbors"]
                                   if n["network_dist"] > 0]))
            if rescued else 0,
        },
    }

    result_file = RESULTS / "rescue_protein_analysis.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {result_file.name}")

    # Figures
    print("\n  Generating figures...")
    plot_fig76_rescue_characterisation(
        enrichment, topo_comp, metrics, metrics_nr, rescued, n_categories
    )
    plot_fig77_network_distance_analysis(rescued)

    print(f"\n{BANNER}")
    print("Phase 15 complete.")
    print(BANNER)


if __name__ == "__main__":
    run()
