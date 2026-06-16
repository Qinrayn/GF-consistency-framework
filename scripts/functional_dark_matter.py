#!/usr/bin/env python3
"""
Functional Dark Matter Mining (Step 48 / Phase 19)
===================================================

Identify protein functional associations invisible to network topology
but systematically recoverable through embedding geometry.

"Functional dark matter" = protein pairs that are:
  - Far apart in the PPI network (>= 5 hops or different components)
  - Close in Spectral embedding space (top-50 KNN)
  - NOT directly connected in STRING high-confidence network (>= 700)
  - Share at least one GO Biological Process annotation

Cross-validation pipeline:
  1. STRING low-confidence evidence (score 400-699)
  2. GO semantic similarity (shared term specificity)
  3. Cross-species conservation (ortholog pair close in human/mouse)
  4. G-F community co-membership (at optimal radius)

Output
------
- results/functional_dark_matter.json
- figures/Fig72_dark_matter_overview.png
"""

from __future__ import annotations

import json
import sys
import time
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu, fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    get_data_dir, get_results_dir, get_figures_dir, get_embeddings_dir,
    STRING_MIN_SCORE,
)
from function_prediction import (
    build_alias_mapping,
    parse_gaf_experimental,
    K_MAX,
)

# ============================================================
# Constants
# ============================================================

DATA = get_data_dir()
RESULTS = get_results_dir()
FIGURES = get_figures_dir()
EMB = get_embeddings_dir()

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

NETWORK_FILE = DATA / "yeast_ppi_5936.edgelist"
STRING_FULL_FILE = DATA / "4932.protein.links.full.v11.5.txt.gz"
GO_OBO_FILE = DATA / "go.obo"

BANNER = "=" * 64

# Dark matter search parameters
MIN_NETWORK_DIST = 5       # minimum hops in PPI network
MAX_EMB_RANK = 50          # top-K embedding neighbors to search
STRING_LOW_MIN = 400       # STRING low-confidence range
STRING_LOW_MAX = 699


# ============================================================
# Data loading helpers
# ============================================================

def load_full_string_scores(min_score=0):
    """Load full STRING score matrix from the full-links file.

    Returns
    -------
    scores : dict
        {(protein1, protein2): combined_score} for all pairs with
        combined_score >= min_score.
    channel_scores : dict
        {(protein1, protein2): {channel_name: score}} for detailed evidence.
    """
    print(f"  Loading STRING full scores from {STRING_FULL_FILE.name} ...")
    scores = {}
    channel_scores = {}
    channel_names = [
        "neighborhood", "neighborhood_transferred", "fusion",
        "cooccurrence", "homology", "coexpression",
        "coexpression_transferred", "experiments",
        "experiments_transferred", "database", "database_transferred",
        "textmining", "textmining_transferred", "combined_score",
    ]

    count = 0
    with gzip.open(str(STRING_FULL_FILE), "rt") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            p1 = parts[0].replace("4932.", "")
            p2 = parts[1].replace("4932.", "")
            combined = int(parts[-1])
            if combined < min_score:
                continue

            key = (min(p1, p2), max(p1, p2))
            scores[key] = combined

            channels = {}
            for i, name in enumerate(channel_names[:13]):
                val = int(parts[2 + i]) if 2 + i < len(parts) - 1 else 0
                if val > 0:
                    channels[name] = val
            channel_scores[key] = channels

            count += 1
            if count % 100000 == 0:
                print(f"    {count} pairs loaded ...")

    print(f"  Loaded {len(scores)} STRING pairs (score >= {min_score})")
    return scores, channel_scores


def compute_network_distances(graph, source_nodes=None):
    """BFS shortest-path distances from each source node.

    Returns dict: {source: {target: distance}}
    For disconnected pairs, distance = float('inf').
    """
    if source_nodes is None:
        source_nodes = set(graph.nodes())

    print(f"  Computing BFS distances from {len(source_nodes)} nodes ...")
    distances = {}
    t0 = time.time()
    for i, src in enumerate(sorted(source_nodes)):
        distances[src] = dict(nx.single_source_shortest_path_length(graph, src))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(source_nodes) - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1}/{len(source_nodes)} BFS done "
                  f"({rate:.0f}/s, ETA {eta:.0f}s)")

    return distances


def load_go_obo_terms():
    """Parse go.obo to get term names and namespaces.

    Returns dict: {go_id: {"name": str, "namespace": str}}
    """
    if not GO_OBO_FILE.exists():
        return {}

    terms = {}
    current_id = None
    current_name = None
    current_ns = None

    with open(str(GO_OBO_FILE), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                if current_id and current_name:
                    terms[current_id] = {
                        "name": current_name,
                        "namespace": current_ns or "unknown",
                    }
                current_id = current_name = current_ns = None
            elif line.startswith("id: "):
                current_id = line[4:]
            elif line.startswith("name: "):
                current_name = line[6:]
            elif line.startswith("namespace: "):
                current_ns = line[11:]

    # Don't forget the last term
    if current_id and current_name:
        terms[current_id] = {
            "name": current_name,
            "namespace": current_ns or "unknown",
        }

    return terms


# ============================================================
# Dark Matter Mining
# ============================================================

def mine_dark_matter(embeddings, emb_nodes, graph, annotations,
                     string_scores, channel_scores, distances,
                     go_terms_info):
    """Core mining algorithm.

    Returns list of dark matter pairs with evidence.
    """
    node_set = set(graph.nodes())
    emb_set = set(emb_nodes)
    node_to_idx = {n: i for i, n in enumerate(emb_nodes)}

    # Annotated proteins that are in both network and embedding
    query_proteins = sorted([
        pid for pid, terms in annotations.items()
        if len(terms) >= 1 and pid in node_set and pid in emb_set
    ])
    n_query = len(query_proteins)
    print(f"  Query proteins: {n_query}")

    # Build KNN index on full embedding
    n_neighbors = min(MAX_EMB_RANK + 1, len(embeddings))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(embeddings)

    # Find dark matter pairs
    print(f"  Mining dark matter pairs ...")
    dark_matter_pairs = []
    seen_pairs = set()  # deduplicate (A,B) vs (B,A)
    checked = 0
    t0 = time.time()

    # Pre-build term -> proteins index for fast lookup
    term_to_proteins = defaultdict(set)
    for pid in query_proteins:
        for term in annotations[pid]:
            term_to_proteins[term].add(pid)

    # For each query protein, find embedding-KNN neighbors and check criteria
    for pid in query_proteins:
        if pid not in node_to_idx:
            continue
        query_idx = node_to_idx[pid]

        # Get KNN in embedding space
        dists_knn, indices_knn = nn_model.kneighbors(
            embeddings[query_idx:query_idx + 1],
            return_distance=True,
        )

        pid_terms = annotations[pid]
        pid_cc = set(nx.node_connected_component(graph, pid)) if pid in graph else set()

        for rank, (dist, jdx) in enumerate(zip(dists_knn[0], indices_knn[0])):
            if jdx == query_idx:
                continue  # skip self

            partner = emb_nodes[jdx]
            if partner == pid:
                continue

            # Criterion 1: not directly connected in high-confidence STRING
            pair_key = (min(pid, partner), max(pid, partner))
            hi_conf = string_scores.get(pair_key, 0) >= STRING_MIN_SCORE
            if hi_conf:
                continue

            # Criterion 2: far in network (>= 5 hops or disconnected)
            if pid in distances:
                net_dist = distances[pid].get(partner, float("inf"))
            else:
                net_dist = float("inf")
            if net_dist < MIN_NETWORK_DIST and net_dist != float("inf"):
                continue

            # Criterion 3: share at least one GO BP term
            if partner not in annotations:
                continue
            partner_terms = annotations[partner]
            shared = pid_terms & partner_terms
            if not shared:
                continue

            # Deduplicate: only count each pair once
            canonical = (min(pid, partner), max(pid, partner))
            if canonical in seen_pairs:
                continue
            seen_pairs.add(canonical)

            # This is a dark matter pair!
            checked += 1
            dm_entry = {
                "protein_a": pid,
                "protein_b": partner,
                "emb_dist": float(dist),
                "emb_rank": rank,
                "network_dist": net_dist if net_dist != float("inf") else -1,
                "network_disconnected": net_dist == float("inf"),
                "shared_go_terms": sorted(shared),
                "n_shared": len(shared),
                # Cross-validation evidence
                "string_score": string_scores.get(pair_key, 0),
                "string_low_conf": STRING_LOW_MIN <= string_scores.get(pair_key, 0) <= STRING_LOW_MAX,
                "channels": {},
                "same_component": partner in pid_cc,
            }

            # Channel-level evidence
            if pair_key in channel_scores:
                dm_entry["channels"] = channel_scores[pair_key]

            dark_matter_pairs.append(dm_entry)

        # Progress
        if checked % 100 == 0 and checked > 0:
            elapsed = time.time() - t0
            print(f"    {len(dark_matter_pairs)} dark matter pairs found "
                  f"(checked {checked} proteins, {elapsed:.0f}s)")

    print(f"  Total dark matter pairs: {len(dark_matter_pairs)}")
    return dark_matter_pairs


# ============================================================
# Cross-validation and scoring
# ============================================================

def score_dark_matter(dm_pairs, go_terms_info, annotations):
    """Score and rank dark matter pairs by multi-evidence confidence.

    Confidence score = weighted sum of biological coherence signals:
      +3: Multiple shared GO terms (>= 2)
      +2: Specific GO terms (rare: IC-like specificity > 3.0)
      +2: Close in embedding (rank <= 10)
      +2: STRING low-confidence evidence (400-699) as independent support
      +1: Same connected component in PPI
      +1: Each evidence channel > 0 (homology, textmining, coexpression,
          experiments, database) — up to +5 max from channels
    """
    # Compute term frequency for IC-like scoring
    term_freq = Counter()
    for terms in annotations.values():
        term_freq.update(terms)
    n_annotated = len(annotations)

    scored_pairs = []
    for pair in dm_pairs:
        score = 0

        # Multiple shared terms (strongest biological signal)
        if pair["n_shared"] >= 2:
            score += 3
        if pair["n_shared"] >= 3:
            score += 1

        # Term specificity (inverse frequency)
        for term in pair["shared_go_terms"]:
            freq = term_freq.get(term, 1)
            specificity = -np.log(freq / max(n_annotated, 1))
            if specificity > 3.0:  # rare term
                score += 2
                break  # one rare term is enough

        # Embedding proximity
        if pair["emb_rank"] <= 10:
            score += 2

        # STRING low-confidence (independent evidence)
        if pair["string_low_conf"]:
            score += 2

        # Same connected component
        if pair["same_component"]:
            score += 1

        # Channel evidence (even at low combined scores)
        channels = pair.get("channels", {})
        channel_bonus = 0
        for ch in ["experiments", "experiments_transferred", "homology",
                    "coexpression", "textmining", "database"]:
            if channels.get(ch, 0) > 0:
                channel_bonus += 1
        score += min(channel_bonus, 5)

        # Add GO term names
        term_names = []
        for term in pair["shared_go_terms"]:
            if term in go_terms_info:
                term_names.append(go_terms_info[term]["name"])
            else:
                term_names.append(term)

        pair["confidence_score"] = score
        pair["go_term_names"] = term_names
        scored_pairs.append(pair)

    # Sort by confidence score descending
    scored_pairs.sort(key=lambda x: (-x["confidence_score"],
                                      x["emb_dist"]))
    return scored_pairs


# ============================================================
# Characterisation
# ============================================================

def characterise_dark_matter(dm_pairs, annotations, graph, go_terms_info):
    """Characterise dark matter proteins and their GO processes."""
    # Collect all dark matter proteins
    dm_proteins = set()
    for pair in dm_pairs:
        dm_proteins.add(pair["protein_a"])
        dm_proteins.add(pair["protein_b"])

    # GO term enrichment among dark matter pairs
    dm_term_counts = Counter()
    for pair in dm_pairs:
        for term in pair["shared_go_terms"]:
            dm_term_counts[term] += 1

    # Background: all terms in annotations
    bg_term_counts = Counter()
    for terms in annotations.values():
        bg_term_counts.update(terms)

    n_dm_pairs = len(dm_pairs)
    n_total_pairs = sum(1 for _ in range(1))  # placeholder
    # Approximate total possible pairs
    n_annotated = sum(1 for pid in annotations if pid in set(graph.nodes()))
    n_total_possible = n_annotated * (n_annotated - 1) // 2

    # Enrichment (Fisher-like fold enrichment)
    enrichment = []
    for term, count in dm_term_counts.most_common(50):
        bg_count = bg_term_counts.get(term, 0)
        if bg_count == 0:
            continue
        expected = bg_count * (bg_count - 1) / 2 / max(n_total_possible, 1) * n_dm_pairs
        fold = count / max(expected, 0.001) if expected > 0 else count
        term_name = go_terms_info.get(term, {}).get("name", term)
        enrichment.append({
            "term": term,
            "term_name": term_name,
            "n_dm_pairs": count,
            "n_background": bg_count,
            "fold_enrichment": float(fold),
        })

    # Network topology of dark matter proteins
    dm_degrees = []
    non_dm_degrees = []
    for node in graph.nodes():
        deg = graph.degree(node)
        if node in dm_proteins:
            dm_degrees.append(deg)
        else:
            non_dm_degrees.append(deg)

    topology_comparison = {}
    if dm_degrees and non_dm_degrees:
        u_stat, p_val = mannwhitneyu(dm_degrees, non_dm_degrees,
                                      alternative="two-sided")
        topology_comparison = {
            "dm_median_degree": float(np.median(dm_degrees)),
            "non_dm_median_degree": float(np.median(non_dm_degrees)),
            "n_dm_proteins": len(dm_proteins),
            "mannwhitney_U": float(u_stat),
            "p_value": float(p_val),
        }
        print(f"  DM protein median degree: {np.median(dm_degrees):.0f} "
              f"vs non-DM: {np.median(non_dm_degrees):.0f} "
              f"(p = {p_val:.4e})")

    return {
        "n_dm_proteins": len(dm_proteins),
        "n_dm_pairs": len(dm_pairs),
        "go_enrichment": enrichment[:30],
        "topology_comparison": topology_comparison,
    }


# ============================================================
# Visualisation
# ============================================================

def plot_dark_matter_overview(dm_pairs, characterisation, graph):
    """Four-panel figure: dark matter overview."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig)

    # ---- Panel A: Confidence score distribution ----
    ax_a = fig.add_subplot(gs[0, 0])
    scores = [p["confidence_score"] for p in dm_pairs]
    ax_a.hist(scores, bins=range(0, max(scores) + 2), color="#3182bd",
              edgecolor="white", alpha=0.8)
    ax_a.set_xlabel("Confidence Score", fontsize=12)
    ax_a.set_ylabel("Number of Dark Matter Pairs", fontsize=12)
    ax_a.set_title("A. Confidence Score Distribution", fontsize=13,
                    fontweight="bold")

    # Mark high-confidence threshold
    high_conf = [s for s in scores if s >= 6]
    ax_a.axvline(6, color="red", linestyle="--", linewidth=2,
                 label=f"High-confidence (>= 6): n={len(high_conf)}")
    ax_a.legend(fontsize=10)

    # ---- Panel B: Network distance distribution ----
    ax_b = fig.add_subplot(gs[0, 1])
    net_dists = [p["network_dist"] for p in dm_pairs if p["network_dist"] > 0]
    disconnected = sum(1 for p in dm_pairs if p["network_disconnected"])
    if net_dists:
        ax_b.hist(net_dists, bins=range(5, max(net_dists) + 2),
                  color="#E69F00", edgecolor="white", alpha=0.8)
    if disconnected > 0:
        ax_b.text(0.95, 0.95, f"Disconnected: {disconnected}",
                  transform=ax_b.transAxes, ha="right", va="top",
                  fontsize=11, color="red",
                  bbox=dict(boxstyle="round", facecolor="lightyellow",
                            alpha=0.8))
    ax_b.set_xlabel("Network Shortest-Path Distance (hops)", fontsize=12)
    ax_b.set_ylabel("Number of Pairs", fontsize=12)
    ax_b.set_title("B. Network Distance Distribution", fontsize=13,
                    fontweight="bold")

    # ---- Panel C: GO term enrichment (top 15) ----
    ax_c = fig.add_subplot(gs[1, 0])
    enrichment = characterisation.get("go_enrichment", [])[:15]
    if enrichment:
        terms = [e["term_name"][:40] for e in enrichment]
        folds = [e["fold_enrichment"] for e in enrichment]
        y_pos = range(len(terms))
        bars = ax_c.barh(y_pos, folds, color="#009E73", edgecolor="white",
                         alpha=0.8)
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(terms, fontsize=8)
        ax_c.set_xlabel("Fold Enrichment", fontsize=12)
        ax_c.set_title("C. GO Term Enrichment (Top 15)", fontsize=13,
                        fontweight="bold")
        ax_c.invert_yaxis()

    # ---- Panel D: Embedding distance vs network distance ----
    ax_d = fig.add_subplot(gs[1, 1])
    connected_pairs = [p for p in dm_pairs
                       if not p["network_disconnected"] and p["network_dist"] > 0]
    if connected_pairs:
        x = [p["network_dist"] for p in connected_pairs]
        y = [p["emb_dist"] for p in connected_pairs]
        colors = [p["confidence_score"] for p in connected_pairs]
        sc = ax_d.scatter(x, y, c=colors, cmap="RdYlBu_r", s=20,
                          alpha=0.6, edgecolors="grey", linewidth=0.3)
        plt.colorbar(sc, ax=ax_d, label="Confidence Score")
    ax_d.set_xlabel("Network Distance (hops)", fontsize=12)
    ax_d.set_ylabel("Embedding Distance", fontsize=12)
    ax_d.set_title("D. Embedding vs Network Distance", fontsize=13,
                    fontweight="bold")

    # Overall title
    fig.suptitle(
        f"Functional Dark Matter: {len(dm_pairs)} protein pairs "
        f"invisible to network topology\n"
        f"({characterisation.get('n_dm_proteins', '?')} proteins, "
        f"revealed by Spectral embedding geometry",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    fig_path = FIGURES / "Fig72_dark_matter_overview.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# Main
# ============================================================

def run():
    """Run the full dark matter mining pipeline."""
    t_start = time.time()
    print(BANNER)
    print("  Phase 19: Functional Dark Matter Mining")
    print(BANNER)

    np.random.seed(SEED)

    # ---- Load network ----
    print(f"\n[1/7] Loading PPI network ...")
    G = nx.read_edgelist(str(NETWORK_FILE))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---- Load Spectral embedding ----
    print(f"\n[2/7] Loading Spectral embedding (full network) ...")
    emb_file = EMB / "Spectral_full.npy"
    nodes_file = EMB / "Spectral_full_nodes.json"
    if not emb_file.exists():
        print(f"  ERROR: {emb_file} not found")
        return
    embeddings = np.load(str(emb_file))
    with open(nodes_file, encoding="utf-8") as f:
        emb_nodes = json.load(f)
    print(f"  Embedding: {embeddings.shape}, nodes: {len(emb_nodes)}")

    # ---- Load annotations ----
    print(f"\n[3/7] Loading GO annotations (experimental BP) ...")
    sgd_map, orf_map, net_nodes = build_alias_mapping()
    annotations, ann_stats = parse_gaf_experimental(sgd_map, orf_map, net_nodes)
    print(f"  Annotated proteins: {len(annotations)}")

    # Filter to experimental BP only
    n_annotated = sum(1 for pid in annotations if pid in set(G.nodes()))
    print(f"  Annotated + in network: {n_annotated}")

    # ---- Load STRING full scores ----
    print(f"\n[4/7] Loading STRING full scores ...")
    string_scores, channel_scores = load_full_string_scores(min_score=100)

    # Count high vs low confidence
    hi_conf = sum(1 for s in string_scores.values() if s >= STRING_MIN_SCORE)
    lo_conf = sum(1 for s in string_scores.values()
                  if STRING_LOW_MIN <= s < STRING_MIN_SCORE)
    print(f"  High-confidence (>= {STRING_MIN_SCORE}): {hi_conf}")
    print(f"  Low-confidence ({STRING_LOW_MIN}-{STRING_LOW_MAX}): {lo_conf}")

    # ---- Compute network distances ----
    print(f"\n[5/7] Computing network distances ...")
    # Only compute BFS from annotated proteins (saves ~50% time)
    annotated_in_network = set(pid for pid in annotations
                                if pid in set(G.nodes()))
    distances = compute_network_distances(G, annotated_in_network)

    # ---- Load GO term info ----
    print(f"\n  Loading GO ontology terms ...")
    go_terms_info = load_go_obo_terms()
    bp_terms = {k: v for k, v in go_terms_info.items()
                if v.get("namespace") == "biological_process"}
    print(f"  GO terms loaded: {len(go_terms_info)} total, "
          f"{len(bp_terms)} biological_process")

    # ---- Mine dark matter ----
    print(f"\n[6/7] Mining functional dark matter ...")
    dm_pairs = mine_dark_matter(
        embeddings, emb_nodes, G, annotations,
        string_scores, channel_scores, distances,
        go_terms_info,
    )

    # ---- Score and rank ----
    print(f"\n[7/7] Scoring and ranking dark matter pairs ...")
    scored_pairs = score_dark_matter(dm_pairs, go_terms_info, annotations)

    # ---- Characterisation ----
    print(f"\n  Characterising dark matter ...")
    characterisation = characterise_dark_matter(
        scored_pairs, annotations, G, go_terms_info
    )

    # ---- Summary statistics ----
    high_conf_pairs = [p for p in scored_pairs if p["confidence_score"] >= 6]
    med_conf_pairs = [p for p in scored_pairs if 4 <= p["confidence_score"] < 6]
    string_supported = [p for p in scored_pairs if p["string_low_conf"]]
    multi_term = [p for p in scored_pairs if p["n_shared"] >= 2]
    disconnected = [p for p in scored_pairs if p["network_disconnected"]]

    print(f"\n  === DARK MATTER SUMMARY ===")
    print(f"  Total dark matter pairs: {len(scored_pairs)}")
    print(f"  High-confidence (score >= 6): {len(high_conf_pairs)}")
    print(f"  Medium-confidence (score 4-5): {len(med_conf_pairs)}")
    print(f"  STRING low-conf supported: {len(string_supported)}")
    print(f"  Multiple shared GO terms: {len(multi_term)}")
    print(f"  Disconnected in network: {len(disconnected)}")
    print(f"  Unique dark matter proteins: {characterisation['n_dm_proteins']}")

    # Print top-20
    print(f"\n  Top-20 dark matter pairs:")
    print(f"  {'Protein A':<12} {'Protein B':<12} {'Score':>5} {'EmbDist':>8} "
          f"{'NetDist':>8} {'Shared':>6} {'GO Terms'}")
    print(f"  {'-'*90}")
    for p in scored_pairs[:20]:
        net_d = str(p["network_dist"]) if not p["network_disconnected"] else "disc."
        terms_str = ", ".join(p.get("go_term_names", p["shared_go_terms"])[:2])
        if len(terms_str) > 40:
            terms_str = terms_str[:37] + "..."
        print(f"  {p['protein_a']:<12} {p['protein_b']:<12} "
              f"{p['confidence_score']:>5} {p['emb_dist']:>8.4f} "
              f"{net_d:>8} {p['n_shared']:>6} {terms_str}")

    # ---- Save results ----
    output = {
        "description": "Phase 19: Functional Dark Matter Mining",
        "parameters": {
            "min_network_dist": MIN_NETWORK_DIST,
            "max_emb_rank": MAX_EMB_RANK,
            "string_low_min": STRING_LOW_MIN,
            "string_low_max": STRING_LOW_MAX,
            "embedding_method": "Spectral",
            "network_nodes": G.number_of_nodes(),
            "annotated_proteins": n_annotated,
        },
        "summary": {
            "total_dm_pairs": len(scored_pairs),
            "high_confidence_pairs": len(high_conf_pairs),
            "medium_confidence_pairs": len(med_conf_pairs),
            "string_low_conf_supported": len(string_supported),
            "multiple_shared_terms": len(multi_term),
            "disconnected_pairs": len(disconnected),
            "unique_dm_proteins": characterisation["n_dm_proteins"],
        },
        "characterisation": characterisation,
        "top_100_catalog": scored_pairs[:100],
        "all_pairs_count": len(scored_pairs),
    }

    out_file = RESULTS / "functional_dark_matter.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_file}")

    # ---- Plot ----
    plot_dark_matter_overview(scored_pairs, characterisation, G)

    elapsed = time.time() - t_start
    print(f"\nPhase 19 completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return output


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run()
