#!/usr/bin/env python3
"""
G-F Consistency Framework — Temporal / Dynamic Network Interface
=================================================================
Provides a framework for analysing PPI networks that change over time
(e.g., cell-cycle stages, drug treatment timepoints, disease progression).

A ``TemporalNetwork`` is a sequence of static graph snapshots, each
associated with a time label. The G-F consistency analysis can be
applied to each snapshot independently to track how geometric-functional
consistency evolves over time.

This module provides:
- ``TemporalNetwork``: Container for time-stamped graph snapshots
- ``load_temporal_edgelist``: Parse edge lists with timestamps
- ``temporal_gf_analysis``: Run G-F curves on each snapshot
- ``temporal_consistency_score``: Measure G-F stability across time

This is a **framework skeleton** — the full temporal analysis requires
time-resolved PPI datasets (e.g., from time-course AP-MS experiments).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal Network Container
# ---------------------------------------------------------------------------

@dataclass
class TemporalSnapshot:
    """A single timepoint of a dynamic network."""
    time: float
    graph: nx.Graph
    label: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()


@dataclass
class TemporalNetwork:
    """Ordered sequence of graph snapshots over time.

    Supports iteration, indexing, and basic temporal queries.
    """
    snapshots: list[TemporalSnapshot] = field(default_factory=list)
    name: str = ""
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx) -> TemporalSnapshot:
        return self.snapshots[idx]

    def __iter__(self):
        return iter(self.snapshots)

    @property
    def times(self) -> list[float]:
        return [s.time for s in self.snapshots]

    def add_snapshot(self, time: float, graph: nx.Graph,
                     label: str = "", **metadata) -> None:
        """Append a snapshot at the given timepoint."""
        snap = TemporalSnapshot(time=time, graph=graph, label=label,
                                metadata=metadata)
        self.snapshots.append(snap)
        # Keep sorted by time
        self.snapshots.sort(key=lambda s: s.time)

    def get_window(self, t_start: float, t_end: float) -> "TemporalNetwork":
        """Return a sub-sequence within the time window [t_start, t_end]."""
        filtered = [s for s in self.snapshots if t_start <= s.time <= t_end]
        return TemporalNetwork(snapshots=filtered, name=self.name + "_window")

    def node_persistence(self) -> dict[str, list[float]]:
        """Track which nodes appear at which timepoints.

        Returns
        -------
        dict mapping node_id -> list of times where the node appears
        """
        persistence: dict[str, list[float]] = {}
        for snap in self.snapshots:
            for node in snap.graph.nodes():
                if node not in persistence:
                    persistence[node] = []
                persistence[node].append(snap.time)
        return persistence

    def edge_turnover(self) -> dict[str, dict]:
        """Compute edge appearance/disappearance between consecutive snapshots.

        Returns
        -------
        dict with keys: 'appeared', 'disappeared', 'persistent' per transition
        """
        results = []
        for i in range(1, len(self.snapshots)):
            prev_edges = set(
                tuple(sorted(e)) for e in self.snapshots[i - 1].graph.edges()
            )
            curr_edges = set(
                tuple(sorted(e)) for e in self.snapshots[i].graph.edges()
            )
            results.append({
                "t_prev": self.snapshots[i - 1].time,
                "t_curr": self.snapshots[i].time,
                "appeared": list(curr_edges - prev_edges),
                "disappeared": list(prev_edges - curr_edges),
                "persistent": list(prev_edges & curr_edges),
                "jaccard": len(prev_edges & curr_edges) / len(prev_edges | curr_edges)
                           if (prev_edges | curr_edges) else 1.0,
            })
        return results


# ---------------------------------------------------------------------------
# I/O: Temporal edgelist format
# ---------------------------------------------------------------------------

def load_temporal_edgelist(
    filepath: Path,
    time_column: int = 2,
    node1_column: int = 0,
    node2_column: int = 1,
    delimiter: str = "\t",
    has_header: bool = True,
    min_score_column: Optional[int] = None,
    min_score: int = 0,
) -> TemporalNetwork:
    """Parse a time-stamped edgelist into a TemporalNetwork.

    Expected format (tab-separated by default):
        node1   node2   time   [optional: score]

    Parameters
    ----------
    filepath : path to edgelist file
    time_column : 0-indexed column for the timestamp
    node1_column, node2_column : node identifier columns
    delimiter : column separator
    has_header : whether the first line is a header
    min_score_column : optional column for edge confidence score
    min_score : minimum score to include an edge

    Returns
    -------
    TemporalNetwork with one snapshot per unique timestamp
    """
    filepath = Path(filepath)
    time_edges: dict[float, list[tuple[str, str]]] = {}

    with open(filepath, encoding="utf-8") as f:
        if has_header:
            next(f)
        for line in f:
            parts = line.strip().split(delimiter)
            if len(parts) < max(time_column, node1_column, node2_column) + 1:
                continue

            u, v = parts[node1_column], parts[node2_column]
            t = float(parts[time_column])

            if min_score_column is not None:
                score = float(parts[min_score_column])
                if score < min_score:
                    continue

            if t not in time_edges:
                time_edges[t] = []
            time_edges[t].append((u, v))

    tn = TemporalNetwork(name=filepath.stem)
    for t in sorted(time_edges.keys()):
        G = nx.Graph()
        G.add_edges_from(time_edges[t])
        tn.add_snapshot(t, G, label=f"t={t}")

    logger.info(
        "Loaded temporal network: %d snapshots from %s",
        len(tn), filepath.name,
    )
    return tn


# ---------------------------------------------------------------------------
# Temporal G-F Analysis
# ---------------------------------------------------------------------------

def temporal_gf_analysis(
    temporal_net: TemporalNetwork,
    go_map: dict,
    embedding_method: str = "Spectral",
    r_min: float = 0.05,
    r_max: float = 0.55,
    n_points: int = 100,
) -> list[dict]:
    """Run G-F curve analysis on each temporal snapshot.

    Parameters
    ----------
    temporal_net : temporal network
    go_map : gene -> GO terms mapping
    embedding_method : which embedding to compute per snapshot
    r_min, r_max, n_points : G-F grid parameters

    Returns
    -------
    list of dicts, one per snapshot, with keys:
        time, n_nodes, n_edges, gf_score, peak_purity, plateau_width
    """
    from scripts.utils import (
        compute_gf_curve, compute_gf_score, compute_plateau_width,
        spectral_embedding_from_graph, rescale_coordinates,
    )

    r_vals = np.linspace(r_min, r_max, n_points)
    results = []

    for snap in temporal_net:
        G = snap.graph
        nodes = sorted(G.nodes())
        if len(nodes) < 5:
            results.append({
                "time": snap.time, "n_nodes": len(nodes),
                "n_edges": snap.n_edges, "gf_score": 0.0,
                "peak_purity": 0.0, "plateau_width": 0.0,
                "skipped": True,
            })
            continue

        # Filter to nodes with GO annotations
        valid = [n for n in nodes if n in go_map]
        if len(valid) < 5:
            results.append({
                "time": snap.time, "n_nodes": len(valid),
                "n_edges": snap.n_edges, "gf_score": 0.0,
                "peak_purity": 0.0, "plateau_width": 0.0,
                "skipped": True,
            })
            continue

        G_sub = G.subgraph(valid).copy()

        # Embedding
        coords = spectral_embedding_from_graph(G_sub, nodelist=valid)
        coords = rescale_coordinates(coords)

        # G-F curve
        purities, _ = compute_gf_curve(coords, valid, go_map, r_vals)
        gf_score = compute_gf_score(r_vals, purities)
        pw = compute_plateau_width(r_vals, purities)

        results.append({
            "time": snap.time,
            "n_nodes": len(valid),
            "n_edges": G_sub.number_of_edges(),
            "gf_score": round(gf_score, 4),
            "peak_purity": round(pw["peak_purity"], 4),
            "plateau_width": round(pw["W"], 4),
            "skipped": False,
        })

        logger.info(
            "t=%.2f: %d nodes, G-F=%.4f, W=%.4f",
            snap.time, len(valid), gf_score, pw["W"],
        )

    return results


def temporal_consistency_score(results: list[dict]) -> dict:
    """Measure how stable G-F scores are across timepoints.

    Parameters
    ----------
    results : output from temporal_gf_analysis

    Returns
    -------
    dict with stability metrics
    """
    scores = [r["gf_score"] for r in results if not r.get("skipped")]
    if len(scores) < 2:
        return {"mean_gf": 0.0, "std_gf": 0.0, "cv_gf": 0.0,
                "trend_slope": 0.0, "n_valid_snapshots": len(scores)}

    scores = np.array(scores)
    mean_gf = float(np.mean(scores))
    std_gf = float(np.std(scores, ddof=1))
    cv_gf = std_gf / mean_gf if mean_gf > 1e-10 else 0.0

    # Linear trend
    times = np.array([r["time"] for r in results if not r.get("skipped")])
    if len(times) >= 2:
        slope, intercept = np.polyfit(times, scores, 1)
        trend_slope = float(slope)
    else:
        trend_slope = 0.0

    return {
        "mean_gf": round(mean_gf, 4),
        "std_gf": round(std_gf, 4),
        "cv_gf": round(cv_gf, 4),
        "trend_slope": round(trend_slope, 6),
        "n_valid_snapshots": len(scores),
    }
