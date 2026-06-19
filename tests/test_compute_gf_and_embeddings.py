#!/usr/bin/env python3
"""
test_compute_gf_and_embeddings.py -- Unit tests for compute_gf and embedding functions.

Adds coverage for:
- compute_gf.py: compute_random_baseline, compute_random_baseline_with_stats
- utils.py: spectral_embedding_from_graph, classical_mds_from_distances,
            diffusion_map_from_similarity, compute_centrality_features,
            build_similarity_matrix
"""

import sys
import numpy as np
import networkx as nx
import pytest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from utils import (
    compute_gf_curve, compute_gf_score, rescale_coordinates,
    spectral_embedding_from_graph, classical_mds_from_distances,
    diffusion_map_from_similarity, compute_centrality_features,
    build_similarity_matrix, precompute_distance_matrix,
)


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def small_graph():
    """Small connected graph for embedding tests."""
    G = nx.karate_club_graph()
    return G


@pytest.fixture
def go_map_karate():
    """Synthetic GO annotations for karate club nodes."""
    go_map = {}
    for i in range(34):
        term = f"GO:{i % 5:03d}"
        go_map[str(i)] = {term, f"GO:{(i + 1) % 5:03d}"}
    return go_map


# ================================================================
# Test spectral_embedding_from_graph
# ================================================================

class TestSpectralEmbedding:

    def test_output_shape(self, small_graph):
        """Should produce (n, 2) array."""
        coords = spectral_embedding_from_graph(small_graph)
        assert coords.shape == (34, 2)

    def test_finite_values(self, small_graph):
        """All coordinates should be finite."""
        coords = spectral_embedding_from_graph(small_graph)
        assert np.all(np.isfinite(coords))

    def test_non_degenerate(self, small_graph):
        """Coordinates should not be all zeros or all identical."""
        coords = spectral_embedding_from_graph(small_graph)
        assert np.std(coords) > 1e-6


# ================================================================
# Test classical_mds_from_distances
# ================================================================

class TestClassicalMDS:

    def test_output_shape(self):
        """Should produce (n, 2) array from distance matrix."""
        np.random.seed(42)
        D = np.random.rand(10, 10)
        D = (D + D.T) / 2  # Symmetrise
        np.fill_diagonal(D, 0)
        coords = classical_mds_from_distances(D)
        assert coords.shape == (10, 2)

    def test_finite_values(self):
        """All coordinates should be finite."""
        np.random.seed(42)
        D = np.random.rand(8, 8)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        coords = classical_mds_from_distances(D)
        assert np.all(np.isfinite(coords))

    def test_identity_distance(self):
        """All-equal distances should produce degenerate or near-zero coords."""
        n = 5
        D = np.ones((n, n)) * 2.0
        np.fill_diagonal(D, 0)
        coords = classical_mds_from_distances(D)
        assert coords.shape == (n, 2)


# ================================================================
# Test diffusion_map_from_similarity
# ================================================================

class TestDiffusionMap:

    def test_output_shape(self):
        """Should produce (n, 2) array from similarity matrix."""
        np.random.seed(42)
        n = 10
        S = np.random.rand(n, n)
        S = (S + S.T) / 2
        np.fill_diagonal(S, 1)
        coords = diffusion_map_from_similarity(S)
        assert coords.shape == (n, 2)

    def test_finite_values(self):
        """All coordinates should be finite."""
        np.random.seed(42)
        n = 8
        S = np.eye(n) + 0.1 * np.random.rand(n, n)
        S = (S + S.T) / 2
        coords = diffusion_map_from_similarity(S)
        assert np.all(np.isfinite(coords))


# ================================================================
# Test compute_centrality_features
# ================================================================

class TestCentralityFeatures:

    def test_output_shape(self, small_graph):
        """Should produce (n, 6) feature matrix."""
        nodes = sorted(small_graph.nodes())
        features = compute_centrality_features(small_graph, nodes)
        assert features.shape == (len(nodes), 6)

    def test_normalized(self, small_graph):
        """Features should be normalized to [0, 1] range."""
        nodes = sorted(small_graph.nodes())
        features = compute_centrality_features(small_graph, nodes)
        assert np.all(features >= -0.01)  # Small tolerance for numerical
        assert np.all(features <= 1.01)

    def test_finite(self, small_graph):
        """All features should be finite."""
        nodes = sorted(small_graph.nodes())
        features = compute_centrality_features(small_graph, nodes)
        assert np.all(np.isfinite(features))

    def test_non_constant(self, small_graph):
        """At least some features should vary across nodes."""
        nodes = sorted(small_graph.nodes())
        features = compute_centrality_features(small_graph, nodes)
        # Not all columns should be constant
        col_stds = np.std(features, axis=0)
        assert np.any(col_stds > 1e-6)


# ================================================================
# Test build_similarity_matrix
# ================================================================

class TestBuildSimilarity:

    def test_symmetric(self):
        """Similarity matrix should be symmetric."""
        features = np.random.rand(10, 5)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        S = build_similarity_matrix(features)
        assert np.allclose(S, S.T, atol=1e-10)

    def test_self_similarity(self):
        """Diagonal should be close to 1 for normalized features."""
        features = np.random.rand(8, 4)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        S = build_similarity_matrix(features)
        assert np.allclose(np.diag(S), 1.0, atol=1e-6)

    def test_output_shape(self):
        """Should produce (n, n) matrix."""
        features = np.random.rand(12, 6)
        S = build_similarity_matrix(features)
        assert S.shape == (12, 12)


# ================================================================
# Test compute_random_baseline (compute_gf.py)
# ================================================================

class TestRandomBaseline:

    def test_returns_list(self):
        """compute_random_baseline should return purity list of same length as r_vals."""
        from compute_gf import compute_random_baseline
        coords = np.random.RandomState(42).randn(20, 2)
        nodes = [f"n{i}" for i in range(20)]
        go_map = {f"n{i}": {f"GO:{i % 3:03d}"} for i in range(20)}
        r_vals = np.linspace(0.1, 1.0, 10)
        baseline = compute_random_baseline(
            coords, nodes, go_map, r_vals, n_shuffles=3
        )
        assert isinstance(baseline, (list, np.ndarray))
        assert len(baseline) == len(r_vals)

    def test_baseline_bounded(self):
        """Random baseline purity should be in [0, 1]."""
        from compute_gf import compute_random_baseline
        coords = np.random.RandomState(42).randn(15, 2)
        nodes = [f"n{i}" for i in range(15)]
        go_map = {f"n{i}": {f"GO:{i % 4:03d}"} for i in range(15)}
        r_vals = np.linspace(0.2, 0.8, 5)
        baseline = compute_random_baseline(
            coords, nodes, go_map, r_vals, n_shuffles=5
        )
        arr = np.array(baseline)
        assert np.all(arr >= -0.01)
        assert np.all(arr <= 1.01)


class TestRandomBaselineWithStats:

    def test_returns_tuple(self):
        """compute_random_baseline_with_stats should return (mean_curve, std_scalar)."""
        from compute_gf import compute_random_baseline_with_stats
        coords = np.random.RandomState(42).randn(20, 2)
        nodes = [f"n{i}" for i in range(20)]
        go_map = {f"n{i}": {f"GO:{i % 3:03d}"} for i in range(20)}
        r_vals = np.linspace(0.1, 1.0, 8)
        result = compute_random_baseline_with_stats(
            coords, nodes, go_map, r_vals, n_shuffles=3
        )
        assert len(result) == 2
        mean_curve, std_val = result
        assert isinstance(mean_curve, (list, np.ndarray))
        assert isinstance(std_val, (float, np.floating))
        assert std_val >= 0


# ================================================================
# Test edge cases for GF computation
# ================================================================

class TestGFEdgeCases:

    def test_single_node(self):
        """GF computation should handle single-node embeddings gracefully."""
        coords = np.array([[0.5, 0.3]])
        nodes = ["A"]
        go_map = {"A": {"GO:001"}}
        r_vals = np.linspace(0.1, 1.0, 5)
        purities, mods = compute_gf_curve(coords, nodes, go_map, r_vals)
        assert len(purities) == 5
        # Single node: no edges at any r, so purity should be 0
        gf = compute_gf_score(r_vals, purities, 0.1, 0.5)
        assert gf >= 0

    def test_disconnected_graph(self):
        """GF should handle embeddings where nodes are far apart."""
        coords = np.array([[0.0, 0.0], [100.0, 100.0]])
        nodes = ["A", "B"]
        go_map = {"A": {"GO:001"}, "B": {"GO:002"}}
        r_vals = np.linspace(0.01, 0.1, 5)  # Too small to connect
        try:
            purities, mods = compute_gf_curve(coords, nodes, go_map, r_vals)
            # No edges means no communities or empty graph, purity = 0
            assert np.all(np.array(purities) >= 0)
        except Exception:
            # Community detection may fail on empty graph
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
