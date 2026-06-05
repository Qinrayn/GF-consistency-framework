"""Tests for scripts/utils.py — core computation functions.

These tests use small synthetic graphs to verify the G-F framework's
mathematical correctness without requiring the full yeast PPI dataset.
"""

import numpy as np
import networkx as nx
import pytest

from scripts.utils import (
    _community_purity,
    functional_purity,
    functional_purity_named,
    compute_gf_curve,
    compute_gf_score,
    compute_plateau_width,
    precompute_distance_matrix,
    build_spatial_graph_fast,
    rescale_coordinates,
    check_embedding_collapse,
    align_embedding_to_nodes,
    PLATEAU_RELATIVE_THRESHOLD,
)


# ===================================================================
# _community_purity
# ===================================================================

class TestCommunityPurity:
    """Functional purity for a single community."""

    def test_uniform_community(self):
        """All genes share the same GO term -> purity = 1.0."""
        go_map = {"a": ["GO:1"], "b": ["GO:1"], "c": ["GO:1"]}
        assert _community_purity(["a", "b", "c"], go_map) == pytest.approx(1.0)

    def test_mixed_community(self):
        """2 of 3 terms are GO:1 -> purity = 2/3."""
        go_map = {"a": ["GO:1"], "b": ["GO:1"], "c": ["GO:2"]}
        assert _community_purity(["a", "b", "c"], go_map) == pytest.approx(2.0 / 3.0)

    def test_no_annotations(self, empty_go_map):
        """No GO terms at all -> purity = 0.0."""
        assert _community_purity(["x", "y"], empty_go_map) == 0.0

    def test_single_node(self):
        go_map = {"a": ["GO:1", "GO:2", "GO:3"]}
        # 1 most common / 3 total = 1/3
        assert _community_purity(["a"], go_map) == pytest.approx(1.0 / 3.0)

    def test_multiple_terms_per_gene(self):
        """Gene has multiple GO terms; most common still dominates."""
        go_map = {"a": ["GO:1", "GO:1", "GO:2"], "b": ["GO:1"]}
        # Total: GO:1 x3, GO:2 x1 -> purity = 3/4
        assert _community_purity(["a", "b"], go_map) == pytest.approx(0.75)

    def test_missing_gene_in_go_map(self):
        go_map = {"a": ["GO:1"]}
        # "b" has no annotation -> only "a"'s terms count
        assert _community_purity(["a", "b"], go_map) == pytest.approx(1.0)


# ===================================================================
# functional_purity / functional_purity_named
# ===================================================================

class TestFunctionalPurity:
    """Mean purity across communities."""

    def test_two_perfect_communities(self, triangle_nodes, triangle_go_map):
        """Two communities, both pure."""
        communities = [{0, 1}, {2}]  # {A,B} and {C}
        purity = functional_purity(communities, triangle_go_map, triangle_nodes)
        # {A,B}: GO:0001 x3, GO:0003 x1 -> 3/4; {C}: GO:0002 x1 -> 1.0
        assert purity == pytest.approx((0.75 + 1.0) / 2.0)

    def test_empty_community_list(self, triangle_go_map, triangle_nodes):
        assert functional_purity([], triangle_go_map, triangle_nodes) == 0.0

    def test_named_communities(self, triangle_go_map):
        communities = [{"A", "B"}, {"C"}]
        purity = functional_purity_named(communities, triangle_go_map)
        assert purity == pytest.approx((0.75 + 1.0) / 2.0)


# ===================================================================
# compute_gf_curve
# ===================================================================

class TestGFCompute:
    """G-F curve computation on small synthetic data."""

    def test_returns_correct_length(self, triangle_graph, triangle_coords,
                                     triangle_nodes, triangle_go_map):
        r_vals = np.linspace(0.05, 0.55, 50)
        purities, mods = compute_gf_curve(
            triangle_coords, triangle_nodes, triangle_go_map, r_vals
        )
        assert len(purities) == 50
        assert len(mods) == 50

    def test_purity_bounded(self, triangle_graph, triangle_coords,
                             triangle_nodes, triangle_go_map):
        r_vals = np.linspace(0.01, 1.0, 100)
        purities, _ = compute_gf_curve(
            triangle_coords, triangle_nodes, triangle_go_map, r_vals
        )
        assert all(0.0 <= p <= 1.0 for p in purities)

    def test_small_r_no_edges(self, chain_coords, chain_nodes, chain_go_map):
        """At very small r, no edges -> purity = 0."""
        r_vals = np.array([0.001])
        purities, mods = compute_gf_curve(
            chain_coords, chain_nodes, chain_go_map, r_vals
        )
        assert purities[0] == 0.0


# ===================================================================
# compute_gf_score
# ===================================================================

class TestGFScore:
    """G-F score integration."""

    def test_constant_purity(self):
        """Constant purity curve -> score equals that constant."""
        r = np.linspace(0.05, 0.55, 200)
        p = np.full(200, 0.5)
        score = compute_gf_score(r, p, r_min=0.05, r_max=0.422)
        assert score == pytest.approx(0.5, abs=1e-3)

    def test_zero_purity(self):
        r = np.linspace(0.05, 0.55, 200)
        p = np.zeros(200)
        assert compute_gf_score(r, p) == pytest.approx(0.0)

    def test_linear_purity(self):
        """Linear purity p(r) = r -> integral = mean(r) over [a,b]."""
        r = np.linspace(0.0, 1.0, 1000)
        p = r.copy()
        score = compute_gf_score(r, p, r_min=0.0, r_max=1.0)
        assert score == pytest.approx(0.5, abs=1e-3)

    def test_narrow_interval(self):
        """Score on a sub-interval of a constant curve."""
        r = np.linspace(0.0, 1.0, 500)
        p = np.full(500, 0.3)
        score = compute_gf_score(r, p, r_min=0.2, r_max=0.8)
        assert score == pytest.approx(0.3, abs=1e-3)

    def test_insufficient_points(self):
        """Fewer than 2 points in interval -> 0.0."""
        r = np.array([0.0, 1.0])
        p = np.array([0.5, 0.5])
        score = compute_gf_score(r, p, r_min=0.3, r_max=0.7)
        assert score == 0.0


# ===================================================================
# compute_plateau_width
# ===================================================================

class TestPlateauWidth:
    """Plateau width detection."""

    def test_flat_peak(self):
        """Constant high purity region."""
        r = np.linspace(0.0, 1.0, 100)
        p = np.zeros(100)
        p[20:50] = 0.8
        result = compute_plateau_width(r, p)
        assert result["W"] > 0.0
        assert result["peak_purity"] == pytest.approx(0.8)

    def test_zero_purity(self):
        r = np.linspace(0.0, 1.0, 50)
        p = np.zeros(50)
        result = compute_plateau_width(r, p)
        assert result["W"] == 0.0

    def test_empty_input(self):
        result = compute_plateau_width([], [])
        assert result["W"] == 0.0

    def test_relative_threshold(self):
        """Peak=0.5, threshold=80% -> effective=0.4."""
        r = np.linspace(0.0, 1.0, 200)
        p = np.zeros(200)
        # Set a plateau region at purity=0.5
        p[50:150] = 0.5
        result = compute_plateau_width(r, p, relative_threshold=0.8)
        assert result["effective_threshold"] == pytest.approx(0.4)
        assert result["W"] > 0.0
        assert result["peak_purity"] == pytest.approx(0.5)

    def test_single_point_peak(self):
        """Only one point above threshold."""
        r = np.linspace(0.0, 1.0, 100)
        p = np.zeros(100)
        p[50] = 1.0
        result = compute_plateau_width(r, p, relative_threshold=0.8)
        # Single point: W = 0 (r[50] - r[50])
        assert result["W"] == pytest.approx(0.0)

    def test_dict_keys(self):
        """Returned dict has all expected keys."""
        r = np.linspace(0.0, 1.0, 50)
        p = np.full(50, 0.3)
        result = compute_plateau_width(r, p)
        expected_keys = {"W", "r_min", "r_max", "peak_purity", "effective_threshold"}
        assert set(result.keys()) == expected_keys


# ===================================================================
# Distance / spatial graph utilities
# ===================================================================

class TestSpatialUtils:
    """Distance matrix and spatial graph construction."""

    def test_distance_matrix_symmetric(self, triangle_coords):
        D = precompute_distance_matrix(triangle_coords)
        np.testing.assert_allclose(D, D.T)

    def test_distance_matrix_diagonal_zero(self, triangle_coords):
        D = precompute_distance_matrix(triangle_coords)
        np.testing.assert_allclose(np.diag(D), 0.0)

    def test_distance_matrix_single_point(self):
        D = precompute_distance_matrix(np.array([[0.0, 0.0]]))
        assert D.shape == (1, 1)

    def test_spatial_graph_edges(self):
        """4 points: two close, two far."""
        coords = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
        D = precompute_distance_matrix(coords)
        G = build_spatial_graph_fast(D, r=0.5)
        # Expect edges: (0,1) and (2,3)
        assert G.has_edge(0, 1)
        assert G.has_edge(2, 3)
        assert not G.has_edge(0, 2)
        assert G.number_of_nodes() == 4


# ===================================================================
# Coordinate rescaling
# ===================================================================

class TestRescaling:
    """rescale_coordinates and check_embedding_collapse."""

    def test_rescale_target_std(self):
        coords = np.random.randn(100, 2) * 5.0
        rescaled = rescale_coordinates(coords, target_std=0.3)
        assert np.std(rescaled) == pytest.approx(0.3, abs=0.01)

    def test_rescale_zero_std(self):
        coords = np.zeros((10, 2))
        rescaled = rescale_coordinates(coords)
        np.testing.assert_allclose(rescaled, 0.0)

    def test_collapse_detection_normal(self, triangle_coords):
        result = check_embedding_collapse(triangle_coords, "test")
        assert result["collapsed"] is False

    def test_collapse_detection_collapsed(self):
        coords = np.zeros((10, 2))
        result = check_embedding_collapse(coords, "zeros")
        assert result["collapsed"] is True


# ===================================================================
# Node alignment
# ===================================================================

class TestAlignment:
    """align_embedding_to_nodes."""

    def test_subset_alignment(self):
        coords = np.array([[1, 2], [3, 4], [5, 6]])
        emb_nodes = ["A", "B", "C"]
        target_nodes = ["B", "C"]
        aligned, common = align_embedding_to_nodes(coords, emb_nodes, target_nodes)
        assert common == ["B", "C"]
        np.testing.assert_array_equal(aligned, [[3, 4], [5, 6]])

    def test_no_overlap(self):
        coords = np.array([[1, 2]])
        aligned, common = align_embedding_to_nodes(coords, ["X"], ["A"])
        assert common == []
        assert aligned.shape[0] == 0
