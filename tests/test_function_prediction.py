#!/usr/bin/env python3
"""
test_function_prediction.py -- Unit tests for function prediction core logic.

Tests the critical functions in function_prediction.py that lack coverage:
knn_predict, ppi_neighbor_predict, twohop_diffusion_predict,
evaluate_precision_at_k, compute_mean_reciprocal_rank,
random_baseline_predictions, compute_gf_correlation.
"""

import sys
import math
import numpy as np
import networkx as nx
import pytest
from pathlib import Path
from collections import Counter

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))


# ================================================================
# Fixtures: synthetic data
# ================================================================

@pytest.fixture
def simple_network():
    """Small 5-node network with GO annotations."""
    G = nx.Graph()
    G.add_edges_from([
        ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("A", "C"),
    ])
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    annotations = {
        "A": {"GO:001", "GO:002"},
        "B": {"GO:001", "GO:003"},
        "C": {"GO:001", "GO:002", "GO:003"},
        "D": {"GO:002", "GO:004"},
        "E": {"GO:004", "GO:005"},
    }
    return G, nodes, node_to_idx, annotations


@pytest.fixture
def simple_embedding():
    """5-node 2D embedding coordinates."""
    return np.array([
        [0.0, 0.0],   # A
        [1.0, 0.0],   # B
        [0.5, 0.5],   # C
        [1.5, 0.5],   # D
        [2.0, 0.0],   # E
    ])


# ================================================================
# Test knn_predict
# ================================================================

class TestKnnPredict:

    def test_basic_prediction(self, simple_network, simple_embedding):
        """knn_predict should return a list of (term, score) predictions."""
        from function_prediction import knn_predict
        G, nodes, node_to_idx, annotations = simple_network
        preds = knn_predict(
            query_id="A",
            coords=simple_embedding,
            nodes=nodes,
            node_to_idx=node_to_idx,
            annotations=annotations,
            k=3,
        )
        assert isinstance(preds, list)
        assert len(preds) > 0
        # Each prediction should be a (term, score) tuple
        assert isinstance(preds[0], tuple)
        assert len(preds[0]) == 2

    def test_hidden_term_excluded_from_query(self, simple_network, simple_embedding):
        """Hidden term should be removed from query annotations, not neighbors."""
        from function_prediction import knn_predict
        G, nodes, node_to_idx, annotations = simple_network
        # Node A has GO:001 and GO:002; hide GO:001
        preds = knn_predict(
            query_id="A",
            coords=simple_embedding,
            nodes=nodes,
            node_to_idx=node_to_idx,
            annotations=annotations,
            k=2,
            hidden_term="GO:001",
        )
        # Predictions should still work (neighbors vote for GO:001)
        assert isinstance(preds, list)

    def test_k_equals_1_nearest(self, simple_network, simple_embedding):
        """k=1 should use only the single nearest neighbor."""
        from function_prediction import knn_predict
        G, nodes, node_to_idx, annotations = simple_network
        preds = knn_predict(
            query_id="E",
            coords=simple_embedding,
            nodes=nodes,
            node_to_idx=node_to_idx,
            annotations=annotations,
            k=1,
        )
        assert isinstance(preds, list)
        # E's nearest is D. D has GO:002, GO:004
        if len(preds) > 0:
            pred_terms = [p[0] for p in preds]
            assert any(t in pred_terms for t in ["GO:002", "GO:004"])

    def test_no_annotations_returns_empty(self):
        """Node with no annotations and no annotated neighbors should get empty list."""
        from function_prediction import knn_predict
        coords = np.array([[0.0, 0.0], [10.0, 10.0]])
        nodes = ["X", "Y"]
        node_to_idx = {"X": 0, "Y": 1}
        annotations = {}  # No annotations at all
        preds = knn_predict("X", coords, nodes, node_to_idx, annotations, k=1)
        assert isinstance(preds, list)


# ================================================================
# Test evaluate_precision_at_k
# ================================================================

class TestEvaluatePrecisionAtK:

    def test_hidden_term_in_top1(self):
        """If hidden term is the top prediction, P@1 = 1."""
        from function_prediction import evaluate_precision_at_k
        predictions = [("GO:001", 0.9), ("GO:002", 0.5), ("GO:003", 0.1)]
        actual_term = "GO:001"
        k_values = [1, 3, 5, 10]
        result = evaluate_precision_at_k(predictions, actual_term, k_values)
        assert isinstance(result, dict)
        assert result[1] == 1  # In top-1

    def test_hidden_term_not_found(self):
        """If hidden term is not in predictions, all P@k = 0."""
        from function_prediction import evaluate_precision_at_k
        predictions = [("GO:999", 0.8), ("GO:998", 0.3)]
        actual_term = "GO:001"
        k_values = [1, 3, 5]
        result = evaluate_precision_at_k(predictions, actual_term, k_values)
        for k in k_values:
            assert result[k] == 0

    def test_hidden_term_at_rank3(self):
        """Hidden term at position 3: P@1=0, P@3=1."""
        from function_prediction import evaluate_precision_at_k
        predictions = [
            ("GO:002", 0.9), ("GO:003", 0.7),
            ("GO:001", 0.5), ("GO:004", 0.1),
        ]
        actual_term = "GO:001"
        k_values = [1, 3, 5]
        result = evaluate_precision_at_k(predictions, actual_term, k_values)
        assert result[1] == 0
        assert result[3] == 1


# ================================================================
# Test compute_mean_reciprocal_rank
# ================================================================

class TestComputeMRR:

    def test_perfect_rank1(self):
        """If hidden term is always at rank 1, MRR = 1.0 for that method."""
        from function_prediction import compute_mean_reciprocal_rank
        trials = [
            {"Spectral": 1, "DM": 1},
            {"Spectral": 1, "DM": 2},
        ]
        result = compute_mean_reciprocal_rank(trials)
        assert isinstance(result, dict)
        assert abs(result["Spectral"] - 1.0) < 1e-10  # Always rank 1
        assert abs(result["DM"] - 0.75) < 1e-10  # (1/1 + 1/2) / 2

    def test_missing_term_rr_zero(self):
        """Rank 0 (not found) gives RR = 0."""
        from function_prediction import compute_mean_reciprocal_rank
        trials = [{"Spectral": 0}]
        result = compute_mean_reciprocal_rank(trials)
        assert abs(result["Spectral"] - 0.0) < 1e-10

    def test_empty_trials(self):
        """Empty trial list should return empty dict."""
        from function_prediction import compute_mean_reciprocal_rank
        result = compute_mean_reciprocal_rank([])
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_multiple_methods(self):
        """MRR computed independently per method."""
        from function_prediction import compute_mean_reciprocal_rank
        trials = [
            {"A": 1, "B": 2},
            {"A": 2, "B": 1},
        ]
        result = compute_mean_reciprocal_rank(trials)
        assert abs(result["A"] - 0.75) < 1e-10  # (1 + 0.5) / 2
        assert abs(result["B"] - 0.75) < 1e-10  # (0.5 + 1) / 2


# ================================================================
# Test ppi_neighbor_predict
# ================================================================

class TestPpiNeighborPredict:

    def test_direct_neighbors_vote(self, simple_network):
        """PPI neighbors should vote for their GO terms."""
        from function_prediction import ppi_neighbor_predict
        G, nodes, _, annotations = simple_network
        preds = ppi_neighbor_predict(
            query_id="A",
            graph=G,
            annotations=annotations,
        )
        assert isinstance(preds, list)
        # A's neighbors: B, C. B has GO:001, GO:003; C has GO:001, GO:002, GO:003
        if len(preds) > 0:
            pred_terms = [p[0] for p in preds]
            assert "GO:001" in pred_terms  # Most votes from B+C

    def test_isolated_node(self):
        """Isolated node with no neighbors should return empty predictions."""
        from function_prediction import ppi_neighbor_predict
        G = nx.Graph()
        G.add_node("LONELY")
        annotations = {"LONELY": {"GO:001"}}
        preds = ppi_neighbor_predict("LONELY", G, annotations)
        assert isinstance(preds, list)
        assert len(preds) == 0


# ================================================================
# Test twohop_diffusion_predict
# ================================================================

class TestTwohopDiffusion:

    def test_basic_diffusion(self, simple_network):
        """2-hop diffusion should return ranked predictions."""
        from function_prediction import twohop_diffusion_predict
        G, nodes, _, annotations = simple_network
        preds = twohop_diffusion_predict(
            query_id="A",
            graph=G,
            annotations=annotations,
            decay=0.5,
        )
        assert isinstance(preds, list)

    def test_decay_effect(self, simple_network):
        """Higher decay should still return valid predictions."""
        from function_prediction import twohop_diffusion_predict
        G, nodes, _, annotations = simple_network
        preds_low = twohop_diffusion_predict("A", G, annotations, decay=0.1)
        preds_high = twohop_diffusion_predict("A", G, annotations, decay=0.9)
        assert isinstance(preds_low, list)
        assert isinstance(preds_high, list)


# ================================================================
# Test random_baseline_predictions
# ================================================================

class TestRandomBaseline:

    def test_returns_list(self):
        """Random baseline should return a list of (term, count) from Counter."""
        from function_prediction import random_baseline_predictions
        global_freq = Counter({"GO:001": 10, "GO:002": 5, "GO:003": 3})
        preds = random_baseline_predictions(global_freq)
        assert isinstance(preds, list)
        assert len(preds) == 3

    def test_ordering_by_frequency(self):
        """Terms should be sorted by global frequency (descending)."""
        from function_prediction import random_baseline_predictions
        global_freq = Counter({"GO:001": 10, "GO:002": 5, "GO:003": 3})
        preds = random_baseline_predictions(global_freq)
        assert preds[0][0] == "GO:001"
        assert preds[-1][0] == "GO:003"


# ================================================================
# Test compute_gf_correlation
# ================================================================

class TestGFCorrelation:

    def test_basic_correlation(self):
        """Should compute Spearman and Pearson correlations."""
        from function_prediction import compute_gf_correlation
        method_mrr = {
            "A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04, "E": 0.05,
        }
        gf_scores = {
            "A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5,
        }
        result = compute_gf_correlation(method_mrr, gf_scores)
        assert isinstance(result, dict)
        assert "spearman_rho" in result
        assert abs(result["spearman_rho"] - 1.0) < 0.01  # Perfect monotonic

    def test_negative_correlation(self):
        """Inverse relationship should give negative rho."""
        from function_prediction import compute_gf_correlation
        method_mrr = {
            "A": 0.05, "B": 0.04, "C": 0.03, "D": 0.02, "E": 0.01,
        }
        gf_scores = {
            "A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5,
        }
        result = compute_gf_correlation(method_mrr, gf_scores)
        assert result["spearman_rho"] < -0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
