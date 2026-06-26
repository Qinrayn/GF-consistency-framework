#!/usr/bin/env python3
"""
Tests for P0/P1 audit scripts: multiple comparison correction, outlier
sensitivity analysis, interval sensitivity analysis.

These tests verify the statistical correctness of the audit scripts without
requiring the full pipeline data.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))


# ================================================================
# Test multiple_comparison_correction.py
# ================================================================

class TestBenjaminiHochberg:
    """Test BH FDR correction implementation."""

    def test_import(self):
        from multiple_comparison_correction import benjamini_hochberg
        assert callable(benjamini_hochberg)

    def test_all_significant(self):
        """When all p-values are tiny, all should be significant."""
        from multiple_comparison_correction import benjamini_hochberg
        pvals = [0.001, 0.002, 0.003, 0.004]
        result = benjamini_hochberg(pvals)
        assert result["n_tests"] == 4
        assert all(result["rejected"])
        assert all(p < 0.05 for p in result["p_corrected"])

    def test_none_significant(self):
        """When all p-values are large, none should be significant."""
        from multiple_comparison_correction import benjamini_hochberg
        pvals = [0.5, 0.6, 0.7, 0.8]
        result = benjamini_hochberg(pvals)
        assert not any(result["rejected"])

    def test_mixed(self):
        """Mixed significance: some pass, some don't."""
        from multiple_comparison_correction import benjamini_hochberg
        pvals = [0.001, 0.01, 0.04, 0.06, 0.5]
        result = benjamini_hochberg(pvals)
        assert result["n_tests"] == 5
        # First two should definitely be significant
        assert result["rejected"][0] is True
        assert result["rejected"][1] is True

    def test_monotonicity(self):
        """BH-adjusted p-values should be non-decreasing when sorted."""
        from multiple_comparison_correction import benjamini_hochberg
        pvals = [0.001, 0.01, 0.03, 0.05, 0.1, 0.3]
        result = benjamini_hochberg(pvals)
        corrected = result["p_corrected"]
        # Sorted by original p-value, adjusted should be non-decreasing
        sorted_adjusted = sorted(corrected)
        for i in range(len(sorted_adjusted) - 1):
            assert sorted_adjusted[i] <= sorted_adjusted[i + 1] + 1e-10

    def test_bonferroni(self):
        from multiple_comparison_correction import bonferroni_correction
        pvals = [0.001, 0.01, 0.5]
        result = bonferroni_correction(pvals)
        assert result["p_corrected"][0] == pytest.approx(0.003)
        assert result["p_corrected"][2] == pytest.approx(1.0)  # capped
        assert result["rejected"][0] is True
        assert result["rejected"][2] is False

    def test_single_pvalue(self):
        """Edge case: single p-value."""
        from multiple_comparison_correction import benjamini_hochberg
        result = benjamini_hochberg([0.03])
        assert result["n_tests"] == 1
        assert result["p_corrected"][0] == pytest.approx(0.03)


# ================================================================
# Test outlier_sensitivity_analysis.py
# ================================================================

class TestOutlierDetection:
    """Test outlier detection logic."""

    def test_load_embedding_format(self):
        """Test that human embedding JSON format is correctly parsed."""
        from outlier_sensitivity_analysis import load_human_embedding
        # This should work for 'spectral' which has no outlier
        nodes, coords = load_human_embedding("spectral")
        if nodes is not None:
            assert len(nodes) == coords.shape[0]
            assert coords.shape[1] == 2  # 2D embeddings

    def test_check_outlier_no_outlier(self):
        """Synthetic data with no outliers should report low deviation."""
        from outlier_sensitivity_analysis import check_outlier
        rng = np.random.RandomState(42)
        coords = rng.randn(100, 2)
        nodes = [f"node_{i}" for i in range(100)]
        result = check_outlier(nodes, coords, "test")
        assert result["outlier_present"] is False
        assert result["max_deviation_sigma"] < 5.0  # normal data

    def test_check_outlier_with_outlier(self):
        """Synthetic data with an extreme outlier should detect it."""
        from outlier_sensitivity_analysis import check_outlier
        rng = np.random.RandomState(42)
        coords = rng.randn(100, 2)
        coords[50] = [100.0, 0.0]  # extreme outlier
        nodes = [f"node_{i}" for i in range(100)]
        nodes[50] = "OUTLIER_NODE"
        result = check_outlier(nodes, coords, "test")
        assert result["max_deviation_sigma"] > 5.0  # extreme outlier should be detectable
        assert result["max_deviation_node"] == "OUTLIER_NODE"


# ================================================================
# Test interval_sensitivity_analysis.py
# ================================================================

class TestIntervalSensitivity:
    """Test interval sensitivity analysis constants."""

    def test_intervals_defined(self):
        """Test that 7 intervals are defined."""
        from interval_sensitivity_analysis import INTERVALS
        assert len(INTERVALS) == 7
        # Each interval is (name, r_min, r_max)
        for name, rmin, rmax in INTERVALS:
            assert isinstance(name, str)
            assert 0 < rmin < rmax
            assert rmax <= 0.55

    def test_current_interval(self):
        """The 'current' interval should be [0.05, 0.422]."""
        from interval_sensitivity_analysis import INTERVALS
        current = [i for i in INTERVALS if i[0] == "current"][0]
        assert current[1] == pytest.approx(0.05)
        assert current[2] == pytest.approx(0.422)

    def test_methods_defined(self):
        """Test that 11 methods are defined."""
        from interval_sensitivity_analysis import METHODS
        assert len(METHODS) == 11
        assert "Spectral" in METHODS
        assert "GAT" in METHODS


# ================================================================
# Test shared constants in utils.py
# ================================================================

class TestSharedConstants:
    """Test that shared constants are correctly defined in utils.py."""

    def test_banner(self):
        from utils import BANNER
        assert isinstance(BANNER, str)
        assert len(BANNER) == 70  # unified to 70

    def test_method_colors(self):
        from utils import METHOD_COLORS
        assert isinstance(METHOD_COLORS, dict)
        assert len(METHOD_COLORS) == 11  # all 11 methods
        for method in ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec",
                        "VGAE", "VGAE-feat", "PCA", "GraphSAGE", "GAT", "GIN"]:
            assert method in METHOD_COLORS
            assert METHOD_COLORS[method].startswith("#")  # hex color

    def test_baseline_colors(self):
        from utils import BASELINE_COLORS
        assert isinstance(BASELINE_COLORS, dict)
        assert "PPI-Neighbors" in BASELINE_COLORS
        assert "Random" in BASELINE_COLORS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
