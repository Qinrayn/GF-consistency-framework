"""Tests for scripts/config_loader.py — configuration loading and validation."""

import copy
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.config_loader import (
    load_config,
    get_config_value,
    _deep_merge,
    _validate,
    _DEFAULTS,
)


# ===================================================================
# Deep merge
# ===================================================================

class TestDeepMerge:
    """_deep_merge utility."""

    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {"b": 3})
        assert result == {"a": 1, "b": 3}

    def test_nested_override(self):
        base = {"x": {"y": 1, "z": 2}, "w": 3}
        result = _deep_merge(base, {"x": {"z": 99}})
        assert result == {"x": {"y": 1, "z": 99}, "w": 3}

    def test_none_deletes_key(self):
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {"b": None})
        assert result == {"a": 1}

    def test_list_replacement(self):
        base = {"items": [1, 2, 3]}
        result = _deep_merge(base, {"items": [4, 5]})
        assert result == {"items": [4, 5]}

    def test_base_not_mutated(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 99}})
        assert base["a"]["b"] == 1


# ===================================================================
# Validation
# ===================================================================

class TestValidation:
    """_validate config validation."""

    def test_defaults_pass(self):
        warnings = _validate(copy.deepcopy(_DEFAULTS))
        assert len(warnings) == 0

    def test_bad_seed(self):
        cfg = copy.deepcopy(_DEFAULTS)
        cfg["pipeline"]["seed"] = -1
        warnings = _validate(cfg)
        assert any("seed" in w for w in warnings)

    def test_bad_species(self):
        cfg = copy.deepcopy(_DEFAULTS)
        cfg["pipeline"]["species"] = "zebrafish"
        warnings = _validate(cfg)
        assert any("species" in w or "Unknown" in w for w in warnings)

    def test_r_min_exceeds_r_max(self):
        cfg = copy.deepcopy(_DEFAULTS)
        cfg["gf_score"]["r_min"] = 0.9
        cfg["gf_score"]["r_max"] = 0.1
        warnings = _validate(cfg)
        assert any("r_min" in w for w in warnings)

    def test_type_mismatch(self):
        cfg = copy.deepcopy(_DEFAULTS)
        cfg["gf_score"]["n_points"] = "two hundred"
        warnings = _validate(cfg)
        assert any("n_points" in w for w in warnings)


# ===================================================================
# load_config
# ===================================================================

class TestLoadConfig:
    """load_config public API."""

    def test_load_defaults_no_yaml(self, tmp_path):
        """Without a YAML file, defaults are returned."""
        cfg = load_config(
            config_path=tmp_path / "nonexistent.yaml",
            project_root=tmp_path,
        )
        assert cfg["pipeline"]["seed"] == 42
        assert cfg["gf_score"]["n_points"] == 200

    def test_load_from_yaml(self, tmp_path):
        """YAML file overrides defaults."""
        yaml_content = {
            "pipeline": {"seed": 123, "species": "human"},
            "gf_score": {"n_points": 500},
        }
        yaml_file = tmp_path / "pipeline_config.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        cfg = load_config(config_path=yaml_file, project_root=tmp_path)
        assert cfg["pipeline"]["seed"] == 123
        assert cfg["pipeline"]["species"] == "human"
        assert cfg["gf_score"]["n_points"] == 500
        # Non-overridden values should keep defaults
        assert cfg["gf_score"]["gf_r_min"] == 0.05

    def test_strict_mode_raises(self, tmp_path):
        """strict=True should raise on validation warnings."""
        yaml_content = {"pipeline": {"seed": -5}}
        yaml_file = tmp_path / "pipeline_config.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        with pytest.raises(ValueError, match="warning"):
            load_config(config_path=yaml_file, project_root=tmp_path, strict=True)

    def test_relative_paths_resolved(self, tmp_path):
        cfg = load_config(project_root=tmp_path)
        data_dir = cfg["paths"]["data_dir"]
        assert str(tmp_path) in data_dir

    def test_project_root_stored(self, tmp_path):
        cfg = load_config(project_root=tmp_path)
        assert cfg["_project_root"] == str(tmp_path)


# ===================================================================
# get_config_value
# ===================================================================

class TestGetConfigValue:
    """get_config_value dotted-key accessor."""

    def test_simple_key(self):
        cfg = {"a": {"b": {"c": 42}}}
        assert get_config_value(cfg, "a.b.c") == 42

    def test_missing_key(self):
        cfg = {"a": {"b": 1}}
        assert get_config_value(cfg, "a.x.y", default="fallback") == "fallback"

    def test_top_level_key(self):
        cfg = {"seed": 42}
        assert get_config_value(cfg, "seed") == 42
