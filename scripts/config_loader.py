#!/usr/bin/env python3
"""
G-F Consistency Framework — Configuration Loader
=================================================
Loads, validates, and provides pipeline configuration from YAML files.

Usage
-----
    from scripts.config_loader import load_config

    cfg = load_config()                     # auto-detect pipeline_config.yaml
    cfg = load_config("my_config.yaml")     # explicit path
    cfg = load_config(merge_cli=True)       # merge argparse overrides

The returned object is a plain ``dict`` so that existing code can access
parameters with standard dict syntax (``cfg["pipeline"]["seed"]``).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded defaults — used when no YAML file is found
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "pipeline": {
        "seed": 42,
        "species": "yeast",
        "start_from": 1,
        "skip_gnn": False,
        "skip_extended": False,
        "skip_plots": False,
        "run_human": False,
    },
    "paths": {
        "data_dir": "data",
        "results_dir": "results",
        "figures_dir": "figures",
        "embeddings_dir": "embeddings",
    },
    "network": {
        "curated_network": "curated_153_ppi.edgelist",
        "string_file": "4932.protein.links.v11.5.txt.gz",
        "string_min_score": 700,
        "gene_go_map": "gene_go_map.json",
    },
    "embeddings": {
        "target_std": 0.3,
        "embedding_dim": 2,
        "classical_methods": ["DM", "MDS", "Spectral", "DeepWalk", "Node2Vec", "VGAE"],
        "curated_methods": ["PCA", "VGAE-feat"],
        "gnn_methods": ["GraphSAGE", "GAT", "GIN"],
        "deepwalk": {"walk_length": 20, "walks_per_node": 10, "window_size": 5},
        "node2vec": {"walk_length": 20, "walks_per_node": 10,
                      "window_size": 5, "p": 0.5, "q": 2.0},
        "vgae": {"hidden_dim": 4, "latent_dim": 2,
                 "epochs": 300, "learning_rate": 0.01},
    },
    "gf_score": {
        "r_min": 0.05,
        "r_max": 0.55,
        "n_points": 200,
        "gf_r_min": 0.05,
        "gf_r_max": 0.422,
        "plateau_relative_threshold": 0.80,
        "adaptive": {
            "cv_threshold": 0.1,
            "min_width": 0.15,
            "significance_sigma": 2,
        },
    },
    "robustness": {
        "n_subsets": 30,
        "subset_sizes": [50, 100, 150, 200, "all"],
        "n_points": 30,
        "methods": ["DM", "MDS"],
        "bonferroni_reference_size": 150,
    },
    "link_prediction": {
        "cv_folds": 5,
        "min_methods_for_spearman": 5,
    },
    "downstream_knn": {
        "k_neighbors": 5,
        "min_label_count": 3,
    },
    "randomization": {
        "n_shuffles": 10,
    },
    "sampling_density": {
        "grids": [
            {"name": "coarse", "n_points": 30},
            {"name": "standard", "n_points": 200},
        ],
    },
    "benchmark": {
        "n_repeat": 1,
        "sampling_points": [50, 100, 200],
    },
}

# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

_VALIDATION_RULES: dict[str, list[tuple[str, type, Optional[Any], Optional[Any]]]] = {
    # section: [(dotted_key, expected_type, min_val, max_val)]
    "pipeline": [
        ("seed", int, 0, None),
        ("species", str, None, None),
        ("start_from", int, 1, 72),
    ],
    "gf_score": [
        ("r_min", float, 0.0, 1.0),
        ("r_max", float, 0.0, 2.0),
        ("n_points", int, 10, 5000),
        ("gf_r_min", float, 0.0, 1.0),
        ("gf_r_max", float, 0.0, 2.0),
        ("plateau_relative_threshold", float, 0.0, 1.0),
    ],
    "robustness": [
        ("n_subsets", int, 1, 1000),
        ("n_points", int, 5, 5000),
    ],
    "link_prediction": [
        ("cv_folds", int, 2, 20),
    ],
    "downstream_knn": [
        ("k_neighbors", int, 1, 50),
    ],
    "randomization": [
        ("n_shuffles", int, 1, 1000),
    ],
}


# ---------------------------------------------------------------------------
# Deep merge utility
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*.

    Lists are replaced (not appended).  None values in *override* are
    treated as "delete key" so that users can remove defaults.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(config: dict) -> list[str]:
    """Return a list of warning strings (empty = all good)."""
    warnings: list[str] = []

    for section, rules in _VALIDATION_RULES.items():
        sec = config.get(section, {})
        for dotted_key, expected_type, min_val, max_val in rules:
            parts = dotted_key.split(".")
            obj = sec
            for p in parts:
                if isinstance(obj, dict):
                    obj = obj.get(p)
                else:
                    obj = None
                    break
            if obj is None:
                continue

            # Type check — accept int as float
            if expected_type is float and isinstance(obj, int):
                obj = float(obj)
                # patch back so downstream code gets float
                target = sec
                for p in parts[:-1]:
                    target = target.get(p, {})
                if isinstance(target, dict):
                    target[parts[-1]] = obj

            if not isinstance(obj, expected_type):
                warnings.append(
                    f"Config [{section}].{dotted_key}: expected "
                    f"{expected_type.__name__}, got {type(obj).__name__}"
                )
                continue

            if min_val is not None and obj < min_val:
                warnings.append(
                    f"Config [{section}].{dotted_key}: value {obj} "
                    f"below minimum {min_val}"
                )
            if max_val is not None and obj > max_val:
                warnings.append(
                    f"Config [{section}].{dotted_key}: value {obj} "
                    f"above maximum {max_val}"
                )

    # Cross-field: r_min < r_max
    gf = config.get("gf_score", {})
    if gf.get("r_min", 0) >= gf.get("r_max", 1):
        warnings.append("gf_score.r_min >= gf_score.r_max")
    if gf.get("gf_r_min", 0) >= gf.get("gf_r_max", 1):
        warnings.append("gf_score.gf_r_min >= gf_score.gf_r_max")

    # Species check
    species = config.get("pipeline", {}).get("species", "yeast")
    if species not in ("yeast", "human", "mouse", "fly", "ecoli"):
        warnings.append(f"Unknown species '{species}' (expected yeast, human, mouse, fly, or ecoli)")

    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: Optional[str | Path] = None,
    project_root: Optional[Path] = None,
    *,
    merge_cli: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Load pipeline configuration.

    Parameters
    ----------
    config_path : str or Path, optional
        Explicit path to a YAML config file.  When *None*, the loader
        searches for ``pipeline_config.yaml`` in the project root.
    project_root : Path, optional
        Project root directory (default: auto-detect from this file's
        location).
    merge_cli : bool
        If *True*, parse ``sys.argv`` for known CLI flags and merge them
        on top of the config (CLI wins).
    strict : bool
        If *True*, raise ``ValueError`` on validation warnings instead of
        just logging them.

    Returns
    -------
    dict
        Fully resolved configuration dictionary.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    config_path = Path(config_path) if config_path else project_root / "pipeline_config.yaml"

    # Start from defaults
    config = copy.deepcopy(_DEFAULTS)

    # Try to load YAML
    if config_path.exists():
        try:
            import yaml
        except ImportError:
            logger.warning(
                "PyYAML not installed — using built-in defaults. "
                "Install with: pip install pyyaml"
            )
        else:
            with open(config_path, encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f)
            if isinstance(user_cfg, dict):
                config = _deep_merge(config, user_cfg)
                logger.info("Loaded config from %s", config_path)
            else:
                logger.warning("Config file %s is empty or invalid", config_path)
    else:
        logger.debug("No config file at %s — using defaults", config_path)

    # Resolve relative paths
    paths = config.get("paths", {})
    for key in ("data_dir", "results_dir", "figures_dir", "embeddings_dir"):
        val = paths.get(key, "")
        if val and not Path(val).is_absolute():
            paths[key] = str(project_root / val)
    config["paths"] = paths

    # Store project root for convenience
    config["_project_root"] = str(project_root)

    # Validate
    warnings = _validate(config)
    for w in warnings:
        logger.warning("Config validation: %s", w)
    if strict and warnings:
        raise ValueError(
            f"Configuration has {len(warnings)} warning(s):\n"
            + "\n".join(f"  - {w}" for w in warnings)
        )

    # Optionally merge CLI overrides
    if merge_cli:
        config = _merge_cli_overrides(config)

    return config


def _merge_cli_overrides(config: dict) -> dict:
    """Parse known CLI arguments and overlay them onto the config.

    Only processes arguments that the pipeline recognises; unknown args
    are silently ignored so that downstream argparse calls still work.
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-human", action="store_true", default=None)
    parser.add_argument("--skip-plots", action="store_true", default=None)
    parser.add_argument("--skip-gnn", action="store_true", default=None)
    parser.add_argument("--skip-extended", action="store_true", default=None)
    parser.add_argument("--start-from", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    args, _ = parser.parse_known_args()

    if args.run_human:
        config["pipeline"]["run_human"] = True
    if args.skip_plots:
        config["pipeline"]["skip_plots"] = True
    if args.skip_gnn:
        config["pipeline"]["skip_gnn"] = True
    if args.skip_extended:
        config["pipeline"]["skip_extended"] = True
    if args.start_from is not None:
        config["pipeline"]["start_from"] = args.start_from
    if args.seed is not None:
        config["pipeline"]["seed"] = args.seed
    if args.species is not None:
        config["pipeline"]["species"] = args.species

    return config


def get_config_value(config: dict, dotted_key: str, default: Any = None) -> Any:
    """Retrieve a config value using dotted notation."""
    parts = dotted_key.split(".")
    obj = config
    for p in parts:
        if isinstance(obj, dict):
            obj = obj.get(p)
        else:
            return default
        if obj is None:
            return default
    return obj
