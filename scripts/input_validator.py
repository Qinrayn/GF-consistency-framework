#!/usr/bin/env python3
"""
G-F Consistency Framework — Input Validation & Error Handling
=============================================================
Validates input files, embedding dimensions, GO annotations, and
pipeline parameters before expensive computation begins.

All validation functions return a ``ValidationResult`` dataclass with
``valid`` (bool), ``errors`` (list of str), and ``warnings`` (list of str).
"""

from __future__ import annotations

import json
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ValidationResult:
    """Outcome of a validation check."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another result into this one (in-place)."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False
        return self

    def summary(self) -> str:
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        if not parts:
            return "Validation passed"
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Network file validation
# ---------------------------------------------------------------------------

def validate_edgelist(path: Path, min_nodes: int = 3) -> ValidationResult:
    """Validate a tab/space-separated edgelist file.

    Checks:
    - File exists and is readable
    - At least 2 columns per line
    - Minimum number of unique nodes
    - No self-loops (warning only)
    """
    result = ValidationResult()
    path = Path(path)

    if not path.exists():
        result.valid = False
        result.errors.append(f"Edgelist file not found: {path}")
        return result

    if path.stat().st_size == 0:
        result.valid = False
        result.errors.append(f"Edgelist file is empty: {path}")
        return result

    nodes = set()
    self_loops = 0
    line_errors = 0

    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(str(path), "rt", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    line_errors += 1
                    if line_errors <= 5:
                        result.warnings.append(f"Line {i}: expected ≥2 columns, got {len(parts)}")
                    continue
                u, v = parts[0], parts[1]
                if u == v:
                    self_loops += 1
                nodes.add(u)
                nodes.add(v)
    except Exception as e:
        result.valid = False
        result.errors.append(f"Failed to read edgelist: {e}")
        return result

    if self_loops > 0:
        result.warnings.append(f"Found {self_loops} self-loop(s) (ignored)")
    if line_errors > 5:
        result.warnings.append(f"{line_errors} malformed lines total")

    if len(nodes) < min_nodes:
        result.valid = False
        result.errors.append(
            f"Network has {len(nodes)} nodes, minimum is {min_nodes}"
        )

    return result


# ---------------------------------------------------------------------------
# Embedding validation
# ---------------------------------------------------------------------------

def validate_embedding(coords: np.ndarray, expected_nodes: int,
                       method_name: str = "") -> ValidationResult:
    """Validate an embedding array.

    Checks:
    - Shape: (n, d) with n == expected_nodes
    - No NaN or Inf values
    - Not collapsed (all zeros or all identical)
    """
    result = ValidationResult()
    prefix = f"[{method_name}] " if method_name else ""

    if coords.ndim != 2:
        result.valid = False
        result.errors.append(f"{prefix}Expected 2-D array, got {coords.ndim}-D")
        return result

    if coords.shape[0] != expected_nodes:
        result.valid = False
        result.errors.append(
            f"{prefix}Expected {expected_nodes} nodes, got {coords.shape[0]}"
        )

    n_nan = int(np.isnan(coords).sum())
    n_inf = int(np.isinf(coords).sum())
    if n_nan > 0:
        result.valid = False
        result.errors.append(f"{prefix}Contains {n_nan} NaN value(s)")
    if n_inf > 0:
        result.valid = False
        result.errors.append(f"{prefix}Contains {n_inf} Inf value(s)")

    # Collapse detection
    if coords.shape[0] > 1:
        row_std = np.std(coords, axis=0)
        if np.all(row_std < 1e-10):
            result.warnings.append(
                f"{prefix}Embedding appears collapsed (zero variance)"
            )

    return result


# ---------------------------------------------------------------------------
# GO annotation validation
# ---------------------------------------------------------------------------

def validate_go_map(go_map: dict, expected_nodes: Optional[set] = None) -> ValidationResult:
    """Validate a gene-to-GO-term mapping.

    Checks:
    - Non-empty dict
    - Values are lists of strings
    - Coverage: fraction of expected_nodes with annotations
    """
    result = ValidationResult()

    if not isinstance(go_map, dict):
        result.valid = False
        result.errors.append("GO map must be a dict")
        return result

    if len(go_map) == 0:
        result.valid = False
        result.errors.append("GO map is empty")
        return result

    bad_values = 0
    for gene, terms in go_map.items():
        if not isinstance(terms, list):
            bad_values += 1
        elif not all(isinstance(t, str) for t in terms):
            bad_values += 1

    if bad_values > 0:
        result.warnings.append(
            f"{bad_values} gene(s) have non-list or non-string GO terms"
        )

    if expected_nodes is not None:
        covered = len(set(go_map.keys()) & expected_nodes)
        coverage = covered / len(expected_nodes) if expected_nodes else 0.0
        if coverage < 0.1:
            result.warnings.append(
                f"GO coverage: {coverage:.1%} ({covered}/{len(expected_nodes)} nodes)"
            )
        elif coverage < 0.5:
            result.warnings.append(
                f"GO coverage is moderate: {coverage:.1%}"
            )

    return result


# ---------------------------------------------------------------------------
# STRING file validation
# ---------------------------------------------------------------------------

def validate_string_file(path: Path, min_score: int = 700) -> ValidationResult:
    """Validate a STRING .txt.gz interaction file."""
    result = ValidationResult()
    path = Path(path)

    if not path.exists():
        result.valid = False
        result.errors.append(f"STRING file not found: {path}")
        return result

    try:
        n_edges = 0
        n_valid = 0
        with gzip.open(str(path), "rt", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    n_edges += 1
                    if int(parts[2]) >= min_score:
                        n_valid += 1
                if n_edges > 100000:
                    break  # sample check only
    except Exception as e:
        result.valid = False
        result.errors.append(f"Failed to read STRING file: {e}")
        return result

    if n_edges == 0:
        result.valid = False
        result.errors.append("STRING file contains no edges")
    else:
        frac = n_valid / n_edges if n_edges > 0 else 0.0
        if frac < 0.01:
            result.warnings.append(
                f"Only {frac:.1%} edges pass score threshold {min_score}"
            )

    return result


# ---------------------------------------------------------------------------
# Pipeline parameter validation
# ---------------------------------------------------------------------------

def validate_pipeline_params(config: dict) -> ValidationResult:
    """Validate key pipeline parameters from the config dict."""
    result = ValidationResult()

    gf = config.get("gf_score", {})
    r_min = gf.get("r_min", 0.05)
    r_max = gf.get("r_max", 0.55)
    n_pts = gf.get("n_points", 200)

    if r_min >= r_max:
        result.valid = False
        result.errors.append(f"r_min ({r_min}) must be < r_max ({r_max})")
    if n_pts < 10:
        result.valid = False
        result.errors.append(f"n_points ({n_pts}) must be >= 10")

    gf_r_min = gf.get("gf_r_min", 0.05)
    gf_r_max = gf.get("gf_r_max", 0.422)
    if gf_r_min >= gf_r_max:
        result.valid = False
        result.errors.append(f"gf_r_min ({gf_r_min}) must be < gf_r_max ({gf_r_max})")
    if gf_r_min < r_min or gf_r_max > r_max:
        result.warnings.append(
            "Integration interval extends beyond sampling grid"
        )

    seed = config.get("pipeline", {}).get("seed", 42)
    if not isinstance(seed, int) or seed < 0:
        result.valid = False
        result.errors.append(f"Seed must be a non-negative integer, got {seed}")

    return result


# ---------------------------------------------------------------------------
# Convenience: full pre-flight check
# ---------------------------------------------------------------------------

def preflight_check(config: dict) -> ValidationResult:
    """Run all available validation checks based on the config.

    This is meant to be called at the start of the pipeline to catch
    problems before any expensive computation begins.
    """
    result = ValidationResult()
    paths = config.get("paths", {})
    project_root = Path(config.get("_project_root", "."))

    data_dir = Path(paths.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    # Validate curated network
    net_cfg = config.get("network", {})
    edgelist = data_dir / net_cfg.get("curated_network", "curated_153_ppi.edgelist")
    result.merge(validate_edgelist(edgelist))

    # Validate GO map
    go_file = data_dir / net_cfg.get("gene_go_map", "gene_go_map.json")
    if go_file.exists():
        with open(go_file, encoding="utf-8") as f:
            go_map = json.load(f)
        result.merge(validate_go_map(go_map))
    else:
        result.errors.append(f"GO map not found: {go_file}")
        result.valid = False

    # Validate STRING file (warning only — not required for curated pipeline)
    string_file = data_dir / net_cfg.get("string_file", "4932.protein.links.v11.5.txt.gz")
    if string_file.exists():
        min_score = net_cfg.get("string_min_score", 700)
        result.merge(validate_string_file(string_file, min_score))

    # Validate pipeline params
    result.merge(validate_pipeline_params(config))

    return result
