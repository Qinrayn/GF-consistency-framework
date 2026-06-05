"""Shared pytest fixtures for the G-F Consistency Framework test suite."""

import sys
import json
from pathlib import Path

import numpy as np
import networkx as nx
import pytest

# Ensure the project root is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Small synthetic fixtures (no file I/O needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def triangle_graph():
    """3-node triangle graph."""
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
    return G


@pytest.fixture
def triangle_coords():
    """Equilateral triangle embedding for 3 nodes."""
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866],
    ])


@pytest.fixture
def triangle_nodes():
    return ["A", "B", "C"]


@pytest.fixture
def triangle_go_map():
    """Two communities: {A,B} share GO:1, {C} has GO:2."""
    return {
        "A": ["GO:0001", "GO:0001"],
        "B": ["GO:0001", "GO:0003"],
        "C": ["GO:0002"],
    }


@pytest.fixture
def chain_graph():
    """Linear chain: A - B - C - D - E."""
    G = nx.path_graph(5)
    nx.set_node_attributes(G, {i: str(i) for i in range(5)}, "name")
    mapping = {i: chr(ord("A") + i) for i in range(5)}
    return nx.relabel_nodes(G, mapping)


@pytest.fixture
def chain_coords():
    """5 points on a line, evenly spaced."""
    return np.array([[float(i), 0.0] for i in range(5)])


@pytest.fixture
def chain_nodes():
    return ["A", "B", "C", "D", "E"]


@pytest.fixture
def chain_go_map():
    return {
        "A": ["GO:0001"],
        "B": ["GO:0001"],
        "C": ["GO:0002"],
        "D": ["GO:0002"],
        "E": ["GO:0003"],
    }


@pytest.fixture
def empty_go_map():
    return {}


@pytest.fixture
def r_vals_standard():
    """Standard 200-point grid matching the pipeline default."""
    return np.linspace(0.05, 0.55, 200)


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    """Load the default pipeline configuration."""
    from scripts.config_loader import load_config
    return load_config()
