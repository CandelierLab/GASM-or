"""Shared fixtures and helpers for the GASM test suite."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest


def shuffle_graph(G, seed=0):
    """Return a relabeled copy of ``G`` and the ground-truth mapping ``{a: b}``."""
    rng = np.random.default_rng(seed)
    perm = list(G.nodes())
    rng.shuffle(perm)
    mapping = {node: perm[i] for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping), mapping


@pytest.fixture
def tree():
    return nx.random_labeled_tree(15, seed=1)


@pytest.fixture
def sparse_graph():
    return nx.gnp_random_graph(20, 0.2, seed=2)


@pytest.fixture
def dense_graph():
    return nx.gnp_random_graph(14, 0.7, seed=3)


@pytest.fixture
def directed_graph():
    return nx.gnp_random_graph(12, 0.3, seed=4, directed=True)
