"""End-to-end tests of the GASM matching on the CPU platform."""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest

import gasm
from conftest import shuffle_graph


def _match(G1, G2, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gasm.utils.GASMWarning)
        return gasm.match(G1, G2, platform="CPU", seed=0, **kw)


# -- isomorphic recovery -------------------------------------------------


def test_isomorphic_tree(tree):
    # Random trees have automorphisms, so several valid isomorphisms exist; the
    # matching is structurally perfect even if it differs from the exact
    # ground-truth permutation.
    G2, gt = shuffle_graph(tree, seed=5)
    M = _match(tree, G2)
    assert M.structural_quality(tree, G2) == pytest.approx(1.0)


def test_isomorphic_sparse(sparse_graph):
    G2, gt = shuffle_graph(sparse_graph, seed=6)
    M = _match(sparse_graph, G2)
    assert M.accuracy(gt) == 1.0


def test_isomorphic_dense_adaptive(dense_graph):
    G2, gt = shuffle_graph(dense_graph, seed=7)
    M = _match(dense_graph, G2, convergence="adaptive")
    assert M.accuracy(gt) == 1.0


def test_isomorphic_directed(directed_graph):
    G2, gt = shuffle_graph(directed_graph, seed=8)
    M = _match(directed_graph, G2)
    assert M.accuracy(gt) == 1.0


# -- attributes ----------------------------------------------------------


def test_categorical_attribute(tree):
    for v in tree.nodes():
        tree.nodes[v]["color"] = v % 3
    G2, gt = shuffle_graph(tree, seed=9)
    attr = gasm.Attribute("color", "vertex", "categorical", rho=0.1)
    M = _match(tree, G2, attributes=[attr])
    # Three colours cannot break every automorphism of the tree, but the
    # matching stays structurally perfect.
    assert M.structural_quality(tree, G2) == pytest.approx(1.0)


def test_attributes_only(tree):
    # Distinct labels make an attributes-only matching unambiguous.
    for v in tree.nodes():
        tree.nodes[v]["tag"] = float(v)
    G2, gt = shuffle_graph(tree, seed=10)
    attr = gasm.Attribute("tag", "vertex", "measurable", rho=0.01)
    M = _match(tree, G2, attributes=[attr], structure=False)
    assert M.accuracy(gt) == 1.0


# -- LAP solvers ---------------------------------------------------------


def test_jv_auction_parity(sparse_graph):
    G2, _ = shuffle_graph(sparse_graph, seed=11)
    Mjv = _match(sparse_graph, G2, lap="jv")
    Mau = _match(sparse_graph, G2, lap="auction")
    assert Mjv.score == pytest.approx(Mau.score, rel=1e-6)


# -- Matching object -----------------------------------------------------


def test_matching_api(tree):
    G2, gt = shuffle_graph(tree, seed=12)
    M = _match(tree, G2)
    assert len(M) == tree.number_of_nodes()
    a0, b0 = M.matchups[0]
    assert M.matchup_A(a0) == b0
    assert M.matchup_B(b0) == a0
    assert M.matchup_A([a0]) == [b0]
    assert M.scores.shape == (len(M),)
    assert M.score == pytest.approx(M.scores.sum())
    assert M.score_matrix.shape == (tree.number_of_nodes(), G2.number_of_nodes())


def test_match_on_edges(tree):
    G2, _ = shuffle_graph(tree, seed=13)
    M = _match(tree, G2, match_on="edges")
    assert len(M) == min(tree.number_of_edges(), G2.number_of_edges())


# -- validation ----------------------------------------------------------


def test_mixed_directedness_raises():
    with pytest.raises(ValueError):
        gasm.match(nx.path_graph(3), nx.path_graph(3, create_using=nx.DiGraph), platform="CPU")


def test_empty_graph_raises():
    with pytest.raises(ValueError):
        gasm.match(nx.Graph(), nx.path_graph(2), platform="CPU")


def test_structural_quality_bounds(sparse_graph):
    G2, _ = shuffle_graph(sparse_graph, seed=14)
    M = _match(sparse_graph, G2)
    qs = M.structural_quality(sparse_graph, G2)
    assert 0.0 <= qs <= 1.0
