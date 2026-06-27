"""Matching quality and accuracy metrics (Section 2 of the reference article)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp


def _binary_matching_matrix(matchups, nodes_a, nodes_b):
    """Build the binary matching matrix ``M`` (eq. 1)."""
    ia = {label: i for i, label in enumerate(nodes_a)}
    ib = {label: i for i, label in enumerate(nodes_b)}
    M = sp.lil_matrix((len(nodes_a), len(nodes_b)), dtype=np.float64)
    for a, b in matchups:
        if a in ia and b in ib:
            M[ia[a], ib[b]] = 1.0
    return M.tocsr()


def structural_quality(G1, G2, matching) -> float:
    """Structural quality ``qS`` of a matching (eq. 3).

    Parameters
    ----------
    G1, G2:
        The two networkx graphs that were matched.
    matching:
        A :class:`gasm.matching.Matching`, or any iterable of ``(a, b)`` pairs.

    Returns
    -------
    float
        ``qS`` in ``[0, 1]``; higher is a better structural match. Returns ``0``
        when both graphs have no edge.
    """
    matchups = getattr(matching, "matchups", matching)
    nodes_a = list(G1.nodes())
    nodes_b = list(G2.nodes())

    LA = nx.to_scipy_sparse_array(G1, nodelist=nodes_a, format="csr", dtype=np.float64)
    LB = nx.to_scipy_sparse_array(G2, nodelist=nodes_b, format="csr", dtype=np.float64)
    M = _binary_matching_matrix(matchups, nodes_a, nodes_b)

    Z = LA @ M - M @ LB
    discrepancies = float((Z.multiply(Z)).sum())

    mA = G1.number_of_edges()
    mB = G2.number_of_edges()
    if mA == 0 and mB == 0:
        return 0.0

    if G1.is_directed() or G2.is_directed():
        denom = mA + mB
    else:
        muA = sum(1 for u, v in G1.edges() if u == v)
        muB = sum(1 for u, v in G2.edges() if u == v)
        denom = 2 * (mA + mB) - muA - muB

    if denom == 0:
        return 0.0
    return 1.0 - discrepancies / denom


def accuracy(matching, ground_truth) -> float:
    """Accuracy ``gamma`` of a matching against a ground truth (Section 2.2).

    Parameters
    ----------
    matching:
        A :class:`gasm.matching.Matching`, or any iterable of ``(a, b)`` pairs.
    ground_truth:
        Mapping ``{a: b}`` of the true correspondence, or an iterable of
        ``(a, b)`` pairs.

    Returns
    -------
    float
        Proportion of matched pairs that agree with the ground truth.
    """
    matchups = getattr(matching, "matchups", matching)
    if not isinstance(ground_truth, dict):
        ground_truth = dict(ground_truth)
    if not matchups:
        return 0.0
    correct = sum(1 for a, b in matchups if ground_truth.get(a) == b)
    return correct / len(matchups)
