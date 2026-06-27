"""Graph generators and helpers shared by the GASM benchmarks.

The benchmarks import the *local* package under development (``import gasm``),
never a PyPI release. Run them from a checkout with ``pip install -e .`` so that
``gasm`` resolves to this repository.

The graph families reproduce the spirit of the figures of Candelier, JGAA 29(1),
2025: a binary tree, a star with branches, a circular ladder and Erdos-Renyi
random graphs. Each generator returns a :class:`networkx.Graph`; helpers shuffle
a graph to produce an isomorphic copy together with its ground-truth mapping.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def binary_tree(depth: int) -> nx.Graph:
    """Complete binary tree of the given depth (``2**(depth+1) - 1`` vertices)."""
    return nx.balanced_tree(2, depth)


def star_branched(n_branches: int, branch_length: int) -> nx.Graph:
    """A central vertex with ``n_branches`` paths of length ``branch_length``.

    This family carries strong local symmetries (the branches are
    interchangeable), useful to probe how the noise lifts degeneracies.
    """
    G = nx.Graph()
    G.add_node(0)
    nxt = 1
    for _ in range(n_branches):
        prev = 0
        for _ in range(branch_length):
            G.add_edge(prev, nxt)
            prev = nxt
            nxt += 1
    return G


def circular_ladder(n: int) -> nx.Graph:
    """Circular ladder graph ``CL_n`` (a prism with ``2n`` vertices)."""
    return nx.circular_ladder_graph(n)


def erdos_renyi(n: int, p: float, *, directed: bool = False, seed: int = 0) -> nx.Graph:
    """Erdos-Renyi-Gilbert ``G(n, p)`` random graph."""
    return nx.gnp_random_graph(n, p, seed=seed, directed=directed)


def shuffle_graph(G: nx.Graph, seed: int = 0):
    """Return an isomorphic relabeling of ``G`` and the ground-truth ``{a: b}``."""
    rng = np.random.default_rng(seed)
    perm = list(G.nodes())
    rng.shuffle(perm)
    mapping = {node: perm[i] for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping), mapping


def perturb_graph(G: nx.Graph, n_edits: int, seed: int = 0):
    """Return a perturbed isomorphic copy of ``G`` and its ground truth.

    ``n_edits`` edges are toggled (added if absent, removed if present) after
    relabeling, modelling structural noise on an otherwise isomorphic graph.
    """
    H, mapping = shuffle_graph(G, seed=seed)
    rng = np.random.default_rng(seed + 1)
    nodes = list(H.nodes())
    for _ in range(n_edits):
        u, v = rng.choice(nodes, size=2, replace=False)
        if H.has_edge(u, v):
            H.remove_edge(u, v)
        else:
            H.add_edge(u, v)
    return H, mapping
