"""Internal graph representation for GASM.

Converts a :class:`networkx.Graph` or :class:`networkx.DiGraph` into the sparse
incidence structures used by the iterative procedure, following the notations of
Candelier, *Graph Matching Based on Similarities in Structure and Attributes*,
JGAA 29(1) 289-320 (2025).

Notations
---------
- Undirected graphs use the unoriented incidence matrix ``R`` (eq. 14), of shape
  ``n x m``: each column is an edge with two non-zero entries (one for a
  self-loop).
- Directed graphs use the source-edge matrix ``S`` and terminus-edge matrix
  ``T`` (eq. 24-25), both of shape ``n x m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path


@dataclass
class Graph:
    """Sparse, GASM-ready view of a networkx graph.

    Attributes
    ----------
    nodes:
        Ordered list of the original networkx node labels. The position in this
        list is the internal integer index used by all matrices.
    directed:
        Whether the graph is directed.
    n, m, mu:
        Number of vertices, edges and self-loops.
    edges:
        Ordered list of ``(u, v)`` edges (original labels), one per column of the
        incidence matrices.
    R:
        Unoriented incidence matrix (``n x m``), CSR. ``None`` for directed
        graphs.
    S, T:
        Source-edge and terminus-edge matrices (``n x m``), CSR. ``None`` for
        undirected graphs.
    adjacency:
        Sparse adjacency matrix ``Lambda`` (``n x n``), CSR.
    degree:
        Vertex degree (out-degree for directed graphs), as a dense vector.
    isolated:
        Boolean mask of isolated vertices (degree 0, ignoring self-loops).
    """

    nodes: list
    directed: bool
    n: int
    m: int
    mu: int
    edges: list
    adjacency: sp.csr_matrix
    degree: np.ndarray
    isolated: np.ndarray
    R: sp.csr_matrix | None = None
    S: sp.csr_matrix | None = None
    T: sp.csr_matrix | None = None
    _node_index: dict | None = None
    _raw_node_data: dict | None = None
    _raw_edge_data: dict | None = None

    @property
    def node_index(self) -> dict:
        """Mapping from original node label to internal integer index."""
        if self._node_index is None:
            self._node_index = {label: i for i, label in enumerate(self.nodes)}
        return self._node_index

    @property
    def mean_degree(self) -> float:
        """Average degree (out-degree for directed graphs)."""
        return float(self.m) / self.n if self.n else 0.0

    # -- complement -------------------------------------------------------

    def complement_incidence(self):
        """Return the incidence matrices of the complement graph.

        For undirected graphs returns ``(R_bar, None, None)``; for directed
        graphs returns ``(None, S_bar, T_bar)``. Self-loops are complemented as
        well (a vertex without a self-loop in ``G`` has one in ``G_bar``).
        """
        if self.directed:
            return _directed_complement(self)
        return _undirected_complement(self)


def from_networkx(graph: nx.Graph) -> Graph:
    """Build a :class:`Graph` from a networkx graph.

    Parameters
    ----------
    graph:
        A :class:`networkx.Graph` or :class:`networkx.DiGraph`. Multigraphs are
        not supported.
    """
    if graph.is_multigraph():
        raise ValueError("GASM does not support multigraphs.")

    directed = graph.is_directed()
    nodes = list(graph.nodes())
    n = len(nodes)
    index = {label: i for i, label in enumerate(nodes)}

    edges = list(graph.edges())
    m = len(edges)
    mu = sum(1 for u, v in edges if u == v)

    # Incidence structures.
    rows_s, rows_t, cols = [], [], []
    for j, (u, v) in enumerate(edges):
        rows_s.append(index[u])
        rows_t.append(index[v])
        cols.append(j)

    shape = (n, m)
    if directed:
        S = sp.csr_matrix(
            (np.ones(m), (rows_s, cols)), shape=shape, dtype=np.float64
        )
        T = sp.csr_matrix(
            (np.ones(m), (rows_t, cols)), shape=shape, dtype=np.float64
        )
        R = None
    else:
        data = np.ones(2 * m)
        all_rows = rows_s + rows_t
        all_cols = cols + cols
        R = sp.csr_matrix(
            (data, (all_rows, all_cols)), shape=shape, dtype=np.float64
        )
        # Self-loops must appear once, not twice, in the unoriented incidence.
        for j, (u, v) in enumerate(edges):
            if u == v:
                R[index[u], j] = 1.0
        R = R.tocsr()
        S = T = None

    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=nodes, format="csr", dtype=np.float64
    ).tocsr()

    if directed:
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
    else:
        degree = np.asarray(
            (adjacency - sp.diags(adjacency.diagonal())).sum(axis=1)
        ).ravel() + adjacency.diagonal()

    # Isolated vertices: no incident edge other than possibly a self-loop.
    if directed:
        incident = np.asarray((S + T).sum(axis=1)).ravel() if m else np.zeros(n)
    else:
        incident = np.asarray(R.sum(axis=1)).ravel() if m else np.zeros(n)
    isolated = incident == 0

    g = Graph(
        nodes=nodes,
        directed=directed,
        n=n,
        m=m,
        mu=mu,
        edges=edges,
        adjacency=adjacency,
        degree=degree,
        isolated=isolated,
        R=R,
        S=S,
        T=T,
    )
    g._node_index = index
    g._raw_node_data = {v: dict(graph.nodes[v]) for v in nodes}
    g._raw_edge_data = {e: dict(graph.edges[e]) for e in edges}
    return g


def diameter(graph: Graph) -> int:
    """Return the graph diameter as the largest finite shortest-path distance.

    This robustly handles disconnected graphs by ignoring unreachable pairs,
    and directed graphs by using directed shortest paths (eq. 30).
    """
    if graph.m == 0 or graph.n <= 1:
        return 0
    dist = shortest_path(
        graph.adjacency, method="D", directed=graph.directed, unweighted=True
    )
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return 0
    return int(finite.max())


def use_complement(ga: Graph, gb: Graph) -> bool:
    """Decide whether to use graph complements (eq. 18 / eq. 26).

    Undirected: complement when ``4(mA + mB) > nA(nA+1) + nB(nB+1)``.
    Directed: complement when ``2(mA + mB) > nA^2 + nB^2``.
    """
    na, nb = ga.n, gb.n
    ma, mb = ga.m, gb.m
    if ga.directed:
        return 2 * (ma + mb) > na * na + nb * nb
    return 4 * (ma + mb) > na * (na + 1) + nb * (nb + 1)


def _undirected_complement(g: Graph):
    """Incidence matrix of the undirected complement graph (with self-loops)."""
    n = g.n
    adj = g.adjacency.toarray() > 0
    # Complement adjacency including self-loops, symmetric.
    comp = np.ones((n, n), dtype=bool)
    comp[adj] = False
    comp = np.triu(comp)  # upper triangle incl. diagonal -> unique edges
    rows, cols = np.nonzero(comp)
    m_bar = rows.size
    inc_rows, inc_cols, data = [], [], []
    for j, (u, v) in enumerate(zip(rows, cols)):
        if u == v:
            inc_rows.append(u)
            inc_cols.append(j)
            data.append(1.0)
        else:
            inc_rows.extend([u, v])
            inc_cols.extend([j, j])
            data.extend([1.0, 1.0])
    R_bar = sp.csr_matrix(
        (data, (inc_rows, inc_cols)), shape=(n, m_bar), dtype=np.float64
    )
    return R_bar, None, None


def _directed_complement(g: Graph):
    """Source/terminus matrices of the directed complement graph."""
    n = g.n
    adj = g.adjacency.toarray() > 0
    comp = np.ones((n, n), dtype=bool)
    comp[adj] = False
    rows, cols = np.nonzero(comp)  # rows = source, cols = target
    m_bar = rows.size
    edge_cols = np.arange(m_bar)
    S_bar = sp.csr_matrix(
        (np.ones(m_bar), (rows, edge_cols)), shape=(n, m_bar), dtype=np.float64
    )
    T_bar = sp.csr_matrix(
        (np.ones(m_bar), (cols, edge_cols)), shape=(n, m_bar), dtype=np.float64
    )
    return None, S_bar, T_bar
