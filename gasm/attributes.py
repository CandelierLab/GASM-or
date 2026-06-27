"""Attribute handling for GASM (Section 3.2 of the reference article).

Builds the vertex distance matrix ``V`` (eq. 9) and edge distance matrix ``E``
(eq. 10) from user-specified attributes, each with an uncertainty parameter
``rho``. Categorical attributes use eq. (6)-(7); measurable attributes use
eq. (8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .graph import Graph
from .utils import AttributeWarning, warn


@dataclass
class Attribute:
    """Specification of a graph attribute used for matching.

    Parameters
    ----------
    name:
        Key under which the attribute is stored in the networkx node or edge
        data dictionaries.
    on:
        ``'vertex'`` or ``'edge'``.
    kind:
        ``'measurable'`` (a distance can be defined, eq. 8) or ``'categorical'``
        (only equality is meaningful, eq. 6-7).
    rho:
        Uncertainty over the attribute values. A non-negative float, or
        ``'auto'`` to use the standard deviation of all pairwise comparisons as
        a safe upper bound.
    """

    name: str
    on: Literal["vertex", "edge"]
    kind: Literal["measurable", "categorical"] = "measurable"
    rho: float | str = "auto"

    def __post_init__(self):
        if self.on not in ("vertex", "edge"):
            raise ValueError("Attribute.on must be 'vertex' or 'edge'.")
        if self.kind not in ("measurable", "categorical"):
            raise ValueError(
                "Attribute.kind must be 'measurable' or 'categorical'."
            )
        if not (self.rho == "auto" or (isinstance(self.rho, (int, float)) and self.rho >= 0)):
            raise ValueError("Attribute.rho must be a non-negative float or 'auto'.")


def _coerce(spec) -> Attribute:
    if isinstance(spec, Attribute):
        return spec
    if isinstance(spec, dict):
        return Attribute(**spec)
    raise TypeError(
        "Each attribute must be a gasm.Attribute or a dict, got "
        f"{type(spec).__name__}."
    )


def _vertex_values(g: Graph, name: str):
    missing = [v for v in g.nodes if name not in g._raw_node_data[v]]
    if missing:
        warn(
            f"Vertex attribute '{name}' is missing on {len(missing)} vertices; "
            "those comparisons are treated as dissimilar.",
            AttributeWarning,
        )
    return [g._raw_node_data[v].get(name, _MISSING) for v in g.nodes]


def _edge_values(g: Graph, name: str):
    missing = [e for e in g.edges if name not in g._raw_edge_data[e]]
    if missing:
        warn(
            f"Edge attribute '{name}' is missing on {len(missing)} edges; "
            "those comparisons are treated as dissimilar.",
            AttributeWarning,
        )
    return [g._raw_edge_data[e].get(name, _MISSING) for e in g.edges]


class _Missing:
    def __repr__(self):
        return "<missing>"


_MISSING = _Missing()


def _measurable_matrix(va, vb, rho):
    a = np.asarray(va, dtype=np.float64)[:, None]
    b = np.asarray(vb, dtype=np.float64)[None, :]
    diff = a - b
    if rho == 0:
        return (diff == 0).astype(np.float64)
    return np.exp(-(diff ** 2) / (2.0 * rho ** 2))


def _categorical_matrix(va, vb, rho):
    a = np.asarray(va, dtype=object)[:, None]
    b = np.asarray(vb, dtype=object)[None, :]
    equal = a == b
    if rho == 0:
        return equal.astype(np.float64)
    out = np.full(equal.shape, np.exp(-1.0 / (2.0 * rho ** 2)), dtype=np.float64)
    out[equal] = 1.0
    return out


def _auto_rho(va, vb, kind):
    a = np.asarray(va, dtype=np.float64) if kind == "measurable" else None
    if kind == "measurable":
        diff = a[:, None] - np.asarray(vb, dtype=np.float64)[None, :]
        sigma = float(np.std(diff))
    else:
        eq = (np.asarray(va, dtype=object)[:, None] == np.asarray(vb, dtype=object)[None, :])
        sigma = float(np.std(eq.astype(np.float64)))
    # Guard against a degenerate zero std (all values identical).
    return sigma if sigma > 0 else 0.0


def _attribute_matrix(spec: Attribute, ga: Graph, gb: Graph):
    if spec.on == "vertex":
        va = _vertex_values(ga, spec.name)
        vb = _vertex_values(gb, spec.name)
    else:
        va = _edge_values(ga, spec.name)
        vb = _edge_values(gb, spec.name)

    rho = _auto_rho(va, vb, spec.kind) if spec.rho == "auto" else float(spec.rho)

    if spec.kind == "measurable":
        return _measurable_matrix(va, vb, rho)
    return _categorical_matrix(va, vb, rho)


def _coerce_matrices(mats, shape, on: str):
    """Validate and normalise user-provided similarity matrices.

    Accepts a single 2D array or an iterable of 2D arrays, each of the expected
    ``shape``. Values are clipped to ``[0, 1]`` (eq. 9-10 require the factors to
    lie in this interval); out-of-range values trigger an
    :class:`~gasm.utils.AttributeWarning`.
    """
    arr = np.asarray(mats, dtype=np.float64)
    # A single matrix is wrapped into a one-element list; a stack/list of
    # matrices keeps its leading axis.
    if arr.ndim == 2:
        stack = [arr]
    elif arr.ndim == 3:
        stack = list(arr)
    else:
        raise ValueError(
            f"{on} matrices must be a 2D array or a sequence of 2D arrays."
        )

    out = []
    for A in stack:
        if A.shape != shape:
            raise ValueError(
                f"{on} matrix has shape {A.shape}, expected {shape} "
                "(rows follow G1 node/edge order, columns follow G2)."
            )
        if A.min() < 0.0 or A.max() > 1.0:
            warn(
                f"{on} matrix has values outside [0, 1]; clipping to the "
                "interval required by eq. (9-10).",
                AttributeWarning,
            )
            A = np.clip(A, 0.0, 1.0)
        out.append(A)
    return out


def build_matrices(specs, ga: Graph, gb: Graph, vertex_matrices=None, edge_matrices=None):
    """Build the vertex (``V``) and edge (``E``) distance matrices.

    Parameters
    ----------
    specs:
        Iterable of :class:`Attribute` or dict specifications, or ``None``.
    ga, gb:
        The two graphs being matched.
    vertex_matrices:
        Optional precomputed vertex similarity matrix of shape ``(nA, nB)``, or
        a sequence of such matrices, injected directly as extra Hadamard factors
        of ``V`` (eq. 9). Rows follow the ``ga`` node order, columns the ``gb``
        node order. Values must lie in ``[0, 1]`` and are clipped otherwise.
    edge_matrices:
        Optional precomputed edge similarity matrix of shape ``(mA, mB)``, or a
        sequence of such matrices, injected directly as extra Hadamard factors of
        ``E`` (eq. 10). Rows follow the ``ga`` edge order, columns the ``gb``
        edge order. Values must lie in ``[0, 1]`` and are clipped otherwise.

    Returns
    -------
    V:
        Vertex distance matrix of shape ``(nA, nB)`` (eq. 9). All-ones when no
        vertex attribute is specified.
    E:
        Edge distance matrix of shape ``(mA, mB)`` (eq. 10). All-ones when no
        edge attribute is specified.
    """
    V = np.ones((ga.n, gb.n), dtype=np.float64)
    E = np.ones((ga.m, gb.m), dtype=np.float64)

    if specs:
        for spec in specs:
            spec = _coerce(spec)
            A = _attribute_matrix(spec, ga, gb)
            if spec.on == "vertex":
                V *= A
            else:
                E *= A

    if vertex_matrices is not None:
        for A in _coerce_matrices(vertex_matrices, (ga.n, gb.n), "vertex"):
            V *= A

    if edge_matrices is not None:
        for A in _coerce_matrices(edge_matrices, (ga.m, gb.m), "edge"):
            E *= A

    return V, E
