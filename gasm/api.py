"""Public entry point: :func:`match`."""

from __future__ import annotations

import networkx as nx

from . import attributes as attr_mod
from . import lap as lap_registry
from .cpu import core as cpu_core
from .matching import Matching
from .graph import from_networkx
from .utils import AttributeWarning, PlatformWarning, warn


def match(
    G1,
    G2,
    *,
    platform: str = "GPU",
    attributes=None,
    vertex_matrices=None,
    edge_matrices=None,
    structure: bool = True,
    complement: bool = True,
    lap: str = "auto",
    noise: float = 1e-10,
    convergence: str = "adaptive",
    tol: float = 1e-6,
    patience: int = 2,
    max_iterations: int | None = None,
    normalize: bool = True,
    match_on: str = "vertices",
    return_scores: bool = False,
    seed: int | None = None,
) -> Matching:
    """Match two graphs with the GASM algorithm.

    Parameters
    ----------
    G1, G2:
        The two graphs to match, as :class:`networkx.Graph` or
        :class:`networkx.DiGraph`. Both must share the same directedness.
    platform:
        ``'GPU'`` (default) runs on OpenCL, falling back to the CPU with a
        warning when no device is available; ``'CPU'`` forces the reference
        implementation.
    attributes:
        ``None`` for a purely structural matching, or a list of
        :class:`gasm.Attribute` (or equivalent dicts) describing the vertex and
        edge attributes to use, each with its uncertainty ``rho``.
    vertex_matrices:
        Optional precomputed vertex similarity matrix of shape ``(nA, nB)``, or a
        sequence of such matrices, injected directly as extra Hadamard factors of
        the vertex distance matrix ``V`` (eq. 9). Rows follow the ``G1`` node
        order, columns the ``G2`` node order. Values must lie in ``[0, 1]`` and
        are clipped otherwise.
    edge_matrices:
        Optional precomputed edge similarity matrix of shape ``(mA, mB)``, or a
        sequence of such matrices, injected directly as extra Hadamard factors of
        the edge distance matrix ``E`` (eq. 10). Rows follow the ``G1`` edge
        order, columns the ``G2`` edge order. Values must lie in ``[0, 1]`` and
        are clipped otherwise.
    structure:
        When ``False``, ignore the graph structure and match on attributes only.
    complement:
        Allow the complement procedure (eq. 18 / 26) for dense graphs. ``False``
        always uses the original incidence matrices.
    lap:
        Linear assignment solver: ``'auto'``, ``'jv'`` (Jonker-Volgenant) or
        ``'auction'``.
    noise:
        Amplitude ``eta`` of the symmetry-lifting noise (eq. 11). ``0`` disables
        it.
    convergence:
        ``'adaptive'`` (early stopping) or ``'diameter'`` (fixed number of
        iterations, eq. 30).
    tol, patience:
        Parameters of the adaptive convergence criterion.
    max_iterations:
        Hard cap on the number of iterations; defaults to ``min(diam_A, diam_B)``
        (eq. 30).
    normalize:
        Apply the approximate normalization ``fx = 4 dA dB + 1`` (eq. S2).
    match_on:
        ``'vertices'`` (match on the vertex score matrix ``X``) or ``'edges'``
        (match on the edge score matrix ``Y``).
    return_scores:
        For the GPU platform, transfer the score matrix immediately instead of
        lazily. Ignored on the CPU, where scores are always available.
    seed:
        Seed for the noise generator, for reproducible matchings.

    Returns
    -------
    Matching
        The matching result.
    """
    if match_on not in ("vertices", "edges"):
        raise ValueError("match_on must be 'vertices' or 'edges'.")
    if platform not in ("GPU", "CPU"):
        raise ValueError("platform must be 'GPU' or 'CPU'.")

    if G1.is_directed() != G2.is_directed():
        raise ValueError(
            "Both graphs must share the same directedness "
            "(both directed or both undirected)."
        )
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        raise ValueError("Both graphs must have at least one vertex.")

    has_info = bool(attributes) or vertex_matrices is not None or edge_matrices is not None
    if not structure and not has_info:
        warn(
            "structure=False with no attributes: the matching has no information "
            "to rely on and will be arbitrary.",
            AttributeWarning,
        )

    ga = from_networkx(G1)
    gb = from_networkx(G2)
    V, E = attr_mod.build_matrices(
        attributes, ga, gb, vertex_matrices=vertex_matrices, edge_matrices=edge_matrices
    )

    options = dict(
        structure=structure,
        complement=complement,
        noise=noise,
        convergence=convergence,
        tol=tol,
        patience=patience,
        max_iterations=max_iterations,
        normalize=normalize,
        match_on=match_on,
        seed=seed,
    )

    score_matrix = None
    labels_a = labels_b = None
    used_platform = platform

    if platform == "GPU":
        try:
            from .gpu import core as gpu_core

            score_matrix, labels_a, labels_b, _iters, _loader = gpu_core.run(
                ga, gb, V, E, lap=lap, return_scores=return_scores, **options
            )
        except Exception as exc:  # pragma: no cover - depends on host OpenCL
            warn(
                "GPU platform unavailable "
                f"({type(exc).__name__}: {exc}); falling back to the CPU.",
                PlatformWarning,
            )
            used_platform = "CPU"

    if used_platform == "CPU":
        score_matrix, labels_a, labels_b, _iters = cpu_core.run(ga, gb, V, E, **options)

    solver = lap_registry.get_cpu_solver(lap)
    rows, cols = solver(score_matrix)
    return Matching(
        labels_a,
        labels_b,
        rows,
        cols,
        match_on=match_on,
        score_matrix=score_matrix,
    )
