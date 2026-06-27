"""CPU implementation of the GASM iterative procedure (eq. 15-31).

This is the reference implementation, faithful to Candelier (JGAA 29(1), 2025).
It uses dense NumPy score matrices with sparse incidence multiplications and is
the platform used when ``platform='CPU'`` or when no OpenCL device is available.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from .. import graph as graphmod
from ..convergence import ConvergenceMonitor


def _effective_diameter(g, use_comp: bool) -> int:
    """Diameter of the graph actually propagated during the iterations.

    When the complement is used (eq. 18 / 26) the structural information flows
    along the complement graph, whose diameter governs how many message-passing
    steps are needed. Self-loops are irrelevant to distances and are ignored.
    """
    if not use_comp:
        return graphmod.diameter(g)
    n = g.n
    A = (g.adjacency > 0).toarray()
    comp = np.ones((n, n), dtype=bool)
    np.fill_diagonal(comp, False)
    comp &= ~A
    if not g.directed:
        comp |= comp.T
    if not comp.any():
        return 0
    dist = shortest_path(
        sp.csr_matrix(comp.astype(np.float64)),
        directed=g.directed,
        unweighted=True,
    )
    finite = dist[np.isfinite(dist)]
    return int(finite.max()) if finite.size else 0



def _l(Ssp, D):
    """Sparse-left product ``Ssp @ D`` returning a dense array."""
    return np.asarray(Ssp @ D)


def _r(D, Ssp):
    """Dense-right product ``D @ Ssp`` returning a dense array.

    Computed as ``(Ssp.T @ D.T).T`` to keep the multiplication as
    sparse-times-dense, which SciPy supports efficiently.
    """
    return np.asarray((Ssp.transpose() @ D.T).T)


def _init_structure(ga, gb, E):
    """Structure term of the initialization (eq. 15 / eq. 27)."""
    if ga.directed:
        return _r(_l(ga.S, E), gb.S.transpose()) + _r(_l(ga.T, E), gb.T.transpose())
    return _r(_l(ga.R, E), gb.R.transpose())


def run(
    ga,
    gb,
    V,
    E,
    *,
    structure: bool = True,
    complement: bool = True,
    noise: float = 1e-10,
    convergence: str = "adaptive",
    tol: float = 1e-6,
    patience: int = 2,
    max_iterations: int | None = None,
    normalize: bool = True,
    match_on: str = "vertices",
    seed: int | None = None,
):
    """Run the GASM iterations on the CPU.

    Returns
    -------
    score_matrix:
        The converged score matrix the LAP should be run on: the vertex score
        matrix ``X`` (``match_on='vertices'``) or the edge score matrix ``Y``
        (``match_on='edges'``).
    labels_a, labels_b:
        Labels of the rows and columns of ``score_matrix`` (node labels for
        vertices, ``(u, v)`` edge tuples for edges).
    iterations:
        Number of iterations actually performed.
    """
    rng = np.random.default_rng(seed)
    nA, nB = ga.n, gb.n

    # Noise term H, h_uv ~ U[0, eta] (eq. 11).
    if noise and noise > 0:
        H = rng.uniform(0.0, noise, size=(nA, nB))
    else:
        H = np.zeros((nA, nB))
    Vplus = V + H

    do_iterate = structure and ga.m > 0 and gb.m > 0

    # Initialization (eq. 15 / 27). Always uses the original incidence matrices.
    if do_iterate:
        X = Vplus * _init_structure(ga, gb, E)
    else:
        X = Vplus.copy()
    Y = None

    # Normalization factor fx = 4 dA dB + 1 (eq. S2); fy = 1.
    fx = (4.0 * ga.mean_degree * gb.mean_degree + 1.0) if normalize else 1.0

    # Incidence matrices used in the iterations: complements when dense enough
    # (eq. 18 / 26), unless matching on edges (Y must index real edges).
    use_comp = complement and match_on != "edges" and graphmod.use_complement(ga, gb)
    if ga.directed:
        if use_comp:
            _, SA, TA = ga.complement_incidence()
            _, SB, TB = gb.complement_incidence()
        else:
            SA, TA, SB, TB = ga.S, ga.T, gb.S, gb.T
    else:
        if use_comp:
            RA, _, _ = ga.complement_incidence()
            RB, _, _ = gb.complement_incidence()
        else:
            RA, RB = ga.R, gb.R

    # Convergence cap k_tilde = min(diam_A, diam_B) (eq. 30). In 'diameter' mode
    # this faithfully uses the original graph diameters; in 'adaptive' mode the
    # cap follows the iterated (complement) graph so early stopping is not
    # artificially capped below the number of steps needed to propagate.
    if convergence == "diameter":
        cap = max(min(graphmod.diameter(ga), graphmod.diameter(gb)), 1)
    else:
        cap = max(min(_effective_diameter(ga, use_comp), _effective_diameter(gb, use_comp)), 1)
    monitor = ConvergenceMonitor(
        mode=convergence,
        diameter_cap=cap,
        max_iterations=max_iterations,
        tol=tol,
        patience=patience,
        floor=2,
    )

    k = 1
    if do_iterate:
        while not monitor.update(k, X):
            k += 1
            if ga.directed:
                Y = _r(_l(SA.transpose(), X), SB) + _r(_l(TA.transpose(), X), TB)
                X = _r(_l(SA, Y), SB.transpose()) + _r(_l(TA, Y), TB.transpose())
            else:
                Y = _r(_l(RA.transpose(), X), RB)
                X = _r(_l(RA, Y), RB.transpose())
            if normalize:
                X = X / fx

        # Restore isolated vertices (eq. 31): x_{u,v} = nu_{u,v} / fx^{k-1},
        # with nu = V (the initial vertex distances, without noise).
        iso = ga.isolated[:, None] | gb.isolated[None, :]
        if iso.any():
            X = X.copy()
            X[iso] = V[iso] / (fx ** (k - 1))

    if match_on == "edges":
        if Y is None:
            raise ValueError(
                "match_on='edges' requires a structural iteration; the graphs "
                "have no edges or structure is disabled."
            )
        return Y, ga.edges, gb.edges, k
    return X, ga.nodes, gb.nodes, k
