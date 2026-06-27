"""Bertsekas auction algorithm with epsilon-scaling (CPU reference).

This is a NumPy reference implementation, used as a CPU fallback and to validate
the native OpenCL auction kernel. With epsilon-scaling the auction converges to
an optimal assignment (identical total score to Jonker-Volgenant); the GASM
noise term lifts ties so both solvers return the same matching. For production
CPU use prefer the ``'jv'`` solver, which is considerably faster in pure Python.
"""

from __future__ import annotations

import numpy as np


def _auction_round(C: np.ndarray, eps: float, prices: np.ndarray):
    nr, nc = C.shape
    person_to_obj = np.full(nr, -1, dtype=np.int64)
    obj_to_person = np.full(nc, -1, dtype=np.int64)
    unassigned = list(range(nr))

    while unassigned:
        i = unassigned.pop()
        values = C[i] - prices
        j = int(np.argmax(values))
        best = values[j]
        if nc > 1:
            saved = values[j]
            values[j] = -np.inf
            second = values.max()
            values[j] = saved
        else:
            second = -np.inf
        bid = best - second + eps
        prices[j] += bid
        prev = obj_to_person[j]
        if prev != -1:
            person_to_obj[prev] = -1
            unassigned.append(int(prev))
        obj_to_person[j] = i
        person_to_obj[i] = j

    return person_to_obj


def _auction(C: np.ndarray) -> np.ndarray:
    nr, nc = C.shape
    maxabs = max(1.0, float(np.abs(C).max()))
    eps = maxabs
    # The final assignment is within ``nr * eps`` of the optimum, so eps_final
    # must be small relative to the score scale to recover the exact optimum.
    eps_final = maxabs * 1e-9 / max(nr, 1)
    prices = np.zeros(nc, dtype=np.float64)
    assign = np.full(nr, -1, dtype=np.int64)

    while True:
        assign = _auction_round(C, eps, prices)
        if eps <= eps_final:
            break
        eps = max(eps / 4.0, eps_final)
    return assign


def solve(score: np.ndarray):
    """Return a maximum-score assignment of rows to columns.

    Parameters
    ----------
    score:
        Dense score matrix of shape ``(nA, nB)``; higher is better.

    Returns
    -------
    rows, cols:
        Index arrays of the assignment, of size ``min(nA, nB)``.
    """
    score = np.asarray(score, dtype=np.float64)
    nr, nc = score.shape
    transposed = False
    if nr > nc:
        score = score.T
        transposed = True
        nr, nc = nc, nr

    assign = _auction(score)
    rows = np.arange(nr)
    cols = assign
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]
