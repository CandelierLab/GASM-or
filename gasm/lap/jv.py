"""Jonker-Volgenant LAP solver (CPU), via SciPy.

:func:`scipy.optimize.linear_sum_assignment` implements the rectangular
shortest-augmenting-path algorithm of Crouse (2016), a Jonker-Volgenant
variant, in polynomial time.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(score: np.ndarray):
    """Return a maximum-score assignment of rows to columns.

    Parameters
    ----------
    score:
        Dense score matrix of shape ``(nA, nB)``; higher is better.

    Returns
    -------
    rows, cols:
        Index arrays such that row ``rows[k]`` is assigned to column
        ``cols[k]``. The assignment has size ``min(nA, nB)``.
    """
    rows, cols = linear_sum_assignment(score, maximize=True)
    return rows, cols
