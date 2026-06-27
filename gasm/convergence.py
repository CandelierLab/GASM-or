"""Convergence criteria for the GASM iterative procedure.

The reference article (eq. 30) fixes the number of iterations to
``k_tilde = min(diam_A, diam_B)``. Supp. Fig. S3 shows the accuracy usually
plateaus well before that bound, so this module also provides an adaptive
early-stop criterion that monitors the stabilisation of the row-wise argmax
assignment -- a cheap surrogate for the final LAP that avoids running the LAP at
every iteration -- capped by the diameter for safety.
"""

from __future__ import annotations

import numpy as np


class ConvergenceMonitor:
    """Track convergence of the vertex score matrix across iterations.

    Parameters
    ----------
    mode:
        ``'adaptive'`` (default) for early stopping, or ``'diameter'`` to
        reproduce the fixed-iteration behaviour of the article.
    diameter_cap:
        Hard upper bound on the number of iterations ``k_tilde`` (eq. 30).
    max_iterations:
        Optional manual override of the hard cap.
    tol:
        Relative Frobenius-norm tolerance for the early-stop criterion.
    patience:
        Number of consecutive iterations with an unchanged argmax assignment
        required to declare convergence.
    floor:
        Minimum number of iterations before early stopping is allowed.
    """

    def __init__(
        self,
        mode: str = "adaptive",
        diameter_cap: int = 1,
        max_iterations: int | None = None,
        tol: float = 1e-6,
        patience: int = 2,
        floor: int = 2,
    ):
        if mode not in ("adaptive", "diameter"):
            raise ValueError("convergence must be 'adaptive' or 'diameter'.")
        self.mode = mode
        cap = diameter_cap if max_iterations is None else max_iterations
        self.cap = max(int(cap), 1)
        self.tol = tol
        self.patience = patience
        self.floor = max(floor, 1)
        self._prev_argmax: np.ndarray | None = None
        self._prev_X: np.ndarray | None = None
        self._stable = 0
        self.iterations = 0

    @property
    def max_steps(self) -> int:
        """Maximum number of update iterations that will be performed."""
        return self.cap

    def update(self, k: int, X: np.ndarray) -> bool:
        """Register iteration ``k`` and return ``True`` if iteration should stop.

        Parameters
        ----------
        k:
            Current iteration index (``>= 1``).
        X:
            Current vertex score matrix.
        """
        self.iterations = k
        if k >= self.cap:
            return True
        if self.mode == "diameter":
            return False
        if k < self.floor:
            self._prev_argmax = X.argmax(axis=1)
            self._prev_X = X
            return False

        argmax = X.argmax(axis=1)
        stable = False
        if self._prev_argmax is not None and np.array_equal(argmax, self._prev_argmax):
            self._stable += 1
        else:
            self._stable = 0

        if self._prev_X is not None:
            denom = np.linalg.norm(X)
            if denom > 0:
                rel = np.linalg.norm(X - self._prev_X) / denom
                if rel < self.tol:
                    stable = True

        self._prev_argmax = argmax
        self._prev_X = X
        return stable or self._stable >= self.patience
