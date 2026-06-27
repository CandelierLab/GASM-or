"""The :class:`Matching` result object returned by :func:`gasm.match`."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from . import metrics


class Matching:
    """Result of a graph matching.

    A matching associates vertices (or edges) of graph ``A`` with vertices (or
    edges) of graph ``B``. The full score matrix is transferred from the GPU
    lazily: accessing :attr:`score`, :attr:`scores` or :attr:`score_matrix`
    triggers a single device-to-host transfer that is then cached.

    Parameters
    ----------
    labels_a, labels_b:
        Ordered labels of the matched elements of each graph (vertices or
        edges), indexed by the internal integer indices.
    rows, cols:
        Assignment index arrays: internal index ``rows[k]`` of ``A`` is matched
        with internal index ``cols[k]`` of ``B``.
    match_on:
        ``'vertices'`` or ``'edges'``.
    score_matrix:
        The dense score matrix, if already available on the host (CPU path).
    score_loader:
        Callable returning the dense score matrix, used for lazy GPU transfer.
    """

    def __init__(
        self,
        labels_a,
        labels_b,
        rows,
        cols,
        match_on: str = "vertices",
        score_matrix: np.ndarray | None = None,
        score_loader: Callable[[], np.ndarray] | None = None,
    ):
        self._labels_a = list(labels_a)
        self._labels_b = list(labels_b)
        self._rows = np.asarray(rows, dtype=np.int64)
        self._cols = np.asarray(cols, dtype=np.int64)
        self.match_on = match_on
        self._score_matrix = score_matrix
        self._score_loader = score_loader

    # -- core matchup access ---------------------------------------------

    @property
    def matchups(self) -> list[tuple]:
        """List of ``(a, b)`` matched pairs using original labels."""
        return [
            (self._labels_a[r], self._labels_b[c])
            for r, c in zip(self._rows, self._cols)
        ]

    def matchup_A(self, nodes=None):
        """Matchups keyed by elements of ``A``.

        Parameters
        ----------
        nodes:
            If ``None``, return the full ``{a: b}`` dict. If a single label,
            return its match. If an iterable of labels, return the list of
            matches.
        """
        mapping = {
            self._labels_a[r]: self._labels_b[c]
            for r, c in zip(self._rows, self._cols)
        }
        return self._select(mapping, nodes)

    def matchup_B(self, nodes=None):
        """Matchups keyed by elements of ``B`` (see :meth:`matchup_A`)."""
        mapping = {
            self._labels_b[c]: self._labels_a[r]
            for r, c in zip(self._rows, self._cols)
        }
        return self._select(mapping, nodes)

    @staticmethod
    def _select(mapping, nodes):
        if nodes is None:
            return mapping
        if isinstance(nodes, (list, tuple, set, np.ndarray)):
            return [mapping.get(n) for n in nodes]
        return mapping.get(nodes)

    # -- scores (lazy) ----------------------------------------------------

    def _ensure_score_matrix(self) -> np.ndarray:
        if self._score_matrix is None:
            if self._score_loader is None:
                raise RuntimeError("No score matrix is available for this matching.")
            self._score_matrix = np.asarray(self._score_loader())
        return self._score_matrix

    @property
    def score_matrix(self) -> np.ndarray:
        """The full dense score matrix (lazily transferred from the device)."""
        return self._ensure_score_matrix()

    @property
    def scores(self) -> np.ndarray:
        """Per-matchup scores as a NumPy array (lazily transferred)."""
        X = self._ensure_score_matrix()
        return X[self._rows, self._cols]

    @property
    def score(self) -> float:
        """Global matching score (sum of per-matchup scores, lazily transferred)."""
        return float(self.scores.sum())

    # -- metrics ----------------------------------------------------------

    def accuracy(self, ground_truth) -> float:
        """Accuracy ``gamma`` against a ground truth (see :func:`gasm.accuracy`)."""
        return metrics.accuracy(self, ground_truth)

    def structural_quality(self, G1, G2) -> float:
        """Structural quality ``qS`` (see :func:`gasm.structural_quality`)."""
        return metrics.structural_quality(G1, G2, self)

    # -- dunder -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterable[tuple]:
        return iter(self.matchups)

    def __repr__(self) -> str:
        return (
            f"<Matching {len(self)} pairs on {self.match_on}; "
            f"score {'computed' if self._score_matrix is not None else 'on device'}>"
        )
