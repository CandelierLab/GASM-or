"""GASM -- Graph Attributes and Structure Matching.

A fast graph matching library implementing the GASM algorithm of Candelier,
*Graph Matching Based on Similarities in Structure and Attributes*,
Journal of Graph Algorithms and Applications 29(1), 289-320 (2025),
with GPU (OpenCL) and CPU back-ends.

Typical usage::

    import gasm
    M = gasm.match(G1, G2)
    pairs = M.matchups
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .api import match
from .attributes import Attribute
from .matching import Matching
from .metrics import accuracy, structural_quality

try:
    __version__ = version("GASM-or")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

__all__ = [
    "match",
    "Matching",
    "Attribute",
    "accuracy",
    "structural_quality",
    "__version__",
]
