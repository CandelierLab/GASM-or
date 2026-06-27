"""Linear Assignment Problem (LAP) solver registry.

GASM ends with a LAP on the converged vertex (or edge) score matrix, searching
for a maximum-score assignment. Solvers are registered by name so that new ones
can be added without touching the rest of the code base. The ``lap`` argument of
:func:`gasm.match` selects one of them, ``'auto'`` picking the best available
solver for the active platform.

Currently available:

- ``'jv'``: Jonker-Volgenant, via :func:`scipy.optimize.linear_sum_assignment`
  (CPU).
- ``'auction'``: Bertsekas auction algorithm with epsilon-scaling (native GPU
  implementation; falls back to a NumPy reference on CPU).
"""

from __future__ import annotations

from typing import Callable

_CPU_SOLVERS: dict[str, Callable] = {}


def register_cpu_solver(name: str, func: Callable) -> None:
    """Register a CPU LAP solver under ``name``."""
    _CPU_SOLVERS[name] = func


def get_cpu_solver(name: str) -> Callable:
    """Return the CPU LAP solver registered under ``name``.

    ``'auto'`` resolves to the Jonker-Volgenant solver.
    """
    if name == "auto":
        name = "jv"
    if name not in _CPU_SOLVERS:
        raise ValueError(
            f"Unknown LAP solver '{name}'. Available: {sorted(_CPU_SOLVERS)} or 'auto'."
        )
    return _CPU_SOLVERS[name]


def available() -> list[str]:
    """List the registered CPU LAP solver names."""
    return sorted(_CPU_SOLVERS)


# Register built-in solvers.
from . import jv as _jv  # noqa: E402
from . import auction as _auction  # noqa: E402

register_cpu_solver("jv", _jv.solve)
register_cpu_solver("auction", _auction.solve)
