"""Internal utilities: warnings and validation helpers for GASM."""

from __future__ import annotations

import warnings


class GASMWarning(UserWarning):
    """Base class for all warnings emitted by GASM."""


class PlatformWarning(GASMWarning):
    """Emitted when the requested compute platform is unavailable."""


class AttributeWarning(GASMWarning):
    """Emitted when attribute specifications are inconsistent with the graphs."""


class ConvergenceWarning(GASMWarning):
    """Emitted when the iterative procedure may not have converged."""


def warn(message: str, category: type[GASMWarning] = GASMWarning) -> None:
    """Emit a GASM warning with a controlled stack level."""
    warnings.warn(message, category, stacklevel=3)
