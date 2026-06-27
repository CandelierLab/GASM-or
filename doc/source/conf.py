"""Sphinx configuration for the GASM-or documentation."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

project = "GASM-or"
author = "Raphaël Candelier"
copyright = "2025, Raphaël Candelier"

# The version (release) is read from the installed package so it stays in sync
# with the single source of truth in pyproject.toml. Per the project guidelines,
# check that it is incremented before each documentation build.
try:
    release = _pkg_version("GASM-or")
except Exception:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output: Furo, dark theme ---------------------------------------
html_theme = "furo"
html_title = f"GASM-or {release}"
html_static_path = ["_static"]
html_theme_options = {
    "dark_css_variables": {
        "color-brand-primary": "#8ab4f8",
        "color-brand-content": "#8ab4f8",
    },
}
# Force the dark theme as the default presentation.
html_context = {"default_mode": "dark"}

# -- Autodoc -------------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
# pyopencl is an optional dependency (GPU back-end) and is not installed in the
# documentation build environment, so mock it to let autodoc import gasm.gpu.core.
autodoc_mock_imports = ["pyopencl"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Cross-project references --------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}
