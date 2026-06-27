Developer guide
===============

This page documents the internal organisation of GASM-or for contributors.

Architecture
------------

The package is split into a platform-agnostic core and two back-ends:

* :mod:`gasm.api` -- input validation, attribute assembly and platform dispatch
  for :func:`gasm.match`.
* :mod:`gasm.graph` -- conversion of a :class:`networkx.Graph` into the sparse
  incidence structures ``R`` (undirected) or ``S`` and ``T`` (directed),
  density and diameter computations, and the complement procedure.
* :mod:`gasm.attributes` -- the vertex distance matrix ``V`` and edge distance
  matrix ``E`` (eq. 6-10).
* :mod:`gasm.convergence` -- the adaptive convergence criterion.
* :mod:`gasm.metrics` -- the accuracy and structural-quality metrics.
* :mod:`gasm.matching` -- the :class:`gasm.Matching` result object.
* :mod:`gasm.cpu.core` -- the reference CPU iterations (eq. 15-31).
* :mod:`gasm.gpu.core` -- the OpenCL iterations, with the kernels in
  ``gasm/gpu/kernels/gasm.cl``.
* :mod:`gasm.lap` -- the extensible registry of LAP solvers.

The notations follow the reference article; equation numbers are quoted in the
source where relevant.

Running the tests
-----------------

The test suite lives in ``benchmark/tests`` and uses :mod:`pytest`:

.. code-block:: bash

   pip install -e ".[dev]"
   pytest benchmark/tests

The GPU tests are skipped automatically when no OpenCL device is available.

Benchmarks
----------

The benchmark scripts in ``benchmark`` import the *local* package and offer a
quick mode (fast, indicative results) and a full mode (paper-scale results):

.. code-block:: bash

   python benchmark/accuracy_quality.py --mode quick
   python benchmark/speed.py --mode full --platforms CPU GPU

Building the documentation
--------------------------

The documentation is built with Sphinx:

.. code-block:: bash

   sphinx-build doc/source doc/build

Before each build, check that the version (``release`` in
``doc/source/conf.py``) is consistent with the package version in
``pyproject.toml``; ``conf.py`` reads it from the installed package.

Releasing
---------

Releases are published to PyPI automatically by the GitHub Actions workflow
``.github/workflows/publish.yml`` when a GitHub release is published, using PyPI
Trusted Publishing (OIDC). The release flow is:

#. bump the version in ``pyproject.toml``;
#. create a GitHub release/tag;
#. the workflow builds the distributions and uploads them to PyPI.
