Installation
============

GASM-or is published on PyPI under the name ``GASM-or`` and imported as
``gasm``.

Basic installation
-------------------

.. code-block:: bash

   pip install --upgrade pip
   pip install GASM-or

This installs the CPU back-end and its dependencies (:mod:`numpy`, :mod:`scipy`
and :mod:`networkx`).

Optional features
-----------------

The GPU back-end, the benchmarks and the documentation tooling are available as
optional extras:

.. code-block:: bash

   pip install "GASM-or[gpu]"        # OpenCL back-end (pyopencl)
   pip install "GASM-or[benchmark]"  # matplotlib, for the benchmark scripts
   pip install "GASM-or[doc]"        # sphinx + furo, to build these docs

GPU requirements
----------------

The GPU back-end relies on :mod:`pyopencl` and a working OpenCL runtime with at
least one usable device. When no OpenCL device is available, :func:`gasm.match`
automatically falls back to the CPU back-end and emits a
:class:`gasm.utils.PlatformWarning`.

Development install
-------------------

To work on the package or run the benchmarks against the local sources:

.. code-block:: bash

   git clone https://github.com/CandelierLab/GASM-or.git
   cd GASM-or
   pip install -e ".[dev]"
