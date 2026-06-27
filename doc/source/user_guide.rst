User guide
==========

This guide describes the options of :func:`gasm.match` and the concepts behind
them. All references point to the corresponding entries of the
:doc:`api`.

Platforms
---------

GASM-or ships two back-ends, selected with the ``platform`` argument of
:func:`gasm.match`:

* ``'GPU'`` (default) runs the iterations on an OpenCL device via the
  :mod:`gasm.gpu.core` module. If no device is available, it falls back to the
  CPU and emits a :class:`gasm.utils.PlatformWarning`.
* ``'CPU'`` forces the reference implementation in :mod:`gasm.cpu.core`.

Both back-ends produce the same matching on graphs without symmetries; on graphs
with local symmetries the infinitesimal noise (see below) lifts the degeneracies,
and several equally valid solutions may exist.

Structure and attributes
-------------------------

By default the matching is purely structural. Attributes are added through a list
of :class:`gasm.Attribute`, each describing a vertex or edge attribute with an
uncertainty ``rho``:

* ``kind='measurable'`` uses a Gaussian similarity on the attribute difference
  (eq. 8 of the article);
* ``kind='categorical'`` uses an equality-based similarity (eq. 7);
* ``rho='auto'`` estimates ``rho`` from the spread of the attribute values.

Set ``structure=False`` to match on attributes only. The vertex and edge
similarity matrices are assembled by :func:`gasm.attributes.build_matrices`.

Injecting precomputed similarity matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes a similarity between vertices (or edges) is already available as a
matrix rather than as a per-element attribute -- for instance a temporal
correlation matrix between dynamic vertex quantities. Such matrices can be fed
directly to :func:`gasm.match` through ``vertex_matrices`` and
``edge_matrices``, bypassing the :class:`gasm.Attribute` machinery. They are
injected as additional Hadamard factors :math:`\mathcal{A}_i` of ``V`` (eq. 9)
and ``E`` (eq. 10):

* ``vertex_matrices`` expects a matrix of shape :math:`(n_A, n_B)` (or a list of
  them); rows follow the ``G1`` node order, columns the ``G2`` node order.
* ``edge_matrices`` expects a matrix of shape :math:`(m_A, m_B)` (or a list of
  them); rows follow the ``G1`` edge order, columns the ``G2`` edge order.

Values must lie in :math:`[0, 1]`, with ``0`` meaning dissimilar and ``1``
meaning similar; out-of-range values are clipped with a
:class:`gasm.utils.AttributeWarning`. Precomputed matrices and
:class:`gasm.Attribute` specifications can be combined freely.

The complement procedure
------------------------

For dense graphs it is faster to propagate information along the *complement*
graph. :func:`gasm.match` enables this automatically following the density
criterion of the article (eq. 18 / 26); pass ``complement=False`` to always use
the original incidence matrices. The decision is implemented in
:func:`gasm.graph.use_complement`.

Convergence
-----------

The number of iterations is bounded by the graph diameter (eq. 30). GASM-or adds
an adaptive early-stopping criterion, implemented in
:class:`gasm.convergence.ConvergenceMonitor`, which stops as soon as the row-wise
argmax assignment is stable or the score matrix barely changes:

* ``convergence='adaptive'`` (default) enables early stopping, with parameters
  ``tol`` and ``patience``;
* ``convergence='diameter'`` reproduces the fixed number of iterations of the
  article.

The hard cap can also be set manually with ``max_iterations``.

Linear assignment
-----------------

The final assignment is solved by a linear assignment problem (LAP) solver,
selected with ``lap``:

* ``'jv'`` -- Jonker-Volgenant, via :func:`scipy.optimize.linear_sum_assignment`;
* ``'auction'`` -- Bertsekas auction algorithm with epsilon-scaling;
* ``'auto'`` -- picks a solver automatically.

Solvers are registered in :mod:`gasm.lap`, which can be extended with additional
algorithms.

Matching on edges
-----------------

By default the assignment is computed on the vertex score matrix. Set
``match_on='edges'`` to assign edges instead, using the edge score matrix
``Y``. This option requires the CPU back-end and disables the complement
procedure so that the edge labels remain meaningful.

Evaluation metrics
------------------

* :func:`gasm.accuracy` -- fraction of pairs matching a known ground truth.
* :func:`gasm.structural_quality` -- the structural quality :math:`q_S`
  (eq. 3), in :math:`[0, 1]`, equal to ``1`` for a perfect structural match.
