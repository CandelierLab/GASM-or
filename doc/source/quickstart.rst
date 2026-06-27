Quickstart
==========

Matching two graphs
--------------------

The single entry point is :func:`gasm.match`. It takes two
:class:`networkx.Graph` (or :class:`networkx.DiGraph`) objects and returns a
:class:`gasm.Matching`::

    import gasm
    import networkx as nx

    G1 = nx.gnp_random_graph(30, 0.1, seed=0)
    G2 = nx.relabel_nodes(G1, {i: (i + 5) % 30 for i in G1.nodes()})

    M = gasm.match(G1, G2)

By default the matching runs on the GPU and falls back to the CPU when no OpenCL
device is available. Use ``platform='CPU'`` to force the CPU back-end::

    M = gasm.match(G1, G2, platform='CPU')

Reading the result
------------------

A :class:`gasm.Matching` exposes the matched pairs and their scores::

    M.matchups          # list of (a, b) pairs with the original labels
    M.matchup_A()       # dict {a: b}
    M.matchup_B()       # dict {b: a}
    M.score             # global matching score
    M.scores            # per-pair scores
    M.score_matrix      # full vertex score matrix

On the GPU back-end the score matrix is transferred from the device only when one
of :attr:`~gasm.Matching.score`, :attr:`~gasm.Matching.scores` or
:attr:`~gasm.Matching.score_matrix` is accessed.

Using attributes
----------------

Vertex and edge attributes are declared with :class:`gasm.Attribute`. Each
attribute has an uncertainty ``rho`` (eq. 7-8 of the article), or ``'auto'`` to
estimate it::

    attrs = [
        gasm.Attribute('weight', on='edge', kind='measurable', rho=0.1),
        gasm.Attribute('label', on='vertex', kind='categorical'),
    ]
    M = gasm.match(G1, G2, attributes=attrs)

Evaluating a matching
---------------------

When a ground truth is known, :func:`gasm.accuracy` (or
:meth:`gasm.Matching.accuracy`) reports the fraction of correct pairs, while
:func:`gasm.structural_quality` (or :meth:`gasm.Matching.structural_quality`)
reports the structural quality :math:`q_S`::

    ground_truth = {i: (i + 5) % 30 for i in G1.nodes()}
    M.accuracy(ground_truth)
    M.structural_quality(G1, G2)
