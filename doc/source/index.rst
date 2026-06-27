GASM-or documentation
======================

**GASM-or** is a fast implementation of the *Graph Attributes and Structure
Matching* (GASM) algorithm of Candelier, *Graph Matching Based on Similarities in
Structure and Attributes*, Journal of Graph Algorithms and Applications
**29**\ (1), 289-320 (2025), with both GPU (OpenCL) and CPU back-ends.

Given two graphs, GASM looks for a vertex-to-vertex correspondence that maximises
a similarity score integrating both the connectivity (structure) and the vertex
and edge attributes. The main entry point is :func:`gasm.match`, which returns a
:class:`gasm.Matching` object::

    import gasm
    import networkx as nx

    G1 = nx.random_labeled_tree(20)
    G2 = nx.relabel_nodes(G1, {i: (i + 7) % 20 for i in G1.nodes()})

    M = gasm.match(G1, G2)
    print(M.matchups)

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

.. toctree::
   :maxdepth: 2
   :caption: Developers

   developer

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
