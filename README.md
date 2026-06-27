# GASM Official Repository

A python repository implementing an optimized version of the *Graph Attribute and Structure Matching* ([GASM](https://jgaa.info/index.php/jgaa/article/view/2979)) algorithm on both CPU and GPU.

Check out the [documentation](https://candelierlab.github.io/GASM-or/) !

## Installation

```
pip install --upgrade pip
pip install GASM-or
```

Optional extras:

```
pip install "GASM-or[gpu]"        # OpenCL GPU back-end (pyopencl)
pip install "GASM-or[benchmark]"  # matplotlib, for the benchmark scripts
pip install "GASM-or[doc]"        # sphinx + furo, to build the documentation
```

## Quick start

```python
import gasm
import networkx as nx

G1 = nx.gnp_random_graph(30, 0.1, seed=0)
G2 = nx.relabel_nodes(G1, {i: (i + 5) % 30 for i in G1.nodes()})

M = gasm.match(G1, G2)          # GPU by default, CPU fallback
print(M.matchups)               # list of (a, b) matched pairs
print(M.score)                  # global matching score
```

Force the CPU back-end, add attributes, or evaluate the result:

```python
M = gasm.match(G1, G2, platform="CPU")

attrs = [
    gasm.Attribute("weight", on="edge", kind="measurable", rho=0.1),
    gasm.Attribute("label", on="vertex", kind="categorical"),
]
M = gasm.match(G1, G2, attributes=attrs)

ground_truth = {i: (i + 5) % 30 for i in G1.nodes()}
M.accuracy(ground_truth)        # fraction of correct pairs
M.structural_quality(G1, G2)    # structural quality qS
```

## Features

- Faithful implementation of GASM for undirected and directed graphs.
- GPU (OpenCL) and CPU back-ends, with automatic CPU fallback.
- Vertex and edge attributes, categorical or measurable, with per-attribute uncertainty.
- Structure-only or attributes-only matching.
- Automatic complement procedure for dense graphs.
- Refined adaptive convergence criterion (with the article's fixed-iteration behaviour available on demand).
- Pluggable linear assignment solvers (Jonker-Volgenant, auction).

## Dependencies

Requires `numpy`, `scipy` and `networkx`. The GPU back-end additionally requires `pyopencl` and an OpenCL runtime.

## Benchmarks

The `benchmark/` scripts import the local package and offer quick and full modes:

```
python benchmark/accuracy_quality.py --mode quick
python benchmark/speed.py --mode full --platforms CPU GPU
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the [LICENSE](LICENSE) file for the full text.

## Authors and acknowledgment

Crafted with ❤️ by Raphaël Candelier.
