"""GPU back-end tests, skipped when no OpenCL device is available."""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest

import gasm
from conftest import shuffle_graph


def _opencl_available() -> bool:
    try:
        import pyopencl as cl

        return bool(cl.get_platforms())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _opencl_available(), reason="no OpenCL device available"
)


def _match(G1, G2, platform, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gasm.utils.GASMWarning)
        return gasm.match(G1, G2, platform=platform, seed=0, **kw)


def test_gpu_cpu_parity_undirected():
    G = nx.gnp_random_graph(20, 0.2, seed=2)
    G2, gt = shuffle_graph(G, seed=6)
    Mc = _match(G, G2, "CPU")
    Mg = _match(G, G2, "GPU")
    assert set(Mc.matchups) == set(Mg.matchups)
    assert Mg.accuracy(gt) == 1.0


def test_gpu_cpu_parity_directed():
    G = nx.gnp_random_graph(12, 0.3, seed=4, directed=True)
    G2, gt = shuffle_graph(G, seed=8)
    Mc = _match(G, G2, "CPU")
    Mg = _match(G, G2, "GPU")
    assert set(Mc.matchups) == set(Mg.matchups)


def test_gpu_dense_complement():
    G = nx.gnp_random_graph(14, 0.7, seed=3)
    G2, _ = shuffle_graph(G, seed=7)
    Mg = _match(G, G2, "GPU")
    assert Mg.structural_quality(G, G2) == pytest.approx(1.0)
