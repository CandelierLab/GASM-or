"""GPU (OpenCL/pyopencl) implementation of the GASM iterations.

The expensive part of GASM is the repeated sparse-times-dense products of the
iteration equations (eq. 16-17 / 28-29). This back-end runs them on the device
in single precision, using CSR incidence matrices and the kernels in
``kernels/gasm.cl``. Initialization, isolated-vertex restoration (eq. 31) and the
final LAP are performed on the host.

Importing this module requires :mod:`pyopencl` and at least one usable OpenCL
device; otherwise an exception is raised and :func:`gasm.match` falls back to the
CPU back-end.
"""

from __future__ import annotations

import os

import numpy as np
import pyopencl as cl
import scipy.sparse as sp

from .. import graph as graphmod
from ..cpu.core import _effective_diameter, _init_structure

_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "kernels", "gasm.cl")

_CTX = None
_QUEUE = None
_PROG = None
_KERNELS = None

_KERNEL_NAMES = ("spmm", "transpose", "scale", "hadamard", "add", "row_argmax")


def _ensure_context():
    """Create (once) the OpenCL context, queue and compiled kernels."""
    global _CTX, _QUEUE, _PROG, _KERNELS
    if _CTX is None:
        _CTX = cl.create_some_context(interactive=False)
        _QUEUE = cl.CommandQueue(_CTX)
        with open(_KERNEL_PATH, "r", encoding="utf-8") as fh:
            _PROG = cl.Program(_CTX, fh.read()).build()
        _KERNELS = {name: cl.Kernel(_PROG, name) for name in _KERNEL_NAMES}
    return _CTX, _QUEUE, _KERNELS


class _DenseBuffer:
    """A dense (rows x cols) float32 device buffer, row-major."""

    def __init__(self, ctx, rows, cols, host=None):
        self.rows = rows
        self.cols = cols
        mf = cl.mem_flags
        if host is None:
            self.buf = cl.Buffer(ctx, mf.READ_WRITE, size=rows * cols * 4 or 4)
        else:
            arr = np.ascontiguousarray(host, dtype=np.float32)
            self.buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr)

    def to_host(self, queue):
        out = np.empty((self.rows, self.cols), dtype=np.float32)
        cl.enqueue_copy(queue, out, self.buf)
        return out


class _CSRBuffer:
    """A CSR matrix (M x K) on the device."""

    def __init__(self, ctx, matrix):
        m = matrix.tocsr()
        self.M, self.K = m.shape
        mf = cl.mem_flags
        indptr = m.indptr.astype(np.int32)
        indices = m.indices.astype(np.int32)
        data = m.data.astype(np.float32)
        self.row_ptr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indptr)
        self.col_idx = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indices)
        self.vals = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)


def _spmm(prog, queue, ctx, csr, dense):
    """Return ``csr @ dense`` as a new :class:`_DenseBuffer` (csr is M x K)."""
    assert csr.K == dense.rows
    out = _DenseBuffer(ctx, csr.M, dense.cols)
    prog["spmm"](
        queue,
        (csr.M, dense.cols),
        None,
        np.int32(csr.M),
        np.int32(dense.cols),
        csr.row_ptr,
        csr.col_idx,
        csr.vals,
        dense.buf,
        out.buf,
    )
    return out


def _transpose(prog, queue, ctx, dense):
    out = _DenseBuffer(ctx, dense.cols, dense.rows)
    prog["transpose"](
        queue, (dense.rows, dense.cols), None,
        np.int32(dense.rows), np.int32(dense.cols), dense.buf, out.buf,
    )
    return out


def _scale(prog, queue, dense, f):
    n = dense.rows * dense.cols
    prog["scale"](queue, (n,), None, np.int32(n), dense.buf, np.float32(f))


def _add(prog, queue, a, b):
    n = a.rows * a.cols
    prog["add"](queue, (n,), None, np.int32(n), a.buf, b.buf)


def _row_argmax(prog, queue, ctx, dense):
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=dense.rows * 4 or 4)
    prog["row_argmax"](
        queue, (dense.rows,), None,
        np.int32(dense.rows), np.int32(dense.cols), dense.buf, out,
    )
    host = np.empty(dense.rows, dtype=np.int32)
    cl.enqueue_copy(queue, host, out)
    return host


def _undirected_product(prog, queue, ctx, RAt, RA, RBt, RB, X):
    """Y = RA^T X RB then X = RA Y RB^T, returning the new X buffer."""
    # Y = RAt @ X @ RB  (RAt: mA x nA, RB stored as RBt: mB x nB).
    P = _spmm(prog, queue, ctx, RAt, X)          # mA x nB
    Pt = _transpose(prog, queue, ctx, P)         # nB x mA
    Qt = _spmm(prog, queue, ctx, RBt, Pt)        # mB x mA  (= Y^T)
    Y = _transpose(prog, queue, ctx, Qt)         # mA x mB
    # X = RA @ Y @ RB^T
    U = _spmm(prog, queue, ctx, RA, Y)           # nA x mB
    Ut = _transpose(prog, queue, ctx, U)         # mB x nA
    Wt = _spmm(prog, queue, ctx, RB, Ut)         # nB x nA  (= X^T)
    return _transpose(prog, queue, ctx, Wt)      # nA x nB


def run(
    ga,
    gb,
    V,
    E,
    *,
    lap="auto",
    return_scores=False,
    structure=True,
    complement=True,
    noise=1e-10,
    convergence="adaptive",
    tol=1e-6,
    patience=2,
    max_iterations=None,
    normalize=True,
    match_on="vertices",
    seed=None,
):
    """Run the GASM iterations on the GPU.

    Returns the converged vertex score matrix on the host (float64), the row and
    column labels, the number of iterations performed, and ``None`` (no lazy
    device loader is used in this version; the score matrix is materialised for
    the host LAP).
    """
    if match_on == "edges":
        raise NotImplementedError(
            "match_on='edges' is only available on the CPU back-end."
        )

    ctx, queue, prog = _ensure_context()
    rng = np.random.default_rng(seed)
    nA, nB = ga.n, gb.n

    # Initialization on the host (float64 so the noise is meaningful).
    H = rng.uniform(0.0, noise, size=(nA, nB)) if noise and noise > 0 else 0.0
    Vplus = V + H
    do_iterate = structure and ga.m > 0 and gb.m > 0
    X0 = Vplus * _init_structure(ga, gb, E) if do_iterate else Vplus.copy()

    fx = (4.0 * ga.mean_degree * gb.mean_degree + 1.0) if normalize else 1.0

    if not do_iterate:
        return np.asarray(X0, dtype=np.float64), ga.nodes, gb.nodes, 1, None

    use_comp = complement and graphmod.use_complement(ga, gb)
    if ga.directed:
        if use_comp:
            _, SA, TA = ga.complement_incidence()
            _, SB, TB = gb.complement_incidence()
        else:
            SA, TA, SB, TB = ga.S, ga.T, gb.S, gb.T
        mats = {
            "SAt": _CSRBuffer(ctx, SA.T), "SA": _CSRBuffer(ctx, SA),
            "SBt": _CSRBuffer(ctx, SB.T), "SB": _CSRBuffer(ctx, SB),
            "TAt": _CSRBuffer(ctx, TA.T), "TA": _CSRBuffer(ctx, TA),
            "TBt": _CSRBuffer(ctx, TB.T), "TB": _CSRBuffer(ctx, TB),
        }
    else:
        if use_comp:
            RA, _, _ = ga.complement_incidence()
            RB, _, _ = gb.complement_incidence()
        else:
            RA, RB = ga.R, gb.R
        mats = {
            "RAt": _CSRBuffer(ctx, RA.T), "RA": _CSRBuffer(ctx, RA),
            "RBt": _CSRBuffer(ctx, RB.T), "RB": _CSRBuffer(ctx, RB),
        }

    X = _DenseBuffer(ctx, nA, nB, host=X0)

    cap = (
        max(min(graphmod.diameter(ga), graphmod.diameter(gb)), 1)
        if convergence == "diameter"
        else max(min(_effective_diameter(ga, use_comp), _effective_diameter(gb, use_comp)), 1)
    )
    if max_iterations is not None:
        cap = max(int(max_iterations), 1)

    prev_argmax = None
    stable = 0
    k = 1
    while k < cap:
        # Convergence test (adaptive: argmax stability on device).
        if convergence != "diameter" and k >= 2:
            am = _row_argmax(prog, queue, ctx, X)
            if prev_argmax is not None and np.array_equal(am, prev_argmax):
                stable += 1
            else:
                stable = 0
            prev_argmax = am
            if stable >= patience:
                break
        elif convergence != "diameter":
            prev_argmax = _row_argmax(prog, queue, ctx, X)

        k += 1
        if ga.directed:
            # Y = SAt X SB + TAt X TB ; X = SA Y SB^T + TA Y TB^T
            P = _spmm(prog, queue, ctx, mats["SAt"], X)
            Pt = _transpose(prog, queue, ctx, P)
            Qt = _spmm(prog, queue, ctx, mats["SBt"], Pt)
            Ys = _transpose(prog, queue, ctx, Qt)
            P2 = _spmm(prog, queue, ctx, mats["TAt"], X)
            P2t = _transpose(prog, queue, ctx, P2)
            Q2t = _spmm(prog, queue, ctx, mats["TBt"], P2t)
            Yt = _transpose(prog, queue, ctx, Q2t)
            _add(prog, queue, Ys, Yt)  # Ys = Y
            U = _spmm(prog, queue, ctx, mats["SA"], Ys)
            Ut = _transpose(prog, queue, ctx, U)
            Wt = _spmm(prog, queue, ctx, mats["SB"], Ut)
            Xs = _transpose(prog, queue, ctx, Wt)
            U2 = _spmm(prog, queue, ctx, mats["TA"], Ys)
            U2t = _transpose(prog, queue, ctx, U2)
            W2t = _spmm(prog, queue, ctx, mats["TB"], U2t)
            Xt = _transpose(prog, queue, ctx, W2t)
            _add(prog, queue, Xs, Xt)
            X = Xs
        else:
            X = _undirected_product(
                prog, queue, ctx, mats["RAt"], mats["RA"], mats["RBt"], mats["RB"], X
            )
        if normalize:
            _scale(prog, queue, X, fx)

    queue.finish()
    Xh = X.to_host(queue).astype(np.float64)

    # Restore isolated vertices on the host (eq. 31).
    iso = ga.isolated[:, None] | gb.isolated[None, :]
    if iso.any():
        Xh[iso] = V[iso] / (fx ** (k - 1))

    return Xh, ga.nodes, gb.nodes, k, None
