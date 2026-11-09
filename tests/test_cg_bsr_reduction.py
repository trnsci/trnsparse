"""cg_bsr reductions over N that do not depend on the LNC split (trnsci/trnsolver#128).

The CPU stand-in follows trnsolver's tests/test_cg_reduction.py: ``_Dev``
marks "device" vectors whose full reductions (``dot``, ``vdot``, a 1-D
``linalg.norm``) add in an order set by a layout. Layout 1 reduces all N
entries at once; layout 2 reduces the two halves separately and adds them,
as two cores would. Elementwise ops and the BSR matvec are unchanged (the
matvec stands in for an LNC-invariant kernel), and a copy to the host
returns a plain tensor. Results that are bitwise equal across the layouts
do not depend on the device's reduction order.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

import trnsparse
from trnsparse import _reduce


class _Dev(torch.Tensor):
    layout = 1

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        with torch._C.DisableTorchFunctionSubclass():
            out = _split(name, args, kwargs) if cls.layout == 2 else None
            if out is None:
                out = func(*args, **kwargs)
        if not isinstance(out, torch.Tensor):
            return out
        host = _to_host(name, args, kwargs)
        return out if host else out.as_subclass(_Dev)


def _to_host(name, args, kwargs):
    if name == "cpu":
        return True
    if name != "to":
        return False
    dev = kwargs.get("device")
    for a in args[1:]:
        if isinstance(a, (torch.device, str)):
            dev = a
    return dev is not None and torch.device(dev).type == "cpu"


def _split(name, args, kwargs):
    if name in ("dot", "vdot") and len(args) == 2 and args[0].dim() == 1 and args[0].shape[0] > 1:
        u, v = args
        h = u.shape[0] // 2
        f = torch.vdot if name == "vdot" else torch.dot
        return f(u[:h], v[:h]) + f(u[h:], v[h:])
    if name == "linalg_norm" and len(args) == 1 and not kwargs and args[0].dim() == 1:
        t = args[0]
        if t.shape[0] > 1:
            h = t.shape[0] // 2
            return torch.sqrt(t[:h].square().sum() + t[h:].square().sum())
    return None


@contextlib.contextmanager
def _layout(k):
    old, _Dev.layout = _Dev.layout, k
    try:
        yield
    finally:
        _Dev.layout = old


def _plain(t):
    return t.as_subclass(torch.Tensor)


def _banded_spd_bsr(n, bs=32, seed=0):
    """A banded SPD kernel matrix (compact-support stand-in for a GP) in BSR."""
    g = torch.Generator().manual_seed(seed)
    x = torch.sort(torch.rand(n, generator=g)).values * (n / 50)
    d = (x[:, None] - x[None, :]).abs()
    K = torch.clamp(1 - d / 3, min=0) ** 2  # Wendland-like, compact support
    A = K + 0.1 * torch.eye(n)
    return A, trnsparse.BSRMatrix.from_dense(A.float(), block_size=bs)


def _solve(red, n=2048, M=None):
    A, A_bsr = _banded_spd_bsr(n)
    b = torch.randn(n, generator=torch.Generator().manual_seed(1)).as_subclass(_Dev)
    x, it, rel = trnsparse.cg_bsr(A_bsr, b, tol=1e-6, max_iter=500, M=M, reduction=red)
    return _plain(x), it, rel, A


def _both(fn):
    with _layout(1):
        a = fn()
    with _layout(2):
        b = fn()
    return a, b


class TestLayoutInvariance:
    @pytest.mark.parametrize("red", ["pairwise", "host64"])
    def test_bitwise(self, red):
        (xa, ia, ra, A), (xb, ib, rb, _) = _both(lambda: _solve(red))
        assert torch.equal(xa, xb) and ia == ib and ra == rb
        b = torch.randn(xa.shape[0], generator=torch.Generator().manual_seed(1))
        x64 = torch.linalg.solve(A.double(), b.double())
        assert float((xa.double() - x64).norm() / x64.norm()) < 1e-3
        assert ra < 1e-6

    def test_device_mode_differs(self):
        (xa, ia, ra, _), (xb, ib, rb, _) = _both(lambda: _solve("device"))
        assert not (torch.equal(xa, xb) and ra == rb)

    def test_jacobi_preconditioned_bitwise(self):
        _, A_bsr = _banded_spd_bsr(2048)
        dinv = (1.0 / trnsparse.bsr_diagonal(A_bsr)).as_subclass(_Dev)
        (xa, ia, *_), (xb, ib, *_) = _both(lambda: _solve("pairwise", M=lambda r: dinv * r))
        assert torch.equal(xa, xb) and ia == ib


class TestModes:
    @pytest.mark.parametrize("red", _reduce.REDUCTIONS)
    def test_modes_agree(self, red):
        A, A_bsr = _banded_spd_bsr(1000, seed=2)
        b = torch.randn(1000, generator=torch.Generator().manual_seed(2))
        x, it, rel = trnsparse.cg_bsr(A_bsr, b, tol=1e-6, reduction=red)
        x0, it0, _ = trnsparse.cg_bsr(A_bsr, b, tol=1e-6, reduction="device")
        # On host tensors the folded modes form float64 products while
        # "device" is an fp32 torch.dot, so the modes do different
        # arithmetic here. For this system (cond ~1e3) fp32 CG converges
        # late by an amount that moves with the rounding of the dots and of
        # the fp32 matvec: pairwise / device take 145 / 153 iterations on
        # Linux (MKL) and 151 / 152 on macOS (Accelerate); fp64 CG takes
        # ~108. The counts agree only to a few percent, so the solutions
        # are checked against float64 instead.
        assert rel < 1e-6 and abs(it - it0) <= max(2, it0 // 10), (it, it0)
        x64 = torch.linalg.solve(A.double(), b.double())
        for xs in (x, x0):
            assert float((xs.double() - x64).norm() / x64.norm()) < 1e-4
        assert torch.allclose(x, x0, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("n", [1, 7, 64, 65, 1000, 4097, 30011])
    def test_fold_matches_numpy_tree(self, n):
        g = torch.Generator().manual_seed(n)
        p = torch.randn(n, generator=g)
        lev, m = _reduce.fold_plan(n)
        t = np.zeros(m << lev, dtype=np.float32)
        t[:n] = p.numpy()
        for _ in range(lev):
            h = t.shape[0] // 2
            t = t[:h] + t[h:]
        assert float(_reduce.fold_host64(p)) == float(t.astype(np.float64).sum())

    def test_helpers_accuracy(self):
        g = torch.Generator().manual_seed(3)
        u, v = torch.randn(100003, generator=g), torch.randn(100003, generator=g)
        ref = float((u.double() * v.double()).sum())
        scale = float((u.double() * v.double()).abs().sum())
        nref = float(u.double().norm())
        for red in ("pairwise", "host64"):
            assert abs(_reduce.dot(u, v, red) - ref) <= 1e-12 * scale
            assert abs(_reduce.norm(u, red) - nref) <= 1e-12 * nref
        assert abs(_reduce.dot(u, v, "device") - ref) <= 1e-5 * scale

    def test_grad_path_takes_reduction(self):
        A, A_bsr = _banded_spd_bsr(512, seed=4)
        b = torch.randn(512, generator=torch.Generator().manual_seed(4), requires_grad=True)
        x, _, _ = trnsparse.cg_bsr(A_bsr, b, tol=1e-10, max_iter=500, reduction="pairwise")
        x.sum().backward()
        ref = torch.linalg.solve(A.T.double(), torch.ones(512, dtype=torch.float64))
        assert torch.allclose(b.grad.double(), ref, rtol=1e-3, atol=1e-4)

    def test_env_and_validation(self, monkeypatch):
        monkeypatch.delenv("TRNSPARSE_CG_REDUCTION", raising=False)
        assert _reduce.reduction_mode(None, "TRNSPARSE_CG_REDUCTION", "device") == "device"
        monkeypatch.setenv("TRNSPARSE_CG_REDUCTION", "Pairwise")
        assert _reduce.reduction_mode(None, "TRNSPARSE_CG_REDUCTION", "device") == "pairwise"
        monkeypatch.setenv("TRNSPARSE_CG_REDUCTION", "fast")
        A_bsr = trnsparse.BSRMatrix.from_dense(torch.eye(32) * 2, block_size=32)
        with pytest.raises(ValueError, match="reduction"):
            trnsparse.cg_bsr(A_bsr, torch.ones(32))
        monkeypatch.delenv("TRNSPARSE_CG_REDUCTION")
        with pytest.raises(ValueError, match="reduction"):
            trnsparse.cg_bsr(A_bsr, torch.ones(32), reduction="exact")
