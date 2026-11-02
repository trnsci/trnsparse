"""CPU tests for the fused kernel-matrix product K(X, Y) @ V (#93).

The PyTorch path runs the same augmented-GEMM algorithm as the NKI kernel;
the reference is dense float64 `torch.cdist` -> kernel -> matmul. The NKI
kernel itself is checked under `nki.simulate` and the MLIR verifier in
tests/test_nki_kernel_matmul.py, and on hardware in the neuron-marked tests.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import trnsparse
from trnsparse import KERNELS, KernelOperator, kernel_matmul, kernel_matmul_reference
from trnsparse.nki import kernels as K


def _rel(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-300))


@pytest.fixture(autouse=True)
def _pytorch_backend():
    old = trnsparse.get_backend()
    trnsparse.set_backend("pytorch")
    yield
    trnsparse.set_backend(old)


# Matern-1/2 has a cusp at r = 0, so the fp32 expansion's error at
# near-coincident points enters as sqrt(u |x|^2) / l (see kernel_matmul's
# docstring); the smooth kernels see it at first order only.
_TOL = {"rbf": 2e-5, "matern12": 3e-3, "matern32": 2e-5, "matern52": 2e-5}


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "n,m,d,R",
    [
        (1, 1, 3, 1),
        (127, 129, 3, 7),
        (129, 127, 1, 64),
        (1000, 4097, 3, 1),
        (300, 200, 16, 513),
        (64, 90, 126, 2),
    ],
)
def test_cross_matches_fp64_reference(kernel, n, m, d, R):
    g = torch.Generator().manual_seed(n * 7 + m)
    X = torch.rand(n, d, generator=g)
    Y = torch.rand(m, d, generator=g)
    V = torch.randn(m, R, generator=g)
    ell = 0.5 * d**0.5
    out = kernel_matmul(X, Y, V, kernel, lengthscale=ell, outputscale=1.7)
    ref = kernel_matmul_reference(X, Y, V, kernel, lengthscale=ell, outputscale=1.7)
    assert out.shape == (n, R) and out.dtype == torch.float32
    assert _rel(out, ref) < _TOL[kernel]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("n", [1, 127, 129, 1000])
def test_square_operator_with_noise(kernel, n):
    g = torch.Generator().manual_seed(n)
    X = torch.rand(n, 3, generator=g)
    V = torch.randn(n, 5, generator=g)
    out = kernel_matmul(X, None, V, kernel, 0.3, 1.3, noise=1e-2)
    ref = kernel_matmul_reference(X, None, V, kernel, 0.3, 1.3, noise=1e-2)
    assert _rel(out, ref) < _TOL[kernel]


def test_vector_rhs_and_ard_lengthscale():
    g = torch.Generator().manual_seed(1)
    X = torch.rand(200, 3, generator=g)
    v = torch.randn(200, generator=g)
    ls = torch.tensor([0.2, 0.5, 1.0])
    out = kernel_matmul(X, None, v, "rbf", ls, 2.0, noise=1e-3)
    ref = kernel_matmul_reference(X, None, v, "rbf", ls, 2.0, noise=1e-3)
    assert out.shape == (200,)
    assert _rel(out, ref) < 2e-5
    # ARD with equal entries is the scalar lengthscale.
    same = kernel_matmul(X, None, v, "rbf", torch.full((3,), 0.4), 2.0)
    assert _rel(same, kernel_matmul_reference(X, None, v, "rbf", 0.4, 2.0)) < 2e-5


def test_reference_matches_dense_formula():
    g = torch.Generator().manual_seed(2)
    X = torch.rand(50, 2, generator=g, dtype=torch.float64)
    Y = torch.rand(40, 2, generator=g, dtype=torch.float64)
    V = torch.randn(40, 3, generator=g, dtype=torch.float64)
    r = torch.cdist(X, Y) / 0.7
    s3, s5 = 3**0.5 * r, 5**0.5 * r
    dense = {
        "rbf": torch.exp(-0.5 * r**2),
        "matern12": torch.exp(-r),
        "matern32": (1 + s3) * torch.exp(-s3),
        "matern52": (1 + s5 + s5**2 / 3) * torch.exp(-s5),
    }
    for kernel, Kd in dense.items():
        ref = kernel_matmul_reference(X, Y, V, kernel, 0.7, 1.5)
        torch.testing.assert_close(ref, 1.5 * Kd @ V, rtol=1e-12, atol=1e-12)


def test_float64_auto_stays_float64():
    trnsparse.set_backend("auto")
    g = torch.Generator().manual_seed(3)
    X = torch.rand(300, 3, generator=g, dtype=torch.float64)
    V = torch.randn(300, 2, generator=g, dtype=torch.float64)
    out = kernel_matmul(X, None, V, "rbf", 0.3, 1.0, noise=1e-4)
    assert out.dtype == torch.float64
    ref = kernel_matmul_reference(X, None, V, "rbf", 0.3, 1.0, noise=1e-4)
    assert _rel(out, ref) < 1e-12


def test_operator_matches_function_and_cross():
    g = torch.Generator().manual_seed(4)
    X = torch.rand(500, 3, generator=g) + 10.0  # far from the origin: centring matters
    V = torch.randn(500, 4, generator=g)
    op = KernelOperator(X, "matern52", 0.25, 1.1, noise=1e-3)
    ref = kernel_matmul_reference(X, None, V, "matern52", 0.25, 1.1, noise=1e-3)
    assert _rel(op(V), ref) < 2e-5
    Xs = torch.rand(77, 3, generator=g) + 10.0
    ref_c = kernel_matmul_reference(Xs, X, V, "matern52", 0.25, 1.1)
    assert _rel(op.cross(Xs, V), ref_c) < 2e-5
    rows = op.rows([0, 5, 499])
    dense = kernel_matmul_reference(
        X, None, torch.eye(500, dtype=torch.float64), "matern52", 0.25, 1.1
    )
    torch.testing.assert_close(rows, dense[[0, 5, 499]], rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(op.diag(), torch.full((500,), 1.1, dtype=torch.float64))


def test_validation():
    X = torch.rand(10, 3)
    with pytest.raises(ValueError):
        kernel_matmul(X, None, torch.rand(9, 1))
    with pytest.raises(ValueError):
        kernel_matmul(X, X, torch.rand(10, 1), noise=1.0)
    with pytest.raises(ValueError):
        kernel_matmul(X, None, torch.rand(10, 1), kernel="cauchy")
    with pytest.raises(ValueError):
        kernel_matmul(X, None, torch.rand(10, 1), lengthscale=0.0)
    with pytest.raises(NotImplementedError):
        kernel_matmul(X, None, torch.rand(10, 1), tile_mask=object())


def test_nki_backend_without_device_falls_back(monkeypatch):
    """Explicit nki with no device: the launch fails and the torch path answers (not REQUIRE_NKI)."""
    from trnsparse.nki import dispatch

    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)

    def boom(*a, **k):
        raise RuntimeError("no device")

    monkeypatch.setattr(dispatch, "nki_kernel_matmul", boom)
    X = torch.rand(40, 3)
    V = torch.rand(40, 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = kernel_matmul(X, None, V, "rbf", 0.5)
    assert any("NKI path failed" in str(x.message) for x in w)
    assert _rel(out, kernel_matmul_reference(X, None, V, "rbf", 0.5)) < 2e-5


def test_float64_policy_on_nki(monkeypatch):
    from trnsparse.nki import dispatch

    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    seen = {}

    def fake(x, y, v, params, kind, group):
        seen["dtype"] = x.dtype
        return torch.zeros(x.shape[0], v.shape[1])

    monkeypatch.setattr(dispatch, "nki_kernel_matmul", fake)
    X = torch.rand(20, 3, dtype=torch.float64)
    V = torch.rand(20, 1, dtype=torch.float64)
    monkeypatch.setattr(dispatch, "get_backend", lambda: "nki")
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
    with pytest.warns(UserWarning, match="float32"):
        out = kernel_matmul(X, None, V)
    assert seen["dtype"] == torch.float32 and out.dtype == torch.float64
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    with pytest.raises(TypeError, match="REQUIRE_NKI"):
        kernel_matmul(X, None, V)
    monkeypatch.setattr(dispatch, "get_backend", lambda: "auto")
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
    seen.clear()
    kernel_matmul(X, None, V)  # auto: float64 stays on torch
    assert not seen


def test_params_are_tensor_arguments(monkeypatch):
    """Lengthscale, outputscale and noise reach the kernel as data, not as static arguments."""
    from trnsparse.nki import dispatch

    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    calls = []
    monkeypatch.setattr(
        dispatch,
        "nki_kernel_matmul",
        lambda x, y, v, p, kind, group: (
            calls.append((p.clone(), kind, group)) or torch.zeros(x.shape[0], v.shape[1])
        ),
    )
    X = torch.rand(30, 3)
    V = torch.rand(30, 1)
    kernel_matmul(X, None, V, "rbf", 0.5, 2.0, noise=0.1)
    kernel_matmul(X, None, V, "rbf", 0.25, 3.0, noise=0.2)
    (p1, k1, g1), (p2, k2, g2) = calls
    assert (k1, g1) == (k2, g2)  # same static arguments: one compile
    assert p1.shape == (128, 4)
    torch.testing.assert_close(
        p1[0], torch.tensor([-0.5 / 0.25, torch.log(torch.tensor(2.0)), 0.1, 0.0])
    )
    assert not torch.equal(p1, p2)


# ── Work split across physical cores (pure Python, shared with the kernel) ──


@pytest.mark.parametrize("n", [1, 2, 127, 128, 129, 511, 512, 513, 1000, 1536, 4097, 100_003])
@pytest.mark.parametrize("R", [1, 64, 512, 513, 1500])
@pytest.mark.parametrize("n_prog", [1, 2])
def test_kmv_split_covers_rows_with_equal_trip_counts(n, R, n_prog):
    NI, groups = K.kmv_plan(n, R, n_prog)
    assert NI in (128, 256, 512) and NI % 128 == 0
    assert [c for g in groups for c in g] == [(r0, min(512, R - r0)) for r0 in range(0, R, 512)]
    owner = [None] * n
    trips = set()
    for pid in range(n_prog):
        lo, hi, t0, tn = K.kmv_split(n, NI, n_prog, pid)
        trips.add(hi - lo)
        assert 0 <= tn <= NI
        assert (tn == 0) == (n % (NI * n_prog) == 0)
        rows = [r for b in range(lo, hi) for r in range(b * NI, (b + 1) * NI)] + list(
            range(t0, t0 + tn)
        )
        for r in rows:
            assert owner[r] in (None, pid) or r >= n - 128  # only the last chunk may be recomputed
            owner[r] = pid if owner[r] is None else owner[r]
    assert len(trips) == 1  # same rolled-loop trip count on every core
    assert all(o is not None for o in owner)
