"""Fused kernel-matrix product (#93) on hardware (TorchNeuron, Trn2).

    TRNSPARSE_REQUIRE_NKI=1 pytest tests/test_kernel_matmul_neuron.py -v -m neuron

Accuracy against the float64 reference at ragged sizes, the cached-handle
and custom-op launch paths agreeing bit for bit, and run-to-run bitwise
determinism. With TRNSPARSE_KMV_DUMP=<dir> the determinism case also saves
its output, so runs at NEURON_LOGICAL_NC_CONFIG=1 and =2 can be compared
bitwise (see `benchmarks/bench_kernel_matmul_trn2.py --compare`).
"""

from __future__ import annotations

import os

import pytest
import torch

import trnsparse
from trnsparse.nki import HAS_NKI
from trnsparse.nki import dispatch as _d


def _native() -> bool:
    try:
        return HAS_NKI and not _d._use_simulator() and _d._get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

DEV = "neuron"


@pytest.fixture(autouse=True)
def nki_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


def _rel(a, b):
    return float((a.cpu().double() - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern32", "matern52"])
@pytest.mark.parametrize(
    "n,m,d,R",
    [
        (1, 1, 3, 1),
        (127, 129, 3, 7),
        (129, 127, 1, 64),
        (1000, 4097, 3, 1),
        (4097, 1000, 16, 513),
        (700, 513, 126, 2),
    ],
)
def test_cross_on_device(kernel, n, m, d, R):
    g = torch.Generator().manual_seed(n + m)
    X, Y, V = (
        torch.rand(n, d, generator=g),
        torch.rand(m, d, generator=g),
        torch.randn(m, R, generator=g),
    )
    ell = 0.5 * d**0.5
    ref = trnsparse.kernel_matmul_reference(X, Y, V, kernel, ell, 1.5)
    out = trnsparse.kernel_matmul(X.to(DEV), Y.to(DEV), V.to(DEV), kernel, ell, 1.5)
    assert out.device.type == DEV
    assert _rel(out, ref) < 2e-5


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
@pytest.mark.parametrize("n", [1, 129, 1000, 4097, 20000])
def test_square_operator_on_device(kernel, n):
    g = torch.Generator().manual_seed(n)
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, kernel, 0.1, 1.0, noise=1e-4, device=DEV)
    assert op.route == "nki"
    ref = trnsparse.kernel_matmul_reference(X, None, V, kernel, 0.1, 1.0, noise=1e-4)
    assert _rel(op(V.to(DEV)), ref) < 2e-5


def test_handle_and_custom_op_agree(monkeypatch):
    g = torch.Generator().manual_seed(3)
    X, V = torch.rand(3000, 3, generator=g), torch.randn(3000, 17, generator=g)
    op = trnsparse.KernelOperator(X, "rbf", 0.2, 1.0, noise=1e-3, device=DEV)
    Vd = V.to(DEV)
    a = op(Vd).cpu()
    monkeypatch.setattr(_d, "_EAGER_HANDLE", False)
    b = op(Vd).cpu()
    assert torch.equal(a, b)


def test_bitwise_deterministic():
    g = torch.Generator().manual_seed(4)
    n = 50_000
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, "rbf", 0.05, 1.0, noise=1e-4, device=DEV)
    Vd = V.to(DEV)
    a, b = op(Vd).cpu(), op(Vd).cpu()
    assert torch.equal(a, b)
    out = os.environ.get("TRNSPARSE_KMV_DUMP")
    if out:
        lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "default")
        os.makedirs(out, exist_ok=True)
        torch.save(a, os.path.join(out, f"kmv_n{n}_lnc{lnc}.pt"))


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
def test_row_blocked_is_bitwise_one_launch(monkeypatch, kernel, self_op):
    """Forced small row blocks (#100) == one launch, bit for bit, on the device.

    n = 5000 in 1024-row blocks: four full blocks and a padded ragged one.
    """
    g = torch.Generator().manual_seed(100)
    n, R = 5000, 33
    m = n if self_op else 3000
    X = torch.rand(n, 3, generator=g)
    Y = None if self_op else torch.rand(m, 3, generator=g)
    V = torch.randn(m, R, generator=g)
    op = trnsparse.KernelOperator(
        X if self_op else Y, kernel, 0.1, 1.0, noise=1e-4 if self_op else 0.0, device=DEV
    )
    Vd = V.to(DEV)

    def apply():
        return (op(Vd) if self_op else op.cross(X, Vd)).cpu()

    monkeypatch.delenv("TRNSPARSE_KMV_BLOCK_WORK", raising=False)
    one = apply()
    monkeypatch.setenv("TRNSPARSE_KMV_BLOCK_WORK", str(1024 * m))
    assert _d._kernels.kmv_block_rows(n, m, _d._kmv_block_work()) == 1024
    n_handles = len(_d._eager_handles)
    blocked = apply()
    assert torch.equal(one, blocked)
    assert len(_d._eager_handles) - n_handles <= 1  # one handle for every block shape
    ref = trnsparse.kernel_matmul_reference(
        X, None if self_op else Y, V, kernel, 0.1, 1.0, noise=1e-4 if self_op else 0.0
    )
    assert _rel(blocked, ref) < 2e-5


@pytest.mark.skipif(
    os.environ.get("TRNSPARSE_KMV_BIG") != "1", reason="set TRNSPARSE_KMV_BIG=1 (minutes)"
)
def test_one_million_rows_under_the_runtime_timeout():
    """n = 2^20, R = 33: was one 40-80 s launch (NRT status 5 at 30 s); now 19 launches."""
    import time

    n = 1 << 20
    g = torch.Generator().manual_seed(7)
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, "rbf", 0.05, 1.0, noise=1e-4, device=DEV)
    Vd = V.to(DEV)
    out = op(Vd)  # compile + first run
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    out = op(Vd)
    torch.neuron.synchronize()
    dt = time.perf_counter() - t0
    print(f"\nn={n} R=33 rbf: {dt:.2f} s, {n * n / dt:.3e} entries/s")
    idx = torch.linspace(0, n - 1, 64).long()
    ref = trnsparse.kernel_matmul_reference(X[idx], X, V, "rbf", 0.05, 1.0)
    ref += 1e-4 * V[idx].double()
    assert _rel(out.cpu()[idx], ref) < 2e-5
