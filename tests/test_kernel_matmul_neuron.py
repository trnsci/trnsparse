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
