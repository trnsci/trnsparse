"""Fused SBUF-resident CG kernel on hardware (#22).

Runs `cg_bsr_fused` on a `neuron` device (native TorchNeuron path) and
checks it against the host path of the same algorithm and an fp64 direct
solve. Skipped off-hardware; the CPU behaviour is covered by
tests/test_iterative.py and the kernel's MLIR by tests/test_nki_compile.py.

    NKI_ENABLE_TRACE_CACHE=0 pytest tests/test_nki_cg_fused.py -v -m neuron
"""

from __future__ import annotations

import pytest
import torch

import trnsparse
from trnsparse.nki import HAS_NKI
from trnsparse.nki.dispatch import _get_dispatch_path, _use_simulator


def _native() -> bool:
    try:
        return HAS_NKI and not _use_simulator() and _get_dispatch_path() == "native"
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


def _spd(T: int, band: int = 1, n: int | None = None, seed: int = 0):
    """Block-banded SPD (fp64), condition number in the hundreds."""
    g = torch.Generator().manual_seed(seed)
    N = T * 128
    M = torch.randn(N, N, generator=g, dtype=torch.float64) / N**0.5
    mask = torch.zeros(T, T, dtype=torch.float64)
    for i in range(T):
        mask[i, max(0, i - band) : i + band + 1] = 1.0
    M = M * torch.kron(mask, torch.ones(128, 128, dtype=torch.float64))
    A = (M + M.T) / 2
    ev = torch.linalg.eigvalsh(A)
    A = A + (ev[-1] - 300 * ev[0]) / 299 * torch.eye(N, dtype=torch.float64)  # kappa = 300
    if n is not None:
        A = A[:n, :n].contiguous()
    return A


def _rhs(n, R=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    shape = (n,) if R is None else (n, R)
    return torch.randn(*shape, dtype=torch.float64, generator=g)


def _true_res(A, b, x):
    return float(torch.linalg.norm(b - A @ x.double()) / torch.linalg.norm(b))


@pytest.mark.parametrize("pc", [None, "block_jacobi"])
def test_matches_host_path(pc):
    A = _spd(4)
    b = _rhs(A.shape[0])
    bsr = trnsparse.BSRMatrix.from_dense(A.float(), block_size=128)
    op_dev = trnsparse.prepare_cg_bsr(bsr, preconditioner=pc, device=DEV)
    x, k, rel, info = trnsparse.cg_bsr_fused(op_dev, b, tol=1e-5, return_info=True)
    assert info["path"] == "nki"
    assert x.device.type == "cpu" and x.dtype == torch.float64
    xh, kh, _ = trnsparse.cg_bsr_fused(bsr, b.float(), tol=1e-5, preconditioner=pc)
    assert abs(k - kh) <= 2
    assert rel <= 1e-5
    assert _true_res(A, b, x) < 3e-5
    ref = torch.linalg.solve(A, b)
    assert float(torch.linalg.norm(x - ref) / torch.linalg.norm(ref)) < 1e-4


def test_check_interval_does_not_change_result():
    A = _spd(3, seed=2)
    b = _rhs(A.shape[0], seed=3)
    op = trnsparse.prepare_cg_bsr(
        trnsparse.BSRMatrix.from_dense(A.float(), block_size=128), device=DEV
    )
    runs = [trnsparse.cg_bsr_fused(op, b, tol=1e-5, check_every=ce) for ce in (1, 3, 8)]
    assert len({k for _, k, _ in runs}) == 1
    for x, _, _ in runs[1:]:
        torch.testing.assert_close(x, runs[0][0], atol=1e-6, rtol=1e-5)


def test_iteration_cap_is_exact():
    A = _spd(3, seed=4)
    b = _rhs(A.shape[0], seed=5)
    op = trnsparse.prepare_cg_bsr(
        trnsparse.BSRMatrix.from_dense(A.float(), block_size=128), device=DEV
    )
    _, k, rel, info = trnsparse.cg_bsr_fused(
        op, b, tol=0.0, max_iter=13, check_every=4, return_info=True
    )
    assert k == 13 and info["trips"] == 4 and rel > 0


def test_padded_order_and_initial_guess():
    A = _spd(3, n=300, seed=6)
    b = _rhs(300, seed=7)
    bsr = trnsparse.BSRMatrix.from_dense(A.float(), block_size=128)
    op = trnsparse.prepare_cg_bsr(bsr, preconditioner="block_jacobi", device=DEV)
    x, k, rel = trnsparse.cg_bsr_fused(op, b, tol=1e-5)
    assert x.shape == (300,) and _true_res(A, b, x) < 3e-5
    _, k2, _ = trnsparse.cg_bsr_fused(op, b, x0=x, tol=1e-4)
    assert k2 < k


def test_multiple_rhs():
    A = _spd(4, seed=8)
    B = _rhs(A.shape[0], R=6, seed=9)
    B[:, 2] = 0.0
    B[:, 4] *= 1e4
    op = trnsparse.prepare_cg_bsr(
        trnsparse.BSRMatrix.from_dense(A.float(), block_size=128),
        preconditioner="block_jacobi",
        device=DEV,
    )
    X, iters, rel = trnsparse.cg_bsr_fused(op, B, tol=1e-5)
    assert X.shape == B.shape and int(iters[2]) == 0
    assert torch.count_nonzero(X[:, 2]) == 0
    for c in (0, 1, 3, 4, 5):
        assert _true_res(A, B[:, c], X[:, c]) < 3e-5
        _, k, _ = trnsparse.cg_bsr_fused(op, B[:, c], tol=1e-5)
        assert abs(int(iters[c]) - k) <= 2


def test_streamed_blocks(monkeypatch):
    # A 2 KiB budget keeps 4 blocks per core resident; the rest stream from HBM
    # inside the launch. A distinct structure (band 2) avoids a cached NEFF
    # traced with the default budget.
    monkeypatch.setenv("TRNSPARSE_CG_SBUF_KIB", "2")
    A = _spd(5, band=2, seed=10)
    b = _rhs(A.shape[0], seed=11)
    bsr = trnsparse.BSRMatrix.from_dense(A.float(), block_size=128)
    op = trnsparse.prepare_cg_bsr(bsr, device=DEV)
    assert op.resident_blocks < op.n_blocks
    x, _, rel = trnsparse.cg_bsr_fused(op, b, tol=1e-5)
    assert rel <= 1e-5 and _true_res(A, b, x) < 3e-5
