"""On-hardware NKI SpMM parity + differentiability tests.

Requires Neuron hardware. Run via:

    AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn1

The kernel materializes A as dense then runs NKI GEMM (v0.2.0 Phase 1
path). Row-bucketing for real sparse speedup lands in v0.3.0 (#15).
"""

from __future__ import annotations

import pytest
import torch

import trnsparse
from trnsparse.nki.dispatch import _use_nki

pytestmark = pytest.mark.neuron


ATOL, RTOL = 1e-3, 1e-4


@pytest.fixture
def nki_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


def _random_sparse(M, K, density, seed=0):
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(M, K, generator=g) * (
        torch.rand(M, K, generator=g) < density
    )
    return dense


class TestNkiSpmmParity:
    """PyTorch-vs-NKI agreement across uniform and skewed nnz distributions."""

    @pytest.mark.parametrize(
        "M,K,N,density",
        [
            (128, 128, 128, 0.1),     # tile-aligned, moderate density
            (256, 256, 128, 0.05),    # default case
            (256, 256, 128, 0.01),    # sparse
            (512, 256, 256, 0.05),    # non-square
            (200, 137, 64, 0.05),     # unaligned dimensions (exercise padding)
        ],
    )
    def test_parity(self, nki_backend, M, K, N, density):
        A_dense = _random_sparse(M, K, density)
        A = trnsparse.from_dense(A_dense)
        B = torch.randn(K, N)

        got = trnsparse.spmm(A, B)
        expected = A_dense @ B

        torch.testing.assert_close(got, expected, atol=ATOL, rtol=RTOL)

    def test_dispatch_routes_to_nki(self, nki_backend):
        assert _use_nki(), "set_backend('nki') should enable the NKI path"


class TestNkiSpmmDifferentiability:
    """Satisfies the suite-wide autograd requirement (trnsci/trnsci#3).

    Uses small inputs so torch.autograd.gradcheck's fp64 finite differences
    run fast and produce a clean signal.
    """

    def test_gradcheck(self, nki_backend):
        M, K, N = 16, 16, 8
        density = 0.2
        A_dense = _random_sparse(M, K, density).double().requires_grad_(True)
        B = torch.randn(K, N, dtype=torch.float64, requires_grad=True)

        from trnsparse.nki.dispatch import _SpMMFunction

        def func(a, b):
            return _SpMMFunction.apply(a, b)

        assert torch.autograd.gradcheck(func, (A_dense, B), eps=1e-6, atol=1e-4)

    def test_backward_finite(self, nki_backend):
        """Smoke test: backward produces finite gradients end-to-end."""
        M, K, N = 64, 64, 32
        A_dense = _random_sparse(M, K, 0.1).requires_grad_(True)
        B = torch.randn(K, N, requires_grad=True)

        from trnsparse.nki.dispatch import _SpMMFunction
        C = _SpMMFunction.apply(A_dense, B)
        loss = C.pow(2).sum()
        loss.backward()

        assert A_dense.grad is not None and torch.isfinite(A_dense.grad).all()
        assert B.grad is not None and torch.isfinite(B.grad).all()
