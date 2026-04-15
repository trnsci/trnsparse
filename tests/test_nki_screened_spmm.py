"""On-hardware fused screened SpMM tests (#19).

Requires Neuron hardware. Run via:

    AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn1
"""

from __future__ import annotations

import math

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


class TestNkiScreenedSpmmParity:
    @pytest.mark.parametrize(
        "n,N,threshold",
        [
            (128, 128, 0.0),
            (128, 128, 0.5),
            (256, 128, 0.5),
            (256, 256, 0.1),
            (200, 64, 0.3),
        ],
    )
    def test_parity(self, nki_backend, n, N, threshold):
        torch.manual_seed(42)
        A = torch.randn(n, n)
        diag = torch.abs(torch.randn(n)) * 4.0 + 0.01
        B = torch.randn(n, N)

        got = trnsparse.screened_spmm(A, diag, B, threshold=threshold)

        Q = torch.sqrt(torch.abs(diag))
        mask = (Q.unsqueeze(-1) * Q.unsqueeze(0)) > math.sqrt(max(threshold, 0.0))
        expected = (A * mask.to(A.dtype)) @ B

        torch.testing.assert_close(got, expected, atol=ATOL, rtol=RTOL)

    def test_dispatch_routes_to_nki(self, nki_backend):
        assert _use_nki()


class TestNkiScreenedSpmmDifferentiability:
    """Satisfies the trnsci/trnsci#3 autograd requirement for screened SpMM.

    Mask is non-differentiable (discrete gate); gradients flow to A and B.
    """

    def test_backward_finite(self, nki_backend):
        torch.manual_seed(1)
        n = 128
        A = torch.randn(n, n, requires_grad=True)
        diag = torch.abs(torch.randn(n)) * 4.0 + 0.01
        B = torch.randn(n, 64, requires_grad=True)

        C = trnsparse.screened_spmm(A, diag, B, threshold=0.5)
        loss = C.pow(2).sum()
        loss.backward()

        assert A.grad is not None and torch.isfinite(A.grad).all()
        assert B.grad is not None and torch.isfinite(B.grad).all()

    def test_gradcheck_small(self, nki_backend):
        torch.manual_seed(2)
        n = 128
        A = torch.randn(n, n, dtype=torch.float64, requires_grad=True)
        diag = torch.abs(torch.randn(n, dtype=torch.float64)) * 4.0 + 0.01
        B = torch.randn(n, 8, dtype=torch.float64, requires_grad=True)

        from trnsparse.nki.dispatch import _ScreenedSpMMFunction

        def func(a, b):
            return _ScreenedSpMMFunction.apply(a, diag, 0.5, b)

        assert torch.autograd.gradcheck(func, (A, B), eps=1e-6, atol=1e-4)
