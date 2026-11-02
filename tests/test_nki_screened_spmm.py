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

        # The kernel runs in fp32 (Trainium has no fp64 datapath; dispatch
        # downcasts and casts the result back), so fp64 finite differences at
        # eps=1e-6 would measure fp32 roundoff, not the gradient. The op is
        # linear in each input, so a large eps has no truncation error.
        assert torch.autograd.gradcheck(func, (A, B), eps=1e-2, atol=1e-2, rtol=1e-2)


def _decaying_case(n, N, kept, dtype, seed=0):
    """Sorted, decaying Schwarz bounds with |A_ij| <= Q_i Q_j (block-screened integrals)."""
    g = torch.Generator().manual_seed(seed)
    Q = torch.exp(-8.0 * torch.sort(torch.rand(n, generator=g)).values)
    A = (2 * torch.rand(n, n, generator=g) - 1) * torch.outer(Q, Q)
    B = torch.randn(n, N, generator=g)
    tau = float(torch.quantile(torch.outer(Q, Q).flatten()[:: max(1, n * n // 100000)], 1 - kept))
    return A.to(dtype), Q * Q, B.to(dtype), Q, tau


class TestNkiScreenedTileSkip:
    """Tile-skipping path (#61) against fp64 references at awkward sizes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "n,N,kept",
        [(200, 1, 0.5), (300, 77, 0.3), (1000, 600, 0.1), (1500, 128, 0.6), (2100, 513, 0.05)],
    )
    def test_matches_fp64(self, nki_backend, n, N, kept, dtype):
        from trnsparse.nki.dispatch import screened_tile_structure

        A, diag, B, Q, tau = _decaying_case(n, N, kept, dtype)
        _, cols, _, frac = screened_tile_structure(Q, tau, n)
        assert 0 < frac < 1  # some tiles skipped, some kept
        got = trnsparse.screened_spmm(A.to("neuron"), diag, B.to("neuron"), tau * tau).cpu()
        mask = torch.outer(Q, Q) > tau
        A64, B64 = A.double(), B.double()
        ref = (A64 * mask) @ B64
        tol = 2e-6 if dtype == torch.float32 else 2e-2
        assert float((got.double() - ref).abs().max() / ref.abs().max()) < tol
        # Screening error against the unscreened fp64 product stays within
        # the Schwarz bound tau * sqrt(#dropped) * ||B||_F (plus rounding).
        dropped = int((~mask).sum())
        err = float((got.double() - A64 @ B64).norm())
        bound = tau * math.sqrt(dropped) * float(B64.norm())
        assert err <= bound + tol * float(ref.norm())

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("n,N,kept", [(300, 77, 0.3), (2100, 513, 0.2)])
    def test_symmetric(self, nki_backend, n, N, kept, dtype):
        A, diag, B, Q, tau = _decaying_case(n, N, kept, torch.float32)
        A, B = ((A + A.T) / 2).to(dtype), B.to(dtype)
        got = trnsparse.screened_spmm(
            A.to("neuron"), diag, B.to("neuron"), tau * tau, symmetric=True
        ).cpu()
        mask = torch.outer(Q, Q) > tau
        ref = (A.double() * mask) @ B.double()
        tol = 2e-6 if dtype == torch.float32 else 2e-2
        assert float((got.double() - ref).abs().max() / ref.abs().max()) < tol

    def test_autograd(self, nki_backend):
        A, diag, B, Q, tau = _decaying_case(300, 64, 0.3, torch.float32)
        a = A.to("neuron").requires_grad_()
        b = B.to("neuron").requires_grad_()
        trnsparse.screened_spmm(a, diag, b, tau * tau).sum().backward()
        m = (torch.outer(Q, Q) > tau).float()
        g = torch.ones(300, 64)
        torch.testing.assert_close(a.grad.cpu(), (g @ B.T) * m, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(b.grad.cpu(), (A * m).T @ g, atol=1e-4, rtol=1e-4)

    def test_everything_screened(self, nki_backend):
        A, diag, B, Q, tau = _decaying_case(256, 64, 0.5, torch.float32)
        got = trnsparse.screened_spmm(A.to("neuron"), diag, B.to("neuron"), 1e6).cpu()
        assert torch.count_nonzero(got) == 0
