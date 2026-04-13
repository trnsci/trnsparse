"""On-hardware BSR SpMM parity + gradcheck.

Requires Neuron hardware. Run via:

    AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn1
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


def _block_random(m_blocks, n_blocks, block_size, block_density, seed):
    torch.manual_seed(seed)
    b = block_size
    M, N = m_blocks * b, n_blocks * b
    A = torch.zeros(M, N)
    mask = torch.rand(m_blocks, n_blocks) < block_density
    for i in range(m_blocks):
        for j in range(n_blocks):
            if mask[i, j]:
                A[i * b : (i + 1) * b, j * b : (j + 1) * b] = torch.randn(b, b)
    return A, mask.sum().item()


class TestNkiBSRSpmm:
    @pytest.mark.parametrize(
        "m_blocks,n_blocks,N,block_density",
        [
            (2, 2, 128, 0.5),  # tile-aligned, dense block pattern
            (4, 4, 128, 0.25),  # quarter of blocks nonzero
            (3, 3, 256, 0.5),  # larger RHS
            (4, 2, 128, 0.5),  # rectangular
        ],
    )
    def test_parity(self, nki_backend, m_blocks, n_blocks, N, block_density):
        A, _ = _block_random(
            m_blocks, n_blocks, block_size=128, block_density=block_density, seed=42
        )
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        B = torch.randn(n_blocks * 128, N)

        got = trnsparse.bsr_spmm(bsr, B)
        expected = A @ B

        torch.testing.assert_close(got, expected, atol=ATOL, rtol=RTOL)

    def test_dispatch_routes_to_nki(self, nki_backend):
        assert _use_nki()


class TestNkiBSRDifferentiability:
    """Satisfies the trnsci/trnsci#3 autograd requirement for BSR."""

    def test_backward_finite(self, nki_backend):
        """loss.backward() end-to-end produces finite gradients for both
        the stored BSR blocks and the dense RHS.
        """
        A, _ = _block_random(2, 2, block_size=128, block_density=0.5, seed=1)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        bsr.blocks = bsr.blocks.detach().requires_grad_(True)
        B = torch.randn(2 * 128, 64, requires_grad=True)

        C = trnsparse.bsr_spmm(bsr, B)
        loss = C.pow(2).sum()
        loss.backward()

        assert bsr.blocks.grad is not None
        assert torch.isfinite(bsr.blocks.grad).all()
        assert B.grad is not None
        assert torch.isfinite(B.grad).all()

    def test_gradcheck_small(self, nki_backend):
        """gradcheck on tiny inputs: forward NKI path, backward PyTorch
        path as documented in _BSRSpMMFunction.
        """
        torch.manual_seed(0)
        # Small shape for double-precision finite differences
        A, _ = _block_random(1, 1, block_size=128, block_density=1.0, seed=2)
        bsr = trnsparse.BSRMatrix.from_dense(A.double(), block_size=128)
        bsr.blocks = bsr.blocks.detach().double().requires_grad_(True)
        B = torch.randn(128, 16, dtype=torch.float64, requires_grad=True)

        from trnsparse.nki.dispatch import _BSRSpMMFunction

        def func(blocks, B):
            return _BSRSpMMFunction.apply(
                blocks,
                bsr.block_col_indices,
                bsr.block_row_ptrs,
                bsr.shape,
                bsr.block_size,
                B,
            )

        assert torch.autograd.gradcheck(func, (bsr.blocks, B), eps=1e-6, atol=1e-4)
