"""BSR format + CPU-fallback bsr_spmm tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import trnsparse


def _block_random(m_blocks, n_blocks, block_size, block_density, seed=0):
    """Build a dense (m_blocks*b, n_blocks*b) tensor with only a fraction
    of its 128×128 blocks nonzero.
    """
    torch.manual_seed(seed)
    b = block_size
    M, N = m_blocks * b, n_blocks * b
    A = torch.zeros(M, N)
    mask = torch.rand(m_blocks, n_blocks) < block_density
    for i in range(m_blocks):
        for j in range(n_blocks):
            if mask[i, j]:
                A[i * b:(i + 1) * b, j * b:(j + 1) * b] = torch.randn(b, b)
    return A


class TestBSRConstruction:

    def test_from_dense_roundtrip_aligned(self):
        A = _block_random(2, 3, block_size=8, block_density=0.5, seed=1)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=8)
        assert bsr.shape == A.shape
        assert bsr.block_size == 8
        torch.testing.assert_close(bsr.to_dense(), A)

    def test_from_dense_pads_unaligned(self):
        torch.manual_seed(2)
        A = torch.randn(20, 14)  # not a multiple of 8
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=8)
        assert bsr.shape == (20, 14)
        torch.testing.assert_close(bsr.to_dense(), A)

    def test_density_counts_blocks_not_elements(self):
        A = _block_random(4, 4, block_size=8, block_density=0.25, seed=3)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=8)
        # 16 possible blocks, density ~0.25 → ~4 blocks stored
        # Exact count depends on RNG, but should be well under 16
        assert 1 <= bsr.n_blocks <= 16
        assert 0.0 < bsr.density <= 1.0

    def test_threshold_drops_empty_blocks(self):
        A = torch.zeros(16, 16)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=8)
        assert bsr.n_blocks == 0
        assert bsr.density == 0.0

    def test_from_csr_via_dense(self):
        torch.manual_seed(4)
        A = _block_random(2, 2, block_size=8, block_density=0.5, seed=4)
        csr = trnsparse.from_dense(A)
        bsr = trnsparse.BSRMatrix.from_csr(csr, block_size=8)
        torch.testing.assert_close(bsr.to_dense(), A)


class TestBSRSpMM:
    """PyTorch-fallback path — shared spec for the NKI kernel to match."""

    def test_vs_dense(self):
        A = _block_random(3, 3, block_size=8, block_density=0.5, seed=5)
        B = torch.randn(24, 16)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=8)
        got = trnsparse.bsr_spmm(bsr, B)
        expected = A @ B
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_unaligned_shapes(self):
        torch.manual_seed(6)
        A = torch.randn(20, 14)
        B = torch.randn(14, 10)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=8)
        got = trnsparse.bsr_spmm(bsr, B)
        expected = A @ B
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_sparse_block_pattern(self):
        """Only 10% of blocks nonzero — BSR should skip the zero blocks
        but produce the same result as full dense matmul."""
        A = _block_random(5, 5, block_size=8, block_density=0.1, seed=7)
        B = torch.randn(40, 12)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=8)
        got = trnsparse.bsr_spmm(bsr, B)
        expected = A @ B
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_empty_pattern(self):
        """Zero matrix: n_blocks == 0 means output is zeros."""
        A = torch.zeros(16, 16)
        B = torch.randn(16, 8)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=8)
        got = trnsparse.bsr_spmm(bsr, B)
        torch.testing.assert_close(got, torch.zeros(16, 8), atol=1e-6, rtol=0)
