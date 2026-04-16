"""Tests for block-sparse attention example and pattern helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import trnsparse

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

import block_sparse_attention as demo  # noqa: E402


class TestMaskShapes:
    """Pattern helper sanity checks — block count and coverage."""

    def test_local_window_mask_shape(self):
        seq_len, block_size, window = 512, 128, 2
        mask = demo._local_window_mask(seq_len, block_size, window)
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        # Each block-row should have at most 2*window+1 nonzero blocks.
        n_blocks = seq_len // block_size
        bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=block_size)
        for i in range(n_blocks):
            row_count = (bsr.block_row_ptrs[i + 1] - bsr.block_row_ptrs[i]).item()
            assert row_count <= 2 * window + 1

    def test_local_window_mask_symmetric(self):
        mask = demo._local_window_mask(256, 128, 1)
        assert (mask == mask.T).all(), "local window mask should be symmetric"

    def test_dilated_mask_shape(self):
        seq_len, block_size, stride = 512, 128, 2
        mask = demo._dilated_mask(seq_len, block_size, stride)
        assert mask.shape == (seq_len, seq_len)
        n_blocks = seq_len // block_size
        # For stride=2 and n_blocks=4: blocks 0,2 from row 0; blocks 1,3 from row 1; etc.
        bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=block_size)
        expected_per_row = (n_blocks + stride - 1) // stride
        for i in range(n_blocks):
            row_count = (bsr.block_row_ptrs[i + 1] - bsr.block_row_ptrs[i]).item()
            assert row_count == expected_per_row

    def test_global_token_mask_global_rows_full(self):
        seq_len, block_size, n_global = 384, 128, 1
        mask = demo._global_token_mask(seq_len, block_size, window=1, n_global=n_global)
        # First n_global*block_size rows must be fully attended.
        assert mask[: n_global * block_size, :].all()
        # First n_global*block_size columns must be fully attended by all rows.
        assert mask[:, : n_global * block_size].all()


class TestAttentionParity:
    """bsr_spmm attention matches dense reference."""

    def _run_parity(self, seq_len, head_dim, block_size, mask):
        torch.manual_seed(0)
        Q = torch.randn(seq_len, head_dim)
        K = torch.randn(seq_len, head_dim)
        V = torch.randn(seq_len, head_dim)

        mask_bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=block_size)
        out_bsr = demo.block_sparse_attention(Q, K, V, mask_bsr)
        out_dense = demo._dense_reference(Q, K, V, mask)

        torch.testing.assert_close(out_bsr, out_dense, atol=1e-4, rtol=1e-4)
        assert out_bsr.shape == (seq_len, head_dim)

    def test_local_window_parity(self):
        seq_len, head_dim, block_size = 256, 32, 128
        mask = demo._local_window_mask(seq_len, block_size, window=1)
        self._run_parity(seq_len, head_dim, block_size, mask)

    def test_dilated_parity(self):
        seq_len, head_dim, block_size = 256, 32, 128
        mask = demo._dilated_mask(seq_len, block_size, stride=2)
        self._run_parity(seq_len, head_dim, block_size, mask)

    def test_global_token_parity(self):
        seq_len, head_dim, block_size = 384, 32, 128
        mask = demo._global_token_mask(seq_len, block_size, window=1, n_global=1)
        self._run_parity(seq_len, head_dim, block_size, mask)

    def test_full_attention_is_dense(self):
        """A fully-nonzero BSR mask should match plain dense attention."""
        seq_len, head_dim, block_size = 256, 32, 128
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        self._run_parity(seq_len, head_dim, block_size, mask)
