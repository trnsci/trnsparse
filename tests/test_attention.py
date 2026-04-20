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


class TestTiledAttentionParity:
    """block_sparse_attention_tiled matches block_sparse_attention (naive).

    The tiled two-pass implementation must produce numerically identical
    output to the naive O(seq_len²) reference, within floating-point noise.
    """

    def _run_parity(self, seq_len, head_dim, block_size, mask):
        torch.manual_seed(10)
        Q = torch.randn(seq_len, head_dim)
        K = torch.randn(seq_len, head_dim)
        V = torch.randn(seq_len, head_dim)

        mask_bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=block_size)
        out_naive = demo.block_sparse_attention(Q, K, V, mask_bsr)
        out_tiled = trnsparse.block_sparse_attention_tiled(Q, K, V, mask_bsr)

        torch.testing.assert_close(out_naive, out_tiled, atol=1e-4, rtol=1e-4)
        assert out_tiled.shape == (seq_len, head_dim)

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

    def test_full_mask_parity(self):
        """Fully dense mask: tiled must match naive."""
        seq_len, head_dim, block_size = 256, 32, 128
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        self._run_parity(seq_len, head_dim, block_size, mask)

    def test_head_dim_256_parity(self):
        """head_dim=256 (K-tiling path) matches naive reference."""
        seq_len, head_dim, block_size = 256, 256, 128
        mask = demo._local_window_mask(seq_len, block_size, window=1)
        self._run_parity(seq_len, head_dim, block_size, mask)

    def test_head_dim_256_dilated_parity(self):
        """head_dim=256 dilated pattern matches naive reference."""
        seq_len, head_dim, block_size = 256, 256, 128
        mask = demo._dilated_mask(seq_len, block_size, stride=2)
        self._run_parity(seq_len, head_dim, block_size, mask)


class TestAttnTiledGrad:
    """Backward-pass tests for block_sparse_attention_tiled (v0.5.0).

    Uses float64 for gradcheck precision. pytorch backend is set explicitly
    (NKI kernels don't support float64 on hardware).
    """

    def _local_mask(self, seq_len: int, block_size: int, window: int) -> torch.Tensor:
        n_blocks = seq_len // block_size
        bm = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
        for i in range(n_blocks):
            lo = max(0, i - window)
            hi = min(n_blocks, i + window + 1)
            bm[i, lo:hi] = True
        return bm.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    def _dilated_mask(self, seq_len: int, block_size: int, stride: int) -> torch.Tensor:
        n_blocks = seq_len // block_size
        bm = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
        for i in range(n_blocks):
            for j in range(n_blocks):
                if (i - j) % stride == 0:
                    bm[i, j] = True
        return bm.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    def _run_gradcheck(
        self, mask: torch.Tensor, seq_len: int, head_dim: int, block_size: int, seed: int
    ):
        prev = trnsparse.get_backend()
        trnsparse.set_backend("pytorch")
        try:
            torch.manual_seed(seed)
            mask_bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=block_size)

            # float64 for gradcheck precision
            Q = torch.randn(seq_len, head_dim, dtype=torch.float64, requires_grad=True)
            K = torch.randn(seq_len, head_dim, dtype=torch.float64, requires_grad=True)
            V = torch.randn(seq_len, head_dim, dtype=torch.float64, requires_grad=True)

            def fn(q, k, v):
                return trnsparse.block_sparse_attention_tiled(q, k, v, mask_bsr)

            torch.autograd.gradcheck(fn, (Q, K, V), eps=1e-4, atol=1e-3, rtol=1e-3)
        finally:
            trnsparse.set_backend(prev)

    def test_gradcheck_local(self):
        """gradcheck passes for local-window mask (w=1)."""
        seq_len, head_dim, block_size = 256, 32, 128
        mask = self._local_mask(seq_len, block_size, window=1)
        self._run_gradcheck(mask, seq_len, head_dim, block_size, seed=40)

    def test_gradcheck_dilated(self):
        """gradcheck passes for dilated mask (stride=2)."""
        seq_len, head_dim, block_size = 256, 32, 128
        mask = self._dilated_mask(seq_len, block_size, stride=2)
        self._run_gradcheck(mask, seq_len, head_dim, block_size, seed=41)

    def test_backward_shapes(self):
        """dQ, dK, dV have the same shape as Q, K, V."""
        prev = trnsparse.get_backend()
        trnsparse.set_backend("pytorch")
        try:
            torch.manual_seed(42)
            seq_len, head_dim, block_size = 256, 32, 128
            mask = self._local_mask(seq_len, block_size, window=1)
            mask_bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=block_size)

            Q = torch.randn(seq_len, head_dim, requires_grad=True)
            K = torch.randn(seq_len, head_dim, requires_grad=True)
            V = torch.randn(seq_len, head_dim, requires_grad=True)

            out = trnsparse.block_sparse_attention_tiled(Q, K, V, mask_bsr)
            out.sum().backward()

            assert Q.grad is not None and Q.grad.shape == Q.shape
            assert K.grad is not None and K.grad.shape == K.shape
            assert V.grad is not None and V.grad.shape == V.shape
        finally:
            trnsparse.set_backend(prev)

    def test_backward_parity_finite_diff(self):
        """Analytical backward matches finite-difference at atol=1e-3."""
        prev = trnsparse.get_backend()
        trnsparse.set_backend("pytorch")
        try:
            torch.manual_seed(43)
            seq_len, head_dim, block_size = 256, 32, 128
            mask = self._local_mask(seq_len, block_size, window=1)
            mask_bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=block_size)

            Q = torch.randn(seq_len, head_dim, dtype=torch.float64)
            K = torch.randn(seq_len, head_dim, dtype=torch.float64)
            V = torch.randn(seq_len, head_dim, dtype=torch.float64)
            dO = torch.randn(seq_len, head_dim, dtype=torch.float64)

            # Analytical backward
            Q_a = Q.clone().requires_grad_(True)
            K_a = K.clone().requires_grad_(True)
            V_a = V.clone().requires_grad_(True)
            out = trnsparse.block_sparse_attention_tiled(Q_a, K_a, V_a, mask_bsr)
            out.backward(dO)

            # Finite-difference reference for dV (cheapest to check)
            eps = 1e-4
            dV_fd = torch.zeros_like(V)
            for i in range(min(4, seq_len)):  # spot-check 4 rows
                for j in range(head_dim):
                    Vp = V.clone()
                    Vp[i, j] += eps
                    Vm = V.clone()
                    Vm[i, j] -= eps
                    fp = trnsparse.block_sparse_attention_tiled(Q, K, Vp, mask_bsr)
                    fm = trnsparse.block_sparse_attention_tiled(Q, K, Vm, mask_bsr)
                    dV_fd[i, j] = ((fp - fm) * dO).sum() / (2 * eps)

            torch.testing.assert_close(V_a.grad[:4], dV_fd[:4], atol=1e-3, rtol=1e-3)
        finally:
            trnsparse.set_backend(prev)

    def test_backward_parity_head_dim_256(self):
        """Analytical backward matches finite-diff spot-check at head_dim=256."""
        prev = trnsparse.get_backend()
        trnsparse.set_backend("pytorch")
        try:
            torch.manual_seed(50)
            seq_len, head_dim, block_size = 256, 256, 128
            mask = self._local_mask(seq_len, block_size, window=1)
            mask_bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=block_size)

            Q = torch.randn(seq_len, head_dim, dtype=torch.float64)
            K = torch.randn(seq_len, head_dim, dtype=torch.float64)
            V = torch.randn(seq_len, head_dim, dtype=torch.float64)
            dO = torch.randn(seq_len, head_dim, dtype=torch.float64)

            Q_a = Q.clone().requires_grad_(True)
            K_a = K.clone().requires_grad_(True)
            V_a = V.clone().requires_grad_(True)
            out = trnsparse.block_sparse_attention_tiled(Q_a, K_a, V_a, mask_bsr)
            out.backward(dO)

            eps = 1e-4
            dV_fd = torch.zeros_like(V)
            for i in range(min(4, seq_len)):
                for j in range(min(4, head_dim)):
                    Vp, Vm = V.clone(), V.clone()
                    Vp[i, j] += eps
                    Vm[i, j] -= eps
                    fp = trnsparse.block_sparse_attention_tiled(Q, K, Vp, mask_bsr)
                    fm = trnsparse.block_sparse_attention_tiled(Q, K, Vm, mask_bsr)
                    dV_fd[i, j] = ((fp - fm) * dO).sum() / (2 * eps)

            torch.testing.assert_close(V_a.grad[:4, :4], dV_fd[:4, :4], atol=1e-3, rtol=1e-3)
        finally:
            trnsparse.set_backend(prev)

    def test_backward_shapes_head_dim_256(self):
        """dQ, dK, dV shapes correct for head_dim=256."""
        prev = trnsparse.get_backend()
        trnsparse.set_backend("pytorch")
        try:
            torch.manual_seed(51)
            seq_len, head_dim, block_size = 256, 256, 128
            mask = self._local_mask(seq_len, block_size, window=1)
            mask_bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=block_size)

            Q = torch.randn(seq_len, head_dim, requires_grad=True)
            K = torch.randn(seq_len, head_dim, requires_grad=True)
            V = torch.randn(seq_len, head_dim, requires_grad=True)

            out = trnsparse.block_sparse_attention_tiled(Q, K, V, mask_bsr)
            out.sum().backward()

            assert Q.grad is not None and Q.grad.shape == Q.shape
            assert K.grad is not None and K.grad.shape == K.shape
            assert V.grad is not None and V.grad.shape == V.shape
        finally:
            trnsparse.set_backend(prev)
