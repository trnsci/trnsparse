"""NKI dispatch for sparse operations.

v0.2.0 — `nki_spmm` is the first NKI-dispatched op. Forward path:
materialize the CSR into a padded dense view, run the NKI GEMM kernel
(`kernels._spmm_dense_kernel`) on the XLA device, slice back to the
original shape. Backward runs at the PyTorch level — see
`_SpMMFunction.backward` for the analytic adjoints.

Autograd wrapping is a hard requirement across the trnsci suite (see
trnsci/trnsci#3). This module's `_SpMMFunction` is the first concrete
example in the suite; sibling projects copying this pattern is expected.
"""

from __future__ import annotations

import os

import torch

from .kernels import _TILE_K, _TILE_M, _TILE_N, HAS_NKI

if HAS_NKI:
    from .kernels import _bsr_spmm_kernel, _spmm_dense_kernel  # noqa: F401 — NKI-only

# When set, kernel-path failures re-raise instead of falling back to
# PyTorch. Used by the hardware validation suite.
_REQUIRE_NKI = os.environ.get("TRNSPARSE_REQUIRE_NKI", "").lower() in (
    "1",
    "true",
    "yes",
)

_backend = "auto"


def set_backend(backend: str) -> None:
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires neuronxcc")
    _backend = backend


def get_backend() -> str:
    return _backend


def _use_nki() -> bool:
    if _backend == "nki":
        return True
    if _backend == "pytorch":
        return False
    return HAS_NKI


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _to_xla(*tensors):
    """Move tensors to the XLA device for NKI kernel dispatch."""
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    orig = tensors[0].device
    return [t.to(device) for t in tensors], orig


def _csr_to_dense_padded(A) -> torch.Tensor:
    """Materialize a CSRMatrix into a dense (M, K) tensor.

    v0.2.0 uses the dense view directly as the A operand of the GEMM
    kernel. Row-bucketing (Phase 3, #15) will replace this with a
    per-bucket dense tile keyed on nnz quantile, which is where the
    actual sparse speedup comes from.
    """
    t = torch.sparse_csr_tensor(A.row_ptrs, A.col_indices, A.values, size=A.shape)
    return t.to_dense()


def _nki_spmm_impl(A_dense: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Dispatch a dense (M, K) @ (K, N) matmul through the NKI kernel.

    Pads M/K to TILE_M/TILE_K multiples and N up to TILE_N when N > TILE_N.
    Slices the result back to the caller's (M, N).

    Falls back to `torch.matmul` on kernel errors unless
    `TRNSPARSE_REQUIRE_NKI=1`.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    M, K = A_dense.shape
    _, N = B.shape
    M_pad = _round_up(M, _TILE_M)
    K_pad = _round_up(K, _TILE_K)
    N_pad = N if N <= _TILE_N else _round_up(N, _TILE_N)
    needs_pad = (M_pad != M) or (K_pad != K) or (N_pad != N)

    try:
        if needs_pad:
            A_p = torch.zeros(M_pad, K_pad, dtype=A_dense.dtype, device=A_dense.device)
            A_p[:M, :K] = A_dense
            B_p = torch.zeros(K_pad, N_pad, dtype=B.dtype, device=B.device)
            B_p[:K, :N] = B
            (a, b), orig_device = _to_xla(A_p.contiguous(), B_p.contiguous())
        else:
            (a, b), orig_device = _to_xla(A_dense.contiguous(), B.contiguous())
        c = _spmm_dense_kernel(a, b)
        result = c.to(orig_device)
        return result[:M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        return torch.matmul(A_dense, B)


class _SpMMFunction(torch.autograd.Function):
    """Autograd wrapper for the NKI SpMM forward.

    Forward is NKI-dispatched; backward runs at the PyTorch level.
    Analytic adjoints for C = A @ B with A sparse:

        dL/dA_values = (dL/dC @ Bᵀ)[rows, cols]   (projected onto A's pattern)
        dL/dB        = Aᵀ @ dL/dC

    For v0.2.0 the forward pass materializes A as dense, so the A gradient
    flows back through the materialized view naturally — torch handles the
    projection via the sparse_csr_tensor.to_dense() graph.
    """

    @staticmethod
    def forward(ctx, A_dense: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(A_dense, B)
        return _nki_spmm_impl(A_dense, B)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        A_dense, B = ctx.saved_tensors
        grad_A = grad_out @ B.T if ctx.needs_input_grad[0] else None
        grad_B = A_dense.T @ grad_out if ctx.needs_input_grad[1] else None
        return grad_A, grad_B


def nki_spmm(A, B: torch.Tensor) -> torch.Tensor:
    """SpMM entry point: C = A @ B for a `CSRMatrix` A and dense B.

    Routes through `_SpMMFunction.apply` so `loss.backward()` works.
    On NKI backend, forward runs the kernel; on PyTorch, torch.matmul
    on the densified A is the fallback — the autograd wrapper is kept
    on both paths for API uniformity.
    """
    A_dense = _csr_to_dense_padded(A)
    return _SpMMFunction.apply(A_dense, B)


def _nki_bsr_spmm_impl(
    blocks_pad: torch.Tensor, b_gathered: torch.Tensor, out_rows: int, out_cols: int
) -> torch.Tensor:
    """Dispatch `_bsr_spmm_kernel` on the XLA device.

    `blocks_pad` is `(M_tiles, K_max, 128, 128)` and `b_gathered` is
    `(M_tiles, K_max, 128, N_pad)`. Returns `(M_tiles * 128, N_pad)`
    sliced down to `(out_rows, out_cols)`.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    try:
        (bp, bg), orig_device = _to_xla(blocks_pad.contiguous(), b_gathered.contiguous())
        c = _bsr_spmm_kernel(bp, bg)
        result = c.to(orig_device)
        return result[:out_rows, :out_cols].contiguous()
    except Exception:
        if _REQUIRE_NKI:
            raise
        # Fallback: straight PyTorch reference using the gathered slices.
        M_tiles, K_max, B, _ = blocks_pad.shape
        _, _, _, N_pad = b_gathered.shape
        out = torch.zeros(M_tiles * B, N_pad, dtype=blocks_pad.dtype, device=blocks_pad.device)
        for m in range(M_tiles):
            for k in range(K_max):
                out[m * B : (m + 1) * B] += blocks_pad[m, k] @ b_gathered[m, k]
        return out[:out_rows, :out_cols].contiguous()


def _bsr_pad_and_gather(A, B: torch.Tensor):
    """Host-side prep for the BSR kernel.

    Returns:
        blocks_pad: (M_tiles, K_max, block_size, block_size) zero-padded
        b_gathered: (M_tiles, K_max, block_size, N_pad) B-slices indexed
                     by each block's column index (zero when no block)
        out_rows, out_cols: the caller's original output shape
        K_max: for NEFF cache key visibility
    """
    m, n = A.shape
    _, out_cols = B.shape
    b = A.block_size
    m_pad = ((m + b - 1) // b) * b
    n_pad = ((n + b - 1) // b) * b

    # Pad B rows up to block alignment and columns up to TILE_N alignment
    N_pad = out_cols if out_cols <= _TILE_N else ((out_cols + _TILE_N - 1) // _TILE_N) * _TILE_N
    if N_pad != out_cols or n_pad != n:
        B_p = torch.zeros(n_pad, N_pad, dtype=B.dtype, device=B.device)
        B_p[:n, :out_cols] = B
        B = B_p

    M_tiles = m_pad // b
    nb_cols = n_pad // b

    # Count blocks per row and pad up to K_max
    block_row_ptrs = A.block_row_ptrs
    blocks_per_row = (block_row_ptrs[1:] - block_row_ptrs[:-1]).tolist()
    K_max = max(blocks_per_row) if blocks_per_row else 1

    # Construct a 2D grid indexed by (block_row, k_slot) -> block_column
    # or -1 if no block. Then gather B and blocks via this grid.
    col_grid = torch.full((M_tiles, K_max), -1, dtype=torch.long)
    block_idx_grid = torch.full((M_tiles, K_max), -1, dtype=torch.long)
    for i in range(M_tiles):
        start = block_row_ptrs[i].item()
        end = block_row_ptrs[i + 1].item()
        row_len = end - start
        col_grid[i, :row_len] = A.block_col_indices[start:end]
        block_idx_grid[i, :row_len] = torch.arange(start, end)

    # Pad A with a single zero block at index `n_blocks` for masked slots.
    zero_block = torch.zeros(1, b, b, dtype=A.dtype, device=A.blocks.device)
    blocks_with_zero = torch.cat([A.blocks, zero_block], dim=0)
    masked_idx = torch.where(
        block_idx_grid >= 0, block_idx_grid, torch.tensor(A.n_blocks, dtype=torch.long)
    )
    blocks_pad = blocks_with_zero[masked_idx]  # (M_tiles, K_max, b, b)

    # Gather B slices per slot. For masked slots, a zero slice (columns 0..b)
    # times a zero block contributes zero anyway — we use column 0 as sentinel.
    safe_col = torch.where(col_grid >= 0, col_grid, torch.tensor(0, dtype=torch.long))
    # Build (M_tiles, K_max, b, N_pad) via fancy indexing into B reshaped (nb_cols, b, N_pad).
    B_by_block = B.view(nb_cols, b, N_pad)
    b_gathered = B_by_block[safe_col]  # (M_tiles, K_max, b, N_pad)

    return blocks_pad, b_gathered, m, out_cols, K_max


class _BSRSpMMFunction(torch.autograd.Function):
    """Autograd wrapper for BSR SpMM.

    Forward: dispatched to NKI kernel (uniform-K-padded per block row).
    Backward (PyTorch-level):
        dA_blocks[k] = dC[row_block_of_k] @ B[col_block_of_k].T
                         (pattern stays fixed — no gradient to block indices)
        dB            = A.T @ dC  (via PyTorch dense path for simplicity)

    The backward pass reuses the PyTorch fallback rather than calling the
    kernel in reverse — correct, differentiable, and v0.3.0-scope.
    """

    @staticmethod
    def forward(ctx, A_blocks, A_block_col_indices, A_block_row_ptrs, A_shape, A_block_size, B):
        # Reconstruct a lightweight BSR-like handle for the host-side prep
        class _BSRHandle:
            shape = A_shape
            block_size = A_block_size
            blocks = A_blocks
            block_col_indices = A_block_col_indices
            block_row_ptrs = A_block_row_ptrs

            @property
            def n_blocks(self):
                return self.blocks.shape[0]

            @property
            def dtype(self):
                return self.blocks.dtype

        handle = _BSRHandle()
        blocks_pad, b_gathered, out_rows, out_cols, _ = _bsr_pad_and_gather(handle, B)
        C = _nki_bsr_spmm_impl(blocks_pad, b_gathered, out_rows, out_cols)

        ctx.save_for_backward(A_blocks, A_block_col_indices, A_block_row_ptrs, B)
        ctx.A_shape = A_shape
        ctx.A_block_size = A_block_size
        return C

    @staticmethod
    def backward(ctx, grad_out):
        A_blocks, col_idx, row_ptrs, B = ctx.saved_tensors
        b = ctx.A_block_size

        grad_blocks = None
        grad_B = None

        if ctx.needs_input_grad[0]:
            # dA_blocks[k] = grad_out[rowblock*b : (rowblock+1)*b] @ B[col*b:(col+1)*b].T
            # rowblock is inferred from row_ptrs.
            n_blocks = A_blocks.shape[0]
            grad_blocks = torch.zeros_like(A_blocks)
            M_tiles = row_ptrs.shape[0] - 1
            for i in range(M_tiles):
                start = row_ptrs[i].item()
                end = row_ptrs[i + 1].item()
                for k in range(start, end):
                    j = col_idx[k].item()
                    grad_blocks[k] = grad_out[i * b : (i + 1) * b] @ B[j * b : (j + 1) * b].T

        if ctx.needs_input_grad[5]:
            # dB = A.T @ grad_out — reconstruct A dense and multiply.
            m, n = ctx.A_shape
            m_pad = ((m + b - 1) // b) * b
            n_pad = ((n + b - 1) // b) * b
            A_dense = torch.zeros(m_pad, n_pad, dtype=A_blocks.dtype, device=A_blocks.device)
            M_tiles = row_ptrs.shape[0] - 1
            for i in range(M_tiles):
                start = row_ptrs[i].item()
                end = row_ptrs[i + 1].item()
                for k in range(start, end):
                    j = col_idx[k].item()
                    A_dense[i * b : (i + 1) * b, j * b : (j + 1) * b] = A_blocks[k]
            A_dense = A_dense[:m, :n]
            grad_B = A_dense.T @ grad_out

        # Five non-tensor inputs in forward args: indices/ptrs/shape/block_size/B
        # Only A_blocks (arg 0) and B (arg 5) have grads.
        return grad_blocks, None, None, None, None, grad_B


def nki_bsr_spmm(A, B: torch.Tensor) -> torch.Tensor:
    """BSR SpMM entry point — wraps `_BSRSpMMFunction.apply` for autograd."""
    return _BSRSpMMFunction.apply(
        A.blocks,
        A.block_col_indices,
        A.block_row_ptrs,
        A.shape,
        A.block_size,
        B,
    )
