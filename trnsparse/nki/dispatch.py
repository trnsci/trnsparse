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

import numpy as np
import torch

from .kernels import _TILE_K, _TILE_M, _TILE_N, HAS_NKI

if HAS_NKI:
    import nki

    from .kernels import (  # noqa: F401 — NKI-only
        _attn_bwd_dkdv_kernel,
        _attn_bwd_dq_kernel,
        _attn_out_kernel,
        _attn_stats_kernel,
        _bsr_spmm_kernel,
        _screened_spmm_kernel,
        _spmm_dense_kernel,
    )

# When set, kernel-path failures re-raise instead of falling back to
# PyTorch. Used by the hardware validation suite.
_REQUIRE_NKI = os.environ.get("TRNSPARSE_REQUIRE_NKI", "").lower() in (
    "1",
    "true",
    "yes",
)

# When set, dispatch bypasses torch_xla and runs kernels through
# `nki.simulate(kernel)(np_args)` on CPU. Lets us iterate kernels on any
# x86_64 Linux box without paying the NEFF compile + hardware dispatch
# cost. Semantics follow NKI 0.3.0's simulator: no NEFF compile, no
# SBUF/PSUM capacity checks, no latency/parallelism modelling. For
# correctness iteration only; hardware still owns perf numbers.
_USE_SIMULATOR = os.environ.get("TRNSPARSE_USE_SIMULATOR", "").lower() in (
    "1",
    "true",
    "yes",
)


def _use_simulator() -> bool:
    return _USE_SIMULATOR and HAS_NKI


_backend = "auto"


def set_backend(backend: str) -> None:
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires nki>=0.3.0 (Neuron SDK 2.29+)")
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
            A_feed, B_feed = A_p.contiguous(), B_p.contiguous()
        else:
            A_feed, B_feed = A_dense.contiguous(), B.contiguous()

        if _use_simulator():
            # CPU-side: feed NumPy arrays to nki.simulate(kernel). Bypasses
            # torch_xla entirely.
            out_np = nki.simulate(_spmm_dense_kernel)(A_feed.cpu().numpy(), B_feed.cpu().numpy())
            result = torch.from_numpy(np.asarray(out_np)).to(A_dense.device)
        else:
            (a, b), orig_device = _to_xla(A_feed, B_feed)
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
        bp_feed = blocks_pad.contiguous()
        bg_feed = b_gathered.contiguous()

        if _use_simulator():
            out_np = nki.simulate(_bsr_spmm_kernel)(bp_feed.cpu().numpy(), bg_feed.cpu().numpy())
            result = torch.from_numpy(np.asarray(out_np)).to(blocks_pad.device)
        else:
            (bp, bg), orig_device = _to_xla(bp_feed, bg_feed)
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


def _nki_screened_spmm_impl(
    A: torch.Tensor,
    Q: torch.Tensor,
    threshold_sqrt: float,
    B: torch.Tensor,
) -> torch.Tensor:
    """Dispatch `_screened_spmm_kernel` on the XLA device (or simulator).

    `A` is square (M, M); `Q` is the 1-D Schwarz-bound vector of length M.
    Pads M up to TILE_M and N up to TILE_N when `N > TILE_N`.
    Falls back to `torch.matmul` on masked A if the kernel errors and
    `TRNSPARSE_REQUIRE_NKI` is not set.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    M, K = A.shape
    _, N = B.shape
    assert M == K, f"screened_spmm currently requires square A; got {A.shape}"
    M_pad = _round_up(M, _TILE_M)
    N_pad = N if N <= _TILE_N else _round_up(N, _TILE_N)
    needs_pad = (M_pad != M) or (N_pad != N)

    threshold_sqrt_t = torch.tensor(threshold_sqrt, dtype=A.dtype)

    try:
        if needs_pad:
            A_p = torch.zeros(M_pad, M_pad, dtype=A.dtype, device=A.device)
            A_p[:M, :M] = A
            Q_p = torch.zeros(M_pad, dtype=Q.dtype, device=Q.device)
            Q_p[:M] = Q
            B_p = torch.zeros(M_pad, N_pad, dtype=B.dtype, device=B.device)
            B_p[:M, :N] = B
            A_feed, Q_feed, B_feed = A_p.contiguous(), Q_p.contiguous(), B_p.contiguous()
        else:
            A_feed, Q_feed, B_feed = A.contiguous(), Q.contiguous(), B.contiguous()

        if _use_simulator():
            out_np = nki.simulate(_screened_spmm_kernel)(
                A_feed.cpu().numpy(),
                Q_feed.cpu().numpy(),
                threshold_sqrt_t.cpu().numpy(),
                B_feed.cpu().numpy(),
            )
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            (a, q, b), orig_device = _to_xla(A_feed, Q_feed, B_feed)
            ts = threshold_sqrt_t.to(a.device)
            c = _screened_spmm_kernel(a, q, ts, b)
            result = c.to(orig_device)

        return result[:M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        # Torch fallback computes the mask + matmul directly.
        pair_bound = Q.unsqueeze(-1) * Q.unsqueeze(0)
        mask = pair_bound > threshold_sqrt
        return (A * mask.to(A.dtype)) @ B


class _ScreenedSpMMFunction(torch.autograd.Function):
    """Autograd wrapper for fused screened SpMM.

    Forward: NKI-dispatched (or PyTorch fallback). Backward: PyTorch-level,
    projecting gradients through the mask.

    The mask depends on `diag_integrals` and `threshold` but is discrete —
    no gradient flows back to them. Gradients flow to `A` (masked) and
    `B` (transposed masked A).
    """

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        diag_integrals: torch.Tensor,
        threshold: float,
        B: torch.Tensor,
    ) -> torch.Tensor:
        import math as _math

        Q = torch.sqrt(torch.abs(diag_integrals))
        threshold_sqrt = _math.sqrt(threshold)
        C = _nki_screened_spmm_impl(A, Q, threshold_sqrt, B)

        # Save the effective mask for backward.
        mask = (Q.unsqueeze(-1) * Q.unsqueeze(0)) > threshold_sqrt
        ctx.save_for_backward(A, B, mask)
        return C

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        A, B, mask = ctx.saved_tensors
        m_f = mask.to(A.dtype)
        grad_A = (grad_out @ B.T) * m_f if ctx.needs_input_grad[0] else None
        grad_B = (A * m_f).T @ grad_out if ctx.needs_input_grad[3] else None
        # No gradient to diag_integrals (arg 1) or threshold (arg 2).
        return grad_A, None, None, grad_B


def nki_screened_spmm(
    A: torch.Tensor,
    diag_integrals: torch.Tensor,
    B: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Screened SpMM entry point — wraps `_ScreenedSpMMFunction.apply` for autograd."""
    return _ScreenedSpMMFunction.apply(A, diag_integrals, threshold, B)


def _attn_gather(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask_bsr, scale: float):
    """Host-side gather for the two-pass NKI attention kernels.

    Mirrors `_bsr_pad_and_gather`: builds a (M_tiles, K_max) column grid,
    fancy-indexes K and V blocks into padded (M_tiles, K_max, b, head_dim)
    tensors, and scales Q into (M_tiles, b, head_dim).

    Returns:
        q_scaled_blocks: (M_tiles, b, head_dim)
        k_gathered_pad:  (M_tiles, K_max, b, head_dim)
        v_gathered_pad:  (M_tiles, K_max, b, head_dim)
        K_max:           int — padded number of k-slots per block-row
        M_tiles:         int
    """
    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    M_tiles = seq_len // b

    assert head_dim <= _TILE_K or head_dim % _TILE_K == 0, (
        f"head_dim {head_dim}: must be ≤ {_TILE_K} or a multiple of {_TILE_K}. "
        f"Supported: 32, 64, 128 (single tile) and 256, 512 (K-tiled)."
    )

    # NKI 0.3.0 simulator: nc_matmul requires K = TILE_K exactly. Pad head_dim to
    # TILE_K when running in the simulator with head_dim < TILE_K. The forward
    # output is sliced to [:seq_len, :head_dim] so the padding is transparent.
    head_dim_k = _TILE_K if (_use_simulator() and head_dim < _TILE_K) else head_dim

    block_row_ptrs = mask_bsr.block_row_ptrs
    blocks_per_row = (block_row_ptrs[1:] - block_row_ptrs[:-1]).tolist()
    K_max = max(blocks_per_row) if blocks_per_row else 1

    # Build (M_tiles, K_max) column grid; -1 for padding slots.
    col_grid = torch.full((M_tiles, K_max), -1, dtype=torch.long)
    for i in range(M_tiles):
        start = block_row_ptrs[i].item()
        end = block_row_ptrs[i + 1].item()
        row_len = end - start
        col_grid[i, :row_len] = mask_bsr.block_col_indices[start:end]

    # Gather K and V: reshape (seq_len, head_dim) → (seq_len//b, b, head_dim)
    # then fancy-index by col_grid (use col 0 as sentinel for padded slots).
    nb_rows = seq_len // b
    K_by_block = K.view(nb_rows, b, head_dim)  # (nb_rows, b, head_dim)
    V_by_block = V.view(nb_rows, b, head_dim)

    safe_col = torch.where(col_grid >= 0, col_grid, torch.tensor(0, dtype=torch.long))
    k_gathered_pad = K_by_block[safe_col]  # (M_tiles, K_max, b, head_dim)
    v_gathered_pad = V_by_block[safe_col]  # (M_tiles, K_max, b, head_dim)

    q_scaled_blocks = (Q * scale).view(M_tiles, b, head_dim)

    if head_dim_k != head_dim:
        extra = head_dim_k - head_dim

        def _pad_hd(t: torch.Tensor, ndim_extra: int) -> torch.Tensor:
            return torch.cat([t, t.new_zeros(*t.shape[:-1], ndim_extra)], dim=-1)

        q_scaled_blocks = _pad_hd(q_scaled_blocks, extra)
        k_gathered_pad = _pad_hd(k_gathered_pad, extra)
        v_gathered_pad = _pad_hd(v_gathered_pad, extra)

    return q_scaled_blocks, k_gathered_pad, v_gathered_pad, K_max, M_tiles


def _attn_host_reduction(tile_max: torch.Tensor, tile_sumexp: torch.Tensor) -> tuple:
    """Reduce per-block stats to per-row softmax denominators.

    Args:
        tile_max:    (M_tiles, K_max, 128) — per-block row-wise max
        tile_sumexp: (M_tiles, K_max, 128) — per-block stable exp-sum

    Returns:
        row_max:   (M_tiles, 128) — row-wise global max
        row_denom: (M_tiles, 128) — softmax denominator
    """
    row_max = tile_max.max(dim=1).values  # (M_tiles, 128)
    correction = torch.exp(tile_max - row_max.unsqueeze(1))  # (M_tiles, K_max, 128)
    row_denom = (tile_sumexp * correction).sum(dim=1)  # (M_tiles, 128)
    row_denom = row_denom.clamp(min=1e-12)
    return row_max, row_denom


def nki_bsr_attn_tiled(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask_bsr,
    return_stats: bool = False,
):
    """Two-pass block-sparse attention via NKI kernel pair.

    Orchestrates `_attn_stats_kernel` (pass 1) + host reduction +
    `_attn_out_kernel` (pass 2) for the block-sparse attention forward pass.
    No O(seq_len²) intermediate is allocated.

    head_dim must be ≤ 128 (single tile) or a multiple of 128 (K-tiling:
    256, 512). K-tiling accumulates the score Q @ K.T across TILE_K=128
    chunks, reusing PSUM across nc_matmul calls.

    Args:
        Q, K, V:      (seq_len, head_dim) float tensors.
        mask_bsr:     BSRMatrix with block_size=128 encoding the attention pattern.
        return_stats: If True, returns (out, row_max, row_denom) where row_max
                      and row_denom have shape (M_tiles, block_size). Used by
                      `_AttnTiledFunction.forward` to save stats for backward.

    Returns:
        (seq_len, head_dim) attention output, or a 3-tuple when return_stats=True.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    seq_len, head_dim = Q.shape
    scale = head_dim**-0.5

    q_scaled, k_gathered, v_gathered, K_max, M_tiles = _attn_gather(Q, K, V, mask_bsr, scale)
    b = mask_bsr.block_size

    # Contiguous inputs for kernel dispatch.
    qs = q_scaled.contiguous()
    kg = k_gathered.contiguous()
    vg = v_gathered.contiguous()

    try:
        if _use_simulator():
            tile_max_np, tile_sumexp_np = nki.simulate(_attn_stats_kernel)(
                qs.cpu().numpy(), kg.cpu().numpy()
            )
            tile_max = torch.from_numpy(np.asarray(tile_max_np)).to(Q.device)
            tile_sumexp = torch.from_numpy(np.asarray(tile_sumexp_np)).to(Q.device)
            # NKI 0.3.0 keepdims: tile_max/tile_sumexp are (M_tiles, K_max, 128, 1)
            if tile_max.dim() == 4:
                tile_max = tile_max.squeeze(-1)
                tile_sumexp = tile_sumexp.squeeze(-1)

            row_max, row_denom = _attn_host_reduction(tile_max, tile_sumexp)
            rm = row_max.contiguous()
            rd = row_denom.contiguous()

            out_np = nki.simulate(_attn_out_kernel)(
                qs.cpu().numpy(),
                kg.cpu().numpy(),
                vg.cpu().numpy(),
                rm.cpu().numpy(),
                rd.cpu().numpy(),
            )
            result = torch.from_numpy(np.asarray(out_np)).to(Q.device)
        else:
            (qs_x, kg_x, vg_x), orig_device = _to_xla(qs, kg, vg)
            tile_max_x, tile_sumexp_x = _attn_stats_kernel(qs_x, kg_x)
            tile_max = tile_max_x.to(orig_device)
            tile_sumexp = tile_sumexp_x.to(orig_device)
            if tile_max.dim() == 4:
                tile_max = tile_max.squeeze(-1)
                tile_sumexp = tile_sumexp.squeeze(-1)

            row_max, row_denom = _attn_host_reduction(tile_max, tile_sumexp)
            rm = row_max.contiguous()
            rd = row_denom.contiguous()

            (rm_x, rd_x), _ = _to_xla(rm, rd)
            result_x = _attn_out_kernel(qs_x, kg_x, vg_x, rm_x, rd_x)
            result = result_x.to(orig_device)

        out = result[:seq_len, :head_dim].contiguous()
        if return_stats:
            # row_max/row_denom are already (M_tiles, b) from _attn_host_reduction.
            return out, row_max[:M_tiles, :b].contiguous(), row_denom[:M_tiles, :b].contiguous()
        return out
    except Exception:
        if _REQUIRE_NKI:
            raise
        # PyTorch fallback — import lazily to avoid circular dependency.
        from ..ops import _block_sparse_attn_pytorch

        scale_val = head_dim**-0.5
        if return_stats:
            return _block_sparse_attn_pytorch(Q, K, V, mask_bsr, scale_val, return_stats=True)
        return _block_sparse_attn_pytorch(Q, K, V, mask_bsr, scale_val)


def _attn_bwd_gather(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    O: torch.Tensor,
    mask_bsr,
    scale: float,
    row_max: torch.Tensor,
    row_denom: torch.Tensor,
):
    """Host-side gather for the NKI backward kernels.

    Builds two sets of tensors:
    1. Row-first (for `_attn_bwd_dq_kernel`): same column grid as forward gather.
    2. Column-first (for `_attn_bwd_dkdv_kernel`): BSC transposed view — for each
       column block ki, gather all block-rows m that attend to ki.

    Args:
        Q, K, V, dO, O: (seq_len, head_dim) tensors.
        mask_bsr:        BSRMatrix encoding the attention pattern.
        scale:           QK scale (baked into q_scaled for dQ kernel).
        row_max:         (M_tiles, block_size) saved from forward.
        row_denom:       (M_tiles, block_size) saved from forward.

    Returns a dict with keys:
        row_first: dict with keys q_scaled, k_gathered, v_gathered, do_gathered, D_blocks
        col_first: dict with keys k_blocks, v_blocks, q_gathered, do_gathered, D_gathered,
                                  row_max_gathered, row_denom_gathered
    """
    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    M_tiles = seq_len // b
    nb_rows = M_tiles  # alias
    block_row_ptrs = mask_bsr.block_row_ptrs
    block_col_indices = mask_bsr.block_col_indices

    # NKI 0.3.0 simulator: pad head_dim to TILE_K so nc_matmul gets K=TILE_K.
    # Padding with zeros is correct: [dO|0]·[O|0] = dO·O for D, and the kernel
    # outputs are sliced to [:seq_len, :head_dim] in nki_bsr_attn_bwd.
    if _use_simulator() and head_dim < _TILE_K:
        extra = _TILE_K - head_dim

        def _pad_hd(t: torch.Tensor) -> torch.Tensor:
            return torch.cat([t, t.new_zeros(*t.shape[:-1], extra)], dim=-1)

        Q = _pad_hd(Q)
        K = _pad_hd(K)
        V = _pad_hd(V)
        dO = _pad_hd(dO)
        O = _pad_hd(O)
        head_dim = _TILE_K  # update local for views below

    # ── Row-first tensors (for dQ kernel) ──────────────────────────────────────
    blocks_per_row = (block_row_ptrs[1:] - block_row_ptrs[:-1]).tolist()
    K_max = max(blocks_per_row) if blocks_per_row else 1

    col_grid = torch.full((M_tiles, K_max), -1, dtype=torch.long)
    for i in range(M_tiles):
        start = block_row_ptrs[i].item()
        end = block_row_ptrs[i + 1].item()
        row_len = end - start
        col_grid[i, :row_len] = block_col_indices[start:end]

    safe_col = torch.where(col_grid >= 0, col_grid, torch.tensor(0, dtype=torch.long))

    K_by_block = K.view(nb_rows, b, head_dim)
    V_by_block = V.view(nb_rows, b, head_dim)
    dO_by_block = dO.view(nb_rows, b, head_dim)

    q_scaled = (Q * scale).view(M_tiles, b, head_dim)
    k_gathered = K_by_block[safe_col]  # (M_tiles, K_max, b, head_dim)
    v_gathered = V_by_block[safe_col]  # (M_tiles, K_max, b, head_dim)
    do_gathered = dO_by_block[safe_col]  # (M_tiles, K_max, b, head_dim)

    # D = dO · O row-wise; reshape to (M_tiles, b) for block indexing.
    D_flat = (dO * O).sum(dim=-1)  # (seq_len,)
    D_blocks = D_flat.view(M_tiles, b)  # (M_tiles, b)

    row_first = {
        "q_scaled": q_scaled,
        "k_gathered": k_gathered,
        "v_gathered": v_gathered,
        "do_gathered": do_gathered,
        "D_blocks": D_blocks,
    }

    # ── Column-first tensors (for dK/dV kernel) ────────────────────────────────
    N_col = nb_rows  # number of column blocks = number of row blocks (square)

    # Invert the BSR pattern: for each column block ki, collect all block-rows m.
    col_to_rows: list[list[int]] = [[] for _ in range(N_col)]
    for m in range(M_tiles):
        start = block_row_ptrs[m].item()
        end = block_row_ptrs[m + 1].item()
        for idx in range(start, end):
            ki = block_col_indices[idx].item()
            col_to_rows[ki].append(m)

    K_max_col = max((len(r) for r in col_to_rows), default=1)

    # row_grid_col[ki, slot] = block-row index m, or -1 for padding.
    row_grid_col = torch.full((N_col, K_max_col), -1, dtype=torch.long)
    for ki, rows in enumerate(col_to_rows):
        if rows:
            row_grid_col[ki, : len(rows)] = torch.tensor(rows, dtype=torch.long)

    safe_row = torch.where(row_grid_col >= 0, row_grid_col, torch.tensor(0, dtype=torch.long))

    # K and V in column order: (N_col, b, head_dim) — already row == col block order.
    k_blocks = K.view(N_col, b, head_dim)
    v_blocks = V.view(N_col, b, head_dim)

    # Gather Q_scaled, dO, D by row index; reuse views computed in row-first section.
    q_gathered_col = q_scaled[safe_row]  # (N_col, K_max_col, b, head_dim)
    do_gathered_col = dO_by_block[safe_row]  # (N_col, K_max_col, b, head_dim)
    D_gathered_col = D_blocks[safe_row]  # (N_col, K_max_col, b)
    row_max_gathered_col = row_max[safe_row]  # (N_col, K_max_col, b)
    row_denom_gathered_col = row_denom[safe_row]  # (N_col, K_max_col, b)

    col_first = {
        "k_blocks": k_blocks,
        "v_blocks": v_blocks,
        "q_gathered": q_gathered_col,
        "do_gathered": do_gathered_col,
        "D_gathered": D_gathered_col,
        "row_max_gathered": row_max_gathered_col,
        "row_denom_gathered": row_denom_gathered_col,
    }

    return row_first, col_first


def nki_bsr_attn_bwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    O: torch.Tensor,
    mask_bsr,
    row_max: torch.Tensor,
    row_denom: torch.Tensor,
) -> tuple:
    """NKI backward pass for block-sparse tiled attention.

    Runs two NKI kernels:
      1. `_attn_bwd_dq_kernel`   — row-first, computes dQ.
      2. `_attn_bwd_dkdv_kernel` — column-first, computes dK and dV.

    Returns:
        (dQ, dK, dV) — same shape as (Q, K, V).
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    scale = head_dim**-0.5

    row_first, col_first = _attn_bwd_gather(Q, K, V, dO, O, mask_bsr, scale, row_max, row_denom)

    # Pack contiguous inputs.
    rf = {k: v.contiguous() for k, v in row_first.items()}
    cf = {k: v.contiguous() for k, v in col_first.items()}

    try:
        if _use_simulator():
            dQ_np = nki.simulate(_attn_bwd_dq_kernel)(
                rf["q_scaled"].cpu().numpy(),
                rf["k_gathered"].cpu().numpy(),
                rf["v_gathered"].cpu().numpy(),
                rf["do_gathered"].cpu().numpy(),
                rf["D_blocks"].cpu().numpy(),
                row_max.contiguous().cpu().numpy(),
                row_denom.contiguous().cpu().numpy(),
            )
            dQ_raw = torch.from_numpy(np.asarray(dQ_np)).to(Q.device)

            dK_np, dV_np = nki.simulate(_attn_bwd_dkdv_kernel)(
                cf["k_blocks"].cpu().numpy(),
                cf["v_blocks"].cpu().numpy(),
                cf["q_gathered"].cpu().numpy(),
                cf["do_gathered"].cpu().numpy(),
                cf["D_gathered"].cpu().numpy(),
                cf["row_max_gathered"].cpu().numpy(),
                cf["row_denom_gathered"].cpu().numpy(),
            )
            dK_raw = torch.from_numpy(np.asarray(dK_np)).to(Q.device)
            dV_raw = torch.from_numpy(np.asarray(dV_np)).to(Q.device)
        else:
            (
                (
                    qs_x,
                    kg_x,
                    vg_x,
                    dog_x,
                    db_x,
                    rm_x,
                    rd_x,
                ),
                orig_device,
            ) = _to_xla(
                rf["q_scaled"],
                rf["k_gathered"],
                rf["v_gathered"],
                rf["do_gathered"],
                rf["D_blocks"],
                row_max.contiguous(),
                row_denom.contiguous(),
            )
            dQ_x = _attn_bwd_dq_kernel(qs_x, kg_x, vg_x, dog_x, db_x, rm_x, rd_x)
            dQ_raw = dQ_x.to(orig_device)

            (kb_x, vb_x, qgc_x, dogc_x, dgc_x, rmgc_x, rdgc_x), _ = _to_xla(
                cf["k_blocks"],
                cf["v_blocks"],
                cf["q_gathered"],
                cf["do_gathered"],
                cf["D_gathered"],
                cf["row_max_gathered"],
                cf["row_denom_gathered"],
            )
            dK_x, dV_x = _attn_bwd_dkdv_kernel(kb_x, vb_x, qgc_x, dogc_x, dgc_x, rmgc_x, rdgc_x)
            dK_raw = dK_x.to(orig_device)
            dV_raw = dV_x.to(orig_device)

        return (
            dQ_raw[:seq_len, :head_dim].contiguous(),
            dK_raw[:seq_len, :head_dim].contiguous(),
            dV_raw[:seq_len, :head_dim].contiguous(),
        )
    except Exception:
        if _REQUIRE_NKI:
            raise
        from ..ops import _block_sparse_attn_backward

        return _block_sparse_attn_backward(
            Q, K, V, O, dO, mask_bsr, scale, row_max=row_max, row_denom=row_denom
        )
