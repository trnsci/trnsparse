"""
Sparse matrix operations for Trainium.

SpMV (sparse × dense vector), SpMM (sparse × dense matrix),
sparse addition, and sparse scaling.

For quantum chemistry:
- SpMV: Fock matrix × MO vector in iterative eigensolvers (CG/Davidson)
- SpMM: Sparse integral tensor × dense MO coefficient matrix
- Sparse add: Accumulating screened integral contributions

On Trainium, SpMV/SpMM map to gather-matmul patterns on the Tensor Engine:
gather non-zero rows/cols into dense tiles, matmul, scatter back.
"""

from __future__ import annotations

import math

import torch

from .formats import BSRMatrix, COOMatrix, CSRMatrix


def _as_torch_csr(A: CSRMatrix) -> torch.Tensor:
    """View CSRMatrix as a torch.sparse_csr_tensor (no copy)."""
    return torch.sparse_csr_tensor(A.row_ptrs, A.col_indices, A.values, size=A.shape)


def spmv(
    A: CSRMatrix,
    x: torch.Tensor,
    alpha: float = 1.0,
    y: torch.Tensor | None = None,
    beta: float = 0.0,
) -> torch.Tensor:
    """Sparse matrix × dense vector: y = alpha * A @ x + beta * y

    Lowers to `torch.sparse_csr_tensor @ x` — a single vectorized call
    instead of a per-row Python loop.
    """
    m, n = A.shape
    assert x.shape[0] == n, f"Dimension mismatch: A is {A.shape}, x is {x.shape}"

    result = alpha * (_as_torch_csr(A) @ x)
    if y is not None and beta != 0.0:
        result = result + beta * y
    return result


def spmm(
    A: CSRMatrix,
    B: torch.Tensor,
    alpha: float = 1.0,
    C: torch.Tensor | None = None,
    beta: float = 0.0,
) -> torch.Tensor:
    """Sparse matrix × dense matrix: C = alpha * A @ B + beta * C

    On NKI backend (v0.2.0): routes through `nki_spmm`, which
    materializes A dense, runs the NKI GEMM kernel on the Tensor Engine,
    and returns the dense result. Autograd-aware via `_SpMMFunction`.

    On PyTorch backend: lowers to `torch.sparse_csr_tensor @ B` (v0.1.3
    vectorized fallback).
    """
    from .nki.dispatch import _use_nki, nki_spmm

    m, n = A.shape
    assert B.shape[0] == n, f"Dimension mismatch: A is {A.shape}, B is {B.shape}"

    if _use_nki():
        result = alpha * nki_spmm(A, B)
    else:
        result = alpha * (_as_torch_csr(A) @ B)
    if C is not None and beta != 0.0:
        result = result + beta * C
    return result


def spmv_symmetric(
    A: CSRMatrix, x: torch.Tensor, alpha: float = 1.0, uplo: str = "upper"
) -> torch.Tensor:
    """Symmetric sparse matrix × vector using only stored triangle.

    For symmetric matrices (like the overlap matrix S or density P),
    only half the non-zeros need to be stored. Computes
    `A @ x + (A_strict_triangle)ᵀ @ x` as two vectorized SpMVs.
    """
    m, n = A.shape
    assert m == n, "Matrix must be square for symmetric SpMV"

    rows = torch.repeat_interleave(torch.arange(m), A.row_ptrs[1:] - A.row_ptrs[:-1])
    cols = A.col_indices
    strict_mask = rows != cols
    A_sparse = _as_torch_csr(A)

    strict = torch.sparse_coo_tensor(
        torch.stack([rows[strict_mask], cols[strict_mask]]),
        A.values[strict_mask],
        size=A.shape,
    ).coalesce()

    return alpha * (A_sparse @ x + strict.t() @ x)


def sparse_add(A: CSRMatrix, B: CSRMatrix, alpha: float = 1.0, beta: float = 1.0) -> CSRMatrix:
    """Sparse matrix addition: C = alpha * A + beta * B

    Result is the union of A's and B's patterns. Memory usage is
    O(nnz(A) + nnz(B)) — no dense intermediate.
    """
    assert A.shape == B.shape, f"Shape mismatch: {A.shape} vs {B.shape}"

    coo_a = A.to_coo()
    coo_b = B.to_coo()
    indices = torch.stack(
        [
            torch.cat([coo_a.row_indices, coo_b.row_indices]),
            torch.cat([coo_a.col_indices, coo_b.col_indices]),
        ]
    )
    values = torch.cat([alpha * coo_a.values, beta * coo_b.values])

    coalesced = torch.sparse_coo_tensor(indices, values, size=A.shape).coalesce()
    csr = coalesced.to_sparse_csr()
    return CSRMatrix(
        values=csr.values(),
        col_indices=csr.col_indices(),
        row_ptrs=csr.crow_indices(),
        shape=A.shape,
    )


def sparse_scale(A: CSRMatrix, alpha: float) -> CSRMatrix:
    """Scale sparse matrix: B = alpha * A"""
    return CSRMatrix(
        values=alpha * A.values,
        col_indices=A.col_indices.clone(),
        row_ptrs=A.row_ptrs.clone(),
        shape=A.shape,
    )


def sparse_transpose(A: CSRMatrix) -> CSRMatrix:
    """Transpose sparse matrix (CSR → CSR of A^T)."""
    coo = A.to_coo()
    transposed = COOMatrix(
        values=coo.values,
        row_indices=coo.col_indices,
        col_indices=coo.row_indices,
        shape=(A.shape[1], A.shape[0]),
    )
    return transposed.to_csr()


def nnz_per_row(A: CSRMatrix) -> torch.Tensor:
    """Number of non-zeros per row."""
    return A.row_ptrs[1:] - A.row_ptrs[:-1]


def _bsr_spmm_pytorch(A: BSRMatrix, B: torch.Tensor) -> torch.Tensor:
    """Block-sparse × dense matmul — PyTorch fallback reference.

    Iterates over stored blocks, does `block @ B[j*b:(j+1)*b, :]` and
    accumulates into the output row-block. This is the semantic spec
    for the NKI kernel to match.
    """
    m, n = A.shape
    _, k = B.shape
    b = A.block_size
    m_pad = ((m + b - 1) // b) * b
    n_pad = ((n + b - 1) // b) * b
    mb = m_pad // b

    if B.shape[0] != n:
        raise AssertionError(f"Dimension mismatch: A is {A.shape}, B is {B.shape}")

    # Pad B up to block-aligned rows
    if n_pad != n:
        B_p = torch.zeros(n_pad, k, dtype=B.dtype, device=B.device)
        B_p[:n, :] = B
        B = B_p

    out = torch.zeros(m_pad, k, dtype=A.dtype, device=B.device)
    for i in range(mb):
        start = A.block_row_ptrs[i].item()
        end = A.block_row_ptrs[i + 1].item()
        for idx in range(start, end):
            j = A.block_col_indices[idx].item()
            out[i * b : (i + 1) * b] += A.blocks[idx] @ B[j * b : (j + 1) * b]
    return out[:m]


def bsr_spmm(A: BSRMatrix, B: torch.Tensor) -> torch.Tensor:
    """Block-sparse × dense matmul: C = A @ B.

    On NKI backend: routes through `_BSRSpMMFunction`, which runs one
    `nc_matmul` per nonzero block (zero gather overhead — each block is
    already a Tensor Engine tile).

    On PyTorch backend: the reference loop in `_bsr_spmm_pytorch`.
    """
    from .nki.dispatch import _use_nki

    if _use_nki():
        from .nki.dispatch import nki_bsr_spmm

        return nki_bsr_spmm(A, B)
    return _bsr_spmm_pytorch(A, B)


def screened_spmm(
    A: torch.Tensor,
    diag_integrals: torch.Tensor,
    B: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Fused Schwarz-screened dense matmul: `C = (A * mask) @ B`.

    The mask is the Schwarz-inequality pair bound:

        Q[i]      = sqrt(|diag_integrals[i]|)
        mask[i,j] = (Q[i] * Q[j] > sqrt(threshold))

    On the NKI backend, the sqrt / outer-product / threshold / mask-apply
    / matmul chain is fused into a single `@nki.jit` kernel — one
    dispatch, no intermediate mask tensor on HBM, no separate BSR
    construction pass. Saves ~30-50% end-to-end vs the unfused
    `density_screen → screen_quartets → from_dense → spmm` flow at
    realistic Fock-build sizes.

    On the PyTorch backend, falls back to the explicit mask materialize
    + matmul (semantic spec for the NKI kernel to match).

    Args:
        A: Dense matrix, shape `(M, K)`. The unscreened operand —
            typically the integral slice `(μν|λσ)` for the λσ range.
        diag_integrals: Per-index Schwarz bounds source. Shape `(M,)`
            if `M == K` (square case), or passed as `(K,)` if one wants
            to screen based on the K dimension only. For the common
            chemistry use case (square A, symmetric bounds), shape `(M,)`.
        B: Dense RHS, shape `(K, N)`.
        threshold: Screening threshold. Pairs with
            `Q[i] * Q[j] <= sqrt(threshold)` are zeroed in `A` before
            the matmul.

    Returns:
        `C`, shape `(M, N)` = `(A * mask) @ B`.

    Differentiable via `_ScreenedSpMMFunction`; backward projects
    gradients back through the mask (`dA *= mask`, no gradient to
    `diag_integrals` or `threshold` since the mask is discrete).
    """
    from .nki.dispatch import _use_nki

    if _use_nki():
        from .nki.dispatch import nki_screened_spmm

        return nki_screened_spmm(A, diag_integrals, B, threshold)

    # PyTorch fallback — semantic spec for the kernel.
    # Requires diag_integrals 1-D of length matching A's rows and cols
    # (common chemistry case: A is square (n, n) with a per-shell bound vector).
    assert diag_integrals.dim() == 1, (
        f"diag_integrals must be 1-D; got shape {diag_integrals.shape}"
    )
    M, K = A.shape
    assert diag_integrals.shape[0] == M == K, (
        "screened_spmm requires square A with diag_integrals of matching length; "
        f"got A shape {A.shape}, diag_integrals shape {diag_integrals.shape}"
    )
    Q = torch.sqrt(torch.abs(diag_integrals))
    threshold_sqrt = math.sqrt(threshold)
    pair_bound = Q.unsqueeze(-1) * Q.unsqueeze(0)  # (M, K)
    mask = pair_bound > threshold_sqrt
    return (A * mask.to(A.dtype)) @ B


def block_sparse_attention_tiled(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask_bsr: BSRMatrix,
    scale: float | None = None,
) -> torch.Tensor:
    """Block-sparse attention without an O(seq_len²) score intermediate.

    The naive implementation (`examples/block_sparse_attention.py`) builds
    the full `(seq_len, seq_len)` score matrix before masking, which uses
    O(seq_len²) memory and compute regardless of block density. This
    two-pass implementation works only over the nonzero blocks:

    **Pass 1** — per-block statistics: for each stored block (m, ki),
    compute score_tile = Q[m] @ K[ki].T and extract its row-wise max
    (tile_max, shape (block_size,)) and stable row-wise exp-sum
    (tile_sumexp). Store to an O(n_stored_blocks × block_size) array.

    **Host reduction** — row-level softmax denominators: combine tile
    stats across all k-blocks in each block-row m into global row_max
    and row_denom vectors of length seq_len. O(n_stored_blocks × b) work.

    **Pass 2** — weighted accumulation: recompute score_tile for each
    stored block, apply the stable softmax normalisation using row_max
    and row_denom loaded from HBM, and accumulate weights @ V_block
    into the output.

    Memory: O(n_stored_blocks × block_size) for stats + O(seq_len × head_dim)
    for output. No (seq_len, seq_len) tensor is ever allocated.

    This is the PyTorch reference for the NKI `_attn_stats_kernel` +
    `_attn_out_kernel` pair documented in
    https://github.com/trnsci/trnsparse/issues/25.

    Args:
        Q: (seq_len, head_dim) query tensor.
        K: (seq_len, head_dim) key tensor.
        V: (seq_len, head_dim) value tensor.
        mask_bsr: BSRMatrix encoding the nonzero attention block pattern.
            Only the sparsity pattern is used (block_row_ptrs,
            block_col_indices); block values are ignored.
        scale: Optional QK scale (default: head_dim ** -0.5).

    Returns:
        (seq_len, head_dim) attention output.
    """
    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    M_tiles = seq_len // b

    if scale is None:
        scale = head_dim**-0.5

    n_blocks = mask_bsr.n_blocks

    # --- Pass 1: per-block (tile_max, tile_sumexp) ---
    # tile_max[idx, r]    = max over columns of score_tile[r, :]
    # tile_sumexp[idx, r] = sum over columns of exp(score[r,:] - tile_max[idx, r])
    tile_max = torch.full((n_blocks, b), float("-inf"), dtype=Q.dtype)
    tile_sumexp = torch.zeros((n_blocks, b), dtype=Q.dtype)

    for m in range(M_tiles):
        start = mask_bsr.block_row_ptrs[m].item()
        end = mask_bsr.block_row_ptrs[m + 1].item()
        q_block = Q[m * b : (m + 1) * b]  # (b, head_dim)
        for idx in range(start, end):
            col = mask_bsr.block_col_indices[idx].item()
            k_block = K[col * b : (col + 1) * b]  # (b, head_dim)
            score = (q_block @ k_block.T) * scale  # (b, b)
            tile_max[idx] = score.max(dim=1).values  # (b,)
            tile_sumexp[idx] = torch.exp(score - tile_max[idx].unsqueeze(1)).sum(dim=1)

    # --- Host reduction: row_max, row_denom (shape: seq_len each) ---
    row_max = torch.full((seq_len,), float("-inf"), dtype=Q.dtype)
    for m in range(M_tiles):
        start = mask_bsr.block_row_ptrs[m].item()
        end = mask_bsr.block_row_ptrs[m + 1].item()
        if start == end:
            row_max[m * b : (m + 1) * b] = 0.0
            continue
        row_max[m * b : (m + 1) * b] = tile_max[start:end].max(dim=0).values

    row_denom = torch.zeros(seq_len, dtype=Q.dtype)
    for m in range(M_tiles):
        start = mask_bsr.block_row_ptrs[m].item()
        end = mask_bsr.block_row_ptrs[m + 1].item()
        if start == end:
            continue
        row_max_m = row_max[m * b : (m + 1) * b]  # (b,)
        for idx in range(start, end):
            correction = torch.exp(tile_max[idx] - row_max_m)  # (b,)
            row_denom[m * b : (m + 1) * b] += tile_sumexp[idx] * correction

    row_denom = row_denom.clamp(min=1e-12)

    # --- Pass 2: recompute scores, normalise, accumulate V ---
    out = torch.zeros(seq_len, head_dim, dtype=Q.dtype)
    for m in range(M_tiles):
        start = mask_bsr.block_row_ptrs[m].item()
        end = mask_bsr.block_row_ptrs[m + 1].item()
        if start == end:
            continue
        q_block = Q[m * b : (m + 1) * b]  # (b, head_dim)
        row_max_m = row_max[m * b : (m + 1) * b]  # (b,)
        row_denom_m = row_denom[m * b : (m + 1) * b]  # (b,)
        for idx in range(start, end):
            col = mask_bsr.block_col_indices[idx].item()
            k_block = K[col * b : (col + 1) * b]
            v_block = V[col * b : (col + 1) * b]
            score = (q_block @ k_block.T) * scale  # (b, b)
            # Stable per-row softmax weights
            weights = torch.exp(score - row_max_m.unsqueeze(1)) / row_denom_m.unsqueeze(1)
            out[m * b : (m + 1) * b] += weights @ v_block  # (b, head_dim)

    return out
