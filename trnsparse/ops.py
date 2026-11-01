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

Placement (native TorchNeuron, Trn2/Trn3): sparse structure lives on the
host, dense operands may live on the host or on `device="neuron"`, and
every result lands on the dense operand's device. The `neuron` device has
no sparse CSR tensors (aws-neuron-sdk#1442), so when the dense operand is
on another device than the CSR, A is densified on the host (no torch
sparse tensor) and only the dense data moves.
"""

from __future__ import annotations

import math

import torch

from .formats import BSRMatrix, COOMatrix, CSRMatrix, _csr_to_dense_host


def _as_torch_csr(A: CSRMatrix) -> torch.Tensor:
    """View CSRMatrix as a torch.sparse_csr_tensor (no copy)."""
    return torch.sparse_csr_tensor(A.row_ptrs, A.col_indices, A.values, size=A.shape)


def _torch_sparse_ok(A: CSRMatrix, X: torch.Tensor) -> bool:
    """Can `A @ X` use a torch sparse tensor? Only when A and X share a
    device and that device is not `neuron` (no sparse CSR there,
    aws-neuron-sdk#1442)."""
    same = A.values.device == A.row_ptrs.device == A.col_indices.device == X.device
    return same and X.device.type != "neuron"


def _csr_matmul(A: CSRMatrix, X: torch.Tensor) -> torch.Tensor:
    """`A @ X` on X's device.

    Same (non-neuron) device: the torch sparse CSR matmul, as before.
    Otherwise (a host CSR against a `neuron` operand): densify A on the host
    and move only the dense matrix, because the device has no sparse CSR
    tensors.
    """
    if _torch_sparse_ok(A, X):
        return _as_torch_csr(A) @ X
    return _csr_to_dense_host(A).to(X.device) @ X


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

    Placement: `A` on the host, `x` on the host or on `neuron`. With `x` on
    a device, A is densified on the host and the dense product runs on
    x's device (the `neuron` device has no sparse CSR); the result is on
    x's device. For repeated SpMVs with the same A on device, prefer
    `bsr_spmm` (dense blocks stay resident) or densify once yourself.
    """
    m, n = A.shape
    assert x.shape[0] == n, f"Dimension mismatch: A is {A.shape}, x is {x.shape}"

    result = alpha * _csr_matmul(A, x)
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

    Placement (native TorchNeuron): `A` (structure and values) on the
    host, `B` on the host or on `neuron`. A is densified on the host and
    the dense tile moves to the device for the `trnsparse::spmm_dense`
    kernel op; the result is on B's device. Differentiable w.r.t.
    `A.values` and `B` (autograd registered on the op).
    """
    from .nki.dispatch import _use_nki, nki_spmm

    m, n = A.shape
    assert B.shape[0] == n, f"Dimension mismatch: A is {A.shape}, B is {B.shape}"

    if _use_nki():
        result = alpha * nki_spmm(A, B)
    else:
        result = alpha * _csr_matmul(A, B)
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

    if not _torch_sparse_ok(A, x):
        return _spmv_symmetric_dense(A, x, alpha)

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


def _spmv_symmetric_dense(A: CSRMatrix, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """`spmv_symmetric` for a host CSR and an operand on another device.

    The device has no sparse tensors: build the symmetric dense view
    (stored triangle plus the transpose of its strict part) on the host and
    move it once.
    """
    D = _csr_to_dense_host(A)
    strict = D - torch.diag(torch.diagonal(D))
    return alpha * ((D + strict.T).to(x.device) @ x)


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
    # Structure from host copies; blocks follow B's device.
    row_ptrs = A.block_row_ptrs.cpu()
    col_indices = A.block_col_indices.cpu()
    blocks = A.blocks.to(B.device)
    for i in range(mb):
        start = row_ptrs[i].item()
        end = row_ptrs[i + 1].item()
        for idx in range(start, end):
            j = col_indices[idx].item()
            out[i * b : (i + 1) * b] += blocks[idx] @ B[j * b : (j + 1) * b]
    return out[:m]


def bsr_spmm(A: BSRMatrix, B: torch.Tensor) -> torch.Tensor:
    """Block-sparse × dense matmul: C = A @ B.

    On NKI backend: routes through `_BSRSpMMFunction`, which runs one
    `nc_matmul` per nonzero block (zero gather overhead — each block is
    already a Tensor Engine tile).

    On PyTorch backend: the reference loop in `_bsr_spmm_pytorch`.

    Placement (native TorchNeuron): `A.block_row_ptrs` and
    `A.block_col_indices` on the host; `A.blocks` and `B` on the host or on
    `neuron`. The result is on B's device.
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


def _block_sparse_attn_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask_bsr,
    scale: float,
    return_stats: bool = False,
):
    """Two-pass block-sparse attention — PyTorch reference (no autograd).

    Called by `_AttnTiledFunction.forward` and the no-grad path of
    `block_sparse_attention_tiled`. Not for direct use.

    Args:
        return_stats: If True, returns (out, row_max, row_denom) where
            row_max and row_denom have shape (M_tiles, block_size). Used
            by `_AttnTiledFunction.forward` to save stats for backward.
    """
    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    M_tiles = seq_len // b
    n_blocks = mask_bsr.n_blocks

    dev = Q.device
    tile_max = torch.full((n_blocks, b), float("-inf"), dtype=Q.dtype, device=dev)
    tile_sumexp = torch.zeros((n_blocks, b), dtype=Q.dtype, device=dev)

    for m in range(M_tiles):
        start = mask_bsr.block_row_ptrs[m].item()
        end = mask_bsr.block_row_ptrs[m + 1].item()
        q_block = Q[m * b : (m + 1) * b]
        for idx in range(start, end):
            col = mask_bsr.block_col_indices[idx].item()
            score = (q_block @ K[col * b : (col + 1) * b].T) * scale
            tile_max[idx] = score.max(dim=1).values
            tile_sumexp[idx] = torch.exp(score - tile_max[idx].unsqueeze(1)).sum(dim=1)

    row_max_flat = torch.full((seq_len,), float("-inf"), dtype=Q.dtype, device=dev)
    for m in range(M_tiles):
        start = mask_bsr.block_row_ptrs[m].item()
        end = mask_bsr.block_row_ptrs[m + 1].item()
        if start == end:
            row_max_flat[m * b : (m + 1) * b] = 0.0
            continue
        row_max_flat[m * b : (m + 1) * b] = tile_max[start:end].max(dim=0).values

    row_denom_flat = torch.zeros(seq_len, dtype=Q.dtype, device=dev)
    for m in range(M_tiles):
        start = mask_bsr.block_row_ptrs[m].item()
        end = mask_bsr.block_row_ptrs[m + 1].item()
        if start == end:
            continue
        row_max_m = row_max_flat[m * b : (m + 1) * b]
        for idx in range(start, end):
            row_denom_flat[m * b : (m + 1) * b] += tile_sumexp[idx] * torch.exp(
                tile_max[idx] - row_max_m
            )

    row_denom_flat = row_denom_flat.clamp(min=1e-12)

    out = torch.zeros(seq_len, head_dim, dtype=Q.dtype, device=dev)
    for m in range(M_tiles):
        start = mask_bsr.block_row_ptrs[m].item()
        end = mask_bsr.block_row_ptrs[m + 1].item()
        if start == end:
            continue
        q_block = Q[m * b : (m + 1) * b]
        row_max_m = row_max_flat[m * b : (m + 1) * b]
        row_denom_m = row_denom_flat[m * b : (m + 1) * b]
        for idx in range(start, end):
            col = mask_bsr.block_col_indices[idx].item()
            score = (q_block @ K[col * b : (col + 1) * b].T) * scale
            weights = torch.exp(score - row_max_m.unsqueeze(1)) / row_denom_m.unsqueeze(1)
            out[m * b : (m + 1) * b] += weights @ V[col * b : (col + 1) * b]

    if return_stats:
        # Return stats as (M_tiles, b) for uniform shape with NKI path.
        return out, row_max_flat.view(M_tiles, b), row_denom_flat.view(M_tiles, b)
    return out


def _block_sparse_attn_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    mask_bsr,
    scale: float,
    row_max: torch.Tensor | None = None,
    row_denom: torch.Tensor | None = None,
) -> tuple:
    """Backward pass for block-sparse tiled attention.

    Uses the Flash Attention delta identity: D_i = dO_i · O_i (row-wise
    dot product). For each stored block (m, ki), recomputes P_block from
    Q and K (same two-pass as forward), then applies the softmax backward:

        dS = P * (dP - D_m)    where dP = dO_m @ V_ki.T

    Accumulates dQ, dK, dV without materialising the full attention matrix.

    Args:
        row_max:   Optional (M_tiles, block_size) saved from forward ctx.
                   When provided, skips the row_max recomputation pass.
        row_denom: Optional (M_tiles, block_size) saved from forward ctx.
                   When provided, skips the row_denom recomputation pass.

    Returns:
        (dQ, dK, dV) — same shape as (Q, K, V).
    """
    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    M_tiles = seq_len // b

    # Delta: D_i = dO_i · O_i (row-wise dot product of upstream grad with output)
    D = (dO * O).sum(dim=-1)  # (seq_len,)

    if row_max is not None and row_denom is not None:
        # Stats saved from forward — flatten to (seq_len,) for indexing.
        row_max_flat = row_max.reshape(seq_len)
        row_denom_flat = row_denom.reshape(seq_len)
    else:
        # --- Recompute row_max (one pass over blocks) ---
        row_max_flat = torch.full((seq_len,), float("-inf"), dtype=Q.dtype, device=Q.device)
        for m in range(M_tiles):
            start = mask_bsr.block_row_ptrs[m].item()
            end = mask_bsr.block_row_ptrs[m + 1].item()
            if start == end:
                row_max_flat[m * b : (m + 1) * b] = 0.0
                continue
            q_block = Q[m * b : (m + 1) * b]
            row_max_m = row_max_flat[m * b : (m + 1) * b]
            for idx in range(start, end):
                col = mask_bsr.block_col_indices[idx].item()
                score = (q_block @ K[col * b : (col + 1) * b].T) * scale
                row_max_m = torch.maximum(row_max_m, score.max(dim=1).values)
            row_max_flat[m * b : (m + 1) * b] = row_max_m

        # --- Recompute row_denom (second pass over blocks) ---
        row_denom_flat = torch.zeros(seq_len, dtype=Q.dtype, device=Q.device)
        for m in range(M_tiles):
            start = mask_bsr.block_row_ptrs[m].item()
            end = mask_bsr.block_row_ptrs[m + 1].item()
            if start == end:
                continue
            q_block = Q[m * b : (m + 1) * b]
            row_max_m = row_max_flat[m * b : (m + 1) * b]
            for idx in range(start, end):
                col = mask_bsr.block_col_indices[idx].item()
                score = (q_block @ K[col * b : (col + 1) * b].T) * scale
                row_denom_flat[m * b : (m + 1) * b] += torch.exp(
                    score - row_max_m.unsqueeze(1)
                ).sum(dim=1)

        row_denom_flat = row_denom_flat.clamp(min=1e-12)

    # Use local names for the grad accumulation pass below.
    row_max = row_max_flat
    row_denom = row_denom_flat

    # --- Gradient accumulation (single pass over blocks) ---
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    for m in range(M_tiles):
        start = mask_bsr.block_row_ptrs[m].item()
        end = mask_bsr.block_row_ptrs[m + 1].item()
        if start == end:
            continue
        q_block = Q[m * b : (m + 1) * b]
        do_block = dO[m * b : (m + 1) * b]
        d_block = D[m * b : (m + 1) * b]  # (b,) — Flash delta for this block-row
        row_max_m = row_max[m * b : (m + 1) * b]
        row_denom_m = row_denom[m * b : (m + 1) * b]

        for idx in range(start, end):
            col = mask_bsr.block_col_indices[idx].item()
            k_block = K[col * b : (col + 1) * b]
            v_block = V[col * b : (col + 1) * b]

            # Recompute stable softmax weights
            score = (q_block @ k_block.T) * scale
            P = torch.exp(score - row_max_m.unsqueeze(1)) / row_denom_m.unsqueeze(1)

            # Softmax backward via delta identity: dS = P * (dP - D_m)
            dP = do_block @ v_block.T
            dS = P * (dP - d_block.unsqueeze(1))

            dQ[m * b : (m + 1) * b] += dS @ k_block * scale
            dK[col * b : (col + 1) * b] += dS.T @ q_block * scale
            dV[col * b : (col + 1) * b] += P.T @ do_block

    return dQ, dK, dV


class _AttnTiledFunction(torch.autograd.Function):
    """Autograd wrapper for block-sparse tiled attention (v0.5.0).

    Forward: dispatched to NKI kernel pair or PyTorch two-pass reference.
    Backward: tiled Flash-style gradient — no O(seq_len²) intermediate.

    Saves Q, K, V, O plus the BSRMatrix sparsity pattern components
    (not the block values — only the pattern is needed for backward).

    Used on the PyTorch/XLA path, the simulator and the PyTorch backend. On
    the native TorchNeuron path the same forward/backward pair is the
    `trnsparse::bsr_attn` custom op with `register_autograd`
    (`nki/_native_attn.py`).
    """

    @staticmethod
    def forward(ctx, Q, K, V, mask_col_indices, mask_row_ptrs, mask_shape, mask_block_size, scale):
        class _BSRHandle:
            block_col_indices = mask_col_indices
            block_row_ptrs = mask_row_ptrs
            shape = mask_shape
            block_size = mask_block_size

            @property
            def n_blocks(self):
                return len(self.block_col_indices)

        mask_bsr = _BSRHandle()

        from .nki.dispatch import _use_nki

        if _use_nki():
            from .nki.dispatch import nki_bsr_attn_tiled

            out, row_max, row_denom = nki_bsr_attn_tiled(Q, K, V, mask_bsr, return_stats=True)
        else:
            out, row_max, row_denom = _block_sparse_attn_pytorch(
                Q, K, V, mask_bsr, scale, return_stats=True
            )

        ctx.save_for_backward(Q, K, V, out, row_max, row_denom)
        ctx.mask_col_indices = mask_col_indices
        ctx.mask_row_ptrs = mask_row_ptrs
        ctx.mask_shape = mask_shape
        ctx.mask_block_size = mask_block_size
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, row_max, row_denom = ctx.saved_tensors

        class _BSRHandle:
            block_col_indices = ctx.mask_col_indices
            block_row_ptrs = ctx.mask_row_ptrs
            shape = ctx.mask_shape
            block_size = ctx.mask_block_size

            @property
            def n_blocks(self):
                return len(self.block_col_indices)

        mask_bsr = _BSRHandle()

        from .nki.dispatch import _use_nki

        if _use_nki():
            from .nki.dispatch import nki_bsr_attn_bwd

            dQ, dK, dV = nki_bsr_attn_bwd(Q, K, V, dO, O, mask_bsr, row_max, row_denom)
        else:
            dQ, dK, dV = _block_sparse_attn_backward(
                Q, K, V, O, dO, mask_bsr, ctx.scale, row_max=row_max, row_denom=row_denom
            )
        # 8 inputs: Q, K, V, col_indices, row_ptrs, shape, block_size, scale
        return dQ, dK, dV, None, None, None, None, None


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

    Differentiable (v0.5.0): when any of Q/K/V requires grad, the call is
    routed through `_AttnTiledFunction`, which wraps the tiled Flash-style
    backward. Backward also avoids O(seq_len²) intermediates.

    Args:
        Q: (seq_len, head_dim) query tensor.
        K: (seq_len, head_dim) key tensor.
        V: (seq_len, head_dim) value tensor.
        mask_bsr: BSRMatrix encoding the nonzero attention block pattern.
            Only the sparsity pattern is used (block_row_ptrs,
            block_col_indices); block values are ignored.
        scale: Optional QK scale (default: head_dim ** -0.5).

    Note:
        On the NKI backend, head_dim must be ≤ 128 (single tile) or a
        multiple of 128 (K-tiling, e.g., 256 or 512). The PyTorch backend
        accepts any head_dim.

        Placement (native TorchNeuron): the mask's `block_row_ptrs` /
        `block_col_indices` stay on the host; Q, K, V may be on the host or
        on `neuron`, and the output is on Q's device. The call runs as the
        `trnsparse::bsr_attn` custom op, whose backward (the NKI dQ and
        dK/dV kernels) is attached with `register_autograd`.

    Returns:
        (seq_len, head_dim) attention output.
    """
    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    if seq_len % b:
        # A partial last block row/column is not supported on any backend:
        # the PyTorch reference left the tail query rows zero and the NKI
        # gather raised an opaque view error. Pad Q/K/V and mask yourself
        # (padded keys must be masked out, not zero-padded).
        raise ValueError(
            f"block_sparse_attention_tiled needs seq_len ({seq_len}) to be a "
            f"multiple of mask_bsr.block_size ({b})"
        )

    if scale is None:
        scale = head_dim**-0.5

    from .nki.dispatch import _native_path, _use_nki

    if _use_nki() and _native_path():
        from .nki.dispatch import _get_native_ops

        out, _, _ = _get_native_ops()["bsr_attn"](
            Q,
            K,
            V,
            mask_bsr.block_col_indices.cpu(),
            mask_bsr.block_row_ptrs.cpu(),
            mask_bsr.block_size,
        )
        return out

    if torch.is_grad_enabled() and (Q.requires_grad or K.requires_grad or V.requires_grad):
        return _AttnTiledFunction.apply(
            Q,
            K,
            V,
            mask_bsr.block_col_indices,
            mask_bsr.block_row_ptrs,
            mask_bsr.shape,
            mask_bsr.block_size,
            scale,
        )

    if _use_nki():
        from .nki.dispatch import nki_bsr_attn_tiled

        return nki_bsr_attn_tiled(Q, K, V, mask_bsr)

    return _block_sparse_attn_pytorch(Q, K, V, mask_bsr, scale)
