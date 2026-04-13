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

import torch
from typing import Optional

from .formats import CSRMatrix, COOMatrix


def _as_torch_csr(A: CSRMatrix) -> torch.Tensor:
    """View CSRMatrix as a torch.sparse_csr_tensor (no copy)."""
    return torch.sparse_csr_tensor(
        A.row_ptrs, A.col_indices, A.values, size=A.shape
    )


def spmv(A: CSRMatrix, x: torch.Tensor, alpha: float = 1.0,
         y: Optional[torch.Tensor] = None, beta: float = 0.0) -> torch.Tensor:
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


def spmm(A: CSRMatrix, B: torch.Tensor, alpha: float = 1.0,
         C: Optional[torch.Tensor] = None, beta: float = 0.0) -> torch.Tensor:
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


def spmv_symmetric(A: CSRMatrix, x: torch.Tensor, alpha: float = 1.0,
                   uplo: str = "upper") -> torch.Tensor:
    """Symmetric sparse matrix × vector using only stored triangle.

    For symmetric matrices (like the overlap matrix S or density P),
    only half the non-zeros need to be stored. Computes
    `A @ x + (A_strict_triangle)ᵀ @ x` as two vectorized SpMVs.
    """
    m, n = A.shape
    assert m == n, "Matrix must be square for symmetric SpMV"

    rows = torch.repeat_interleave(
        torch.arange(m), A.row_ptrs[1:] - A.row_ptrs[:-1]
    )
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
    indices = torch.stack([
        torch.cat([coo_a.row_indices, coo_b.row_indices]),
        torch.cat([coo_a.col_indices, coo_b.col_indices]),
    ])
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
