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


def spmv(A: CSRMatrix, x: torch.Tensor, alpha: float = 1.0,
         y: Optional[torch.Tensor] = None, beta: float = 0.0) -> torch.Tensor:
    """Sparse matrix × dense vector: y = alpha * A @ x + beta * y

    CSR row-wise dot products — each row's non-zeros dot with
    corresponding entries of x.
    """
    m, n = A.shape
    assert x.shape[0] == n, f"Dimension mismatch: A is {A.shape}, x is {x.shape}"

    result = torch.zeros(m, dtype=A.dtype)
    for i in range(m):
        start = A.row_ptrs[i].item()
        end = A.row_ptrs[i + 1].item()
        if start < end:
            cols = A.col_indices[start:end]
            vals = A.values[start:end]
            result[i] = torch.dot(vals, x[cols])

    result = alpha * result
    if y is not None and beta != 0.0:
        result = result + beta * y
    return result


def spmm(A: CSRMatrix, B: torch.Tensor, alpha: float = 1.0,
         C: Optional[torch.Tensor] = None, beta: float = 0.0) -> torch.Tensor:
    """Sparse matrix × dense matrix: C = alpha * A @ B + beta * C

    Each row of A selects and weights rows of B.
    On NKI: gather selected B rows into dense tile, matmul, scatter.
    """
    m, n = A.shape
    k = B.shape[1]
    assert B.shape[0] == n, f"Dimension mismatch: A is {A.shape}, B is {B.shape}"

    result = torch.zeros(m, k, dtype=A.dtype)
    for i in range(m):
        start = A.row_ptrs[i].item()
        end = A.row_ptrs[i + 1].item()
        if start < end:
            cols = A.col_indices[start:end]
            vals = A.values[start:end]
            # vals (nnz_row,) × B[cols] (nnz_row, k) → (k,)
            result[i] = vals @ B[cols]

    result = alpha * result
    if C is not None and beta != 0.0:
        result = result + beta * C
    return result


def spmv_symmetric(A: CSRMatrix, x: torch.Tensor, alpha: float = 1.0,
                   uplo: str = "upper") -> torch.Tensor:
    """Symmetric sparse matrix × vector using only stored triangle.

    For symmetric matrices (like the overlap matrix S or density P),
    only half the non-zeros need to be stored. This computes A @ x
    using only the upper or lower triangle entries.
    """
    m, n = A.shape
    assert m == n, "Matrix must be square for symmetric SpMV"
    result = torch.zeros(m, dtype=A.dtype)

    for i in range(m):
        start = A.row_ptrs[i].item()
        end = A.row_ptrs[i + 1].item()
        for idx in range(start, end):
            j = A.col_indices[idx].item()
            v = A.values[idx].item()
            result[i] += v * x[j]
            if i != j:
                result[j] += v * x[i]  # Symmetric contribution

    return alpha * result


def sparse_add(A: CSRMatrix, B: CSRMatrix, alpha: float = 1.0, beta: float = 1.0) -> CSRMatrix:
    """Sparse matrix addition: C = alpha * A + beta * B

    Result may have more non-zeros than either input (union of patterns).
    """
    assert A.shape == B.shape, f"Shape mismatch: {A.shape} vs {B.shape}"
    # Simple implementation via dense (fine for moderate sizes)
    dense = alpha * A.to_dense() + beta * B.to_dense()
    from .formats import from_dense
    return from_dense(dense, threshold=0.0)


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
