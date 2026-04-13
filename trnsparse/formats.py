"""
Sparse matrix formats for Trainium.

CSR (Compressed Sparse Row) and COO (Coordinate) formats with
conversion routines. CSR is the primary compute format; COO is
the natural construction format.

For quantum chemistry: the two-electron integral tensor (μν|λσ) is
extremely sparse after Schwarz screening — typically >99% zero for
large molecules. Storing and operating on it in dense format is
the difference between O(N⁴) and O(N²) effective cost.
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class CSRMatrix:
    """Compressed Sparse Row format.

    For an m×n matrix with nnz non-zeros:
        values: (nnz,) — non-zero values
        col_indices: (nnz,) — column index for each value
        row_ptrs: (m+1,) — row_ptrs[i]:row_ptrs[i+1] span row i's entries
        shape: (m, n)
    """
    values: torch.Tensor
    col_indices: torch.Tensor
    row_ptrs: torch.Tensor
    shape: Tuple[int, int]

    @property
    def nnz(self) -> int:
        return self.values.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    @property
    def density(self) -> float:
        return self.nnz / (self.shape[0] * self.shape[1])

    def to_dense(self) -> torch.Tensor:
        """Convert to dense tensor."""
        t = torch.sparse_csr_tensor(
            self.row_ptrs, self.col_indices, self.values, size=self.shape
        )
        return t.to_dense()

    def to_coo(self) -> COOMatrix:
        """Convert to COO format."""
        rows = []
        for i in range(self.shape[0]):
            start, end = self.row_ptrs[i].item(), self.row_ptrs[i + 1].item()
            rows.extend([i] * (end - start))
        return COOMatrix(
            values=self.values.clone(),
            row_indices=torch.tensor(rows, dtype=torch.long),
            col_indices=self.col_indices.clone(),
            shape=self.shape,
        )

    def __repr__(self) -> str:
        return f"CSRMatrix(shape={self.shape}, nnz={self.nnz}, density={self.density:.4f})"


@dataclass
class COOMatrix:
    """Coordinate (triplet) format.

    values: (nnz,) — non-zero values
    row_indices: (nnz,) — row index for each value
    col_indices: (nnz,) — column index for each value
    shape: (m, n)
    """
    values: torch.Tensor
    row_indices: torch.Tensor
    col_indices: torch.Tensor
    shape: Tuple[int, int]

    @property
    def nnz(self) -> int:
        return self.values.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    @property
    def density(self) -> float:
        return self.nnz / (self.shape[0] * self.shape[1])

    def to_dense(self) -> torch.Tensor:
        m, n = self.shape
        dense = torch.zeros(m, n, dtype=self.dtype)
        dense[self.row_indices, self.col_indices] = self.values
        return dense

    def to_csr(self) -> CSRMatrix:
        """Convert to CSR format."""
        m, n = self.shape
        # Sort by row then column
        sort_idx = torch.argsort(self.row_indices * n + self.col_indices)
        sorted_rows = self.row_indices[sort_idx]
        sorted_cols = self.col_indices[sort_idx]
        sorted_vals = self.values[sort_idx]

        # Build row pointers
        row_ptrs = torch.zeros(m + 1, dtype=torch.long)
        for r in sorted_rows:
            row_ptrs[r + 1] += 1
        row_ptrs = torch.cumsum(row_ptrs, dim=0)

        return CSRMatrix(
            values=sorted_vals,
            col_indices=sorted_cols,
            row_ptrs=row_ptrs,
            shape=self.shape,
        )

    def __repr__(self) -> str:
        return f"COOMatrix(shape={self.shape}, nnz={self.nnz}, density={self.density:.4f})"


# --- Construction helpers ---

def from_dense(A: torch.Tensor, threshold: float = 0.0) -> CSRMatrix:
    """Convert dense matrix to CSR, dropping values with |v| <= threshold."""
    m, n = A.shape
    mask = torch.abs(A) > threshold
    rows, cols = torch.where(mask)
    vals = A[mask]

    # Build CSR
    coo = COOMatrix(values=vals, row_indices=rows, col_indices=cols, shape=(m, n))
    return coo.to_csr()


def from_scipy(sp_matrix) -> CSRMatrix:
    """Convert scipy.sparse matrix to CSRMatrix."""
    import scipy.sparse as sp
    csr = sp.csr_matrix(sp_matrix)
    return CSRMatrix(
        values=torch.tensor(csr.data, dtype=torch.float32),
        col_indices=torch.tensor(csr.indices, dtype=torch.long),
        row_ptrs=torch.tensor(csr.indptr, dtype=torch.long),
        shape=csr.shape,
    )


def eye_sparse(n: int, dtype: torch.dtype = torch.float32) -> CSRMatrix:
    """Sparse identity matrix."""
    return CSRMatrix(
        values=torch.ones(n, dtype=dtype),
        col_indices=torch.arange(n, dtype=torch.long),
        row_ptrs=torch.arange(n + 1, dtype=torch.long),
        shape=(n, n),
    )
