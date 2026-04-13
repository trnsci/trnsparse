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

from dataclasses import dataclass

import torch


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
    shape: tuple[int, int]

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
        t = torch.sparse_csr_tensor(self.row_ptrs, self.col_indices, self.values, size=self.shape)
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
    shape: tuple[int, int]

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


@dataclass
class BSRMatrix:
    """Block-Sparse Row format.

    The Trainium-native sparse representation: every nonzero block is a
    dense `(block_size, block_size)` tile that matches the Tensor Engine's
    stationary-operand partition dim (128). A nonzero block becomes exactly
    one `nc_matmul` call with zero gather overhead — the block is already
    in the shape the systolic array wants.

    For an (M, N) matrix with `block_size = b` and `nb` nonzero blocks:
        blocks:           (nb, b, b)         — stacked dense blocks
        block_col_indices: (nb,)             — column-block index for each block
        block_row_ptrs:   (M/b + 1,)         — row-block i spans
                                               block_row_ptrs[i]:block_row_ptrs[i+1]
        shape:            (M, N)             — element-wise shape (multiples of b)
        block_size:       int                — defaults to 128 (Tensor Engine tile)

    `M` and `N` must be multiples of `block_size`. `from_dense`/`from_csr`
    handle padding for non-aligned inputs.

    BSR is the headline format for v0.3.0+. CSR/COO remain for construction
    and interop; on-device compute should go through BSR wherever the
    matrix has block structure (Fock matrices, FEM stiffness, GNN
    adjacencies, block-sparse attention masks).
    """

    blocks: torch.Tensor
    block_col_indices: torch.Tensor
    block_row_ptrs: torch.Tensor
    shape: tuple[int, int]
    block_size: int = 128

    @property
    def n_blocks(self) -> int:
        return self.blocks.shape[0]

    @property
    def nnz(self) -> int:
        """Effective nnz = n_blocks * block_size^2 (counts zeros inside nonzero blocks)."""
        return self.n_blocks * self.block_size * self.block_size

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks.dtype

    @property
    def density(self) -> float:
        """Block density: fraction of blocks stored (not fraction of elements nonzero)."""
        m, n = self.shape
        total_blocks = (m // self.block_size) * (n // self.block_size)
        return self.n_blocks / total_blocks if total_blocks else 0.0

    @classmethod
    def from_dense(
        cls, A: torch.Tensor, block_size: int = 128, threshold: float = 0.0
    ) -> BSRMatrix:
        """Build a BSR from a dense matrix.

        A block is stored iff it has at least one element with `|v| > threshold`.
        Shapes are padded up to a multiple of `block_size`.
        """
        m, n = A.shape
        # Pad to block alignment
        m_pad = ((m + block_size - 1) // block_size) * block_size
        n_pad = ((n + block_size - 1) // block_size) * block_size
        if (m_pad, n_pad) != (m, n):
            padded = torch.zeros(m_pad, n_pad, dtype=A.dtype, device=A.device)
            padded[:m, :n] = A
            A = padded

        mb = m_pad // block_size
        nb_cols = n_pad // block_size
        # View as a block grid: (mb, block_size, nb_cols, block_size) -> (mb, nb_cols, block_size, block_size)
        block_grid = A.reshape(mb, block_size, nb_cols, block_size).permute(0, 2, 1, 3).contiguous()
        # Nonzero block mask
        block_max = block_grid.abs().amax(dim=(-2, -1))  # (mb, nb_cols)
        nonzero_mask = block_max > threshold

        block_row_ptrs = torch.zeros(mb + 1, dtype=torch.long)
        block_row_ptrs[1:] = nonzero_mask.sum(dim=1).cumsum(dim=0)

        block_col_indices = nonzero_mask.nonzero(as_tuple=False)[:, 1].to(torch.long)
        kept_blocks = block_grid[nonzero_mask]  # (n_blocks, block_size, block_size)

        return cls(
            blocks=kept_blocks.contiguous(),
            block_col_indices=block_col_indices,
            block_row_ptrs=block_row_ptrs,
            shape=(m, n),  # report the original shape, not the padded one
            block_size=block_size,
        )

    @classmethod
    def from_csr(cls, csr: CSRMatrix, block_size: int = 128, threshold: float = 0.0) -> BSRMatrix:
        """Convert CSR → BSR. Simple path: densify then block.

        For large matrices this materializes the dense intermediate; BSR is
        most useful when the caller already has block structure and can
        skip the CSR step entirely (`from_dense` or a direct constructor).
        """
        return cls.from_dense(csr.to_dense(), block_size=block_size, threshold=threshold)

    def to_dense(self) -> torch.Tensor:
        """Expand the stored blocks back into a dense `(M, N)` tensor."""
        m, n = self.shape
        b = self.block_size
        # Round up to block alignment for the internal layout
        m_pad = ((m + b - 1) // b) * b
        n_pad = ((n + b - 1) // b) * b
        mb = m_pad // b
        nb_cols = n_pad // b

        grid = torch.zeros(mb, nb_cols, b, b, dtype=self.dtype)
        for i in range(mb):
            start = self.block_row_ptrs[i].item()
            end = self.block_row_ptrs[i + 1].item()
            for k in range(start, end):
                j = self.block_col_indices[k].item()
                grid[i, j] = self.blocks[k]
        dense = grid.permute(0, 2, 1, 3).reshape(m_pad, n_pad)
        return dense[:m, :n].contiguous()

    def to_csr(self) -> CSRMatrix:
        """Convert BSR → CSR via dense intermediate. See from_csr caveat."""
        from . import formats  # noqa: F401 — avoid circular at module import

        return from_dense(self.to_dense(), threshold=0.0)

    def __repr__(self) -> str:
        return (
            f"BSRMatrix(shape={self.shape}, block_size={self.block_size}, "
            f"n_blocks={self.n_blocks}, block_density={self.density:.4f})"
        )


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
