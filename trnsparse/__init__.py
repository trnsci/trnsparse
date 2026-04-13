"""
trnsparse — Sparse matrix operations for AWS Trainium via NKI.

CSR/COO formats, SpMV, SpMM, and integral screening for
sparse scientific computing. Part of the trnsci scientific computing suite.
"""

__version__ = "0.2.0"

from .formats import CSRMatrix, COOMatrix, BSRMatrix, from_dense, from_scipy, eye_sparse
from .ops import (spmv, spmm, spmv_symmetric, bsr_spmm, sparse_add,
                  sparse_scale, sparse_transpose, nnz_per_row)
from .screening import schwarz_bounds, screen_quartets, density_screen, sparsity_stats
from .nki import HAS_NKI, set_backend, get_backend

__all__ = [
    "CSRMatrix", "COOMatrix", "BSRMatrix",
    "from_dense", "from_scipy", "eye_sparse",
    "spmv", "spmm", "spmv_symmetric", "bsr_spmm",
    "sparse_add", "sparse_scale", "sparse_transpose", "nnz_per_row",
    "schwarz_bounds", "screen_quartets", "density_screen", "sparsity_stats",
    "HAS_NKI", "set_backend", "get_backend",
]
