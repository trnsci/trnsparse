"""
trnsparse — Sparse matrix operations for AWS Trainium via NKI.

CSR/COO formats, SpMV, SpMM, and integral screening for
sparse scientific computing. Part of the trnsci scientific computing suite.
"""

__version__ = "0.4.1"

from .formats import BSRMatrix, COOMatrix, CSRMatrix, eye_sparse, from_dense, from_scipy
from .iterative import bsr_diagonal, cg_bsr, jacobi_preconditioner_bsr, power_iteration_bsr
from .nki import HAS_NKI, get_backend, set_backend
from .ops import (
    bsr_spmm,
    nnz_per_row,
    screened_spmm,
    sparse_add,
    sparse_scale,
    sparse_transpose,
    spmm,
    spmv,
    spmv_symmetric,
)
from .screening import density_screen, schwarz_bounds, screen_quartets, sparsity_stats

__all__ = [
    "CSRMatrix",
    "COOMatrix",
    "BSRMatrix",
    "from_dense",
    "from_scipy",
    "eye_sparse",
    "spmv",
    "spmm",
    "spmv_symmetric",
    "bsr_spmm",
    "screened_spmm",
    "sparse_add",
    "sparse_scale",
    "sparse_transpose",
    "nnz_per_row",
    "schwarz_bounds",
    "screen_quartets",
    "density_screen",
    "sparsity_stats",
    "cg_bsr",
    "power_iteration_bsr",
    "jacobi_preconditioner_bsr",
    "bsr_diagonal",
    "HAS_NKI",
    "set_backend",
    "get_backend",
]
