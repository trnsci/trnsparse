# API Reference

## Formats

- `CSRMatrix(indptr, indices, data, shape)` — compressed sparse row
  - `.from_dense(dense, threshold=0)` — construct by thresholding a dense tensor
  - `.to_dense()` — materialize back to a dense `torch.Tensor`
  - `.transpose()` — transpose to CSR (via CSC)
  - `.nnz`, `.shape`, `.dtype`, `.device`
- `COOMatrix(rows, cols, vals, shape)` — coordinate format
  - `.to_csr()` / `CSRMatrix.to_coo()` — roundtrip conversions

## Ops

- `spmv(A, x)` — sparse matrix × dense vector
- `spmv_symmetric(A, x)` — for symmetric A; skips duplicate work
- `spmm(A, B)` — sparse matrix × dense matrix (gather-matmul-scatter on NKI)
- `sparse_add(A, B, alpha=1.0, beta=1.0)` — `α A + β B`
- `sparse_scale(A, s)` — scalar multiply
- `transpose(A)` — transpose of CSR / COO

## Screening

- `schwarz_bounds(shell_pair_integrals)` — Cauchy-Schwarz bounds $Q_{pq} = \sqrt{(pq|pq)}$
- `screen_quartets(Q, threshold)` — keep only quartets with $Q_{pq}Q_{rs} > \text{threshold}$
- `density_screen(Q, P, threshold)` — density-weighted screening
- `sparsity_stats(mask)` — density, nnz, per-row nnz distribution

## Dispatch

- `set_backend("auto" | "pytorch" | "nki")` — select compute backend
- `get_backend()` — current backend
- `HAS_NKI` — module-level flag set at import time
