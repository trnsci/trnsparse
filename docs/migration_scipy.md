# Migrating from `scipy.sparse`

`scipy.sparse` is the reference sparse-matrix API for the Python
scientific ecosystem. trnsparse's constructor and op names are close but
not identical — this guide covers what stays the same and what changes
when you port code across.

## API mapping

| scipy.sparse | trnsparse | Notes |
|---|---|---|
| `csr_matrix((data, indices, indptr))` | `CSRMatrix(data, indices, indptr, shape)` | Same CSR layout, explicit shape required in trnsparse. |
| `coo_matrix((data, (row, col)))` | `COOMatrix(data, row, col, shape)` | Same COO triple layout. |
| `csr_matrix(dense_array)` | `from_dense(torch_tensor, threshold=0.0)` | Densify → sparsify helper; `threshold` filters below-tolerance values. |
| `csr_matrix(scipy_matrix)` (passthrough) | `from_scipy(sp_matrix)` | Ingests an existing `scipy.sparse` matrix directly — zero-copy where possible. |
| `eye(n, format='csr')` | `eye_sparse(n, dtype=torch.float32)` | Identity CSR. |
| `A @ x` (SpMV) | `spmv(A, x)` | Function call, not operator; supports `alpha` / `beta` scaling. |
| `A @ B` (SpMM) | `spmm(A, B)` | Same; supports `alpha` / `beta`. |
| `A @ x` with symmetric A | `spmv_symmetric(A, x)` | Exploits symmetry for ~2× throughput when you can guarantee it. |
| `alpha * A + beta * B` | `sparse_add(A, B, alpha=alpha, beta=beta)` | Combined scale + add. |
| `alpha * A` | `sparse_scale(A, alpha)` | In-place-style scalar scaling. |
| `A.T` | `sparse_transpose(A)` | Returns a new CSRMatrix. |
| `A.getnnz(axis=1)` | `nnz_per_row(A)` | Returns a tensor, not a numpy array. |

## Schwarz / density screening

trnsparse adds quantum-chemistry-flavored screening helpers that don't
have a direct `scipy.sparse` equivalent:

| trnsparse | Purpose |
|---|---|
| `schwarz_bounds(diagonal_integrals)` | Bound `(ij|kl)` integrals via `|(ij|ij)|^{1/2}`. |
| `screen_quartets(...)` | Apply Schwarz + density threshold to an ERI quartet list. |
| `density_screen(D, threshold)` | Drop rows/cols of density matrix below threshold. |
| `sparsity_stats(Q, threshold)` | Report nnz, mean/max row density, etc. |

## What's the same

- **CSR layout bit-for-bit.** The `indptr`, `indices`, `data` arrays
  match `scipy.sparse.csr_matrix` exactly — same dtype conventions
  (`int32`/`int64` for indices, user-specified for data) and the same
  zero-indexed, sorted-column-within-row invariants. You can round-trip
  via `from_scipy` / `.to_scipy()` without layout surprises.
- **Matrix-vector semantics.** `spmv(A, x)` produces the same output as
  `A @ x` in scipy (modulo floating-point summation order). Symmetric
  and triangular variants use the same conventions.
- **Broadcasting rules for `spmm`.** `spmm(A, B)` treats `B` as dense
  `(k, n)` and returns dense `(m, n)` — same as scipy's `A @ B_dense`.

## What changes

**Backend.** trnsparse uses `torch.Tensor` throughout instead of
`numpy.ndarray`. `CSRMatrix.data`, `.indices`, `.indptr` are all
`torch.Tensor`. Device placement follows torch conventions:
`A.data.to('cuda')` or similar. On Trainium, use `set_backend("nki")`
to dispatch `spmv` / `spmm` to the on-device kernel path.

**Constructor signature.** trnsparse requires an explicit `shape`
argument; scipy infers it from the indices when possible. Be explicit:

```python
# scipy
A = csr_matrix((data, indices, indptr))              # shape inferred

# trnsparse
A = CSRMatrix(data, indices, indptr, shape=(m, n))   # shape required
```

**Operator access.** scipy overloads `@`, `+`, `*` on its matrix
classes; trnsparse exposes everything as module-level functions (`spmv`,
`sparse_add`, `sparse_scale`). This keeps the API explicit and makes
backend dispatch easier to reason about.

**Missing ops.** Not yet implemented in trnsparse (workarounds noted):

| Not yet | Workaround |
|---|---|
| `A.multiply(B)` (elementwise) | Densify one side: `torch.sparse.mm(A.to_torch_sparse(), B)` with dense mask. |
| `A.power(n)` | Materialize via `from_dense((A.to_dense() ** n).to(...))`. |
| Slicing (`A[i:j, :]`) | Extract on the CPU side via `from_scipy(A.to_scipy()[i:j])`. |
| Elementwise division | Not supported — use dense path. |

Track these in the [issue tracker](https://github.com/trnsci/trnsparse/issues)
if they block a port.

**Dtype policy.** trnsparse respects the input `data` tensor's dtype.
scipy defaults to `float64`; if you want `float32` throughout, cast
before construction:

```python
A = CSRMatrix(data.to(torch.float32), indices, indptr, shape)
```

## Example: porting a CG solver

```python
# Before (scipy)
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

A = csr_matrix(...)
x, info = cg(A, b, tol=1e-6)

# After (trnsparse + trnsolver)
import torch
import trnsparse
import trnsolver

A = trnsparse.from_scipy(scipy_matrix)            # zero-copy ingestion
b = torch.as_tensor(b_numpy, dtype=torch.float32)
x = trnsolver.cg(A, b, tol=1e-6)                  # uses trnsparse.spmv
```

The hot path (`spmv` inside CG) dispatches to NKI when
`trnsparse.set_backend("nki")` is active, without any CG code changes.
