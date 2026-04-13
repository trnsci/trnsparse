# Architecture

```
trnsparse/
├── trnsparse/
│   ├── __init__.py
│   ├── formats.py       # CSRMatrix, COOMatrix, conversions, from_dense
│   ├── ops.py           # spmv, spmm, spmv_symmetric, add, scale, transpose
│   ├── screening.py     # schwarz_bounds, screen_quartets, density_screen
│   └── nki/
│       ├── __init__.py
│       └── dispatch.py  # Gather-matmul-scatter SpMM kernel
├── tests/
├── examples/
│   └── sparse_fock.py   # Screened Fock build demo
```

## NKI SpMM strategy

SpMM on Trainium uses a gather-matmul-scatter pattern:

1. **DMA engine**: gather non-zero column indices into dense SBUF tiles
2. **Tensor Engine**: matmul the dense tile against B columns
3. **DMA engine**: scatter results back to output rows

This is the same pattern used in sparse attention. The efficiency depends on the nnz distribution per row — uniform nnz maps cleanly to fixed-size tiles; highly variable nnz needs row-bucketing.

## Formats

- **CSR** (compressed sparse row): preferred for SpMV with many right-hand sides and for SpMM. Row pointer + column indices + values.
- **COO** (coordinate): preferred for construction and permutation. Three parallel 1-D tensors.

Conversions `csr_to_coo()` / `coo_to_csr()` are cheap (bucket sort).

## Dispatch

`nki/dispatch.py` exposes `HAS_NKI`, `set_backend("auto"|"pytorch"|"nki")`, and `get_backend()`. The backend selector is scaffolded but not yet wired through the public ops — in v0.1.x, `spmv` / `spmm` / screening always run the PyTorch path regardless of `set_backend`. Routing through `_use_nki()` and the gather-matmul-scatter kernel lands in v0.2.0 once validated on trn1 / trn2 (see [issue #7](https://github.com/trnsci/trnsparse/issues/7) and [#2](https://github.com/trnsci/trnsparse/issues/2)).
