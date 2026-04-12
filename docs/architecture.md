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

All compute entry points (`spmv`, `spmm`, screening) route through `nki/dispatch.py`, which picks `pytorch` or `nki` based on `set_backend(...)` and hardware detection.
