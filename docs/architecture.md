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

`nki/dispatch.py` exposes `HAS_NKI`, `set_backend("auto"|"pytorch"|"nki")`, `get_backend()`, and the NKI entry points. In v0.2.0, `spmm` routes through `_use_nki()` and calls `_spmm_dense_kernel` on the Tensor Engine. `spmv`, `spmv_symmetric`, and screening still run the PyTorch path (single-column NKI matmul doesn't pay off).

### v0.2.0 SpMM path

Forward — `_SpMMFunction.forward`:

1. Materialize the CSR into a dense `(M, K)` tile (host-side).
2. Pad `M`, `K` up to 128-multiples and `N` up to a 512-multiple (only when `N > 512`).
3. Move padded A and B to the XLA device; dispatch `_spmm_dense_kernel`.
4. Slice the result back to the caller's `(M, N)`.

The kernel is the trnblas GEMM pattern: stationary A-tile on the systolic array, streaming B tiles via `nisa.nc_matmul`, PSUM accumulation across K-tiles, one store per output tile.

Backward — `_SpMMFunction.backward`, PyTorch-level:

- `dL/dA_dense = dL/dC @ Bᵀ` (projects back through the `to_dense()` graph onto the original CSR values)
- `dL/dB = A_denseᵀ @ dL/dC`

This wrapping satisfies [`trnsci/trnsci#3`](https://github.com/trnsci/trnsci/issues/3) — the suite-wide requirement that every NKI kernel live inside a `torch.autograd.Function` so training-time `loss.backward()` works. `torch.autograd.gradcheck` on small inputs is part of the hardware test matrix.

### Known limits (v0.2.0)

- **No sparsity exploitation.** Materialize-then-GEMM pays the full `M × K` cost. Row-bucketing is the v0.3.0 ([#15](https://github.com/trnsci/trnsparse/issues/15)) Phase 3 story. See [Benchmarks](benchmarks.md) for where NKI sits today vs scipy / torch.sparse.
- **SpMV stays PyTorch.** A single output column on the Tensor Engine doesn't justify the compile + dispatch overhead.
