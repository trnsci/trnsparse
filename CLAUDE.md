# trnsparse

Sparse matrix operations for AWS Trainium via NKI.
Part of the trnsci scientific computing suite.

## What This Is

A cuSPARSE-equivalent for Trainium. CSR/COO formats, SpMV, SpMM,
and integral screening for sparse scientific computing.

**Primary use case:** Schwarz-screened Fock builds for DF-MP2 quantum chemistry.
At >3000 basis functions, >99% of shell quartets screen to zero.
Storing and operating on the integral tensor in dense format wastes
both memory and compute. trnsparse makes the sparsity explicit.

## Architecture

```
trnsparse/
├── trnsparse/
│   ├── __init__.py
│   ├── formats.py       # CSRMatrix, COOMatrix, conversions, from_dense
│   ├── ops.py           # spmv, spmm, spmv_symmetric, add, scale, transpose
│   ├── screening.py     # schwarz_bounds, screen_quartets, density_screen
│   └── nki/
│       ├── __init__.py
│       └── dispatch.py  # Gather-matmul-scatter pattern for SpMM
├── tests/
│   ├── test_formats.py  # CSR/COO construction, roundtrips
│   ├── test_ops.py      # SpMV, SpMM vs dense reference
│   └── test_screening.py
├── examples/
│   └── sparse_fock.py   # Screened Fock build demo
```

## NKI SpMM Strategy

**v0.2.0 (current):** SpMM materializes the CSR into a dense `(M, K)` tile
and runs stationary-tile-reuse GEMM on the Tensor Engine. Validated on
trn1; forward + `torch.autograd.Function` backward both pass parity +
`gradcheck` at `atol=1e-4`. The dense materialization means no sparsity
advantage yet — NKI is slower than the PyTorch fallback at small sparse
sizes (see `docs/benchmarks.md`).

**v0.3.0 (planned, #15):** gather-matmul-scatter with row-bucketing.
For each bucket of rows with similar nnz:
1. **DMA engine**: gather non-zero column indices into dense SBUF tiles.
2. **Tensor Engine**: matmul the tile against B columns.
3. **DMA engine**: scatter results back to output rows.

Same pattern used in sparse attention. Uniform nnz maps cleanly to fixed-
size tiles; variable nnz is handled by bucketing rows by nnz quantile so
each bucket pads only within itself.

## Dependencies

- `torch>=2.1`, `numpy>=1.24`
- `neuronxcc` (optional)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
python examples/sparse_fock.py --demo
```
