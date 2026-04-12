# trnsparse

Sparse matrix operations for AWS Trainium via NKI.
Part of the trn-* scientific computing suite by Playground Logic.

## What This Is

A cuSPARSE-equivalent for Trainium. CSR/COO formats, SpMV, SpMM,
and integral screening for sparse scientific computing.

**Primary use case:** Schwarz-screened Fock builds for Janesko/TCU.
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

SpMM on Trainium uses a gather-matmul-scatter pattern:
1. **DMA engine**: gather non-zero column indices into dense SBUF tiles
2. **Tensor Engine**: matmul the dense tile against B columns
3. **DMA engine**: scatter results back to output rows

This is the same pattern used in sparse attention. The efficiency depends
on the nnz distribution per row — uniform nnz maps cleanly to fixed-size
tiles; highly variable nnz needs row-bucketing.

## Dependencies

- `torch>=2.1`, `numpy>=1.24`
- `neuronxcc` (optional)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
python examples/sparse_fock.py --demo
```
