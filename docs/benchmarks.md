# Benchmarks

Performance results for trnsparse — CSR/COO conversion, SpMV, SpMM, and Schwarz screening — comparing the PyTorch CPU fallback and NKI Trainium path.

## Status

Baseline PyTorch-fallback numbers run on every CI build. NKI numbers are pending on-hardware validation on trn1 / trn2 — the gather-matmul-scatter SpMM kernel is scaffolded but not yet validated. See [AWS Setup](aws_setup.md) for the on-hardware CI flow.

## Reproducing locally

```bash
pytest benchmarks/ --benchmark-only
```

## Results table (placeholder)

| Op | Shape / nnz | PyTorch (CPU) | NKI (Trainium) | Speedup |
|---|---|---|---|---|
| spmv (CSR) | 10k × 10k / 1% | TBD | TBD | TBD |
| spmv (CSR) | 100k × 100k / 0.1% | TBD | TBD | TBD |
| spmm (CSR) | 10k × 10k / 1%, N=128 | TBD | TBD | TBD |
| schwarz_bounds | 3000 shell pairs | TBD | TBD | TBD |

Numbers will be populated once the NKI SpMM kernel validates on trn1 / trn2.
