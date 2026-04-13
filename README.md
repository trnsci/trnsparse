# trnsparse

[![CI](https://github.com/trnsci/trnsparse/actions/workflows/ci.yml/badge.svg)](https://github.com/trnsci/trnsparse/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trnsparse)](https://pypi.org/project/trnsparse/)
[![Python](https://img.shields.io/pypi/pyversions/trnsparse)](https://pypi.org/project/trnsparse/)
[![License](https://img.shields.io/github/license/trnsci/trnsparse)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-trnsci.dev-blue)](https://trnsci.dev/trnsparse/)

Sparse matrix operations for AWS Trainium via NKI.

CSR/COO formats, SpMV, SpMM, and integral screening for sparse scientific computing on Trainium. Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Current phase

trnsparse follows the [trnsci 5-phase roadmap](https://trnsci.dev/roadmap/). Active work is tracked in phase-labeled GitHub issues:

- **[Phase 1 — correctness](https://github.com/trnsci/trnsparse/issues/14)** ✅ v0.2.0: NKI SpMM validated on trn1 via densify-then-GEMM; first `torch.autograd.Function`-wrapped NKI kernel in the suite (see [`trnsci/trnsci#3`](https://github.com/trnsci/trnsci/issues/3)). Benchmarks in [`docs/benchmarks.md`](https://trnsci.dev/trnsparse/benchmarks/).
- **[Phase 3 — perf](https://github.com/trnsci/trnsparse/issues/15)**: nnz-bucketing SpMM, streaming large-sparse, NEFF cache reuse.
- **[Phase 4 — multi-chip](https://github.com/trnsci/trnsparse/issues/16)**: sharded sparse matrices across chips.
- **[Phase 5 — generation](https://github.com/trnsci/trnsparse/issues/17)**: trn2 DMA bandwidth exploitation.

_(No Phase 2 for trnsparse — the precision story is inherited from trnblas.)_

Suite-wide tracker: [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).

## Install

```bash
pip install trnsparse
```

## Usage

```python
import torch
import trnsparse

# Dense → sparse
A = torch.randn(100, 100)
A[torch.abs(A) < 1.0] = 0.0
csr = trnsparse.from_dense(A)

# SpMV: y = A @ x
y = trnsparse.spmv(csr, x, alpha=2.0)

# SpMM: C = A @ B
C = trnsparse.spmm(csr, B)

# Integral screening
Q = trnsparse.schwarz_bounds(diagonal_integrals)
mask = trnsparse.screen_quartets(Q, threshold=1e-10)
stats = trnsparse.sparsity_stats(Q)
```

## Operations

| Operation | Description |
|-----------|-------------|
| `spmv` | Sparse × dense vector |
| `spmm` | Sparse × dense matrix |
| `spmv_symmetric` | Symmetric SpMV (half storage) |
| `sparse_add` | C = αA + βB |
| `sparse_scale` | B = αA |
| `sparse_transpose` | A^T |
| `schwarz_bounds` | Schwarz screening bounds |
| `screen_quartets` | Shell quartet significance mask |
| `density_screen` | Density-weighted screening |

## License

Apache 2.0 — Copyright 2026 Scott Friedman
