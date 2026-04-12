# trnsparse

Sparse matrix operations for AWS Trainium via NKI.

A cuSPARSE-equivalent for Trainium: CSR/COO formats, SpMV, SpMM, and integral screening for sparse scientific computing.

**Primary use case:** Schwarz-screened Fock builds for large-basis quantum chemistry. At >3000 basis functions, >99% of shell quartets screen to zero. Storing and operating on the integral tensor in dense format wastes both memory and compute. `trnsparse` makes the sparsity explicit.

## Install

```bash
pip install trnsparse
pip install trnsparse[neuron]   # on Neuron hardware
```

## Quick example

```python
import torch
import trnsparse

dense = torch.randn(1024, 1024)
dense[dense.abs() < 0.5] = 0.0

A = trnsparse.CSRMatrix.from_dense(dense)
x = torch.randn(1024)
y = trnsparse.spmv(A, x)
```

## Status

Alpha. CSR/COO formats, SpMV/SpMM, and Schwarz screening are functional via PyTorch fallback. NKI gather-matmul-scatter SpMM kernel is scaffolded; on-hardware validation is the next milestone.

Part of the [trnsci](https://github.com/trnsci/trnsci) scientific computing suite.
