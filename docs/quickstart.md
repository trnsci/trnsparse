# Quickstart

## Sparse matrix construction

```python
import torch
import trnsparse

# From a dense tensor (zeros dropped)
dense = torch.randn(512, 512)
dense[dense.abs() < 0.5] = 0.0
A = trnsparse.CSRMatrix.from_dense(dense)
print(f"density: {A.nnz / (A.shape[0] * A.shape[1]):.2%}")

# From explicit COO triples
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([1, 2, 0])
vals = torch.tensor([1.0, 2.0, 3.0])
B = trnsparse.COOMatrix(rows, cols, vals, shape=(3, 3))
```

## SpMV and SpMM

```python
x = torch.randn(512)
y = trnsparse.spmv(A, x)

X = torch.randn(512, 128)
Y = trnsparse.spmm(A, X)
```

## Schwarz screening

For a Fock-build with two-electron integrals $(\mu\nu|\rho\sigma)$:

```python
# Shell-pair Schwarz bounds: |(pq|rs)| <= sqrt((pq|pq)(rs|rs))
Q = trnsparse.schwarz_bounds(shell_pair_integrals)
keep = trnsparse.screen_quartets(Q, threshold=1e-10)
```

## Backend selection

```python
trnsparse.set_backend("auto")     # default
trnsparse.set_backend("pytorch")  # force PyTorch fallback
trnsparse.set_backend("nki")      # force NKI (requires Neuron hardware)
```
