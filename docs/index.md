# trnsparse

Sparse matrix operations for AWS Trainium via NKI.

A cuSPARSE-equivalent for Trainium: CSR/COO formats, SpMV, SpMM, and integral screening for sparse scientific computing.

**Primary use case:** Schwarz-screened Fock builds for large-basis quantum chemistry. At >3000 basis functions, >99% of shell quartets screen to zero. Storing and operating on the integral tensor in dense format wastes both memory and compute. `trnsparse` makes the sparsity explicit.

## Install

```bash
pip install trnsparse

# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trnsparse[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trnsparse[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

See [Installation](installation.md) for the two lineages and where sparse
structure lives on the native `neuron` device.

## Quick example

```python
import torch
import trnsparse

dense = torch.randn(1024, 1024)
dense[dense.abs() < 0.5] = 0.0

A = trnsparse.from_dense(dense)
x = torch.randn(1024)
y = trnsparse.spmv(A, x)
```

## Status

Alpha (0.x). CSR/COO are the construction and interop formats; `BSRMatrix`
at `block_size=128` is the NKI compute format. BSR SpMM, screened SpMM,
block-sparse attention (forward and backward), fused CG / PCG
(`cg_bsr_fused`) and `ShardedBSR` across the logical NeuronCores of a chip
are validated on Trn2 with native TorchNeuron (Neuron SDK 2.33). Without
Neuron hardware every API runs on the PyTorch path. See
[Benchmarks](benchmarks.md) for measured crossovers against dense aten.

Part of the [trnsci](https://github.com/trnsci/trnsci) scientific computing suite.
