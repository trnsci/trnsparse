# trnsparse

[![CI](https://github.com/trnsci/trnsparse/actions/workflows/ci.yml/badge.svg)](https://github.com/trnsci/trnsparse/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/trnsci/trnsparse/graph/badge.svg)](https://codecov.io/gh/trnsci/trnsparse)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/trnsparse)](https://pypi.org/project/trnsparse/)
[![Python](https://img.shields.io/pypi/pyversions/trnsparse)](https://pypi.org/project/trnsparse/)
[![License](https://img.shields.io/github/license/trnsci/trnsparse)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-trnsci.dev-blue)](https://trnsci.dev/trnsparse/)

Sparse matrix operations for AWS Trainium via NKI.

CSR/COO formats, SpMV, SpMM, and integral screening for sparse scientific computing on Trainium. Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Current phase

trnsparse follows the [trnsci 5-phase roadmap](https://trnsci.dev/roadmap/). Active work is tracked in phase-labeled GitHub issues:

- **[Phase 1 — correctness](https://github.com/trnsci/trnsparse/issues/14)** ✅ v0.2.0: NKI SpMM validated on trn1 via densify-then-GEMM (see [`trnsci/trnsci#3`](https://github.com/trnsci/trnsci/issues/3)).
- **v0.3.0** ✅ [`BSRMatrix`](https://trnsci.dev/trnsparse/architecture/) — Trainium-native 128×128 block-sparse format; `bsr_spmm` NKI kernel. CSR becomes interop; BSR is the compute path.
- **v0.3.2** ✅ `cg_bsr`, `power_iteration_bsr` — iterative solvers over BSR (Python loop; fused kernel gated on NKI capability).
- **v0.4.0** ✅ `screened_spmm` — fused Schwarz-screened SpMM in one NKI dispatch.
- **v0.4.2** ✅ Block-sparse attention — `BSRMatrix` + `bsr_spmm` as the primitive; `examples/block_sparse_attention.py` + [`docs/sparse_attention.md`](https://trnsci.dev/trnsparse/sparse_attention/).
- **v0.4.3** ✅ Architecture-friendly alternatives: `chebyshev_bsr` / `richardson_bsr` (fixed-K solvers, no inner products); `block_sparse_attention_tiled` (two-pass, no O(seq²) intermediate).
- **[Phase 3 — perf](https://github.com/trnsci/trnsparse/issues/15)**: nnz-bucketing, NKI attention kernel pair — parked on NKI indirect DMA gather / hardware validation.
- **[Phase 4 — multi-chip](https://github.com/trnsci/trnsparse/issues/16)**: sharded BSR across NeuronCores.
- **[Phase 5 — generation](https://github.com/trnsci/trnsparse/issues/17)**: trn2 DMA bandwidth exploitation.

_(No Phase 2 for trnsparse — precision inherited from trnblas.)_

Suite-wide tracker: [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).

## Install

```bash
pip install trnsparse

# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trnsparse[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trnsparse[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

Kernels dispatch on whichever Neuron PyTorch lineage is installed:

- **TorchNeuron (native)** — Trn2/Trn3, Neuron SDK 2.33+, Python ≥ 3.12,
  `torch` 2.14, `torch-neuronx` 2.14.*. Each kernel is a PyTorch custom op
  (`torch.ops.trnsparse.*`) on the `neuron` device, with its backward
  attached via `register_autograd`; no `torch_xla`.
- **PyTorch/XLA** — Trn1 (and any host on the `torch-neuronx` 2.9 lineage).
  Kernels dispatch through `torch_xla`, unchanged.

**Placement on the native path:** the `neuron` device has no sparse CSR
tensors, so keep sparse *structure* (CSR `row_ptrs` / `col_indices`, BSR
`block_row_ptrs` / `block_col_indices`) on the host and put dense operands
(and BSR `blocks`) on `device="neuron"`. Results come back on the dense
operand's device. See [installation](docs/installation.md#placement-sparse-structure-on-the-host).

Set `TRNSCI_NKI_DISPATCH=native|xla|cpu` (shared by all trnsci libraries)
to force a path. With an editable install (`pip install -e`), set
`NKI_ENABLE_TRACE_CACHE=0`: the NKI kernel cache only versions installed
wheels and can serve stale kernels.

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
| `spmv` | Sparse × dense vector (CSR) |
| `spmm` | Sparse × dense matrix (CSR, PyTorch fallback) |
| `bsr_spmm` | Block-sparse × dense (BSR-native, Tensor Engine) |
| `screened_spmm` | Fused Schwarz-screened matmul (one NKI dispatch) |
| `spmv_symmetric` | Symmetric SpMV (half storage) |
| `sparse_add` | C = αA + βB |
| `sparse_scale` | B = αA |
| `sparse_transpose` | Aᵀ |
| `cg_bsr` | Conjugate Gradient on BSR matrix |
| `chebyshev_bsr` | Fixed-K Chebyshev semi-iteration (no inner products) |
| `richardson_bsr` | Fixed-K Richardson iteration |
| `power_iteration_bsr` | Dominant eigenpair via power iteration |
| `jacobi_preconditioner_bsr` | Diagonal preconditioner for `cg_bsr` |
| `bsr_diagonal` | Extract main diagonal from BSR matrix |
| `block_sparse_attention_tiled` | Two-pass sparse attention, no O(seq²) intermediate |
| `schwarz_bounds` | Schwarz screening bounds |
| `screen_quartets` | Shell quartet significance mask |
| `density_screen` | Density-weighted screening |

## License

Apache 2.0 — Copyright 2026 Scott Friedman


## Disclaimer

trnsci is an **independent open-source project**. It is not sponsored by, endorsed by, or affiliated with Amazon.com, Inc., Amazon Web Services, Inc., or Annapurna Labs Ltd.

"AWS", "Amazon", "Trainium", "Inferentia", "NeuronCore", "Neuron SDK", and related identifiers are trademarks of their respective owners and are used here solely for descriptive and interoperability purposes. Use does not imply endorsement, partnership, or any other relationship.

All work, opinions, analyses, benchmark results, architectural commentary, and editorial judgments in this repository and on [trnsci.dev](https://trnsci.dev) are those of the project's contributors. They do not represent the views, positions, or commitments of Amazon, AWS, or Annapurna Labs.

Feedback directed at the Neuron SDK or Trainium hardware is good-faith ecosystem commentary from independent users. It is not privileged information, is not pre-reviewed by AWS, and should not be read as authoritative about product roadmap, behavior, or quality.

For official AWS guidance, see [aws-neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/) and the [AWS Trainium product page](https://aws.amazon.com/ai/machine-learning/trainium/).
