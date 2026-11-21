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
- **v0.3.2** ✅ `cg_bsr`, `power_iteration_bsr` — iterative solvers over BSR (Python loop; the fused kernel shipped in 0.11.0, below).
- **v0.4.0** ✅ `screened_spmm` — fused Schwarz-screened SpMM in one NKI dispatch.
- **v0.4.2** ✅ Block-sparse attention — `BSRMatrix` + `bsr_spmm` as the primitive; `examples/block_sparse_attention.py` + [`docs/sparse_attention.md`](https://trnsci.dev/trnsparse/sparse_attention/).
- **v0.4.3** ✅ Architecture-friendly alternatives: `chebyshev_bsr` / `richardson_bsr` (fixed-K solvers, no inner products); `block_sparse_attention_tiled` (two-pass, no O(seq²) intermediate).
- **v0.4.4–v0.6.0** ✅ NKI attention kernel pair, NKI backward, `head_dim` multiples of 128 (K-tiling).
- **v0.8.0–v0.9.0** ✅ Native TorchNeuron path (Trn2/Trn3): static-structure BSR SpMM and attention kernels, split across both LNC=2 cores, bf16 operands on fp32 PSUM.
- **v0.10.0** ✅ `ShardedBSR` / `ShardedAttention` across the logical NeuronCores of one chip; pre-transposed BSR SpMM beats dense aten up to ~82–98% block density on one core.
- **v0.11.0** ✅ `cg_bsr_fused` / `prepare_cg_bsr` — one-launch CG and block-Jacobi PCG with A in SBUF (5–56× faster than CPU on Trn2), `power_iteration_bsr_fused`.
- **[Phase 3 — perf](https://github.com/trnsci/trnsparse/issues/15)**: nnz-bucketing CSR, streaming large-sparse — parked on NKI indirect DMA gather.
- **[Phase 4 — multi-chip](https://github.com/trnsci/trnsparse/issues/16)**: sharded BSR across chips (within one chip: `ShardedBSR`, v0.10.0).
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

### Known platform issue: NKI trace cache and kernel factories

`import torch_neuronx` turns on the NKI trace cache, and the cache key ignores
constants captured in a kernel's closure: two `@nki.jit` kernels built by one
factory, differing only in a captured value, share one trace, and the second
silently returns the first's results
([aws-neuron/aws-neuron-sdk#1454](https://github.com/aws-neuron/aws-neuron-sdk/issues/1454)).
trnsparse's kernels do not rely on closure constants (or on module globals that
can differ between processes): every value that specialises a kernel is a
kernel argument, and `tests/test_trace_cache_safety.py` enforces this. If you
write your own NKI kernels with a factory, set `NKI_ENABLE_TRACE_CACHE=0`
before importing `torch_neuronx`.

## Known issues in earlier releases

Silent wrong results fixed in the 2026-10-31 to 2026-11-04 releases. If you
use an affected version, upgrade. Details are in the [CHANGELOG](CHANGELOG.md).

| Bug | Affected | Fixed in |
|---|---|---|
| `block_sparse_attention_tiled` returned zero rows for a partial last block (PyTorch path) (#44) | 0.6.0–0.10.0 | 0.11.0 |
| `block_sparse_attention_tiled` ignored a custom `scale` on the NKI paths (#47) | 0.6.0–0.11.0 | 0.12.0 |
| `spmv_symmetric` ignored `uplo` and used both triangles (#52) | through 0.12.0 | 0.12.1 |
| `COOMatrix.to_dense` dropped duplicate entries instead of summing them (#56) | 0.1.1–0.12.1 | 0.13.0 |
| `from_scipy` dropped the imaginary part of complex matrices (#58) | 0.1.1–0.12.1 | 0.13.0 |
| Second derivatives of `block_sparse_attention_tiled` were wrong (#65) | 0.6.0–0.12.1 | 0.13.0 |
| `spmv` / `spmm` accepted a `y` / `C` that broadcast the output to a larger shape (#67) | 0.1.1–0.12.1 | 0.13.0 |
| `chebyshev_coeffs` / `chebyshev_bsr` were not the Chebyshev polynomial for K >= 2 (#73) | 0.4.3–0.12.1 | 0.13.0 |
| `bsr_diagonal` returned length m (zero-padded) for a tall matrix (#74) | 0.3.2–0.12.1 | 0.13.0 |
| `bsr_spmm` dropped the imaginary part of complex operands on the NKI backend (#78) | 0.8.0–0.12.1 | 0.13.0 |
| `spmv` / `spmm` returned float32 for integer operands on the device (#79) | through 0.12.1 (device CSR path) | 0.13.0 |
| `block_sparse_attention_tiled` accepted a mask smaller than `seq_len` on the NKI path (#82) | 0.10.0–0.12.1 | 0.13.0 |
| Integer, bool and complex operands reached the fp32/bf16 NKI kernels: the static attention kernels dropped the imaginary part of complex Q/K/V, and the static `bsr_spmm` and attention kernels cast integers to fp32 and back (trnsci/trnsci#35) | every release with NKI kernels through 0.13.0, NKI / `auto` backends | 0.14.0 |
| `from_scipy` downcast float64 to float32 (~3e-8 rel. err), int64 to float32 (exact only up to 2^24) and complex128 to complex64 (#69) | 0.1.1–0.13.0 | 0.14.0 |
| `cg_bsr` gave wrong gradients: autograd differentiated a frozen-coefficient polynomial, not the solve (`b.grad` ~20% off) (#70) | 0.3.2–0.13.0 | 0.14.0 |
| `power_iteration_bsr` gave wrong gradients (frozen normalisation; `v0` got a spurious gradient) (#96) | 0.3.2–0.13.0 | 0.14.0 |
| `cg_bsr` reported `iters = max_iter` after a breakdown, e.g. 500 for a one-step solve (#71) | 0.3.2–0.13.0 | 0.14.0 |
| `cg_bsr_fused` / `power_iteration_bsr_fused` dropped the imaginary part of complex input and truncated integer / bool `b`, on every backend (#85) | 0.11.0–0.13.0 | 0.14.0 |

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
| `spmm` | Sparse × dense matrix (CSR; densify-then-GEMM on NKI, `torch.sparse` otherwise) |
| `bsr_spmm` | Block-sparse × dense (BSR-native, Tensor Engine; NKI at `block_size=128`, other sizes run on PyTorch) |
| `screened_spmm` | Fused Schwarz-screened matmul (one NKI dispatch) |
| `spmv_symmetric` | Symmetric SpMV; references only the `uplo` triangle, as in BLAS `symv` (0.12.1) |
| `sparse_add` | C = αA + βB |
| `sparse_scale` | B = αA |
| `sparse_transpose` | Aᵀ |
| `cg_bsr` | Conjugate Gradient on BSR matrix |
| `cg_bsr_fused` / `prepare_cg_bsr` | One-launch CG / block-Jacobi PCG, A in SBUF, vector-stationary TensorE matvec, device-side convergence check, multi-RHS, opt-in bf16 A |
| `chebyshev_bsr` | Fixed-K Chebyshev semi-iteration (no inner products) |
| `richardson_bsr` | Fixed-K Richardson iteration |
| `power_iteration_bsr` | Dominant eigenpair via power iteration |
| `power_iteration_bsr_fused` | Power iteration in one launch (the fused CG kernel's machinery) |
| `jacobi_preconditioner_bsr` | Diagonal preconditioner for `cg_bsr` |
| `ShardedBSR` / `ShardedAttention` | BSR SpMM / attention forward split across logical NeuronCores (`trnsparse.multicore`) |
| `bsr_diagonal` | Extract main diagonal from BSR matrix |
| `block_sparse_attention_tiled` | Two-pass sparse attention, no O(seq²) intermediate, autograd; `seq_len` must be a multiple of the block size |
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
