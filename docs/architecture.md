# Architecture

## Why BSR is Trainium-native

Trainium's Tensor Engine is a 128×128 systolic array. `nisa.nc_matmul`
moves a 128-partition stationary tile against a moving operand of up to
128 × 512, all on-chip. The natural "unit of sparse work" on this
architecture is therefore not a single nonzero — it's a 128×128 block.

**BSR (block-sparse row)** at `block_size=128` stores a matrix as a list
of these blocks plus a block-level CSR pattern. Every stored block is
already in the shape the systolic array wants: one `nc_matmul` per block,
no gather step, no padding-within-tile waste. The architectural match is
exact.

Contrast with CSR, which stores individual nonzero elements. The v0.2.0
NKI path for CSR materializes the matrix into a dense `(M, K)` tile
before the matmul (see the [SpMM path](#v020-csr-spmm-path) below) —
necessary for correctness, but it pays the full `M × K` cost whether the
matrix is sparse or not. BSR skips this because blocks are already dense.

**Consequence for the library's shape:** CSR remains the construction and
interop format (scipy compatibility, PyTorch's `torch.sparse_csr_tensor`
interop). BSR is the compute format for the NKI path. For matrices with
real block structure — Fock/ERI tensors after Schwarz screening, FEM
stiffness matrices, graph adjacencies, block-sparse attention masks —
BSR is strictly preferred. For truly unstructured sparse (random CSR),
the PyTorch fallback is already within 2× of scipy; NKI adds nothing.

See [Benchmarks](benchmarks.md) for numbers that validate this framing.


```
trnsparse/
├── trnsparse/
│   ├── __init__.py
│   ├── formats.py       # CSRMatrix, COOMatrix, BSRMatrix, from_dense
│   ├── ops.py           # spmv, spmm, bsr_spmm, screened_spmm, ...
│   ├── screening.py     # schwarz_bounds, screen_quartets, density_screen
│   ├── iterative.py     # cg_bsr, power_iteration_bsr, jacobi_preconditioner_bsr
│   └── nki/
│       ├── __init__.py
│       ├── kernels.py   # _bsr_spmm_kernel, _screened_spmm_kernel
│       └── dispatch.py  # torch.autograd.Function wrappers + routing
├── tests/
├── examples/
│   ├── sparse_fock.py             # Schwarz-screened Fock (3 paths + trnblas)
│   ├── pyscf_bridge.py            # Real AO integrals via PySCF
│   └── block_sparse_attention.py  # Block-sparse attention via bsr_spmm
```

## NKI dispatch hierarchy

CSR and COO are interop and PyTorch-fallback formats. BSR is the NKI
compute format. `nki/dispatch.py` exposes `HAS_NKI`,
`set_backend("auto"|"pytorch"|"nki")`, `get_backend()`, and the NKI
entry points.

### `spmm(csr, B)` — PyTorch fallback

Routes through `torch.sparse_csr_tensor` operations. Benchmarked within
2× of scipy at typical shapes; the NKI per-op overhead doesn't pay off
for CSR SpMM until the matrix is large enough that the v0.2.0 CSR kernel
(densify-then-GEMM) would dominate anyway. SpMV also stays on this path.

### `bsr_spmm(bsr, B)` — NKI, `_bsr_spmm_kernel`

One `nisa.nc_matmul` per nonzero block. Host-side preamble pads each
block-row to the same `K_max` so the kernel's `affine_range` bounds are
fixed. `torch.autograd.Function`-wrapped (`_BSRSpMMFunction`); gradcheck
passes at `atol=1e-4` on hardware.

### `screened_spmm(A, diag, B, threshold)` — NKI, fused

Fuses Schwarz bound (outer-product pair bound), threshold mask, and
matmul into one dispatch. Saves ~30–50% vs the unfused
`schwarz_bounds → screen_quartets → from_dense → spmm` flow on
Fock-build-sized inputs.

### Block-sparse attention

`bsr_spmm` applied to the post-softmax attention weight matrix. No new
kernel — a `BSRMatrix` with a local-window or dilated block pattern IS
an attention mask. See [sparse_attention.md](sparse_attention.md).

## Formats

- **CSR**: row pointer + column indices + values. Preferred for construction,
  interop with scipy/PyTorch, and SpMV.
- **COO**: three parallel 1-D tensors. Preferred for construction and permutation.
- **BSR** (`block_size=128`): stacked 128×128 blocks + block-level CSR pattern.
  Every block maps to one `nc_matmul`; zero gather overhead.

All three support `from_dense`, `to_dense`, and interconversion.

## Autograd wrapping

Every NKI kernel lives inside a `torch.autograd.Function` (satisfies
[`trnsci/trnsci#3`](https://github.com/trnsci/trnsci/issues/3)).
Backward passes run at the PyTorch level via block-gradient projection.
Block-selection is non-differentiable by construction; `grad_out` is
routed into exactly the stored blocks. `gradcheck` at `atol=1e-4` is
part of the hardware test matrix for all three NKI kernels.

### Fused screened SpMM (v0.4.0)

`screened_spmm(A, diag_integrals, B, threshold)` fuses the
chemistry-screened SpMM pipeline — Schwarz bound from the diagonal
integrals, pair-bound threshold mask, masked matmul — into a single
NKI kernel. The unfused equivalent does four host passes (sqrt, outer
product, threshold, mask-apply) plus a separate `from_dense` + `spmm`
call; the fused kernel collapses all of that into one dispatch.

Mask semantics: `mask[i,j] = sqrt(|diag[i]|) * sqrt(|diag[j]|) >
sqrt(threshold)`, matching `schwarz_bounds` + `screen_quartets`
composed. No gradient flows back to `diag_integrals` or `threshold` —
the mask is treated as a discrete gate; `grad_A *= mask` and
`grad_B = (A * mask).T @ grad_C`.

Restricted to square A (`M == K`) with 1-D `diag_integrals` in v0.4.0
— the common Fock-build case. Rectangular / asymmetric-bounds
extension is a follow-up if asked for.

## Known limits

- **Row-bucketing CSR** ([#15](https://github.com/trnsci/trnsparse/issues/15)) — parked. Requires NKI indirect-DMA gather (not exposed as of NKI 0.3.0). The CSR PyTorch fallback is within 2× of scipy; that's the current story.
- **Fused tile-level attention scores** ([#25](https://github.com/trnsci/trnsparse/issues/25)) — parked on the same primitive. The current example materializes the full `(seq_len, seq_len)` score matrix before masking.
- **Fused CG/power-iteration kernel** ([#22](https://github.com/trnsci/trnsparse/issues/22)) — parked on `nl.affine_range` lacking `break` and iteration-carried scalar state.
- **Multi-chip sharded BSR** ([#16](https://github.com/trnsci/trnsparse/issues/16)) — gated on suite-level multi-chip collectives.
- **SpMV stays PyTorch.** Single output column on the Tensor Engine doesn't amortize compile + dispatch overhead.
