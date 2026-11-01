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
before the matmul (see the [dispatch hierarchy](#nki-dispatch-hierarchy) below) —
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
│   ├── iterative.py     # cg_bsr, cg_bsr_fused, power_iteration_bsr(_fused), chebyshev/richardson
│   ├── multicore.py     # ShardedBSR, ShardedAttention (logical NeuronCores of one chip)
│   └── nki/
│       ├── __init__.py
│       ├── kernels.py       # BSR / screened / attention / fused CG kernels
│       ├── dispatch.py      # path selection, torch.autograd.Function wrappers (XLA/CPU)
│       ├── _native.py       # TorchNeuron custom ops + register_autograd
│       └── _native_attn.py  # trnsparse::bsr_attn / bsr_attn_bwd composite ops
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

### `spmm(csr, B)` — densify-then-GEMM on NKI, `torch.sparse` otherwise

On the `pytorch` backend (and under `auto` without NKI) this routes
through `torch.sparse_csr_tensor` operations, benchmarked within 2× of
scipy at typical shapes. Under `nki`, and under `auto` when NKI is
available, A is densified on the host and the v0.2.0 dense GEMM kernel
runs on the Tensor Engine (`trnsparse::spmm_dense` on the native path). It
pays the full `M × K` cost, so on Trn2 it stays behind dense aten and far
behind `torch.sparse` on CPU at low density ([Benchmarks](benchmarks.md)):
for repeated products on the device, use BSR. SpMV stays on the
`torch.sparse` path (densified on the host when `x` is on a device).

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

### Static structure on the native path (Trn2/Trn3)

TorchNeuron has no device sparse CSR, so the structure stays on the host
(#30). The native path turns that into an advantage: `bsr_spmm` and
block-sparse attention pass `block_row_ptrs` / `block_col_indices` to the
kernel as compile-time constants (`trnsparse::bsr_spmm_static`,
`attn_fwd_static`, `attn_bwd_static`). The trace unrolls exactly the stored
blocks with static DMA addresses: no `(M_tiles, K_max)` padding (one dense
global-token row no longer makes every row cost `K_max` blocks), no gathered
copies of B / K / V, and the attention softmax statistics stay on chip. The
cost is one compile per distinct structure (seconds), which a reused pattern
amortizes. `TRNSPARSE_STATIC_STRUCTURE=0` restores the gather path. Measured
crossovers against dense aten: `docs/benchmarks.md`.

Under LNC=2 (the Trn2 default) NKI traces each kernel once per physical
core, with `program_id` 0 and 1 as Python ints. The static kernels split
their output tiles between the two traces, balanced by stored blocks:

- SpMM: block rows and 512-column tiles.
- Attention forward: block rows.
- Attention backward: one pool of dQ block rows and dK/dV block columns.

Each output tile has exactly one writer. `TRNSPARSE_CORE_SPLIT=0` turns the
split off, for A/B measurements.

The static kernels take fp32 or bf16 operands. When every operand is bf16,
the matmuls run in bf16 and accumulate in fp32 PSUM. Scores, softmax
statistics and score-space arithmetic stay fp32; only P and dS are rounded
to bf16 for their matmuls. Outputs come back in bf16, and the attention
statistics (`row_max`, `row_denom`) are always fp32. Mixed fp32/bf16
operands run in fp32.

## Formats

- **CSR**: row pointer + column indices + values. Preferred for construction,
  interop with scipy/PyTorch, and SpMV.
- **COO**: three parallel 1-D tensors. Preferred for construction and permutation.
- **BSR** (`block_size=128`): stacked 128×128 blocks + block-level CSR pattern.
  Every block maps to one `nc_matmul`; zero gather overhead. Other block
  sizes work everywhere on the PyTorch path. The NKI kernels take 128 only, so
  `bsr_spmm` and `block_sparse_attention_tiled` send any other block size
  straight to PyTorch, or raise `NotImplementedError` under
  `TRNSPARSE_REQUIRE_NKI=1`.

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

- **BSR block size.** The NKI kernels take `block_size=128` only. Other block
  sizes run on the PyTorch path under `auto` and `nki` (correct, no NKI, no
  failed compile since 0.12.0), and raise `NotImplementedError` under
  `TRNSPARSE_REQUIRE_NKI=1`. `cg_bsr_fused` / `prepare_cg_bsr` raise
  `ValueError` for any other block size.
- **Row-bucketing CSR** ([#15](https://github.com/trnsci/trnsparse/issues/15)) — parked.
  Requires NKI indirect-DMA gather (not exposed as of NKI 0.3.0). The CSR PyTorch
  fallback is within 2× of scipy; that's the current story.
- **Fused tile-level attention scores** ([#25](https://github.com/trnsci/trnsparse/issues/25)) —
  shipped: the NKI kernel pair in v0.4.4, the NKI backward in 0.5.1, and
  K-tiling for `head_dim` a multiple of 128 (256, 512) in 0.6.0.
  `block_sparse_attention_tiled` routes through NKI on the `nki` backend; the
  PyTorch two-pass reference remains on the `pytorch` backend. `seq_len` must
  be a multiple of the mask's block size (`ValueError` since 0.11.0). bf16
  inputs raise on the gather kernels used for `head_dim > 128`
  ([#49](https://github.com/trnsci/trnsparse/issues/49)); the static path
  (`head_dim ≤ 128`) takes bf16.
- **Fused iterative solvers** ([#22](https://github.com/trnsci/trnsparse/issues/22)) —
  shipped in 0.11.0: `cg_bsr_fused` runs CG or block-Jacobi PCG in one launch
  with A SBUF-resident and a device-side `nl.while_loop` convergence check, and
  `power_iteration_bsr_fused` uses the same kernel. See
  [Iterative solvers](iterative_solvers.md). `chebyshev_bsr` and
  `richardson_bsr` (v0.4.3) remain host-driven loops over `bsr_spmm`.
- **Multi-chip sharded BSR** ([#16](https://github.com/trnsci/trnsparse/issues/16)) —
  across the logical NeuronCores of one Trn2 chip, `ShardedBSR` and
  `ShardedAttention` shipped in 0.10.0 (`trnsparse.multicore`). Sharding across
  chips is still gated on suite-level multi-chip collectives.
- **SpMV stays PyTorch.** Single output column on the Tensor Engine doesn't amortize
  compile + dispatch overhead.
