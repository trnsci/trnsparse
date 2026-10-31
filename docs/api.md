# API Reference

## Formats

- `CSRMatrix(indptr, indices, data, shape)` — compressed sparse row
  - `.from_dense(dense, threshold=0)` — construct by thresholding a dense tensor
  - `.to_dense()` — materialize back to a dense `torch.Tensor`
  - `.transpose()` — transpose to CSR (via CSC)
  - `.nnz`, `.shape`, `.dtype`, `.device`
- `COOMatrix(rows, cols, vals, shape)` — coordinate format
  - `.to_csr()` / `CSRMatrix.to_coo()` — roundtrip conversions

## Ops

- `spmv(A, x)` — sparse matrix × dense vector
- `spmv_symmetric(A, x)` — for symmetric A; skips duplicate work
- `spmm(A, B)` — sparse matrix × dense matrix (gather-matmul-scatter on NKI)
- `sparse_add(A, B, alpha=1.0, beta=1.0)` — `α A + β B`
- `sparse_scale(A, s)` — scalar multiply
- `transpose(A)` — transpose of CSR / COO

## Screening

- `schwarz_bounds(shell_pair_integrals)` — Cauchy-Schwarz bounds $Q_{pq} = \sqrt{(pq|pq)}$
- `screen_quartets(Q, threshold)` — keep only quartets with $Q_{pq}Q_{rs} > \text{threshold}$
- `density_screen(Q, P, threshold)` — density-weighted screening
- `sparsity_stats(mask)` — density, nnz, per-row nnz distribution

## Dispatch

- `set_backend("auto" | "pytorch" | "nki")` — select compute backend
- `get_backend()` — current backend
- `HAS_NKI` — module-level flag set at import time

On Neuron hardware the NKI backend runs on one of two `torch-neuronx`
lineages, detected on the first kernel launch (force with
`TRNSCI_NKI_DISPATCH=native|xla|cpu`):

- **Native (TorchNeuron, Trn2/Trn3)** — each kernel is a custom op on
  `device="neuron"`: `torch.ops.trnsparse.spmm_dense`, `bsr_spmm`,
  `screened_spmm`, `attn_stats`, `attn_out`, `attn_bwd_dq`,
  `attn_bwd_dkdv`, and the composite `bsr_attn` / `bsr_attn_bwd`. Backwards
  are attached with `torch.library.register_autograd`; `spmm` also traces
  under `torch.compile(backend="neuron", fullgraph=True)`. The ops are
  registered only once this path is selected.
- **XLA (Trn1)** — kernels run on the `torch_xla` device, as before.

**Placement on the native path:** sparse structure (CSR `row_ptrs` /
`col_indices` / `values`, BSR `block_row_ptrs` / `block_col_indices`) stays
on the host; dense operands and BSR `blocks` may be on `neuron`; results are
on the dense operand's device. See
[Installation](installation.md#placement-sparse-structure-on-the-host).

## Multi-core (`trnsparse.multicore`, native path)

A Trn2 chip at LNC=2 has four logical NeuronCores, which TorchNeuron
exposes as `neuron:0` … `neuron:3` (`NEURON_RT_VISIBLE_CORES=0-3`). Each
static kernel already splits its work across the two physical cores of one
logical core; these classes split it across logical cores.

- `logical_cores(n=None)`: the visible logical cores as devices.
- `partition_block_rows(row_ptrs, n_parts)`: block rows split into groups
  balanced by stored blocks (the same greedy as the in-kernel split).
- `ShardedBSR(A, devices=None, pretransposed=True)`: A split by block rows,
  with each core's blocks resident on that core.
  - `.replicate(B)` copies B to every core.
  - `.spmm(B_replicas)` returns one output per core, issuing one kernel
    launch per core without waiting.
  - `.gather(outs)` assembles C on the host.
  - `pretransposed=True` runs the pre-transposed kernel (no in-kernel
    transpose, fewer DMAs, and B tiles shared by groups of up to 8 block
    rows). Gradients reach each core's B replica and block copy.
- `ShardedAttention(mask, devices=None)`: a single-head block-sparse
  attention forward, split by query block rows. Use `.shard_rows(Q)`,
  `.replicate_kv(K, V)`, `.forward(Qts, KVs)` and `.gather(outs)`. It is
  forward only. For several heads, put whole heads on different cores and
  call `block_sparse_attention_tiled` there; that path has autograd.

TorchNeuron does not copy tensors between logical cores. A `neuron:0`
tensor's `.to("neuron:1")` comes back still on core 0, and `copy_` across
cores raises. So per-core operands are built on the host and copied once,
and results return through the host. The kernel ops run on the logical core
of their operands, and operands on two different cores are an error.
