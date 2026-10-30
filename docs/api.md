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
