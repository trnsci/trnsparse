# trnsparse

Sparse matrix operations for AWS Trainium via NKI.
Part of the trnsci scientific computing suite.

## What This Is

A cuSPARSE-equivalent for Trainium. CSR/COO formats, SpMV, SpMM,
and integral screening for sparse scientific computing.

**Primary use case:** Schwarz-screened Fock builds for DF-MP2 quantum chemistry.
At >3000 basis functions, >99% of shell quartets screen to zero.
Storing and operating on the integral tensor in dense format wastes
both memory and compute. trnsparse makes the sparsity explicit.

## Architecture

```
trnsparse/
├── trnsparse/
│   ├── __init__.py
│   ├── formats.py       # CSRMatrix, COOMatrix, BSRMatrix, from_dense
│   ├── ops.py           # spmv, spmm, bsr_spmm, screened_spmm, add, scale, transpose
│   ├── screening.py     # schwarz_bounds, screen_quartets, density_screen
│   ├── iterative.py     # cg_bsr, power_iteration_bsr, jacobi_preconditioner_bsr
│   └── nki/
│       ├── __init__.py
│       ├── kernels.py   # _bsr_spmm_kernel, _screened_spmm_kernel
│       └── dispatch.py  # torch.autograd.Function wrappers + NKI/PyTorch routing
├── tests/
├── examples/
│   ├── sparse_fock.py             # Schwarz-screened Fock build (3 paths + trnblas)
│   ├── pyscf_bridge.py            # Real AO integrals via PySCF (optional dep)
│   └── block_sparse_attention.py  # Block-sparse attention via bsr_spmm
```

## NKI compute strategy (v0.4.x current)

**Formats:** CSR/COO are interop and PyTorch-fallback formats.
BSR at `block_size=128` is the NKI compute format — every nonzero block
is already a 128×128 Tensor Engine tile, zero gather overhead.

**Dispatch hierarchy:**
1. `spmm(csr, B)` — PyTorch `torch.sparse_csr_tensor` fallback (within
   2× of scipy; NKI overhead not worth it for CSR at current sizes).
2. `bsr_spmm(bsr, B)` — NKI kernel `_bsr_spmm_kernel`; one `nc_matmul`
   per nonzero block. `torch.autograd.Function`-wrapped; gradcheck passes.
3. `screened_spmm(A, diag, B, threshold)` — fused Schwarz + mask + matmul
   in one NKI dispatch; avoids 4 host passes + CSR build + separate SpMM.
4. `block_sparse_attention(Q, K, V, mask_bsr)` — `bsr_spmm` applied to
   the attention weight matrix; no new kernel, mask is a BSRMatrix.

**Open gaps (NKI capability-gated):** row-bucketing CSR (#15), sharded BSR
across NeuronCores (#16), fused tile-level attention scores (#25).

## Dependencies

- `torch>=2.1`, `numpy>=1.24`
- `neuronxcc` (optional)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
python examples/sparse_fock.py --demo
```

## Blog posts

When asked to draft a blog post for this library for the [trnsci blog](https://trnsci.dev/blog/):

1. Read the editorial brief at [`docs/blog/AUTHOR_BRIEF.md`](https://github.com/trnsci/trnsci/blob/main/docs/blog/AUTHOR_BRIEF.md) in the umbrella repo (trnsci/trnsci). It defines voice (authorless, library-as-subject), stance (architecture-first, transparency-always), and the nine required section headings.

2. Find the prompt block for this library in [`BLOG_PROMPTS.md`](https://github.com/trnsci/trnsci/blob/main/BLOG_PROMPTS.md) at the umbrella repo root. It carries library-specific context and suggested architectural angles.

3. Draft the post following the brief. Open a PR against `trnsci/trnsci` at `docs/blog/posts/<YYYY-MM-DD>-<slug>.md`. Scott (suite director) reviews before merge.

The umbrella repo — not this one — owns the blog. Per-library retrospective posts are unsigned; library is the subject, no byline. See the brief for the full set of rules.
