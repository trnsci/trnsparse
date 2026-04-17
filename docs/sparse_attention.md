# Block-sparse attention

## The connection

Transformer attention computes `O = softmax(Q @ K.T / √d) @ V`. The expensive
part at long sequence lengths is the `(seq_len, seq_len)` score matrix and the
final `attn_weights @ V` matmul. Sparse attention restricts which positions can
attend to each other, zeroing out large regions of the score matrix. What
remains is a sparse × dense matmul: exactly `bsr_spmm`.

The block size that matters is 128. Trainium's Tensor Engine is a 128-partition
systolic array; `nc_matmul` consumes a 128×K×N tile in one call. A local-window
attention mask with window granularity of 128 tokens has exactly `2w+1` nonzero
128×128 blocks per block-row — each nonzero block maps one-to-one to one
`nc_matmul` call, with no gather overhead. The tile size of the mask and the
tile size of the hardware coincide.

cuSPARSE's BSR is a specialization added on top of CSR. Here it's the other
way: BSR is what the attention mask asks for, and the hardware is built around
that shape.

## Building patterns

### Local window (Longformer-style)

Each token block attends to its `window` nearest neighbors in block space:

```python
import trnsparse

def local_window_mask(seq_len: int, block_size: int, window: int) -> trnsparse.BSRMatrix:
    """BSRMatrix for sliding-window attention."""
    import torch
    n_blocks = seq_len // block_size
    block_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    for i in range(n_blocks):
        lo, hi = max(0, i - window), min(n_blocks, i + window + 1)
        block_mask[i, lo:hi] = True
    mask_dense = block_mask.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return trnsparse.BSRMatrix.from_dense(mask_dense.float(), block_size=block_size)
```

At `seq_len=4096` and `window=2`: `5` nonzero blocks per row, `5/32 ≈ 15.6%`
block density vs 100% for dense attention. Memory for the stored blocks:
`32 rows × 5 blocks × 128 × 128 × 4 bytes ≈ 10 MB` vs 64 MB for the full
float32 attention matrix.

### Dilated sparse

Every `stride`-th block column is attended to. Block `(i, j)` is nonzero
iff `(i - j) % stride == 0`:

```python
def dilated_mask(seq_len: int, block_size: int, stride: int) -> trnsparse.BSRMatrix:
    import torch
    n_blocks = seq_len // block_size
    block_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    for i in range(n_blocks):
        for j in range(n_blocks):
            if (i - j) % stride == 0:
                block_mask[i, j] = True
    mask_dense = block_mask.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return trnsparse.BSRMatrix.from_dense(mask_dense.float(), block_size=block_size)
```

### Global tokens (BigBird-style)

The first `n_global` token-blocks attend to and are attended to by every
block. Remaining block-rows follow a local window:

```python
def global_token_mask(
    seq_len: int, block_size: int, window: int, n_global: int
) -> trnsparse.BSRMatrix:
    import torch
    n_blocks = seq_len // block_size
    block_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    block_mask[:n_global, :] = True   # global rows attend everywhere
    block_mask[:, :n_global] = True   # global columns attended by all
    for i in range(n_global, n_blocks):
        lo, hi = max(0, i - window), min(n_blocks, i + window + 1)
        block_mask[i, lo:hi] = True
    mask_dense = block_mask.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return trnsparse.BSRMatrix.from_dense(mask_dense.float(), block_size=block_size)
```

## The computation

Given a `mask_bsr` constructed above:

```python
import torch

def block_sparse_attention(Q, K, V, mask_bsr):
    """Q, K, V: (seq_len, head_dim) → (seq_len, head_dim)."""
    scale = Q.shape[-1] ** -0.5
    scores = (Q @ K.T) * scale

    # Mask out unattended positions before softmax.
    mask_dense = mask_bsr.to_dense().bool()
    masked_scores = scores.masked_fill(~mask_dense, float("-inf"))
    attn_weights = torch.softmax(masked_scores, dim=-1)

    # Positions outside the mask go to zero after softmax; from_dense
    # with threshold=0 will still store them unless they're exactly zero.
    # Use threshold slightly above zero to keep only the attended blocks.
    attn_bsr = trnsparse.BSRMatrix.from_dense(
        attn_weights, block_size=mask_bsr.block_size, threshold=1e-9
    )
    return trnsparse.bsr_spmm(attn_bsr, V)
```

**What this materializes**: the full `(seq_len, seq_len)` score matrix. At
`seq_len=4096` that's 64 MB of float32 even before the matmul. See
[What's next](#whats-next) for the fused-tile path that avoids this.

**Gradients**: `bsr_spmm` is differentiable (see `architecture.md`). The
backward through `block_sparse_attention` works out-of-the-box with
`torch.autograd`; the block-selection step (`from_dense`) is non-differentiable
by construction and is treated as a constant by the autograd graph.

## Block density arithmetic

For `seq_len = S`, `block_size = b`, `window = w` local-window mask:

```
blocks per row   = 2w + 1  (clamped at edges)
block density    = (2w+1) / (S/b)
```

| seq_len | window | n_blocks_per_row | block density |
|--------:|-------:|-----------------:|:--------------|
| 1024    | 2      | 5                | 39.1%         |
| 2048    | 2      | 5                | 19.5%         |
| 4096    | 2      | 5                | 9.8%          |
| 4096    | 4      | 9                | 17.6%         |
| 8192    | 2      | 5                | 4.9%          |

The win is at long sequences. At `seq_len=1024`, a window-2 mask still stores
40% of blocks — dispatch overhead dominates on Trainium at that size. At
`seq_len=8192`, 5% density means 95% of the `nc_matmul` calls are skipped and
the per-block compute becomes the bottleneck, which is where the Tensor Engine
thrives.

## Tiled two-pass path (v0.4.3)

`block_sparse_attention_tiled` eliminates the `O(seq_len²)` score intermediate
without any new NKI primitives. The algorithm uses two passes over the nonzero
blocks:

**Pass 1 — per-block statistics**: For each stored block `(m, ki)`, compute
the score tile `Q[m] @ K[ki].T` and extract its row-wise max (`tile_max`,
shape `(block_size,)`) and stable row-wise exp-sum (`tile_sumexp`). Every
`(m, ki)` pair is independent — no iteration-carried state.

**Host reduction**: For each block-row m, take the max of `tile_max` across all
its nonzero k-blocks (`row_max`) and compute the corrected denominator
`row_denom`. This is O(n_stored_blocks × block_size) work — at seq_len=4096
with window=2, about 160 block pairs × 128 = 20K operations, negligible.

**Pass 2 — stable softmax + accumulation**: Recompute each score tile, apply
`exp(score - row_max) / row_denom`, and accumulate `weights @ V_block` into
the output via a PSUM-style loop.

```python
out = trnsparse.block_sparse_attention_tiled(Q, K, V, mask_bsr)
```

Memory: `n_stored_blocks × block_size × 4` bytes for stats (5 KB at seq_len=512,
window=1) vs `seq_len² × 4` bytes for the naive dense score matrix (1 MB at
seq_len=512, 64 MB at seq_len=4096).

**On CPU**, the Python-level loops make this path slower than the vectorized
naive implementation — that's expected. The value of `block_sparse_attention_tiled`
is algorithmic: it validates the two-pass decomposition that the NKI kernel pair
(`_attn_stats_kernel` + `_attn_out_kernel`) will implement on hardware, where
per-block `nc_matmul` calls amortize the dispatch cost and the O(seq_len²)
allocation is the binding constraint.

The NKI implementation path is documented in
[#25](https://github.com/trnsci/trnsparse/issues/25).

## NKI kernel pair (v0.4.4)

`block_sparse_attention_tiled` now routes through the NKI kernel pair on the
`nki` backend. The `pytorch` backend retains the Python-loop reference path.

`_attn_stats_kernel` (pass 1) and `_attn_out_kernel` (pass 2) implement the
two-pass decomposition in `trnsparse/nki/kernels.py`:

- **Pass 1** (`_attn_stats_kernel`): each `(m, ki)` block pair is independent;
  no carry. Uses `nl.max` and `nl.sum` reductions within the 128×128 SBUF tile.
  Q block is loaded once as a transposed tile (stationary) and reused across all
  ki blocks in the row.
- **Pass 2** (`_attn_out_kernel`): recomputes scores from Q and K to avoid
  storing the full score tensor; uses `row_max` / `row_denom` from the host
  reduction (loaded by static `affine_range` offset — legal). Accumulates
  `weights @ V` into a PSUM tile that spans all ki blocks.

**Constraint:** `head_dim ≤ 128` in v0.4.4 (`nc_matmul` partition limit).
`head_dim=256` requires K-tiling and is a follow-up.

A runnable reference for both the naive and tiled paths is in
[`examples/block_sparse_attention.py`](https://github.com/trnsci/trnsparse/blob/main/examples/block_sparse_attention.py).
