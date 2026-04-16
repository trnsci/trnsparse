"""Block-sparse attention via BSRMatrix and bsr_spmm.

The final matmul in sparse attention — `attn_weights @ V` — is a sparse ×
dense matrix multiplication. When the attention mask has 128×128 block
structure (local window, dilated, global tokens), that matmul is exactly
`bsr_spmm(attn_weights_bsr, V)`. No new kernel; BSRMatrix captures the
sparsity pattern, and the Tensor Engine sees a sequence of dense 128×128
tiles with zero gather overhead.

Three mask patterns are demonstrated:

  local  — sliding-window (Longformer-style): each token attends to the
            nearest `window` blocks. Block (i, j) is stored iff |i-j| ≤ window.
  dilated — every `stride`-th block column is stored. Covers dilated sparse
            attention variants.
  global — first `n_global` token blocks attend to and are attended to by
            everyone; remaining rows follow a local window.

Usage:
    python examples/block_sparse_attention.py --demo
    python examples/block_sparse_attention.py --seq-len 512 --pattern dilated
    python examples/block_sparse_attention.py --seq-len 1024 --head-dim 64 --pattern global

Note on materialization: this example computes the full score matrix
`Q @ K.T` before masking. A production path would compute only the nonzero
tiles of S = Q @ K.T, avoiding the O(seq_len²) intermediate. That fused
tile-level score computation is the architectural follow-up; here the claim
being demonstrated is that `bsr_spmm` handles the attn_weights × V step.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import trnsparse

# ---------------------------------------------------------------------------
# Mask pattern helpers
# ---------------------------------------------------------------------------


def _local_window_mask(seq_len: int, block_size: int, window: int) -> torch.Tensor:
    """Boolean mask for local-window (Longformer-style) attention.

    Block (i, j) in block-row i, block-column j is nonzero iff |i - j| ≤ window.

    Returns:
        bool tensor of shape (seq_len, seq_len).
    """
    assert seq_len % block_size == 0, "seq_len must be a multiple of block_size"
    n_blocks = seq_len // block_size
    block_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    for i in range(n_blocks):
        lo = max(0, i - window)
        hi = min(n_blocks, i + window + 1)
        block_mask[i, lo:hi] = True
    # Expand each block indicator to (block_size, block_size)
    return block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)


def _dilated_mask(seq_len: int, block_size: int, stride: int) -> torch.Tensor:
    """Boolean mask for dilated sparse attention.

    Block (i, j) is nonzero iff (i - j) % stride == 0 (i.e., block column j
    is a multiple of stride away from block row i, including j == i).

    Returns:
        bool tensor of shape (seq_len, seq_len).
    """
    assert seq_len % block_size == 0
    n_blocks = seq_len // block_size
    block_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    for i in range(n_blocks):
        for j in range(n_blocks):
            if (i - j) % stride == 0:
                block_mask[i, j] = True
    return block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)


def _global_token_mask(seq_len: int, block_size: int, window: int, n_global: int) -> torch.Tensor:
    """Boolean mask with global tokens + local window (BigBird-style).

    First `n_global` token-blocks attend to all blocks and are attended to
    by all blocks. Remaining rows follow a local window of `window` blocks.

    Returns:
        bool tensor of shape (seq_len, seq_len).
    """
    assert seq_len % block_size == 0
    n_blocks = seq_len // block_size
    assert 0 <= n_global <= n_blocks
    block_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    # Global rows: attend everywhere
    block_mask[:n_global, :] = True
    # Global columns: attended to by everyone
    block_mask[:, :n_global] = True
    # Local window for non-global rows
    for i in range(n_global, n_blocks):
        lo = max(0, i - window)
        hi = min(n_blocks, i + window + 1)
        block_mask[i, lo:hi] = True
    return block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)


# ---------------------------------------------------------------------------
# Attention computation
# ---------------------------------------------------------------------------


def block_sparse_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask_bsr: trnsparse.BSRMatrix,
) -> torch.Tensor:
    """Compute sparse attention output using bsr_spmm for the final matmul.

    Args:
        Q: (seq_len, head_dim) query tensor.
        K: (seq_len, head_dim) key tensor.
        V: (seq_len, head_dim) value tensor.
        mask_bsr: BSRMatrix encoding which attention blocks are nonzero.
            The blocks themselves are ignored — only the sparsity pattern
            (block_row_ptrs, block_col_indices) is used. The actual weights
            come from softmax(Q @ K.T).

    Returns:
        (seq_len, head_dim) attention output.

    Note: this materializes the full (seq_len, seq_len) score matrix Q @ K.T
    before masking. See module docstring on the production alternative.
    """
    seq_len, head_dim = Q.shape
    scale = head_dim**-0.5

    # Full score matrix — O(seq_len²). A production path computes only the
    # nonzero tiles of S to avoid this intermediate.
    scores = (Q @ K.T) * scale  # (seq_len, seq_len)

    # Apply additive mask: positions outside the pattern → -inf before softmax.
    mask_dense = mask_bsr.to_dense().bool()
    masked_scores = scores.masked_fill(~mask_dense, float("-inf"))

    # Row-wise softmax over all attended positions (positions outside mask → 0).
    attn_weights = torch.softmax(masked_scores, dim=-1)  # (seq_len, seq_len)

    # Convert the post-softmax weights to BSR (same pattern, real values now).
    attn_bsr = trnsparse.BSRMatrix.from_dense(attn_weights, block_size=mask_bsr.block_size)

    # Final matmul via bsr_spmm: (seq_len, seq_len) sparse × (seq_len, head_dim) dense.
    return trnsparse.bsr_spmm(attn_bsr, V)  # (seq_len, head_dim)


def _dense_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Dense reference: full softmax(Q @ K.T) @ V with additive mask."""
    scale = Q.shape[-1] ** -0.5
    scores = (Q @ K.T) * scale
    masked = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(masked, dim=-1)
    return attn @ V


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run a small demo (seq_len=512)")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--pattern",
        choices=["local", "dilated", "global"],
        default="local",
        help="attention mask pattern",
    )
    parser.add_argument(
        "--window", type=int, default=2, help="block window for local/global patterns"
    )
    parser.add_argument("--stride", type=int, default=2, help="block stride for dilated pattern")
    parser.add_argument(
        "--n-global", type=int, default=2, help="global token blocks for global pattern"
    )
    parser.add_argument("--block-size", type=int, default=128, help="BSR block size")
    args = parser.parse_args()

    if args.demo:
        args.seq_len = 512
        args.head_dim = 64
        args.pattern = "local"
        args.window = 1

    seq_len = args.seq_len
    head_dim = args.head_dim
    block_size = args.block_size

    if seq_len % block_size != 0:
        print(f"Error: seq_len ({seq_len}) must be a multiple of block_size ({block_size}).")
        sys.exit(1)

    torch.manual_seed(42)
    Q = torch.randn(seq_len, head_dim)
    K = torch.randn(seq_len, head_dim)
    V = torch.randn(seq_len, head_dim)

    # Build the mask.
    if args.pattern == "local":
        mask = _local_window_mask(seq_len, block_size, args.window)
        pattern_desc = f"local window (w={args.window})"
    elif args.pattern == "dilated":
        mask = _dilated_mask(seq_len, block_size, args.stride)
        pattern_desc = f"dilated (stride={args.stride})"
    else:
        mask = _global_token_mask(seq_len, block_size, args.window, args.n_global)
        pattern_desc = f"global tokens (n_global={args.n_global}, w={args.window})"

    mask_bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=block_size)
    n_blocks_total = (seq_len // block_size) ** 2
    block_density = mask_bsr.n_blocks / n_blocks_total

    print("Block-sparse attention via bsr_spmm:")
    print(f"  seq_len:      {seq_len}")
    print(f"  head_dim:     {head_dim}")
    print(f"  block_size:   {block_size}")
    print(f"  pattern:      {pattern_desc}")
    print(
        f"  n_blocks:     {mask_bsr.n_blocks} / {n_blocks_total}  ({block_density:.1%} block density)"
    )
    print()

    # Dense reference.
    out_dense = _dense_reference(Q, K, V, mask)

    # BSR path.
    out_bsr = block_sparse_attention(Q, K, V, mask_bsr)

    max_err = (out_bsr - out_dense).abs().max().item()
    print(f"  Numerical agreement: max |Δ| = {max_err:.2e}")
    assert max_err < 1e-3, f"BSR and dense outputs disagree: max_err={max_err}"

    # Timing: attn_weights @ V, BSR vs dense.
    # Pre-compute attn_weights for a fair comparison of just the matmul step.
    scale = head_dim**-0.5
    masked_scores = (Q @ K.T * scale).masked_fill(~mask, float("-inf"))
    attn_weights = torch.softmax(masked_scores, dim=-1)
    attn_bsr_timed = trnsparse.BSRMatrix.from_dense(attn_weights, block_size=block_size)

    N_REPS = 20

    t0 = time.perf_counter()
    for _ in range(N_REPS):
        _ = trnsparse.bsr_spmm(attn_bsr_timed, V)
    t_bsr = (time.perf_counter() - t0) / N_REPS

    t0 = time.perf_counter()
    for _ in range(N_REPS):
        _ = attn_weights @ V
    t_dense = (time.perf_counter() - t0) / N_REPS

    print(f"  Timing (attn_weights @ V, {N_REPS}-rep mean):")
    print(f"    bsr_spmm:   {t_bsr * 1e6:8.1f} μs")
    print(f"    dense @:    {t_dense * 1e6:8.1f} μs")
    print()
    print("  Note: on CPU the PyTorch fallback is used. bsr_spmm wins at")
    print("  large seq_len on Trainium, where dispatch overhead amortizes")
    print("  against the tile-level nc_matmul throughput advantage.")


if __name__ == "__main__":
    main()
