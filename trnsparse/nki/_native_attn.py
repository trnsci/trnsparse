"""
Native TorchNeuron block-sparse attention as autograd-aware custom ops.

`ops._AttnTiledFunction` (a `torch.autograd.Function`) wraps the two-pass
NKI forward and the dQ / dK-dV NKI backward on the PyTorch/XLA path. On the
native path the same pair is two composite custom ops, linked with
`register_autograd`:

- `trnsparse::bsr_attn(Q, K, V, block_col_indices, block_row_ptrs, block_size, scale=None)`
  -> `(out, row_max, row_denom)`: `dispatch.nki_bsr_attn_tiled`, i.e. the
  `trnsparse::attn_stats` and `trnsparse::attn_out` kernel ops plus the
  host-side softmax reduction.
- `trnsparse::bsr_attn_bwd(...)` -> `(dQ, dK, dV)`: `dispatch.nki_bsr_attn_bwd`,
  i.e. the `trnsparse::attn_bwd_dq` / `attn_bwd_dkdv` kernel ops.

They are composite (several kernel launches plus host-side gathers over the
mask structure), so they are plain `torch.library.custom_op`s rather than
`nki_op`s. The mask structure arguments are host tensors (the `neuron`
device has no sparse CSR, and the gather loops read them as scalars); Q, K,
V may be on `neuron`. Outputs are on Q's device.

Kept apart from `_native.py` so it registers once per process even when
`_native` is re-imported, and so it imports nothing from torch_neuronx (the
CPU tests exercise these ops). Like `_native.py`, no
`from __future__ import annotations`: the schema comes from the live
annotations. The dispatch functions are looked up on `dispatch` at call
time.
"""

import torch

from . import dispatch as _d


def _stats_dtype(dtype: torch.dtype) -> torch.dtype:
    """Softmax statistics dtype: fp32 for low-precision Q (bf16 / fp16), else Q's.

    The backward recomputes P = exp(S - row_max) / row_denom; a bf16
    row_max would scale every P of the row by up to exp(2^-9 · |row_max|).
    """
    return torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype


def _pattern(block_col_indices, block_row_ptrs, seq_len, block_size):
    return _d._MaskPattern(block_col_indices, block_row_ptrs, (seq_len, seq_len), block_size)


@torch.library.custom_op("trnsparse::bsr_attn", mutates_args=())
def bsr_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_col_indices: torch.Tensor,
    block_row_ptrs: torch.Tensor,
    block_size: int,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = _pattern(block_col_indices, block_row_ptrs, Q.shape[0], block_size)
    out, row_max, row_denom = _d.nki_bsr_attn_tiled(Q, K, V, mask, return_stats=True, scale=scale)
    # Fixed output contract (Q's device; out in Q's dtype, the statistics in
    # `_stats_dtype`) to match the fake.
    sdt = _stats_dtype(Q.dtype)
    return (
        out.to(device=Q.device, dtype=Q.dtype),
        row_max.to(device=Q.device, dtype=sdt),
        row_denom.to(device=Q.device, dtype=sdt),
    )


@bsr_attn.register_fake
def _bsr_attn_fake(Q, K, V, block_col_indices, block_row_ptrs, block_size, scale=None):
    M_tiles = Q.shape[0] // block_size
    sdt = _stats_dtype(Q.dtype)
    return (
        torch.empty_like(Q),
        Q.new_empty((M_tiles, block_size), dtype=sdt),
        Q.new_empty((M_tiles, block_size), dtype=sdt),
    )


@torch.library.custom_op("trnsparse::bsr_attn_bwd", mutates_args=())
def bsr_attn_bwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    O: torch.Tensor,
    block_col_indices: torch.Tensor,
    block_row_ptrs: torch.Tensor,
    block_size: int,
    row_max: torch.Tensor,
    row_denom: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = _pattern(block_col_indices, block_row_ptrs, Q.shape[0], block_size)
    dQ, dK, dV = _d.nki_bsr_attn_bwd(Q, K, V, dO, O, mask, row_max, row_denom, scale=scale)
    return tuple(t.to(device=Q.device, dtype=Q.dtype) for t in (dQ, dK, dV))


@bsr_attn_bwd.register_fake
def _bsr_attn_bwd_fake(Q, K, V, dO, O, block_col_indices, block_row_ptrs, block_size, *rest):
    return torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V)


def _bsr_attn_setup_context(ctx, inputs, output):
    Q, K, V, block_col_indices, block_row_ptrs, block_size, scale = inputs
    out, row_max, row_denom = output
    ctx.save_for_backward(Q, K, V, out, row_max, row_denom, block_col_indices, block_row_ptrs)
    ctx.block_size = block_size
    ctx.scale = scale


def _bsr_attn_backward(ctx, d_out, d_row_max, d_row_denom):
    # row_max / row_denom are saved softmax statistics, not differentiable
    # outputs: their incoming gradients are ignored, as in _AttnTiledFunction.
    Q, K, V, O, row_max, row_denom, block_col_indices, block_row_ptrs = ctx.saved_tensors
    dQ, dK, dV = torch.ops.trnsparse.bsr_attn_bwd(
        Q,
        K,
        V,
        d_out.contiguous(),
        O,
        block_col_indices,
        block_row_ptrs,
        ctx.block_size,
        row_max,
        row_denom,
        ctx.scale,
    )
    return dQ, dK, dV, None, None, None, None


bsr_attn.register_autograd(_bsr_attn_backward, setup_context=_bsr_attn_setup_context)
