"""
Native TorchNeuron registration of the trnsparse NKI kernels.

On the TorchNeuron lineage (Neuron SDK 2.33+, torch-neuronx 2.14,
PyTorch 2.14, Trn2/Trn3), Trainium is a native PyTorch `neuron` device
and an `@nki.jit` kernel is exposed to PyTorch as a custom operator via
`torch_neuronx.nki_op`. Each wrapper below calls an existing kernel from
`kernels.py` unchanged; the op schema is inferred from the annotations.

Autograd: the PyTorch/XLA path wraps its kernels in
`torch.autograd.Function`s. Here the backward is attached to the op itself
with `register_autograd` (the `CustomOpDef` that `nki_op` returns), so it
also holds under `torch.compile`:

- `trnsparse::spmm_dense`    C = A @ B            dA = dC Bᵀ,  dB = Aᵀ dC
- `trnsparse::bsr_spmm`      C[m] = Σ_k A[m,k] B[m,k]  (per k-slot adjoints)
- `trnsparse::screened_spmm` C = (A ⊙ mask) @ B   dA = (dC Bᵀ) ⊙ mask, dB = (A ⊙ mask)ᵀ dC
  (no gradient to the Schwarz bounds or the threshold: the mask is discrete)

Gradients reach `CSRMatrix.values`, `BSRMatrix.blocks` and the dense
operands through the (differentiable) host-side densify / pad / gather.
The attention kernels are the inner pieces of the `trnsparse::bsr_attn`
op, whose autograd lives in `_native_attn.py`.

Importing this module registers the ops, so `dispatch.py` imports it only
after selecting the native backend. CPU-only installs and the Trn1 / XLA
lineage (whose `torch_neuronx` has no `nki_op`) never load it.

This module deliberately does not use `from __future__ import annotations`:
`nki_op` builds the op schema from the live annotation objects.

The kernels are looked up on `kernels` at call time rather than imported
by name, so registration itself does not trace or compile anything; the
first call per input shape does.
"""

import torch
from torch_neuronx import nki_op

from . import kernels as _k

_P = _k._TILE_M  # partition tile; every kernel's output rows are multiples of it

# Fake (meta) implementations give output shape/dtype only, with no kernel
# trace. Used by torch.compile tracing and for meta-device inputs.


@nki_op(
    "trnsparse::spmm_dense",
    mutates_args={},
    fake_fn=lambda a, b: a.new_empty((a.shape[0], b.shape[1])),
)
def spmm_dense(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _k._spmm_dense_kernel(a, b)


@nki_op(
    "trnsparse::bsr_spmm",
    mutates_args={},
    fake_fn=lambda blocks_pad, b_gathered: blocks_pad.new_empty(
        (blocks_pad.shape[0] * _P, b_gathered.shape[-1])
    ),
)
def bsr_spmm(blocks_pad: torch.Tensor, b_gathered: torch.Tensor) -> torch.Tensor:
    return _k._bsr_spmm_kernel(blocks_pad, b_gathered)


@nki_op(
    "trnsparse::screened_spmm",
    mutates_args={},
    fake_fn=lambda a, q, threshold_sqrt, b: a.new_empty((a.shape[0], b.shape[1])),
)
def screened_spmm(
    a: torch.Tensor, q: torch.Tensor, threshold_sqrt: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return _k._screened_spmm_kernel(a, q, threshold_sqrt, b)


def _fake_attn_stats(q_scaled_blocks, k_gathered_pad):
    M_tiles, K_max = k_gathered_pad.shape[:2]
    shape = (M_tiles, K_max, _P, 1)
    return q_scaled_blocks.new_empty(shape), q_scaled_blocks.new_empty(shape)


@nki_op("trnsparse::attn_stats", mutates_args={}, fake_fn=_fake_attn_stats)
def attn_stats(
    q_scaled_blocks: torch.Tensor, k_gathered_pad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _k._attn_stats_kernel(q_scaled_blocks, k_gathered_pad)


@nki_op(
    "trnsparse::attn_out",
    mutates_args={},
    fake_fn=lambda q_scaled_blocks, k_gathered_pad, v_gathered_pad, row_max, row_denom: (
        q_scaled_blocks.new_empty((k_gathered_pad.shape[0] * _P, k_gathered_pad.shape[3]))
    ),
)
def attn_out(
    q_scaled_blocks: torch.Tensor,
    k_gathered_pad: torch.Tensor,
    v_gathered_pad: torch.Tensor,
    row_max: torch.Tensor,
    row_denom: torch.Tensor,
) -> torch.Tensor:
    return _k._attn_out_kernel(q_scaled_blocks, k_gathered_pad, v_gathered_pad, row_max, row_denom)


@nki_op(
    "trnsparse::attn_bwd_dq",
    mutates_args={},
    fake_fn=lambda q_scaled_blocks, k_gathered_pad, *rest: q_scaled_blocks.new_empty(
        (k_gathered_pad.shape[0] * _P, k_gathered_pad.shape[3])
    ),
)
def attn_bwd_dq(
    q_scaled_blocks: torch.Tensor,
    k_gathered_pad: torch.Tensor,
    v_gathered_pad: torch.Tensor,
    do_gathered_pad: torch.Tensor,
    D_blocks: torch.Tensor,
    row_max: torch.Tensor,
    row_denom: torch.Tensor,
) -> torch.Tensor:
    return _k._attn_bwd_dq_kernel(
        q_scaled_blocks,
        k_gathered_pad,
        v_gathered_pad,
        do_gathered_pad,
        D_blocks,
        row_max,
        row_denom,
    )


def _fake_attn_bwd_dkdv(k_blocks, v_blocks, q_gathered_col, *rest):
    N_col, _, _, head_dim = q_gathered_col.shape
    shape = (N_col * _P, head_dim)
    return k_blocks.new_empty(shape), k_blocks.new_empty(shape)


@nki_op("trnsparse::attn_bwd_dkdv", mutates_args={}, fake_fn=_fake_attn_bwd_dkdv)
def attn_bwd_dkdv(
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    q_gathered_col: torch.Tensor,
    do_gathered_col: torch.Tensor,
    D_gathered_col: torch.Tensor,
    row_max_gathered_col: torch.Tensor,
    row_denom_gathered_col: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _k._attn_bwd_dkdv_kernel(
        k_blocks,
        v_blocks,
        q_gathered_col,
        do_gathered_col,
        D_gathered_col,
        row_max_gathered_col,
        row_denom_gathered_col,
    )


# ── Autograd (register_autograd) ─────────────────────────────────────────────
#
# Backward formulas run as plain torch ops on the forward's device. They
# are module-level so the CPU tests can check them against torch.autograd.


def _save_inputs(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _spmm_dense_backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a = grad @ b.T if ctx.needs_input_grad[0] else None
    grad_b = a.T @ grad if ctx.needs_input_grad[1] else None
    return grad_a, grad_b


def _bsr_spmm_backward(ctx, grad):
    # out[m*P:(m+1)*P] = Σ_k blocks_pad[m, k] @ b_gathered[m, k]
    blocks_pad, b_gathered = ctx.saved_tensors
    M_tiles, _, bs, _ = blocks_pad.shape
    g = grad.reshape(M_tiles, 1, bs, grad.shape[-1])  # broadcast over k-slots
    grad_blocks = g @ b_gathered.transpose(-1, -2) if ctx.needs_input_grad[0] else None
    grad_b = blocks_pad.transpose(-1, -2) @ g if ctx.needs_input_grad[1] else None
    return grad_blocks, grad_b


def _screened_mask(a, q, threshold_sqrt):
    # Same predicate as the kernel: Q[i] * Q[j] > sqrt(threshold).
    return ((q * q.reshape(1, -1)) > threshold_sqrt.reshape(1, 1)).to(a.dtype)


def _screened_spmm_backward(ctx, grad):
    a, q, threshold_sqrt, b = ctx.saved_tensors
    mask = _screened_mask(a, q, threshold_sqrt)
    grad_a = (grad @ b.T) * mask if ctx.needs_input_grad[0] else None
    grad_b = (a * mask).T @ grad if ctx.needs_input_grad[3] else None
    return grad_a, None, None, grad_b


spmm_dense.register_autograd(_spmm_dense_backward, setup_context=_save_inputs)
bsr_spmm.register_autograd(_bsr_spmm_backward, setup_context=_save_inputs)
screened_spmm.register_autograd(_screened_spmm_backward, setup_context=_save_inputs)


# Composite attention ops (forward/backward over Q, K, V with host mask
# structure). Registered once per process; see `_native_attn.py`.
from . import _native_attn  # noqa: E402

# Keyed by the short op name used in `dispatch._launch` / `dispatch._run`.
OPS = {
    "spmm_dense": spmm_dense,
    "bsr_spmm": bsr_spmm,
    "screened_spmm": screened_spmm,
    "attn_stats": attn_stats,
    "attn_out": attn_out,
    "attn_bwd_dq": attn_bwd_dq,
    "attn_bwd_dkdv": attn_bwd_dkdv,
    "bsr_attn": _native_attn.bsr_attn,
    "bsr_attn_bwd": _native_attn.bsr_attn_bwd,
}
