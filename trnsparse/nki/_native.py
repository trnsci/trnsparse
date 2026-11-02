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


def _fake_bsr_spmm_static(blocks, b, row_ptrs, cols):
    return b.new_empty(((len(row_ptrs) - 1) * _P, b.shape[1]))


@nki_op("trnsparse::bsr_spmm_static", mutates_args={}, fake_fn=_fake_bsr_spmm_static)
def bsr_spmm_static(
    blocks: torch.Tensor, b: torch.Tensor, row_ptrs: list[int], cols: list[int]
) -> torch.Tensor:
    # Structure arrives as int lists and reaches the kernel as tuples, i.e.
    # compile-time constants (torch_neuronx keys the compiled kernel on them).
    return _k._bsr_spmm_static_kernel(blocks, b, tuple(row_ptrs), tuple(cols))


def _fake_bsr_spmm_static_t(blocks_t, b, row_ptrs, cols):
    return b.new_empty(((len(row_ptrs) - 1) * _P, b.shape[1]))


@nki_op("trnsparse::bsr_spmm_static_t", mutates_args={}, fake_fn=_fake_bsr_spmm_static_t)
def bsr_spmm_static_t(
    blocks_t: torch.Tensor, b: torch.Tensor, row_ptrs: list[int], cols: list[int]
) -> torch.Tensor:
    # blocks_t: (128, n_blocks * 128), block idx's transpose in columns idx*128..
    return _k._bsr_spmm_static_t_kernel(blocks_t, b, tuple(row_ptrs), tuple(cols))


def _fake_keyed_spmm(blocks, b, key):
    return b.new_empty(((len(_k._STRUCTURES[key][0]) - 1) * _P, b.shape[1]))


# Keyed variants: the structure is registered once (`kernels.register_structure`)
# and crosses the op boundary as one int, so a launch's host cost does not
# grow with the number of stored blocks.


@nki_op("trnsparse::bsr_spmm_keyed", mutates_args={}, fake_fn=_fake_keyed_spmm)
def bsr_spmm_keyed(blocks: torch.Tensor, b: torch.Tensor, key: int) -> torch.Tensor:
    return _k._bsr_spmm_keyed_kernel(blocks, b, int(key))


@nki_op("trnsparse::bsr_spmm_t_keyed", mutates_args={}, fake_fn=_fake_keyed_spmm)
def bsr_spmm_t_keyed(blocks_t: torch.Tensor, b: torch.Tensor, key: int) -> torch.Tensor:
    return _k._bsr_spmm_t_keyed_kernel(blocks_t, b, int(key))


def _fake_attn_fwd_keyed(q_t, k_t, v, key):
    seq = q_t.shape[1]
    hd = v.shape[1]
    return v.new_empty((seq, hd)), v.new_empty((seq, 1)), v.new_empty((seq, 1))


@nki_op("trnsparse::attn_fwd_keyed", mutates_args={}, fake_fn=_fake_attn_fwd_keyed)
def attn_fwd_keyed(
    q_t: torch.Tensor, k_t: torch.Tensor, v: torch.Tensor, key: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _k._attn_fwd_keyed_kernel(q_t, k_t, v, int(key))


def _fake_attn_fwd_static(q_t, k_t, v, row_ptrs, cols):
    seq = v.shape[0]
    return v.new_empty(v.shape), v.new_empty((seq, 1)), v.new_empty((seq, 1))


@nki_op("trnsparse::attn_fwd_static", mutates_args={}, fake_fn=_fake_attn_fwd_static)
def attn_fwd_static(
    q_t: torch.Tensor, k_t: torch.Tensor, v: torch.Tensor, row_ptrs: list[int], cols: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _k._attn_fwd_static_kernel(q_t, k_t, v, tuple(row_ptrs), tuple(cols))


def _fake_attn_bwd_static(q_t, k_t, v_t, do_t, q_s, *rest):
    return q_s.new_empty(q_s.shape), q_s.new_empty(q_s.shape), q_s.new_empty(q_s.shape)


@nki_op("trnsparse::attn_bwd_static", mutates_args={}, fake_fn=_fake_attn_bwd_static)
def attn_bwd_static(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    do_t: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    rm_col: torch.Tensor,
    rinv_col: torch.Tensor,
    d_col: torch.Tensor,
    rm_row: torch.Tensor,
    rinv_row: torch.Tensor,
    d_row: torch.Tensor,
    row_ptrs: list[int],
    cols: list[int],
    col_ptrs: list[int],
    rows: list[int],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _k._attn_bwd_static_kernel(
        q_t,
        k_t,
        v_t,
        do_t,
        q_s,
        k,
        do,
        rm_col,
        rinv_col,
        d_col,
        rm_row,
        rinv_row,
        d_row,
        tuple(row_ptrs),
        tuple(cols),
        tuple(col_ptrs),
        tuple(rows),
        float(scale),
    )


def _fake_attn_stats(q_scaled_blocks, k_gathered_pad):
    M_tiles, K_max = k_gathered_pad.shape[:2]
    shape = (M_tiles, K_max, _P, 1)
    # The statistics are fp32 for fp32 and bf16 Q (#49).
    return (
        q_scaled_blocks.new_empty(shape, dtype=torch.float32),
        q_scaled_blocks.new_empty(shape, dtype=torch.float32),
    )


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


def _fake_cg_bsr_keyed(
    blocks_t,
    minv_t,
    state,
    key,
    check_every,
    precond,
    sbuf_kib,
    core_split,
    matvec,
    precision,
    variant=0,
):
    T = len(_k._STRUCTURES[key][0]) - 1
    R = (state.shape[1] - 2) // (2 * T)
    return state.new_empty((_P, T * R + 3 * R + 1))


@nki_op("trnsparse::cg_bsr_keyed", mutates_args={}, fake_fn=_fake_cg_bsr_keyed)
def cg_bsr_keyed(
    blocks_t: torch.Tensor,
    minv_t: torch.Tensor,
    state: torch.Tensor,
    key: int,
    check_every: int,
    precond: int,
    sbuf_kib: int,
    core_split: int,
    matvec: int,
    precision: int,
    variant: int = 0,
) -> torch.Tensor:
    # Fused SBUF-resident CG (#22); no autograd (a solver, not a layer).
    # variant: 0 = standard CG, 1 = Chronopoulos-Gear recurrence (opt-in).
    return _k._cg_bsr_keyed_kernel(
        blocks_t,
        minv_t,
        state,
        int(key),
        int(check_every),
        int(precond),
        int(sbuf_kib),
        int(core_split),
        int(matvec),
        int(precision),
        int(variant),
    )


def _fake_power_bsr_keyed(
    blocks_t, state, key, check_every, sbuf_kib, core_split, matvec, precision
):
    T = len(_k._STRUCTURES[key][0]) - 1
    return state.new_empty((_P, T + 4))


@nki_op("trnsparse::power_bsr_keyed", mutates_args={}, fake_fn=_fake_power_bsr_keyed)
def power_bsr_keyed(
    blocks_t: torch.Tensor,
    state: torch.Tensor,
    key: int,
    check_every: int,
    sbuf_kib: int,
    core_split: int,
    matvec: int,
    precision: int,
) -> torch.Tensor:
    # Fused power iteration (#22): the CG kernel's machinery, algo=1.
    return _k._power_bsr_keyed_kernel(
        blocks_t,
        state,
        int(key),
        int(check_every),
        int(sbuf_kib),
        int(core_split),
        int(matvec),
        int(precision),
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


def _save_static(ctx, inputs, output):
    blocks, b, row_ptrs, cols = inputs
    ctx.save_for_backward(blocks, b)
    ctx.row_ptrs = row_ptrs
    ctx.cols = cols


def _bsr_spmm_static_backward(ctx, grad):
    # out[m] = sum_{idx in row m} blocks[idx] @ b[cols[idx]]
    blocks, b = ctx.saved_tensors
    row_ptrs, cols = ctx.row_ptrs, ctx.cols
    n_blocks, bs, _ = blocks.shape
    N = b.shape[1]
    counts = [row_ptrs[i + 1] - row_ptrs[i] for i in range(len(row_ptrs) - 1)]
    rows = torch.repeat_interleave(
        torch.arange(len(counts)), torch.tensor(counts, dtype=torch.long)
    )
    rows = rows.to(grad.device)
    cols_t = torch.tensor(cols, dtype=torch.long, device=grad.device)
    g_rows = grad.reshape(-1, bs, N)[rows]  # (n_blocks, bs, N)
    b_cols = b.reshape(-1, bs, N)[cols_t]  # (n_blocks, bs, N)
    grad_blocks = g_rows @ b_cols.transpose(-1, -2) if ctx.needs_input_grad[0] else None
    grad_b = None
    if ctx.needs_input_grad[1]:
        contrib = blocks.transpose(-1, -2) @ g_rows  # (n_blocks, bs, N)
        grad_b = torch.zeros(b.shape[0] // bs, bs, N, dtype=b.dtype, device=b.device)
        grad_b = grad_b.index_add(0, cols_t, contrib.to(b.dtype)).reshape(b.shape)
    return grad_blocks, grad_b, None, None


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
bsr_spmm_static.register_autograd(_bsr_spmm_static_backward, setup_context=_save_static)


def _save_keyed(ctx, inputs, output):
    blocks, b, key = inputs
    ctx.save_for_backward(blocks, b)
    ctx.row_ptrs, ctx.cols = _k._STRUCTURES[key]


def _bsr_spmm_keyed_backward(ctx, grad):
    grad_blocks, grad_b, _, _ = _bsr_spmm_static_backward(ctx, grad)
    return grad_blocks, grad_b, None


bsr_spmm_keyed.register_autograd(_bsr_spmm_keyed_backward, setup_context=_save_keyed)


def _save_t_keyed(ctx, inputs, output):
    # blocks_t (bs, n_blocks * bs) holds block idx's transpose in columns
    # idx*bs..; the backward works on the (n_blocks, bs, bs) blocks.
    blocks_t, b, key = inputs
    bs = blocks_t.shape[0]
    ctx.save_for_backward(blocks_t.reshape(bs, -1, bs).permute(1, 2, 0), b)
    ctx.row_ptrs, ctx.cols = _k._STRUCTURES[key]


def _bsr_spmm_t_keyed_backward(ctx, grad):
    grad_blocks, grad_b, _, _ = _bsr_spmm_static_backward(ctx, grad)
    grad_t = None
    if grad_blocks is not None:
        grad_t = grad_blocks.permute(2, 0, 1).reshape(grad_blocks.shape[-1], -1)
    return grad_t, grad_b, None


bsr_spmm_t_keyed.register_autograd(_bsr_spmm_t_keyed_backward, setup_context=_save_t_keyed)
# attn_fwd_static / attn_bwd_static have no register_autograd of their own:
# they run inside the composite trnsparse::bsr_attn / bsr_attn_bwd ops, whose
# autograd link is attached in _native_attn.


# Composite attention ops (forward/backward over Q, K, V with host mask
# structure). Registered once per process; see `_native_attn.py`.
from . import _native_attn  # noqa: E402

# Keyed by the short op name used in `dispatch._launch` / `dispatch._run`.
OPS = {
    "spmm_dense": spmm_dense,
    "bsr_spmm": bsr_spmm,
    "bsr_spmm_static": bsr_spmm_static,
    "bsr_spmm_static_t": bsr_spmm_static_t,
    "bsr_spmm_keyed": bsr_spmm_keyed,
    "bsr_spmm_t_keyed": bsr_spmm_t_keyed,
    "attn_fwd_keyed": attn_fwd_keyed,
    "cg_bsr_keyed": cg_bsr_keyed,
    "power_bsr_keyed": power_bsr_keyed,
    "attn_fwd_static": attn_fwd_static,
    "attn_bwd_static": attn_bwd_static,
    "screened_spmm": screened_spmm,
    "attn_stats": attn_stats,
    "attn_out": attn_out,
    "attn_bwd_dq": attn_bwd_dq,
    "attn_bwd_dkdv": attn_bwd_dkdv,
    "bsr_attn": _native_attn.bsr_attn,
    "bsr_attn_bwd": _native_attn.bsr_attn_bwd,
}
