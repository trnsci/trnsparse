"""``torch.cat`` without one device concatenation past 2^32 bytes (#150).

On Trn2 at LNC=1 (Neuron SDK 2.33.1), device aten ``cat`` into an output
above 2^32 bytes writes to wrong places, including its own inputs' memory
(aws-neuron/aws-neuron-sdk#1502; the same class as #1457, offsets past 2^32
bytes). At LNC=2, or below 2^32 bytes, it is exact.

`cat` is ``torch.cat`` up to `_CAT_MAX_BYTES`. Above that it preallocates the
output once and copies each input into its slice, one row block (at most
`_SLICE_BYTES` of the output, `row_blocks`) at a time, so no copy addresses a
view spanning more than 2^31 bytes (device aten copies a column slice of a
view past 2^32 bytes about 170x slower, trnsparse#141). Along dim 0 each copy
writes one contiguous block. The values are the inputs' bits either way. The
backward hands each input its slice of the cotangent, as ``torch.cat``'s does.
"""

from __future__ import annotations

import functools

import torch

# Largest output (bytes) built with one `torch.cat`.
_CAT_MAX_BYTES = 2**32
# Largest view (bytes of the output's rows) one copy addresses.
_SLICE_BYTES = 2**31


def row_blocks(rows: int, row_bytes: int, limit: int | None = None) -> list[tuple[int, int]]:
    """``[(r0, r1), ...]`` covering ``range(rows)``, each block at most `limit`
    bytes (default `_SLICE_BYTES`) of rows ``row_bytes`` wide, and at least one row."""
    step = max(1, (_SLICE_BYTES if limit is None else limit) // max(row_bytes, 1))
    return [(r0, min(r0 + step, rows)) for r0 in range(0, rows, step)]


def _out_shape(tensors, dim: int) -> list[int]:
    shape = list(tensors[0].shape)
    for t in tensors[1:]:
        if t.dim() != len(shape) or any(
            a != b for d, (a, b) in enumerate(zip(t.shape, shape, strict=True)) if d != dim
        ):
            raise RuntimeError(
                f"trnsparse cat: shapes {[tuple(x.shape) for x in tensors]} do not match "
                f"outside dim {dim}"
            )
    shape[dim] = sum(int(t.shape[dim]) for t in tensors)
    return shape


def _assemble(tensors, dim: int, shape, dtype) -> torch.Tensor:
    out = tensors[0].new_empty(shape, dtype=dtype)
    if out.numel() == 0:
        return out
    row_bytes = out.element_size() * (out.numel() // out.shape[0])
    off = 0
    for t in tensors:
        k = int(t.shape[dim])
        if k == 0:
            continue
        if dim == 0:  # contiguous blocks of the output
            for r0, r1 in row_blocks(k, row_bytes):
                out[off + r0 : off + r1].copy_(t[r0:r1])
        else:
            for r0, r1 in row_blocks(out.shape[0], row_bytes):
                out[r0:r1].narrow(dim, off, k).copy_(t[r0:r1])
        off += k
    return out


class _BlockCat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim, shape, dtype, *tensors):
        ctx.dim = dim
        ctx.sizes = [int(t.shape[dim]) for t in tensors]
        ctx.dtypes = [t.dtype for t in tensors]
        return _assemble(tensors, dim, shape, dtype)

    @staticmethod
    def backward(ctx, grad):
        grads, off = [], 0
        for k, dt in zip(ctx.sizes, ctx.dtypes, strict=True):
            grads.append(grad.narrow(ctx.dim, off, k).to(dt))
            off += k
        return (None, None, None, *grads)


def cat(tensors, dim: int = 0) -> torch.Tensor:
    """``torch.cat(tensors, dim)``, with no single device ``cat`` past
    `_CAT_MAX_BYTES` (see the module docstring). Same values, dtype and
    gradients."""
    tensors = list(tensors)
    if not tensors:
        return torch.cat(tensors, dim)
    dtype = functools.reduce(torch.promote_types, (t.dtype for t in tensors))
    nbytes = sum(t.numel() for t in tensors) * dtype.itemsize
    if nbytes <= _CAT_MAX_BYTES:
        return torch.cat(tensors, dim)
    dim = dim % tensors[0].dim()
    return _BlockCat.apply(dim, _out_shape(tensors, dim), dtype, *tensors)


def stack(tensors, dim: int = 0) -> torch.Tensor:
    """``torch.stack(tensors, dim)`` through `cat`."""
    tensors = list(tensors)
    if not tensors:
        return torch.stack(tensors, dim)
    dim = dim % (tensors[0].dim() + 1)
    return cat([t.unsqueeze(dim) for t in tensors], dim)
