"""`trnsparse._cat`: concatenation without one device ``torch.cat`` past 2^32 bytes (#150).

The bound is patched down so the block path runs at test sizes: the result
must equal ``torch.cat`` bit for bit, in dtype, shape and gradients, and the
block path must not call ``torch.cat``.
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from trnsparse import _cat


@pytest.fixture
def block_path(monkeypatch):
    """Every non-empty cat takes the block path, in 3-row blocks, without torch.cat."""
    monkeypatch.setattr(_cat, "_CAT_MAX_BYTES", 0)
    real = torch.cat

    def forbid(*a, **k):
        raise AssertionError("torch.cat on the block path")

    def run(fn, *a, **k):
        with monkeypatch.context() as mp:
            mp.setattr(torch, "cat", forbid)
            return fn(*a, **k)

    return real, run


def _parts(shape_fn, sizes, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*shape_fn(k), generator=g).to(dtype) for k in sizes]


@pytest.mark.parametrize(
    "shape_fn,dim",
    [
        (lambda k: (k,), 0),
        (lambda k: (k, 7), 0),
        (lambda k: (11, k), 1),
        (lambda k: (11, k), -1),
        (lambda k: (5, k, 3), 1),
        (lambda k: (5, 4, k), 2),
        (lambda k: (k, 4, 3), 0),
    ],
)
def test_cat_bitwise(monkeypatch, block_path, shape_fn, dim):
    real, run = block_path
    monkeypatch.setattr(_cat, "_SLICE_BYTES", 3 * 4 * 7 * 4 * 3)  # a few rows per copy
    parts = _parts(shape_fn, (4, 0, 9, 1))
    got = run(_cat.cat, parts, dim)
    ref = real(parts, dim)
    assert got.shape == ref.shape and got.dtype == ref.dtype and got.is_contiguous()
    assert torch.equal(got, ref)


def test_cat_promotes_like_torch(block_path):
    real, run = block_path
    a = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    b = torch.randn(2, 2, dtype=torch.float64)
    got = run(_cat.cat, [a, b], 1)
    assert got.dtype == torch.float64 and torch.equal(got, real([a, b], 1))


def test_cat_complex_parts(block_path):
    # `[Re B | Im B]` as the complex BSR SpMM and kernel_matmul build it.
    real, run = block_path
    B = torch.randn(13, 5, dtype=torch.complex64)
    assert torch.equal(run(_cat.cat, [B.real, B.imag], 1), real([B.real, B.imag], 1))


def test_cat_shape_mismatch_raises(block_path):
    _, run = block_path
    with pytest.raises(RuntimeError, match="do not match"):
        run(_cat.cat, [torch.zeros(2, 3), torch.zeros(3, 3)], 1)


def test_cat_gradients(block_path):
    _, run = block_path
    a = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    c = torch.randn(4, 1, dtype=torch.float64)
    assert run(gradcheck, lambda x, y: _cat.cat([x, y, c], 1), (a, b))
    assert run(gradgradcheck, lambda x, y: _cat.cat([x, c, y], 1), (a, b))
    # Mixed dtypes: each input's gradient comes back in its own dtype.
    f = torch.randn(4, 3, dtype=torch.float32, requires_grad=True)
    out = run(_cat.cat, [f, b], 1)
    out.sum().backward()
    assert f.grad.dtype == torch.float32 and torch.equal(f.grad, torch.ones(4, 3))


def test_stack_bitwise(block_path):
    real, run = block_path
    rows = [torch.arange(5), torch.arange(5, 10)]
    assert torch.equal(run(_cat.stack, rows), torch.stack(rows))
    assert torch.equal(run(_cat.stack, rows, 1), torch.stack(rows, 1))


def test_small_cat_is_torch_cat(monkeypatch):
    # Below the bound the helper is torch.cat itself (the default for every
    # existing call site below 2^32 bytes).
    calls = []
    real = torch.cat
    monkeypatch.setattr(torch, "cat", lambda *a, **k: calls.append(1) or real(*a, **k))
    _cat.cat([torch.zeros(2), torch.ones(3)])
    assert calls == [1]


def test_row_blocks():
    assert _cat.row_blocks(10, 4, 12) == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert _cat.row_blocks(3, 100, 12) == [(0, 1), (1, 2), (2, 3)]  # at least one row
    assert _cat.row_blocks(0, 4) == []
