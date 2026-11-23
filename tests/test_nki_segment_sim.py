"""Simulator-backed tests of the segment-sum / gather kernels (#139).

Run with ``TRNSPARSE_USE_SIMULATOR=1`` (CI also sets TRNSPARSE_REQUIRE_NKI=1,
so a kernel failure is an error, not a fallback) on a host with ``nki``:

    TRNSPARSE_USE_SIMULATOR=1 TRNSPARSE_REQUIRE_NKI=1 pytest tests/test_nki_segment_sim.py

The kernels run through ``nki.simulate`` on the plan's tables: values against
the fp64 incidence product, the gather bitwise against ``index_select``, both
backwards, every exemplar width, the dense and compact tile layouts, several
chunks per tile, ragged tiles and the exemplar's graph shapes. The simulator
does not model the hardware's summation order or its LNC split; the bitwise
LNC and rerun checks are in tests/test_segment_neuron.py.
"""

from __future__ import annotations

import os

import pytest
import torch

import trnsparse

pytestmark = pytest.mark.nki_simulator


@pytest.fixture(autouse=True)
def _simulator_enabled():
    if os.environ.get("TRNSPARSE_USE_SIMULATOR", "").lower() not in ("1", "true", "yes"):
        pytest.skip("TRNSPARSE_USE_SIMULATOR=1 required")
    from trnsparse.nki.dispatch import HAS_NKI

    if not HAS_NKI:
        pytest.skip("nki package not importable on this host")


@pytest.fixture(autouse=True)
def nki_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


def _ref(m, idx, n):
    A = torch.zeros(n, idx.numel(), dtype=torch.float64)
    A[idx, torch.arange(idx.numel())] = 1.0
    return A @ m.double().reshape(idx.numel(), -1)


def _rel(got, ref):
    got, ref = got.detach().double(), ref.double()
    return float((got - ref).norm() / ref.norm())


def _launches(monkeypatch):
    from trnsparse.nki import dispatch

    calls = []
    real = dispatch.nki_segment_op

    def op(name, x, plan):
        calls.append(name)
        return real(name, x, plan)

    monkeypatch.setattr(dispatch, "nki_segment_op", op)
    return calls


@pytest.mark.parametrize(
    "n,e,F,lo,hi",
    [
        (16, 150, 3, 0, 16),  # the CH3COOH dimer shape, force assembly width
        (64, 2156, 128, 0, 64),  # Cu(111) slab, layer-0 feature width
        (129, 600, 1, 0, 129),  # ragged last tile, density width
        (200, 1500, 1100, 0, 200),  # two column groups, a ragged PSUM tile
        (128 * 40, 500, 4, 128 * 30, 128 * 30 + 300),  # compact layout
        (3, 4096, 64, 0, 1),  # one heavy row: 32 slots, four chunks
    ],
)
def test_segment_sum_and_gather(monkeypatch, n, e, F, lo, hi):
    calls = _launches(monkeypatch)
    g = torch.Generator().manual_seed(n + e)
    idx = torch.randint(lo, hi, (e,), generator=g)
    plan = trnsparse.segment_plan(idx, n)
    m = torch.randn(e, F, generator=g, requires_grad=True)
    out = trnsparse.segment_sum(m, idx, n, plan=plan)
    assert _rel(out, _ref(m, idx, n)) <= 1e-6
    go = torch.randn(n, F, generator=g)
    (gm,) = torch.autograd.grad(out, m, go)
    assert torch.equal(gm, go[idx])
    h = torch.randn(n, F, generator=g, requires_grad=True)
    got = trnsparse.gather(h, idx, plan=plan)
    assert torch.equal(got, h.detach()[idx])
    ge = torch.randn(e, F, generator=g)
    (gh,) = torch.autograd.grad(got, h, ge)
    assert _rel(gh, _ref(ge, idx, n)) <= 1e-6
    assert calls == ["segment_sum_tiles", "gather_rows", "gather_rows", "segment_sum_tiles"]


@pytest.mark.parametrize("F", [1, 3, 128, 512, 2048, 5120])
def test_every_exemplar_width(F):
    g = torch.Generator().manual_seed(F)
    n, e = 40, 300
    idx = torch.randint(0, n, (e,), generator=g)
    m = torch.randn(e, F, generator=g)
    assert _rel(trnsparse.segment_sum(m, idx, n), _ref(m, idx, n)) <= 1e-6
    h = torch.randn(n, F, generator=g)
    assert torch.equal(trnsparse.gather(h, idx), h[idx])


def test_empty_segments_and_padding_edges():
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(1, 30, (200,), generator=g)
    idx[idx == 5] = 6
    m = torch.randn(200, 64, generator=g)
    out = trnsparse.segment_sum(m, idx, 31)
    assert out[[0, 5, 30]].abs().max() == 0
    assert _rel(out, _ref(m, idx, 31)) <= 1e-6
    # E-bucket padding with zero messages. On the last segment it leaves every
    # other row's slots where they were, so the result is bitwise unchanged.
    # On an earlier segment it moves later rows' edges to other slots and
    # chunks (a different, still fixed, grouping): equal to fp32 rounding.
    zm = torch.cat([m, torch.zeros(56, 64)])
    last = trnsparse.segment_sum(zm, torch.cat([idx, torch.full((56,), 30)]), 31)
    assert torch.equal(last, out)
    first = trnsparse.segment_sum(zm, torch.cat([idx, torch.zeros(56, dtype=torch.long)]), 31)
    assert _rel(first, _ref(m, idx, 31)) <= 1e-6


def test_index_values_above_2_24():
    # N = 2^24 + 3 (compact layout, 2 computed tiles, ~131k zero tiles): run
    # the kernel on the occupied tiles only by checking the touched rows.
    N = (1 << 24) + 3
    idx = torch.tensor([N - 1, N - 2, (1 << 24) + 1, 1 << 24, 0, N - 1])
    plan = trnsparse.segment_plan(idx, N)
    assert plan.compact
    m = torch.randn(len(idx), 1, generator=torch.Generator().manual_seed(1))
    out = trnsparse.segment_sum(m, idx, N, plan=plan)
    for row in sorted({int(i) for i in idx}):
        torch.testing.assert_close(out[row, 0].double(), m[idx == row, 0].double().sum())
    assert int(torch.count_nonzero(out)) == len({int(i) for i in idx})
