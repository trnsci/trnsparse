"""CPU tests for the graph gather / segment-sum pair (#139, MLIP exemplar trnsci/trnsci#170).

- The pytorch path against the exemplar's reference blocks (``index_add_`` /
  ``index_select``, examples/mlip/blocks.py in trnsci/trnsci) on a vendored
  slice of the MACE-MP-0b2 medium capture, against ``torch.index_add`` and
  against an fp64 one-hot product.
- fp64 gradcheck / gradgradcheck, bitwise determinism, unsorted receivers,
  empty segments, zero edges, heavy repetition, every feature width of the
  exemplar (1, 3, 128, 512, 2048, 5120).
- The `SegmentPlan` tile layout, checked by running the kernel's loops in
  torch on the plan's tables (`_emulate_*`). This covers the plan logic on
  hosts without NKI. The kernels themselves run in tests/test_nki_segment_sim.py
  (simulator) and tests/test_segment_neuron.py (hardware).
- Index values above 2^24 and E*F above 2^24: the int64 host math and the
  int32 tables stay exact.

The fixture ``fixtures/mlip_graph_tiny.pt`` holds the graph (B) blocks of the
exemplar's committed CI fixture (16-atom CH3COOH dimer, public MACE-MP-0b2
medium checkpoint, mace-torch 0.3.15).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import trnsparse as ts
from trnsparse import segment as seg
from trnsparse.nki import dispatch

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mlip_graph_tiny.pt"
WIDTHS = (1, 3, 128, 512, 2048, 5120)
P = 128


@pytest.fixture(autouse=True)
def _pytorch_backend():
    prev = ts.get_backend()
    ts.set_backend("pytorch")
    yield
    ts.set_backend(prev)


def rel(got, ref) -> float:
    got, ref = got.detach().double(), ref.detach().double()
    den = ref.norm()
    return float((got - ref).norm() / den) if den else float((got - ref).norm())


# The exemplar's reference blocks (examples/mlip/blocks.py), verbatim semantics.
def ref_segment_sum(m, idx, n):
    out = m.new_zeros((n,) + tuple(m.shape[1:]))
    return out.index_add_(0, idx, m)


def ref_gather(h, idx):
    return h.index_select(0, idx)


def onehot_fp64(m, idx, n):
    """Independent reference: the dense incidence product in float64."""
    A = torch.zeros(n, idx.numel(), dtype=torch.float64)
    A[idx, torch.arange(idx.numel())] = 1.0
    return A @ m.double().reshape(idx.numel(), -1)


def _graph(n, e, seed=0, lo=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, n, (e,), generator=g), g


# ── exemplar fixture ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cap():
    return torch.load(FIXTURE, map_location="cpu", weights_only=True)


def test_fixture_contract(cap):
    assert cap["sender"].dtype == torch.int64 and cap["receiver"].dtype == torch.int64
    assert cap["stats"]["sender_sorted"] and not cap["stats"]["receiver_sorted"]
    assert FIXTURE.stat().st_size < 200 * 1024


def test_fixture_segment_sum_matches_capture(cap):
    for s in cap["B_scatter"]:
        m = s["src"].clone().requires_grad_(True)
        out = ts.segment_sum(m, s["index"], s["dim_size"])
        assert torch.equal(out, ref_segment_sum(s["src"], s["index"], s["dim_size"])), s["name"]
        assert rel(out, s["out"]) <= 2e-5, s["name"]
        (gm,) = torch.autograd.grad(out, m, s["grad_out"])
        assert rel(gm, s["grad_src"]) <= 2e-5, s["name"]
        assert torch.equal(gm, ref_gather(s["grad_out"], s["index"])), s["name"]


def test_fixture_gather_matches_capture(cap):
    for gth in cap["B_gather"]:
        h = gth["src"].clone().requires_grad_(True)
        out = ts.gather(h, gth["index"])
        assert torch.equal(out, gth["out"]), gth["name"]
        if gth["grad_src"] is not None:
            (gh,) = torch.autograd.grad(out, h, gth["grad_out"])
            assert rel(gh, gth["grad_src"]) <= 2e-5, gth["name"]


def test_fixture_force_assembly(cap):
    # F = segsum(g, sender) - segsum(g, receiver), g = dE/d(edge vector).
    g, snd, rcv = cap["edge_vectors_grad"], cap["sender"], cap["receiver"]
    n = cap["stats"]["atoms"]
    f = ts.segment_sum(g, snd, n) - ts.segment_sum(g, rcv, n)
    assert torch.equal(f, ref_segment_sum(g, snd, n) - ref_segment_sum(g, rcv, n))
    assert rel(f, cap["forces"]) <= 2e-5


def test_fixture_plans_reused(cap):
    # One plan per index, reused across every call of the step (sorted senders
    # need no permutation).
    n = cap["stats"]["atoms"]
    ps, pr = ts.segment_plan(cap["sender"], n), ts.segment_plan(cap["receiver"], n)
    assert ps.is_sorted and not pr.is_sorted
    assert torch.equal(ps.perm, torch.arange(cap["sender"].numel()))
    g = cap["edge_vectors_grad"]
    f = ts.segment_sum(g, cap["sender"], n, plan=ps) - ts.segment_sum(
        g, cap["receiver"], n, plan=pr
    )
    assert rel(f, cap["forces"]) <= 2e-5


# ── against torch.index_add / fp64 ──────────────────────────────────────────


@pytest.mark.parametrize("F", WIDTHS)
def test_segment_sum_every_width(F):
    idx, g = _graph(40, 300, seed=F)
    m = torch.randn(300, F, generator=g)
    out = ts.segment_sum(m, idx, 40)
    assert torch.equal(out, torch.zeros(40, F).index_add(0, idx, m))
    assert rel(out, onehot_fp64(m, idx, 40)) <= 1e-6


@pytest.mark.parametrize("F", WIDTHS)
def test_gather_every_width(F):
    idx, g = _graph(40, 300, seed=F)
    h = torch.randn(40, F, generator=g)
    assert torch.equal(ts.gather(h, idx), h[idx])


def test_int32_index_and_trailing_dims():
    idx, g = _graph(9, 50)
    m = torch.randn(50, 2, 3, generator=g)
    out = ts.segment_sum(m, idx.to(torch.int32), 9)
    assert out.shape == (9, 2, 3)
    assert torch.equal(out, ref_segment_sum(m, idx, 9))
    v = torch.randn(50, generator=g)  # 1-D values (F = 1, squeezed)
    assert torch.equal(ts.segment_sum(v, idx, 9), ref_segment_sum(v, idx, 9))
    h = torch.randn(9, 4, 2, generator=g)
    assert torch.equal(ts.gather(h, idx.to(torch.int32)), h[idx])


def test_unsorted_equals_sorted_bitwise():
    # A stable sort keeps each segment's edges in their relative order, so the
    # sequential sum is the same bits.
    idx, g = _graph(50, 2000, seed=3)
    m = torch.randn(2000, 64, generator=g)
    perm = torch.argsort(idx, stable=True)
    assert torch.equal(ts.segment_sum(m, idx, 50), ts.segment_sum(m[perm], idx[perm], 50))


def test_deterministic_rerun():
    idx, g = _graph(216, 10392, seed=4)  # the TiC(001) slab's shape, receiver unsorted
    m = torch.randn(10392, 128, generator=g)
    a = ts.segment_sum(m, idx, 216)
    for _ in range(3):
        assert torch.equal(a, ts.segment_sum(m, idx, 216))
    mm = m.clone().requires_grad_(True)
    go = torch.randn(216, 128, generator=g)
    (g1,) = torch.autograd.grad(ts.segment_sum(mm, idx, 216), mm, go)
    (g2,) = torch.autograd.grad(ts.segment_sum(mm, idx, 216), mm, go)
    assert torch.equal(g1, g2)


# ── edge cases (check.py --edge-cases) ───────────────────────────────────────


def test_empty_segments():
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(1, 30, (200,), generator=g)
    idx[idx == 5] = 6
    m = torch.randn(200, 64, generator=g, requires_grad=True)
    out = ts.segment_sum(m, idx, 31)
    assert out[[0, 5, 30]].abs().max() == 0
    assert rel(out, onehot_fp64(m, idx, 31)) <= 1e-6
    go = torch.randn(31, 64, generator=g)
    (gm,) = torch.autograd.grad(out, m, go)
    assert torch.equal(gm, go[idx])


def test_zero_edges():
    m = torch.zeros(0, 64, requires_grad=True)
    idx = torch.zeros(0, dtype=torch.long)
    out = ts.segment_sum(m, idx, 7)
    assert out.shape == (7, 64) and out.abs().max() == 0
    (gm,) = torch.autograd.grad(out, m, torch.ones(7, 64))
    assert gm.shape == (0, 64)
    h = torch.randn(7, 16, requires_grad=True)
    out = ts.gather(h, idx)
    assert out.shape == (0, 16)
    (gh,) = torch.autograd.grad(out.sum() + h.sum() * 0, h)
    assert gh.shape == (7, 16)


def test_one_segment_heavy_repetition():
    g = torch.Generator().manual_seed(1)
    m = torch.randn(4096, 64, generator=g)
    idx = torch.zeros(4096, dtype=torch.long)
    out = ts.segment_sum(m, idx, 3)
    assert rel(out, onehot_fp64(m, idx, 3)) <= 1e-5
    assert out[1:].abs().max() == 0


def test_self_image_edges_and_repeated_gather():
    # sender == receiver (a self-image edge) needs no special case.
    snd = torch.tensor([0, 0, 1, 2, 2, 2])
    rcv = torch.tensor([0, 1, 1, 2, 0, 2])
    gen = torch.Generator().manual_seed(14)
    g = torch.randn(6, 3, generator=gen)
    f = ts.segment_sum(g, snd, 3) - ts.segment_sum(g, rcv, 3)
    assert torch.equal(f, ref_segment_sum(g, snd, 3) - ref_segment_sum(g, rcv, 3))
    h = torch.randn(40, 16, generator=gen, requires_grad=True)
    idx = torch.tensor([3, 3, 3, 0, 39, 7, 3])
    out = ts.gather(h, idx)
    assert torch.equal(out, h.detach()[idx])
    go = torch.randn(7, 16, generator=gen)
    (gh,) = torch.autograd.grad(out, h, go)
    assert torch.equal(gh, ref_segment_sum(go, idx, 40))


def test_padded_edges_add_exactly_zero():
    # E-bucket padding: zero messages on extra edges change nothing (bitwise).
    idx, g = _graph(30, 100)
    m = torch.randn(100, 8, generator=g)
    pad_idx = torch.cat([idx, torch.zeros(28, dtype=torch.long)])
    pad_m = torch.cat([m, torch.zeros(28, 8)])
    assert torch.equal(ts.segment_sum(m, idx, 30), ts.segment_sum(pad_m, pad_idx, 30))


def test_flat_index_above_2_24():
    # E * F = 20.99M > 2^24 (a 2^24 boundary in flattened scatter lowering).
    E, F = 4100, 5120
    idx, g = _graph(120, E, seed=7)
    m = torch.randn(E, F, generator=g)
    out = ts.segment_sum(m, idx, 120)
    assert torch.equal(out, torch.zeros(120, F).index_add(0, idx, m))
    plan = ts.segment_plan(idx, 120)
    assert rel(_emulate_segment_sum(m, plan), onehot_fp64(m, idx, 120)) <= 1e-6


def test_index_values_above_2_24():
    # N = 2^24 + 3 rows: index values not exactly representable in fp32 stay exact.
    N = (1 << 24) + 3
    idx = torch.tensor([N - 1, N - 2, (1 << 24) + 1, 1 << 24, 0, N - 1])
    assert len({float(torch.tensor(float(i), dtype=torch.float32)) for i in idx[1:4]}) < 3
    m = torch.randn(len(idx), 1, generator=torch.Generator().manual_seed(2))
    out = ts.segment_sum(m, idx, N)
    assert torch.equal(out[N - 1], m[0] + m[5])
    # N - 2 == 2^24 + 1: rows 1 and 2 share a segment; 2^24 + 1 rounds to 2^24 in fp32.
    assert torch.equal(out[(1 << 24) + 1], m[1] + m[2]) and torch.equal(out[1 << 24], m[3])
    nz = torch.nonzero(out[:, 0]).flatten()
    assert torch.equal(nz, torch.tensor(sorted({int(i) for i in idx})))
    h = torch.arange(N, dtype=torch.float64).reshape(N, 1)
    assert torch.equal(ts.gather(h, idx)[:, 0], idx.double())
    # The plan: compact (two occupied tiles out of 131073), int32 rows exact.
    plan = ts.segment_plan(idx, N)
    assert plan.compact and plan.computed_tiles == 2
    gidx, src, local, orow, zrow = plan.tables()
    assert gidx.dtype == src.dtype == orow.dtype == zrow.dtype == torch.int32
    assert torch.equal(seg._gather_natural(gidx)[: len(idx), 0].long(), idx)
    rows = torch.cat([orow, zrow[: plan.zero_tiles * P]]).flatten().long()
    assert torch.equal(torch.sort(rows).values, torch.arange(plan.rows_out))
    got = _emulate_segment_sum(m, plan)
    assert torch.equal(got[N - 1], (m[0] + m[5]).double()) and got.shape[0] == N
    assert torch.equal(_emulate_gather(h, plan)[:, 0], idx.double())


# ── autograd ─────────────────────────────────────────────────────────────────


def test_gradcheck_fp64():
    idx, g = _graph(7, 25, seed=5)
    m = torch.randn(25, 3, dtype=torch.float64, generator=g, requires_grad=True)
    h = torch.randn(7, 3, dtype=torch.float64, generator=g, requires_grad=True)
    assert gradcheck(lambda x: ts.segment_sum(x, idx, 7), (m,))
    assert gradgradcheck(lambda x: ts.segment_sum(x, idx, 7), (m,))
    assert gradcheck(lambda x: ts.gather(x, idx), (h,))
    assert gradgradcheck(lambda x: ts.gather(x, idx), (h,))


def test_each_is_the_others_backward():
    idx, g = _graph(11, 60, seed=6)
    m = torch.randn(60, 4, generator=g, requires_grad=True)
    h = torch.randn(11, 4, generator=g, requires_grad=True)
    go_n, go_e = torch.randn(11, 4, generator=g), torch.randn(60, 4, generator=g)
    (gm,) = torch.autograd.grad(ts.segment_sum(m, idx, 11), m, go_n)
    assert torch.equal(gm, ts.gather(go_n, idx))
    (gh,) = torch.autograd.grad(ts.gather(h, idx), h, go_e)
    assert torch.equal(gh, ts.segment_sum(go_e, idx, 11))


def test_forces_through_autograd_grad():
    # The exemplar's checker differentiates through the candidate with
    # torch.autograd.grad: positions -> edge vectors -> energy.
    g = torch.Generator().manual_seed(8)
    n = 12
    pos = torch.randn(n, 3, generator=g, dtype=torch.float64, requires_grad=True)
    snd, _ = _graph(n, 80, seed=9)
    snd = torch.sort(snd).values
    rcv, _ = _graph(n, 80, seed=10)
    vec = ts.gather(pos, rcv) - ts.gather(pos, snd)
    w = ts.segment_sum((vec**2).sum(1, keepdim=True), rcv, n)
    (f,) = torch.autograd.grad(-(w**2).sum(), pos)
    vec_r = pos[rcv] - pos[snd]
    w_r = ref_segment_sum((vec_r**2).sum(1, keepdim=True), rcv, n)
    (f_r,) = torch.autograd.grad(-(w_r**2).sum(), pos)
    assert rel(f, f_r) <= 1e-12


def test_no_grad_to_index():
    idx, g = _graph(5, 20)
    m = torch.randn(20, 2, generator=g, requires_grad=True)
    out = ts.segment_sum(m, idx, 5)
    assert out.requires_grad and not idx.requires_grad


def test_integer_values():
    idx, g = _graph(6, 30)
    m = torch.randint(-5, 6, (30, 2), generator=g)
    assert torch.equal(ts.segment_sum(m, idx, 6), ref_segment_sum(m, idx, 6))
    assert torch.equal(ts.gather(m, idx[:6]), m[idx[:6]])


# ── validation ───────────────────────────────────────────────────────────────


def test_validation_errors():
    m = torch.randn(10, 2)
    with pytest.raises(ValueError):
        ts.segment_sum(m, torch.zeros(9, dtype=torch.long), 3)
    with pytest.raises(ValueError):
        ts.segment_sum(m, torch.zeros(10, 1, dtype=torch.long), 3)
    with pytest.raises(TypeError):
        ts.segment_plan(torch.zeros(10), 3)
    with pytest.raises(IndexError):
        ts.segment_plan(torch.tensor([0, 3]), 3)
    with pytest.raises(IndexError):
        ts.segment_plan(torch.tensor([-1, 0]), 3)
    plan = ts.segment_plan(torch.zeros(10, dtype=torch.long), 3)
    with pytest.raises(ValueError, match="plan is for"):
        ts.segment_sum(m, torch.zeros(10, dtype=torch.long), 4, plan=plan)
    with pytest.raises(TypeError):
        ts.segment_sum(m, torch.zeros(10, dtype=torch.long), 3, plan="nope")


def test_plan_rejects_int32_overflow(monkeypatch):
    monkeypatch.setattr(seg, "_INT32_MAX", 100)
    with pytest.raises(ValueError, match="2\\^31"):
        ts.segment_plan(torch.zeros(101, dtype=torch.long), 3)


# ── the plan's kernel layout, emulated in torch ──────────────────────────────


def _emulate_segment_sum(m: torch.Tensor, plan) -> torch.Tensor:
    """`_segment_sum_kernel`'s loops in torch (fp64), on the plan's own tables.

    Tile k, chunk c, slot j: onehot[p, i] = (local[p] == i); out_tile +=
    onehot^T @ m[src[p]]; each computed tile stored at its `orow` rows, each
    zero tile at its `zrow` rows; every output row written exactly once.
    """
    gidx, src, local, orow, zrow = plan.tables()
    m2 = m.reshape(m.shape[0], -1).double()
    F = m2.shape[1]
    K = orow.shape[0] // P
    nc = src.shape[0] // (K * P)
    out = torch.full((plan.rows_out, F), float("nan"), dtype=torch.float64)
    written = torch.zeros(plan.rows_out, dtype=torch.int64)
    iota = torch.arange(P, dtype=torch.float32)
    for k in range(K):
        acc = torch.zeros(P, F, dtype=torch.float64)
        for c in range(nc):
            r0 = (k * nc + c) * P
            for j in range(src.shape[1]):
                rows = m2[src[r0 : r0 + P, j].long()]
                oh = (local[r0 : r0 + P, j : j + 1] == iota).double()
                acc += oh.T @ rows
        dst = orow[k * P : (k + 1) * P, 0].long()
        out[dst] = acc
        written[dst] += 1
    for z in range(plan.zero_tiles):
        dst = zrow[z * P : (z + 1) * P, 0].long()
        out[dst] = 0.0
        written[dst] += 1
    assert torch.equal(written, torch.ones_like(written)), "a row written != once"
    return out[: plan.num_segments]


def _emulate_gather(h: torch.Tensor, plan) -> torch.Tensor:
    """The gather kernel's reads: per block of each level, the (128, M) index
    tile in column-major (DMA batching) order."""
    from trnsparse.nki.kernels import _gather_split

    gidx = plan.tables()[0]
    ep = gidx.shape[0]
    assert ep % (P * 2) == 0
    order, t = [], 0
    for t0, M, nb in _gather_split(ep):
        assert t0 == t and nb % 2 == 0
        gv = gidx.reshape(ep // M, M)
        order += [gv[(t0 // M + b) * P : (t0 // M + b + 1) * P].T.flatten() for b in range(nb)]
        t += nb * M
    rows = torch.cat(order).long()
    assert rows.numel() == ep
    return h.reshape(h.shape[0], -1)[rows][: plan.num_edges]


@pytest.mark.parametrize(
    "n,e,lo,sorted_",
    [
        (16, 150, 0, False),  # the exemplar's CH3COOH dimer shape
        (64, 2156, 0, True),  # Cu(111) slab, sorted senders
        (216, 10392, 0, False),  # TiC(001) slab
        (300, 1, 0, False),  # one edge
        (129, 600, 0, False),  # a ragged last tile
        (128 * 40, 500, 128 * 30, False),  # mostly empty tiles: compact layout
        (128 * 41, 500, 128 * 39, True),  # compact, odd occupied count
        (3, 4096, 0, True),  # one row, 32 slots: four chunks
        (1728, 16384, 0, False),  # gather: two 32-column blocks only
        (1728, 128 * 126, 0, False),  # gather: one block pair at every level
    ],
)
def test_plan_layout_reproduces_segment_sum(n, e, lo, sorted_):
    idx, g = _graph(n, e, seed=n + e, lo=lo)
    if n == 3:
        idx = torch.zeros(e, dtype=torch.long)
    if sorted_:
        idx = torch.sort(idx).values
    plan = ts.segment_plan(idx, n)
    assert plan.is_sorted == bool((idx[1:] >= idx[:-1]).all())
    assert plan.slots % 8 == 0 and plan.computed_tiles % 2 == 0 and plan.zero_tiles % 2 == 0
    m = torch.randn(e, 5, generator=g)
    assert rel(_emulate_segment_sum(m, plan), onehot_fp64(m, idx, n)) <= 1e-12
    h = torch.randn(n, 5, generator=g)
    assert torch.equal(_emulate_gather(h, plan), h[idx])


def test_plan_tables_dtypes_and_ranges():
    idx, _ = _graph(216, 10392, seed=1)
    plan = ts.segment_plan(idx, 216)
    gidx, src, local, orow, zrow = plan.tables()
    for t in (gidx, src, orow, zrow):
        assert t.dtype == torch.int32  # ids never int64 or float on the device
    assert local.dtype == torch.float32
    assert set(torch.unique(local).tolist()) <= set(range(-1, P))
    assert int(src.min()) >= 0 and int(src.max()) < 10392
    # Every edge appears exactly once among the real slots, in stable order.
    real = local >= 0
    assert torch.equal(torch.sort(src[real].long()).values, torch.arange(10392))
    assert plan.slot_fill > 0.5


def test_plan_slots_bucket_small_changes():
    # 10,392 -> 10,378 edges (an MD rattle): the same slot count, so the same
    # compiled kernel shape apart from E itself.
    a, g = _graph(216, 10392, seed=2)
    keep = torch.randperm(10392, generator=g)[:10378]
    pa, pb = ts.segment_plan(a, 216), ts.segment_plan(a[keep], 216)
    assert (pa.tiles, pa.slots, pa.rows_out) == (pb.tiles, pb.slots, pb.rows_out)


def test_plan_cache_by_content(monkeypatch):
    seg._PLAN_CACHE.clear()
    idx, _ = _graph(50, 400)
    p1 = seg._plan_for(idx, 50)
    assert seg._plan_for(idx.clone(), 50) is p1
    assert seg._plan_for(idx, 51) is not p1
    idx2 = idx.clone()
    idx2[0] = (idx2[0] + 1) % 50
    assert seg._plan_for(idx2, 50) is not p1
    for i in range(seg._PLAN_CACHE_MAX + 2):
        seg._plan_for(torch.full((3,), i % 50), 50)
    assert len(seg._PLAN_CACHE) == seg._PLAN_CACHE_MAX


def test_pytorch_path_builds_no_plan(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the pytorch path needs no plan")

    monkeypatch.setattr(seg, "_plan_for", boom)
    idx, g = _graph(10, 40)
    ts.segment_sum(torch.randn(40, 2, generator=g), idx, 10)
    ts.gather(torch.randn(10, 2, generator=g), idx)


# ── routing (no NKI needed) ──────────────────────────────────────────────────


@pytest.fixture
def no_env_flags(monkeypatch):
    """Routing tests start from TRNSPARSE_REQUIRE_NKI / _USE_SIMULATOR unset."""
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)


def test_fp64_auto_takes_torch(monkeypatch, no_env_flags):
    # The suite fp64 policy: under auto, float64 computes in torch.
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_backend", "auto")
    calls = []
    monkeypatch.setattr(dispatch, "nki_segment_op", lambda *a: calls.append(a))
    idx, g = _graph(10, 40)
    m = torch.randn(40, 2, dtype=torch.float64, generator=g)
    assert torch.equal(ts.segment_sum(m, idx, 10), ref_segment_sum(m, idx, 10))
    assert torch.equal(ts.gather(m[:10], idx), m[idx])
    assert calls == []


def test_kernel_failure_falls_back_unless_required(monkeypatch, no_env_flags):
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_backend", "nki")
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
    monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")

    def fail(*a):
        raise RuntimeError("no device")

    monkeypatch.setattr(dispatch, "nki_segment_op", fail)
    idx, g = _graph(10, 40)
    m = torch.randn(40, 2, generator=g)
    assert torch.equal(ts.segment_sum(m, idx, 10), ref_segment_sum(m, idx, 10))
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    with pytest.raises(RuntimeError, match="no device"):
        ts.segment_sum(m, idx, 10)


def test_oversized_launch_routes_to_torch(monkeypatch, no_env_flags):
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_backend", "nki")
    monkeypatch.setattr(seg, "_SEG_MAX_BYTES", 1024)
    monkeypatch.setattr(dispatch, "nki_segment_op", lambda *a: pytest.fail("kernel launched"))
    idx, g = _graph(10, 400)
    m = torch.randn(400, 2, generator=g)
    assert torch.equal(ts.segment_sum(m, idx, 10), ref_segment_sum(m, idx, 10))
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    with pytest.raises(NotImplementedError, match="byte addressing"):
        ts.segment_sum(m, idx, 10)


def test_kernel_route_through_emulated_launch(monkeypatch, no_env_flags):
    # The non-native kernel route (XLA / simulator shape) with the launch
    # replaced by the torch emulation: plan, padding, slicing, dtype casts
    # and the autograd.Function pair.
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_backend", "nki")
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
    monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")

    def launch(op_name, x, plan):
        assert x.dtype == torch.float32 and x.is_contiguous()
        if op_name == "segment_sum_tiles":
            full = torch.zeros(plan.rows_out, x.shape[1])
            full[: plan.num_segments] = _emulate_segment_sum(x, plan).float()
            return full
        gidx = seg._gather_natural(plan.tables()[0])
        return x[gidx[:, 0].long()]

    monkeypatch.setattr(dispatch, "nki_segment_op", launch)
    idx, g = _graph(216, 3000, seed=12)
    for dt in (torch.float32, torch.bfloat16):
        m = torch.randn(3000, 7, generator=g).to(dt).requires_grad_(True)
        out = ts.segment_sum(m, idx, 216)
        assert out.dtype == dt and out.shape == (216, 7)
        assert rel(out, onehot_fp64(m.detach(), idx, 216)) <= (
            1e-6 if dt == torch.float32 else 1e-2
        )
        go = torch.randn(216, 7, generator=g).to(dt)
        (gm,) = torch.autograd.grad(out, m, go)
        assert torch.equal(gm, go[idx])
        h = torch.randn(216, 7, generator=g).to(dt).requires_grad_(True)
        out = ts.gather(h, idx)
        assert torch.equal(out, h.detach()[idx])
        ge = torch.randn(3000, 7, generator=g).to(dt)
        (gh,) = torch.autograd.grad(out, h, ge)
        assert rel(gh, onehot_fp64(ge, idx, 216)) <= (1e-6 if dt == torch.float32 else 1e-2)


# ── feature-axis chunks (#141) ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "e,n,f,w",
    [
        (96278, 1728, 5120, 5120),  # 1728 atoms, layer 1: one launch (just)
        (229376, 4096, 5120, 2048),  # 4096 atoms: 2048 + 2048 + 1024
        (458752, 8000, 5120, 1024),  # 8000 atoms: five 1024-column launches
        (1_000_000, 17857, 2048, 512),  # E = 1e6: 512-column launches
        (1_000_000, 17857, 512, 512),  # fits
        (10_000_000, 178_571, 128, 53),  # narrower than a PSUM bank: unaligned
        (2**29, 10, 1, 0),  # one column exceeds a launch
    ],
)
def test_feature_chunk_widths(e, n, f, w):
    assert seg._feature_chunk(e, n, f) == w
    if w:
        assert seg._launch_rows(e, n) * w * 4 < seg._SEG_MAX_BYTES


def _reference_launch(calls):
    """`nki_segment_op` replaced by an exact column-independent reference that
    honours the plan's padding (index_add_ in edge order), recording widths."""

    def launch(op_name, x, plan):
        assert x.dtype == torch.float32 and x.is_contiguous()
        calls.append((op_name, int(x.shape[1])))
        if op_name == "segment_sum_tiles":
            full = torch.zeros(plan.rows_out, x.shape[1])
            full[: plan.num_segments] = seg._torch_segment_sum(
                x, plan.index, plan.num_segments, plan
            )
            return full
        gidx = seg._gather_natural(plan.tables()[0])
        return x[gidx[:, 0].long()]

    return launch


@pytest.fixture
def kernel_route(monkeypatch, no_env_flags):
    """The non-native kernel route with a recording reference launch."""
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_backend", "nki")
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")
    calls = []
    monkeypatch.setattr(dispatch, "nki_segment_op", _reference_launch(calls))
    return calls


def _sum_gather_and_grads(idx, n, F, plan, seed):
    g = torch.Generator().manual_seed(seed)
    e = idx.numel()
    m = torch.randn(e, F, generator=g, requires_grad=True)
    out = ts.segment_sum(m, idx, n, plan=plan)
    (gm,) = torch.autograd.grad(out, m, torch.randn(n, F, generator=g))
    h = torch.randn(n, F, generator=g, requires_grad=True)
    got = ts.gather(h, idx, plan=plan)
    (gh,) = torch.autograd.grad(got, h, torch.randn(e, F, generator=g))
    return out.detach(), gm, got.detach(), gh


def test_wide_call_splits_features_bitwise(monkeypatch, kernel_route):
    idx, _ = _graph(216, 3000, seed=21)
    plan = ts.segment_plan(idx, 216)
    one = _sum_gather_and_grads(idx, 216, 5120, plan, seed=4)
    assert {w for _, w in kernel_route} == {5120} and len(kernel_route) == 4
    # A launch budget of 2999 columns at 3072 rows: 2048 + 2048 + 1024.
    monkeypatch.setattr(seg, "_SEG_MAX_BYTES", seg._launch_rows(3000, 216) * 4 * 3000)
    kernel_route.clear()
    split = _sum_gather_and_grads(idx, 216, 5120, plan, seed=4)
    assert [w for _, w in kernel_route] == [2048, 2048, 1024] * 4
    assert all(torch.equal(a, b) for a, b in zip(one, split, strict=True))
    # Unaligned chunks (narrower than a PSUM bank) are bitwise too.
    monkeypatch.setattr(seg, "_SEG_MAX_BYTES", seg._launch_rows(3000, 216) * 4 * 300)
    kernel_route.clear()
    narrow = _sum_gather_and_grads(idx, 216, 1000, plan, seed=5)
    assert [w for _, w in kernel_route[:4]] == [299, 299, 299, 103]
    monkeypatch.setattr(seg, "_SEG_MAX_BYTES", 2**31)
    assert all(
        torch.equal(a, b)
        for a, b in zip(narrow, _sum_gather_and_grads(idx, 216, 1000, plan, seed=5), strict=True)
    )


def test_split_call_matches_independent_reference(monkeypatch, kernel_route):
    idx, g = _graph(300, 2000, seed=22)
    monkeypatch.setattr(seg, "_SEG_MAX_BYTES", seg._launch_rows(2000, 300) * 4 * 700)
    m = torch.randn(2000, 1500, generator=g)
    out = ts.segment_sum(m, idx, 300)
    assert [w for _, w in kernel_route] == [512, 512, 476]
    assert rel(out, onehot_fp64(m, idx, 300)) <= 1e-6
    h = torch.randn(300, 1500, generator=g)
    assert torch.equal(ts.gather(h, idx), h[idx])


# ── padding mask (#141) ──────────────────────────────────────────────────────


def _padded_graph(n=1728, e=20000, pad=4000, seed=5):
    idx, g = _graph(n, e, seed=seed)
    pidx = torch.cat([idx, torch.zeros(pad, dtype=torch.long)])  # all on node 0
    mask = torch.cat([torch.zeros(e, dtype=torch.bool), torch.ones(pad, dtype=torch.bool)])
    return idx, pidx, mask, g


def test_padding_mask_takes_no_slots():
    idx, pidx, mask, _ = _padded_graph()
    real = ts.segment_plan(idx, 1728)
    piled = ts.segment_plan(pidx, 1728)
    masked = ts.segment_plan(pidx, 1728, padding=mask)
    assert piled.slots > real.slots  # node-0 padding inflates every tile
    assert (masked.slots, masked.rows_out, masked.compact) == (
        real.slots,
        real.rows_out,
        real.compact,
    )
    assert masked.num_padding == 4000 and masked.num_edges == 24000
    # Trailing padding: the summing tables are those of the real edges alone.
    for a, b in zip(real.tables()[1:], masked.tables()[1:], strict=True):
        assert torch.equal(a, b)
    assert masked.tables()[0].shape[0] >= 24000  # the gather covers every edge
    spread = ts.segment_plan(torch.cat([idx, torch.arange(4000) % 1728]), 1728)
    assert spread.slots < piled.slots  # the no-mask alternative


def test_padding_mask_interleaved_matches_real_edges():
    # Padding anywhere (not only trailing): the kernel layout sums the real
    # edges in their stable order, bitwise equal to the plan over them alone.
    idx, g = _graph(400, 5000, seed=9)
    mask = torch.rand(5000, generator=g) < 0.2
    plan = ts.segment_plan(idx, 400, padding=mask)
    m = torch.randn(5000, 6, generator=g)
    m[mask] = float("nan")  # never read by a sum
    got = _emulate_segment_sum(m, plan)
    ref = _emulate_segment_sum(m[~mask], ts.segment_plan(idx[~mask], 400))
    assert torch.equal(got, ref)
    assert rel(got, onehot_fp64(m[~mask], idx[~mask], 400)) <= 1e-12
    h = torch.randn(400, 6, generator=g)
    assert torch.equal(_emulate_gather(h, plan), h[idx])


def test_padding_contract_pytorch_path():
    idx, pidx, mask, g = _padded_graph(n=300, e=3000, pad=500)
    plan = ts.segment_plan(pidx, 300, padding=mask)
    m = torch.randn(3500, 4, generator=g, requires_grad=True)
    with torch.no_grad():
        m[3000:] = float("nan")
    out = ts.segment_sum(m, pidx, 300, plan=plan)
    assert torch.equal(out, ref_segment_sum(m.detach()[:3000], idx, 300))
    go = torch.randn(300, 4, generator=g)
    (gm,) = torch.autograd.grad(out, m, go)
    assert torch.equal(gm, go[pidx])  # padded rows are filled, as documented
    h = torch.randn(300, 4, generator=g, requires_grad=True)
    got = ts.gather(h, pidx, plan=plan)
    assert torch.equal(got, h.detach()[pidx])
    ge = torch.randn(3500, 4, generator=g)
    (gh,) = torch.autograd.grad(got, h, ge)
    assert torch.equal(gh, ref_segment_sum(ge[:3000], idx, 300))  # padded cotangent ignored


def test_padding_contract_kernel_route(kernel_route):
    idx, pidx, mask, g = _padded_graph(n=300, e=3000, pad=500)
    plan = ts.segment_plan(pidx, 300, padding=mask)
    m = torch.randn(3500, 4, generator=g)
    m[3000:] = float("nan")
    assert torch.equal(ts.segment_sum(m, pidx, 300, plan=plan), ref_segment_sum(m[:3000], idx, 300))
    h = torch.randn(300, 4, generator=g, requires_grad=True)
    ge = torch.randn(3500, 4, generator=g)
    (gh,) = torch.autograd.grad(ts.gather(h, pidx, plan=plan), h, ge)
    assert torch.equal(gh, ref_segment_sum(ge[:3000], idx, 300))


def test_padding_mask_validation():
    idx, _ = _graph(10, 40)
    with pytest.raises(ValueError, match="bool"):
        ts.segment_plan(idx, 10, padding=torch.zeros(40, dtype=torch.long))
    with pytest.raises(ValueError, match="bool"):
        ts.segment_plan(idx, 10, padding=torch.zeros(39, dtype=torch.bool))
    plan = ts.segment_plan(idx, 10, padding=torch.zeros(40, dtype=torch.bool))
    assert plan.padding is None and plan.keep() is None
    every = ts.segment_plan(idx, 10, padding=torch.ones(40, dtype=torch.bool))
    assert every.num_padding == 40 and not every.compact
    m = torch.randn(40, 3)
    assert torch.equal(_emulate_segment_sum(m, every), torch.zeros(10, 3, dtype=torch.float64))
