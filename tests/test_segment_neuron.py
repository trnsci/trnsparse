"""Graph segment sum / gather (#139) on hardware (TorchNeuron, Trn2).

The MLIP exemplar's graph block (trnsci/trnsci#170) on the device:

    TRNSPARSE_REQUIRE_NKI=1 pytest tests/test_segment_neuron.py -v -m neuron

- correctness at the exemplar's 216-atom (TiC(001) slab) and 1728-atom (TiC
  bulk) shapes, every feature width, forward and both backwards;
- bitwise determinism over reruns, and across LNC settings: with
  TRNSPARSE_SEG_DUMP=<dir> the determinism case saves its outputs, and a run
  with TRNSPARSE_SEG_COMPARE=<dir> compares against them bit for bit
  (run once at NEURON_LOGICAL_NC_CONFIG=1, once at =2);
- index values above 2^24 (N = 2^24 + 3) and more than 2^24 edges;
- feature-axis chunks (#141): forced chunks bitwise equal to one launch at
  1728 atoms, and a 4096-atom layer-1 call (4.7e9 bytes) on the kernel; the
  padding mask bitwise equal to the plan over the real edges;
- the exemplar checker (``examples/mlip/check.py --candidate trnsparse``)
  when TRNSCI_MLIP_DIR points at a trnsci checkout's ``examples/mlip``;
- a timing harness against aten ``index_add_`` / ``scatter_add_`` (MACE's
  expanded-index pattern) / ``index_select`` on the device (``-k timing -s``;
  TRNSPARSE_SEG_BENCH_JSON=<path> writes the table).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch

import trnsparse
from trnsparse.nki import HAS_NKI
from trnsparse.nki import dispatch as _d


def _native() -> bool:
    try:
        return HAS_NKI and not _d._use_simulator() and _d._get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

DEV = "neuron"
WIDTHS = (1, 3, 128, 512, 2048, 5120)
# (atoms, edges): the exemplar's TiC(001) slab and TiC bulk.
SHAPES = {"tic001_216": (216, 10392), "tic_bulk_1728": (1728, 96278)}


@pytest.fixture(autouse=True)
def nki_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


def _graph(n, e, seed=0):
    """Receivers unsorted, senders sorted (matscipy order), about e / n per atom."""
    g = torch.Generator().manual_seed(seed)
    rcv = torch.randint(0, n, (e,), generator=g)
    snd = torch.sort(torch.randint(0, n, (e,), generator=g)).values
    return snd, rcv, g


def _ref_segsum(m, idx, n, block=128):
    """fp64 host reference, in column blocks (the 1728 x 5120 case is 3.9 GB in fp64)."""
    out = torch.empty(n, m.shape[1], dtype=torch.float64)
    for c in range(0, m.shape[1], block):
        out[:, c : c + block] = torch.zeros(
            n, min(block, m.shape[1] - c), dtype=torch.float64
        ).index_add_(0, idx, m[:, c : c + block].double())
    return out


def _rel(got, ref):
    got = got.detach().cpu().double()
    return float((got - ref).norm() / ref.norm())


def _sync():
    try:
        import torch_neuronx

        torch_neuronx.synchronize()
    except Exception:
        pass


@pytest.mark.parametrize("F", WIDTHS)
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_segment_sum_gather_at_exemplar_shapes(shape, F):
    n, e = SHAPES[shape]
    snd, rcv, g = _graph(n, e, seed=F)
    for idx in (rcv, snd):
        plan = trnsparse.segment_plan(idx, n).to(DEV)
        m = torch.randn(e, F, generator=g)
        md = m.to(DEV).requires_grad_(True)
        out = trnsparse.segment_sum(md, idx.to(DEV), n, plan=plan)
        assert out.device.type == DEV
        assert _rel(out, _ref_segsum(m, idx, n)) <= 2e-5
        go = torch.randn(n, F, generator=g)
        (gm,) = torch.autograd.grad(out, md, go.to(DEV))
        assert torch.equal(gm.cpu(), go[idx])  # the backward is a gather: exact
        h = torch.randn(n, F, generator=g)
        hd = h.to(DEV).requires_grad_(True)
        got = trnsparse.gather(hd, idx.to(DEV), plan=plan)
        assert torch.equal(got.cpu(), h[idx])
        ge = torch.randn(e, F, generator=g)
        (gh,) = torch.autograd.grad(got, hd, ge.to(DEV))
        assert _rel(gh, _ref_segsum(ge, idx, n)) <= 2e-5


def test_force_assembly_1728():
    n, e = SHAPES["tic_bulk_1728"]
    snd, rcv, g = _graph(n, e, seed=3)
    ps, pr = trnsparse.segment_plan(snd, n).to(DEV), trnsparse.segment_plan(rcv, n).to(DEV)
    ev = torch.randn(e, 3, generator=g)
    evd = ev.to(DEV)
    f = trnsparse.segment_sum(evd, snd.to(DEV), n, plan=ps) - trnsparse.segment_sum(
        evd, rcv.to(DEV), n, plan=pr
    )
    ref = _ref_segsum(ev, snd, n) - _ref_segsum(ev, rcv, n)
    assert _rel(f, ref) <= 2e-5


def test_determinism_and_lnc_dump():
    """Five reruns bitwise equal; optionally saved / compared across LNC settings."""
    n, e = SHAPES["tic001_216"]
    snd, rcv, g = _graph(n, e, seed=11)
    plan = trnsparse.segment_plan(rcv, n).to(DEV)
    results = {}
    for F in (3, 2048, 5120):
        m = torch.randn(e, F, generator=g)
        md = m.to(DEV).requires_grad_(True)
        go = torch.randn(n, F, generator=g).to(DEV)
        runs = []
        for _ in range(5):
            out = trnsparse.segment_sum(md, rcv.to(DEV), n, plan=plan)
            (gm,) = torch.autograd.grad(out, md, go)
            h = trnsparse.gather(out.detach(), rcv.to(DEV), plan=plan)
            runs.append((out.detach().cpu(), gm.cpu(), h.cpu()))
        for r in runs[1:]:
            assert all(torch.equal(a, b) for a, b in zip(runs[0], r, strict=True)), F
        results[f"F{F}"] = runs[0]
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "default")
    dump = os.environ.get("TRNSPARSE_SEG_DUMP")
    if dump:
        Path(dump).mkdir(parents=True, exist_ok=True)
        torch.save(results, Path(dump) / f"segment_lnc{lnc}.pt")
    cmp_dir = os.environ.get("TRNSPARSE_SEG_COMPARE")
    if cmp_dir:
        for other in sorted(Path(cmp_dir).glob("segment_lnc*.pt")):
            saved = torch.load(other, weights_only=True)
            for k, v in results.items():
                assert all(torch.equal(a, b) for a, b in zip(v, saved[k], strict=True)), (
                    f"{k} differs from {other.name} (this run: LNC={lnc})"
                )


def test_index_values_above_2_24():
    # N = 2^24 + 3: compact layout (2 computed tiles, ~131k zero tiles).
    N = (1 << 24) + 3
    idx = torch.tensor([N - 1, N - 2, (1 << 24) + 1, 1 << 24, 0, N - 1])
    m = torch.randn(len(idx), 1, generator=torch.Generator().manual_seed(1))
    out = trnsparse.segment_sum(m.to(DEV), idx.to(DEV), N).cpu()
    ref = torch.zeros(N, 1).index_add_(0, idx, m)
    assert torch.equal(out, ref)
    h = torch.arange(N, dtype=torch.float32).reshape(N, 1)  # exact below 2^24 only
    hi = torch.arange(N, dtype=torch.int32).view(torch.float32).reshape(N, 1)  # bit patterns
    got = trnsparse.gather(hi.to(DEV), idx.to(DEV)).cpu()
    assert torch.equal(got.view(torch.int32)[:, 0].long(), idx)  # rows above 2^24 exact
    assert torch.equal(trnsparse.gather(h.to(DEV), idx.to(DEV)).cpu(), h[idx])


def test_more_than_2_24_edges():
    # E = 2^24 + 1000 edges (about 3e5 atoms at 56 per atom), F = 1.
    e = (1 << 24) + 1000
    n = 300_000
    g = torch.Generator().manual_seed(2)
    idx = torch.randint(0, n, (e,), generator=g)
    m = torch.randn(e, 1, generator=g)
    plan = trnsparse.segment_plan(idx, n)
    md = m.to(DEV).requires_grad_(True)
    out = trnsparse.segment_sum(md, idx.to(DEV), n, plan=plan)
    assert _rel(out, _ref_segsum(m, idx, n)) <= 2e-5
    go = torch.randn(n, 1, generator=g)
    (gm,) = torch.autograd.grad(out, md, go.to(DEV))
    assert torch.equal(gm.cpu(), go[idx])  # edge ids above 2^24 gathered exactly


def _run_pair(idx, n, F, plan, g):
    """segment_sum, its grad, gather and its grad on the device (host results)."""
    e = idx.numel()
    idx_d = idx.to(DEV)
    m, go = torch.randn(e, F, generator=g), torch.randn(n, F, generator=g)
    md = m.to(DEV).requires_grad_(True)
    out = trnsparse.segment_sum(md, idx_d, n, plan=plan)
    (gm,) = torch.autograd.grad(out, md, go.to(DEV))
    res = {"sum": out.detach().cpu(), "gm": gm.cpu()}
    del md, out, gm
    h, ge = torch.randn(n, F, generator=g), torch.randn(e, F, generator=g)
    hd = h.to(DEV).requires_grad_(True)
    got = trnsparse.gather(hd, idx_d, plan=plan)
    (gh,) = torch.autograd.grad(got, hd, ge.to(DEV))
    res.update(got=got.detach().cpu(), gh=gh.cpu())
    return res, (m, go, h, ge)


def _no_pytorch_path(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("took the pytorch path")

    monkeypatch.setattr(trnsparse.segment, "_torch_segment_sum", boom)
    monkeypatch.setattr(trnsparse.segment, "_torch_gather", boom)


def test_feature_split_is_bitwise_1728(monkeypatch):
    """#141: forced feature chunks (aligned 2048 / unaligned 1000 columns)
    equal the one-launch kernel bit for bit, forward and both backwards."""
    seg = trnsparse.segment
    _no_pytorch_path(monkeypatch)
    n, e = SHAPES["tic_bulk_1728"]
    _, rcv, _ = _graph(n, e, seed=141)
    plan = trnsparse.segment_plan(rcv, n).to(DEV)
    one, (m, go, h, ge) = _run_pair(rcv, n, 5120, plan, torch.Generator().manual_seed(1))
    assert _rel(one["sum"], _ref_segsum(m, rcv, n)) <= 2e-5
    assert torch.equal(one["gm"], go[rcv]) and torch.equal(one["got"], h[rcv])
    rows = seg._launch_rows(e, n)
    for patch in (("_SEG_MAX_BYTES", rows * 4 * 3000), ("_feature_chunk", lambda e, n, f: 1000)):
        with monkeypatch.context() as mp:
            mp.setattr(seg, *patch)
            split, _ = _run_pair(rcv, n, 5120, plan, torch.Generator().manual_seed(1))
        for k, v in one.items():
            assert torch.equal(v, split[k]), (patch[0], k)


def test_4096_atoms_layer1_splits_features(monkeypatch):
    """#141: N = 4096, E = 229376 (the exemplar's padded bucket), F = 5120 is
    4.7e9 bytes, above one launch: it runs as 2048 + 2048 + 1024 columns on the
    kernel, never on the pytorch path."""
    _no_pytorch_path(monkeypatch)
    calls = []
    real = _d.nki_segment_op
    monkeypatch.setattr(
        _d, "nki_segment_op", lambda o, x, p: calls.append(x.shape[1]) or real(o, x, p)
    )
    n, e = 4096, 229376
    _, rcv, _ = _graph(n, e, seed=4096)
    plan = trnsparse.segment_plan(rcv, n).to(DEV)
    res, (m, go, h, ge) = _run_pair(rcv, n, 5120, plan, torch.Generator().manual_seed(2))
    assert calls == [2048, 2048, 1024] * 2
    assert _rel(res["sum"], _ref_segsum(m, rcv, n)) <= 2e-5
    assert torch.equal(res["gm"], go[rcv])
    assert torch.equal(res["got"], h[rcv])
    assert _rel(res["gh"], _ref_segsum(ge, rcv, n)) <= 2e-5


def test_padding_mask_bitwise_1728():
    """#141: E padded to a 98304 bucket on node 0. With the mask the plan has
    the real edges' slot count and the sums are bitwise those of the real
    edges (NaN padding messages are never read)."""
    n, e = SHAPES["tic_bulk_1728"]
    _, rcv, g = _graph(n, e, seed=17)
    pad = 98304 - e
    pidx = torch.cat([rcv, torch.zeros(pad, dtype=torch.long)])
    mask = torch.cat([torch.zeros(e, dtype=torch.bool), torch.ones(pad, dtype=torch.bool)])
    real = trnsparse.segment_plan(rcv, n).to(DEV)
    masked = trnsparse.segment_plan(pidx, n, padding=mask).to(DEV)
    assert masked.slots == real.slots < trnsparse.segment_plan(pidx, n).slots
    m = torch.randn(e, 512, generator=g)
    mp = torch.cat([m, torch.full((pad, 512), float("nan"))])
    a = trnsparse.segment_sum(m.to(DEV), rcv.to(DEV), n, plan=real).cpu()
    b = trnsparse.segment_sum(mp.to(DEV), pidx.to(DEV), n, plan=masked).cpu()
    assert torch.equal(a, b)
    hd = torch.randn(n, 512, generator=g).to(DEV).requires_grad_(True)
    ge = torch.randn(e, 512, generator=g)
    gep = torch.cat([ge, torch.full((pad, 512), float("nan"))])
    (ga,) = torch.autograd.grad(trnsparse.gather(hd, rcv.to(DEV), plan=real), hd, ge.to(DEV))
    (gb,) = torch.autograd.grad(trnsparse.gather(hd, pidx.to(DEV), plan=masked), hd, gep.to(DEV))
    assert torch.equal(ga.cpu(), gb.cpu())


def test_exemplar_checker():
    """``check.py --candidate trnsparse --device neuron --blocks B --edge-cases``."""
    root = os.environ.get("TRNSCI_MLIP_DIR")
    if not root:
        pytest.skip("set TRNSCI_MLIP_DIR to a trnsci checkout's examples/mlip")
    sys.path.insert(0, root)
    import check  # noqa: E402

    captures = sorted(Path(root).glob("fixtures/*.pt"))
    extra = os.environ.get("TRNSCI_MLIP_CAPTURES")
    if extra:
        captures += sorted(Path(extra).glob("*.pt"))
    chk = check.Checker(trnsparse, DEV, 2e-5)
    for path in captures:
        chk.graph(check.load(str(path)))
    check.edge_cases(trnsparse, DEV, chk)
    bad = [r for r in chk.rows if r[3] != "ok"]
    print("\n".join(f"{st:4s} {err:9.2e}  {name}: {q}" for name, q, err, st in chk.rows))
    assert not bad, bad


# ── timing harness ───────────────────────────────────────────────────────────

# aten scatter_add_ with an expanded index on trn2 (SDK 2.33): at the 216-atom
# shape the first call (compile) took 46 s at F = 512, 108 s at F = 2048 and
# 267 s at F = 5120, and each later call 1.0 s, 3.9 s and 9.9 s (#140).
# Larger E * F would take most of a timing window.
_SCATTER_ADD_MAX_EF = 6 * 10**7


def _time(fn, reps=10):
    t0 = time.perf_counter()
    out = fn()
    _sync()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    fn()
    _sync()
    if time.perf_counter() - t0 > 0.25:
        # aten scatter_add_ with an expanded index takes seconds per call at
        # E * F ~ 1e8 on trn2 (#140); two timed calls are enough.
        reps = 2
    else:
        fn()
        _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    _sync()
    return first, (time.perf_counter() - t0) / reps, out


@pytest.mark.parametrize("E", [10_000, 100_000, 1_000_000])
def test_timing_vs_aten(E):
    """Time per call (and first call, which includes compile) against device aten.

    aten candidates: ``zeros.index_add_(0, idx, m)``, MACE's
    ``zeros.scatter_add_(0, idx.expand_as(m), m)`` (stride-0 int64 index) and
    ``h.index_select(0, idx)``. Also records whether each is bitwise
    deterministic over two calls and its error against the host fp64 sum.
    """
    n = max(1, E // 56)
    rows = []
    g = torch.Generator().manual_seed(E)
    idx = torch.randint(0, n, (E,), generator=g)
    idx_d = idx.to(DEV)
    plan = trnsparse.segment_plan(idx, n).to(DEV)
    for F in (1, 3, 512, 2048, 5120):
        if E * F * 4 >= 2**31:
            # Feature-split launches (#141); aten at this size is not timed.
            rows.append({"E": E, "F": F, "note": "above one launch's 2^31 bytes: split on F"})
            continue
        m = torch.randn(E, F, generator=g)
        md = m.to(DEV)
        ref = _ref_segsum(m, idx, n)
        cands = {
            "trnsparse.segment_sum": lambda md=md: trnsparse.segment_sum(md, idx_d, n, plan=plan),
            "aten.index_add_": lambda md=md: torch.zeros(n, md.shape[1], device=DEV).index_add_(
                0, idx_d, md
            ),
            "aten.scatter_add_(expand)": lambda md=md: torch.zeros(
                n, md.shape[1], device=DEV
            ).scatter_add_(0, idx_d.unsqueeze(1).expand_as(md), md),
        }
        h = md[:n].contiguous() if E >= n else torch.randn(n, F).to(DEV)
        gcands = {
            "trnsparse.gather": lambda h=h: trnsparse.gather(h, idx_d, plan=plan),
            "aten.index_select": lambda h=h: h.index_select(0, idx_d),
        }
        for name, fn in {**cands, **gcands}.items():
            row = {"E": E, "F": F, "op": name}
            if name.startswith("aten.scatter_add_") and E * F > _SCATTER_ADD_MAX_EF:
                row["note"] = "not run: E*F above the size where its compile takes minutes"
                rows.append(row)
                continue
            try:
                first, per, out = _time(fn)
                again = fn()
                _sync()
                row.update(
                    first_ms=1e3 * first,
                    per_call_ms=1e3 * per,
                    bitwise_rerun=bool(torch.equal(out.cpu(), again.cpu())),
                )
                if name in cands:
                    row["rel_err_vs_fp64"] = _rel(out, ref)
                else:
                    row["exact"] = bool(torch.equal(out.cpu(), h.cpu()[idx]))
            except Exception as exc:  # noqa: BLE001 - a device aten failure is a result
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rows.append(row)
    for r in rows:
        print(json.dumps(r))
    path = os.environ.get("TRNSPARSE_SEG_BENCH_JSON")
    if path:
        prev = json.loads(Path(path).read_text()) if Path(path).exists() else []
        Path(path).write_text(json.dumps(prev + rows, indent=1))
    ours = [r for r in rows if r.get("op", "").startswith("trnsparse")]
    assert all("error" not in r for r in ours), ours
    assert all(r["bitwise_rerun"] for r in ours), ours
