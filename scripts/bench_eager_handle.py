"""Host cost of one native kernel call: the `trnsparse::` custom op against the
cached eager handle (`TRNSPARSE_NKI_EAGER_HANDLE`, trnblas#92), with aten for
scale, on device-resident inputs.

``issue`` is the host time per call of n back-to-back calls; ``e2e`` the
median of one call plus a synchronize; ms. ``same_as_op`` compares the
handle's result with the op's bit for bit. Inputs are created on the host
and copied to the device (aws-neuron/aws-neuron-sdk#1439).

    python scripts/bench_eager_handle.py
"""

from __future__ import annotations

import statistics
import time

import torch


def _sync():
    import torch_neuronx

    sync = getattr(torch_neuronx, "synchronize", None)
    if sync is None:
        sync = torch_neuronx.neuron.synchronize
    sync()


def _issue_e2e(fn, n=200, e2e_reps=100):
    for _ in range(10):
        fn()
    _sync()
    issue = []
    for _ in range(5):
        t0 = time.perf_counter()
        rs = [fn() for _ in range(n)]
        issue.append((time.perf_counter() - t0) / n * 1e3)
        _sync()
        del rs
    e2e = []
    for _ in range(e2e_reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        e2e.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(issue), statistics.median(e2e)


def _host(out):
    if isinstance(out, tuple | list):
        return tuple(_host(o) for o in out)
    return out.detach().cpu() if isinstance(out, torch.Tensor) else out


def _same(a, b):
    if isinstance(a, tuple):
        return all(_same(x, y) for x, y in zip(a, b, strict=True))
    return torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b


def run(D, cases, n=200):
    """cases: (name, fn, aten) — aten=True runs once (no op/handle toggle)."""
    print(f"{'case':44s} {'path':7s} {'issue':>8s} {'e2e':>8s}  same_as_op", flush=True)
    for name, fn, aten, *kw in cases:
        reps = kw[0] if kw else n
        paths = ("aten",) if aten else ("op", "handle")
        ref = None
        for path in paths:
            D._EAGER_HANDLE_ENABLED = path != "op"
            out = _host(fn())
            same = ""
            if path == "op":
                ref = out
            elif path == "handle":
                same = "yes" if _same(out, ref) else "NO"
            del out
            issue, e2e = _issue_e2e(fn, n=reps, e2e_reps=min(100, 5 * reps))
            print(f"{name:44s} {path:7s} {issue:8.4f} {e2e:8.4f}  {same}", flush=True)
    D._EAGER_HANDLE_ENABLED = True


def main():
    import trnsparse
    from trnsparse.nki import dispatch as D

    trnsparse.set_backend("nki")
    P = 128
    g = torch.Generator().manual_seed(0)
    a, b = (
        torch.randn(256, 256, generator=g).to("neuron"),
        torch.randn(256, 128, generator=g).to("neuron"),
    )
    bm = torch.zeros(6, 4, dtype=torch.bool)
    bm[0, :] = True
    for i in range(1, 6):
        bm[i, i % 4] = bm[i, (3 * i + 1) % 4] = True
    dense = torch.randn(6 * P, 4 * P, generator=g) * bm.repeat_interleave(P, 0).repeat_interleave(
        P, 1
    )
    Ab = trnsparse.BSRMatrix.from_dense(dense, block_size=P)
    Ab.blocks = Ab.blocks.to("neuron")
    Bb = torch.randn(4 * P, 256, generator=g).to("neuron")
    seq = 3 * P
    pat = torch.zeros(seq, seq)
    for i in range(3):
        pat[i * P : (i + 1) * P, : (i + 1) * P] = 1.0
    mask = trnsparse.BSRMatrix.from_dense(pat, block_size=P)
    Q, K, V = (torch.randn(seq, 64, generator=g).to("neuron") for _ in range(3))
    M = torch.randn(256, 256, generator=g)
    Sp = trnsparse.BSRMatrix.from_dense(M @ M.T / 256 + torch.eye(256) * 4.0, block_size=P)
    rhs = torch.randn(256, generator=g).to("neuron")
    run(
        D,
        [
            ("aten a@b 256x256x128", lambda: a @ b, True),
            (
                "spmm_dense 256x256x128 (_launch)",
                lambda: D._launch(None, "spmm_dense", a, b),
                False,
            ),
            (
                "bsr_spmm 6x4 blocks, N=256 (keyed, int key)",
                lambda: trnsparse.bsr_spmm(Ab, Bb),
                False,
            ),
            (
                "attention fwd seq=384 hd=64 (keyed)",
                lambda: trnsparse.block_sparse_attention_tiled(Q, K, V, mask),
                False,
                50,
            ),
            (
                "cg_bsr_fused n=256 (keyed, int settings)",
                lambda: trnsparse.cg_bsr_fused(Sp, rhs, tol=1e-5, max_iter=200),
                False,
                20,
            ),
        ],
    )


if __name__ == "__main__":
    main()
