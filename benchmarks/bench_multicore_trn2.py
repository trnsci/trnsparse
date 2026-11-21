"""Multi-core (logical NeuronCore) scaling benchmark for Trn2, native TorchNeuron.

Run with every logical core of the chip visible to one process:

    NEURON_RT_VISIBLE_CORES=0-3 NKI_ENABLE_TRACE_CACHE=0 \\
        python benchmarks/bench_multicore_trn2.py --section bsr --size 8192 --widths 512

Sections:
    bsr       static BSR SpMM on 1, 2 and 4 logical cores (`ShardedBSR`,
              block rows balanced by stored blocks) vs dense aten `A @ B` on
              one core and with A's rows split across 2 and 4 cores.
    attn      single-head block-sparse attention forward, query block rows
              split across 1, 2 and 4 cores (`ShardedAttention`), vs dense
              and masked SDPA on one core and with the query rows split
              across 4 cores.
    heads     multi-head block-sparse attention (``--heads``) forward and
              forward+backward with whole heads on different cores, vs
              dense SDPA over the same heads on one core and split across
              cores.

Every case keeps its operands resident on the cores that use them (built
on the host, copied once; TorchNeuron does not copy between logical cores),
issues one launch per core from this thread, and then synchronizes every
core used. That is the time of a sharded call whose inputs and outputs stay
on the cores. The host gather of a sharded result is reported separately
(`*_gather_ms`). Warm median of ``--reps`` calls; the first calls compile.
"""

# ruff: noqa: B023  (every closure is benchmarked before its loop variables change)
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_native_trn2 import (  # noqa: E402
    block_mask,
    mask_bsr_from_block_mask,
    random_bsr,
)

import trnsparse  # noqa: E402
from trnsparse.multicore import ShardedAttention, ShardedBSR, logical_cores  # noqa: E402

P = 128
F = torch.nn.functional


def pick(cores, k):
    """k devices: the first k cores (wrapping onto the visible ones for smoke runs)."""
    return [cores[i % len(cores)] for i in range(k)]


def sync(devs):
    for i in sorted({d.index for d in devs}):
        torch.neuron.synchronize(i)


def bench(fn, devs, warmup=2, reps=10):
    for _ in range(warmup):
        fn()
    sync(devs)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(devs)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def safe(fn, devs, **kw):
    try:
        return bench(fn, devs, **kw)
    except Exception as e:  # report, don't abort the section
        print(f"# bench failed: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
        return None


def issue_time(fn, devs, reps=5):
    """Host time to issue the launches (before any synchronization), median."""
    ts = []
    for _ in range(reps):
        sync(devs)
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    sync(devs)
    return statistics.median(ts)


def cpu_bench(fn, reps=2):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def rel_err(x, ref):
    x = x.detach().cpu().double()
    ref = ref.detach().cpu().double()
    return float((x - ref).norm() / ref.norm().clamp_min(1e-30))


def emit(row, out):
    line = json.dumps(row)
    print(line, flush=True)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


def free():
    gc.collect()


# ── BSR ──────────────────────────────────────────────────────────────────────


def _row_check(A, B, rows):
    """fp64 reference for block rows `rows` of A @ B (host)."""
    ptrs = A.block_row_ptrs.tolist()
    cols = A.block_col_indices.tolist()
    Bd = B.double()
    out = []
    for m in rows:
        acc = torch.zeros(P, B.shape[1], dtype=torch.float64)
        for idx in range(ptrs[m], ptrs[m + 1]):
            j = cols[idx]
            acc += A.blocks[idx].double() @ Bd[j * P : (j + 1) * P]
        out.append(acc)
    return torch.cat(out)


def section_bsr(args, cores):
    n = args.size
    core_counts = [int(c) for c in args.cores.split(",")]
    for width in [int(x) for x in args.widths.split(",")]:
        g = torch.Generator().manual_seed(1)
        Bc = torch.randn(n, width, generator=g)
        B_on = {d: Bc.to(d) for d in cores[: max(core_counts)]}
        # Dense aten, one core and A's rows split across k cores (B on every core).
        Ac = torch.randn(n, n, generator=g) / n**0.5
        dense = {}
        for k in core_counts:
            devs = pick(cores, k)
            slabs = [Ac[i * n // k : (i + 1) * n // k].to(devs[i]) for i in range(k)]
            dense[k] = safe(
                lambda: [s @ B_on[d] for s, d in zip(slabs, devs, strict=True)],
                devs,
                reps=args.reps,
            )
            slabs = None
            free()
        cpu_dense = cpu_bench(lambda: Ac @ Bc) if args.cpu else None
        Ac = None
        free()
        for dens in [float(x) for x in args.densities.split(",")]:
            A = random_bsr(n, dens)
            counts = (A.block_row_ptrs[1:] - A.block_row_ptrs[:-1]).tolist()
            row = {
                "section": "bsr",
                "n": n,
                "width": width,
                "block_density": dens,
                "n_blocks": A.n_blocks,
                "K_max": max(counts),
                "cpu_dense_ms": cpu_dense,
            }
            for k in core_counts:
                row[f"dense_{k}c_ms"] = dense[k]
            # The public one-core API (as in #34 / #36), for continuity.
            d0 = cores[0]
            A0 = trnsparse.BSRMatrix(
                blocks=A.blocks.to(d0),
                block_col_indices=A.block_col_indices,
                block_row_ptrs=A.block_row_ptrs,
                shape=A.shape,
                block_size=P,
            )
            trnsparse.bsr_spmm(A0, B_on[d0])
            row["api_1c_ms"] = safe(lambda: trnsparse.bsr_spmm(A0, B_on[d0]), [d0], reps=args.reps)
            A0 = None
            variants = [("", False)] if not args.only_pretransposed else []
            if args.pretransposed or args.only_pretransposed:
                variants.append(("t_", True))
            for tag, pre in variants:
                for k in core_counts:
                    devs = pick(cores, k)
                    t0 = time.perf_counter()
                    S = ShardedBSR(A, devs, pretransposed=pre)
                    reps_B = [B_on[d] for d in S.devices]
                    outs = S.spmm(reps_B)
                    sync(S.devices)
                    row[f"sparse_{tag}{k}c_first_ms"] = (
                        time.perf_counter() - t0
                    ) * 1e3  # incl. compile
                    row[f"sparse_{tag}{k}c_blocks"] = S.stored_blocks
                    # correctness: two block rows from every shard
                    chk = []
                    for p, rows in enumerate(S.rows):
                        sel = rows[:1] + rows[-1:]
                        o = outs[p].cpu().reshape(len(rows), P, width)
                        pos = [0, len(rows) - 1]
                        chk.append(rel_err(o[pos].reshape(-1, width), _row_check(A, Bc, sel)))
                    row[f"sparse_{tag}{k}c_rel_err"] = max(chk)
                    row[f"sparse_{tag}{k}c_ms"] = safe(
                        lambda: S.spmm(reps_B), S.devices, reps=args.reps
                    )
                    row[f"sparse_{tag}{k}c_issue_ms"] = issue_time(
                        lambda: S.spmm(reps_B), S.devices
                    )
                    if k == max(core_counts) and not pre:
                        row[f"sparse_{tag}{k}c_gather_ms"] = safe(
                            lambda: S.gather(S.spmm(reps_B)), S.devices, reps=max(3, args.reps // 2)
                        )
                    outs = S = None
                    free()
            k1 = min(core_counts)
            for k in core_counts:
                if row.get(f"sparse_t_{k}c_ms") and row.get(f"sparse_{k}c_ms"):
                    row[f"t_speedup_{k}c"] = row[f"sparse_{k}c_ms"] / row[f"sparse_t_{k}c_ms"]
                if row.get(f"sparse_{k}c_ms") and row.get(f"sparse_{k1}c_ms"):
                    row[f"sparse_scaling_{k}c"] = row[f"sparse_{k1}c_ms"] / row[f"sparse_{k}c_ms"]
                if row.get(f"sparse_{k}c_ms") and dense.get(k1):
                    row[f"dense_{k1}c_over_sparse_{k}c"] = dense[k1] / row[f"sparse_{k}c_ms"]
                if row.get(f"sparse_{k}c_ms") and dense.get(k):
                    row[f"dense_{k}c_over_sparse_{k}c"] = dense[k] / row[f"sparse_{k}c_ms"]
            emit(row, args.out)
        B_on = None
        free()


# ── Attention: one head, query rows split ────────────────────────────────────


def _sdpa(q, k, v, m=None):
    return F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=m)[0]


def section_attn(args, cores):
    seq = args.size
    nb = seq // P
    core_counts = [int(c) for c in args.cores.split(",")]
    kmax = max(core_counts)
    for hd in [int(x) for x in args.head_dims.split(",")]:
        g = torch.Generator().manual_seed(0)
        Qc, Kc, Vc = (torch.randn(seq, hd, generator=g) for _ in range(3))
        K_on = {d: Kc.to(d) for d in pick(cores, kmax)}
        V_on = {d: Vc.to(d) for d in pick(cores, kmax)}
        dense = {}
        for k in core_counts:
            devs = pick(cores, k)
            qs = [Qc[i * seq // k : (i + 1) * seq // k].to(devs[i]) for i in range(k)]
            dense[k] = safe(
                lambda: [_sdpa(q, K_on[d], V_on[d]) for q, d in zip(qs, devs, strict=True)],
                devs,
                reps=args.reps,
            )
            qs = None
        for kind in args.masks.split(","):
            bm = block_mask(kind, nb)
            mb = mask_bsr_from_block_mask(bm)
            counts = bm.sum(1).tolist()
            row = {
                "section": "attn",
                "seq": seq,
                "head_dim": hd,
                "mask": kind,
                "block_density": float(bm.float().mean()),
                "K_max": max(counts),
            }
            for k in core_counts:
                row[f"sdpa_dense_{k}c_ms"] = dense[k]
            # masked SDPA on one core and with rows split over kmax cores
            em = bm.repeat_interleave(P, 0).repeat_interleave(P, 1)
            for k in (1, kmax):
                devs = pick(cores, k)
                qs = [Qc[i * seq // k : (i + 1) * seq // k].to(devs[i]) for i in range(k)]
                ems = [em[i * seq // k : (i + 1) * seq // k].to(devs[i]) for i in range(k)]
                row[f"sdpa_masked_{k}c_ms"] = safe(
                    lambda: [
                        _sdpa(q, K_on[d], V_on[d], m) for q, m, d in zip(qs, ems, devs, strict=True)
                    ],
                    devs,
                    reps=args.reps,
                )
                qs = ems = None
                free()
            ref = None
            if args.check:
                ref = _sdpa(Qc.double(), Kc.double(), Vc.double(), em)
            em = None
            for k in core_counts:
                devs = pick(cores, k)
                SA = ShardedAttention(mb, devs)
                Qs = SA.shard_rows(Qc)
                KVs = SA.replicate_kv(Kc, Vc)
                outs = SA.forward(Qs, KVs)
                sync(SA.devices)
                if ref is not None:
                    row[f"sparse_{k}c_rel_err"] = rel_err(SA.gather(outs), ref)
                row[f"sparse_{k}c_fwd_ms"] = safe(
                    lambda: SA.forward(Qs, KVs), SA.devices, reps=args.reps
                )
                row[f"sparse_{k}c_issue_ms"] = issue_time(lambda: SA.forward(Qs, KVs), SA.devices)
                if k == kmax:
                    row[f"sparse_{k}c_gather_ms"] = safe(
                        lambda: SA.gather(SA.forward(Qs, KVs)),
                        SA.devices,
                        reps=max(3, args.reps // 2),
                    )
                if k == 1:
                    Q1, K1, V1 = (t.to(devs[0]) for t in (Qc, Kc, Vc))
                    row["api_1c_fwd_ms"] = safe(
                        lambda: trnsparse.block_sparse_attention_tiled(Q1, K1, V1, mb),
                        devs,
                        reps=args.reps,
                    )
                    Q1 = K1 = V1 = None
                KVs = None
                SA = Qs = outs = None
                free()
            k1 = min(core_counts)
            for k in core_counts:
                if row.get(f"sparse_{k}c_fwd_ms") and row.get(f"sparse_{k1}c_fwd_ms"):
                    row[f"sparse_scaling_{k}c"] = (
                        row[f"sparse_{k1}c_fwd_ms"] / row[f"sparse_{k}c_fwd_ms"]
                    )
            emit(row, args.out)
        K_on = V_on = None
        free()


# ── Attention: heads across cores ────────────────────────────────────────────


def section_heads(args, cores):
    seq = args.size
    nb = seq // P
    H = args.heads
    core_counts = [int(c) for c in args.cores.split(",")]
    for hd in [int(x) for x in args.head_dims.split(",")]:
        g = torch.Generator().manual_seed(0)
        Qc, Kc, Vc, dOc = (torch.randn(H, seq, hd, generator=g) for _ in range(4))

        def place(k):
            """Head h on core h % k: per core, lists of that core's heads."""
            devs = pick(cores, k)
            out = []
            for i, d in enumerate(devs):
                hs = list(range(i, H, k))
                out.append((d, [tuple(t[h].to(d) for t in (Qc, Kc, Vc, dOc)) for h in hs]))
            return devs, out

        def place_dense(k):
            devs = pick(cores, k)
            out = []
            for i, d in enumerate(devs):
                hs = list(range(i, H, k))
                out.append(tuple(t[hs][None].to(d) for t in (Qc, Kc, Vc, dOc)))
            return devs, out

        dense_fwd, dense_fb = {}, {}
        for k in core_counts:
            devs, pl = place_dense(k)
            dense_fwd[k] = safe(
                lambda: [F.scaled_dot_product_attention(q, kk, v) for q, kk, v, _ in pl],
                devs,
                reps=args.reps,
            )

            def dfb():
                for q, kk, v, do in pl:
                    q, kk, v = (t.detach().requires_grad_() for t in (q, kk, v))
                    F.scaled_dot_product_attention(q, kk, v).backward(do)

            dense_fb[k] = safe(dfb, devs, reps=args.reps)
            pl = None
            free()
        for kind in args.masks.split(","):
            bm = block_mask(kind, nb)
            mb = mask_bsr_from_block_mask(bm)
            row = {
                "section": "heads",
                "seq": seq,
                "head_dim": hd,
                "heads": H,
                "mask": kind,
                "block_density": float(bm.float().mean()),
            }
            for k in core_counts:
                row[f"sdpa_dense_{k}c_fwd_ms"] = dense_fwd[k]
                row[f"sdpa_dense_{k}c_fwd_bwd_ms"] = dense_fb[k]
            for k in core_counts:
                devs, pl = place(k)

                def fwd():
                    return [
                        trnsparse.block_sparse_attention_tiled(q, kk, v, mb)
                        for _, hs in pl
                        for q, kk, v, _ in hs
                    ]

                def fb():
                    for _, hs in pl:
                        for q, kk, v, do in hs:
                            q, kk, v = (t.detach().requires_grad_() for t in (q, kk, v))
                            trnsparse.block_sparse_attention_tiled(q, kk, v, mb).backward(do)

                row[f"sparse_{k}c_fwd_ms"] = safe(fwd, devs, reps=args.reps)
                if not args.no_bwd:
                    row[f"sparse_{k}c_fwd_bwd_ms"] = safe(fb, devs, reps=args.reps)
                pl = None
                free()
            k1 = min(core_counts)
            for k in core_counts:
                for ph in ("fwd", "fwd_bwd"):
                    a, b = row.get(f"sparse_{k1}c_{ph}_ms"), row.get(f"sparse_{k}c_{ph}_ms")
                    if a and b:
                        row[f"sparse_scaling_{k}c_{ph}"] = a / b
            emit(row, args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True, choices=["bsr", "attn", "heads"])
    ap.add_argument("--size", type=int, default=8192)
    ap.add_argument("--densities", default="0.01,0.05,0.1,0.25,0.5")
    ap.add_argument("--widths", default="128,512")
    ap.add_argument("--head-dims", default="64,128")
    ap.add_argument("--masks", default="local,dilated,global")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--cores", default="1,2,4")
    ap.add_argument("--no-bwd", action="store_true")
    ap.add_argument("--check", action="store_true", help="fp64 reference for attention")
    ap.add_argument("--cpu", action="store_true", help="CPU dense baseline")
    ap.add_argument(
        "--pretransposed", action="store_true", help="also the pre-transposed BSR kernel"
    )
    ap.add_argument(
        "--only-pretransposed", action="store_true", help="only the pre-transposed BSR kernel"
    )
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    trnsparse.set_backend("nki")
    cores = logical_cores()
    print(f"# {len(cores)} logical cores visible", file=sys.stderr)
    {"bsr": section_bsr, "attn": section_attn, "heads": section_heads}[args.section](args, cores)


if __name__ == "__main__":
    main()
