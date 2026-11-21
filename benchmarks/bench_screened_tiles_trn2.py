"""Trn2: Schwarz-screened SpMM with tile skipping (#61) vs dense aten.

Chemistry-shaped inputs: bounds Q sorted in decreasing order (exponential
decay, as for shells ordered by extent), symmetric A with |A_ij| <= Q_i Q_j,
and tau chosen
so that a target fraction of the 128 x 128 tiles survives. Times (ms,
median of --reps, `torch.neuron.synchronize()` after every call):

- tiles:  `screened_spmm` (this PR: kept tiles only)
- tiles sym: `screened_spmm(..., symmetric=True)` (no Tensor Engine transposes)
- fused:  `screened_spmm` with TRNSPARSE_SCREEN_SKIP_TILES off (main's kernel)
- dense:  aten `A @ B` on the device (no screening)
- masked: aten `(A * mask) @ B` on the device, mask precomputed on the device

Errors are against fp64 on the host: `par` = max|C - C_masked64| / max|C_masked64|
(parity with the screened result), and `bound` = ||C - A64 B64||_F divided by
the Schwarz bound tau * sqrt(#dropped) * ||B||_F (<= 1 means within the
threshold-times-norm bound, up to operand rounding).

Usage: python bench_screened_tiles_trn2.py [--sizes 2048,4096,8192] [--widths 512]
       [--kept 0.05,0.2,0.5,0.8] [--dtypes fp32,bf16] [--no-fused]
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

import torch

import trnsparse
from trnsparse.nki import dispatch as _d

DEV = "neuron"
P = 128


def _sync():
    if hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def bench(fn, warmup=2, reps=10):
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def make_case(n, width, kept, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.sort(torch.rand(n, generator=g)).values
    Q = torch.exp(-8.0 * x)  # decreasing bounds
    diag = Q * Q
    R = 2 * torch.rand(n, n, generator=g) - 1
    A = (R + R.T) / 2 * torch.outer(Q, Q)  # symmetric, like (mu nu | lambda sigma)
    B = torch.randn(n, width, generator=g)
    nt = (n + P - 1) // P
    qpad = torch.zeros(nt * P)
    qpad[:n] = Q
    qmax = qpad.reshape(nt, P).amax(1)
    tile_max = torch.sort(torch.outer(qmax, qmax).flatten(), descending=True).values
    k = max(1, min(len(tile_max) - 1, round(kept * len(tile_max))))
    tau = float((tile_max[k - 1] + tile_max[k]) / 2)  # keeps exactly k tiles
    return A, diag, B, Q, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2048,4096,8192")
    ap.add_argument("--widths", default="512")
    ap.add_argument("--kept", default="0.05,0.2,0.5,0.8")
    ap.add_argument("--dtypes", default="fp32,bf16")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-fused", action="store_true")
    args = ap.parse_args()
    trnsparse.set_backend("nki")
    _d._REQUIRE_NKI = True
    dts = {"fp32": torch.float32, "bf16": torch.bfloat16}
    print(
        "| n | width | dtype | kept tiles | partial | tiles (ms) | tiles sym (ms) | fused main (ms) "
        "| dense aten (ms) | masked aten (ms) | best/dense | par err | bound ratio | compile+1st (s) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in [int(s) for s in args.sizes.split(",")]:
        for width in [int(s) for s in args.widths.split(",")]:
            for dname in args.dtypes.split(","):
                for kept in [float(s) for s in args.kept.split(",")]:
                    run_case(n, width, dname, dts[dname], kept, args)


def run_case(n, width, dname, dt, kept, args):
    A, diag, B, Q, tau = make_case(n, width, kept)
    thr = tau * tau
    A = A.to(dt)
    B = B.to(dt)
    _, _, part, frac = _d.screened_tile_structure(Q, tau, n)
    mask = (Q[:, None] * Q[None, :]) > tau
    a_d, b_d = A.to(DEV), B.to(DEV)
    m_d = mask.to(dt).to(DEV)

    def tiles():
        return trnsparse.screened_spmm(a_d, diag, b_d, thr)

    def tiles_sym():
        return trnsparse.screened_spmm(a_d, diag, b_d, thr, symmetric=True)

    t0 = time.perf_counter()
    C = tiles().cpu()
    t_first = time.perf_counter() - t0
    t_tiles = bench(tiles, reps=args.reps)
    C_sym = tiles_sym().cpu()
    t_sym = bench(tiles_sym, reps=args.reps)
    t_fused = float("nan")
    if not args.no_fused and dt == torch.float32:  # main's kernel is fp32-only
        _d._SCREEN_SKIP_TILES = False
        try:
            t_fused = bench(tiles, reps=args.reps)
        except Exception as e:
            print(f"# fused failed: {str(e)[:120]}", file=sys.stderr)
        _d._SCREEN_SKIP_TILES = True
    t_dense = bench(lambda: a_d @ b_d, reps=args.reps)
    t_masked = bench(lambda: (a_d * m_d) @ b_d, reps=args.reps)
    A64, B64 = A.double(), B.double()
    ref_m = (A64 * mask) @ B64
    par = max(float((c.double() - ref_m).abs().max() / ref_m.abs().max()) for c in (C, C_sym))
    full = A64 @ B64
    dropped = int((~mask).sum())
    bound = tau * math.sqrt(max(dropped, 1)) * float(B64.norm())
    ratio = float((C.double() - full).norm()) / bound
    best = min(t_tiles, t_sym)
    print(
        f"| {n} | {width} | {dname} | {frac:.0%} | {sum(part)} | **{t_tiles:.2f}** "
        f"| **{t_sym:.2f}** | {t_fused:.2f} | {t_dense:.2f} | {t_masked:.2f} "
        f"| {best / t_dense:.2f} | {par:.1e} | {ratio:.2f} | {t_first:.1f} |",
        flush=True,
    )
    _sync()


if __name__ == "__main__":
    main()
