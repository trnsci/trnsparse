"""Fused kernel-matrix product K(X, X) @ V (#93): kernel entries per second.

Modes (rates in kernel entries/s = n * m / time; one application):

- nki:    `KernelOperator` on the device (one fused launch, K never formed)
- aten:   the same augmented-GEMM algorithm in aten ops on the device
          (blocked addmm + exp + matmul, the PyTorch path of kernel_matmul)
- cpu2:   the aten algorithm on the host CPU, 2 threads (the suite's
          same-box convention, trnsci/trnsci#35)
- cpuall: the same with all host vCPUs

CPU modes time a block of `--cpu-rows` output rows against all n points
(the rate is per entry, so this measures the same thing in bounded time);
device modes apply the full operator. Each device result is checked
against float64 on the host on sampled rows.

Usage:
    python benchmarks/bench_kernel_matmul_trn2.py --device \
        [--sizes 131072,262144,524288,1048576] [--rhs 1,16,64,128] \
        [--modes nki,aten,cpu2,cpuall] [--kernel rbf] [--json out.json]
    python benchmarks/bench_kernel_matmul_trn2.py --compare A.pt B.pt   # bitwise LNC check
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

import trnsparse
from trnsparse.kernel_matmul import _torch_kernel_matmul

DEV = "neuron"


def _sync(device):
    if device == DEV and hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def _time(fn, device, warmup, reps):
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _check(out, X, V, kernel, ell, noise, rows=256):
    idx = torch.linspace(0, X.shape[0] - 1, rows).long()
    ref = (
        trnsparse.kernel_matmul_reference(X[idx], X, V, kernel, ell, 1.0) + noise * V[idx].double()
    )
    got = out.cpu()[idx].double()
    return float((got - ref).abs().max() / ref.abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true", help="run the nki and aten modes on neuron")
    ap.add_argument("--sizes", default="131072,262144,524288,1048576")
    ap.add_argument("--rhs", default="1,16,64,128")
    ap.add_argument("--modes", default="nki,aten,cpu2,cpuall")
    ap.add_argument("--kernel", default="rbf")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--cpu-rows", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        a, b = (torch.load(p) for p in args.compare)
        same = torch.equal(a, b)
        print(f"bitwise equal: {same}; max |diff| = {float((a - b).abs().max()):.3e}")
        sys.exit(0 if same else 1)

    modes = [m for m in args.modes.split(",") if m]
    if not args.device:
        modes = [m for m in modes if m.startswith("cpu")]
    sizes = [int(s) for s in args.sizes.split(",")]
    rhs = [int(r) for r in args.rhs.split(",")]
    ell, noise = 0.05, 1e-4
    rows = []
    print(
        f"kernel={args.kernel} d={args.d} lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG', 'default')}"
    )
    print(f"{'n':>9} {'R':>4} {'mode':>7} {'time_s':>10} {'entries/s':>11} {'relerr':>9}")
    for n in sizes:
        g = torch.Generator().manual_seed(n)
        X = torch.rand(n, args.d, generator=g)
        for R in rhs:
            V = torch.randn(n, R, generator=g)
            for mode in modes:
                rec = {"n": n, "R": R, "mode": mode, "kernel": args.kernel, "d": args.d}
                if mode == "nki":
                    trnsparse.set_backend("nki")
                    op = trnsparse.KernelOperator(X, args.kernel, ell, 1.0, noise=noise, device=DEV)
                    Vd = V.to(DEV)
                    t_cold = time.perf_counter()
                    op(Vd)
                    _sync(DEV)
                    rec["first_call_s"] = time.perf_counter() - t_cold
                    t = _time(lambda op=op, Vd=Vd: op(Vd), DEV, 1, args.reps)
                    rec["relerr"] = _check(op(Vd), X, V, args.kernel, ell, noise)
                    entries = n * n
                elif mode == "aten":
                    trnsparse.set_backend("pytorch")
                    op = trnsparse.KernelOperator(X, args.kernel, ell, 1.0, noise=noise, device=DEV)
                    Vd = V.to(DEV)
                    t = _time(lambda op=op, Vd=Vd: op(Vd), DEV, 1, args.reps)
                    rec["relerr"] = _check(op(Vd), X, V, args.kernel, ell, noise)
                    entries = n * n
                else:
                    threads = 2 if mode == "cpu2" else os.cpu_count()
                    torch.set_num_threads(threads)
                    rec["threads"] = threads
                    trnsparse.set_backend("pytorch")
                    op = trnsparse.KernelOperator(X, args.kernel, ell, 1.0, noise=noise)
                    k = min(args.cpu_rows, n)
                    Xa = op.X[:k].contiguous()
                    t = _time(
                        lambda Xa=Xa, op=op, V=V: _torch_kernel_matmul(
                            Xa, op.X, V, op.kind, op._a, 1.0, 0.0
                        ),
                        "cpu",
                        1,
                        args.reps,
                    )
                    entries = k * n
                    rec["rows_timed"] = k
                rec["time_s"] = t
                rec["entries_per_s"] = entries / t
                rec["full_apply_s"] = n * n / rec["entries_per_s"]
                rows.append(rec)
                print(
                    f"{n:>9} {R:>4} {mode:>7} {t:>10.4f} {rec['entries_per_s']:>11.3e} "
                    f"{rec.get('relerr', float('nan')):>9.2e}",
                    flush=True,
                )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
