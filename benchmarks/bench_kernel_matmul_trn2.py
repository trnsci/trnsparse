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
        [--modes nki,aten,cpu2,cpuall] [--kernel rbf[,matern12,...]] [--json out.json]
    python benchmarks/bench_kernel_matmul_trn2.py --compare A.pt B.pt   # bitwise LNC check

Kernel variants (#119): `--variants 0,1,3` runs the nki mode once per
variant on the same operands. A variant is a bit mask: 1 sets
TRNSPARSE_KMV_SUBTILE=1 (row-tiled distance matmuls), 2 sets
TRNSPARSE_KMV_NORM_BIAS=1 (|y|^2 in the PSUM reader), 0 is the default
kernel. Each rate is printed against `--baseline` (5.06e10 entries/s per
logical core, #108: group 32, R = 128), with whether the result equals the
first variant's bit for bit:

    python benchmarks/bench_kernel_matmul_trn2.py --device --modes nki \
        --sizes 131072 --rhs 128 --kernel rbf,matern12,matern52 --variants 0,1,3 \
        --json kmv_variants.json

`--dump DIR` saves every nki result, so two checkouts (or LNC settings) can
be compared with `--compare`. The package version and the
`TRNSPARSE_KMV_GROUP` in effect are printed with the header.
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


_VARIANT_ENV = ((1, "TRNSPARSE_KMV_SUBTILE"), (2, "TRNSPARSE_KMV_NORM_BIAS"))


def _one(args, mode, kernel, n, R, X, V, ell, noise, variant):
    """One timed application; `variant` (a mask, None: the environment as set) sets the env."""
    rec = {"n": n, "R": R, "mode": mode, "kernel": kernel, "d": args.d}
    prev = {name: os.environ.get(name) for _, name in _VARIANT_ENV}
    if variant is not None:
        for bit, name in _VARIANT_ENV:
            os.environ[name] = "1" if variant & bit else "0"
        rec["variant"] = variant
    try:
        if mode == "nki":
            trnsparse.set_backend("nki")
            op = trnsparse.KernelOperator(X, kernel, ell, 1.0, noise=noise, device=DEV)
            Vd = V.to(DEV)
            t_cold = time.perf_counter()
            op(Vd)
            _sync(DEV)
            rec["first_call_s"] = time.perf_counter() - t_cold
            t = _time(lambda op=op, Vd=Vd: op(Vd), DEV, 1, args.reps)
            res = op(Vd).cpu()
            rec["relerr"] = _check(res, X, V, kernel, ell, noise)
            rec["_res"] = res
            if args.dump:
                os.makedirs(args.dump, exist_ok=True)
                tag = "" if variant is None else f"_v{variant}"
                torch.save(res, os.path.join(args.dump, f"kmv_{kernel}_n{n}_R{R}{tag}.pt"))
            entries = n * n
        elif mode == "aten":
            trnsparse.set_backend("pytorch")
            op = trnsparse.KernelOperator(X, kernel, ell, 1.0, noise=noise, device=DEV)
            Vd = V.to(DEV)
            t = _time(lambda op=op, Vd=Vd: op(Vd), DEV, 1, args.reps)
            rec["relerr"] = _check(op(Vd), X, V, kernel, ell, noise)
            entries = n * n
        else:
            threads = 2 if mode == "cpu2" else os.cpu_count()
            torch.set_num_threads(threads)
            rec["threads"] = threads
            trnsparse.set_backend("pytorch")
            op = trnsparse.KernelOperator(X, kernel, ell, 1.0, noise=noise)
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
    finally:
        for name, val in prev.items():
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val
    rec["time_s"] = t
    rec["entries_per_s"] = entries / t
    rec["x_baseline"] = rec["entries_per_s"] / args.baseline
    rec["full_apply_s"] = n * n / rec["entries_per_s"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true", help="run the nki and aten modes on neuron")
    ap.add_argument("--sizes", default="131072,262144,524288,1048576")
    ap.add_argument("--rhs", default="1,16,64,128")
    ap.add_argument("--modes", default="nki,aten,cpu2,cpuall")
    ap.add_argument("--kernel", default="rbf", help="one kernel or a comma-separated list")
    ap.add_argument(
        "--variants",
        default=None,
        help="comma-separated kernel variant masks for the nki mode, e.g. 0,1,3 (#119):"
        " 1 = TRNSPARSE_KMV_SUBTILE, 2 = TRNSPARSE_KMV_NORM_BIAS",
    )
    ap.add_argument(
        "--baseline",
        type=float,
        default=5.06e10,
        help="entries/s per logical core to compare the nki mode with (#108)",
    )
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--cpu-rows", type=int, default=2048)
    ap.add_argument(
        "--aten-max", type=int, default=262144, help="largest n for the aten mode (it is slow)"
    )
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument(
        "--dump", default=None, help="save each nki result to DIR/kmv_n{n}_R{R}.pt (for --compare)"
    )
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
    kernels = [k for k in args.kernel.split(",") if k]
    variants = [int(s) for s in args.variants.split(",")] if args.variants else [None]
    ell, noise = 0.05, 1e-4
    rows = []
    print(
        f"kernel={args.kernel} d={args.d} lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG', 'default')}"
        f" trnsparse={trnsparse.__version__} ({trnsparse.__file__})"
        f" group={os.environ.get('TRNSPARSE_KMV_GROUP', 'auto')}"
        f" variants={args.variants or 'env'}"
        f" (SUBTILE={os.environ.get('TRNSPARSE_KMV_SUBTILE', 'unset')}"
        f" NORM_BIAS={os.environ.get('TRNSPARSE_KMV_NORM_BIAS', 'unset')})"
        f" baseline={args.baseline:.3e}"
    )
    print(
        f"{'kernel':>8} {'n':>9} {'R':>4} {'mode':>9} {'time_s':>10} {'entries/s':>11}"
        f" {'x_base':>7} {'relerr':>9} {'bitwise':>7}"
    )
    for kernel in kernels:
        for n in sizes:
            g = torch.Generator().manual_seed(n)
            X = torch.rand(n, args.d, generator=g)
            for R in rhs:
                V = torch.randn(n, R, generator=g)
                for mode in modes:
                    if mode == "aten" and n > args.aten_max:
                        continue
                    first = None  # the nki result of the first variant, for the bitwise column
                    for st in variants if mode == "nki" else [None]:
                        rec = _one(args, mode, kernel, n, R, X, V, ell, noise, st)
                        res = rec.pop("_res", None)
                        if res is not None:
                            if first is None:
                                first = res
                            else:
                                rec["bitwise_first"] = bool(torch.equal(res, first))
                        rows.append(rec)
                        label = mode if st is None else f"{mode}/v{st}"
                        bw = rec.get("bitwise_first")
                        print(
                            f"{kernel:>8} {n:>9} {R:>4} {label:>9} {rec['time_s']:>10.4f}"
                            f" {rec['entries_per_s']:>11.3e}"
                            f" {rec['entries_per_s'] / args.baseline:>7.3f}"
                            f" {rec.get('relerr', float('nan')):>9.2e}"
                            f" {'-' if bw is None else str(bw):>7}",
                            flush=True,
                        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
