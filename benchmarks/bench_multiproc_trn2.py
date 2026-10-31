"""One process per logical NeuronCore: BSR SpMM and dense aten, Trn2.

The companion of `bench_multicore_trn2.py`, which issues every core's launch
from one Python thread. Here each of k processes owns one logical core
(`NEURON_RT_VISIBLE_CORES=<i>`), its share of the block rows (the same
`partition_block_rows` split as `ShardedBSR`) and its own copy of B, so the
host-side launch costs run in parallel. Each rep: all processes meet at a
barrier, launch, synchronize, and meet again; the coordinator times the
span between the two barriers (the barrier cost itself is measured with an
empty step and reported as ``barrier_ms``).

    NKI_ENABLE_TRACE_CACHE=0 python benchmarks/bench_multiproc_trn2.py --size 8192 --width 512 --procs 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def worker(i, k, core, args, bar, q):
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import torch
    import torch_neuronx  # noqa: F401
    from bench_native_trn2 import random_bsr

    import trnsparse
    from trnsparse.multicore import ShardedBSR, partition_block_rows

    trnsparse.set_backend("nki")
    dev = torch.device("neuron", 0)
    n, w = args.size, args.width
    g = torch.Generator().manual_seed(1)
    Bc = torch.randn(n, w, generator=g)
    Ac = torch.randn(n, n, generator=g) / n**0.5
    B = Bc.to(dev)
    slab = Ac[i * n // k : (i + 1) * n // k].to(dev)
    Ac = None

    def sync():
        torch.neuron.synchronize()

    def steps():
        yield "barrier", lambda: None
        yield "dense", lambda: slab @ B
        for dens in [float(x) for x in args.densities.split(",")]:
            A = random_bsr(n, dens)
            rows = partition_block_rows(A.block_row_ptrs.tolist(), k)[i]
            # This process's rows only: a one-core ShardedBSR over the sub-matrix.
            from trnsparse.multicore import _sub_bsr

            sub = _sub_bsr(A, rows, "cpu")
            S = ShardedBSR(sub, [dev], pretransposed=args.pretransposed)
            yield f"sparse_{dens}", lambda S=S: S.spmm([B])

    for name, fn in steps():
        fn()
        fn()
        sync()
        for _ in range(args.reps):
            bar.wait()
            fn()
            sync()
            bar.wait()
        if i == 0:
            q.put(("done", name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8192)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--densities", default="0.01,0.05,0.1,0.25,0.5")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--cores", default="", help="comma list of core ids (default 0..procs-1)")
    ap.add_argument("--pretransposed", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    k = args.procs
    cores = [int(c) for c in args.cores.split(",")] if args.cores else list(range(k))
    ctx = mp.get_context("spawn")
    bar = ctx.Barrier(k + 1)
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(i, k, cores[i], args, bar, q)) for i in range(k)]
    for p in ps:
        p.start()
    results = {}
    names = ["barrier", "dense"] + [f"sparse_{d}" for d in args.densities.split(",")]
    for name in names:
        ts = []
        for _ in range(args.reps):
            bar.wait()
            t0 = time.perf_counter()
            bar.wait()
            ts.append((time.perf_counter() - t0) * 1e3)
        results[name] = statistics.median(ts)
        q.get()
    for p in ps:
        p.join()
    row = {
        "section": "bsr_multiproc",
        "n": args.size,
        "width": args.width,
        "procs": k,
        "pretransposed": args.pretransposed,
        "barrier_ms": results["barrier"],
        "dense_ms": results["dense"],
    }
    for d in args.densities.split(","):
        row[f"sparse_{float(d)}_ms"] = results[f"sparse_{d}"]
    line = json.dumps(row)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
