"""Fused power iteration (`power_iteration_bsr_fused`, #22) on Trn2.

Standalone script, one matrix per process, one JSON line per method,
appended to ``--out``. A fixed number of iterations (tol = 0), so every
method does the same work:

    NKI_ENABLE_TRACE_CACHE=0 taskset -c 10,11 \\
        python benchmarks/bench_power_fused_trn2.py --tiles 32 --density 0.25

Matrices: the synthetic block-screened Gaussian-overlap matrix of
`bench_cg_fused_trn2.py` (fp64; the device and fp32 columns get its fp32
rounding). Methods:

    fused_column / fused_row   power_iteration_bsr_fused, one launch
    host_bsr                   power_iteration_bsr on the device (the
                               shipped loop: BSR SpMM op + two .item() syncs
                               per iteration)
    torch_cpu_fp64 / fp32      the power_iteration_bsr_fused host path
                               (block bmm), same masked iteration
    scipy                      the same iteration with a scipy BSR matvec

Each row reports ms, µs per iteration, and lam against an fp64 eigvalsh.
"""

# ruff: noqa: B023
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_cg_fused_trn2 import P, cluster_overlap, timed  # noqa: E402

import trnsparse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, required=True)
    ap.add_argument("--density", type=float, required=True)
    ap.add_argument("--kappa", type=float, default=1e3)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--methods", default="fused_column,fused_row,host_bsr,torch_cpu,scipy")
    ap.add_argument("--out", default="power_fused.jsonl")
    args = ap.parse_args()
    methods = args.methods.split(",")
    K = args.iters

    A64, _ = cluster_overlap(args.tiles, args.density, args.kappa)
    n = A64.shape[0]
    top = float(np.linalg.eigvalsh(A64)[-1])
    A32 = torch.from_numpy(A64).float()
    bsr = trnsparse.BSRMatrix.from_dense(A32, block_size=P)
    v0 = torch.from_numpy(np.random.default_rng(2).standard_normal(n))
    base = dict(
        n=n,
        tiles=args.tiles,
        n_blocks=bsr.n_blocks,
        density=round(bsr.n_blocks / args.tiles**2, 4),
        iters=K,
        lam_ref=top,
        lnc=os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"),
    )

    def emit(method, secs, lam, extra=None):
        row = dict(base, method=method, ms=round(secs * 1e3, 3))
        row.update(us_per_iter=round(secs * 1e6 / K, 2), lam=lam, lam_rel_err=abs(lam - top) / top)
        if extra:
            row.update(extra)
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

    trnsparse.set_backend("nki")
    dev = torch.device("neuron")
    op = trnsparse.prepare_cg_bsr(bsr, device=dev)
    for mv in ("column", "row"):
        if f"fused_{mv}" not in methods:
            continue

        def fn():
            return trnsparse.power_iteration_bsr_fused(
                op, v0=v0, tol=0.0, max_iter=K, matvec=mv, return_info=True
            )

        secs, (lam, _, k, info) = timed(fn, args.reps)
        assert k == K
        emit(f"fused_{mv}", secs, lam, dict(resident_blocks=info["resident_blocks"]))
    if "host_bsr" in methods:
        bd = trnsparse.BSRMatrix(
            bsr.blocks.to(dev), bsr.block_col_indices, bsr.block_row_ptrs, bsr.shape, P
        )
        vd = v0.float().to(dev)
        secs, (lam, _, k) = timed(
            lambda: trnsparse.power_iteration_bsr(bd, v0=vd, max_iter=K, tol=0.0), 1
        )
        emit("host_bsr", secs, lam)
    op = None
    if "torch_cpu" in methods:
        op_cpu = trnsparse.prepare_cg_bsr(bsr, device="cpu")
        for dt in (torch.float64, torch.float32):
            vv = v0.to(dt)
            secs, (lam, _, _) = timed(
                lambda: trnsparse.power_iteration_bsr_fused(op_cpu, v0=vv, tol=0.0, max_iter=K),
                args.reps,
            )
            emit(f"torch_cpu_{'fp64' if dt == torch.float64 else 'fp32'}", secs, lam)
    if "scipy" in methods:
        import scipy.sparse as sp

        As = sp.bsr_matrix(A64, blocksize=(P, P))

        def run():
            v = v0.numpy() / np.linalg.norm(v0.numpy())
            lam = 0.0
            for _ in range(K):
                q = As @ v
                lam = float(v @ q)
                v = q / np.linalg.norm(q)
            return lam

        secs, lam = timed(run, args.reps)
        emit("scipy", secs, lam)


if __name__ == "__main__":
    main()
