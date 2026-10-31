"""Time-to-tolerance for the fused SBUF-resident CG (`cg_bsr_fused`, #22).

Standalone script (not pytest-benchmark), one matrix per process, one JSON
line per (method, preconditioner) row, appended to ``--out``:

    NKI_ENABLE_TRACE_CACHE=0 taskset -c 10,11 \\
        python benchmarks/bench_cg_fused_trn2.py --tiles 32 --density 0.25

Test matrices are synthetic and chemistry-flavoured: the overlap matrix of
normalised s-type Gaussians on a synthetic molecular cluster. Each 128-row
block row is one "fragment" (128 Gaussians with jittered centres and
log-uniform exponents around a fragment centre); fragment centres are
random in a box. Blocks between fragments farther apart than a cutoff are
dropped (block screening), the cutoff chosen to hit ``--density``; then a
level shift sets the condition number to ``--kappa`` exactly, which keeps
the screened matrix SPD. A is built in fp64; every solver gets the fp32
rounding of it except the fp64 CPU baselines.

Methods (each row reports wall time to tolerance, iterations, and the true
residual ||b - A x|| / ||b|| and the error against an fp64 direct solve):

    fused        cg_bsr_fused: one launch, A's blocks in SBUF, device-side
                 convergence check every --check-every steps; one row per
                 --matvec variant (row, column, column:bf16, column:bf16x2)
    host_bsr     host-driven CG over the keyed pre-transposed BSR SpMM op,
                 device tensors, no host sync except every --check-every
                 steps
    cg_bsr       trnsparse.cg_bsr as shipped (one .item() sync per dot)
    dense_aten   CG with torch.mv on a dense fp32 A on the device, synced
                 every --check-every steps
    scipy        scipy.sparse.linalg.cg on a BSR (fp64) matrix, on the host
    torch_cpu    the cg_bsr_fused host path (fp64 and fp32 block bmm)

Plain CG and block-Jacobi PCG (exact inverses of the 128x128 diagonal
blocks) for fused, host_bsr, scipy and torch_cpu. Host threads are whatever
the process is pinned to (``taskset``), with OMP_NUM_THREADS matching.
"""

# ruff: noqa: B023  (every closure is benchmarked before its loop variables change)
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trnsparse  # noqa: E402
from trnsparse.iterative import prepare_cg_bsr  # noqa: E402

P = 128


def cluster_overlap(T: int, density: float, kappa: float, seed: int = 0):
    """Block-screened Gaussian overlap of a synthetic cluster, shifted to kappa."""
    rng = np.random.default_rng(seed)
    n = T * P
    side = 3.5 * T ** (1 / 3)  # about one fragment per 3.5^3 volume
    centres = rng.uniform(0, side, size=(T, 3))
    pos = np.repeat(centres, P, axis=0) + rng.normal(0, 0.8, size=(n, 3))
    expo = np.exp(rng.uniform(np.log(0.15), np.log(3.0), size=n))
    d = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
    off = np.sort(d[np.triu_indices(T, 1)])
    n_pairs = max(0, round((density * T * T - T) / 2))
    keep = d <= (off[n_pairs - 1] if n_pairs > 0 else -1.0)
    np.fill_diagonal(keep, True)
    A = np.zeros((n, n))
    for i in range(T):
        si = slice(i * P, (i + 1) * P)
        for j in range(i, T):
            if not keep[i, j]:
                continue
            sj = slice(j * P, (j + 1) * P)
            ai, aj = expo[si, None], expo[None, sj]
            r2 = ((pos[si, None, :] - pos[None, sj, :]) ** 2).sum(-1)
            s = (2 * np.sqrt(ai * aj) / (ai + aj)) ** 1.5 * np.exp(-(ai * aj / (ai + aj)) * r2)
            A[si, sj] = s
            A[sj, si] = s.T
    ev = np.linalg.eigvalsh(A) if n <= 8192 else None
    lo, hi = ev[0], ev[-1]
    mu = (hi - kappa * lo) / (kappa - 1)
    A[np.diag_indices(n)] += mu
    return A, keep


def _sync():
    if hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def timed(fn, reps):
    out = fn()  # warm (compiles)
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        out = fn()
        _sync()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), out


def host_cg(matvec, precond, b, tol, max_iter, check_every):
    """CG on device tensors (b: (n, R), one solve per column) with a host
    sync only every `check_every` steps, when all columns are tested."""
    x = torch.zeros_like(b)
    r = b.clone()
    z = precond(r) if precond else r
    p = z.clone()
    rz = (r * z).sum(0)
    thr = tol * tol * (b * b).sum(0).cpu()
    k = 0
    while k < max_iter:
        for _ in range(check_every):
            q = matvec(p)
            alpha = rz / (p * q).sum(0).clamp_min(1e-30)
            x = x + alpha * p
            r = r - alpha * q
            z = precond(r) if precond else r
            rz_new = (r * z).sum(0)
            p = z + (rz_new / rz.clamp_min(1e-30)) * p
            rz = rz_new
            k += 1
        if bool(((r * r).sum(0).cpu() <= thr).all()):
            break
    return x, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, required=True, help="block rows T (n = 128 T)")
    ap.add_argument("--density", type=float, required=True, help="target block density")
    ap.add_argument("--kappa", type=float, default=1e3)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--check-every", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--methods", default="fused,host_bsr,cg_bsr,dense_aten,scipy,torch_cpu")
    ap.add_argument("--precond", default="none,block_jacobi")
    ap.add_argument("--out", default="cg_fused.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--rhs", type=int, default=1, help="right-hand sides solved together")
    ap.add_argument(
        "--matvec",
        default="auto",
        help="fused variants, comma-separated matvec[:precision], e.g. row,column,column:bf16",
    )
    args = ap.parse_args()
    methods = args.methods.split(",")
    preconds = [None if p == "none" else p for p in args.precond.split(",")]

    t0 = time.time()
    A64, keep = cluster_overlap(args.tiles, args.density, args.kappa)
    n = A64.shape[0]
    rng = np.random.default_rng(1)
    R = args.rhs
    b64 = rng.standard_normal((n, R))
    x_ref = np.linalg.solve(A64, b64)
    gen_s = time.time() - t0
    A32 = torch.from_numpy(A64).float()
    bsr = trnsparse.BSRMatrix.from_dense(A32, block_size=P)
    b = torch.from_numpy(b64 if R > 1 else b64[:, 0])
    base = dict(
        n=n,
        tiles=args.tiles,
        n_blocks=bsr.n_blocks,
        density=round(bsr.n_blocks / args.tiles**2, 4),
        kappa=args.kappa,
        tol=args.tol,
        check_every=args.check_every,
        tag=args.tag,
        rhs=R,
        lnc=os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"),
    )
    print(f"# n={n} blocks={bsr.n_blocks} gen {gen_s:.1f}s", file=sys.stderr, flush=True)

    def emit(method, pc, secs, x, iters, extra=None):
        x = np.asarray(x.detach().cpu().double().numpy() if torch.is_tensor(x) else x)
        x = x.reshape(n, R)
        iters = int(np.max(np.asarray(iters)))
        row = dict(base, method=method, precond=pc or "none", ms=round(secs * 1e3, 3))
        res = np.linalg.norm(b64 - A64 @ x, axis=0) / np.linalg.norm(b64, axis=0)
        err = np.linalg.norm(x - x_ref, axis=0) / np.linalg.norm(x_ref, axis=0)
        row.update(
            ms_per_rhs=round(secs * 1e3 / R, 4),
            iters=iters,
            us_per_iter=round(secs * 1e6 / max(iters, 1), 2),
            true_rel_res=float(res.max()),
            rel_err=float(err.max()),
        )
        if extra:
            row.update(extra)
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

    from trnsparse.nki.dispatch import _get_native_ops

    trnsparse.set_backend("nki")
    ops = _get_native_ops()
    dev = torch.device("neuron")

    for pc in preconds:
        op_dev = prepare_cg_bsr(bsr, preconditioner=pc, device=dev)
        for spec in args.matvec.split(",") if "fused" in methods else []:
            mv, _, prec = spec.partition(":")
            prec = prec or "fp32"
            fn = lambda: trnsparse.cg_bsr_fused(  # noqa: E731
                op_dev,
                b,
                tol=args.tol,
                max_iter=args.max_iter,
                check_every=args.check_every,
                return_info=True,
                matvec=mv,
                precision=prec,
            )
            tc = time.time()
            op_dev._device_operands(prec)
            fn()
            compile_s = time.time() - tc
            secs, (x, it, rel, info) = timed(fn, args.reps)
            it = int(torch.as_tensor(it).max())
            # fixed cost: same launch with max_iter=0 (A load, r0, transfers)
            secs0, _ = timed(
                lambda: trnsparse.cg_bsr_fused(
                    op_dev, b, tol=args.tol, max_iter=0, matvec=mv, precision=prec
                ),
                args.reps,
            )
            emit(
                "fused",
                pc,
                secs,
                x,
                it,
                dict(
                    matvec=info["matvec"],
                    precision=prec,
                    resident_blocks=info["resident_blocks"],
                    trips=info["trips"],
                    ms_fixed=round(secs0 * 1e3, 3),
                    us_per_iter_marginal=round((secs - secs0) * 1e6 / max(it, 1), 2),
                    first_call_s=round(compile_s, 1),
                    rec_rel_res=float(torch.as_tensor(rel).max()),
                ),
            )
        if "host_bsr" in methods:
            bt, mt, key = op_dev._device_operands()
            T = args.tiles
            dkey = trnsparse.nki.kernels.register_structure(list(range(T + 1)), list(range(T)))
            mv = lambda v: ops["bsr_spmm_t_keyed"](bt, v, key)  # noqa: E731
            pre = (lambda v: ops["bsr_spmm_t_keyed"](mt, v, dkey)) if pc else None  # noqa: E731
            bd = b.float().reshape(n, R).to(dev)
            secs, (x, it) = timed(
                lambda: host_cg(mv, pre, bd, args.tol, args.max_iter, args.check_every), args.reps
            )
            emit("host_bsr", pc, secs, x, it)
        if "cg_bsr" in methods and pc is None and R == 1:
            blocks_dev = trnsparse.BSRMatrix(
                bsr.blocks.to(dev), bsr.block_col_indices, bsr.block_row_ptrs, bsr.shape, P
            )
            bd = b.float().to(dev)
            secs, (x, it, rel) = timed(
                lambda: trnsparse.cg_bsr(blocks_dev, bd, tol=args.tol, max_iter=args.max_iter),
                1,
            )
            emit("cg_bsr", pc, secs, x, it)
        if "dense_aten" in methods and pc is None:
            Ad = A32.to(dev)
            bd = b.float().reshape(n, R).to(dev)
            secs, (x, it) = timed(
                lambda: host_cg(
                    lambda v: Ad @ v, None, bd, args.tol, args.max_iter, args.check_every
                ),
                args.reps,
            )
            emit("dense_aten", pc, secs, x, it)
            del Ad
        if "scipy" in methods:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla

            As = sp.bsr_matrix(A64, blocksize=(P, P))
            M = None
            if pc:
                minv = op_dev._minv64.numpy()

                def mfun(v):
                    return np.einsum("tij,tj->ti", minv, v.reshape(-1, P)).reshape(-1)

                M = spla.LinearOperator((n, n), matvec=mfun)
            cnt = [0]

            def cb(_):
                cnt[0] += 1

            def run():  # one solve per column (scipy's cg takes a vector)
                cnt[0] = 0
                cols = [
                    spla.cg(
                        As,
                        b64[:, c],
                        rtol=args.tol,
                        atol=0.0,
                        maxiter=args.max_iter,
                        M=M,
                        callback=cb,
                    )[0]
                    for c in range(R)
                ]
                return np.stack(cols, axis=1)

            secs, x = timed(run, args.reps)
            emit("scipy", pc, secs, x, cnt[0] // R, dict(threads=torch.get_num_threads()))
        if "torch_cpu" in methods:
            op_cpu = prepare_cg_bsr(bsr, preconditioner=pc, device="cpu")
            for dt in (torch.float64, torch.float32):
                bb = b.to(dt)
                secs, (x, it, rel) = timed(
                    lambda: trnsparse.cg_bsr_fused(
                        op_cpu, bb, tol=args.tol, max_iter=args.max_iter
                    ),
                    args.reps,
                )
                emit(
                    f"torch_cpu_{'fp64' if dt == torch.float64 else 'fp32'}",
                    pc,
                    secs,
                    x,
                    it,
                    dict(threads=torch.get_num_threads()),
                )
        op_dev = None  # free the device operands before the next case
        _sync()


if __name__ == "__main__":
    main()
