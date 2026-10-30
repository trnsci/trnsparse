"""Warm-latency benchmark for the native TorchNeuron path (Trn2/Trn3).

Not a pytest-benchmark file: a standalone script, one section per process,
so each run stays short and a compile failure does not take the rest down.
Every row is printed as one JSON line (and appended to ``--out``).

    NKI_ENABLE_TRACE_CACHE=0 python benchmarks/bench_native_trn2.py --section bsr --size 4096

Sections:
    bsr     BSR (128x128) SpMM vs dense aten matmul on the device, CPU dense,
            and torch.sparse BSR/CSR on CPU. Block densities x RHS widths.
    csr     CSR SpMM (host structure, densify, NKI dense kernel) at element
            densities, vs the same baselines.
    screen  Screened SpMM vs dense masked matmul.
    attn    Block-sparse attention (local / dilated / global masks), forward
            and forward+backward, vs dense SDPA on the device and on CPU.

Options for the bsr and attn sections:
    --dtype bf16  run the operands in bf16 (fp32 PSUM inside the kernels)
                  and the dense aten baselines in bf16 too, so each side has
                  matched precision. Errors are against fp64 on the
                  bf16-rounded inputs.
    --fast        only the device columns (trnsparse e2e, static kernel,
                  dense aten); skips the gather path and the CPU baselines.
    TRNSPARSE_CORE_SPLIT=0 in the environment keeps each static kernel on
    one physical core's partition (the whole kernel on both cores, as
    before the split), for A/B runs.

Timing: each callable is run ``--warmup`` times (the first call compiles),
then ``--reps`` times with ``torch.neuron.synchronize()`` after every call;
the median is reported in milliseconds. Structure stays on the host, dense
data (blocks, operands) lives on the device, as `docs/installation.md`
recommends for the native path.
"""

# ruff: noqa: B023  (every closure is benchmarked before its loop variables change)
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trnsparse  # noqa: E402
from trnsparse.nki import dispatch as _d  # noqa: E402

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


def safe_bench(fn, **kw):
    try:
        return bench(fn, **kw)
    except Exception as e:  # report, don't abort the section
        print(f"# bench failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return None


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


# ── BSR ──────────────────────────────────────────────────────────────────────


def random_bsr(n, density, seed=0):
    g = torch.Generator().manual_seed(seed)
    nb = n // P
    total = nb * nb
    k = max(1, round(density * total))
    flat = torch.randperm(total, generator=g)[:k].sort().values
    rows, cols = flat // nb, flat % nb
    row_ptrs = torch.zeros(nb + 1, dtype=torch.long)
    row_ptrs[1:] = torch.bincount(rows, minlength=nb).cumsum(0)
    blocks = torch.randn(k, P, P, generator=g) / math.sqrt(P)
    return trnsparse.BSRMatrix(
        blocks=blocks,
        block_col_indices=cols.to(torch.long),
        block_row_ptrs=row_ptrs,
        shape=(n, n),
        block_size=P,
    )


_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def section_bsr(args):
    densities = [float(x) for x in args.densities.split(",")]
    widths = [int(x) for x in args.widths.split(",")]
    n = args.size
    dt = _DTYPES[args.dtype]
    for width in widths:
        g = torch.Generator().manual_seed(1)
        Bc = torch.randn(n, width, generator=g).to(dt).float()  # values exact in dt
        Bd = Bc.to(DEV, dt)
        dense_dev_ms = None
        cpu_dense_ms = None
        for dens in densities:
            A = random_bsr(n, dens)
            A.blocks = A.blocks.to(dt).float()
            Ac = A.to_dense()
            Ad_dev = trnsparse.BSRMatrix(
                blocks=A.blocks.to(DEV, dt),
                block_col_indices=A.block_col_indices,
                block_row_ptrs=A.block_row_ptrs,
                shape=A.shape,
                block_size=P,
            )
            counts = (A.block_row_ptrs[1:] - A.block_row_ptrs[:-1]).tolist()
            K_max = max(counts)
            row = {
                "section": "bsr",
                "dtype": args.dtype,
                "core_split": _core_split(),
                "n": n,
                "width": width,
                "block_density": dens,
                "n_blocks": A.n_blocks,
                "K_max": K_max,
                "mean_blocks_per_row": sum(counts) / len(counts),
            }
            ref = Ac.double() @ Bc.double()
            out = trnsparse.bsr_spmm(Ad_dev, Bd)
            row["rel_err"] = rel_err(out, ref)
            row["trnsparse_ms"] = safe_bench(lambda: trnsparse.bsr_spmm(Ad_dev, Bd), reps=args.reps)
            ops = _d._get_native_ops()
            if args.fast:
                rp, cl = _d._bsr_structure_lists(A)
                bl = Ad_dev.blocks
                row["static_kernel_ms"] = safe_bench(
                    lambda: ops["bsr_spmm_static"](bl, Bd, rp, cl), reps=args.reps
                )
                Ad = Ac.to(DEV, dt)
                if dense_dev_ms is None:
                    dense_dev_ms = safe_bench(lambda: Ad @ Bd, reps=args.reps)
                row["dense_dev_ms"] = dense_dev_ms
                Ad = None
                if row["trnsparse_ms"] and dense_dev_ms:
                    row["speedup_vs_dense_dev"] = dense_dev_ms / row["trnsparse_ms"]
                emit(row, args.out)
                continue

            # Split: device-side prep (gather) vs the kernel launch.
            def prep():
                return _d._bsr_pad_and_gather(_d._BlockView(Ad_dev, Ad_dev.blocks), Bd)

            bp, bg, *_ = prep()
            # Gather path (main before trnsparse#30 static kernels): device-side
            # pad + gather, then the K_max-padded kernel.
            row["gather_prep_ms"] = safe_bench(prep, reps=args.reps)
            row["gather_kernel_ms"] = safe_bench(lambda: ops["bsr_spmm"](bp, bg), reps=args.reps)
            bp = bg = None  # free the gathered tiles
            if "bsr_spmm_static" in ops:
                rp, cl = _d._bsr_structure_lists(A)
                bl = Ad_dev.blocks
                row["static_kernel_ms"] = safe_bench(
                    lambda: ops["bsr_spmm_static"](bl, Bd, rp, cl), reps=args.reps
                )

            Ad = Ac.to(DEV, dt)
            if dense_dev_ms is None:
                dense_dev_ms = safe_bench(lambda: Ad @ Bd, reps=args.reps)
            row["dense_dev_ms"] = dense_dev_ms
            Ad = None
            if cpu_dense_ms is None:
                cpu_dense_ms = safe_bench(lambda: Ac @ Bc, warmup=1, reps=args.cpu_reps)
            row["cpu_dense_ms"] = cpu_dense_ms
            try:
                Absr = Ac.to_sparse_bsr((P, P))
                row["cpu_torch_bsr_ms"] = safe_bench(
                    lambda: Absr @ Bc, warmup=1, reps=args.cpu_reps
                )
            except Exception as e:
                row["cpu_torch_bsr_ms"] = f"unsupported: {type(e).__name__}"
            Acsr = Ac.to_sparse_csr()
            row["cpu_torch_csr_ms"] = safe_bench(lambda: Acsr @ Bc, warmup=1, reps=args.cpu_reps)
            # trnsparse on CPU (pytorch backend): the loop-over-blocks fallback.
            prev = trnsparse.get_backend()
            trnsparse.set_backend("pytorch")
            row["cpu_trnsparse_ms"] = safe_bench(
                lambda: trnsparse.bsr_spmm(A, Bc), warmup=1, reps=args.cpu_reps
            )
            trnsparse.set_backend(prev)
            if row["trnsparse_ms"] and row["dense_dev_ms"]:
                row["speedup_vs_dense_dev"] = row["dense_dev_ms"] / row["trnsparse_ms"]
            emit(row, args.out)


# ── CSR ──────────────────────────────────────────────────────────────────────


def section_csr(args):
    densities = [float(x) for x in args.densities.split(",")]
    widths = [int(x) for x in args.widths.split(",")]
    n = args.size
    for width in widths:
        Bc = torch.randn(n, width, generator=torch.Generator().manual_seed(1))
        Bd = Bc.to(DEV)
        for dens in densities:
            g = torch.Generator().manual_seed(0)
            Ac = torch.randn(n, n, generator=g) * (torch.rand(n, n, generator=g) < dens)
            A = trnsparse.from_dense(Ac)
            row = {"section": "csr", "n": n, "width": width, "density": dens, "nnz": A.nnz}
            out = trnsparse.spmm(A, Bd)
            row["rel_err"] = rel_err(out, Ac.double() @ Bc.double())
            row["trnsparse_ms"] = safe_bench(lambda: trnsparse.spmm(A, Bd), reps=args.reps)
            Ad = Ac.to(DEV)
            row["dense_dev_ms"] = safe_bench(lambda: Ad @ Bd, reps=args.reps)
            row["densify_host_h2d_ms"] = safe_bench(
                lambda: trnsparse.formats._csr_to_dense_host(A).to(DEV), reps=args.reps
            )
            if hasattr(_d, "_csr_to_dense_on"):
                row["densify_on_device_ms"] = safe_bench(
                    lambda: _d._csr_to_dense_on(A, DEV), reps=args.reps
                )
            row["cpu_dense_ms"] = safe_bench(lambda: Ac @ Bc, warmup=1, reps=args.cpu_reps)
            Acsr = Ac.to_sparse_csr()
            row["cpu_torch_csr_ms"] = safe_bench(lambda: Acsr @ Bc, warmup=1, reps=args.cpu_reps)
            emit(row, args.out)
            Ad = None


# ── Screened SpMM ────────────────────────────────────────────────────────────


def section_screen(args):
    widths = [int(x) for x in args.widths.split(",")]
    n = args.size
    g = torch.Generator().manual_seed(0)
    # Exponentially decaying Schwarz bounds, so a threshold keeps a band.
    pos = torch.arange(n, dtype=torch.float32)
    diag = torch.exp(-pos / (n / 8)) + 1e-12
    Ac = torch.randn(n, n, generator=g)
    for width in widths:
        Bc = torch.randn(n, width, generator=g)
        Bd, Ad, diag_d = Bc.to(DEV), Ac.to(DEV), diag.to(DEV)
        for thr in [float(x) for x in args.thresholds.split(",")]:
            Q = diag.sqrt()
            mask = (Q[:, None] * Q[None, :]) > math.sqrt(thr)
            ref = (Ac.double() * mask) @ Bc.double()
            row = {
                "section": "screen",
                "n": n,
                "width": width,
                "threshold": thr,
                "kept_fraction": float(mask.float().mean()),
            }
            out = trnsparse.screened_spmm(Ad, diag_d, Bd, thr)
            row["rel_err"] = rel_err(out, ref)
            row["trnsparse_ms"] = safe_bench(
                lambda: trnsparse.screened_spmm(Ad, diag_d, Bd, thr), reps=args.reps
            )
            mask_d = mask.to(DEV, torch.float32)
            row["dense_masked_dev_ms"] = safe_bench(lambda: (Ad * mask_d) @ Bd, reps=args.reps)
            row["dense_dev_ms"] = safe_bench(lambda: Ad @ Bd, reps=args.reps)
            mf = mask.float()
            row["cpu_masked_ms"] = safe_bench(lambda: (Ac * mf) @ Bc, warmup=1, reps=args.cpu_reps)
            emit(row, args.out)


# ── Attention ────────────────────────────────────────────────────────────────


def block_mask(kind, nb):
    m = torch.zeros(nb, nb, dtype=torch.bool)
    w = 1  # local window radius in blocks (3 blocks per row)
    if kind == "local":
        for i in range(nb):
            m[i, max(0, i - w) : i + w + 1] = True
    elif kind == "dilated":
        stride = 4
        for i in range(nb):
            m[i, (torch.arange(nb) - i) % stride == 0] = True
    elif kind == "global":
        for i in range(nb):
            m[i, max(0, i - w) : i + w + 1] = True
        m[0, :] = True
        m[:, 0] = True
    else:
        raise ValueError(kind)
    return m


def mask_bsr_from_block_mask(bm):
    nb = bm.shape[0]
    row_ptrs = torch.zeros(nb + 1, dtype=torch.long)
    row_ptrs[1:] = bm.sum(1).cumsum(0)
    cols = bm.nonzero()[:, 1].to(torch.long)
    return trnsparse.BSRMatrix(
        blocks=torch.ones(len(cols), 1, 1),  # values unused by attention
        block_col_indices=cols,
        block_row_ptrs=row_ptrs,
        shape=(nb * P, nb * P),
        block_size=P,
    )


def section_attn(args):
    seq = args.size
    nb = seq // P
    F = torch.nn.functional
    dt = _DTYPES[args.dtype]
    for hd in [int(x) for x in args.head_dims.split(",")]:
        g = torch.Generator().manual_seed(0)
        Qc, Kc, Vc = (torch.randn(seq, hd, generator=g).to(dt).float() for _ in range(3))
        dOc = torch.randn(seq, hd, generator=g).to(dt).float()
        Qd, Kd, Vd, dOd = (t.to(DEV, dt) for t in (Qc, Kc, Vc, dOc))

        def sdpa_fwd(q, k, v, m=None):
            return F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=m)[0]

        dense_fwd = safe_bench(lambda: sdpa_fwd(Qd, Kd, Vd), reps=args.reps)

        def dense_fwd_bwd(m=None):
            q, k, v = (t.detach().requires_grad_() for t in (Qd, Kd, Vd))
            o = sdpa_fwd(q, k, v, m)
            o.backward(dOd)
            return q.grad

        dense_fb = safe_bench(dense_fwd_bwd, reps=args.reps)
        cpu_fwd = None
        if not args.fast:
            cpu_fwd = safe_bench(lambda: sdpa_fwd(Qc, Kc, Vc), warmup=1, reps=args.cpu_reps)
        for kind in args.masks.split(","):
            bm = block_mask(kind, nb)
            mb = mask_bsr_from_block_mask(bm)
            elem_mask = bm.repeat_interleave(P, 0).repeat_interleave(P, 1)
            counts = bm.sum(1).tolist()
            row = {
                "section": "attn",
                "dtype": args.dtype,
                "core_split": _core_split(),
                "seq": seq,
                "head_dim": hd,
                "mask": kind,
                "block_density": float(bm.float().mean()),
                "K_max": max(counts),
                "mean_blocks_per_row": sum(counts) / len(counts),
            }
            ref = sdpa_fwd(Qc.double(), Kc.double(), Vc.double(), elem_mask)
            out = trnsparse.block_sparse_attention_tiled(Qd, Kd, Vd, mb)
            row["fwd_rel_err"] = rel_err(out, ref)
            row["trnsparse_fwd_ms"] = safe_bench(
                lambda: trnsparse.block_sparse_attention_tiled(Qd, Kd, Vd, mb), reps=args.reps
            )

            def ts_fwd_bwd():
                q, k, v = (t.detach().requires_grad_() for t in (Qd, Kd, Vd))
                o = trnsparse.block_sparse_attention_tiled(q, k, v, mb)
                o.backward(dOd)
                return q.grad, k.grad, v.grad

            if not args.no_bwd:
                try:
                    gq, gk, gv = ts_fwd_bwd()
                    qr, kr, vr = (t.double().requires_grad_() for t in (Qc, Kc, Vc))
                    sdpa_fwd(qr, kr, vr, elem_mask).backward(dOc.double())
                    row["bwd_rel_err"] = max(
                        rel_err(gq, qr.grad), rel_err(gk, kr.grad), rel_err(gv, vr.grad)
                    )
                    row["trnsparse_fwd_bwd_ms"] = safe_bench(ts_fwd_bwd, reps=args.reps)
                except Exception as e:
                    row["trnsparse_fwd_bwd_ms"] = f"failed: {type(e).__name__}: {str(e)[:120]}"
            row["sdpa_dense_dev_fwd_ms"] = dense_fwd
            row["sdpa_dense_dev_fwd_bwd_ms"] = dense_fb
            em_d = elem_mask.to(DEV)
            row["sdpa_masked_dev_fwd_ms"] = safe_bench(
                lambda: sdpa_fwd(Qd, Kd, Vd, em_d), reps=args.reps
            )
            if not args.fast:
                row["sdpa_cpu_fwd_ms"] = cpu_fwd
                row["sdpa_masked_cpu_fwd_ms"] = safe_bench(
                    lambda: sdpa_fwd(Qc, Kc, Vc, elem_mask), warmup=1, reps=args.cpu_reps
                )
            emit(row, args.out)
            em_d = None


def _core_split():
    from trnsparse.nki import kernels

    return kernels._core_split_enabled()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True, choices=["bsr", "csr", "screen", "attn"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--densities", default="0.01,0.05,0.1,0.25,0.5")
    ap.add_argument("--widths", default="128,512")
    ap.add_argument("--thresholds", default="1e-12,1e-4,1e-2")
    ap.add_argument("--head-dims", default="64,128")
    ap.add_argument("--masks", default="local,dilated,global")
    ap.add_argument("--no-bwd", action="store_true")
    ap.add_argument("--dtype", default="fp32", choices=sorted(_DTYPES))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--cpu-reps", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    trnsparse.set_backend("nki")
    {"bsr": section_bsr, "csr": section_csr, "screen": section_screen, "attn": section_attn}[
        args.section
    ](args)


if __name__ == "__main__":
    main()
