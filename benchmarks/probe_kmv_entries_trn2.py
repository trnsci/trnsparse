"""Per-entry accuracy of the fused kernel-matrix tile on the device (trnsci/trnsci#173).

Isolates the kernel-function evaluation (sqrt, exp, polynomial) from the
distance expansion: X and Y lie on a dyadic grid (coordinates k / 16), so
every term of |x|^2 + |y|^2 - 2 x.y is exact in fp32 and D is exact, and
V = I (m = 128 columns), so out = K(X, Y) entry by entry with no
summation rounding. Each entry is compared with float64 k(D).

The host centring of `kernel_matmul` is bypassed (it would round the grid):
the kernel launch is called directly with the scale coefficient `a` as a
kernel argument (powers of two, so a * D is exact too).

Usage (on the device):
    python benchmarks/probe_kmv_entries_trn2.py [--n 4096] [--a 1,16] [--json out.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from trnsparse.kernel_matmul import _params
from trnsparse.nki import dispatch as _d
from trnsparse.nki.kernels import _KMV_KINDS, kmv_group


def _grid(rng, n, d):
    return (rng.integers(-16, 17, size=(n, d)) / 16.0).astype(np.float32)


def _ref(kind, D, a):
    if kind == 0:
        return np.exp(a * D)
    t = np.sqrt(a * D)
    e = np.exp(-t)
    return [None, e, (1 + t) * e, (1 + t + t * t / 3) * e][kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--a", default="1,16", help="Matern a values (rbf uses -a / 2)")
    ap.add_argument("--kinds", default=",".join(_KMV_KINDS))
    ap.add_argument("--json", default=None)
    ap.add_argument("--sim", action="store_true", help="NKI simulator on the host")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    m = 128
    X = _grid(rng, args.n, args.d)
    Y = _grid(rng, m, args.d)
    D = ((X[:, None, :].astype(np.float64) - Y[None, :, :]) ** 2).sum(-1)
    V = torch.eye(m, dtype=torch.float32)
    dev = "cpu" if args.sim else "neuron"
    xd, yd, vd = (torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev), V.to(dev))
    rows = []
    print(f"n={args.n} m={m} d={args.d}")
    print(f"{'kernel':>9} {'a':>5} {'max_rel':>9} {'p99.9':>9} {'median':>9} {'k>=1e-3 max':>11}")
    for name in args.kinds.split(","):
        kind = _KMV_KINDS.index(name)
        for a in (float(v) for v in args.a.split(",")):
            aa = -0.5 * a if kind == 0 else a
            prm = _params(aa, 1.0, 0.0, dev)
            out = _d._kmv_launch(xd, yd, vd, None, prm, kind, kmv_group(m), args.sim).cpu()
            got = out.double().numpy()
            ref = _ref(kind, D, aa)
            ok = ref > 1e-30
            rel = np.abs(got - ref)[ok] / ref[ok]
            big = ref >= 1e-3
            rel_big = np.abs(got - ref)[big] / ref[big]
            rec = {
                "kernel": name,
                "a": a,
                "max_rel": float(rel.max()),
                "p999_rel": float(np.quantile(rel, 0.999)),
                "median_rel": float(np.median(rel)),
                "max_rel_k_ge_1e-3": float(rel_big.max()),
                "entries": int(ok.sum()),
            }
            rows.append(rec)
            print(
                f"{name:>9} {a:>5g} {rec['max_rel']:>9.2e} {rec['p999_rel']:>9.2e} "
                f"{rec['median_rel']:>9.2e} {rec['max_rel_k_ge_1e-3']:>11.2e}",
                flush=True,
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
