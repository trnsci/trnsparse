"""Relative error of ScalarE sqrt and GpSimd rsqrt * x by input range, on the device (trnsci/trnsci#173).

One 128 x 2048 tile of fp32 inputs, log-uniform over [2^-124, 2^124],
through `nisa.activation(sqrt)` (ScalarE) and through
`tensor_scalar(rsqrt)` on GpSimd times x, against float64 sqrt. Prints the
overall maximum and the maximum per 8 binades. On Trn2 (SDK 2.33.1) ScalarE
sqrt is 8.4e-7 relative in [2^-52, 2^52], and its 6.2e-3 worst case is
above 2^92; GpSimd rsqrt * x is 1.1e-7 everywhere.

Usage (on the device):
    python benchmarks/probe_scalare_sqrt_range_trn2.py
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch

_P, _F = 128, 2048
_LO, _HI, _STEP = -124, 124, 8


@nki.jit
def _scalare_sqrt(x):
    out = nl.ndarray((_P, _F), dtype=nl.float32, buffer=nl.shared_hbm)
    t = nl.load(x)
    r = nl.ndarray((_P, _F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(r, nl.sqrt, t)
    nl.store(out, value=r)
    return out


@nki.jit
def _gpsimd_rsqrt_x(x):
    out = nl.ndarray((_P, _F), dtype=nl.float32, buffer=nl.shared_hbm)
    t = nl.load(x)
    r = nl.ndarray((_P, _F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(r, t, op0=nl.rsqrt, operand0=0.0, engine=nisa.gpsimd_engine)
    nisa.tensor_tensor(r, t, r, op=nl.multiply)
    nl.store(out, value=r)
    return out


def main():
    rng = np.random.default_rng(0)
    x = np.exp2(rng.uniform(_LO, _HI, size=(_P, _F))).astype(np.float32)
    x64 = x.astype(np.float64)
    ref = np.sqrt(x64)
    e = np.floor(np.log2(x64))
    xd = torch.from_numpy(x).to("neuron")
    for name, kern in (("ScalarE sqrt", _scalare_sqrt), ("GpSimd rsqrt * x", _gpsimd_rsqrt_x)):
        rel = np.abs(kern(xd).cpu().double().numpy() - ref) / ref
        print(f"{name}: max {rel.max():.2e} at x = {x.flat[rel.argmax()]:.3e}")
        for b in range(_LO, _HI, _STEP):
            m = (e >= b) & (e < b + _STEP)
            print(f"  [2^{b}, 2^{b + _STEP}): max {rel[m].max():.2e}")


if __name__ == "__main__":
    main()
