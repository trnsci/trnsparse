"""Reductions over n that do not depend on the LNC split (trnsci/trnsolver#128).

Device aten reductions over n (``dot``, ``norm``, ``sum``, matmul) add in an
order that depends on how the compiler splits the work across logical cores:
on Trn2 at n = 1e5 a column sum differs by about 5e-4 between LNC=1 and
LNC=2 (bitwise reproducible within one LNC). Elementwise ops are bitwise
equal across LNC. This mirrors ``trnsolver.block`` (same tree, same leaf
size), kept local so trnsparse has no trnsolver runtime dependency:

- ``"pairwise"``: zero-pad to m · 2^L rows (m ≤ 64), fold with L elementwise
  adds of contiguous halves on the device (a fixed tree that depends on n
  only), copy the m leaves to the host and sum them in float64.
- ``"host64"``: copy the elementwise product to the host, sum in float64.
- ``"device"``: ``torch.dot`` / ``torch.linalg.norm`` on the device.

Host (CPU) operands are multiplied in float64 in the folded modes.
"""

from __future__ import annotations

import math
import os

import torch

REDUCTIONS = ("pairwise", "host64", "device")
_FOLD_LEAF = 64
_HOST = torch.device("cpu")


def reduction_mode(reduction: str | None, env: str, default: str) -> str:
    """Resolve ``reduction=...``; ``None`` reads the ``env`` variable (else ``default``)."""
    if reduction is None:
        reduction = os.environ.get(env, "").strip().lower() or default
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}; got {reduction!r}")
    return reduction


def fold_plan(n: int, leaf: int = _FOLD_LEAF) -> tuple[int, int]:
    """(levels L, leaves m) with m = ceil(n / 2^L) ≤ ``leaf``."""
    lev = 0
    while -(-n >> lev) > leaf:
        lev += 1
    return lev, -(-n >> lev)


def fold_host64(t: torch.Tensor, leaf: int = _FOLD_LEAF) -> torch.Tensor:
    """Σ over dim 0 by a fixed pairwise tree (device halvings, then host float64)."""
    n = t.shape[0]
    lev, m = fold_plan(n, leaf)
    npad = m << lev
    if npad != n:
        t = torch.cat([t, t.new_zeros((npad - n, *t.shape[1:]))], 0)
    for _ in range(lev):
        h = t.shape[0] // 2
        t = t[:h] + t[h:]
    return t.to(device=_HOST, dtype=torch.float64).sum(dim=0)


def _real_prod(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if u.is_complex() or v.is_complex():
        u, v = torch.view_as_real(u.resolve_conj()), torch.view_as_real(v.resolve_conj())
        return u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1]
    return u * v


def dot(u: torch.Tensor, v: torch.Tensor, reduction: str) -> float:
    """Re(uᴴ v) of two vectors as a Python float."""
    if reduction == "device":
        return torch.dot(u, v).item()
    if u.device.type == "cpu":
        wide = torch.complex128 if (u.is_complex() or v.is_complex()) else torch.float64
        u, v = u.to(wide), v.to(wide)
    P = _real_prod(u, v)
    if reduction == "host64":
        return float(P.to(device=_HOST, dtype=torch.float64).sum())
    return float(fold_host64(P))


def norm(u: torch.Tensor, reduction: str) -> float:
    """‖u‖₂ as a Python float."""
    if reduction == "device":
        return torch.linalg.norm(u).item()
    if reduction == "host64":
        w = torch.complex128 if u.is_complex() else torch.float64
        return float(torch.linalg.vector_norm(u.to(device=_HOST, dtype=w)))
    return math.sqrt(max(dot(u, u, "pairwise"), 0.0))
