"""Iterative linear solvers for BSR sparse matrices.

v0.3.2 — Phase 1 plumbing (#22). Python-level CG and power iteration
on top of `bsr_spmm` as the matvec. A gets reloaded from HBM on every
iteration; v0.4.0 (follow-up) delivers the SBUF-resident fused NKI
kernel that keeps A on-chip across iterations — the architectural win.

Algorithm body for CG mirrors `trnsolver.iterative.cg` at commit-time,
kept local to avoid a trnsolver runtime dep.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .formats import BSRMatrix
from .ops import bsr_spmm


def bsr_diagonal(A: BSRMatrix) -> torch.Tensor:
    """Extract the main diagonal of a BSR matrix as a dense vector.

    The diagonal lives inside the diagonal blocks (block row i at block
    column i). Blocks that aren't diagonal contribute nothing. Returns
    a zero vector if no diagonal block is stored (a fully zero-diagonal
    matrix would be degenerate as a preconditioner anyway).
    """
    m, _ = A.shape
    b = A.block_size
    M_tiles = (m + b - 1) // b
    out = torch.zeros(M_tiles * b, dtype=A.dtype)
    for i in range(M_tiles):
        start = A.block_row_ptrs[i].item()
        end = A.block_row_ptrs[i + 1].item()
        for k in range(start, end):
            if A.block_col_indices[k].item() == i:
                out[i * b : (i + 1) * b] = torch.diagonal(A.blocks[k])
                break
    return out[:m]


def _bsr_matvec(A: BSRMatrix) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap `bsr_spmm` so CG / power iteration can pass it a vector."""

    def matvec(x: torch.Tensor) -> torch.Tensor:
        return bsr_spmm(A, x.unsqueeze(1)).squeeze(1)

    return matvec


def cg_bsr(
    A: BSRMatrix,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    M: Callable[[torch.Tensor], torch.Tensor] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Conjugate Gradient solver for `A @ x = b` with A SPD and BSR-stored.

    Thin specialization of the standard CG loop: the matvec goes through
    `bsr_spmm`, so on the NKI backend each iteration dispatches one
    kernel call. That amortizes poorly across many iterations — the
    v0.4.0 fused kernel fuses the loop and keeps A SBUF-resident.

    Args:
        A: SPD BSR matrix (N, N).
        b: Right-hand side, shape (N,).
        x0: Initial guess (default: zeros).
        tol: Relative-residual tolerance on `||r|| / ||b||`.
        max_iter: Iteration cap.
        M: Preconditioner — callable `M(r) → M^{-1} r` or a dense
            inverse tensor. Build a Jacobi preconditioner from
            `bsr_diagonal(A)` for the common diagonally-dominant case.

    Returns:
        `(x, iters, rel_residual)` — solution, iteration count, final
        relative residual.
    """
    n = b.shape[0]
    matvec = _bsr_matvec(A)

    if M is None:
        precond: Callable[[torch.Tensor], torch.Tensor] | None = None
    elif callable(M):
        precond = M
    else:
        M_tensor = M

        def precond(r: torch.Tensor) -> torch.Tensor:
            return torch.mv(M_tensor, r)

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
    r = b - matvec(x)
    b_norm = torch.linalg.norm(b).item()
    if b_norm < 1e-15:
        return x, 0, 0.0

    z = precond(r) if precond is not None else r.clone()
    p = z.clone()
    rz = torch.dot(r, z).item()

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = torch.dot(p, Ap).item()
        if abs(pAp) < 1e-30:
            break

        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.linalg.norm(r).item()
        rel = r_norm / b_norm
        if rel < tol:
            return x, k + 1, rel

        z = precond(r) if precond is not None else r
        rz_new = torch.dot(r, z).item()
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, max_iter, r_norm / b_norm


def jacobi_preconditioner_bsr(
    A: BSRMatrix,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a Jacobi (diagonal) preconditioner for a BSR matrix.

    Returns a callable `M(r) = r / diag(A)`. Cheap + effective when A
    is diagonally dominant.
    """
    d = bsr_diagonal(A)
    if torch.any(d.abs() < 1e-15):
        raise ValueError("Jacobi preconditioner: diagonal has near-zero entries")
    inv_d = 1.0 / d

    def precond(r: torch.Tensor) -> torch.Tensor:
        return r * inv_d

    return precond


def power_iteration_bsr(
    A: BSRMatrix,
    v0: torch.Tensor | None = None,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> tuple[float, torch.Tensor, int]:
    """Find the dominant eigenpair (λ, v) of a BSR matrix via power iteration.

    Converges to the eigenvector with largest `|λ|`. Tolerance is on
    the Rayleigh-quotient change between iterations.

    Args:
        A: Square BSR matrix (N, N). Doesn't need to be SPD — power
            iteration works for any matrix with a unique dominant
            eigenvalue, but convergence slows as the spectral gap
            shrinks.
        v0: Initial vector (default: random unit vector).
        max_iter: Iteration cap.
        tol: Convergence on `|λ_k - λ_{k-1}|`.

    Returns:
        `(eigenvalue, eigenvector, iters)`.
    """
    n = A.shape[0]
    matvec = _bsr_matvec(A)

    v = v0.clone() if v0 is not None else torch.randn(n, dtype=A.dtype)
    v = v / torch.linalg.norm(v)

    lam_prev = 0.0
    for k in range(max_iter):
        Av = matvec(v)
        lam = torch.dot(v, Av).item()
        norm = torch.linalg.norm(Av).item()
        if norm < 1e-30:
            return 0.0, v, k + 1
        v = Av / norm

        if abs(lam - lam_prev) < tol:
            return lam, v, k + 1
        lam_prev = lam

    return lam, v, max_iter
