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


def chebyshev_coeffs(
    lam_min: float,
    lam_max: float,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chebyshev semi-iteration step coefficients for K fixed iterations.

    Returns (alpha, beta) each of shape (K,) such that the iteration

        x_{k+1} = x_k + alpha[k] * r_k + beta[k] * (x_k - x_{k-1})

    (with r_k = b - A @ x_k) minimises the Chebyshev residual polynomial
    over [lam_min, lam_max] after K steps.

    Algorithm: Templates for the Solution of Linear Systems (Barrett et
    al. 1994), Algorithm 2.1 converted from direction-vector form to the
    momentum / heavy-ball form. c = half-range, d = centre.

    Args:
        lam_min: Lower eigenvalue bound (must be > 0 for SPD A).
        lam_max: Upper eigenvalue bound.
        K: Number of iterations for which coefficients are needed.

    Returns:
        (alpha, beta): float32 tensors of length K.
    """
    c = (lam_max - lam_min) / 2.0  # half-range
    d = (lam_max + lam_min) / 2.0  # centre

    alpha = torch.zeros(K, dtype=torch.float64)
    beta = torch.zeros(K, dtype=torch.float64)

    # k=0: pure Richardson at the optimal single-step rate.
    alpha_prev = 1.0 / d
    alpha[0] = alpha_prev
    beta[0] = 0.0

    for k in range(1, K):
        # Templates step: beta_t = (c * alpha_prev / 2)^2
        #                  alpha_new = 1 / (d - beta_t / alpha_prev)
        beta_t = (c * alpha_prev / 2.0) ** 2
        alpha_new = 1.0 / (d - beta_t / alpha_prev)
        # Convert to momentum form:
        #   alpha_k^(momentum) = alpha_new
        #   beta_k^(momentum)  = alpha_new * beta_t / alpha_prev
        alpha[k] = alpha_new
        beta[k] = alpha_new * beta_t / alpha_prev
        alpha_prev = alpha_new

    return alpha.float(), beta.float()


def chebyshev_bsr(
    A: BSRMatrix,
    b: torch.Tensor,
    lam_min: float,
    lam_max: float,
    K: int = 50,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Chebyshev semi-iteration for `A @ x = b` with A SPD and BSR-stored.

    Runs exactly K iterations with pre-computed coefficients derived from
    eigenvalue bounds [lam_min, lam_max]. No inner products are computed
    during the iteration — all coefficients are determined before the
    first step. This is the key structural difference from `cg_bsr`:
    the loop body is purely matvec + elementwise ops, which maps directly
    to the fixed-iteration NKI kernel pattern.

    Obtain lam_max from `power_iteration_bsr`. A conservative lam_min
    estimate (e.g. lam_max / condition_number_estimate) is sufficient;
    over-estimating the spectral range only slows convergence slightly,
    not correctness.

    Args:
        A: SPD BSR matrix (N, N).
        b: Right-hand side, shape (N,).
        lam_min: Lower eigenvalue bound (> 0).
        lam_max: Upper eigenvalue bound.
        K: Fixed iteration count.
        x0: Initial guess (default: zeros).

    Returns:
        `(x, K, rel_residual)` — solution, iteration count (always K),
        final relative residual ||b - A @ x|| / ||b||.
    """
    n = b.shape[0]
    matvec = _bsr_matvec(A)

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
    x_prev = x.clone()
    b_norm = torch.linalg.norm(b).item()

    alpha, beta = chebyshev_coeffs(lam_min, lam_max, K)

    for k in range(K):
        r = b - matvec(x)
        x_new = x + alpha[k].item() * r
        if k > 0:
            x_new = x_new + beta[k].item() * (x - x_prev)
        x_prev = x
        x = x_new

    r_final = b - matvec(x)
    rel = torch.linalg.norm(r_final).item() / max(b_norm, 1e-15)
    return x, K, rel


def richardson_bsr(
    A: BSRMatrix,
    b: torch.Tensor,
    omega: float,
    K: int = 100,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Fixed-K Richardson iteration: x_{k+1} = x_k + omega * (b - A @ x_k).

    The simplest fixed-point iteration with no adaptive coefficients.
    Converges when 0 < omega < 2 / lam_max. The optimal static step is
    omega = 2 / (lam_min + lam_max).

    Like `chebyshev_bsr`, the loop runs for exactly K iterations with no
    adaptive stopping — the right structure for eventual NKI fusion.

    Args:
        A: BSR matrix (N, N).
        b: Right-hand side, shape (N,).
        omega: Relaxation parameter. Use 2 / (lam_min + lam_max) for
            optimal convergence on SPD systems.
        K: Fixed iteration count.
        x0: Initial guess (default: zeros).

    Returns:
        `(x, K, rel_residual)`.
    """
    n = b.shape[0]
    matvec = _bsr_matvec(A)

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
    b_norm = torch.linalg.norm(b).item()

    for _ in range(K):
        r = b - matvec(x)
        x = x + omega * r

    r_final = b - matvec(x)
    rel = torch.linalg.norm(r_final).item() / max(b_norm, 1e-15)
    return x, K, rel


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
