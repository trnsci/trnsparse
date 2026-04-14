"""Iterative solver tests on BSR matrices (CPU plumbing — #22 Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import trnsparse


def _random_spd_bsr(
    n: int, block_size: int = 128, seed: int = 0
) -> tuple[torch.Tensor, trnsparse.BSRMatrix]:
    """Build a dense SPD matrix of size (n, n), return (dense, BSR view).

    `A = M @ M.T + n*I` guarantees SPD. Block-storing a dense SPD matrix
    is valid — BSR just chooses which blocks to keep; SPD-ness is a
    per-element property that survives the block partition.
    """
    torch.manual_seed(seed)
    M = torch.randn(n, n)
    A_dense = M @ M.T + n * torch.eye(n)
    return A_dense, trnsparse.BSRMatrix.from_dense(A_dense, block_size=block_size)


class TestCgBsr:
    def test_identity_converges_immediately(self):
        """A = I → x = b in one iteration (or zero for zero RHS)."""
        n = 128
        I_dense = torch.eye(n)
        I_bsr = trnsparse.BSRMatrix.from_dense(I_dense, block_size=128)
        b = torch.randn(n)

        x, iters, rel = trnsparse.cg_bsr(I_bsr, b, tol=1e-8, max_iter=50)
        torch.testing.assert_close(x, b, atol=1e-5, rtol=1e-5)
        assert iters <= 2, f"identity should converge in 1 iter, took {iters}"
        assert rel < 1e-8

    def test_parity_vs_scipy(self):
        """Solution from cg_bsr matches scipy.sparse.linalg.cg within tol."""
        sp = pytest.importorskip("scipy.sparse")
        spla = pytest.importorskip("scipy.sparse.linalg")

        n = 128
        A_dense, A_bsr = _random_spd_bsr(n, block_size=128, seed=1)
        b = torch.randn(n)

        x_ours, iters_ours, rel_ours = trnsparse.cg_bsr(A_bsr, b, tol=1e-8, max_iter=2 * n)
        # scipy's cg takes rtol (relative to ||b||) by default in modern scipy
        A_scipy = sp.csr_matrix(A_dense.numpy())
        x_scipy, info = spla.cg(A_scipy, b.numpy(), rtol=1e-8, maxiter=2 * n)

        assert info == 0, "scipy CG should converge on this well-conditioned SPD system"
        np.testing.assert_allclose(x_ours.numpy(), x_scipy, atol=1e-4, rtol=1e-4)
        assert rel_ours < 1e-7

    def test_jacobi_preconditioner_fewer_iters(self):
        """A strongly diagonally-dominant system should converge faster
        with Jacobi preconditioning than without.
        """
        torch.manual_seed(2)
        n = 256
        # Diagonally dominant: large diagonal, small off-diag.
        A_dense = 0.01 * torch.randn(n, n)
        A_dense = 0.5 * (A_dense + A_dense.T)  # symmetric
        A_dense = A_dense + torch.diag(1.0 + 10.0 * torch.rand(n))  # SPD
        A_bsr = trnsparse.BSRMatrix.from_dense(A_dense, block_size=128)
        b = torch.randn(n)

        _, iters_plain, _ = trnsparse.cg_bsr(A_bsr, b, tol=1e-8, max_iter=500)
        M = trnsparse.jacobi_preconditioner_bsr(A_bsr)
        _, iters_precond, _ = trnsparse.cg_bsr(A_bsr, b, tol=1e-8, max_iter=500, M=M)

        assert iters_precond <= iters_plain, (
            f"Jacobi preconditioner should not make it worse "
            f"(plain={iters_plain}, precond={iters_precond})"
        )

    def test_zero_rhs_returns_zero(self):
        """b = 0 → x = 0 in 0 iterations."""
        _, A_bsr = _random_spd_bsr(128, block_size=128, seed=3)
        b = torch.zeros(128)
        x, iters, rel = trnsparse.cg_bsr(A_bsr, b)
        torch.testing.assert_close(x, b)
        assert iters == 0
        assert rel == 0.0


class TestPowerIterationBsr:
    def test_dominant_eigenvalue_diagonal(self):
        """Diagonal matrix: dominant eigenvalue is max(diag); eigenvector
        is the corresponding basis vector.
        """
        n = 128
        diag = torch.linspace(0.5, 10.0, n)  # strictly increasing; max = 10
        A_dense = torch.diag(diag)
        A_bsr = trnsparse.BSRMatrix.from_dense(A_dense, block_size=128)

        lam, v, iters = trnsparse.power_iteration_bsr(A_bsr, max_iter=500, tol=1e-10)
        assert abs(lam - 10.0) < 1e-3, f"expected ~10.0, got {lam}"
        # Eigenvector aligns with e_{n-1} (corresponding to max diag entry).
        assert abs(abs(v[-1].item()) - 1.0) < 1e-3

    def test_dominant_eigenvalue_vs_torch(self):
        """Match torch.linalg.eigvalsh's largest eigenvalue within 1e-3."""
        torch.manual_seed(4)
        n = 128
        A_dense, A_bsr = _random_spd_bsr(n, block_size=128, seed=4)

        lam_ours, _, _ = trnsparse.power_iteration_bsr(A_bsr, max_iter=2000, tol=1e-10)
        lam_ref = torch.linalg.eigvalsh(A_dense).max().item()

        # Power iteration on a random SPD matrix has a spectral gap;
        # 1e-3 relative tolerance is generous.
        assert abs(lam_ours - lam_ref) / lam_ref < 1e-3, f"ours={lam_ours}, ref={lam_ref}"


class TestBsrDiagonal:
    def test_diagonal_of_identity(self):
        I_bsr = trnsparse.BSRMatrix.from_dense(torch.eye(128), block_size=128)
        d = trnsparse.bsr_diagonal(I_bsr)
        torch.testing.assert_close(d, torch.ones(128))

    def test_diagonal_of_off_diagonal_blocks_only(self):
        """A matrix whose only stored blocks are off-diagonal returns
        zeros on the diagonal.
        """
        torch.manual_seed(5)
        A_dense = torch.zeros(256, 256)
        A_dense[:128, 128:] = torch.randn(128, 128)
        A_dense[128:, :128] = torch.randn(128, 128)
        A_bsr = trnsparse.BSRMatrix.from_dense(A_dense, block_size=128)
        d = trnsparse.bsr_diagonal(A_bsr)
        torch.testing.assert_close(d, torch.zeros(256))
