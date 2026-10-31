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


class TestRichardsonBsr:
    def test_identity_converges_one_step(self):
        """A = I, omega = 1 → x = b after exactly one iteration."""
        n = 128
        I_bsr = trnsparse.BSRMatrix.from_dense(torch.eye(n), block_size=128)
        b = torch.randn(n)
        x, iters, rel = trnsparse.richardson_bsr(I_bsr, b, omega=1.0, K=1)
        torch.testing.assert_close(x, b, atol=1e-5, rtol=1e-5)
        assert iters == 1
        assert rel < 1e-5

    def test_spd_parity_vs_cg(self):
        """Richardson (large K, optimal omega) matches cg_bsr solution."""
        n = 128
        A_dense, A_bsr = _random_spd_bsr(n, block_size=128, seed=6)
        b = torch.randn(n)

        lam = torch.linalg.eigvalsh(A_dense)
        lam_min, lam_max = lam.min().item(), lam.max().item()
        omega = 2.0 / (lam_min + lam_max)  # optimal static step

        # Richardson needs more iterations than CG on ill-conditioned A.
        # For this well-conditioned synthetic matrix K=500 is sufficient.
        x_rich, _, rel_rich = trnsparse.richardson_bsr(A_bsr, b, omega=omega, K=500)
        x_cg, _, _ = trnsparse.cg_bsr(A_bsr, b, tol=1e-8, max_iter=200)
        torch.testing.assert_close(x_rich, x_cg, atol=1e-3, rtol=1e-3)
        assert rel_rich < 1e-4

    def test_omega_too_large_diverges(self):
        """omega > 2 / lam_max → iteration diverges (residual grows)."""
        n = 128
        A_dense, A_bsr = _random_spd_bsr(n, block_size=128, seed=7)
        b = torch.randn(n)
        lam_max = torch.linalg.eigvalsh(A_dense).max().item()

        omega_bad = 2.5 / lam_max  # exceeds stability bound
        x0 = torch.zeros(n)
        r0_norm = torch.linalg.norm(b - A_dense @ x0).item()

        x_div, _, rel = trnsparse.richardson_bsr(A_bsr, b, omega=omega_bad, K=20, x0=x0)
        r_norm = torch.linalg.norm(b - A_dense @ x_div).item()
        assert r_norm > r0_norm, "Unstable omega should cause divergence"


class TestChebyshevBsr:
    def test_parity_vs_cg(self):
        """chebyshev_bsr solution matches cg_bsr within atol=1e-3."""
        n = 128
        A_dense, A_bsr = _random_spd_bsr(n, block_size=128, seed=8)
        b = torch.randn(n)

        lam = torch.linalg.eigvalsh(A_dense)
        lam_min, lam_max = lam.min().item(), lam.max().item()

        x_cheb, _, rel = trnsparse.chebyshev_bsr(A_bsr, b, lam_min, lam_max, K=100)
        x_cg, _, _ = trnsparse.cg_bsr(A_bsr, b, tol=1e-8, max_iter=200)
        torch.testing.assert_close(x_cheb, x_cg, atol=1e-3, rtol=1e-3)
        assert rel < 1e-4

    def test_coeffs_shape(self):
        """chebyshev_coeffs returns (K,) float32 tensors; alpha[0] = 1/d."""
        K = 10
        lam_min, lam_max = 1.0, 100.0
        alpha, beta = trnsparse.chebyshev_coeffs(lam_min, lam_max, K)
        assert alpha.shape == (K,)
        assert beta.shape == (K,)
        assert alpha.dtype == torch.float32
        d = (lam_max + lam_min) / 2.0
        assert abs(alpha[0].item() - 1.0 / d) < 1e-5
        assert abs(beta[0].item()) < 1e-6  # first step has no momentum

    def test_lower_residual_than_richardson_same_K(self):
        """For well-conditioned SPD, Chebyshev beats Richardson at same K."""
        n = 128
        A_dense, A_bsr = _random_spd_bsr(n, block_size=128, seed=9)
        b = torch.randn(n)

        lam = torch.linalg.eigvalsh(A_dense)
        lam_min, lam_max = lam.min().item(), lam.max().item()
        omega = 2.0 / (lam_min + lam_max)

        K = 30
        _, _, rel_rich = trnsparse.richardson_bsr(A_bsr, b, omega=omega, K=K)
        _, _, rel_cheb = trnsparse.chebyshev_bsr(A_bsr, b, lam_min, lam_max, K=K)

        assert rel_cheb <= rel_rich, (
            f"Chebyshev should converge at least as fast as Richardson "
            f"(cheb={rel_cheb:.4f}, rich={rel_rich:.4f})"
        )


# ── cg_bsr_fused: host path (#22). The NKI kernel runs the same algorithm;
# tests/test_nki_cg_fused.py checks it against this path on hardware. ────────


def _block_spd(T: int, band: int = 1, n: int | None = None, seed: int = 0, shift: float = 0.5):
    """Block-banded SPD matrix (fp64) with 128x128 blocks; optional order n < 128 T."""
    g = torch.Generator().manual_seed(seed)
    N = T * 128
    M = torch.randn(N, N, generator=g, dtype=torch.float64) / N**0.5
    mask = torch.zeros(T, T, dtype=torch.float64)
    for i in range(T):
        mask[i, max(0, i - band) : i + band + 1] = 1.0
    M = M * torch.kron(mask, torch.ones(128, 128, dtype=torch.float64))
    A = (M + M.T) / 2
    A = A + (torch.linalg.eigvalsh(A)[0].abs() + shift) * torch.eye(N, dtype=torch.float64)
    if n is not None:
        A = A[:n, :n].contiguous()
    return A


class TestCgBsrFused:
    def test_matches_direct_solve(self):
        A = _block_spd(3)
        b = torch.randn(A.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        x, iters, rel = trnsparse.cg_bsr_fused(bsr, b, tol=1e-10, max_iter=500)
        assert x.dtype == torch.float64
        assert rel < 1e-10
        torch.testing.assert_close(x, torch.linalg.solve(A, b), atol=1e-8, rtol=1e-8)
        assert 0 < iters < 500

    def test_parity_vs_scipy_cg(self):
        sp = pytest.importorskip("scipy.sparse")
        spla = pytest.importorskip("scipy.sparse.linalg")
        A = _block_spd(4, band=1, seed=2)
        b = torch.randn(A.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        x, iters, _ = trnsparse.cg_bsr_fused(bsr, b, tol=1e-8, max_iter=1000)
        count = [0]
        xs, info = spla.cg(
            sp.csr_matrix(A.numpy()),
            b.numpy(),
            rtol=1e-8,
            atol=0.0,
            maxiter=1000,
            callback=lambda _: count.__setitem__(0, count[0] + 1),
        )
        assert info == 0
        np.testing.assert_allclose(x.numpy(), xs, atol=1e-6, rtol=1e-6)
        assert abs(iters - count[0]) <= 1

    def test_block_jacobi_fewer_iterations(self):
        # Badly scaled diagonal blocks: block-Jacobi removes the scaling.
        A = _block_spd(4, seed=4)
        s = torch.repeat_interleave(torch.tensor([1.0, 30.0, 0.05, 4.0], dtype=torch.float64), 128)
        A = s[:, None] ** 0.5 * A * s[None, :] ** 0.5
        b = torch.randn(A.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        x0, k0, _ = trnsparse.cg_bsr_fused(bsr, b, tol=1e-8, max_iter=2000)
        x1, k1, r1 = trnsparse.cg_bsr_fused(
            bsr, b, tol=1e-8, max_iter=2000, preconditioner="block_jacobi"
        )
        assert r1 < 1e-8
        assert k1 < k0
        ref = torch.linalg.solve(A, b)
        torch.testing.assert_close(x1, ref, atol=1e-6, rtol=1e-6)

    def test_order_not_multiple_of_128(self):
        A = _block_spd(3, n=300, seed=6)
        b = torch.randn(300, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        assert bsr.shape == (300, 300)  # last block row/column zero-padded
        # the operator puts identity on the padded diagonal (SPD padded system)
        x, _, rel = trnsparse.cg_bsr_fused(bsr, b, tol=1e-10, preconditioner="block_jacobi")
        assert x.shape == (300,) and rel < 1e-10
        torch.testing.assert_close(x, torch.linalg.solve(A, b), atol=1e-8, rtol=1e-8)

    def test_initial_guess_and_cap(self):
        A = _block_spd(2, seed=8)
        b = torch.randn(A.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(9))
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        ref = torch.linalg.solve(A, b)
        x, iters, rel = trnsparse.cg_bsr_fused(bsr, b, x0=ref, tol=1e-6)
        assert iters == 0 and rel < 1e-6
        x, iters, rel, info = trnsparse.cg_bsr_fused(
            bsr, b, tol=1e-14, max_iter=5, check_every=4, return_info=True
        )
        assert iters == 5 and rel > 1e-14
        assert info["path"] == "pytorch" and info["trips"] == 2

    def test_zero_rhs(self):
        bsr = trnsparse.BSRMatrix.from_dense(_block_spd(1), block_size=128)
        x, iters, rel = trnsparse.cg_bsr_fused(bsr, torch.zeros(128, dtype=torch.float64))
        assert iters == 0 and rel == 0.0 and torch.count_nonzero(x) == 0

    def test_operator_reuse_and_fp32(self):
        A = _block_spd(2, seed=10)
        bsr = trnsparse.BSRMatrix.from_dense(A.float(), block_size=128)
        op = trnsparse.prepare_cg_bsr(bsr, preconditioner="block_jacobi")
        assert isinstance(op, trnsparse.FusedCGOperator)
        assert op.resident_blocks == op.n_blocks == bsr.n_blocks
        for seed in (11, 12):
            b = torch.randn(256, generator=torch.Generator().manual_seed(seed))
            x, _, rel = trnsparse.cg_bsr_fused(op, b, tol=1e-5)
            assert x.dtype == torch.float32
            assert rel < 1e-5
            res = torch.linalg.norm(b.double() - A @ x.double()) / torch.linalg.norm(b.double())
            assert res < 1e-4

    def test_resident_block_budget(self, monkeypatch):
        from trnsparse.nki.kernels import cg_resident_blocks

        # 160 KiB / partition, 512 B per fp32 block per partition
        assert cg_resident_blocks(1000, 32, False) == 320
        assert cg_resident_blocks(1000, 32, True) == 288
        assert cg_resident_blocks(100, 32, True) == 100
        monkeypatch.setenv("TRNSPARSE_CG_SBUF_KIB", "8")
        assert cg_resident_blocks(100, 4, False) == 16

    def test_rejects_bad_inputs(self):
        bsr = trnsparse.BSRMatrix.from_dense(torch.eye(64), block_size=32)
        with pytest.raises(ValueError, match="block_size=128"):
            trnsparse.prepare_cg_bsr(bsr)
        rect = trnsparse.BSRMatrix.from_dense(torch.ones(128, 256), block_size=128)
        with pytest.raises(ValueError, match="square"):
            trnsparse.prepare_cg_bsr(rect)
        off = torch.zeros(256, 256)
        off[:128, 128:] = 1.0
        off[128:, :128] = 1.0
        offb = trnsparse.BSRMatrix.from_dense(off, block_size=128)
        with pytest.raises(ValueError, match="diagonal block"):
            trnsparse.prepare_cg_bsr(offb, preconditioner="block_jacobi")
        sq = trnsparse.BSRMatrix.from_dense(torch.eye(128), block_size=128)
        with pytest.raises(ValueError, match="preconditioner"):
            trnsparse.prepare_cg_bsr(sq, preconditioner="ilu")
        with pytest.raises(ValueError, match="shape"):
            trnsparse.cg_bsr_fused(sq, torch.ones(64))

    def test_multiple_rhs_match_single_solves(self):
        A = _block_spd(3, seed=13)
        g = torch.Generator().manual_seed(14)
        B = torch.randn(A.shape[0], 4, dtype=torch.float64, generator=g)
        B[:, 1] *= 1e3  # columns are scaled independently
        B[:, 2] = 0.0  # a zero column converges at once
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        X, iters, rel = trnsparse.cg_bsr_fused(
            bsr, B, tol=1e-9, preconditioner="block_jacobi", max_iter=500
        )
        assert X.shape == B.shape and iters.shape == (4,) and rel.shape == (4,)
        assert int(iters[2]) == 0 and torch.count_nonzero(X[:, 2]) == 0
        for c in (0, 1, 3):
            x, k, r = trnsparse.cg_bsr_fused(
                bsr, B[:, c], tol=1e-9, preconditioner="block_jacobi", max_iter=500
            )
            assert int(iters[c]) == k
            torch.testing.assert_close(X[:, c], x, atol=1e-9 * float(B[:, c].norm()), rtol=1e-9)
        torch.testing.assert_close(X, torch.linalg.solve(A, B), atol=1e-6, rtol=1e-6)

    def test_rhs_count_limits(self):
        sq = trnsparse.BSRMatrix.from_dense(torch.eye(128), block_size=128)
        with pytest.raises(ValueError, match="right-hand sides"):
            trnsparse.cg_bsr_fused(sq, torch.ones(128, 513))
        with pytest.raises(ValueError, match="shape"):
            trnsparse.cg_bsr_fused(sq, torch.ones(128, 2, 2))

    def test_matvec_and_precision_options(self):
        A = _block_spd(2, seed=15)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        b = torch.randn(256, dtype=torch.float64, generator=torch.Generator().manual_seed(16))
        _, _, _, info = trnsparse.cg_bsr_fused(bsr, b, return_info=True)
        assert info["matvec"] == "column" and info["precision"] == "fp32"
        _, _, _, info = trnsparse.cg_bsr_fused(bsr, b[:, None].repeat(1, 33), return_info=True)
        assert info["matvec"] == "row"  # "auto": more than 32 right-hand sides
        with pytest.raises(ValueError, match="matvec"):
            trnsparse.cg_bsr_fused(bsr, b, matvec="diagonal")
        with pytest.raises(ValueError, match="precision"):
            trnsparse.cg_bsr_fused(bsr, b, precision="fp16")
        with pytest.raises(ValueError, match="column"):
            trnsparse.cg_bsr_fused(bsr, b, matvec="row", precision="bf16")
        with pytest.raises(ValueError, match="right-hand sides"):
            trnsparse.cg_bsr_fused(bsr, b[:, None].repeat(1, 17), precision="bf16x2")

    def test_bf16_emulation_solves_the_rounded_system(self):
        # The host path emulates the bf16 column matvec: CG converges for the
        # bf16-rounded A, so the residual against A itself stalls near the
        # rounding (~2^-9), and bf16 hi + lo (~2^-17) lands close to fp32.
        from trnsparse.iterative import _bf16_parts

        A = _block_spd(2, seed=17, shift=5.0)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        b = torch.randn(256, dtype=torch.float64, generator=torch.Generator().manual_seed(18))
        op = trnsparse.prepare_cg_bsr(bsr)

        def true_res(x):
            return float(torch.linalg.norm(b - A @ x) / torch.linalg.norm(b))

        x32, _, _ = trnsparse.cg_bsr_fused(op, b, tol=1e-10, max_iter=500)
        x16, _, rel16 = trnsparse.cg_bsr_fused(op, b, tol=1e-10, max_iter=500, precision="bf16")
        x2, _, _ = trnsparse.cg_bsr_fused(op, b, tol=1e-10, max_iter=500, precision="bf16x2")
        assert rel16 < 1e-6  # the recursive residual converges ...
        assert 1e-5 < true_res(x16) < 3e-2  # ... to the bf16-rounded system
        assert true_res(x2) < 1e-4 and true_res(x32) < 1e-9
        hi, lo = _bf16_parts(op._blocks64, 2)
        assert hi.dtype == lo.dtype == torch.bfloat16
        err = (hi.double() + lo.double() - op._blocks64).abs().max()
        assert err <= 2.0**-15 * op._blocks64.abs().max()

    def test_resident_blocks_bf16(self):
        from trnsparse.nki.kernels import cg_resident_blocks

        # bf16 A takes 256 B per block per partition: twice the blocks
        assert cg_resident_blocks(1000, 32, False, block_bytes=256) == 640
        assert cg_resident_blocks(1000, 32, True, block_bytes=256) == 576
        assert cg_resident_blocks(1000, 32, False, block_bytes=512) == 320


class TestPowerIterationFused:
    def test_matches_power_iteration_bsr(self):
        A = _block_spd(3, seed=20)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        v0 = torch.randn(
            A.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(21)
        )
        lam, v, k = trnsparse.power_iteration_bsr_fused(bsr, v0=v0, tol=1e-10, max_iter=3000)
        lam_ref, _, k_ref = trnsparse.power_iteration_bsr(bsr, v0=v0, tol=1e-10, max_iter=3000)
        assert abs(lam - lam_ref) < 1e-8 * abs(lam_ref) and abs(k - k_ref) <= 1
        top = float(torch.linalg.eigvalsh(A)[-1])
        assert abs(lam - top) < 1e-4 * top
        assert v.dtype == torch.float64 and abs(float(torch.linalg.norm(v)) - 1) < 1e-12

    def test_cap_padding_and_info(self):
        A = _block_spd(2, n=200, seed=22)
        bsr = trnsparse.BSRMatrix.from_dense(A, block_size=128)
        v0 = torch.ones(200, dtype=torch.float64)
        lam, v, k, info = trnsparse.power_iteration_bsr_fused(
            bsr, v0=v0, tol=0.0, max_iter=7, check_every=4, return_info=True
        )
        assert k == 7 and v.shape == (200,) and info["path"] == "pytorch"
        assert info["matvec"] == "column" and info["trips"] == 2
        with pytest.raises(ValueError, match="v0"):
            trnsparse.power_iteration_bsr_fused(bsr, v0=torch.ones(10))
        with pytest.raises(ValueError, match="column"):
            trnsparse.power_iteration_bsr_fused(bsr, matvec="row", precision="bf16")
