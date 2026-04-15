"""Test integral screening utilities."""

import numpy as np
import pytest
import torch

import trnsparse


class TestSchwarz:
    def test_bounds_positive(self):
        diag = torch.rand(10, 10) * 0.1
        Q = trnsparse.schwarz_bounds(diag)
        assert torch.all(Q >= 0)

    def test_screening(self):
        Q = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        mask = trnsparse.screen_quartets(Q, threshold=0.3)
        # Q > sqrt(0.3) ≈ 0.548 → only the 1.0 entries pass
        assert mask[0, 0].item() is True
        assert mask[0, 1].item() is False

    def test_high_threshold_screens_everything(self):
        Q = torch.ones(5, 5) * 0.01
        mask = trnsparse.screen_quartets(Q, threshold=1.0)
        assert mask.sum().item() == 0

    def test_low_threshold_keeps_everything(self):
        Q = torch.ones(5, 5)
        mask = trnsparse.screen_quartets(Q, threshold=1e-20)
        assert mask.sum().item() == 25


class TestDensityScreen:
    """Coverage for the density-weighted pair bound (#10)."""

    def test_low_threshold_keeps_all(self):
        Q = torch.rand(8, 8)
        P = torch.rand(8, 8)
        mask = trnsparse.density_screen(Q, P, threshold=0.0)
        assert mask.all()

    def test_high_threshold_drops_all(self):
        Q = torch.rand(8, 8)
        P = torch.rand(8, 8)
        mask = trnsparse.density_screen(Q, P, threshold=1e30)
        assert mask.sum().item() == 0

    def test_no_false_negatives_vs_quartet_reference(self):
        """Every pair dropped by `density_screen` must have a true upper bound
        below the threshold when checked via the full quartet expression
        `|P_λσ| * Q_μν * Q_λσ`. Catches regressions where the 2D bound becomes
        too aggressive.
        """
        torch.manual_seed(0)
        n = 6
        Q = torch.abs(torch.randn(n, n))
        P = torch.abs(torch.randn(n, n))
        threshold = 0.5

        mask = trnsparse.density_screen(Q, P, threshold=threshold)

        # True per-pair upper bound: max_λσ(|P_λσ| * Q_μν * Q_λσ)
        PQ = torch.abs(P) * Q
        max_PQ = PQ.max().item()
        true_upper = Q * max_PQ  # (n, n)

        # Every pair the mask drops must have true upper bound <= threshold
        dropped = ~mask
        assert (true_upper[dropped] <= threshold).all(), (
            "density_screen dropped a pair whose true upper bound exceeds "
            "the threshold — false negative (would miss a significant quartet)"
        )


class TestScreenedSpmm:
    """PyTorch-fallback path for the fused screened SpMM (#19)."""

    def test_threshold_zero_equals_plain_matmul(self):
        """threshold=0 keeps all entries → screened_spmm == A @ B."""
        torch.manual_seed(0)
        n = 64
        A = torch.randn(n, n)
        diag = torch.abs(torch.randn(n)) + 0.1
        B = torch.randn(n, 16)

        got = trnsparse.screened_spmm(A, diag, B, threshold=0.0)
        torch.testing.assert_close(got, A @ B, atol=1e-5, rtol=1e-5)

    def test_huge_threshold_zeros_output(self):
        """threshold → ∞ drops all entries → screened_spmm returns zeros."""
        torch.manual_seed(1)
        n = 64
        A = torch.randn(n, n)
        diag = torch.abs(torch.randn(n))
        B = torch.randn(n, 16)

        got = trnsparse.screened_spmm(A, diag, B, threshold=1e30)
        torch.testing.assert_close(got, torch.zeros(n, 16), atol=0, rtol=0)

    def test_parity_vs_explicit_mask(self):
        """Matches (A * mask) @ B for a non-trivial threshold."""
        import math

        torch.manual_seed(2)
        n = 64
        A = torch.randn(n, n)
        diag = torch.abs(torch.randn(n)) * 4.0
        B = torch.randn(n, 16)
        threshold = 0.5

        got = trnsparse.screened_spmm(A, diag, B, threshold=threshold)

        Q = torch.sqrt(torch.abs(diag))
        mask = (Q.unsqueeze(-1) * Q.unsqueeze(0)) > math.sqrt(threshold)
        expected = (A * mask.to(A.dtype)) @ B

        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_non_trivial_mask_setup(self):
        """Guard: the chosen threshold must drop some but not all entries
        on the test distribution — otherwise other tests are vacuous.
        """
        import math

        torch.manual_seed(3)
        n = 64
        diag = torch.abs(torch.randn(n))
        threshold = 0.5
        Q = torch.sqrt(torch.abs(diag))
        mask = (Q.unsqueeze(-1) * Q.unsqueeze(0)) > math.sqrt(threshold)
        assert not mask.all()
        assert mask.any()


class TestSparsityStats:
    def test_fully_dense(self):
        Q = torch.ones(10, 10)
        stats = trnsparse.sparsity_stats(Q, threshold=1e-20)
        assert stats["pair_sparsity"] == pytest.approx(0.0)

    def test_keys(self):
        Q = torch.rand(5, 5)
        stats = trnsparse.sparsity_stats(Q)
        assert "n_shells" in stats
        assert "significant_pairs" in stats
        assert "pair_sparsity" in stats
