"""Test integral screening utilities."""

import pytest
import torch
import numpy as np
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
