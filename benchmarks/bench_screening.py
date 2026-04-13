"""Schwarz screening benchmarks.

These ops don't depend on the NKI backend (pure elementwise) — included
to complete the picture alongside SpMV/SpMM.
"""

import pytest
import torch

import trnsparse


@pytest.fixture(params=[512, 2048, 8192])
def n_shells(request):
    return request.param


@pytest.fixture
def diagonal_integrals(n_shells):
    torch.manual_seed(3)
    return torch.abs(torch.randn(n_shells, n_shells))


def test_schwarz_bounds(benchmark, diagonal_integrals):
    benchmark(lambda: trnsparse.schwarz_bounds(diagonal_integrals))


def test_screen_quartets(benchmark, diagonal_integrals):
    Q = trnsparse.schwarz_bounds(diagonal_integrals)
    benchmark(lambda: trnsparse.screen_quartets(Q, threshold=1e-4))


def test_sparsity_stats(benchmark, diagonal_integrals):
    Q = trnsparse.schwarz_bounds(diagonal_integrals)
    benchmark(lambda: trnsparse.sparsity_stats(Q, threshold=1e-4))
