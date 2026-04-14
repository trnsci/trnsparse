"""Iterative-solver benchmarks: trnsparse.cg_bsr vs scipy baseline.

The v0.3.2 plumbing dispatches one `bsr_spmm` call per CG iteration.
On NKI that means one kernel launch + HBM round-trip per iteration.
Expect the current path to be dominated by dispatch overhead compared
to scipy's compiled C loop, which motivates the v0.4.0 fused-kernel
follow-up.
"""

from __future__ import annotations

import pytest
import torch

import trnsparse


@pytest.fixture(params=[128, 256])
def iter_size(request):
    return request.param


@pytest.fixture
def spd_bsr_and_dense(iter_size):
    torch.manual_seed(0)
    n = iter_size
    M = torch.randn(n, n)
    A_dense = M @ M.T + n * torch.eye(n)
    A_bsr = trnsparse.BSRMatrix.from_dense(A_dense, block_size=128)
    b = torch.randn(n)
    return A_dense, A_bsr, b


def test_cg_bsr_trnsparse(benchmark, spd_bsr_and_dense):
    _, A_bsr, b = spd_bsr_and_dense
    benchmark(lambda: trnsparse.cg_bsr(A_bsr, b, tol=1e-8, max_iter=2 * b.shape[0]))


def test_cg_scipy(benchmark, spd_bsr_and_dense):
    sp = pytest.importorskip("scipy.sparse")
    spla = pytest.importorskip("scipy.sparse.linalg")
    A_dense, _, b = spd_bsr_and_dense
    A_scipy = sp.csr_matrix(A_dense.numpy())
    b_np = b.numpy()
    benchmark(lambda: spla.cg(A_scipy, b_np, rtol=1e-8, maxiter=2 * len(b_np)))


def test_power_iteration_trnsparse(benchmark, spd_bsr_and_dense):
    _, A_bsr, _ = spd_bsr_and_dense
    benchmark(lambda: trnsparse.power_iteration_bsr(A_bsr, max_iter=500, tol=1e-9))
