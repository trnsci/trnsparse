"""BSR SpMM benchmarks: Trainium-native block-sparse path.

Covers the architectural claim for v0.3.0 — that block-structured sparse
matmul on the Tensor Engine matches dense-GEMM throughput at moderate
block densities, which is where CSR-path NKI loses.

Run:

    pytest benchmarks/bench_bsr_spmm.py --benchmark-only

Fixtures parametrize `block_density` (fraction of 128×128 blocks stored),
not element-level density. A 10% block-dense matrix is effectively
~10% sparse at the element level (because each block is fully dense).
"""

from __future__ import annotations

import pytest
import torch

import trnsparse
from trnsparse.nki.dispatch import HAS_NKI

M_BLOCKS = [4, 8]  # matrix has M_BLOCKS * 128 rows
N_BLOCKS = [4, 8]
BLOCK_DENSITIES = [0.1, 0.25, 0.5]
RHS_COLS = [128, 256]


@pytest.fixture(params=M_BLOCKS)
def m_blocks(request):
    return request.param


@pytest.fixture(params=N_BLOCKS)
def n_blocks(request):
    return request.param


@pytest.fixture(params=BLOCK_DENSITIES)
def block_density(request):
    return request.param


@pytest.fixture(params=RHS_COLS)
def bsr_rhs_cols(request):
    return request.param


@pytest.fixture
def bsr_and_B(m_blocks, n_blocks, block_density, bsr_rhs_cols):
    torch.manual_seed(0)
    b = 128
    M, N = m_blocks * b, n_blocks * b
    A = torch.zeros(M, N)
    mask = torch.rand(m_blocks, n_blocks) < block_density
    for i in range(m_blocks):
        for j in range(n_blocks):
            if mask[i, j]:
                A[i * b : (i + 1) * b, j * b : (j + 1) * b] = torch.randn(b, b)
    bsr = trnsparse.BSRMatrix.from_dense(A, block_size=b)
    B = torch.randn(N, bsr_rhs_cols)
    return bsr, B, A


@pytest.fixture
def force_pytorch_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("pytorch")
    yield
    trnsparse.set_backend(prev)


@pytest.fixture
def force_nki_backend():
    if not HAS_NKI:
        pytest.skip("NKI not available")
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


def test_bsr_spmm_pytorch(benchmark, force_pytorch_backend, bsr_and_B):
    bsr, B, _ = bsr_and_B
    benchmark(lambda: trnsparse.bsr_spmm(bsr, B))


def test_bsr_spmm_nki(benchmark, force_nki_backend, bsr_and_B):
    bsr, B, _ = bsr_and_B
    benchmark(lambda: trnsparse.bsr_spmm(bsr, B))


def test_dense_matmul(benchmark, bsr_and_B):
    """Dense-GEMM ceiling: what `torch.matmul(A_dense, B)` takes. BSR at
    high block density should approach this. Never go below this — if it
    does, BSR isn't pulling its weight.
    """
    _, B, A = bsr_and_B
    benchmark(lambda: A @ B)
