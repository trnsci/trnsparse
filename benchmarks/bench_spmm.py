"""SpMM benchmarks: trnsparse vs scipy.sparse vs torch.sparse.

Run:

    pytest benchmarks/bench_spmm.py --benchmark-only

Four comparisons in a single pass:
  - `trnsparse_nki` — NKI dispatch (skipped if HAS_NKI is False)
  - `trnsparse_pytorch` — v0.1.3 vectorized torch.sparse fallback
  - `scipy` — scipy.sparse CSR matmul
  - `torch_sparse` — torch.sparse_csr matmul

On Trainium + neuronxcc installed, all four run in one pytest invocation.
"""

import pytest
import trnsparse
from trnsparse.nki.dispatch import HAS_NKI


@pytest.fixture
def force_pytorch_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("pytorch")
    yield
    trnsparse.set_backend(prev)


@pytest.fixture
def force_nki_backend():
    if not HAS_NKI:
        pytest.skip("NKI not available in this environment")
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


def test_spmm_trnsparse_pytorch(benchmark, force_pytorch_backend, csr, dense_rhs_mat):
    benchmark(lambda: trnsparse.spmm(csr, dense_rhs_mat))


def test_spmm_trnsparse_nki(benchmark, force_nki_backend, csr, dense_rhs_mat):
    benchmark(lambda: trnsparse.spmm(csr, dense_rhs_mat))


def test_spmm_scipy(benchmark, scipy_csr, dense_rhs_mat):
    B = dense_rhs_mat.numpy()
    benchmark(lambda: scipy_csr @ B)


def test_spmm_torch_sparse(benchmark, torch_sparse_csr, dense_rhs_mat):
    benchmark(lambda: torch_sparse_csr @ dense_rhs_mat)
