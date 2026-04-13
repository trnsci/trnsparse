"""SpMV benchmarks: trnsparse vs scipy.sparse vs torch.sparse.

Run:

    pytest benchmarks/bench_spmv.py --benchmark-only

Baseline is the PyTorch fallback in trnsparse. scipy and torch.sparse
give the ecosystem reference points.
"""

import trnsparse


def test_spmv_trnsparse(benchmark, csr, dense_rhs_vec):
    benchmark(lambda: trnsparse.spmv(csr, dense_rhs_vec))


def test_spmv_scipy(benchmark, scipy_csr, dense_rhs_vec):
    x = dense_rhs_vec.numpy()
    benchmark(lambda: scipy_csr @ x)


def test_spmv_torch_sparse(benchmark, torch_sparse_csr, dense_rhs_vec):
    benchmark(lambda: torch_sparse_csr @ dense_rhs_vec)
