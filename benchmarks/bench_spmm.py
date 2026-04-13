"""SpMM benchmarks: trnsparse vs scipy.sparse vs torch.sparse.

Run:

    pytest benchmarks/bench_spmm.py --benchmark-only

SpMM is the primary target for the NKI path — these PyTorch numbers are
the baseline that NKI on trn1/trn2 has to beat.
"""

import trnsparse


def test_spmm_trnsparse(benchmark, csr, dense_rhs_mat):
    benchmark(lambda: trnsparse.spmm(csr, dense_rhs_mat))


def test_spmm_scipy(benchmark, scipy_csr, dense_rhs_mat):
    B = dense_rhs_mat.numpy()
    benchmark(lambda: scipy_csr @ B)


def test_spmm_torch_sparse(benchmark, torch_sparse_csr, dense_rhs_mat):
    benchmark(lambda: torch_sparse_csr @ dense_rhs_mat)
