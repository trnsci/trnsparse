"""Benchmark fixtures for trnsparse.

Sizes and densities are parametrized on the fixtures below. Each fixture
returns matrices in multiple formats so different backends can be timed
against the same numeric data.

Set `TRNSPARSE_BACKEND=nki` (e.g. inside `scripts/run_benchmarks.sh`) to
run `trnsparse` ops via the NKI dispatch. Default is `auto` / PyTorch.
"""

import os

import pytest
import torch

import trnsparse


def pytest_configure(config):
    backend = os.environ.get("TRNSPARSE_BACKEND", "auto")
    if backend != "auto":
        trnsparse.set_backend(backend)


SIZES = [256, 1024, 4096]
DENSITIES = [0.001, 0.01, 0.1]


@pytest.fixture(params=SIZES)
def size(request):
    return request.param


@pytest.fixture(params=DENSITIES)
def density(request):
    return request.param


@pytest.fixture
def sparse_dense(size, density):
    """A dense (size, size) tensor with `density` fraction of nonzeros."""
    torch.manual_seed(0)
    mask = torch.rand(size, size) < density
    return torch.randn(size, size) * mask


@pytest.fixture
def csr(sparse_dense):
    return trnsparse.from_dense(sparse_dense)


@pytest.fixture
def scipy_csr(sparse_dense):
    try:
        import scipy.sparse as sp
    except ImportError:
        pytest.skip("scipy not installed")
    return sp.csr_matrix(sparse_dense.numpy())


@pytest.fixture
def torch_sparse_csr(sparse_dense):
    return sparse_dense.to_sparse_csr()


@pytest.fixture
def dense_rhs_vec(size):
    torch.manual_seed(1)
    return torch.randn(size)


@pytest.fixture(params=[32, 128])
def rhs_cols(request):
    return request.param


@pytest.fixture
def dense_rhs_mat(size, rhs_cols):
    torch.manual_seed(2)
    return torch.randn(size, rhs_cols)
