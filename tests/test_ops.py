"""Test sparse matrix operations."""

import numpy as np
import pytest
import torch

import trnsparse


class TestSpMV:
    def test_identity(self):
        I = trnsparse.eye_sparse(4)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = trnsparse.spmv(I, x)
        np.testing.assert_allclose(y.numpy(), x.numpy())

    def test_vs_dense(self):
        A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
        x = torch.tensor([1.0, 2.0, 3.0])
        csr = trnsparse.from_dense(A)
        y_sparse = trnsparse.spmv(csr, x)
        y_dense = A @ x
        np.testing.assert_allclose(y_sparse.numpy(), y_dense.numpy(), atol=1e-6)

    def test_alpha_beta(self):
        A = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        csr = trnsparse.from_dense(A)
        x = torch.tensor([1.0, 1.0])
        y = torch.tensor([10.0, 20.0])
        result = trnsparse.spmv(csr, x, alpha=2.0, y=y, beta=0.5)
        # 2.0 * [2, 3] + 0.5 * [10, 20] = [4, 6] + [5, 10] = [9, 16]
        np.testing.assert_allclose(result.numpy(), [9.0, 16.0])

    def test_large_sparse(self):
        torch.manual_seed(42)
        n = 100
        A = torch.randn(n, n)
        A[torch.abs(A) < 1.5] = 0.0  # ~87% sparse
        x = torch.randn(n)
        csr = trnsparse.from_dense(A)
        y_sparse = trnsparse.spmv(csr, x)
        y_dense = A @ x
        np.testing.assert_allclose(y_sparse.numpy(), y_dense.numpy(), atol=1e-5)


class TestSpMM:
    def test_identity(self):
        I = trnsparse.eye_sparse(3)
        B = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = trnsparse.spmm(I, B)
        np.testing.assert_allclose(result.numpy(), B.numpy())

    def test_vs_dense(self):
        A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
        csr = trnsparse.from_dense(A)
        result = trnsparse.spmm(csr, B)
        expected = A @ B
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)

    def test_alpha_beta(self):
        A = torch.eye(2)
        csr = trnsparse.from_dense(A)
        B = torch.ones(2, 3)
        C = torch.ones(2, 3) * 10
        result = trnsparse.spmm(csr, B, alpha=2.0, C=C, beta=0.5)
        np.testing.assert_allclose(result.numpy(), np.full((2, 3), 7.0))


class TestSymmetricSpMV:
    def test_symmetric(self):
        A = torch.tensor([[4.0, 1.0, 0.0], [1.0, 3.0, 2.0], [0.0, 2.0, 5.0]])
        # Store only upper triangle
        upper = torch.triu(A)
        csr = trnsparse.from_dense(upper)
        x = torch.tensor([1.0, 2.0, 3.0])
        result = trnsparse.spmv_symmetric(csr, x, uplo="upper")
        expected = A @ x
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


class TestSparseAdd:
    def test_basic(self):
        A = trnsparse.from_dense(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
        B = trnsparse.from_dense(torch.tensor([[0.0, 3.0], [4.0, 0.0]]))
        C = trnsparse.sparse_add(A, B)
        expected = torch.tensor([[1.0, 3.0], [4.0, 2.0]])
        np.testing.assert_allclose(C.to_dense().numpy(), expected.numpy())


class TestSparseScale:
    def test_basic(self):
        A = trnsparse.from_dense(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
        B = trnsparse.sparse_scale(A, 3.0)
        expected = torch.tensor([[3.0, 0.0], [0.0, 6.0]])
        np.testing.assert_allclose(B.to_dense().numpy(), expected.numpy())


class TestNnzPerRow:
    def test_basic(self):
        A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
        csr = trnsparse.from_dense(A)
        nnz = trnsparse.nnz_per_row(csr)
        np.testing.assert_array_equal(nnz.numpy(), [2, 0, 3])
