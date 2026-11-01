"""Test sparse matrix formats and conversions."""

import numpy as np
import pytest
import torch

import trnsparse
from trnsparse import COOMatrix, CSRMatrix


class TestCSR:
    def test_from_dense(self):
        A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
        csr = trnsparse.from_dense(A)
        assert csr.nnz == 5
        assert csr.shape == (3, 3)

    def test_to_dense_roundtrip(self):
        A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        csr = trnsparse.from_dense(A)
        recovered = csr.to_dense()
        np.testing.assert_allclose(recovered.numpy(), A.numpy())

    def test_density(self):
        A = torch.zeros(10, 10)
        A[0, 0] = 1.0
        A[5, 5] = 2.0
        csr = trnsparse.from_dense(A)
        assert csr.density == pytest.approx(0.02)

    def test_threshold(self):
        A = torch.tensor([[1.0, 1e-12, 0.5], [0.0, 2.0, 1e-15]])
        csr = trnsparse.from_dense(A, threshold=1e-10)
        assert csr.nnz == 3  # Only 1.0, 0.5, 2.0

    def test_identity(self):
        I = trnsparse.eye_sparse(4)
        assert I.nnz == 4
        np.testing.assert_allclose(I.to_dense().numpy(), np.eye(4))


class TestCOO:
    def test_coo_to_csr_roundtrip(self):
        A = torch.tensor([[0.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
        csr = trnsparse.from_dense(A)
        coo = csr.to_coo()
        csr2 = coo.to_csr()
        np.testing.assert_allclose(csr2.to_dense().numpy(), A.numpy())

    def test_coo_to_dense(self):
        coo = COOMatrix(
            values=torch.tensor([1.0, 2.0, 3.0]),
            row_indices=torch.tensor([0, 1, 2]),
            col_indices=torch.tensor([2, 0, 1]),
            shape=(3, 3),
        )
        expected = torch.tensor([[0, 0, 1], [2, 0, 0], [0, 3, 0]], dtype=torch.float32)
        np.testing.assert_allclose(coo.to_dense().numpy(), expected.numpy())


class TestTranspose:
    def test_transpose(self):
        A = torch.tensor([[1.0, 2.0], [0.0, 3.0], [4.0, 0.0]])
        csr = trnsparse.from_dense(A)
        csr_t = trnsparse.sparse_transpose(csr)
        np.testing.assert_allclose(csr_t.to_dense().numpy(), A.T.numpy())


class TestFromScipy:
    def test_complex_keeps_imaginary_part(self):
        # Through 0.12.1 complex input was cast to float32, dropping the
        # imaginary part (#58).
        sp = pytest.importorskip("scipy.sparse")
        M = sp.random(15, 11, density=0.4, random_state=0, format="csr")
        M = (M + 1j * sp.random(15, 11, density=0.4, random_state=1, format="csr")).tocsr()
        A = trnsparse.from_scipy(M)
        assert A.dtype == torch.complex64
        np.testing.assert_allclose(A.to_dense().numpy(), M.toarray(), rtol=1e-6, atol=1e-6)
        x = np.random.default_rng(2).standard_normal(11).astype(np.complex64)
        np.testing.assert_allclose(
            trnsparse.spmv(A, torch.from_numpy(x)).numpy(), M @ x, rtol=1e-5, atol=1e-5
        )

    def test_real_stays_float32(self):
        sp = pytest.importorskip("scipy.sparse")
        M = sp.random(6, 6, density=0.5, random_state=3, format="csr")
        A = trnsparse.from_scipy(M)
        assert A.dtype == torch.float32
        np.testing.assert_allclose(A.to_dense().numpy(), M.toarray(), rtol=1e-6)
