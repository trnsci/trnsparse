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

    @staticmethod
    def _case(store, seed=3):
        g = torch.Generator().manual_seed(seed)
        S = torch.randn(9, 9, generator=g, dtype=torch.float64)
        S[torch.rand(9, 9, generator=g) > 0.5] = 0
        S.fill_diagonal_(1.5)
        stored = {"upper": torch.triu(S), "lower": torch.tril(S), "both": S}[store]
        x = torch.randn(9, generator=g, dtype=torch.float64)
        return S, stored, x

    @staticmethod
    def _sym(T):
        return T + T.T - torch.diag(torch.diagonal(T))

    @pytest.mark.parametrize("dense_branch", [False, True])
    @pytest.mark.parametrize("uplo", ["upper", "lower"])
    @pytest.mark.parametrize("store", ["upper", "lower", "both"])
    def test_uplo_and_storage(self, monkeypatch, store, uplo, dense_branch):
        """Both triangles stored: only `uplo` is referenced (#52). One triangle
        stored: that triangle is used whatever `uplo` says, as through 0.11.0,
        with a DeprecationWarning when `uplo` names the empty triangle."""
        import warnings

        from trnsparse import ops

        if dense_branch:  # the device-operand branch, on the host
            monkeypatch.setattr(ops, "_torch_sparse_ok", lambda A, x: False)
        S, stored, x = self._case(store)
        tri = uplo if store == "both" else store
        want = 0.7 * self._sym(torch.triu(S) if tri == "upper" else torch.tril(S)) @ x
        csr = trnsparse.from_dense(stored)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            got = trnsparse.spmv_symmetric(csr, x, alpha=0.7, uplo=uplo)
        torch.testing.assert_close(got, want)
        dep = [m for m in w if issubclass(m.category, DeprecationWarning)]
        assert bool(dep) == (store != "both" and store != uplo)

    def test_diagonal_only_no_warning(self):
        import warnings

        csr = trnsparse.from_dense(torch.diag(torch.arange(1.0, 6.0)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = trnsparse.spmv_symmetric(csr, torch.ones(5), uplo="lower")
        torch.testing.assert_close(got, torch.arange(1.0, 6.0))

    def test_bad_uplo(self):
        csr = trnsparse.from_dense(torch.eye(3))
        with pytest.raises(ValueError):
            trnsparse.spmv_symmetric(csr, torch.ones(3), uplo="U")


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


class TestEpilogueShape:
    """y / C must broadcast *to* the product's shape (torch.addmv / addmm rule).

    Through 0.12.1 a y of shape (m, 1) turned spmv's (m,) result into (m, m)
    without an error (#67).
    """

    def setup_method(self):
        g = torch.Generator().manual_seed(67)
        self.D = torch.randn(12, 10, generator=g)
        self.A = trnsparse.from_dense(self.D)
        self.x = torch.randn(10, generator=g)
        self.B = torch.randn(10, 4, generator=g)

    def test_spmv_rejects_enlarging_y(self):
        with pytest.raises(ValueError, match="does not broadcast"):
            trnsparse.spmv(self.A, self.x, y=torch.ones(12, 1), beta=1.0)

    def test_spmm_rejects_enlarging_c(self):
        with pytest.raises(ValueError, match="does not broadcast"):
            trnsparse.spmm(self.A, self.B, C=torch.ones(1, 12, 4), beta=1.0)
        with pytest.raises(ValueError, match="does not broadcast"):
            trnsparse.spmm(self.A, self.x, C=torch.ones(12, 1), beta=1.0)

    @pytest.mark.parametrize("yshape", [(12,), (1,), ()])
    def test_spmv_broadcasting_y_still_works(self, yshape):
        y = torch.full(yshape, 2.0)
        out = trnsparse.spmv(self.A, self.x, y=y, beta=0.5)
        torch.testing.assert_close(out, torch.addmv(y.expand(12), self.D, self.x, beta=0.5))

    @pytest.mark.parametrize("cshape", [(12, 4), (12, 1), (4,), ()])
    def test_spmm_broadcasting_c_still_works(self, cshape):
        C = torch.full(cshape, 2.0)
        out = trnsparse.spmm(self.A, self.B, C=C, beta=0.5)
        torch.testing.assert_close(out, torch.addmm(C, self.D, self.B, beta=0.5))


class TestIntegerDtypeOnDensePath:
    """The device dense path (no torch sparse CSR) keeps an integer product's
    dtype under the default alpha=1.0 / beta=1.0 (#79). Forced on CPU here."""

    @pytest.fixture
    def dense_path(self, monkeypatch):
        import trnsparse.ops as ops

        monkeypatch.setattr(ops, "_torch_sparse_ok", lambda *a: False)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_spmv_spmm_keep_int_dtype(self, dense_path, dtype):
        g = torch.Generator().manual_seed(0)
        Ad = torch.randint(-5, 6, (37, 29), generator=g).to(dtype)
        Ad[torch.rand(37, 29, generator=g) < 0.7] = 0
        A = trnsparse.from_dense(Ad)
        x = torch.randint(-5, 6, (29,), generator=g).to(dtype)
        B = torch.randint(-5, 6, (29, 6), generator=g).to(dtype)
        y = torch.randint(-5, 6, (37,), generator=g).to(dtype)
        C = torch.randint(-5, 6, (37, 6), generator=g).to(dtype)
        for got, want in [
            (trnsparse.spmv(A, x), Ad @ x),
            (trnsparse.spmv(A, x, y=y, beta=1.0), Ad @ x + y),
            (trnsparse.spmm(A, B), Ad @ B),
            (trnsparse.spmm(A, B, C=C, beta=1.0), Ad @ B + C),
        ]:
            assert got.dtype == dtype
            assert torch.equal(got, want)

    def test_float_alpha_still_scales(self, dense_path):
        Ad = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        A = trnsparse.from_dense(Ad)
        x = torch.tensor([3.0, 4.0])
        y = torch.tensor([1.0, 1.0])
        torch.testing.assert_close(
            trnsparse.spmv(A, x, alpha=0.5, y=y, beta=-2.0), 0.5 * (Ad @ x) - 2.0 * y
        )
