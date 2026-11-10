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

    def test_coo_to_dense_sums_duplicates(self):
        # Duplicate (row, col) entries are summed, like to_csr() and scipy.sparse.
        # Through 0.12.1 to_dense kept one of them (the other was dropped).
        sp = pytest.importorskip("scipy.sparse")
        rows, cols, vals = [0, 1, 0, 2], [1, 2, 1, 0], [1.0, 2.0, 10.0, 3.0]
        coo = COOMatrix(
            values=torch.tensor(vals, dtype=torch.float64),
            row_indices=torch.tensor(rows),
            col_indices=torch.tensor(cols),
            shape=(3, 3),
        )
        expected = sp.coo_matrix((vals, (rows, cols)), shape=(3, 3)).toarray()
        np.testing.assert_allclose(coo.to_dense().numpy(), expected)
        np.testing.assert_allclose(coo.to_csr().to_dense().numpy(), expected)

    def test_csr_with_duplicates_to_coo_to_dense(self):
        csr = CSRMatrix(
            values=torch.tensor([5.0, 7.0, 1.0]),
            col_indices=torch.tensor([1, 1, 0]),
            row_ptrs=torch.tensor([0, 2, 3]),
            shape=(2, 2),
        )
        np.testing.assert_allclose(csr.to_coo().to_dense().numpy(), csr.to_dense().numpy())


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
        A = trnsparse.from_scipy(M.astype(np.complex64))
        assert A.dtype == torch.complex64
        np.testing.assert_allclose(A.to_dense().numpy(), M.toarray(), rtol=1e-6, atol=1e-6)
        x = np.random.default_rng(2).standard_normal(11).astype(np.complex64)
        np.testing.assert_allclose(
            trnsparse.spmv(A, torch.from_numpy(x)).numpy(), M @ x, rtol=1e-5, atol=1e-5
        )

    def test_float32_stays_float32(self):
        sp = pytest.importorskip("scipy.sparse")
        M = sp.random(6, 6, density=0.5, random_state=3, format="csr", dtype=np.float32)
        A = trnsparse.from_scipy(M)
        assert A.dtype == torch.float32
        np.testing.assert_allclose(A.to_dense().numpy(), M.toarray(), rtol=1e-6)

    @pytest.mark.parametrize("npdt", [np.float64, np.int64, np.int32, np.complex128])
    def test_preserves_dtype_exactly(self, npdt):
        # Through 0.13.0 every real input was stored as float32 (rounding
        # float64, and int64 above 2**24), and complex as complex64 (#69).
        sp = pytest.importorskip("scipy.sparse")
        M = sp.random(9, 7, density=0.5, random_state=4, format="csr") * 1e9
        if np.issubdtype(npdt, np.complexfloating):
            M = M + 1j * M
        M = (M + sp.eye(9, 7) * (1 / 3)).astype(npdt).tocsr()
        A = trnsparse.from_scipy(M)
        assert A.dtype == torch.from_numpy(M.data).dtype
        np.testing.assert_array_equal(A.to_dense().numpy(), M.toarray())

    def test_csr_float64_is_zero_copy(self):
        sp = pytest.importorskip("scipy.sparse")
        M = sp.random(8, 8, density=0.4, random_state=5, format="csr")
        A = trnsparse.from_scipy(M)
        assert np.shares_memory(A.values.numpy(), M.data)

    def test_other_formats_copy(self):
        sp = pytest.importorskip("scipy.sparse")
        M = sp.random(8, 8, density=0.4, random_state=6, format="coo")
        A = trnsparse.from_scipy(M)
        assert A.dtype == torch.float64
        np.testing.assert_array_equal(A.to_dense().numpy(), M.toarray())


# from_dense on the device (trnsci/trnsci#135): TorchNeuron's nonzero /
# torch.where / A[mask] round flat indices above 2^24 through fp32, so a
# device dense matrix with more than 2^24 elements got wrong column indices
# (and the values at those wrong positions). The device path takes the
# indices from a host nonzero instead.
_BIG_SHAPE = (2100, 8100)  # 17.0M elements > 2^24


def _big_sparse_dense(seed=2, nnz=4000):
    g = torch.Generator().manual_seed(seed)
    m, n = _BIG_SHAPE
    D = torch.zeros(m, n)
    r = torch.randint(0, m, (nnz,), generator=g)
    c = torch.randint(0, n, (nnz,), generator=g)
    D[r, c] = torch.randn(nnz, generator=g)
    D[m - 1, n - 3 :] = torch.tensor([1.5, 2.5, 3.5])  # flat indices above 2^24
    return D


def _assert_same_csr(got, ref):
    assert got.shape == ref.shape
    assert torch.equal(got.col_indices.cpu(), ref.col_indices)
    assert torch.equal(got.row_ptrs.cpu(), ref.row_ptrs)
    assert torch.equal(got.values.cpu(), ref.values)


class TestFromDenseHostIndex:
    def test_host_index_path_matches_where(self):
        from trnsparse.formats import _dense_nonzeros

        D = _big_sparse_dense()
        mask = D != 0
        r0, c0, v0 = _dense_nonzeros(D, mask, host_index=False)
        r1, c1, v1 = _dense_nonzeros(D, mask, host_index=True)
        assert torch.equal(r0, r1) and torch.equal(c0, c1) and torch.equal(v0, v1)
        assert int(c1[-1]) == _BIG_SHAPE[1] - 1 and float(v1[-1]) == 3.5

    @pytest.mark.parametrize("span", [1, 4, 6, 1 << 23])
    def test_host_index_path_row_blocks(self, monkeypatch, span):
        from trnsparse import formats
        from trnsparse.formats import _dense_nonzeros

        monkeypatch.setattr(formats, "_GATHER_SPAN", span)  # span 1: every row on the host
        g = torch.Generator().manual_seed(3)
        A = torch.randn(7, 5, generator=g).masked_fill(torch.rand(7, 5, generator=g) < 0.5, 0.0)
        A[3] = 0.0  # an empty row block
        A.requires_grad_(span == 1 << 23)
        mask = A != 0
        r1, c1, v1 = _dense_nonzeros(A, mask, host_index=True)
        r0, c0 = torch.where(mask)
        assert torch.equal(r0, r1) and torch.equal(c0, c1) and torch.equal(v1, A[mask])
        if A.requires_grad:
            v1.sum().backward()
            assert torch.equal(A.grad, mask.to(A.dtype))

    def test_host_index_path_threshold_and_dtype(self):
        from trnsparse.formats import _dense_nonzeros

        A = torch.tensor([[0.5, -2.0, 0.0], [3.0, 0.1, -0.2]], dtype=torch.float64)
        mask = A.abs() > 0.3
        r, c, v = _dense_nonzeros(A, mask, host_index=True)
        assert r.tolist() == [0, 0, 1] and c.tolist() == [0, 1, 0]
        assert v.dtype == torch.float64 and v.tolist() == [0.5, -2.0, 3.0]


def _neuron_device():
    try:
        import torch_neuronx  # noqa: F401
    except ImportError:
        pytest.skip("requires torch-neuronx")
    return "neuron"


@pytest.mark.neuron
def test_from_dense_on_device_above_2_24_elements():
    dev = _neuron_device()
    D = _big_sparse_dense()
    _assert_same_csr(trnsparse.from_dense(D.to(dev)), trnsparse.from_dense(D))


@pytest.mark.neuron
def test_bsr_from_dense_on_device():
    dev = _neuron_device()
    g = torch.Generator().manual_seed(7)
    D = torch.zeros(1024, 1024)
    D[:128, 896:] = torch.randn(128, 128, generator=g)
    D[512:640, 128:256] = torch.randn(128, 128, generator=g)
    ref = trnsparse.BSRMatrix.from_dense(D, block_size=128)
    got = trnsparse.BSRMatrix.from_dense(D.to(dev), block_size=128)
    assert torch.equal(got.block_col_indices.cpu(), ref.block_col_indices)
    assert torch.equal(got.block_row_ptrs.cpu(), ref.block_row_ptrs)
    assert torch.equal(got.blocks.cpu(), ref.blocks)
