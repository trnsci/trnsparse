"""Reference-conformance suite (trnsci/trnsci#105).

Every public trnsparse API is checked against an *independent* reference:
dense torch formulas in float64 / complex128, `scipy.sparse`, the dense
masked-softmax attention formula and `torch.linalg` solvers. Never against
another trnsparse backend, because a bug every backend shares (trnsparse#52,
`spmv_symmetric` ignoring `uplo`) is invisible to a differential test.

Axes: awkward sizes (1, 127, 129, 1000), empty and size-1 shapes, empty rows,
non-canonical CSR, COO duplicates, dtypes, every value of the keyword
arguments, non-contiguous dense operands, fp64 gradcheck / gradgradcheck,
input mutation and aliasing, and determinism.

Known open behaviour is marked `xfail(strict=True)` with its issue number, so
the case flips to a failure (and the marker has to go) once it is decided or
fixed.

CPU CI runs this module with the default pytorch backend. The same cases run
on a Trn2 box against the NKI paths:

    pytest tests/test_conformance.py --backend nki  --conformance-device neuron
    pytest tests/test_conformance.py --backend auto --conformance-device neuron
    TRNSPARSE_REQUIRE_NKI=1 pytest tests/test_conformance.py --backend nki \
        --conformance-device neuron

Dense operands go to `--conformance-device`; sparse structure (CSR / BSR
indices) stays on the host, which is the documented placement. With a
non-pytorch backend, float64 / complex128 results are held to fp32
tolerances (the kernels compute in fp32 by design) and the fp64 gradchecks
are skipped on a device. Under `TRNSPARSE_REQUIRE_NKI=1` the BSR paths raise
`NotImplementedError` for `block_size != 128` by design (#46); those cases
are skipped in that configuration.
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import trnsparse as ts

# ---------------------------------------------------------------- config

TOL = {
    torch.float64: 1e-10,
    torch.float32: 2e-5,
    torch.float16: 4e-3,
    torch.bfloat16: 3e-2,
    torch.complex128: 1e-10,
    torch.complex64: 2e-5,
}
FP32_TOL = 2e-4
REQUIRE_NKI = os.environ.get("TRNSPARSE_REQUIRE_NKI", "").lower() in ("1", "true", "yes")

_STATE: dict = {"backend": "pytorch", "device": torch.device("cpu")}


@pytest.fixture(scope="module", autouse=True)
def _conformance_backend(request):
    backend = request.config.getoption("--backend")
    device = torch.device(request.config.getoption("--conformance-device"))
    prev = ts.get_backend()
    ts.set_backend(backend)
    _STATE.update(backend=backend, device=device)
    yield
    ts.set_backend(prev)
    _STATE.update(backend="pytorch", device=torch.device("cpu"))


def on_device() -> bool:
    return _STATE["device"].type != "cpu"


def nki_backend() -> bool:
    return _STATE["backend"] != "pytorch"


def D(t: torch.Tensor) -> torch.Tensor:
    """Dense operand placement for this run."""
    return t.to(_STATE["device"])


def H(t):
    return t.detach().cpu() if isinstance(t, torch.Tensor) else t


def needs_pytorch_block(b: int):
    """Skip a non-128 BSR case under REQUIRE_NKI on an NKI backend (raises by design, #46)."""
    if b != 128 and REQUIRE_NKI and nki_backend():
        pytest.skip("block_size != 128 raises under TRNSPARSE_REQUIRE_NKI by design (#46)")


def ref_dtype(dt: torch.dtype) -> torch.dtype:
    if dt.is_complex:
        return torch.complex128
    if dt.is_floating_point:
        return torch.float64
    return torch.int64


def tol_for(dt: torch.dtype) -> float:
    t = TOL.get(dt, 0.0)
    if nki_backend() and dt in (torch.float64, torch.complex128):
        return FP32_TOL
    if nki_backend() and dt in (torch.float32, torch.complex64):
        return max(t, 1e-4)
    return t


def assert_close(got, ref, dtype=None, tol=None, check_dtype=True, what=""):
    got = H(got)
    ref = H(ref)
    dtype = dtype if dtype is not None else ref.dtype
    assert tuple(got.shape) == tuple(ref.shape), (
        f"{what} shape {tuple(got.shape)} != {tuple(ref.shape)}"
    )
    if check_dtype:
        assert got.dtype == dtype, f"{what} dtype {got.dtype} != {dtype}"
    if not (dtype.is_floating_point or dtype.is_complex):
        assert torch.equal(got.to(torch.int64), ref.to(torch.int64)), (
            f"{what} integer result differs"
        )
        return
    if got.numel() == 0:
        return
    rd = torch.complex128 if (got.is_complex() or ref.is_complex()) else torch.float64
    g, r = got.to(rd), ref.to(rd)
    assert torch.isfinite(g).all() or not torch.isfinite(r).all(), f"{what} non-finite result"
    err = (g - r).abs().max().item() / max(r.abs().max().item(), 1e-30)
    t = tol if tol is not None else tol_for(dtype)
    if nki_backend():
        t = max(t, tol_for(dtype))
    assert err <= t, f"{what} rel err {err:.3g} > tol {t:.1g}"


def value_or_raise(fn, ref, dtype, tol=None):
    """A dtype torch's sparse kernels lack may raise; it must not return a wrong value."""
    try:
        got = fn()
    except (NotImplementedError, RuntimeError, TypeError):
        return
    assert_close(got, ref, dtype, tol=tol)


# ---------------------------------------------------------------- builders


def rand_dense(m, n, density=0.3, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    if dtype.is_complex:
        A = torch.complex(
            torch.randn(m, n, generator=g, dtype=torch.float64),
            torch.randn(m, n, generator=g, dtype=torch.float64),
        )
    elif dtype.is_floating_point:
        A = torch.randn(m, n, generator=g, dtype=torch.float64)
    else:
        A = torch.randint(-5, 6, (m, n), generator=g).to(torch.float64)
    keep = torch.rand(m, n, generator=g) < density
    return (A * keep).to(dtype)


def rand_vec(shape, dtype=torch.float64, seed=1):
    g = torch.Generator().manual_seed(seed)
    if dtype.is_complex:
        return torch.complex(
            torch.randn(*shape, generator=g, dtype=torch.float64),
            torch.randn(*shape, generator=g, dtype=torch.float64),
        ).to(dtype)
    if dtype.is_floating_point:
        return torch.randn(*shape, generator=g, dtype=torch.float64).to(dtype)
    return torch.randint(-5, 6, shape, generator=g).to(dtype)


def csr_from_triplets(rows, cols, vals, shape, dtype=torch.float32):
    """Raw CSR without trnsparse code: unsorted columns and duplicates are kept."""
    order = sorted(range(len(rows)), key=lambda i: rows[i])
    rp = [0] * (shape[0] + 1)
    for i in order:
        rp[rows[i] + 1] += 1
    for i in range(shape[0]):
        rp[i + 1] += rp[i]
    return ts.CSRMatrix(
        values=torch.tensor([vals[i] for i in order], dtype=dtype),
        col_indices=torch.tensor([cols[i] for i in order], dtype=torch.long),
        row_ptrs=torch.tensor(rp, dtype=torch.long),
        shape=shape,
    )


def dense_of_triplets(rows, cols, vals, shape, dtype=torch.float64):
    out = torch.zeros(shape, dtype=dtype)
    for r, c, v in zip(rows, cols, vals, strict=True):
        out[r, c] += v
    return out


def bsr_ref_dense(A: ts.BSRMatrix) -> torch.Tensor:
    """Independent BSR -> dense (no trnsparse code)."""
    m, n = A.shape
    b = A.block_size
    out = torch.zeros(-(-m // b) * b, -(-n // b) * b, dtype=ref_dtype(A.blocks.dtype))
    rp = A.block_row_ptrs.cpu().tolist()
    ci = A.block_col_indices.cpu().tolist()
    blk = A.blocks.detach().cpu().to(out.dtype)
    for i in range(len(rp) - 1):
        for k in range(rp[i], rp[i + 1]):
            out[i * b : (i + 1) * b, ci[k] * b : (ci[k] + 1) * b] += blk[k]
    return out[:m, :n]


def mask_bsr(n_tiles, b, pattern, dtype=torch.float32):
    pattern = sorted(set(pattern))
    rp = [0] * (n_tiles + 1)
    for r, _ in pattern:
        rp[r + 1] += 1
    for i in range(n_tiles):
        rp[i + 1] += rp[i]
    return ts.BSRMatrix(
        blocks=torch.ones(len(pattern), b, b, dtype=dtype),
        block_col_indices=torch.tensor([c for _, c in pattern], dtype=torch.long),
        block_row_ptrs=torch.tensor(rp, dtype=torch.long),
        shape=(n_tiles * b, n_tiles * b),
        block_size=b,
    )


def attn_dense(Q, K, V, nt, b, pattern, scale):
    """Dense masked-softmax attention (differentiable); fully masked rows -> 0."""
    M = torch.zeros(nt * b, nt * b, dtype=torch.bool)
    for r, c in pattern:
        M[r * b : (r + 1) * b, c * b : (c + 1) * b] = True
    S = (Q @ K.T) * scale
    P = torch.softmax(S.masked_fill(~M, float("-inf")), dim=-1)
    P = torch.where(M.any(1, keepdim=True), P, torch.zeros_like(P))
    return P @ V


def attn_ref(Q, K, V, nt, b, pattern, scale):
    Q, K, V = (H(t).to(torch.float64) for t in (Q, K, V))
    return attn_dense(Q, K, V, nt, b, pattern, scale)


def spd_dense(n, b, seed=3, band=1):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    A = A @ A.T / n
    i = torch.arange(n)
    mask = ((i[:, None] // b) - (i[None, :] // b)).abs() <= band
    A = A * mask + (n * 0.1 + 1.0) * torch.eye(n, dtype=torch.float64)
    return (A + A.T) / 2


def spd_bsr(n, b, dtype=torch.float64, seed=3, band=1):
    A = spd_dense(n, b, seed, band)
    return ts.BSRMatrix.from_dense(A.to(dtype), block_size=b), A


def sym_ref(A: torch.Tensor, uplo: str) -> torch.Tensor:
    """BLAS symv semantics: only the `uplo` triangle of A is referenced."""
    T = torch.triu(A) if uplo == "upper" else torch.tril(A)
    S = torch.triu(A, 1) if uplo == "upper" else torch.tril(A, -1)
    return T + S.T


# ---------------------------------------------------------------- shapes

SHAPES = [(1, 1), (1, 129), (129, 1), (127, 129), (129, 127), (1000, 1000)]
EDGE_SHAPES = [(0, 5), (5, 0), (0, 0)]
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
LOWP_INT = [torch.bfloat16, torch.float16, torch.int32, torch.int64]


def _density(m, n):
    return 0.01 if m * n >= 10**6 else 0.3


def _sid(s):
    return f"{s[0]}x{s[1]}"


# ---------------------------------------------------------------- formats


@pytest.mark.parametrize("shape", SHAPES + EDGE_SHAPES, ids=_sid)
@pytest.mark.parametrize("dtype", DTYPES + LOWP_INT, ids=str)
def test_from_dense_roundtrip(shape, dtype):
    Ad = rand_dense(*shape, density=_density(*shape), dtype=dtype, seed=11)
    A = ts.from_dense(Ad)
    assert A.shape == shape
    assert A.nnz == int((Ad != 0).sum())
    assert A.dtype == dtype
    assert_close(A.to_dense(), Ad, dtype)
    assert_close(A.to_coo().to_dense(), Ad, dtype)
    assert_close(A.to_coo().to_csr().to_dense(), Ad, dtype)
    total = shape[0] * shape[1]
    if total:
        assert A.density == pytest.approx(A.nnz / total)


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.5])
def test_from_dense_threshold(threshold):
    Ad = rand_dense(127, 129, 0.5, torch.float64, 12)
    ref = Ad * (Ad.abs() > threshold)
    assert_close(ts.from_dense(Ad, threshold=threshold).to_dense(), ref)


@pytest.mark.parametrize("view", ["T", "step", "expanded"])
def test_from_dense_noncontiguous(view):
    base = rand_dense(60, 40, 0.5, torch.float32, 13)
    Av = {"T": base.T, "step": base[::2, ::3], "expanded": base[:, :1].expand(60, 7)}[view]
    assert_close(ts.from_dense(Av).to_dense(), Av)
    assert_close(ts.BSRMatrix.from_dense(Av, block_size=8).to_dense(), Av)


def test_noncanonical_csr():
    rows = [0, 0, 0, 1, 2, 2, 2, 3]
    cols = [3, 0, 2, 1, 2, 0, 2, 3]  # row 0 unsorted, (2, 2) twice, an explicit zero
    vals = [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    A = csr_from_triplets(rows, cols, vals, (4, 4))
    Dr = dense_of_triplets(rows, cols, vals, (4, 4))
    x = rand_vec((4,), torch.float32, 14)
    assert_close(A.to_dense(), Dr, torch.float32)
    assert_close(A.to_coo().to_dense(), Dr, torch.float32)
    assert_close(ts.spmv(A, D(x)), Dr @ x.double(), torch.float32)
    assert_close(
        ts.spmm(A, D(x[:, None].repeat(1, 3))), Dr @ x.double()[:, None].repeat(1, 3), torch.float32
    )
    assert_close(ts.sparse_transpose(A).to_dense(), Dr.T, torch.float32)
    assert_close(ts.sparse_add(A, A).to_dense(), 2 * Dr, torch.float32)
    assert_close(ts.sparse_scale(A, -2.0).to_dense(), -2 * Dr, torch.float32)
    assert_close(bsr_ref_dense(ts.BSRMatrix.from_csr(A, block_size=2)), Dr, check_dtype=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_coo_duplicates_sum(dtype):
    import scipy.sparse as sp

    r, c, v = [0, 1, 0, 2, 2], [1, 2, 1, 0, 0], [1.0, 2.0, 10.0, 3.0, -1.0]
    coo = ts.COOMatrix(torch.tensor(v, dtype=dtype), torch.tensor(r), torch.tensor(c), (3, 3))
    ref = torch.from_numpy(sp.coo_matrix((v, (r, c)), shape=(3, 3)).toarray()).to(dtype)
    assert coo.nnz == 5
    assert coo.dtype == dtype
    assert_close(coo.to_dense(), ref, dtype)
    assert_close(coo.to_csr().to_dense(), ref, dtype)
    assert_close(ts.spmv(coo.to_csr(), D(torch.ones(3, dtype=dtype))), ref.sum(1), dtype)


_SCIPY_DTYPE_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="trnsci/trnsparse#69 needs decision: from_scipy stores float32/complex64 "
    "whatever the scipy dtype",
)


@pytest.mark.parametrize(
    "npdt",
    [
        np.float32,
        np.complex64,
        pytest.param(np.float64, marks=_SCIPY_DTYPE_XFAIL),
        pytest.param(np.complex128, marks=_SCIPY_DTYPE_XFAIL),
        pytest.param(np.int64, marks=_SCIPY_DTYPE_XFAIL),
    ],
    ids=lambda d: d.__name__,
)
@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
def test_from_scipy(npdt, fmt):
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    M = sp.random(127, 129, density=0.1, random_state=rng, format="csr")
    if np.issubdtype(npdt, np.complexfloating):
        M = M + 1j * sp.random(127, 129, density=0.1, random_state=rng, format="csr")
    if np.issubdtype(npdt, np.integer):
        M = M * 1e8
    M = M.astype(npdt).asformat(fmt)
    ref = torch.from_numpy(M.toarray())
    got = ts.from_scipy(M)
    assert got.shape == M.shape
    assert got.dtype == ref.dtype, f"dtype {got.dtype} != scipy {ref.dtype}"
    assert_close(got.to_dense(), ref, ref.dtype)


@pytest.mark.parametrize("n", [1, 129])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.int32])
def test_eye_sparse(n, dtype):
    E = ts.eye_sparse(n, dtype=dtype)
    assert_close(E.to_dense(), torch.eye(n, dtype=dtype), dtype)


def test_eye_sparse_default_dtype():
    assert ts.eye_sparse(4).dtype == torch.float32


BSR_SHAPES = [
    (1, 1, 1),
    (10, 7, 4),
    (3, 3, 4),
    (127, 129, 8),
    (1, 130, 128),
    (130, 1, 128),
    (1000, 1000, 128),
]


@pytest.mark.parametrize("m,n,b", BSR_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_bsr_roundtrip(m, n, b, dtype):
    Ad = rand_dense(m, n, _density(m, n), dtype, 15)
    A = ts.BSRMatrix.from_dense(Ad, block_size=b)
    assert A.shape == (m, n)
    assert A.block_size == b
    assert A.dtype == dtype
    assert A.n_blocks == A.blocks.shape[0] == A.block_col_indices.numel()
    assert_close(bsr_ref_dense(A), Ad, ref_dtype(dtype), check_dtype=False)
    assert_close(A.to_dense(), Ad, dtype)
    assert_close(A.to_csr().to_dense(), Ad, dtype)
    assert_close(ts.BSRMatrix.from_csr(ts.from_dense(Ad), block_size=b).to_dense(), Ad, dtype)


@pytest.mark.parametrize("threshold", [0.0, 1.0, 10.0])
def test_bsr_threshold(threshold):
    Ad = rand_dense(16, 16, 0.3, torch.float64, 16)
    Ad[:4, :4] *= 0.01
    A = ts.BSRMatrix.from_dense(Ad, block_size=4, threshold=threshold)
    grid = Ad.reshape(4, 4, 4, 4).permute(0, 2, 1, 3).abs().amax((-2, -1))
    keep = grid > threshold
    assert A.n_blocks == int(keep.sum())
    ref = Ad * keep.repeat_interleave(4, 0).repeat_interleave(4, 1)
    assert_close(A.to_dense(), ref)


# ---------------------------------------------------------------- spmv / spmm


@pytest.mark.parametrize("shape", SHAPES + EDGE_SHAPES, ids=_sid)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_spmv(shape, dtype):
    m, n = shape
    Ad = rand_dense(m, n, _density(m, n), dtype, 21)
    x = rand_vec((n,), dtype, 22)
    R = ref_dtype(dtype)
    assert_close(ts.spmv(ts.from_dense(Ad), D(x)), Ad.to(R) @ x.to(R), dtype)


@pytest.mark.parametrize("alpha", [1.0, -2.5, 0.0])
@pytest.mark.parametrize("beta", [0.0, 1.0, 3.0])
@pytest.mark.parametrize("y_kind", ["none", "vec", "scalar_bcast"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex128], ids=str)
def test_spmv_alpha_beta(alpha, beta, y_kind, dtype):
    Ad = rand_dense(129, 127, 0.2, dtype, 23)
    x = rand_vec((127,), dtype, 24)
    R = ref_dtype(dtype)
    y = {
        "none": None,
        "vec": rand_vec((129,), dtype, 25),
        "scalar_bcast": rand_vec((1,), dtype, 26),
    }[y_kind]
    ref = alpha * (Ad.to(R) @ x.to(R))
    if y is not None and beta != 0.0:
        ref = ref + beta * y.to(R)
    got = ts.spmv(ts.from_dense(Ad), D(x), alpha=alpha, y=None if y is None else D(y), beta=beta)
    assert_close(got, ref, dtype)


@pytest.mark.parametrize("dtype", LOWP_INT, ids=str)
def test_spmv_spmm_lowp_int_value_or_raise(dtype):
    # torch's sparse CSR kernels lack these dtypes: raising is fine, a wrong value is not.
    Ad = rand_dense(37, 29, 0.3, dtype, 27)
    A = ts.from_dense(Ad)
    x = rand_vec((29,), dtype, 28)
    B = rand_vec((29, 6), dtype, 29)
    R = ref_dtype(dtype)
    value_or_raise(lambda: ts.spmv(A, D(x)), Ad.to(R) @ x.to(R), dtype)
    value_or_raise(lambda: ts.spmm(A, D(B)), Ad.to(R) @ B.to(R), dtype)


@pytest.mark.parametrize("shape", SHAPES + EDGE_SHAPES, ids=_sid)
@pytest.mark.parametrize("k", [1, 7])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_spmm(shape, k, dtype):
    m, n = shape
    Ad = rand_dense(m, n, _density(m, n), dtype, 31)
    B = rand_vec((n, k), dtype, 32)
    R = ref_dtype(dtype)
    assert_close(ts.spmm(ts.from_dense(Ad), D(B)), Ad.to(R) @ B.to(R), dtype)


@pytest.mark.parametrize("alpha", [1.0, 0.5])
@pytest.mark.parametrize("beta", [0.0, -1.0])
@pytest.mark.parametrize("c_shape", ["full", "row", "col", "none"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64], ids=str)
def test_spmm_alpha_beta(alpha, beta, c_shape, dtype):
    Ad = rand_dense(129, 127, 0.2, dtype, 33)
    B = rand_vec((127, 5), dtype, 34)
    R = ref_dtype(dtype)
    C = {
        "full": rand_vec((129, 5), dtype, 35),
        "row": rand_vec((5,), dtype, 36),
        "col": rand_vec((129, 1), dtype, 37),
        "none": None,
    }[c_shape]
    ref = alpha * (Ad.to(R) @ B.to(R))
    if C is not None and beta != 0.0:
        ref = ref + beta * C.to(R)
    got = ts.spmm(ts.from_dense(Ad), D(B), alpha=alpha, C=None if C is None else D(C), beta=beta)
    assert_close(got, ref, dtype)


def test_spmv_spmm_epilogue_must_not_enlarge():
    # torch.addmv / addmm reject a y / C that broadcasts to a larger output (#67).
    A = ts.from_dense(rand_dense(12, 10, 0.4, torch.float32, 38))
    with pytest.raises(ValueError):
        ts.spmv(A, D(torch.ones(10)), y=D(torch.ones(12, 1)), beta=1.0)
    with pytest.raises(ValueError):
        ts.spmm(A, D(torch.ones(10, 5)), C=D(torch.ones(12, 5, 1)), beta=1.0)


def _views(n, k, seed):
    big = rand_vec((2 * n, 3 * k + 3), torch.float32, seed)
    return {
        "colslice": big[:n, :k],
        "T": rand_vec((k, n), torch.float32, seed + 1).T,
        "step": big[::2, ::3][:n, :k],
        "expanded": rand_vec((n, 1), torch.float32, seed + 2).expand(n, k),
        "as_strided": torch.as_strided(
            rand_vec((n * k * 4,), torch.float32, seed + 3), (n, k), (3, 7)
        ),
        "neg_stride": rand_vec((n, k), torch.float32, seed + 4).flip(0),
    }


@pytest.mark.parametrize("view", list(_views(4, 2, 0)))
def test_spmv_spmm_noncontiguous(view):
    Ad = rand_dense(129, 127, 0.2, torch.float32, 41)
    A = ts.from_dense(Ad)
    Bv = _views(127, 8, 42)[view]
    assert_close(ts.spmm(A, D(Bv)), Ad.double() @ Bv.double(), torch.float32)
    assert_close(ts.spmv(A, D(Bv[:, 1])), Ad.double() @ Bv[:, 1].double(), torch.float32)
    # CSR whose values are a strided view, and int32 indices
    vals = torch.stack([A.values, torch.zeros_like(A.values)], 1)[:, 0]
    Av = ts.CSRMatrix(vals, A.col_indices.int(), A.row_ptrs.int(), A.shape)
    assert_close(ts.spmm(Av, D(Bv)), Ad.double() @ Bv.double(), torch.float32)


def test_spmv_spmm_batch_semantics():
    Ad = rand_dense(12, 10, 0.4, torch.float32, 43)
    A = ts.from_dense(Ad)
    X = rand_vec((10, 5), torch.float32, 44)
    assert_close(ts.spmv(A, D(X)), Ad.double() @ X.double(), torch.float32)
    assert_close(ts.spmm(A, D(X[:, 0])), Ad.double() @ X[:, 0].double(), torch.float32)


def test_spmv_spmm_empty_rows_and_nnz0():
    Ae = torch.zeros(6, 6)
    Ae[0, 1], Ae[3, 3], Ae[5, 0] = 1.0, 2.0, 3.0
    assert_close(
        ts.spmm(ts.from_dense(Ae), D(torch.ones(6, 2))),
        Ae.double() @ torch.ones(6, 2).double(),
        torch.float32,
    )
    Z = ts.from_dense(torch.zeros(4, 4))
    assert_close(ts.spmv(Z, D(torch.ones(4))), torch.zeros(4), torch.float32)
    assert_close(ts.spmm(Z, D(torch.ones(4, 3))), torch.zeros(4, 3), torch.float32)
    A5 = ts.from_dense(rand_dense(6, 5, 0.5, torch.float32, 45))
    assert_close(ts.spmm(A5, D(torch.zeros(5, 0))), torch.zeros(6, 0), torch.float32)


# ---------------------------------------------------------------- spmv_symmetric


@pytest.mark.parametrize("n", [1, 127, 129, 1000])
@pytest.mark.parametrize("uplo", ["upper", "lower"])
@pytest.mark.parametrize("store", ["upper", "lower", "full"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex128], ids=str)
def test_spmv_symmetric(n, uplo, store, dtype):
    Ad = rand_dense(n, n, 0.01 if n == 1000 else 0.2, dtype, 51) + torch.eye(n, dtype=dtype)
    if store != "full":
        Ad = torch.triu(Ad) if store == "upper" else torch.tril(Ad)
    x = rand_vec((n,), dtype, 52)
    R = ref_dtype(dtype)
    if store != "full" and store != uplo and n > 1:
        # Compatibility path (#52): the other triangle is used, with a DeprecationWarning.
        ref = sym_ref(Ad.to(R), store) @ x.to(R)
        with pytest.warns(DeprecationWarning):
            got = ts.spmv_symmetric(ts.from_dense(Ad), D(x), uplo=uplo)
    else:
        ref = sym_ref(Ad.to(R), uplo) @ x.to(R)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            got = ts.spmv_symmetric(ts.from_dense(Ad), D(x), uplo=uplo)
    assert_close(got, ref, dtype)


@pytest.mark.parametrize("alpha", [1.0, -0.5, 0.0])
def test_spmv_symmetric_alpha_default_uplo(alpha):
    Ad = rand_dense(129, 129, 0.2, torch.float64, 53)
    x = rand_vec((129,), torch.float64, 54)
    assert_close(
        ts.spmv_symmetric(ts.from_dense(Ad), D(x), alpha=alpha), alpha * (sym_ref(Ad, "upper") @ x)
    )


def test_spmv_symmetric_edge_and_2d():
    Z = ts.from_dense(torch.zeros(4, 4))
    assert_close(ts.spmv_symmetric(Z, D(torch.ones(4))), torch.zeros(4), torch.float32)
    assert_close(
        ts.spmv_symmetric(ts.from_dense(torch.tensor([[2.5]])), D(torch.tensor([2.0]))),
        torch.tensor([5.0]),
    )
    Dg = torch.diag(torch.arange(1.0, 6.0))
    assert_close(ts.spmv_symmetric(ts.from_dense(Dg), D(torch.ones(5))), torch.arange(1.0, 6.0))
    S = torch.triu(rand_dense(10, 10, 0.5, torch.float32, 55))
    X = rand_vec((10, 5), torch.float32, 56)
    assert_close(
        ts.spmv_symmetric(ts.from_dense(S), D(X)),
        sym_ref(S.double(), "upper") @ X.double(),
        torch.float32,
    )
    # non-canonical: unsorted columns + a duplicate in the upper triangle
    r, c, v = [0, 0, 1, 1, 2], [2, 0, 1, 1, 2], [1.0, 2.0, 3.0, 4.0, 5.0]
    Ds = dense_of_triplets(r, c, v, (3, 3))
    assert_close(
        ts.spmv_symmetric(csr_from_triplets(r, c, v, (3, 3)), D(torch.ones(3))),
        sym_ref(Ds, "upper").sum(1),
        torch.float32,
    )
    with pytest.raises(ValueError):
        ts.spmv_symmetric(Z, D(torch.ones(4)), uplo="U")


# ---------------------------------------------------------------- CSR algebra


@pytest.mark.parametrize("shape", [(1, 1), (127, 129), (1000, 1000), (0, 5)], ids=_sid)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_csr_algebra(shape, dtype):
    Ad = rand_dense(*shape, _density(*shape), dtype, 61)
    Bd = rand_dense(*shape, _density(*shape), dtype, 62)
    A, B = ts.from_dense(Ad), ts.from_dense(Bd)
    R = ref_dtype(dtype)
    assert_close(ts.sparse_transpose(A).to_dense(), Ad.T, dtype)
    assert ts.sparse_transpose(A).shape == (shape[1], shape[0])
    for s in (1.0, 3.0, -0.5, 0.0):
        assert_close(ts.sparse_scale(A, s).to_dense(), s * Ad.to(R), dtype)
    for alpha, beta in ((1.0, 1.0), (2.0, -1.0), (0.0, 1.0), (1.0, 0.0)):
        got = ts.sparse_add(A, B, alpha=alpha, beta=beta).to_dense()
        assert_close(got, alpha * Ad.to(R) + beta * Bd.to(R), dtype)
    assert_close(ts.nnz_per_row(A), (Ad != 0).sum(1), torch.int64)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.int32, torch.int64], ids=str
)
def test_csr_algebra_lowp_int(dtype):
    Ad = rand_dense(37, 29, 0.3, dtype, 63)
    Bd = rand_dense(37, 29, 0.3, dtype, 64)
    A = ts.from_dense(Ad)
    R = ref_dtype(dtype)
    assert_close(ts.sparse_transpose(A).to_dense(), Ad.T, dtype)
    assert_close(ts.sparse_scale(A, 3).to_dense(), 3 * Ad.to(R), dtype)
    assert_close(
        ts.sparse_add(A, ts.from_dense(Bd), alpha=2, beta=-1).to_dense(),
        2 * Ad.to(R) - Bd.to(R),
        dtype,
    )


# ---------------------------------------------------------------- bsr_spmm


@pytest.mark.parametrize("m,n,b", BSR_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64], ids=str)
def test_bsr_spmm(m, n, b, k, dtype):
    needs_pytorch_block(b)
    Ad = rand_dense(m, n, _density(m, n) / 3, dtype, 71)
    A = ts.BSRMatrix.from_dense(Ad, block_size=b)
    B = rand_vec((n, k), dtype, 72)
    R = ref_dtype(dtype)
    assert_close(ts.bsr_spmm(A, D(B)), Ad.to(R) @ B.to(R), dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("b", [8, 128])
def test_bsr_spmm_lowp(dtype, b):
    needs_pytorch_block(b)
    Ad = rand_dense(256, 384, 0.05, dtype, 73)
    B = rand_vec((384, 64), dtype, 74)
    value_or_raise(
        lambda: ts.bsr_spmm(ts.BSRMatrix.from_dense(Ad, block_size=b), D(B)),
        Ad.double() @ B.double(),
        dtype,
    )


@pytest.mark.parametrize("view", list(_views(4, 2, 0)))
@pytest.mark.parametrize("b", [8, 128])
def test_bsr_spmm_noncontiguous(view, b):
    needs_pytorch_block(b)
    Ad = rand_dense(256, 256, 0.02, torch.float32, 75)
    A = ts.BSRMatrix.from_dense(Ad, block_size=b)
    Bv = _views(256, 8, 76)[view]
    assert_close(ts.bsr_spmm(A, D(Bv)), Ad.double() @ Bv.double(), torch.float32)


def test_bsr_spmm_transposed_block_view_and_empty():
    Ad = rand_dense(256, 256, 0.05, torch.float32, 77)
    Ab = ts.BSRMatrix.from_dense(Ad, block_size=128)
    At = ts.BSRMatrix(
        Ab.blocks.transpose(1, 2), Ab.block_col_indices, Ab.block_row_ptrs, Ab.shape, 128
    )
    B = rand_vec((256, 16), torch.float32, 78)
    assert_close(ts.bsr_spmm(At, D(B)), bsr_ref_dense(At) @ B.double(), torch.float32)
    Az = ts.BSRMatrix.from_dense(torch.zeros(256, 256), block_size=128)
    assert Az.n_blocks == 0
    assert_close(ts.bsr_spmm(Az, D(torch.ones(256, 4))), torch.zeros(256, 4), torch.float32)


@pytest.mark.parametrize("m,n,b", [(12, 12, 5), (8, 12, 4), (12, 8, 4), (129, 129, 128)])
def test_bsr_diagonal(m, n, b):
    Ad = rand_dense(m, n, 0.5, torch.float64, 79) + torch.eye(m, n, dtype=torch.float64)
    assert_close(ts.bsr_diagonal(ts.BSRMatrix.from_dense(Ad, block_size=b)), torch.diagonal(Ad))


# ---------------------------------------------------------------- screening


def _schwarz_mask(dg, thr):
    q = dg.double().abs().sqrt()
    return (q[:, None] * q[None, :]) > math.sqrt(thr)


@pytest.mark.parametrize("n", [1, 127, 128, 129])
@pytest.mark.parametrize("threshold", [0.0, 0.3, 1e6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_screened_spmm(n, threshold, dtype):
    A = rand_vec((n, n), dtype, 81)
    dg = rand_vec((n,), dtype, 82)  # signed: the bound uses |diag|
    B = rand_vec((n, 7), dtype, 83)
    m = _schwarz_mask(dg, threshold)
    got = ts.screened_spmm(D(A), D(dg), D(B), threshold)
    assert_close(got, (A.double() * m) @ B.double(), dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_screened_spmm_lowp(dtype):
    A = rand_vec((128, 128), dtype, 84)
    dg = rand_vec((128,), dtype, 85).abs()
    B = rand_vec((128, 32), dtype, 86)
    # The mask is a threshold compare: evaluate it in the input dtype.
    q = torch.sqrt(torch.abs(dg))
    m = (q[:, None] * q[None, :]) > math.sqrt(0.3)
    assert_close(ts.screened_spmm(D(A), D(dg), D(B), 0.3), (A.double() * m) @ B.double(), dtype)


def test_screened_spmm_noncontiguous_and_1d():
    A = rand_vec((128, 128), torch.float32, 87)
    dg = rand_vec((256,), torch.float32, 88).abs()[::2]
    B = rand_vec((64, 128), torch.float32, 89).T
    m = _schwarz_mask(dg, 0.2)
    assert_close(
        ts.screened_spmm(D(A.T), D(dg), D(B), 0.2), (A.T.double() * m) @ B.double(), torch.float32
    )
    assert_close(
        ts.screened_spmm(D(A), D(torch.ones(128)), D(torch.ones(128)), 0.5),
        A.double() @ torch.ones(128, dtype=torch.float64),
        torch.float32,
    )


@pytest.mark.parametrize("threshold", [1e-10, 0.25, 4.0])
def test_screening_utilities(threshold):
    Qd = rand_vec((129, 129), torch.float64, 91)
    Pm = rand_vec((129, 129), torch.float64, 92)
    Q = Qd.abs().sqrt()
    assert_close(ts.schwarz_bounds(Qd), Q)
    sig = Q > math.sqrt(threshold)
    assert torch.equal(ts.screen_quartets(Q, threshold), sig)
    assert torch.equal(ts.density_screen(Q, Pm, threshold), Q * (Pm.abs() * Q).max() > threshold)
    s = ts.sparsity_stats(Q, threshold)
    k = int(sig.sum())
    assert s["n_shells"] == 129
    assert s["total_pairs"] == 129**2
    assert s["significant_pairs"] == k
    assert s["pair_sparsity"] == pytest.approx(1 - k / 129**2)
    assert s["total_quartets"] == 129**4
    assert s["significant_quartets_upper"] == k * k


def test_screening_defaults():
    Q = torch.tensor([[1e-6, 1e-4], [1.0, 0.0]], dtype=torch.float64)
    assert torch.equal(ts.screen_quartets(Q), Q > 1e-5)
    assert torch.equal(ts.density_screen(Q, torch.ones(2, 2)), Q * Q.max() > 1e-10)


# ---------------------------------------------------------------- attention

ATTN_CASES = [
    # (block, n_tiles, head_dim, pattern)
    (4, 3, 5, [(0, 0), (1, 0), (1, 2), (2, 2)]),
    (4, 2, 3, [(0, 1), (1, 0), (1, 1)]),
    (128, 1, 1, [(0, 0)]),
    (128, 3, 64, [(0, 0), (1, 0), (1, 1), (2, 2), (2, 0)]),
    (128, 3, 32, [(0, 0), (2, 1), (2, 2)]),  # block row 1 empty -> zero rows
]


@pytest.mark.parametrize(
    "b,nt,hd,pat", ATTN_CASES, ids=lambda v: str(v) if not isinstance(v, list) else ""
)
@pytest.mark.parametrize("scale", [None, 0.1, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_attention(b, nt, hd, pat, scale, dtype):
    needs_pytorch_block(b)
    Q, K, V = (rand_vec((nt * b, hd), dtype, s) for s in (101, 102, 103))
    sc = hd**-0.5 if scale is None else scale
    got = ts.block_sparse_attention_tiled(D(Q), D(K), D(V), mask_bsr(nt, b, pat), scale=scale)
    assert_close(got, attn_ref(Q, K, V, nt, b, pat, sc), dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_attention_lowp(dtype):
    pat = [(0, 0), (1, 0), (1, 1), (2, 2), (2, 0)]
    Q, K, V = (rand_vec((384, 64), dtype, s) for s in (104, 105, 106))
    got = ts.block_sparse_attention_tiled(D(Q), D(K), D(V), mask_bsr(3, 128, pat))
    tol = {torch.bfloat16: 6e-2, torch.float16: 1e-2}[dtype]
    assert_close(got, attn_ref(Q, K, V, 3, 128, pat, 64**-0.5), dtype, tol=tol)


def test_attention_noncontiguous_and_cross_length():
    pat = [(0, 0), (1, 0), (1, 1)]
    Q = rand_vec((64, 256), torch.float32, 107).T
    K = rand_vec((256, 128), torch.float32, 108)[:, ::2]
    V = rand_vec((256, 64), torch.float32, 109).flip(0)
    got = ts.block_sparse_attention_tiled(D(Q), D(K), D(V), mask_bsr(2, 128, pat))
    assert_close(got, attn_ref(Q, K, V, 2, 128, pat, 64**-0.5), torch.float32)
    # S_k != S_q: one block row over two key blocks
    Qc = rand_vec((128, 16), torch.float32, 110)
    Kc = rand_vec((256, 16), torch.float32, 111)
    m = ts.BSRMatrix(
        torch.ones(2, 128, 128), torch.tensor([0, 1]), torch.tensor([0, 2]), (128, 256), 128
    )
    ref = torch.softmax((Qc.double() @ Kc.double().T) * 16**-0.5, -1) @ Kc.double()
    assert_close(ts.block_sparse_attention_tiled(D(Qc), D(Kc), D(Kc), m), ref, torch.float32)


def test_attention_rejects_partial_block():
    Q = torch.randn(130, 8)
    with pytest.raises(ValueError):
        ts.block_sparse_attention_tiled(D(Q), D(Q), D(Q), mask_bsr(2, 128, [(0, 0)]))


# ---------------------------------------------------------------- iterative solvers


@pytest.mark.parametrize("n,b", [(1, 1), (64, 8), (129, 8), (300, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("precond", ["none", "jacobi_callable", "dense_inverse"])
def test_cg_bsr(n, b, dtype, precond):
    needs_pytorch_block(b)
    A, Ad = spd_bsr(n, b, dtype)
    rhs = rand_vec((n,), dtype, 121)
    M = {
        "none": None,
        "jacobi_callable": ts.jacobi_preconditioner_bsr(A),
        "dense_inverse": torch.diag(1.0 / torch.diagonal(Ad)).to(dtype),
    }[precond]
    tol = 1e-12 if dtype == torch.float64 else 1e-6
    x, it, rel = ts.cg_bsr(A, D(rhs), tol=tol, max_iter=2000, M=M)
    xs = torch.linalg.solve(Ad, rhs.double())
    assert_close(x, xs, dtype, tol=1e-8 if dtype == torch.float64 else 1e-4)
    true_rel = (
        torch.linalg.norm(rhs.double() - Ad @ H(x).double()) / torch.linalg.norm(rhs.double())
    ).item()
    assert 0 < it <= 2000
    assert rel == pytest.approx(true_rel, rel=1e-2, abs=10 * tol)


def test_cg_bsr_x0_and_zero_b():
    A, Ad = spd_bsr(300, 128)
    rhs = rand_vec((300,), torch.float64, 122)
    xs = torch.linalg.solve(Ad, rhs)
    x, _, _ = ts.cg_bsr(A, D(rhs), x0=D(xs), tol=1e-8)
    assert_close(x, xs, tol=1e-8)
    x, it, rel = ts.cg_bsr(A, D(torch.zeros(300, dtype=torch.float64)))
    assert_close(x, torch.zeros(300, dtype=torch.float64))
    assert it == 0


@pytest.mark.xfail(
    strict=True, raises=UnboundLocalError, reason="trnsci/trnsparse#71: cg_bsr(max_iter=0) raises"
)
def test_cg_bsr_max_iter_zero():
    A, Ad = spd_bsr(64, 8)
    rhs = rand_vec((64,), torch.float64, 123)
    x, it, rel = ts.cg_bsr(A, D(rhs), max_iter=0)
    assert_close(x, torch.zeros(64, dtype=torch.float64))
    assert it == 0
    assert rel == pytest.approx(1.0)


@pytest.mark.xfail(
    strict=True, reason="trnsci/trnsparse#71: on breakdown cg_bsr reports iters=max_iter"
)
def test_cg_bsr_breakdown_iteration_count():
    # tol=0 cannot be met; CG on n=8 converges exactly in <= 8 steps, then breaks down.
    A, Ad = spd_bsr(8, 4, band=2)
    rhs = rand_vec((8,), torch.float64, 124)
    x, it, rel = ts.cg_bsr(A, D(rhs), tol=0.0, max_iter=500)
    assert_close(x, torch.linalg.solve(Ad, rhs), tol=1e-8)
    assert it <= 8 + 1, f"reported {it} iterations for an 8x8 system"


@pytest.mark.parametrize("n", [128, 300])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"preconditioner": "block_jacobi"},
        {"recurrence": "chronopoulos_gear"},
        {"matvec": "row"},
        {"matvec": "column", "check_every": 1},
        {"precision": "bf16x2", "matvec": "column"},
    ],
    ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()) or "default",
)
def test_cg_bsr_fused(n, dtype, kw):
    A, Ad = spd_bsr(n, 128, dtype)
    rhs = rand_vec((n,), dtype, 125)
    tol = 1e-12 if dtype == torch.float64 else 1e-7
    x, it, rel = ts.cg_bsr_fused(A, D(rhs), tol=tol, max_iter=2000, **kw)
    xs = torch.linalg.solve(Ad, rhs.double())
    etol = 1e-8 if dtype == torch.float64 and not nki_backend() else 1e-4
    if kw.get("recurrence"):
        etol *= 10
    if kw.get("precision"):
        etol = max(etol, 1e-4)
    assert_close(x, xs, dtype, tol=etol)
    assert isinstance(it, int) and 0 < it <= 2000
    assert isinstance(rel, float)


def test_cg_bsr_fused_bf16_precision():
    # bf16 storage of A: CG solves the bf16-rounded system (documented, ~2^-9 relative).
    A, Ad = spd_bsr(256, 128, torch.float32)
    rhs = rand_vec((256,), torch.float32, 126)
    x, _, _ = ts.cg_bsr_fused(A, D(rhs), tol=1e-6, precision="bf16", matvec="column")
    assert_close(x, torch.linalg.solve(Ad, rhs.double()), torch.float32, tol=2e-2)


@pytest.mark.parametrize("R", [1, 7, 40])
def test_cg_bsr_fused_multi_rhs(R):
    A, Ad = spd_bsr(256, 128)
    Bm = rand_vec((256, R), torch.float64, 127) * torch.logspace(-6, 6, R, dtype=torch.float64)
    X, it, rel = ts.cg_bsr_fused(A, D(Bm), tol=1e-12)
    assert_close(X, torch.linalg.solve(Ad, Bm), tol=1e-7 if not nki_backend() else 1e-4)
    assert tuple(H(it).shape) == (R,) and tuple(H(rel).shape) == (R,)
    for j in {0, R // 2, R - 1}:
        xj, _, _ = ts.cg_bsr_fused(A, D(Bm[:, j]), tol=1e-12)
        assert_close(H(X)[:, j], xj, tol=1e-10 if not nki_backend() else 1e-4)


def test_cg_bsr_fused_x0_zero_b_strided_and_prepared():
    A, Ad = spd_bsr(300, 128)
    rhs = rand_vec((300,), torch.float64, 128)
    xs = torch.linalg.solve(Ad, rhs)
    assert_close(ts.cg_bsr_fused(A, D(rhs), x0=D(xs), tol=1e-10)[0], xs, tol=1e-8)
    assert_close(
        ts.cg_bsr_fused(A, D(torch.zeros(300, dtype=torch.float64)))[0],
        torch.zeros(300, dtype=torch.float64),
    )
    bb = rand_vec((600,), torch.float64, 129)[::2]
    assert_close(ts.cg_bsr_fused(A, D(bb), tol=1e-12)[0], torch.linalg.solve(Ad, bb), tol=1e-8)
    BB = rand_vec((4, 300), torch.float64, 130).T
    assert_close(ts.cg_bsr_fused(A, D(BB), tol=1e-12)[0], torch.linalg.solve(Ad, BB), tol=1e-8)
    for pc in (None, "block_jacobi"):
        op = ts.prepare_cg_bsr(A, preconditioner=pc)
        assert isinstance(op, ts.FusedCGOperator)
        x1, _, _, info = ts.cg_bsr_fused(op, D(rhs), tol=1e-12, return_info=True)
        x2 = ts.cg_bsr_fused(op, D(rhs), tol=1e-12)[0]
        assert torch.equal(H(x1), H(x2))
        assert_close(x1, xs, tol=1e-8 if not nki_backend() else 1e-4)
        assert {"resident_blocks", "n_blocks", "trips", "path"} <= set(info)


def test_cg_bsr_fused_rejects_bad_kwargs():
    A, _ = spd_bsr(128, 128)
    rhs = torch.ones(128, dtype=torch.float64)
    for kw in (
        {"matvec": "diag"},
        {"precision": "fp8"},
        {"recurrence": "pipelined"},
        {"preconditioner": "ilu"},
    ):
        with pytest.raises(ValueError):
            ts.cg_bsr_fused(A, D(rhs), **kw)
    with pytest.raises(ValueError):
        ts.cg_bsr_fused(A, D(rhs), recurrence="chronopoulos_gear", preconditioner="block_jacobi")
    with pytest.raises(ValueError):
        ts.cg_bsr_fused(
            ts.BSRMatrix.from_dense(torch.eye(16, dtype=torch.float64), block_size=8),
            D(torch.ones(16)),
        )


def _cheb_residual_ref(lam, lmin, lmax, K):
    """Scaled Chebyshev residual polynomial T_K((d - lam)/c) / T_K(d/c)."""
    d, c = (lmax + lmin) / 2, (lmax - lmin) / 2
    T = np.polynomial.chebyshev.Chebyshev.basis(K)
    return torch.from_numpy(T(((d - lam) / c).numpy()) / T(d / c))


@pytest.mark.parametrize("K", [1, 2, 7])
@pytest.mark.parametrize("bounds", [(0.5, 4.0), (1e-2, 1.0)])
def test_chebyshev_coeffs(K, bounds):
    lmin, lmax = bounds
    alpha, beta = ts.chebyshev_coeffs(lmin, lmax, K)
    assert alpha.shape == (K,) and beta.shape == (K,)
    # A = diag(lam), b = 1: the residual after K steps is p_K(lam).
    lam = torch.linspace(lmin, lmax, 33, dtype=torch.float64)
    x_prev = x = torch.zeros_like(lam)
    for k in range(K):
        x, x_prev = x + float(alpha[k]) * (1 - lam * x) + float(beta[k]) * (x - x_prev), x
    # alpha/beta are float32 (the documented return dtype); their rounding
    # perturbs p_K by a few K * u32 relative to max |p_K| (1.05e-6 measured at
    # K = 7, kappa = 100), so the tolerance is float32-coefficient level.
    assert_close(1 - lam * x, _cheb_residual_ref(lam, lmin, lmax, K), tol=4e-6)


def test_chebyshev_bsr_fixed_k_residual():
    # Small fixed K (the smoother use): the residual must meet the Chebyshev bound.
    A, Ad = spd_bsr(64, 8, band=2)
    lmin, lmax = _ev_bounds(Ad)
    rhs = rand_vec((64,), torch.float64, 135)
    K = 6
    x, _, rel = ts.chebyshev_bsr(A, D(rhs), lmin, lmax, K=K)
    lam, V = torch.linalg.eigh(Ad)
    ref = V @ (_cheb_residual_ref(lam, lmin, lmax, K) * (V.T @ rhs))  # p_K(A) b
    assert_close(rhs - Ad @ H(x).double(), ref, tol=1e-5)


def test_chebyshev_coeffs_k1():
    assert_close(ts.chebyshev_coeffs(1.0, 3.0, 1)[0], torch.tensor([0.5]), check_dtype=False)


def _ev_bounds(Ad):
    ev = torch.linalg.eigvalsh(Ad)
    return ev.min().item(), ev.max().item()


@pytest.mark.parametrize("b", [8, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("use_x0", [False, True])
def test_chebyshev_richardson(b, dtype, use_x0):
    needs_pytorch_block(b)
    A, Ad = spd_bsr(256, b, dtype)
    rhs = rand_vec((256,), dtype, 131)
    x0 = D(torch.ones(256, dtype=dtype)) if use_x0 else None
    lmin, lmax = _ev_bounds(Ad)
    xs = torch.linalg.solve(Ad, rhs.double())
    x, k, rel = ts.chebyshev_bsr(A, D(rhs), lmin, lmax, K=200, x0=x0)
    assert k == 200
    true_rel = (
        torch.linalg.norm(rhs.double() - Ad @ H(x).double()) / torch.linalg.norm(rhs.double())
    ).item()
    assert rel == pytest.approx(true_rel, rel=1e-3, abs=1e-6)
    assert_close(x, xs, dtype, tol=1e-4)
    x, k, rel = ts.richardson_bsr(A, D(rhs), 2 / (lmin + lmax), K=400, x0=x0)
    assert k == 400
    assert_close(x, xs, dtype, tol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("fused", [False, True])
def test_power_iteration(dtype, fused):
    _, Ad = spd_bsr(256, 128)
    u = rand_vec((256,), torch.float64, 132)
    u = u / u.norm()
    Agap = Ad + 20.0 * torch.outer(u, u)
    A = ts.BSRMatrix.from_dense(Agap.to(dtype), block_size=128)
    evals, evecs = torch.linalg.eigh(Agap)
    v0 = D(rand_vec((256,), dtype, 133))
    fn = ts.power_iteration_bsr_fused if fused else ts.power_iteration_bsr
    lam, v, it = fn(A, v0=v0, max_iter=5000, tol=1e-12)[:3]
    assert abs(lam - evals[-1].item()) / evals[-1].item() < (
        1e-4 if dtype == torch.float32 else 1e-8
    )
    cos = (H(v).double() @ evecs[:, -1]).abs() / H(v).double().norm()
    assert cos > 1 - 1e-4


def test_power_iteration_seeded_default_v0():
    A, _ = spd_bsr(256, 128, torch.float32)
    for fn in (ts.power_iteration_bsr, ts.power_iteration_bsr_fused):
        torch.manual_seed(7)
        a = H(fn(A, max_iter=20)[1])
        torch.manual_seed(7)
        assert torch.equal(a, H(fn(A, max_iter=20)[1]))


def test_jacobi_preconditioner():
    A, Ad = spd_bsr(129, 8)
    r = rand_vec((129,), torch.float64, 134)
    assert_close(ts.jacobi_preconditioner_bsr(A)(D(r)), r / torch.diagonal(Ad))
    Z = ts.BSRMatrix.from_dense(torch.zeros(8, 8, dtype=torch.float64), block_size=4)
    with pytest.raises(ValueError):
        ts.jacobi_preconditioner_bsr(Z)


# ---------------------------------------------------------------- autograd


def _host_only():
    if on_device():
        pytest.skip("fp64 gradcheck runs on the host only")


def _csr_parts(m=9, n=7, seed=141):
    Ad = rand_dense(m, n, 0.5, torch.float64, seed)
    A = ts.from_dense(Ad)
    rows = torch.repeat_interleave(torch.arange(m), A.row_ptrs[1:] - A.row_ptrs[:-1])

    def mk(vals):
        return ts.CSRMatrix(vals, A.col_indices, A.row_ptrs, A.shape)

    def dense_of(vals):
        return torch.zeros(m, n, dtype=vals.dtype).index_put(
            (rows, A.col_indices), vals, accumulate=True
        )

    return A, mk, dense_of


def _grad_vs_dense(fn_ts, fn_dense, inputs, tol=1e-9):
    ins = [t.detach().clone().requires_grad_() for t in inputs]
    out = fn_ts(*ins)
    w = rand_vec(tuple(out.shape), torch.float64, 199).to(out.dtype)
    (out * D(w)).sum().backward()
    ins2 = [t.detach().clone().requires_grad_() for t in inputs]
    (fn_dense(*ins2) * w).sum().backward()
    for i, (a, b2) in enumerate(zip(ins, ins2, strict=True)):
        assert a.grad is not None, f"input {i} got no gradient"
        assert_close(a.grad, b2.grad, tol=tol if not nki_backend() else FP32_TOL, what=f"grad[{i}]")


def test_autograd_csr_ops():
    _host_only()
    A, mk, dense_of = _csr_parts()
    v = A.values.clone().requires_grad_()
    B = rand_vec((7, 3), torch.float64, 142).requires_grad_()
    x = rand_vec((7,), torch.float64, 143).requires_grad_()
    y = torch.ones(9, dtype=torch.float64)
    assert gradcheck(lambda vals, Bm: ts.spmm(mk(vals), Bm), (v, B))
    assert gradcheck(lambda vals, xx: ts.spmv(mk(vals), xx), (v, x))
    assert gradcheck(lambda vals, xx: ts.spmv(mk(vals), xx, alpha=-1.5, y=y, beta=2.0), (v, x))
    assert gradcheck(
        lambda vals, Bm: ts.spmm(
            mk(vals), Bm, alpha=0.5, C=torch.ones(9, 3, dtype=torch.float64), beta=-1.0
        ),
        (v, B),
    )
    _grad_vs_dense(
        lambda vals, Bm: ts.spmm(mk(vals), Bm), lambda vals, Bm: dense_of(vals) @ Bm, [v, B]
    )
    _grad_vs_dense(
        lambda vals, xx: ts.spmv(mk(vals), xx), lambda vals, xx: dense_of(vals) @ xx, [v, x]
    )
    for fn_ts, fn_d in (
        (lambda vals: ts.sparse_scale(mk(vals), 2.5).to_dense(), lambda vals: 2.5 * dense_of(vals)),
        (lambda vals: ts.sparse_transpose(mk(vals)).to_dense(), lambda vals: dense_of(vals).T),
        (
            lambda vals: ts.sparse_add(mk(vals), A, 2.0, 1.0).to_dense(),
            lambda vals: 2.0 * dense_of(vals) + A.to_dense(),
        ),
    ):
        _grad_vs_dense(fn_ts, fn_d, [v])
    Ad = A.to_dense()
    _grad_vs_dense(lambda dn: ts.from_dense(dn).to_dense(), lambda dn: dn * (Ad != 0), [Ad])
    _grad_vs_dense(
        lambda dn: ts.BSRMatrix.from_dense(dn, block_size=4).to_dense(), lambda dn: dn * 1.0, [Ad]
    )


@pytest.mark.parametrize("uplo", ["upper", "lower"])
def test_autograd_spmv_symmetric(uplo):
    _host_only()
    Sd = rand_dense(7, 7, 0.6, torch.float64, 144)
    Sd = torch.triu(Sd) if uplo == "upper" else torch.tril(Sd)
    S = ts.from_dense(Sd)
    rows = torch.repeat_interleave(torch.arange(7), S.row_ptrs[1:] - S.row_ptrs[:-1])
    x = rand_vec((7,), torch.float64, 145).requires_grad_()
    sv = S.values.clone().requires_grad_()

    def f(vals, xx):
        return ts.spmv_symmetric(
            ts.CSRMatrix(vals, S.col_indices, S.row_ptrs, S.shape), xx, uplo=uplo
        )

    def fd(vals, xx):
        T = torch.zeros(7, 7, dtype=torch.float64).index_put(
            (rows, S.col_indices), vals, accumulate=True
        )
        return sym_ref(T, uplo) @ xx

    assert gradcheck(f, (sv, x))
    _grad_vs_dense(f, fd, [sv, x])


@pytest.mark.parametrize("b,m,n", [(4, 12, 8), (8, 24, 16)])
def test_autograd_bsr_spmm(b, m, n):
    _host_only()
    needs_pytorch_block(b)
    Ab = ts.BSRMatrix.from_dense(rand_dense(m, n, 0.2, torch.float64, 146), block_size=b)
    blk = Ab.blocks.clone().requires_grad_()
    B = rand_vec((n, 3), torch.float64, 147).requires_grad_()

    def f(bl, Bm):
        return ts.bsr_spmm(
            ts.BSRMatrix(bl, Ab.block_col_indices, Ab.block_row_ptrs, Ab.shape, b), Bm
        )

    def fd(bl, Bm):
        return bsr_ref_dense_diff(bl, Ab) @ Bm

    assert gradcheck(f, (blk, B))
    _grad_vs_dense(f, fd, [blk, B])


def bsr_ref_dense_diff(bl, Ab):
    m, n = Ab.shape
    b = Ab.block_size
    out = torch.zeros(-(-m // b) * b, -(-n // b) * b, dtype=bl.dtype)
    rp = Ab.block_row_ptrs.tolist()
    ci = Ab.block_col_indices.tolist()
    for i in range(len(rp) - 1):
        for k in range(rp[i], rp[i + 1]):
            idx = (
                torch.arange(i * b, (i + 1) * b)[:, None],
                torch.arange(ci[k] * b, (ci[k] + 1) * b)[None, :],
            )
            out = out.index_put(idx, bl[k], accumulate=True)
    return out[:m, :n]


def test_autograd_screened_spmm():
    _host_only()
    A = rand_vec((16, 16), torch.float64, 148).requires_grad_()
    dg = rand_vec((16,), torch.float64, 149).abs()
    B = rand_vec((16, 5), torch.float64, 150).requires_grad_()
    m = _schwarz_mask(dg, 0.3).to(torch.float64)
    assert gradcheck(lambda a, bm: ts.screened_spmm(a, dg, bm, 0.3), (A, B))
    _grad_vs_dense(
        lambda a, bm: ts.screened_spmm(a, dg, bm, 0.3), lambda a, bm: (a * m) @ bm, [A, B]
    )


@pytest.mark.parametrize("b,nt,hd,pat", [ATTN_CASES[0], ATTN_CASES[1]], ids=["b4_3x5", "b4_2x3"])
@pytest.mark.parametrize("scale", [None, 0.7])
def test_autograd_attention(b, nt, hd, pat, scale):
    _host_only()
    needs_pytorch_block(b)
    msk = mask_bsr(nt, b, pat, dtype=torch.float64)
    ins = tuple(rand_vec((nt * b, hd), torch.float64, s).requires_grad_() for s in (151, 152, 153))
    sc = hd**-0.5 if scale is None else scale

    def f(q, k, v):
        return ts.block_sparse_attention_tiled(q, k, v, msk, scale=scale)

    assert gradcheck(f, ins)
    assert gradgradcheck(f, ins)
    _grad_vs_dense(f, lambda q, k, v: attn_dense(q, k, v, nt, b, pat, sc), list(ins))


def test_autograd_attention_b128_grad():
    # b=128 is the shape the NKI kernels take: first-order gradients vs dense.
    pat = [(0, 0), (1, 0), (1, 1)]
    msk = mask_bsr(2, 128, pat, dtype=torch.float64)
    ins = [rand_vec((256, 32), torch.float64, s) for s in (154, 155, 156)]
    _grad_vs_dense(
        lambda q, k, v: ts.block_sparse_attention_tiled(D(q), D(k), D(v), msk),
        lambda q, k, v: attn_dense(q, k, v, 2, 128, pat, 32**-0.5),
        ins,
        tol=1e-9,
    )


def test_autograd_fixed_coefficient_iterations():
    # chebyshev / richardson use coefficients fixed in advance: their gradient is
    # the exact derivative of the returned iterate, which converges to A^{-1} g.
    A, Ad = spd_bsr(64, 8, band=2)
    lmin, lmax = _ev_bounds(Ad)
    g = rand_vec((64,), torch.float64, 157)
    for fn in (
        lambda bb: ts.chebyshev_bsr(A, bb, lmin, lmax, K=300)[0],
        lambda bb: ts.richardson_bsr(A, bb, 2 / (lmin + lmax), K=2000)[0],
    ):
        bb = D(rand_vec((64,), torch.float64, 158)).requires_grad_()
        (fn(bb) * D(g)).sum().backward()
        assert_close(bb.grad, torch.linalg.solve(Ad, g), tol=1e-5)


def test_autograd_cg_bsr_generic_b():
    A, Ad = spd_bsr(64, 8, band=2)
    bb = D(rand_vec((64,), torch.float64, 159)).requires_grad_()
    x, _, _ = ts.cg_bsr(A, bb, tol=1e-12, max_iter=500)
    g = rand_vec((64,), torch.float64, 160)
    if x.requires_grad:
        (x * D(g)).sum().backward()
        assert_close(bb.grad, torch.linalg.solve(Ad, g), tol=1e-6)


_CG_GRAD_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="trnsci/trnsparse#70 needs decision: cg_bsr gradients use frozen (.item()) step sizes",
)


@_CG_GRAD_XFAIL
def test_autograd_cg_bsr_wrt_b_eigenvector():
    # b along an eigenvector: CG converges in one step, so the frozen-coefficient
    # graph is far from A^{-1}. Correct (implicit) gradients, or no graph at all, pass.
    A, Ad = spd_bsr(64, 8, band=2)
    bb = D(torch.linalg.eigh(Ad)[1][:, -1].clone()).requires_grad_()
    x, _, _ = ts.cg_bsr(A, bb, tol=1e-12, max_iter=500)
    if x.requires_grad:
        g = rand_vec((64,), torch.float64, 161)
        (x * D(g)).sum().backward()
        assert_close(bb.grad, torch.linalg.solve(Ad, g), tol=1e-6)


@_CG_GRAD_XFAIL
def test_autograd_cg_bsr_wrt_blocks():
    A, Ad = spd_bsr(64, 8, band=2)
    rhs = torch.linalg.eigh(Ad)[1][:, -1].clone()
    blk = A.blocks.clone().requires_grad_()
    A2 = ts.BSRMatrix(blk, A.block_col_indices, A.block_row_ptrs, A.shape, A.block_size)
    x, _, _ = ts.cg_bsr(A2, D(rhs), tol=1e-12, max_iter=500)
    if x.requires_grad:
        g = rand_vec((64,), torch.float64, 162)
        (x * D(g)).sum().backward()
        lam = torch.linalg.solve(Ad, g)
        Gd = -torch.outer(lam, torch.linalg.solve(Ad, rhs))
        b = A.block_size
        rp, ci = A.block_row_ptrs.tolist(), A.block_col_indices.tolist()
        ref = torch.stack(
            [
                Gd[i * b : (i + 1) * b, ci[k] * b : (ci[k] + 1) * b]
                for i in range(len(rp) - 1)
                for k in range(rp[i], rp[i + 1])
            ]
        )
        assert_close(blk.grad, ref, tol=1e-6)


# ---------------------------------------------------------------- mutation / aliasing


def _storages(o):
    if isinstance(o, ts.CSRMatrix):
        return [o.values, o.col_indices, o.row_ptrs]
    if isinstance(o, ts.COOMatrix):
        return [o.values, o.row_indices, o.col_indices]
    if isinstance(o, ts.BSRMatrix):
        return [o.blocks]
    if isinstance(o, (tuple, list)):
        return [t for x in o for t in _storages(x)]
    return [o] if isinstance(o, torch.Tensor) else []


def _check_unchanged(make_inputs, call):
    ins = make_inputs()
    snap = [t.detach().clone() for t in ins]
    out = call(*ins)
    for i, (a, s) in enumerate(zip(ins, snap, strict=True)):
        assert torch.equal(H(a), H(s)), f"input {i} mutated"
    for t in _storages(out):
        for i, a in enumerate(ins):
            if t.numel() and a.numel():
                assert t.untyped_storage().data_ptr() != a.untyped_storage().data_ptr(), (
                    f"output shares storage with input {i}"
                )


_Ad = rand_dense(12, 10, 0.4, torch.float32, 171)
_A = ts.from_dense(_Ad)
_A1 = rand_dense(128, 128, 0.2, torch.float32, 172)
_Ab1 = ts.BSRMatrix.from_dense(_A1, block_size=128)
_Asp, _Adsp = spd_bsr(256, 128, torch.float32)


def _parts():
    return [_A.values.clone(), _A.col_indices.clone(), _A.row_ptrs.clone()]


def _mk(v, c, r):
    return ts.CSRMatrix(v, c, r, _A.shape)


def _bsr1(bl):
    return ts.BSRMatrix(bl, _Ab1.block_col_indices, _Ab1.block_row_ptrs, _Ab1.shape, 128)


_x = rand_vec((10,), torch.float32, 173)
_y = rand_vec((12,), torch.float32, 174)
_B = rand_vec((10, 4), torch.float32, 175)
_C = rand_vec((12, 4), torch.float32, 176)
_bsp = rand_vec((256,), torch.float32, 177)

ALIAS_CASES = {
    "spmv(y,beta)": (
        lambda: _parts() + [D(_x.clone()), D(_y.clone())],
        lambda v, c, r, xx, yy: ts.spmv(_mk(v, c, r), xx, 2.0, yy, 0.5),
    ),
    "spmv": (lambda: _parts() + [D(_x.clone())], lambda v, c, r, xx: ts.spmv(_mk(v, c, r), xx)),
    "spmm(C,beta)": (
        lambda: _parts() + [D(_B.clone()), D(_C.clone())],
        lambda v, c, r, bb, cc: ts.spmm(_mk(v, c, r), bb, 2.0, cc, 0.5),
    ),
    "sparse_scale(1)": (_parts, lambda v, c, r: ts.sparse_scale(_mk(v, c, r), 1.0)),
    "sparse_transpose": (_parts, lambda v, c, r: ts.sparse_transpose(_mk(v, c, r))),
    "sparse_add": (_parts, lambda v, c, r: ts.sparse_add(_mk(v, c, r), _mk(v, c, r))),
    "to_coo": (_parts, lambda v, c, r: _mk(v, c, r).to_coo()),
    "to_dense": (_parts, lambda v, c, r: _mk(v, c, r).to_dense()),
    "from_dense": (lambda: [_Ad.clone()], lambda a: ts.from_dense(a)),
    "bsr_from_dense": (lambda: [_A1.clone()], lambda a: ts.BSRMatrix.from_dense(a, block_size=128)),
    "bsr_to_dense": (lambda: [_Ab1.blocks.clone()], lambda bl: _bsr1(bl).to_dense()),
    "bsr_to_csr": (lambda: [_Ab1.blocks.clone()], lambda bl: _bsr1(bl).to_csr()),
    "bsr_spmm": (
        lambda: [D(_Ab1.blocks.clone()), D(rand_vec((128, 8), torch.float32, 178))],
        lambda bl, bb: ts.bsr_spmm(_bsr1(bl), bb),
    ),
    "bsr_diagonal": (lambda: [_Ab1.blocks.clone()], lambda bl: ts.bsr_diagonal(_bsr1(bl))),
    "screened_spmm": (
        lambda: [
            D(_A1.clone()),
            D(rand_vec((128,), torch.float32, 179).abs()),
            D(rand_vec((128, 4), torch.float32, 180)),
        ],
        lambda a, d, bb: ts.screened_spmm(a, d, bb, 0.1),
    ),
    "attention": (
        lambda: [D(rand_vec((256, 16), torch.float32, s)) for s in (181, 182, 183)],
        lambda q, k, v: ts.block_sparse_attention_tiled(
            q, k, v, mask_bsr(2, 128, [(0, 0), (1, 0), (1, 1)])
        ),
    ),
    "cg_bsr(x0)": (
        lambda: [D(_bsp.clone()), D(torch.ones(256))],
        lambda bb, x0: ts.cg_bsr(_Asp, bb, x0=x0, tol=1e-6)[0],
    ),
    "cg_bsr_fused(x0)": (
        lambda: [D(_bsp.clone()), D(torch.ones(256))],
        lambda bb, x0: ts.cg_bsr_fused(_Asp, bb, x0=x0, tol=1e-6)[0],
    ),
    "chebyshev(x0)": (
        lambda: [D(_bsp.clone()), D(torch.ones(256))],
        lambda bb, x0: ts.chebyshev_bsr(_Asp, bb, 1.0, 50.0, K=5, x0=x0)[0],
    ),
    "richardson(x0)": (
        lambda: [D(_bsp.clone()), D(torch.ones(256))],
        lambda bb, x0: ts.richardson_bsr(_Asp, bb, 0.01, K=5, x0=x0)[0],
    ),
    "power_iteration(v0)": (
        lambda: [D(_bsp.clone())],
        lambda v0: ts.power_iteration_bsr(_Asp, v0=v0, max_iter=5)[1],
    ),
    "power_iteration_fused(v0)": (
        lambda: [D(_bsp.clone())],
        lambda v0: ts.power_iteration_bsr_fused(_Asp, v0=v0, max_iter=5)[1],
    ),
}


@pytest.mark.parametrize("name", list(ALIAS_CASES))
def test_no_mutation_no_alias(name):
    make, call = ALIAS_CASES[name]
    _check_unchanged(make, call)


def test_outputs_independent_of_later_mutation():
    a = _Ad.clone()
    S = ts.from_dense(a)
    a.mul_(0)
    assert_close(S.to_dense(), _Ad)
    a = _A1.clone()
    Sb = ts.BSRMatrix.from_dense(a, block_size=128)
    a.mul_(0)
    assert_close(Sb.to_dense(), _A1)
    S = ts.from_dense(_Ad.clone())
    ts.sparse_scale(S, 1.0).values.mul_(0)
    ts.sparse_transpose(S).values.mul_(0)
    assert_close(S.to_dense(), _Ad)


# ---------------------------------------------------------------- determinism


def test_determinism():
    Ad = rand_dense(300, 300, 0.05, torch.float32, 191)
    A = ts.from_dense(Ad)
    Ab = ts.BSRMatrix.from_dense(Ad, block_size=128)
    B = D(rand_vec((300, 64), torch.float32, 192))
    Q = D(rand_vec((256, 32), torch.float32, 193))
    msk = mask_bsr(2, 128, [(0, 0), (1, 0), (1, 1)])
    bb = D(rand_vec((256,), torch.float32, 194))
    for f in (
        lambda: ts.spmm(A, B),
        lambda: ts.spmv(A, B[:, 0]),
        lambda: ts.bsr_spmm(Ab, B),
        lambda: ts.block_sparse_attention_tiled(Q, Q, Q, msk),
        lambda: ts.cg_bsr_fused(_Asp, bb, tol=1e-6)[0],
        lambda: ts.cg_bsr(_Asp, bb, tol=1e-6)[0],
    ):
        assert torch.equal(H(f()), H(f()))
    # BSR and CSR products of the same matrix agree; column split does not change bsr_spmm.
    assert_close(ts.bsr_spmm(Ab, B), H(ts.spmm(A, B)).double(), torch.float32, tol=1e-5)
    full = H(ts.bsr_spmm(Ab, B))
    part = torch.cat(
        [H(ts.bsr_spmm(Ab, B[:, :32].contiguous())), H(ts.bsr_spmm(Ab, B[:, 32:].contiguous()))], 1
    )
    assert_close(full, part.double(), torch.float32, tol=1e-6)


def test_public_api_covered():
    # Guard: a new public name must get a conformance case here.
    covered = {
        "CSRMatrix", "COOMatrix", "BSRMatrix", "from_dense", "from_scipy", "eye_sparse", "spmv", "spmm",
        "spmv_symmetric", "bsr_spmm", "screened_spmm", "block_sparse_attention_tiled", "sparse_add",
        "sparse_scale", "sparse_transpose", "nnz_per_row", "schwarz_bounds", "screen_quartets",
        "density_screen", "sparsity_stats", "cg_bsr", "cg_bsr_fused", "prepare_cg_bsr", "FusedCGOperator",
        "chebyshev_bsr", "chebyshev_coeffs", "richardson_bsr", "power_iteration_bsr",
        "power_iteration_bsr_fused", "jacobi_preconditioner_bsr", "bsr_diagonal", "HAS_NKI", "set_backend",
        "get_backend",
    }  # fmt: skip
    assert set(ts.__all__) == covered
