"""CPU tests for the fused kernel-matrix product K(X, Y) @ V (#93).

The PyTorch path runs the same augmented-GEMM algorithm as the NKI kernel;
the reference is dense float64 `torch.cdist` -> kernel -> matmul. The NKI
kernel itself is checked under `nki.simulate` and the MLIR verifier in
tests/test_nki_kernel_matmul.py, and on hardware in the neuron-marked tests.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import trnsparse
from trnsparse import KERNELS, KernelOperator, kernel_matmul, kernel_matmul_reference
from trnsparse.nki import kernels as K


def _rel(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-300))


@pytest.fixture(autouse=True)
def _pytorch_backend():
    old = trnsparse.get_backend()
    trnsparse.set_backend("pytorch")
    yield
    trnsparse.set_backend(old)


# Matern-1/2 has a cusp at r = 0, so the fp32 expansion's error at
# near-coincident points enters as sqrt(u |x|^2) / l (see kernel_matmul's
# docstring); the smooth kernels see it at first order only.
_TOL = {"rbf": 2e-5, "matern12": 3e-3, "matern32": 2e-5, "matern52": 2e-5}


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "n,m,d,R",
    [
        (1, 1, 3, 1),
        (127, 129, 3, 7),
        (129, 127, 1, 64),
        (1000, 4097, 3, 1),
        (300, 200, 16, 513),
        (64, 90, 126, 2),
    ],
)
def test_cross_matches_fp64_reference(kernel, n, m, d, R):
    g = torch.Generator().manual_seed(n * 7 + m)
    X = torch.rand(n, d, generator=g)
    Y = torch.rand(m, d, generator=g)
    V = torch.randn(m, R, generator=g)
    ell = 0.5 * d**0.5
    out = kernel_matmul(X, Y, V, kernel, lengthscale=ell, outputscale=1.7)
    ref = kernel_matmul_reference(X, Y, V, kernel, lengthscale=ell, outputscale=1.7)
    assert out.shape == (n, R) and out.dtype == torch.float32
    assert _rel(out, ref) < _TOL[kernel]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("n", [1, 127, 129, 1000])
def test_square_operator_with_noise(kernel, n):
    g = torch.Generator().manual_seed(n)
    X = torch.rand(n, 3, generator=g)
    V = torch.randn(n, 5, generator=g)
    out = kernel_matmul(X, None, V, kernel, 0.3, 1.3, noise=1e-2)
    ref = kernel_matmul_reference(X, None, V, kernel, 0.3, 1.3, noise=1e-2)
    assert _rel(out, ref) < _TOL[kernel]


def test_vector_rhs_and_ard_lengthscale():
    g = torch.Generator().manual_seed(1)
    X = torch.rand(200, 3, generator=g)
    v = torch.randn(200, generator=g)
    ls = torch.tensor([0.2, 0.5, 1.0])
    out = kernel_matmul(X, None, v, "rbf", ls, 2.0, noise=1e-3)
    ref = kernel_matmul_reference(X, None, v, "rbf", ls, 2.0, noise=1e-3)
    assert out.shape == (200,)
    assert _rel(out, ref) < 2e-5
    # ARD with equal entries is the scalar lengthscale.
    same = kernel_matmul(X, None, v, "rbf", torch.full((3,), 0.4), 2.0)
    assert _rel(same, kernel_matmul_reference(X, None, v, "rbf", 0.4, 2.0)) < 2e-5


def test_reference_matches_dense_formula():
    g = torch.Generator().manual_seed(2)
    X = torch.rand(50, 2, generator=g, dtype=torch.float64)
    Y = torch.rand(40, 2, generator=g, dtype=torch.float64)
    V = torch.randn(40, 3, generator=g, dtype=torch.float64)
    r = torch.cdist(X, Y) / 0.7
    s3, s5 = 3**0.5 * r, 5**0.5 * r
    dense = {
        "rbf": torch.exp(-0.5 * r**2),
        "matern12": torch.exp(-r),
        "matern32": (1 + s3) * torch.exp(-s3),
        "matern52": (1 + s5 + s5**2 / 3) * torch.exp(-s5),
    }
    for kernel, Kd in dense.items():
        ref = kernel_matmul_reference(X, Y, V, kernel, 0.7, 1.5)
        torch.testing.assert_close(ref, 1.5 * Kd @ V, rtol=1e-12, atol=1e-12)


def test_float64_auto_stays_float64():
    trnsparse.set_backend("auto")
    g = torch.Generator().manual_seed(3)
    X = torch.rand(300, 3, generator=g, dtype=torch.float64)
    V = torch.randn(300, 2, generator=g, dtype=torch.float64)
    out = kernel_matmul(X, None, V, "rbf", 0.3, 1.0, noise=1e-4)
    assert out.dtype == torch.float64
    ref = kernel_matmul_reference(X, None, V, "rbf", 0.3, 1.0, noise=1e-4)
    assert _rel(out, ref) < 1e-12


def test_operator_matches_function_and_cross():
    g = torch.Generator().manual_seed(4)
    X = torch.rand(500, 3, generator=g) + 10.0  # far from the origin: centring matters
    V = torch.randn(500, 4, generator=g)
    op = KernelOperator(X, "matern52", 0.25, 1.1, noise=1e-3)
    ref = kernel_matmul_reference(X, None, V, "matern52", 0.25, 1.1, noise=1e-3)
    assert _rel(op(V), ref) < 2e-5
    Xs = torch.rand(77, 3, generator=g) + 10.0
    ref_c = kernel_matmul_reference(Xs, X, V, "matern52", 0.25, 1.1)
    assert _rel(op.cross(Xs, V), ref_c) < 2e-5
    rows = op.rows([0, 5, 499])
    dense = kernel_matmul_reference(
        X, None, torch.eye(500, dtype=torch.float64), "matern52", 0.25, 1.1
    )
    torch.testing.assert_close(rows, dense[[0, 5, 499]], rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(op.diag(), torch.full((500,), 1.1, dtype=torch.float64))


def test_validation():
    X = torch.rand(10, 3)
    with pytest.raises(ValueError):
        kernel_matmul(X, None, torch.rand(9, 1))
    with pytest.raises(ValueError):
        kernel_matmul(X, X, torch.rand(10, 1), noise=1.0)
    with pytest.raises(ValueError):
        kernel_matmul(X, None, torch.rand(10, 1), kernel="cauchy")
    with pytest.raises(ValueError):
        kernel_matmul(X, None, torch.rand(10, 1), lengthscale=0.0)
    with pytest.raises(NotImplementedError):
        kernel_matmul(X, None, torch.rand(10, 1), tile_mask=object())


def test_nki_backend_without_device_falls_back(monkeypatch):
    """Explicit nki with no device: the launch fails and the torch path answers (not REQUIRE_NKI)."""
    from trnsparse.nki import dispatch

    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)

    def boom(*a, **k):
        raise RuntimeError("no device")

    monkeypatch.setattr(dispatch, "nki_kernel_matmul", boom)
    X = torch.rand(40, 3)
    V = torch.rand(40, 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = kernel_matmul(X, None, V, "rbf", 0.5)
    assert any("NKI path failed" in str(x.message) for x in w)
    assert _rel(out, kernel_matmul_reference(X, None, V, "rbf", 0.5)) < 2e-5


def test_float64_policy_on_nki(monkeypatch):
    from trnsparse.nki import dispatch

    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    seen = {}

    def fake(x, y, v, params, kind, group):
        seen["dtype"] = x.dtype
        return torch.zeros(x.shape[0], v.shape[1])

    monkeypatch.setattr(dispatch, "nki_kernel_matmul", fake)
    X = torch.rand(20, 3, dtype=torch.float64)
    V = torch.rand(20, 1, dtype=torch.float64)
    monkeypatch.setattr(dispatch, "get_backend", lambda: "nki")
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
    with pytest.warns(UserWarning, match="float32"):
        out = kernel_matmul(X, None, V)
    assert seen["dtype"] == torch.float32 and out.dtype == torch.float64
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    with pytest.raises(TypeError, match="REQUIRE_NKI"):
        kernel_matmul(X, None, V)
    monkeypatch.setattr(dispatch, "get_backend", lambda: "auto")
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
    seen.clear()
    kernel_matmul(X, None, V)  # auto: float64 stays on torch
    assert not seen


def test_params_are_tensor_arguments(monkeypatch):
    """Lengthscale, outputscale and noise reach the kernel as data, not as static arguments."""
    from trnsparse.nki import dispatch

    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    calls = []
    monkeypatch.setattr(
        dispatch,
        "nki_kernel_matmul",
        lambda x, y, v, p, kind, group: (
            calls.append((p.clone(), kind, group)) or torch.zeros(x.shape[0], v.shape[1])
        ),
    )
    X = torch.rand(30, 3)
    V = torch.rand(30, 1)
    kernel_matmul(X, None, V, "rbf", 0.5, 2.0, noise=0.1)
    kernel_matmul(X, None, V, "rbf", 0.25, 3.0, noise=0.2)
    (p1, k1, g1), (p2, k2, g2) = calls
    assert (k1, g1) == (k2, g2)  # same static arguments: one compile
    assert p1.shape == (128, 4)
    torch.testing.assert_close(
        p1[0], torch.tensor([-0.5 / 0.25, torch.log(torch.tensor(2.0)), 0.1, 0.0])
    )
    assert not torch.equal(p1, p2)


# ── Work split across physical cores (pure Python, shared with the kernel) ──


@pytest.mark.parametrize("n", [1, 2, 127, 128, 129, 511, 512, 513, 1000, 1536, 4097, 100_003])
@pytest.mark.parametrize("R", [1, 64, 512, 513, 1500])
@pytest.mark.parametrize("n_prog", [1, 2])
def test_kmv_split_covers_rows_with_equal_trip_counts(n, R, n_prog):
    NI, groups = K.kmv_plan(n, R, n_prog)
    assert NI in (128, 256, 512) and NI % 128 == 0
    assert [c for g in groups for c in g] == [(r0, min(128, R - r0)) for r0 in range(0, R, 128)]
    # One PSUM bank per accumulator, at least two banks left for distance tiles.
    assert all(1 <= len(g) <= K._KMV_PSUM_BANKS - 2 for g in groups)
    assert len(groups) == (1 if R <= 768 else -(-R // 768))
    owner = [None] * n
    trips = set()
    for pid in range(n_prog):
        lo, hi, t0, tn = K.kmv_split(n, NI, n_prog, pid)
        trips.add(hi - lo)
        assert 0 <= tn <= NI
        assert (tn == 0) == (n % (NI * n_prog) == 0)
        rows = [r for b in range(lo, hi) for r in range(b * NI, (b + 1) * NI)] + list(
            range(t0, t0 + tn)
        )
        for r in rows:
            assert owner[r] in (None, pid) or r >= n - 128  # only the last chunk may be recomputed
            owner[r] = pid if owner[r] is None else owner[r]
    assert len(trips) == 1  # same rolled-loop trip count on every core
    assert all(o is not None for o in owner)


# ── Y tiles per loop body (#103) ──


def test_kmv_group_rule(monkeypatch):
    from trnsparse.kernel_matmul import _kmv_group

    # Small m: the fixed 4-tile body of earlier releases (same summation, same bits).
    for m in (1, 128, 1000, 3199):
        assert K.kmv_group(m) == 4
    # About sqrt(m / 128) Y tiles in between, capped at 16.
    assert K.kmv_group(3200) == 5
    assert K.kmv_group(8192) == 8
    assert [K.kmv_group(m) for m in (32768, 131072, 1 << 20)] == [16, 16, 16]
    assert all(K.kmv_group(m) <= K.kmv_group(m + 128) for m in range(0, 40000, 128))
    monkeypatch.delenv("TRNSPARSE_KMV_GROUP", raising=False)
    assert _kmv_group(131072) == 16
    monkeypatch.setenv("TRNSPARSE_KMV_GROUP", "7")
    assert _kmv_group(131072) == 7
    monkeypatch.setenv("TRNSPARSE_KMV_GROUP", "100")
    assert _kmv_group(10) == K._KMV_MAX_GROUP


def test_group_follows_m_not_the_row_block(monkeypatch):
    """Square, row-blocked and cross products of one Y share a body size (same bits)."""
    from trnsparse.nki import dispatch

    monkeypatch.delenv("TRNSPARSE_KMV_GROUP", raising=False)
    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    groups = []
    monkeypatch.setattr(
        dispatch,
        "nki_kernel_matmul",
        lambda x, y, v, p, kind, group: groups.append(group) or torch.zeros(x.shape[0], v.shape[1]),
    )
    X = torch.rand(9000, 3)
    op = KernelOperator(X, "rbf", 0.5)
    op(torch.rand(9000, 2))
    op.cross(torch.rand(10, 3), torch.rand(9000, 2))
    assert groups == [K.kmv_group(9000)] * 2 == [8, 8]


# ── Row-blocked launches (#100): no launch reaches the 30 s runtime timeout ──


def test_kmv_block_rows_rule():
    W = K._KMV_BLOCK_WORK
    # Under the budget: one launch, blocking is a no-op.
    assert K.kmv_block_rows(1000, 1000, W) == 1000
    # n = 1M, d = 3, R = 33 (the Trn2 timeout of #100): 2048-row blocks, 512 launches.
    # The budget counts kernel entries only (the rate is flat in R, #100).
    assert W == 6 * 10**10
    # n = m = 2^20 (the Trn2 timeout of #100): 19 launches of 55296 rows.
    n = 1 << 20
    rows = K.kmv_block_rows(n, n, W)
    assert rows == 55296 and rows % K._KMV_BLOCK_QUANTUM == 0 and rows * n <= W
    assert -(-n // rows) == 19 == -(-n // (W // n // 1024 * 1024))  # fewest launches
    # n = m = 524288: 5 launches of 105472 rows (one was 10.1 s at LNC=2).
    assert K.kmv_block_rows(524_288, 524_288, W) == 105_472
    # Just under / over the budget: two blocks of 100352.
    assert K.kmv_block_rows(200_000, 300_000, W) == 200_000
    assert K.kmv_block_rows(200_001, 300_000, W) == 100_352
    for n, m in [
        (1 << 20, 1 << 20),
        (10**6, 10**7),
        (10**7, 10**6),
        (10**4, 10**9),
        (3 << 20, 77_777),
    ]:
        rows = K.kmv_block_rows(n, m, W)
        assert rows % 128 == 0 and 128 <= rows < n
        assert rows * m <= W or rows == 128
        nblk = -(-n // rows)
        cap = max(
            128, W // m // (1024 if W // m >= 1024 else 128) * (1024 if W // m >= 1024 else 128)
        )
        assert nblk == -(-n // cap)  # as few launches as the budget allows
        assert (
            nblk * rows - n < (1024 if rows >= 1024 else 128) * nblk
        )  # padding < one quantum per block
    # Tiny budgets round to 128-row tiles, never below one tile.
    assert K.kmv_block_rows(1000, 50, 50 * 300) == 256
    assert K.kmv_block_rows(1000, 50, 1) == 128
    assert K.kmv_block_rows(100, 50, 1) == 100


def _fake_launch(calls):
    """Stand-in for one kernel launch: row-local float64 math, one row at a time.

    Every output row is computed by the same operations whatever block it is
    in, so blocked == unblocked bitwise exactly when the host-side blocking
    (slicing, padding, the noise rows, stitching) is right.
    """

    def launch(x, y, v, vn, params, kind, group, sim):
        calls.append(
            (
                tuple(x.shape),
                None if y is None else tuple(y.shape),
                vn is not None,
                (None if y is None else y.data_ptr(), v.data_ptr(), params.data_ptr()),
            )
        )
        Y = x if y is None else y
        a, noise = float(params[0, 0]), float(params[0, 2])
        Y64, V64 = Y.double(), v.double()
        out = torch.empty(x.shape[0], v.shape[1], dtype=torch.float64)
        for i in range(x.shape[0]):
            D = ((x[i].double() - Y64) ** 2).sum(1)
            out[i] = torch.exp(a * D) @ V64
        if y is None:
            out += noise * V64
        elif vn is not None:
            out += noise * vn.double()
        return out.float()

    return launch


@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
@pytest.mark.parametrize("n", [1, 128, 300, 513, 1000])
def test_row_blocked_launch_matches_one_launch(monkeypatch, self_op, n):
    from trnsparse.kernel_matmul import _params
    from trnsparse.nki import dispatch as D

    g = torch.Generator().manual_seed(n)
    m = n if self_op else 77
    x = torch.rand(n, 3, generator=g)
    y = None if self_op else torch.rand(m, 3, generator=g)
    v = torch.randn(m, 5, generator=g)
    params = _params(-0.5 / 0.3**2, 1.0, 1e-2 if self_op else 0.0, "cpu")
    calls = []
    monkeypatch.setattr(D, "_kmv_launch", _fake_launch(calls))
    monkeypatch.setattr(D, "_use_simulator", lambda: True)  # host tensors, no device move

    monkeypatch.delenv("TRNSPARSE_KMV_BLOCK_WORK", raising=False)
    one = D.nki_kernel_matmul(x, y, v, params, 0, 4)
    assert len(calls) == 1 and calls[0][2] is False  # under the budget: one plain launch

    calls.clear()
    monkeypatch.setenv("TRNSPARSE_KMV_BLOCK_WORK", "1")  # force 128-row blocks
    blocked = D.nki_kernel_matmul(x, y, v, params, 0, 4)
    assert torch.equal(one, blocked)
    rows = K.kmv_block_rows(n, m, 1)
    assert len(calls) == -(-n // rows)
    if n > rows:
        # Every launch has the block shape (ragged tail padded), Y whole, and
        # the square operator's blocks carry their rows of V for the noise.
        assert {c[0] for c in calls} == {(rows, 3)}
        assert {c[1] for c in calls} == {(m, 3)}
        assert all(c[2] is self_op for c in calls)
        # Y, V and params are placed once: every launch gets the same storage.
        Y_in = x if self_op else y
        assert {c[3] for c in calls} == {(Y_in.data_ptr(), v.data_ptr(), params.data_ptr())}


def test_kernel_operator_row_blocked_under_nki_backend(monkeypatch):
    """KernelOperator on the nki route goes through the blocked launcher."""
    from trnsparse.nki import dispatch as D

    calls = []
    monkeypatch.setattr(D, "_kmv_launch", _fake_launch(calls))
    monkeypatch.setattr(D, "_use_simulator", lambda: True)
    monkeypatch.setattr(D, "_use_nki", lambda: True)
    g = torch.Generator().manual_seed(9)
    X, V = torch.rand(700, 3, generator=g), torch.randn(700, 4, generator=g)
    op = KernelOperator(X, "rbf", 0.3, 1.0, noise=1e-3)
    assert op.route == "nki"
    monkeypatch.delenv("TRNSPARSE_KMV_BLOCK_WORK", raising=False)
    a = op(V)
    monkeypatch.setenv("TRNSPARSE_KMV_BLOCK_WORK", str(256 * 700))
    b = op(V)
    assert len(calls) == 1 + 3  # three 256-row blocks: 256, 256, 188 (padded)
    assert torch.equal(a, b)


# ── Row blocks across logical NeuronCores (#102) ─────────────────────────────


def test_kmv_block_rows_parts():
    W = K._KMV_BLOCK_WORK
    n = 1 << 20
    # parts=1 is the single-core rule, unchanged.
    assert K.kmv_block_rows(n, n, W, parts=1) == K.kmv_block_rows(n, n, W) == 55296
    # 4 cores at n = m = 2^20: 20 blocks of 53248 rows, 5 per core, under budget.
    rows = K.kmv_block_rows(n, n, W, parts=4)
    assert rows == 53248 and rows * n <= W and -(-n // rows) == 20
    for n, m, work, parts in [
        (1 << 20, 1 << 20, W, 2),
        (1 << 20, 1 << 20, W, 3),
        (524_288, 524_288, W, 4),  # 5 single-core blocks -> 8, 2 per core
        (300_000, 300_000, W, 4),  # 2 single-core blocks -> 4
        (5000, 5000, W, 4),  # under budget: still split into 4
        (5000, 3000, 1024 * 3000, 4),
        (1000, 77, 1, 3),
    ]:
        rows = K.kmv_block_rows(n, m, work, parts=parts)
        nblk = -(-n // rows)
        assert rows % 128 == 0, (n, m, parts, rows)
        assert nblk % parts == 0 or rows == 128  # at the one-tile floor: as many as fit
        assert rows * m <= max(work, 128 * m)
        assert nblk >= -(-n // K.kmv_block_rows(n, m, work))  # never fewer launches
    # Fewer than one 128-row tile per core: as many tiles as there are.
    assert K.kmv_block_rows(300, 50, 10**9, parts=4) == 128
    assert K.kmv_block_rows(100, 50, 10**9, parts=4) == 100


def test_kmv_core_count_and_env(monkeypatch):
    from trnsparse.nki import dispatch as D

    W = K._KMV_BLOCK_WORK
    big, small = (1 << 20, 1 << 20), (1000, 1000)
    # auto: every visible core once the product is row-blocked, else one.
    assert D._kmv_core_count(*big, 4, "auto", W) == 4
    assert D._kmv_core_count(*small, 4, "auto", W) == 1
    assert D._kmv_core_count(*big, 1, "auto", W) == 1
    # all / k: always split, capped at the visible count; 1 is single-core.
    assert D._kmv_core_count(*small, 4, "all", W) == 4
    assert D._kmv_core_count(*small, 4, 2, W) == 2
    assert D._kmv_core_count(*big, 2, 8, W) == 2
    assert D._kmv_core_count(*big, 4, 1, W) == 1
    for raw, want in [("", "auto"), ("auto", "auto"), ("ALL", "all"), (" 3 ", 3), ("1", 1)]:
        monkeypatch.setenv("TRNSPARSE_KMV_CORES", raw)
        assert D._kmv_cores_setting() == want
    for bad in ["0", "-2", "four", "1.5"]:
        monkeypatch.setenv("TRNSPARSE_KMV_CORES", bad)
        with pytest.raises(ValueError, match="TRNSPARSE_KMV_CORES"):
            D._kmv_cores_setting()
    # Off the native path there is nothing to split over.
    monkeypatch.setenv("TRNSPARSE_KMV_CORES", "all")
    monkeypatch.setattr(D, "_get_dispatch_path", lambda: "cpu")
    assert D._kmv_core_devices(*big, (torch.zeros(1),)) == []


def _multicore_standin(monkeypatch, k, threads=False):
    """Route `nki_kernel_matmul` through the multi-core path on host "cores".

    Core 0 keeps the caller's tensors; every other core gets its own copies
    (`_kmv_place` -> clone), so the operand storage a launch sees names the
    core it ran on.
    """
    from trnsparse.nki import dispatch as D

    monkeypatch.setattr(D, "_use_simulator", lambda: False)
    monkeypatch.setattr(D, "_kmv_core_devices", lambda n, m, t: [torch.device("cpu")] * k)
    placed = []

    def place(t, dev):
        c = t.clone()
        placed.append(c)
        return c

    monkeypatch.setattr(D, "_kmv_place", place)
    if threads:
        monkeypatch.setenv("TRNSPARSE_KMV_THREADS", "1")
    else:
        monkeypatch.delenv("TRNSPARSE_KMV_THREADS", raising=False)
    return placed


@pytest.mark.parametrize("threads", [False, True], ids=["async", "threads"])
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
@pytest.mark.parametrize("n", [300, 1000, 2500])
def test_multicore_row_blocks_match_single_core(monkeypatch, self_op, n, k, threads):
    import threading

    from trnsparse.kernel_matmul import _params
    from trnsparse.nki import dispatch as D

    g = torch.Generator().manual_seed(n + k)
    m = n if self_op else 77
    x = torch.rand(n, 3, generator=g)
    y = None if self_op else torch.rand(m, 3, generator=g)
    v = torch.randn(m, 5, generator=g)
    params = _params(-0.5 / 0.3**2, 1.0, 1e-2 if self_op else 0.0, "cpu")
    fake = _fake_launch(calls := [])
    monkeypatch.setattr(D, "_kmv_launch", fake)

    # Single core: one launch, and the single-core 128-row blocking.
    monkeypatch.setattr(D, "_use_simulator", lambda: True)
    monkeypatch.delenv("TRNSPARSE_KMV_BLOCK_WORK", raising=False)
    one = D.nki_kernel_matmul(x, y, v, params, 0, 4)
    monkeypatch.setenv("TRNSPARSE_KMV_BLOCK_WORK", "1")
    single = D.nki_kernel_matmul(x, y, v, params, 0, 4)
    assert torch.equal(one, single)

    # k host "cores", on the same forced 128-row-tile budget.
    placed = _multicore_standin(monkeypatch, k, threads)
    on_thread = []

    def launch(*a):
        on_thread.append(threading.current_thread() is threading.main_thread())
        return fake(*a)

    monkeypatch.setattr(D, "_kmv_launch", launch)
    calls.clear()
    multi = D.nki_kernel_matmul(x, y, v, params, 0, 4)
    assert torch.equal(multi, single)  # gather in row order, bitwise
    assert multi.device == x.device

    rows = K.kmv_block_rows(n, m, 1, parts=k)
    nblk = -(-n // rows)
    kk = min(k, nblk)
    assert len(calls) == nblk
    assert {c[0] for c in calls} == {(rows, 3)}  # one block shape (ragged tail padded)
    assert all(c[2] is self_op for c in calls)
    # Y, V and params placed once per core: copies for cores 1.., none for core 0.
    assert len(placed) == (kk - 1) * (3 if self_op else 4)
    Y_in = x if self_op else y
    core_ops = [c[3] for c in calls[:kk]]
    assert core_ops[0] == (Y_in.data_ptr(), v.data_ptr(), params.data_ptr())
    assert len(set(core_ops)) == kk
    # Round robin: block b on core b % k, so each core gets the same count.
    if threads:
        by_core = {}
        for c in calls:
            by_core.setdefault(c[3], 0)
            by_core[c[3]] += 1
        assert set(by_core) == set(core_ops)
        assert all(on_thread[:kk])  # first block of every core traced on the caller
    else:
        assert [c[3] for c in calls] == [core_ops[b % kk] for b in range(nblk)]
        assert all(on_thread)  # every launch from the calling thread
    counts = [sum(1 for c in calls if c[3] == ops) for ops in core_ops]
    assert max(counts) - min(counts) <= (0 if nblk % kk == 0 else 1)


def test_multicore_under_budget_auto_stays_single(monkeypatch):
    """`auto` below the blocking budget is one launch on one core, as before."""
    from trnsparse.kernel_matmul import _params
    from trnsparse.nki import dispatch as D

    calls = []
    monkeypatch.setattr(D, "_kmv_launch", _fake_launch(calls))
    monkeypatch.setattr(D, "_use_simulator", lambda: False)
    monkeypatch.setattr(D, "_get_dispatch_path", lambda: "native")
    monkeypatch.setattr(D, "_visible_cores", lambda: 4)
    monkeypatch.setattr(D, "_to_device", lambda *t: (list(t), t[0].device))
    monkeypatch.delenv("TRNSPARSE_KMV_CORES", raising=False)
    monkeypatch.delenv("TRNSPARSE_KMV_BLOCK_WORK", raising=False)
    x, v = torch.rand(500, 3), torch.randn(500, 2)
    D.nki_kernel_matmul(x, None, v, _params(-1.0, 1.0, 1e-2, "cpu"), 0, 4)
    assert len(calls) == 1 and calls[0][0] == (500, 3)
