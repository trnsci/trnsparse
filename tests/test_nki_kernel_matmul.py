"""The fused kernel-matrix product kernel (#93) under nki.simulate and the compiler, on CPU.

- `nki.simulate` at LNC=1 and LNC=2 (both physical-core traces) against the
  float64 reference, over ragged n, m, d and R, every kernel family;
- bit-identical results at LNC=1 and LNC=2 (the reduction over Y tiles runs
  in the same order on any core count);
- the dispatch path (`kernel_matmul` with backend "nki") under
  TRNSPARSE_USE_SIMULATOR=1;
- MLIR verification of both entry points at LNC=1 and LNC=2;
- with neuronx-cc on PATH, the full NEFF compile at LNC=1 and LNC=2, with
  NKI's own allocation (no silent fallback to compiler allocation, which
  turns off NKI scheduling).

Needs `nki` but no Neuron device.
"""

from __future__ import annotations

import inspect
import re
import shutil
import warnings

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.nki_simulator

from trnsparse.nki.kernels import HAS_NKI  # noqa: E402

if not HAS_NKI:
    pytest.skip("nki not importable on this host", allow_module_level=True)

from nki._simulator import simulate_kernel  # noqa: E402
from nki.compiler.frontend import FrontendKind, source_to_mlir  # noqa: E402

import trnsparse  # noqa: E402
from trnsparse import kernel_matmul_reference  # noqa: E402
from trnsparse.kernel_matmul import _params, _scale_coeff  # noqa: E402
from trnsparse.nki import dispatch  # noqa: E402
from trnsparse.nki import kernels as K  # noqa: E402

_KINDS = ("rbf", "matern12", "matern32", "matern52")


class _ColumnwiseNumpy:
    """numpy, with `matmul` summed in contraction order, one fp32 add per term.

    The TensorEngine computes every output column of a matmul the same way
    whatever the moving operand's width. numpy's BLAS (and einsum) do not:
    their blocking, and the dot/GEMV paths of a one-row or one-column
    product, follow the shape and the memory layout, so a column can round
    differently in a wider product. The kernel's @V matmul moves all X
    points of a block (#103), and that width differs between LNC=1 and
    LNC=2 in the ragged tail and for small n, so the bitwise checks here
    run the simulator's matmul as an explicit fp32 sum over the contraction
    in order, which depends on nothing but the column's own operands.
    """

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def matmul(a, b):
        a = np.asarray(a, np.float32)  # (M, K): the stationary operand, transposed
        b = np.asarray(b, np.float32)  # (K, N): the moving operand
        out = np.zeros((a.shape[0], b.shape[1]), np.float32)
        for k in range(a.shape[1]):
            out += np.multiply.outer(a[:, k], b[k])
        return out


@pytest.fixture(autouse=True)
def _columnwise_sim_matmul(monkeypatch):
    from nki._backends.simulator import matmul as sim_matmul

    monkeypatch.setattr(sim_matmul, "np", _ColumnwiseNumpy())


def _inputs(n, m, d, R, kernel, self_op, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, d)) - 0.5).astype(np.float32)
    y = x if self_op else (rng.random((m, d)) - 0.5).astype(np.float32)
    v = rng.standard_normal((m, R)).astype(np.float32)
    ell = 0.4 * d**0.5
    a = _scale_coeff(kernel) / ell**2
    noise = 1e-2 if self_op else 0.0
    params = _params(a, 1.3, noise, "cpu").numpy()
    ref = kernel_matmul_reference(
        torch.from_numpy(x),
        None if self_op else torch.from_numpy(y),
        torch.from_numpy(v),
        kernel,
        ell,
        1.3,
        noise,
    ).numpy()
    return x, y, v, params, ref


def _run(x, y, v, params, kind, lnc, self_op, group=4, variant=0):
    extra = (variant,) if variant else ()
    if self_op:
        args = (x, v, params, kind, group, *extra)
        return np.asarray(simulate_kernel(K._kmv_self_kernel, args, {}, lnc=lnc, target="trn2"))
    args = (x, y, v, params, kind, group, *extra)
    return np.asarray(simulate_kernel(K._kmv_cross_kernel, args, {}, lnc=lnc, target="trn2"))


# (n, m, d, R, kernel): ragged tiles (1, 127, 129, 4097), contraction lengths
# d + 2 up to 128, R across one and several 512-column PSUM tiles.
_SIM_CASES = [
    (1, 1, 3, 1, "rbf"),
    (127, 127, 3, 5, "matern52"),
    (129, 129, 1, 1, "rbf"),
    (1000, 1000, 3, 7, "rbf"),
    (600, 600, 16, 64, "matern32"),
    (4097, 4097, 3, 3, "matern12"),
    (300, 200, 3, 513, "rbf"),
    (700, 513, 126, 2, "rbf"),
    (128, 100, 3, 2, "matern32"),
]


@pytest.mark.parametrize("case", _SIM_CASES, ids=lambda c: "-".join(map(str, c)))
@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
def test_simulate_matches_reference_lnc1_lnc2(case, self_op):
    n, m, d, R, kernel = case
    if self_op and n != m:
        pytest.skip("square operator needs m = n")
    x, y, v, params, ref = _inputs(n, m, d, R, kernel, self_op)
    kind = _KINDS.index(kernel)
    tol = 5e-6  # the square operator's diagonal is exact, Matern-1/2 included (#143)
    outs = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # e.g. PSUM accumulate / ScalarE range warnings
        for lnc in (1, 2):
            out = _run(x, y, v, params, kind, lnc, self_op)
            assert out.shape == (n, R)
            assert np.abs(out - ref).max() / np.abs(ref).max() < tol
            outs.append(out)
    # Same Y-tile order on any core count (the simulator's matmul sums in
    # contraction order, `_ColumnwiseNumpy`).
    assert np.array_equal(outs[0], outs[1])


def test_simulate_deterministic_across_runs():
    x, y, v, params, _ = _inputs(1000, 1000, 3, 33, "rbf", True)
    a = _run(x, y, v, params, 0, 2, True)
    b = _run(x, y, v, params, 0, 2, True)
    assert np.array_equal(a, b)


def test_group_size_does_not_change_the_answer():
    x, y, v, params, ref = _inputs(1500, 1500, 3, 4, "rbf", True)
    for g in (1, 2, 7, 16, 32):
        out = _run(x, y, v, params, 0, 1, True, group=g)
        assert np.abs(out - ref).max() / np.abs(ref).max() < 5e-6


@pytest.mark.parametrize("R", [1, 130])
def test_large_bodies_lnc_bitwise_and_accurate(R):
    """Bodies of 16 tiles, a ragged Y tile, two V column tiles: LNC=1 == LNC=2, accurate."""
    x, y, v, params, ref = _inputs(700, 2100 + 37, 3, R, "matern32", False, seed=R)
    outs = [_run(x, y, v, params, 2, lnc, False, group=16) for lnc in (1, 2)]
    assert np.array_equal(outs[0], outs[1])
    assert np.abs(outs[0] - ref).max() / np.abs(ref).max() < 5e-6


@pytest.mark.skipif(not dispatch._use_simulator(), reason="needs TRNSPARSE_USE_SIMULATOR=1")
def test_dispatch_path_under_simulator():
    old = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    try:
        g = torch.Generator().manual_seed(5)
        X = torch.rand(300, 3, generator=g) + 5.0
        V = torch.randn(300, 3, generator=g)
        op = trnsparse.KernelOperator(X, "rbf", 0.3, 1.2, noise=1e-3)
        assert op.route == "nki"
        ref = kernel_matmul_reference(X, None, V, "rbf", 0.3, 1.2, 1e-3)
        out = op(V)
        assert float((out.double() - ref).abs().max() / ref.abs().max()) < 1e-5
        Xs = torch.rand(50, 3, generator=g) + 5.0
        ref_c = kernel_matmul_reference(Xs, X, V, "rbf", 0.3, 1.2)
        out_c = op.cross(Xs, V)
        assert float((out_c.double() - ref_c).abs().max() / ref_c.abs().max()) < 1e-5
    finally:
        trnsparse.set_backend(old)


def _run_blocked(x, y, v, params, kind, lnc, self_op, rows, group=4, variant=0):
    """The product as zero-padded row blocks of `rows`, as `nki_kernel_matmul` launches it (#100)."""
    n, d = x.shape
    yy = x if self_op else y
    outs = []
    for i0 in range(0, n, rows):
        nb = min(rows, n - i0)
        xb = np.zeros((rows, d), np.float32)
        xb[:nb] = x[i0 : i0 + nb]
        extra = (variant,) if variant else ()
        if self_op:
            vb = np.zeros((rows, v.shape[1]), np.float32)
            vb[:nb] = v[i0 : i0 + nb]
            args = (xb, yy, v, vb, params, kind, group, *extra)
            kern = K._kmv_selfblk_kernel
        else:
            args = (xb, yy, v, params, kind, group, *extra)
            kern = K._kmv_cross_kernel
        ob = np.asarray(simulate_kernel(kern, args, {}, lnc=lnc, target="trn2"))
        outs.append(ob[:nb])
    return np.concatenate(outs)


def _assert_blocked_matches(one, blocked):
    """Bitwise: every output column is computed the same way at any block width.

    (The simulator's matmul sums in contraction order, `_ColumnwiseNumpy`;
    the neuron-marked test checks the same bit for bit on the device.)
    """
    assert np.array_equal(one, blocked)


_TILE = 128


@pytest.mark.parametrize("kernel", _KINDS)
def test_simulate_square_diagonal_is_exact(kernel):
    """K_ii = s on the square operator, one launch and row-blocked (#143).

    Far-from-centre points (|x| ~ 4) and l = 0.05 put the expansion's
    D_ii ~ u |x|^2 under Matern-1/2's cusp: 3e-3 relative in K_ii before
    the correction. V = I reads K + noise I.
    """
    n = 300
    rng = np.random.default_rng(1)
    x = (4.0 * rng.random((n, 3))).astype(np.float32)
    v = np.eye(n, dtype=np.float32)
    ell, s, noise = 0.05, 1.3, 1e-2
    params = _params(_scale_coeff(kernel) / ell**2, s, noise, "cpu").numpy()
    ref = kernel_matmul_reference(
        torch.from_numpy(x), None, torch.from_numpy(v), kernel, ell, s, noise
    ).numpy()
    kind = _KINDS.index(kernel)
    # Off the diagonal: the cross kernel on (X, X) bit for bit (the
    # expansion's error at near-duplicate pairs stays there, uncentred).
    p0 = _params(_scale_coeff(kernel) / ell**2, s, 0.0, "cpu").numpy()
    cross = _run(x, x, v, p0, kind, 2, False)
    off = ~np.eye(n, dtype=bool)
    for out in (
        _run(x, None, v, params, kind, 2, True),
        _run_blocked(x, None, v, params, kind, 1, True, 128),
    ):
        np.testing.assert_allclose(np.diagonal(out), s + noise, rtol=3e-7, atol=0)
        assert np.array_equal(out[off], cross[off])
    assert np.abs(np.diagonal(ref) - (s + noise)).max() < 1e-12


@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
@pytest.mark.parametrize("n,rows", [(1000, 256), (1280, 1024), (1300, 1024), (513, 128)])
def test_simulate_row_blocked_is_bitwise_one_launch(n, rows, kernel, self_op, lnc):
    """Row blocks (ragged tail zero-padded) give the one-launch result bit for bit (#100)."""
    m = n if self_op else 700
    x, y, v, params, ref = _inputs(n, m, 3, 33, kernel, self_op, seed=n)
    kind = _KINDS.index(kernel)
    one = _run(x, y, v, params, kind, lnc, self_op)
    blocked = _run_blocked(x, y, v, params, kind, lnc, self_op, rows)
    _assert_blocked_matches(one, blocked)


@pytest.mark.skipif(not dispatch._use_simulator(), reason="needs TRNSPARSE_USE_SIMULATOR=1")
@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
def test_dispatch_row_blocked_under_simulator(monkeypatch, self_op):
    """`nki_kernel_matmul` with a forced tiny budget == the one-launch kernel, bitwise."""
    n = 640 + 60  # five 128-row chunks and a ragged one
    m = n if self_op else 300
    x, y, v, params, _ = _inputs(n, m, 3, 5, "rbf", self_op, seed=11)
    args = [torch.from_numpy(a) for a in (x, v, params)]
    xt, vt, pt = args
    yt = None if self_op else torch.from_numpy(y)
    monkeypatch.delenv("TRNSPARSE_KMV_BLOCK_WORK", raising=False)
    one = dispatch.nki_kernel_matmul(xt, yt, vt, pt, 0, 4)
    monkeypatch.setenv("TRNSPARSE_KMV_BLOCK_WORK", str(256 * m))  # 256-row blocks
    assert K.kmv_block_rows(n, m, dispatch._kmv_block_work()) == 256
    blocked = dispatch.nki_kernel_matmul(xt, yt, vt, pt, 0, 4)
    _assert_blocked_matches(one.numpy(), blocked.numpy())
    direct = _run(x, y, v, params, 0, 1, self_op)
    assert np.array_equal(one.numpy(), direct)


# ── Row-tiled distance matmuls (#119, TRNSPARSE_KMV_SUBTILE=1) ───────────────
#
# The variant moves each Y tile's augmented coordinates (and a copy of the X
# block) to the SBUF partitions of a TensorEngine row tile and issues four
# distance matmuls back to back; the arithmetic and every summation order are
# the default kernel's, so the simulator must give the default kernel's bits:
# 32-row tiles (d + 2 <= 32), 64-row tiles (d = 40), no tiling (d = 126),
# ragged Y tiles, bodies that are not a multiple of 4 tiles, V passes of 4
# column tiles instead of 6 (R = 700), the exact diagonal and row blocks.

_ST = K._KMV_SUBTILE

_SUBTILE_CASES = _SIM_CASES + [
    (600, 600, 40, 3, "matern12"),
    (300, 1100, 3, 700, "rbf"),
    (513, 2100 + 37, 5, 130, "matern52"),
]


@pytest.mark.parametrize("case", _SUBTILE_CASES, ids=lambda c: "-".join(map(str, c)))
@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
def test_subtile_is_bitwise_default_kernel(case, self_op):
    n, m, d, R, kernel = case
    if self_op and n != m:
        pytest.skip("square operator needs m = n")
    x, y, v, params, ref = _inputs(n, m, d, R, kernel, self_op)
    kind = _KINDS.index(kernel)
    lncs = (1, 2) if n * m <= 10**6 else (2,)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for lnc in lncs:
            for group in (4, 7):
                base = _run(x, y, v, params, kind, lnc, self_op, group)
                st = _run(x, y, v, params, kind, lnc, self_op, group, variant=_ST)
                assert np.array_equal(st, base), (lnc, group)
    assert np.abs(st - ref).max() / np.abs(ref).max() < 5e-6


@pytest.mark.parametrize("kernel", _KINDS)
def test_subtile_square_diagonal_is_exact(kernel):
    """K_ii = s with row-tiled distance matmuls, one launch and row-blocked (#143, #119)."""
    n = 300
    rng = np.random.default_rng(1)
    x = (4.0 * rng.random((n, 3))).astype(np.float32)
    v = np.eye(n, dtype=np.float32)
    ell, s, noise = 0.05, 1.3, 1e-2
    params = _params(_scale_coeff(kernel) / ell**2, s, noise, "cpu").numpy()
    kind = _KINDS.index(kernel)
    base = _run(x, None, v, params, kind, 2, True)
    for out in (
        _run(x, None, v, params, kind, 2, True, variant=_ST),
        _run_blocked(x, None, v, params, kind, 1, True, 128, variant=_ST),
    ):
        np.testing.assert_allclose(np.diagonal(out), s + noise, rtol=3e-7, atol=0)
        assert np.array_equal(out, base)


@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
def test_subtile_row_blocked_is_bitwise_one_launch(self_op):
    n, rows = 1300, 1024
    m = n if self_op else 700
    x, y, v, params, _ = _inputs(n, m, 3, 33, "matern52", self_op, seed=n)
    one = _run(x, y, v, params, 3, 2, self_op)
    blocked = _run_blocked(x, y, v, params, 3, 2, self_op, rows, variant=_ST)
    _assert_blocked_matches(one, blocked)


def test_subtile_plan_leaves_four_distance_banks():
    """Every V pass of the variant leaves >= 4 PSUM banks for the distance tiles in flight."""
    for R in (1, 128, 512, 513, 700, 1500):
        _, groups = K.kmv_plan(1000, R, 1, K._KMV_SUBTILE_PASS_RT)
        assert all(K._KMV_PSUM_BANKS - len(g) >= 4 for g in groups)
        assert sum(rr for g in groups for _, rr in g) == R
    assert [K.kmv_pe_tile(s) for s in (1, 5, 32, 33, 64, 65, 128)] == [32, 32, 32, 64, 64, 128, 128]


# The norm fold (`_KMV_NORM_BIAS`, TRNSPARSE_KMV_NORM_BIAS=1) moves |y|^2 from
# the GEMM to the PSUM reader: not the default bits (rbf's exp now rounds
# a D' + b), but the same accuracy, the same bits at LNC=1 and LNC=2, with
# and without row tiling, and an exact diagonal.

_NB = K._KMV_NORM_BIAS


@pytest.mark.parametrize("case", _SUBTILE_CASES, ids=lambda c: "-".join(map(str, c)))
@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
def test_norm_bias_accurate_and_lnc_bitwise(case, self_op):
    n, m, d, R, kernel = case
    if self_op and n != m:
        pytest.skip("square operator needs m = n")
    x, y, v, params, ref = _inputs(n, m, d, R, kernel, self_op)
    kind = _KINDS.index(kernel)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fb = _run(x, y, v, params, kind, 2, self_op, variant=_NB)
        assert np.abs(fb - ref).max() / np.abs(ref).max() < 5e-6
        assert np.array_equal(_run(x, y, v, params, kind, 2, self_op, variant=_NB | _ST), fb)
        if n * m <= 10**6:
            assert np.array_equal(_run(x, y, v, params, kind, 1, self_op, variant=_NB), fb)


@pytest.mark.parametrize("kernel", _KINDS)
def test_norm_bias_square_diagonal_is_exact(kernel):
    n = 300
    rng = np.random.default_rng(1)
    x = (4.0 * rng.random((n, 3))).astype(np.float32)
    v = np.eye(n, dtype=np.float32)
    ell, s, noise = 0.05, 1.3, 1e-2
    params = _params(_scale_coeff(kernel) / ell**2, s, noise, "cpu").numpy()
    p0 = _params(_scale_coeff(kernel) / ell**2, s, 0.0, "cpu").numpy()
    kind = _KINDS.index(kernel)
    var = _NB | _ST
    cross = _run(x, x, v, p0, kind, 2, False, variant=var)
    off = ~np.eye(n, dtype=bool)
    for out in (
        _run(x, None, v, params, kind, 2, True, variant=var),
        _run_blocked(x, None, v, params, kind, 1, True, 128, variant=var),
    ):
        np.testing.assert_allclose(np.diagonal(out), s + noise, rtol=3e-7, atol=0)
        assert np.array_equal(out[off], cross[off])


@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
def test_norm_bias_row_blocked_is_bitwise_one_launch(self_op):
    n, rows = 1300, 1024
    m = n if self_op else 700
    x, y, v, params, _ = _inputs(n, m, 3, 33, "rbf", self_op, seed=n)
    var = _NB | _ST
    one = _run(x, y, v, params, 0, 2, self_op, variant=var)
    blocked = _run_blocked(x, y, v, params, 0, 2, self_op, rows, variant=var)
    _assert_blocked_matches(one, blocked)


@pytest.mark.skipif(not dispatch._use_simulator(), reason="needs TRNSPARSE_USE_SIMULATOR=1")
@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
def test_dispatch_subtile_env_under_simulator(monkeypatch, self_op):
    """TRNSPARSE_KMV_SUBTILE=1 through `nki_kernel_matmul`: the default kernel's bits."""
    n = 700
    m = n if self_op else 300
    x, y, v, params, _ = _inputs(n, m, 3, 5, "matern12", self_op, seed=12)
    xt, vt, pt = (torch.from_numpy(a) for a in (x, v, params))
    yt = None if self_op else torch.from_numpy(y)
    monkeypatch.delenv("TRNSPARSE_KMV_SUBTILE", raising=False)
    base = dispatch.nki_kernel_matmul(xt, yt, vt, pt, 1, 4)
    monkeypatch.setenv("TRNSPARSE_KMV_SUBTILE", "1")
    st = dispatch.nki_kernel_matmul(xt, yt, vt, pt, 1, 4)
    assert np.array_equal(st.numpy(), base.numpy())
    monkeypatch.setenv("TRNSPARSE_KMV_NORM_BIAS", "1")
    fb = dispatch.nki_kernel_matmul(xt, yt, vt, pt, 1, 4).numpy()
    direct = _run(x, y, v, params, 1, 1, self_op, 4, variant=_ST | _NB)
    assert np.array_equal(fb, direct)


def _compile_cases():
    f = np.zeros
    p = f((128, 4), np.float32)
    cases = {}
    for n, R, kind, g in [
        (1000, 33, 0, 4),
        (4096, 64, 0, 7),
        (1000, 512, 0, 4),
        (1000, 513, 3, 4),
        (129, 1, 1, 4),
        (1, 1, 0, 4),
        (131072, 33, 0, 4),
        (131072, 33, 0, 16),
        (131072, 128, 3, 16),
        (8192, 768, 0, 32),
        (8192, 769, 0, 16),
        # The auto group reaches 32 at m >= 131072 (#108): every V pass width
        # (one tile, the 64 KiB batched-V limit at 512 columns, a full
        # 6-tile pass, two passes) with the most SBUF temporaries (Matern-5/2).
        # Without the scratch ring, Matern-5/2 fell back to neuronx-cc
        # allocation from 24 tiles (20 at R >= 384).
        (131072, 1, 3, 32),
        (131072, 128, 3, 32),
        (131072, 512, 3, 32),
        (131072, 513, 3, 32),
        (131072, 768, 3, 32),
        (131072, 1500, 3, 32),
        (131072, 1500, 0, 32),
    ]:
        cases[f"self-n{n}-R{R}-k{kind}-g{g}"] = (
            "_kmv_self_kernel",
            lambda n=n, R=R, kind=kind, g=g: (
                f((n, 3), np.float32),
                f((n, R), np.float32),
                p,
                kind,
                g,
            ),
        )
    for n, m, d, R, kind in [
        (700, 513, 3, 7, 0),
        (512, 8192, 3, 1500, 1),
        (1024, 1000, 3, 33, 2),
        (300, 4097, 126, 2, 0),
        (1000, 1100, 16, 1500, 3),
    ]:
        cases[f"cross-n{n}-m{m}-d{d}-R{R}-k{kind}"] = (
            "_kmv_cross_kernel",
            lambda n=n, m=m, d=d, R=R, kind=kind: (
                f((n, d), np.float32),
                f((m, d), np.float32),
                f((m, R), np.float32),
                p,
                kind,
                K.kmv_group(m),
            ),
        )
    # Row blocks of the square operator (#100): 55296 rows of n = 2^20 at R = 33,
    # and a small ragged-free block.
    for nb, m, R in [(55296, 1 << 20, 33), (256, 1000, 7)]:
        cases[f"selfblk-nb{nb}-m{m}-R{R}"] = (
            "_kmv_selfblk_kernel",
            lambda nb=nb, m=m, R=R: (
                f((nb, 3), np.float32),
                f((m, 3), np.float32),
                f((m, R), np.float32),
                f((nb, R), np.float32),
                p,
                0,
                4,
            ),
        )
    return cases


_COMPILE = _compile_cases()


_VARIANTS = [0, K._KMV_SUBTILE, K._KMV_SUBTILE | K._KMV_NORM_BIAS]
_VARIANT_IDS = ["default", "subtile", "subtile-normbias"]


def _variant_args(mk, variant):
    return (*mk(), variant) if variant else mk()


@pytest.mark.parametrize("variant", _VARIANTS, ids=_VARIANT_IDS)
@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("case", sorted(_COMPILE))
def test_kmv_mlir_verifies(case, lnc, variant):
    name, mk = _COMPILE[case]
    result = source_to_mlir(
        getattr(K, name),
        *_variant_args(mk, variant),
        target="trn2",
        lnc=lnc,
        frontend=FrontendKind.TRACER,
    )
    result.verify()
    if variant:
        # The distance matmuls carry their row tile (#119) when d + 2 <= 64;
        # with two or more Y tiles (m >= 256) the first body uses tile 1 too.
        args = mk()
        d = args[0].shape[1]
        m = args[0 if name == "_kmv_self_kernel" else 1].shape[0]
        tiled = bool(re.search(r"row_pos=index %c(32|64|96)\b", str(result)))
        if d + 2 > 64:
            assert not tiled
        elif m >= 256:
            assert tiled


@pytest.mark.skipif(shutil.which("neuronx-cc") is None, reason="neuronx-cc not on PATH")
@pytest.mark.parametrize("variant", _VARIANTS, ids=_VARIANT_IDS)
@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("case", sorted(_COMPILE))
def test_kmv_neff_compiles_with_nki_allocation(case, lnc, variant, tmp_path):
    try:
        from nki.compiler import driver
        from nki.compiler.frontend import TracerFrontend
        from nki.compiler.ncc_driver import CompileOptions
    except ImportError as e:
        pytest.skip(f"nki compiler driver API not available: {e}")
    name, mk = _COMPILE[case]
    fn = getattr(K, name).func
    args = _variant_args(mk, variant)
    params = list(inspect.signature(fn).parameters)[: len(args)]
    inputs = dict(zip(params, args, strict=True))
    opts = CompileOptions(
        target="trn2", lnc=lnc, artifacts_dir=str(tmp_path), output_path=str(tmp_path / "file.neff")
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bir = driver.compile_to_bir(
            fn, frontend=TracerFrontend(), inputs=inputs, compile_opts=opts, enable_cache=False
        )
        driver._compile_bir_to_neff(bir, opts, inputs)
    assert (tmp_path / "file.neff").stat().st_size > 0
    fallback = [str(x.message) for x in w if "allocation failed" in str(x.message)]
    assert not fallback, fallback[0][:800]
