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


def _run(x, y, v, params, kind, lnc, self_op, group=4):
    if self_op:
        args = (x, v, params, kind, group)
        return np.asarray(simulate_kernel(K._kmv_self_kernel, args, {}, lnc=lnc, target="trn2"))
    args = (x, y, v, params, kind, group)
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
    tol = 5e-3 if kernel == "matern12" and self_op else 5e-6  # Matern-1/2 cusp, see kernel_matmul
    outs = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # e.g. PSUM accumulate / ScalarE range warnings
        for lnc in (1, 2):
            out = _run(x, y, v, params, kind, lnc, self_op)
            assert out.shape == (n, R)
            assert np.abs(out - ref).max() / np.abs(ref).max() < tol
            outs.append(out)
    if d <= 3:
        # Same Y-tile order on any core count. (The simulator's matmul is a
        # BLAS call whose summation order can depend on the tile width, so
        # this is asserted where the contraction is short enough not to.)
        assert np.array_equal(outs[0], outs[1])


def test_simulate_deterministic_across_runs():
    x, y, v, params, _ = _inputs(1000, 1000, 3, 33, "rbf", True)
    a = _run(x, y, v, params, 0, 2, True)
    b = _run(x, y, v, params, 0, 2, True)
    assert np.array_equal(a, b)


def test_group_size_does_not_change_the_answer():
    x, y, v, params, ref = _inputs(1500, 1500, 3, 4, "rbf", True)
    for g in (1, 2, 7):
        out = _run(x, y, v, params, 0, 1, True, group=g)
        assert np.abs(out - ref).max() / np.abs(ref).max() < 5e-6


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


def _run_blocked(x, y, v, params, kind, lnc, self_op, rows, group=4):
    """The product as zero-padded row blocks of `rows`, as `nki_kernel_matmul` launches it (#100)."""
    n, d = x.shape
    yy = x if self_op else y
    outs = []
    for i0 in range(0, n, rows):
        nb = min(rows, n - i0)
        xb = np.zeros((rows, d), np.float32)
        xb[:nb] = x[i0 : i0 + nb]
        if self_op:
            vb = np.zeros((rows, v.shape[1]), np.float32)
            vb[:nb] = v[i0 : i0 + nb]
            args = (xb, yy, v, vb, params, kind, group)
            kern = K._kmv_selfblk_kernel
        else:
            args = (xb, yy, v, params, kind, group)
            kern = K._kmv_cross_kernel
        ob = np.asarray(simulate_kernel(kern, args, {}, lnc=lnc, target="trn2"))
        outs.append(ob[:nb])
    return np.concatenate(outs)


def _assert_blocked_matches(one, blocked):
    """Bitwise, except the simulator's ragged last 128-row chunk (n % 128 rows).

    In one launch that chunk is a matmul of n % 128 columns; in a padded
    block it is part of a full 128-column one. The simulator runs each as a
    numpy matmul, whose summation order changes with the width (a 1-row
    product is a GEMV), so those rows may differ in the last bits there. The
    TensorEngine computes every output column the same way at any width,
    which the neuron-marked test checks bit for bit on the device.
    """
    n = one.shape[0]
    head = n - n % _TILE
    assert np.array_equal(one[:head], blocked[:head])
    # fp32 rounding of a different summation order: within the kernel's
    # own accuracy bound against float64 (5e-6 of the largest entry).
    assert np.abs(blocked[head:] - one[head:]).max(initial=0.0) <= 5e-6 * np.abs(one).max()


_TILE = 128


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
                4,
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


@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("case", sorted(_COMPILE))
def test_kmv_mlir_verifies(case, lnc):
    name, mk = _COMPILE[case]
    result = source_to_mlir(
        getattr(K, name), *mk(), target="trn2", lnc=lnc, frontend=FrontendKind.TRACER
    )
    result.verify()


@pytest.mark.skipif(shutil.which("neuronx-cc") is None, reason="neuronx-cc not on PATH")
@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("case", sorted(_COMPILE))
def test_kmv_neff_compiles_with_nki_allocation(case, lnc, tmp_path):
    try:
        from nki.compiler import driver
        from nki.compiler.frontend import TracerFrontend
        from nki.compiler.ncc_driver import CompileOptions
    except ImportError as e:
        pytest.skip(f"nki compiler driver API not available: {e}")
    name, mk = _COMPILE[case]
    fn = getattr(K, name).func
    inputs = dict(zip(inspect.signature(fn).parameters, mk(), strict=True))
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
