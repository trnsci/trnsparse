"""CPU compile check: every NKI kernel lowers to MLIR that passes verification.

`nki.simulate` does not run the MLIR verifier, so a kernel can pass every
simulator test and still fail to compile for hardware. trnsparse#29 was such
a case on NKI 0.7.0: `nl.divide` by a `(P, 1)` softmax denominator lowered to
`nisa.tensor_scalar_arith`, which no longer supports `divide`, and the
screened-SpMM outer product `(P, 1) * (1, F)` lowered to a tensor-scalar op
whose 1-partition destination did not match its 128-partition operand.

This module runs the NKI frontend (trace -> MLIR -> verify) on CPU for every
kernel and target. It needs `nki` but no Neuron device and no `neuronx-cc`.
MLIR verification cannot catch errors raised later, in the backend: a
tensor_tensor `divide` passes `verify()` but fails neuronx-cc ISA validation
(`s3s3d3_tt_valid_op`). When `neuronx-cc` is on PATH, `test_kernel_neff_compiles`
additionally runs the full NEFF compile on CPU (no device needed).
"""

from __future__ import annotations

import inspect
import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.nki_simulator

from trnsparse.nki.kernels import HAS_NKI  # noqa: E402

try:
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    source_to_mlir = None

_P = 128


def _f32(*shape):
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape).astype(np.float32)


def _attn_inputs(hd):
    # (M_tiles=2, K_max=2) — shapes match nki_bsr_attn_tiled / nki_bsr_attn_bwd.
    return {
        "_attn_stats_kernel": (_f32(2, _P, hd), _f32(2, 2, _P, hd)),
        "_attn_out_kernel": (
            _f32(2, _P, hd),
            _f32(2, 2, _P, hd),
            _f32(2, 2, _P, hd),
            _f32(2, _P, 1),
            np.abs(_f32(2, _P, 1)) + 1.0,
        ),
        "_attn_bwd_dq_kernel": (
            _f32(2, _P, hd),
            _f32(2, 2, _P, hd),
            _f32(2, 2, _P, hd),
            _f32(2, 2, _P, hd),
            _f32(2, _P, 1),
            _f32(2, _P, 1),
            np.abs(_f32(2, _P, 1)) + 1.0,
        ),
        "_attn_bwd_dkdv_kernel": (
            _f32(2, _P, hd),
            _f32(2, _P, hd),
            _f32(2, 2, _P, hd),
            _f32(2, 2, _P, hd),
            _f32(2, 2, _P, 1),
            _f32(2, 2, _P, 1),
            np.abs(_f32(2, 2, _P, 1)) + 1.0,
        ),
    }


def _cases():
    cases = {
        ("_spmm_dense_kernel", "base"): lambda: (_f32(256, 256), _f32(256, 128)),
        ("_bsr_spmm_kernel", "base"): lambda: (_f32(2, 2, _P, _P), _f32(2, 2, _P, 128)),
        ("_screened_spmm_kernel", "base"): lambda: (
            _f32(256, 256),
            np.abs(_f32(256, 1)),
            np.full((1, 1), 0.5, dtype=np.float32),
            _f32(256, 128),
        ),
    }
    # Static-structure kernels: structure as tuple constants. An empty block
    # row and a reused column exercise the unrolled loops' edge cases.
    cases[("_bsr_spmm_static_kernel", "n128")] = lambda: (
        _f32(3, _P, _P),
        _f32(3 * _P, 128),
        (0, 2, 2, 3),
        (0, 2, 2),
    )
    cases[("_bsr_spmm_static_kernel", "n1024")] = lambda: (
        _f32(3, _P, _P),
        _f32(3 * _P, 1024),
        (0, 2, 2, 3),
        (0, 2, 2),
    )
    for hd in (64, 128):
        cases[("_attn_fwd_static_kernel", f"hd{hd}")] = lambda hd=hd: (
            _f32(hd, 3 * _P),
            _f32(hd, 3 * _P),
            _f32(3 * _P, hd),
            (0, 2, 2, 5),
            (0, 1, 0, 1, 2),
        )
        cases[("_attn_bwd_static_kernel", f"hd{hd}")] = lambda hd=hd: (
            _f32(hd, 3 * _P),
            _f32(hd, 3 * _P),
            _f32(hd, 3 * _P),
            _f32(hd, 3 * _P),
            _f32(3 * _P, hd),
            _f32(3 * _P, hd),
            _f32(3 * _P, hd),
            _f32(3 * _P, 1),
            np.abs(_f32(3 * _P, 1)),
            _f32(3 * _P, 1),
            _f32(1, 3 * _P),
            np.abs(_f32(1, 3 * _P)),
            _f32(1, 3 * _P),
            (0, 2, 2, 5),
            (0, 1, 0, 1, 2),
            (0, 2, 4, 5),
            (0, 2, 0, 2, 2),
            0.125,
        )
    # head_dim 64 / 128: single-tile path; head_dim=256: K-tiled path.
    for hd in (64, 128, 256):
        for name in _attn_inputs(hd):
            cases[(name, f"hd{hd}")] = lambda name=name, hd=hd: _attn_inputs(hd)[name]
    return cases


_CASES = _cases()


@pytest.fixture(autouse=True)
def _frontend_available():
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    if source_to_mlir is None:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")


@pytest.mark.parametrize("target", ["trn1", "trn2"])
@pytest.mark.parametrize("case", sorted(_CASES), ids=lambda c: f"{c[0]}-{c[1]}")
def test_kernel_mlir_verifies(case, target):
    from trnsparse.nki import kernels

    kernel = getattr(kernels, case[0])
    result = source_to_mlir(kernel, *_CASES[case](), target=target, frontend=FrontendKind.TRACER)
    result.verify()


@pytest.mark.skipif(shutil.which("neuronx-cc") is None, reason="neuronx-cc not on PATH")
@pytest.mark.parametrize("target", ["trn1", "trn2"])
@pytest.mark.parametrize("case", sorted(_CASES), ids=lambda c: f"{c[0]}-{c[1]}")
def test_kernel_neff_compiles(case, target, tmp_path):
    """Full CPU compile (MLIR -> BIR -> neuronx-cc NEFF); catches ISA-validation errors."""
    try:
        from nki.compiler import driver
        from nki.compiler.frontend import TracerFrontend
        from nki.compiler.ncc_driver import CompileOptions
    except ImportError as e:
        pytest.skip(f"nki compiler driver API not available: {e}")
    if not hasattr(driver, "_compile_bir_to_neff"):
        pytest.skip("nki.compiler.driver._compile_bir_to_neff not available in this nki")

    from trnsparse.nki import kernels

    kernel = getattr(kernels, case[0])
    fn = getattr(kernel, "func", kernel)
    inputs = dict(zip(inspect.signature(fn).parameters, _CASES[case](), strict=True))
    opts = CompileOptions(
        target=target,
        lnc=1,
        artifacts_dir=str(tmp_path),
        output_path=str(tmp_path / "file.neff"),
    )
    bir = driver.compile_to_bir(
        fn, frontend=TracerFrontend(), inputs=inputs, compile_opts=opts, enable_cache=False
    )
    driver._compile_bir_to_neff(bir, opts, inputs)
    assert (tmp_path / "file.neff").stat().st_size > 0


# ── Static kernels on both physical cores, fp32 and bf16 (#30) ───────────────
#
# Under LNC=2 the static kernels are traced once per physical core and split
# their output tiles between the two traces. Verify both traces for every
# kernel and input dtype, including structures with fewer units than cores
# (one block row: the BSR kernel splits columns instead) and skewed rows.


def _bf16(*shape):
    import ml_dtypes

    return _f32(*shape).astype(ml_dtypes.bfloat16)


def _static_lnc2_cases():
    cases = {}
    for dtype, mk in (("f32", _f32), ("bf16", _bf16)):
        cases[("_bsr_spmm_static_kernel", f"skew-n512-{dtype}")] = lambda mk=mk: (
            mk(6, _P, _P),
            mk(4 * _P, 512),
            (0, 4, 4, 5, 6),
            (0, 1, 2, 3, 0, 3),
        )
        cases[("_bsr_spmm_static_kernel", f"n1024-{dtype}")] = lambda mk=mk: (
            mk(3, _P, _P),
            mk(3 * _P, 1024),
            (0, 2, 2, 3),
            (0, 2, 2),
        )
        cases[("_bsr_spmm_static_t_kernel", f"skew-n512-{dtype}")] = lambda mk=mk: (
            mk(_P, 6 * _P),
            mk(4 * _P, 512),
            (0, 4, 4, 5, 6),
            (0, 1, 2, 3, 0, 3),
        )
        cases[("_bsr_spmm_static_t_kernel", f"long-row-n128-{dtype}")] = lambda mk=mk: (
            mk(_P, 20 * _P),
            mk(20 * _P, 128),
            (0, 20, 20),
            tuple(range(20)),
        )
        cases[("_bsr_spmm_static_kernel", f"onerow-n128-{dtype}")] = lambda mk=mk: (
            mk(2, _P, _P),
            mk(2 * _P, 128),
            (0, 2),
            (0, 1),
        )
        for hd in (64, 128):
            cases[("_attn_fwd_static_kernel", f"hd{hd}-{dtype}")] = lambda hd=hd, mk=mk: (
                mk(hd, 3 * _P),
                mk(hd, 3 * _P),
                mk(3 * _P, hd),
                (0, 2, 2, 5),
                (0, 1, 0, 1, 2),
            )
            cases[("_attn_bwd_static_kernel", f"hd{hd}-{dtype}")] = lambda hd=hd, mk=mk: (
                mk(hd, 3 * _P),
                mk(hd, 3 * _P),
                mk(hd, 3 * _P),
                mk(hd, 3 * _P),
                mk(3 * _P, hd),
                mk(3 * _P, hd),
                mk(3 * _P, hd),
                _f32(3 * _P, 1),
                np.abs(_f32(3 * _P, 1)),
                _f32(3 * _P, 1),
                _f32(1, 3 * _P),
                np.abs(_f32(1, 3 * _P)),
                _f32(1, 3 * _P),
                (0, 2, 2, 5),
                (0, 1, 0, 1, 2),
                (0, 2, 4, 5),
                (0, 2, 0, 2, 2),
                0.125,
            )
    # Keyed entry points: the structure is a registered key (an int constant).
    from trnsparse.nki.kernels import register_structure

    for dtype, mk in (("f32", _f32), ("bf16", _bf16)):
        key = register_structure((0, 4, 4, 5, 6), (0, 1, 2, 3, 0, 3))
        cases[("_bsr_spmm_keyed_kernel", f"skew-n512-{dtype}")] = lambda mk=mk, key=key: (
            mk(6, _P, _P),
            mk(4 * _P, 512),
            key,
        )
        cases[("_bsr_spmm_t_keyed_kernel", f"skew-n512-{dtype}")] = lambda mk=mk, key=key: (
            mk(_P, 6 * _P),
            mk(4 * _P, 512),
            key,
        )
        # 10 rows sharing columns: two row groups, B-tile reuse within each.
        g_rows = [sorted({m % 8, (m + 1) % 8, 0, 3, 5, 7}) for m in range(10)]
        gp = tuple([0] + list(np.cumsum([len(r) for r in g_rows])))
        gc = tuple(c for r in g_rows for c in r)
        gkey = register_structure(gp, gc)
        for w in (128, 512):
            cases[("_bsr_spmm_t_keyed_kernel", f"groups-n{w}-{dtype}")] = (
                lambda mk=mk, gkey=gkey, w=w, nb=len(gc): (mk(_P, nb * _P), mk(8 * _P, w), gkey)
            )
        akey = register_structure((0, 2, 2, 5), (0, 1, 0, 1, 2))
        cases[("_attn_fwd_keyed_kernel", f"hd64-{dtype}")] = lambda mk=mk, akey=akey: (
            mk(64, 3 * _P),
            mk(64, 3 * _P),
            mk(3 * _P, 64),
            akey,
        )
    return cases


_STATIC_LNC2 = _static_lnc2_cases()


@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("case", sorted(_STATIC_LNC2), ids=lambda c: f"{c[0]}-{c[1]}")
def test_static_kernel_lnc_mlir_verifies(case, lnc):
    from trnsparse.nki import kernels

    if case[1].endswith("bf16"):
        pytest.importorskip("ml_dtypes")
    kernel = getattr(kernels, case[0])
    result = source_to_mlir(
        kernel, *_STATIC_LNC2[case](), target="trn2", lnc=lnc, frontend=FrontendKind.TRACER
    )
    result.verify()


@pytest.mark.skipif(shutil.which("neuronx-cc") is None, reason="neuronx-cc not on PATH")
@pytest.mark.parametrize("case", sorted(_STATIC_LNC2), ids=lambda c: f"{c[0]}-{c[1]}")
def test_static_kernel_lnc2_neff_compiles(case, tmp_path):
    """Full CPU NEFF compile of both per-core traces (LNC=2, trn2)."""
    try:
        from nki.compiler import driver
        from nki.compiler.frontend import TracerFrontend
        from nki.compiler.ncc_driver import CompileOptions
    except ImportError as e:
        pytest.skip(f"nki compiler driver API not available: {e}")
    if not hasattr(driver, "_compile_bir_to_neff"):
        pytest.skip("nki.compiler.driver._compile_bir_to_neff not available in this nki")
    if case[1].endswith("bf16"):
        pytest.importorskip("ml_dtypes")

    from trnsparse.nki import kernels

    kernel = getattr(kernels, case[0])
    fn = getattr(kernel, "func", kernel)
    inputs = dict(zip(inspect.signature(fn).parameters, _STATIC_LNC2[case](), strict=True))
    opts = CompileOptions(
        target="trn2",
        lnc=2,
        artifacts_dir=str(tmp_path),
        output_path=str(tmp_path / "file.neff"),
    )
    bir = driver.compile_to_bir(
        fn, frontend=TracerFrontend(), inputs=inputs, compile_opts=opts, enable_cache=False
    )
    driver._compile_bir_to_neff(bir, opts, inputs)
    assert (tmp_path / "file.neff").stat().st_size > 0


# ── Fused SBUF-resident CG (#22) ─────────────────────────────────────────────
#
# Device-side while loop, PSUM dot products, both physical cores at LNC=2
# (rows split between the cores, same control flow on each, as neuronx-cc
# requires). "stream" shrinks the SBUF budget so most of A streams from HBM
# inside the iteration; "rhs*" solve several right-hand sides at once;
# "nosplit" runs the whole solve on both cores.


def _cg_inputs(T: int, precond: int, check_every: int, R: int = 1, sbuf_kib=160, split=1):
    from trnsparse.nki.kernels import register_structure

    rp, cl = [0], []
    for m in range(T):
        cl += sorted({m, (m + 1) % T, (m - 1) % T})
        rp.append(len(cl))
    key = register_structure(rp, cl)
    minv = _f32(_P, T * _P) if precond else _f32(_P, _P)
    state = _f32(_P, 2 * T * R + 2)
    return (_f32(_P, len(cl) * _P), minv, state, key, check_every, precond, sbuf_kib, split)


_CG_CASES = {
    "plain": (dict(T=3, precond=0, check_every=2),),
    "bjacobi": (dict(T=3, precond=1, check_every=2),),
    "stream": (dict(T=5, precond=1, check_every=1, sbuf_kib=2),),
    "nosplit": (dict(T=4, precond=1, check_every=2, split=0),),
    "one-tile": (dict(T=1, precond=0, check_every=3),),
    "rhs4": (dict(T=3, precond=1, check_every=2, R=4),),
    # 3 x 200 columns: the matvec spans two PSUM banks
    "rhs200-stream": (dict(T=3, precond=0, check_every=1, R=200, sbuf_kib=2),),
}


@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("case", sorted(_CG_CASES))
def test_cg_kernel_lnc_mlir_verifies(case, lnc):
    from trnsparse.nki import kernels

    (kw,) = _CG_CASES[case]
    result = source_to_mlir(
        kernels._cg_bsr_keyed_kernel,
        *_cg_inputs(**kw),
        target="trn2",
        lnc=lnc,
        frontend=FrontendKind.TRACER,
    )
    result.verify()


@pytest.mark.skipif(shutil.which("neuronx-cc") is None, reason="neuronx-cc not on PATH")
@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("case", ["plain", "bjacobi", "rhs4", "nosplit"])
def test_cg_kernel_neff_compiles(case, lnc, tmp_path):
    """Full CPU NEFF compile. At LNC=2 this is what rejects per-core programs
    whose control flow differs (NCC_IXGM002), which MLIR verify does not see."""
    try:
        from nki.compiler import driver
        from nki.compiler.frontend import TracerFrontend
        from nki.compiler.ncc_driver import CompileOptions
    except ImportError as e:
        pytest.skip(f"nki compiler driver API not available: {e}")
    if not hasattr(driver, "_compile_bir_to_neff"):
        pytest.skip("nki.compiler.driver._compile_bir_to_neff not available in this nki")

    from trnsparse.nki import kernels

    fn = getattr(kernels._cg_bsr_keyed_kernel, "func", kernels._cg_bsr_keyed_kernel)
    (kw,) = _CG_CASES[case]
    inputs = dict(zip(inspect.signature(fn).parameters, _cg_inputs(**kw), strict=True))
    opts = CompileOptions(
        target="trn2",
        lnc=lnc,
        artifacts_dir=str(tmp_path),
        output_path=str(tmp_path / "file.neff"),
    )
    bir = driver.compile_to_bir(
        fn, frontend=TracerFrontend(), inputs=inputs, compile_opts=opts, enable_cache=False
    )
    driver._compile_bir_to_neff(bir, opts, inputs)
    assert (tmp_path / "file.neff").stat().st_size > 0
