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
