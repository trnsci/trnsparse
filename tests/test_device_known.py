"""The device-known category rules of tests/test_conformance.py (trnsci/trnsci#35).

CPU only: checks that representative conformance IDs map to their category
(and others to None) without a device, and that the conftest summary groups
xfail reasons by category.
"""

from types import SimpleNamespace

import pytest
import torch
from conftest import _expect_category, _expect_summary
from test_conformance import DEVICE_KNOWN, _device_known_category

c64, c128, f16, f32, f64 = (
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.float32,
    torch.float64,
)


def node(name, **params):
    return SimpleNamespace(originalname=name, name=name, callspec=SimpleNamespace(params=params))


@pytest.mark.parametrize(
    "item,cat",
    [
        (node("test_spmv", shape=(127, 129), dtype=c64), "complex-mv"),
        (node("test_spmv", shape=(1000, 1000), dtype=c128), "complex-mv"),
        (node("test_spmv", shape=(0, 5), dtype=c128), None),
        (node("test_spmv", shape=(127, 129), dtype=f64), None),
        (node("test_spmv_alpha_beta", alpha=1.0, beta=0.0, y_kind="vec", dtype=c128), "complex-mv"),
        (node("test_spmv_symmetric", n=1, uplo="upper", store="full", dtype=c128), "complex-mv"),
        (node("test_spmv_symmetric", n=1, uplo="upper", store="full", dtype=f32), None),
        (node("test_coo_duplicates_sum", dtype=c64), "complex-mv"),
        (node("test_cg_bsr", n=64, b=8, dtype=f64, precond="none"), "fp64-at-fp32"),
        (node("test_cg_bsr", n=64, b=8, dtype=f64, precond="jacobi_callable"), "fp64-at-fp32"),
        (node("test_cg_bsr", n=64, b=8, dtype=f32, precond="none"), None),
        (
            node("test_cg_bsr", n=1, b=1, dtype=f32, precond="dense_inverse"),
            "harness-dense-inverse",
        ),
        (node("test_power_iteration", dtype=f64, fused=True), "fp64-at-fp32"),
        (node("test_power_iteration", dtype=f32, fused=True), None),
        (SimpleNamespace(name="test_cg_bsr_breakdown_iteration_count"), "fp64-at-fp32"),
        (SimpleNamespace(name="test_chebyshev_bsr_fixed_k_residual"), "fp64-at-fp32"),
        (node("test_screened_spmm_lowp", dtype=f16), "screened-fp16-mask"),
        (node("test_screened_spmm_lowp", dtype=torch.bfloat16), None),
        (SimpleNamespace(name="test_noncanonical_csr"), None),
    ],
)
def test_category_rules(item, cat):
    got = _device_known_category(item)
    assert got == cat
    assert got is None or got in DEVICE_KNOWN


def test_summary_groups_by_category():
    def rep(nodeid, reason=None):
        return SimpleNamespace(nodeid=nodeid, when="call", wasxfail=reason)

    assert _expect_category("[device-known:complex-mv] trnsci/trnsparse#130: x") == "complex-mv"
    assert _expect_category("trnsci/trnsparse#70: autograd") == "other xfail"
    s = _expect_summary(
        {
            "passed": [rep("a"), rep("b")],
            "xfailed": [rep("c", "[device-known:fp64-at-fp32] #137: y"), rep("d", "plain")],
            "xpassed": [rep("e", "[device-known:complex-mv] #130: x")],
            "failed": [rep("f")],
        }
    )
    assert s["passed"] == 2
    assert s["xfailed"] == {"fp64-at-fp32": ["c"], "other xfail": ["d"]}
    assert s["xpassed"] == {"complex-mv": ["e"]}
    assert s["new_failures"] == ["f"]
