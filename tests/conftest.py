"""Test configuration."""

import atexit
import os
import shutil
import tempfile

import pytest

# NKI trace cache (aws-neuron/aws-neuron-sdk#1454, trnsci/trnsci#35).
# `import torch_neuronx` turns the cache on (NKI_ENABLE_TRACE_CACHE=1), so
# that is how users run these kernels, and the neuron tests run the same way.
# The cache lives on disk and is versioned by wheel .dist-info, which an
# editable install or source checkout does not change, so a shared cache could
# serve a trace of an older kernel. Each session therefore gets a fresh,
# empty cache directory (NKI_TRACE_CACHE_URL), removed at exit.
# tests/test_trace_cache_safety.py checks that no kernel depends on a value
# the cache key misses (closure or global). Overrides: NKI_ENABLE_TRACE_CACHE=0
# runs with the cache off; an explicit NKI_TRACE_CACHE_URL is kept as is.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "1")
if os.environ["NKI_ENABLE_TRACE_CACHE"] != "0" and "NKI_TRACE_CACHE_URL" not in os.environ:
    _trace_cache_dir = tempfile.mkdtemp(prefix="trnsparse-nki-trace-cache-")
    os.environ["NKI_TRACE_CACHE_URL"] = _trace_cache_dir
    atexit.register(shutil.rmtree, _trace_cache_dir, ignore_errors=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")
    config.addinivalue_line(
        "markers",
        "nki_simulator: runs NKI kernels via nki.simulate on CPU "
        "(requires TRNSPARSE_USE_SIMULATOR=1 + nki>=0.3.0)",
    )


def pytest_addoption(parser):
    # tests/test_conformance.py runs its reference cases on the selected
    # backend and device, so a Trn2 run executes the same cases as CPU CI:
    #   pytest tests/test_conformance.py --backend nki --conformance-device neuron
    group = parser.getgroup("trnsparse conformance")
    group.addoption(
        "--backend",
        default="pytorch",
        choices=("pytorch", "nki", "auto"),
        help="trnsparse backend for tests/test_conformance.py (default: pytorch)",
    )
    group.addoption(
        "--conformance-device",
        default="cpu",
        help="device for the dense operands in tests/test_conformance.py (default: cpu)",
    )


_FP64_POLICY_MARK = "Trainium computes float64 at fp32 precision"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Under TRNSPARSE_REQUIRE_NKI, float64 / complex128 kernel calls raise
    # NotImplementedError by design (the fp64 policy, #84). The conformance
    # suite builds most references in float64, so report those cases as
    # skips there rather than failures (as needs_pytorch_block does for #46).
    outcome = yield
    report = outcome.get_result()
    if (
        call.excinfo is not None
        and report.failed
        and item.path.name == "test_conformance.py"
        and os.environ.get("TRNSPARSE_REQUIRE_NKI", "").lower() in ("1", "true", "yes")
        and call.excinfo.errisinstance(NotImplementedError)
        and _FP64_POLICY_MARK in str(call.excinfo.value)
    ):
        report.outcome = "skipped"
        report.longrepr = (
            str(item.path),
            item.location[1] or 0,
            "Skipped: float64 raises under TRNSPARSE_REQUIRE_NKI by design (#84)",
        )
