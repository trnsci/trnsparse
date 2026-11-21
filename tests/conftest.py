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
    group.addoption(
        "--conformance-json",
        default=None,
        metavar="PATH",
        help="write the device conformance summary (passed, xfailed/XPASS by category, "
        "NEW failures) as JSON",
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


# ---------------------------------------------------------------- device conformance summary
# Device conformance runs (``--backend nki --conformance-device neuron``) mark
# known platform limits ``xfail(strict=False)`` at run time, with a reason that
# starts ``[device-known:<category>]`` (see DEVICE_KNOWN in
# tests/test_conformance.py). On a non-CPU device run this prints passed,
# xfailed and XPASS by category, and the NEW failures (anything not expected).
# ``--conformance-json PATH`` also writes it as JSON; merge the JSON from
# chunked runs with ``python scripts/conformance_expect.py run1.json ...``.

_KNOWN_PREFIX = "[device-known:"


def _expect_category(reason):
    reason = reason or ""
    if reason.startswith(_KNOWN_PREFIX):
        return reason[len(_KNOWN_PREFIX) :].split("]", 1)[0]
    return "other xfail"


def _expect_summary(stats):
    def ids(key):
        return [
            r.nodeid for r in stats.get(key, []) if getattr(r, "when", "call") in ("call", "setup")
        ]

    by_cat = {}
    for key in ("xfailed", "xpassed"):
        for r in stats.get(key, []):
            cat = _expect_category(getattr(r, "wasxfail", ""))
            by_cat.setdefault(key, {}).setdefault(cat, []).append(r.nodeid)
    return {
        "passed": len(ids("passed")),
        "skipped": len(stats.get("skipped", [])),
        "xfailed": by_cat.get("xfailed", {}),
        "xpassed": by_cat.get("xpassed", {}),
        "new_failures": ids("failed") + [r.nodeid for r in stats.get("error", [])],
    }


def _print_expect_summary(write, s):
    write("conformance expectations (device run)")
    write(f"  passed: {s['passed']}   skipped: {s['skipped']}")
    for key, label in (("xfailed", "xfailed"), ("xpassed", "XPASS (known category now passes)")):
        total = sum(len(v) for v in s[key].values())
        write(f"  {label}: {total}")
        for cat, nodeids in sorted(s[key].items()):
            write(f"    {cat}: {len(nodeids)}")
    write(f"  NEW failures: {len(s['new_failures'])}")
    for nodeid in s["new_failures"]:
        write(f"    {nodeid}")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    path = config.getoption("--conformance-json", None)
    if config.getoption("--conformance-device", "cpu") == "cpu" and not path:
        return
    s = _expect_summary(terminalreporter.stats)
    _print_expect_summary(lambda line: terminalreporter.write_line(line), s)
    if path:
        import json

        with open(path, "w") as fh:
            json.dump(s, fh, indent=1)
