"""Test configuration."""

import pytest  # noqa: F401 — imported for marker side-effect exposure


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")
    config.addinivalue_line(
        "markers",
        "nki_simulator: runs NKI kernels via nki.simulate on CPU "
        "(requires TRNSPARSE_USE_SIMULATOR=1 + nki>=0.3.0)",
    )
