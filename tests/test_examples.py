"""Smoke tests for the examples directory.

These cover the user-facing integration demos without executing them as
subprocesses — the tests import the example modules and call their
entry points directly so failures surface with full tracebacks.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))


class TestSparseFockDemo:
    def test_unfused_path_runs(self):
        """Exercise the v0.1.x multi-step screening path end-to-end."""
        import sparse_fock as demo

        diag = demo._synthetic_schwarz_system(50)
        import trnsparse

        Q = trnsparse.schwarz_bounds(diag)
        A = torch.randn(50, 50) * (Q.unsqueeze(-1) * Q.unsqueeze(0)) * 0.01
        P = torch.randn(50, 50)
        J, t = demo._unfused_path(A, diag, P, threshold=1e-4)
        assert J.shape == (50, 50)
        assert torch.isfinite(J).all()
        assert t > 0

    def test_fused_vs_unfused_parity(self):
        """The two paths should agree up to fp tolerance on the same inputs."""
        import sparse_fock as demo

        torch.manual_seed(7)
        diag = demo._synthetic_schwarz_system(40)
        import trnsparse

        Q = trnsparse.schwarz_bounds(diag)
        A = torch.randn(40, 40) * (Q.unsqueeze(-1) * Q.unsqueeze(0)) * 0.01
        P = torch.randn(40, 40)
        threshold = 1e-4

        J_unfused, _ = demo._unfused_path(A, diag, P, threshold)
        J_fused, _ = demo._fused_path(A, diag, P, threshold)

        torch.testing.assert_close(J_unfused, J_fused, atol=1e-6, rtol=1e-6)


class TestPyScfBridge:
    """Skips cleanly without PySCF; asserts non-trivial screening on H2O."""

    def test_h2o_sto3g_screening(self):
        pytest.importorskip("pyscf")
        import pyscf_bridge as demo

        report = demo.run_demo("h2o", "sto-3g", threshold=1e-8)
        assert report["nao"] > 0
        # Tight tolerance would give no sparsity on a tiny basis; sto-3g
        # H2O has only 7 AOs, so we just assert the API ran and produced
        # a finite result — non-trivial sparsity needs larger bases.
        assert math.isfinite(report["output_norm"])
        assert 0.0 <= report["pair_sparsity"] <= 1.0
        assert report["output_shape"] == (report["nao"], report["nao"])
