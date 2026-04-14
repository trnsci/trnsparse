"""Simulator-backed kernel correctness tests (NKI 0.3.0 Stable).

Run with `TRNSPARSE_USE_SIMULATOR=1` on any x86_64 Linux host that has
`nki>=0.3.0` installed. Bypasses torch_xla + NEFF compile; routes
kernel dispatch through `nki.simulate(kernel)(np_args)`.

Intentionally curated to small shapes — the CPU simulator is slow at
1024³ and above. Correctness parity with hardware at these scales is
what we're verifying, not perf.
"""

from __future__ import annotations

import os

import pytest
import torch

import trnsparse

pytestmark = pytest.mark.nki_simulator


@pytest.fixture(autouse=True)
def _simulator_enabled():
    """Skip the whole module if TRNSPARSE_USE_SIMULATOR isn't set.

    The marker alone isn't sufficient — users may `pytest -m
    nki_simulator` on a host where nki isn't importable or the env
    var hasn't been set. Fail loudly vs silently falling back.
    """
    if os.environ.get("TRNSPARSE_USE_SIMULATOR", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("TRNSPARSE_USE_SIMULATOR=1 required")

    from trnsparse.nki.dispatch import HAS_NKI

    if not HAS_NKI:
        pytest.skip("nki package not importable on this host")


ATOL, RTOL = 1e-3, 1e-4


@pytest.fixture
def nki_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


class TestCsrSpmmSimulator:
    """CSR SpMM path through the simulator. Mirrors the hardware tests
    in `test_nki_spmm.py` at smaller shapes that fit the CPU simulator
    comfortably.
    """

    def test_aligned_128(self, nki_backend):
        torch.manual_seed(0)
        A_dense = torch.randn(128, 128)
        A_dense[torch.abs(A_dense) < 0.5] = 0.0
        A = trnsparse.from_dense(A_dense)
        B = torch.randn(128, 64)

        got = trnsparse.spmm(A, B)
        torch.testing.assert_close(got, A_dense @ B, atol=ATOL, rtol=RTOL)

    def test_rectangular(self, nki_backend):
        torch.manual_seed(1)
        A_dense = torch.randn(256, 128)
        A_dense *= (torch.rand(256, 128) < 0.1).float()
        A = trnsparse.from_dense(A_dense)
        B = torch.randn(128, 32)

        got = trnsparse.spmm(A, B)
        torch.testing.assert_close(got, A_dense @ B, atol=ATOL, rtol=RTOL)


class TestBsrSpmmSimulator:
    """BSR SpMM through the simulator. Block size 128 matches the Tensor
    Engine tile; we pick a minimal (2, 2) block grid so the simulator
    runs in seconds rather than minutes.
    """

    def test_block_dense_2x2(self, nki_backend):
        torch.manual_seed(2)
        b = 128
        # Fill both blocks in each row — exercises the non-zero-block path.
        A_dense = torch.randn(2 * b, 2 * b)
        bsr = trnsparse.BSRMatrix.from_dense(A_dense, block_size=b)
        assert bsr.n_blocks == 4

        B = torch.randn(2 * b, 64)
        got = trnsparse.bsr_spmm(bsr, B)
        torch.testing.assert_close(got, A_dense @ B, atol=ATOL, rtol=RTOL)

    def test_block_sparse_2x2(self, nki_backend):
        """Drop one block to exercise the zero-block-padding path."""
        torch.manual_seed(3)
        b = 128
        A_dense = torch.zeros(2 * b, 2 * b)
        A_dense[:b, :b] = torch.randn(b, b)
        A_dense[b:, b:] = torch.randn(b, b)  # block-diagonal
        bsr = trnsparse.BSRMatrix.from_dense(A_dense, block_size=b)
        assert bsr.n_blocks == 2

        B = torch.randn(2 * b, 32)
        got = trnsparse.bsr_spmm(bsr, B)
        torch.testing.assert_close(got, A_dense @ B, atol=ATOL, rtol=RTOL)
