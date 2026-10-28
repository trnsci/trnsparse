"""
Native TorchNeuron dispatch on hardware (Trn2/Trn3, torch-neuronx 2.14+).

Skipped unless the dispatch path resolves to "native": these do not run on
the XLA lineage (Trn1) or off-hardware. The existing hardware suites
(test_nki_spmm, test_nki_bsr, test_nki_attn, ...) also exercise the native
path automatically on a TorchNeuron host, with host-resident operands; this
file covers what is specific to it: custom-op registration, no torch_xla,
host CSR structure against `device="neuron"` dense operands (the neuron
device has no sparse CSR, aws-neuron-sdk#1442), register_autograd
backwards, and torch.compile.

    NKI_ENABLE_TRACE_CACHE=0 TRNSPARSE_REQUIRE_NKI=1 pytest tests/test_nki_native.py -v -m neuron
"""

import subprocess
import sys

import pytest
import torch

import trnsparse
from trnsparse.nki import HAS_NKI
from trnsparse.nki.dispatch import _get_dispatch_path, _get_native_ops, _use_simulator


def _native() -> bool:
    try:
        return HAS_NKI and not _use_simulator() and _get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

DEV = "neuron"
P = 128
ATOL, RTOL = 1e-3, 1e-4


@pytest.fixture(autouse=True)
def nki_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


def _random_csr(M, K, density, seed=0):
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(M, K, generator=g) * (torch.rand(M, K, generator=g) < density)
    return dense, trnsparse.from_dense(dense)  # host CSR


def _block_causal_mask(seq_len):
    nb = seq_len // P
    pattern = torch.zeros(seq_len, seq_len)
    for i in range(nb):
        for j in range(i + 1):
            pattern[i * P : (i + 1) * P, j * P : (j + 1) * P] = 1.0
    return trnsparse.BSRMatrix.from_dense(pattern, block_size=P)


class TestNativeRegistration:
    def test_native_path_never_imports_torch_xla(self):
        code = (
            "import sys, torch, trnsparse; "
            "from trnsparse.nki.dispatch import _get_dispatch_path; "
            "trnsparse.set_backend('nki'); "
            "A = trnsparse.from_dense(torch.eye(128)); "
            "C = trnsparse.spmm(A, torch.ones(128, 4, device='neuron')); "
            "assert C.device.type == 'neuron'; "
            "assert _get_dispatch_path() == 'native'; "
            "assert 'torch_xla' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_ops_registered_in_torch_ops(self):
        _get_native_ops()
        for name in (
            "spmm_dense",
            "bsr_spmm",
            "screened_spmm",
            "attn_stats",
            "attn_out",
            "attn_bwd_dq",
            "attn_bwd_dkdv",
            "bsr_attn",
            "bsr_attn_bwd",
        ):
            assert hasattr(torch.ops.trnsparse, name), name


class TestHostStructureDeviceOperands:
    @pytest.mark.parametrize("M,K,N", [(256, 256, 128), (200, 137, 64), (4097, 300, 8)])
    def test_spmm_host_csr_device_b(self, M, K, N):
        dense, A = _random_csr(M, K, 0.05, seed=M)
        B = torch.randn(K, N)
        C = trnsparse.spmm(A, B.to(DEV))
        assert C.device.type == DEV
        torch.testing.assert_close(C.cpu(), dense @ B, atol=ATOL, rtol=RTOL)

    def test_spmm_grad_host_values_device_b(self):
        dense, A = _random_csr(256, 192, 0.05, seed=1)
        A.values.requires_grad_(True)
        B = torch.randn(192, 64).to(DEV).requires_grad_(True)
        trnsparse.spmm(A, B).sum().backward()
        assert B.grad.device.type == DEV
        dense_ref = dense.clone().requires_grad_(True)
        B_ref = B.detach().cpu().requires_grad_(True)
        (dense_ref @ B_ref).sum().backward()
        torch.testing.assert_close(B.grad.cpu(), B_ref.grad, atol=ATOL, rtol=RTOL)
        rows = torch.repeat_interleave(torch.arange(256), A.row_ptrs[1:] - A.row_ptrs[:-1])
        torch.testing.assert_close(
            A.values.grad, dense_ref.grad[rows, A.col_indices], atol=ATOL, rtol=RTOL
        )

    def test_spmv_host_csr_device_x(self):
        dense, A = _random_csr(300, 300, 0.05, seed=2)
        x = torch.randn(300)
        y = trnsparse.spmv(A, x.to(DEV))
        assert y.device.type == DEV
        torch.testing.assert_close(y.cpu(), dense @ x, atol=ATOL, rtol=RTOL)

    def test_bsr_spmm_device_blocks_host_structure_and_grad(self):
        g = torch.Generator().manual_seed(3)
        dense = torch.randn(256, 384, generator=g)
        dense[128:, :256] = 0
        A = trnsparse.BSRMatrix.from_dense(dense, block_size=P)
        A.blocks = A.blocks.to(DEV).requires_grad_(True)  # dense blocks on device
        B = torch.randn(384, 32, generator=g)
        B_d = B.to(DEV).requires_grad_(True)
        C = trnsparse.bsr_spmm(A, B_d)
        assert C.device.type == DEV
        torch.testing.assert_close(C.detach().cpu(), dense @ B, atol=ATOL, rtol=RTOL)
        C.sum().backward()
        torch.testing.assert_close(
            B_d.grad.cpu(), dense.T @ torch.ones(256, 32), atol=ATOL, rtol=RTOL
        )
        assert A.blocks.grad is not None and A.blocks.grad.device.type == DEV

    def test_screened_spmm_device_operands_and_grad(self):
        g = torch.Generator().manual_seed(4)
        Am = torch.randn(200, 200, generator=g)
        diag = torch.rand(200, generator=g)
        B = torch.randn(200, 48, generator=g)
        A_d = Am.to(DEV).requires_grad_(True)
        C = trnsparse.screened_spmm(A_d, diag.to(DEV), B.to(DEV), threshold=0.2)
        Q = diag.sqrt()
        mask = ((Q[:, None] * Q[None, :]) > 0.2**0.5).float()
        torch.testing.assert_close(C.detach().cpu(), (Am * mask) @ B, atol=ATOL, rtol=RTOL)
        C.sum().backward()
        torch.testing.assert_close(
            A_d.grad.cpu(), (torch.ones(200, 48) @ B.T) * mask, atol=ATOL, rtol=RTOL
        )

    def test_attention_device_qkv_forward_and_backward(self):
        from trnsparse.ops import _block_sparse_attn_backward, _block_sparse_attn_pytorch

        torch.manual_seed(5)
        seq_len, hd = 3 * P, 64
        mask = _block_causal_mask(seq_len)
        Q, K, V = (torch.randn(seq_len, hd) for _ in range(3))
        Qd, Kd, Vd = (t.to(DEV).requires_grad_(True) for t in (Q, K, V))
        out = trnsparse.block_sparse_attention_tiled(Qd, Kd, Vd, mask)
        assert out.device.type == DEV
        scale = hd**-0.5
        ref, rm, rd = _block_sparse_attn_pytorch(Q, K, V, mask, scale, return_stats=True)
        torch.testing.assert_close(out.detach().cpu(), ref, atol=ATOL, rtol=RTOL)

        dO = torch.randn(seq_len, hd)
        out.backward(dO.to(DEV))
        dQ, dK, dV = _block_sparse_attn_backward(
            Q, K, V, ref, dO, mask, scale, row_max=rm, row_denom=rd
        )
        for got, exp in ((Qd.grad, dQ), (Kd.grad, dK), (Vd.grad, dV)):
            torch.testing.assert_close(got.cpu(), exp, atol=1e-2, rtol=1e-3)

    def test_cg_bsr_device_rhs(self):
        n = 256
        g = torch.Generator().manual_seed(6)
        M = torch.randn(n, n, generator=g)
        S = M @ M.T / n + torch.eye(n) * 4.0
        A = trnsparse.BSRMatrix.from_dense(S, block_size=P)
        b = torch.randn(n, generator=g)
        x, iters, _ = trnsparse.cg_bsr(A, b.to(DEV), tol=1e-5, max_iter=200)
        assert x.device.type == DEV
        torch.testing.assert_close(x.cpu(), torch.linalg.solve(S, b), atol=1e-3, rtol=1e-3)


class TestTorchCompile:
    def test_spmm_dense_op_fullgraph(self):
        spmm_dense = _get_native_ops()["spmm_dense"]
        a, b = torch.randn(256, 256), torch.randn(256, 128)

        @torch.compile(backend="neuron", fullgraph=True)
        def f(a, b):
            return spmm_dense(a, b) * 2.0

        out = f(a.to(DEV), b.to(DEV)).cpu()
        torch.testing.assert_close(out, (a @ b) * 2.0, atol=ATOL, rtol=RTOL)

    def test_spmm_host_csr_structure_fullgraph(self):
        # The B6 compile-coverage rows failed because their CSR inputs were
        # device tensors. Here the CSR (structure and values) stays on the
        # host and only B is a neuron tensor; densify, pad, kernel op and
        # slice are all in one graph.
        dense, A = _random_csr(256, 256, 0.05, seed=7)
        B = torch.randn(256, 64)

        @torch.compile(backend="neuron", fullgraph=True, dynamic=False)
        def f(B):
            return trnsparse.spmm(A, B)

        out = f(B.to(DEV))
        assert out.device.type == DEV
        torch.testing.assert_close(out.cpu(), dense @ B, atol=ATOL, rtol=RTOL)

    def test_spmm_host_csr_structure_passed_as_inputs(self):
        # Same as B6's spmm(CSR, B) row, with the CSR component tensors
        # (on the host) as graph inputs and B on the device.
        dense, A = _random_csr(256, 256, 0.05, seed=8)
        B = torch.randn(256, 64)

        @torch.compile(backend="neuron", fullgraph=True, dynamic=False)
        def f(v, ci, rp, B):
            return trnsparse.spmm(
                trnsparse.CSRMatrix(values=v, col_indices=ci, row_ptrs=rp, shape=(256, 256)), B
            )

        out = f(A.values, A.col_indices, A.row_ptrs, B.to(DEV))
        torch.testing.assert_close(out.cpu(), dense @ B, atol=ATOL, rtol=RTOL)
