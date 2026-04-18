"""Hardware tests for the NKI attention kernels (v0.4.4 forward + v0.5.1 backward, #25).

Run on a trn1 instance:
    AWS_PROFILE=aws pytest tests/test_nki_attn.py -m neuron -v

Forward tests validate parity between the NKI kernel pair
(`_attn_stats_kernel` + `_attn_out_kernel`) and the PyTorch reference.
Backward tests validate parity between the NKI backward kernel pair
(`_attn_bwd_dq_kernel` + `_attn_bwd_dkdv_kernel`) and the PyTorch
autograd backward.
"""

from __future__ import annotations

import pytest
import torch

import trnsparse

pytestmark = pytest.mark.neuron


@pytest.fixture
def nki_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


def _local_mask(seq_len: int, block_size: int, window: int) -> torch.Tensor:
    n_blocks = seq_len // block_size
    bm = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    for i in range(n_blocks):
        lo = max(0, i - window)
        hi = min(n_blocks, i + window + 1)
        bm[i, lo:hi] = True
    return bm.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)


def _dilated_mask(seq_len: int, block_size: int, stride: int) -> torch.Tensor:
    n_blocks = seq_len // block_size
    bm = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    for i in range(n_blocks):
        for j in range(n_blocks):
            if (i - j) % stride == 0:
                bm[i, j] = True
    return bm.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)


def _global_token_mask(seq_len: int, block_size: int, window: int, n_global: int) -> torch.Tensor:
    n_blocks = seq_len // block_size
    bm = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    bm[:n_global, :] = True
    bm[:, :n_global] = True
    for i in range(n_global, n_blocks):
        lo = max(0, i - window)
        hi = min(n_blocks, i + window + 1)
        bm[i, lo:hi] = True
    return bm.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)


ATOL, RTOL = 1e-3, 1e-4

SEQ_LEN = 512
HEAD_DIM = 64
BLOCK_SIZE = 128


class TestAttnTiledHardware:
    """NKI attention kernel pair on trn1 hardware."""

    def _run(self, mask: torch.Tensor):
        seq_len, head_dim = SEQ_LEN, HEAD_DIM
        Q = torch.randn(seq_len, head_dim)
        K = torch.randn(seq_len, head_dim)
        V = torch.randn(seq_len, head_dim)

        mask_bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=BLOCK_SIZE)

        trnsparse.set_backend("pytorch")
        ref = trnsparse.block_sparse_attention_tiled(Q, K, V, mask_bsr)

        trnsparse.set_backend("nki")
        got = trnsparse.block_sparse_attention_tiled(Q, K, V, mask_bsr)

        torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL)

    def test_local_window_parity(self):
        torch.manual_seed(30)
        self._run(_local_mask(SEQ_LEN, BLOCK_SIZE, window=2))

    def test_dilated_parity(self):
        torch.manual_seed(31)
        self._run(_dilated_mask(SEQ_LEN, BLOCK_SIZE, stride=2))

    def test_global_token_parity(self):
        torch.manual_seed(32)
        self._run(_global_token_mask(SEQ_LEN, BLOCK_SIZE, window=2, n_global=2))


class TestAttnBwdHardware:
    """NKI backward kernel pair on trn1 hardware (v0.5.1).

    Verifies that dQ, dK, dV from the NKI kernel pair match the PyTorch
    backward at atol=1e-3. Uses seq_len=512, head_dim=64 (realistic shapes).
    """

    def _pytorch_grads(self, Q, K, V, mask_bsr, seed):
        torch.manual_seed(seed)
        prev = trnsparse.get_backend()
        trnsparse.set_backend("pytorch")
        try:
            Qr = Q.clone().requires_grad_(True)
            Kr = K.clone().requires_grad_(True)
            Vr = V.clone().requires_grad_(True)
            out = trnsparse.block_sparse_attention_tiled(Qr, Kr, Vr, mask_bsr)
            dO = torch.randn_like(out)
            out.backward(dO)
            return Qr.grad.detach(), Kr.grad.detach(), Vr.grad.detach(), dO
        finally:
            trnsparse.set_backend(prev)

    def _run_bwd(self, mask: torch.Tensor, seed: int):
        Q = torch.randn(SEQ_LEN, HEAD_DIM)
        K = torch.randn(SEQ_LEN, HEAD_DIM)
        V = torch.randn(SEQ_LEN, HEAD_DIM)
        mask_bsr = trnsparse.BSRMatrix.from_dense(mask.float(), block_size=BLOCK_SIZE)

        dQ_ref, dK_ref, dV_ref, dO = self._pytorch_grads(Q, K, V, mask_bsr, seed)

        trnsparse.set_backend("nki")
        Qr = Q.clone().requires_grad_(True)
        Kr = K.clone().requires_grad_(True)
        Vr = V.clone().requires_grad_(True)
        out = trnsparse.block_sparse_attention_tiled(Qr, Kr, Vr, mask_bsr)
        out.backward(dO)

        torch.testing.assert_close(Qr.grad, dQ_ref, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(Kr.grad, dK_ref, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(Vr.grad, dV_ref, atol=ATOL, rtol=RTOL)

    def test_bwd_local_parity(self):
        torch.manual_seed(40)
        self._run_bwd(_local_mask(SEQ_LEN, BLOCK_SIZE, window=2), seed=40)

    def test_bwd_dilated_parity(self):
        torch.manual_seed(41)
        self._run_bwd(_dilated_mask(SEQ_LEN, BLOCK_SIZE, stride=2), seed=41)
