"""Hardware tests for the NKI two-pass attention kernel pair (v0.4.4, #25).

Run on a trn1 instance:
    AWS_PROFILE=aws pytest tests/test_nki_attn.py -m neuron -v

These tests validate parity between the NKI kernel pair
(`_attn_stats_kernel` + `_attn_out_kernel`) and the PyTorch reference
(`block_sparse_attention_tiled` on pytorch backend) at realistic shapes.
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
