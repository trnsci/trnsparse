"""Host-side gather for the NKI attention kernels (CPU, no nki needed).

Regression tests for two #29 bugs the simulator tests did not exercise:
the dQ gather fed dO by column block instead of row block, and padded
k-slots (rows with fewer than K_max blocks) gathered block 0 and were
counted in the softmax.
"""

from __future__ import annotations

import torch

import trnsparse
from trnsparse.nki.dispatch import _attn_bwd_gather, _attn_gather, _attn_slot_valid

B = 128


def _local_bsr(n_blocks: int, window: int):
    bm = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    for i in range(n_blocks):
        bm[i, max(0, i - window) : i + window + 1] = True
    dense = bm.repeat_interleave(B, 0).repeat_interleave(B, 1).float()
    return trnsparse.BSRMatrix.from_dense(dense, block_size=B)


def test_bwd_gather_do_is_row_block():
    torch.manual_seed(0)
    S, H = 4 * B, 64
    Q, K, V, dO, O = (torch.randn(S, H) for _ in range(5))
    mb = _local_bsr(4, window=1)
    stats = torch.zeros(4, B)
    rf, _ = _attn_bwd_gather(Q, K, V, dO, O, mb, H**-0.5, stats, stats + 1)
    do_g = rf["do_gathered"]
    for m in range(4):
        for slot in range(do_g.shape[1]):
            torch.testing.assert_close(do_g[m, slot], dO[m * B : (m + 1) * B])


def test_padded_slots_zeroed():
    torch.manual_seed(1)
    S, H = 4 * B, 64
    Q, K, V = (torch.randn(S, H) for _ in range(3))
    mb = _local_bsr(4, window=2)  # rows hold 3, 4, 4, 3 blocks -> K_max = 4
    _, kg, vg, K_max, M_tiles = _attn_gather(Q, K, V, mb, H**-0.5)
    valid = _attn_slot_valid(mb, M_tiles, K_max)
    assert K_max == 4
    assert valid.sum(dim=1).tolist() == [3, 4, 4, 3]
    assert (kg[~valid] == 0).all() and (vg[~valid] == 0).all()
