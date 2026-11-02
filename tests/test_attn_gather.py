"""Host-side gather for the NKI attention kernels (CPU, no nki needed).

Regression tests for two #29 bugs the simulator tests did not exercise:
the dQ gather fed dO by column block instead of row block, and padded
k-slots (rows with fewer than K_max blocks) gathered block 0 and were
counted in the softmax.
"""

from __future__ import annotations

import pytest
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


# ── #49: bf16 operands and head_dim padding for the gather kernels ───────────


@pytest.fixture
def no_simulator(monkeypatch):
    """Hardware head_dim rules (the NKI 0.3.0 simulator also pads head_dim < 128)."""
    from trnsparse.nki import dispatch

    monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)


def test_kernel_head_dim_pads_to_whole_k_tiles(no_simulator):
    from trnsparse.nki.dispatch import _attn_kernel_head_dim

    assert [_attn_kernel_head_dim(h) for h in (64, 128, 192, 256, 320, 512)] == [
        64,
        128,
        256,
        256,
        384,
        512,
    ]


def test_operands_share_one_dtype():
    from trnsparse.nki.dispatch import _attn_operands, _kernel_dtype

    bf, f32, f64 = (
        torch.zeros(2, 2, dtype=d) for d in (torch.bfloat16, torch.float32, torch.float64)
    )
    assert [t.dtype for t in _attn_operands(_kernel_dtype(bf, bf), bf, bf)] == [torch.bfloat16] * 2
    # Mixed: bf16 upcast (lossless) to fp32; fp64 left for _kernel_input as before.
    mixed = _attn_operands(_kernel_dtype(bf, f32, f64), bf, f32, f64)
    assert [t.dtype for t in mixed] == [torch.float32, torch.float32, torch.float64]
    assert mixed[1] is f32


def test_gather_head_dim_192_zero_padded(no_simulator):
    torch.manual_seed(2)
    S, H = 2 * B, 192
    Q, K, V = (torch.randn(S, H) for _ in range(3))
    mb = _local_bsr(2, window=1)
    qs, kg, vg, _, _ = _attn_gather(Q, K, V, mb, H**-0.5)
    assert qs.shape[-1] == kg.shape[-1] == vg.shape[-1] == 256
    assert (qs[..., H:] == 0).all() and (kg[..., H:] == 0).all() and (vg[..., H:] == 0).all()
    torch.testing.assert_close(qs[..., :H].reshape(S, H), Q * H**-0.5)

    dO, O = torch.randn(S, H), torch.randn(S, H)
    stats = torch.zeros(2, B)
    rf, cf = _attn_bwd_gather(Q, K, V, dO, O, mb, H**-0.5, stats, stats + 1)
    for t in (
        rf["q_scaled"],
        rf["k_gathered"],
        rf["do_gathered"],
        cf["q_gathered"],
        cf["v_blocks"],
    ):
        assert t.shape[-1] == 256 and (t[..., H:] == 0).all()
    torch.testing.assert_close(rf["D_blocks"].reshape(S), (dO * O).sum(-1))


def test_bf16_gather_scales_in_fp32_and_keeps_d_fp32(no_simulator):
    torch.manual_seed(3)
    S, H = 2 * B, 256
    Q, K, V, dO, O = (torch.randn(S, H).bfloat16() for _ in range(5))
    mb = _local_bsr(2, window=1)
    scale = H**-0.5
    qs, kg, _, _, _ = _attn_gather(Q, K, V, mb, scale)
    assert qs.dtype == kg.dtype == torch.bfloat16
    assert torch.equal(qs.reshape(S, H), (Q.float() * scale).bfloat16())
    stats = torch.zeros(2, B)
    rf, _ = _attn_bwd_gather(Q, K, V, dO, O, mb, scale, stats, stats + 1)
    assert rf["q_scaled"].dtype == torch.bfloat16
    assert rf["D_blocks"].dtype == torch.float32
    torch.testing.assert_close(rf["D_blocks"].reshape(S), (dO.float() * O.float()).sum(-1))


def test_fp32_gather_unchanged(no_simulator):
    """fp32 inputs: q_scaled and D are computed exactly as before (#49 touches bf16 only)."""
    torch.manual_seed(4)
    S, H = 2 * B, 64
    Q, K, V, dO, O = (torch.randn(S, H) for _ in range(5))
    mb = _local_bsr(2, window=1)
    qs, *_ = _attn_gather(Q, K, V, mb, H**-0.5)
    assert torch.equal(qs.reshape(S, H), Q * H**-0.5)
    stats = torch.zeros(2, B)
    rf, _ = _attn_bwd_gather(Q, K, V, dO, O, mb, H**-0.5, stats, stats + 1)
    assert torch.equal(rf["D_blocks"].reshape(S), (dO * O).sum(dim=-1))
