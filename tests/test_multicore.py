"""Tests for `trnsparse.multicore` (work split across logical NeuronCores, #30).

The CPU tests check the partition and the shard/gather bookkeeping with
host "devices". The `neuron` tests run the sharded kernels on every visible
logical core (one core under the usual single-core run, all four with
`NEURON_RT_VISIBLE_CORES=0-3`):

    NEURON_RT_VISIBLE_CORES=0-3 TRNSPARSE_REQUIRE_NKI=1 pytest tests/test_multicore.py -m neuron
"""

from __future__ import annotations

import pytest
import torch

import trnsparse
from trnsparse.multicore import ShardedAttention, ShardedBSR, partition_block_rows
from trnsparse.nki import HAS_NKI
from trnsparse.nki import dispatch as _d

P = 128


def _random_bsr(nb, density, seed=0, skew=False):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(nb, nb, generator=g) < density
    mask[torch.arange(nb), torch.arange(nb)] = True
    if skew:
        mask[0, :] = True  # a global-token row
    rows, cols = mask.nonzero(as_tuple=True)
    row_ptrs = torch.zeros(nb + 1, dtype=torch.long)
    row_ptrs[1:] = torch.bincount(rows, minlength=nb).cumsum(0)
    blocks = torch.randn(len(cols), P, P, generator=g) / P**0.5
    return trnsparse.BSRMatrix(
        blocks=blocks,
        block_col_indices=cols.to(torch.long),
        block_row_ptrs=row_ptrs,
        shape=(nb * P, nb * P),
        block_size=P,
    )


@pytest.mark.parametrize("n_parts", [1, 2, 3, 4, 7])
def test_partition_block_rows_covers_and_balances(n_parts):
    ptrs = [0, 12, 13, 15, 15, 18, 20, 21, 25]  # counts 12,1,2,0,3,2,1,4
    parts = partition_block_rows(ptrs, n_parts)
    assert len(parts) == n_parts
    assert sorted(m for p in parts for m in p) == list(range(8))
    assert all(p == sorted(p) for p in parts)
    if n_parts == 2:
        counts = [ptrs[i + 1] - ptrs[i] for i in range(8)]
        loads = [sum(counts[m] for m in p) for p in parts]
        assert sorted(loads) == [12, 13]  # 25 blocks: the best split


def test_sharded_bsr_on_host_devices():
    A = _random_bsr(9, 0.3, skew=True)
    B = torch.randn(A.shape[1], 40)
    prev = trnsparse.get_backend()
    trnsparse.set_backend("pytorch")
    try:
        S = ShardedBSR(A, devices=[torch.device("cpu")] * 4)
        assert len(S.shards) == 4
        assert sum(S.stored_blocks) == A.n_blocks
        assert sorted(m for r in S.rows for m in r) == list(range(9))
        C = S.gather(S.spmm(B))
    finally:
        trnsparse.set_backend(prev)
    torch.testing.assert_close(C, A.to_dense() @ B, atol=1e-4, rtol=1e-4)


def test_sharded_bsr_more_cores_than_rows():
    A = _random_bsr(2, 1.0)
    S = ShardedBSR(A, devices=[torch.device("cpu")] * 4)
    assert len(S.devices) == 2  # cores with no rows are dropped


def test_sharded_attention_masks_partition_the_structure():
    mask = _random_bsr(10, 0.25, skew=True)
    SA = ShardedAttention(mask, devices=[torch.device("cpu")] * 3)
    ptrs = mask.block_row_ptrs.tolist()
    cols = mask.block_col_indices.tolist()
    for rows, sub in zip(SA.rows, SA.masks, strict=True):
        assert sub.shape == (len(rows) * P, 10 * P)
        sp = sub.block_row_ptrs.tolist()
        sc = sub.block_col_indices.tolist()
        for i, m in enumerate(rows):
            assert sc[sp[i] : sp[i + 1]] == cols[ptrs[m] : ptrs[m + 1]]
    Q = torch.randn(10 * P, 16)
    Qts = SA.shard_rows(Q, scale=1.0)
    torch.testing.assert_close(SA.gather([q.T for q in Qts]), Q)  # inverse row maps


def test_kernel_device_follows_operands(monkeypatch):
    monkeypatch.setattr(_d, "_NEURON_DEVICE", "meta")
    host = torch.zeros(2)
    meta = torch.zeros(2, device="meta")
    assert _d._kernel_device(host, host) == torch.device("meta")
    assert _d._kernel_device(host, meta) == meta.device


# ── hardware ─────────────────────────────────────────────────────────────────


def _native() -> bool:
    try:
        return HAS_NKI and not _d._use_simulator() and _d._get_dispatch_path() == "native"
    except Exception:
        return False


@pytest.fixture
def nki_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


def _cores():
    from trnsparse.multicore import logical_cores

    return logical_cores()


@pytest.mark.neuron
@pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path")
@pytest.mark.parametrize("pretransposed", [False, True])
@pytest.mark.parametrize("width", [128, 256, 512, 1024])
def test_sharded_bsr_neuron(nki_backend, width, pretransposed):
    # Dense enough that row groups share columns (B-tile reuse), skewed, and
    # with more stored blocks per row than one DMA chunk.
    cores = _cores()
    A = _random_bsr(24, 0.5, seed=3, skew=True)
    B = torch.randn(A.shape[1], width)
    ref = A.to_dense().double() @ B.double()
    S = ShardedBSR(A, cores, pretransposed=pretransposed)
    outs = S.spmm(S.replicate(B))
    assert [o.device for o in outs] == S.devices
    torch.testing.assert_close(S.gather(outs).double(), ref, atol=1e-3, rtol=1e-4)


@pytest.mark.neuron
@pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path")
def test_bsr_spmm_on_every_core(nki_backend):
    # Each logical core runs the kernel on its own operands (no implicit move).
    A = _random_bsr(8, 0.3, seed=4)
    B = torch.randn(A.shape[1], 256)
    ref = A.to_dense().double() @ B.double()
    for d in _cores():
        Ad = trnsparse.BSRMatrix(
            blocks=A.blocks.to(d),
            block_col_indices=A.block_col_indices,
            block_row_ptrs=A.block_row_ptrs,
            shape=A.shape,
            block_size=P,
        )
        out = trnsparse.bsr_spmm(Ad, B.to(d))
        assert out.device == d
        torch.testing.assert_close(out.cpu().double(), ref, atol=1e-3, rtol=1e-4)


@pytest.mark.neuron
@pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path")
def test_sharded_attention_neuron(nki_backend):
    cores = _cores()
    mask = _random_bsr(12, 0.2, seed=5, skew=True)
    seq, hd = 12 * P, 64
    g = torch.Generator().manual_seed(6)
    Q, K, V = (torch.randn(seq, hd, generator=g) for _ in range(3))
    elem = (
        torch.zeros(12, 12, dtype=torch.bool)
        .index_put_(
            (
                torch.repeat_interleave(
                    torch.arange(12), mask.block_row_ptrs[1:] - mask.block_row_ptrs[:-1]
                ),
                mask.block_col_indices,
            ),
            torch.tensor(True),
        )
        .repeat_interleave(P, 0)
        .repeat_interleave(P, 1)
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.double()[None], K.double()[None], V.double()[None], attn_mask=elem
    )[0]
    SA = ShardedAttention(mask, cores)
    out = SA.gather(SA.forward(SA.shard_rows(Q), SA.replicate_kv(K, V)))
    torch.testing.assert_close(out.double(), ref, atol=1e-3, rtol=1e-4)


@pytest.mark.neuron
@pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path")
@pytest.mark.parametrize("pretransposed", [False, True])
def test_sharded_bsr_grad_neuron(nki_backend, pretransposed):
    # Gradient to each core's B replica: sum over cores of A[rows_p]^T dC_p.
    A = _random_bsr(8, 0.4, seed=7)
    B = torch.randn(A.shape[1], 128)
    S = ShardedBSR(A, _cores(), pretransposed=pretransposed)
    reps = [b.requires_grad_() for b in S.replicate(B)]
    outs = S.spmm(reps)
    sum(o.sum() for o in outs).backward()
    got = sum(r.grad.cpu().double() for r in reps)
    ref = A.to_dense().double().T @ torch.ones(A.shape[0], 128, dtype=torch.float64)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-4)
