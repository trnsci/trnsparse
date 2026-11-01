"""
Static-structure BSR SpMM and block-sparse attention across logical NeuronCores.

On Trn2 at LNC=2 one chip has four logical NeuronCores, and TorchNeuron
exposes each one as a device (`neuron:0` ... `neuron:3`; set
`NEURON_RT_VISIBLE_CORES=0-3` for one process to see all four). Each
static kernel already splits its output tiles across the two physical
cores of its logical core (`kernels._assign_units`). This module adds the
next level: it partitions the work across logical cores.

- `ShardedBSR` splits a BSR matrix by block rows. Each logical core gets
  the block rows it owns (balanced by stored-block count, with the same
  longest-processing-time greedy as the in-kernel split) as its own
  compact BSR matrix, with the blocks resident on that core. A product
  needs the whole right-hand side on every core.
- `ShardedAttention` splits a single-head block-sparse attention forward by
  query block rows. Each core gets its rows of Q and the whole of K and V,
  prepared on the host in the layout the kernel takes, so a call is one
  kernel launch per core.
  For several heads or a batch, put whole heads on different cores and call
  `block_sparse_attention_tiled` there; that path is differentiable.

Data placement. TorchNeuron does not copy tensors between logical cores:
`t.to("neuron:1")` on a `neuron:0` tensor returns a tensor still on core 0,
and `copy_` across cores raises. So the per-core operands are built on the
host and copied to each core (`replicate`, `shard_rows`), and results come
back through the host (`gather`). Keep the shards resident between calls
and keep outputs on their cores where you can: each host round trip costs
about as much as the kernel (~1.3 ms per 16 MB on Trn2).

Launches are asynchronous: `spmm` and `forward` issue one kernel per core
and return without waiting, so the cores run concurrently. Synchronize (or
read the results) before timing.

Neither class records autograd across cores: `ShardedBSR.spmm` outputs do
carry each shard's `bsr_spmm` gradient on its own core, but `gather` goes
through the host. `ShardedAttention.forward` is forward-only.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .formats import BSRMatrix
from .nki import dispatch as _d
from .nki import kernels as _k
from .nki.kernels import _EMPTY_UNIT_COST, _assign_units, _row_block_counts


def logical_cores(n: int | None = None) -> list[torch.device]:
    """The visible logical NeuronCores as devices, `neuron:0` ... `neuron:{n-1}`.

    `n=None` returns every visible core. Raises if TorchNeuron is not
    available or fewer than `n` cores are visible.
    """
    import torch_neuronx  # noqa: F401  (registers the neuron device)

    count = torch.neuron.device_count()
    n = count if n is None else n
    if n > count:
        raise ValueError(
            f"{n} logical NeuronCores requested, {count} visible (NEURON_RT_VISIBLE_CORES)"
        )
    return [torch.device("neuron", i) for i in range(n)]


def partition_block_rows(row_ptrs: Sequence[int], n_parts: int, weight: int = 1) -> list[list[int]]:
    """Split block rows into `n_parts` groups balanced by stored blocks.

    `row_ptrs` are BSR block-row pointers. Each row costs its stored-block
    count times `weight` (empty rows cost a small constant: they only store
    a zero tile). The greedy is `kernels._assign_units`, so it is
    deterministic, and each group lists its rows in ascending order. Some
    groups are empty when there are fewer rows than parts.
    """
    counts = _row_block_counts(list(row_ptrs))
    costs = [weight * c if c else _EMPTY_UNIT_COST for c in counts]
    return [_assign_units(costs, n_parts, p) for p in range(n_parts)]


def _sub_bsr(A: BSRMatrix, rows: list[int], device) -> BSRMatrix:
    """The BSR matrix of block rows `rows` of A (compact; blocks on `device`)."""
    ptrs = A.block_row_ptrs.cpu().tolist()
    cols = A.block_col_indices.cpu()
    idx: list[int] = []
    sub_ptrs = [0]
    for m in rows:
        idx.extend(range(ptrs[m], ptrs[m + 1]))
        sub_ptrs.append(len(idx))
    sel = torch.tensor(idx, dtype=torch.long)
    blocks = A.blocks.detach().cpu()[sel] if idx else A.blocks.new_zeros(0, *A.blocks.shape[1:])
    return BSRMatrix(
        blocks=blocks.contiguous().to(device),
        block_col_indices=cols[sel] if idx else torch.zeros(0, dtype=torch.long),
        block_row_ptrs=torch.tensor(sub_ptrs, dtype=torch.long),
        shape=(len(rows) * A.block_size, A.shape[1]),
        block_size=A.block_size,
    )


class ShardedBSR:
    """A BSR matrix split by block rows across logical NeuronCores.

    Args:
        A: BSR matrix (`block_size` 128; structure on the host). Its blocks
            may be on the host or a device; each shard's blocks are copied
            to its core once, here.
        devices: the logical cores (`neuron:k` devices). Defaults to all
            visible cores (`logical_cores()`).
        pretransposed: keep each shard's blocks transposed, side by side
            (`(128, n_blocks * 128)`), and run the pre-transposed kernel: no
            in-kernel transpose, one DMA per up to 16 of a row's blocks,
            B tiles shared by groups of up to 8 block rows, one PSUM copy
            (`kernels._bsr_spmm_static_t_body`). `False` keeps the blocks
            as stored and runs `_bsr_spmm_static_kernel`. Both ops carry
            autograd, to the per-core block copies and B replicas.

    Attributes:
        devices: the cores that own at least one block row.
        rows: per core, the block rows of A it owns (ascending).
        shards: per core, the compact BSR matrix of those rows.

    `spmm(B)` returns one output per core (its rows of C = A @ B, on that
    core); `gather` assembles them into C.
    """

    def __init__(
        self,
        A: BSRMatrix,
        devices: Sequence[torch.device] | None = None,
        pretransposed: bool = True,
    ):
        if A.block_size != _d._TILE_M:
            raise ValueError(f"ShardedBSR needs block_size {_d._TILE_M}, got {A.block_size}")
        devices = list(devices) if devices is not None else logical_cores()
        parts = partition_block_rows(A.block_row_ptrs.cpu().tolist(), len(devices))
        self.shape = A.shape
        self.block_size = A.block_size
        self.devices = [d for d, r in zip(devices, parts, strict=True) if r]
        self.rows = [r for r in parts if r]
        self.shards = [_sub_bsr(A, r, d) for r, d in zip(self.rows, self.devices, strict=True)]
        # The static kernel's compile-time constants, built once.
        self._keys = [_k.register_structure(*_d._bsr_structure_lists(S)) for S in self.shards]
        self.pretransposed = pretransposed
        self._blocks_t = None
        if pretransposed:
            # (128, n_blocks * 128): block idx's transpose in columns idx*128..
            self._blocks_t = [
                S.blocks.detach()
                .cpu()
                .permute(2, 0, 1)
                .reshape(S.block_size, -1)
                .contiguous()
                .to(d)
                for S, d in zip(self.shards, self.devices, strict=True)
            ]

    @property
    def stored_blocks(self) -> list[int]:
        """Stored blocks per core (the balance the partition achieved)."""
        return [s.n_blocks for s in self.shards]

    def replicate(self, B: torch.Tensor) -> list[torch.Tensor]:
        """B copied (through the host) to every core of the partition."""
        Bh = B.detach().cpu() if B.device.type != "cpu" else B
        return [Bh.to(d) for d in self.devices]

    def _direct(self, b: torch.Tensor, S: BSRMatrix) -> bool:
        """True when `b` can go straight to the static kernel op (one launch)."""
        N = b.shape[1]
        return (
            _d._native_path()
            and b.device.type != "cpu"
            and b.shape[0] == S.shape[1]
            and S.shape[1] % S.block_size == 0
            and (N <= _d._TILE_N or N % _d._TILE_N == 0)
            and b.dtype == S.blocks.dtype
            and b.dtype in (torch.float32, torch.bfloat16)
            and b.is_contiguous()
            and _d._static_structure_ok(S.n_blocks * max(1, N // _d._TILE_N))
        )

    def spmm(self, B: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Per-core outputs of C = A @ B, without waiting for the kernels.

        `B` is one tensor (copied to every core first, through the host) or
        a list with one replica per core (`replicate(B)`), which is what to
        keep resident across calls. Output `p` is the rows `self.rows[p]`
        of C, in that order, on `self.devices[p]`. With aligned, resident
        replicas each core costs exactly one kernel launch (the static
        `trnsparse::bsr_spmm_keyed` op, autograd attached; the structure
        crosses the op boundary as one registered key); anything else
        goes through `bsr_spmm`.
        """
        if isinstance(B, torch.Tensor):
            B = self.replicate(B)
        if len(B) != len(self.shards):
            raise ValueError(f"{len(B)} replicas of B for {len(self.shards)} cores")
        from .ops import bsr_spmm

        outs = []
        for p, (S, key, b) in enumerate(zip(self.shards, self._keys, B, strict=True)):
            if self._direct(b, S):
                if self.pretransposed:
                    op = _d._get_native_ops()["bsr_spmm_t_keyed"]
                    outs.append(op(self._blocks_t[p], b, key))
                else:
                    op = _d._get_native_ops()["bsr_spmm_keyed"]
                    outs.append(op(S.blocks, b, key))
            else:
                outs.append(bsr_spmm(S, b))
        return outs

    def gather(self, outs: Sequence[torch.Tensor], device="cpu") -> torch.Tensor:
        """Assemble per-core outputs into C (rows in A's order) on `device`."""
        bs = self.block_size
        n_cols = outs[0].shape[-1]
        # Block rows cover A's rows padded up to the block size; the partial
        # last block row is sliced off at the end.
        m_pad = -(-self.shape[0] // bs) * bs
        C = torch.zeros(m_pad, n_cols, dtype=outs[0].dtype)
        for rows, o in zip(self.rows, outs, strict=True):
            o = o.detach().cpu().reshape(len(rows), bs, n_cols)
            C.view(-1, bs, n_cols)[torch.tensor(rows, dtype=torch.long)] = o
        return C[: self.shape[0]].to(device)


class ShardedAttention:
    """Single-head block-sparse attention forward, split by query block rows.

    Each logical core computes the output rows of the query block rows it
    owns (balanced by stored mask blocks), against the whole of K and V.
    Forward only, no autograd. For multi-head or batched attention, place
    whole heads on different cores instead (see the module docstring).

    Args:
        mask_bsr: the block mask (`block_size` 128, square, structure on the
            host; block values are ignored).
        devices: the logical cores; defaults to all visible.
    """

    def __init__(self, mask_bsr: BSRMatrix, devices: Sequence[torch.device] | None = None):
        if mask_bsr.block_size != _d._TILE_M:
            raise ValueError(f"ShardedAttention needs block_size {_d._TILE_M}")
        if mask_bsr.shape[0] % _d._TILE_M:
            raise ValueError(
                f"ShardedAttention needs seq_len ({mask_bsr.shape[0]}) to be a multiple of {_d._TILE_M}"
            )
        devices = list(devices) if devices is not None else logical_cores()
        ptrs = mask_bsr.block_row_ptrs.cpu().tolist()
        cols = mask_bsr.block_col_indices.cpu().tolist()
        parts = partition_block_rows(ptrs, len(devices), weight=2)
        self.seq_len = mask_bsr.shape[0]
        self.block_size = mask_bsr.block_size
        self.devices = [d for d, r in zip(devices, parts, strict=True) if r]
        self.rows = [r for r in parts if r]
        self.masks = []
        for rows in self.rows:
            sub_ptrs = [0]
            sub_cols: list[int] = []
            for m in rows:
                sub_cols.extend(cols[ptrs[m] : ptrs[m + 1]])
                sub_ptrs.append(len(sub_cols))
            self.masks.append(
                _d._MaskPattern(
                    torch.tensor(sub_cols, dtype=torch.long),
                    torch.tensor(sub_ptrs, dtype=torch.long),
                    (len(rows) * self.block_size, self.seq_len),
                    self.block_size,
                )
            )
        self._keys = [_k.register_structure(*_d._bsr_structure_lists(m)) for m in self.masks]

    def shard_rows(self, Q: torch.Tensor, scale: float | None = None) -> list[torch.Tensor]:
        """Each core's query rows as the kernel's `(Q * scale)^T` (head_dim, rows).

        Built on the host (scale, row selection and transpose) and copied
        once to each core. `scale` defaults to head_dim ** -0.5.
        """
        bs = self.block_size
        hd = Q.shape[-1]
        if hd > _d._TILE_K:
            raise ValueError(f"ShardedAttention needs head_dim <= {_d._TILE_K}")
        scale = hd**-0.5 if scale is None else scale
        Qh = (Q.detach().cpu().float() * scale).reshape(-1, bs, hd)
        return [
            Qh[torch.tensor(r, dtype=torch.long)].reshape(-1, hd).T.contiguous().to(d)
            for r, d in zip(self.rows, self.devices, strict=True)
        ]

    def replicate_kv(self, K: torch.Tensor, V: torch.Tensor) -> list[tuple]:
        """Per core, `(K^T, V)` as the kernel takes them, copied through the host."""
        Kt = K.detach().cpu().float().T.contiguous()
        Vh = V.detach().cpu().float().contiguous()
        return [(Kt.to(d), Vh.to(d)) for d in self.devices]

    def forward(self, Qts: Sequence[torch.Tensor], KVs: Sequence[tuple]) -> list[torch.Tensor]:
        """Per-core attention outputs, without waiting for the kernels.

        `Qts` from `shard_rows(Q)`, `KVs` from `replicate_kv(K, V)`: one
        kernel launch per core. Output `p` (rows, head_dim) holds the rows
        `self.rows[p]`, on `self.devices[p]`.
        """
        if not _d._native_path():
            raise RuntimeError("ShardedAttention needs the native TorchNeuron path")
        op = _d._get_native_ops()["attn_fwd_keyed"]
        return [
            op(q_t, k_t, v, key)[0] for q_t, (k_t, v), key in zip(Qts, KVs, self._keys, strict=True)
        ]

    def gather(self, outs: Sequence[torch.Tensor], device="cpu") -> torch.Tensor:
        """Assemble per-core outputs into (seq_len, head_dim) on `device`."""
        bs = self.block_size
        hd = outs[0].shape[-1]
        out = torch.zeros(self.seq_len, hd, dtype=outs[0].dtype)
        for rows, o in zip(self.rows, outs, strict=True):
            out.view(-1, bs, hd)[torch.tensor(rows, dtype=torch.long)] = (
                o.detach().cpu().reshape(len(rows), bs, hd)
            )
        return out.to(device)
