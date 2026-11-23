"""Graph gather and segment sum, each the other's backward (#139).

The graph block of a message-passing interatomic potential (the MLIP
exemplar, trnsci/trnsci#170), in the convention of MACE's ``scatter_sum``:

- ``gather(h, idx)``: ``out[e] = h[idx[e]]``. ``h`` is ``[N, ...]`` and
  ``idx`` is ``[E]``. The backward is ``segment_sum(grad, idx, N)``.
- ``segment_sum(m, idx, N)``: ``out[i] = sum of m[e] over the edges with
  idx[e] == i``. ``m`` is ``[E, ...]``. Rows with no edge are 0. The
  backward is ``gather(grad, idx)``.

Both take int64 (or int32) indices, as a neighbour list produces them,
sorted or not. Gradients flow to the float operand only.

**Plans.** ``segment_plan(idx, N)`` sorts the edges by segment once on the
host (stable, int64) and builds the kernel's tile layout. Pass the plan to
every call over that index: the neighbour list is fixed across one energy +
forces step, and in MD until the next rebuild. Without ``plan=`` the NKI path
builds one per call, keyed by the index's content (a small LRU cache). That
means one device-to-host copy and one hash of the index per call. The
pytorch path needs no plan.

**Backends** (``trnsparse.set_backend``):

- ``"pytorch"`` (the reference): ``zeros(N, F).index_add_(0, idx, m)`` and
  ``h.index_select(0, idx)``. On the CPU, ``index_add_`` adds the edges of
  each row sequentially in edge order, so it is deterministic and matches
  the exemplar's reference blocks bitwise.
- ``"nki"`` / ``"auto"`` with NKI: the ``_segment_sum_kernel`` and
  ``_gather_rows_kernel`` NKI kernels, launched as the ``trnsparse::`` custom
  ops ``segment_sum_tiles`` and ``gather_rows`` on TorchNeuron. Each op
  registers the other as its backward (``register_autograd``). Other routes
  (XLA, simulator) use a ``torch.autograd.Function`` with the same
  backward.

**Summation order (NKI path, deterministic).** A plan cuts the segments into
tiles of 128 rows and lays each tile's edges out in their stable sorted
order (ascending edge id within a segment) as slots of 128 edges. The
kernel reduces each slot on the TensorEngine: a 0/1 incidence tile times
the slot's message rows. Slot partials accumulate in fp32 PSUM over chunks
of 8 slots, and the chunk partials add to an fp32 SBUF sum in chunk order.
The order is a function of the plan only, so reruns are bitwise identical.
Each tile has one writer and runs the same instructions on either physical
core, so results do not depend on the LNC setting. The pytorch path adds in
edge order (a different, also fixed, order). The two agree to fp32
rounding.

**Static shapes.** Indices are runtime data (the plan's tables), never
compiled in. A kernel compiles per (E, F, tiles, slots): the slot count is
rounded up to a multiple of 8 (1024 edges per tile), so small changes in
connectivity between MD steps usually keep it. Pad E to buckets with zero
messages on the last segment: no other row's edges move, so their sums are
bitwise unchanged. (Padding an earlier segment moves later rows to other
slots, which regroups their sums: equal to fp32 rounding, not bitwise.)

**Index safety.** Index arithmetic stays on the host in int64. The device
sees only int32 edge and row ids, moved by DMA without conversion, and fp32
row offsets in [-1, 127] (exact). There is no flattened E*F or N*F index
and no int64 transpose, mask or ``index_select`` on the device
(aws-neuron-sdk#1466, #1467, #1468). Range limits are checked on the host:
E and N below 2^31, and the indexed tensor below 2^31 bytes per launch
(beyond that, ``auto`` uses the pytorch path).

**dtypes.** The kernels compute in fp32 with fp32 accumulation: bf16 / fp16
inputs are upcast exactly, and the result comes back in the input dtype.
float64 follows the suite fp64 policy (#84): host torch under ``auto``, an
fp32 cast with a warning under an explicit ``nki``, and an error under
TRNSPARSE_REQUIRE_NKI. Integer, bool and complex values take the pytorch
path.
"""

from __future__ import annotations

import collections
import hashlib
import math

import torch

from .nki import dispatch as _d
from .nki.kernels import _SEG_CHUNK, _SEG_P, _SEG_PROGRAMS

_INT32_MAX = 2**31 - 1
# Largest indexed tensor (bytes) one kernel launch addresses: the DMA adds
# int32 row index x row stride, kept within a signed 32-bit byte offset.
_SEG_MAX_BYTES = 2**31


def _host_index(idx: torch.Tensor) -> torch.Tensor:
    """`idx` as a contiguous host int64 tensor (one D2H copy if on a device)."""
    if not isinstance(idx, torch.Tensor):
        idx = torch.as_tensor(idx)
    if idx.dim() != 1:
        raise ValueError(f"index must be 1-D, got shape {tuple(idx.shape)}")
    if idx.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"index must be int64 or int32, got {idx.dtype}")
    return idx.detach().to("cpu", torch.int64).contiguous()


class SegmentPlan:
    """Host plan for `segment_sum` / `gather` over one index vector.

    Build once per graph (`segment_plan(idx, num_segments)`) and pass it as
    ``plan=`` to every `segment_sum(m, idx, N)` and `gather(h, idx)` over the
    same ``idx``. All index arithmetic happens here, on the host, in int64.

    Attributes:
        num_segments: N, the number of segments (rows of the sum).
        num_edges: E, the length of the index.
        is_sorted: True when ``idx`` is non-decreasing (no permutation needed).
        perm: int64 ``[E]``, the stable sort of the edges by segment.
        tiles: 128-row output tiles, ``ceil(N / 128)``.
        slots: 128-edge slots per tile in the kernel layout (a multiple of
            8: the largest tile's edges, rounded up). Shapes are static per
            (tiles, slots, E), so a small change in connectivity usually
            reuses the compiled kernel.
        compact: True when the kernel computes only the occupied tiles and
            zero-fills the rest (more than a quarter of the tiles empty).
        slot_fill: E / (computed slots x 128), the useful fraction of the
            kernel's work.
    """

    def __init__(self, idx: torch.Tensor, num_segments: int):
        index = _host_index(idx)
        n = int(num_segments)
        e = int(index.numel())
        if n < 0:
            raise ValueError(f"num_segments must be >= 0, got {n}")
        if e > _INT32_MAX or n > _INT32_MAX - 2 * _SEG_P:
            raise ValueError(f"E = {e} and N = {n} must stay below 2^31 for the int32 device ids")
        if e:
            lo, hi = int(index.min()), int(index.max())
            if lo < 0 or hi >= n:
                raise IndexError(f"index values must lie in [0, {n}), got [{lo}, {hi}]")
        self.num_segments = n
        self.num_edges = e
        self.index = index
        self.is_sorted = e < 2 or bool((index[1:] >= index[:-1]).all())
        if self.is_sorted:
            self.perm = torch.arange(e, dtype=torch.int64)
        else:
            self.perm = torch.argsort(index, stable=True)
        self._build_layout()
        self._device_tables: dict[str, tuple[torch.Tensor, ...]] = {}

    def _build_layout(self) -> None:
        """Tile / slot tables for the kernels (host int64 math, int32 / fp32 results)."""
        P, e, n = _SEG_P, self.num_edges, self.num_segments
        T = max(1, -(-n // P))
        srt = self.index[self.perm]  # segment ids in sorted order
        tile_of = torch.div(srt, P, rounding_mode="floor")
        tcount = torch.bincount(tile_of, minlength=T) if e else torch.zeros(T, dtype=torch.int64)
        tstart = torch.cumsum(tcount, 0) - tcount
        occupied = torch.nonzero(tcount, as_tuple=False).flatten()
        empty = torch.nonzero(tcount == 0, as_tuple=False).flatten()
        S = max(1, -(-int(tcount.max()) // P)) if e else 1
        S = -(-S // _SEG_CHUNK) * _SEG_CHUNK
        compact = 4 * int(empty.numel()) > T
        if not compact:
            # Every tile computed (empty ones give zeros), padded to an even count.
            K = -(-T // _SEG_PROGRAMS) * _SEG_PROGRAMS
            tiles = torch.arange(K, dtype=torch.int64)
            zeros = torch.zeros(0, dtype=torch.int64)
            rows_out = K * P
        else:
            # Occupied tiles computed, empty ones zero-filled. Both counts
            # even: an empty tile joins the computed ones (it computes 0), a
            # spare tile past N joins the zero tiles.
            tiles, zeros = occupied, empty
            if tiles.numel() % _SEG_PROGRAMS:
                tiles, zeros = torch.cat([tiles, zeros[:1]]), zeros[1:]
            rows_tiles = T
            if zeros.numel() % _SEG_PROGRAMS:
                zeros = torch.cat([zeros, torch.tensor([T], dtype=torch.int64)])
                rows_tiles = T + 1
            rows_out = rows_tiles * P
            K = int(tiles.numel())
        kpos = torch.full((max(T, int(tiles.max()) + 1),), -1, dtype=torch.int64)
        kpos[tiles] = torch.arange(K, dtype=torch.int64)

        # Slot q of tile k holds sorted edge tstart[t] + q at (row k*P + q % P,
        # column q // P). Padding slots repeat the tile's first edge (any real
        # row) with local -1, so they add exactly 0.
        first = torch.zeros(K, dtype=torch.int64)
        real = tiles < T
        if e:
            has = torch.zeros(K, dtype=torch.bool)
            has[real] = tcount[tiles[real]] > 0
            first[has] = self.perm[tstart[tiles[has]]]
        src = first.repeat_interleave(P).unsqueeze(1).expand(K * P, S).clone()
        local = torch.full((K * P, S), -1.0, dtype=torch.float32)
        if e:
            q = torch.arange(e, dtype=torch.int64) - tstart[tile_of]
            k = kpos[tile_of]
            rows = k * P + torch.remainder(q, P)
            cols = torch.div(q, P, rounding_mode="floor")
            src[rows, cols] = self.perm
            local[rows, cols] = (srt - tile_of * P).to(torch.float32)  # in [0, 127]: exact
        p = torch.arange(P, dtype=torch.int64)
        orow = (tiles.unsqueeze(1) * P + p).reshape(-1, 1)
        nz = int(zeros.numel())
        zt = zeros if nz else torch.tensor([T - 1 if T else 0], dtype=torch.int64)
        zrow = (zt.unsqueeze(1) * P + p).reshape(-1, 1)
        ep = -(-max(e, 1) // (P * _SEG_PROGRAMS)) * (P * _SEG_PROGRAMS)
        gidx = torch.zeros(ep, 1, dtype=torch.int64)
        gidx[:e, 0] = self.index
        for name, t in (("src", src), ("orow", orow), ("zrow", zrow), ("gidx", gidx)):
            if t.numel() and (int(t.min()) < 0 or int(t.max()) > _INT32_MAX):
                raise AssertionError(f"segment plan table {name} out of int32 range")
        self.tiles, self.slots, self.compact = T, S, compact
        self.computed_tiles, self.zero_tiles, self.rows_out = K, nz, rows_out
        self.slot_fill = e / float(K * S * P)
        # int64 -> int32 casts of values checked above: exact.
        # Kernel layout: tile k, chunk c (8 slots), partition p is row
        # (k * NC + c) * P + p, slot j of the chunk is column j.
        nc = S // _SEG_CHUNK

        def chunked(t):
            return t.reshape(K, P, nc, _SEG_CHUNK).permute(0, 2, 1, 3).reshape(-1, _SEG_CHUNK)

        self._tables = (
            gidx.to(torch.int32),
            chunked(src).to(torch.int32).contiguous(),
            chunked(local).contiguous(),
            orow.to(torch.int32),
            zrow.to(torch.int32),
        )

    def tables(self, device=None) -> tuple[torch.Tensor, ...]:
        """(gidx, src, local, orow, zrow) on `device`, copied there once and cached."""
        if device is None or torch.device(device).type == "cpu":
            return self._tables
        key = str(torch.device(device))
        if key not in self._device_tables:
            self._device_tables[key] = tuple(t.to(device) for t in self._tables)
        return self._device_tables[key]

    def to(self, device) -> SegmentPlan:
        """Stage the kernel tables on `device` now (otherwise on the first call)."""
        self.tables(device)
        return self

    @property
    def statics(self) -> tuple[int, int]:
        """(rows_out, zero tiles): the kernels' static arguments."""
        return self.rows_out, self.zero_tiles

    def __repr__(self) -> str:
        return (
            f"SegmentPlan(N={self.num_segments}, E={self.num_edges}, sorted={self.is_sorted}, "
            f"tiles={self.tiles}, slots={self.slots}, compact={self.compact}, "
            f"slot_fill={self.slot_fill:.2f})"
        )


def segment_plan(idx: torch.Tensor, num_segments: int) -> SegmentPlan:
    """Plan for `segment_sum(m, idx, num_segments)` and `gather(h, idx)`.

    ``idx`` is ``[E]`` int64 (or int32), on any device (copied to the host
    once), values in ``[0, num_segments)``. Build it once per neighbour
    list and pass it as ``plan=``.
    """
    return SegmentPlan(idx, num_segments)


_PLAN_CACHE: collections.OrderedDict = collections.OrderedDict()
_PLAN_CACHE_MAX = 8


def _plan_for(idx: torch.Tensor, n: int) -> SegmentPlan:
    """A plan for `idx`, cached by its content (blake2b of the int64 values) and N."""
    host = _host_index(idx)
    key = (int(n), host.numel(), hashlib.blake2b(host.numpy().tobytes(), digest_size=16).digest())
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = SegmentPlan(host, n)
        _PLAN_CACHE[key] = plan
        while len(_PLAN_CACHE) > _PLAN_CACHE_MAX:
            _PLAN_CACHE.popitem(last=False)
    else:
        _PLAN_CACHE.move_to_end(key)
    return plan


# ── pytorch reference and the autograd pair (non-native routes) ─────────────


def _torch_segment_sum(m: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    out = m.new_zeros((n, m.shape[1]))
    if m.shape[0] == 0:
        return out
    return out.index_add_(0, idx.to(m.device), m)


def _torch_gather(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if idx.numel() == 0:
        return h.new_zeros((0, h.shape[1]))
    return h.index_select(0, idx.to(h.device))


def _kernel_segment_sum(m: torch.Tensor, plan: SegmentPlan) -> torch.Tensor:
    out = _d.nki_segment_op("segment_sum_tiles", m, plan)
    return out[: plan.num_segments]


def _kernel_gather(h: torch.Tensor, plan: SegmentPlan) -> torch.Tensor:
    out = _d.nki_segment_op("gather_rows", h, plan)
    return out[: plan.num_edges]


class _SegmentSumFn(torch.autograd.Function):
    """segment_sum on the pytorch / XLA / simulator routes; backward = gather."""

    @staticmethod
    def forward(ctx, m, idx, n, plan, kernel):
        ctx.idx, ctx.plan, ctx.kernel = idx, plan, kernel
        if kernel:
            return _kernel_segment_sum(m, plan)
        return _torch_segment_sum(m, idx, n)

    @staticmethod
    def backward(ctx, grad):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        g = _GatherFn.apply(grad.contiguous(), ctx.idx, ctx.plan, ctx.kernel)
        return g, None, None, None, None


class _GatherFn(torch.autograd.Function):
    """gather on the pytorch / XLA / simulator routes; backward = segment_sum."""

    @staticmethod
    def forward(ctx, h, idx, plan, kernel):
        ctx.idx, ctx.plan, ctx.kernel, ctx.n = idx, plan, kernel, h.shape[0]
        if kernel:
            return _kernel_gather(h, plan)
        return _torch_gather(h, idx)

    @staticmethod
    def backward(ctx, grad):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        g = _SegmentSumFn.apply(grad.contiguous(), ctx.idx, ctx.n, ctx.plan, ctx.kernel)
        return g, None, None, None


# ── Public API ───────────────────────────────────────────────────────────────


def _kernel_ok(what: str, x: torch.Tensor, e: int, n: int, f: int) -> bool:
    """True when this call takes the NKI kernel route (see the module docstring)."""
    if not _d._use_nki() or e == 0 or n == 0 or f == 0:
        return False
    if not _d._kernel_route(what, x):
        return False
    rows = max(e, n + 2 * _SEG_P)
    if rows * f * 4 >= _SEG_MAX_BYTES:
        if _d._REQUIRE_NKI:
            raise NotImplementedError(
                f"trnsparse {what}: {rows} x {f} fp32 rows exceed one kernel launch's "
                f"{_SEG_MAX_BYTES} byte addressing range. Unset TRNSPARSE_REQUIRE_NKI to "
                "use the pytorch path, or split the call over edge blocks."
            )
        return False
    return True


def _check_plan(plan: SegmentPlan, e: int, n: int) -> SegmentPlan:
    if not isinstance(plan, SegmentPlan):
        raise TypeError(f"plan must be a SegmentPlan, got {type(plan).__name__}")
    if plan.num_edges != e or plan.num_segments != n:
        raise ValueError(
            f"plan is for E={plan.num_edges}, N={plan.num_segments}; this call has E={e}, N={n}"
        )
    return plan


def _kernel_dtype_in(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float32 else x.float()


@_d._fp64_host_guard("segment_sum")
def segment_sum(
    m: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
    *,
    plan: SegmentPlan | None = None,
) -> torch.Tensor:
    """``out[i] = sum of m[e] over the edges e with idx[e] == i``; empty rows are 0.

    Args:
        m: ``[E, ...]`` values (messages), one row per edge.
        idx: ``[E]`` int64 / int32 segment of each edge, in ``[0, num_segments)``,
            sorted or not, repeats allowed.
        num_segments: N, the number of output rows.
        plan: a `SegmentPlan` for ``(idx, num_segments)``, built once per graph
            with `segment_plan` (NKI path only; built per call otherwise).

    Returns:
        ``[N, ...]`` in ``m``'s dtype, on ``m``'s device. Differentiable with
        respect to ``m``; the backward is `gather` of the cotangent over the
        same index (and the same plan). Deterministic on every path: see the
        module docstring for the summation order.
    """
    n = int(num_segments)
    e = int(m.shape[0]) if m.dim() else 0
    if m.dim() < 1:
        raise ValueError("m must have at least one dimension (edges first)")
    if idx.dim() != 1 or idx.shape[0] != e:
        raise ValueError(f"idx must be [E] = [{e}], got shape {tuple(idx.shape)}")
    rest = tuple(m.shape[1:])
    m2 = m.reshape(e, math.prod(rest))
    f = int(m2.shape[1])
    if _kernel_ok("segment_sum", m2, e, n, f):
        p = _check_plan(plan, e, n) if plan is not None else _plan_for(idx, n)
        x = _kernel_dtype_in(m2).contiguous()
        try:
            if _d._native_path():
                out = _kernel_segment_sum(x, p)
            else:
                out = _SegmentSumFn.apply(x, idx, n, p, True)
            return out.to(m.dtype).reshape((n,) + rest)
        except Exception:
            if _d._REQUIRE_NKI:
                raise
    if plan is not None:
        _check_plan(plan, e, n)
    return _SegmentSumFn.apply(m2, idx, n, None, False).reshape((n,) + rest)


@_d._fp64_host_guard("gather")
def gather(
    h: torch.Tensor,
    idx: torch.Tensor,
    *,
    plan: SegmentPlan | None = None,
) -> torch.Tensor:
    """``out[e] = h[idx[e]]``: rows of ``h`` by an int64 / int32 index.

    Args:
        h: ``[N, ...]`` node values.
        idx: ``[E]`` row of each edge, in ``[0, N)``, sorted or not.
        plan: a `SegmentPlan` for ``(idx, N)`` (NKI path only; built per call
            otherwise). The same plan serves `segment_sum` over ``idx``.

    Returns:
        ``[E, ...]`` in ``h``'s dtype, on ``h``'s device. Differentiable with
        respect to ``h``; the backward is `segment_sum` of the cotangent over
        the same index.
    """
    if h.dim() < 1:
        raise ValueError("h must have at least one dimension (rows first)")
    if idx.dim() != 1:
        raise ValueError(f"idx must be 1-D, got shape {tuple(idx.shape)}")
    n = int(h.shape[0])
    e = int(idx.shape[0])
    rest = tuple(h.shape[1:])
    h2 = h.reshape(n, math.prod(rest))
    f = int(h2.shape[1])
    if _kernel_ok("gather", h2, e, n, f):
        p = _check_plan(plan, e, n) if plan is not None else _plan_for(idx, n)
        x = _kernel_dtype_in(h2).contiguous()
        try:
            if _d._native_path():
                out = _kernel_gather(x, p)
            else:
                out = _GatherFn.apply(x, idx, p, True)
            return out.to(h.dtype).reshape((e,) + rest)
        except Exception:
            if _d._REQUIRE_NKI:
                raise
    if plan is not None:
        _check_plan(plan, e, n)
    return _GatherFn.apply(h2, idx, None, False).reshape((e,) + rest)
