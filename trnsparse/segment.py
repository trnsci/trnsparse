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
connectivity between MD steps usually keep it.

**Padding E to a compile bucket.** Mark the padding edges:
``segment_plan(idx, N, padding=mask)``. The plan gives them no slots, so
the slot count and every real row's summation order are those of the real
edges alone, and the sums are bitwise equal to the plan over the real edges,
whichever segments the padding points at. The contract: the sums ignore
``m`` (and the gather cotangent) at padded edges; `gather` and the backward
of `segment_sum` still fill padded rows (see `segment_plan`). Without a
mask, padding edges are ordinary edges: they must carry zero messages, and
piling them onto one segment makes that segment's tile set the slot count
for every tile (56 -> 72 slots at 1728 atoms, 2.9x at 8000, #141). Spread
them over segments 0, 1, 2, ... instead.

**Index safety.** Index arithmetic stays on the host in int64. The device
sees only int32 edge and row ids, moved by DMA without conversion, and fp32
row offsets in [-1, 127] (exact). There is no flattened E*F or N*F index
and no int64 transpose, mask or ``index_select`` on the device
(aws-neuron-sdk#1466, #1467, #1468). Range limits are checked on the host:
E and N below 2^31, and every tensor a launch addresses below
``_SEG_MAX_BYTES`` (2^34 bytes; one launch is verified on trn2 well past
2^32 bytes, #141). A call wider than that splits the feature axis into
chunks, one launch per chunk over the same plan: columns are independent, so
the result is bitwise equal to one launch. Device aten is no fallback at
these sizes (its ``index_add_`` / ``index_select`` fail on trn2 at 4.7e9
bytes), and E below 2^31 means a single column always fits, so ``auto``
keeps every fp32 call on the kernel.

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
from .nki.kernels import _SEG_CHUNK, _SEG_GROUP, _SEG_P, _SEG_PROGRAMS, _SEG_PSUM_N, _gather_split

_INT32_MAX = 2**31 - 1
# Largest tensor (bytes) one kernel launch addresses. Wider calls split the
# feature axis (`_feature_chunk`). #140 set this to 2^31 on the assumption
# that the indirect DMA forms row x stride as a signed 32-bit byte offset.
# On trn2 (SDK 2.33.1) one launch is bitwise equal to the 2^31-byte chunks
# at 4.7e9 bytes and matches the fp64 sum (gather exact) at 9.4e9 bytes:
# past 2^32 bytes and 2^31 fp32 elements (#141). The bound is the next
# power-of-two boundary, 2^34 bytes (2^32 fp32 elements); a 1.7e10-byte
# operand no longer fits one logical core at LNC=2.
_SEG_MAX_BYTES = 2**34
# Column chunks are cut from row blocks of at most this many bytes: device
# aten copies a column slice of a view spanning more than 2^32 bytes about
# 170x slower (2.0 s against 12 ms at 229376 x 5120 fp32, #141).
_SEG_SLICE_BYTES = 2**31


def _launch_rows(e: int, n: int) -> int:
    """Rows of the largest tensor a launch over (E, N) addresses, either op or
    its backward: the padded edge count (messages, gather output) or the padded
    output tiles (sum output, gather input), whichever is larger."""
    ep = -(-max(e, 1) // (_SEG_P * _SEG_PROGRAMS)) * (_SEG_P * _SEG_PROGRAMS)
    return max(ep, n + 2 * _SEG_P)


def _feature_chunk(e: int, n: int, f: int) -> int:
    """Columns per launch for an (E, N, F) call: F when one launch fits, else
    the widest chunk that does (a multiple of the kernel's 2048-column group,
    or of a 512-column PSUM bank, when that wide), 0 when one column does not.

    Columns are independent in both kernels (the incidence tiles and the slot
    order depend on the plan only), so a split call is bitwise equal to one
    launch. Aligned chunks also keep each column in the same group and bank
    position as in the unsplit kernel."""
    w = (_SEG_MAX_BYTES - 1) // (_launch_rows(e, n) * 4)
    if w >= f:
        return f
    for align in (_SEG_GROUP, _SEG_PSUM_N):
        if w >= align:
            return w // align * align
    return w


def _host_index(idx: torch.Tensor) -> torch.Tensor:
    """`idx` as a contiguous host int64 tensor (one D2H copy if on a device)."""
    if not isinstance(idx, torch.Tensor):
        idx = torch.as_tensor(idx)
    if idx.dim() != 1:
        raise ValueError(f"index must be 1-D, got shape {tuple(idx.shape)}")
    if idx.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"index must be int64 or int32, got {idx.dtype}")
    return idx.detach().to("cpu", torch.int64).contiguous()


def _gather_layout(gidx: torch.Tensor) -> torch.Tensor:
    """The gather kernel's index table from edge-order ids ``[Ep, 1]``.

    Each block of M x 128 edges (the levels of `_gather_split`) is stored so
    that the kernel's contiguous (128, M) load of it, read column-major (the
    DMA batching order), gives the block's edges in order: entry
    ``(b * 128 + p) * M + j`` of the block region holds its edge
    ``j * 128 + p``. `_gather_natural` is the inverse."""
    out = gidx.clone()
    for t0, M, nb in _gather_split(gidx.shape[0]):
        r = slice(t0 * _SEG_P, (t0 + nb * M) * _SEG_P)
        out[r] = gidx[r].reshape(nb, M, _SEG_P).transpose(1, 2).reshape(-1, 1)
    return out


def _gather_natural(table: torch.Tensor) -> torch.Tensor:
    """Edge-order ids ``[Ep, 1]`` from a gather index table (inverse of `_gather_layout`)."""
    out = table.clone()
    for t0, M, nb in _gather_split(table.shape[0]):
        r = slice(t0 * _SEG_P, (t0 + nb * M) * _SEG_P)
        out[r] = table[r].reshape(nb, _SEG_P, M).transpose(1, 2).reshape(-1, 1)
    return out


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
        num_padding: edges marked in ``padding`` (0 without a mask).
    """

    def __init__(self, idx: torch.Tensor, num_segments: int, padding: torch.Tensor | None = None):
        index = _host_index(idx)
        n = int(num_segments)
        e = int(index.numel())
        pad = None
        if padding is not None:
            if not isinstance(padding, torch.Tensor):
                padding = torch.as_tensor(padding)
            if padding.dtype != torch.bool or padding.shape != (e,):
                raise ValueError(
                    f"padding must be a bool [E] = [{e}] mask, got {padding.dtype} "
                    f"{tuple(padding.shape)}"
                )
            pad = padding.detach().to("cpu").contiguous()
            if not bool(pad.any()):
                pad = None
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
        self.padding = pad
        self.num_padding = int(pad.sum()) if pad is not None else 0
        self.is_sorted = e < 2 or bool((index[1:] >= index[:-1]).all())
        if self.is_sorted:
            self.perm = torch.arange(e, dtype=torch.int64)
        else:
            self.perm = torch.argsort(index, stable=True)
        self._build_layout()
        self._device_tables: dict[str, tuple[torch.Tensor, ...]] = {}
        self._device_keep: dict[str, torch.Tensor] = {}

    def _build_layout(self) -> None:
        """Tile / slot tables for the kernels (host int64 math, int32 / fp32 results)."""
        P, n = _SEG_P, self.num_segments
        e_all = self.num_edges
        # The summed edges in stable sorted order: every edge but the padding.
        summed = self.perm if self.padding is None else self.perm[~self.padding[self.perm]]
        e = int(summed.numel())
        T = max(1, -(-n // P))
        srt = self.index[summed]  # segment ids in sorted order
        tile_of = torch.div(srt, P, rounding_mode="floor")
        tcount = torch.bincount(tile_of, minlength=T) if e else torch.zeros(T, dtype=torch.int64)
        tstart = torch.cumsum(tcount, 0) - tcount
        occupied = torch.nonzero(tcount, as_tuple=False).flatten()
        empty = torch.nonzero(tcount == 0, as_tuple=False).flatten()
        S = max(1, -(-int(tcount.max()) // P)) if e else 1
        S = -(-S // _SEG_CHUNK) * _SEG_CHUNK
        compact = e > 0 and 4 * int(empty.numel()) > T
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
            first[has] = summed[tstart[tiles[has]]]
        src = first.repeat_interleave(P).unsqueeze(1).expand(K * P, S).clone()
        local = torch.full((K * P, S), -1.0, dtype=torch.float32)
        if e:
            q = torch.arange(e, dtype=torch.int64) - tstart[tile_of]
            k = kpos[tile_of]
            rows = k * P + torch.remainder(q, P)
            cols = torch.div(q, P, rounding_mode="floor")
            src[rows, cols] = summed
            local[rows, cols] = (srt - tile_of * P).to(torch.float32)  # in [0, 127]: exact
        p = torch.arange(P, dtype=torch.int64)
        orow = (tiles.unsqueeze(1) * P + p).reshape(-1, 1)
        nz = int(zeros.numel())
        zt = zeros if nz else torch.tensor([T - 1 if T else 0], dtype=torch.int64)
        zrow = (zt.unsqueeze(1) * P + p).reshape(-1, 1)
        ep = -(-max(e_all, 1) // (P * _SEG_PROGRAMS)) * (P * _SEG_PROGRAMS)
        gidx = torch.zeros(ep, 1, dtype=torch.int64)
        gidx[:e_all, 0] = self.index
        gidx = _gather_layout(gidx)
        for name, t in (("src", src), ("orow", orow), ("zrow", zrow), ("gidx", gidx)):
            if t.numel() and (int(t.min()) < 0 or int(t.max()) > _INT32_MAX):
                raise AssertionError(f"segment plan table {name} out of int32 range")
        self.tiles, self.slots, self.compact = T, S, compact
        self.computed_tiles, self.zero_tiles, self.rows_out = K, nz, rows_out
        self.slot_fill = e / float(K * S * P)  # summed (non-padding) edges
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

    def keep(self, device=None) -> torch.Tensor | None:
        """``[E, 1]`` bool, False at the padding edges (None without padding), on `device`."""
        if self.padding is None:
            return None
        key = str(torch.device(device)) if device is not None else "cpu"
        if key not in self._device_keep:
            self._device_keep[key] = (~self.padding).unsqueeze(1).to(device or "cpu")
        return self._device_keep[key]

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
            f"padding={self.num_padding}, "
            f"tiles={self.tiles}, slots={self.slots}, compact={self.compact}, "
            f"slot_fill={self.slot_fill:.2f})"
        )


def segment_plan(
    idx: torch.Tensor, num_segments: int, *, padding: torch.Tensor | None = None
) -> SegmentPlan:
    """Plan for `segment_sum(m, idx, num_segments)` and `gather(h, idx)`.

    ``idx`` is ``[E]`` int64 (or int32), on any device (copied to the host
    once), values in ``[0, num_segments)``. Build it once per neighbour
    list and pass it as ``plan=``.

    ``padding`` is an optional ``[E]`` bool mask of edges that pad E to a
    compile bucket. The plan leaves them out of every sum it computes, on
    every backend: `segment_sum` ignores ``m`` at padded edges (as if it were
    zero, so a NaN there does not spread), and so does the backward of
    `gather` (the segment sum of the cotangent). `gather` itself still returns
    ``h[idx[e]]`` at padded edges, and the backward of `segment_sum` still
    returns ``grad[idx[e]]`` there: both are rows the caller discards or
    multiplies by zero. Exact whenever the padded messages and gather
    cotangents are zero, which is what bucket padding means. Padded edges
    take no slots, so the kernel shape, the slot count and every real row's
    summation order are those of the plan over the real edges alone: the sums
    are bitwise equal to it, whichever segments the padding points at. (Without
    a mask, padding edges are summed like any other: put them all on one
    segment and that segment's tile sets the slot count for every tile.)
    """
    return SegmentPlan(idx, num_segments, padding)


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


def _torch_segment_sum(m: torch.Tensor, idx: torch.Tensor, n: int, plan=None) -> torch.Tensor:
    out = m.new_zeros((n, m.shape[1]))
    if m.shape[0] == 0:
        return out
    keep = plan.keep(m.device) if plan is not None else None
    if keep is not None:
        m = torch.where(keep, m, m.new_zeros(()))  # padding edges add +0
    return out.index_add_(0, idx.to(m.device), m)


def _torch_gather(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if idx.numel() == 0:
        return h.new_zeros((0, h.shape[1]))
    return h.index_select(0, idx.to(h.device))


def _kernel_launches(op_name: str, x: torch.Tensor, plan: SegmentPlan, rows: int) -> torch.Tensor:
    """`op_name` over `x`, one launch per feature chunk (`_feature_chunk`), all
    with the same plan; the first `rows` rows of each, joined on the feature
    axis. `split` and `cat` differentiate to each other, so on the native path
    each chunk's backward is the other op on that chunk."""
    f = int(x.shape[1])
    w = _feature_chunk(plan.num_edges, plan.num_segments, f)
    if w < 1:
        raise NotImplementedError(f"trnsparse {op_name}: one feature column exceeds a launch")
    if w >= f:
        return _d.nki_segment_op(op_name, x, plan)[:rows]
    rb = max(1, _SEG_SLICE_BYTES // (f * x.element_size()))
    blocks = [r.split(w, dim=1) for r in x.split(rb, dim=0)]
    parts = []
    for j in range(len(blocks[0])):
        c = torch.cat([b[j] for b in blocks]) if len(blocks) > 1 else blocks[0][j].contiguous()
        parts.append(_d.nki_segment_op(op_name, c, plan)[:rows])
    return torch.cat(parts, dim=1)


def _kernel_segment_sum(m: torch.Tensor, plan: SegmentPlan) -> torch.Tensor:
    return _kernel_launches("segment_sum_tiles", m, plan, plan.num_segments)


def _kernel_gather(h: torch.Tensor, plan: SegmentPlan) -> torch.Tensor:
    return _kernel_launches("gather_rows", h, plan, plan.num_edges)


class _SegmentSumFn(torch.autograd.Function):
    """segment_sum on the pytorch / XLA / simulator routes; backward = gather."""

    @staticmethod
    def forward(ctx, m, idx, n, plan, kernel):
        ctx.idx, ctx.plan, ctx.kernel = idx, plan, kernel
        if kernel:
            return _kernel_segment_sum(m, plan)
        return _torch_segment_sum(m, idx, n, plan)

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
    if _feature_chunk(e, n, f) < 1:
        # Even a one-column launch over these rows exceeds the range (not
        # reachable at the default bound, with E and N below 2^31); wider
        # calls split the feature axis instead.
        if _d._REQUIRE_NKI:
            raise NotImplementedError(
                f"trnsparse {what}: {_launch_rows(e, n)} fp32 rows of one column exceed one "
                f"kernel launch's {_SEG_MAX_BYTES} byte addressing range. Unset "
                "TRNSPARSE_REQUIRE_NKI to use the pytorch path, or split the call over "
                "edge blocks."
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
    return _SegmentSumFn.apply(m2, idx, n, plan, False).reshape((n,) + rest)


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
    return _GatherFn.apply(h2, idx, plan, False).reshape((e,) + rest)
