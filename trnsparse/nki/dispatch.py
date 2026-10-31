"""NKI dispatch for sparse operations.

v0.2.0 — `nki_spmm` is the first NKI-dispatched op. Forward path:
materialize the CSR into a padded dense view, run the NKI GEMM kernel
(`kernels._spmm_dense_kernel`) on the Neuron device, slice back to the
original shape. Backward runs at the PyTorch level — see
`_SpMMFunction.backward` for the analytic adjoints.

Autograd wrapping is a hard requirement across the trnsci suite (see
trnsci/trnsci#3). On the PyTorch/XLA path (Trn1) and the CPU fallback the
wrappers are `torch.autograd.Function`s (`_SpMMFunction`, `_BSRSpMMFunction`,
`_ScreenedSpMMFunction`, `ops._AttnTiledFunction`). On the native
TorchNeuron path (Trn2/Trn3) every kernel is a `trnsparse::` custom op and
its backward is attached with `register_autograd` (see `_native.py` and
`_native_attn.py`), so the host wrappers below call the forward directly.

Placement on the native path: the `neuron` device has no sparse CSR
tensors (aws-neuron-sdk#1442). CSR / BSR structure (row pointers, column
indices, block grids) stays on the host; only dense operands, dense blocks
and densified tiles go to `device="neuron"`. Results come back on the
device of the dense operand.
"""

from __future__ import annotations

import importlib.util
import os
import warnings

import numpy as np
import torch

from .kernels import _TILE_K, _TILE_M, _TILE_N, HAS_NKI

if HAS_NKI:
    import nki

    from .kernels import (  # noqa: F401 — NKI-only
        _attn_bwd_dkdv_kernel,
        _attn_bwd_dq_kernel,
        _attn_out_kernel,
        _attn_stats_kernel,
        _bsr_spmm_kernel,
        _screened_spmm_kernel,
        _spmm_dense_kernel,
    )

# When set, kernel-path failures re-raise instead of falling back to
# PyTorch. Used by the hardware validation suite.
_REQUIRE_NKI = os.environ.get("TRNSPARSE_REQUIRE_NKI", "").lower() in (
    "1",
    "true",
    "yes",
)

# When set, dispatch bypasses torch_xla and runs kernels through
# `nki.simulate(kernel)(np_args)` on CPU. Lets us iterate kernels on any
# x86_64 Linux box without paying the NEFF compile + hardware dispatch
# cost. Semantics follow NKI 0.3.0's simulator: no NEFF compile, no
# SBUF/PSUM capacity checks, no latency/parallelism modelling. For
# correctness iteration only; hardware still owns perf numbers.
_USE_SIMULATOR = os.environ.get("TRNSPARSE_USE_SIMULATOR", "").lower() in (
    "1",
    "true",
    "yes",
)


def _use_simulator() -> bool:
    return _USE_SIMULATOR and HAS_NKI


# Static-structure kernels on the native path (trnsparse#30): the host-side
# block structure becomes compile-time constants of the kernel instead of
# padded, gathered device tensors. TRNSPARSE_STATIC_STRUCTURE=0 restores the
# gather path. Structures above _STATIC_MAX_UNROLL unrolled block matmuls
# (n_blocks x output column tiles) also take the gather path, to bound the
# trace / compile time (8192x8192 at 50% block density, 2048 blocks,
# compiles in ~5 s on Trn2).
_STATIC_STRUCTURE = os.environ.get("TRNSPARSE_STATIC_STRUCTURE", "1").lower() not in (
    "0",
    "false",
    "no",
)
_STATIC_MAX_UNROLL = int(os.environ.get("TRNSPARSE_STATIC_MAX_UNROLL", "16384"))


def _static_structure_ok(n_unrolled: int) -> bool:
    return _STATIC_STRUCTURE and n_unrolled <= _STATIC_MAX_UNROLL


_backend = "auto"


def set_backend(backend: str) -> None:
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires nki>=0.3.0 (Neuron SDK 2.29+)")
    _backend = backend


def get_backend() -> str:
    return _backend


def _use_nki() -> bool:
    if _backend == "nki":
        return True
    if _backend == "pytorch":
        return False
    return HAS_NKI


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _to_xla(*tensors):
    """Move tensors to the XLA device for NKI kernel dispatch."""
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    orig = tensors[0].device
    return [t.to(device) for t in tensors], orig


# ── Device dispatch path: native TorchNeuron vs XLA ──────────────────────────
#
# Two Neuron PyTorch lineages coexist:
#   native  torch-neuronx 2.14+ (TorchNeuron, Trn2/Trn3): Trainium is a
#           PyTorch `neuron` device; kernels are custom ops registered via
#           `torch_neuronx.nki_op` (see `_native.py`).
#   xla     torch-neuronx 2.9 lineage (PyTorch/XLA; Trn1 and Inf2 stay
#           here): kernels dispatch on the torch_xla device via `_to_xla()`.
# Anything else is "cpu": no device path, so a kernel launch outside the
# simulator fails in `_to_xla` and the wrapper falls back to PyTorch (or
# re-raises under TRNSPARSE_REQUIRE_NKI), exactly as before the native path
# existed. Detection runs on the first launch, not at import, and the
# native probe never imports torch_xla.

_NEURON_DEVICE = "neuron"
_dispatch_path: str | None = None
_native_ops: dict | None = None
_xla_deprecation_warned = False

_XLA_DEPRECATION_MESSAGE = (
    "the PyTorch/XLA path is deprecated on Trn2/Trn3; use TorchNeuron (native). "
    "Trn1/Inf2 remain supported in maintenance mode."
)


def _native_available() -> bool:
    try:
        import torch_neuronx
    except Exception:
        return False
    # The XLA-lineage torch_neuronx has no `nki_op`.
    if not hasattr(torch_neuronx, "nki_op"):
        return False
    try:
        torch.device(_NEURON_DEVICE)  # torch_neuronx renames PrivateUse1 to "neuron"
        return torch_neuronx.device_count() > 0
    except Exception:
        return False


def _detect_dispatch_path() -> str:
    # TRNSCI_NKI_DISPATCH=auto|native|xla|cpu is shared by every trnsci
    # library, so one variable pins the whole suite on hosts that have both
    # lineages.
    forced = os.environ.get("TRNSCI_NKI_DISPATCH", "auto").strip().lower()
    if forced == "native":
        if not _native_available():
            raise RuntimeError(
                "TRNSCI_NKI_DISPATCH=native, but no TorchNeuron backend with a "
                "'neuron' device is available"
            )
        return "native"
    if forced == "xla":
        if importlib.util.find_spec("torch_xla") is None:
            raise RuntimeError("TRNSCI_NKI_DISPATCH=xla, but torch_xla is not installed")
        return "xla"
    if forced == "cpu":
        return "cpu"
    if forced != "auto":
        raise ValueError(f"TRNSCI_NKI_DISPATCH must be auto, native, xla, or cpu; got {forced!r}")
    if _native_available():
        return "native"
    if importlib.util.find_spec("torch_xla") is not None:
        return "xla"
    return "cpu"


def _maybe_warn_xla_deprecated() -> None:
    # Dormant until TorchNeuron GA: the suite turns it on with
    # TRNSCI_XLA_DEPRECATION=1 (trnsci/trnsci#35). Fires once per process.
    global _xla_deprecation_warned
    if _xla_deprecation_warned or os.environ.get("TRNSCI_XLA_DEPRECATION", "") != "1":
        return
    _xla_deprecation_warned = True
    warnings.warn(_XLA_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


def _get_dispatch_path() -> str:
    """Return "native", "xla", or "cpu" (cached after the first call)."""
    global _dispatch_path
    if _dispatch_path is None:
        _dispatch_path = _detect_dispatch_path()
    if _dispatch_path == "xla":
        _maybe_warn_xla_deprecated()
    return _dispatch_path


def _get_native_ops() -> dict:
    """Import `_native` (registering the trnsparse:: custom ops) on first use."""
    global _native_ops
    if _native_ops is None:
        from . import _native

        _native_ops = _native.OPS
    return _native_ops


def _native_path() -> bool:
    """True when kernels launch as native `trnsparse::` custom ops.

    On this path autograd comes from `register_autograd` on those ops, so the
    host wrappers skip their `torch.autograd.Function`s. The simulator keeps
    its own (autograd.Function) path.
    """
    return HAS_NKI and not _use_simulator() and _get_dispatch_path() == "native"


def _to_device(*tensors: torch.Tensor):
    """Move kernel operands to the device of the selected dispatch path.

    Returns `(device_tensors, orig_device)`. The XLA (and "cpu") path is
    `_to_xla` unchanged.
    """
    if _get_dispatch_path() == "native":
        orig = tensors[0].device
        return [t.to(_NEURON_DEVICE) for t in tensors], orig
    return _to_xla(*tensors)


def _run(kernel, op_name: str, *device_tensors: torch.Tensor):
    """Call one kernel on operands already placed by `_to_device`.

    `op_name` is the `trnsparse::<op_name>` custom op used on the native
    path; the XLA path calls `kernel` directly on torch_xla tensors.
    """
    if _get_dispatch_path() == "native":
        return _get_native_ops()[op_name](*device_tensors)
    return kernel(*device_tensors)


def _launch(kernel, op_name: str, *tensors: torch.Tensor):
    """Run one kernel launch on the Neuron device; results return to the
    device of the first operand. Every host wrapper goes through this (or
    through its two halves, `_to_device` + `_run`, when several kernels
    share operands that should cross to the device once)."""
    xs, orig = _to_device(*tensors)
    out = _run(kernel, op_name, *xs)
    if isinstance(out, tuple | list):
        return tuple(o.to(orig) for o in out)
    return out.to(orig)


class _MaskPattern:
    """Structure-only BSR view (no block values): what attention needs.

    Index tensors are kept on the host (copied there if they arrive on a
    device) so the per-block-row loops below never touch a device tensor.
    """

    def __init__(self, block_col_indices, block_row_ptrs, shape, block_size):
        self.block_col_indices = block_col_indices.cpu()
        self.block_row_ptrs = block_row_ptrs.cpu()
        self.shape = tuple(shape)
        self.block_size = int(block_size)

    @property
    def n_blocks(self) -> int:
        return len(self.block_col_indices)


def _kernel_input(t: torch.Tensor) -> torch.Tensor:
    """Cast a kernel operand to a dtype the NeuronCore datapath supports.

    Trainium has no fp64 datapath; the native device silently hands fp64
    operands to the kernel as fp32, and the kernel output (whose dtype
    follows its input) then comes back as fp32. Downcast explicitly so the
    kernel always sees fp32, and have callers cast the result back to the
    caller's dtype — otherwise fp64 autograd inputs (gradcheck) hit a
    float/double mix in the PyTorch-level backward.
    """
    return t.float() if t.dtype == torch.float64 else t


def _kernel_dtype(*ts: torch.Tensor) -> torch.dtype:
    """Common kernel operand dtype: bf16 only when every operand is bf16, else fp32.

    The static kernels take fp32 or bf16 matmul operands (fp32 PSUM either
    way) and need all operands of one call in the same dtype.
    """
    return torch.bfloat16 if all(t.dtype == torch.bfloat16 for t in ts) else torch.float32


def _csr_to_dense_padded(A) -> torch.Tensor:
    """Materialize a CSRMatrix into a dense (M, K) tensor.

    v0.2.0 uses the dense view directly as the A operand of the GEMM
    kernel. Row-bucketing (Phase 3, #15) will replace this with a
    per-bucket dense tile keyed on nnz quantile, which is where the
    actual sparse speedup comes from.
    """
    t = torch.sparse_csr_tensor(A.row_ptrs, A.col_indices, A.values, size=A.shape)
    return t.to_dense()


# Densify on the device below this element density, on the host above it.
# Trn2 crossover (4096^2 / 8192^2): a device scatter costs ~15 ns per stored
# entry, a host densify + H2D copy ~1.4 ns per matrix element.
_DEVICE_DENSIFY_MAX_DENSITY = 0.1


def _csr_to_dense_on(A, device) -> torch.Tensor:
    """Densify a host-structure CSRMatrix for a kernel on `device`.

    Sparse enough (`nnz < _DEVICE_DENSIFY_MAX_DENSITY * M * K`): the row
    index of each entry is computed on the host, only the nnz-sized
    (flat index, value) pairs cross to the device, and the scatter into a
    zero tile runs there. Otherwise the dense matrix is built on the host and
    copied (`formats._csr_to_dense_host`), as before. Before this, every
    call copied the full host-built dense matrix (156 ms of a 157 ms
    `spmm` at 8192 x 8192, 0.1% dense, on Trn2).

    Canonical CSR (strictly increasing columns per row, as `from_dense`
    builds) uses `scatter`; duplicates fall back to `index_add` (summed, as
    `torch.sparse_csr_tensor(...).to_dense()`; ~8x slower on Trn2).
    Differentiable w.r.t. `A.values` either way.
    """
    from ..formats import _csr_to_dense_host

    m, n = A.shape
    nnz = A.values.shape[0]
    if torch.device(device).type == "cpu" or nnz >= _DEVICE_DENSIFY_MAX_DENSITY * m * n:
        return _csr_to_dense_host(A).to(device)
    row_ptrs = A.row_ptrs.cpu()
    positions = torch.arange(nnz, dtype=row_ptrs.dtype)
    rows = (torch.searchsorted(row_ptrs, positions, right=True) - 1).to(torch.long)
    flat = rows * n + A.col_indices.cpu().to(torch.long)
    values = A.values.to(device)
    dense = torch.zeros(m * n, dtype=values.dtype, device=device)
    flat_d = flat.to(device)
    # Under torch.compile the duplicate check would be a data-dependent
    # branch; index_add is correct either way.
    canonical = not torch.compiler.is_compiling() and (
        nnz < 2 or bool((flat[1:] > flat[:-1]).all())
    )
    if canonical:
        dense = dense.scatter(0, flat_d, values)
    else:
        dense = dense.index_add(0, flat_d, values)
    return dense.view(m, n)


def _nki_spmm_impl(A_dense: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Dispatch a dense (M, K) @ (K, N) matmul through the NKI kernel.

    Pads M/K to TILE_M/TILE_K multiples and N up to TILE_N when N > TILE_N.
    Slices the result back to the caller's (M, N).

    Falls back to `torch.matmul` on kernel errors unless
    `TRNSPARSE_REQUIRE_NKI=1`.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    M, K = A_dense.shape
    _, N = B.shape
    M_pad = _round_up(M, _TILE_M)
    K_pad = _round_up(K, _TILE_K)
    N_pad = N if N <= _TILE_N else _round_up(N, _TILE_N)
    needs_pad = (M_pad != M) or (K_pad != K) or (N_pad != N)

    try:
        if needs_pad:
            A_p = torch.zeros(M_pad, K_pad, dtype=A_dense.dtype, device=A_dense.device)
            A_p[:M, :K] = A_dense
            B_p = torch.zeros(K_pad, N_pad, dtype=B.dtype, device=B.device)
            B_p[:K, :N] = B
            A_feed, B_feed = A_p.contiguous(), B_p.contiguous()
        else:
            A_feed, B_feed = A_dense.contiguous(), B.contiguous()
        A_feed, B_feed = _kernel_input(A_feed), _kernel_input(B_feed)

        if _use_simulator():
            # CPU-side: feed NumPy arrays to nki.simulate(kernel). Bypasses
            # torch_xla entirely.
            out_np = nki.simulate(_spmm_dense_kernel)(A_feed.cpu().numpy(), B_feed.cpu().numpy())
            result = torch.from_numpy(np.asarray(out_np)).to(A_dense.device)
        else:
            result = _launch(_spmm_dense_kernel, "spmm_dense", A_feed, B_feed)

        result = result.to(A_dense.dtype)
        return result[:M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        return torch.matmul(A_dense, B)


class _SpMMFunction(torch.autograd.Function):
    """Autograd wrapper for the NKI SpMM forward.

    Forward is NKI-dispatched; backward runs at the PyTorch level.
    Analytic adjoints for C = A @ B with A sparse:

        dL/dA_values = (dL/dC @ Bᵀ)[rows, cols]   (projected onto A's pattern)
        dL/dB        = Aᵀ @ dL/dC

    For v0.2.0 the forward pass materializes A as dense, so the A gradient
    flows back through the materialized view naturally — torch handles the
    projection via the sparse_csr_tensor.to_dense() graph.
    """

    @staticmethod
    def forward(ctx, A_dense: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(A_dense, B)
        return _nki_spmm_impl(A_dense, B)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        A_dense, B = ctx.saved_tensors
        grad_A = grad_out @ B.T if ctx.needs_input_grad[0] else None
        grad_B = A_dense.T @ grad_out if ctx.needs_input_grad[1] else None
        return grad_A, grad_B


def nki_spmm(A, B: torch.Tensor) -> torch.Tensor:
    """SpMM entry point: C = A @ B for a `CSRMatrix` A and dense B.

    Routes through `_SpMMFunction.apply` so `loss.backward()` works.
    On NKI backend, forward runs the kernel; on PyTorch, torch.matmul
    on the densified A is the fallback — the autograd wrapper is kept
    on both paths for API uniformity.

    Native TorchNeuron path: A's CSR structure stays on the host and is
    densified there (no torch sparse tensor), the dense A moves to B's
    device, and the `trnsparse::spmm_dense` op (autograd registered with
    `register_autograd`) runs the kernel. B may live on the host or on
    `neuron`; the result is on B's device.
    """
    if _native_path():
        A_dense = _csr_to_dense_on(A, B.device)
        return _nki_spmm_impl(A_dense, B)
    A_dense = _csr_to_dense_padded(A)
    return _SpMMFunction.apply(A_dense, B)


def _nki_bsr_spmm_impl(
    blocks_pad: torch.Tensor, b_gathered: torch.Tensor, out_rows: int, out_cols: int
) -> torch.Tensor:
    """Dispatch `_bsr_spmm_kernel` on the XLA device.

    `blocks_pad` is `(M_tiles, K_max, 128, 128)` and `b_gathered` is
    `(M_tiles, K_max, 128, N_pad)`. Returns `(M_tiles * 128, N_pad)`
    sliced down to `(out_rows, out_cols)`.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    try:
        bp_feed = _kernel_input(blocks_pad.contiguous())
        bg_feed = _kernel_input(b_gathered.contiguous())

        if _use_simulator():
            out_np = nki.simulate(_bsr_spmm_kernel)(bp_feed.cpu().numpy(), bg_feed.cpu().numpy())
            result = torch.from_numpy(np.asarray(out_np)).to(blocks_pad.device)
        else:
            result = _launch(_bsr_spmm_kernel, "bsr_spmm", bp_feed, bg_feed)

        return result[:out_rows, :out_cols].to(blocks_pad.dtype).contiguous()
    except Exception:
        if _REQUIRE_NKI:
            raise
        # Fallback: straight PyTorch reference using the gathered slices.
        M_tiles, K_max, B, _ = blocks_pad.shape
        _, _, _, N_pad = b_gathered.shape
        out = torch.zeros(M_tiles * B, N_pad, dtype=blocks_pad.dtype, device=blocks_pad.device)
        for m in range(M_tiles):
            for k in range(K_max):
                out[m * B : (m + 1) * B] += blocks_pad[m, k] @ b_gathered[m, k]
        return out[:out_rows, :out_cols].contiguous()


def _bsr_pad_and_gather(A, B: torch.Tensor):
    """Host-side prep for the BSR kernel.

    Returns:
        blocks_pad: (M_tiles, K_max, block_size, block_size) zero-padded
        b_gathered: (M_tiles, K_max, block_size, N_pad) B-slices indexed
                     by each block's column index (zero when no block)
        out_rows, out_cols: the caller's original output shape
        K_max: for NEFF cache key visibility
    """
    m, n = A.shape
    _, out_cols = B.shape
    b = A.block_size
    m_pad = ((m + b - 1) // b) * b
    n_pad = ((n + b - 1) // b) * b

    # Pad B rows up to block alignment and columns up to TILE_N alignment
    N_pad = out_cols if out_cols <= _TILE_N else ((out_cols + _TILE_N - 1) // _TILE_N) * _TILE_N
    if N_pad != out_cols or n_pad != n:
        B_p = torch.zeros(n_pad, N_pad, dtype=B.dtype, device=B.device)
        B_p[:n, :out_cols] = B
        B = B_p

    M_tiles = m_pad // b
    nb_cols = n_pad // b

    # Count blocks per row and pad up to K_max. The structure is read from
    # host copies (a no-op when it already lives on the host), so the loop
    # below never does scalar reads of device tensors.
    block_row_ptrs = A.block_row_ptrs.cpu()
    block_col_indices = A.block_col_indices.cpu()
    blocks_per_row = (block_row_ptrs[1:] - block_row_ptrs[:-1]).tolist()
    K_max = max(blocks_per_row) if blocks_per_row else 1

    # Construct a 2D grid indexed by (block_row, k_slot) -> block_column
    # or -1 if no block. Then gather B and blocks via this grid.
    col_grid = torch.full((M_tiles, K_max), -1, dtype=torch.long)
    block_idx_grid = torch.full((M_tiles, K_max), -1, dtype=torch.long)
    for i in range(M_tiles):
        start = block_row_ptrs[i].item()
        end = block_row_ptrs[i + 1].item()
        row_len = end - start
        col_grid[i, :row_len] = block_col_indices[start:end]
        block_idx_grid[i, :row_len] = torch.arange(start, end)

    # Pad A with a single zero block at index `n_blocks` for masked slots.
    zero_block = torch.zeros(1, b, b, dtype=A.dtype, device=A.blocks.device)
    blocks_with_zero = torch.cat([A.blocks, zero_block], dim=0)
    masked_idx = torch.where(
        block_idx_grid >= 0, block_idx_grid, torch.tensor(A.n_blocks, dtype=torch.long)
    )
    # The index grids are built on the host; move them to the data's device
    # for the gather (a no-op on the host).
    blocks_pad = blocks_with_zero[masked_idx.to(blocks_with_zero.device)]  # (M_tiles, K_max, b, b)

    # Gather B slices per slot. For masked slots, a zero slice (columns 0..b)
    # times a zero block contributes zero anyway — we use column 0 as sentinel.
    safe_col = torch.where(col_grid >= 0, col_grid, torch.tensor(0, dtype=torch.long))
    # Build (M_tiles, K_max, b, N_pad) via fancy indexing into B reshaped (nb_cols, b, N_pad).
    B_by_block = B.view(nb_cols, b, N_pad)
    b_gathered = B_by_block[safe_col.to(B.device)]  # (M_tiles, K_max, b, N_pad)

    return blocks_pad, b_gathered, m, out_cols, K_max


class _BSRSpMMFunction(torch.autograd.Function):
    """Autograd wrapper for BSR SpMM.

    Forward: dispatched to NKI kernel (uniform-K-padded per block row).
    Backward (PyTorch-level):
        dA_blocks[k] = dC[row_block_of_k] @ B[col_block_of_k].T
                         (pattern stays fixed — no gradient to block indices)
        dB            = A.T @ dC  (via PyTorch dense path for simplicity)

    The backward pass reuses the PyTorch fallback rather than calling the
    kernel in reverse — correct, differentiable, and v0.3.0-scope.
    """

    @staticmethod
    def forward(ctx, A_blocks, A_block_col_indices, A_block_row_ptrs, A_shape, A_block_size, B):
        # Reconstruct a lightweight BSR-like handle for the host-side prep
        class _BSRHandle:
            shape = A_shape
            block_size = A_block_size
            blocks = A_blocks
            block_col_indices = A_block_col_indices
            block_row_ptrs = A_block_row_ptrs

            @property
            def n_blocks(self):
                return self.blocks.shape[0]

            @property
            def dtype(self):
                return self.blocks.dtype

        handle = _BSRHandle()
        blocks_pad, b_gathered, out_rows, out_cols, _ = _bsr_pad_and_gather(handle, B)
        C = _nki_bsr_spmm_impl(blocks_pad, b_gathered, out_rows, out_cols)

        ctx.save_for_backward(A_blocks, A_block_col_indices, A_block_row_ptrs, B)
        ctx.A_shape = A_shape
        ctx.A_block_size = A_block_size
        return C

    @staticmethod
    def backward(ctx, grad_out):
        A_blocks, col_idx, row_ptrs, B = ctx.saved_tensors
        b = ctx.A_block_size

        grad_blocks = None
        grad_B = None

        if ctx.needs_input_grad[0]:
            # dA_blocks[k] = grad_out[rowblock*b : (rowblock+1)*b] @ B[col*b:(col+1)*b].T
            # rowblock is inferred from row_ptrs.
            n_blocks = A_blocks.shape[0]
            grad_blocks = torch.zeros_like(A_blocks)
            M_tiles = row_ptrs.shape[0] - 1
            for i in range(M_tiles):
                start = row_ptrs[i].item()
                end = row_ptrs[i + 1].item()
                for k in range(start, end):
                    j = col_idx[k].item()
                    grad_blocks[k] = grad_out[i * b : (i + 1) * b] @ B[j * b : (j + 1) * b].T

        if ctx.needs_input_grad[5]:
            # dB = A.T @ grad_out — reconstruct A dense and multiply.
            m, n = ctx.A_shape
            m_pad = ((m + b - 1) // b) * b
            n_pad = ((n + b - 1) // b) * b
            A_dense = torch.zeros(m_pad, n_pad, dtype=A_blocks.dtype, device=A_blocks.device)
            M_tiles = row_ptrs.shape[0] - 1
            for i in range(M_tiles):
                start = row_ptrs[i].item()
                end = row_ptrs[i + 1].item()
                for k in range(start, end):
                    j = col_idx[k].item()
                    A_dense[i * b : (i + 1) * b, j * b : (j + 1) * b] = A_blocks[k]
            A_dense = A_dense[:m, :n]
            grad_B = A_dense.T @ grad_out

        # Five non-tensor inputs in forward args: indices/ptrs/shape/block_size/B
        # Only A_blocks (arg 0) and B (arg 5) have grads.
        return grad_blocks, None, None, None, None, grad_B


def nki_bsr_spmm(A, B: torch.Tensor) -> torch.Tensor:
    """BSR SpMM entry point — wraps `_BSRSpMMFunction.apply` for autograd.

    Native TorchNeuron path: the block structure (`block_row_ptrs`,
    `block_col_indices`) stays on the host; `A.blocks` (dense) moves to B's
    device and the gather into kernel tiles runs there. The
    `trnsparse::bsr_spmm` op carries the autograd (gradients reach
    `A.blocks` and `B` through the gather), so no per-block Python loop
    runs on device tensors. The result is on B's device.
    """
    if _native_path():
        out_cols = B.shape[1]
        n_tiles = 1 if out_cols <= _TILE_N else -(-out_cols // _TILE_N)
        if A.block_size == _TILE_M and _static_structure_ok(A.n_blocks * n_tiles):
            try:
                return _nki_bsr_spmm_static(A, B)
            except Exception:
                if _REQUIRE_NKI:
                    raise
        blocks_pad, b_gathered, out_rows, out_cols, _ = _bsr_pad_and_gather(
            _BlockView(A, A.blocks.to(B.device)), B
        )
        return _nki_bsr_spmm_impl(blocks_pad, b_gathered, out_rows, out_cols)
    return _BSRSpMMFunction.apply(
        A.blocks,
        A.block_col_indices,
        A.block_row_ptrs,
        A.shape,
        A.block_size,
        B,
    )


def _bsr_structure_lists(A) -> tuple[list[int], list[int]]:
    """Host BSR structure as Python int lists (the static kernels' constants)."""
    return A.block_row_ptrs.cpu().tolist(), A.block_col_indices.cpu().tolist()


def _nki_bsr_spmm_static(A, B: torch.Tensor) -> torch.Tensor:
    """BSR SpMM through `trnsparse::bsr_spmm_static` (native path).

    No (M_tiles, K_max) padding and no gather: `A.blocks` and B go to the
    device as they are (B zero-padded to block / 512-column alignment only
    when needed) and the kernel walks the stored blocks, addressing B by
    each block's column index. Autograd to `A.blocks` and B is registered
    on the op.

    bf16 when both `A.blocks` and B are bf16 (fp32 PSUM accumulation, bf16
    result); otherwise fp32. The kernel splits its output tiles across the
    physical cores of the logical NeuronCore.
    """
    m, n = A.shape
    b = A.block_size
    out_cols = B.shape[1]
    n_pad = _round_up(n, b)
    N_pad = out_cols if out_cols <= _TILE_N else _round_up(out_cols, _TILE_N)
    if N_pad != out_cols or n_pad != n:
        B_p = torch.zeros(n_pad, N_pad, dtype=B.dtype, device=B.device)
        B_p[:n, :out_cols] = B
        B = B_p
    row_ptrs, cols = _bsr_structure_lists(A)
    # Kernel operands go to the neuron device (a no-op when they are already
    # there); the result returns to B's device.
    orig = B.device
    kdt = _kernel_dtype(A.blocks, B)
    out_dtype = torch.promote_types(A.blocks.dtype, B.dtype)
    blocks = A.blocks.contiguous().to(device=_NEURON_DEVICE, dtype=kdt)
    b_dev = B.contiguous().to(device=_NEURON_DEVICE, dtype=kdt)
    out = _get_native_ops()["bsr_spmm_static"](blocks, b_dev, row_ptrs, cols)
    return out[:m, :out_cols].to(device=orig, dtype=out_dtype)


class _BlockView:
    """A BSR matrix with its dense blocks placed on another device."""

    def __init__(self, A, blocks: torch.Tensor):
        self.shape = A.shape
        self.block_size = A.block_size
        self.blocks = blocks
        self.block_col_indices = A.block_col_indices
        self.block_row_ptrs = A.block_row_ptrs

    @property
    def n_blocks(self) -> int:
        return self.blocks.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks.dtype


def _nki_screened_spmm_impl(
    A: torch.Tensor,
    Q: torch.Tensor,
    threshold_sqrt: float,
    B: torch.Tensor,
) -> torch.Tensor:
    """Dispatch `_screened_spmm_kernel` on the XLA device (or simulator).

    `A` is square (M, M); `Q` is the 1-D Schwarz-bound vector of length M.
    Pads M up to TILE_M and N up to TILE_N when `N > TILE_N`.
    Falls back to `torch.matmul` on masked A if the kernel errors and
    `TRNSPARSE_REQUIRE_NKI` is not set.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    M, K = A.shape
    _, N = B.shape
    assert M == K, f"screened_spmm currently requires square A; got {A.shape}"
    M_pad = _round_up(M, _TILE_M)
    N_pad = N if N <= _TILE_N else _round_up(N, _TILE_N)
    needs_pad = (M_pad != M) or (N_pad != N)

    # NKI 0.3.0: all tensors must be ≥2D; reshape scalar to (1,1).
    threshold_sqrt_t = torch.tensor([[threshold_sqrt]], dtype=A.dtype)

    try:
        if needs_pad:
            A_p = torch.zeros(M_pad, M_pad, dtype=A.dtype, device=A.device)
            A_p[:M, :M] = A
            Q_p = torch.zeros(M_pad, dtype=Q.dtype, device=Q.device)
            Q_p[:M] = Q
            B_p = torch.zeros(M_pad, N_pad, dtype=B.dtype, device=B.device)
            B_p[:M, :N] = B
            A_feed, Q_feed, B_feed = A_p.contiguous(), Q_p.contiguous(), B_p.contiguous()
        else:
            A_feed, Q_feed, B_feed = A.contiguous(), Q.contiguous(), B.contiguous()
        # NKI 0.3.0: nl.load requires 2D tensors; unsqueeze Q from (M,) to (M,1)
        Q_feed = Q_feed.unsqueeze(-1).contiguous()
        A_feed, Q_feed, B_feed = _kernel_input(A_feed), _kernel_input(Q_feed), _kernel_input(B_feed)
        threshold_sqrt_t = _kernel_input(threshold_sqrt_t)

        if _use_simulator():
            out_np = nki.simulate(_screened_spmm_kernel)(
                A_feed.cpu().numpy(),
                Q_feed.cpu().numpy(),
                threshold_sqrt_t.cpu().numpy(),
                B_feed.cpu().numpy(),
            )
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            result = _launch(
                _screened_spmm_kernel, "screened_spmm", A_feed, Q_feed, threshold_sqrt_t, B_feed
            )

        result = result.to(A.dtype)
        return result[:M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        # Torch fallback computes the mask + matmul directly.
        pair_bound = Q.unsqueeze(-1) * Q.unsqueeze(0)
        mask = pair_bound > threshold_sqrt
        return (A * mask.to(A.dtype)) @ B


class _ScreenedSpMMFunction(torch.autograd.Function):
    """Autograd wrapper for fused screened SpMM.

    Forward: NKI-dispatched (or PyTorch fallback). Backward: PyTorch-level,
    projecting gradients through the mask.

    The mask depends on `diag_integrals` and `threshold` but is discrete —
    no gradient flows back to them. Gradients flow to `A` (masked) and
    `B` (transposed masked A).
    """

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        diag_integrals: torch.Tensor,
        threshold: float,
        B: torch.Tensor,
    ) -> torch.Tensor:
        import math as _math

        Q = torch.sqrt(torch.abs(diag_integrals))
        threshold_sqrt = _math.sqrt(threshold)
        C = _nki_screened_spmm_impl(A, Q, threshold_sqrt, B)

        # Save the effective mask for backward.
        mask = (Q.unsqueeze(-1) * Q.unsqueeze(0)) > threshold_sqrt
        ctx.save_for_backward(A, B, mask)
        return C

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        A, B, mask = ctx.saved_tensors
        m_f = mask.to(A.dtype)
        grad_A = (grad_out @ B.T) * m_f if ctx.needs_input_grad[0] else None
        grad_B = (A * m_f).T @ grad_out if ctx.needs_input_grad[3] else None
        # No gradient to diag_integrals (arg 1) or threshold (arg 2).
        return grad_A, None, None, grad_B


def nki_screened_spmm(
    A: torch.Tensor,
    diag_integrals: torch.Tensor,
    B: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Screened SpMM entry point — wraps `_ScreenedSpMMFunction.apply` for autograd.

    Native TorchNeuron path: all operands are dense; `A` and
    `diag_integrals` follow B's device, and the `trnsparse::screened_spmm`
    op carries the autograd (no gradient to `diag_integrals` or
    `threshold`). The result is on B's device.
    """
    if _native_path():
        import math as _math

        A = A.to(B.device)
        Q = torch.sqrt(torch.abs(diag_integrals.to(B.device)))
        return _nki_screened_spmm_impl(A, Q, _math.sqrt(threshold), B)
    return _ScreenedSpMMFunction.apply(A, diag_integrals, threshold, B)


def _attn_slot_valid(mask_bsr, M_tiles: int, K_max: int) -> torch.Tensor:
    """(M_tiles, K_max) bool: True where the k-slot holds a real block.

    Rows with fewer than K_max blocks are padded with sentinel slots that
    gather column block 0. Those slots must contribute nothing: callers zero
    the gathered K/V (forward, dQ) or Q/dO (dK/dV) there and drop their
    stats from the host softmax reduction.
    """
    ptrs = mask_bsr.block_row_ptrs.cpu()
    counts = (ptrs[1:] - ptrs[:-1]).to(torch.long)
    return torch.arange(K_max).unsqueeze(0) < counts[:M_tiles].unsqueeze(1)


def _zero_invalid_slots(t: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Zero `t[i, j, ...]` wherever `valid[i, j]` is False."""
    shape = valid.shape + (1,) * (t.dim() - valid.dim())
    return t * valid.view(shape).to(device=t.device, dtype=t.dtype)


def _attn_gather(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask_bsr, scale: float):
    """Host-side gather for the two-pass NKI attention kernels.

    Mirrors `_bsr_pad_and_gather`: builds a (M_tiles, K_max) column grid,
    fancy-indexes K and V blocks into padded (M_tiles, K_max, b, head_dim)
    tensors, and scales Q into (M_tiles, b, head_dim).

    Returns:
        q_scaled_blocks: (M_tiles, b, head_dim)
        k_gathered_pad:  (M_tiles, K_max, b, head_dim)
        v_gathered_pad:  (M_tiles, K_max, b, head_dim)
        K_max:           int — padded number of k-slots per block-row
        M_tiles:         int
    """
    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    M_tiles = seq_len // b

    assert head_dim <= _TILE_K or head_dim % _TILE_K == 0, (
        f"head_dim {head_dim}: must be ≤ {_TILE_K} or a multiple of {_TILE_K}. "
        f"Supported: 32, 64, 128 (single tile) and 256, 512 (K-tiled)."
    )

    # NKI 0.3.0 simulator: nc_matmul requires K = TILE_K exactly. Pad head_dim to
    # TILE_K when running in the simulator with head_dim < TILE_K. The forward
    # output is sliced to [:seq_len, :head_dim] so the padding is transparent.
    head_dim_k = _TILE_K if (_use_simulator() and head_dim < _TILE_K) else head_dim

    # Structure on the host (no-op if it already is); grids are host tensors.
    block_row_ptrs = mask_bsr.block_row_ptrs.cpu()
    block_col_indices = mask_bsr.block_col_indices.cpu()
    blocks_per_row = (block_row_ptrs[1:] - block_row_ptrs[:-1]).tolist()
    K_max = max(blocks_per_row) if blocks_per_row else 1

    # Build (M_tiles, K_max) column grid; -1 for padding slots.
    col_grid = torch.full((M_tiles, K_max), -1, dtype=torch.long)
    for i in range(M_tiles):
        start = block_row_ptrs[i].item()
        end = block_row_ptrs[i + 1].item()
        row_len = end - start
        col_grid[i, :row_len] = block_col_indices[start:end]

    # Gather K and V: reshape (seq_len, head_dim) → (seq_len//b, b, head_dim)
    # then fancy-index by col_grid (use col 0 as sentinel for padded slots).
    nb_rows = seq_len // b
    K_by_block = K.view(nb_rows, b, head_dim)  # (nb_rows, b, head_dim)
    V_by_block = V.view(nb_rows, b, head_dim)

    safe_col = torch.where(col_grid >= 0, col_grid, torch.tensor(0, dtype=torch.long))
    # Padded slots (col_grid == -1) gather block 0; zero K/V there so the
    # slot adds nothing to weights @ V (its stats are dropped on the host).
    valid = col_grid >= 0
    safe_col = safe_col.to(K.device)
    k_gathered_pad = _zero_invalid_slots(K_by_block[safe_col], valid)  # (M_tiles, K_max, b, hd)
    v_gathered_pad = _zero_invalid_slots(V_by_block[safe_col.to(V.device)], valid)

    q_scaled_blocks = (Q * scale).view(M_tiles, b, head_dim)

    if head_dim_k != head_dim:
        extra = head_dim_k - head_dim

        def _pad_hd(t: torch.Tensor, ndim_extra: int) -> torch.Tensor:
            return torch.cat([t, t.new_zeros(*t.shape[:-1], ndim_extra)], dim=-1)

        q_scaled_blocks = _pad_hd(q_scaled_blocks, extra)
        k_gathered_pad = _pad_hd(k_gathered_pad, extra)
        v_gathered_pad = _pad_hd(v_gathered_pad, extra)

    return q_scaled_blocks, k_gathered_pad, v_gathered_pad, K_max, M_tiles


def _attn_host_reduction(
    tile_max: torch.Tensor, tile_sumexp: torch.Tensor, valid: torch.Tensor | None = None
) -> tuple:
    """Reduce per-block stats to per-row softmax denominators.

    Args:
        tile_max:    (M_tiles, K_max, 128) — per-block row-wise max
        tile_sumexp: (M_tiles, K_max, 128) — per-block stable exp-sum

    Returns:
        row_max:   (M_tiles, 128) — row-wise global max
        row_denom: (M_tiles, 128) — softmax denominator

    `valid` (M_tiles, K_max) bool, if given, drops padded k-slots from the
    reduction (rows with no valid slot keep all slots, as before).
    """
    if valid is not None:
        valid = valid.to(tile_max.device)
        valid = valid | ~valid.any(dim=1, keepdim=True)
        v = valid.unsqueeze(-1)
        tile_max = torch.where(v, tile_max, torch.full_like(tile_max, float("-inf")))
        tile_sumexp = torch.where(v, tile_sumexp, torch.zeros_like(tile_sumexp))
    row_max = tile_max.max(dim=1).values  # (M_tiles, 128)
    correction = torch.exp(tile_max - row_max.unsqueeze(1))  # (M_tiles, K_max, 128)
    row_denom = (tile_sumexp * correction).sum(dim=1)  # (M_tiles, 128)
    row_denom = row_denom.clamp(min=1e-12)
    return row_max, row_denom


def nki_bsr_attn_tiled(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask_bsr,
    return_stats: bool = False,
):
    """Two-pass block-sparse attention via NKI kernel pair.

    Orchestrates `_attn_stats_kernel` (pass 1) + host reduction +
    `_attn_out_kernel` (pass 2) for the block-sparse attention forward pass.
    No O(seq_len²) intermediate is allocated.

    head_dim must be ≤ 128 (single tile) or a multiple of 128 (K-tiling:
    256, 512). K-tiling accumulates the score Q @ K.T across TILE_K=128
    chunks, reusing PSUM across nc_matmul calls.

    Args:
        Q, K, V:      (seq_len, head_dim) float tensors.
        mask_bsr:     BSRMatrix with block_size=128 encoding the attention pattern.
        return_stats: If True, returns (out, row_max, row_denom) where row_max
                      and row_denom have shape (M_tiles, block_size). Used by
                      `_AttnTiledFunction.forward` to save stats for backward.

    Returns:
        (seq_len, head_dim) attention output, or a 3-tuple when return_stats=True.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    seq_len, head_dim = Q.shape
    scale = head_dim**-0.5

    if (
        _native_path()
        and head_dim <= _TILE_K
        and mask_bsr.block_size == _TILE_M
        and seq_len % _TILE_M == 0
        and _static_structure_ok(2 * mask_bsr.n_blocks)
    ):
        try:
            return _nki_attn_fwd_static(Q, K, V, mask_bsr, scale, return_stats)
        except Exception:
            if _REQUIRE_NKI:
                raise

    q_scaled, k_gathered, v_gathered, K_max, M_tiles = _attn_gather(Q, K, V, mask_bsr, scale)
    slot_valid = _attn_slot_valid(mask_bsr, M_tiles, K_max)
    b = mask_bsr.block_size

    # Contiguous inputs for kernel dispatch.
    qs = _kernel_input(q_scaled.contiguous())
    kg = _kernel_input(k_gathered.contiguous())
    vg = _kernel_input(v_gathered.contiguous())

    try:
        if _use_simulator():
            tile_max_np, tile_sumexp_np = nki.simulate(_attn_stats_kernel)(
                qs.cpu().numpy(), kg.cpu().numpy()
            )
            tile_max = torch.from_numpy(np.asarray(tile_max_np)).to(Q.device)
            tile_sumexp = torch.from_numpy(np.asarray(tile_sumexp_np)).to(Q.device)
            # NKI 0.3.0 keepdims: tile_max/tile_sumexp are (M_tiles, K_max, 128, 1)
            if tile_max.dim() == 4:
                tile_max = tile_max.squeeze(-1)
                tile_sumexp = tile_sumexp.squeeze(-1)

            row_max, row_denom = _attn_host_reduction(tile_max, tile_sumexp, slot_valid)
            # NKI 0.3.0: row vectors must be 2D for nl.load; unsqueeze (M,128) → (M,128,1)
            rm = row_max.unsqueeze(-1).contiguous()
            rd = row_denom.unsqueeze(-1).contiguous()

            out_np = nki.simulate(_attn_out_kernel)(
                qs.cpu().numpy(),
                kg.cpu().numpy(),
                vg.cpu().numpy(),
                rm.cpu().numpy(),
                rd.cpu().numpy(),
            )
            result = torch.from_numpy(np.asarray(out_np)).to(Q.device)
        else:
            # Q/K/V tiles cross to the device once and feed both passes.
            (qs_x, kg_x, vg_x), orig_device = _to_device(qs, kg, vg)
            tile_max_x, tile_sumexp_x = _run(_attn_stats_kernel, "attn_stats", qs_x, kg_x)
            tile_max = tile_max_x.to(orig_device)
            tile_sumexp = tile_sumexp_x.to(orig_device)
            if tile_max.dim() == 4:
                tile_max = tile_max.squeeze(-1)
                tile_sumexp = tile_sumexp.squeeze(-1)

            row_max, row_denom = _attn_host_reduction(tile_max, tile_sumexp, slot_valid)
            rm = row_max.unsqueeze(-1).contiguous()
            rd = row_denom.unsqueeze(-1).contiguous()

            (rm_x, rd_x), _ = _to_device(rm, rd)
            result_x = _run(_attn_out_kernel, "attn_out", qs_x, kg_x, vg_x, rm_x, rd_x)
            result = result_x.to(orig_device)

        out = result[:seq_len, :head_dim].to(Q.dtype).contiguous()
        if return_stats:
            # row_max/row_denom are already (M_tiles, b) from _attn_host_reduction.
            return out, row_max[:M_tiles, :b].contiguous(), row_denom[:M_tiles, :b].contiguous()
        return out
    except Exception:
        if _REQUIRE_NKI:
            raise
        # PyTorch fallback — import lazily to avoid circular dependency.
        from ..ops import _block_sparse_attn_pytorch

        scale_val = head_dim**-0.5
        if return_stats:
            return _block_sparse_attn_pytorch(Q, K, V, mask_bsr, scale_val, return_stats=True)
        return _block_sparse_attn_pytorch(Q, K, V, mask_bsr, scale_val)


def _nki_attn_fwd_static(Q, K, V, mask_bsr, scale: float, return_stats: bool):
    """Block-sparse attention forward via `trnsparse::attn_fwd_static`.

    One kernel launch over exactly the stored mask blocks (no K_max padding,
    no K/V gather, no host softmax reduction between two launches). Q and K
    are passed transposed (head_dim on partitions) so every load is a plain
    strided DMA.

    bf16 when Q, K and V are all bf16 (fp32 scores, statistics and PSUM;
    P rounded to bf16 for P @ V), otherwise fp32. The statistics come back
    in fp32 either way.
    """
    seq_len, head_dim = Q.shape
    M_tiles = seq_len // mask_bsr.block_size
    row_ptrs, cols = _bsr_structure_lists(mask_bsr)
    orig, dtype = Q.device, Q.dtype
    kdt = _kernel_dtype(Q, K, V)
    Qd, Kd, Vd = (t.to(_NEURON_DEVICE) for t in (Q, K, V))
    q_t = (Qd.float() * scale).to(kdt).T.contiguous()
    Kd, Vd = Kd.to(kdt), Vd.to(kdt)
    out, row_max, row_denom = _get_native_ops()["attn_fwd_static"](
        q_t, Kd.T.contiguous(), Vd.contiguous(), row_ptrs, cols
    )
    out, row_max, row_denom = (t.to(orig) for t in (out, row_max, row_denom))
    out = out.to(dtype)
    if return_stats:
        return out, row_max.reshape(M_tiles, -1), row_denom.reshape(M_tiles, -1)
    return out


def _column_structure(row_ptrs: list[int], cols: list[int], n_col: int):
    """Transpose a block structure: (col_ptrs, rows), rows ascending per column."""
    by_col: list[list[int]] = [[] for _ in range(n_col)]
    for m in range(len(row_ptrs) - 1):
        for idx in range(row_ptrs[m], row_ptrs[m + 1]):
            by_col[cols[idx]].append(m)
    col_ptrs = [0]
    rows: list[int] = []
    for r in by_col:
        rows.extend(r)
        col_ptrs.append(len(rows))
    return col_ptrs, rows


def _nki_attn_bwd_static(Q, K, V, dO, O, mask_bsr, row_max, row_denom, scale: float):
    """Block-sparse attention backward via `trnsparse::attn_bwd_static`.

    One launch for dQ, dK and dV over exactly the stored blocks (row-first
    for dQ, column-first for dK / dV, both structures as kernel constants).
    D = rowsum(dO * O) and 1 / row_denom are computed with device aten ops,
    in fp32.

    bf16 operands when Q, K, V and dO are all bf16 (fp32 statistics, scores
    and PSUM), otherwise fp32.
    """
    seq_len, head_dim = Q.shape
    orig, dtype = Q.device, Q.dtype
    dev = _NEURON_DEVICE
    row_ptrs, cols = _bsr_structure_lists(mask_bsr)
    col_ptrs, rows = _column_structure(row_ptrs, cols, seq_len // mask_bsr.block_size)
    kdt = _kernel_dtype(Q, K, V, dO)
    Q, K, V, dO, O = (t.to(dev) for t in (Q, K, V, dO, O))
    q_s = (Q.float() * scale).to(kdt).contiguous()
    k = K.to(kdt).contiguous()
    do = dO.to(kdt).contiguous()
    rm = row_max.to(device=dev, dtype=torch.float32).reshape(seq_len, 1)
    rinv = 1.0 / row_denom.to(device=dev, dtype=torch.float32).reshape(seq_len, 1)
    d = (dO.float() * O.float()).sum(dim=-1, keepdim=True)
    dQ, dK, dV = _get_native_ops()["attn_bwd_static"](
        q_s.T.contiguous(),
        k.T.contiguous(),
        V.to(kdt).T.contiguous(),
        do.T.contiguous(),
        q_s,
        k,
        do,
        rm.contiguous(),
        rinv.contiguous(),
        d.contiguous(),
        rm.reshape(1, seq_len).contiguous(),
        rinv.reshape(1, seq_len).contiguous(),
        d.reshape(1, seq_len).contiguous(),
        row_ptrs,
        cols,
        col_ptrs,
        rows,
        float(scale),
    )
    return tuple(t.to(device=orig, dtype=dtype) for t in (dQ, dK, dV))


def _attn_bwd_gather(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    O: torch.Tensor,
    mask_bsr,
    scale: float,
    row_max: torch.Tensor,
    row_denom: torch.Tensor,
):
    """Host-side gather for the NKI backward kernels.

    Builds two sets of tensors:
    1. Row-first (for `_attn_bwd_dq_kernel`): same column grid as forward gather.
    2. Column-first (for `_attn_bwd_dkdv_kernel`): BSC transposed view — for each
       column block ki, gather all block-rows m that attend to ki.

    Args:
        Q, K, V, dO, O: (seq_len, head_dim) tensors.
        mask_bsr:        BSRMatrix encoding the attention pattern.
        scale:           QK scale (baked into q_scaled for dQ kernel).
        row_max:         (M_tiles, block_size) saved from forward.
        row_denom:       (M_tiles, block_size) saved from forward.

    Returns a dict with keys:
        row_first: dict with keys q_scaled, k_gathered, v_gathered, do_gathered, D_blocks
        col_first: dict with keys k_blocks, v_blocks, q_gathered, do_gathered, D_gathered,
                                  row_max_gathered, row_denom_gathered
    """
    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    M_tiles = seq_len // b
    nb_rows = M_tiles  # alias
    # Structure on the host (no-op if it already is): the grid loops below
    # read scalars from these, never from device tensors.
    block_row_ptrs = mask_bsr.block_row_ptrs.cpu()
    block_col_indices = mask_bsr.block_col_indices.cpu()

    # NKI 0.3.0 simulator: pad head_dim to TILE_K so nc_matmul gets K=TILE_K.
    # Padding with zeros is correct: [dO|0]·[O|0] = dO·O for D, and the kernel
    # outputs are sliced to [:seq_len, :head_dim] in nki_bsr_attn_bwd.
    if _use_simulator() and head_dim < _TILE_K:
        extra = _TILE_K - head_dim

        def _pad_hd(t: torch.Tensor) -> torch.Tensor:
            return torch.cat([t, t.new_zeros(*t.shape[:-1], extra)], dim=-1)

        Q = _pad_hd(Q)
        K = _pad_hd(K)
        V = _pad_hd(V)
        dO = _pad_hd(dO)
        O = _pad_hd(O)
        head_dim = _TILE_K  # update local for views below

    # ── Row-first tensors (for dQ kernel) ──────────────────────────────────────
    blocks_per_row = (block_row_ptrs[1:] - block_row_ptrs[:-1]).tolist()
    K_max = max(blocks_per_row) if blocks_per_row else 1

    col_grid = torch.full((M_tiles, K_max), -1, dtype=torch.long)
    for i in range(M_tiles):
        start = block_row_ptrs[i].item()
        end = block_row_ptrs[i + 1].item()
        row_len = end - start
        col_grid[i, :row_len] = block_col_indices[start:end]

    safe_col = torch.where(col_grid >= 0, col_grid, torch.tensor(0, dtype=torch.long))

    K_by_block = K.view(nb_rows, b, head_dim)
    V_by_block = V.view(nb_rows, b, head_dim)
    dO_by_block = dO.view(nb_rows, b, head_dim)

    q_scaled = (Q * scale).view(M_tiles, b, head_dim)
    # Padded slots gather block 0; zero K/V there so dS @ K and dO @ V.T
    # contribute nothing to dQ.
    row_valid = col_grid >= 0
    k_gathered = _zero_invalid_slots(K_by_block[safe_col.to(K.device)], row_valid)
    v_gathered = _zero_invalid_slots(V_by_block[safe_col.to(V.device)], row_valid)
    # dP = dO_m @ V_ki.T needs the *row* block dO_m in every k-slot of row m
    # (gathering by column index, as before, fed dO_ki and broke dQ whenever
    # a row attends to a block other than its own).
    do_gathered = dO_by_block.unsqueeze(1).expand(M_tiles, K_max, b, head_dim)

    # D = dO · O row-wise; reshape to (M_tiles, b) for block indexing.
    D_flat = (dO * O).sum(dim=-1)  # (seq_len,)
    D_blocks = D_flat.view(M_tiles, b)  # (M_tiles, b)

    row_first = {
        "q_scaled": q_scaled,
        "k_gathered": k_gathered,
        "v_gathered": v_gathered,
        "do_gathered": do_gathered,
        "D_blocks": D_blocks,
    }

    # ── Column-first tensors (for dK/dV kernel) ────────────────────────────────
    N_col = nb_rows  # number of column blocks = number of row blocks (square)

    # Invert the BSR pattern: for each column block ki, collect all block-rows m.
    col_to_rows: list[list[int]] = [[] for _ in range(N_col)]
    for m in range(M_tiles):
        start = block_row_ptrs[m].item()
        end = block_row_ptrs[m + 1].item()
        for idx in range(start, end):
            ki = block_col_indices[idx].item()
            col_to_rows[ki].append(m)

    K_max_col = max((len(r) for r in col_to_rows), default=1)

    # row_grid_col[ki, slot] = block-row index m, or -1 for padding.
    row_grid_col = torch.full((N_col, K_max_col), -1, dtype=torch.long)
    for ki, rows in enumerate(col_to_rows):
        if rows:
            row_grid_col[ki, : len(rows)] = torch.tensor(rows, dtype=torch.long)

    safe_row = torch.where(row_grid_col >= 0, row_grid_col, torch.tensor(0, dtype=torch.long))

    # K and V in column order: (N_col, b, head_dim) — already row == col block order.
    k_blocks = K.view(N_col, b, head_dim)
    v_blocks = V.view(N_col, b, head_dim)

    # Gather Q_scaled, dO, D by row index; reuse views computed in row-first section.
    # Padded slots gather block-row 0; zero Q/dO there so dS.T @ Q and
    # P.T @ dO contribute nothing to dK/dV.
    col_valid = row_grid_col >= 0
    q_gathered_col = _zero_invalid_slots(q_scaled[safe_row.to(q_scaled.device)], col_valid)
    do_gathered_col = _zero_invalid_slots(dO_by_block[safe_row.to(dO.device)], col_valid)
    D_gathered_col = D_blocks[safe_row.to(D_blocks.device)]  # (N_col, K_max_col, b)
    row_max_gathered_col = row_max[safe_row.to(row_max.device)]  # (N_col, K_max_col, b)
    row_denom_gathered_col = row_denom[safe_row.to(row_denom.device)]  # (N_col, K_max_col, b)

    col_first = {
        "k_blocks": k_blocks,
        "v_blocks": v_blocks,
        "q_gathered": q_gathered_col,
        "do_gathered": do_gathered_col,
        "D_gathered": D_gathered_col,
        "row_max_gathered": row_max_gathered_col,
        "row_denom_gathered": row_denom_gathered_col,
    }

    return row_first, col_first


def nki_bsr_attn_bwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    O: torch.Tensor,
    mask_bsr,
    row_max: torch.Tensor,
    row_denom: torch.Tensor,
) -> tuple:
    """NKI backward pass for block-sparse tiled attention.

    Runs two NKI kernels:
      1. `_attn_bwd_dq_kernel`   — row-first, computes dQ.
      2. `_attn_bwd_dkdv_kernel` — column-first, computes dK and dV.

    Returns:
        (dQ, dK, dV) — same shape as (Q, K, V).
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    scale = head_dim**-0.5

    if (
        _native_path()
        and head_dim <= _TILE_K
        and b == _TILE_M
        and seq_len % _TILE_M == 0
        and _static_structure_ok(2 * mask_bsr.n_blocks)
    ):
        try:
            return _nki_attn_bwd_static(Q, K, V, dO, O, mask_bsr, row_max, row_denom, scale)
        except Exception:
            if _REQUIRE_NKI:
                raise

    row_first, col_first = _attn_bwd_gather(Q, K, V, dO, O, mask_bsr, scale, row_max, row_denom)

    # Pack contiguous inputs. NKI: per-row stat vectors (D_blocks, D/row_max/
    # row_denom gathered, trailing dim = b) must be ≥2D in the kernel —
    # unsqueeze (..., b) → (..., b, 1). Select them by name: a shape test
    # (`ndim <= 3 and shape[-1] == b`) also matched q_scaled / k_blocks /
    # v_blocks, which are (tiles, b, head_dim) and collide when head_dim == b
    # (head_dim=128, or the simulator's padded head_dim).
    _stat_keys = {"D_blocks", "D_gathered", "row_max_gathered", "row_denom_gathered"}

    def _u(key: str, t: torch.Tensor) -> torch.Tensor:
        t = _kernel_input(t)
        if key in _stat_keys:
            return t.unsqueeze(-1).contiguous()
        return t.contiguous()

    rf = {k: _u(k, v) for k, v in row_first.items()}
    cf = {k: _u(k, v) for k, v in col_first.items()}
    rm_in = _kernel_input(row_max).unsqueeze(-1).contiguous()
    rd_in = _kernel_input(row_denom).unsqueeze(-1).contiguous()

    try:
        if _use_simulator():
            dQ_np = nki.simulate(_attn_bwd_dq_kernel)(
                rf["q_scaled"].cpu().numpy(),
                rf["k_gathered"].cpu().numpy(),
                rf["v_gathered"].cpu().numpy(),
                rf["do_gathered"].cpu().numpy(),
                rf["D_blocks"].cpu().numpy(),
                rm_in.cpu().numpy(),
                rd_in.cpu().numpy(),
            )
            dQ_raw = torch.from_numpy(np.asarray(dQ_np)).to(Q.device)

            dK_np, dV_np = nki.simulate(_attn_bwd_dkdv_kernel)(
                cf["k_blocks"].cpu().numpy(),
                cf["v_blocks"].cpu().numpy(),
                cf["q_gathered"].cpu().numpy(),
                cf["do_gathered"].cpu().numpy(),
                cf["D_gathered"].cpu().numpy(),
                cf["row_max_gathered"].cpu().numpy(),
                cf["row_denom_gathered"].cpu().numpy(),
            )
            dK_raw = torch.from_numpy(np.asarray(dK_np)).to(Q.device)
            dV_raw = torch.from_numpy(np.asarray(dV_np)).to(Q.device)
        else:
            (
                (
                    qs_x,
                    kg_x,
                    vg_x,
                    dog_x,
                    db_x,
                    rm_x,
                    rd_x,
                ),
                orig_device,
            ) = _to_device(
                rf["q_scaled"],
                rf["k_gathered"],
                rf["v_gathered"],
                rf["do_gathered"],
                rf["D_blocks"],
                rm_in,
                rd_in,
            )
            dQ_x = _run(
                _attn_bwd_dq_kernel, "attn_bwd_dq", qs_x, kg_x, vg_x, dog_x, db_x, rm_x, rd_x
            )
            dQ_raw = dQ_x.to(orig_device)

            (kb_x, vb_x, qgc_x, dogc_x, dgc_x, rmgc_x, rdgc_x), _ = _to_device(
                cf["k_blocks"],
                cf["v_blocks"],
                cf["q_gathered"],
                cf["do_gathered"],
                cf["D_gathered"],
                cf["row_max_gathered"],
                cf["row_denom_gathered"],
            )
            dK_x, dV_x = _run(
                _attn_bwd_dkdv_kernel,
                "attn_bwd_dkdv",
                kb_x,
                vb_x,
                qgc_x,
                dogc_x,
                dgc_x,
                rmgc_x,
                rdgc_x,
            )
            dK_raw = dK_x.to(orig_device)
            dV_raw = dV_x.to(orig_device)

        # dQ needs scale: kernel gives dS@K (gradient w.r.t. Q_scaled=Q*scale),
        # but dL/dQ = dL/d(Q_scaled)*scale. dK already has scale via q_sbuf=Q*scale.
        return (
            dQ_raw[:seq_len, :head_dim].to(Q.dtype).contiguous() * scale,
            dK_raw[:seq_len, :head_dim].to(Q.dtype).contiguous(),
            dV_raw[:seq_len, :head_dim].to(Q.dtype).contiguous(),
        )
    except Exception:
        if _REQUIRE_NKI:
            raise
        from ..ops import _block_sparse_attn_backward

        return _block_sparse_attn_backward(
            Q, K, V, O, dO, mask_bsr, scale, row_max=row_max, row_denom=row_denom
        )
