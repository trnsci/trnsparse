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

import dataclasses
import functools
import importlib.util
import os
import sys
import warnings

import numpy as np
import torch

from . import kernels as _kernels
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

# float64 / complex128 on the `neuron` device under `auto` (#84): device aten
# computes them at fp32 precision (measured on Trn2), so the op runs in host
# torch instead and the result comes back to the device (`_fp64_host_guard`).
# TRNSPARSE_FP64_ON_DEVICE=1 keeps device aten (fp32 precision, no copies).
_FP64_ON_DEVICE = os.environ.get("TRNSPARSE_FP64_ON_DEVICE", "").lower() in (
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


# Block sizes the BSR / attention kernels take. Every kernel treats a
# block as one 128x128 Tensor Engine tile (`_TILE_M` is built into its
# shapes), so any other block size fails to lower on every call (#46).
_NKI_BLOCK_SIZES = (_TILE_M,)


def _nki_block_ok(op: str, block_size: int) -> bool:
    """True when the NKI kernels behind `op` take `block_size`.

    Callers route to the PyTorch path on False, without tracing a kernel
    that cannot lower. Under `TRNSPARSE_REQUIRE_NKI` an unsupported block
    size raises `NotImplementedError` instead.
    """
    if int(block_size) in _NKI_BLOCK_SIZES:
        return True
    if _REQUIRE_NKI:
        sizes = ", ".join(str(b) for b in _NKI_BLOCK_SIZES)
        raise NotImplementedError(
            f"trnsparse {op}: the NKI kernels support block_size {sizes} only, "
            f"got {block_size}. Rebuild the BSR matrix with block_size={_TILE_M}, "
            "or unset TRNSPARSE_REQUIRE_NKI to use the PyTorch path."
        )
    return False


# ── Kernel operand dtypes: one guard at the kernel boundary ─────────────────
#
# Every NKI kernel computes in fp32 / bf16 (fp32 PSUM). float64, complex,
# integer and bool tensors must never reach one: before 0.12.2 some did, and
# came back as garbage, a dropped imaginary part, or the wrong dtype
# (#78, #79, trnsci/trnsci#35). Two layers:
#
# 1. `_kernel_route` (with `_no_kernel_for_dtype` / `_fp64_route`), at each
#    public entry: which value dtypes a kernel path can take. Complex (outside
#    `bsr_spmm`), integer and bool values have no kernel: they take the PyTorch
#    path (what `set_backend("pytorch")` does), or raise `NotImplementedError`
#    under TRNSPARSE_REQUIRE_NKI. float64 (and complex128 in `bsr_spmm`'s
#    complex kernel path) follows the suite fp64 policy (#84): Trainium
#    computes fp64 at fp32 precision, so under `auto` it takes the PyTorch
#    path at float64 accuracy; under an explicit `set_backend("nki")` it is
#    cast to fp32 for the kernel with a one-time `UserWarning` per op; under
#    TRNSPARSE_REQUIRE_NKI it raises `NotImplementedError`. The result dtype
#    is always the pytorch backend's for the same inputs.
# 2. `_check_kernel_operands`, enforced by every launch itself (the native op
#    table from `_get_native_ops`, `_run` on the XLA path, `_simulate` on the
#    simulator): each tensor operand of each op, by position, must have a dtype
#    in `_KERNEL_OPERAND_DTYPES[op]`, else `TypeError`. A wrapper that forgets
#    layer 1 fails loudly instead of handing a kernel an unsupported dtype.

_F32 = frozenset({torch.float32})
_MM = frozenset({torch.float32, torch.bfloat16})  # static / attention matmul operands
_MM16 = _MM | {torch.float16}  # the gather GEMM kernels also take fp16 (as before)
_ANYF = _MM16 | {torch.float64}  # composite ops: cast internally per kernel
_IDX = frozenset({torch.int32, torch.int64})

# Allowed dtypes per tensor operand, in argument order (non-tensor arguments
# such as structure lists, keys and flags are skipped). Keyed by the short op
# name used in `_native.OPS`, `_launch`, `_run` and `_simulate`.
_KERNEL_OPERAND_DTYPES: dict[str, tuple[frozenset, ...]] = {
    "spmm_dense": (_MM16, _MM16),
    "bsr_spmm": (_MM16, _MM16),
    "screened_spmm": (_MM16, _MM16, _MM16, _MM16),
    "screened_tiles_keyed": (_MM, _F32, _F32, _F32, _MM),  # A, Q col / row, tau, B
    "bsr_spmm_static": (_MM, _MM),
    "bsr_spmm_static_t": (_MM, _MM),
    "bsr_spmm_keyed": (_MM, _MM),
    "bsr_spmm_t_keyed": (_MM, _MM),
    "attn_fwd_keyed": (_MM, _MM, _MM),
    "attn_fwd_static": (_MM, _MM, _MM),
    "attn_bwd_static": (_MM,) * 7 + (_F32,) * 6,
    "attn_stats": (_MM, _MM),
    "attn_out": (_MM, _MM, _MM, _F32, _F32),
    "attn_bwd_dq": (_MM,) * 4 + (_F32,) * 3,
    "attn_bwd_dkdv": (_MM,) * 4 + (_F32,) * 3,
    "cg_bsr_keyed": (_MM, _F32, _F32),
    "power_bsr_keyed": (_MM, _F32),
    # kernel_matmul (#93): X, (Y,) V, (V's noise rows,) params; fp32 throughout.
    "kmv_self": (_F32,) * 3,
    "kmv_cross": (_F32,) * 4,
    "kmv_selfblk": (_F32,) * 5,
    # Composite ops (several kernel launches; each launch is checked again).
    "bsr_attn": (_ANYF, _ANYF, _ANYF, _IDX, _IDX),
    "bsr_attn_bwd": (_ANYF,) * 5 + (_IDX, _IDX) + (_ANYF, _ANYF),
}

# Value dtypes a kernel path takes at the public entry (float64 only through
# the explicit-nki fp32 cast, see `_fp64_route`).
_KERNEL_VALUE_DTYPES = _ANYF

# Dtypes the device computes at fp32 precision only (#84).
_FP64_DTYPES = frozenset({torch.float64, torch.complex128})

# Ops that already warned about the explicit-nki fp32 cast (one warning per op).
_fp64_warned: set[str] = set()


def _check_kernel_operands(op_name: str, args) -> None:
    """Raise `TypeError` unless every tensor in `args` has an allowed dtype for `op_name`."""
    allowed = _KERNEL_OPERAND_DTYPES.get(op_name)
    if allowed is None:
        raise TypeError(f"trnsparse kernel {op_name!r} has no operand dtype table entry")
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    if len(tensors) != len(allowed) or any(
        t.dtype not in ok for t, ok in zip(tensors, allowed, strict=False)
    ):
        got = [str(t.dtype) for t in tensors]
        want = [sorted(str(d) for d in ok) for ok in allowed]
        raise TypeError(
            f"trnsparse kernel {op_name!r}: operand dtypes {got} not allowed (expected {want})"
        )


def _guard_op(op_name: str, op):
    """`op` with its operand dtypes checked on every call (see `_check_kernel_operands`)."""

    def guarded(*args, **kwargs):
        _check_kernel_operands(op_name, args)
        return op(*args, **kwargs)

    guarded.__name__ = guarded.__qualname__ = op_name
    guarded.__wrapped__ = op
    return guarded


def _kernel_values_ok(*tensors: torch.Tensor) -> bool:
    """True when every value operand has a kernel dtype (float64 through a cast)."""
    return all(t.dtype in _KERNEL_VALUE_DTYPES for t in tensors)


def _no_kernel_for_dtype(what: str, *tensors: torch.Tensor) -> None:
    """Under TRNSPARSE_REQUIRE_NKI, refuse value dtypes the kernels do not take.

    Without it the caller runs the PyTorch path instead (as `set_backend("pytorch")`).
    """
    if _REQUIRE_NKI:
        dts = sorted({str(t.dtype) for t in tensors})
        raise NotImplementedError(
            f"trnsparse {what}: the NKI kernels take float32/bfloat16/float16 "
            f"values, got {dts}. Unset TRNSPARSE_REQUIRE_NKI to compute in torch."
        )


def _fp64_route(what: str, *dtypes: torch.dtype) -> bool:
    """The fp64 policy (#84) for value dtypes headed to a kernel.

    True: a kernel path (no float64 / complex128, or an explicit
    `set_backend("nki")`, where the caller casts to fp32 / complex64 and
    this warns once per op). False: the PyTorch path at float64 accuracy
    (backend `auto`). Under TRNSPARSE_REQUIRE_NKI, float64 / complex128
    raise `NotImplementedError` (the kernels cannot give fp64 accuracy).
    """
    wide = sorted({str(d) for d in dtypes if d in _FP64_DTYPES})
    if not wide:
        return True
    if _REQUIRE_NKI:
        raise NotImplementedError(
            f"trnsparse {what}: no NKI kernel computes {', '.join(wide)} at full "
            "precision (Trainium computes float64 at fp32 precision). Unset "
            "TRNSPARSE_REQUIRE_NKI to compute in torch at float64 accuracy, or pass "
            "float32 operands."
        )
    if _backend != "nki":
        return False
    key = what.removesuffix(" backward")
    if key not in _fp64_warned:
        _fp64_warned.add(key)
        warnings.warn(
            f"trnsparse {key}: {', '.join(wide)} operands on the explicit NKI backend "
            "are cast to float32 / complex64 for the kernel, so precision is reduced "
            "to fp32 (the result keeps the pytorch-backend dtype). Use "
            'set_backend("auto") for float64 accuracy in torch.',
            UserWarning,
            stacklevel=3,
        )
    return True


def _kernel_route(what: str, *tensors: torch.Tensor) -> bool:
    """True: a kernel path. False: the PyTorch path.

    Complex / integer / bool values: the PyTorch path (raises under
    REQUIRE_NKI). float64: `_fp64_route` (torch under `auto`, an fp32 cast
    with a one-time warning under explicit `nki`, raise under REQUIRE_NKI).
    """
    if not _kernel_values_ok(*tensors):
        _no_kernel_for_dtype(what, *tensors)
        return False
    return _fp64_route(what, *(t.dtype for t in tensors))


# ── Device-resident float64 under `auto`: host torch (#84) ───────────────────
#
# `auto` promises float64 accuracy for float64 (and complex128) operands. On
# the host that is host torch; on the `neuron` device the torch fallback
# would be device aten, which computes float64 at fp32 precision. So every
# public op is wrapped in `_fp64_host_guard`: under `auto` (not
# TRNSPARSE_REQUIRE_NKI, which keeps raising, and not
# TRNSPARSE_FP64_ON_DEVICE), a float64 / complex128 tensor operand on a
# `neuron` device (directly, or as a CSR / COO / BSR container's values)
# sends the whole call to the host: neuron-resident operands are copied to
# the CPU (indices included), the op runs there (host torch at float64),
# and tensor results go back to the device of the dense operand, keeping
# their dtype. The copies are `.to()` calls, so autograd flows through.
# One `UserWarning` per op.

_fp64_host_warned: set[str] = set()


def _on_neuron_device(t: torch.Tensor) -> bool:
    """True when `t` lives on a `neuron` device (patched in the CPU tests)."""
    return t.device.type == "neuron"


_SPARSE_CONTAINERS: tuple[type, ...] | None = None


def _sparse_containers() -> tuple[type, ...]:
    global _SPARSE_CONTAINERS
    if _SPARSE_CONTAINERS is None:
        from ..formats import BSRMatrix, COOMatrix, CSRMatrix

        _SPARSE_CONTAINERS = (CSRMatrix, COOMatrix, BSRMatrix)
    return _SPARSE_CONTAINERS


def _operand_tensors(x):
    """The tensors in one operand (a tensor, sparse container, tuple or list)."""
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, _sparse_containers()):
        for f in dataclasses.fields(x):
            v = getattr(x, f.name)
            if isinstance(v, torch.Tensor):
                yield v
    elif isinstance(x, tuple | list):
        for y in x:
            yield from _operand_tensors(y)


def _host_copy(t: torch.Tensor) -> torch.Tensor:
    """`t` copied to the CPU (differentiable)."""
    return t.to("cpu")


def _move_neuron_to_host(x):
    """`x` with every neuron-resident tensor (also inside containers) on the CPU."""
    if isinstance(x, torch.Tensor):
        return _host_copy(x) if _on_neuron_device(x) else x
    if isinstance(x, _sparse_containers()):
        moved = {
            f.name: _move_neuron_to_host(getattr(x, f.name))
            for f in dataclasses.fields(x)
            if isinstance(getattr(x, f.name), torch.Tensor)
        }
        return dataclasses.replace(x, **moved)
    if isinstance(x, tuple | list):
        return type(x)(_move_neuron_to_host(y) for y in x)
    return x


def _move_to(x, device: torch.device):
    """Result `x` (tensors, containers, tuples, lists) with its tensors on `device`."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, _sparse_containers()):
        moved = {
            f.name: getattr(x, f.name).to(device)
            for f in dataclasses.fields(x)
            if isinstance(getattr(x, f.name), torch.Tensor)
        }
        return dataclasses.replace(x, **moved)
    if isinstance(x, tuple | list):
        return type(x)(_move_to(y, device) for y in x)
    return x


def _warn_fp64_host(op: str, stacklevel: int = 3) -> None:
    """One UserWarning per op: device-resident float64 computed on the host."""
    if op in _fp64_host_warned:
        return
    _fp64_host_warned.add(op)
    warnings.warn(
        f"trnsparse {op}: float64 on the neuron device is computed on the "
        "host for float64 accuracy (device arithmetic is fp32); pass "
        "float32 to stay on device, or set TRNSPARSE_FP64_ON_DEVICE=1 to "
        "keep device aten at fp32 precision.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _fp64_device_on_host(op: str) -> bool:
    """True, with the one-time warning, when float64 bound for the neuron
    device computes on the host instead: backend `auto`, no
    TRNSPARSE_REQUIRE_NKI, no TRNSPARSE_FP64_ON_DEVICE. For a route that
    `_fp64_host_guard` cannot see (`KernelOperator` keeps its operands)."""
    if _backend != "auto" or _REQUIRE_NKI or _FP64_ON_DEVICE:
        return False
    _warn_fp64_host(op, stacklevel=4)
    return True


def _fp64_host_guard(op: str):
    """Decorator for a public op: device-resident float64 runs in host torch.

    See the section comment above. The result goes back to the device of the
    first neuron-resident dense (non-container) tensor operand: trnsparse
    results land on the dense operand's device. When only a sparse container
    was on the device, the dense operand is on the host and so is the result.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _backend != "auto" or _REQUIRE_NKI or _FP64_ON_DEVICE:
                return fn(*args, **kwargs)
            operands = (*args, *kwargs.values())
            if not any(
                t.dtype in _FP64_DTYPES and _on_neuron_device(t)
                for x in operands
                for t in _operand_tensors(x)
            ):
                return fn(*args, **kwargs)
            home = next(
                (
                    t.device
                    for x in operands
                    if not isinstance(x, _sparse_containers())
                    for t in _operand_tensors(x)
                    if _on_neuron_device(t)
                ),
                None,
            )
            _warn_fp64_host(op, stacklevel=3)
            out = fn(
                *(_move_neuron_to_host(a) for a in args),
                **{k: _move_neuron_to_host(v) for k, v in kwargs.items()},
            )
            return out if home is None else _move_to(out, home)

        return wrapper

    return deco


def _simulate(kernel, op_name: str, *tensors: torch.Tensor):
    """`nki.simulate(kernel)` on host numpy views of `tensors`, dtype-checked."""
    _check_kernel_operands(op_name, tensors)
    return nki.simulate(kernel)(*(_sim_np(t) for t in tensors))


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

        missing = set(_native.OPS) - set(_KERNEL_OPERAND_DTYPES)
        if missing:
            raise TypeError(f"trnsparse native ops without an operand dtype entry: {missing}")
        _native_ops = {name: _guard_op(name, op) for name, op in _native.OPS.items()}
    return _native_ops


def _native_path() -> bool:
    """True when kernels launch as native `trnsparse::` custom ops.

    On this path autograd comes from `register_autograd` on those ops, so the
    host wrappers skip their `torch.autograd.Function`s. The simulator keeps
    its own (autograd.Function) path.
    """
    return HAS_NKI and not _use_simulator() and _get_dispatch_path() == "native"


def _kernel_device(*tensors: torch.Tensor) -> torch.device:
    """The `neuron` device a native kernel launch runs on.

    The logical NeuronCore of the operands that already live on a `neuron`
    device (`neuron:k`, any k), else the default `neuron` device. Host
    operands follow. Operands on two different logical cores are an error:
    TorchNeuron does not move tensors between logical cores (a
    `neuron:0 -> neuron:1` `.to()` returns the tensor still on core 0), so
    route such data through the host explicitly (`trnsparse.multicore`).
    """
    kind = torch.device(_NEURON_DEVICE).type
    found = None
    for t in tensors:
        if t.device.type == kind:
            if found is None:
                found = t.device
            elif t.device != found:
                raise ValueError(
                    f"kernel operands are on two logical NeuronCores ({found} and "
                    f"{t.device}); place them on one core, or use trnsparse.multicore"
                )
    return found if found is not None else torch.device(_NEURON_DEVICE)


def _to_device(*tensors: torch.Tensor):
    """Move kernel operands to the device of the selected dispatch path.

    Returns `(device_tensors, orig_device)`. On the native path that is the
    logical NeuronCore of the operands already on `neuron` (`_kernel_device`).
    The XLA (and "cpu") path is `_to_xla` unchanged.
    """
    if _get_dispatch_path() == "native":
        orig = tensors[0].device
        dev = _kernel_device(*tensors)
        return [t.to(dev) for t in tensors], orig
    return _to_xla(*tensors)


# ── Eager call handles (native path) ─────────────────────────────────────────
#
# Same fix as trnblas#92. A `trnsparse::` custom-op call costs ~0.10 ms of host
# time before the kernel is queued; an aten op costs ~0.02 ms (trn2, SDK 2.33).
# Most of the gap is per-call work that never changes for a given kernel:
#   - nki's `Kernel.__call__` rebuilds the torch handle on every call
#     (`inspect.signature`, a fresh `TorchNativeKernel` hashed into the
#     registry, LNC resolution): ~0.06 ms;
#   - the `torch.library` custom-op layer: ~0.01-0.02 ms.
# So eager calls on plain neuron tensors build a handle once per kernel with
# the public `torch_neuronx.wrap_nki` and call it directly (~0.02 ms). The
# handle is gridless, so NKI resolves the LNC grid itself.
# Static arguments (keyed-structure ids, solver settings, structure lists) pass
# through it with the same conversions the op body applies (`int(key)`,
# `tuple(row_ptrs)`, `float(scale)`), listed per op in `_native.KERNELS`.
# Only ops whose body is a single kernel call qualify: `_native.KERNELS`
# names the kernel each one calls, and the handle wraps that kernel (looked
# up when called, as the op body does; not the caller's), so it runs exactly
# what the op body runs. Ops that are not listed keep the op.
# Under torch.compile, for tensor subclasses (fake / functional), tensors that
# require grad, and int64 inputs (the custom op applies the TorchNeuron int64
# policy), the custom op is used as before.
# `TRNSPARSE_NKI_EAGER_HANDLE=0` turns the fast path off.
#
# This is the one handle cache in the module: `_native_call` (one handle per
# kernel, `shapes=()`) and the `kernel_matmul` launches (`_kmv_launch`, one
# handle per kernel and launch shape, shared by every row block and every
# logical core, #100 / #102) both use `_eager_handle` and `_eager_call_ok`,
# so the env toggle covers both.

_EAGER_HANDLE_ENABLED: bool = os.environ.get(
    "TRNSPARSE_NKI_EAGER_HANDLE", "1"
).strip().lower() not in ("0", "false", "no", "off")
# (id(kernel), shapes) -> (kernel, handle); holding the kernel keeps its id unique.
_eager_handles: dict[tuple, tuple[object, object]] = {}
_wrap_nki: object = None  # torch_neuronx.wrap_nki, resolved on first use; False if absent


def _eager_handle(kernel, shapes=()):
    """Torch call handle for `kernel`, built once per (kernel, `shapes`); None if
    unavailable. `_native_call` passes no shapes (one handle per kernel);
    `_kmv_launch` passes the operand shapes (one handle per launch shape)."""
    key = (id(kernel), tuple(shapes))
    entry = _eager_handles.get(key)
    if entry is not None:
        return entry[1]
    global _wrap_nki
    if _wrap_nki is None:
        try:
            import torch_neuronx

            _wrap_nki = getattr(torch_neuronx, "wrap_nki", False)
        except ImportError:
            _wrap_nki = False
    if _wrap_nki is False or not hasattr(kernel, "func"):
        return None  # no torch_neuronx.wrap_nki, or not an @nki.jit kernel
    handle = _wrap_nki(kernel)
    _eager_handles[key] = (kernel, handle)
    return handle


def _eager_call_ok(tensors) -> bool:
    """True when calling the cached handle is equivalent to the custom op."""
    if not _EAGER_HANDLE_ENABLED or torch.compiler.is_compiling():
        return False
    for t in tensors:
        if (
            type(t) is not torch.Tensor
            or t.device.type != _NEURON_DEVICE
            or t.requires_grad
            or t.dtype == torch.int64
        ):
            return False
    return True


def _native_kernel(op_name: str):
    """`(kernel, conversions)` for the kernel the `trnsparse::<op_name>` op body
    calls (looked up now, as the body does), or None for an op that is not a
    single kernel call. `conversions[i]` is what the body applies to
    positional argument i (`int`, `tuple`, `float`), or None to pass it as is."""
    native = sys.modules.get(__package__ + "._native")  # loaded by _get_native_ops
    entry = None if native is None else native.KERNELS.get(op_name)
    if entry is None:
        return None
    kernel = getattr(entry[0], entry[1], None)
    return None if kernel is None else (kernel, entry[2] if len(entry) > 2 else ())


def _native_call(op_name: str, args):
    """Run `trnsparse::<op_name>` on `args` (tensors already on the device, then
    the op's static arguments): through the cached handle when
    that is equivalent (see above), else the custom op."""
    ops = _get_native_ops()
    if _eager_call_ok([a for a in args if isinstance(a, torch.Tensor)]):
        entry = _native_kernel(op_name)
        if entry is not None:
            # The handle skips the op body, so check the operands here, as the
            # guarded op does (`_guard_op`, trnsci/trnsci#35).
            _check_kernel_operands(op_name, args)
        handle = None if entry is None else _eager_handle(entry[0])
        if handle is not None:
            conv = entry[1]
            if conv:
                args = [
                    conv[i](a) if i < len(conv) and conv[i] is not None else a
                    for i, a in enumerate(args)
                ]
            return handle(*args)
    return ops[op_name](*args)


def _run(kernel, op_name: str, *device_tensors: torch.Tensor):
    """Call one kernel on operands already placed by `_to_device`.

    `op_name` is the `trnsparse::<op_name>` custom op used on the native
    path (eager calls on plain tensors skip the op layer through a cached
    handle, see `_native_call`); the XLA path calls `kernel` directly on
    torch_xla tensors.
    """
    if _get_dispatch_path() == "native":
        return _native_call(op_name, device_tensors)
    _check_kernel_operands(op_name, device_tensors)
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

    float64 reaches a kernel wrapper only on the explicit `nki` backend
    (`_fp64_route`, #84; `auto` takes torch). Trainium has no fp64 datapath; the native device silently hands fp64
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


def _attn_operands(kdt: torch.dtype, *ts: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Gather-attention operands in the kernel dtype `kdt` (from `_kernel_dtype`).

    The gather kernels, like the static ones, need every matmul operand of a
    call in one dtype (#49). Lower-precision operands of a mixed call are
    upcast (lossless); fp64 is left for `_kernel_input` to downcast, as before.
    """
    return tuple(t if t.dtype in (kdt, torch.float64) else t.to(kdt) for t in ts)


def _stat_input(t: torch.Tensor) -> torch.Tensor:
    """Softmax statistics / D for the gather kernels: fp32 (fp64 downcast as before)."""
    return t if t.dtype == torch.float32 else t.float()


def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Host arithmetic dtype for scale and D = rowsum(dO * O): at least fp32."""
    return torch.promote_types(dtype, torch.float32)


def _sim_np(t: torch.Tensor) -> np.ndarray:
    """Host numpy view of a kernel operand for `nki.simulate` (bf16 via ml_dtypes)."""
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        import ml_dtypes

        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _sim_torch(a, device) -> torch.Tensor:
    """`nki.simulate` output as a torch tensor (bf16 arrays widen to fp32, exactly)."""
    a = np.asarray(a)
    if a.dtype.name == "bfloat16":
        a = a.astype(np.float32)
    return torch.from_numpy(a).to(device)


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
            out_np = _simulate(_spmm_dense_kernel, "spmm_dense", A_feed, B_feed)
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
    if not _kernel_route("spmm", A.values, B):
        from ..ops import _csr_matmul

        return _csr_matmul(A, B)
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
            out_np = _simulate(_bsr_spmm_kernel, "bsr_spmm", bp_feed, bg_feed)
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
    if not _nki_block_ok("bsr_spmm", A.block_size):
        from ..ops import _bsr_spmm_pytorch

        return _bsr_spmm_pytorch(A, B)
    if A.blocks.is_complex() or B.is_complex():
        # complex128 / float64 parts: the fp64 policy (#84) before the real launches.
        if not _fp64_route("bsr_spmm", A.blocks.dtype, B.dtype):
            from ..ops import _bsr_spmm_pytorch

            return _bsr_spmm_pytorch(A, B)
        return _nki_bsr_spmm_complex(A, B)  # real kernel launches (#78)
    if not _kernel_route("bsr_spmm", A.blocks, B):
        from ..ops import _bsr_spmm_pytorch

        return _bsr_spmm_pytorch(A, B)
    if _native_path():
        out_cols = B.shape[1]
        n_tiles = 1 if out_cols <= _TILE_N else -(-out_cols // _TILE_N)
        if _static_structure_ok(A.n_blocks * n_tiles):
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


def _nki_bsr_spmm_complex(A, B: torch.Tensor) -> torch.Tensor:
    """Complex BSR SpMM as real kernel launches (#78).

    The kernels are real-only. Through 0.12.1 the static path cast complex
    operands to fp32, which silently dropped the imaginary part. Here
    C_re = Ar·Br − Ai·Bi and C_im = Ar·Bi + Ai·Br, with one launch per real
    part of A against `[Br | Bi]` (columns side by side), so a complex A and
    complex B take two launches. `.real` / `.imag` / `torch.complex` are
    differentiable, so autograd reaches complex `A.blocks` and `B`.
    """
    out_dtype = torch.promote_types(A.blocks.dtype, B.dtype)
    n = B.shape[1]
    b_parts = torch.cat([B.real, B.imag], dim=1) if B.is_complex() else B
    blocks = A.blocks
    a_parts = (blocks.real, blocks.imag) if blocks.is_complex() else (blocks,)
    prods = [nki_bsr_spmm(_BlockView(A, a), b_parts) for a in a_parts]
    if not B.is_complex():  # complex A, real B
        re, im = prods
    elif len(prods) == 1:  # real A, complex B
        re, im = prods[0][:, :n], prods[0][:, n:]
    else:
        (rr, ri), (ir, ii) = ((p[:, :n], p[:, n:]) for p in prods)
        re, im = rr - ii, ri + ir
    return torch.complex(re.contiguous(), im.contiguous()).to(out_dtype)


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
    dev = _kernel_device(B, A.blocks)
    blocks = A.blocks.contiguous().to(device=dev, dtype=kdt)
    b_dev = B.contiguous().to(device=dev, dtype=kdt)
    key = _kernels.register_structure(row_ptrs, cols)
    out = _native_call("bsr_spmm_keyed", [blocks, b_dev, key])
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
            out_np = _simulate(
                _screened_spmm_kernel, "screened_spmm", A_feed, Q_feed, threshold_sqrt_t, B_feed
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


# Screened SpMM with tile skipping (#61, native path). Off restores the
# all-tiles fused kernel.
_SCREEN_SKIP_TILES = os.environ.get("TRNSPARSE_SCREEN_SKIP_TILES", "1").lower() not in (
    "0",
    "false",
    "no",
)


def screened_tile_structure(Q: torch.Tensor, threshold_sqrt: float, n: int | None = None):
    """Classify the 128 x 128 tiles of a Schwarz-screened square matrix.

    `Q` is the bound vector (`sqrt(|diag|)`), `mask[i, j] = Q[i] Q[j] > tau`
    with `tau = threshold_sqrt`, in fp32 as the kernel computes it. Tile
    (m, k) is skipped when its largest pair bound `max(Q_m) max(Q_k)` is
    <= tau, full when its smallest `min(Q_m) min(Q_k)` (over the real
    indices; padding is zero in A) is > tau, else partial. Q >= 0 and fp32
    rounding is monotone, so the classification is exact: a skipped tile
    has every element masked and a full tile none.

    Returns `(row_ptrs, cols, partial)` as int lists, BSR-style over the
    kept tiles, and the kept-tile fraction.
    """
    q = Q.detach().to(device="cpu", dtype=torch.float32).abs().reshape(-1)
    n = q.numel() if n is None else n
    T = _TILE_M
    nt = (n + T - 1) // T
    q_pad = torch.zeros(nt * T, dtype=torch.float32)
    q_pad[:n] = q[:n]
    big = torch.full((nt * T,), float("inf"), dtype=torch.float32)
    big[:n] = q[:n]
    qmax = q_pad.reshape(nt, T).amax(dim=1)
    qmin = big.reshape(nt, T).amin(dim=1)
    tau = torch.tensor(threshold_sqrt, dtype=torch.float32)
    keep = (qmax[:, None] * qmax[None, :]) > tau
    full = (qmin[:, None] * qmin[None, :]) > tau
    counts = keep.sum(dim=1)
    row_ptrs = [0] + torch.cumsum(counts, 0).tolist()
    mi, kj = keep.nonzero(as_tuple=True)
    cols = kj.tolist()
    partial = (~full[mi, kj]).to(torch.int64).tolist()
    frac = float(keep.sum()) / float(nt * nt) if nt else 0.0
    return row_ptrs, cols, partial, frac


# Screening plans by bound vector (#61): the tile classification, its key
# and the device copies of Q / tau, reused while the same `diag_integrals`
# tensor (same object, unmodified) comes back with the same threshold.
# Each entry holds a reference to its tensor, so the id cannot be reused.
_SCREEN_PLANS: dict = {}
_SCREEN_PLANS_MAX = 16


def _screen_plan(diag_integrals: torch.Tensor, threshold: float, M: int, dev, symmetric=False):
    """(key, n_kept, q_col, q_row, thr) for a square screened product."""
    import math as _math

    ck = (id(diag_integrals), float(threshold), int(M), str(dev), bool(symmetric))
    hit = _SCREEN_PLANS.get(ck)
    if hit is not None and hit[0] is diag_integrals and hit[1] == diag_integrals._version:
        return hit[2]
    tau = _math.sqrt(threshold)
    Q = torch.sqrt(torch.abs(diag_integrals.detach().to(device="cpu", dtype=torch.float32)))
    row_ptrs, cols, partial, _ = screened_tile_structure(Q, tau, M)
    M_pad = _round_up(M, _TILE_M)
    q_h = torch.zeros(M_pad, dtype=torch.float32)
    q_h[:M] = Q.reshape(-1)[:M]
    plan = (
        _kernels.register_structure(row_ptrs, cols, partial, (int(bool(symmetric)),))
        if cols
        else None,
        len(cols),
        q_h.reshape(M_pad, 1).to(dev),
        q_h.reshape(1, M_pad).to(dev),
        torch.full((_TILE_M, 1), tau, dtype=torch.float32).to(dev),
    )
    if len(_SCREEN_PLANS) >= _SCREEN_PLANS_MAX:
        _SCREEN_PLANS.pop(next(iter(_SCREEN_PLANS)))
    _SCREEN_PLANS[ck] = (diag_integrals, diag_integrals._version, plan)
    return plan


def _nki_screened_tiles(
    A: torch.Tensor, diag_integrals: torch.Tensor, threshold: float, B, symmetric: bool = False
):
    """Screened SpMM through `trnsparse::screened_tiles_keyed` (native path).

    A, B stay dense; only the tile classification of the bounds is a
    compile-time key (`screened_tile_structure`). Returns None when the
    structure is too large to unroll, or when nothing survives the screen
    and A or B needs a gradient (the caller then takes the all-tiles kernel).
    """
    M, _ = A.shape
    _, N = B.shape
    orig = B.device
    dev = _kernel_device(B, A)
    key, n_kept, q_col, q_row, thr = _screen_plan(diag_integrals, threshold, M, dev, symmetric)
    n_nt = 1 if N <= _TILE_N else _round_up(N, _TILE_N) // _TILE_N
    if not _static_structure_ok(n_kept * n_nt):
        return None
    out_dtype = torch.promote_types(A.dtype, B.dtype)
    if not n_kept:
        if A.requires_grad or B.requires_grad:
            return None
        return torch.zeros(M, N, dtype=out_dtype, device=orig)
    kdt = _kernel_dtype(A, B)
    M_pad = _round_up(M, _TILE_M)
    N_pad = N if N <= _TILE_N else _round_up(N, _TILE_N)
    if M_pad != M:
        a_dev = torch.zeros(M_pad, M_pad, dtype=kdt, device=dev)
        a_dev[:M, :M] = A.to(device=dev, dtype=kdt)
    else:
        a_dev = A.contiguous().to(device=dev, dtype=kdt)
    if M_pad != M or N_pad != N:
        b_dev = torch.zeros(M_pad, N_pad, dtype=kdt, device=dev)
        b_dev[:M, :N] = B.to(device=dev, dtype=kdt)
    else:
        b_dev = B.contiguous().to(device=dev, dtype=kdt)
    out = _get_native_ops()["screened_tiles_keyed"](a_dev, q_col, q_row, thr, b_dev, key)
    if M_pad != M or N_pad != N:
        out = out[:M, :N]
    return out.to(device=orig, dtype=out_dtype)


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
    symmetric: bool = False,
) -> torch.Tensor:
    """Screened SpMM entry point — wraps `_ScreenedSpMMFunction.apply` for autograd.

    Native TorchNeuron path: all operands are dense; `A` and
    `diag_integrals` follow B's device, and the `trnsparse::screened_spmm`
    op carries the autograd (no gradient to `diag_integrals` or
    `threshold`). The result is on B's device.
    """
    if not _kernel_route("screened_spmm", A, B):
        from ..ops import _screened_spmm_pytorch

        return _screened_spmm_pytorch(A, diag_integrals, B, threshold)
    if B.dim() == 1:  # matrix-vector: one right-hand side
        return nki_screened_spmm(A, diag_integrals, B[:, None], threshold, symmetric)[:, 0]
    if _native_path():
        import math as _math

        if (
            _SCREEN_SKIP_TILES
            and A.dim() == 2
            and A.shape[0] == A.shape[1]
            and B.dim() == 2
            and diag_integrals.dim() == 1
            and diag_integrals.shape[0] == A.shape[0]
            # The bounds are compared in fp32 on chip; a bf16 / fp16 bound
            # vector's mask (compared in its own dtype) takes the fused kernel.
            and diag_integrals.dtype in (torch.float32, torch.float64)
        ):
            out = _nki_screened_tiles(A, diag_integrals, threshold, B, symmetric)
            if out is not None:
                return out
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


def _attn_kernel_head_dim(head_dim: int) -> int:
    """head_dim the gather attention kernels run at; inputs are zero-padded to it.

    The K-tiled kernels take head_dim in whole 128-column chunks, so a head_dim
    above 128 that is not a multiple of 128 (e.g. 192) is padded up to the next
    one (#49); before, it raised. Zero columns change no score, add zero
    output / gradient columns, and are sliced off. NKI 0.3.0 simulator:
    nc_matmul requires K = TILE_K exactly, so there head_dim < 128 is padded
    to 128 as well.
    """
    if head_dim > _TILE_K:
        return -(-head_dim // _TILE_K) * _TILE_K
    if _use_simulator():
        return _TILE_K
    return head_dim


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

    head_dim_k = _attn_kernel_head_dim(head_dim)

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

    # Scale in at least fp32, then round to Q's dtype (bf16), as the static path does.
    q_scaled_blocks = (Q.to(_acc_dtype(Q.dtype)) * scale).to(Q.dtype).view(M_tiles, b, head_dim)

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
    scale: float | None = None,
):
    """Two-pass block-sparse attention via NKI kernel pair.

    Orchestrates `_attn_stats_kernel` (pass 1) + host reduction +
    `_attn_out_kernel` (pass 2) for the block-sparse attention forward pass.
    No O(seq_len²) intermediate is allocated.

    head_dim ≤ 128 runs as a single tile; above 128 it is K-tiled in 128-column
    chunks (256, 512; other sizes such as 192 are zero-padded to the next
    multiple of 128). K-tiling accumulates the score Q @ K.T across TILE_K=128
    chunks, reusing PSUM across nc_matmul calls.

    Args:
        Q, K, V:      (seq_len, head_dim) float tensors.
        mask_bsr:     BSRMatrix with block_size=128 encoding the attention pattern.
        return_stats: If True, returns (out, row_max, row_denom) where row_max
                      and row_denom have shape (M_tiles, block_size). Used by
                      `_AttnTiledFunction.forward` to save stats for backward.
        scale:        QK scale; None means head_dim ** -0.5. Applied to Q on
                      the host/device before the kernels (they take Q * scale).

    Returns:
        (seq_len, head_dim) attention output, or a 3-tuple when return_stats=True.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    seq_len, head_dim = Q.shape
    scale = head_dim**-0.5 if scale is None else float(scale)

    if not _nki_block_ok("block_sparse_attention_tiled", mask_bsr.block_size):
        from ..ops import _block_sparse_attn_pytorch

        return _block_sparse_attn_pytorch(Q, K, V, mask_bsr, scale, return_stats=return_stats)
    if not _kernel_route("block_sparse_attention_tiled", Q, K, V):
        from ..ops import _block_sparse_attn_pytorch

        return _block_sparse_attn_pytorch(Q, K, V, mask_bsr, scale, return_stats=return_stats)

    if (
        _native_path()
        and head_dim <= _TILE_K
        and seq_len % _TILE_M == 0
        and _static_structure_ok(2 * mask_bsr.n_blocks)
    ):
        try:
            return _nki_attn_fwd_static(Q, K, V, mask_bsr, scale, return_stats)
        except Exception:
            if _REQUIRE_NKI:
                raise

    # One operand dtype for the gather kernels: bf16 when Q, K and V all are,
    # else fp32. Statistics, scores and PSUM are fp32 either way (#49).
    Qk, Kk, Vk = _attn_operands(_kernel_dtype(Q, K, V), Q, K, V)
    q_scaled, k_gathered, v_gathered, K_max, M_tiles = _attn_gather(Qk, Kk, Vk, mask_bsr, scale)
    slot_valid = _attn_slot_valid(mask_bsr, M_tiles, K_max)
    b = mask_bsr.block_size

    # Contiguous inputs for kernel dispatch.
    qs = _kernel_input(q_scaled.contiguous())
    kg = _kernel_input(k_gathered.contiguous())
    vg = _kernel_input(v_gathered.contiguous())

    try:
        if _use_simulator():
            tile_max_np, tile_sumexp_np = _simulate(_attn_stats_kernel, "attn_stats", qs, kg)
            tile_max = _sim_torch(tile_max_np, Q.device)
            tile_sumexp = _sim_torch(tile_sumexp_np, Q.device)
            # NKI 0.3.0 keepdims: tile_max/tile_sumexp are (M_tiles, K_max, 128, 1)
            if tile_max.dim() == 4:
                tile_max = tile_max.squeeze(-1)
                tile_sumexp = tile_sumexp.squeeze(-1)

            row_max, row_denom = _attn_host_reduction(tile_max, tile_sumexp, slot_valid)
            # NKI 0.3.0: row vectors must be 2D for nl.load; unsqueeze (M,128) → (M,128,1)
            rm = row_max.unsqueeze(-1).contiguous()
            rd = row_denom.unsqueeze(-1).contiguous()

            out_np = _simulate(_attn_out_kernel, "attn_out", qs, kg, vg, rm, rd)
            result = _sim_torch(out_np, Q.device)
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

        if return_stats:
            return _block_sparse_attn_pytorch(Q, K, V, mask_bsr, scale, return_stats=True)
        return _block_sparse_attn_pytorch(Q, K, V, mask_bsr, scale)


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
    dev = _kernel_device(Q, K, V)
    Qd, Kd, Vd = (t.to(dev) for t in (Q, K, V))
    q_t = (Qd.float() * scale).to(kdt).T.contiguous()
    Kd, Vd = Kd.to(kdt), Vd.to(kdt)
    key = _kernels.register_structure(row_ptrs, cols)
    out, row_max, row_denom = _native_call(
        "attn_fwd_keyed", [q_t, Kd.T.contiguous(), Vd.contiguous(), key]
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
    dev = _kernel_device(Q, K, V, dO, O)
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
    dQ, dK, dV = _native_call(
        "attn_bwd_static",
        [
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
        ],
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

    # Pad head_dim to what the kernels take (`_attn_kernel_head_dim`: a whole
    # number of 128 chunks above 128; 128 in the NKI 0.3.0 simulator).
    # Padding with zeros is correct: [dO|0]·[O|0] = dO·O for D, and the kernel
    # outputs are sliced to [:seq_len, :head_dim] in nki_bsr_attn_bwd.
    head_dim_k = _attn_kernel_head_dim(head_dim)
    if head_dim_k != head_dim:
        extra = head_dim_k - head_dim

        def _pad_hd(t: torch.Tensor) -> torch.Tensor:
            return torch.cat([t, t.new_zeros(*t.shape[:-1], extra)], dim=-1)

        Q = _pad_hd(Q)
        K = _pad_hd(K)
        V = _pad_hd(V)
        dO = _pad_hd(dO)
        O = _pad_hd(O)
        head_dim = head_dim_k  # update local for views below

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

    # Scale and D in at least fp32 (Q's dtype for the kernel operands), as
    # the static path does (#49).
    acc = _acc_dtype(Q.dtype)
    q_scaled = (Q.to(acc) * scale).to(Q.dtype).view(M_tiles, b, head_dim)
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
    D_flat = (dO.to(acc) * O.to(acc)).sum(dim=-1)  # (seq_len,)
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
    scale: float | None = None,
) -> tuple:
    """NKI backward pass for block-sparse tiled attention.

    Runs two NKI kernels:
      1. `_attn_bwd_dq_kernel`   — row-first, computes dQ.
      2. `_attn_bwd_dkdv_kernel` — column-first, computes dK and dV.

    `scale` must be the forward's QK scale (None means head_dim ** -0.5).

    Returns:
        (dQ, dK, dV) — same shape as (Q, K, V).
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    seq_len, head_dim = Q.shape
    b = mask_bsr.block_size
    scale = head_dim**-0.5 if scale is None else float(scale)

    if not _nki_block_ok("block_sparse_attention_tiled backward", b):
        from ..ops import _block_sparse_attn_backward

        return _block_sparse_attn_backward(
            Q, K, V, O, dO, mask_bsr, scale, row_max=row_max, row_denom=row_denom
        )
    if not _kernel_route("block_sparse_attention_tiled backward", Q, K, V, dO, O):
        from ..ops import _block_sparse_attn_backward

        return _block_sparse_attn_backward(
            Q, K, V, O, dO, mask_bsr, scale, row_max=row_max, row_denom=row_denom
        )

    if (
        _native_path()
        and head_dim <= _TILE_K
        and seq_len % _TILE_M == 0
        and _static_structure_ok(2 * mask_bsr.n_blocks)
    ):
        try:
            return _nki_attn_bwd_static(Q, K, V, dO, O, mask_bsr, row_max, row_denom, scale)
        except Exception:
            if _REQUIRE_NKI:
                raise

    # One operand dtype for the gather kernels: bf16 when Q, K, V and dO all
    # are, else fp32. The statistics and D are fp32 either way (#49).
    Qk, Kk, Vk, dOk = _attn_operands(_kernel_dtype(Q, K, V, dO), Q, K, V, dO)
    row_first, col_first = _attn_bwd_gather(
        Qk, Kk, Vk, dOk, O, mask_bsr, scale, _stat_input(row_max), _stat_input(row_denom)
    )

    # Pack contiguous inputs. NKI: per-row stat vectors (D_blocks, D/row_max/
    # row_denom gathered, trailing dim = b) must be ≥2D in the kernel —
    # unsqueeze (..., b) → (..., b, 1). Select them by name: a shape test
    # (`ndim <= 3 and shape[-1] == b`) also matched q_scaled / k_blocks /
    # v_blocks, which are (tiles, b, head_dim) and collide when head_dim == b
    # (head_dim=128, or the simulator's padded head_dim).
    _stat_keys = {"D_blocks", "D_gathered", "row_max_gathered", "row_denom_gathered"}

    def _u(key: str, t: torch.Tensor) -> torch.Tensor:
        if key in _stat_keys:
            return _stat_input(t).unsqueeze(-1).contiguous()
        return _kernel_input(t).contiguous()

    rf = {k: _u(k, v) for k, v in row_first.items()}
    cf = {k: _u(k, v) for k, v in col_first.items()}
    rm_in = _stat_input(row_max).unsqueeze(-1).contiguous()
    rd_in = _stat_input(row_denom).unsqueeze(-1).contiguous()

    try:
        if _use_simulator():
            dQ_np = _simulate(
                _attn_bwd_dq_kernel,
                "attn_bwd_dq",
                rf["q_scaled"],
                rf["k_gathered"],
                rf["v_gathered"],
                rf["do_gathered"],
                rf["D_blocks"],
                rm_in,
                rd_in,
            )
            dQ_raw = _sim_torch(dQ_np, Q.device)

            dK_np, dV_np = _simulate(
                _attn_bwd_dkdv_kernel,
                "attn_bwd_dkdv",
                cf["k_blocks"],
                cf["v_blocks"],
                cf["q_gathered"],
                cf["do_gathered"],
                cf["D_gathered"],
                cf["row_max_gathered"],
                cf["row_denom_gathered"],
            )
            dK_raw = _sim_torch(dK_np, Q.device)
            dV_raw = _sim_torch(dV_np, Q.device)
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
        # The scale is applied in at least fp32 before rounding to Q's dtype.
        dq = dQ_raw[:seq_len, :head_dim]
        return (
            (dq.to(_acc_dtype(Q.dtype)) * scale).to(Q.dtype).contiguous(),
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


# ── Fused kernel-matrix product K(X, Y) @ V (#93) ────────────────────────────
#
# Eager native launches go through a `torch_neuronx.wrap_nki` handle built
# once per kernel (the trnblas#92 pattern, aws-neuron-sdk#1456): the custom-op
# layer and nki `Kernel.__call__` rebuild a handle on every call, about
# 0.08 ms. A CG iteration is one launch, so that floor is per iteration.
# The cache and gate are the module's (`_eager_handle` / `_eager_call_ok`,
# above), keyed here by (kernel, operand shapes).
# TRNSPARSE_NKI_EAGER_HANDLE=0 uses the `trnsparse::kmv_*` custom ops instead.


def _kmv_block_work() -> int:
    """Per-launch budget in kernel entries, rows x m (`kernels._KMV_BLOCK_WORK`).

    TRNSPARSE_KMV_BLOCK_WORK overrides it (tests force tiny blocks with it).
    """
    env = os.environ.get("TRNSPARSE_KMV_BLOCK_WORK", "").strip()
    return max(1, int(float(env))) if env else _kernels._KMV_BLOCK_WORK


def _kmv_launch(x, y, v, vn, params, kind: int, group: int, sim: bool) -> torch.Tensor:
    """One kernel launch on operands already placed (device, or host for the simulator)."""
    from . import kernels as _k

    if y is None:
        kernel, name, tensors = _k._kmv_self_kernel, "kmv_self", (x, v, params)
    elif vn is None:
        kernel, name, tensors = _k._kmv_cross_kernel, "kmv_cross", (x, y, v, params)
    else:
        kernel, name, tensors = _k._kmv_selfblk_kernel, "kmv_selfblk", (x, y, v, vn, params)
    # Every route below (simulator, eager handle, custom op, XLA) checks the
    # operand dtypes first, as `_launch` / `_run` / `_simulate` do.
    _check_kernel_operands(name, tensors)
    if sim:
        out = nki.simulate(kernel)(*(_sim_np(t) for t in tensors), int(kind), int(group))
        return _sim_torch(out, v.device)
    if _get_dispatch_path() == "native":
        shapes = tuple(tuple(t.shape) for t in tensors)
        handle = _eager_handle(kernel, shapes) if _eager_call_ok(tensors) else None
        if handle is not None:
            return handle(*tensors, int(kind), int(group))
        return _get_native_ops()[name](*tensors, int(kind), int(group))
    return kernel(*tensors, int(kind), int(group))


# ── Row blocks across logical NeuronCores (#102) ─────────────────────────────
#
# The row blocks of one product are independent, so on a chip with several
# visible logical cores (trn2.3xlarge at LNC=2: neuron:0..3 with
# NEURON_RT_VISIBLE_CORES=0-3) they are spread over the cores. Placement
# follows `trnsparse.multicore` and trnblas#24: TorchNeuron does not copy
# between logical cores, and a copy out of a busy core waits behind its
# queued kernels, so the operands are staged through the host once, before
# any kernel is queued, and each core gets its own Y, V and params. Launches
# are asynchronous: all of them are issued from the calling thread,
# round-robin over the cores, so the cores run together and NKI tracing
# (not thread-safe, aws-neuron-sdk#1453) stays on one thread. The results
# come back through the host in row order.
#
# TRNSPARSE_KMV_CORES: "auto" (default) uses every visible core when the
# product is row-blocked anyway (n * m above `_kmv_block_work()`), and one
# core below that; "all" or an int k always splits over that many cores
# (at most the visible count). "1" keeps everything on the operands' core.
# TRNSPARSE_KMV_THREADS=1 issues each core's blocks from its own worker
# thread instead (a fallback in case launches block the host); the first
# block of every core is launched on the calling thread first, so the
# cached `wrap_nki` handle is traced there.

_KMV_CORES_ENV = "TRNSPARSE_KMV_CORES"
_kmv_pool = None


def _kmv_cores_setting() -> int | str:
    """`TRNSPARSE_KMV_CORES` parsed: "auto", "all" or a positive int."""
    raw = os.environ.get(_KMV_CORES_ENV, "").strip().lower()
    if raw in ("", "auto"):
        return "auto"
    if raw == "all":
        return "all"
    try:
        k = int(raw)
    except ValueError:
        k = 0
    if k < 1:
        raise ValueError(f"{_KMV_CORES_ENV} must be a positive int, 'all' or 'auto'; got {raw!r}")
    return k


def _kmv_threads() -> bool:
    raw = os.environ.get("TRNSPARSE_KMV_THREADS", "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _visible_cores() -> int:
    """Visible logical NeuronCores (`torch_neuronx.device_count()`), 1 if unknown."""
    try:
        import torch_neuronx

        return max(1, int(torch_neuronx.device_count()))
    except Exception:
        return 1


def _kmv_core_count(n: int, m: int, visible: int, setting: int | str, work: int) -> int:
    """Logical cores one n x m product is split over (1: the single-core path)."""
    if visible <= 1 or setting == 1:
        return 1
    if setting == "auto":
        k = visible if n * m > work else 1
    elif setting == "all":
        k = visible
    else:
        k = min(int(setting), visible)
    return max(1, k)


def _kmv_core_devices(n: int, m: int, tensors) -> list[torch.device]:
    """The cores for one product, the operands' core first; one entry = single core."""
    if _get_dispatch_path() != "native":
        return []
    k = _kmv_core_count(n, m, _visible_cores(), _kmv_cores_setting(), _kmv_block_work())
    if k <= 1:
        return []
    home = _kernel_device(*tensors)
    if home.index is None:
        try:
            import torch_neuronx

            home = torch.device(_NEURON_DEVICE, int(torch_neuronx.current_device()))
        except Exception:
            home = torch.device(_NEURON_DEVICE, 0)
    visible = _visible_cores()
    return [torch.device(_NEURON_DEVICE, (home.index + i) % visible) for i in range(k)]


def _kmv_place(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """A host operand copied to one core (the CPU tests swap in a copy)."""
    return t.to(device)


def _kmv_block(xs, ys, vs, ps, i0: int, rows: int, n: int, kind, group, sim):
    """Launch the row block of x starting at i0 on the core its operands are on.

    `ys` None is the square operator (Y = X): the block takes its rows of V
    for the noise term (`_kmv_selfblk_kernel`).

    The ragged last block is zero-padded to `rows` (padding rows are computed
    and dropped), so every block has one shape and one handle. Returns the
    (rows, R) block output; rows past n are padding.
    """
    nb = min(rows, n - i0)
    self_op = ys is None
    xb = xs[i0 : i0 + nb]
    vb = vs[i0 : i0 + nb] if self_op else None
    if nb < rows:
        xb = torch.cat([xb, xb.new_zeros(rows - nb, xs.shape[1])])
        if self_op:
            vb = torch.cat([vb, vb.new_zeros(rows - nb, vs.shape[1])])
    return _kmv_launch(xb, xs if self_op else ys, vs, vb, ps, kind, group, sim)


def _kmv_multicore(tensors, self_op: bool, n: int, R: int, rows: int, devices, kind, group):
    """Row blocks of one product spread over `devices`, gathered on the host.

    Block b runs on core b % k (k = len(devices)), with the operands placed
    on that core once. `rows` comes from `kmv_block_rows(..., parts=k)`, so
    with a block count that is a multiple of k every core gets the same
    number of blocks. Every block is the same kernel at one shape, and a row
    of the product runs the same instructions whatever block it is in (the
    #100/#101 bitwise check on Trn2), so the result is bit-identical to
    single-core.
    """
    orig = tensors[0].device
    # Stage through the host first: a D2H copy queues behind the kernels of
    # the core it reads from, so nothing is launched until every core has
    # its operands.
    host = [t.detach().cpu() for t in tensors]
    placed = []
    for p, dev in enumerate(devices):
        if p == 0:
            placed.append(
                [
                    t if t.device == dev else _kmv_place(h, dev)
                    for t, h in zip(tensors, host, strict=True)
                ]
            )
        else:
            placed.append([_kmv_place(h, dev) for h in host])
    k = len(devices)
    starts = list(range(0, n, rows))
    outs: list = [None] * len(starts)

    def launch(b):
        ops = placed[b % k]
        xs, ys, vs, ps = (ops[0], None, *ops[1:]) if self_op else ops
        outs[b] = _kmv_block(xs, ys, vs, ps, starts[b], rows, n, kind, group, False)

    if _kmv_threads() and len(starts) > k:
        global _kmv_pool
        # The first block of every core on this thread: traces the one block
        # shape into the cached handle before any worker launches.
        for b in range(k):
            launch(b)
        if _kmv_pool is None or _kmv_pool._max_workers < k:
            from concurrent.futures import ThreadPoolExecutor

            _kmv_pool = ThreadPoolExecutor(max_workers=k, thread_name_prefix="trnsparse-kmv")

        def run_core(p):
            for b in range(k + p, len(starts), k):
                launch(b)

        for f in [_kmv_pool.submit(run_core, p) for p in range(k)]:
            f.result()
    else:
        for b in range(len(starts)):
            launch(b)

    out = torch.empty(n, R, dtype=torch.float32)
    for b, i0 in enumerate(starts):
        nb = min(rows, n - i0)
        out[i0 : i0 + nb] = outs[b][:nb].cpu()
    return out.to(orig)


def nki_kernel_matmul(
    x: torch.Tensor,
    y: torch.Tensor | None,
    v: torch.Tensor,
    params: torch.Tensor,
    kind: int,
    group: int,
) -> torch.Tensor:
    """The fused kernel-matrix product (see `kernels._kmv_body`), row-blocked.

    x (n, d), y (m, d) or None (Y = X, noise added), v (m, R) and params
    (128, 4) are fp32 and contiguous, prepared by `trnsparse.kernel_matmul`.
    The result is fp32 on v's device. Raises on any kernel-path failure;
    the caller decides whether to fall back.

    A product of more than `_kmv_block_work()` kernel entries (n * m)
    runs as several launches over row blocks of x (`kernels.kmv_block_rows`),
    so no launch reaches the runtime's 30 s execution timeout (#100). Below
    the budget it is one launch, exactly as before. Rows of the product are
    independent and each row runs the same instructions in a block as in one
    launch, so the blocked result is bit-identical. The last, ragged block
    is zero-padded to the block shape (padding rows are computed and
    dropped): every launch has one shape, one compile and one cached handle.

    With several visible logical cores the blocks are spread over them
    (`TRNSPARSE_KMV_CORES`, #102; see the section comment above): the
    block count rounds up to a multiple of the core count, each core gets
    its own copy of Y, V and params, and the result is gathered on the
    host and returned on the operands' device, bit-identical to one core.
    """
    n, d = x.shape
    m = n if y is None else y.shape[0]
    R = v.shape[1]
    sim = _use_simulator()
    tensors = (x, v, params) if y is None else (x, y, v, params)
    devices = [] if sim else _kmv_core_devices(n, m, tensors)
    if len(devices) > 1:
        rows = _kernels.kmv_block_rows(n, m, _kmv_block_work(), parts=len(devices))
        if rows < n:
            k = min(len(devices), -(-n // rows))
            return _kmv_multicore(tensors, y is None, n, R, rows, devices[:k], kind, group)
    rows = _kernels.kmv_block_rows(n, m, _kmv_block_work())
    # Placed once: every block launch gets the same device Y, V and params,
    # and its X (and noise) rows as views of the placed X (and V).
    if sim:
        placed, orig = list(tensors), v.device
    else:
        placed, orig = _to_device(*tensors)
    if y is None:
        xs, vs, ps = placed
        ys = None
    else:
        xs, ys, vs, ps = placed
    if rows >= n:
        return _kmv_launch(xs, ys, vs, None, ps, kind, group, sim).to(orig)

    out = torch.empty(n, R, dtype=torch.float32, device=vs.device)
    for i0 in range(0, n, rows):
        nb = min(rows, n - i0)
        ob = _kmv_block(xs, ys, vs, ps, i0, rows, n, kind, group, sim)
        out[i0 : i0 + nb] = ob[:nb].to(out.device)
    return out.to(orig)
