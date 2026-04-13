"""NKI dispatch for sparse operations.

v0.2.0 — `nki_spmm` is the first NKI-dispatched op. Forward path:
materialize the CSR into a padded dense view, run the NKI GEMM kernel
(`kernels._spmm_dense_kernel`) on the XLA device, slice back to the
original shape. Backward runs at the PyTorch level — see
`_SpMMFunction.backward` for the analytic adjoints.

Autograd wrapping is a hard requirement across the trnsci suite (see
trnsci/trnsci#3). This module's `_SpMMFunction` is the first concrete
example in the suite; sibling projects copying this pattern is expected.
"""

from __future__ import annotations

import os

import torch

from .kernels import HAS_NKI, _TILE_K, _TILE_M, _TILE_N

if HAS_NKI:
    from .kernels import _spmm_dense_kernel  # noqa: F401 — NKI-only symbol

# When set, kernel-path failures re-raise instead of falling back to
# PyTorch. Used by the hardware validation suite.
_REQUIRE_NKI = os.environ.get("TRNSPARSE_REQUIRE_NKI", "").lower() in (
    "1", "true", "yes",
)

_backend = "auto"


def set_backend(backend: str) -> None:
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires neuronxcc")
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


def _csr_to_dense_padded(A) -> torch.Tensor:
    """Materialize a CSRMatrix into a dense (M, K) tensor.

    v0.2.0 uses the dense view directly as the A operand of the GEMM
    kernel. Row-bucketing (Phase 3, #15) will replace this with a
    per-bucket dense tile keyed on nnz quantile, which is where the
    actual sparse speedup comes from.
    """
    t = torch.sparse_csr_tensor(
        A.row_ptrs, A.col_indices, A.values, size=A.shape
    )
    return t.to_dense()


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
            (a, b), orig_device = _to_xla(A_p.contiguous(), B_p.contiguous())
        else:
            (a, b), orig_device = _to_xla(A_dense.contiguous(), B.contiguous())
        c = _spmm_dense_kernel(a, b)
        result = c.to(orig_device)
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
    """
    A_dense = _csr_to_dense_padded(A)
    return _SpMMFunction.apply(A_dense, B)
