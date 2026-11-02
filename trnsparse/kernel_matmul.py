"""
Fused kernel-matrix products K(X, Y) @ V without forming K (trnsparse#93).

The operator of matrix-free Gaussian-process regression and kernel ridge
regression (trnsci/trnsci#109): each application costs O(n m (d + R)) work
on O((n + m)(d + R)) data, and K (n x m) is never written anywhere.

Kernels (stationary, of the distance r = |x - y| / lengthscale):

    rbf       s * exp(-r^2 / 2)
    matern12  s * exp(-r)
    matern32  s * (1 + sqrt(3) r) exp(-sqrt(3) r)
    matern52  s * (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)

with s the outputscale. A tensor lengthscale of shape (d,) gives per-dimension
(ARD) lengthscales by rescaling X and Y on the host.

On the NKI backend one kernel launch computes the whole product: squared
distances as one TensorEngine matmul per 128 x 512 tile on augmented
coordinates, the kernel function on ScalarE reading PSUM, and the product
with V on the TensorEngine, accumulated in PSUM (see
`trnsparse.nki.kernels._kmv_body`). Lengthscale, outputscale and noise are
kernel arguments (a tensor), not trace constants, so changing them never
recompiles (aws-neuron-sdk#1454).

Precision. Inputs are centred on the host (the kernel is translation
invariant, so this is exact) and the distance uses the expansion
|x|^2 + |y|^2 - 2 x.y in fp32. Its absolute error is about
u * (|x|^2 + |y|^2) with u = 2^-24, so the relative error of a distance is
about u * (|x|^2 + |y|^2) / |x - y|^2: small for well-separated points and
large only for near-duplicates, where the kernel value is ~s anyway (the
error in k is then about u * (|x|^2 + |y|^2) / (2 l^2), relative). Centring
keeps |x|^2 small. float64 inputs follow the 2026-11-02 suite policy:
backend "auto" computes them with torch in float64, explicit "nki" casts
to float32 with a warning, and TRNSPARSE_REQUIRE_NKI raises.
"""

from __future__ import annotations

import math
import os
import warnings

import torch

from .nki import dispatch as _dispatch
from .nki.kernels import _KMV_GROUP, _KMV_KINDS, _KMV_MAX_D

__all__ = [
    "KERNELS",
    "KernelOperator",
    "kernel_matmul",
    "kernel_matmul_reference",
]

KERNELS = _KMV_KINDS
_MATERN_A = {"matern12": 1.0, "matern32": 3.0, "matern52": 5.0}

# Elements of one row block of K on the PyTorch path (64 MiB in fp32).
_TORCH_BLOCK_ELEMS = 1 << 24


def _kind(kernel: str) -> int:
    if kernel not in _KMV_KINDS:
        raise ValueError(f"kernel must be one of {', '.join(_KMV_KINDS)}; got {kernel!r}")
    return _KMV_KINDS.index(kernel)


def _scale_coeff(kernel: str) -> float:
    """a with k = s * exp(a D) (rbf) or t = sqrt(a D) (Matern), D the squared distance in lengthscale units."""
    return -0.5 if kernel == "rbf" else _MATERN_A[kernel]


def _kernel_of_sqdist(kind: int, D: torch.Tensor, a: float, s: float) -> torch.Tensor:
    """k(D) for a squared-distance block D (already >= 0), in D's dtype."""
    if kind == 0:
        return torch.exp(D * a).mul_(s)
    t = torch.sqrt(D * a)
    e = torch.exp(-t).mul_(s)
    if kind == 1:
        return e
    if kind == 2:
        return t.add_(1.0).mul_(e)
    return (t * (t / 3.0 + 1.0)).add_(1.0).mul_(e)


def _lengthscale(lengthscale, d: int, dtype, device) -> tuple[torch.Tensor | None, float]:
    """(per-dimension scale or None, scalar lengthscale)."""
    if isinstance(lengthscale, torch.Tensor) and lengthscale.numel() > 1:
        ls = lengthscale.detach().to(device=device, dtype=dtype).reshape(-1)
        if ls.numel() != d:
            raise ValueError(f"lengthscale has {ls.numel()} entries for d = {d}")
        if not bool((ls > 0).all()):
            raise ValueError("lengthscale must be positive")
        return ls, 1.0
    ell = float(lengthscale)
    if not ell > 0:
        raise ValueError(f"lengthscale must be positive, got {ell}")
    return None, ell


def _check(X, Y, V, noise):
    if X.dim() != 2:
        raise ValueError(f"X must be (n, d), got shape {tuple(X.shape)}")
    if Y is not None and (Y.dim() != 2 or Y.shape[1] != X.shape[1]):
        raise ValueError(f"Y must be (m, {X.shape[1]}), got shape {tuple(Y.shape)}")
    m = X.shape[0] if Y is None else Y.shape[0]
    if V.dim() != 2 or V.shape[0] != m:
        raise ValueError(f"V must be ({m}, R) or ({m},), got shape {tuple(V.shape)}")
    if noise and Y is not None:
        raise ValueError("noise applies to the square operator only (Y=None)")


def kernel_matmul_reference(
    X: torch.Tensor,
    Y: torch.Tensor | None,
    V: torch.Tensor,
    kernel: str = "rbf",
    lengthscale=1.0,
    outputscale: float = 1.0,
    noise: float = 0.0,
    *,
    block_rows: int = 2048,
) -> torch.Tensor:
    """Dense float64 reference: `torch.cdist` -> kernel -> matmul, on the CPU.

    Blocked over rows of X so memory stays O(block_rows * m). Y=None means
    Y = X and adds noise * V. Returns float64 (n, R) (or (n,) for 1-D V).
    """
    vec = V.dim() == 1
    X = X.detach().to("cpu", torch.float64)
    Y = X if Y is None else Y.detach().to("cpu", torch.float64)
    V64 = V.detach().to("cpu", torch.float64)
    V64 = V64[:, None] if vec else V64
    ls, ell = _lengthscale(lengthscale, X.shape[1], torch.float64, "cpu")
    if ls is not None:
        X, Y = X / ls, Y / ls
    kind = _kind(kernel)
    a = _scale_coeff(kernel) / ell**2
    out = torch.empty(X.shape[0], V64.shape[1], dtype=torch.float64)
    for i0 in range(0, X.shape[0], block_rows):
        D = torch.cdist(X[i0 : i0 + block_rows], Y).square_()
        out[i0 : i0 + block_rows] = _kernel_of_sqdist(kind, D, a, float(outputscale)) @ V64
    if noise:
        out += float(noise) * V64
    return out[:, 0] if vec else out


def _torch_kernel_matmul(Xc, Yc, V, kind, a, s, noise):
    """PyTorch path: the same augmented-GEMM algorithm in V's dtype, blocked over rows.

    Runs on any device, so on `neuron` it is the aten baseline (tiled
    distance GEMM + exp + matmul) for the device tables of #93.
    """
    self_op = Yc is None
    Y = Xc if self_op else Yc
    xn = (Xc * Xc).sum(1)
    yn = xn if self_op else (Y * Y).sum(1)
    m = Y.shape[0]
    block = max(1, _TORCH_BLOCK_ELEMS // max(m, 1))
    out = torch.empty(Xc.shape[0], V.shape[1], dtype=V.dtype, device=V.device)
    for i0 in range(0, Xc.shape[0], block):
        xb = Xc[i0 : i0 + block]
        D = torch.addmm(yn[None, :], xb, Y.T, alpha=-2.0).add_(xn[i0 : i0 + block, None])
        D.clamp_(min=0.0)
        out[i0 : i0 + block] = _kernel_of_sqdist(kind, D, a, s) @ V
    if self_op and noise:
        out += noise * V
    return out


def _params(a: float, s: float, noise: float, device) -> torch.Tensor:
    """(128, 4) fp32 [a, log s, noise, 0], replicated per partition (kernel argument)."""
    row = torch.tensor([a, math.log(s), noise, 0.0], dtype=torch.float32)
    return row.expand(128, 4).contiguous().to(device)


def _kmv_group() -> int:
    return max(1, int(os.environ.get("TRNSPARSE_KMV_GROUP", str(_KMV_GROUP))))


def _route(dtype: torch.dtype, d: int) -> str:
    """'nki' or 'torch' for one product, applying the float64 policy."""
    if not _dispatch._use_nki() or d > _KMV_MAX_D:
        if _dispatch._REQUIRE_NKI and d > _KMV_MAX_D:
            raise NotImplementedError(
                f"trnsparse kernel_matmul: the NKI kernel takes d <= {_KMV_MAX_D}, got {d}"
            )
        return "torch"
    if dtype == torch.float64:
        if _dispatch._REQUIRE_NKI:
            raise TypeError(
                "trnsparse kernel_matmul: float64 under TRNSPARSE_REQUIRE_NKI; the NKI "
                "kernel computes in float32. Pass float32, or unset TRNSPARSE_REQUIRE_NKI."
            )
        if _dispatch.get_backend() == "auto":
            return "torch"
        warnings.warn(
            "trnsparse kernel_matmul: float64 inputs on the nki backend are computed in "
            "float32 (the NeuronCore has no float64 datapath)",
            stacklevel=3,
        )
    elif dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"trnsparse kernel_matmul: unsupported dtype {dtype}")
    return "nki"


class KernelOperator:
    """The matrix-free operator V -> (K(X, X) + noise I) V, one fused launch per call.

    Prepares X once (centred, ARD-rescaled, fp32 on the target device) and
    keeps the per-call constants in a small device tensor, so repeated
    applications (CG iterations) move nothing but V. `cross(Y, W)` gives
    K(Y, X) W, the posterior mean at test points Y for W = alpha.

    Args:
        X: (n, d) training inputs.
        kernel: one of `KERNELS`.
        lengthscale: float, or a (d,) tensor of per-dimension lengthscales.
        outputscale: s > 0, the kernel's value at distance 0.
        noise: added to the diagonal (sigma^2 of the Gaussian likelihood).
        device: where the operator runs (default: X's device).
        dtype: working dtype (default: float32 for the NKI path, X's dtype
            on the PyTorch path when it is float64).
    """

    def __init__(
        self,
        X: torch.Tensor,
        kernel: str = "rbf",
        lengthscale=1.0,
        outputscale: float = 1.0,
        noise: float = 0.0,
        *,
        device=None,
        dtype: torch.dtype | None = None,
    ):
        if X.dim() != 2:
            raise ValueError(f"X must be (n, d), got shape {tuple(X.shape)}")
        if not outputscale > 0:
            raise ValueError(f"outputscale must be positive, got {outputscale}")
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        self.kernel = kernel
        self.kind = _kind(kernel)
        self.outputscale = float(outputscale)
        self.noise = float(noise)
        self.device = torch.device(device) if device is not None else X.device
        n, d = X.shape
        self.route = _route(dtype or X.dtype, d)
        if self.route == "nki":
            dtype = torch.float32  # the kernel computes in fp32
        elif dtype is None:
            dtype = X.dtype
        if dtype not in (torch.float32, torch.float64):
            dtype = torch.float32
        self.dtype = dtype
        X64 = X.detach().to("cpu", torch.float64)
        ls, ell = _lengthscale(lengthscale, d, torch.float64, "cpu")
        self._ard = ls
        if ls is not None:
            X64 = X64 / ls
        self.lengthscale = lengthscale
        self._ell = ell
        self.center = X64.mean(0)
        self._a = _scale_coeff(kernel) / ell**2
        self.X = (X64 - self.center).to(device=self.device, dtype=dtype).contiguous()
        self.params = _params(self._a, self.outputscale, self.noise, self.device)
        self._Xh = None
        self.shape = (n, n)

    def _prep_points(self, Y: torch.Tensor) -> torch.Tensor:
        Y64 = Y.detach().to("cpu", torch.float64)
        if self._ard is not None:
            Y64 = Y64 / self._ard
        return (Y64 - self.center).to(device=self.device, dtype=self.dtype).contiguous()

    def _apply(self, Xa: torch.Tensor, Yb: torch.Tensor | None, V: torch.Tensor, noise: float):
        vec = V.dim() == 1
        out_dtype = V.dtype
        Vk = (V[:, None] if vec else V).to(device=self.device, dtype=self.dtype).contiguous()
        out = None
        if self.route == "nki":
            try:
                # The cross kernel (Yb given) never reads the noise entry.
                out = _dispatch.nki_kernel_matmul(Xa, Yb, Vk, self.params, self.kind, _kmv_group())
            except Exception as e:
                if _dispatch._REQUIRE_NKI:
                    raise
                warnings.warn(
                    f"trnsparse kernel_matmul: NKI path failed ({type(e).__name__}: {e}); "
                    "using the PyTorch path",
                    stacklevel=3,
                )
                self.route = "torch"
        if out is None:
            out = _torch_kernel_matmul(Xa, Yb, Vk, self.kind, self._a, self.outputscale, noise)
        out = out.to(dtype=out_dtype) if out_dtype.is_floating_point else out
        return out[:, 0] if vec else out

    def __call__(self, V: torch.Tensor) -> torch.Tensor:
        """(K(X, X) + noise I) @ V for V of shape (n,) or (n, R)."""
        if V.shape[0] != self.shape[0]:
            raise ValueError(f"V must have {self.shape[0]} rows, got {V.shape[0]}")
        return self._apply(self.X, None, V, self.noise)

    matmul = __call__

    def cross(self, Y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """K(Y, X) @ W (no noise) for new points Y (q, d) and W (n,) or (n, R)."""
        if Y.dim() != 2 or Y.shape[1] != self.X.shape[1]:
            raise ValueError(f"Y must be (q, {self.X.shape[1]}), got shape {tuple(Y.shape)}")
        if W.shape[0] != self.shape[0]:
            raise ValueError(f"W must have {self.shape[0]} rows, got {W.shape[0]}")
        return self._apply(self._prep_points(Y), self.X, W, 0.0)

    def diag(self) -> torch.Tensor:
        """Diagonal of K(X, X) without the noise: the outputscale everywhere (float64, host)."""
        return torch.full((self.shape[0],), self.outputscale, dtype=torch.float64)

    def rows(self, idx) -> torch.Tensor:
        """Rows K(X[idx], X) without the noise, float64 on the host (pivoted Cholesky)."""
        idx = torch.as_tensor(idx, dtype=torch.long).reshape(-1).cpu()
        if self._Xh is None:  # host float64 copy, made once (one row per pivot step)
            self._Xh = self.X.detach().to("cpu", torch.float64)
        Xh = self._Xh
        D = torch.cdist(Xh[idx], Xh).square_()
        return _kernel_of_sqdist(self.kind, D, self._a, self.outputscale)


def kernel_matmul(
    X: torch.Tensor,
    Y: torch.Tensor | None,
    V: torch.Tensor,
    kernel: str = "rbf",
    lengthscale=1.0,
    outputscale: float = 1.0,
    noise: float = 0.0,
    tile_mask=None,
) -> torch.Tensor:
    """K(X, Y) @ V without forming K; Y=None gives (K(X, X) + noise I) @ V.

    Args:
        X: (n, d) points.
        Y: (m, d) points, or None for Y = X.
        V: (m, R) or (m,).
        kernel: "rbf", "matern12", "matern32" or "matern52".
        lengthscale: float, or a (d,) tensor of per-dimension lengthscales.
        outputscale: s, the kernel's value at distance 0.
        noise: diagonal term, Y=None only.
        tile_mask: reserved for static BSR-128 tile skipping (#93, later);
            must be None.

    Returns:
        (n, R) (or (n,)) in V's dtype, on V's device. For repeated products
        with the same X (iterative solvers) build a `KernelOperator` once.
    """
    if tile_mask is not None:
        raise NotImplementedError("kernel_matmul: tile_mask is not implemented yet (#93)")
    _check(X, Y, V[:, None] if V.dim() == 1 else V, noise)
    dtype = torch.promote_types(X.dtype, V.dtype)
    op = KernelOperator(
        X,
        kernel,
        lengthscale,
        outputscale,
        noise,
        device=V.device,
        dtype=None if dtype != torch.float64 else torch.float64,
    )
    if Y is None:
        return op(V)
    # K(X, Y) V: X are the output rows, Y the summed points. Centre both by X's mean.
    return op._apply(op.X, op._prep_points(Y), V, 0.0)
