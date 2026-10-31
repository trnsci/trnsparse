"""Iterative linear solvers for BSR sparse matrices.

v0.3.2 — Phase 1 plumbing (#22). Python-level CG and power iteration
on top of `bsr_spmm` as the matvec. A gets reloaded from HBM on every
iteration; v0.4.0 (follow-up) delivers the SBUF-resident fused NKI
kernel that keeps A on-chip across iterations — the architectural win.

Algorithm body for CG mirrors `trnsolver.iterative.cg` at commit-time,
kept local to avoid a trnsolver runtime dep.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .formats import BSRMatrix
from .ops import bsr_spmm


def bsr_diagonal(A: BSRMatrix) -> torch.Tensor:
    """Extract the main diagonal of a BSR matrix as a dense vector.

    The diagonal lives inside the diagonal blocks (block row i at block
    column i). Blocks that aren't diagonal contribute nothing. Returns
    a zero vector if no diagonal block is stored (a fully zero-diagonal
    matrix would be degenerate as a preconditioner anyway).
    """
    m, _ = A.shape
    b = A.block_size
    M_tiles = (m + b - 1) // b
    # Find each block row's diagonal block on host copies of the structure,
    # then gather all of them in one indexing op on the blocks' device (no
    # per-element reads or writes of device tensors).
    row_ptrs = A.block_row_ptrs.cpu()
    col_indices = A.block_col_indices.cpu()
    rows, ks = [], []
    for i in range(M_tiles):
        start = row_ptrs[i].item()
        end = row_ptrs[i + 1].item()
        for k in range(start, end):
            if col_indices[k].item() == i:
                rows.append(i)
                ks.append(k)
                break
    dev = A.blocks.device
    out = torch.zeros(M_tiles, b, dtype=A.dtype, device=dev)
    if rows:
        diag = torch.diagonal(A.blocks[torch.tensor(ks, device=dev)], dim1=-2, dim2=-1)
        out[torch.tensor(rows, device=dev)] = diag
    return out.reshape(-1)[:m]


def _bsr_matvec(A: BSRMatrix) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap `bsr_spmm` so CG / power iteration can pass it a vector."""

    def matvec(x: torch.Tensor) -> torch.Tensor:
        return bsr_spmm(A, x.unsqueeze(1)).squeeze(1)

    return matvec


def cg_bsr(
    A: BSRMatrix,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    M: Callable[[torch.Tensor], torch.Tensor] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Conjugate Gradient solver for `A @ x = b` with A SPD and BSR-stored.

    Thin specialization of the standard CG loop: the matvec goes through
    `bsr_spmm`, so on the NKI backend each iteration dispatches one
    kernel call. That amortizes poorly across many iterations — the
    v0.4.0 fused kernel fuses the loop and keeps A SBUF-resident.

    Args:
        A: SPD BSR matrix (N, N).
        b: Right-hand side, shape (N,).
        x0: Initial guess (default: zeros).
        tol: Relative-residual tolerance on `||r|| / ||b||`.
        max_iter: Iteration cap.
        M: Preconditioner — callable `M(r) → M^{-1} r` or a dense
            inverse tensor. Build a Jacobi preconditioner from
            `bsr_diagonal(A)` for the common diagonally-dominant case.

    Returns:
        `(x, iters, rel_residual)` — solution, iteration count, final
        relative residual.
    """
    n = b.shape[0]
    matvec = _bsr_matvec(A)

    if M is None:
        precond: Callable[[torch.Tensor], torch.Tensor] | None = None
    elif callable(M):
        precond = M
    else:
        M_tensor = M

        def precond(r: torch.Tensor) -> torch.Tensor:
            return torch.mv(M_tensor, r)

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
    r = b - matvec(x)
    b_norm = torch.linalg.norm(b).item()
    if b_norm < 1e-15:
        return x, 0, 0.0

    z = precond(r) if precond is not None else r.clone()
    p = z.clone()
    rz = torch.dot(r, z).item()

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = torch.dot(p, Ap).item()
        if abs(pAp) < 1e-30:
            break

        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.linalg.norm(r).item()
        rel = r_norm / b_norm
        if rel < tol:
            return x, k + 1, rel

        z = precond(r) if precond is not None else r
        rz_new = torch.dot(r, z).item()
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, max_iter, r_norm / b_norm


def jacobi_preconditioner_bsr(
    A: BSRMatrix,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a Jacobi (diagonal) preconditioner for a BSR matrix.

    Returns a callable `M(r) = r / diag(A)`. Cheap + effective when A
    is diagonally dominant.
    """
    d = bsr_diagonal(A)
    if torch.any(d.abs() < 1e-15):
        raise ValueError("Jacobi preconditioner: diagonal has near-zero entries")
    inv_d = 1.0 / d

    def precond(r: torch.Tensor) -> torch.Tensor:
        # No-op when r is on A.blocks' device (the usual case).
        return r * inv_d.to(r.device)

    return precond


def chebyshev_coeffs(
    lam_min: float,
    lam_max: float,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chebyshev semi-iteration step coefficients for K fixed iterations.

    Returns (alpha, beta) each of shape (K,) such that the iteration

        x_{k+1} = x_k + alpha[k] * r_k + beta[k] * (x_k - x_{k-1})

    (with r_k = b - A @ x_k) minimises the Chebyshev residual polynomial
    over [lam_min, lam_max] after K steps.

    Algorithm: Templates for the Solution of Linear Systems (Barrett et
    al. 1994), Algorithm 2.1 converted from direction-vector form to the
    momentum / heavy-ball form. c = half-range, d = centre.

    Args:
        lam_min: Lower eigenvalue bound (must be > 0 for SPD A).
        lam_max: Upper eigenvalue bound.
        K: Number of iterations for which coefficients are needed.

    Returns:
        (alpha, beta): float32 tensors of length K.
    """
    c = (lam_max - lam_min) / 2.0  # half-range
    d = (lam_max + lam_min) / 2.0  # centre

    alpha = torch.zeros(K, dtype=torch.float64)
    beta = torch.zeros(K, dtype=torch.float64)

    # k=0: pure Richardson at the optimal single-step rate.
    alpha_prev = 1.0 / d
    alpha[0] = alpha_prev
    beta[0] = 0.0

    for k in range(1, K):
        # Templates step: beta_t = (c * alpha_prev / 2)^2
        #                  alpha_new = 1 / (d - beta_t / alpha_prev)
        beta_t = (c * alpha_prev / 2.0) ** 2
        alpha_new = 1.0 / (d - beta_t / alpha_prev)
        # Convert to momentum form:
        #   alpha_k^(momentum) = alpha_new
        #   beta_k^(momentum)  = alpha_new * beta_t / alpha_prev
        alpha[k] = alpha_new
        beta[k] = alpha_new * beta_t / alpha_prev
        alpha_prev = alpha_new

    return alpha.float(), beta.float()


def chebyshev_bsr(
    A: BSRMatrix,
    b: torch.Tensor,
    lam_min: float,
    lam_max: float,
    K: int = 50,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Chebyshev semi-iteration for `A @ x = b` with A SPD and BSR-stored.

    Runs exactly K iterations with pre-computed coefficients derived from
    eigenvalue bounds [lam_min, lam_max]. No inner products are computed
    during the iteration — all coefficients are determined before the
    first step. This is the key structural difference from `cg_bsr`:
    the loop body is purely matvec + elementwise ops, which maps directly
    to the fixed-iteration NKI kernel pattern.

    Obtain lam_max from `power_iteration_bsr`. A conservative lam_min
    estimate (e.g. lam_max / condition_number_estimate) is sufficient;
    over-estimating the spectral range only slows convergence slightly,
    not correctness.

    Args:
        A: SPD BSR matrix (N, N).
        b: Right-hand side, shape (N,).
        lam_min: Lower eigenvalue bound (> 0).
        lam_max: Upper eigenvalue bound.
        K: Fixed iteration count.
        x0: Initial guess (default: zeros).

    Returns:
        `(x, K, rel_residual)` — solution, iteration count (always K),
        final relative residual ||b - A @ x|| / ||b||.
    """
    n = b.shape[0]
    matvec = _bsr_matvec(A)

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
    x_prev = x.clone()
    b_norm = torch.linalg.norm(b).item()

    alpha, beta = chebyshev_coeffs(lam_min, lam_max, K)

    for k in range(K):
        r = b - matvec(x)
        x_new = x + alpha[k].item() * r
        if k > 0:
            x_new = x_new + beta[k].item() * (x - x_prev)
        x_prev = x
        x = x_new

    r_final = b - matvec(x)
    rel = torch.linalg.norm(r_final).item() / max(b_norm, 1e-15)
    return x, K, rel


def richardson_bsr(
    A: BSRMatrix,
    b: torch.Tensor,
    omega: float,
    K: int = 100,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Fixed-K Richardson iteration: x_{k+1} = x_k + omega * (b - A @ x_k).

    The simplest fixed-point iteration with no adaptive coefficients.
    Converges when 0 < omega < 2 / lam_max. The optimal static step is
    omega = 2 / (lam_min + lam_max).

    Like `chebyshev_bsr`, the loop runs for exactly K iterations with no
    adaptive stopping — the right structure for eventual NKI fusion.

    Args:
        A: BSR matrix (N, N).
        b: Right-hand side, shape (N,).
        omega: Relaxation parameter. Use 2 / (lam_min + lam_max) for
            optimal convergence on SPD systems.
        K: Fixed iteration count.
        x0: Initial guess (default: zeros).

    Returns:
        `(x, K, rel_residual)`.
    """
    n = b.shape[0]
    matvec = _bsr_matvec(A)

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
    b_norm = torch.linalg.norm(b).item()

    for _ in range(K):
        r = b - matvec(x)
        x = x + omega * r

    r_final = b - matvec(x)
    rel = torch.linalg.norm(r_final).item() / max(b_norm, 1e-15)
    return x, K, rel


def power_iteration_bsr(
    A: BSRMatrix,
    v0: torch.Tensor | None = None,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> tuple[float, torch.Tensor, int]:
    """Find the dominant eigenpair (λ, v) of a BSR matrix via power iteration.

    Converges to the eigenvector with largest `|λ|`. Tolerance is on
    the Rayleigh-quotient change between iterations.

    Args:
        A: Square BSR matrix (N, N). Doesn't need to be SPD — power
            iteration works for any matrix with a unique dominant
            eigenvalue, but convergence slows as the spectral gap
            shrinks.
        v0: Initial vector (default: random unit vector).
        max_iter: Iteration cap.
        tol: Convergence on `|λ_k - λ_{k-1}|`.

    Returns:
        `(eigenvalue, eigenvector, iters)`.
    """
    n = A.shape[0]
    matvec = _bsr_matvec(A)

    v = v0.clone() if v0 is not None else torch.randn(n, dtype=A.dtype)
    v = v / torch.linalg.norm(v)

    lam_prev = 0.0
    for k in range(max_iter):
        Av = matvec(v)
        lam = torch.dot(v, Av).item()
        norm = torch.linalg.norm(Av).item()
        if norm < 1e-30:
            return 0.0, v, k + 1
        v = Av / norm

        if abs(lam - lam_prev) < tol:
            return lam, v, k + 1
        lam_prev = lam

    return lam, v, max_iter


# ── Fused, SBUF-resident CG (#22) ───────────────────────────────────────────

_P = 128
# The fused kernel unrolls `check_every` CG steps (one matmul per stored
# block each) per device-side convergence check; cap the unrolled matmul
# count so trace / compile time stays bounded.
_FUSED_MAX_UNROLL = 8192


class FusedCGOperator:
    """An SPD BSR matrix prepared for `cg_bsr_fused`.

    Holds A's blocks pre-transposed side by side (the kernel's stationary
    layout), the optional block-Jacobi inverses, and the registered static
    structure key. Build it once with `prepare_cg_bsr` and reuse it across
    right-hand sides: the host-side preparation (block transposes, the
    block-Jacobi Cholesky inverses, the host-to-device copy) is then paid
    once.

    Attributes:
        n: matrix order. n_tiles: block rows (n padded to a multiple of 128).
        n_blocks: stored blocks. resident_blocks: blocks the kernel keeps in
        SBUF (the rest stream from HBM on every iteration).
        preconditioner: None or "block_jacobi".
    """

    def __init__(self, A: BSRMatrix, preconditioner: str | None, device):
        if A.block_size != _P:
            raise ValueError(f"cg_bsr_fused needs block_size=128, got {A.block_size}")
        m, n = A.shape
        if m != n:
            raise ValueError(f"cg_bsr_fused needs a square matrix, got {A.shape}")
        if preconditioner not in (None, "block_jacobi"):
            raise ValueError(
                f"preconditioner must be None or 'block_jacobi', got {preconditioner!r}"
            )
        T = (n + _P - 1) // _P
        if T > 512:
            raise ValueError("cg_bsr_fused supports n <= 65536 (one PSUM bank per vector)")
        row_ptrs = A.block_row_ptrs.cpu().tolist()
        cols = A.block_col_indices.cpu().tolist()
        blocks = A.blocks.detach().to("cpu", torch.float64).clone()
        diag = []
        for t in range(T):
            ks = [k for k in range(row_ptrs[t], row_ptrs[t + 1]) if cols[k] == t]
            diag.append(ks[0] if ks else None)
        pad = T * _P - n
        if pad:
            # Identity on the padded diagonal keeps the padded system SPD;
            # b is zero there, so x stays zero there.
            if diag[T - 1] is None:
                raise ValueError("cg_bsr_fused: the last block row stores no diagonal block")
            i = torch.arange(_P - pad, _P)
            blocks[diag[T - 1], i, i] = 1.0
        self.n = n
        self.n_tiles = T
        self.n_blocks = len(cols)
        self.row_ptrs = row_ptrs
        self.cols = cols
        self.preconditioner = preconditioner
        self.dtype = A.dtype
        self.device = torch.device(device) if device is not None else A.blocks.device
        # Host fp64 copies drive the CPU path.
        self._blocks64 = blocks
        self._rows = torch.repeat_interleave(
            torch.arange(T), torch.tensor([row_ptrs[t + 1] - row_ptrs[t] for t in range(T)])
        )
        self._cols = torch.tensor(cols, dtype=torch.long)
        self._minv64 = None
        if preconditioner == "block_jacobi":
            if any(d is None for d in diag):
                raise ValueError("block-Jacobi needs every diagonal block stored")
            D = blocks[torch.tensor(diag)]
            L, info = torch.linalg.cholesky_ex(D)
            if bool((info != 0).any()):
                raise ValueError("block-Jacobi: a diagonal block is not SPD")
            self._minv64 = torch.cholesky_inverse(L)
        from .nki.kernels import _cg_sbuf_budget_bytes, _core_split_enabled

        # Read once here; the kernel gets them as arguments (compile-cache key).
        self.sbuf_kib = _cg_sbuf_budget_bytes() // 1024
        self.core_split = _core_split_enabled()
        self.resident_blocks = cg_resident_blocks_for(self)
        self._dev = None  # device operands, built on first device solve
        self._host_cache: dict = {}

    def _device_operands(self):
        if self._dev is None:
            from .nki.kernels import register_structure

            def side_by_side(blk):  # (k, 128, 128) -> (128, k*128) of A_idx^T
                return blk.permute(2, 0, 1).reshape(_P, -1)

            bt = side_by_side(self._blocks64).to(torch.float32).contiguous()
            if self._minv64 is not None:
                mt = side_by_side(self._minv64).to(torch.float32).contiguous()
            else:
                mt = torch.zeros(_P, _P, dtype=torch.float32)
            key = register_structure(self.row_ptrs, self.cols)
            self._dev = (bt.to(self.device), mt.to(self.device), key)
        return self._dev

    # CPU path helpers on (T, 128, R) tiles
    def _cast(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        key = (name, dtype)
        if key not in self._host_cache:
            self._host_cache[key] = getattr(self, name).to(dtype)
        return self._host_cache[key]

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:  # x: (T, 128, R)
        blk = self._cast("_blocks64", x.dtype)
        return torch.zeros_like(x).index_add_(0, self._rows, torch.bmm(blk, x[self._cols]))

    def _precond(self, r: torch.Tensor) -> torch.Tensor:
        if self._minv64 is None:
            return r
        return torch.bmm(self._cast("_minv64", r.dtype), r)


def prepare_cg_bsr(A: BSRMatrix, preconditioner: str | None = None, device=None) -> FusedCGOperator:
    """Prepare an SPD BSR matrix (block_size 128) for repeated `cg_bsr_fused` solves.

    `device` is where the kernel runs (default: A.blocks' device). Pass
    `"neuron"` (or `"neuron:k"`) to run the fused kernel with host-built A.
    """
    return FusedCGOperator(A, preconditioner, device)


def _use_fused_kernel(op: FusedCGOperator, b: torch.Tensor) -> bool:
    from .nki.dispatch import _native_path, _use_nki

    if not _use_nki():
        return False
    try:
        native = _native_path()
    except Exception:
        return False
    on_neuron = op.device.type != "cpu" or b.device.type not in ("cpu",)
    return native and on_neuron


def cg_bsr_fused(
    A: BSRMatrix | FusedCGOperator,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    preconditioner: str | None = None,
    check_every: int = 8,
    return_info: bool = False,
):
    """CG for SPD BSR A in one launch, with A resident in SBUF (#22).

    The fused NKI kernel loads A's stored blocks into SBUF once and runs
    the whole solve on the NeuronCore: x, r, p live in SBUF, dot products
    reduce in PSUM, and a device-side `while` loop re-checks
    `||r|| <= tol * ||b||` every `check_every` steps without returning to
    the host. Steps past convergence inside a trip are masked, so the
    iteration count is exact. Blocks beyond the SBUF budget
    (`TRNSPARSE_CG_SBUF_KIB`, default 160 KiB per partition) stream from
    HBM on every iteration, still inside the launch.

    `b` may hold several right-hand sides as columns, `(n, R)`: they are
    solved as R independent CG iterations sharing every load of A (one
    matmul per stored block streams all R columns), each with its own
    step sizes, convergence test and iteration count.

    The kernel runs when the native TorchNeuron path is active, the NKI
    backend is selected, and the operator (or b) is on a `neuron` device.
    Otherwise the same algorithm runs in PyTorch on the host, in b's dtype
    (fp64 stays fp64). The device kernel is fp32 throughout.

    Args:
        A: SPD BSR matrix with block_size 128, or a `prepare_cg_bsr` operator.
        b: right-hand side(s), (n,) or (n, R) with R <= 512.
        x0: initial guess, b's shape (default zeros).
        tol: relative residual tolerance on the recursive residual, per column.
        max_iter: iteration cap.
        preconditioner: None or "block_jacobi" (exact inverses of the
            128x128 diagonal blocks; ignored when A is an operator).
        check_every: CG steps between convergence checks (device only).
        return_info: also return a dict (resident_blocks, trips, path).

    Returns:
        `(x, iters, rel_residual)` like `cg_bsr`, plus the info dict when
        `return_info`. rel_residual is the recursive ||r|| / ||b||. For a
        2-D b, iters and rel_residual are per-column tensors.
    """
    op = A if isinstance(A, FusedCGOperator) else prepare_cg_bsr(A, preconditioner)
    n, T = op.n, op.n_tiles
    vector = b.dim() == 1
    if b.dim() not in (1, 2) or b.shape[0] != n:
        raise ValueError(f"b must have shape ({n},) or ({n}, R), got {tuple(b.shape)}")
    B = b.reshape(n, -1)
    R = B.shape[1]
    if not 1 <= R <= 512:
        raise ValueError(f"cg_bsr_fused takes 1..512 right-hand sides, got {R}")
    X0 = None if x0 is None else x0.reshape(n, R)
    check_every = max(1, int(check_every))
    per_step = op.n_blocks + (T if op.preconditioner else 0)
    if check_every * per_step > _FUSED_MAX_UNROLL:
        check_every = max(1, _FUSED_MAX_UNROLL // per_step)
    out_dtype = b.dtype
    # Columns are solved scaled to unit norm, so one threshold (tol^2) fits all.
    norms = torch.linalg.norm(B.detach().to("cpu", torch.float64), dim=0)
    scale = torch.where(norms > 1e-300, norms, torch.ones_like(norms))
    info = {
        "resident_blocks": cg_resident_blocks_for(op, R),
        "n_blocks": op.n_blocks,
        "check_every": check_every,
        "n_rhs": R,
    }

    def finish(x, iters, rel, trips, path):
        info.update(trips=trips, path=path)
        x = x.reshape(b.shape)
        if vector:
            iters, rel = int(iters[0]), float(rel[0])
        return (x, iters, rel, info) if return_info else (x, iters, rel)

    def to_tiles(v):  # (n, R) / scale -> (T, 128, R) fp64 on the host
        v = v.detach().to("cpu", torch.float64) / scale
        out = torch.zeros(T * _P, R, dtype=torch.float64)
        out[:n] = v
        return out.reshape(T, _P, R)

    if _use_fused_kernel(op, b):
        from .nki.dispatch import _get_native_ops

        bt, mt, key = op._device_operands()
        TR = T * R
        # Pack [B | X0 | tol^2, max_iter] on the host: one copy each way.
        # Column t*R + c of a (128, T R) block holds rows t*128.. of column c.
        state = torch.zeros(_P, 2 * TR + 2, dtype=torch.float32)
        state[:, :TR] = to_tiles(B).permute(1, 0, 2).reshape(_P, TR)
        if X0 is not None:
            state[:, TR : 2 * TR] = to_tiles(X0).permute(1, 0, 2).reshape(_P, TR)
        state[:, 2 * TR] = tol * tol
        state[:, 2 * TR + 1] = float(max_iter)
        out = _get_native_ops()["cg_bsr_keyed"](
            bt,
            mt,
            state.to(bt.device),
            key,
            check_every,
            1 if op.preconditioner else 0,
            op.sbuf_kib,
            1 if op.core_split else 0,
        ).cpu()
        xt = out[:, :TR].double().reshape(_P, T, R).permute(1, 0, 2).reshape(T * _P, R)
        x = (xt[:n] * scale).to(device=b.device, dtype=out_dtype)
        iters = out[0, TR : TR + R].round().long()
        rel = out[0, TR + R : TR + 2 * R].double().clamp_min(0).sqrt()
        return finish(x, iters, rel, int(out[0, TR + 3 * R].item()), "nki")

    # Host path: the same masked algorithm in b's dtype on (T, 128, R) tiles.
    dt = b.dtype if b.dtype == torch.float64 else torch.float32
    bv = to_tiles(B).to(dt)
    xv = to_tiles(X0).to(dt) if X0 is not None else torch.zeros_like(bv)

    def dot(u, v):
        return (u * v).sum(dim=(0, 1))

    r = bv - op._matvec(xv)
    z = op._precond(r)
    p = z.clone()
    rz = dot(r, z)
    rr = dot(r, r)
    thr = tol * tol
    cnt = torch.zeros(R, dtype=torch.long)
    while True:
        act = (rr > thr) & (cnt < max_iter)
        if not bool(act.any()):
            break
        a = act.to(dt)
        q = op._matvec(p)
        alpha = a * rz / dot(p, q).clamp_min(1e-30)
        xv = xv + alpha * p
        r = r - alpha * q
        z = op._precond(r)
        rz_new = dot(r, z)
        rr = dot(r, r)
        p = z + (a * rz_new / rz.clamp_min(1e-30)) * p
        rz = rz_new
        cnt += act.long()
    x = (xv.reshape(T * _P, R)[:n] * scale.to(dt)).to(device=b.device, dtype=out_dtype)
    rel = rr.clamp_min(0).sqrt().double()
    trips = -(-int(cnt.max()) // check_every) if R else 0
    return finish(x, cnt, rel, trips, "pytorch")


def cg_resident_blocks_for(op: FusedCGOperator, n_rhs: int = 1, lnc: int | None = None) -> int:
    """Stored blocks of `op` the fused kernel keeps in SBUF (summed over the
    physical cores, which split the block rows at LNC=2)."""
    import os

    from .nki.kernels import cg_core_rows, cg_resident_blocks

    if lnc is None:
        lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    pc = op.preconditioner is not None
    total = 0
    for o0, o1 in cg_core_rows(op.row_ptrs, pc, lnc, op.core_split):
        nb = op.row_ptrs[o1] - op.row_ptrs[o0]
        w = op.n_tiles if n_rhs > 1 else o1 - o0
        total += cg_resident_blocks(nb, w, pc, n_rhs, budget_kib=op.sbuf_kib)
    return total
