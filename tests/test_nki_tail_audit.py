"""Tail-tile audit on hardware (trnsci/trnfft#100 class of bug).

Every NKI kernel reachable with `set_backend("nki")`, at awkward sizes:
dims of 1, 127, 129, 200, 1000 and 4097, odd right-hand-side widths, and
free dims that are not a multiple of 512. Each case is checked against an
fp64 reference, so a dropped partial tile (rows, contraction terms or
columns) shows up as a large error rather than passing on friendly sizes.

    NKI_ENABLE_TRACE_CACHE=0 TRNSPARSE_REQUIRE_NKI=1 \\
        pytest tests/test_nki_tail_audit.py -v -m neuron

`CASES` is also the input of the standalone probe, which reports each case
as OK, SILENT (wrong result) or LOUD (exception) instead of asserting.
"""

from __future__ import annotations

import itertools

import pytest
import torch

import trnsparse
from trnsparse.nki import HAS_NKI
from trnsparse.nki.dispatch import _get_dispatch_path, _use_simulator


def _native() -> bool:
    try:
        return HAS_NKI and not _use_simulator() and _get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

DEV = "neuron"
P = 128
REQUIRE_NKI_PATH = True  # the probe can clear it to check references on the host


def _g(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max abs error over the reference's max abs value (fp64, on the host)."""
    got = got.detach().to("cpu", torch.float64)
    ref = ref.detach().to("cpu", torch.float64)
    assert got.shape == ref.shape, f"shape {tuple(got.shape)} != {tuple(ref.shape)}"
    return float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def _block_sparse(m: int, n: int, b: int, seed: int, keep: float = 0.5) -> torch.Tensor:
    """(m, n) fp64 with about `keep` of its b x b blocks nonzero.

    The partial last block row and column are always kept, so a dropped
    tail cannot hide behind an empty block.
    """
    g = _g(seed)
    mb, nb = -(-m // b), -(-n // b)
    bm = torch.rand(mb, nb, generator=g) < keep
    bm[-1, :] = True
    bm[:, -1] = True
    mask = bm.repeat_interleave(b, 0).repeat_interleave(b, 1)[:m, :n]
    return torch.randn(m, n, generator=g, dtype=torch.float64) * mask


# ── CSR SpMM (densified, `_spmm_dense_kernel`) ────────────────────────────────

SPMM_SHAPES = [
    (1, 1, 1),
    (127, 129, 7),
    (129, 127, 513),
    (200, 200, 600),
    (1000, 300, 33),
    (4097, 129, 5),
]


def case_spmm(M, K, N, seed=0):
    g = _g(seed)
    A = torch.randn(M, K, generator=g, dtype=torch.float64)
    A = A * (torch.rand(M, K, generator=g) < 0.3)
    A[-1, -1] = 1.0  # tail row / tail contraction term always present
    B = torch.randn(K, N, generator=g, dtype=torch.float64)
    C = trnsparse.spmm(trnsparse.from_dense(A.float()), B.float().to(DEV))
    return _rel(C, A @ B), 1e-4


# ── BSR SpMM (static-structure and gather kernels) ────────────────────────────

BSR_SHAPES = [
    (1, 1, 1),
    (127, 129, 7),
    (129, 200, 513),
    (200, 1000, 600),
    (1000, 1000, 33),
    (4097, 257, 5),
]


def case_bsr(m, n, N, static=True, block_size=P, dtype=torch.float32, seed=1):
    import trnsparse.nki.dispatch as d

    A = _block_sparse(m, n, block_size, seed)
    B = torch.randn(n, N, generator=_g(seed + 100), dtype=torch.float64)
    bsr = trnsparse.BSRMatrix.from_dense(A.to(dtype), block_size=block_size)
    prev = d._STATIC_STRUCTURE
    d._STATIC_STRUCTURE = static
    try:
        C = trnsparse.bsr_spmm(bsr, B.to(dtype).to(DEV))
    finally:
        d._STATIC_STRUCTURE = prev
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    return _rel(C, A @ B), tol


def case_bsr_unsupported(m, n, N, block_size):
    """Block sizes the kernels do not take (#46): straight to torch, or a
    NotImplementedError naming block_size 128 under TRNSPARSE_REQUIRE_NKI."""
    import trnsparse.nki.dispatch as d

    if not d._REQUIRE_NKI:
        return case_bsr(m, n, N, block_size=block_size)
    with pytest.raises(NotImplementedError, match="block_size 128 only"):
        case_bsr(m, n, N, block_size=block_size)
    return 0.0, 1e-4


# ── Screened SpMM (`_screened_spmm_kernel`) ───────────────────────────────────

SCREENED_SHAPES = [(1, 1), (127, 7), (129, 513), (200, 600), (1000, 33), (4097, 5)]


def case_screened(M, N, seed=2):
    g = _g(seed)
    A = torch.randn(M, M, generator=g, dtype=torch.float64)
    diag = torch.rand(M, generator=g, dtype=torch.float64) + 0.5
    diag[: M // 3] = 1e-12  # screen out the first third, keep the tail
    B = torch.randn(M, N, generator=g, dtype=torch.float64)
    thr = 1e-6
    q = diag.abs().sqrt()
    ref = (A * ((q[:, None] * q[None, :]) > thr**0.5)) @ B
    C = trnsparse.screened_spmm(A.float(), diag.float(), B.float().to(DEV), thr)
    return _rel(C, ref), 1e-4


# ── Block-sparse attention (static and gather kernels, fwd + bwd) ─────────────

ATTN_SHAPES = [(128, 1), (384, 33), (1152, 127), (256, 128), (384, 256)]


def _attn_mask(seq: int, seed: int) -> trnsparse.BSRMatrix:
    nb = -(-seq // P)
    bm = (torch.rand(nb, nb, generator=_g(seed)) < 0.4) | torch.eye(nb, dtype=torch.bool)
    bm[-1, :] = True  # last query block row sees every key block
    dense = bm.repeat_interleave(P, 0).repeat_interleave(P, 1)[:seq, :seq].double()
    return trnsparse.BSRMatrix.from_dense(dense, block_size=P)


def _attn_ref(Q, K, V, mask, scale=None):
    dense = mask.to_dense().bool()[: Q.shape[0], : Q.shape[0]]
    s = (Q @ K.T) * (Q.shape[1] ** -0.5 if scale is None else scale)
    s = s.masked_fill(~dense, float("-inf"))
    return torch.softmax(s, dim=-1) @ V


def case_attn(seq, hd, static=True, backward=False, seed=3, scale=None):
    import trnsparse.nki.dispatch as d

    g = _g(seed)
    Q, K, V = (torch.randn(seq, hd, generator=g, dtype=torch.float64) for _ in range(3))
    mask = _attn_mask(seq, seed)
    ref = _attn_ref(Q, K, V, mask, scale)
    prev = d._STATIC_STRUCTURE
    d._STATIC_STRUCTURE = static
    try:
        if not backward:
            out = trnsparse.block_sparse_attention_tiled(
                *(t.float().to(DEV) for t in (Q, K, V)), mask, scale=scale
            )
            return _rel(out, ref), 1e-4
        dO = torch.randn(seq, hd, generator=g, dtype=torch.float64)
        Qr, Kr, Vr = (t.detach().clone().requires_grad_(True) for t in (Q, K, V))
        (_attn_ref(Qr, Kr, Vr, mask, scale) * dO).sum().backward()
        Qd, Kd, Vd = (t.float().to(DEV).requires_grad_(True) for t in (Q, K, V))
        out = trnsparse.block_sparse_attention_tiled(Qd, Kd, Vd, mask, scale=scale)
        (out * dO.float().to(DEV)).sum().backward()
        err = max(_rel(a.grad, b.grad) for a, b in ((Qd, Qr), (Kd, Kr), (Vd, Vr)))
        return err, 1e-3
    finally:
        d._STATIC_STRUCTURE = prev


def case_attn_partial(seq=200, hd=16, seed=4):
    """seq_len not a multiple of the block: must raise, never be silently wrong."""
    g = _g(seed)
    Q, K, V = (torch.randn(seq, hd, generator=g, dtype=torch.float64) for _ in range(3))
    mask = trnsparse.BSRMatrix.from_dense(torch.ones(seq, seq, dtype=torch.float64))
    out = trnsparse.block_sparse_attention_tiled(*(t.float().to(DEV) for t in (Q, K, V)), mask)
    return _rel(out, _attn_ref(Q, K, V, mask)), 1e-4


# ── Fused CG and power iteration (`_cg_bsr_keyed_kernel`, `_power_bsr_keyed`) ─

FUSED_N = [1, 127, 129, 200, 1000, 4097]


def _spd(n: int, seed: int, band: int = 1) -> torch.Tensor:
    """Block-banded, strictly diagonally dominant SPD (fp64), kappa < ~50."""
    g = _g(seed)
    T = -(-n // P)
    M = torch.randn(T * P, T * P, generator=g, dtype=torch.float64) / (P * (2 * band + 1)) ** 0.5
    bm = torch.zeros(T, T, dtype=torch.float64)
    for i in range(T):
        bm[i, max(0, i - band) : i + band + 1] = 1.0
    M = (M * torch.kron(bm, torch.ones(P, P, dtype=torch.float64)))[:n, :n]
    A = (M + M.T) / 2
    return A + (A.abs().sum(dim=1).max() + 1.0) * torch.eye(n, dtype=torch.float64)


def case_cg(n, R=1, pc=None, matvec="auto", precision="fp32", seed=5):
    A = _spd(n, seed)
    shape = (n,) if R == 1 else (n, R)
    b = torch.randn(*shape, generator=_g(seed + 1), dtype=torch.float64)
    b[-min(n, 3) :] *= 10.0  # weight the tail rows
    op = trnsparse.prepare_cg_bsr(
        trnsparse.BSRMatrix.from_dense(A.float(), block_size=P), preconditioner=pc, device=DEV
    )
    x, _, _, info = trnsparse.cg_bsr_fused(
        op, b, tol=1e-6, max_iter=500, matvec=matvec, precision=precision, return_info=True
    )
    assert info["path"] == "nki" or not REQUIRE_NKI_PATH, info
    ref = torch.linalg.solve(A, b)
    tol = 1e-3 if precision == "fp32" else 3e-2
    return _rel(x, ref), tol


def case_power(n, matvec="column", seed=6):
    A = _spd(n, seed)
    A[n - 1, n - 1] += 4.0 * float(A.diagonal().max())  # dominant pair lives in the tail
    v0 = torch.randn(n, generator=_g(seed + 1), dtype=torch.float64)
    op = trnsparse.prepare_cg_bsr(
        trnsparse.BSRMatrix.from_dense(A.float(), block_size=P), device=DEV
    )
    lam, v, _, info = trnsparse.power_iteration_bsr_fused(
        op, v0=v0, tol=1e-4, max_iter=3000, matvec=matvec, return_info=True
    )
    assert info["path"] == "nki" or not REQUIRE_NKI_PATH, info
    ev, V = torch.linalg.eigh(A)
    lam_err = abs(lam - float(ev[-1])) / abs(float(ev[-1]))
    vec_err = 1.0 - abs(float(v.double() @ V[:, -1]))
    return max(lam_err, vec_err), 1e-4


# ── Case table (shared with the probe) ────────────────────────────────────────


def _s(*dims) -> str:
    return "x".join(str(d) for d in dims)


CASES: list[tuple[str, str, object]] = []
for M, K, N in SPMM_SHAPES:
    CASES.append(("spmm_dense", _s(M, K, N), lambda M=M, K=K, N=N: case_spmm(M, K, N)))
for m, n, N in BSR_SHAPES:
    CASES.append(("bsr_spmm_static", _s(m, n, N), lambda m=m, n=n, N=N: case_bsr(m, n, N)))
    CASES.append(
        ("bsr_spmm_gather", _s(m, n, N), lambda m=m, n=n, N=N: case_bsr(m, n, N, static=False))
    )
for m, n, N in [(129, 200, 513), (4097, 257, 5)]:
    CASES.append(
        (
            "bsr_spmm_static_bf16",
            _s(m, n, N),
            lambda m=m, n=n, N=N: case_bsr(m, n, N, dtype=torch.bfloat16),
        )
    )
for b, (m, n, N) in itertools.product(
    (16, 32, 64), [(200, 200, 7), (256, 256, 513), (1000, 300, 33)]
):
    CASES.append(
        (
            f"bsr_spmm_block{b}",
            _s(m, n, N),
            lambda m=m, n=n, N=N, b=b: case_bsr_unsupported(m, n, N, b),
        )
    )
for M, N in SCREENED_SHAPES:
    CASES.append(("screened_spmm", _s(M, M, N), lambda M=M, N=N: case_screened(M, N)))
for seq, hd in ATTN_SHAPES:
    if hd <= P:
        CASES.append(("attn_fwd_static", _s(seq, hd), lambda s=seq, h=hd: case_attn(s, h)))
        CASES.append(
            ("attn_bwd_static", _s(seq, hd), lambda s=seq, h=hd: case_attn(s, h, backward=True))
        )
    CASES.append(
        ("attn_fwd_gather", _s(seq, hd), lambda s=seq, h=hd: case_attn(s, h, static=False))
    )
    CASES.append(
        (
            "attn_bwd_gather",
            _s(seq, hd),
            lambda s=seq, h=hd: case_attn(s, h, static=False, backward=True),
        )
    )
# Non-default QK scale (trnsparse#47: it was dropped on every NKI path).
for path, static in (("static", True), ("gather", False)):
    for direction, bwd in (("fwd", False), ("bwd", True)):
        for scale in (0.1, 2.0):
            CASES.append(
                (
                    f"attn_{direction}_{path}_scale{scale}",
                    _s(384, 64),
                    lambda st=static, b=bwd, sc=scale: case_attn(
                        384, 64, static=st, backward=b, scale=sc
                    ),
                )
            )
CASES.append(("attn_fwd_gather_scale0.1", _s(384, 256), lambda: case_attn(384, 256, scale=0.1)))
for n in FUSED_N:
    for R in (1, 3, 33):
        CASES.append(("cg_bsr_fused", f"n={n},R={R}", lambda n=n, R=R: case_cg(n, R)))
    CASES.append(("cg_bsr_fused_jacobi", f"n={n},R=1", lambda n=n: case_cg(n, pc="block_jacobi")))
    CASES.append(("cg_bsr_fused_row", f"n={n},R=1", lambda n=n: case_cg(n, matvec="row")))
    CASES.append(("power_bsr_fused", f"n={n}", lambda n=n: case_power(n)))
    CASES.append(("power_bsr_fused_row", f"n={n}", lambda n=n: case_power(n, matvec="row")))
for n in (200, 4097):
    CASES.append(
        (
            "cg_bsr_fused_bf16x2",
            f"n={n},R=3",
            lambda n=n: case_cg(n, R=3, matvec="column", precision="bf16x2"),
        )
    )

# Cases where a loud error is the documented behaviour (never a silent result).
LOUD_OK = {("attn_partial", "200x16")}
CASES.append(("attn_partial", "200x16", case_attn_partial))


@pytest.fixture(autouse=True)
def nki_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


@pytest.mark.parametrize("kernel,shape,fn", CASES, ids=[f"{k}-{s}" for k, s, _ in CASES])
def test_tail(kernel, shape, fn):
    if (kernel, shape) in LOUD_OK:
        with pytest.raises(ValueError):
            fn()
        return
    err, tol = fn()
    assert err <= tol, f"{kernel} {shape}: rel err {err:.3e} > {tol:.0e}"
