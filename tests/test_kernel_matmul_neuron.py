"""Fused kernel-matrix product (#93) on hardware (TorchNeuron, Trn2).

    TRNSPARSE_REQUIRE_NKI=1 pytest tests/test_kernel_matmul_neuron.py -v -m neuron

Accuracy against the float64 reference at ragged sizes, the cached-handle
and custom-op launch paths agreeing bit for bit, and run-to-run bitwise
determinism. With TRNSPARSE_KMV_DUMP=<dir> the determinism case also saves
its output, so runs at NEURON_LOGICAL_NC_CONFIG=1 and =2 can be compared
bitwise (see `benchmarks/bench_kernel_matmul_trn2.py --compare`).

`test_subtile_*` run the row-tiled distance matmuls (#119,
TRNSPARSE_KMV_SUBTILE=1) against the same tolerances and the default kernel;
`test_norm_bias_on_device` adds TRNSPARSE_KMV_NORM_BIAS=1.

The multi-core case (#102) needs several visible logical cores and skips
otherwise:

    NEURON_RT_VISIBLE_CORES=0-3 TRNSPARSE_REQUIRE_NKI=1 \\
        pytest tests/test_kernel_matmul_neuron.py -v -m neuron -k multicore
"""

from __future__ import annotations

import os

import pytest
import torch

import trnsparse
from trnsparse.nki import HAS_NKI
from trnsparse.nki import dispatch as _d


def _native() -> bool:
    try:
        return HAS_NKI and not _d._use_simulator() and _d._get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

DEV = "neuron"


@pytest.fixture(autouse=True)
def nki_backend():
    prev = trnsparse.get_backend()
    trnsparse.set_backend("nki")
    yield
    trnsparse.set_backend(prev)


def _rel(a, b):
    return float((a.cpu().double() - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern32", "matern52"])
@pytest.mark.parametrize(
    "n,m,d,R",
    [
        (1, 1, 3, 1),
        (127, 129, 3, 7),
        (129, 127, 1, 64),
        (1000, 4097, 3, 1),
        (4097, 1000, 16, 513),
        (700, 513, 126, 2),
    ],
)
def test_cross_on_device(kernel, n, m, d, R):
    g = torch.Generator().manual_seed(n + m)
    X, Y, V = (
        torch.rand(n, d, generator=g),
        torch.rand(m, d, generator=g),
        torch.randn(m, R, generator=g),
    )
    ell = 0.5 * d**0.5
    ref = trnsparse.kernel_matmul_reference(X, Y, V, kernel, ell, 1.5)
    out = trnsparse.kernel_matmul(X.to(DEV), Y.to(DEV), V.to(DEV), kernel, ell, 1.5)
    assert out.device.type == DEV
    assert _rel(out, ref) < 2e-5


@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern52"])
@pytest.mark.parametrize("n", [1, 129, 1000, 4097, 20000])
def test_square_operator_on_device(kernel, n):
    g = torch.Generator().manual_seed(n)
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, kernel, 0.1, 1.0, noise=1e-4, device=DEV)
    assert op.route == "nki"
    ref = trnsparse.kernel_matmul_reference(X, None, V, kernel, 0.1, 1.0, noise=1e-4)
    # Matern-1/2 has a cusp at D = 0: near-duplicate off-diagonal pairs keep the
    # augmented-GEMM cancellation error (~u|x|^2 in D, amplified by sqrt). The
    # diagonal is exact (#143); the off-diagonal residue is #119's recentring item.
    # Measured on trn2: 2.9e-5 at n=4097, l=0.1 (window 7).
    tol = 5e-5 if kernel == "matern12" else 2e-5
    assert _rel(op(V.to(DEV)), ref) < tol


@pytest.mark.parametrize("blocked", [False, True], ids=["one-launch", "row-blocked"])
@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern32", "matern52"])
def test_square_operator_diagonal_exact_on_device(monkeypatch, kernel, blocked):
    """K_ii = s on the device (#143), one launch and in row blocks (`_kmv_selfblk_kernel`).

    l = 0.05 and |x| up to 3.5 after centring: the expansion's D_ii ~ u |x|^2
    put up to ~5e-3 relative error on K_ii for Matern-1/2 before the
    correction. V = I reads K + noise I. Off the diagonal the square
    operator is the cross kernel on (X, X) bit for bit.
    """
    n = 1000 + 37
    if blocked:
        monkeypatch.setenv("TRNSPARSE_KMV_BLOCK_WORK", str(256 * n))
    g = torch.Generator().manual_seed(7)
    X, V = 4.0 * torch.rand(n, 3, generator=g), torch.eye(n)
    op = trnsparse.KernelOperator(X, kernel, 0.05, 1.3, noise=1e-2, device=DEV)
    out = op(V.to(DEV)).cpu()
    torch.testing.assert_close(out.diagonal(), torch.full((n,), 1.3 + 1e-2), rtol=3e-7, atol=0)
    cross = op.cross(X, V.to(DEV)).cpu()
    off = ~torch.eye(n, dtype=torch.bool)
    assert torch.equal(out[off], cross[off])


@pytest.mark.parametrize("R", [1, 33, 130])
@pytest.mark.parametrize("group", [4, 8, 16, 32])
def test_group_sizes_on_device(monkeypatch, group, R):
    """Every loop-body size (#103: default `kmv_group(m)`, 8 here) is accurate on the device."""
    monkeypatch.setenv("TRNSPARSE_KMV_GROUP", str(group))
    g = torch.Generator().manual_seed(R)
    n = 8192 + 77  # a ragged Y tile and ragged rows
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, R, generator=g)
    op = trnsparse.KernelOperator(X, "matern32", 0.1, 1.0, noise=1e-4, device=DEV)
    ref = trnsparse.kernel_matmul_reference(X, None, V, "matern32", 0.1, 1.0, noise=1e-4)
    assert _rel(op(V.to(DEV)), ref) < 2e-5


@pytest.mark.parametrize("R", [1, 128, 1500])
@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern32", "matern52"])
def test_auto_group_32_on_device(monkeypatch, kernel, R):
    """m >= 131072 picks the 32-tile body (#108): accurate for every kernel and V width."""
    from trnsparse.nki.kernels import kmv_group

    monkeypatch.delenv("TRNSPARSE_KMV_GROUP", raising=False)
    g = torch.Generator().manual_seed(R)
    m, q = 131072 + 77, 600
    assert kmv_group(m) == 32
    X, Y = torch.rand(m, 3, generator=g), torch.rand(q, 3, generator=g)
    W = torch.randn(m, R, generator=g)
    op = trnsparse.KernelOperator(X, kernel, 0.05, 1.0, device=DEV)
    ref = trnsparse.kernel_matmul_reference(Y, X, W, kernel, 0.05, 1.0)
    # Matern-1/2's cusp at 0 errs like sqrt(u |x|^2) / l whatever the group
    # (about 5e-5 here at group 16 or 32); the smooth kernels stay near 1e-5.
    tol = 2e-4 if kernel == "matern12" else 3e-5
    assert _rel(op.cross(Y, W.to(DEV)), ref) < tol


def test_handle_and_custom_op_agree(monkeypatch):
    g = torch.Generator().manual_seed(3)
    X, V = torch.rand(3000, 3, generator=g), torch.randn(3000, 17, generator=g)
    op = trnsparse.KernelOperator(X, "rbf", 0.2, 1.0, noise=1e-3, device=DEV)
    Vd = V.to(DEV)
    a = op(Vd).cpu()
    monkeypatch.setattr(_d, "_EAGER_HANDLE_ENABLED", False)
    b = op(Vd).cpu()
    assert torch.equal(a, b)


def test_bitwise_deterministic():
    g = torch.Generator().manual_seed(4)
    n = 50_000
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, "rbf", 0.05, 1.0, noise=1e-4, device=DEV)
    Vd = V.to(DEV)
    a, b = op(Vd).cpu(), op(Vd).cpu()
    assert torch.equal(a, b)
    out = os.environ.get("TRNSPARSE_KMV_DUMP")
    if out:
        lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "default")
        os.makedirs(out, exist_ok=True)
        torch.save(a, os.path.join(out, f"kmv_n{n}_lnc{lnc}.pt"))


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
def test_row_blocked_is_bitwise_one_launch(monkeypatch, kernel, self_op):
    """Forced small row blocks (#100) == one launch, bit for bit, on the device.

    n = 5000 in 1024-row blocks: four full blocks and a padded ragged one.
    """
    g = torch.Generator().manual_seed(100)
    n, R = 5000, 33
    m = n if self_op else 3000
    X = torch.rand(n, 3, generator=g)
    Y = None if self_op else torch.rand(m, 3, generator=g)
    V = torch.randn(m, R, generator=g)
    op = trnsparse.KernelOperator(
        X if self_op else Y, kernel, 0.1, 1.0, noise=1e-4 if self_op else 0.0, device=DEV
    )
    Vd = V.to(DEV)

    def apply():
        return (op(Vd) if self_op else op.cross(X, Vd)).cpu()

    monkeypatch.delenv("TRNSPARSE_KMV_BLOCK_WORK", raising=False)
    one = apply()
    monkeypatch.setenv("TRNSPARSE_KMV_BLOCK_WORK", str(1024 * m))
    assert _d._kernels.kmv_block_rows(n, m, _d._kmv_block_work()) == 1024
    n_handles = len(_d._eager_handles)
    blocked = apply()
    assert torch.equal(one, blocked)
    assert len(_d._eager_handles) - n_handles <= 1  # one handle for every block shape
    ref = trnsparse.kernel_matmul_reference(
        X, None if self_op else Y, V, kernel, 0.1, 1.0, noise=1e-4 if self_op else 0.0
    )
    assert _rel(blocked, ref) < 2e-5


@pytest.mark.skipif(
    os.environ.get("TRNSPARSE_KMV_BIG") != "1", reason="set TRNSPARSE_KMV_BIG=1 (minutes)"
)
def test_one_million_rows_under_the_runtime_timeout():
    """n = 2^20, R = 33: was one 40-80 s launch (NRT status 5 at 30 s); now 19 launches."""
    import time

    n = 1 << 20
    g = torch.Generator().manual_seed(7)
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, "rbf", 0.05, 1.0, noise=1e-4, device=DEV)
    Vd = V.to(DEV)
    out = op(Vd)  # compile + first run
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    out = op(Vd)
    torch.neuron.synchronize()
    dt = time.perf_counter() - t0
    W = _d._kmv_block_work()
    cores = _d._kmv_core_count(n, n, _d._visible_cores(), _d._kmv_cores_setting(), W)
    rows = _d._kernels.kmv_block_rows(n, n, W, parts=cores)
    print(
        f"\nn={n} R=33 rbf: {dt:.2f} s, {n * n / dt:.3e} entries/s, "
        f"{cores} core(s), {-(-n // rows)} blocks of {rows}"
    )
    dump = os.environ.get("TRNSPARSE_KMV_DUMP")
    if dump:  # compare runs at different TRNSPARSE_KMV_CORES bitwise (#102)
        os.makedirs(dump, exist_ok=True)
        torch.save(out.cpu(), os.path.join(dump, f"kmv_n{n}_cores{cores}.pt"))
    idx = torch.linspace(0, n - 1, 64).long()
    ref = trnsparse.kernel_matmul_reference(X[idx], X, V, "rbf", 0.05, 1.0)
    ref += 1e-4 * V[idx].double()
    assert _rel(out.cpu()[idx], ref) < 2e-5


@pytest.mark.parametrize("threads", ["0", "1"], ids=["async", "threads"])
@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
def test_multicore_is_bitwise_single_core(monkeypatch, kernel, self_op, threads):
    """Row blocks over every visible logical core (#102) == one core, bit for bit.

    n = 5000 in forced small blocks: 1024-row blocks on one core, and the
    block count rounded up to a multiple of the core count across cores.
    Needs NEURON_RT_VISIBLE_CORES covering 2+ logical cores.
    """
    visible = _d._visible_cores()
    if visible < 2:
        pytest.skip("needs >= 2 visible logical NeuronCores (NEURON_RT_VISIBLE_CORES=0-3)")
    g = torch.Generator().manual_seed(102)
    n, R = 5000, 33
    m = n if self_op else 3000
    X = torch.rand(n, 3, generator=g)
    Y = None if self_op else torch.rand(m, 3, generator=g)
    V = torch.randn(m, R, generator=g)
    op = trnsparse.KernelOperator(
        X if self_op else Y, kernel, 0.1, 1.0, noise=1e-4 if self_op else 0.0, device=DEV
    )
    Vd = V.to(DEV)

    def apply():
        out = op(Vd) if self_op else op.cross(X, Vd)
        assert out.device == Vd.device
        return out.cpu()

    used = []
    real = _d._kmv_multicore

    def spy(tensors, self_op_, n_, R_, rows, devices, *a):
        used.append((rows, list(devices)))
        return real(tensors, self_op_, n_, R_, rows, devices, *a)

    monkeypatch.setattr(_d, "_kmv_multicore", spy)
    monkeypatch.setenv("TRNSPARSE_KMV_THREADS", threads)
    monkeypatch.delenv("TRNSPARSE_KMV_BLOCK_WORK", raising=False)
    monkeypatch.setenv("TRNSPARSE_KMV_CORES", "1")
    one = apply()
    monkeypatch.setenv("TRNSPARSE_KMV_BLOCK_WORK", str(1024 * m))
    single = apply()
    assert not used
    monkeypatch.setenv("TRNSPARSE_KMV_CORES", "all")
    n_handles = len(_d._eager_handles)
    multi = apply()
    again = apply()
    assert len(used) == 2
    rows, devices = used[0]
    assert len(devices) == min(visible, -(-n // rows)) >= 2
    assert len({d.index for d in devices}) == len(devices)
    assert len(_d._eager_handles) - n_handles <= 1  # one handle for every core
    assert torch.equal(single, one)
    assert torch.equal(multi, single)
    assert torch.equal(again, multi)
    ref = trnsparse.kernel_matmul_reference(
        X, None if self_op else Y, V, kernel, 0.1, 1.0, noise=1e-4 if self_op else 0.0
    )
    assert _rel(multi, ref) < 2e-5


@pytest.mark.parametrize("self_op", [True, False], ids=["square", "cross"])
@pytest.mark.parametrize("on_device", [True, False], ids=["device_v", "host_v"])
def test_complex_v_on_device(self_op, on_device):
    """complex64 V (#120): one real launch on [Re V, Im V], complex64 result,
    against a dense complex128 reference (through 0.15.0: K @ Re(V), real)."""
    g = torch.Generator().manual_seed(120)
    n, m, d, R = 300, 257, 3, 5
    X = torch.rand(n, d, generator=g)
    Y = None if self_op else torch.rand(m, d, generator=g)
    rows = n if self_op else m
    V = torch.complex(torch.randn(rows, R, generator=g), torch.randn(rows, R, generator=g))
    Xd = X.double()
    Kd = torch.exp(-torch.cdist(Xd, Xd if self_op else Y.double()).square() / (2 * 0.6**2))
    ref = 1.5 * Kd.to(torch.complex128) @ V.to(torch.complex128)
    if self_op:
        ref += 0.1 * V.to(torch.complex128)
    Vin = V.to(DEV) if on_device else V
    out = trnsparse.kernel_matmul(
        X.to(DEV),
        None if self_op else Y.to(DEV),
        Vin,
        "rbf",
        0.6,
        1.5,
        noise=0.1 if self_op else 0.0,
    )
    assert out.dtype == torch.complex64 and out.shape == (n, R)
    assert out.device.type == Vin.device.type
    got = out.cpu().to(torch.complex128)
    err = float((got - ref).abs().max() / ref.abs().max())
    assert err < 2e-5, err


# ── Row-tiled distance matmuls (#119, TRNSPARSE_KMV_SUBTILE=1) ───────────────
#
# Same arithmetic and summation order as the default kernel (bitwise in the
# simulator); on the device the gate is the default kernel's tolerances, and
# the bitwise comparison with the default kernel is reported separately.


def _both(monkeypatch, fn):
    """fn() with the default kernel, then with TRNSPARSE_KMV_SUBTILE=1."""
    monkeypatch.delenv("TRNSPARSE_KMV_NORM_BIAS", raising=False)
    monkeypatch.delenv("TRNSPARSE_KMV_SUBTILE", raising=False)
    base = fn()
    monkeypatch.setenv("TRNSPARSE_KMV_SUBTILE", "1")
    return base, fn()


@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern32", "matern52"])
@pytest.mark.parametrize(
    "n,m,d,R",
    [
        (127, 129, 3, 7),
        (1000, 4097, 3, 1),
        (4097, 1000, 16, 513),
        (700, 513, 40, 2),
        (300, 1100, 3, 700),
    ],
)
def test_subtile_cross_on_device(monkeypatch, kernel, n, m, d, R):
    g = torch.Generator().manual_seed(n + m)
    X, Y = torch.rand(n, d, generator=g), torch.rand(m, d, generator=g)
    V = torch.randn(m, R, generator=g)
    ell = 0.5 * d**0.5
    ref = trnsparse.kernel_matmul_reference(X, Y, V, kernel, ell, 1.5)
    Xd, Yd, Vd = X.to(DEV), Y.to(DEV), V.to(DEV)
    base, st = _both(
        monkeypatch, lambda: trnsparse.kernel_matmul(Xd, Yd, Vd, kernel, ell, 1.5).cpu()
    )
    assert _rel(st, ref) < 2e-5
    assert _rel(st, base.double()) < 2e-5


@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern52"])
@pytest.mark.parametrize("n", [129, 4097, 20000])
def test_subtile_square_operator_on_device(monkeypatch, kernel, n):
    g = torch.Generator().manual_seed(n)
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, kernel, 0.1, 1.0, noise=1e-4, device=DEV)
    ref = trnsparse.kernel_matmul_reference(X, None, V, kernel, 0.1, 1.0, noise=1e-4)
    Vd = V.to(DEV)
    base, st = _both(monkeypatch, lambda: op(Vd).cpu())
    tol = 5e-5 if kernel == "matern12" else 2e-5  # as test_square_operator_on_device
    assert _rel(st, ref) < tol
    assert _rel(st, base.double()) < tol


@pytest.mark.parametrize("blocked", [False, True], ids=["one-launch", "row-blocked"])
@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern32", "matern52"])
def test_subtile_diagonal_exact_on_device(monkeypatch, kernel, blocked):
    """K_ii = s with row-tiled distance matmuls (#143), one launch and row-blocked."""
    n = 1000 + 37
    if blocked:
        monkeypatch.setenv("TRNSPARSE_KMV_BLOCK_WORK", str(256 * n))
    monkeypatch.setenv("TRNSPARSE_KMV_SUBTILE", "1")
    g = torch.Generator().manual_seed(7)
    X, V = 4.0 * torch.rand(n, 3, generator=g), torch.eye(n)
    op = trnsparse.KernelOperator(X, kernel, 0.05, 1.3, noise=1e-2, device=DEV)
    out = op(V.to(DEV)).cpu()
    torch.testing.assert_close(out.diagonal(), torch.full((n,), 1.3 + 1e-2), rtol=3e-7, atol=0)
    cross = op.cross(X, V.to(DEV)).cpu()
    off = ~torch.eye(n, dtype=torch.bool)
    assert torch.equal(out[off], cross[off])


@pytest.mark.parametrize("R", [1, 128, 1500])
@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern52"])
def test_subtile_auto_group_32_on_device(monkeypatch, kernel, R):
    """The 32-tile body at m >= 131072 (#108) with row tiling; R = 1500 runs passes of 4 tiles."""
    monkeypatch.delenv("TRNSPARSE_KMV_GROUP", raising=False)
    monkeypatch.setenv("TRNSPARSE_KMV_SUBTILE", "1")
    g = torch.Generator().manual_seed(R)
    m, q = 131072 + 77, 600
    X, Y = torch.rand(m, 3, generator=g), torch.rand(q, 3, generator=g)
    W = torch.randn(m, R, generator=g)
    op = trnsparse.KernelOperator(X, kernel, 0.05, 1.0, device=DEV)
    ref = trnsparse.kernel_matmul_reference(Y, X, W, kernel, 0.05, 1.0)
    tol = 2e-4 if kernel == "matern12" else 3e-5  # as test_auto_group_32_on_device
    assert _rel(op.cross(Y, W.to(DEV)), ref) < tol


@pytest.mark.parametrize("group", [4, 7, 32])
def test_subtile_group_sizes_on_device(monkeypatch, group):
    monkeypatch.setenv("TRNSPARSE_KMV_GROUP", str(group))
    monkeypatch.setenv("TRNSPARSE_KMV_SUBTILE", "1")
    g = torch.Generator().manual_seed(group)
    n = 8192 + 77
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, "matern32", 0.1, 1.0, noise=1e-4, device=DEV)
    ref = trnsparse.kernel_matmul_reference(X, None, V, "matern32", 0.1, 1.0, noise=1e-4)
    assert _rel(op(V.to(DEV)), ref) < 2e-5


def test_subtile_bitwise_deterministic(monkeypatch):
    monkeypatch.setenv("TRNSPARSE_KMV_SUBTILE", "1")
    g = torch.Generator().manual_seed(4)
    n = 50_000
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, "rbf", 0.05, 1.0, noise=1e-4, device=DEV)
    Vd = V.to(DEV)
    a, b = op(Vd).cpu(), op(Vd).cpu()
    assert torch.equal(a, b)
    out = os.environ.get("TRNSPARSE_KMV_DUMP")
    if out:
        lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "default")
        os.makedirs(out, exist_ok=True)
        torch.save(a, os.path.join(out, f"kmv_subtile_n{n}_lnc{lnc}.pt"))


@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern32", "matern52"])
@pytest.mark.parametrize("n", [4097, 20000])
def test_norm_bias_on_device(monkeypatch, kernel, n):
    """TRNSPARSE_KMV_NORM_BIAS=1 (|y|^2 in the PSUM reader, #119), with row tiling:
    the default kernel's tolerances, an exact diagonal, and run-to-run bits."""
    monkeypatch.setenv("TRNSPARSE_KMV_SUBTILE", "1")
    monkeypatch.setenv("TRNSPARSE_KMV_NORM_BIAS", "1")
    g = torch.Generator().manual_seed(n)
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, kernel, 0.1, 1.0, noise=1e-4, device=DEV)
    ref = trnsparse.kernel_matmul_reference(X, None, V, kernel, 0.1, 1.0, noise=1e-4)
    Vd = V.to(DEV)
    a, b = op(Vd).cpu(), op(Vd).cpu()
    assert torch.equal(a, b)
    tol = 5e-5 if kernel == "matern12" else 2e-5
    assert _rel(a, ref) < tol
    q = 1000 + 37
    Xq = 4.0 * torch.rand(q, 3, generator=g)
    opq = trnsparse.KernelOperator(Xq, kernel, 0.05, 1.3, noise=1e-2, device=DEV)
    out = opq(torch.eye(q).to(DEV)).cpu()
    torch.testing.assert_close(out.diagonal(), torch.full((q,), 1.3 + 1e-2), rtol=3e-7, atol=0)


@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern52"])
def test_subtile_bitwise_default_kernel_on_device(monkeypatch, kernel):
    """Informative: the TensorEngine's row tiles give the full array's bits.

    Not required (the gate is the tolerances above), but expected: the
    arithmetic and every summation order are the default kernel's.
    """
    g = torch.Generator().manual_seed(5)
    n = 20_000
    X, V = torch.rand(n, 3, generator=g), torch.randn(n, 33, generator=g)
    op = trnsparse.KernelOperator(X, kernel, 0.05, 1.0, noise=1e-4, device=DEV)
    Vd = V.to(DEV)
    base, st = _both(monkeypatch, lambda: op(Vd).cpu())
    diff = float((st - base).abs().max())
    assert torch.equal(st, base), f"max |subtile - default| = {diff:.3e}"
