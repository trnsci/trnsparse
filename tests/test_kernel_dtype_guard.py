"""No NKI kernel is reached with an operand dtype it does not take (CPU).

Through 0.12.1 float64, complex, integer and bool tensors could reach the
fp32 / bf16 kernels unchanged and come back as garbage, a dropped imaginary
part, or the wrong dtype (#78, #79, trnsci/trnsci#35). Here every
`torch_neuronx.nki_op` is replaced by a stub that records each call's
tensor operand dtypes, checks them against what the kernel takes (the
table below, written from the kernels, not copied from `dispatch`), and
returns one-filled outputs from the op's fake. Every public API that can
reach a kernel is then driven with fp32, fp64, bf16, fp16, int32, int64,
bool, complex64 and complex128 values on the native path (static and
gather kernels) and the XLA path, with backend "nki" and "auto", with and
without TRNSPARSE_REQUIRE_NKI. The result dtype must match the pytorch
backend's for the same inputs.

Stub violations are recorded rather than raised, because the host wrappers
fall back to PyTorch on a kernel exception when REQUIRE_NKI is off.
"""

import inspect
import re
import sys
import types

import pytest
import torch

import trnsparse
import trnsparse.nki
from trnsparse import iterative
from trnsparse.nki import dispatch

if not hasattr(torch.library, "custom_op"):
    pytest.skip("torch.library.custom_op needs torch >= 2.4", allow_module_level=True)

F32 = {torch.float32}
MM = {torch.float32, torch.bfloat16}
MM16 = MM | {torch.float16}
ANYF = MM16 | {torch.float64}
IDX = {torch.int32, torch.int64}

# What each kernel takes, per tensor operand (non-tensor arguments skipped).
KERNEL_TAKES = {
    "spmm_dense": (MM16, MM16),
    "bsr_spmm": (MM16, MM16),
    "screened_spmm": (MM16,) * 4,
    "bsr_spmm_static": (MM, MM),
    "bsr_spmm_static_t": (MM, MM),
    "bsr_spmm_keyed": (MM, MM),
    "bsr_spmm_t_keyed": (MM, MM),
    "attn_fwd_keyed": (MM, MM, MM),
    "attn_fwd_static": (MM, MM, MM),
    "attn_bwd_static": (MM,) * 7 + (F32,) * 6,
    "attn_stats": (MM, MM),
    "attn_out": (MM, MM, MM, F32, F32),
    "attn_bwd_dq": (MM,) * 4 + (F32,) * 3,
    "attn_bwd_dkdv": (MM,) * 4 + (F32,) * 3,
    "cg_bsr_keyed": (MM, F32, F32),
    "power_bsr_keyed": (MM, F32),
}
# Composite ops (not kernels; each inner launch is checked by its own stub).
COMPOSITE = {"bsr_attn", "bsr_attn_bwd"}
# No public API launches these today (the keyed variants replaced them).
UNREACHED = {"bsr_spmm_static", "bsr_spmm_static_t", "attn_fwd_static", "bsr_spmm_t_keyed"}

DTYPES = [
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
]
NO_KERNEL = {torch.int32, torch.int64, torch.bool, torch.complex64, torch.complex128}
P = 128


class _Log:
    def __init__(self):
        self.calls = []  # (op, dtypes)
        self.violations = []


def _filled(out):
    # Ones, not zeros: cg_bsr needs p.Ap != 0 to take a step.
    if isinstance(out, tuple | list):
        return tuple(torch.ones_like(o) for o in out)
    return torch.ones_like(out)


def _check(log, name, args):
    dts = [a.dtype for a in args if isinstance(a, torch.Tensor)]
    log.calls.append((name, tuple(dts)))
    takes = KERNEL_TAKES[name]
    if len(dts) != len(takes) or any(d not in ok for d, ok in zip(dts, takes, strict=False)):
        log.violations.append((name, dts))


class _StubOp:
    def __init__(self, name, fake, log):
        self.name, self.fake, self.log = name, fake, log

    def __call__(self, *args):
        _check(self.log, self.name, args)
        return _filled(self.fake(*args))

    def register_autograd(self, *a, **k):
        pass


def _stub_torch_neuronx(log):
    mod = types.ModuleType("torch_neuronx")

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        def dec(f):
            return _StubOp(name.split("::")[1], fake_fn, log)

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: 1
    return mod


_XLA_KERNELS = {
    "_spmm_dense_kernel": "spmm_dense",
    "_bsr_spmm_kernel": "bsr_spmm",
    "_screened_spmm_kernel": "screened_spmm",
    "_attn_stats_kernel": "attn_stats",
    "_attn_out_kernel": "attn_out",
    "_attn_bwd_dq_kernel": "attn_bwd_dq",
    "_attn_bwd_dkdv_kernel": "attn_bwd_dkdv",
}


@pytest.fixture(params=["native-static", "native-gather", "xla"])
def path(request):
    return request.param


@pytest.fixture(params=["nki", "auto"])
def backend(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["fallback", "require"])
def require(request):
    return request.param


@pytest.fixture
def stubs(monkeypatch, path, backend, require):
    log = _install_stubs(monkeypatch, path, backend, require)
    yield log
    assert not log.violations, log.violations


def _install_stubs(monkeypatch, path, backend, require):
    """Native (or XLA) dispatch on CPU with every kernel replaced by a checking stub."""
    log = _Log()
    monkeypatch.setitem(sys.modules, "torch_neuronx", _stub_torch_neuronx(log))
    # A fresh `_native` under the stub; removed again afterwards (monkeypatch).
    monkeypatch.delitem(sys.modules, "trnsparse.nki._native", raising=False)
    monkeypatch.delattr(trnsparse.nki, "_native", raising=False)
    monkeypatch.setattr(dispatch, "_native_ops", None)
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", require)
    monkeypatch.setattr(dispatch, "_backend", backend)
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setattr(dispatch, "_STATIC_STRUCTURE", path == "native-static")
    # The fused solvers run their kernel where the operator lives: CPU here.
    monkeypatch.setattr(iterative, "_use_fused_kernel", lambda op: dispatch._native_path())
    from trnsparse.nki import _native

    # The host wrappers name the gather kernels (imported only with nki):
    # on the XLA path `_run` calls them, so they are the checking stubs too.
    for attr, name in _XLA_KERNELS.items():
        stub = _native.OPS[name]  # a _StubOp: same check, same fake
        monkeypatch.setattr(dispatch, attr, stub, raising=False)
    if path == "xla":
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        monkeypatch.setattr(dispatch, "_to_xla", lambda *t: (list(t), t[0].device))
    else:
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")

        def no_xla(*t):
            raise AssertionError("_to_xla on the native path")

        monkeypatch.setattr(dispatch, "_to_xla", no_xla)

    # Layer 2 must never be what stops an API call: record its refusals too.
    real_check = getattr(dispatch, "_check_kernel_operands", None)

    def check(op_name, args):
        try:
            real_check(op_name, args)
        except TypeError as e:
            log.violations.append(("guard", str(e)))
            raise

    if real_check is not None:  # (absent before 0.12.2)
        monkeypatch.setattr(dispatch, "_check_kernel_operands", check)
    return log


def _rnd(shape, dtype, g):
    if dtype.is_complex:
        re_, im_ = torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)
        return torch.complex(re_, im_).to(dtype)
    if dtype == torch.bool:
        return torch.randn(*shape, generator=g) > 0
    if not dtype.is_floating_point:
        return torch.randint(-3, 4, shape, generator=g).to(dtype)
    return torch.randn(*shape, generator=g).to(dtype)


def _spd_bsr():
    g = torch.Generator().manual_seed(3)
    M = torch.randn(256, 256, generator=g) * 0.01
    A = M @ M.T + 2 * torch.eye(256)
    A[:128, 128:] = 0
    A[128:, :128] = 0
    return trnsparse.BSRMatrix.from_dense(A, block_size=P)


def _case(api, dtype):
    """(callable, routed) for one API and value dtype; routed: no kernel takes it."""
    g = torch.Generator().manual_seed(0)
    if api == "spmm":
        Ad = _rnd((200, 150), dtype, g)
        Ad[:, ::3] = 0
        A = trnsparse.from_dense(Ad)
        B = _rnd((150, 7), dtype, g)
        return lambda: trnsparse.spmm(A, B)
    if api == "spmm_f32_rhs":  # mixed: values in `dtype`, fp32 B
        Ad = _rnd((200, 150), dtype, g)
        A = trnsparse.from_dense(Ad)
        B = torch.randn(150, 4, generator=g)
        return lambda: trnsparse.spmm(A, B)
    if api == "bsr_spmm":
        Ad = _rnd((256, 384), dtype, g)
        Ad[:128, 128:256] = 0
        A = trnsparse.BSRMatrix.from_dense(Ad, block_size=P)
        B = _rnd((384, 5), dtype, g)
        return lambda: trnsparse.bsr_spmm(A, B)
    if api == "screened_spmm":
        A = _rnd((200, 200), dtype, g)
        diag = _rnd((200,), dtype, g)
        B = _rnd((200, 6), dtype, g)
        return lambda: trnsparse.screened_spmm(A, diag, B, 0.25)
    if api in ("attention", "attention_grad"):
        mask = trnsparse.BSRMatrix.from_dense(
            torch.tensor([[1.0, 0.0], [1.0, 1.0]]).repeat_interleave(P, 0).repeat_interleave(P, 1),
            block_size=P,
        )
        Q, K, V = (_rnd((256, 64), dtype, g) for _ in range(3))
        if api == "attention":
            return lambda: trnsparse.block_sparse_attention_tiled(Q, K, V, mask)

        def fwd_bwd():
            q, k, v = (t.detach().clone().requires_grad_(True) for t in (Q, K, V))
            out = trnsparse.block_sparse_attention_tiled(q, k, v, mask)
            out.float().sum().backward()
            return q.grad

        return fwd_bwd
    if api == "cg_bsr":
        A = _spd_bsr()
        b = _rnd((256,), dtype, g)
        return lambda: trnsparse.cg_bsr(A, b, max_iter=3)[0]
    if api == "cg_bsr_fused":
        A = _spd_bsr()
        b = _rnd((256,), dtype, g)
        return lambda: trnsparse.cg_bsr_fused(A, b, max_iter=3)[0]
    if api == "power_fused":
        Ad = _spd_bsr().to_dense().to(dtype) if dtype.is_floating_point else None
        if Ad is None:
            pytest.skip("power iteration takes a float matrix")
        A = trnsparse.BSRMatrix.from_dense(Ad, block_size=P)
        return lambda: trnsparse.power_iteration_bsr_fused(A, max_iter=3)[1]
    raise AssertionError(api)


# APIs whose value dtype decides kernel vs PyTorch (REQUIRE_NKI raises for
# complex / integer / bool). The fused solvers pack b into an fp32 kernel
# state on every backend (see the needs-decision issue in the PR).
ROUTED_APIS = {"spmm", "spmm_f32_rhs", "bsr_spmm", "screened_spmm", "attention", "attention_grad"}
APIS = sorted(ROUTED_APIS | {"cg_bsr", "cg_bsr_fused", "power_fused"})


def _outcome(fn):
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001 — the pytorch backend may refuse a dtype
        return None, e


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("api", APIS)
def test_no_unsupported_dtype_reaches_a_kernel(stubs, monkeypatch, api, dtype, require):
    if api == "attention_grad" and not dtype.is_floating_point:
        pytest.skip("gradients need a float dtype")
    try:
        fn = _case(api, dtype)
    except (RuntimeError, TypeError, ValueError) as e:
        pytest.skip(f"inputs not constructible for {dtype}: {e}")

    with monkeypatch.context() as m:
        m.setattr(dispatch, "_backend", "pytorch")
        ref, ref_err = _outcome(fn)

    got, err = _outcome(fn)
    assert not stubs.violations, stubs.violations
    routed = api in ROUTED_APIS and dtype in NO_KERNEL
    if api == "bsr_spmm" and dtype.is_complex and hasattr(dispatch, "_nki_bsr_spmm_complex"):
        routed = False  # #80: complex BSR SpMM as real kernel launches
    if require and routed:
        assert isinstance(err, NotImplementedError), (err, got)
        return
    if ref_err is not None:
        return  # the pytorch backend refuses this dtype; raising or a value are both fine
    assert err is None, f"{api} {dtype}: pytorch backend gives {ref.dtype}, nki raised {err!r}"
    assert got.dtype == ref.dtype, (api, dtype, got.dtype, ref.dtype)
    assert got.shape == ref.shape


def test_every_native_op_has_a_dtype_table_entry(stubs):
    from trnsparse.nki import _native

    ops = dispatch._get_native_ops()
    assert set(_native.OPS) == set(ops) == set(KERNEL_TAKES) | COMPOSITE
    assert set(dispatch._KERNEL_OPERAND_DTYPES) == set(ops)
    for name, takes in KERNEL_TAKES.items():
        assert tuple(map(set, dispatch._KERNEL_OPERAND_DTYPES[name])) == takes, name


@pytest.mark.parametrize("bad", [torch.float64, torch.int32, torch.bool, torch.complex64], ids=str)
def test_guarded_ops_refuse_unsupported_dtypes(stubs, monkeypatch, bad):
    ops = dispatch._get_native_ops()
    monkeypatch.setattr(dispatch, "_check_kernel_operands", _real_check)
    for name, takes in KERNEL_TAKES.items():
        args = [torch.zeros(1, 1, dtype=bad)] + [torch.zeros(1, 1)] * (len(takes) - 1)
        with pytest.raises(TypeError, match=name):
            ops[name](*args)
    assert stubs.calls == []  # refused before the kernel stub


_real_check = getattr(dispatch, "_check_kernel_operands", None)


def test_xla_and_simulator_launches_check_the_table(monkeypatch):
    monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
    monkeypatch.setattr(dispatch, "_to_xla", lambda *t: (list(t), t[0].device))

    def kernel(*t):
        raise AssertionError("kernel reached")

    a = torch.zeros(2, 2, dtype=torch.float64)
    with pytest.raises(TypeError, match="spmm_dense"):
        dispatch._launch(kernel, "spmm_dense", a, a)
    with pytest.raises(TypeError, match="no operand dtype table entry"):
        dispatch._launch(kernel, "new_kernel", a.float())
    with pytest.raises(TypeError, match="attn_stats"):
        dispatch._simulate(kernel, "attn_stats", a, a)
    # Every simulator call goes through `_simulate`, every launch names a table op.
    src = inspect.getsource(dispatch)
    assert set(re.findall(r"nki\.simulate\((\w+)\)\(", src)) == {"kernel"}
    names = re.findall(r'(?:_launch|_run|_simulate)\(\s*_\w+_kernel,\s*"(\w+)"', src)
    assert names and set(names) <= set(KERNEL_TAKES)


def test_every_reachable_kernel_is_exercised(monkeypatch):
    """The APIs above reach every kernel op except the listed unreached ones."""
    seen = set()
    for path, backend in (("native-static", "nki"), ("native-gather", "nki"), ("xla", "nki")):
        with monkeypatch.context() as m:
            log = _install_stubs(m, path, backend, True)
            for api in APIS:
                _case(api, torch.float32)()
            seen |= {name for name, _ in log.calls}
            assert not log.violations
    assert seen == set(KERNEL_TAKES) - UNREACHED, seen ^ (set(KERNEL_TAKES) - UNREACHED)
