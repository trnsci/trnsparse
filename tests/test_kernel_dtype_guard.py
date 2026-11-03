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

float64 (and complex128 in `bsr_spmm`, whose complex path is real kernel
launches) follows the suite fp64 policy (#84): under `auto` it computes in
torch at float64 accuracy and reaches no kernel; under an explicit `nki`
it is cast to fp32 for the kernel with one `UserWarning` per op; under
TRNSPARSE_REQUIRE_NKI it raises `NotImplementedError`.

Stub violations are recorded rather than raised, because the host wrappers
fall back to PyTorch on a kernel exception when REQUIRE_NKI is off.
"""

import inspect
import re
import sys
import types
import warnings

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
    "screened_tiles_keyed": (MM, F32, F32, F32, MM),
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
    # Record the real state first, so teardown removes the stub-backed `_native`
    # imported below (a later test in the session would otherwise get stub ops).
    monkeypatch.setitem(sys.modules, "trnsparse.nki._native", None)
    monkeypatch.setattr(trnsparse.nki, "_native", None, raising=False)
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


# The fp64 policy (#84): the dtypes it covers per API. complex128 only where
# a complex kernel path exists (bsr_spmm, and cg_bsr through it); elsewhere
# complex has no kernel at all (NO_KERNEL). The fused solvers decide on
# float64 only (complex is #85).
FP64_APIS = {api: {torch.float64} for api in APIS}
FP64_APIS["bsr_spmm"] = FP64_APIS["cg_bsr"] = {torch.float64, torch.complex128}
FUSED = {"cg_bsr_fused", "power_fused"}


def _fp64_policy(api, dtype, path):
    """True when the fp64 policy decides this call (it would reach a kernel otherwise)."""
    if dtype not in FP64_APIS[api]:
        return False
    # The fused kernels run on the native path only (the XLA path is the host).
    return not (api in FUSED and path == "xla")


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("api", APIS)
def test_no_unsupported_dtype_reaches_a_kernel(stubs, monkeypatch, api, dtype, require, path):
    if api == "attention_grad" and not dtype.is_floating_point:
        pytest.skip("gradients need a float dtype")
    try:
        fn = _case(api, dtype)
    except (RuntimeError, TypeError, ValueError) as e:
        pytest.skip(f"inputs not constructible for {dtype}: {e}")

    with monkeypatch.context() as m:
        m.setattr(dispatch, "_backend", "pytorch")
        ref, ref_err = _outcome(fn)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the explicit-nki fp64 warning
        got, err = _outcome(fn)
    assert not stubs.violations, stubs.violations
    routed = api in ROUTED_APIS and dtype in NO_KERNEL
    if api == "bsr_spmm" and dtype.is_complex and hasattr(dispatch, "_nki_bsr_spmm_complex"):
        routed = False  # #80: complex BSR SpMM as real kernel launches
    if require and (routed or _fp64_policy(api, dtype, path)):
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


# ── The fp64 policy (#84) ────────────────────────────────────────────────────

FP64_CASES = [
    (api, dtype)
    for api in APIS
    for dtype in (torch.float64, torch.complex128)
    if dtype in FP64_APIS[api]
]
FP64_IDS = [f"{a}-{d}" for a, d in FP64_CASES]


def _fp64_case(api, dtype, path):
    if not _fp64_policy(api, dtype, path):
        pytest.skip("the fused kernels do not run on the XLA path")
    return _case(api, dtype)


def _seeded(fn):
    torch.manual_seed(0)  # power iteration starts from a random vector
    return fn()


def _pytorch_ref(monkeypatch, fn):
    with monkeypatch.context() as m:
        m.setattr(dispatch, "_backend", "pytorch")
        return _outcome(lambda: _seeded(fn))


def _policy_warnings(caught):
    return [
        w for w in caught if issubclass(w.category, UserWarning) and "reduced" in str(w.message)
    ]


@pytest.mark.parametrize("api,dtype", FP64_CASES, ids=FP64_IDS)
def test_fp64_auto_computes_in_torch_at_fp64_accuracy(monkeypatch, path, api, dtype):
    log = _install_stubs(monkeypatch, path, "auto", False)
    fn = _fp64_case(api, dtype, path)
    ref, ref_err = _pytorch_ref(monkeypatch, fn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got, err = _outcome(lambda: _seeded(fn))
    assert log.calls == [], log.calls  # no kernel launch
    assert not _policy_warnings(caught)
    if ref_err is not None:  # the pytorch backend refuses (mixed dtypes): so does auto
        assert type(err) is type(ref_err), (err, ref_err)
        return
    assert err is None, err
    assert got.dtype == ref.dtype
    # The same torch code as the pytorch backend: float64 accuracy.
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("api,dtype", FP64_CASES, ids=FP64_IDS)
def test_fp64_explicit_nki_casts_and_warns_once_per_op(monkeypatch, path, api, dtype):
    log = _install_stubs(monkeypatch, path, "nki", False)
    monkeypatch.setattr(dispatch, "_fp64_warned", set())
    fn = _fp64_case(api, dtype, path)
    ref, ref_err = _pytorch_ref(monkeypatch, fn)
    if ref_err is not None:
        pytest.skip(f"the pytorch backend refuses these operands: {ref_err}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outs = [_seeded(fn) for _ in range(3)]
    assert not log.violations, log.violations
    assert log.calls, "explicit nki must launch the kernel"
    for name, dts in log.calls:  # fp32 operands only (no bf16 / fp16 in these calls)
        assert all(d == torch.float32 for d in dts if d.is_floating_point), (name, dts)
    ws = _policy_warnings(caught)
    assert len(ws) == 1, [str(w.message) for w in ws]  # one per op, across calls
    assert "fp32" in str(ws[0].message)
    for got in outs:
        assert got.dtype == ref.dtype and got.shape == ref.shape


def test_fp64_warning_is_per_op(monkeypatch):
    _install_stubs(monkeypatch, "native-static", "nki", False)
    monkeypatch.setattr(dispatch, "_fp64_warned", set())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for api in ("spmm", "bsr_spmm", "screened_spmm", "spmm", "bsr_spmm"):
            _case(api, torch.float64)()
    ws = _policy_warnings(caught)
    assert len(ws) == 3
    assert sorted(w.message.args[0].split(":")[0] for w in ws) == [
        "trnsparse bsr_spmm",
        "trnsparse screened_spmm",
        "trnsparse spmm",
    ]


@pytest.mark.parametrize("backend_", ["nki", "auto"])
@pytest.mark.parametrize("api,dtype", FP64_CASES, ids=FP64_IDS)
def test_fp64_require_nki_raises(monkeypatch, path, backend_, api, dtype):
    log = _install_stubs(monkeypatch, path, backend_, True)
    fn = _fp64_case(api, dtype, path)
    with pytest.raises(NotImplementedError, match="float64|complex128"):
        fn()
    assert log.calls == []


# Result dtype == the pytorch backend's: same dtypes and mixed promotion.
MIXED = [
    (torch.float32, torch.float32),
    (torch.float64, torch.float64),
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.float32, torch.float64),
    (torch.float64, torch.float32),
    (torch.bfloat16, torch.float32),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float64),
    (torch.complex64, torch.float64),
]


def _mixed_case(api, da, db):
    g = torch.Generator().manual_seed(5)
    if api == "spmm":
        A = trnsparse.from_dense(_rnd((200, 150), da, g))
        B = _rnd((150, 7), db, g)
        return lambda: trnsparse.spmm(A, B)
    if api == "bsr_spmm":
        Ad = _rnd((256, 384), da, g)
        Ad[:128, 128:256] = 0
        A = trnsparse.BSRMatrix.from_dense(Ad, block_size=P)
        B = _rnd((384, 5), db, g)
        return lambda: trnsparse.bsr_spmm(A, B)
    if api == "screened_spmm":
        A, B = _rnd((200, 200), da, g), _rnd((200, 6), db, g)
        diag = _rnd((200,), torch.float32, g)
        return lambda: trnsparse.screened_spmm(A, diag, B, 0.25)
    raise AssertionError(api)


@pytest.mark.parametrize("da,db", MIXED, ids=lambda d: str(d).replace("torch.", ""))
@pytest.mark.parametrize("api", ["spmm", "bsr_spmm", "screened_spmm"])
def test_result_dtype_matches_pytorch_backend(monkeypatch, path, backend, api, da, db):
    log = _install_stubs(monkeypatch, path, backend, False)
    try:
        fn = _mixed_case(api, da, db)
    except (RuntimeError, TypeError, ValueError) as e:
        pytest.skip(f"inputs not constructible: {e}")
    ref, ref_err = _pytorch_ref(monkeypatch, fn)
    if ref_err is not None:
        pytest.skip(f"the pytorch backend refuses {da} x {db}: {ref_err}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        got = fn()
    assert not log.violations, log.violations
    assert got.dtype == ref.dtype == torch.promote_types(da, db), (got.dtype, ref.dtype)
    assert got.shape == ref.shape
    if torch.float64 in (da, db) and backend == "auto":
        assert log.calls == []  # float64 anywhere: torch under auto


# ── Device-resident float64 under `auto`: host torch (#84) ───────────────────
#
# Device aten computes float64 at fp32 precision (measured on Trn2), so under
# `auto` a float64 / complex128 operand on the `neuron` device sends the op to
# the host (`dispatch._fp64_host_guard`). Here the "neuron" device is CPU
# tensors marked `_fake_neuron`: `_on_neuron_device` checks the mark,
# `_host_copy` (the device -> host copy) records itself and returns an
# unmarked clone, and `_move_to` (the result's way back) records its target
# and marks what it returns.

import dataclasses  # noqa: E402

HOST_WARNING = (
    "float64 on the neuron device is computed on the host for float64 accuracy "
    "(device arithmetic is fp32); pass float32 to stay on device"
)


def _neuron(t):
    t = t.detach().clone()
    t._fake_neuron = True
    return t


def _neuron_mat(A):
    return dataclasses.replace(
        A,
        **{
            f.name: _neuron(getattr(A, f.name))
            for f in dataclasses.fields(A)
            if isinstance(getattr(A, f.name), torch.Tensor)
        },
    )


def _is_neuron(t):
    return getattr(t, "_fake_neuron", False)


class _Transfers:
    def __init__(self):
        self.to_host = []  # dtypes copied to the host
        self.back = []  # devices results were moved to


def _fake_neuron(monkeypatch, backend="auto", require=False, on_device=False):
    log = _install_stubs(monkeypatch, "native-gather", backend, require)
    monkeypatch.setattr(dispatch, "_fp64_host_warned", set())
    monkeypatch.setattr(dispatch, "_fp64_warned", set())
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", on_device)
    monkeypatch.setattr(dispatch, "_on_neuron_device", _is_neuron)
    tr = _Transfers()

    def host_copy(t):
        assert _is_neuron(t)
        tr.to_host.append(t.dtype)
        return t.clone()  # unmarked: a host tensor (and differentiable)

    real_move_to = dispatch._move_to

    def move_to(x, device):
        tr.back.append(device)
        out = real_move_to(x, device)
        if isinstance(out, torch.Tensor):
            out._fake_neuron = True
        return out

    monkeypatch.setattr(dispatch, "_host_copy", host_copy)
    monkeypatch.setattr(dispatch, "_move_to", move_to)
    return log, tr


def _sym_csr(n, dtype, g):
    M = _rnd((n, n), dtype, g)
    M[:, ::3] = 0
    return trnsparse.from_dense(M + M.T.conj() if dtype.is_complex else M + M.T)


def _spd_bsr_dtype(dtype):
    return trnsparse.BSRMatrix.from_dense(_spd_bsr().to_dense().to(dtype), block_size=P)


def _device_case(api, dtype):
    """(fn, args): `fn(*args)` is the public call; args hold host operands."""
    g = torch.Generator().manual_seed(11)
    if api == "spmv":
        A = trnsparse.from_dense(_rnd((200, 150), dtype, g))
        return (lambda A, x: trnsparse.spmv(A, x)), (A, _rnd((150,), dtype, g))
    if api == "spmm":
        A = trnsparse.from_dense(_rnd((200, 150), dtype, g))
        return (lambda A, B: trnsparse.spmm(A, B)), (A, _rnd((150, 7), dtype, g))
    if api == "spmv_symmetric":
        A = _sym_csr(150, dtype, g)
        return (lambda A, x: trnsparse.spmv_symmetric(A, x)), (A, _rnd((150,), dtype, g))
    if api == "bsr_spmm":
        Ad = _rnd((256, 384), dtype, g)
        Ad[:128, 128:256] = 0
        A = trnsparse.BSRMatrix.from_dense(Ad, block_size=P)
        return (lambda A, B: trnsparse.bsr_spmm(A, B)), (A, _rnd((384, 5), dtype, g))
    if api == "screened_spmm":
        args = (_rnd((200, 200), dtype, g), _rnd((200,), dtype, g), _rnd((200, 6), dtype, g))
        return (lambda A, d, B: trnsparse.screened_spmm(A, d, B, 0.25)), args
    if api in ("attention", "attention_grad"):
        mask = trnsparse.BSRMatrix.from_dense(
            torch.tensor([[1.0, 0.0], [1.0, 1.0]]).repeat_interleave(P, 0).repeat_interleave(P, 1),
            block_size=P,
        )
        qkv = tuple(_rnd((256, 64), dtype, g) for _ in range(3))
        if api == "attention":
            return (lambda Q, K, V: trnsparse.block_sparse_attention_tiled(Q, K, V, mask)), qkv

        def fwd_bwd(Q, K, V):
            q, k, v = (t.requires_grad_(True) for t in (Q, K, V))
            out = trnsparse.block_sparse_attention_tiled(q, k, v, mask)
            out.sum().backward()
            return q.grad

        return fwd_bwd, qkv
    b = _rnd((256,), dtype, g)
    if api == "cg_bsr":
        return (lambda A, b: trnsparse.cg_bsr(A, b, max_iter=5)[0]), (_spd_bsr_dtype(dtype), b)
    if api == "chebyshev_bsr":
        fn = lambda A, b: trnsparse.chebyshev_bsr(A, b, 1.0, 3.0, K=4)[0]  # noqa: E731
        return fn, (_spd_bsr_dtype(dtype), b)
    if api == "richardson_bsr":
        fn = lambda A, b: trnsparse.richardson_bsr(A, b, 0.4, K=4)[0]  # noqa: E731
        return fn, (_spd_bsr_dtype(dtype), b)
    if api == "power_iteration_bsr":
        fn = lambda A, v0: trnsparse.power_iteration_bsr(A, v0, max_iter=4)[1]  # noqa: E731
        return fn, (_spd_bsr_dtype(dtype), b)
    if api == "jacobi_preconditioner_bsr":
        A = _spd_bsr_dtype(dtype)
        return (lambda r: trnsparse.jacobi_preconditioner_bsr(A)(r)), (b,)
    raise AssertionError(api)


DEVICE_APIS = [
    "spmv",
    "spmm",
    "spmv_symmetric",
    "bsr_spmm",
    "screened_spmm",
    "attention",
    "attention_grad",
    "cg_bsr",
    "chebyshev_bsr",
    "richardson_bsr",
    "power_iteration_bsr",
    "jacobi_preconditioner_bsr",
]
DEVICE_CASES = [(a, torch.float64) for a in DEVICE_APIS] + [
    (a, torch.complex128) for a in ("spmv", "spmm", "bsr_spmm", "cg_bsr")
]
DEVICE_IDS = [f"{a}-{d}" for a, d in DEVICE_CASES]


def _on_device(args):
    return tuple(
        _neuron(a) if isinstance(a, torch.Tensor) else _neuron_mat(a) if a is not None else a
        for a in args
    )


def _host_warnings(caught):
    return [w for w in caught if HOST_WARNING in str(w.message)]


def _call(fn, args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args)
    return out, _host_warnings(caught)


def _host_ref(monkeypatch, fn, args):
    with monkeypatch.context() as m:
        m.setattr(dispatch, "_backend", "pytorch")
        return fn(*args)


@pytest.mark.parametrize("api,dtype", DEVICE_CASES, ids=DEVICE_IDS)
def test_fp64_on_device_auto_computes_on_host(monkeypatch, api, dtype):
    log, tr = _fake_neuron(monkeypatch)
    fn, args = _device_case(api, dtype)
    ref = _host_ref(monkeypatch, fn, args)
    outs = []
    for _ in range(3):
        out, ws = _call(fn, _on_device(args))
        outs.append((out, ws))
    assert log.calls == [], log.calls  # no kernel launch
    assert dtype in tr.to_host  # device operands were copied to the host
    assert tr.back and all(d == torch.device("cpu") for d in tr.back)
    # The warning fires once for the op, on the first call only.
    assert len(outs[0][1]) == 1 and all(not ws for _, ws in outs[1:])
    msg = str(outs[0][1][0].message)
    assert "TRNSPARSE_FP64_ON_DEVICE" in msg
    for out, _ in outs:
        if api != "attention_grad":  # q.grad accumulates on its device leaf
            assert _is_neuron(out), "result must come back to the device"
        assert out.dtype == dtype
        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)


def test_fp64_on_device_warning_is_per_op(monkeypatch):
    _fake_neuron(monkeypatch)
    caught_ops = []
    for api in ("spmm", "bsr_spmm", "spmm", "screened_spmm", "bsr_spmm"):
        fn, args = _device_case(api, torch.float64)
        _, ws = _call(fn, _on_device(args))
        caught_ops += [str(w.message).split(":")[0] for w in ws]
    assert caught_ops == ["trnsparse spmm", "trnsparse bsr_spmm", "trnsparse screened_spmm"]


def test_fp64_on_device_result_follows_the_dense_operand(monkeypatch):
    # Only the sparse container on the device: the dense operand (and so the
    # result, as trnsparse places it) is on the host.
    _, tr = _fake_neuron(monkeypatch)
    fn, (A, B) = _device_case("bsr_spmm", torch.float64)
    out, ws = _call(fn, (_neuron_mat(A), B))
    assert len(ws) == 1 and torch.float64 in tr.to_host
    assert not _is_neuron(out) and tr.back == []
    torch.testing.assert_close(out, _host_ref(monkeypatch, fn, (A, B)), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("api,dtype", DEVICE_CASES, ids=DEVICE_IDS)
def test_fp64_on_device_override_keeps_device_aten(monkeypatch, api, dtype):
    log, tr = _fake_neuron(monkeypatch, on_device=True)
    fn, args = _device_case(api, dtype)
    ref = _host_ref(monkeypatch, fn, args)
    out, ws = _call(fn, _on_device(args))
    assert ws == [] and tr.to_host == [] and tr.back == []  # no host round trip
    assert log.calls == []  # still torch (device aten), not a kernel
    assert out.dtype == dtype and out.shape == ref.shape


@pytest.mark.parametrize("value,expect", [("1", True), ("true", True), ("", False), ("0", False)])
def test_fp64_on_device_env_flag(value, expect):
    import os
    import subprocess

    env = {**os.environ, "TRNSPARSE_FP64_ON_DEVICE": value}
    code = "from trnsparse.nki import dispatch; print(dispatch._FP64_ON_DEVICE)"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == str(expect)


@pytest.mark.parametrize("api,dtype", [("spmm", torch.float64), ("bsr_spmm", torch.complex128)])
def test_fp64_on_device_require_nki_still_raises(monkeypatch, api, dtype):
    log, tr = _fake_neuron(monkeypatch, require=True)
    fn, args = _device_case(api, dtype)
    with pytest.raises(NotImplementedError, match="float64|complex128"):
        fn(*_on_device(args))
    assert tr.to_host == [] and log.calls == []


def test_fp64_on_device_explicit_nki_unchanged(monkeypatch):
    log, tr = _fake_neuron(monkeypatch, backend="nki")
    fn, args = _device_case("spmm", torch.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*_on_device(args))
    assert tr.to_host == [] and not _host_warnings(caught)
    assert _policy_warnings(caught)  # the fp32-cast warning, as before
    assert log.calls and out.dtype == torch.float64


def test_fp64_on_device_pytorch_backend_unchanged(monkeypatch):
    _, tr = _fake_neuron(monkeypatch, backend="pytorch")
    fn, args = _device_case("bsr_spmm", torch.float64)
    _, ws = _call(fn, _on_device(args))
    assert ws == [] and tr.to_host == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp32_on_device_stays_on_device(monkeypatch, dtype):
    _, tr = _fake_neuron(monkeypatch)
    fn, args = _device_case("bsr_spmm", dtype)
    _, ws = _call(fn, _on_device(args))
    assert ws == [] and tr.to_host == []


def test_fp64_on_host_unchanged(monkeypatch):
    # The real predicate: CPU tensors are host tensors.
    _install_stubs(monkeypatch, "native-gather", "auto", False)
    monkeypatch.setattr(dispatch, "_fp64_host_warned", set())
    calls = []
    monkeypatch.setattr(dispatch, "_host_copy", lambda t: calls.append(t) or t)
    for api in DEVICE_APIS:
        fn, args = _device_case(api, torch.float64)
        _, ws = _call(fn, args)
        assert ws == [], api
    assert calls == []
