"""
Dispatch-path selection (native TorchNeuron vs XLA vs CPU), lazy
registration of the `trnsparse::` custom ops, their `register_autograd`
backward formulas, and host-structure / device-operand placement.

CPU-only: `torch_neuronx` and `torch_xla` are replaced with stand-ins via
`sys.modules` / `find_spec` monkeypatching, and the "neuron" device with
the CPU, so these run in the default `pytest -m "not neuron"` job on any
host.
"""

import importlib.util
import itertools
import re
import subprocess
import sys
import types
import warnings

import pytest
import torch

import trnsparse
import trnsparse.nki
from trnsparse.nki import dispatch, kernels

_REAL_FIND_SPEC = importlib.util.find_spec
_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")

needs_custom_op = pytest.mark.skipif(
    not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch >= 2.4"
)

KERNEL_OPS = {
    "spmm_dense",
    "bsr_spmm",
    "screened_spmm",
    "attn_stats",
    "attn_out",
    "attn_bwd_dq",
    "attn_bwd_dkdv",
}
# Static-structure kernel ops: structure passed as int lists (compile-time
# constants), launched directly rather than through `_launch` / `_run`.
STATIC_OPS = {"bsr_spmm_static", "attn_fwd_static", "attn_bwd_static", "bsr_spmm_static_t"}
# Keyed variants: the structure is one registered int key (kernels.register_structure).
KEYED_OPS = {"bsr_spmm_keyed", "bsr_spmm_t_keyed", "attn_fwd_keyed", "cg_bsr_keyed"}
STATIC_OPS |= KEYED_OPS
AUTOGRAD_OPS = {
    "spmm_dense",
    "bsr_spmm",
    "screened_spmm",
    "bsr_spmm_static",
    "bsr_spmm_keyed",
    "bsr_spmm_t_keyed",
}
COMPOSITE_OPS = {"bsr_attn", "bsr_attn_bwd"}
P = 128


@pytest.fixture(autouse=True)
def reset_dispatch_state(monkeypatch):
    """Each test sees an undetected path and an unregistered op table."""
    monkeypatch.setattr(dispatch, "_dispatch_path", None)
    monkeypatch.setattr(dispatch, "_native_ops", None)
    monkeypatch.setattr(dispatch, "_xla_deprecation_warned", False)
    monkeypatch.delenv("TRNSCI_NKI_DISPATCH", raising=False)
    monkeypatch.delenv("TRNSCI_XLA_DEPRECATION", raising=False)
    # Drop any `_native` left by an earlier test so registration re-runs.
    # (`_native_attn` registers real ops once per process and stays.)
    monkeypatch.delitem(sys.modules, "trnsparse.nki._native", raising=False)
    monkeypatch.delattr(trnsparse.nki, "_native", raising=False)
    yield


class _RecordedOp:
    """What the stand-in `nki_op` returns: callable, records register_autograd."""

    def __init__(self, fn, recorded, name):
        self._fn = fn
        self._recorded = recorded
        self._name = name

    def __call__(self, *args):
        return self._fn(*args)

    def register_autograd(self, backward, setup_context=None):
        self._recorded[self._name]["autograd"] = (backward, setup_context)


def _fake_native_torch_neuronx(device_count=1, recorded=None):
    """Stand-in for the TorchNeuron `torch_neuronx` (exports `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")
    rec = {} if recorded is None else recorded

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        def dec(f):
            rec[name] = {"fn": f, "mutates": mutates_args, "fake": fake_fn, "autograd": None}
            return _RecordedOp(f, rec, name)

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: device_count
    return mod


_NS = itertools.count()


def _real_custom_op_torch_neuronx():
    """Stand-in whose `nki_op` registers a real custom op (fresh namespace).

    Mirrors torch_neuronx.nki_op: `torch.library.custom_op` with an inferred
    schema plus `register_fake`, returning the CustomOpDef.
    """
    mod = types.ModuleType("torch_neuronx")
    ns = f"trnsparse_t{next(_NS)}"

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        def dec(f):
            op = torch.library.custom_op(ns + "::" + name.split("::")[1], f, mutates_args=())
            op.register_fake(fake_fn if fake_fn is not None else f)
            return op

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: 1
    return mod


def _xla_torch_neuronx():
    """Stand-in for the XLA-lineage `torch_neuronx` (no `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")
    mod.trace = lambda *a, **k: None
    return mod


def _set_torch_xla_present(monkeypatch, present: bool):
    def find_spec(name, *args, **kwargs):
        if name == "torch_xla":
            return object() if present else None
        return _REAL_FIND_SPEC(name, *args, **kwargs)

    monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)


def _accept_neuron_device(monkeypatch):
    # Off-hardware, "neuron" is not a registered torch device type.
    real_device = torch.device

    def device(spec, *args, **kwargs):
        if spec == dispatch._NEURON_DEVICE:
            return real_device("cpu")
        return real_device(spec, *args, **kwargs)

    monkeypatch.setattr(dispatch.torch, "device", device)


# Torch references for the kernels, used when a test runs the native path
# on CPU with real (test-namespace) custom ops.
def _ref_spmm_dense(a, b):
    return a @ b


def _ref_bsr_spmm(blocks_pad, b_gathered):
    M_tiles = blocks_pad.shape[0]
    out = torch.einsum("mkij,mkjn->min", blocks_pad, b_gathered)
    return out.reshape(M_tiles * blocks_pad.shape[2], b_gathered.shape[-1])


def _ref_bsr_spmm_static(blocks, b, row_ptrs, cols):
    bs = blocks.shape[-1]
    out = b.new_zeros(((len(row_ptrs) - 1) * bs, b.shape[1]))
    for m in range(len(row_ptrs) - 1):
        for idx in range(row_ptrs[m], row_ptrs[m + 1]):
            j = cols[idx]
            out[m * bs : (m + 1) * bs] += blocks[idx] @ b[j * bs : (j + 1) * bs]
    return out


def _ref_attn_fwd_static(q_t, k_t, v, row_ptrs, cols):
    seq = v.shape[0]
    out = v.new_zeros(v.shape)
    rmax = v.new_zeros((seq, 1))
    rden = v.new_ones((seq, 1))
    for m in range(len(row_ptrs) - 1):
        js = cols[row_ptrs[m] : row_ptrs[m + 1]]
        if not js:
            continue
        rows = slice(m * P, (m + 1) * P)
        kidx = torch.cat([torch.arange(j * P, (j + 1) * P) for j in js])
        s = q_t[:, rows].T @ k_t[:, kidx]
        mx = s.max(dim=1, keepdim=True).values
        e = torch.exp(s - mx)
        den = e.sum(dim=1, keepdim=True)
        out[rows] = (e @ v[kidx]) / den
        rmax[rows], rden[rows] = mx, den
    return out, rmax, rden


def _ref_bsr_spmm_static_t(blocks_t, b, row_ptrs, cols):
    bs = blocks_t.shape[0]
    blocks = blocks_t.reshape(bs, -1, bs).permute(1, 2, 0)  # back to (n_blocks, bs, bs)
    return _ref_bsr_spmm_static(blocks, b, row_ptrs, cols)


def _ref_bsr_spmm_keyed(blocks, b, key):
    return _ref_bsr_spmm_static(blocks, b, *kernels._STRUCTURES[key])


def _ref_bsr_spmm_t_keyed(blocks_t, b, key):
    return _ref_bsr_spmm_static_t(blocks_t, b, *kernels._STRUCTURES[key])


def _ref_attn_fwd_keyed(q_t, k_t, v, key):
    return _ref_attn_fwd_static(q_t, k_t, v, *kernels._STRUCTURES[key])


def _ref_attn_bwd_static(
    q_t, k_t, v_t, do_t, q_s, k, do, rm_col, rinv_col, d_col, rm_row, rinv_row, d_row,
    row_ptrs, cols, col_ptrs, rows, scale,
):  # fmt: skip
    dQ, dK, dV = torch.zeros_like(q_s), torch.zeros_like(q_s), torch.zeros_like(q_s)
    for m in range(len(row_ptrs) - 1):
        r = slice(m * P, (m + 1) * P)
        for idx in range(row_ptrs[m], row_ptrs[m + 1]):
            c = slice(cols[idx] * P, (cols[idx] + 1) * P)
            p = torch.exp(q_s[r] @ k[c].T - rm_col[r]) * rinv_col[r]
            ds = p * (do[r] @ v_t[:, c] - d_col[r])
            dQ[r] += scale * ds @ k[c]
            dK[c] += ds.T @ q_s[r]
            dV[c] += p.T @ do[r]
    return dQ, dK, dV


def _ref_screened_spmm(a, q, threshold_sqrt, b):
    mask = (q * q.reshape(1, -1)) > threshold_sqrt.reshape(1, 1)
    return (a * mask.to(a.dtype)) @ b


@pytest.fixture
def cpu_native(monkeypatch):
    """Run the native path on CPU: real custom ops backed by torch references."""
    if not _HAS_CUSTOM_OP:
        pytest.skip("torch.library.custom_op needs torch >= 2.4")
    monkeypatch.setitem(sys.modules, "torch_neuronx", _real_custom_op_torch_neuronx())
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    monkeypatch.setattr(dispatch, "_backend", "nki")
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setattr(dispatch, "_to_xla", _no_xla)
    for name, fn in (
        ("_spmm_dense_kernel", _ref_spmm_dense),
        ("_bsr_spmm_kernel", _ref_bsr_spmm),
        ("_screened_spmm_kernel", _ref_screened_spmm),
        ("_bsr_spmm_static_kernel", _ref_bsr_spmm_static),
        ("_attn_fwd_static_kernel", _ref_attn_fwd_static),
        ("_attn_bwd_static_kernel", _ref_attn_bwd_static),
        ("_bsr_spmm_static_t_kernel", _ref_bsr_spmm_static_t),
        ("_bsr_spmm_keyed_kernel", _ref_bsr_spmm_keyed),
        ("_bsr_spmm_t_keyed_kernel", _ref_bsr_spmm_t_keyed),
        ("_attn_fwd_keyed_kernel", _ref_attn_fwd_keyed),
    ):
        # `kernels` for the op bodies; `dispatch` because the host wrappers
        # name the kernel (only imported there when nki is installed).
        monkeypatch.setattr(kernels, name, fn, raising=False)
        monkeypatch.setattr(dispatch, name, fn, raising=False)
    yield


def _no_xla(*t):
    raise AssertionError("_to_xla must not run on the native path")


class TestDispatchPathSelection:
    def test_native_when_nki_op_and_device(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "native"

    def test_native_probe_does_not_touch_torch_xla(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch_xla", None)

        def find_spec(name, *args, **kwargs):
            assert name != "torch_xla", "native probe must not look up torch_xla"
            return _REAL_FIND_SPEC(name, *args, **kwargs)

        monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)
        assert dispatch._get_dispatch_path() == "native"

    def test_xla_when_torch_neuronx_has_no_nki_op(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_xla_when_native_has_no_device(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(device_count=0)
        )
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_not_native_when_neuron_device_type_unregistered(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_not_native_when_device_count_raises(self, monkeypatch):
        mod = _fake_native_torch_neuronx()

        def boom():
            raise RuntimeError("no runtime")

        mod.device_count = boom
        monkeypatch.setitem(sys.modules, "torch_neuronx", mod)
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_cpu_when_nothing_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", None)  # import raises
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_detection_is_cached(self, monkeypatch):
        calls = []

        def detect():
            calls.append(1)
            return "cpu"

        monkeypatch.setattr(dispatch, "_detect_dispatch_path", detect)
        assert dispatch._get_dispatch_path() == "cpu"
        assert dispatch._get_dispatch_path() == "cpu"
        assert len(calls) == 1

    def test_detection_is_lazy(self):
        # Fresh interpreter: importing trnsparse must not run detection.
        code = (
            "import trnsparse; from trnsparse.nki import dispatch; "
            "assert dispatch._dispatch_path is None"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


class TestDispatchOverride:
    """TRNSCI_NKI_DISPATCH=auto|native|xla|cpu (suite-wide) pins the path."""

    def _both_lineages(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)

    def test_auto_prefers_native(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "auto")
        assert dispatch._get_dispatch_path() == "native"

    def test_force_xla_when_both_available(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "XLA")
        assert dispatch._get_dispatch_path() == "xla"

    def test_force_cpu_when_both_available(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "cpu")
        assert dispatch._get_dispatch_path() == "cpu"

    def test_force_native_without_backend_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "native")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=native"):
            dispatch._get_dispatch_path()

    def test_force_xla_without_torch_xla_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "xla")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=xla"):
            dispatch._get_dispatch_path()

    def test_unknown_value_raises(self, monkeypatch):
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "cuda")
        with pytest.raises(ValueError, match="auto, native, xla, or cpu"):
            dispatch._get_dispatch_path()


class TestXlaDeprecationHook:
    def _xla(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)

    def test_off_by_default(self, monkeypatch):
        self._xla(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert dispatch._get_dispatch_path() == "xla"
            assert dispatch._get_dispatch_path() == "xla"

    def test_fires_once_when_enabled(self, monkeypatch):
        self._xla(monkeypatch)
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dispatch._get_dispatch_path()
            dispatch._get_dispatch_path()
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert str(dep[0].message) == (
            "the PyTorch/XLA path is deprecated on Trn2/Trn3; use TorchNeuron (native). "
            "Trn1/Inf2 remain supported in maintenance mode."
        )

    def test_silent_on_native_path(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert dispatch._get_dispatch_path() == "native"


class TestLaunchRouting:
    def test_native_launch_uses_custom_op_not_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        seen = {}

        def op(a, b):
            seen["args"] = (a, b)
            return a + b

        monkeypatch.setattr(dispatch, "_native_ops", {"fake": op})
        monkeypatch.setattr(dispatch, "_to_xla", _no_xla)

        def no_kernel(*t):
            raise AssertionError("raw kernel must not be called on the native path")

        out = dispatch._launch(no_kernel, "fake", torch.ones(2, 1), torch.ones(2, 1))
        assert torch.equal(out, torch.full((2, 1), 2.0))
        assert out.device.type == "cpu"

    def test_native_launch_returns_tuple_outputs(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        monkeypatch.setattr(dispatch, "_native_ops", {"pair": lambda a: (a * 2, a * 3)})
        monkeypatch.setattr(dispatch, "_to_xla", _no_xla)
        out = dispatch._launch(None, "pair", torch.ones(3, 1))
        assert isinstance(out, tuple) and len(out) == 2
        assert torch.equal(out[1], torch.full((3, 1), 3.0))

    def test_xla_launch_calls_kernel_via_to_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        moved = []

        def fake_to_xla(*tensors):
            moved.append(len(tensors))
            return list(tensors), tensors[0].device

        monkeypatch.setattr(dispatch, "_to_xla", fake_to_xla)

        def no_ops():
            raise AssertionError("native ops must not load on the XLA path")

        monkeypatch.setattr(dispatch, "_get_native_ops", no_ops)
        out = dispatch._launch(lambda x: x * 3, "unused", torch.ones(4, 1))
        assert moved == [1]
        assert torch.equal(out, torch.full((4, 1), 3.0))

    def test_cpu_launch_keeps_pre_native_behaviour(self, monkeypatch):
        # No device path: the launch goes through `_to_xla`, which raises
        # when torch_xla is absent — the same failure as before.
        monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")
        monkeypatch.setitem(sys.modules, "torch_xla", None)
        with pytest.raises(ImportError):
            dispatch._launch(lambda x: x, "unused", torch.ones(1, 1))

    def test_every_kernel_launch_site_uses_a_registered_op_name(self):
        # Each `_launch` / `_run` call in dispatch.py names an op in KERNEL_OPS.
        import inspect
        import re

        src = inspect.getsource(dispatch)
        names = set(re.findall(r'(?:_launch|_run)\(\s*_\w+_kernel,\s*"(\w+)"', src))
        assert names == KERNEL_OPS


class TestLazyRegistration:
    def test_import_does_not_register_ops(self):
        # Fresh interpreter: importing trnsparse (and the dispatch module)
        # must not load `_native` / `_native_attn` or import torch_neuronx /
        # torch_xla.
        code = (
            "import sys, trnsparse, trnsparse.nki.dispatch; "
            "bad = [m for m in ('trnsparse.nki._native', 'trnsparse.nki._native_attn', "
            "'torch_neuronx', 'torch_xla') if m in sys.modules]; "
            "assert not bad, bad"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    @needs_custom_op
    def test_ops_registered_on_first_native_use(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        assert "trnsparse.nki._native" not in sys.modules
        assert recorded == {}

        ops = dispatch._get_native_ops()

        assert set(ops) == KERNEL_OPS | STATIC_OPS | COMPOSITE_OPS
        assert set(recorded) == {f"trnsparse::{name}" for name in KERNEL_OPS | STATIC_OPS}
        assert all(r["mutates"] == {} for r in recorded.values())
        assert all(r["fake"] is not None for r in recorded.values())
        with_autograd = {n.split("::")[1] for n, r in recorded.items() if r["autograd"]}
        assert with_autograd == AUTOGRAD_OPS
        assert hasattr(torch.ops.trnsparse, "bsr_attn")
        assert hasattr(torch.ops.trnsparse, "bsr_attn_bwd")

        # Second call reuses the table; no re-registration.
        recorded.clear()
        assert dispatch._get_native_ops() is ops
        assert recorded == {}

    @needs_custom_op
    def test_op_signatures_build_a_schema(self, monkeypatch):
        # nki_op derives the schema with torch.library.infer_schema; check
        # every wrapper is fully annotated (Tensor args, Tensor returns).
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        dispatch._get_native_ops()
        assert len(recorded) == len(KERNEL_OPS | STATIC_OPS)
        for name, r in recorded.items():
            schema = torch.library.infer_schema(r["fn"], mutates_args=r["mutates"])
            ret = schema.split("->")[1].strip()
            assert ret in ("Tensor", "(Tensor, Tensor)", "(Tensor, Tensor, Tensor)"), (name, schema)
            if name.split("::")[1] in KEYED_OPS:
                assert re.search(r"(Sym)?[Ii]nt key[,)]", schema), (name, schema)
            elif name.split("::")[1] in STATIC_OPS:
                # list[int] infers as SymInt[] (int[] on older torch).
                assert re.search(r"(Sym)?[Ii]nt\[\] row_ptrs, (Sym)?[Ii]nt\[\] cols", schema), (
                    name,
                    schema,
                )
            assert "Tensor" in schema.split("->")[0], (name, schema)

    @needs_custom_op
    def test_fake_fns_give_kernel_output_shapes(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        dispatch._get_native_ops()

        def fake(name):
            return recorded[f"trnsparse::{name}"]["fake"]

        def m(*shape):
            return torch.empty(*shape, device="meta")

        assert fake("spmm_dense")(m(256, 384), m(384, 64)).shape == (256, 64)
        assert fake("bsr_spmm")(m(3, 2, P, P), m(3, 2, P, 96)).shape == (3 * P, 96)
        assert fake("screened_spmm")(m(256, 256), m(256, 1), m(1, 1), m(256, 32)).shape == (
            256,
            32,
        )
        tmax, tsum = fake("attn_stats")(m(2, P, 64), m(2, 3, P, 64))
        assert tmax.shape == tsum.shape == (2, 3, P, 1)
        args = (m(2, P, 64), m(2, 3, P, 64), m(2, 3, P, 64), m(2, P, 1), m(2, P, 1))
        assert fake("attn_out")(*args).shape == (2 * P, 64)
        dq_args = (m(2, P, 64), m(2, 3, P, 64), m(2, 3, P, 64), m(2, 3, P, 64)) + (m(2, P, 1),) * 3
        assert fake("attn_bwd_dq")(*dq_args).shape == (2 * P, 64)
        dk_args = (m(2, P, 64), m(2, P, 64), m(2, 4, P, 64), m(2, 4, P, 64)) + (m(2, 4, P, 1),) * 3
        dK, dV = fake("attn_bwd_dkdv")(*dk_args)
        assert dK.shape == dV.shape == (2 * P, 64)
        assert fake("bsr_spmm_static")(m(3, P, P), m(2 * P, 96), [0, 1, 1, 3], [0, 0, 1]).shape == (
            3 * P,
            96,
        )
        o, rm, rd = fake("attn_fwd_static")(
            m(64, 2 * P), m(64, 2 * P), m(2 * P, 64), [0, 1, 3], [0, 0, 1]
        )
        assert o.shape == (2 * P, 64) and rm.shape == rd.shape == (2 * P, 1)
        # fused CG: T = 3 block rows, R = 4 right-hand sides
        key = kernels.register_structure([0, 1, 2, 3], [0, 1, 2])
        out = fake("cg_bsr_keyed")(m(P, 3 * P), m(P, P), m(P, 2 * 3 * 4 + 2), key, 8, 0, 160, 1)
        assert out.shape == (P, 3 * 4 + 3 * 4 + 1)


class _Ctx:
    def __init__(self, tensors, needs):
        self.saved_tensors = tensors
        self.needs_input_grad = needs


@needs_custom_op
class TestRegisteredBackwardFormulas:
    """The register_autograd backwards match torch.autograd on references."""

    @pytest.fixture
    def native(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        dispatch._get_native_ops()
        return sys.modules["trnsparse.nki._native"]

    def _check(self, backward, ref, inputs, needs):
        leaves = [t.clone().requires_grad_(n) for t, n in zip(inputs, needs, strict=True)]
        out = ref(*leaves)
        g = torch.randn_like(out)
        out.backward(g)
        got = backward(_Ctx(tuple(inputs), needs), g)
        for leaf, grad, n in zip(leaves, got, needs, strict=True):
            if n:
                torch.testing.assert_close(grad, leaf.grad)
            else:
                assert grad is None

    def test_spmm_dense(self, native):
        torch.manual_seed(0)
        inputs = (torch.randn(256, 128), torch.randn(128, 64))
        self._check(native._spmm_dense_backward, _ref_spmm_dense, inputs, (True, True))
        self._check(native._spmm_dense_backward, _ref_spmm_dense, inputs, (False, True))

    def test_bsr_spmm(self, native):
        torch.manual_seed(1)
        inputs = (torch.randn(3, 2, P, P), torch.randn(3, 2, P, 40))
        self._check(native._bsr_spmm_backward, _ref_bsr_spmm, inputs, (True, True))

    def test_bsr_spmm_static(self, native):
        torch.manual_seed(3)
        row_ptrs, cols = [0, 2, 2, 3], [0, 2, 2]  # empty middle row, column 2 reused
        blocks, b = torch.randn(3, P, P), torch.randn(3 * P, 24)
        for needs in ((True, True), (False, True), (True, False)):
            leaves = [t.clone().requires_grad_(n) for t, n in zip((blocks, b), needs, strict=True)]
            out = _ref_bsr_spmm_static(*leaves, row_ptrs, cols)
            g = torch.randn_like(out)
            out.backward(g)
            ctx = _Ctx((blocks, b), needs + (False, False))
            ctx.row_ptrs, ctx.cols = row_ptrs, cols
            got = native._bsr_spmm_static_backward(ctx, g)
            assert got[2] is None and got[3] is None
            for leaf, grad, n in zip(leaves, got[:2], needs, strict=True):
                if n:
                    torch.testing.assert_close(grad, leaf.grad, atol=1e-4, rtol=1e-4)
                else:
                    assert grad is None

    def test_bsr_spmm_t_keyed(self, native):
        # Pre-transposed layout: grads come back in the (bs, n_blocks * bs) layout.
        torch.manual_seed(4)
        row_ptrs, cols = [0, 2, 2, 3], [0, 2, 2]
        key = kernels.register_structure(row_ptrs, cols)
        blocks_t, b = torch.randn(P, 3 * P), torch.randn(3 * P, 24)
        leaves = [t.clone().requires_grad_() for t in (blocks_t, b)]
        out = _ref_bsr_spmm_t_keyed(*leaves, key)
        g = torch.randn_like(out)
        out.backward(g)
        ctx = _Ctx((), (True, True, False))
        ctx.save_for_backward = lambda *t: setattr(ctx, "saved_tensors", t)
        native._save_t_keyed(ctx, (blocks_t, b, key), out)
        got = native._bsr_spmm_t_keyed_backward(ctx, g)
        assert got[2] is None
        for leaf, grad in zip(leaves, got[:2], strict=True):
            torch.testing.assert_close(grad, leaf.grad, atol=1e-4, rtol=1e-4)

    def test_screened_spmm(self, native):
        torch.manual_seed(2)
        q = torch.rand(128, 1)
        inputs = (torch.randn(128, 128), q, torch.tensor([[0.3]]), torch.randn(128, 16))
        self._check(
            native._screened_spmm_backward,
            _ref_screened_spmm,
            inputs,
            (True, False, False, True),
        )


def _random_csr(M, K, density, seed=0):
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(M, K, generator=g) * (torch.rand(M, K, generator=g) < density)
    return dense, trnsparse.from_dense(dense)


class TestHostCsrPlacement:
    def test_host_densify_matches_to_dense(self):
        dense, A = _random_csr(37, 53, 0.2, seed=3)
        from trnsparse.formats import _csr_to_dense_host

        torch.testing.assert_close(_csr_to_dense_host(A), A.to_dense())
        torch.testing.assert_close(_csr_to_dense_host(A), dense)

    def test_host_densify_is_differentiable(self):
        from trnsparse.formats import _csr_to_dense_host

        _, A = _random_csr(16, 16, 0.3, seed=4)
        A.values.requires_grad_(True)
        _csr_to_dense_host(A).sum().backward()
        assert torch.equal(A.values.grad, torch.ones_like(A.values))

    def test_host_densify_empty_rows(self):
        from trnsparse.formats import _csr_to_dense_host

        dense = torch.zeros(5, 4)
        dense[1, 2] = 1.0
        dense[4, 0] = 2.0
        torch.testing.assert_close(_csr_to_dense_host(trnsparse.from_dense(dense)), dense)

    def test_spmv_spmm_accept_dense_operand_on_another_device(self):
        # "meta" stands in for "neuron": a host CSR against a device operand
        # takes the densify-on-host path and returns on the operand's device.
        _, A = _random_csr(20, 30, 0.2, seed=5)
        x = torch.empty(30, device="meta")
        B = torch.empty(30, 4, device="meta")
        prev = trnsparse.get_backend()
        trnsparse.set_backend("pytorch")
        try:
            y = trnsparse.spmv(A, x)
            C = trnsparse.spmm(A, B)
        finally:
            trnsparse.set_backend(prev)
        assert y.device.type == "meta" and y.shape == (20,)
        assert C.device.type == "meta" and C.shape == (20, 4)

    def test_spmv_symmetric_dense_branch_matches_sparse(self):
        g = torch.Generator().manual_seed(6)
        S = torch.randn(12, 12, generator=g)
        A = trnsparse.from_dense(torch.triu(S + S.T))
        x = torch.randn(12, generator=g)
        from trnsparse.ops import _spmv_symmetric_dense

        torch.testing.assert_close(
            _spmv_symmetric_dense(A, x, 2.0), trnsparse.spmv_symmetric(A, x, alpha=2.0)
        )
        got = trnsparse.spmv_symmetric(A, x.to("meta"))
        assert got.device.type == "meta" and got.shape == (12,)


class TestNativePathOnCpu:
    """The native wrappers end to end on CPU, with real custom ops whose
    bodies are torch references (no NKI): routing, host CSR structure, and
    gradients through the register_autograd backwards."""

    def test_spmm_host_csr_forward_and_grad(self, cpu_native):
        dense, A = _random_csr(200, 137, 0.05, seed=7)
        B = torch.randn(137, 64, requires_grad=True)
        A.values.requires_grad_(True)
        C = trnsparse.spmm(A, B)
        torch.testing.assert_close(C, dense @ B.detach(), atol=1e-4, rtol=1e-4)
        C.sum().backward()

        dense_ref = dense.clone().requires_grad_(True)
        B_ref = B.detach().clone().requires_grad_(True)
        (dense_ref @ B_ref).sum().backward()
        torch.testing.assert_close(B.grad, B_ref.grad)
        rows = torch.repeat_interleave(torch.arange(200), A.row_ptrs[1:] - A.row_ptrs[:-1])
        torch.testing.assert_close(A.values.grad, dense_ref.grad[rows, A.col_indices])

    def test_spmm_host_csr_traces_fullgraph(self, cpu_native):
        # CPU stand-in for the neuron compile test: the native spmm path
        # (host densify without sparse tensors, pad, op, slice) is one graph,
        # with the host CSR as a closure or as graph inputs, and the
        # registered backward traces too.
        dense, A = _random_csr(256, 256, 0.05, seed=13)
        B = torch.randn(256, 64)
        f = torch.compile(lambda B: trnsparse.spmm(A, B), backend="eager", fullgraph=True)
        torch.testing.assert_close(f(B), dense @ B, atol=1e-4, rtol=1e-4)

        def from_parts(v, ci, rp, B):
            return trnsparse.spmm(trnsparse.CSRMatrix(v, ci, rp, (256, 256)), B)

        f2 = torch.compile(from_parts, backend="eager", fullgraph=True)
        out = f2(A.values, A.col_indices, A.row_ptrs, B)
        torch.testing.assert_close(out, dense @ B, atol=1e-4, rtol=1e-4)

        Bg = B.clone().requires_grad_(True)
        f3 = torch.compile(
            lambda B: trnsparse.spmm(A, B).sum(), backend="aot_eager", fullgraph=True
        )
        f3(Bg).backward()
        torch.testing.assert_close(Bg.grad, dense.T @ torch.ones(256, 64), atol=1e-4, rtol=1e-4)

    def test_spmm_skips_autograd_function(self, cpu_native, monkeypatch):
        def boom(*a):
            raise AssertionError("_SpMMFunction must not run on the native path")

        monkeypatch.setattr(dispatch._SpMMFunction, "apply", boom)
        _, A = _random_csr(128, 128, 0.1, seed=8)
        trnsparse.spmm(A, torch.randn(128, 8))

    def test_bsr_spmm_forward_and_grad(self, cpu_native, monkeypatch):
        def boom(*a):
            raise AssertionError("_BSRSpMMFunction must not run on the native path")

        monkeypatch.setattr(dispatch._BSRSpMMFunction, "apply", boom)
        g = torch.Generator().manual_seed(9)
        dense = torch.randn(256, 384, generator=g)
        dense[:128, 128:256] = 0
        dense[128:, :256] = 0  # block row 1 keeps one block: exercises a padded k-slot
        A = trnsparse.BSRMatrix.from_dense(dense, block_size=P)
        A.blocks.requires_grad_(True)
        B = torch.randn(384, 32, generator=g, requires_grad=True)
        C = trnsparse.bsr_spmm(A, B)
        torch.testing.assert_close(C, A.to_dense().detach() @ B.detach(), atol=1e-4, rtol=1e-4)
        C.square().sum().backward()

        blocks_ref = A.blocks.detach().clone().requires_grad_(True)
        B_ref = B.detach().clone().requires_grad_(True)
        ref = trnsparse.BSRMatrix(
            blocks_ref, A.block_col_indices, A.block_row_ptrs, A.shape, A.block_size
        )
        from trnsparse.ops import _bsr_spmm_pytorch

        _bsr_spmm_pytorch(ref, B_ref).square().sum().backward()
        torch.testing.assert_close(A.blocks.grad, blocks_ref.grad, atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(B.grad, B_ref.grad, atol=1e-3, rtol=1e-4)

    def test_bsr_spmm_static_is_default_and_gather_path_still_works(self, cpu_native, monkeypatch):
        g = torch.Generator().manual_seed(11)
        dense = torch.randn(256, 384, generator=g)
        dense[:128, 128:] = 0
        A = trnsparse.BSRMatrix.from_dense(dense, block_size=P)
        B = torch.randn(384, 600, generator=g)  # > 512 columns: padded to 1024
        calls = []
        real = dispatch._nki_bsr_spmm_static
        monkeypatch.setattr(
            dispatch, "_nki_bsr_spmm_static", lambda *a: calls.append(1) or real(*a)
        )
        C = trnsparse.bsr_spmm(A, B)
        assert calls == [1]
        torch.testing.assert_close(C, dense @ B, atol=1e-3, rtol=1e-4)
        # Disabled (or over the unroll cap): the gather path, same result.
        monkeypatch.setattr(dispatch, "_STATIC_STRUCTURE", False)
        torch.testing.assert_close(trnsparse.bsr_spmm(A, B), C, atol=1e-3, rtol=1e-4)
        assert calls == [1]

    @pytest.mark.parametrize("duplicates", [False, True])
    def test_csr_densify_on_device_matches_host(self, monkeypatch, duplicates):
        dense, A = _random_csr(37, 53, 0.05, seed=4)
        if duplicates:  # repeat each row's first entry: must be summed
            v, ci, rp = [], [], [0]
            for i in range(37):
                lo, hi = A.row_ptrs[i].item(), A.row_ptrs[i + 1].item()
                if hi > lo:
                    v += [A.values[lo].item()]
                    ci += [A.col_indices[lo].item()]
                v += A.values[lo:hi].tolist()
                ci += A.col_indices[lo:hi].tolist()
                rp.append(len(v))
            A = trnsparse.CSRMatrix(torch.tensor(v), torch.tensor(ci), torch.tensor(rp), A.shape)
            dense = A.to_dense()
        # Exercise the device branch on CPU: treat "cpu" as a device target.
        monkeypatch.setattr(
            dispatch.torch, "device", lambda d: types.SimpleNamespace(type="neuron")
        )
        A.values.requires_grad_(True)
        D = dispatch._csr_to_dense_on(A, "cpu")
        torch.testing.assert_close(D, dense)
        D.sum().backward()
        torch.testing.assert_close(A.values.grad, torch.ones_like(A.values))

    def test_attention_static_forward(self, cpu_native):
        from trnsparse.ops import _block_sparse_attn_pytorch

        torch.manual_seed(13)
        seq_len, hd = 4 * P, 64
        bm = torch.eye(4, dtype=torch.bool)
        bm[:, 0] = True
        bm[0, :] = True  # global-token row: 4 blocks, others 2
        pattern = bm.repeat_interleave(P, 0).repeat_interleave(P, 1).float()
        mask = trnsparse.BSRMatrix.from_dense(pattern, block_size=P)
        Q, K, V = (torch.randn(seq_len, hd) for _ in range(3))
        out, rm, rd = dispatch.nki_bsr_attn_tiled(Q, K, V, mask, return_stats=True)
        ref, rm_ref, rd_ref = _block_sparse_attn_pytorch(Q, K, V, mask, hd**-0.5, return_stats=True)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(rm, rm_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(rd, rd_ref, atol=1e-3, rtol=1e-4)

        from trnsparse.ops import _block_sparse_attn_backward

        dO = torch.randn(seq_len, hd)
        got = dispatch.nki_bsr_attn_bwd(Q, K, V, dO, out, mask, rm, rd)
        want = _block_sparse_attn_backward(
            Q, K, V, ref, dO, mask, hd**-0.5, row_max=rm_ref, row_denom=rd_ref
        )
        for g_, w_ in zip(got, want, strict=True):
            torch.testing.assert_close(g_, w_, atol=1e-4, rtol=1e-4)

    def test_static_kernels_bf16_operand_contract(self, cpu_native, monkeypatch):
        """bf16 inputs reach the static kernels as bf16 operands; statistics stay fp32."""
        seen = {}

        def recorder(name, ref, out_dtype_of):
            def run(*args):
                seen[name] = [a.dtype for a in args if isinstance(a, torch.Tensor)]
                f32 = [a.float() if isinstance(a, torch.Tensor) else a for a in args]
                res = ref(*f32)
                dt = out_dtype_of(args)
                if isinstance(res, tuple):
                    return (res[0].to(dt),) + tuple(
                        r.to(dt if name == "bwd" else torch.float32) for r in res[1:]
                    )
                return res.to(dt)

            return run

        for attr, name, ref, odt in (
            ("_bsr_spmm_keyed_kernel", "spmm", _ref_bsr_spmm_keyed, lambda a: a[1].dtype),
            ("_attn_fwd_keyed_kernel", "fwd", _ref_attn_fwd_keyed, lambda a: a[2].dtype),
            ("_attn_bwd_static_kernel", "bwd", _ref_attn_bwd_static, lambda a: a[4].dtype),
        ):
            fn = recorder(name, ref, odt)
            monkeypatch.setattr(kernels, attr, fn, raising=False)
            monkeypatch.setattr(dispatch, attr, fn, raising=False)

        bf = torch.bfloat16
        g = torch.Generator().manual_seed(21)
        dense = torch.randn(256, 256, generator=g)
        dense[:128, 128:] = 0
        A = trnsparse.BSRMatrix.from_dense(dense, block_size=P)
        A.blocks = A.blocks.to(bf)
        C = trnsparse.bsr_spmm(A, torch.randn(256, 64, generator=g).to(bf))
        assert C.dtype == bf and seen["spmm"] == [bf, bf]
        # Mixed precision runs in fp32 and returns the promoted dtype.
        C = trnsparse.bsr_spmm(A, torch.randn(256, 64, generator=g))
        assert C.dtype == torch.float32 and seen["spmm"] == [torch.float32] * 2

        seq_len, hd = 2 * P, 64
        mask = trnsparse.BSRMatrix.from_dense(torch.ones(seq_len, seq_len), block_size=P)
        Q, K, V, dO = (torch.randn(seq_len, hd, generator=g).to(bf) for _ in range(4))
        out, rm, rd = dispatch.nki_bsr_attn_tiled(Q, K, V, mask, return_stats=True)
        assert out.dtype == bf and seen["fwd"] == [bf] * 3
        assert rm.dtype == torch.float32 and rd.dtype == torch.float32
        o2, rm2, rd2 = torch.ops.trnsparse.bsr_attn(
            Q, K, V, mask.block_col_indices, mask.block_row_ptrs, P
        )
        assert o2.dtype == bf and rm2.dtype == torch.float32 and rd2.dtype == torch.float32
        dQ, dK, dV = dispatch.nki_bsr_attn_bwd(Q, K, V, dO, out, mask, rm, rd)
        assert seen["bwd"] == [bf] * 7 + [torch.float32] * 6
        assert {t.dtype for t in (dQ, dK, dV)} == {bf}

    def test_column_structure(self):
        assert dispatch._column_structure([0, 2, 2, 5], [0, 1, 0, 1, 2], 3) == (
            [0, 2, 4, 5],
            [0, 2, 0, 2, 2],
        )

    def test_screened_spmm_forward_and_grad(self, cpu_native, monkeypatch):
        def boom(*a):
            raise AssertionError("_ScreenedSpMMFunction must not run on the native path")

        monkeypatch.setattr(dispatch._ScreenedSpMMFunction, "apply", boom)
        g = torch.Generator().manual_seed(10)
        Am = torch.randn(150, 150, generator=g, requires_grad=True)
        diag = torch.rand(150, generator=g)
        B = torch.randn(150, 24, generator=g, requires_grad=True)
        C = trnsparse.screened_spmm(Am, diag, B, threshold=0.2)
        C.sum().backward()

        Q = diag.sqrt()
        mask = ((Q[:, None] * Q[None, :]) > 0.2**0.5).float()
        A_ref = Am.detach().clone().requires_grad_(True)
        B_ref = B.detach().clone().requires_grad_(True)
        ref = (A_ref * mask) @ B_ref
        torch.testing.assert_close(C, ref, atol=1e-4, rtol=1e-4)
        ref.sum().backward()
        torch.testing.assert_close(Am.grad, A_ref.grad)
        torch.testing.assert_close(B.grad, B_ref.grad, atol=1e-4, rtol=1e-4)


@needs_custom_op
class TestNativeAttentionOps:
    """`trnsparse::bsr_attn` / `bsr_attn_bwd` (register_autograd) on CPU,
    with the NKI forward/backward replaced by the PyTorch references."""

    @pytest.fixture
    def attn(self, monkeypatch):
        from trnsparse.nki import _native_attn  # registers once per process
        from trnsparse.ops import _block_sparse_attn_backward, _block_sparse_attn_pytorch

        def fwd(Q, K, V, mask, return_stats=False):
            return _block_sparse_attn_pytorch(
                Q, K, V, mask, Q.shape[1] ** -0.5, return_stats=return_stats
            )

        def bwd(Q, K, V, dO, O, mask, row_max, row_denom):
            return _block_sparse_attn_backward(
                Q, K, V, O, dO, mask, Q.shape[1] ** -0.5, row_max=row_max, row_denom=row_denom
            )

        monkeypatch.setattr(dispatch, "nki_bsr_attn_tiled", fwd)
        monkeypatch.setattr(dispatch, "nki_bsr_attn_bwd", bwd)
        return _native_attn

    def _mask(self, seq_len):
        nb = seq_len // P
        pattern = torch.zeros(seq_len, seq_len)
        for i in range(nb):
            for j in range(nb):
                if j <= i:  # block-causal
                    pattern[i * P : (i + 1) * P, j * P : (j + 1) * P] = 1.0
        return trnsparse.BSRMatrix.from_dense(pattern, block_size=P)

    def test_forward_and_backward_match_autograd_function(self, attn):
        torch.manual_seed(12)
        seq_len, hd = 3 * P, 32
        mask = self._mask(seq_len)
        Q, K, V = (torch.randn(seq_len, hd, requires_grad=True) for _ in range(3))
        out, row_max, row_denom = torch.ops.trnsparse.bsr_attn(
            Q, K, V, mask.block_col_indices, mask.block_row_ptrs, P
        )
        assert row_max.shape == row_denom.shape == (3, P)
        g = torch.randn_like(out)
        out.backward(g)

        Qr, Kr, Vr = (t.detach().clone().requires_grad_(True) for t in (Q, K, V))
        prev = trnsparse.get_backend()
        trnsparse.set_backend("pytorch")
        try:
            ref = trnsparse.block_sparse_attention_tiled(Qr, Kr, Vr, mask)
        finally:
            trnsparse.set_backend(prev)
        ref.backward(g)
        torch.testing.assert_close(out, ref)
        for got, exp in ((Q.grad, Qr.grad), (K.grad, Kr.grad), (V.grad, Vr.grad)):
            torch.testing.assert_close(got, exp)

    def test_fake_shapes(self, attn):
        m = torch.empty(2 * P, 64, device="meta")
        idx = torch.zeros(3, dtype=torch.long, device="meta")
        out, rm, rd = attn._bsr_attn_fake(m, m, m, idx, idx, P)
        assert out.shape == (2 * P, 64) and rm.shape == rd.shape == (2, P)

    def test_block_sparse_attention_tiled_routes_to_op(self, attn, monkeypatch):
        calls = []

        def op(*args):
            calls.append(args)
            return torch.ops.trnsparse.bsr_attn(*args)

        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
        monkeypatch.setattr(dispatch, "_backend", "nki")
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_native_ops", {"bsr_attn": op})
        mask = self._mask(2 * P)
        Q = torch.randn(2 * P, 64)
        out = trnsparse.block_sparse_attention_tiled(Q, Q, Q, mask)
        assert out.shape == Q.shape
        assert len(calls) == 1
        # Mask structure is passed as host tensors.
        assert calls[0][3].device.type == "cpu" and calls[0][4].device.type == "cpu"
