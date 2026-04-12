"""
NKI dispatch for sparse operations.

SpMM is the primary NKI target: gather non-zero column indices into
dense tiles, run matmul on Tensor Engine, scatter results.

The gather/scatter pattern uses the DMA engine for indirect memory
access, while the matmul runs on the systolic array. This is the
same pattern used in sparse attention implementations.
"""

from __future__ import annotations

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

_backend = "auto"


def set_backend(backend: str):
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
