"""NKI dispatch for sparse operations."""
from .dispatch import HAS_NKI, set_backend, get_backend
__all__ = ["HAS_NKI", "set_backend", "get_backend"]
