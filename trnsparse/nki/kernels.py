"""NKI kernels for trnsparse.

v0.2.0 ships the "densify-then-GEMM" SpMM path: the host gathers the
CSR into a padded dense `(M, K)` tile, the NKI kernel runs stationary
tile reuse GEMM against `B`, and the dense result is returned.

This is correctness-first (Phase 1). Realistic sparse matrices pay a
materialization cost proportional to `M*K`; the row-bucketing path
(Phase 3 / #15) removes that intermediate. The benefit in v0.2.0 is
that the full Neuron toolchain — compile, load, execute, autograd — is
wired end-to-end, the kernel is hardware-validated, and the dispatch
surface matches what row-bucketing will hook into.

Kernel structure mirrors `trnblas/trnblas/nki/dispatch.py::_gemm_kernel`
(stationary A tile, streaming B, PSUM accumulation). Tile dimensions are
pinned to NKI 2.24 partition limits: TILE_M = TILE_K = 128, TILE_N = 512.
"""

from __future__ import annotations

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    HAS_NKI = True
except ImportError:
    HAS_NKI = False


_TILE_M = 128
_TILE_K = 128
_TILE_N = 512


if HAS_NKI:

    @nki.jit
    def _spmm_dense_kernel(a, b):
        """Densified SpMM: C = A @ B with stationary A-tile reuse.

        Caller guarantees A (the host-materialized dense view of the CSR)
        has M and K padded to multiples of 128 and B has K padded likewise
        and N either ≤ 512 or a multiple of 512. PSUM accumulates across
        K-tiles before the single store per (m, n) tile pair.

        NKI 2.24 calling convention (`nisa.nc_matmul`):
            stationary: (TILE_K, TILE_M)  partition=K ≤ 128, free ≤ 128
            moving:     (TILE_K, TILE_N)  partition=K, free ≤ 512
            psum:       (TILE_M, TILE_N)  fp32, in nl.psum
        """
        M, K = a.shape
        _, N = b.shape

        TILE_M = _TILE_M
        TILE_K = _TILE_K
        TILE_N = N if N <= _TILE_N else _TILE_N

        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N

                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K

                    a_t = nl.load_transpose2d(
                        a[m_off:m_off + TILE_M, k_off:k_off + TILE_K]
                    )
                    b_tile = nl.load(
                        b[k_off:k_off + TILE_K, n_off:n_off + TILE_N]
                    )

                    psum[...] += nisa.nc_matmul(a_t, b_tile)

                c_sbuf = nl.copy(psum, dtype=a.dtype)
                nl.store(
                    c[m_off:m_off + TILE_M, n_off:n_off + TILE_N],
                    value=c_sbuf,
                )

        return c
