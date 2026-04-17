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
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False


_TILE_M = 128
_TILE_K = 128
_TILE_N = 512


if HAS_NKI:

    @nki.jit
    def _bsr_spmm_kernel(blocks_pad, b_gathered):
        """Block-sparse × dense matmul via per-block `nc_matmul`.

        Trainium-native SpMM (v0.3.0). Each nonzero block of A is already
        a dense 128×128 Tensor-Engine tile — no gather required. For each
        output block-row, accumulate `A_block @ B_slice` over the row's
        nonzero blocks in PSUM, then store.

        Caller guarantees uniform padding: every block row holds the same
        `K_max` blocks (zero-padded as needed). Host-side `bsr_spmm` does
        the B-slice gather and the zero-padding.

        Shapes:
            blocks_pad:  (M_tiles, K_max, 128, 128)
            b_gathered:  (M_tiles, K_max, 128, N)
        Returns:
            out:         (M_tiles * 128, N)
        """
        M_tiles, K_max, _, _ = blocks_pad.shape
        _, _, _, N = b_gathered.shape

        TILE_M = _TILE_M  # 128, fixed by BSR block_size
        TILE_N = N if N <= _TILE_N else _TILE_N

        out = nl.ndarray((M_tiles * TILE_M, N), dtype=blocks_pad.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M_tiles):
            for n in nl.affine_range(N // TILE_N):
                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K_max):
                    a_t = nl.load_transpose2d(blocks_pad[m, k, :, :])
                    b_tile = nl.load(b_gathered[m, k, :, n * TILE_N : (n + 1) * TILE_N])
                    psum[...] += nisa.nc_matmul(a_t, b_tile)

                c_sbuf = nl.copy(psum, dtype=blocks_pad.dtype)
                nl.store(
                    out[m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N],
                    value=c_sbuf,
                )

        return out

    @nki.jit
    def _screened_spmm_kernel(a, q, threshold_sqrt, b):
        """Fused Schwarz-screened dense matmul: `C = (A * mask) @ B`.

        mask[i,j] = `Q[i] * Q[j] > threshold_sqrt`, where Q is
        `sqrt(|diag_integrals|)` pre-computed on the host.

        Fuses: outer-product pair bound → threshold → mask-apply → nc_matmul
        into one kernel. Saves one mask-memory pass + one kernel dispatch
        vs the unfused flow.

        Caller guarantees: A is square (M, K) with M==K, padded to
        TILE_M=TILE_K=128. B has K padded likewise and N either ≤ 512
        or a multiple of 512. `q` is the 1-D Schwarz bounds of length M.
        `threshold_sqrt` is a 0-d fp32 tensor (scalar).
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

                # Row Q slice used for every k-tile in this (m, n) output tile.
                q_m = nl.load(q[m_off : m_off + TILE_M])  # (TILE_M,)

                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K

                    a_tile = nl.load(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    q_k = nl.load(q[k_off : k_off + TILE_K])  # (TILE_K,)

                    # Outer-product pair bound (TILE_M, TILE_K). nl broadcasting
                    # via explicit reshape — partition-dim-safe.
                    pair_bound = q_m.reshape((TILE_M, 1)) * q_k.reshape((1, TILE_K))
                    mask = nl.greater(pair_bound, threshold_sqrt)
                    a_masked = nl.multiply(a_tile, mask.astype(a.dtype))

                    # Transpose for stationary-A nc_matmul via a staging buffer.
                    # nl.load_transpose2d loads+transposes from HBM, but a_masked
                    # is already in SBUF, so we need to store-and-reload or use
                    # an in-SBUF transpose primitive. nl.transpose is available
                    # in NKI 0.3.0; if the simulator rejects, fall back to
                    # storing to an HBM staging tile and load_transpose2d-ing.
                    a_t = nl.transpose(a_masked)
                    b_tile = nl.load(b[k_off : k_off + TILE_K, n_off : n_off + TILE_N])

                    psum[...] += nisa.nc_matmul(a_t, b_tile)

                c_sbuf = nl.copy(psum, dtype=a.dtype)
                nl.store(
                    c[m_off : m_off + TILE_M, n_off : n_off + TILE_N],
                    value=c_sbuf,
                )

        return c

    @nki.jit
    def _attn_stats_kernel(q_scaled_blocks, k_gathered_pad):
        """Pass-1 attention kernel: per-block row-wise max and stable exp-sum.

        Each (m, ki) pair is independent — no carry between iterations.
        Q block is stationary per block-row: loaded once as a transposed
        (head_dim, 128) tile, reused across all ki blocks in that row.

        Shapes:
            q_scaled_blocks:  (M_tiles, 128, head_dim)  — Q * scale
            k_gathered_pad:   (M_tiles, K_max, 128, head_dim) — K at nonzero cols
        Returns:
            tile_max:         (M_tiles, K_max, 128) — per-block row-wise max
            tile_sumexp:      (M_tiles, K_max, 128) — per-block stable exp-sum
        """
        M_tiles, K_max, _, _ = k_gathered_pad.shape
        _, _, head_dim = q_scaled_blocks.shape

        tile_max = nl.ndarray(
            (M_tiles, K_max, _TILE_M), dtype=q_scaled_blocks.dtype, buffer=nl.shared_hbm
        )
        tile_sumexp = nl.ndarray(
            (M_tiles, K_max, _TILE_M), dtype=q_scaled_blocks.dtype, buffer=nl.shared_hbm
        )

        for m in nl.affine_range(M_tiles):
            q_t = nl.load_transpose2d(q_scaled_blocks[m, :, :])  # (head_dim, 128) stationary

            for ki in nl.affine_range(K_max):
                k_t = nl.load_transpose2d(k_gathered_pad[m, ki, :, :])  # (head_dim, 128)

                score_psum = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
                score_psum[...] += nisa.nc_matmul(q_t, k_t)  # Q @ K.T = (128, 128)
                score = nl.copy(score_psum, dtype=q_scaled_blocks.dtype)

                t_max = nl.max(score, axis=1)  # (128,)
                stable = score - t_max.reshape((_TILE_M, 1))
                t_sum = nl.sum(nl.exp(stable), axis=1)  # (128,)

                nl.store(tile_max[m, ki, :], value=t_max)
                nl.store(tile_sumexp[m, ki, :], value=t_sum)

        return tile_max, tile_sumexp

    @nki.jit
    def _attn_out_kernel(q_scaled_blocks, k_gathered_pad, v_gathered_pad, row_max, row_denom):
        """Pass-2 attention kernel: stable softmax weights × V accumulation.

        Recomputes scores from Q and K (same as pass 1) to avoid storing the
        full (M_tiles, K_max, 128, 128) score tensor. Uses row_max / row_denom
        from the host reduction to apply stable normalisation.

        Nested PSUM strategy:
          - score_psum (128, 128): initialised per (m, ki) block, drained to SBUF.
          - out_psum (128, head_dim): accumulates across all ki for block-row m.

        Shapes:
            q_scaled_blocks:  (M_tiles, 128, head_dim)
            k_gathered_pad:   (M_tiles, K_max, 128, head_dim)
            v_gathered_pad:   (M_tiles, K_max, 128, head_dim)
            row_max:          (M_tiles, 128)
            row_denom:        (M_tiles, 128)
        Returns:
            out:              (M_tiles * 128, head_dim)
        """
        M_tiles, K_max, _, head_dim = k_gathered_pad.shape

        out = nl.ndarray(
            (M_tiles * _TILE_M, head_dim), dtype=q_scaled_blocks.dtype, buffer=nl.shared_hbm
        )

        for m in nl.affine_range(M_tiles):
            q_t = nl.load_transpose2d(q_scaled_blocks[m, :, :])  # (head_dim, 128) stationary
            row_max_m = nl.load(row_max[m, :])  # (128,)
            row_denom_m = nl.load(row_denom[m, :])  # (128,)

            out_psum = nl.zeros((_TILE_M, head_dim), dtype=nl.float32, buffer=nl.psum)

            for ki in nl.affine_range(K_max):
                k_t = nl.load_transpose2d(k_gathered_pad[m, ki, :, :])  # (head_dim, 128)
                v_tile = nl.load(v_gathered_pad[m, ki, :, :])  # (128, head_dim)

                score_psum = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
                score_psum[...] += nisa.nc_matmul(q_t, k_t)
                score = nl.copy(score_psum, dtype=q_scaled_blocks.dtype)

                # Stable softmax weights: exp(score - row_max) / row_denom
                stable = score - row_max_m.reshape((_TILE_M, 1))
                weights = nl.exp(stable) / row_denom_m.reshape((_TILE_M, 1))  # (128, 128)

                # weights @ V: nc_matmul expects (K, M) stationary × (K, N) moving
                # weights is (128, 128) in SBUF; transpose to (128, 128) for nc_matmul
                weights_t = nl.transpose(weights)  # (128, 128), now (head_dim_k, seq) form
                out_psum[...] += nisa.nc_matmul(weights_t, v_tile)

            out_sbuf = nl.copy(out_psum, dtype=q_scaled_blocks.dtype)
            nl.store(out[m * _TILE_M : (m + 1) * _TILE_M, :], value=out_sbuf)

        return out

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

                    a_t = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    b_tile = nl.load(b[k_off : k_off + TILE_K, n_off : n_off + TILE_N])

                    psum[...] += nisa.nc_matmul(a_t, b_tile)

                c_sbuf = nl.copy(psum, dtype=a.dtype)
                nl.store(
                    c[m_off : m_off + TILE_M, n_off : n_off + TILE_N],
                    value=c_sbuf,
                )

        return c
