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


import hashlib
import os

_TILE_M = 128
_TILE_K = 128
_TILE_N = 512

# Split the static-structure kernels across the physical cores of a logical
# NeuronCore (LNC=2 on Trn2). `TRNSPARSE_CORE_SPLIT=0` keeps the whole kernel
# on every core (the pre-split behaviour; both cores then write identical
# results), for A/B measurements. Read at trace time.
_CORE_SPLIT_ENV = "TRNSPARSE_CORE_SPLIT"


def _core_split_enabled() -> bool:
    return os.environ.get(_CORE_SPLIT_ENV, "1").lower() not in ("0", "false", "no")


# Block rows per group in the pre-transposed BSR kernel (one PSUM tile each;
# 8 fills the eight 2 KiB PSUM banks at 512 fp32 columns). 1 = no grouping.
_ROW_GROUP_ENV = "TRNSPARSE_ROW_GROUP"


def _row_group() -> int:
    return max(1, min(8, int(os.environ.get(_ROW_GROUP_ENV, "8"))))


def _assign_units(costs, n_prog: int, pid: int) -> list[int]:
    """Indices of the work units that program `pid` of `n_prog` owns.

    Each unit writes a disjoint output tile, so any partition is race-free;
    this one balances the static cost (stored blocks per unit) with a
    longest-processing-time greedy: units in decreasing cost order go to the
    least-loaded program, ties to the lower program id and the lower unit
    index. Deterministic, so both traces (one per core) agree on the
    partition. Returned indices are ascending (each core walks its units in
    structure order, which keeps neighbouring DMAs together).
    """
    if n_prog <= 1:
        return list(range(len(costs)))
    load = [0.0] * n_prog
    owner = [0] * len(costs)
    for u in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        k = min(range(n_prog), key=lambda q: (load[q], q))
        owner[u] = k
        load[k] += costs[u]
    return [u for u in range(len(costs)) if owner[u] == pid]


def _row_block_counts(ptrs) -> list[int]:
    return [ptrs[i + 1] - ptrs[i] for i in range(len(ptrs) - 1)]


# Registered static structures, keyed by content hash (`register_structure`).
_STRUCTURES: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}


def register_structure(row_ptrs, cols) -> int:
    """Key for a block structure, for the `*_keyed_kernel`s.

    The key is the first 62 bits of the SHA-256 of the structure, so equal
    structures share a key in every process (safe under a persistent NKI
    compile cache) and different structures get different kernels.
    """
    rp = tuple(int(x) for x in row_ptrs)
    cl = tuple(int(x) for x in cols)
    digest = hashlib.sha256(repr((rp, cl)).encode()).digest()
    key = int.from_bytes(digest[:8], "big") >> 2
    _STRUCTURES.setdefault(key, (rp, cl))
    return key


# Relative cost of a unit with no stored blocks (it only stores a zero tile).
_EMPTY_UNIT_COST = 0.25


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
                    nisa.nc_matmul(psum, a_t, b_tile, accumulate=True)

                _pp = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32)
                _pn = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32)
                nisa.activation(_pp, nl.relu, psum)
                nisa.activation(_pn, nl.relu, psum, scale=-1.0)
                c_sbuf = nl.copy(nl.subtract(_pp, _pn), dtype=blocks_pad.dtype)
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

        # threshold_sqrt is a (1, 1) HBM tensor. NKI 0.7 rejects a 1-partition
        # scalar operand against a 128-partition tile, so broadcast it to a
        # (TILE_M, 1) per-partition scalar once, up front.
        thr = nl.broadcast_to(nl.load(threshold_sqrt), shape=(TILE_M, 1))

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N

                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                # Q[m-tile] as a (1, TILE_M) row, broadcast over the k partitions.
                q_m_row = nl.broadcast_to(
                    nl.load_transpose2d(q[m_off : m_off + TILE_M, :]), shape=(TILE_K, TILE_M)
                )

                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K

                    # Work in the transposed frame (k on partitions): the masked
                    # tile is then directly the stationary operand. The mask is
                    # symmetric in (i, j), so mask^T[k, m] = Q[k] Q[m] > thr.
                    # Before this, each tile was masked in the (m, k) frame and
                    # round-tripped through HBM to transpose it (4-5x slower
                    # than dense masked matmul on Trn2).
                    a_t = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    q_k = nl.load(q[k_off : k_off + TILE_K, :])  # (TILE_K, 1)
                    pair_bound = nl.multiply(q_m_row, q_k)  # (P, F) x (P, 1)
                    # Soft 0/1 mask: relu(diff)/(relu(diff)+eps) = 0 where masked, ~1 where not.
                    relu_diff = nl.relu(nl.subtract(pair_bound, thr))
                    # NKI 0.7 / neuronx-cc 2.28: tensor_tensor `divide` fails ISA
                    # validation (s3s3d3_tt_valid_op) - multiply by the reciprocal.
                    mask_f = nl.multiply(relu_diff, nl.reciprocal(nl.add(relu_diff, 1e-10)))
                    a_t = nl.multiply(a_t, mask_f)
                    b_tile = nl.load(b[k_off : k_off + TILE_K, n_off : n_off + TILE_N])

                    nisa.nc_matmul(psum, a_t, b_tile, accumulate=True)

                _pp = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32)
                _pn = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32)
                nisa.activation(_pp, nl.relu, psum)
                nisa.activation(_pn, nl.relu, psum, scale=-1.0)
                c_sbuf = nl.copy(nl.subtract(_pp, _pn), dtype=a.dtype)
                nl.store(
                    c[m_off : m_off + TILE_M, n_off : n_off + TILE_N],
                    value=c_sbuf,
                )

        return c

    @nki.jit
    def _attn_stats_kernel(q_scaled_blocks, k_gathered_pad):
        """Pass-1 attention kernel: per-block row-wise max and stable exp-sum.

        For head_dim ≤ 128: Q is loaded once per block-row as a stationary
        (head_dim, 128) tile, reused across all ki blocks (single nc_matmul).
        For head_dim > 128 (K-tiling): score is accumulated across TILE_K=128
        chunks of head_dim via an inner affine_range loop.

        Shapes:
            q_scaled_blocks:  (M_tiles, 128, head_dim)  — Q * scale
            k_gathered_pad:   (M_tiles, K_max, 128, head_dim) — K at nonzero cols
        Returns:
            tile_max:         (M_tiles, K_max, 128) — per-block row-wise max
            tile_sumexp:      (M_tiles, K_max, 128) — per-block stable exp-sum

        Q and K share a dtype (fp32 or bf16). The scores and the statistics are
        fp32 either way, as in the static kernels (#49).
        """
        M_tiles, K_max, _, _ = k_gathered_pad.shape
        _, _, head_dim = q_scaled_blocks.shape

        tile_max = nl.ndarray((M_tiles, K_max, _TILE_M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        tile_sumexp = nl.ndarray(
            (M_tiles, K_max, _TILE_M, 1), dtype=nl.float32, buffer=nl.shared_hbm
        )

        for m in nl.affine_range(M_tiles):
            for ki in nl.affine_range(K_max):
                score_psum = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
                if head_dim <= _TILE_K:
                    # NKI 0.3.0 simulator: nc_matmul's moving arg cannot be loaded with
                    # load_transpose2d — use nl.load + nl.transpose for K (moving tile).
                    q_t = nl.load_transpose2d(q_scaled_blocks[m, :, :])  # stationary
                    k_t = nl.load_transpose2d(k_gathered_pad[m, ki, :, :])  # moving
                    nisa.nc_matmul(score_psum, q_t, k_t, accumulate=True)
                else:
                    for hd in nl.affine_range(head_dim // _TILE_K):
                        q_c = nl.load_transpose2d(
                            q_scaled_blocks[m, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        k_c = nl.load_transpose2d(
                            k_gathered_pad[m, ki, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        nisa.nc_matmul(score_psum, q_c, k_c, accumulate=True)

                _ssp = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                _ssn = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                nisa.activation(_ssp, nl.relu, score_psum)
                nisa.activation(_ssn, nl.relu, score_psum, scale=-1.0)
                score = nl.subtract(_ssp, _ssn)
                t_max = nl.max(score, axis=1, keepdims=True)
                stable = nl.subtract(score, t_max)
                t_sum = nl.sum(nl.exp(stable), axis=1, keepdims=True)

                nl.store(tile_max[m, ki, :, :], value=t_max)
                nl.store(tile_sumexp[m, ki, :, :], value=t_sum)

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

        For head_dim ≤ 128: Q is stationary (loaded once per m, reused across ki).
        For head_dim > 128: K-tiling — Q/K loaded in TILE_K=128 chunks per ki.
        The weights @ V accumulation (K=128 block dim) is unchanged for all head_dims.

        Precision (#49): Q, K and V share a dtype (fp32 or bf16); row_max and
        row_denom are fp32. Scores, softmax weights and PSUM are fp32; the
        weights are rounded to V's dtype for the weights @ V matmul (both
        nc_matmul operands must be fp32, or neither), as in the static kernel.

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
        vdt = v_gathered_pad.dtype

        out = nl.ndarray(
            (M_tiles * _TILE_M, head_dim), dtype=q_scaled_blocks.dtype, buffer=nl.shared_hbm
        )

        for m in nl.affine_range(M_tiles):
            row_max_m = nl.load(row_max[m, :, :])  # (128, 1)
            row_denom_m = nl.load(row_denom[m, :, :])
            # NKI 0.7: tensor_scalar_arith has no `divide`; multiply by the
            # reciprocal of the (128, 1) softmax denominator instead.
            row_denom_inv_m = nl.reciprocal(row_denom_m)

            out_psum = nl.zeros((_TILE_M, head_dim), dtype=nl.float32, buffer=nl.psum)

            for ki in nl.affine_range(K_max):
                v_tile = nl.load(v_gathered_pad[m, ki, :, :])  # (128, head_dim)

                score_psum = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
                if head_dim <= _TILE_K:
                    q_t = nl.load_transpose2d(q_scaled_blocks[m, :, :])  # stationary
                    k_t = nl.load_transpose2d(k_gathered_pad[m, ki, :, :])  # moving
                    nisa.nc_matmul(score_psum, q_t, k_t, accumulate=True)
                else:
                    for hd in nl.affine_range(head_dim // _TILE_K):
                        q_c = nl.load_transpose2d(
                            q_scaled_blocks[m, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        k_c = nl.load_transpose2d(
                            k_gathered_pad[m, ki, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        nisa.nc_matmul(score_psum, q_c, k_c, accumulate=True)

                _ssp = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                _ssn = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                nisa.activation(_ssp, nl.relu, score_psum)
                nisa.activation(_ssn, nl.relu, score_psum, scale=-1.0)
                score = nl.subtract(_ssp, _ssn)
                stable = nl.subtract(score, row_max_m)
                weights = nl.multiply(nl.exp(stable), row_denom_inv_m)

                # NKI 0.3.0: nl.transpose gives PSUM which nc_matmul rejects as stationary.
                # Round-trip weights through HBM so load_transpose2d produces SBUF.
                # Rounded to V's dtype: nc_matmul takes fp32 operands only in pairs.
                _wh = nl.ndarray((_TILE_M, _TILE_M), dtype=vdt, buffer=nl.shared_hbm)
                nl.store(_wh, value=_cast(weights, vdt, nl.float32))
                weights_t = nl.load_transpose2d(_wh)  # (128,128) SBUF — stationary for nc_matmul
                nisa.nc_matmul(out_psum, weights_t, v_tile, accumulate=True)

            _op = nl.ndarray((_TILE_M, head_dim), dtype=nl.float32)
            _on = nl.ndarray((_TILE_M, head_dim), dtype=nl.float32)
            nisa.activation(_op, nl.relu, out_psum)
            nisa.activation(_on, nl.relu, out_psum, scale=-1.0)
            out_sbuf = nl.copy(nl.subtract(_op, _on), dtype=q_scaled_blocks.dtype)
            nl.store(out[m * _TILE_M : (m + 1) * _TILE_M, :], value=out_sbuf)

        return out

    @nki.jit
    def _attn_bwd_dq_kernel(
        q_scaled_blocks,
        k_gathered_pad,
        v_gathered_pad,
        do_gathered_pad,
        D_blocks,
        row_max,
        row_denom,
    ):
        """Backward kernel for dQ — row-first traversal.

        Mirrors `_attn_out_kernel`: iterates block-rows in the outer loop
        and k-slots in the inner loop. For each stored (m, ki) block,
        recomputes P from Q and K (same as forward), then accumulates:

            dS = P * (dP - D_m)   where dP = dO_m @ V_ki.T
            dQ_m += dS @ K_ki * scale  (scale already baked into q_scaled_blocks)

        Shapes:
            q_scaled_blocks:  (M_tiles, 128, head_dim)  — Q * scale
            k_gathered_pad:   (M_tiles, K_max, 128, head_dim)
            v_gathered_pad:   (M_tiles, K_max, 128, head_dim)
            do_gathered_pad:  (M_tiles, K_max, 128, head_dim)
            D_blocks:         (M_tiles, 128)  — Flash delta per block-row
            row_max:          (M_tiles, 128)
            row_denom:        (M_tiles, 128)
        Returns:
            dQ:               (M_tiles * 128, head_dim)

        Precision (#49): Q, K, V and dO share a dtype (fp32 or bf16); the
        statistics and D are fp32. P, dP and dS are fp32; dS is rounded to
        K's dtype for the dS @ K matmul, as in the static kernel.
        """
        M_tiles, K_max, _, head_dim = k_gathered_pad.shape
        kdt = k_gathered_pad.dtype

        dQ = nl.ndarray(
            (M_tiles * _TILE_M, head_dim), dtype=q_scaled_blocks.dtype, buffer=nl.shared_hbm
        )

        for m in nl.affine_range(M_tiles):
            row_max_m = nl.load(row_max[m, :, :])  # (128, 1)
            row_denom_m = nl.load(row_denom[m, :, :])
            # NKI 0.7: tensor_scalar_arith has no `divide`; multiply by the
            # reciprocal of the (128, 1) softmax denominator instead.
            row_denom_inv_m = nl.reciprocal(row_denom_m)
            d_m = nl.load(D_blocks[m, :, :])  # (128, 1)

            dq_psum = nl.zeros((_TILE_M, head_dim), dtype=nl.float32, buffer=nl.psum)

            for ki in nl.affine_range(K_max):
                # k_sbuf = K_ki as (128, head_dim) for nc_matmul(nl.transpose(dS), k_sbuf) = dS @ K
                k_sbuf = nl.load(k_gathered_pad[m, ki, :, :])  # (128, head_dim) moving for dQ

                # score = Q_m @ K_ki.T
                score_psum = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
                if head_dim <= _TILE_K:
                    q_t = nl.load_transpose2d(q_scaled_blocks[m, :, :])  # stationary
                    k_t = nl.load_transpose2d(k_gathered_pad[m, ki, :, :])  # moving
                    nisa.nc_matmul(score_psum, q_t, k_t, accumulate=True)
                else:
                    for hd in nl.affine_range(head_dim // _TILE_K):
                        q_c = nl.load_transpose2d(
                            q_scaled_blocks[m, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        k_c = nl.load_transpose2d(
                            k_gathered_pad[m, ki, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        nisa.nc_matmul(score_psum, q_c, k_c, accumulate=True)

                _ssp = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                _ssn = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                nisa.activation(_ssp, nl.relu, score_psum)
                nisa.activation(_ssn, nl.relu, score_psum, scale=-1.0)
                score = nl.subtract(_ssp, _ssn)
                stable = nl.subtract(score, row_max_m)
                P = nl.multiply(nl.exp(stable), row_denom_inv_m)

                # dP = dO_m @ V_ki.T
                dp_psum = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
                if head_dim <= _TILE_K:
                    do_t = nl.load_transpose2d(do_gathered_pad[m, ki, :, :])  # stationary
                    v_t = nl.load_transpose2d(v_gathered_pad[m, ki, :, :])  # moving
                    nisa.nc_matmul(dp_psum, do_t, v_t, accumulate=True)
                else:
                    for hd in nl.affine_range(head_dim // _TILE_K):
                        do_c = nl.load_transpose2d(
                            do_gathered_pad[m, ki, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        v_c = nl.load_transpose2d(
                            v_gathered_pad[m, ki, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        nisa.nc_matmul(dp_psum, do_c, v_c, accumulate=True)

                _dpp = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                _dpn = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                nisa.activation(_dpp, nl.relu, dp_psum)
                nisa.activation(_dpn, nl.relu, dp_psum, scale=-1.0)
                dP = nl.subtract(_dpp, _dpn)
                dS = nl.multiply(P, nl.subtract(dP, d_m))

                # Round-trip dS through HBM so load_transpose2d gives SBUF stationary.
                _dsh = nl.ndarray((_TILE_M, _TILE_M), dtype=kdt, buffer=nl.shared_hbm)
                nl.store(_dsh, value=_cast(dS, kdt, nl.float32))
                dS_t = nl.load_transpose2d(_dsh)  # SBUF (transposed) — stationary
                nisa.nc_matmul(dq_psum, dS_t, k_sbuf, accumulate=True)

            _dqp = nl.ndarray((_TILE_M, head_dim), dtype=nl.float32)
            _dqn = nl.ndarray((_TILE_M, head_dim), dtype=nl.float32)
            nisa.activation(_dqp, nl.relu, dq_psum)
            nisa.activation(_dqn, nl.relu, dq_psum, scale=-1.0)
            dq_sbuf = nl.copy(nl.subtract(_dqp, _dqn), dtype=q_scaled_blocks.dtype)
            nl.store(dQ[m * _TILE_M : (m + 1) * _TILE_M, :], value=dq_sbuf)

        return dQ

    @nki.jit
    def _attn_bwd_dkdv_kernel(
        k_blocks,
        v_blocks,
        q_gathered_col,
        do_gathered_col,
        D_gathered_col,
        row_max_gathered_col,
        row_denom_gathered_col,
    ):
        """Backward kernel for dK and dV — column-first traversal.

        For each column block ki, iterate over all block-rows m that attend
        to ki (pre-gathered by host). Accumulates:

            dV_ki += P.T @ dO_m
            dK_ki += dS.T @ Q_m * scale  (scale baked into q_gathered_col)

        Shapes:
            k_blocks:               (N_col, 128, head_dim)  — K in column order
            v_blocks:               (N_col, 128, head_dim)  — V in column order
            q_gathered_col:         (N_col, K_max_col, 128, head_dim)  — Q_m per (ki,mi) pair
            do_gathered_col:        (N_col, K_max_col, 128, head_dim)
            D_gathered_col:         (N_col, K_max_col, 128)
            row_max_gathered_col:   (N_col, K_max_col, 128)
            row_denom_gathered_col: (N_col, K_max_col, 128)
        Returns:
            dK: (N_col * 128, head_dim)
            dV: (N_col * 128, head_dim)

        Precision (#49): K, V, Q and dO share a dtype (fp32 or bf16); the
        statistics and D are fp32. P and dS are fp32 and are rounded to the
        operand dtype as the stationary operands of dS^T @ Q and P^T @ dO,
        as in the static kernel.
        """
        N_col, K_max_col, _, head_dim = q_gathered_col.shape
        qdt = q_gathered_col.dtype

        dK = nl.ndarray((N_col * _TILE_M, head_dim), dtype=k_blocks.dtype, buffer=nl.shared_hbm)
        dV = nl.ndarray((N_col * _TILE_M, head_dim), dtype=k_blocks.dtype, buffer=nl.shared_hbm)

        for ki in nl.affine_range(N_col):
            dk_psum = nl.zeros((_TILE_M, head_dim), dtype=nl.float32, buffer=nl.psum)
            dv_psum = nl.zeros((_TILE_M, head_dim), dtype=nl.float32, buffer=nl.psum)

            for mi in nl.affine_range(K_max_col):
                # All tile loads local to this (ki, mi) iteration — NKI simulator requires
                # nc_matmul args to be local to the innermost affine_range loop.
                if head_dim <= _TILE_K:
                    q_t_mi = nl.load_transpose2d(q_gathered_col[ki, mi, :, :])  # stationary
                    q_sbuf = nl.load(q_gathered_col[ki, mi, :, :])  # (128, hd) moving for dK
                    do_t_mi = nl.load_transpose2d(do_gathered_col[ki, mi, :, :])  # stationary
                    do_sbuf = nl.load(do_gathered_col[ki, mi, :, :])  # (128, hd) moving for dV
                    k_t = nl.load_transpose2d(k_blocks[ki, :, :])  # moving for score
                    v_t = nl.load_transpose2d(v_blocks[ki, :, :])  # moving for dP
                else:
                    q_sbuf = nl.load(q_gathered_col[ki, mi, :, :])  # (128, head_dim)
                    do_sbuf = nl.load(do_gathered_col[ki, mi, :, :])  # (128, head_dim)

                d_mi = nl.load(D_gathered_col[ki, mi, :, :])  # (128, 1)
                row_max_mi = nl.load(row_max_gathered_col[ki, mi, :, :])
                row_denom_mi = nl.load(row_denom_gathered_col[ki, mi, :, :])

                # score = Q_m @ K_ki.T
                score_psum = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
                if head_dim <= _TILE_K:
                    nisa.nc_matmul(score_psum, q_t_mi, k_t, accumulate=True)
                else:
                    for hd in nl.affine_range(head_dim // _TILE_K):
                        q_c = nl.load_transpose2d(
                            q_gathered_col[ki, mi, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        k_c = nl.load_transpose2d(
                            k_blocks[ki, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        nisa.nc_matmul(score_psum, q_c, k_c, accumulate=True)

                _ssp = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                _ssn = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                nisa.activation(_ssp, nl.relu, score_psum)
                nisa.activation(_ssn, nl.relu, score_psum, scale=-1.0)
                score = nl.subtract(_ssp, _ssn)
                stable_s = nl.subtract(score, row_max_mi)
                P = nl.multiply(nl.exp(stable_s), nl.reciprocal(row_denom_mi))

                # dP = dO_m @ V_ki.T
                dp_psum = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
                if head_dim <= _TILE_K:
                    nisa.nc_matmul(dp_psum, do_t_mi, v_t, accumulate=True)
                else:
                    for hd in nl.affine_range(head_dim // _TILE_K):
                        do_c = nl.load_transpose2d(
                            do_gathered_col[ki, mi, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        v_c = nl.load_transpose2d(
                            v_blocks[ki, :, hd * _TILE_K : (hd + 1) * _TILE_K]
                        )
                        nisa.nc_matmul(dp_psum, do_c, v_c, accumulate=True)

                _dpp = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                _dpn = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
                nisa.activation(_dpp, nl.relu, dp_psum)
                nisa.activation(_dpn, nl.relu, dp_psum, scale=-1.0)
                dP = nl.subtract(_dpp, _dpn)
                dS = nl.multiply(P, nl.subtract(dP, d_mi))

                # nc_matmul(dS, q_sbuf) = dS.T @ Q_m (scale baked into q_gathered_col)
                nisa.nc_matmul(dk_psum, _cast(dS, qdt, nl.float32), q_sbuf, accumulate=True)

                # nc_matmul(P, do_sbuf) = P.T @ dO_m
                nisa.nc_matmul(dv_psum, _cast(P, qdt, nl.float32), do_sbuf, accumulate=True)

            _dkp = nl.ndarray((_TILE_M, head_dim), dtype=nl.float32)
            _dkn = nl.ndarray((_TILE_M, head_dim), dtype=nl.float32)
            nisa.activation(_dkp, nl.relu, dk_psum)
            nisa.activation(_dkn, nl.relu, dk_psum, scale=-1.0)
            dk_sbuf = nl.copy(nl.subtract(_dkp, _dkn), dtype=k_blocks.dtype)
            nl.store(dK[ki * _TILE_M : (ki + 1) * _TILE_M, :], value=dk_sbuf)

            _dvp = nl.ndarray((_TILE_M, head_dim), dtype=nl.float32)
            _dvn = nl.ndarray((_TILE_M, head_dim), dtype=nl.float32)
            nisa.activation(_dvp, nl.relu, dv_psum)
            nisa.activation(_dvn, nl.relu, dv_psum, scale=-1.0)
            dv_sbuf = nl.copy(nl.subtract(_dvp, _dvn), dtype=k_blocks.dtype)
            nl.store(dV[ki * _TILE_M : (ki + 1) * _TILE_M, :], value=dv_sbuf)

        return dK, dV

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

                    nisa.nc_matmul(psum, a_t, b_tile, accumulate=True)

                _pp = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32)
                _pn = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32)
                nisa.activation(_pp, nl.relu, psum)
                nisa.activation(_pn, nl.relu, psum, scale=-1.0)
                c_sbuf = nl.copy(nl.subtract(_pp, _pn), dtype=a.dtype)
                nl.store(
                    c[m_off : m_off + TILE_M, n_off : n_off + TILE_N],
                    value=c_sbuf,
                )

        return c

    # ── Static-structure kernels (native path) ───────────────────────────────
    #
    # On TorchNeuron the sparse structure lives on the host (the `neuron`
    # device has no sparse CSR, trnsparse#30). These kernels take that
    # structure as compile-time constants (`row_ptrs`, `cols`: tuples of
    # Python ints) instead of as padded, gathered device tensors. The trace
    # unrolls exactly the stored blocks: no (M_tiles, K_max) padding (a
    # global-token row no longer makes every row cost K_max blocks), no
    # device-side gather of B / K / V before the launch, and every DMA
    # address is static. The cost is one compile per distinct structure
    # (a few seconds; cached in-process by the constant key), which is the
    # right trade when a pattern is reused: iterative solvers, Fock builds,
    # fixed attention masks.
    #
    # Both physical cores (#30 follow-up). Under LNC=2 (the Trn2 default)
    # NKI traces a kernel once per physical core, with `program_id` 0 and 1
    # as Python ints. Each kernel splits its output tiles (block rows, plus
    # 512-column tiles for wide B; block columns for dK/dV) between the
    # cores with `_assign_units`, balanced by stored blocks. Every output
    # tile has exactly one writer, so the cores never share a write.
    #
    # Precision. Inputs may be fp32 or bf16 (all matmul operands of one
    # call share a dtype). Every product accumulates in an fp32 PSUM tile;
    # the softmax statistics and the score-space arithmetic (exp, max, sums,
    # P * (dP - D)) stay fp32, and only the stationary operands built from
    # them (P, dS) are rounded to the input dtype before their matmul, as in
    # flash attention. Outputs take the input dtype; the attention statistics
    # are always fp32.

    def _program_split():
        """(n_programs, program_id) of the current trace, as Python ints."""
        if not _core_split_enabled():
            return 1, 0
        n_prog = int(nl.num_programs()) if nl.program_ndim() > 0 else 1
        pid = int(nl.program_id(0)) if n_prog > 1 else 0
        return n_prog, pid

    def _cast(t, dtype, like_dtype):
        """`t` as `dtype`, or `t` itself when it already has that dtype."""
        return t if dtype == like_dtype else nl.copy(t, dtype=dtype)

    def _bsr_spmm_static_body(blocks, b, row_ptrs, cols):
        """C = A @ B for a BSR A whose structure is static.

        Shapes:
            blocks:   (n_blocks, 128, 128)  stored blocks, row-major
            b:        (n_pad, N)            N <= 512 or a multiple of 512
            row_ptrs: tuple[int]            block-row pointers (M_tiles + 1)
            cols:     tuple[int]            block-column index per block
        blocks and b share a dtype (fp32 or bf16); PSUM accumulates in fp32.
        Returns:
            out:      (M_tiles * 128, N), b's dtype

        Work units are (block row, column tile) output tiles, split between
        the physical cores by stored-block count. A product with fewer
        output tiles than cores splits its columns instead.
        """
        _, N = b.shape
        M_tiles = len(row_ptrs) - 1
        TILE_N = N if N <= _TILE_N else _TILE_N
        n_prog, pid = _program_split()
        if M_tiles * (N // TILE_N) < n_prog and TILE_N % (n_prog * 64) == 0:
            TILE_N //= n_prog
        n_nt = N // TILE_N
        counts = _row_block_counts(row_ptrs)
        units = [(m, nt) for m in range(M_tiles) for nt in range(n_nt)]
        costs = [counts[m] if counts[m] else _EMPTY_UNIT_COST for m, _ in units]

        out = nl.ndarray((M_tiles * _TILE_M, N), dtype=b.dtype, buffer=nl.shared_hbm)

        for u in _assign_units(costs, n_prog, pid):
            m, nt = units[u]
            lo = row_ptrs[m]
            hi = row_ptrs[m + 1]
            r0 = m * _TILE_M
            n0 = nt * TILE_N
            if hi == lo:
                nl.store(
                    out[r0 : r0 + _TILE_M, n0 : n0 + TILE_N],
                    value=nl.zeros((_TILE_M, TILE_N), dtype=b.dtype),
                )
                continue
            psum = nl.zeros((_TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for idx in range(lo, hi):
                j = cols[idx]
                a_t = nl.load_transpose2d(blocks[idx, :, :])
                b_tile = nl.load(b[j * _TILE_K : (j + 1) * _TILE_K, n0 : n0 + TILE_N])
                nisa.nc_matmul(psum, a_t, b_tile, accumulate=True)
            _pp = nl.ndarray((_TILE_M, TILE_N), dtype=nl.float32)
            _pn = nl.ndarray((_TILE_M, TILE_N), dtype=nl.float32)
            nisa.activation(_pp, nl.relu, psum)
            nisa.activation(_pn, nl.relu, psum, scale=-1.0)
            c_sbuf = nl.copy(nl.subtract(_pp, _pn), dtype=b.dtype)
            nl.store(out[r0 : r0 + _TILE_M, n0 : n0 + TILE_N], value=c_sbuf)

        return out

    # Stored blocks per row loaded by one DMA in the pre-transposed kernel
    # (16 x 128 x 4 B = 8 KiB per partition in fp32).
    _BLOCKS_PER_DMA = 16
    # Stored blocks of a row group held in SBUF at once (64 KiB per
    # partition in fp32).
    _GROUP_MAX_BLOCKS = 128

    def _bsr_spmm_static_t_body(blocks_t, b, row_ptrs, cols):
        """C = A @ B for a static-structure BSR A with pre-transposed blocks.

        Same work split and accumulation as `_bsr_spmm_static_kernel`, with
        the per-block overheads cut (#30):

        - The blocks arrive transposed, side by side, as one (128,
          n_blocks * 128) array: block idx's A^T is columns
          [idx * 128, (idx + 1) * 128). The stationary operand is a plain
          load, with no `load_transpose2d`.
        - A block row's stored blocks are contiguous there, so they arrive
          in one DMA per up to `_BLOCKS_PER_DMA` blocks rather than one each.
        - Block rows are processed in groups (up to 8, one PSUM tile each):
          each B tile is loaded once per group and multiplied into every
          row of the group that stores a block in its column.
        - The PSUM tile is read out by one copy (in b's dtype) instead of
          two activations and a subtract.

        Shapes:
            blocks_t: (128, n_blocks * 128)  A_idx^T side by side
            b:        (n_pad, N)             N <= 512 or a multiple of 512
            row_ptrs, cols: as in `_bsr_spmm_static_kernel`
        Returns:
            out:      (M_tiles * 128, N), b's dtype
        """
        _, N = b.shape
        M_tiles = len(row_ptrs) - 1
        TILE_N = N if N <= _TILE_N else _TILE_N
        n_prog, pid = _program_split()
        if M_tiles * (N // TILE_N) < n_prog and TILE_N % (n_prog * 64) == 0:
            TILE_N //= n_prog
        n_nt = N // TILE_N
        counts = _row_block_counts(row_ptrs)
        units = [(m, nt) for m in range(M_tiles) for nt in range(n_nt)]
        costs = [counts[m] if counts[m] else _EMPTY_UNIT_COST for m, _ in units]

        out = nl.ndarray((M_tiles * _TILE_M, N), dtype=b.dtype, buffer=nl.shared_hbm)

        # Row groups: up to `_row_group()` block rows (one PSUM tile each) and
        # `_GROUP_MAX_BLOCKS` stored blocks, same column tile. Within a group
        # each B tile is loaded once and applied to every row that stores a
        # block in that column, so B traffic drops by the column overlap.
        G = _row_group()
        groups = []
        cur, cur_blocks, cur_nt = [], 0, None
        for u in _assign_units(costs, n_prog, pid):
            m, nt = units[u]
            if counts[m] == 0:
                nl.store(
                    out[m * _TILE_M : (m + 1) * _TILE_M, nt * TILE_N : (nt + 1) * TILE_N],
                    value=nl.zeros((_TILE_M, TILE_N), dtype=b.dtype),
                )
                continue
            if cur and (
                nt != cur_nt or len(cur) == G or cur_blocks + counts[m] > _GROUP_MAX_BLOCKS
            ):
                groups.append((cur_nt, cur))
                cur, cur_blocks = [], 0
            cur.append(m)
            cur_blocks += counts[m]
            cur_nt = nt
        if cur:
            groups.append((cur_nt, cur))

        for nt, rows in groups:
            n0 = nt * TILE_N
            if len(rows) == 1:
                # A row alone (no reuse to gain, possibly more blocks than
                # fit in SBUF at once): stream its blocks chunk by chunk.
                m = rows[0]
                psum = nl.zeros((_TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for c0 in range(row_ptrs[m], row_ptrs[m + 1], _BLOCKS_PER_DMA):
                    c1 = min(row_ptrs[m + 1], c0 + _BLOCKS_PER_DMA)
                    a_all = nl.load(blocks_t[:, c0 * _TILE_K : c1 * _TILE_K])
                    for idx in range(c0, c1):
                        j = cols[idx]
                        i0 = (idx - c0) * _TILE_K
                        b_tile = nl.load(b[j * _TILE_K : (j + 1) * _TILE_K, n0 : n0 + TILE_N])
                        nisa.nc_matmul(psum, a_all[:, i0 : i0 + _TILE_K], b_tile, accumulate=True)
                nl.store(
                    out[m * _TILE_M : (m + 1) * _TILE_M, n0 : n0 + TILE_N],
                    value=nl.copy(psum, dtype=b.dtype),
                )
                continue
            psums = [nl.zeros((_TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum) for _ in rows]
            a_of = {}
            by_col = {}
            for r, m in enumerate(rows):
                lo = row_ptrs[m]
                hi = row_ptrs[m + 1]
                for c0 in range(lo, hi, _BLOCKS_PER_DMA):
                    c1 = min(hi, c0 + _BLOCKS_PER_DMA)
                    a_all = nl.load(blocks_t[:, c0 * _TILE_K : c1 * _TILE_K])
                    for idx in range(c0, c1):
                        a_of[idx] = (a_all, (idx - c0) * _TILE_K)
                        by_col.setdefault(cols[idx], []).append((r, idx))
            for j in sorted(by_col):
                b_tile = nl.load(b[j * _TILE_K : (j + 1) * _TILE_K, n0 : n0 + TILE_N])
                for r, idx in by_col[j]:
                    a_all, i0 = a_of[idx]
                    nisa.nc_matmul(psums[r], a_all[:, i0 : i0 + _TILE_K], b_tile, accumulate=True)
            for r, m in enumerate(rows):
                nl.store(
                    out[m * _TILE_M : (m + 1) * _TILE_M, n0 : n0 + TILE_N],
                    value=nl.copy(psums[r], dtype=b.dtype),
                )

        return out

    def _attn_fwd_static_body(q_t, k_t, v, row_ptrs, cols):
        """Block-sparse attention forward, one launch, static mask structure.

        Per block-row m (128 queries), over exactly the row's stored blocks:

        pass 1: S = Q_m K_j^T (queries on partitions); online row max and
                softmax denominator, kept on chip.
        pass 2: S^T = K_j Q_m^T (keys on partitions), P^T = exp(S^T - max)
                with the row max transposed once to a (1, 128) row and
                broadcast over partitions; P^T is directly the stationary
                operand of out += P @ V_j. No score round trip through HBM
                and no PE transpose per block.
        out = acc * (1 / denom), the reciprocal as rsqrt^2. (`nl.reciprocal`
        on ScalarE is accurate to ~1.5e-6 relative, but only within roughly
        |x| in [2^-42, 2^41]: outside it silently returns 1e7 or 0.
        `nisa.reciprocal` on VectorE is correctly rounded over all of fp32.)

        Block rows are split between the physical cores by stored-block
        count. q_t, k_t and v share a dtype (fp32 or bf16); S and the
        statistics are fp32, P^T is rounded to v's dtype for the P @ V
        matmul (fp32 PSUM).

        Shapes:
            q_t:  (head_dim, seq)  (Q * scale)^T, head_dim <= 128
            k_t:  (head_dim, seq)  K^T
            v:    (seq, head_dim)
            row_ptrs, cols: block-mask structure (tuples of int)
        Returns:
            out (seq, head_dim) in v's dtype, row_max (seq, 1) and
            row_denom (seq, 1) in fp32 (row_denom = sum exp(s - row_max),
            as the backward expects).
        """
        head_dim, seq = q_t.shape
        M_tiles = len(row_ptrs) - 1
        n_prog, pid = _program_split()
        counts = _row_block_counts(row_ptrs)
        costs = [2 * c if c else _EMPTY_UNIT_COST for c in counts]

        out = nl.ndarray((seq, head_dim), dtype=v.dtype, buffer=nl.shared_hbm)
        rmax_o = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        rden_o = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.shared_hbm)

        for m in _assign_units(costs, n_prog, pid):
            lo = row_ptrs[m]
            hi = row_ptrs[m + 1]
            r0 = m * _TILE_M
            if hi == lo:
                nl.store(
                    out[r0 : r0 + _TILE_M, :], value=nl.zeros((_TILE_M, head_dim), dtype=v.dtype)
                )
                nl.store(
                    rmax_o[r0 : r0 + _TILE_M, :], value=nl.zeros((_TILE_M, 1), dtype=nl.float32)
                )
                nl.store(
                    rden_o[r0 : r0 + _TILE_M, :], value=nl.full((_TILE_M, 1), 1.0, dtype=nl.float32)
                )
                continue
            qm = nl.load(q_t[:, r0 : r0 + _TILE_M])  # (head_dim, 128)
            rm = None
            den = None
            for idx in range(lo, hi):
                j = cols[idx]
                kj = nl.load(k_t[:, j * _TILE_M : (j + 1) * _TILE_M])
                s_ps = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(s_ps, qm, kj, accumulate=True)
                s = nl.copy(s_ps, dtype=nl.float32)
                t_max = nl.max(s, axis=1, keepdims=True)
                if rm is None:
                    rm = t_max
                    den = nl.sum(nl.exp(nl.subtract(s, rm)), axis=1, keepdims=True)
                else:
                    rm_new = nl.maximum(rm, t_max)
                    alpha = nl.exp(nl.subtract(rm, rm_new))
                    t_sum = nl.sum(nl.exp(nl.subtract(s, rm_new)), axis=1, keepdims=True)
                    den = nl.add(nl.multiply(den, alpha), t_sum)
                    rm = rm_new
            nl.store(rmax_o[r0 : r0 + _TILE_M, :], value=rm)
            nl.store(rden_o[r0 : r0 + _TILE_M, :], value=den)

            rm_ps = nl.ndarray((1, _TILE_M), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(rm_ps, rm, engine=nisa.engine.tensor)
            rm_b = nl.broadcast_to(nl.copy(rm_ps, dtype=nl.float32), shape=(_TILE_M, _TILE_M))
            rs = nl.rsqrt(den)
            rinv = nl.multiply(rs, rs)

            o_ps = nl.zeros((_TILE_M, head_dim), dtype=nl.float32, buffer=nl.psum)
            for idx in range(lo, hi):
                j = cols[idx]
                kj = nl.load(k_t[:, j * _TILE_M : (j + 1) * _TILE_M])
                vj = nl.load(v[j * _TILE_M : (j + 1) * _TILE_M, :])
                st_ps = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(st_ps, kj, qm, accumulate=True)  # S^T: keys on partitions
                p_t = nl.exp(nl.subtract(nl.copy(st_ps, dtype=nl.float32), rm_b), dtype=v.dtype)
                nisa.nc_matmul(o_ps, p_t, vj, accumulate=True)  # out += P @ V_j
            o = nl.multiply(nl.copy(o_ps, dtype=nl.float32), rinv)
            nl.store(out[r0 : r0 + _TILE_M, :], value=nl.copy(o, dtype=v.dtype))

        return out, rmax_o, rden_o

    # ── jit entry points ─────────────────────────────────────────────────────
    #
    # Each static kernel takes its structure either as tuples (`*_kernel`)
    # or as a registered structure key (`*_keyed_kernel`, see
    # `register_structure`). The key is a content hash, so it is just as
    # much a compile-time constant as the tuples, but it crosses the
    # PyTorch custom-op boundary as one int: the host cost of a launch no
    # longer grows with the number of stored blocks (Trn2: 0.13 ms at 41
    # blocks, 0.67 ms at 2048 with the tuples).

    @nki.jit
    def _bsr_spmm_static_kernel(blocks, b, row_ptrs, cols):
        return _bsr_spmm_static_body(blocks, b, row_ptrs, cols)

    @nki.jit
    def _bsr_spmm_static_t_kernel(blocks_t, b, row_ptrs, cols):
        return _bsr_spmm_static_t_body(blocks_t, b, row_ptrs, cols)

    @nki.jit
    def _attn_fwd_static_kernel(q_t, k_t, v, row_ptrs, cols):
        return _attn_fwd_static_body(q_t, k_t, v, row_ptrs, cols)

    @nki.jit
    def _bsr_spmm_keyed_kernel(blocks, b, key):
        return _bsr_spmm_static_body(blocks, b, *_STRUCTURES[key])

    @nki.jit
    def _bsr_spmm_t_keyed_kernel(blocks_t, b, key):
        return _bsr_spmm_static_t_body(blocks_t, b, *_STRUCTURES[key])

    @nki.jit
    def _attn_fwd_keyed_kernel(q_t, k_t, v, key):
        return _attn_fwd_static_body(q_t, k_t, v, *_STRUCTURES[key])

    @nki.jit
    def _attn_bwd_static_kernel(
        q_t,
        k_t,
        v_t,
        do_t,
        q_s,
        k,
        do,
        rm_col,
        rinv_col,
        d_col,
        rm_row,
        rinv_row,
        d_row,
        row_ptrs,
        cols,
        col_ptrs,
        rows,
        scale,
    ):
        """Block-sparse attention backward, one launch, static mask structure.

        With Qs = Q * scale, S = Qs K^T, P = exp(S - row_max) / row_denom,
        D = rowsum(dO * O), per stored block (m, j):

            dS = P * (dO_m V_j^T - D_m)
            dQ_m += scale * dS K_j        (row-first loop, dS^T on partitions)
            dV_j += P^T dO_m              (column-first loop, P on partitions)
            dK_j += dS^T Qs_m

        Each output tile accumulates in PSUM over exactly its stored blocks.
        The row-first loop works in the transposed (keys-on-partitions)
        frame, so its per-query statistics come in as (1, seq) rows broadcast
        over partitions; the column-first loop uses (seq, 1) columns.
        `rinv` is 1 / row_denom computed by the caller (an aten division
        on the device).

        The dQ block rows and the dK/dV block columns form one pool of
        output tiles, split between the physical cores by matmul count
        (3 per block for a dQ row, 4 for a dK/dV column). The seven operand
        tensors share a dtype (fp32 or bf16); the statistics are fp32, and
        P, dS are rounded to the operand dtype only for their matmuls.

        Shapes: *_t (head_dim, seq); q_s, k, do (seq, head_dim);
        *_col (seq, 1); *_row (1, seq). head_dim <= 128.
        Returns dQ, dK, dV, each (seq, head_dim) in q_s's dtype.
        """
        head_dim, seq = q_t.shape
        M_tiles = len(row_ptrs) - 1
        N_col = len(col_ptrs) - 1
        T = _TILE_M
        dt = q_s.dtype
        n_prog, pid = _program_split()
        row_counts = _row_block_counts(row_ptrs)
        col_counts = _row_block_counts(col_ptrs)
        units = [(0, m) for m in range(M_tiles)] + [(1, j) for j in range(N_col)]
        costs = [3 * c if c else _EMPTY_UNIT_COST for c in row_counts] + [
            4 * c if c else _EMPTY_UNIT_COST for c in col_counts
        ]
        mine = [units[u] for u in _assign_units(costs, n_prog, pid)]

        dQ = nl.ndarray((seq, head_dim), dtype=dt, buffer=nl.shared_hbm)
        dK = nl.ndarray((seq, head_dim), dtype=dt, buffer=nl.shared_hbm)
        dV = nl.ndarray((seq, head_dim), dtype=dt, buffer=nl.shared_hbm)

        # ── dQ: row-first ────────────────────────────────────────────────────
        for m in [i for kind, i in mine if kind == 0]:
            lo = row_ptrs[m]
            hi = row_ptrs[m + 1]
            r0 = m * T
            if hi == lo:
                nl.store(dQ[r0 : r0 + T, :], value=nl.zeros((T, head_dim), dtype=dt))
                continue
            qm_t = nl.load(q_t[:, r0 : r0 + T])  # (hd, q)
            dom_t = nl.load(do_t[:, r0 : r0 + T])  # (hd, q)
            rm_b = nl.broadcast_to(nl.load(rm_row[:, r0 : r0 + T]), shape=(T, T))
            ri_b = nl.broadcast_to(nl.load(rinv_row[:, r0 : r0 + T]), shape=(T, T))
            d_b = nl.broadcast_to(nl.load(d_row[:, r0 : r0 + T]), shape=(T, T))
            dq_ps = nl.zeros((T, head_dim), dtype=nl.float32, buffer=nl.psum)
            for idx in range(lo, hi):
                j = cols[idx]
                c0 = j * T
                kj_t = nl.load(k_t[:, c0 : c0 + T])
                vj_t = nl.load(v_t[:, c0 : c0 + T])
                kj = nl.load(k[c0 : c0 + T, :])  # (k, hd)
                st_ps = nl.zeros((T, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(st_ps, kj_t, qm_t, accumulate=True)  # S^T (k, q)
                pt = nl.multiply(nl.exp(nl.subtract(nl.copy(st_ps, dtype=nl.float32), rm_b)), ri_b)
                dpt_ps = nl.zeros((T, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dpt_ps, vj_t, dom_t, accumulate=True)  # dP^T (k, q)
                dst = nl.multiply(pt, nl.subtract(nl.copy(dpt_ps, dtype=nl.float32), d_b), dtype=dt)
                nisa.nc_matmul(dq_ps, dst, kj, accumulate=True)  # dS @ K_j (q, hd)
            dq = nl.multiply(nl.copy(dq_ps, dtype=nl.float32), scale)
            nl.store(dQ[r0 : r0 + T, :], value=nl.copy(dq, dtype=dt))

        # ── dK, dV: column-first ─────────────────────────────────────────────
        for j in [i for kind, i in mine if kind == 1]:
            lo = col_ptrs[j]
            hi = col_ptrs[j + 1]
            c0 = j * T
            if hi == lo:
                zero = nl.zeros((T, head_dim), dtype=dt)
                nl.store(dK[c0 : c0 + T, :], value=zero)
                nl.store(dV[c0 : c0 + T, :], value=zero)
                continue
            kj_t = nl.load(k_t[:, c0 : c0 + T])
            vj_t = nl.load(v_t[:, c0 : c0 + T])
            dk_ps = nl.zeros((T, head_dim), dtype=nl.float32, buffer=nl.psum)
            dv_ps = nl.zeros((T, head_dim), dtype=nl.float32, buffer=nl.psum)
            for idx in range(lo, hi):
                m = rows[idx]
                r0 = m * T
                qm_t = nl.load(q_t[:, r0 : r0 + T])
                dom_t = nl.load(do_t[:, r0 : r0 + T])
                qm = nl.load(q_s[r0 : r0 + T, :])  # (q, hd)
                dom = nl.load(do[r0 : r0 + T, :])  # (q, hd)
                rm_m = nl.load(rm_col[r0 : r0 + T, :])
                ri_m = nl.load(rinv_col[r0 : r0 + T, :])
                d_m = nl.load(d_col[r0 : r0 + T, :])
                s_ps = nl.zeros((T, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(s_ps, qm_t, kj_t, accumulate=True)  # S (q, k)
                p = nl.multiply(nl.exp(nl.subtract(nl.copy(s_ps, dtype=nl.float32), rm_m)), ri_m)
                dp_ps = nl.zeros((T, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dp_ps, dom_t, vj_t, accumulate=True)  # dP (q, k)
                ds = nl.multiply(p, nl.subtract(nl.copy(dp_ps, dtype=nl.float32), d_m), dtype=dt)
                nisa.nc_matmul(dv_ps, _cast(p, dt, nl.float32), dom, accumulate=True)  # P^T dO_m
                nisa.nc_matmul(dk_ps, ds, qm, accumulate=True)  # dS^T Qs_m (k, hd)
            nl.store(dK[c0 : c0 + T, :], value=nl.copy(dk_ps, dtype=dt))
            nl.store(dV[c0 : c0 + T, :], value=nl.copy(dv_ps, dtype=dt))

        return dQ, dK, dV


# ── Fused, SBUF-resident conjugate gradient over static BSR (#22) ───────────
#
# One launch runs the whole solve: A's stored blocks are DMA'd into SBUF
# once and stay there for every iteration; x, r, p (and z) live in SBUF in
# a column layout (element t*128 + p of right-hand side c at [p, t*R + c]);
# every dot product is a VectorE multiply-reduce along the free axis
# followed by one TensorE all-ones matmul that sums over partitions and
# broadcasts the result to all 128 partitions at once, in PSUM.
# Convergence is checked on the device: an `nl.while_loop` runs
# `check_every` CG steps per trip and re-tests ||r||^2 > tol^2 ||b||^2 (and
# the iteration cap) through a sequencer register, so the host is never in
# the iteration loop. Inside a trip the steps are masked: once a column has
# converged (or hit the cap) its alpha = beta = 0 and its state stops
# changing, so iteration counts and solutions are exact, not rounded up to a
# multiple of `check_every`.
#
# R > 1 right-hand sides are R independent CG solves sharing A: each stored
# block is one matmul for all R columns (R stationary columns in the column
# matvec, R moving columns in the row matvec; see `block_matvec_col`).

# SBUF bytes per partition the CG kernel may use for resident A blocks,
# block-Jacobi inverses and (for R > 1) its vectors. Trn2 SBUF is 192 KiB
# per partition; the rest holds stream buffers and compiler temporaries.
_CG_SBUF_ENV = "TRNSPARSE_CG_SBUF_KIB"
# Stored blocks per streaming DMA when A does not fit (8 KiB / partition).
_CG_STREAM_BLOCKS = 16
_CG_TINY = 1e-30
# fp32 (128, R) vectors the multi-RHS kernel keeps in SBUF per block row.
_CG_VECTORS = 8
# Column ("vector-stationary") matvec: block rows per PSUM bank, and the
# most right-hand sides it takes (one 32-column PE tile).
_CG_COL_ROWS = 16
_CG_COL_MAX_RHS = 32
# Bytes per partition (fp32) up to which LNC=2 exchanges use GpSimd's DMA.
_CG_GPSIMD_DMA_BYTES = 1024
# Most bytes per partition of one "spread" matvec source vector (see
# `_cg_bsr_body`): caps the zero gaps, so very many block rows use more chunks.
_CG_SPREAD_BYTES = 8192


def _cg_sbuf_budget_bytes() -> int:
    return int(float(os.environ.get(_CG_SBUF_ENV, "160")) * 1024)


def cg_resident_blocks(
    n_blocks: int,
    n_tiles: int,
    precond: bool,
    n_rhs: int = 1,
    itemsize: int = 4,
    budget_kib: float | None = None,
    block_bytes: int | None = None,
) -> int:
    """How many of A's stored blocks the fused CG kernel keeps in SBUF.

    Blocks past this count are streamed from HBM on every iteration (still
    inside the one launch). The block-Jacobi inverses (one block per block
    row) are always resident and come out of the same budget, as do the
    vectors when there is more than one right-hand side. `block_bytes` is
    the per-partition size of one stored block of A when it differs from
    the fp32 inverses (bf16 A: 256; bf16 hi + lo: 512).
    """
    per_block = _TILE_K * itemsize  # bytes per partition per 128x128 block
    a_block = per_block if block_bytes is None else block_bytes
    total = _cg_sbuf_budget_bytes() if budget_kib is None else int(budget_kib * 1024)
    budget = total - (n_tiles * per_block if precond else 0)
    if n_rhs > 1:
        budget -= _CG_VECTORS * n_tiles * n_rhs * itemsize
    return max(0, min(n_blocks, budget // a_block))


def cg_core_rows(
    row_ptrs, precond: bool, n_prog: int, enabled: bool | None = None
) -> list[tuple[int, int]]:
    """Contiguous block-row range of each physical core in the fused CG.

    With two cores (LNC=2, core split on) the rows split where the stored
    blocks (plus one block-Jacobi inverse per row) balance best; each core
    then keeps only its rows' blocks in SBUF and computes their part of
    every matvec and dot product. One core, or one block row, keeps all.
    """
    T = len(row_ptrs) - 1
    if enabled is None:
        enabled = _core_split_enabled()
    if n_prog != 2 or T < 2 or not enabled:
        return [(0, T)]
    extra = 1 if precond else 0
    cum = [row_ptrs[t] + extra * t for t in range(T + 1)]
    total = cum[T]
    s = min(range(1, T), key=lambda k: (abs(2 * cum[k] - total), k))
    return [(0, s), (s, T)]


def cg_row_order(row_ptrs, lo: int, hi: int) -> list[int]:
    """Order of block rows lo..hi-1 in the fused CG's vector layout.

    Position i of a core's rows goes to TensorE column group i % 4 of the
    column matvec; the four groups run concurrently, so a matvec takes as
    long as the busiest one. Rows are dealt to the groups largest first,
    each to the least loaded group with room (group g holds
    ceil((n - g) / 4) rows), which balances the stored blocks per group much
    better than i % 4 on the original order. The layout is internal: b, x0
    and x are permuted on the way in and out of the kernel.
    """
    n = hi - lo
    caps = [(n - g + 3) // 4 for g in range(4)]
    loads = [0] * 4
    members: list[list[int]] = [[] for _ in range(4)]
    for m in sorted(range(lo, hi), key=lambda m: (-(row_ptrs[m + 1] - row_ptrs[m]), m)):
        g = min((g for g in range(4) if len(members[g]) < caps[g]), key=lambda g: (loads[g], g))
        members[g].append(m)
        loads[g] += row_ptrs[m + 1] - row_ptrs[m]
    order = [0] * n
    for g in range(4):
        for c, m in enumerate(sorted(members[g])):
            order[4 * c + g] = m
    return order


if HAS_NKI:

    def _cg_bsr_body(
        blocks_t,
        minv_t,
        state,
        row_ptrs,
        cols,
        check_every,
        precond,
        sbuf_kib,
        core_split,
        matvec=0,
        precision=0,
        algo=0,
        variant=0,
    ):
        """Solve A X = B by (block-Jacobi preconditioned) CG in one launch.

        With algo=1 the same machinery (resident A, column matvec, PSUM dot
        products, LNC=2 row split, device-side while loop) runs power
        iteration instead; see `power_step` below.

        Shapes (fp32), T = block rows, R = right-hand sides:
            blocks_t: (128, n_blocks * 128)  A_idx^T side by side
            minv_t:   (128, T * 128)         D_t^{-1 T} per block row (precond),
                                             else any (128, 128) placeholder
            state:    (128, 2 T R + 2)       [B | X0 | tol^2, max_iter]; vector
                                             element t*128 + p of column c at
                                             [p, t R + c]; the two parameters
                                             replicated on every partition
            row_ptrs, cols: static block structure (T + 1, n_blocks)
            check_every: CG steps per device-side convergence check
            precond:  1 = block-Jacobi z = D^{-1} r, 0 = plain CG
            sbuf_kib: SBUF budget per partition for resident blocks (and,
                      with R > 1, the vectors); the rest stream from HBM
            core_split: 1 = split block rows between the physical cores
                      at LNC=2 (see below), 0 = both run the whole solve
            matvec:   0 = "row": each stored block is the TensorE stationary
                      operand and the vector streams through it (~0.43 us
                      per fp32 block, the weight load); 1 = "column": the
                      vector is stationary and the block streams through
                      (see `block_matvec_col`); R <= 32 only, else row
            precision: A's storage in the column matvec (vectors, dot
                      products and accumulation stay fp32): 0 = fp32;
                      1 = "bf16": bf16 A, (128, n_blocks * 128); 2 =
                      "bf16x2": bf16 hi and lo parts of A, block idx at
                      columns 256 idx (hi) and 256 idx + 128 (lo). The
                      vector is split into bf16 hi + lo on the device, so
                      the products are exact and only A's rounding is
                      lost. Needs the column matvec and R <= 16.
        Residency and the split are kernel arguments (not read from the
        environment at trace time) so they are part of the compile-cache key.
        Returns:
            out: (128, T R + 3 R + 1)
                 [X | iterations (R) | ||r||^2 (R) | r.z (R) | trips]
        One input and one output tensor, so a solve costs one host-to-device
        and one device-to-host copy besides the (reusable) matrix operands.
        The tolerance applies to columns scaled to ||b_c|| = 1.

        Under LNC=2 the two physical cores split the block rows
        (`cg_core_rows`): each holds its rows' blocks in SBUF, computes its
        rows of A p and its share of p.Ap, and the cores exchange twice per
        step with `sendrecv`: the partial p.Ap (both add the same pair, and
        a + b == b + a exactly) and their rows of r (and z) (`share_rz`),
        after which both hold the full r and z and compute r.z, r.r and
        p = z + beta p over the full width with the same instructions. So
        they agree bit for bit on every step size and on when to stop, and
        run the same device-side loop, which neuronx-cc requires of the
        per-core programs. (Power iteration swaps its rows of v instead.)
        core_split=0 runs the whole solve on both cores instead (only core 0
        stores).
        """
        P = _TILE_M
        f32 = nl.float32
        T = len(row_ptrs) - 1
        R = (state.shape[1] - 2) // (2 * T)
        TR = T * R

        n_prog = int(nl.num_programs()) if nl.program_ndim() > 0 else 1
        pid = int(nl.program_id(0)) if n_prog > 1 else 0
        ranges = cg_core_rows(row_ptrs, bool(precond), n_prog, bool(core_split))
        split = len(ranges) == 2
        o0, o1 = ranges[pid] if split else ranges[0]
        w0, w1 = ranges[1 - pid] if split else (0, 0)  # the other core's rows
        b_lo = row_ptrs[o0]  # this core's stored blocks are b_lo .. b_hi - 1
        b_hi = row_ptrs[o1]
        # with R > 1 the vectors (full width on every core) share the budget
        lowp = int(precision) > 0
        parts = 2 if int(precision) == 2 else 1  # stored parts per block of A
        adt = nl.bfloat16 if lowp else f32
        BW = parts * P  # columns per stored block in blocks_t

        def resident(a0, a1):
            return cg_resident_blocks(
                row_ptrs[a1] - row_ptrs[a0],
                T if R > 1 else a1 - a0,
                bool(precond),
                R,
                budget_kib=sbuf_kib,
                block_bytes=BW * (2 if lowp else 4),
            )

        n_res = resident(o0, o1)
        # Does either core stream blocks from HBM? Then the matvec is paced
        # by the stream: the rows keep their order and the column matvec
        # works in chunks of at most 32 rows, which interleave with the
        # stream DMAs better (measured: n = 8192, 512 blocks, one core). Both cores
        # must agree, since they share the vector layout.
        streams = any(resident(a0, a1) < row_ptrs[a1] - row_ptrs[a0] for a0, a1 in ranges)

        out = nl.ndarray((P, TR + 3 * R + 1), dtype=f32, buffer=nl.shared_hbm)

        # ── resident operands ────────────────────────────────────────────
        a_res = None
        if n_res:
            a_res = nl.ndarray((P, n_res * BW), dtype=adt, buffer=nl.sbuf)
            for c0 in range(0, n_res, _CG_STREAM_BLOCKS):
                c1 = min(n_res, c0 + _CG_STREAM_BLOCKS)
                nisa.dma_copy(
                    dst=a_res[:, c0 * BW : c1 * BW],
                    src=blocks_t[:, (b_lo + c0) * BW : (b_lo + c1) * BW],
                )
        m_res = None
        if precond:
            m_res = nl.ndarray((P, (o1 - o0) * P), dtype=f32, buffer=nl.sbuf)
            for c0 in range(o0, o1, _CG_STREAM_BLOCKS):
                c1 = min(o1, c0 + _CG_STREAM_BLOCKS)
                nisa.dma_copy(
                    dst=m_res[:, (c0 - o0) * P : (c1 - o0) * P], src=minv_t[:, c0 * P : c1 * P]
                )

        ones = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.memset(ones, 1.0)
        st = nl.ndarray((P, 2 * TR + 2), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=st, src=state)
        thr = st[:, 2 * TR : 2 * TR + 1]
        cap = st[:, 2 * TR + 1 : 2 * TR + 2]

        # Vectors hold block rows in layout order (`cg_row_order`, per core):
        # position i holds block row lay[i]; b, x0 and x move between the
        # standard order of `state` / `out` and the layout with DMAs (one per
        # run of consecutive rows), alongside the loads of A.
        lay = []
        for a0, a1 in ranges:
            lay += list(range(a0, a1)) if streams else cg_row_order(row_ptrs, a0, a1)
        pos = [0] * T
        for i, m in enumerate(lay):
            pos[m] = i

        def runs(i0, i1):
            """(position, row, count) runs of consecutive rows in [i0, i1)."""
            out_runs = []
            for i in range(i0, i1):
                if out_runs and lay[i] == out_runs[-1][1] + out_runs[-1][2]:
                    out_runs[-1][2] += 1
                else:
                    out_runs.append([i, lay[i], 1])
            return out_runs

        def gather(dst, src, base, i0, i1):
            """dst (layout positions i0..i1-1) <- rows of the standard-layout
            HBM tensor src starting at column base."""
            for i, m, n in runs(i0, i1):
                nisa.dma_copy(
                    dst=dst[:, (i - i0) * R : (i - i0 + n) * R],
                    src=src[:, base + m * R : base + (m + n) * R],
                )

        res = nl.ndarray((P, TR + 3 * R + 1), dtype=f32, buffer=nl.sbuf)
        x = res[:, 0:TR]
        gather(x, state, TR, 0, T)

        def own(v):  # this core's block rows of a layout-order (P, T R) vector
            return v[:, o0 * R : o1 * R]

        W = (o1 - o0) * R  # own width
        bb = nl.ndarray((P, W), dtype=f32, buffer=nl.sbuf)
        gather(bb, state, 0, o0, o1)
        n_own = o1 - o0
        plain = not precond and not algo
        cgcg = bool(variant)  # Chronopoulos-Gear recurrence (see cgcg_step)

        # ── layout of the full-width vectors ─────────────────────────────
        # p (and under the split q and r, or z) hold every block row. Under
        # the split they use a "segment" layout: core 0's rows [0, s), then
        # core 1's rows [s, T), each in a segment of SW columns (WXr R row
        # columns, zero padded to the longer core, then E extra columns). Each
        # core's rows are then one contiguous segment, so an exchange is a
        # single sendrecv of the own segment into the other segment, with no
        # copies, and both cores lay out (and sum) every vector identically.
        # Without the split there is one segment: the standard layout.
        if split:
            seg_lo = (0, ranges[0][1])
            seg_n = (ranges[0][1], T - ranges[0][1])
        else:
            seg_lo, seg_n = (0,), (T,)
        NSEG = len(seg_n)
        col_mv = bool(matvec) and R <= _CG_COL_MAX_RHS
        SPREAD = col_mv and not lowp and R in (1, 2, 4, 8)  # see below

        def ceil4(n):
            return (n + 3) // 4 * 4

        # block rows per segment; under the split with SPREAD a multiple of 4,
        # so that a chunk's rows can be copied out of PSUM whole (the padding
        # rows are 0)
        WXr = ceil4(max(seg_n)) if (SPREAD and split) else max(seg_n)
        SWr = WXr * R  # row columns per segment
        # PCG: the partial r.z and r.r ride in the extra columns of z's segment
        E = 2 * R if (split and precond and not algo) else 0
        SW = SWr + E
        FW = NSEG * SW  # full-layout width (== T R without the split)
        me = pid if split else 0

        def col_pos(i):  # first column of layout position i in the full layout
            g = 1 if (split and i >= seg_lo[1]) else 0
            return g * SW + (i - seg_lo[g]) * R

        def col(j):  # first column of block row j in the full layout
            return col_pos(pos[j])

        assert col_mv or not lowp, "bf16 A needs the column matvec"
        assert not lowp or 2 * R <= 32, "bf16 A takes at most 16 right-hand sides"
        # "Spread" matvec sources (column matvec, fp32 A, R in 1, 2, 4, 8):
        # each (P, R) block column of p (and x0, and r for the preconditioner)
        # is stored after (G - 1) R zero columns, so the stationary operand
        # for slot c (c < G) of a 4 G-row chunk can be the (c + 1) R columns
        # ending at the block: c R zeros, then the block. Own row 4 c + g of
        # the chunk then lands at PSUM partitions 32 g + c R + k of one
        # 128-column region (the zeros add 0 to the other slots' rows), so a
        # chunk needs one PSUM-to-SBUF copy and one TensorE transpose, against
        # one of each per four rows without it. G (a power of two, G R <= 32)
        # is the smallest that fits either core's rows in one chunk, within
        # _CG_SPREAD_BYTES per spread vector. The zero gaps are never written.
        SX = 1
        if SPREAD:
            SX = 4
            while (
                (not streams or SX < 8)
                and 4 * SX < WXr  # the longer core's rows: both cores use one layout
                and 2 * SX * R <= 32
                and 2 * SX * FW * 4 <= _CG_SPREAD_BYTES
            ):
                SX *= 2
        CROWS = 4 * SX if SPREAD else _CG_COL_ROWS  # block rows per chunk

        def vec(width=FW):
            return nl.ndarray((P, width), dtype=f32, buffer=nl.sbuf)

        def svec(width=FW):
            """A matvec source vector (spread under SPREAD), zeroed: pads and
            gaps must stay 0, and a deferred update may read it (times a zero
            step size) before it is first written."""
            t = vec(SX * width)
            nisa.memset(t, 0.0)
            return t

        def d3(v):  # (P, K, R) view of a dense (P, K R) vector or slice
            return v.reshape((P, v.shape[1] // R, R))

        def sv(v, a, b):
            """(P, (b - a) / R, R) view of dense columns [a, b) of a source
            vector (block aligned)."""
            if not SPREAD:
                return v[:, a:b].reshape((P, (b - a) // R, R))
            G = SX
            return v[:, G * a : G * b].reshape((P, (b - a) // R, G * R))[:, :, (G - 1) * R :]

        def rows3(v, src=False):  # this core's rows of a full-layout vector
            if src:
                return sv(v, me * SW, me * SW + W)
            return d3(v[:, me * SW : me * SW + W])

        def full3(v, src=False):  # every column of a full-layout vector
            return sv(v, 0, FW) if src else d3(v)

        def seg(v, g, src=False):  # segment g of a full-layout vector
            s = SX if src else 1
            return v[:, s * g * SW : s * (g + 1) * SW]

        def to_full(dst, src):
            """Standard-layout (P, T R) src -> full-layout source vector dst
            (pads and gaps zero: svec memsets it)."""
            for g in range(NSEG):
                a0, a1 = seg_lo[g] * R, (seg_lo[g] + seg_n[g]) * R
                nisa.tensor_copy(sv(dst, g * SW, g * SW + a1 - a0), d3(src[:, a0:a1]))

        # With an even check interval p alternates between two tiles, so the
        # update x += alpha p of step k can wait until step k + 1 has started
        # its matvec (off the critical path); it is flushed after the loop.
        defer_x = not algo and not cgcg and check_every % 2 == 0
        p_bufs = [svec(), svec()] if defer_x else [svec()]
        p = p_bufs[0]
        junk = vec(FW)
        tmp = vec(FW) if R > 1 else None
        # matvec outputs: own rows, padded to a multiple of 4 rows with SPREAD
        Wp = ceil4(n_own) * R if SPREAD else W
        q_sb = vec(Wp) if R > 1 else None
        z_sb = vec(Wp) if (R > 1 and precond) else None
        r_full = q_full = z_full = None
        if precond and not algo:
            r = svec(W)  # own rows; the preconditioner's matvec source
        else:
            r = vec(W)
        if split:
            if plain:
                # q is exchanged and r updated over the full width (one
                # exchange per step); pads stay zero
                q_full = vec()
                r_full = vec()
                nisa.memset(q_full, 0.0)
                nisa.memset(r_full, 0.0)
                r = r_full[:, me * SW : me * SW + W]
            elif precond and not algo:
                z_full = vec()
                nisa.memset(z_full, 0.0)
            pq_send = nl.ndarray((P, 3 * R), dtype=f32, buffer=nl.sbuf)
            pq_recv = nl.ndarray((P, 3 * R), dtype=f32, buffer=nl.sbuf)
        elif plain:
            r_full = r
        r3 = sv(r, 0, W) if (precond and not algo) else d3(r)
        # Chronopoulos-Gear CG (variant=1, plain CG only): r is the matvec
        # source (full layout, two tiles alternating with an even check
        # interval), w = A r and s = A p are full width, p own rows.
        if cgcg:
            assert plain, "Chronopoulos-Gear CG: no preconditioner (and not power iteration)"
            cg_defer = check_every % 2 == 0
            r_bufs = [svec(), svec()] if cg_defer else [svec()]
            w_full = vec()
            s_full = vec()
            p_own = vec(W)
            for t in (w_full, s_full, p_own):
                nisa.memset(t, 0.0)

        def scal():  # one value per right-hand side, replicated on partitions
            return nl.ndarray((P, R), dtype=f32, buffer=nl.sbuf)

        rr = scal()
        # plain CG: r.z is r.r, one tile (two copies of one PSUM sum into two
        # tiles get merged by neuronx-cc, and the merged write can be lost)
        rz = rr if plain else scal()
        cnt = scal()
        act = scal()
        trips = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(cnt, 0.0)
        nisa.memset(trips, 0.0)
        # (P, R) views of the replicated (P, 1) parameters
        thr_r = thr if R == 1 else thr.broadcast(1, R)
        cap_r = cap if R == 1 else cap.broadcast(1, R)

        # Block rows per PSUM bank (2 KiB = 512 fp32 per partition).
        rows_per_bank = max(1, min(n_own, _TILE_N // R))

        # PSUM: four tiles per scope (the setup, and each trip of the device
        # loop; PSUM tiles cannot live across the loop), re-zeroed on every
        # use, rather than fresh tiles in every unrolled step, which ran NKI's
        # allocator out of PSUM (eight 2 KiB banks) so that the whole kernel
        # fell back to neuronx-cc allocation and scheduling.
        pss = {}

        def psum_scope():
            pss["mv"] = nl.ndarray((P, _TILE_N), dtype=f32, buffer=nl.psum)
            pss["pt"] = nl.ndarray((P, _TILE_N), dtype=f32, buffer=nl.psum)
            pss["dot"] = [nl.ndarray((P, 3 * R), dtype=f32, buffer=nl.psum) for _ in range(2)]
            pss["n"] = 0

        def psum_zeros(name, width):
            t = pss[name]
            if name == "dot":
                t = t[pss["n"] % 2]
                pss["n"] += 1
            v = t[:, 0:width]
            nisa.memset(v, 0.0)
            return v

        def block_matvec(dst, src, mats, ptrs, cl, cof):
            """Own rows of M @ src for a static block structure; block column
            j of src starts at column `cof(j)`. M's blocks come from
            `mats(idx)` (an SBUF (128,128) view, A^T layout). With one
            right-hand side the product stays in PSUM (and is returned) when
            dst is None: VectorE reads PSUM operands directly, which saves a
            copy on the critical path. Otherwise it goes to `dst` in SBUF one
            PSUM bank at a time."""
            if R == 1 and dst is None:
                ps = psum_zeros("mv", n_own)
                for i in range(o0, o1):
                    m = lay[i]
                    for idx in range(ptrs[m], ptrs[m + 1]):
                        c = cof(cl[idx])
                        nisa.nc_matmul(
                            ps[:, i - o0 : i - o0 + 1],
                            mats(idx),
                            src[:, c : c + 1],
                            accumulate=True,
                        )
                return ps
            for m0 in range(o0, o1, rows_per_bank):
                m1 = min(o1, m0 + rows_per_bank)
                ps = psum_zeros("mv", (m1 - m0) * R)
                for i in range(m0, m1):
                    o = (i - m0) * R
                    m = lay[i]
                    for idx in range(ptrs[m], ptrs[m + 1]):
                        c = cof(cl[idx])
                        nisa.nc_matmul(
                            ps[:, o : o + R],
                            mats(idx),
                            src[:, c : c + R],
                            accumulate=True,
                        )
                nisa.tensor_copy(dst[:, (m0 - o0) * R : (m1 - o0) * R], ps)
            return dst[:, 0:W]

        if algo and q_sb is None:  # power iteration reads A v twice (q.q)
            q_sb = vec(Wp)
        if lowp:
            # bf16 hi / lo parts of the stationary vector: the block column at
            # column c of src at [2c, 2c + R) (hi) and [2c + R, 2c + 2R) (lo)
            src_b = nl.ndarray((P, 2 * FW), dtype=nl.bfloat16, buffer=nl.sbuf)
            src_res = vec()
        if col_mv:
            # q_sb / z_sb: the column matvec always lands in SBUF
            if q_sb is None:
                q_sb = vec(Wp)
            if precond and z_sb is None:
                z_sb = vec(Wp)
            q_row = nl.ndarray((P, 4 * P), dtype=f32, buffer=nl.sbuf)

        def split_bf16(src):
            """src (fp32, full layout) -> src_b = bf16 hi | lo per block column.
            hi = bf16(src), lo = bf16(src - hi): src to ~2^-17 relative."""
            K = FW // R
            sb3 = src_b.reshape((P, K, 2 * R))
            s3 = src.reshape((P, K, R))
            nisa.tensor_copy(sb3[:, :, 0:R], s3)
            nisa.tensor_tensor(src_res.reshape((P, K, R)), s3, sb3[:, :, 0:R], op=nl.subtract)
            nisa.tensor_copy(sb3[:, :, R : 2 * R], src_res.reshape((P, K, R)))
            return src_b

        def block_matvec_col(dst, src, mats, ptrs, cl, order, cof, low=False):
            """Own rows of M @ src into dst (SBUF), vector-stationary.

            For block (m, j) the stationary operand is src's (128, R) block j
            (a 1..32-column weight load, at column `cof(j)`) and the moving
            operand is the stored block itself, A_idx^T: dst = p_j^T A_idx^T =
            (A_idx p_j)^T, a row. The fp32 cost is the 128 moving columns, not
            a 128x128 weight load. Column tiling (tile_size (128, 32)) splits
            the PE array into four 32-column tiles that run concurrently, so
            blocks are issued round-robin over four row groups: own row
            i = 4 c + g of a 16-row chunk goes to tile g. Without SPREAD it
            lands at PSUM partition 32 g (+ k for right-hand side k), free
            offset 128 c, so a chunk fills four 128-column slots, each copied
            to SBUF and transposed back to the column layout on TensorE
            (bit-exact). With SPREAD (see above) it lands at partition
            32 g + c R + k of a single slot: one copy and one transpose per
            chunk. Then the rows are copied into dst.
            `order(m0, m1)` gives the block issue order for rows m0..m1-1.
            With `low` (bf16 A) `mats(idx)` gives the block's `parts` bf16
            parts side by side, the stationary operand is the bf16 hi | lo
            split of src (2R columns), and the hi and lo rows are added in
            the final copy."""
            SWl = 2 * R if low else R
            stat = split_bf16(src) if low else src
            for m0 in range(o0, o1, CROWS):
                m1 = min(o1, m0 + CROWS)
                nr = m1 - m0
                nc = (nr + 3) // 4  # slots used (4 block rows each)
                nsl = 1 if SPREAD else nc  # 128-column PSUM slots
                ps = psum_zeros("mv", nsl * P)
                for m, idx in order(m0, m1, ptrs):
                    i = m - m0
                    g, c = i % 4, i // 4
                    blk = mats(idx)
                    if SPREAD:
                        e = SX * (cof(cl[idx]) + R)  # end of the block's spread group
                        nisa.nc_matmul(
                            ps[32 * g : 32 * g + (c + 1) * R, 0:P],
                            src[:, e - (c + 1) * R : e],
                            blk,
                            accumulate=True,
                            tile_position=(0, 32 * g),
                            tile_size=(P, 32),
                        )
                        continue
                    j0 = cof(cl[idx]) * (2 if low else 1)
                    for part in range(parts if low else 1):
                        nisa.nc_matmul(
                            ps[32 * g : 32 * g + SWl, c * P : (c + 1) * P],
                            stat[:, j0 : j0 + SWl],
                            blk[:, part * P : (part + 1) * P],
                            accumulate=True,
                            tile_position=(0, 32 * g),
                            tile_size=(P, 32),
                        )
                nisa.tensor_copy(q_row[:, 0 : nsl * P], ps)
                pt = pss["pt"][:, 0 : nsl * P]  # the transposes fill it
                for c in range(nsl):
                    nisa.nc_transpose(
                        pt[:, c * P : (c + 1) * P],
                        q_row[:, c * P : (c + 1) * P],
                        engine=nisa.tensor_engine,
                    )
                d0 = (m0 - o0) * R
                if SPREAD:
                    # pt[:, 32 g + c R + k] = row 4 c + g, right-hand side k
                    pg = pt.reshape((P, 4, 32 // R, R))
                    # whole groups of four rows when dst has room for the
                    # chunk's padding rows (0 in PSUM): one copy; else two
                    nf = (nr + 3) // 4 if d0 + 4 * ((nr + 3) // 4) * R <= dst.shape[1] else nr // 4
                    if nf:
                        nisa.tensor_copy(
                            dst[:, d0 : d0 + 4 * nf * R]
                            .reshape((P, nf, 4, R))
                            .permute((0, 2, 1, 3)),
                            pg[:, :, 0:nf, :],
                        )
                    if 4 * nf < nr:
                        a = d0 + 4 * nf * R
                        nisa.tensor_copy(
                            dst[:, a : a + (nr - 4 * nf) * R].reshape((P, nr - 4 * nf, R)),
                            pg[:, 0 : nr - 4 * nf, nf, :],
                        )
                    continue
                # pt[:, 128 c + 32 g + k] = row 4 c + g, right-hand side k
                pt3 = pt.reshape((P, nc * 4, 32))
                d3v = dst[:, d0 : d0 + nr * R].reshape((P, nr, R))
                nisa.tensor_copy(d3v, pt3[:, 0:nr, 0:R])
                if low:  # + the products with the vector's bf16 lo half
                    nisa.tensor_tensor(d3v, d3v, pt3[:, 0:nr, R : 2 * R], op=nl.add)
            return dst[:, 0:W]

        def col_order(stream_of):
            """Round-robin over the four row groups of a chunk, keeping the
            streamed blocks' DMA chunks in increasing order."""

            def order(m0, m1, ptrs):
                # (layout position, block index) pairs of positions m0..m1-1
                lists = [[] for _ in range(4)]
                for i in range(m0, m1):
                    m = lay[i]
                    for idx in range(ptrs[m], ptrs[m + 1]):
                        lists[(i - m0) % 4].append((i, idx))
                seq = []
                for k in range(max(len(v) for v in lists)):
                    for v in lists:
                        if k < len(v):
                            seq.append(v[k])
                pos = {e: n for n, e in enumerate(seq)}
                return sorted(seq, key=lambda e: (stream_of(e[1]), pos[e]))

            return order

        def a_matvec(dst, src):
            """Own rows of A @ src (src a full-layout source vector)."""
            chunk = {}

            def mats(idx):
                i = idx - b_lo
                if i < n_res:
                    return a_res[:, i * BW : (i + 1) * BW]
                c0 = n_res + ((i - n_res) // _CG_STREAM_BLOCKS) * _CG_STREAM_BLOCKS
                if chunk.get("c0") != c0:
                    c1 = min(b_hi - b_lo, c0 + _CG_STREAM_BLOCKS)
                    t = nl.ndarray((P, (c1 - c0) * BW), dtype=adt, buffer=nl.sbuf)
                    nisa.dma_copy(dst=t, src=blocks_t[:, (b_lo + c0) * BW : (b_lo + c1) * BW])
                    chunk["c0"] = c0
                    chunk["t"] = t
                i0 = (i - c0) * BW
                return chunk["t"][:, i0 : i0 + BW]

            if col_mv:

                def stream_of(idx):
                    i = idx - b_lo
                    return -1 if i < n_res else (i - n_res) // _CG_STREAM_BLOCKS

                return block_matvec_col(
                    dst, src, mats, row_ptrs, cols, col_order(stream_of), col, low=lowp
                )
            return block_matvec(dst, src, mats, row_ptrs, cols, col)

        diag_ptrs = tuple(range(T + 1))
        diag_cols = tuple(range(T))

        def precondition(dst):
            """Own rows of z = D^{-1} r into dst (PSUM for R = 1 when dst is
            None and the row matvec is used).

            r holds only own rows, so the diagonal blocks address it with a
            local column index."""
            mats = lambda t: m_res[:, (t - o0) * P : (t - o0 + 1) * P]  # noqa: E731
            cof = lambda j: (pos[j] - o0) * R  # noqa: E731
            if col_mv:
                return block_matvec_col(
                    dst, r, mats, diag_ptrs, diag_cols, col_order(lambda idx: -1), cof
                )
            return block_matvec(dst, r, mats, diag_ptrs, diag_cols, cof)

        def exchange(src, dst):
            """Swap src for the other core's src (into dst). Up to 1 KiB per
            partition goes through GpSimd's internal SBUF-to-SBUF DMA, about
            0.9 us for a few columns against 2.4 us for the default DMA path
            (it falls back to that above the limit)."""
            fast = src.shape[1] * 4 <= _CG_GPSIMD_DMA_BYTES
            nisa.sendrecv(
                src=src,
                dst=dst,
                send_to_rank=1 - pid,
                recv_from_rank=1 - pid,
                pipe_id=0,
                dma_engine=nisa.dma_engine.gpsimd_dma if fast else nisa.dma_engine.dma,
            )

        def share(v, src=False):
            """Under the split, swap this core's segment of a full-layout v for
            the other core's: both then hold all of v, bit for bit."""
            if split:
                exchange(seg(v, me, src), seg(v, 1 - me, src))

        def partial_dots(pairs):
            """(P, k R) PSUM tile: per-column sum(u_i * v_i) over the given
            (P, K, R) views, one matmul with a ones matrix to sum (and
            replicate) over partitions."""
            k = len(pairs)
            parts = nl.ndarray((P, k * R), dtype=f32, buffer=nl.sbuf)
            for i, (u, v) in enumerate(pairs):
                K = u.shape[1]
                jk = nl.ndarray((P, K, R), dtype=f32, buffer=nl.sbuf)
                if R == 1:
                    nisa.scalar_tensor_tensor(
                        dst=jk,
                        data=u,
                        op0=nl.multiply,
                        operand0=1.0,
                        op1=nl.multiply,
                        operand1=v,
                        reduce_res=parts[:, i : i + 1],
                    )
                else:
                    nisa.tensor_tensor(jk, u, v, op=nl.multiply)
                    nisa.tensor_reduce(
                        parts[:, i * R : (i + 1) * R], nl.add, jk.permute((0, 2, 1)), axis=2
                    )
            ps = psum_zeros("dot", k * R)
            nisa.nc_matmul(ps, ones, parts, accumulate=True, is_stationary_onezero=True)
            return ps

        def dots(pairs):
            """Sums over this core's rows: under the split the cores exchange
            their partial sums and add them (a + b == b + a, so both get the
            same bits). Returns (P, k R)."""
            ps = partial_dots(pairs)
            if not split:
                return ps
            kR = len(pairs) * R
            nisa.tensor_copy(pq_send[:, 0:kR], ps)
            exchange(pq_send[:, 0:kR], pq_recv[:, 0:kR])
            tot = nl.ndarray((P, kR), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(tot, pq_send[:, 0:kR], pq_recv[:, 0:kR], op=nl.add)
            return tot

        def full_dots(pairs):
            """Sums over full-layout operands every core holds in full (pads
            are zero): no exchange, the same bits on both cores."""
            return partial_dots(pairs)

        rz_act = scal()

        def update_active():
            # act = (rr > thr) * (cnt < cap), as 1.0 / 0.0 per column; and
            # rz_act = rz * act for the next step size (off the critical path)
            g = scal()
            lt = scal()
            nisa.tensor_tensor(g, rr, thr_r, op=nl.greater)
            nisa.tensor_tensor(lt, cnt, cap_r, op=nl.less)
            nisa.tensor_tensor(act, g, lt, op=nl.multiply)
            nisa.tensor_tensor(rz_act, rz, act, op=nl.multiply)

        def step_size(dst, pq):
            """dst = act * rz / pq = min(1 / pq, 1 / tiny) * rz_act: the same
            bits as `safe_ratio` for pq >= 0, one VectorE op fewer on the
            critical path (rz_act is ready before p.q is)."""
            inv = scal()
            nisa.reciprocal(inv, pq)
            if R == 1:
                nisa.tensor_scalar(
                    dst,
                    inv,
                    op0=nl.minimum,
                    operand0=1.0 / _CG_TINY,
                    op1=nl.multiply,
                    operand1=rz_act,
                )
            else:
                nisa.tensor_scalar(dst, inv, op0=nl.minimum, operand0=1.0 / _CG_TINY)
                nisa.tensor_tensor(dst, dst, rz_act, op=nl.multiply)

        def safe_ratio(dst, num, den):
            # dst = act * num / max(den, tiny)   (reciprocal on VectorE)
            dm = scal()
            inv = scal()
            nisa.tensor_scalar(dm, den, op0=nl.maximum, operand0=_CG_TINY)
            nisa.reciprocal(inv, dm)
            if R == 1:
                nisa.scalar_tensor_tensor(
                    dst=dst, data=inv, op0=nl.multiply, operand0=num, op1=nl.multiply, operand1=act
                )
            else:
                nisa.tensor_tensor(dst, inv, num, op=nl.multiply)
                nisa.tensor_tensor(dst, dst, act, op=nl.multiply)

        def axpy(dst, a, xv, yv, op):
            """dst = yv + a * xv (op=nl.add) or yv - a * xv (op=nl.subtract)
            over (P, K, R) views."""
            if R == 1:
                nisa.scalar_tensor_tensor(
                    dst=dst,
                    data=xv,
                    op0=nl.multiply,
                    operand0=a,
                    op1=op,
                    operand1=yv,
                    reverse1=op == nl.subtract,
                )
                return
            K = dst.shape[1]
            t3 = tmp[:, 0 : K * R].reshape((P, K, R))
            a3 = a.reshape((P, 1, R)).broadcast(1, K)
            nisa.tensor_tensor(t3, xv, a3, op=nl.multiply)
            nisa.tensor_tensor(dst, yv, t3, op=op)

        def power_init():
            """Power iteration (algo=1, R = 1, no preconditioner): the iterate v
            lives in p (full layout, from the B slot of `state`, unit norm),
            rz holds the last Rayleigh quotient and rr its last squared
            change; thr is tol^2 on that change."""
            v0 = vec(TR)
            gather(v0, state, 0, 0, T)
            to_full(p, v0)
            nisa.memset(rz, 0.0)
            nisa.memset(rr, 0.0)
            nisa.tensor_tensor(act, cnt, cap_r, op=nl.less)

        def power_step():
            """v <- A v / ||A v||, lam = v.Av / v.v (Rayleigh quotient);
            continue while (lam - lam_prev)^2 >= tol^2 and under the cap.
            Masked like the CG step: an inactive iterate does not change."""
            q = a_matvec(q_sb, p)
            pv, q3 = rows3(p, src=True), d3(q)
            tot = dots([(pv, q3), (q3, q3), (pv, pv)])
            vq, qq, vv = tot[:, 0:1], tot[:, 1:2], tot[:, 2:3]
            lam = scal()
            safe_ratio(lam, vq, vv)  # act * vq / vv
            keep = scal()  # 1 - act
            nisa.tensor_scalar(keep, act, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            # lam = act * lam_new + (1 - act) * lam_prev; d = lam - lam_prev
            nisa.scalar_tensor_tensor(
                dst=lam, data=rz, op0=nl.multiply, operand0=keep, op1=nl.add, operand1=lam
            )
            d = scal()
            nisa.tensor_tensor(d, lam, rz, op=nl.subtract)
            nisa.tensor_tensor(d, d, d, op=nl.multiply)
            # rr keeps the last active step's d^2 (a masked step has d = 0)
            nisa.scalar_tensor_tensor(
                dst=rr, data=rr, op0=nl.multiply, operand0=keep, op1=nl.add, operand1=d
            )
            # s = act / sqrt(max(qq, tiny)); v = keep * v + s * q (own rows)
            sq = scal()
            nisa.tensor_scalar(sq, qq, op0=nl.maximum, operand0=_CG_TINY)
            nisa.activation(dst=sq, op=nl.sqrt, data=sq)
            sc = scal()
            nisa.reciprocal(sc, sq)
            nisa.tensor_tensor(sc, sc, act, op=nl.multiply)
            jw = junk[:, 0:W]
            nisa.tensor_scalar(jw, q, op0=nl.multiply, operand0=sc)
            nisa.scalar_tensor_tensor(
                dst=pv, data=pv, op0=nl.multiply, operand0=keep, op1=nl.add, operand1=d3(jw)
            )
            share(p, src=True)
            nisa.tensor_tensor(cnt, cnt, act, op=nl.add)
            nisa.tensor_copy(rz, lam)
            # act = (d^2 >= thr) * (cnt < cap); a masked step leaves d = 0
            g = scal()
            lt = scal()
            nisa.tensor_tensor(g, d, thr_r, op=nl.greater_equal)
            nisa.tensor_tensor(lt, cnt, cap_r, op=nl.less)
            nisa.tensor_tensor(g, g, lt, op=nl.multiply)
            nisa.tensor_tensor(act, g, act, op=nl.multiply)

        def precond_dots():
            """PCG: z = D^{-1} r (own rows) and the full r.z, r.r. Under the
            split z goes straight into this core's segment of z_full, the
            partial sums into its extra columns, and one exchange of the
            segment gives both cores all of z and the other core's partial
            sums; seg 0's + seg 1's sums are then the same bits on both cores.
            Returns (z full layout, (P, 2R) tile [r.z | r.r])."""
            if not split:
                z = precondition(z_sb)
                return z, dots([(r3, d3(z)), (r3, r3)])
            z = precondition(z_full[:, me * SW : me * SW + SWr])
            ps = partial_dots([(r3, d3(z)), (r3, r3)])
            nisa.tensor_copy(z_full[:, me * SW + SWr : (me + 1) * SW], ps)
            share(z_full)
            tot = nl.ndarray((P, 2 * R), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(tot, z_full[:, SWr:SW], z_full[:, SW + SWr : 2 * SW], op=nl.add)
            return z_full, tot

        # ── r0 = b - A x0, z0 = M r0, p0 = z0 ────────────────────────────
        def cg_init():
            xs = x
            if split or SPREAD:
                xs = svec()
                to_full(xs, x)
            q0 = a_matvec(q_sb, xs)
            nisa.tensor_tensor(r3, d3(bb), d3(q0), op=nl.subtract)
            if plain:
                share(r_full)
                nisa.tensor_copy(full3(p, src=True), full3(r_full))
                tot = full_dots([(full3(r_full), full3(r_full))])
                nisa.tensor_copy(rr, tot)  # rz is rr in plain CG
            else:
                zf, tot = precond_dots()
                # (p's extra columns, past the rows, are never read)
                nisa.tensor_copy(full3(p, src=True), full3(zf))
                nisa.tensor_copy(rz, tot[:, 0:R])
                nisa.tensor_copy(rr, tot[:, R : 2 * R])
            update_active()

        if cgcg:
            alpha_c = scal()
            beta_c = scal()
            gact = scal()  # gamma * act
            c_t = scal()  # beta gamma / alpha_prev

        def cgcg_init():
            """r0 = b - A x0, w0 = A r0, gamma0 = r0.r0, delta0 = w0.r0,
            alpha0 = gamma0 / delta0, beta0 = 0, s0 = w0, p = 0."""
            xs = svec()
            to_full(xs, x)
            q0 = a_matvec(q_sb, xs)
            r0 = r_bufs[0]
            nisa.tensor_tensor(rows3(r0, src=True), d3(bb), d3(q0), op=nl.subtract)
            share(r0, src=True)
            a_matvec(w_full[:, me * SW : me * SW + SWr], r0)
            share(w_full)
            rf = full3(r0, src=True)
            tot = full_dots([(rf, rf), (d3(w_full), rf)])
            nisa.tensor_copy(rr, tot[:, 0:R])
            update_active()
            safe_ratio(alpha_c, tot[:, 0:R], tot[:, R : 2 * R])
            nisa.memset(beta_c, 0.0)
            nisa.tensor_copy(s_full, w_full)

        def cgcg_step(k):
            """One masked Chronopoulos-Gear step (k-th of the trip): one
            reduction phase per step instead of two. With alpha, beta, s from
            the previous step:
                r' = r - alpha s;  p = r + beta p;  x += alpha p
                w = A r';  gamma' = r'.r';  delta = w.r'
                beta' = gamma' / gamma;  alpha' = gamma' / (delta - beta' gamma' / alpha)
                s' = w + beta' s
            gamma' (and so beta') is ready while the matvec runs; only delta
            (one dot) sits between the matvec and the next r update. Under
            the split the cores swap their segments of w (one exchange) and
            then compute everything over the full width identically."""
            r_old = r_bufs[k % len(r_bufs)]
            r_new = r_bufs[(k + 1) % len(r_bufs)]
            rf_old, rf = full3(r_old, src=True), full3(r_new, src=True)
            po, ox = d3(p_own), d3(own(x))

            def p_and_x():
                axpy(po, beta_c, po, rows3(r_old, src=True), nl.add)  # p = r + beta p
                axpy(ox, alpha_c, po, ox, nl.add)  # x += alpha p

            if not cg_defer:  # r is updated in place: p first
                p_and_x()
            axpy(rf, alpha_c, d3(s_full), rf_old, nl.subtract)  # r' = r - alpha s
            if cg_defer:  # off the critical path, while TensorE runs the matvec
                p_and_x()
            g_parts = nl.ndarray((P, R), dtype=f32, buffer=nl.sbuf)
            jg = nl.ndarray((P, FW // R, R), dtype=f32, buffer=nl.sbuf)
            if R == 1:
                nisa.scalar_tensor_tensor(
                    dst=jg,
                    data=rf,
                    op0=nl.multiply,
                    operand0=1.0,
                    op1=nl.multiply,
                    operand1=rf,
                    reduce_res=g_parts,
                )
            else:
                nisa.tensor_tensor(jg, rf, rf, op=nl.multiply)
                nisa.tensor_reduce(g_parts, nl.add, jg.permute((0, 2, 1)), axis=2)
            # w = A r' (own rows), then the exchange
            a_matvec(w_full[:, me * SW : me * SW + SWr], r_new)
            share(w_full)
            gps = psum_zeros("dot", R)
            nisa.nc_matmul(gps, ones, g_parts, accumulate=True, is_stationary_onezero=True)
            # beta' = act' gamma' / gamma, c = beta' gamma' / alpha (old gamma = rr)
            inv_g = scal()
            nisa.tensor_scalar(inv_g, rr, op0=nl.maximum, operand0=_CG_TINY)
            nisa.reciprocal(inv_g, inv_g)
            inv_a = scal()
            nisa.tensor_scalar(inv_a, alpha_c, op0=nl.maximum, operand0=_CG_TINY)
            nisa.reciprocal(inv_a, inv_a)
            nisa.tensor_copy(rr, gps)
            nisa.tensor_tensor(cnt, cnt, act, op=nl.add)
            update_active()  # act' from gamma' (rz_act = gamma' act')
            nisa.tensor_tensor(beta_c, rz_act, inv_g, op=nl.multiply)
            nisa.tensor_tensor(c_t, beta_c, rr, op=nl.multiply)
            nisa.tensor_tensor(c_t, c_t, inv_a, op=nl.multiply)
            # delta = w . r', then alpha' = min(1 / (delta - c), 1 / tiny) gamma' act'
            dps = full_dots([(d3(w_full), rf)])
            sf = d3(s_full)
            axpy(sf, beta_c, sf, d3(w_full), nl.add)  # s' = w + beta' s
            tden = scal()
            nisa.tensor_tensor(tden, dps, c_t, op=nl.subtract)
            nisa.reciprocal(tden, tden)
            if R == 1:
                nisa.tensor_scalar(
                    alpha_c,
                    tden,
                    op0=nl.minimum,
                    operand0=1.0 / _CG_TINY,
                    op1=nl.multiply,
                    operand1=rz_act,
                )
            else:
                nisa.tensor_scalar(alpha_c, tden, op0=nl.minimum, operand0=1.0 / _CG_TINY)
                nisa.tensor_tensor(alpha_c, alpha_c, rz_act, op=nl.multiply)

        psum_scope()
        if algo:
            assert R == 1 and not precond, "power iteration: one vector, no preconditioner"
            power_init()
        elif cgcg:
            cgcg_init()
        else:
            cg_init()

        alpha_prev = None
        if defer_x:
            alpha_prev = scal()
            nisa.memset(alpha_prev, 0.0)

        def cg_step(k):
            """One masked CG step (k-th of the trip). Plain CG under the split
            exchanges once: each core computes its rows of q = A p straight
            into its segment of q_full and the cores swap segments; from there
            both compute p.q, r -= alpha q, r.r and p = r + beta p over the
            full width with the same instructions on the same data (pads are
            zero). PCG exchanges the partial p.q (a few columns) and then z's
            segments with the partial r.z, r.r (`precond_dots`)."""
            p = p_bufs[k % len(p_bufs)]
            p_new = p_bufs[(k + 1) % len(p_bufs)]
            ox = d3(own(x))
            if defer_x:  # last step's x += alpha p, while TensorE runs this matvec
                axpy(ox, alpha_prev, rows3(p_new, src=True), ox, nl.add)
                alpha = alpha_prev
            else:
                alpha = scal()
            beta = scal()
            if plain:
                q = a_matvec(q_sb if not split else q_full[:, me * SW : me * SW + SWr], p)
                if split:
                    share(q_full)
                    q = q_full
                pf, q3, rf = full3(p, src=True), d3(q), full3(r_full)
                pq = full_dots([(pf, q3)])
                step_size(alpha, pq)
                axpy(rf, alpha, q3, rf, nl.subtract)  # r -= alpha q
                tot = full_dots([(rf, rf)])
                safe_ratio(beta, tot, rz)
                if not defer_x:
                    axpy(ox, alpha, rows3(p, src=True), ox, nl.add)  # x += alpha p
                # in program order: the p update (which the next matvec waits
                # for) ahead of the bookkeeping on VectorE
                with nl.no_reorder():
                    axpy(full3(p_new, src=True), beta, pf, rf, nl.add)  # p = r + beta p
                    nisa.tensor_tensor(cnt, cnt, act, op=nl.add)
                    nisa.tensor_copy(rr, tot)
                    update_active()
                return
            q = a_matvec(q_sb, p)
            q3 = d3(q)
            pq = dots([(rows3(p, src=True), q3)])
            step_size(alpha, pq)
            axpy(r3, alpha, q3, r3, nl.subtract)  # r -= alpha q
            zf, tot = precond_dots()
            rz_new = tot[:, 0:R]
            safe_ratio(beta, rz_new, rz)
            if not defer_x:
                axpy(ox, alpha, rows3(p, src=True), ox, nl.add)  # x += alpha p
            # p = z + beta p over the full width (p's extra columns, past the
            # rows, are never read)
            with nl.no_reorder():
                axpy(full3(p_new, src=True), beta, full3(p, src=True), full3(zf), nl.add)
                nisa.tensor_tensor(cnt, cnt, act, op=nl.add)
                nisa.tensor_copy(rz, rz_new)
                nisa.tensor_copy(rr, tot[:, R : 2 * R])
                update_active()

        def flag_register():
            # continue while any column is active
            anyact = act
            if R > 1:
                anyact = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_reduce(anyact, nl.max, act, axis=1)
            fi = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(fi, anyact[0:1, 0:1])
            reg = nisa.register_alloc()
            nisa.register_load(reg, fi)
            return reg

        def trip(_r):
            psum_scope()
            for k in range(check_every):
                if algo:
                    power_step()
                else:
                    (cgcg_step if cgcg else cg_step)(k)
            nisa.tensor_scalar(trips, trips, op0=nl.add, operand0=1.0)
            return flag_register()

        nl.while_loop(flag_register(), trip)
        if defer_x:  # the last step's x += alpha p (its p is in p_bufs[1])
            axpy(d3(own(x)), alpha_prev, rows3(p_bufs[1], src=True), d3(own(x)), nl.add)

        if algo:  # the iterate is returned in the X slot
            nisa.tensor_copy(d3(own(x)), rows3(p, src=True))
        nisa.tensor_copy(res[:, TR : TR + R], cnt)
        nisa.tensor_copy(res[:, TR + R : TR + 2 * R], rr)
        nisa.tensor_copy(res[:, TR + 2 * R : TR + 3 * R], rz)
        nisa.tensor_copy(res[:, TR + 3 * R : TR + 3 * R + 1], trips)
        # x back to the standard order (each core its own rows)
        if split or pid == 0:
            for i, m, n in runs(o0, o1):
                nisa.dma_copy(dst=out[:, m * R : (m + n) * R], src=res[:, i * R : (i + n) * R])
        if pid == 0:
            nisa.dma_copy(dst=out[:, TR:], src=res[:, TR:])
        return out

    @nki.jit
    def _power_bsr_keyed_kernel(
        blocks_t, state, key, check_every, sbuf_kib, core_split, matvec, precision
    ):
        """Fused power iteration (see `_cg_bsr_body`, algo=1). state is
        (128, 2 T + 2): [v0 (unit norm) | unused | tol^2, max_iter]; out is
        (128, T + 4): [v | iterations | (last change of lam)^2 | lam | trips]."""
        return _cg_bsr_body(
            blocks_t,
            blocks_t,
            state,
            *_STRUCTURES[key],
            int(check_every),
            0,
            int(sbuf_kib),
            int(core_split),
            int(matvec),
            int(precision),
            algo=1,
        )

    @nki.jit
    def _cg_bsr_keyed_kernel(
        blocks_t,
        minv_t,
        state,
        key,
        check_every,
        precond,
        sbuf_kib,
        core_split,
        matvec=0,
        precision=0,
        variant=0,
    ):
        return _cg_bsr_body(
            blocks_t,
            minv_t,
            state,
            *_STRUCTURES[key],
            int(check_every),
            int(precond),
            int(sbuf_kib),
            int(core_split),
            int(matvec),
            int(precision),
            variant=int(variant),
        )
