"""Sparse Fock matrix build with integral screening.

Three paths, side by side:

1. **v0.1.x unfused** — `schwarz_bounds` → `screen_quartets` → mask-apply
   → `from_dense` → `spmm`. What the library shipped before v0.4.0.
   Four host passes over an (n, n) mask + a separate CSR build + a
   separate SpMM dispatch.
2. **v0.4.0 fused** — `screened_spmm(A, diag_integrals, B, threshold)`.
   One NKI kernel on the NKI backend; explicit mask + matmul on CPU.
   Same numeric result, fewer HBM round-trips, no mask tensor on HBM.
3. **Full Fock build** — the coulomb J from path 2 contracted against
   MO coefficients via trnblas: `F_MO = C.T @ J @ C`. Demonstrates the
   suite composition — trnsparse hands off to trnblas's GEMM once the
   sparse step is done.

Schwarz bounds here are synthetic (Gaussian-distance decay) — realistic
enough to show non-trivial sparsity. For real AO integrals from a
molecule, see `examples/pyscf_bridge.py`.

Usage:
    python examples/sparse_fock.py --demo
    python examples/sparse_fock.py --nbasis 200 --threshold 1e-8
"""

from __future__ import annotations

import argparse
import time

import torch

import trnsparse


def _synthetic_schwarz_system(n: int, seed: int = 42) -> torch.Tensor:
    """Return synthetic diagonal integrals `(μμ|μμ)` for a 1-D molecular chain.

    Chemistry convention: the Schwarz bound for `(μν|μν)` factors as
    `Q[μ] * Q[ν]` where `Q[i] = sqrt((ii|ii))`. This demo generates
    per-shell magnitudes that span several orders of magnitude so the
    outer-product bound `Q[i] * Q[j]` produces non-trivial sparsity at
    a realistic threshold.
    """
    torch.manual_seed(seed)
    # Shells arranged in a 1-D chain; Gaussian-like decay in magnitude
    # from a central "heavy" region so there's a tail of small Q values.
    idx = torch.arange(n, dtype=torch.float32)
    center = n / 2.0
    diag_integrals = torch.exp(-((idx - center) ** 2) / (2.0 * (n / 6.0) ** 2)) + 0.01
    return diag_integrals


def _unfused_path(
    integrals_dense: torch.Tensor, diag_integrals: torch.Tensor, P: torch.Tensor, threshold: float
):
    """Path 1: explicit Schwarz bound + mask + from_dense + spmm. v0.1.x flow."""
    import math

    t0 = time.perf_counter()
    Q = trnsparse.schwarz_bounds(diag_integrals)  # (n,)
    pair_bound = Q.unsqueeze(-1) * Q.unsqueeze(0)  # (n, n)
    mask = pair_bound > math.sqrt(threshold)
    integrals_masked = integrals_dense * mask.to(integrals_dense.dtype)
    integrals_sparse = trnsparse.from_dense(integrals_masked)
    J = trnsparse.spmm(integrals_sparse, P)
    return J, time.perf_counter() - t0


def _fused_path(
    integrals_dense: torch.Tensor, diag_integrals: torch.Tensor, P: torch.Tensor, threshold: float
):
    """Path 2: v0.4.0 fused screened_spmm."""
    t0 = time.perf_counter()
    J = trnsparse.screened_spmm(integrals_dense, diag_integrals, P, threshold=threshold)
    return J, time.perf_counter() - t0


def _full_fock_build(J: torch.Tensor, C: torch.Tensor):
    """Path 3: transform the coulomb J into the MO basis via trnblas.

    F_MO = C.T @ J @ C  — two GEMMs. Falls back to torch.matmul if
    trnblas isn't importable (it's an optional suite dep; pure-trnsparse
    users don't need it).
    """
    t0 = time.perf_counter()
    try:
        import trnblas

        Jt = trnblas.gemm(1.0, C, J, transA=True)  # C.T @ J
        F_MO = trnblas.gemm(1.0, Jt, C)  # (C.T @ J) @ C
        backend = "trnblas"
    except ImportError:
        F_MO = C.T @ J @ C
        backend = "torch.matmul (trnblas not installed)"
    return F_MO, time.perf_counter() - t0, backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run a small demo")
    parser.add_argument("--nbasis", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=1e-4)
    args = parser.parse_args()

    if args.demo:
        args.nbasis = 50

    n = args.nbasis
    print("Sparse Fock build:")
    print(f"  Basis functions: {n}")
    print(f"  Threshold:       {args.threshold:.0e}")

    diag_integrals = _synthetic_schwarz_system(n)
    Q = trnsparse.schwarz_bounds(diag_integrals)  # 1-D Schwarz bounds

    # Unscreened ERI slice — random, scaled by outer-product Schwarz
    # (chemistry-realistic: integrals tracking the bound).
    torch.manual_seed(0)
    integrals_dense = torch.randn(n, n) * (Q.unsqueeze(-1) * Q.unsqueeze(0)) * 0.01

    # Density matrix — random SPD.
    M = torch.randn(n, n) * 0.1
    P = M @ M.T

    # MO coefficients (for the trnblas transform in path 3) — orthonormal.
    U, _ = torch.linalg.qr(torch.randn(n, n))
    C = U

    # --- Path 1: unfused ---
    J_unfused, t_unfused = _unfused_path(integrals_dense, diag_integrals, P, args.threshold)

    # --- Path 2: fused ---
    J_fused, t_fused = _fused_path(integrals_dense, diag_integrals, P, args.threshold)

    # --- Path 3: trnblas MO transform ---
    F_MO, t_transform, backend = _full_fock_build(J_fused, C)

    # --- Sparsity stats for context ---
    # Build the pair-bound matrix for reporting stats at the matmul scale.
    pair_bound = Q.unsqueeze(-1) * Q.unsqueeze(0)
    stats = trnsparse.sparsity_stats(pair_bound, args.threshold**0.5)

    print()
    print("  Sparsity statistics:")
    print(f"    Total shell pairs:       {stats['total_pairs']}")
    print(f"    Significant pairs:       {stats['significant_pairs']}")
    print(f"    Pair sparsity:           {stats['pair_sparsity']:.1%}")
    print()
    print("  Coulomb build timings:")
    print(f"    Path 1 (unfused, 4-step):   {t_unfused * 1e3:8.3f} ms")
    print(
        f"    Path 2 (fused screened_spmm): {t_fused * 1e3:8.3f} ms  ({t_unfused / t_fused:.2f}x vs unfused)"
    )
    print()
    print("  Full Fock build (trnsparse → trnblas):")
    print(f"    MO transform (C.T @ J @ C): {t_transform * 1e3:8.3f} ms via {backend}")
    print()
    print(f"  Unfused/fused J agreement: max |ΔJ| = {(J_unfused - J_fused).abs().max().item():.2e}")
    print(f"  F_MO shape: {tuple(F_MO.shape)}, mean |F_MO| = {F_MO.abs().mean().item():.3e}")


if __name__ == "__main__":
    main()
