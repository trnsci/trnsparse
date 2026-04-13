"""
Sparse Fock matrix build with integral screening.

Demonstrates how Schwarz screening reduces the Fock build from O(N⁴) to
effectively O(N²) for large molecules. The sparsity pattern is stored
as a CSR matrix, and SpMM handles the screened contraction.

Usage:
    python examples/sparse_fock.py --demo
    python examples/sparse_fock.py --nbasis 200
"""

import argparse
import time

import torch

import trnsparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--nbasis", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=1e-10)
    args = parser.parse_args()

    if args.demo:
        args.nbasis = 50

    n = args.nbasis
    print("Sparse Fock build:")
    print(f"  Basis functions: {n}")
    print(f"  Threshold:       {args.threshold:.0e}")

    torch.manual_seed(42)

    # Simulate Schwarz bounds (decay with distance for realistic sparsity)
    positions = torch.rand(n, 3) * 10.0  # Random 3D positions
    distances = torch.cdist(positions, positions)
    Q = torch.exp(-0.5 * distances)  # Gaussian decay

    # Screen
    stats = trnsparse.sparsity_stats(Q, args.threshold)
    print("\n  Sparsity statistics:")
    print(f"    Total shell pairs:       {stats['total_pairs']}")
    print(f"    Significant pairs:       {stats['significant_pairs']}")
    print(f"    Pair sparsity:           {stats['pair_sparsity']:.1%}")
    print(f"    Quartet sparsity (lower): {stats['quartet_sparsity_lower']:.1%}")

    # Build sparse integral matrix (simulated)
    mask = trnsparse.screen_quartets(Q, args.threshold)
    integrals_dense = torch.randn(n, n) * Q * 0.01
    integrals_dense[~mask] = 0.0
    integrals_sparse = trnsparse.from_dense(integrals_dense)
    print(f"    Integral matrix nnz:     {integrals_sparse.nnz} / {n * n}")

    # Density matrix (random SPD for demo)
    P = torch.randn(n, n) * 0.1
    P = P @ P.T

    # Sparse Fock build: J_μν = Σ_λσ P_λσ * (μν|λσ)
    # Approximated here as SpMM: J ≈ sparse_integrals @ P
    t0 = time.perf_counter()
    J_sparse = trnsparse.spmm(integrals_sparse, P)
    t_sparse = time.perf_counter() - t0

    # Dense reference
    t0 = time.perf_counter()
    J_dense = integrals_dense @ P
    t_dense = time.perf_counter() - t0

    error = torch.linalg.norm(J_sparse - J_dense).item()
    print(f"\n  Sparse SpMM:  {t_sparse:.4f}s")
    print(f"  Dense matmul: {t_dense:.4f}s")
    print(f"  Error:        {error:.2e}")


if __name__ == "__main__":
    main()
