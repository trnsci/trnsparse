"""
Integral screening utilities for sparse quantum chemistry.

Schwarz screening: |(μν|λσ)| ≤ √(μν|μν) √(λσ|λσ)
Density screening: |P_λσ (μν|λσ)| ≤ |P_λσ| Q_μν Q_λσ

These bounds determine which shell quartets contribute significantly
to the Fock matrix, turning O(N⁴) integral evaluation into O(N²) for
large molecules. The sparsity pattern feeds into trnsparse CSR matrices.
"""

from __future__ import annotations

import torch
from typing import Tuple

from .formats import CSRMatrix, from_dense


def schwarz_bounds(diagonal_integrals: torch.Tensor) -> torch.Tensor:
    """Compute Schwarz upper bounds Q_μν = √(μν|μν).

    Args:
        diagonal_integrals: (nshells, nshells) diagonal ERI values (μν|μν)

    Returns:
        Q: (nshells, nshells) Schwarz bounds
    """
    return torch.sqrt(torch.abs(diagonal_integrals))


def screen_quartets(
    Q: torch.Tensor,
    threshold: float = 1e-10,
) -> torch.Tensor:
    """Screen shell quartets using Schwarz inequality.

    Returns a boolean mask (nshells, nshells) for significant shell pairs.
    A quartet (μν|λσ) is significant if Q_μν * Q_λσ > threshold.
    """
    # Significant pairs: Q_μν > √threshold (since Q_μν * Q_λσ > threshold)
    pair_threshold = threshold ** 0.5
    return Q > pair_threshold


def density_screen(
    Q: torch.Tensor,
    P: torch.Tensor,
    threshold: float = 1e-10,
) -> torch.Tensor:
    """Density-weighted screening.

    A quartet contributes if |P_λσ| * Q_μν * Q_λσ > threshold.
    Returns significant (μν) mask given density P and bounds Q.

    Args:
        Q: (n, n) Schwarz bounds
        P: (n, n) density matrix
        threshold: screening threshold

    Returns:
        mask: (n, n) boolean — True for significant shell pairs
    """
    # Max |P_λσ * Q_λσ| over λσ gives the bound for each μν
    PQ = torch.abs(P) * Q
    max_PQ = PQ.max()  # Conservative: use global max
    return Q * max_PQ > threshold


def sparsity_stats(Q: torch.Tensor, threshold: float = 1e-10) -> dict:
    """Report sparsity statistics for a set of Schwarz bounds.

    Returns dict with total quartets, significant quartets, sparsity fraction.
    """
    n = Q.shape[0]
    mask = screen_quartets(Q, threshold)
    n_sig_pairs = mask.sum().item()
    n_total_pairs = n * n
    n_total_quartets = n_total_pairs * n_total_pairs
    n_sig_quartets = n_sig_pairs * n_sig_pairs  # Upper bound

    return {
        "n_shells": n,
        "total_pairs": n_total_pairs,
        "significant_pairs": n_sig_pairs,
        "pair_sparsity": 1.0 - n_sig_pairs / n_total_pairs,
        "total_quartets": n_total_quartets,
        "significant_quartets_upper": n_sig_quartets,
        "quartet_sparsity_lower": 1.0 - n_sig_quartets / n_total_quartets,
    }
