# Iterative solvers over BSR

trnsparse v0.3.2 adds `cg_bsr` and `power_iteration_bsr` — Conjugate
Gradient and power iteration on block-sparse row matrices. The API is
stable; the architectural story below explains why there's a v0.4.0
follow-up.

## Why this matters

Large SPD linear systems and dominant-eigenpair problems show up
across scientific computing:

- **Quantum chemistry**: Hamiltonian eigenvalue problems (HF, DFT,
  CI), response equations (CPSCF).
- **PDE discretizations**: stiffness-matrix solves for finite element
  methods, graph Laplacian systems.
- **Graph learning**: spectral embeddings, PageRank-like iterations.

The matrix `A` in each case is typically block-sparse (Fock matrices
after Schwarz screening; FEM stiffness tied to mesh connectivity;
graph adjacency). BSR is the Trainium-native representation for those
matrices (see `architecture.md`).

## v0.3.2 — plumbing

```python
import trnsparse

A = trnsparse.BSRMatrix.from_dense(fock_matrix, block_size=128)
b = compute_rhs()

x, iters, rel = trnsparse.cg_bsr(A, b, tol=1e-6, max_iter=1000)
# Jacobi-preconditioned variant:
M = trnsparse.jacobi_preconditioner_bsr(A)
x, iters, rel = trnsparse.cg_bsr(A, b, tol=1e-6, M=M)

lam, v, iters = trnsparse.power_iteration_bsr(A, max_iter=500)
```

Under the hood, each CG iteration calls `bsr_spmm(A, x.unsqueeze(1))`
once. On the NKI backend that's one kernel dispatch + one HBM
round-trip per iteration. On CPU it's `torch.sparse`-backed and
roughly on par with `scipy.sparse.linalg.cg` (benchmarked: 369 μs vs
310 μs at 128×128 SPD, 1.19× slower).

## v0.4.0 — fused kernel with SBUF-resident A

The architectural claim from
[#22](https://github.com/trnsci/trnsparse/issues/22): Trainium's 32 GB
SBUF per NeuronCore fits a 5000×5000 BSR Hamiltonian on-chip. CG
doesn't need to round-trip `A` to HBM at all — only `x`, `r`, and `p`.

The shape of the v0.4.0 kernel:

```python
@nki.jit
def _cg_spd_kernel(A_blocks, A_cols, A_row_ptrs, b, max_iter):
    # Load A blocks once into SBUF at the top.
    A_sbuf = nl.load(A_blocks)

    # State: x, r, p in SBUF registers.
    x = nl.zeros(...)
    r = nl.copy(b)
    p = nl.copy(r)
    rr = nl.reduce(r * r)

    for k in nl.affine_range(max_iter):
        Ap = _bsr_matvec_sbuf(A_sbuf, A_cols, A_row_ptrs, p)   # all SBUF
        pAp = nl.reduce(p * Ap)
        alpha = rr / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = nl.reduce(r * r)
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    return x, residual_norm_history
```

Fixed `max_iter`, no early exit — returning the full residual history
lets the host pick the convergence point post-hoc (standard
NKI-inside-loop constraint: no dynamic control flow).

**Expected regime where this wins**: when
`max_iter × dispatch_overhead > fused_kernel_cost + hbm_load_A_once`.
For a 4000×4000 BSR Hamiltonian with ~100 iterations to convergence,
that's roughly an order of magnitude of wall-time reduction.

**What needs to land first**:

1. Simulator-iterated kernel skeleton (now tractable thanks to the
   `nki-simulator` CI gate — see the NKI 0.3.0 migration in v0.3.1).
2. SBUF sizing model for `A` — some workloads overflow 32 GB; then
   fall back to the v0.3.2 plumbing.
3. Dispatcher logic in `cg_bsr` that picks the fused kernel when
   `max_iter * n` exceeds a threshold.

Tracked in a dedicated sub-issue on `trnsci/trnsparse`.

## Why CG and power iteration, not GMRES / Lanczos / Davidson

v0.3.2 covers the two most common algorithm families:

- **CG** for SPD systems — the workhorse for Hamiltonian solves,
  stiffness systems, and many PDE discretizations.
- **Power iteration** for dominant eigenpairs — the starting point
  for spectral methods, PageRank-like fixed points, and the iteration
  core inside Lanczos / Arnoldi / Davidson.

GMRES (general non-symmetric systems), Lanczos / Arnoldi (full
spectrum), and Davidson (interior eigenvalues) are follow-ups when
users ask for them. The v0.4.0 fused kernel's structure (load A once,
iterate on-chip) generalizes trivially to those variants — it's the
same architectural pattern.
