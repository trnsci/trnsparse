# Iterative solvers over BSR

trnsparse v0.3.2 adds `cg_bsr` and `power_iteration_bsr` — Conjugate
Gradient and power iteration on block-sparse row matrices. The API is
stable. `cg_bsr_fused` (below) is the one-launch, SBUF-resident CG.

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

## Fused CG: `cg_bsr_fused` (#22)

`cg_bsr_fused` runs the whole CG (or block-Jacobi PCG) solve in **one
kernel launch**:

```python
op = trnsparse.prepare_cg_bsr(A, preconditioner="block_jacobi", device="neuron")
x, iters, rel = trnsparse.cg_bsr_fused(op, b, tol=1e-5, max_iter=2000)
X, iters, rel = trnsparse.cg_bsr_fused(op, B)  # B: (n, R), R independent solves
```

What the kernel does on the NeuronCore:

- **A's stored blocks are DMA'd into SBUF once** and every iteration
  reads them from there. The budget is `TRNSPARSE_CG_SBUF_KIB` (default
  160 of the 192 KiB per partition on Trn2, i.e. 20 MiB of the 24 MiB
  SBUF), so about 320 fp32 128x128 blocks per physical core. Blocks
  beyond the budget stream from HBM in 16-block DMAs on every iteration,
  still inside the launch.
- x, r, p (and z) stay in SBUF in a column layout (element `t*128 + p`
  at `[p, t]`). The matvec is one `nc_matmul` per stored block into a
  PSUM column; VectorE reads A p and z = D⁻¹ r straight from PSUM.
- Each dot product is a VectorE multiply-reduce along the free axis and
  one all-ones TensorE matmul that sums over the 128 partitions and
  broadcasts the scalar to all of them at once.
- **Convergence is checked on the device.** An `nl.while_loop` (NKI 0.7
  dynamic control flow) runs `check_every` steps per trip and then
  re-tests `||r||² > tol²` and the iteration cap through a sequencer
  register. The host is not in the loop. Steps after convergence inside
  a trip are masked (alpha = beta = 0), so iteration counts and
  solutions are exact rather than rounded up to `check_every`.
- **At LNC=2 the two physical cores split the block rows.** Each core
  holds only its rows' blocks (so the SBUF capacity doubles), computes
  its rows of A p and its half of every dot product, and the cores swap
  partial sums and halves of p with `nisa.sendrecv`, three exchanges per
  step. Both cores add the two partial sums in the same pair (a + b ==
  b + a exactly), so they agree bit for bit on every step size and on
  when to stop. That is what lets them run the same device-side loop:
  neuronx-cc requires both per-core programs to have the same control
  flow (`NCC_IXGM002`). `TRNSPARSE_CORE_SPLIT=0` runs the whole solve
  on both cores instead.
- **Several right-hand sides** `(n, R)` are R independent CG solves that
  share every load of A: each stored block is one matmul with R moving
  columns. On Trn2 a 128x128 fp32 stationary block costs about 0.5 µs
  with 1 moving column and about 0.52 µs with 64, so up to about 64
  right-hand sides cost little more than one.

The kernel is fp32 (PSUM accumulates in fp32). The host path (no
`neuron` device, or `set_backend("pytorch")`) runs the same masked
algorithm in PyTorch in b's dtype, so fp64 stays fp64.

What limits it (measured on Trn2, see `benchmarks.md`): per iteration the
cost is about 0.5 µs per stored block. That is the TensorE fp32
stationary-weight load for a matvec with one moving column, and it is
the same whether the block comes from SBUF or streams from HBM. 64 KiB
per 0.5 µs is about 128 GB/s, which the DMA queues sustain inside the
loop. So at one right-hand side, SBUF residency is not what makes this
fast. The win is removing roughly ten host launches and a sync from
every iteration.

## Why CG and power iteration, not GMRES / Lanczos / Davidson

v0.3.2 covers the two most common algorithm families:

- **CG** for SPD systems — the workhorse for Hamiltonian solves,
  stiffness systems, and many PDE discretizations.
- **Power iteration** for dominant eigenpairs — the starting point
  for spectral methods, PageRank-like fixed points, and the iteration
  core inside Lanczos / Arnoldi / Davidson.

GMRES (general non-symmetric systems), Lanczos / Arnoldi (full
spectrum), and Davidson (interior eigenvalues) are follow-ups when
users ask for them. The fused kernel's structure (load A once, iterate
on-chip, device-side `while` convergence test) carries over to those
variants.
