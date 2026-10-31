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
  at `[p, t]`). The matvec is one `nc_matmul` per stored block; see
  "Matvec orientation" below.
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
  its rows of A p and its half of p.Ap, and the cores swap with
  `nisa.sendrecv` twice per step: the partial p.Ap (both cores add the
  same pair, and a + b == b + a exactly), and their rows of r (and z),
  after which each core holds the full r and z and computes r.z, r.r and
  p = z + beta p over the full width itself. Either way both cores agree
  bit for bit on every step size and on when to stop. (#40 used three
  exchanges: partial r.z / r.r and p separately.) That is what lets them
  run the same device-side loop:
  neuronx-cc requires both per-core programs to have the same control
  flow (`NCC_IXGM002`). `TRNSPARSE_CORE_SPLIT=0` runs the whole solve
  on both cores instead.
- **Several right-hand sides** `(n, R)` are R independent CG solves that
  share every matmul over a stored block (R stationary columns in the
  column matvec, R moving columns in the row matvec).

### Matvec orientation (`matvec=`)

`nc_matmul(dst, stationary, moving)` computes `stationaryᵀ @ moving`. The
stationary operand is loaded into the 128x128 PE array, then the moving
operand streams through it. Per stored fp32 128x128 block on Trn2
(microbenchmark, one physical core):

| orientation | stationary | moving | ns per block |
|---|---|---|---:|
| `"row"` (#40) | A block (128x128) | p_j (128x1) | 425 |
| vector-stationary, one PE tile | p_j (128x1) | A block | 206 |
| `"column"`: vector-stationary, 4 column tiles in turn | p_j (128x1) | A block | 50 |

The row form pays the fp32 weight load of the whole block for a single
moving column. In the column form the stationary operand is p_j and the
block streams through as the moving operand, so the result is a row,
`(A_idx p_j)ᵀ`. With column tiling (`tile_size=(128, 32)`) the PE array
splits into four 32-column tiles that work concurrently. The kernel
issues blocks round-robin over four row groups, so block row i of a
16-row chunk lands at PSUM partition `32 (i % 4)`, free offset
`128 ((i // 4) % 4)`. Each chunk is then copied to SBUF and put back in
the column layout with one TensorE transpose per four block rows
(`nc_transpose`, bit-exact; about 110 ns for 128x128 fp32). Same fp32
products, different summation order. No relayout of the stored blocks is
needed: the `Aᵀ` blocks the row form uses as stationary are exactly the
moving operand the column form needs. `"column"` takes R <= 32 (one
32-column tile). `"auto"` (the default) picks it for R <= 32 and `"row"`
above that.

### bf16 A (`precision=`, opt-in)

`precision="bf16"` stores A's blocks in bf16 (half the SBUF and HBM bytes,
so twice the resident blocks). The vector stays fp32 and is split on the
device into bf16 hi + lo (`hi = bf16(p)`, `lo = bf16(p - hi)`), so the
products are exact and accumulate in fp32 PSUM. The only loss is A's bf16
rounding (~2^-9 relative): CG converges for the rounded matrix, and the
true residual against A stalls near 5e-3. At κ = 1e5 the rounded matrix
can be indefinite and CG breaks down. `precision="bf16x2"` stores A as
bf16 hi + lo (~2^-17 relative, the same bytes as fp32, two matmuls per
block). Both need `matvec="column"` and R <= 16. The host path emulates
both. Measured on Trn2: `"bf16"` is 1.0–1.4x faster than fp32 where fp32 A
fits in SBUF and about 2.3x faster where it does not (n = 8192 with 1024
blocks, where all 1024 bf16 blocks stay resident). Its true residual is
1e-3 to 7e-2 at κ = 1e3, and at κ = 1e5 the error is about 0.9, or the
solve breaks down. `"bf16x2"` costs about the same as fp32 and lands at
1–15x tol at κ = 1e3 and 5–70x tol at κ = 1e5, so it does not pay. Neither
replaces fp32; bf16 is at best an inner solver or a preconditioner. See `benchmarks.md`.

The kernel is fp32 (PSUM accumulates in fp32). The host path (no
`neuron` device, or `set_backend("pytorch")`) runs the same masked
algorithm in PyTorch in b's dtype, so fp64 stays fp64.

What limits it (measured on Trn2, see `benchmarks.md`): with the row
matvec (#40), each iteration costs about 0.5 µs per stored block (the
fp32 weight load), resident or streamed alike. The column matvec brings
that to about 0.05 µs per resident block. Now three other costs set the
time per iteration: a fixed cost of about 8 µs (LNC=1) or 11 µs (LNC=2,
two exchanges) for the chain of small VectorE / TensorE steps and the
`sendrecv`s; the TensorE transposes (one per four block rows); and, for
blocks past the SBUF budget, the HBM stream (64 KiB per fp32 block). So
residency matters again: at n = 8192 with 1024 blocks, the 384 streamed
blocks set the pace.

## Fused power iteration: `power_iteration_bsr_fused`

```python
op = trnsparse.prepare_cg_bsr(A, device="neuron")  # A need not be SPD here
lam, v, iters = trnsparse.power_iteration_bsr_fused(op, tol=1e-5, max_iter=1000)
```

The same kernel (resident A, column matvec, PSUM dot products, LNC=2 row
split, device-side `while` loop) runs v <- A v / ||A v||. The Rayleigh
quotient lam = v.Av / v.v (with ||Av||² from the same reduction, one
exchange) is the eigenvalue estimate. The loop stops when
|lam_k - lam_(k-1)| < tol, as in `power_iteration_bsr`, or at `max_iter`.
The rows of v are swapped with one more exchange. On the device it is
fp32, so lam settles to about 1e-7 |lam|.

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
