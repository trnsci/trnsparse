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
  holds only its rows' blocks (so the SBUF capacity doubles) and computes
  its rows of every matvec. Full-width vectors use a *segment* layout:
  core 0's rows, zero padded, then core 1's, so each core's rows are one
  contiguous segment and an exchange is a single `nisa.sendrecv` of the
  own segment into the other one, with no copies. Exchanges of up to
  1 KiB per partition go through GpSimd's SBUF-to-SBUF DMA
  (`dma_engine=gpsimd_dma`): about 0.9 µs for a few columns against
  2.4 µs on the default DMA path.
  - Plain CG exchanges **once per step**: the cores swap their rows of
    q = A p, and then both compute p.q, r -= alpha q, r.r and
    p = r + beta p over the full width with the same instructions on the
    same data.
  - PCG exchanges twice: the partial p.q (a few columns; both cores add
    the same pair, and a + b == b + a exactly), then their rows of
    z = D⁻¹ r together with the partial r.z and r.r in the segment's
    extra columns.
  Either way both cores agree bit for bit on every step size and on when
  to stop. That is what lets them run the same device-side loop:
  neuronx-cc requires both per-core programs to have the same control
  flow (`NCC_IXGM002`). `TRNSPARSE_CORE_SPLIT=0` runs the whole solve
  on both cores instead.
- **Row order.** When A is resident, each core's block rows are
  reordered inside the kernel (`cg_row_order`) so that the four
  concurrent TensorE column groups of the column matvec get about the
  same number of stored blocks; the matvec takes as long as its busiest
  group. b, x0 and x are permuted by DMA on the way in and out. When
  blocks stream from HBM the stream sets the pace, and the original order
  with chunks of at most 32 rows interleaves better with the stream DMAs,
  so that is used instead.
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
issues blocks round-robin over four row groups: row i = 4 c + g of a
chunk goes to tile g. The result has to go back to the column layout
with a TensorE transpose (`nc_transpose`, bit-exact):

- In #41 row i landed at PSUM partition `32 g`, free offset `128 c`, so a
  16-row chunk filled four 128-column slots: four PSUM-to-SBUF copies and
  four transposes.
- Now (fp32 A, R in 1, 2, 4, 8) the matvec's source vectors are kept
  *spread*: each (128, R) block column sits after (G - 1) R zero columns.
  The stationary operand for slot c is then the (c + 1) R columns ending
  at the block, c R zeros and the block, so its product lands at PSUM
  partitions `32 g + c R + k` of a single 128-column slot (the zeros add
  0 to the other slots' rows). A chunk of up to 4 G rows (G R <= 32, so
  up to 128 rows for R = 1; 32 rows when blocks stream) needs one copy
  and one transpose. The zero gaps are never written.

Same fp32 products, different summation order. No relayout of the stored blocks is
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

### Chronopoulos–Gear recurrence (`recurrence=`, opt-in)

`recurrence="chronopoulos_gear"` (plain CG only) runs the
Chronopoulos–Gear form of CG: the matvec is applied to r, s = A p is
updated by recurrence, and

    r' = r - alpha s;  p = r + beta p;  x += alpha p
    w = A r';  gamma' = r'.r';  delta = w.r'
    beta' = gamma' / gamma;  alpha' = gamma' / (delta - beta' gamma' / alpha)
    s' = w + beta' s

so both dot products of a step form one reduction phase, and r'.r' is
ready while the matvec runs: only one dot product (delta) sits between
the matvec and the next r update, against two in standard CG. In exact
arithmetic it is CG. In floating point the recurrence for s drifts from
A p, so the recursive residual can drift further from b - A x and the
attainable accuracy is somewhat worse; on the benchmark matrices (κ up
to 1e5, tol 1e-5) the true residual and the iteration counts match
standard CG (`benchmarks.md`), but it stays opt-in. The host path runs
the same recurrence.

### What limits it

Measured on Trn2 (see `benchmarks.md`):

- With the row matvec (#40) each iteration costs about 0.5 µs per stored
  block (the fp32 weight load), resident or streamed alike.
- The column matvec costs one fp32 pass of 128 moving columns per block,
  about 270 ns, on each of four concurrent column groups: about 70 ns per
  block when the groups are balanced (hence the row order).
- The fixed cost per iteration, the chain of small VectorE / TensorE steps
  and the exchanges, is now about 4 µs at LNC=2 and 3.5 µs at LNC=1 for
  plain CG (#41: 11 and 8 µs). It is about 20 dependent instructions of
  150–300 ns each: the PSUM copy, transpose and copy that finish each
  matvec, the exchange, two dot products (multiply-reduce on VectorE, sum
  over partitions on TensorE), the step sizes, and the r and p updates.
  The Chronopoulos–Gear recurrence takes about 1 µs more off.
- Blocks past the SBUF budget stream from HBM (64 KiB per fp32 block), so
  residency matters: at n = 8192 with 1024 blocks, the streamed blocks set
  the pace.

The kernel's PSUM tiles (four per scope) are allocated once per trip of
the device loop and re-zeroed, so NKI's allocator no longer runs out of
PSUM and falls back to neuronx-cc allocation and scheduling.

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
