# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Guard against the NKI trace-cache closure bug ([aws-neuron/aws-neuron-sdk#1454](https://github.com/aws-neuron/aws-neuron-sdk/issues/1454),
  trnsci/trnsci#35).** `tests/test_trace_cache_safety.py` (CPU) finds every
  `@nki.jit` kernel and fails if its body, or a helper it calls, reads a
  closure variable or a module global that is not a constant expression
  (reviewed exceptions are allow-listed with a reason), and checks each
  kernel's `__closure__` at runtime against a stub `nki`.
- Tests: `conftest.py` no longer turns the NKI trace cache off. Neuron tests
  run with it on, as under `import torch_neuronx`, with a fresh
  `NKI_TRACE_CACHE_URL` directory per session (no stale editable-install
  traces). `NKI_ENABLE_TRACE_CACHE=0` still turns it off.
- README: known platform issue note for #1454.
### Fixed

- **`block_sparse_attention_tiled` accepted a mask smaller than `seq_len` on
  the NKI path** (silent, #82). With a `(256, 256)` mask and `seq_len = 384`,
  the native `bsr_attn` op returned output rows 256..383 that no mask block
  covers (garbage values, no error). The PyTorch path raised an opaque
  `IndexError`. Every backend now raises `ValueError` unless
  `mask_bsr.shape == (Q rows, K rows)` and K and V have the same length.
  Affects 0.10.0 through 0.12.1. Found by the round-3 device harness on Trn2
  (trnsci/trnsci#35).
- **`bsr_diagonal` returned length m for a tall matrix** (silent; every
  release 0.3.2–0.12.1, every backend). For a rectangular (m, n) BSR matrix
  with m > n the result was zero-padded to length m; it now has length
  min(m, n), as `torch.diagonal` and scipy do. Square and wide matrices are
  unchanged (#74).
- **`COOMatrix.to_dense` dropped duplicate entries** (silent; every release
  0.1.1–0.12.1, backend-independent). Repeated `(row, col)` entries were
  assigned rather than summed, so all but one was lost, while `to_csr()`,
  `torch.sparse_coo_tensor` and `scipy.sparse` sum them. This also affected
  `CSRMatrix.to_coo().to_dense()` for a CSR with a repeated column in a row.
  Duplicates are now summed; matrices without duplicates are unchanged (#56).
- **`chebyshev_coeffs` / `chebyshev_bsr` did not produce the Chebyshev
  polynomial for K >= 2** (silent; every release 0.4.3–0.12.1, every
  backend, host-side coefficients). The conversion from the Templates
  listing used beta_1 = (c alpha_0 / 2)^2 instead of the special-case
  ½(c alpha_0)^2, and every later coefficient inherited the error, so the
  worst-case residual after K steps was up to ~3x the Chebyshev bound
  (K = 7). The coefficients now follow Saad's three-term recurrence
  (Alg. 12.1) and the residual is T_K((d - lam)/c) / T_K(d/c) (#73).
### Fixed

- **`from_scipy` discarded the imaginary part of complex matrices** (silent;
  every release 0.1.1–0.12.1). Values were always cast to float32, so a
  complex scipy matrix became its real part (one C++ `UserWarning`, no
  error). Complex input now gives `complex64` values; real input is
  unchanged (float32) (#58).
### Fixed

- **Second derivatives of `block_sparse_attention_tiled` were wrong**
  (silent; 0.6.0–0.12.1, the releases with attention autograd, on the
  PyTorch backend and the PyTorch/XLA / simulator paths). The backward
  reused the forward's softmax statistics as constants, so differentiating
  it (`create_graph=True`: `gradgradcheck`, gradient penalties, Hessian-
  vector products) dropped their dependence on Q and K, with no error.
  Double backward now differentiates a graph-carrying recomputation; first-
  order gradients are unchanged (#65).
### Fixed

- **`spmv` / `spmm` accepted a `y` / `C` that enlarged the output** (silent;
  every release 0.1.1–0.12.1, all backends). `beta * y` was added with plain
  broadcasting, so a `(m, 1)` `y` turned the `(m,)` product into `(m, m)`
  with no error, where `torch.addmv` / `addmm` raise. Such shapes now raise
  `ValueError`; every `y` / `C` that broadcasts to the product's shape works
  as before (#67).

### Fixed

- **bf16 Q / K / V raised on the gather attention kernels** (loud; #49).
  Affects released versions **0.4.4 through 0.12.1** with `backend="nki"`
  (or `"auto"` on Neuron): every bf16 call that took the gather kernels,
  i.e. `head_dim > 128` (K-tiled, since 0.6.0), the XLA path, or the native
  path with the static structure off. The weights @ V, dS @ K, dSᵀ @ Q and
  Pᵀ @ dO matmuls paired fp32 P / dS with bf16 operands, and NKI rejects
  mixed fp32 / bf16 `nc_matmul` operands ("if one input is tfloat32/float32,
  both must be"). With `TRNSPARSE_REQUIRE_NKI=1` the call raised. Without it,
  the call fell back to the PyTorch path, which gives correct results but
  runs off the NKI path. The gather kernels now handle precision like the
  static kernels: one operand dtype per call (bf16 when Q, K, V (and dO) are
  all bf16, else fp32); fp32 scores, softmax statistics, D and PSUM; and P /
  dS rounded to the operand dtype before their matmul. The scale and D are
  computed on the host in fp32. fp32 and fp64 inputs give the same values as
  before.
- **`head_dim` above 128 that is not a multiple of 128 (e.g. 192) raised on
  the NKI attention path** (`AssertionError` in the gather, released 0.6.0
  through 0.12.1). It is now zero-padded to the next multiple of 128 for the
  K-tiled kernels and sliced back; zero columns change no score and no
  gradient (#49).

## [0.12.1] — 2026-11-01

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: **correctness fix:** in every release through 0.12.0, `spmv_symmetric` ignored `uplo`, so a fully stored symmetric matrix gave `A@x + strict(A)ᵀ@x` (rel. err 0.85–1.9). Now only the `uplo` triangle is referenced, as in BLAS `symv`. A matrix storing only the opposite triangle still gives the old (correct) result with a `DeprecationWarning` (#53, #52).

## [0.12.0] — 2026-11-01

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: **correctness fix — upgrade recommended if you pass `scale` to block-sparse attention.** In 0.6.0–0.11.0 a custom `scale` passed to `block_sparse_attention_tiled` was silently ignored on the NKI paths (forward and dQ/dK/dV; 29–99% relative error at scale 0.1–2.0). The native `trnsparse::bsr_attn` op was affected from 0.9.0. The PyTorch backend and `ShardedAttention` were correct, and the default `scale=None` was unaffected (#50, #47). Also: BSR block sizes other than 128 go straight to the torch path instead of paying a failed NKI compile on every call (7–41 s first call, 6.7–8.3× steady), which covers `bsr_spmm`, attention and the iterative solvers (#48, #46).

### Fixed

- **`spmv_symmetric` ignored `uplo`** (silent). Affects **every release through
  0.11.0**, on every backend. Every stored entry was used, so a fully stored
  symmetric matrix (or any entry outside the intended triangle) gave
  `A @ x + strict(A)ᵀ @ x` with no error (rel. err 0.85–1.9 on a 20×20 test).
  When both triangles hold entries, only the `uplo` triangle is now
  referenced, as in BLAS `symv`, on the host and device paths. Other `uplo`
  values raise. Callers that store one triangle get the same results as
  before, including those whose `uplo` names the *other* triangle (the old
  code ignored it): that triangle is still used, now with a
  `DeprecationWarning` asking for the matching `uplo` (#52).

- **`block_sparse_attention_tiled` silently ignored a custom `scale` on the
  NKI paths** (released 0.6.0 through 0.11.0; #47). With `backend="nki"` or
  `"auto"` on Neuron, the forward and all three gradients always used
  `head_dim ** -0.5`, and no error was raised. On Trn2 (seq 384, head_dim 64,
  relative error against dense fp64 attention) the forward was off by 29%
  (`scale=0.1`), 93% (`1.0`) and 95% (`2.0`), and dQ / dK by up to 99%. The
  XLA / simulator path (`_AttnTiledFunction` and the no-grad call) was
  affected from 0.6.0, and the native TorchNeuron `trnsparse::bsr_attn` op
  from 0.9.0. The PyTorch backend was always correct. `scale` now reaches
  `nki_bsr_attn_tiled`, `nki_bsr_attn_bwd` and the `bsr_attn` /
  `bsr_attn_bwd` ops, each of which takes a new optional `scale` argument
  (default `head_dim ** -0.5`). The kernels are unchanged: they already take
  `Q * scale`, and the static backward takes `scale` as a kernel argument.
  With the default `scale=None`, outputs and gradients are bit-identical to
  0.11.0 on Trn2. `ShardedAttention.shard_rows(Q, scale)` was not affected.
- `bsr_spmm` and `block_sparse_attention_tiled` with a `block_size` other than
  128 (16, 32 and 64 in the audit) no longer try an NKI kernel first. The
  kernels take 128x128 blocks only, so that trace always failed, and the call
  fell back to torch only after paying for it: 7-9 s on the first `bsr_spmm`
  call and 27-41 s on the first attention call on Trn2. These block sizes now
  go straight to the PyTorch path under `auto` (and under `nki` when
  `TRNSPARSE_REQUIRE_NKI` is not set). Under `TRNSPARSE_REQUIRE_NKI` they raise
  `NotImplementedError` naming the supported block size. The unfused
  iterative solvers (`cg_bsr`, `power_iteration_bsr`, `chebyshev_bsr`,
  `richardson_bsr`) also benefit, because they call `bsr_spmm` on every
  iteration (#46).

## [0.11.0] — 2026-11-01

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: fused CG/PCG (`cg_bsr_fused`) with a column matvec is 5–56× faster than CPU, and the fixed per-iteration cost is halved (#41, #43); fused power iteration (#41). **Fix:** in 0.6.0–0.10.0, `block_sparse_attention_tiled` silently returned zero rows for a partial last block on the PyTorch path; it now raises (#44). A host `BSRMatrix` with device `b` in `cg_bsr_fused` now works (#42).

### Fixed

- **`block_sparse_attention_tiled` silently returned zero rows for a partial
  last block** (released 0.6.0 through 0.10.0, PyTorch path). With
  `seq_len % block_size != 0` the query rows past the last whole block were
  left at 0 (seq_len 200, block 128: rows 128-199 zero, max error 0.79). It now
  raises `ValueError` on every backend; partial blocks need -inf key masking,
  not zero padding. `ShardedAttention` raises the same error.
- `ShardedBSR.gather` (0.10.0) raised a `RuntimeError` from `view` when the row
  count was not a multiple of 128. It now gathers at the padded row count and
  slices back.

### Added

- `tests/test_nki_tail_audit.py`: every NKI kernel (CSR and BSR SpMM, screened
  SpMM, attention, fused CG and power iteration) at awkward sizes (1, 127, 129,
  200, 1000, 4097; odd N; free dims not a multiple of 512) against fp64
  references (trnsci/trnsci#35).

#### Fused SBUF-resident CG (#22)

- `cg_bsr_fused(A, b, ...)` and `prepare_cg_bsr(A, preconditioner=None |
  "block_jacobi", device=...)` (a reusable `FusedCGOperator`): the whole
  CG or block-Jacobi PCG solve runs in one NKI launch
  (`trnsparse::cg_bsr_keyed`). A's blocks are loaded into SBUF once
  (`TRNSPARSE_CG_SBUF_KIB`, default 160 KiB per partition; any overflow
  streams from HBM inside the loop). Vectors stay in SBUF, dot products
  reduce through PSUM, and an `nl.while_loop` checks convergence on the
  device every `check_every` steps, with masked steps so the iteration
  count is exact. At LNC=2 the two physical cores split the block rows
  and exchange partial dots and halves of p with `sendrecv`. `b` may be
  `(n, R)`: R independent solves share each matmul.
- Host fallback with the same algorithm (fp64 stays fp64), CPU tests,
  MLIR verify and NEFF compile at LNC=1/2, and hardware tests
  (`tests/test_nki_cg_fused.py`).
- `benchmarks/bench_cg_fused_trn2.py`: time to tolerance against a
  host-driven loop over the BSR SpMM op, `cg_bsr`, dense aten CG on the
  device, scipy and torch on the CPU, on synthetic block-screened
  Gaussian-overlap matrices.

#### Fused CG follow-ups (#22): column matvec, bf16 A, fused power iteration

- `cg_bsr_fused(..., matvec="auto" | "row" | "column")`: the
  **vector-stationary ("column") matvec** is the new default for up to 32
  right-hand sides. The vector is the TensorE stationary operand, the
  stored block streams through as the moving operand, and four 32-column
  PE tiles work in turn. Each 16 block rows are then transposed back to
  the column layout on TensorE (bit-exact). That is about 50 ns per
  resident fp32 block against 425 ns for the row form, with the same
  fp32 arithmetic in a different summation order. It needs no relayout
  of the stored blocks. More than 32 right-hand sides keep `"row"`.
- Two `sendrecv` exchanges per CG step at LNC=2 instead of three. The
  cores swap their rows of r (and z) and then compute r.z, r.r and
  p = z + beta p over the full width, bit-identically on both cores.
  This replaces separate exchanges of the partial dots and of p.
- `precision="bf16" | "bf16x2"` (opt-in, column matvec, R <= 16): bf16 A,
  or bf16 hi + lo A, with fp32 vectors (split into bf16 hi + lo on the
  device) and fp32 accumulation. The host path emulates both.
- `power_iteration_bsr_fused(A, v0, max_iter, tol, ...)`
  (`trnsparse::power_bsr_keyed`): power iteration in one launch on the
  same kernel (Rayleigh quotient, device-side stop on |Δλ| < tol, LNC=2
  row split).
- `benchmarks/bench_cg_fused_trn2.py --matvec row,column,column:bf16,...`
  and `benchmarks/bench_power_fused_trn2.py`.

#### Fused CG: fixed per-iteration cost (#22)

- Plain CG at LNC=2 exchanges **once per step**: the cores swap their rows
  of q = A p and then compute p.q, r, r.r and p over the full width,
  bit-identically. PCG still exchanges twice, but the second exchange
  carries z with the partial r.z and r.r. Full-width vectors use a
  segment layout (each core's rows contiguous), so every exchange is one
  `sendrecv` with no copies, and exchanges up to 1 KiB per partition use
  GpSimd's SBUF-to-SBUF DMA (about 0.9 µs against 2.4 µs).
- Column matvec: "spread" source vectors (each block column after
  (G - 1) R zero columns) put up to 4 G block rows into one PSUM slot, so
  a chunk takes one PSUM-to-SBUF copy and one TensorE transpose instead of
  one of each per four rows (fp32 A, R in 1, 2, 4, 8).
- `cg_row_order`: each core's block rows are reordered so the four
  concurrent TensorE column groups carry about the same number of blocks
  (b, x0 and x are permuted by DMA in the kernel).
- Shorter step-size chain (no `max` before the reciprocal of p.Ap; r.z x
  act formed off the critical path), the x update deferred into the next
  matvec, the all-ones reduction matmul flagged `is_stationary_onezero`.
- PSUM tiles are allocated once per trip of the device loop and re-zeroed,
  so NKI's allocator no longer falls back to neuronx-cc allocation and
  scheduling.
- `cg_bsr_fused(..., recurrence="chronopoulos_gear")` (opt-in, plain CG):
  the Chronopoulos–Gear recurrence, one reduction phase per step. The host
  path runs the same recurrence.
- Measured on Trn2 (#22, `docs/benchmarks.md`): the fixed cost per plain
  CG iteration falls from about 11 µs to about 4 µs at LNC=2 and from about
  8 µs to about 3.5 µs at LNC=1.

### Fixed

- `cg_bsr_fused` / `power_iteration_bsr_fused` with a host `BSRMatrix` and
  `b` (or `v0`) on the `neuron` device raised `ModuleNotFoundError:
  torch_neuronx.pyhlo`: the kernel got host tensors and NKI fell back to
  its torch-xla framework (#42). A `BSRMatrix` is now prepared on b's
  device, and the fused kernel runs only for an operator on a `neuron`
  device (a host operator takes the host path).

## [0.10.0] — 2026-10-31

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: pre-transposed static BSR SpMM beats dense aten up to ~82–98% block density on one core (was 26–49%), plus `ShardedBSR`/`ShardedAttention` across logical NeuronCores (#38).

### Added

#### Static BSR SpMM and block-sparse attention across logical NeuronCores (#30)

- `trnsparse.multicore`: `ShardedBSR` (block rows split across the logical
  cores of a Trn2 chip, balanced by stored blocks), `ShardedAttention`
  (single-head attention forward split by query block rows),
  `partition_block_rows` and `logical_cores`. Per-core operands are built
  on the host and stay resident; one kernel launch per core, issued
  without waiting.
- Pre-transposed BSR kernel (`bsr_spmm_t_keyed`, the `ShardedBSR`
  default). The blocks are stored transposed, side by side, so the kernel
  needs no in-kernel `load_transpose2d`. It loads up to 16 of a row's
  blocks per DMA, reads PSUM out with one copy, and shares each B tile
  across groups of up to 8 block rows, one PSUM bank each
  (`TRNSPARSE_ROW_GROUP`). The kernel has autograd.
- `benchmarks/bench_multicore_trn2.py` (1, 2 and 4 logical cores, against
  dense aten on 1 core and split across cores) and
  `benchmarks/bench_multiproc_trn2.py` (one process per core).

### Changed

- The static kernels take their structure as a registered key, a content
  hash (`kernels.register_structure`), instead of int lists. The key
  crosses the custom-op boundary as one int, so the host cost of a launch
  no longer grows with the stored-block count: on Trn2 it was 0.13 ms at
  41 blocks and 0.67 ms at 2048, and is now about 0.12 ms at any size. The
  public `bsr_spmm` and attention-forward paths use the keyed ops.
- Native kernels run on the logical core that holds their operands
  (`neuron:k`) instead of always on the default `neuron` device. Operands
  spread over two cores raise an error. TorchNeuron does not copy between
  logical cores: `.to()` returns the tensor still on its old core.
- Kernel docstrings now give the measured `nl.reciprocal` behaviour:
  about 1.5e-6 relative error in range, but it silently returns 1e7 or 0
  outside roughly |x| in [2^-42, 2^41].

## [0.9.0] — 2026-10-30

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron; LNC=1 and LNC=2) — see trnsci/trnsci#35.

Headline performance: static BSR SpMM and block-sparse attention split across both LNC=2 cores, with bf16 operands on fp32 PSUM; the fp32 BSR crossover against dense aten at n=8192, width 512 rises from ~27% to ~49% block density, and at seq 8192 attention forward plus backward beats unmasked dense SDPA for all three masks (#36).

### Changed

#### Native path: static kernels on both LNC=2 cores, bf16 operands (#30)

- `bsr_spmm_static`, `attn_fwd_static` and `attn_bwd_static` split their
  output tiles between the two physical cores of an LNC=2 logical core
  (`program_id`), balanced by stored blocks; each output tile has one
  writer. `TRNSPARSE_CORE_SPLIT=0` turns the split off.
- bf16 inputs (all operands bf16) run bf16 matmuls with fp32 PSUM
  accumulation; scores and softmax statistics stay fp32, and the attention
  statistics returned by `trnsparse::bsr_attn` are fp32 for bf16 / fp16 Q.
  Mixed fp32/bf16 operands run in fp32.
- `benchmarks/bench_native_trn2.py`: `--dtype bf16` (matched-precision
  dense aten baselines) and `--fast` (device columns only).

## [0.8.0] — 2026-10-30

Native TorchNeuron dispatch (Trn2/Trn3) alongside the PyTorch/XLA path (Trn1, maintenance); validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0) — see trnsci/trnsci#35.

Headline performance: static-structure BSR SpMM, 6.8× faster than dense aten `matmul` at 1% block density, n=8192 (#34).

### Changed

#### Native path: static-structure kernels (#30)

On TorchNeuron the sparse structure is host-resident, so the native path now
hands it to the kernels as compile-time constants (`int[]` op arguments,
tuples inside the kernel) instead of padded, gathered device tensors.
Measured on Trn2 (`benchmarks/bench_native_trn2.py`; tables on #30):

- `bsr_spmm` → `trnsparse::bsr_spmm_static`: walks exactly the stored
  blocks and addresses B by block column; no `(M_tiles, K_max)` padding and
  no device-side gather. 8192×8192 @ 8192×512: 3.1–12.5 ms → 0.7–3.9 ms;
  beats dense aten `matmul` on the device up to ~13% (width 128) to ~34%
  (width 512) block density at n = 8192.
- Block-sparse attention forward → `trnsparse::attn_fwd_static`: one launch,
  two passes with the softmax statistics on chip; no K/V gather, no host
  reduction, no per-block score round trip through HBM. Backward →
  `trnsparse::attn_bwd_static`: one launch for dQ (row-first) and dK/dV
  (column-first). seq 8192 global mask, fwd+bwd: 127–195 ms → ~4 ms.
- Screened SpMM masks each tile in the transposed frame, so the masked tile
  is directly the stationary operand (no HBM round trip per tile).
- `spmm` with host CSR scatters the nnz triplets on the device when that is
  cheaper than copying a host-built dense matrix.

One compile per distinct structure (seconds; cached in-process by the
structure). `TRNSPARSE_STATIC_STRUCTURE=0` restores the gather path;
structures above `TRNSPARSE_STATIC_MAX_UNROLL` (default 16384 unrolled
block matmuls) use it automatically.

### Added

- `benchmarks/bench_native_trn2.py`: warm-latency benchmark for the native
  path against dense aten on the device, CPU, and `torch.sparse` on CPU.

#### Native PyTorch dispatch on TorchNeuron (#29, #30)

On TorchNeuron (Neuron SDK 2.33+, `torch-neuronx` 2.14, PyTorch 2.14,
Trn2/Trn3), every NKI kernel is now registered as a PyTorch custom op with
`torch_neuronx.nki_op` and dispatched on `device="neuron"`, with no
`torch_xla` import. Ops: `trnsparse::spmm_dense`, `bsr_spmm`,
`screened_spmm`, `attn_stats`, `attn_out`, `attn_bwd_dq`, `attn_bwd_dkdv`,
plus the composite `trnsparse::bsr_attn` / `bsr_attn_bwd`. Kernel bodies
are unchanged.

- Autograd on the native path uses `register_autograd` on the ops instead
  of `torch.autograd.Function`: `spmm_dense`, `bsr_spmm` and
  `screened_spmm` carry their analytic adjoints, and `bsr_attn` carries the
  NKI dQ / dK-dV backward (via `bsr_attn_bwd`). Gradients reach
  `CSRMatrix.values`, `BSRMatrix.blocks` and the dense operands through the
  host-side densify / pad / gather. `_SpMMFunction`, `_BSRSpMMFunction`,
  `_ScreenedSpMMFunction` and `_AttnTiledFunction` remain for the XLA path,
  the simulator and the PyTorch backend. The native BSR backward has no
  per-block Python loop.
- Placement (#30): the `neuron` device has no sparse CSR tensors
  (aws-neuron-sdk#1442), so sparse structure stays on the host and only
  dense data moves. `spmm`, `spmv`, `spmv_symmetric`, `bsr_spmm`,
  `screened_spmm`, `block_sparse_attention_tiled`, `cg_bsr` and
  `jacobi_preconditioner_bsr` accept host CSR/BSR structure with dense
  operands (and BSR `blocks`) on `neuron`, and return results on the dense
  operand's device. A host CSR against a device operand is densified on the
  host without a torch sparse tensor (`formats._csr_to_dense_host`, static
  shapes), so `spmm` with host-CSR structure traces under
  `torch.compile(..., fullgraph=True)`. Index tensors found on a device are
  read through host copies. Documented in `docs/installation.md` and the
  format and op docstrings.
- The dispatch path is detected on the first kernel launch: native when
  `torch_neuronx` exports `nki_op` and a `neuron` device is available, XLA
  when `torch_xla` is installed, otherwise none. The XLA path (Trn1) is
  unchanged, as are the PyTorch fallback and the `TRNSPARSE_USE_SIMULATOR`
  path. All kernel launches share one helper (`dispatch._launch`, or its
  halves `_to_device` / `_run` where several attention kernels share
  operands).
- Custom ops are registered lazily (`trnsparse/nki/_native.py` and
  `_native_attn.py` are imported only when the native path is selected).
- `TRNSCI_NKI_DISPATCH=auto|native|xla|cpu` forces the dispatch path (a
  suite-wide variable). Forcing an unavailable path raises instead of
  falling back. `TRNSCI_XLA_DEPRECATION=1` makes the XLA path emit a
  one-time `DeprecationWarning` (off by default).
- New `[torchneuron]` extra (`torch>=2.14`, `torch-neuronx>=2.14`,
  `nki>=0.6`); `[neuron]` stays the PyTorch/XLA lineage.
- Tests: `conftest.py` sets `NKI_ENABLE_TRACE_CACHE=0` by default.
  `tests/test_native_dispatch.py` (CPU) covers path selection, launch
  routing, lazy registration, op schemas and fake shapes, the registered
  backward formulas, the native wrappers end to end on CPU with real custom
  ops, and a CPU `fullgraph=True` trace of host-CSR `spmm`.
  `tests/test_nki_native.py` (`@pytest.mark.neuron`, native only) covers
  host-structure / device-operand calls, gradients, and
  `torch.compile(backend="neuron", fullgraph=True)` of `spmm` with
  host-CSR structure. Not yet run on hardware.

## [0.7.0] — 2026-10-28

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron device) — see trnsci/trnsci#35.

### Fixed

- PyTorch reference attention (`_block_sparse_attn_pytorch` and the
  recompute branch of `_block_sparse_attn_backward`) allocates on Q's
  device instead of the default device.

- **NKI 0.7.0 / neuronx-cc 2.28 kernel compatibility (#29).** Kernels that
  passed `nki.simulate` failed to compile on Trn2 (Neuron SDK 2.33):
  - `_attn_out_kernel`, `_attn_bwd_dq_kernel`, `_attn_bwd_dkdv_kernel`:
    softmax normalisation `nl.divide(exp, row_denom)` by a `(128, 1)`
    denominator lowered to `tensor_scalar_arith`, which no longer supports
    `divide`. Now multiplies by `nl.reciprocal(row_denom)` (hoisted per
    block-row in the forward and dQ kernels).
  - `_screened_spmm_kernel`: the `(128, 1) * (1, 128)` Schwarz outer product
    and the `(1, 1)` threshold subtract produced partition-count mismatches
    (`operand0 partition total elements 128 != dst partition total elements
    1`). `q_k` is now `nl.broadcast_to` `(128, 128)` and the threshold is
    loaded once and broadcast to `(128, 1)`. The soft-mask tensor-tensor
    divide (fails ISA validation `s3s3d3_tt_valid_op`) is now a reciprocal
    multiply.
  - `nki_bsr_attn_bwd`: the host-side `(..., b) → (..., b, 1)` unsqueeze
    for per-row stats matched `q_scaled` / `k_blocks` / `v_blocks` by shape
    whenever `head_dim == 128` (and in the simulator's padded path), feeding
    the kernels a 4-D tensor. Stats are now selected by name.
  - Dispatch: kernels always receive fp32 (Trainium has no fp64 datapath;
    the native device downcast fp64 silently) and results are cast back to
    the caller's dtype. Fixes `expected m1 and m2 to have the same dtype,
    but got: float != double` in the PyTorch-level backward under fp64
    gradcheck. The hardware gradcheck tests now use `eps=1e-2,
    atol=rtol=1e-2`, sized for an fp32 forward (the ops are linear in each
    input, so the larger eps adds no truncation error).

- **Block-sparse attention: two host-side gather bugs** (pre-existing, masked
  on Trn2 by the compile failures above; found with neuronx-cc BIRSim):
  - Rows with fewer than `K_max` blocks were padded with sentinel k-slots that
    gathered column block 0, so block 0 was counted again in the softmax and
    in `weights @ V` (local-window masks: ~49% of outputs wrong). Padded slots
    now have K/V (forward, dQ) or Q/dO (dK/dV) zeroed, and their stats are
    dropped in `_attn_host_reduction`.
  - `_attn_bwd_gather` gathered `dO` by column block instead of the row block
    `dO_m`, breaking dQ whenever a row attends to a block other than itself
    (~99% of dQ wrong). The two simulator xfails attributed to a "PSUM
    accumulate bug" (`test_bwd_dq_parity_kmax2`, `test_backward_head_dim_256`)
    were this bug and now pass.

### Added

- `tests/test_attn_gather.py`: CPU regression tests for both gather bugs.
- `tests/test_nki_compile.py` (`nki_simulator` marker): CPU MLIR
  verification (`source_to_mlir(...).verify()`) of every kernel for trn1
  and trn2, attention kernels at head_dim 64, 128 and 256; plus a full NEFF
  compile per kernel/target when `neuronx-cc` is on PATH.

## [0.6.0] — 2026-04-17

### Added

- **K-tiling for head_dim > 128** — NKI attention kernels now support
  `head_dim=256` and `head_dim=512` via an inner K-tile loop that splits
  the `Q @ K.T` score computation across `TILE_K=128`-sized chunks of
  `head_dim`, accumulating each into the existing PSUM.
  - All 4 NKI kernels updated: `_attn_stats_kernel`, `_attn_out_kernel`,
    `_attn_bwd_dq_kernel`, `_attn_bwd_dkdv_kernel`.
  - For `head_dim ≤ 128`: single-tile path preserved unchanged — Q is
    loaded once per block-row as a stationary tile (no regression).
  - For `head_dim > 128`: K-tile inner loop (`affine_range(head_dim //
    128)`). `head_dim` must be a multiple of 128 (covers 256, 512).
  - `weights @ V` and all dK/dV accumulation matmuls are unchanged (K=128
    block dimension, unaffected by head_dim growth).
- `tests/test_attention.py`: `test_head_dim_256_parity`,
  `test_head_dim_256_dilated_parity`, `test_gradcheck_head_dim_256`,
  `test_backward_shapes_head_dim_256` in PyTorch path.
- `tests/test_nki_sim.py`: `TestAttnKTilingSimulator` — simulator parity
  tests for forward and backward at head_dim=256.
- `tests/test_nki_attn.py`: `TestAttnHardwareLargeDim` — hardware parity
  tests at seq_len=512, head_dim=256.

### Changed

- `_attn_gather` assertion relaxed from `head_dim ≤ 128` to `head_dim ≤ 128
  or head_dim % 128 == 0`.
- `block_sparse_attention_tiled` and `nki_bsr_attn_tiled` docstrings updated
  to document K-tiling support.

## [0.5.1] — 2026-04-16

### Added

- **NKI backward kernel pair** (`_attn_bwd_dq_kernel` + `_attn_bwd_dkdv_kernel`)
  in `trnsparse/nki/kernels.py`. When the `nki` backend is active,
  `_AttnTiledFunction.backward` now dispatches to these kernels instead of
  the Python-loop backward.
  - `_attn_bwd_dq_kernel`: row-first traversal, mirrors `_attn_out_kernel`.
    Accumulates `dQ_m += dS @ K_ki * scale` in a PSUM tile per block-row.
  - `_attn_bwd_dkdv_kernel`: column-first traversal (BSC view built on host).
    For each column block ki, iterates over all block-rows m that attend to ki,
    accumulating `dK_ki += dS.T @ Q_m * scale` and `dV_ki += P.T @ dO_m`.
  - No atomic scatter required — column-first gather on host eliminates NKI's
    lack of cross-block accumulation primitives.
- **Stats saving in forward** — `_AttnTiledFunction.forward` now saves
  `(Q, K, V, O, row_max, row_denom)` in ctx (6 tensors vs 4). Backward
  receives the pre-computed row stats and skips the two recomputation passes.
  Applies to both PyTorch and NKI paths.
- `_attn_bwd_gather` and `nki_bsr_attn_bwd` in `trnsparse/nki/dispatch.py`
  — host-side BSC construction and NKI backward orchestration.
- `tests/test_nki_sim.py`: `TestAttnBwdSimulator` — simulator correctness tests
  for dQ/dK/dV against PyTorch backward (local window + dilated patterns).
- `tests/test_nki_attn.py`: `TestAttnBwdHardware` — hardware parity tests
  at seq_len=512, head_dim=64.

### Changed

- `nki_bsr_attn_tiled` accepts `return_stats=True` kwarg (default False) to
  return `(out, row_max, row_denom)`.
- `_block_sparse_attn_pytorch` accepts `return_stats=True` kwarg similarly.
- `_block_sparse_attn_backward` accepts optional `row_max` / `row_denom`
  kwargs; when provided, skips the two recomputation BSR passes.

## [0.5.0] — 2026-04-16

### Added

- **Block-sparse attention autograd** — `block_sparse_attention_tiled` is now
  differentiable. When any of Q/K/V has `requires_grad=True`, the call routes
  through `_AttnTiledFunction`, which provides a tiled backward using the Flash
  Attention delta identity (`D_i = dO_i · O_i`).
- **`_block_sparse_attn_backward(Q, K, V, O, dO, mask_bsr, scale)`** — internal
  backward function. Single pass over nonzero BSR blocks; no O(seq_len²)
  intermediate. Row stats (row_max/row_denom) are recomputed in backward.
- `gradcheck` passes at `atol=1e-3` for local-window and dilated patterns
  (float64, seq_len=256, head_dim=32). See `tests/test_attention.py::TestAttnTiledGrad`.

### Notes

- NKI backward kernel is a follow-up (v0.5.x). The current backward runs
  on PyTorch even when the NKI forward path is active.
- Row stats (row_max/row_denom) are recomputed in backward rather than saved
  from forward — avoids O(n_blocks × b) storage in the autograd graph at the
  cost of two extra BSR passes.

## [0.4.4] — 2026-04-16

### Added

- **NKI attention kernel pair** (`_attn_stats_kernel` + `_attn_out_kernel`,
  closes [#25](https://github.com/trnsci/trnsparse/issues/25)). Two-pass
  block-sparse attention now runs on the Tensor Engine via these kernels when
  the `nki` backend is active. The `pytorch` backend retains the Python-loop
  reference path from v0.4.3.
  - Pass 1 (`_attn_stats_kernel`): each `(m, ki)` block pair computed
    independently via `nc_matmul`; `nl.max` / `nl.sum` within-tile reductions
    produce `(tile_max, tile_sumexp)` with no carry between iterations.
  - Pass 2 (`_attn_out_kernel`): recomputes scores, applies stable softmax
    using `row_max` / `row_denom` loaded per block-row (static HBM offset),
    accumulates `weights @ V` into a PSUM tile spanning all ki blocks.
  - **Constraint:** `head_dim ≤ 128` (`nc_matmul` partition limit). `head_dim=256`
    requires K-tiling and is a follow-up.
- `_attn_gather`, `_attn_host_reduction`, `nki_bsr_attn_tiled` in
  `trnsparse/nki/dispatch.py` — host-side gather and orchestration for the kernel
  pair, mirroring the `_bsr_pad_and_gather` pattern.
- `tests/test_nki_sim.py`: `TestAttnTiledSimulator` — simulator correctness
  tests (local window, dilated; seq_len=256, head_dim=32).
- `tests/test_nki_attn.py`: hardware tests for local window, dilated, and
  global-token patterns at seq_len=512, head_dim=64.

### Notes

No autograd wrapping in v0.4.4 — forward-only, matching the existing PyTorch
path. Attention backward (block-sparse) is a follow-up.

## [0.4.3] — 2026-04-16

### Added

- **`chebyshev_bsr(A, b, lam_min, lam_max, K)`** — fixed-K Chebyshev
  semi-iteration. Pre-computes all step coefficients from eigenvalue bounds
  before the loop; no inner products during iteration. Same convergence rate
  as CG without the adaptive coefficient computation that blocks NKI fusion.
  Returns `(x, K, rel_residual)` matching the `cg_bsr` signature.
- **`chebyshev_coeffs(lam_min, lam_max, K)`** — returns the `(alpha, beta)`
  coefficient tensors for the momentum update `x_{k+1} = x_k + α_k r_k +
  β_k (x_k - x_{k-1})`. Useful for inspecting or caching coefficients across
  solves with the same matrix spectrum.
- **`richardson_bsr(A, b, omega, K)`** — fixed-K Richardson iteration; the
  simplest fixed-point iteration with no coefficient computation at all.
  Optimal omega = `2 / (lam_min + lam_max)`. Baseline comparator for Chebyshev.
- **`block_sparse_attention_tiled(Q, K, V, mask_bsr)`** — two-pass sparse
  attention that avoids the O(seq_len²) score intermediate. Pass 1 computes
  per-block `(tile_max, tile_sumexp)` statistics over nonzero blocks only
  (O(n_blocks × block_size) stats array, e.g. 5 KB at seq_len=512 vs 1 MB
  dense). Pass 2 recomputes scores, applies stable softmax using host-reduced
  row statistics, and accumulates V. PyTorch reference for the NKI kernel pair
  documented in [#25](https://github.com/trnsci/trnsparse/issues/25).
- **`docs/sparse_attention.md`**: new "Tiled two-pass path" section explaining
  the algorithm, memory trade-offs, and the NKI kernel follow-up path.
- **`docs/architecture.md`**: "Known limits" updated — #22 and #25 now have
  v0.4.3 PyTorch references with clear notes on remaining NKI gate conditions.

### Notes

Addresses the architecture-friendly workarounds identified in the analysis of
parked NKI-capability-gated issues (#22, #25). Both solvers and the tiled
attention path share the structural property of fixed-iteration loops with
statically-determined coefficients — the pattern that maps to `nl.affine_range`
without scalar carry.

## [0.4.2] — 2026-04-15

### Added

- **`examples/block_sparse_attention.py`** — block-sparse attention
  reference using `BSRMatrix` + `bsr_spmm`. Three mask patterns
  (local window, dilated, global tokens); verifies against a dense
  reference; reports block density and timing for the `bsr_spmm` step.
  Closes [#21](https://github.com/trnsci/trnsparse/issues/21).
- **`docs/sparse_attention.md`** — writeup: how BSR-128 maps to
  Longformer/BigBird-style attention masks, block density arithmetic,
  pattern construction helpers, and the fused-tile follow-up.
- **`docs/iterative_solvers.md`** now linked in `mkdocs.yml` nav
  (was present but missing from navigation).
- **`tests/test_attention.py`** — 8 CPU tests: mask shape/symmetry
  checks + parity against dense reference at `atol=1e-4` for all three
  patterns and the full-attention edge case.

### Notes

No API changes, no kernel changes. The claim in #21 is: `bsr_spmm` is
the block-sparse attention primitive; `BSRMatrix` captures the mask.
The example and docs make that explicit.

## [0.4.1] — 2026-04-14

### Added

- **`examples/sparse_fock.py`** rewritten around v0.4.0's
  `screened_spmm`. Three paths side-by-side on the same inputs:
  (1) v0.1.x unfused `schwarz_bounds → screen → from_dense → spmm`;
  (2) v0.4.0 fused `screened_spmm` (one call);
  (3) full Fock build — the coulomb from path 2 contracted against MO
  coefficients via `trnblas.gemm` for `F_MO = C.T @ J @ C` (falls
  back to `torch.matmul` if trnblas isn't installed). On a 50-basis
  synthetic system, the fused path is ~130× faster than the unfused
  (dominated by eliminating the Python `from_dense` CSR construction).
  Closes #6.
- **`examples/pyscf_bridge.py`** (new) — optional PySCF-driven demo.
  Builds H2O (or benzene, or H2), pulls real AO ERIs via
  `mol.intor("int2e")`, feeds the `(μμ|μμ)` diagonal into
  `schwarz_bounds` + `screened_spmm` against a mock density matrix.
  Reports realistic sparsity at `threshold=1e-8`. Requires
  `pip install pyscf`; tests skip cleanly if not available.
  Closes #13.
- **`tests/test_examples.py`** — 2 CPU smoke tests plus a
  PySCF-gated test. Exercises the `sparse_fock` unfused + fused
  paths end-to-end and asserts parity (`atol=1e-6`).

### Notes

No API changes, no kernel changes — pure integration demo release.
Users already on v0.4.0 can stay there; upgrade to v0.4.1 only to
pick up the new examples.

## [0.4.0] — 2026-04-14

### Added

- **`screened_spmm(A, diag_integrals, B, threshold)`** — fused Schwarz-
  screened dense matmul. One NKI kernel fuses the full pipeline —
  outer-product pair bound → threshold → mask-apply → `nc_matmul` —
  into a single dispatch. Saves ~30–50% end-to-end vs the unfused
  `density_screen + from_dense + spmm` flow on Fock-build-sized inputs.
  Closes #19.
- **`_screened_spmm_kernel`** — new `@nki.jit` kernel in
  `trnsparse/nki/kernels.py`. Stationary-A-tile-reuse GEMM extended with
  a per-tile pair-bound mask built from the 1-D Schwarz-bound vector.
- **`_ScreenedSpMMFunction`** — `torch.autograd.Function` wrapper.
  Third differentiable NKI kernel in the trnsci suite (after v0.2.0
  CSR SpMM and v0.3.0 BSR SpMM). `torch.autograd.gradcheck` passes at
  `atol=1e-4` on hardware. Mask is non-differentiable (discrete gate);
  gradients flow to `A` (masked) and `B` (transposed masked A) only.
- **Tests**: 4 CPU (`TestScreenedSpmm`), 2 simulator
  (`TestScreenedSpmmSimulator`), 7 hardware
  (`TestNkiScreenedSpmmParity` + `TestNkiScreenedSpmmDifferentiability`).
  All green on `trn1.2xlarge`.
- **`docs/architecture.md`** — new "Fused screened SpMM" section.

### Closed

- [#24](https://github.com/trnsci/trnsparse/issues/24) — fused-CG NKI
  kernel was not buildable under NKI 2.24/0.3.0 constraints (no break,
  no iteration-carried scalar state across `affine_range`, no nested
  kernels). Per-iteration `_cg_step_kernel` reframe evaluated and
  found to save only 5–20% — not worth the authoring cost relative to
  #19's genuine 30–50% savings. See #24 close comment for the audit.

### Known limits

- Restricted to square `A` (`M == K`) with 1-D `diag_integrals`.
  Rectangular / asymmetric-bounds extension is a follow-up if asked for.

## [0.3.2] — 2026-04-14

### Added

- **`cg_bsr`** and **`power_iteration_bsr`** — Conjugate Gradient and
  power iteration on block-sparse row matrices. Plumbing on top of
  `bsr_spmm` (one kernel dispatch per iteration). Closes Phase 1 of
  #22 on-chip iterative solvers.
- **`jacobi_preconditioner_bsr(A)`** — builds a diagonal preconditioner
  for `cg_bsr`'s `M=` argument.
- **`bsr_diagonal(A)`** — extracts the main diagonal from a BSR matrix.
- **`docs/iterative_solvers.md`** — design note covering the v0.3.2
  plumbing and the v0.4.0 fused-kernel goal (#24). Explains the
  architectural win Trainium offers (A SBUF-resident across iterations)
  vs the current per-iteration HBM round-trip.
- **`tests/test_iterative.py`** — 8 CPU tests including scipy parity at
  `atol=1e-4` on a 128×128 SPD system.
- **`benchmarks/bench_iterative.py`** — cg_bsr vs scipy.sparse.linalg.cg.
  At 128×128 SPD: scipy 310 μs, trnsparse 369 μs (1.19×).

### Notes

- Algorithm body for CG is a local copy of `trnsolver.iterative.cg`;
  kept local to avoid a cross-repo runtime dependency for one function.
- v0.4.0 will layer the fused CG/power-iteration NKI kernel on top —
  tracked in #24. The API stays stable across the transition; users
  upgrading from v0.3.2 get the fused-kernel speedup automatically
  when the fused path is available.

## [0.3.1] — 2026-04-14

### Changed

- **Migrated NKI imports to the `nki.*` namespace** (NKI 0.3.0 Stable,
  Neuron SDK 2.29, April 2026). Legacy `neuronxcc.nki.*` shim is no
  longer used. `pyproject.toml` `[neuron]` extra gains `nki>=0.3.0`
  alongside the existing `neuronxcc>=2.24` and `torch-neuronx>=2.9`.
  Hosts without an `nki` wheel (macOS, non-Linux archs) still hit
  `HAS_NKI=False` and get the torch fallback. Kernel bodies unchanged —
  the trnblas audit confirmed the positional `nisa.nc_matmul` +
  `nl.copy(psum, ...)` pattern complies with NKI 0.3.0.
- `test` CI job now filters `-m "not neuron and not nki_simulator"` so
  each test runs in exactly one job.

### Added

- **`TRNSPARSE_USE_SIMULATOR=1` dispatch branch** through
  `nki.simulate(kernel)(np_args)`. Bypasses torch_xla + NEFF compile;
  kernels run on CPU for correctness iteration. Hardware still owns
  perf numbers.
- **`nki-simulator` CI job on `ubuntu-latest`** — installs `nki>=0.3.0`
  from the AWS pip index and runs the simulator suite on every push/PR.
  Kernel correctness gate without AWS cost. Catches Python-trace-level
  errors (bad kwargs, dropped ops, shape mismatches); MLIR verifier
  errors remain hardware-only (NKI 0.3.0 has no documented device-free
  NEFF compile API).
- `tests/test_nki_sim.py` — curated simulator suite (4 tests: CSR
  aligned + rectangular, BSR block-dense + block-diagonal). Skips
  cleanly off-hardware.
- `scripts/run_simulator_tests.sh` — SSM runner mirroring
  `run_neuron_tests.sh` with `TRNSPARSE_USE_SIMULATOR=1` in the env.
- `tests/conftest.py` — registers the `nki_simulator` pytest marker.

Addresses [trnsci/trnsparse#23](https://github.com/trnsci/trnsparse/issues/23).
Follows the trnblas reference commits `c693561`, `f24993b`, `77eeb82`
(suite-wide coordination in `trnsci/trnsci#5`).

## [0.3.0] — 2026-04-13

### Added

- **`BSRMatrix`** — block-sparse row format at 128×128 (the Tensor-Engine
  tile size). Every nonzero block is already a dense tile that maps
  one-to-one to `nisa.nc_matmul`. Conversions from `CSRMatrix` and dense
  plus back. See `docs/architecture.md` for why BSR is the Trainium-native
  sparse representation.
- **`bsr_spmm(A_bsr, B)`** with NKI + PyTorch dispatch. On NKI, routes
  through `_BSRSpMMFunction` (suite-second `torch.autograd.Function`-
  wrapped kernel after v0.2.0's CSR SpMM). Per-block `nc_matmul` with
  zero gather overhead — uniform K_max per block-row via host-side
  zero-padding.
- **`tests/test_bsr.py`** — 9 CPU tests (format roundtrips, SpMM parity
  across block densities + rectangular shapes).
- **`tests/test_nki_bsr.py`** — 7 `@pytest.mark.neuron` tests including
  `torch.autograd.gradcheck` at `atol=1e-4`. Validated on
  `trn1.2xlarge`.
- **`benchmarks/bench_bsr_spmm.py`** — BSR PyTorch + BSR NKI + dense
  GEMM ceiling across `(m_blocks, n_blocks, block_density, N)`.
- **`docs/benchmarks.md`** BSR section with real hardware numbers + an
  honest reading of why v0.3.0 BSR NKI doesn't beat CPU at small sizes
  (kernel dispatch overhead dominates; architectural wins are in
  follow-up issues #19, #20, #21).
- **`docs/architecture.md`** lede rewritten around "why BSR is
  Trainium-native."
- **`sparse_add`** no longer materializes an `N×N` dense intermediate —
  uses `torch.sparse_coo_tensor.coalesce()` for pattern union. Closes #8.
- **`density_screen`** test coverage (false-negative check + degenerate
  thresholds). Closes #10.

### Changed

- Issue #15 (row-bucketing CSR) demoted to backlog. Under the
  architectural frame, the CSR path is served by the PyTorch fallback
  (v0.1.3) and BSR is the NKI-side story. Row-bucketing would only help
  if NKI 2.24 exposed an indirect-DMA primitive, which it doesn't.

### Closed

- #8 sparse_add pattern union
- #9 density_screen bound clarification (current 2D bound is already tight)
- #10 density_screen test coverage
- #12 scipy migration guide
- #18 BSR format + NKI block-sparse SpMM

### New (architectural roadmap)

- #19 Fused screen + matmul NKI kernel
- #20 On-chip iterative solvers over BSR (SBUF-resident A)
- #21 Block-sparse attention primitive

## [0.2.0] — 2026-04-13

### Added

- **NKI SpMM kernel** validated on `trn1.2xlarge`. `set_backend("nki")`
  routes `spmm` through `trnsparse.nki.kernels._spmm_dense_kernel`,
  which runs stationary-tile-reuse GEMM on the Tensor Engine.
- **`torch.autograd.Function` wrapping** (`_SpMMFunction`) with analytic
  backward — the first differentiable NKI kernel in the trnsci suite.
  Satisfies [`trnsci/trnsci#3`](https://github.com/trnsci/trnsci/issues/3);
  validated via `torch.autograd.gradcheck` at `atol=1e-4`.
- `tests/test_nki_spmm.py` — hardware-gated parity + gradcheck coverage
  for tile-aligned, unaligned, and low-density inputs.
- `benchmarks/bench_spmm.py` — four-backend SpMM table
  (scipy / torch.sparse / trnsparse pytorch / trnsparse nki) in one
  pytest pass.
- `docs/benchmarks.md` populated with real `trn1.2xlarge` numbers.

### Changed

- `docs/architecture.md` describes the v0.2.0 SpMM dispatch path end to
  end (materialize → pad → NKI GEMM → slice) and documents the known
  dense-materialization cost that lands for row-bucketing (#15) in v0.3.0.

### Known limits

- SpMM NKI is slower than CPU backends in v0.2.0 — the dense materialization
  removes the sparsity advantage. This is Phase 1 (correctness). Row-
  bucketing in Phase 3 (#15) is where sparse speedups live.
- SpMV stays on the PyTorch path — single-output-column NKI dispatch
  doesn't amortize compile + HBM round-trip cost.

Closes #14 (Phase 1). Addresses #4 (NKI column populated). Unblocks #15
(Phase 3 perf).

## [0.1.3] — 2026-04-12

### Changed

- `spmv`, `spmm`, `spmv_symmetric`, and `CSRMatrix.to_dense` now lower
  to `torch.sparse_csr_tensor` operations instead of per-row Python loops.
- Measured on CPU (256×256, density 0.01) the change is 26× faster for
  SpMV (958 μs → 37 μs) and 52–88× faster for SpMM (1.2 ms → 13–24 μs
  depending on RHS width), putting trnsparse's PyTorch fallback within
  2× of `torch.sparse`.

Does not affect public API or numeric outputs — existing tests pass
unchanged. NKI backend remains scaffolded (routing lands in v0.2.0).

### Added

- `benchmarks/` directory (`conftest.py`, `bench_spmv.py`, `bench_spmm.py`,
  `bench_screening.py`) running trnsparse vs `scipy.sparse` vs
  `torch.sparse` on the same numeric inputs. Closes #11; partial #4.

## [0.1.2] — 2026-04-12

### Changed

- Sync `trnsparse.__version__` with `pyproject.toml` (both now `0.1.2`).
  Previously `__init__.py` reported `0.1.0` while the package version was `0.1.1`.
- Docs badge in `README.md` and `site_url` in `mkdocs.yml` point at
  `trnsci.dev/trnsparse/` instead of `trnsci.github.io/trnsparse/`. Per-repo
  GitHub Pages is superseded by the centralized trnsci.dev site.
- `docs/architecture.md` clarifies that the NKI backend is scaffolded only —
  the PyTorch path runs regardless of `set_backend` in v0.1.x. Routing +
  on-hardware validation land in v0.2.0.

## [0.1.1] — 2026-04-12

### Added

- mkdocs site with `index`, `installation`, `quickstart`, `api`, `architecture`, `aws_setup`
- `infra/terraform/` for on-hardware CI instance provisioning
- `scripts/run_neuron_tests.sh` and benchmark helpers
- GitHub Actions `ci.yml` for CPU-only pytest matrix
- `Issues` URL in pyproject.toml

### Changed

- Bumped `neuronxcc` floor from `>=2.15` to `>=2.24` to unify with the
  rest of the trnsci suite. `torch-neuronx` floor bumped to `>=2.9`.

## [0.1.0] — 2026-04-12

### Added

- Initial scaffold: CSRMatrix / COOMatrix, SpMV / SpMM, Schwarz screening
- NKI dispatch with gather-matmul-scatter kernel stub
- `examples/sparse_fock.py` — screened Fock build demo
