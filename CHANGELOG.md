# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
