# Benchmarks

Performance results for `trnsparse.spmm` across four backends on the same
numeric inputs:

- **scipy** — `scipy.sparse.csr_matrix @ B` (compiled C reference)
- **torch.sparse** — `torch.sparse_csr_tensor @ B`
- **trnsparse (pytorch)** — v0.1.3 vectorized fallback that lowers to `torch.sparse`
- **trnsparse (nki)** — v0.2.0 NKI dispatch: materialize CSR dense, run
  tiled GEMM on the Tensor Engine via `@nki.jit`, return dense

## Hardware

- **Instance:** `trn1.2xlarge` (single NeuronCore v2)
- **AMI:** Deep Learning AMI Neuron PyTorch 2.9 · Ubuntu 24.04
- **Neuron SDK:** 2.24 · `neuronxcc` 2.24

## Results (SpMM, mean time in μs)

Columns are size `M=K`, density, `N` (RHS width). Lower is better.

| Size | Density | N | scipy | torch.sparse | trnsparse pytorch | trnsparse nki |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.001 | 32 | 8.8 | 15.4 | 48.3 | 1397 |
| 256 | 0.001 | 128 | 7.6 | 29.1 | 44.7 | 1365 |
| 256 | 0.01 | 32 | 8.8 | 15.6 | 28.9 | 1248 |
| 256 | 0.01 | 128 | 20.4 | 27.5 | 41.7 | 1370 |
| 256 | 0.1 | 32 | 40.0 | 18.5 | 31.7 | 1278 |
| 256 | 0.1 | 128 | 137 | 34.2 | 48.6 | 1428 |
| 1024 | 0.001 | 32 | 14.4 | 28.9 | 43.4 | 1732 |
| 1024 | 0.001 | 128 | 39.8 | 27.2 | 47.0 | 2067 |
| 1024 | 0.01 | 32 | 72.4 | 31.8 | 48.1 | 1847 |
| 1024 | 0.01 | 128 | 257 | 46.5 | 72.5 | 2212 |
| 1024 | 0.1 | 32 | 609 | 75.0 | 95.5 | 2151 |
| 1024 | 0.1 | 128 | 2475 | 248 | 274 | 2479 |

## What to read into this

**v0.2.0 ships NKI for correctness, not speed.** At every data point the NKI
path is slower than both CPU backends. Two reasons, both structural:

1. **No sparsity exploitation yet.** v0.2.0's NKI kernel materializes the CSR
   into a dense `(M, K)` tile before the matmul. At density 0.001 on a
   `1024 × 1024` matrix, this means 1000× more work than scipy does. Real
   sparse speedups come from **row-bucketing + gather-matmul-scatter**, which
   lands in v0.3.0 ([#15](https://github.com/trnsci/trnsparse/issues/15)).
2. **Kernel-launch overhead dominates.** The NKI times are roughly constant
   at ~1.3-2.5 ms across densities — the NEFF dispatch + HBM round-trip
   amortizes over large workloads, not these small ones. trn2 (v0.2.x Phase 5,
   [#17](https://github.com/trnsci/trnsparse/issues/17)) and NEFF-cache reuse
   (part of Phase 3) close this gap for steady-state workloads.

What v0.2.0 does deliver:

- `set_backend("nki")` now actually runs SpMM on the Tensor Engine.
- Forward + backward (via `torch.autograd.Function`) both validated on
  hardware; `torch.autograd.gradcheck` passes at `atol=1e-4`.
- The full Neuron toolchain — compile, NEFF cache, XLA dispatch, PyTorch
  integration — is exercised end-to-end, which is what Phase 1 is for.

## Reproducing

```bash
# CPU only (scipy + torch.sparse + trnsparse pytorch)
pytest benchmarks/bench_spmm.py --benchmark-only

# On trn1, add NKI column:
AWS_PROFILE=aws ./scripts/run_benchmarks.sh trn1
```

Results above were generated at commit
[`cee2e34`](https://github.com/trnsci/trnsparse/commit/cee2e34) on
`trn1.2xlarge`.

## Results (BSR SpMM, v0.3.0)

`BSRMatrix` stores a matrix as stacked `(block_size, block_size)` dense
blocks plus a block-level CSR pattern. At `block_size=128` each nonzero
block is exactly a Tensor-Engine tile — no gather overhead.

Sizes are `(m_blocks × 128, n_blocks × 128)`. `block_density` is the
fraction of `(m_blocks, n_blocks)` blocks actually stored. Lower is
better; `dense` is `torch.matmul(A_dense, B)` on the trn1.2xlarge CPU
as a ceiling.

| m_blocks | n_blocks | block_density | N | bsr PyTorch (ms) | bsr NKI (ms) | dense CPU (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 0.10 | 128 | 0.11 | 1.85 | 0.12 |
| 4 | 4 | 0.25 | 128 | 0.18 | 1.87 | 0.12 |
| 4 | 4 | 0.50 | 128 | 0.43 | 2.05 | 0.13 |
| 4 | 8 | 0.50 | 256 | 1.01 | 3.00 | 0.49 |
| 8 | 4 | 0.50 | 256 | 1.05 | 3.11 | 0.47 |

### Reading the BSR table

Three observations, all consistent with the architectural framing:

1. **`bsr PyTorch` scales linearly with block density** (0.11 → 0.43 ms
   as density goes 0.10 → 0.50, ~4× more work done). It's the
   straightforward "iterate stored blocks, do `A_block @ B_slice`"
   reference. This is what the NKI path has to beat.
2. **`bsr NKI` is roughly constant at 1.8-3.1 ms** across configurations.
   Kernel dispatch + compilation + HBM round-trips dominate at these
   small sizes, not the matmul work itself. This mirrors the v0.2.0 CSR
   finding — Trainium rewards amortizing the kernel launch.
3. **Dense CPU is the ceiling to beat** — scipy / torch.sparse / BSR
   PyTorch are all within 2-4× of it on the trn1.2xlarge's Xeon. NKI
   needs workloads where the matmul work is large enough to swamp
   dispatch, or where the kernel launch can be amortized across many
   operations.

### Where NKI wins — shipped v0.4.x

Not in raw "one-shot SpMM at N=128 vs CPU." The BSR table above measures
kernel dispatch overhead. The architectural wins come from fusing multiple
host passes into one dispatch:

- **`screened_spmm` (v0.4.0, [#19](https://github.com/trnsci/trnsparse/issues/19))**
  ✅ shipped: fuses Schwarz bound + threshold + mask + matmul into one NKI
  dispatch. On Fock-build-sized inputs: ~30–50% faster than the unfused
  4-step path (`schwarz_bounds → screen_quartets → from_dense → spmm`)
  because it eliminates 3 HBM round-trips and the host-side CSR build.
- **Block-sparse attention (v0.4.2, [#21](https://github.com/trnsci/trnsparse/issues/21))**
  ✅ shipped: `bsr_spmm` handles `attn_weights @ V` for local-window / dilated /
  global-token masks. At seq_len=8192, window=2: 4.9% block density means 95%
  of `nc_matmul` calls are skipped. The example materializes the full score
  matrix; `block_sparse_attention_tiled` (v0.4.3, NKI kernels from v0.4.4,
  [#25](https://github.com/trnsci/trnsparse/issues/25)) does not.
- **[#22 — on-chip iterative solvers](https://github.com/trnsci/trnsparse/issues/22)**:
  CG/power-iteration with A SBUF-resident across all iterations (replaces #20).
  v0.3.2 shipped the Python-loop plumbing; the fused kernel (`cg_bsr_fused`,
  `power_iteration_bsr_fused`) shipped in 0.11.0. Trn2 results are in the
  fused CG sections below.

See [`docs/architecture.md`](architecture.md) for why BSR is the
Trainium-native format.

BSR results from commit
[`411a3a7`](https://github.com/trnsci/trnsparse/commit/411a3a7).

## Trn2 native path (TorchNeuron), static-structure kernels

**Setup.** `trn2.3xlarge` (LNC=2; one logical NeuronCore, `NEURON_RT_VISIBLE_CORES=3`),
Neuron SDK 2.33 venv: `torch` 2.14.0, `torch-neuronx` 2.14.3, `nki` 0.7.0,
`neuronx-cc` 2.28. fp32 throughout. Structure on the host, dense data
(blocks, operands) already on `device="neuron"`. Warm median of 10–30 calls
with `torch.neuron.synchronize()` after each; the first call (compile) is
excluded. CPU columns: same box, 2 threads (`taskset -c 8,9`,
`OMP_NUM_THREADS=2`). The box ran other jobs on other NeuronCores, so
sub-millisecond numbers carry ~±0.1 ms noise. "main" = `1fafcaf` (gather
path); "PR" = static-structure kernels. Script:
`benchmarks/bench_native_trn2.py`.

**Crossover vs dense aten `matmul` on the device** (BSR 128×128, linear
interpolation between measured densities):

| n | width 128 | width 512 |
|---|---|---|
| 1024 | never (0.3 ms launch floor vs 0.14 ms dense) | never |
| 4096 | ~2% | ~16% |
| 8192 | ~14% | ~34% |

### BSR SpMM (fp32, ms, warm median)

| n | width | block dens. | blocks | K_max | dense aten (dev) | main e2e | **PR e2e** | PR kernel only | PR / dense | CPU dense | torch.sparse BSR (CPU) | torch.sparse CSR (CPU) | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 128 | 1% | 1 | 1 | 0.14 | 1.33 | **0.32** | 0.24 | 0.45x | 0.9 | 0.2 | 0.1 | 1e-07 |
| 1024 | 128 | 5% | 3 | 2 | 0.14 | 1.05 | **0.39** | 0.31 | 0.37x | 0.9 | 0.3 | 0.2 | 1e-07 |
| 1024 | 128 | 10% | 6 | 3 | 0.14 | 1.10 | **0.32** | 0.31 | 0.44x | 0.9 | 0.4 | 0.3 | 1e-07 |
| 1024 | 128 | 25% | 16 | 4 | 0.14 | 1.10 | **0.37** | 0.27 | 0.38x | 0.9 | 1.3 | 0.8 | 1e-07 |
| 1024 | 128 | 50% | 32 | 6 | 0.14 | 1.10 | **0.29** | 0.25 | 0.50x | 0.9 | 2.5 | 1.5 | 1e-07 |
| 1024 | 512 | 1% | 1 | 1 | 0.16 | 1.10 | **0.27** | 0.28 | 0.59x | 3.0 | 0.8 | 0.4 | 1e-07 |
| 1024 | 512 | 5% | 3 | 2 | 0.16 | 1.08 | **0.32** | 0.27 | 0.51x | 3.0 | 1.3 | 0.7 | 1e-07 |
| 1024 | 512 | 10% | 6 | 3 | 0.16 | 1.65 | **0.31** | 0.25 | 0.53x | 3.0 | 2.2 | 1.9 | 1e-07 |
| 1024 | 512 | 25% | 16 | 4 | 0.16 | 1.16 | **0.28** | 0.25 | 0.58x | 3.0 | 5.3 | 3.4 | 1e-07 |
| 1024 | 512 | 50% | 32 | 6 | 0.16 | 1.18 | **0.31** | 0.29 | 0.52x | 3.0 | 9.8 | 7.4 | 1e-07 |
| 4096 | 128 | 1% | 10 | 2 | 0.31 | 1.39 | **0.29** | 0.28 | 1.05x | 13.8 | 1.2 | 0.6 | 1e-07 |
| 4096 | 128 | 5% | 51 | 4 | 0.31 | 1.53 | **0.36** | 0.40 | 0.86x | 13.8 | 5.3 | 2.5 | 1e-07 |
| 4096 | 128 | 10% | 102 | 7 | 0.31 | 1.65 | **0.37** | 0.35 | 0.82x | 13.8 | 8.9 | 4.6 | 1e-07 |
| 4096 | 128 | 25% | 256 | 14 | 0.31 | 2.18 | **0.60** | 0.59 | 0.51x | 13.8 | 19.5 | 11.9 | 1e-07 |
| 4096 | 128 | 50% | 512 | 20 | 0.31 | 3.71 | **0.81** | 0.64 | 0.38x | 13.8 | 37.6 | 32.7 | 2e-07 |
| 4096 | 512 | 1% | 10 | 2 | 0.64 | 1.70 | **0.33** | 0.29 | 1.95x | 45.8 | 4.6 | 2.5 | 1e-07 |
| 4096 | 512 | 5% | 51 | 4 | 0.64 | 1.84 | **0.36** | 0.34 | 1.75x | 45.8 | 20.7 | 10.8 | 1e-07 |
| 4096 | 512 | 10% | 102 | 7 | 0.64 | 2.00 | **0.43** | 0.42 | 1.47x | 45.8 | 35.7 | 22.5 | 1e-07 |
| 4096 | 512 | 25% | 256 | 14 | 0.64 | 2.69 | **0.94** | 0.70 | 0.68x | 45.8 | 76.2 | 100.8 | 1e-07 |
| 4096 | 512 | 50% | 512 | 20 | 0.64 | 3.73 | **1.00** | 1.08 | 0.64x | 45.8 | 149.2 | 361.4 | 2e-07 |
| 8192 | 128 | 1% | 41 | 4 | 0.85 | 2.30 | **0.31** | 0.31 | 2.73x | 57.8 | 4.0 | 2.5 | 1e-07 |
| 8192 | 128 | 5% | 205 | 9 | 0.85 | 2.58 | **0.43** | 0.43 | 2.01x | 57.8 | 17.2 | 10.0 | 1e-07 |
| 8192 | 128 | 10% | 410 | 16 | 0.85 | 3.90 | **0.65** | 0.66 | 1.31x | 57.8 | 32.6 | 20.4 | 1e-07 |
| 8192 | 128 | 25% | 1024 | 25 | 0.85 | 6.08 | **1.46** | 1.05 | 0.59x | 57.8 | 79.6 | 67.3 | 2e-07 |
| 8192 | 128 | 50% | 2048 | 41 | 0.85 | 7.22 | **2.27** | 2.34 | 0.38x | 57.8 | 155.1 | 232.3 | 2e-07 |
| 8192 | 512 | 1% | 41 | 4 | 2.69 | 3.07 | **0.40** | 0.36 | 6.77x | 181.7 | 16.4 | 10.0 | 1e-07 |
| 8192 | 512 | 5% | 205 | 9 | 2.69 | 5.09 | **0.93** | 1.07 | 2.89x | 181.7 | 66.8 | 49.3 | 1e-07 |
| 8192 | 512 | 10% | 410 | 16 | 2.69 | 5.41 | **1.41** | 1.37 | 1.91x | 181.7 | 129.9 | 157.3 | 1e-07 |
| 8192 | 512 | 25% | 1024 | 25 | 2.69 | 8.97 | **1.96** | 2.30 | 1.37x | 181.7 | 305.6 | 728.8 | 2e-07 |
| 8192 | 512 | 50% | 2048 | 41 | 2.69 | 12.46 | **3.96** | 3.67 | 0.68x | 181.7 | 598.5 | 1538.1 | 2e-07 |

### CSR SpMM (host structure)

| n | width | elem dens. | dense aten (dev) | main e2e | **PR e2e** | host densify+H2D | device densify | CPU dense | torch.sparse CSR (CPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 128 | 0.1% | 0.13 | 1.59 | **1.16** | 4.17 | 0.64 | 0.9 | 0.03 |
| 1024 | 128 | 1% | 0.17 | 5.18 | **5.28** | 0.97 | 0.94 | 0.9 | 0.06 |
| 1024 | 128 | 10% | 0.10 | 2.61 | **4.02** | 1.98 | 2.17 | 0.9 | 0.34 |
| 1024 | 512 | 0.1% | 0.16 | 1.39 | **5.39** | 1.89 | 0.62 | 3.0 | 0.19 |
| 1024 | 512 | 1% | 0.13 | 5.36 | **5.91** | 0.94 | 1.27 | 3.0 | 0.31 |
| 1024 | 512 | 10% | 0.22 | 5.05 | **5.99** | 4.00 | 2.06 | 2.9 | 1.64 |
| 4096 | 128 | 0.1% | 0.30 | 41.15 | **7.27** | 39.33 | 6.19 | 13.6 | 0.26 |
| 4096 | 128 | 1% | 0.64 | 44.78 | **13.47** | 42.85 | 12.93 | 13.3 | 0.89 |
| 4096 | 128 | 10% | 0.34 | 39.18 | **43.05** | 38.14 | 40.31 | 13.9 | 7.36 |
| 4096 | 512 | 0.1% | 0.63 | 16.95 | **7.37** | 15.12 | 6.10 | 46.6 | 1.12 |
| 4096 | 512 | 1% | 0.67 | 19.96 | **13.96** | 17.24 | 6.17 | 46.3 | 5.32 |
| 4096 | 512 | 10% | 0.65 | 39.27 | **45.16** | 36.97 | 41.09 | 46.4 | 61.42 |
| 8192 | 128 | 0.1% | 0.85 | 156.79 | **13.95** | 156.32 | 11.01 | 55.9 | 0.86 |
| 8192 | 128 | 1% | 0.98 | 174.46 | **27.86** | 171.84 | 25.92 | 56.7 | 4.86 |
| 8192 | 128 | 10% | 0.90 | 158.89 | **168.36** | 155.27 | 159.16 | 56.2 | 55.52 |
| 8192 | 512 | 0.1% | 2.47 | 75.19 | **18.62** | 65.08 | 4.05 | 225.0 | 4.15 |
| 8192 | 512 | 1% | 2.76 | 89.87 | **30.98** | 80.89 | 24.81 | 182.4 | 27.42 |
| 8192 | 512 | 10% | 2.69 | 163.90 | **174.65** | 157.15 | 170.41 | 183.6 | 278.46 |

### Screened SpMM

| n | width | kept | main (dev) | **PR (dev)** | dense masked aten (dev) | dense aten (dev) | CPU masked | PR rel err |
|---|---|---|---|---|---|---|---|---|
| 1024 | 128 | 100% | 0.76 | **0.72** | 0.27 | 0.13 | 1.1 | 6e-07 |
| 1024 | 128 | 64% | 0.80 | **0.76** | 0.26 | 0.14 | 1.0 | 6e-07 |
| 1024 | 128 | 17% | 0.76 | **0.70** | 0.26 | 0.13 | 1.0 | 6e-07 |
| 1024 | 512 | 100% | 0.82 | **0.73** | 0.28 | 0.16 | 3.1 | 6e-07 |
| 1024 | 512 | 64% | 0.83 | **0.76** | 0.29 | 0.16 | 3.1 | 6e-07 |
| 1024 | 512 | 17% | 0.82 | **0.75** | 0.28 | 0.16 | 3.1 | 6e-07 |
| 4096 | 128 | 100% | 3.30 | **1.99** | 0.72 | 0.30 | 33.6 | 6e-07 |
| 4096 | 128 | 64% | 3.39 | **2.36** | 0.72 | 0.30 | 34.0 | 7e-07 |
| 4096 | 128 | 17% | 3.37 | **1.95** | 0.91 | 0.30 | 34.8 | 6e-07 |
| 4096 | 512 | 100% | 4.01 | **2.52** | 1.12 | 0.72 | 70.2 | 6e-07 |
| 4096 | 512 | 64% | 3.97 | **2.49** | 1.07 | 0.71 | 67.0 | 7e-07 |
| 4096 | 512 | 17% | 3.95 | **2.54** | 1.18 | 0.99 | 67.3 | 6e-07 |
| 8192 | 128 | 100% | 10.96 | **5.63** | 2.90 | 0.95 | 158.1 | 6e-07 |
| 8192 | 128 | 64% | 10.95 | **4.97** | 3.13 | 0.82 | 155.1 | 9e-07 |
| 8192 | 128 | 17% | 11.29 | **5.33** | 2.95 | 1.30 | 136.0 | 6e-07 |
| 8192 | 512 | 100% | 14.37 | **9.40** | 4.30 | 2.36 | 289.9 | 6e-07 |
| 8192 | 512 | 64% | 14.56 | **9.56** | 4.76 | 2.74 | 257.7 | 9e-07 |
| 8192 | 512 | 17% | 14.55 | **9.35** | 4.67 | 2.74 | 261.1 | 6e-07 |

### Block-sparse attention (fp32, ms)

| seq | hd | mask | block dens. | K_max | SDPA dense fwd (dev) | SDPA masked fwd (dev) | main fwd | **PR fwd** | SDPA dense fwd+bwd (dev) | main fwd+bwd | **PR fwd+bwd** | SDPA CPU fwd | PR fwd err | PR bwd err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 64 | dilated | 25.0% | 2 | 0.33 | 0.50 | 2.78 | **0.60** | 0.89 | 6.44 | **2.39** | 1.0 | 8e-07 | 8e-07 |
| 1024 | 64 | global | 53.1% | 8 | 0.33 | 0.58 | 3.17 | **0.58** | 0.89 | 8.08 | **2.15** | 1.0 | 9e-07 | 9e-07 |
| 1024 | 64 | local | 34.4% | 3 | 0.33 | 0.53 | 3.28 | **0.59** | 0.89 | 7.90 | **2.10** | 1.0 | 8e-07 | 9e-07 |
| 1024 | 128 | dilated | 25.0% | 2 | 0.29 | 0.51 | 2.91 | **0.57** | 0.73 | 7.06 | **2.10** | 2.2 | 8e-07 | 8e-07 |
| 1024 | 128 | global | 53.1% | 8 | 0.29 | 0.56 | 3.12 | **0.68** | 0.73 | 8.16 | **2.12** | 2.2 | 9e-07 | 9e-07 |
| 1024 | 128 | local | 34.4% | 3 | 0.29 | 0.50 | 2.98 | **0.57** | 0.73 | 6.85 | **2.13** | 2.2 | 8e-07 | 8e-07 |
| 4096 | 64 | dilated | 25.0% | 8 | 0.43 | 1.58 | 4.61 | **1.26** | 2.01 | 11.94 | **3.51** | 15.6 | 9e-07 | 9e-07 |
| 4096 | 64 | global | 15.0% | 32 | 0.43 | 1.49 | 9.99 | **1.09** | 2.01 | 23.54 | **2.98** | 15.6 | 9e-07 | 9e-07 |
| 4096 | 64 | local | 9.2% | 3 | 0.43 | 1.54 | 3.74 | **0.87** | 2.01 | 8.68 | **2.70** | 15.6 | 9e-07 | 9e-07 |
| 4096 | 128 | dilated | 25.0% | 8 | 0.42 | 1.44 | 4.76 | **1.49** | 2.20 | 12.05 | **3.48** | 25.5 | 9e-07 | 9e-07 |
| 4096 | 128 | global | 15.0% | 32 | 0.42 | 1.44 | 11.42 | **1.25** | 2.20 | 26.93 | **2.82** | 25.5 | 9e-07 | 9e-07 |
| 4096 | 128 | local | 9.2% | 3 | 0.42 | 1.40 | 3.56 | **1.03** | 2.20 | 8.95 | **2.77** | 25.5 | 9e-07 | 9e-07 |
| 8192 | 64 | dilated | 25.0% | 16 | 0.77 | 5.17 | 10.36 | **3.52** | 6.25 | 26.77 | **11.16** | 61.7 | 1e-06 | 1e-06 |
| 8192 | 64 | global | 7.7% | 64 | 0.77 | 5.11 | 34.71 | **1.60** | 6.25 | 127.13 | **4.03** | 61.7 | 9e-07 | 9e-07 |
| 8192 | 64 | local | 4.6% | 3 | 0.77 | 5.15 | 4.42 | **1.37** | 6.25 | 14.27 | **3.49** | 61.7 | 9e-07 | 9e-07 |
| 8192 | 128 | dilated | 25.0% | 16 | 0.77 | 5.15 | 11.64 | **3.24** | 6.64 | 31.53 | **11.01** | 104.2 | 9e-07 | 1e-06 |
| 8192 | 128 | global | 7.7% | 64 | 0.77 | 5.00 | 39.67 | **1.92** | 6.64 | 195.24 | **4.15** | 104.2 | 9e-07 | 9e-07 |
| 8192 | 128 | local | 4.6% | 3 | 0.77 | 5.01 | 5.20 | **1.66** | 6.64 | 12.95 | **3.65** | 104.2 | 9e-07 | 9e-07 |

### Reading the Trn2 tables

- **BSR SpMM pays off at scale and moderate sparsity.** At n = 8192 the
  static kernel beats dense aten up to ~14% block density (width 128) and
  ~34% (width 512); at 1% it is 2.7–6.8× faster. Its cost is linear in
  stored blocks (~1.9 µs per 128×128×512 block vs ~0.66 µs per block for
  dense aten), so the crossover moves up with n and with width. At
  n = 1024 nothing sparse wins: one launch costs ~0.3 ms and dense aten
  is 0.14 ms. Against CPU (dense or `torch.sparse`) the device wins
  everywhere above n = 1024.
- **The gather path was the bottleneck, not the Tensor Engine.** On main,
  device-side pad + gather took 60% of every call and the `K_max` padding
  made each row cost as much as the densest row. With the structure as
  kernel constants both are gone: 3–6× faster end to end at 8192.
- **Attention forward never beats unmasked dense SDPA** (a fused kernel at
  0.77 ms for seq 8192), but it is what to compare against when the mask
  is real: masked SDPA costs 5.0–5.2 ms at 8192 and the static kernel
  1.4–3.5 ms. Forward + backward beats unmasked dense SDPA at seq 8192 for
  local and global masks (3.5–4.2 ms vs 6.3–6.6 ms), not for 25% dilated
  (11 ms) and not at seq ≤ 4096.
- **Global-token masks** were the pathological case for the gather path
  (`K_max` = all blocks): seq 8192 fwd+bwd went from 127–195 ms to ~4 ms.
- **Screened SpMM** is 1.5–2× faster than on main but still 2–3× slower
  than dense aten `(A * mask) @ B` on the device: the fused kernel does not
  pay on Trn2 in fp32. It has no tile skipping, so it never benefits from
  screening sparsity.
- **CSR SpMM** (host structure) is now 4–11× faster at ≤ 1% density
  (device scatter of nnz triplets instead of a full dense H2D copy), but
  stays far behind `torch.sparse` on CPU at low density and behind dense
  aten when A is already dense on the device. On Trainium, use BSR.


## Trn2: across the logical NeuronCores of one chip (#30)

**Setup.**
- Hardware: `trn2.3xlarge` at LNC=2, all four logical NeuronCores visible
  to one process (`NEURON_RT_VISIBLE_CORES=0-3`), with the whole chip
  reserved for these runs.
- Software: Neuron SDK 2.33 venv with `torch` 2.14.0, `torch-neuronx`
  2.14.3, `nki` 0.7.0 and `neuronx-cc` 2.28. fp32 throughout.
- Host: `taskset -c 8,9`, `OMP_NUM_THREADS=2`.
- Script: `benchmarks/bench_multicore_trn2.py`.
- Operands are built on the host and copied once to every core that
  uses them. The time covers issuing one launch per core from one Python
  thread and synchronizing every core used; host gathers are excluded.
- **Dense baselines are split too:** "dense kc" is aten `A @ B` with A's
  rows split over k cores and B on each. SDPA "4c" splits the query rows
  (one head) or the heads (four heads) the same way.
- "API 1c" is the public one-core `bsr_spmm`, the path from #36, now with
  keyed ops. "plain" is `ShardedBSR(..., pretransposed=False)`, the #36
  kernel. "pre-T" is the new pre-transposed, row-grouped kernel.
- Rounds: BSR and single-head attention show the lower of two rounds'
  warm medians (15 calls each). The four-head runs are one round.
  Sub-millisecond cells carry about ±0.1–0.2 ms of noise. The crossovers
  interpolate linearly between the measured densities.

**Crossover density vs dense aten** (BSR 128×128):

| n | width | API 1c vs dense 1c (#36 path) | pre-T 1c vs dense 1c | best of pre-T 1/2/4c vs dense 1c | pre-T 4c vs dense 4c | best sparse (any cores) vs best dense (any cores) |
|---|---|---|---|---|---|---|
| 8192 | 128 | ~33% | ~82% | ~96% | ~20% | ~62% |
| 8192 | 512 | ~65% | >100% | >100% | ~69% | ~69% |
| 16384 | 128 | ~69% | ~97% | >100% | ~74% | ~77% |
| 16384 | 512 | ~69% | ~98% | >100% | ~50% | ~50% |

### BSR SpMM (fp32, ms)

| n | width | dens. | blocks | dense 1c | dense 2c | dense 4c | API 1c (#36) | plain 1c | plain 4c | **pre-T 1c** | **pre-T 2c** | **pre-T 4c** | pre-T 4c issue | pre-T 1c->4c | dense 1c / best sparse | dense 4c / pre-T 4c | CPU dense | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 128 | 1% | 41 | 0.85 | 0.60 | 0.63 | 0.28 | 0.24 | 0.50 | **0.24** | **0.36** | **0.53** | 0.43 | 0.45x | 3.57x | 1.21x | 56.36 | 1e-07 |
| 8192 | 128 | 5% | 205 | 0.85 | 0.60 | 0.63 | 0.36 | 0.32 | 0.55 | **0.27** | **0.37** | **0.54** | 0.45 | 0.49x | 3.19x | 1.17x | 56.36 | 1e-07 |
| 8192 | 128 | 10% | 410 | 0.85 | 0.60 | 0.63 | 0.44 | 0.42 | 0.63 | **0.29** | **0.36** | **0.57** | 0.35 | 0.51x | 2.91x | 1.11x | 56.36 | 1e-07 |
| 8192 | 128 | 25% | 1024 | 0.85 | 0.60 | 0.63 | 0.70 | 0.81 | 0.90 | **0.37** | **0.39** | **0.68** | 0.39 | 0.55x | 2.28x | 0.94x | 56.36 | 2e-07 |
| 8192 | 128 | 50% | 2048 | 0.85 | 0.60 | 0.63 | 1.58 | 1.30 | 0.94 | **0.58** | **0.51** | **0.76** | 0.38 | 0.76x | 1.66x | 0.83x | 56.36 | 2e-07 |
| 8192 | 128 | 75% | 3072 | 0.85 | 0.58 | 0.48 | 1.84 | - | - | **0.74** | **0.59** | **0.81** | 0.38 | 0.91x | 1.44x | 0.59x | - | 2e-07 |
| 8192 | 128 | 100% | 4096 | 0.85 | 0.58 | 0.48 | 2.35 | - | - | **1.44** | **0.94** | **1.04** | 0.59 | 1.39x | 0.91x | 0.46x | - | 2e-07 |
| 8192 | 512 | 1% | 41 | 2.69 | 1.53 | 1.46 | 0.32 | 0.29 | 0.51 | **0.25** | **0.31** | **0.50** | 0.37 | 0.50x | 10.75x | 2.93x | 185.06 | 1e-07 |
| 8192 | 512 | 5% | 205 | 2.69 | 1.53 | 1.46 | 0.50 | 0.72 | 0.58 | **0.31** | **0.35** | **0.53** | 0.39 | 0.58x | 8.79x | 2.75x | 185.06 | 1e-07 |
| 8192 | 512 | 10% | 410 | 2.69 | 1.53 | 1.46 | 0.60 | 0.99 | 0.93 | **0.39** | **0.56** | **0.56** | 0.37 | 0.69x | 6.95x | 2.60x | 185.06 | 1e-07 |
| 8192 | 512 | 25% | 1024 | 2.69 | 1.53 | 1.46 | 1.41 | 1.41 | 0.96 | **1.04** | **0.81** | **0.89** | 0.42 | 1.17x | 3.32x | 1.64x | 185.06 | 2e-07 |
| 8192 | 512 | 50% | 2048 | 2.69 | 1.53 | 1.46 | 2.28 | 1.97 | 1.30 | **1.50** | **1.14** | **1.08** | 0.38 | 1.39x | 2.50x | 1.36x | 185.06 | 2e-07 |
| 8192 | 512 | 75% | 3072 | 2.65 | 1.48 | 1.14 | 2.99 | - | - | **2.07** | **1.48** | **1.30** | 0.59 | 1.59x | 2.04x | 0.87x | - | 2e-07 |
| 8192 | 512 | 100% | 4096 | 2.65 | 1.48 | 1.14 | 3.73 | - | - | **2.59** | **1.78** | **1.71** | 0.48 | 1.51x | 1.55x | 0.66x | - | 2e-07 |
| 16384 | 128 | 1% | 164 | 5.79 | 2.37 | 1.92 | 0.38 | 0.57 | 0.64 | **0.31** | **0.41** | **0.65** | 0.42 | 0.47x | 18.87x | 2.95x | 228.78 | 1e-07 |
| 16384 | 128 | 5% | 819 | 5.79 | 2.37 | 1.92 | 0.66 | 0.71 | 0.80 | **0.41** | **0.43** | **0.61** | 0.41 | 0.67x | 14.09x | 3.12x | 228.78 | 1e-07 |
| 16384 | 128 | 10% | 1638 | 5.79 | 2.37 | 1.92 | 1.03 | 1.01 | 1.02 | **0.54** | **0.47** | **0.71** | 0.54 | 0.76x | 12.28x | 2.70x | 228.78 | 2e-07 |
| 16384 | 128 | 25% | 4096 | 5.79 | 2.37 | 1.92 | 2.67 | 2.21 | 1.18 | **1.19** | **1.00** | **0.85** | 0.41 | 1.40x | 6.79x | 2.25x | 228.78 | 2e-07 |
| 16384 | 128 | 50% | 8192 | 5.79 | 2.37 | 1.92 | 4.46 | 3.58 | 2.24 | **3.39** | **1.48** | **1.47** | 0.62 | 2.30x | 3.93x | 1.30x | 228.78 | 2e-07 |
| 16384 | 128 | 75% | 12288 | 5.76 | 2.29 | 1.83 | 6.38 | - | - | **3.29** | **1.79** | **1.85** | 0.58 | 1.78x | 3.22x | 0.99x | - | 3e-07 |
| 16384 | 128 | 100% | 16384 | 5.76 | 2.29 | 1.83 | 8.27 | - | - | **6.47** | **3.65** | **2.90** | 0.67 | 2.24x | 1.99x | 0.63x | - | 3e-07 |
| 16384 | 512 | 1% | 164 | 10.64 | 5.54 | 2.90 | 0.48 | 0.68 | 0.60 | **0.33** | **0.39** | **0.55** | 0.38 | 0.61x | 31.88x | 5.28x | 746.80 | 1e-07 |
| 16384 | 512 | 5% | 819 | 10.64 | 5.54 | 2.90 | 1.03 | 1.06 | 0.88 | **0.95** | **0.71** | **0.71** | 0.57 | 1.33x | 15.00x | 4.07x | 746.80 | 1e-07 |
| 16384 | 512 | 10% | 1638 | 10.64 | 5.54 | 2.90 | 2.12 | 1.96 | 1.20 | **1.08** | **0.96** | **0.95** | 0.42 | 1.14x | 11.19x | 3.05x | 746.80 | 2e-07 |
| 16384 | 512 | 25% | 4096 | 10.64 | 5.54 | 2.90 | 4.00 | 3.57 | 2.27 | **2.73** | **1.66** | **1.70** | 0.60 | 1.61x | 6.42x | 1.71x | 746.80 | 2e-07 |
| 16384 | 512 | 50% | 8192 | 10.64 | 5.54 | 2.90 | 7.23 | 6.46 | 3.73 | **5.63** | **2.91** | **2.92** | 0.60 | 1.93x | 3.66x | 0.99x | 746.80 | 2e-07 |
| 16384 | 512 | 75% | 12288 | 10.98 | 5.68 | 3.01 | 12.95 | - | - | **7.44** | **4.20** | **4.17** | 0.46 | 1.78x | 2.63x | 0.72x | - | 3e-07 |
| 16384 | 512 | 100% | 16384 | 10.98 | 5.68 | 3.01 | 16.80 | - | - | **11.45** | **6.28** | **5.34** | 0.68 | 2.14x | 2.06x | 0.56x | - | 3e-07 |

### Block-sparse attention forward, one head, query rows split (fp32, ms)

| seq | hd | mask | block dens. | SDPA dense 1c | SDPA dense 4c (rows split) | SDPA masked 1c | SDPA masked 4c | API 1c | **sparse 1c** | **sparse 2c** | **sparse 4c** | 1c->4c | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 64 | dilated | 25.0% | 0.72 | 1.05 | 5.13 | 4.22 | 1.97 | **1.55** | **1.14** | **1.29** | 1.20x | 1e-06 |
| 8192 | 64 | global | 7.7% | 0.72 | 1.05 | 5.14 | 4.20 | 1.09 | **0.76** | **0.84** | **0.84** | 0.91x | 9e-07 |
| 8192 | 64 | local | 4.6% | 0.72 | 1.05 | 5.04 | 3.85 | 0.85 | **0.55** | **0.57** | **0.90** | 0.61x | 9e-07 |
| 8192 | 128 | dilated | 25.0% | 0.77 | 1.06 | 5.03 | 4.22 | 1.88 | **1.42** | **1.30** | **1.25** | 1.14x | 9e-07 |
| 8192 | 128 | global | 7.7% | 0.77 | 1.06 | 5.13 | 4.17 | 1.21 | **0.89** | **0.68** | **0.93** | 0.95x | 9e-07 |
| 8192 | 128 | local | 4.6% | 0.77 | 1.06 | 5.04 | 4.09 | 1.03 | **0.72** | **0.70** | **0.85** | 0.85x | 9e-07 |
| 16384 | 64 | dilated | 25.0% | 2.04 | 1.82 | 32.17 | 9.23 | 6.17 | **5.68** | **3.10** | **2.99** | 1.90x | 1e-06 |
| 16384 | 64 | global | 3.9% | 2.04 | 1.82 | 32.46 | 9.17 | 1.86 | **1.34** | **0.93** | **1.14** | 1.18x | 9e-07 |
| 16384 | 64 | local | 2.3% | 2.04 | 1.82 | 31.89 | 6.58 | 1.18 | **0.85** | **0.91** | **1.00** | 0.85x | 8e-07 |
| 16384 | 128 | dilated | 25.0% | 2.22 | 1.96 | 32.94 | 10.67 | 5.71 | **5.17** | **2.91** | **2.88** | 1.80x | 1e-06 |
| 16384 | 128 | global | 3.9% | 2.22 | 1.96 | 32.61 | 8.39 | 2.16 | **1.64** | **1.01** | **1.16** | 1.42x | 9e-07 |
| 16384 | 128 | local | 2.3% | 2.22 | 1.96 | 32.58 | 9.37 | 1.80 | **1.58** | **1.03** | **1.10** | 1.44x | 9e-07 |

### Block-sparse attention, four heads, whole heads per core (fp32, ms)

| seq | hd | mask | block dens. | SDPA fwd 1c | SDPA fwd 4c | **sparse fwd 1c** | **sparse fwd 4c** | SDPA fwd+bwd 1c | SDPA fwd+bwd 4c | **sparse fwd+bwd 1c** | **sparse fwd+bwd 2c** | **sparse fwd+bwd 4c** | fwd+bwd 1c->4c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 64 | dilated | 25.0% | 2.09 | 1.64 | **6.93** | **4.25** | 12.99 | 12.46 | **18.70** | **14.94** | **13.22** | 1.42x |
| 8192 | 64 | global | 7.7% | 2.09 | 1.64 | **3.77** | **2.92** | 12.99 | 12.46 | **10.45** | **10.23** | **9.58** | 1.09x |
| 8192 | 64 | local | 4.6% | 2.09 | 1.64 | **2.93** | **2.48** | 12.99 | 12.46 | **9.70** | **9.28** | **7.56** | 1.28x |
| 8192 | 128 | dilated | 25.0% | 2.17 | 1.65 | **6.32** | **3.72** | 13.60 | 10.51 | **18.44** | **14.63** | **13.24** | 1.39x |
| 8192 | 128 | global | 7.7% | 2.17 | 1.65 | **4.35** | **3.17** | 13.60 | 10.51 | **12.94** | **10.39** | **8.83** | 1.47x |
| 8192 | 128 | local | 4.6% | 2.17 | 1.65 | **3.70** | **2.67** | 13.60 | 10.51 | **10.94** | **9.18** | **8.65** | 1.26x |
| 16384 | 64 | dilated | 25.0% | 7.47 | 4.18 | **22.28** | **10.06** | 48.65 | 27.60 | **77.94** | **49.78** | **37.63** | 2.07x |
| 16384 | 64 | global | 3.9% | 7.47 | 4.18 | **6.02** | **3.85** | 48.65 | 27.60 | **17.87** | **13.01** | **14.06** | 1.27x |
| 16384 | 64 | local | 2.3% | 7.47 | 4.18 | **4.23** | **3.37** | 48.65 | 27.60 | **14.30** | **11.82** | **10.45** | 1.37x |
| 16384 | 128 | dilated | 25.0% | 7.88 | 4.50 | **19.86** | **9.14** | 50.32 | 25.92 | **76.25** | **52.44** | **40.18** | 1.90x |
| 16384 | 128 | global | 3.9% | 7.88 | 4.50 | **6.69** | **4.10** | 50.32 | 25.92 | **19.22** | **13.80** | **13.23** | 1.45x |
| 16384 | 128 | local | 2.3% | 7.88 | 4.50 | **6.03** | **3.61** | 50.32 | 25.92 | **15.97** | **11.87** | **11.92** | 1.34x |

### One process per core (BSR, n = 8192, width 512, pre-T, ms)

`benchmarks/bench_multiproc_trn2.py`: each process owns one core and its
share of the block rows, and all processes are pinned to the same two host
CPUs.

| processes | barrier | dense (rows split) | 1% | 5% | 10% | 25% | 50% |
|---|---|---|---|---|---|---|---|
| 1 | 0.02 | 2.31 | 0.28 | 0.32 | 0.41 | 0.70 | 1.20 |
| 2 | 0.05 | 1.54 | 0.44 | 0.39 | 0.36 | 0.69 | 0.89 |
| 4 | 0.11 | 1.66 | 0.84 | 1.02 | 0.78 | 1.10 | 1.12 |

### Reading the multi-core tables

- **The per-block fix moved the crossover more than extra cores did.**
  - On one core, the pre-transposed, row-grouped kernel is 1.1–2.2×
    faster than the #36 kernel at 25–50% density.
  - It beats dense aten on that same core up to ~82–98% block density,
    and still beats it at 100% for n = 8192, width 512. In #36 the
    crossover was ~26–49% on two physical cores.
  - Where the time goes: B-tile loads, shared across up to 8 block rows
    of a group, and the in-kernel transpose, now gone.
- **Four logical cores pay only for large work per call.**
  - Speedup from one core to four (pre-T): 1.4–2.3× at n = 16384 and
    25–100% density, 1.2–1.6× at n = 8192 and width 512 above 25%.
  - It is a *slowdown* (0.45–0.8×) below ~0.6 ms of single-core work.
- **Why the speedup stops there:**
  - One Python thread issues the launches, at about 0.1 ms each (0.4–0.7 ms
    for four).
  - Concurrent kernels on different logical cores also slow each other.
    At n = 16384, width 512, 50% density, one shard takes 1.15–1.3 ms alone
    and 1.9–2.9 ms with all four running. Dense aten slabs barely contend
    (2.5 ms alone, 3.1 ms for four).
  - Two cores reach most of the four-core gain.
  - One process per core does not help with two host CPUs: four processes
    are no faster than one.
- **Against dense split the same way, the crossover drops.** Dense aten
  scales better across cores (3.7× from one to four at n = 16384, width
  512), so the four-core crossovers are ~50–74% at n = 16384 and ~69% at
  n = 8192, width 512. At n = 8192, width 128 both sides sit on the launch
  floor (0.5–0.7 ms), so splitting is pointless there.
- **Attention, one head:**
  - The forward on one core now beats unmasked dense SDPA at seq 16384
    for local and global masks: 0.85–1.6 ms vs 2.0–2.2 ms.
  - At seq 8192 it ties: 0.55–0.89 ms vs 0.72–0.77 ms.
  - Splitting query rows over cores pays only for the 25% dilated mask at
    seq 16384: 5.2–5.7 → 2.9–3.0 ms, but that is still slower than dense
    SDPA at 1.8–2.2 ms.
  - Masked SDPA (32–33 ms at 16384 on one core, 6.6–11 ms on four) loses
    to every sparse configuration.
- **Attention, four heads (fwd+bwd):**
  - With whole heads per core, sparse beats dense SDPA split the same way
    at seq 16384 for local and global masks: 10.5–14 ms vs 26–28 ms.
  - At seq 8192 the margin is narrower: 7.6–9.6 vs 10.5–12.5 ms.
  - The 25% dilated mask loses at both lengths.
  - Core scaling is 1.1–2.1×, held back by about 10–20 host-issued aten
    ops per head on the public path.

## Trn2: fused SBUF-resident CG (`cg_bsr_fused`, #22)

trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron, one logical NeuronCore
(LNC=2: two physical cores) unless noted. Host threads pinned to 2 CPUs
(`taskset`, `OMP_NUM_THREADS=2`) for every column, CPU baselines included.
Wall time from the host call to the solution on the host, warm median of 3,
tolerance 1e-5 on ||r|| / ||b||. Reproduce with
`benchmarks/bench_cg_fused_trn2.py --tiles T --density d`.

Matrices: the synthetic block-screened overlap matrix of normalised
s-type Gaussians on a synthetic molecular cluster, one 128-function
fragment per block row, with blocks between distant fragments dropped
to hit the block density and a level shift setting κ exactly. A is built
in fp64. The device and the fp32 columns get its fp32 rounding. "true res"
is ||b − A x|| / ||b|| with the fp64 A, and "err" is against an fp64
direct solve.

Columns: `fused` = `cg_bsr_fused` (one launch). `host_bsr` = CG over the
keyed pre-transposed BSR SpMM op with device tensors, synced every 8 steps.
`cg_bsr` = the shipped Python loop (one `.item()` per dot). `dense_aten` =
CG with `torch.mv` on a dense fp32 A on the device, synced every 8 steps.
`scipy` = `scipy.sparse.linalg.cg` on fp64 BSR. `torch fp64/fp32` = the
`cg_bsr_fused` host path (block `bmm`). "resident" = blocks held in SBUF
(summed over the two physical cores); the rest stream from HBM inside the
launch. "µs/iter" = (fused − fused with max_iter=0) / iterations.

#### Plain CG (ms to tol 1e-5, warm median)

| n | blocks | density | κ | resident | fused | host_bsr | cg_bsr | dense_aten | scipy | torch fp64 | torch fp32 | iters | µs/iter | true res | err vs fp64 | vs best CPU | vs host_bsr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 0.250 | 1e+03 | 16 | 2.5 | 145.9 | 159.2 | 140.0 | 12.1 | 18.2 | 22.5 | 118 | 15 | 1.6e-05 | 3.6e-06 | 4.8x | 58x |
| 1024 | 32 | 0.500 | 1e+03 | 32 | 1.3 | 49.0 | 49.4 | 48.8 | 6.9 | 5.6 | 11.2 | 36 | 20 | 8.8e-06 | 3.9e-06 | 4.3x | 38x |
| 1024 | 64 | 1.000 | 1e+03 | 64 | 3.6 | 147.3 | 157.5 | 151.4 | 37.8 | 26.8 | 33.8 | 118 | 26 | 1.3e-05 | 3.5e-06 | 7.5x | 41x |
| 2048 | 32 | 0.125 | 1e+03 | 32 | 1.2 | 34.9 | 33.0 | 34.2 | 5.6 | 4.5 | 6.4 | 25 | 23 | 5.4e-06 | 1.6e-06 | 3.8x | 30x |
| 2048 | 64 | 0.250 | 1e+03 | 64 | 1.7 | 43.2 | 44.2 | 41.4 | 12.8 | 18.9 | 16.5 | 33 | 34 | 7.2e-06 | 2.9e-06 | 7.6x | 26x |
| 2048 | 128 | 0.500 | 1e+03 | 128 | 3.2 | 70.7 | 84.6 | 67.0 | 41.1 | 26.1 | 27.3 | 62 | 41 | 7.5e-06 | 5.2e-06 | 8.2x | 22x |
| 2048 | 256 | 1.000 | 1e+03 | 256 | 10.9 | 169.6 | 177.3 | 163.8 | 168.1 | 95.8 | 150.5 | 130 | 69 | 1.8e-05 | 3.3e-06 | 8.8x | 16x |
| 4096 | 64 | 0.062 | 1e+03 | 64 | 1.3 | 45.9 | 47.7 | 37.4 | 11.9 | 17.8 | 6.9 | 29 | 25 | 9.0e-06 | 5.2e-06 | 5.2x | 34x |
| 4096 | 128 | 0.125 | 1e+03 | 128 | 2.7 | 70.8 | 70.3 | 67.3 | 39.7 | 25.0 | 33.4 | 54 | 40 | 7.0e-06 | 5.4e-06 | 9.2x | 26x |
| 4096 | 256 | 0.250 | 1e+03 | 256 | 6.0 | 92.9 | 114.3 | 85.5 | 95.8 | 57.1 | 46.1 | 66 | 61 | 8.1e-06 | 5.1e-06 | 7.7x | 15x |
| 4096 | 512 | 0.500 | 1e+03 | 512 | 16.5 | 151.8 | 215.2 | 143.9 | 332.5 | 192.5 | 166.1 | 118 | 124 | 1.0e-05 | 3.9e-06 | 10.1x | 9x |
| 8192 | 122 | 0.030 | 1e+03 | 122 | 2.1 | 49.1 | 45.3 | 65.8 | 25.7 | 16.6 | 14.0 | 34 | 42 | 6.4e-06 | 3.5e-06 | 6.8x | 24x |
| 8192 | 256 | 0.062 | 1e+03 | 256 | 4.1 | 55.4 | 86.7 | 77.9 | 71.8 | 44.6 | 32.1 | 48 | 48 | 6.8e-06 | 8.4e-06 | 7.9x | 14x |
| 8192 | 512 | 0.125 | 1e+03 | 512 | 11.2 | 104.5 | 162.0 | 135.0 | 262.8 | 159.7 | 119.5 | 75 | 136 | 9.2e-06 | 1.1e-05 | 10.7x | 9x |
| 8192 | 1024 | 0.250 | 1e+03 | 640 | 30.6 | 167.3 | 239.5 | 193.6 | 1334.2 | 667.1 | 322.1 | 115 | 252 | 9.1e-06 | 1.1e-05 | 10.5x | 5x |
| 2048 | 64 | 0.250 | 1e+05 | 64 | 2.0 | 54.2 | 71.8 | 51.8 | 15.1 | 10.5 | 18.6 | 43 | 24 | 4.1e-05 | 1.7e-04 | 5.2x | 27x |
| 4096 | 128 | 0.125 | 1e+05 | 128 | 3.6 | 79.3 | 87.1 | 81.8 | 48.4 | 31.1 | 26.4 | 67 | 44 | 2.0e-05 | 1.3e-04 | 7.3x | 22x |
| 8192 | 256 | 0.062 | 1e+05 | 256 | 12.4 | 80.0 | 82.8 | 105.6 | 93.1 | 54.8 | 43.6 | 60 | 178 | 1.3e-05 | 3.1e-04 | 3.5x | 6x |

#### Block-Jacobi PCG (ms to tol 1e-5, warm median)

| n | blocks | density | κ | resident | fused | host_bsr | scipy | torch fp64 | torch fp32 | iters | µs/iter | true res | err vs fp64 | vs best CPU | vs host_bsr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 0.250 | 1e+03 | 16 | 2.5 | 134.5 | 16.0 | 12.9 | 23.2 | 98 | 19 | 1.1e-05 | 6.3e-06 | 5.3x | 55x |
| 1024 | 32 | 0.500 | 1e+03 | 32 | 1.3 | 41.6 | 7.2 | 12.0 | 10.5 | 26 | 26 | 9.5e-06 | 5.3e-06 | 5.6x | 33x |
| 1024 | 64 | 1.000 | 1e+03 | 64 | 4.3 | 158.2 | 46.6 | 34.3 | 35.9 | 114 | 21 | 9.5e-06 | 5.9e-06 | 8.0x | 37x |
| 2048 | 32 | 0.125 | 1e+03 | 32 | 1.1 | 29.2 | 6.7 | 11.8 | 7.4 | 21 | 28 | 7.2e-06 | 1.6e-06 | 6.0x | 26x |
| 2048 | 64 | 0.250 | 1e+03 | 64 | 1.9 | 38.9 | 13.2 | 10.0 | 16.8 | 27 | 37 | 3.6e-06 | 2.7e-06 | 5.3x | 21x |
| 2048 | 128 | 0.500 | 1e+03 | 128 | 3.1 | 71.0 | 41.9 | 26.3 | 28.5 | 51 | 50 | 9.5e-06 | 4.3e-06 | 8.4x | 23x |
| 2048 | 256 | 1.000 | 1e+03 | 256 | 13.4 | 238.7 | 235.1 | 142.7 | 199.4 | 158 | 76 | 1.8e-05 | 5.1e-06 | 10.6x | 18x |
| 4096 | 64 | 0.062 | 1e+03 | 64 | 1.4 | 37.1 | 14.5 | 11.1 | 17.8 | 24 | 34 | 4.7e-06 | 2.5e-06 | 7.9x | 26x |
| 4096 | 128 | 0.125 | 1e+03 | 128 | 2.9 | 66.7 | 40.6 | 28.0 | 33.3 | 44 | 52 | 7.2e-06 | 3.3e-06 | 9.7x | 23x |
| 4096 | 256 | 0.250 | 1e+03 | 256 | 5.6 | 82.1 | 83.0 | 51.2 | 40.1 | 50 | 97 | 6.0e-06 | 4.1e-06 | 7.2x | 15x |
| 4096 | 512 | 0.500 | 1e+03 | 512 | 13.5 | 129.1 | 284.8 | 164.0 | 124.9 | 82 | 143 | 9.1e-06 | 6.4e-06 | 9.2x | 10x |
| 8192 | 122 | 0.030 | 1e+03 | 122 | 2.3 | 47.0 | 31.7 | 22.6 | 18.3 | 30 | 55 | 6.0e-06 | 4.1e-06 | 8.0x | 20x |
| 8192 | 256 | 0.062 | 1e+03 | 256 | 5.6 | 62.5 | 77.4 | 49.3 | 36.6 | 42 | 101 | 8.0e-06 | 1.4e-05 | 6.5x | 11x |
| 8192 | 512 | 0.125 | 1e+03 | 512 | 10.7 | 99.5 | 259.9 | 186.3 | 116.0 | 61 | 145 | 7.8e-06 | 8.7e-06 | 10.8x | 9x |
| 8192 | 1024 | 0.250 | 1e+03 | 576 | 21.0 | 131.8 | 986.7 | 483.1 | 217.6 | 74 | 253 | 8.9e-06 | 1.3e-05 | 10.3x | 6x |
| 2048 | 64 | 0.250 | 1e+05 | 64 | 2.2 | 48.7 | 16.4 | 12.1 | 18.9 | 33 | 41 | 3.5e-05 | 1.0e-04 | 5.6x | 23x |
| 4096 | 128 | 0.125 | 1e+05 | 128 | 3.2 | 67.4 | 48.0 | 34.3 | 26.0 | 54 | 39 | 1.7e-05 | 3.6e-05 | 8.2x | 21x |
| 8192 | 256 | 0.062 | 1e+05 | 256 | 5.9 | 72.9 | 108.6 | 66.0 | 49.5 | 53 | 85 | 1.5e-05 | 3.2e-04 | 8.3x | 12x |

#### LNC=2 row split vs whole solve on one physical core (fused, plain CG, µs/iter)

| n | blocks | split (LNC=2) | no split (LNC=2) | LNC=1 | split speedup |
|---:|---:|---:|---:|---:|---:|
| 2048 | 256 | 69 | 114 | 119 | 1.66x |
| 4096 | 256 | 61 | 130 | 139 | 2.12x |
| 4096 | 512 | 124 | 232 | 230 | 1.87x |
| 8192 | 512 | 136 | 241 | 244 | 1.77x |
| 8192 | 1024 | 252 | 460 | 461 | 1.82x |

#### Check interval (n=4096, 256 blocks, fused, ms)

| check_every | plain | PCG |
|---:|---:|---:|
| 1 | 5.9 | 5.1 |
| 4 | 5.4 | 4.8 |
| 8 | 6.0 | 5.6 |
| 32 | 8.5 | 5.4 |

#### Several right-hand sides (ms per right-hand side, warm median)

| n (256 blocks) | precond | R | fused total ms | fused | host_bsr | dense_aten | scipy | torch fp64 | torch fp32 | fused vs best CPU |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | block_jacobi | 1 | 13.4 | 13.43 | 238.68 | - | 235.07 | 142.72 | 199.45 | 11x |
| 2048 | block_jacobi | 64 | 18.9 | 0.30 | 3.23 | - | 234.57 | 32.81 | 12.85 | 44x |
| 2048 | none | 1 | 10.9 | 10.90 | 169.61 | 163.79 | 168.07 | 95.80 | 150.50 | 9x |
| 2048 | none | 64 | 28.6 | 0.45 | 2.44 | 2.40 | 166.45 | 20.72 | 10.22 | 23x |
| 4096 | block_jacobi | 1 | 5.6 | 5.59 | 82.14 | - | 82.96 | 51.20 | 40.10 | 7x |
| 4096 | block_jacobi | 4 | 9.8 | 2.44 | 17.34 | - | 93.92 | 15.04 | 11.88 | 5x |
| 4096 | block_jacobi | 16 | 9.8 | 0.61 | 4.24 | - | 88.37 | 5.70 | 3.72 | 6x |
| 4096 | block_jacobi | 64 | 11.5 | 0.18 | 1.12 | - | 86.64 | 12.34 | 2.63 | 15x |
| 4096 | block_jacobi | 128 | 12.5 | 0.10 | - | - | - | 11.73 | 6.40 | 66x |
| 4096 | none | 1 | 6.0 | 6.00 | 92.92 | 85.49 | 95.81 | 57.12 | 46.10 | 8x |
| 4096 | none | 4 | 10.5 | 2.63 | 20.31 | 23.74 | 95.09 | 16.47 | 13.20 | 5x |
| 4096 | none | 16 | 25.9 | 1.62 | 5.27 | 5.81 | 98.70 | 7.00 | 4.25 | 3x |
| 4096 | none | 64 | 12.2 | 0.19 | 1.28 | 1.38 | 115.96 | 13.96 | 3.20 | 17x |
| 4096 | none | 128 | 12.7 | 0.10 | - | - | - | 14.57 | 7.54 | 76x |

### Reading the CG tables

- **The win is removing the host from the iteration loop.** The fused
  solve is 3.5–11x faster than the best 2-thread CPU solver at every size
  and density here. It is 5–58x faster than the same CG driven from the
  host over the already fast BSR SpMM op: that loop costs about 1–1.5 ms
  per iteration in launches and syncs, against 15–250 µs on the device.
- **A matvec costs about 0.5 µs per stored block per physical core**,
  whether the block is resident or streamed. That is the TensorE load of
  a 128x128 fp32 stationary block for one moving column (a
  microbenchmark gives 497 ns per matmul at 1 column, 524 ns at 64,
  985 ns at 512; bf16 is 100 ns). The DMA queues keep up streaming
  64 KiB blocks from HBM at that rate (about 128 GB/s), so at n = 8192
  with 1024 blocks, the 384 non-resident blocks did not cost extra time
  per block. **SBUF residency stops fitting at 320 fp32 blocks per
  physical core (640 per logical core with the split), but past that
  point nothing falls off a cliff.** Residency matters only when the
  per-block compute gets cheaper than the DMA (bf16, or many more moving
  columns).
- **The LNC=2 row split is 1.7–2.1x per iteration** over running the
  whole solve on one physical core. LNC=1 matches the unsplit LNC=2
  kernel.
- **Several right-hand sides are nearly free.** R columns share each
  stationary load: 128 right-hand sides take 12.7 ms against 6.0 ms for
  one (n = 4096), i.e. 0.10 ms per solve. The best CPU per-solve time
  here is 2.6 ms (block `bmm`, R = 64), so that is about 26x.
- **Checking convergence on the device every 1–8 steps costs little.**
  The device-side `while` test adds about 3 µs per check.
- **Precision is fp32.** The recursive residual reaches tol. The true
  residual against the fp64 A lands at 0.5–2x tol for κ = 1e3, and the
  error against fp64 is 1e-6 to 1e-5. At κ = 1e5 the error grows to
  1e-4 to 3e-4 (fp32 rounding of A times κ), where the fp64 CPU solvers
  do not have that floor. fp32 CG also takes about 5–25% more iterations
  than fp64 CG.

#### Independent solves, one process per logical core (whole chip)

n = 4096, 256 blocks, plain CG, 40 solves per process, every process
pinned to the same 2 host CPUs (driver script posted on #22). The rate
is solves / the slowest process's solve-loop time.

| logical cores | solves/s | scaling | per solve (ms) |
|---:|---:|---:|---:|
| 1 | 148 | 1.00x | 6.8 |
| 2 | 333 | 2.25x | 6.0 |
| 4 | 356 | 2.41x | 11.2 |

For reference, the same solve on the 2 host CPUs takes 46 ms in torch
fp32 (22 solves/s) and 57 ms in torch fp64 (block `bmm`, both threads
busy). Going from 2 to 4 cores is limited by the 2 host CPUs that four
processes share (per-solve host packing and copies) and by cores slowing
each other. On these 2 CPUs, batching right-hand sides into one launch
(above) is the better way to raise throughput.

## Trn2: fused CG follow-ups: column matvec, bf16 A, fused power iteration (#22)

Same machine, matrices, pinning and method as the section above (SDK
2.33, one logical core at LNC=2, host on 2 CPUs, tol 1e-5, warm median of
3). `row (#40)` is #40's matvec, still available as `matvec="row"`.
`column` is the new default (`matvec="auto"` picks it for R <= 32). Both
use the two-exchange LNC=2 step. "µs/iter" = (fused − fused with
max_iter=0) / iterations ("n/a" where the fixed cost's run-to-run noise
exceeds the iterations' time). Reproduce with `bench_cg_fused_trn2.py
--matvec row,column,column:bf16,column:bf16x2` and
`bench_power_fused_trn2.py`.

#### TensorE microbenchmark (ns per fp32 128x128 block, one physical core)

| form | stationary | moving | ns |
|---|---|---|---:|
| row (#40) | A block | p_j (128x1) | 425 |
| row, 4 moving columns | A block | 4 columns | 443 |
| vector-stationary, one PE tile | p_j (128x1) | A block | 206 |
| vector-stationary, 4 column tiles in turn (`column`) | p_j | A block | 50 |
| same, 32-column stationary | (128x32) | A block | 55 |
| same, always tile 0 | p_j | A block | 229 |
| TensorE transpose, fp32 128x128 (`nc_transpose`) | | | 109 |
| K=1 matmul transposing one 128-element row | | | 420 |

#### Plain CG, fp32 (ms to tol 1e-5, warm median of 3)

| n | blocks | density | κ | resident | row (#40) | column | column vs row | scipy | torch fp64 | torch fp32 | column vs best CPU | iters (row / column) | µs/iter (row / column) | true res (row / column) | err (row / column) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 0.250 | 1e+03 | 16 | 2.37 | 2.28 | 1.04x | 11.9 | 22.4 | 23.8 | 5.2x | 117 / 117 | 11 / 7 | 1.4e-05 / 1.9e-05 | 3.6e-06 / 3.6e-06 |
| 1024 | 32 | 0.500 | 1e+03 | 32 | 1.79 | 1.45 | 1.23x | 10.1 | 8.65 | 18.5 | 5.9x | 36 / 36 | 27 / 19 | 1.0e-05 / 6.0e-06 | 4.2e-06 / 3.8e-06 |
| 1024 | 64 | 1.000 | 1e+03 | 64 | 4.10 | 3.00 | 1.37x | 50.5 | 38.2 | 52.0 | 12.7x | 120 / 116 | 24 / 16 | 1.4e-05 / 1.5e-05 | 3.0e-06 / 3.7e-06 |
| 2048 | 32 | 0.125 | 1e+03 | 32 | 1.45 | 1.30 | 1.11x | 7.93 | 15.9 | 12.0 | 6.1x | 25 / 25 | 23 / 20 | 5.3e-06 / 6.2e-06 | 2.1e-06 / 1.1e-06 |
| 2048 | 64 | 0.250 | 1e+03 | 64 | 1.93 | 1.83 | 1.05x | 17.9 | 14.4 | 25.9 | 7.8x | 33 / 33 | 31 / 18 | 6.9e-06 / 6.9e-06 | 3.3e-06 / 3.1e-06 |
| 2048 | 128 | 0.500 | 1e+03 | 128 | 3.48 | 2.10 | 1.66x | 56.8 | 35.2 | 38.6 | 16.8x | 62 / 61 | 42 / n/a | 7.9e-06 / 9.2e-06 | 5.1e-06 / 6.1e-06 |
| 2048 | 256 | 1.000 | 1e+03 | 256 | 9.46 | 3.70 | 2.56x | 215.2 | 126.7 | 155.7 | 34.2x | 128 / 128 | 65 / 22 | 1.2e-05 / 1.5e-05 | 3.4e-06 / 3.5e-06 |
| 4096 | 64 | 0.062 | 1e+03 | 64 | 2.07 | 1.75 | 1.18x | 16.4 | 12.5 | 12.0 | 6.8x | 29 / 29 | 29 / 20 | 8.6e-06 / 8.7e-06 | 5.1e-06 / 5.1e-06 |
| 4096 | 128 | 0.125 | 1e+03 | 128 | 3.39 | 2.05 | 1.65x | 40.8 | 36.2 | 30.5 | 14.9x | 54 / 53 | 39 / 21 | 8.5e-06 / 8.5e-06 | 5.2e-06 / 6.1e-06 |
| 4096 | 256 | 0.250 | 1e+03 | 256 | 6.45 | 2.59 | 2.49x | 92.6 | 66.7 | 57.7 | 22.2x | 66 / 66 | 73 / 23 | 9.0e-06 / 8.0e-06 | 5.2e-06 / 5.1e-06 |
| 4096 | 512 | 0.500 | 1e+03 | 512 | 16.1 | 5.03 | 3.20x | 469.6 | 298.7 | 232.9 | 46.3x | 118 / 117 | 123 / 36 | 1.0e-05 / 8.9e-06 | 3.8e-06 / 4.4e-06 |
| 8192 | 122 | 0.030 | 1e+03 | 122 | 2.68 | 1.67 | 1.61x | 25.2 | 18.5 | 31.4 | 11.1x | 34 / 34 | 51 / 23 | 6.8e-06 / 9.8e-06 | 3.6e-06 / 4.2e-06 |
| 8192 | 256 | 0.062 | 1e+03 | 256 | 4.61 | 2.07 | 2.23x | 94.3 | 52.8 | 41.5 | 20.0x | 48 / 48 | 69 / 27 | 6.8e-06 / 7.0e-06 | 8.4e-06 / 8.5e-06 |
| 8192 | 512 | 0.125 | 1e+03 | 512 | 11.2 | 3.82 | 2.94x | 258.6 | 154.8 | 93.8 | 24.5x | 75 / 75 | 131 / 33 | 7.6e-06 / 8.0e-06 | 1.0e-05 / 1.0e-05 |
| 8192 | 1024 | 0.250 | 1e+03 | 640 | 29.7 | 11.9 | 2.51x | 1728.3 | 976.6 | 559.8 | 47.2x | 114 / 116 | 250 / 93 | 9.1e-06 / 9.4e-06 | 1.1e-05 / 9.7e-06 |
| 2048 | 64 | 0.250 | 1e+05 | 64 | 2.56 | 1.74 | 1.47x | 20.7 | 35.0 | 31.1 | 11.9x | 43 / 43 | 35 / 15 | 3.5e-05 / 5.0e-05 | 2.0e-04 / 2.3e-04 |
| 4096 | 128 | 0.125 | 1e+05 | 128 | 4.01 | 2.32 | 1.73x | 48.5 | 38.2 | 37.4 | 16.1x | 67 / 67 | 40 / 22 | 1.9e-05 / 1.4e-05 | 1.4e-04 / 1.2e-04 |
| 8192 | 256 | 0.062 | 1e+05 | 256 | 5.50 | 2.58 | 2.13x | 89.5 | 70.9 | 55.8 | 21.6x | 61 / 60 | 69 / 27 | 1.4e-05 / 1.3e-05 | 3.5e-04 / 2.8e-04 |
| 8192 | 1024 | 0.250 | 1e+05 | 640 | 52.0 | 19.6 | 2.65x | 2876.2 | 1796.4 | 1103.0 | 56.2x | 215 / 213 | 238 / 88 | 9.5e-06 / 1.0e-05 | 1.2e-05 / 1.5e-05 |

#### Block-Jacobi PCG, fp32 (ms to tol 1e-5, warm median of 3)

| n | blocks | density | κ | resident | row (#40) | column | column vs row | scipy | torch fp64 | torch fp32 | column vs best CPU | iters (row / column) | µs/iter (row / column) | true res (row / column) | err (row / column) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 0.250 | 1e+03 | 16 | 2.81 | 2.32 | 1.21x | 20.2 | 33.9 | 31.5 | 8.7x | 98 / 98 | 20 / 16 | 1.8e-05 / 2.0e-05 | 6.3e-06 / 6.4e-06 |
| 1024 | 32 | 0.500 | 1e+03 | 32 | 1.91 | 1.43 | 1.33x | 9.68 | 9.11 | 17.1 | 6.4x | 26 / 26 | 35 / 23 | 9.4e-06 / 9.5e-06 | 5.3e-06 / 5.3e-06 |
| 1024 | 64 | 1.000 | 1e+03 | 64 | 3.76 | 2.51 | 1.50x | 66.0 | 50.0 | 53.1 | 19.9x | 113 / 112 | 28 / 16 | 1.4e-05 / 1.2e-05 | 6.2e-06 / 6.4e-06 |
| 2048 | 32 | 0.125 | 1e+03 | 32 | 1.20 | 1.31 | 0.92x | 9.67 | 18.2 | 16.6 | 7.4x | 21 / 21 | 29 / 26 | 7.2e-06 / 7.2e-06 | 2.5e-06 / 1.8e-06 |
| 2048 | 64 | 0.250 | 1e+03 | 64 | 1.86 | 1.57 | 1.18x | 18.6 | 14.0 | 25.1 | 8.9x | 27 / 27 | 35 / 26 | 3.7e-06 / 3.6e-06 | 2.7e-06 / 2.0e-06 |
| 2048 | 128 | 0.500 | 1e+03 | 128 | 3.75 | 2.51 | 1.49x | 57.4 | 39.0 | 35.6 | 14.2x | 51 / 50 | 50 / 27 | 9.1e-06 / 9.1e-06 | 3.6e-06 / 6.5e-06 |
| 2048 | 256 | 1.000 | 1e+03 | 256 | 12.2 | 4.81 | 2.54x | 309.2 | 171.5 | 235.8 | 35.7x | 158 / 157 | 72 / 24 | 1.1e-05 / 1.2e-05 | 4.9e-06 / 5.3e-06 |
| 4096 | 64 | 0.062 | 1e+03 | 64 | 1.78 | 1.58 | 1.12x | 19.1 | 16.5 | 26.7 | 10.4x | 24 / 24 | 32 / 25 | 4.8e-06 / 6.1e-06 | 2.6e-06 / 2.7e-06 |
| 4096 | 128 | 0.125 | 1e+03 | 128 | 3.13 | 2.01 | 1.55x | 51.4 | 38.7 | 32.9 | 16.3x | 44 / 43 | 50 / 27 | 6.1e-06 / 1.0e-05 | 3.2e-06 / 5.6e-06 |
| 4096 | 256 | 0.250 | 1e+03 | 256 | 5.95 | 2.45 | 2.43x | 83.1 | 59.5 | 50.1 | 20.4x | 49 / 49 | 84 / 33 | 9.7e-06 / 9.8e-06 | 4.7e-06 / 5.0e-06 |
| 4096 | 512 | 0.500 | 1e+03 | 512 | 12.4 | 4.62 | 2.69x | 681.2 | 238.2 | 170.2 | 36.8x | 81 / 85 | 143 / 41 | 9.8e-06 / 7.3e-06 | 6.9e-06 / 4.1e-06 |
| 8192 | 122 | 0.030 | 1e+03 | 122 | 2.76 | 1.80 | 1.53x | 31.7 | 27.7 | 29.4 | 15.3x | 30 / 30 | 50 / 31 | 6.0e-06 / 6.0e-06 | 4.0e-06 / 3.7e-06 |
| 8192 | 256 | 0.062 | 1e+03 | 256 | 5.41 | 2.72 | 1.99x | 100.3 | 65.2 | 54.0 | 19.9x | 41 / 41 | 97 / 39 | 9.6e-06 / 9.4e-06 | 2.0e-05 / 2.0e-05 |
| 8192 | 512 | 0.125 | 1e+03 | 512 | 10.8 | 3.42 | 3.16x | 350.9 | 182.4 | 110.9 | 32.4x | 61 / 61 | 149 / 22 | 7.9e-06 / 7.8e-06 | 8.7e-06 / 8.7e-06 |
| 8192 | 1024 | 0.250 | 1e+03 | 576 | 20.5 | 9.25 | 2.21x | 1298.7 | 682.4 | 370.8 | 40.1x | 74 / 76 | 259 / 106 | 9.3e-06 / 7.8e-06 | 1.3e-05 / 1.1e-05 |
| 2048 | 64 | 0.250 | 1e+05 | 64 | 2.49 | 1.85 | 1.34x | 22.8 | 17.8 | 30.5 | 9.6x | 33 / 33 | 38 / 26 | 3.5e-05 / 3.5e-05 | 2.5e-04 / 2.3e-04 |
| 4096 | 128 | 0.125 | 1e+05 | 128 | 3.52 | 2.44 | 1.45x | 48.0 | 43.1 | 37.7 | 15.5x | 54 / 54 | 48 / 25 | 1.9e-05 / 1.5e-05 | 1.1e-04 / 1.1e-04 |
| 8192 | 256 | 0.062 | 1e+05 | 256 | 5.60 | 2.69 | 2.08x | 96.8 | 85.8 | 63.9 | 23.8x | 54 / 54 | 72 / 33 | 1.3e-05 / 1.5e-05 | 3.3e-04 / 3.2e-04 |
| 8192 | 1024 | 0.250 | 1e+05 | 576 | 35.1 | 13.7 | 2.56x | 1850.7 | 1211.8 | 616.0 | 44.9x | 130 / 132 | 255 / 95 | 1.0e-05 / 1.3e-05 | 1.1e-05 / 1.8e-05 |

#### bf16 A (opt-in), column matvec (ms to tol 1e-5; true residual and error against the fp64 A)

| n | blocks | κ | precond | fp32 ms | bf16 ms | bf16x2 ms | resident fp32 / bf16 | iters fp32 / bf16 / bf16x2 | true res fp32 | true res bf16 | true res bf16x2 | err fp32 | err bf16 | err bf16x2 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 1e+03 | PCG | 2.32 | 2.26 | 2.34 | 16 / 16 | 98 / 96 / 99 | 2.0e-05 | 7.5e-02 | 1.5e-04 | 6.4e-06 | 8.5e-02 | 1.3e-04 |
| 1024 | 16 | 1e+03 | CG | 2.28 | 2.50 | 2.74 | 16 / 16 | 117 / 139 / 124 | 1.9e-05 | 7.5e-02 | 1.2e-04 | 3.6e-06 | 8.5e-02 | 1.3e-04 |
| 1024 | 32 | 1e+03 | PCG | 1.43 | 1.62 | 1.61 | 32 / 32 | 26 / 27 / 27 | 9.5e-06 | 3.5e-03 | 9.1e-06 | 5.3e-06 | 2.8e-02 | 2.8e-05 |
| 1024 | 32 | 1e+03 | CG | 1.45 | 1.58 | 1.74 | 32 / 32 | 36 / 37 / 37 | 6.0e-06 | 3.5e-03 | 8.6e-06 | 3.8e-06 | 2.8e-02 | 2.8e-05 |
| 1024 | 64 | 1e+03 | PCG | 2.51 | 2.88 | 3.04 | 64 / 64 | 112 / 114 / 114 | 1.2e-05 | 7.4e-02 | 1.6e-04 | 6.4e-06 | 8.3e-02 | 1.2e-04 |
| 1024 | 64 | 1e+03 | CG | 3.00 | 2.60 | 2.90 | 64 / 64 | 116 / 137 / 123 | 1.5e-05 | 7.4e-02 | 1.1e-04 | 3.7e-06 | 8.3e-02 | 1.2e-04 |
| 2048 | 32 | 1e+03 | PCG | 1.31 | 1.20 | 1.41 | 32 / 32 | 21 / 21 / 21 | 7.2e-06 | 5.8e-03 | 9.8e-06 | 1.8e-06 | 1.4e-01 | 5.3e-05 |
| 2048 | 32 | 1e+03 | CG | 1.30 | 1.31 | 1.39 | 32 / 32 | 25 / 25 / 25 | 6.2e-06 | 5.8e-03 | 1.0e-05 | 1.1e-06 | 1.4e-01 | 5.3e-05 |
| 2048 | 64 | 1e+03 | PCG | 1.57 | 1.20 | 1.61 | 64 / 64 | 27 / 27 / 27 | 3.6e-06 | 2.9e-03 | 9.7e-06 | 2.0e-06 | 3.1e-02 | 2.9e-05 |
| 2048 | 64 | 1e+03 | CG | 1.83 | 1.68 | 1.67 | 64 / 64 | 33 / 35 / 34 | 6.9e-06 | 2.9e-03 | 8.1e-06 | 3.1e-06 | 3.1e-02 | 2.9e-05 |
| 2048 | 128 | 1e+03 | PCG | 2.51 | 1.92 | 2.21 | 128 / 128 | 50 / 52 / 52 | 9.1e-06 | 3.4e-03 | 1.0e-05 | 6.5e-06 | 4.5e-03 | 1.0e-05 |
| 2048 | 128 | 1e+03 | CG | 2.10 | 2.19 | 2.24 | 128 / 128 | 61 / 65 / 64 | 9.2e-06 | 3.4e-03 | 1.1e-05 | 6.1e-06 | 4.5e-03 | 1.0e-05 |
| 2048 | 256 | 1e+03 | PCG | 4.81 | 4.11 | 4.49 | 256 / 256 | 157 / 162 / 160 | 1.2e-05 | 5.9e-02 | 1.1e-04 | 5.3e-06 | 6.6e-02 | 9.6e-05 |
| 2048 | 256 | 1e+03 | CG | 3.70 | 3.22 | 3.59 | 256 / 256 | 128 / 147 / 136 | 1.5e-05 | 5.9e-02 | 9.4e-05 | 3.5e-06 | 6.6e-02 | 9.6e-05 |
| 4096 | 64 | 1e+03 | PCG | 1.58 | 1.66 | 1.70 | 64 / 64 | 24 / 24 / 24 | 6.1e-06 | 1.4e-03 | 8.7e-06 | 2.7e-06 | 5.4e-02 | 9.8e-06 |
| 4096 | 64 | 1e+03 | CG | 1.75 | 1.77 | 1.78 | 64 / 64 | 29 / 30 / 29 | 8.7e-06 | 1.4e-03 | 9.2e-06 | 5.1e-06 | 5.4e-02 | 1.1e-05 |
| 4096 | 128 | 1e+03 | PCG | 2.01 | 1.87 | 2.29 | 128 / 128 | 43 / 44 / 44 | 1.0e-05 | 2.5e-03 | 1.1e-05 | 5.6e-06 | 6.2e-02 | 2.1e-04 |
| 4096 | 128 | 1e+03 | CG | 2.05 | 1.78 | 1.85 | 128 / 128 | 53 / 55 / 54 | 8.5e-06 | 2.5e-03 | 1.3e-05 | 6.1e-06 | 6.2e-02 | 2.1e-04 |
| 4096 | 256 | 1e+03 | PCG | 2.45 | 2.29 | 2.29 | 256 / 256 | 49 / 49 / 49 | 9.8e-06 | 5.5e-03 | 1.2e-05 | 5.0e-06 | 6.4e-02 | 7.0e-06 |
| 4096 | 256 | 1e+03 | CG | 2.59 | 2.55 | 2.52 | 256 / 256 | 66 / 68 / 66 | 8.0e-06 | 5.5e-03 | 1.1e-05 | 5.1e-06 | 6.4e-02 | 7.7e-06 |
| 4096 | 512 | 1e+03 | PCG | 4.62 | 3.06 | 3.99 | 512 / 512 | 85 / 85 / 84 | 7.3e-06 | 1.2e-02 | 2.4e-05 | 4.1e-06 | 1.2e-02 | 2.0e-05 |
| 4096 | 512 | 1e+03 | CG | 5.03 | 3.88 | 4.26 | 512 / 512 | 117 / 121 / 121 | 8.9e-06 | 1.2e-02 | 2.2e-05 | 4.4e-06 | 1.2e-02 | 1.9e-05 |
| 8192 | 122 | 1e+03 | PCG | 1.80 | 2.07 | 1.82 | 122 / 122 | 30 / 30 / 30 | 6.0e-06 | 2.4e-03 | 7.2e-06 | 3.7e-06 | 1.1e-01 | 9.1e-05 |
| 8192 | 122 | 1e+03 | CG | 1.67 | 1.62 | 1.71 | 122 / 122 | 34 / 35 / 34 | 9.8e-06 | 2.4e-03 | 1.0e-05 | 4.2e-06 | 1.1e-01 | 9.1e-05 |
| 8192 | 256 | 1e+03 | PCG | 2.72 | 2.32 | 2.58 | 256 / 256 | 41 / 40 / 43 | 9.4e-06 | 3.7e-03 | 7.9e-06 | 2.0e-05 | 1.2e-01 | 9.3e-05 |
| 8192 | 256 | 1e+03 | CG | 2.07 | 2.13 | 1.99 | 256 / 256 | 48 / 45 / 48 | 7.0e-06 | 3.7e-03 | 1.0e-05 | 8.5e-06 | 1.2e-01 | 9.4e-05 |
| 8192 | 512 | 1e+03 | PCG | 3.42 | 2.81 | 3.66 | 512 / 512 | 61 / 63 / 63 | 7.8e-06 | 2.9e-03 | 1.1e-05 | 8.7e-06 | 8.9e-03 | 8.7e-06 |
| 8192 | 512 | 1e+03 | CG | 3.82 | 2.92 | 3.19 | 512 / 512 | 75 / 75 / 75 | 8.0e-06 | 2.9e-03 | 1.1e-05 | 1.0e-05 | 8.9e-03 | 1.4e-05 |
| 8192 | 1024 | 1e+03 | PCG | 9.25 | 4.40 | 8.15 | 576 / 1024 | 76 / 74 / 75 | 7.8e-06 | 5.4e-03 | 1.6e-05 | 1.1e-05 | 5.8e-03 | 1.6e-05 |
| 8192 | 1024 | 1e+03 | CG | 11.9 | 5.03 | 11.1 | 640 / 1024 | 116 / 118 / 118 | 9.4e-06 | 5.4e-03 | 1.4e-05 | 9.7e-06 | 5.8e-03 | 1.2e-05 |
| 2048 | 64 | 1e+05 | PCG | 1.85 | 1.61 | 2.00 | 64 / 64 | 33 / 30 / 33 | 3.5e-05 | 2.9e-02 | 7.7e-04 | 2.3e-04 | 9.4e-01 | 2.4e-02 |
| 2048 | 64 | 1e+05 | CG | 1.74 | 1.83 | 1.86 | 64 / 64 | 43 / 38 / 43 | 5.0e-05 | 2.9e-02 | 7.5e-04 | 2.3e-04 | 9.4e-01 | 2.4e-02 |
| 4096 | 128 | 1e+05 | PCG | 2.44 | 2.04 | 2.30 | 128 / 128 | 54 / 51 / 54 | 1.5e-05 | 1.8e-02 | 1.5e-04 | 1.1e-04 | 8.0e-01 | 7.5e-03 |
| 4096 | 128 | 1e+05 | CG | 2.32 | 1.84 | 2.25 | 128 / 128 | 67 / 64 / 68 | 1.4e-05 | 1.8e-02 | 1.4e-04 | 1.2e-04 | 8.0e-01 | 7.6e-03 |
| 8192 | 256 | 1e+05 | PCG | 2.69 | 2.11 | 2.67 | 256 / 256 | 54 / 50 / 54 | 1.5e-05 | 8.8e-03 | 1.1e-04 | 3.2e-04 | 8.6e-01 | 9.9e-03 |
| 8192 | 256 | 1e+05 | CG | 2.58 | 2.20 | 2.20 | 256 / 256 | 60 / 56 / 61 | 1.3e-05 | 8.8e-03 | 1.1e-04 | 2.8e-04 | 8.6e-01 | 9.9e-03 |
| 8192 | 1024 | 1e+05 | PCG | 13.7 | 3.45 | 13.4 | 576 / 1024 | 132 / 63 / 133 | 1.3e-05 | 6.5e+32 | 7.4e-05 | 1.8e-05 | 5.4e+33 | 6.9e-04 |
| 8192 | 1024 | 1e+05 | CG | 19.6 | 3.96 | 17.8 | 640 / 1024 | 213 / 101 / 218 | 1.0e-05 | nan | 7.4e-05 | 1.5e-05 | nan | 7.1e-04 |

#### Small cases, warm median of 11 (ms)

| n | blocks | precond | row | column | column vs row |
|---:|---:|---|---:|---:|---:|
| 1024 | 16 | block_jacobi | 2.53 | 2.32 | 1.09x |
| 1024 | 16 | none | 2.66 | 2.45 | 1.09x |
| 1024 | 32 | block_jacobi | 1.52 | 1.27 | 1.19x |
| 1024 | 32 | none | 1.52 | 1.42 | 1.07x |
| 2048 | 32 | block_jacobi | 1.40 | 1.33 | 1.05x |
| 2048 | 32 | none | 1.45 | 1.32 | 1.10x |
| 2048 | 64 | block_jacobi | 1.89 | 1.56 | 1.21x |
| 2048 | 64 | none | 1.89 | 1.55 | 1.22x |
| 4096 | 64 | block_jacobi | 1.64 | 1.53 | 1.07x |
| 4096 | 64 | none | 1.64 | 1.58 | 1.03x |

#### µs per CG iteration (plain, fixed iterations: (t(576) − t(64)) / 512)

| n | blocks | row, LNC=2 split | column, LNC=2 split | column, LNC=2 no split | column, LNC=1 | row, LNC=1 | column vs row (split) | split vs LNC=1 (column) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 256 | 63.4 | 19.3 | 22.6 | 21.3 | 113.3 | 3.29x | 1.10x |
| 4096 | 256 | 64.8 | 21.8 | 25.6 | 25.2 | 113.3 | 2.98x | 1.16x |
| 4096 | 512 | 118.6 | 31.4 | 71.3 | 67.5 | 222.8 | 3.78x | 2.15x |
| 8192 | 512 | 118.7 | 30.3 | 62.5 | 58.3 | 222.9 | 3.92x | 1.92x |
| 8192 | 1024 | 232.9 | 87.0 | 186.5 | 167.7 | 441.0 | 2.68x | 1.93x |

#### Several right-hand sides (n = 4096, 256 blocks; ms total, ms per right-hand side)

| precond | R | row total | column total | row per RHS | column per RHS | torch fp64 per RHS | torch fp32 per RHS | best fused vs best CPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| block_jacobi | 1 | 5.3 | 2.6 | 5.305 | 2.568 | 84.290 | 71.733 | 28x |
| block_jacobi | 4 | 9.9 | 3.1 | 2.488 | 0.763 | 17.023 | 14.869 | 19x |
| block_jacobi | 16 | 10.0 | 8.0 | 0.624 | 0.498 | 8.295 | 4.783 | 10x |
| block_jacobi | 32 | 9.8 | 8.3 | 0.307 | 0.260 | 8.782 | 3.910 | 15x |
| block_jacobi | 64 | 11.9 | (row) | 0.186 | - | 14.245 | 3.482 | 19x |
| block_jacobi | 128 | 19.7 | (row) | 0.154 | - | 15.291 | 8.452 | 55x |
| none | 1 | 6.0 | 2.5 | 5.958 | 2.546 | 80.064 | 75.837 | 30x |
| none | 4 | 10.7 | 7.5 | 2.672 | 1.865 | 18.938 | 17.136 | 9x |
| none | 16 | 9.7 | 3.2 | 0.607 | 0.199 | 7.785 | 5.284 | 27x |
| none | 32 | 10.8 | 4.8 | 0.338 | 0.149 | 9.915 | 5.259 | 35x |
| none | 64 | 32.5 | (row) | 0.507 | - | 19.169 | 4.587 | 9x |
| none | 128 | 22.9 | (row) | 0.179 | - | 19.703 | 10.094 | 56x |

#### Power iteration, 200 iterations (ms; λ relative error against fp64 eigvalsh)

| n | blocks | fused column | fused row | host_bsr (`power_iteration_bsr` on device) | torch fp64 | torch fp32 | scipy | fused µs/iter | column vs best CPU | column vs host_bsr | λ err fused | λ err fp64 CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 32 | 3.2 | 4.1 | 136 | 29 | 40 | 41 | 16 | 9x | 42x | 2.6e-08 | 1.0e-09 |
| 2048 | 64 | 4.0 | 5.7 | 150 | 79 | 101 | 168 | 20 | 20x | 38x | 5.2e-08 | 2.6e-09 |
| 2048 | 256 | 4.9 | 13.6 | 168 | 213 | 235 | 306 | 25 | 43x | 34x | 9.5e-08 | 5.4e-10 |
| 4096 | 256 | 5.5 | 14.6 | 225 | 262 | 200 | 506 | 28 | 36x | 41x | 3.1e-08 | 6.0e-11 |
| 4096 | 512 | 6.7 | 25.0 | 201 | 540 | 344 | 769 | 33 | 51x | 30x | 3.0e-08 | 2.2e-10 |
| 8192 | 256 | 5.0 | 13.3 | 159 | 207 | 242 | 326 | 25 | 42x | 32x | 2.8e-08 | 1.0e-09 |
| 8192 | 1024 | 18.5 | 49.0 | 258 | 1381 | 633 | 2671 | 92 | 34x | 14x | 1.6e-08 | 2.7e-10 |

#### Independent solves, one process per logical core (whole chip, n = 4096, 256 blocks, 40 solves per process, 2 shared host CPUs)

| matvec | cores | solves/s | scaling |
|---|---:|---:|---:|
| column | 1 | 294 | 1.00x |
| column | 2 | 632 | 2.15x |
| column | 4 | 699 | 2.38x |
| row | 1 | 147 | 1.00x |
| row | 2 | 321 | 2.18x |
| row | 4 | 333 | 2.26x |

### Reading the follow-up tables

- **The column matvec beats the row matvec in every case measured.** It
  is 1.03–1.2x faster at the smallest sizes (median of 11 runs), up to
  3.2x faster to tolerance, and 2.7–3.9x faster per iteration. It is as
  accurate: true residuals and errors match the row matvec within
  run-to-run variation, and iteration counts differ by at most 4. That is
  why it is the new default for R <= 32.
- **Against the best 2-thread CPU solver, fused CG is now 5–56x faster to
  tolerance.** In #40 it was 3.5–11x.
- **The per-block cost is no longer the floor.** At about 50 ns per
  resident block, an iteration costs about 11 µs fixed at LNC=2 (8 µs at
  LNC=1) plus the blocks. The fixed part is the chain of small VectorE /
  TensorE steps and the `sendrecv`s. Cutting the exchanges from three to
  two saved 2–3 µs per iteration. So the LNC=2 row split now gives
  1.1–1.2x up to 256 blocks and about 2x at 512–1024 blocks.
- **SBUF residency matters again.** At n = 8192 with 1024 blocks, 384 of
  the fp32 blocks (64 KiB each) stream from HBM and set the pace: 87 µs
  per iteration, against 30 µs with 512 resident blocks. bf16 A keeps all
  1024 blocks resident and is 2.3x faster there, but at bf16 accuracy.
- **bf16 A does not pay as a replacement for fp32** (opt-in only). The
  vectors stay fp32 (split into bf16 hi + lo) and accumulation is fp32,
  so the recursive residual converges, but for the rounded matrix. At
  κ = 1e3 the true residual against the fp64 A stalls at 1e-3 to 7e-2. At
  κ = 1e5 the error is about 0.9, or CG breaks down because the rounded
  matrix is indefinite (n = 8192, 1024 blocks). bf16 hi + lo (`bf16x2`)
  recovers most of the accuracy at κ = 1e3 (1–15x tol) but is 5–70x tol
  at κ = 1e5. It also costs as much as fp32: two matmuls and the same
  bytes per block.
- **Several right-hand sides.** The column matvec takes up to 32 and is
  1.2–3.3x faster per solve than the row matvec there. Above 32 the row
  matvec is used.
- **Fused power iteration** (`power_iteration_bsr_fused`) runs 200
  iterations in 3–19 ms. That is 9–51x faster than the best CPU baseline
  and 14–42x faster than `power_iteration_bsr` on the device, which syncs
  with the host twice per iteration. λ matches fp64 eigvalsh to about
  3e-8 (fp32).
- **One process per logical core.** The column matvec doubles the
  single-core rate (294 against 147 solves/s). Going to 2 and 4 cores
  scales 2.15x and 2.38x, limited by the 2 shared host CPUs as in #40.


## Trn2: fused CG fixed per-iteration cost (#22)

Same machine, matrices, pinning and method as the sections above (SDK
2.33, one logical core, host on 2 CPUs, tol 1e-5). "#41" is `main` after
#41; "new" is this change with the default (standard) recurrence;
"Chronopoulos–Gear" is the opt-in `recurrence="chronopoulos_gear"` (plain
CG only). LNC=1 runs on one physical core (`NEURON_LOGICAL_NC_CONFIG=1`).
Time to tolerance is a warm median of 9 solves; per-iteration times are
(t(576) − t(64)) / 512 with tol = 0, median of 7. True residual and error
are against the fp64 A and an fp64 direct solve.

#### Where one plain CG iteration goes (n = 4096, 256 blocks, LNC=2, device profile, slower core)

| step | #41 (µs) | new (µs) |
|---|---:|---:|
| matvec (TensorE, ~131 blocks on this core) | 10.3 | 9.3 (column groups balanced) |
| finish the matvec: PSUM copies, transposes, copy | 2.2 (4 + 4 + 1 ops) | 0.95 (1 + 1 + 1) |
| exchange 1 | 2.9 (partial p.Ap, default DMA) | 0.85 (rows of A p, GpSimd DMA) |
| p.Ap (multiply-reduce, ones matmul) | 0.9 | 0.6 |
| step size, x and r updates | 1.2 | 0.65 (x moved into the next matvec) |
| exchange 2 (+ copies) | 2.4 (rows of r) | — |
| r.r | 1.0 | 0.6 |
| beta, p update | 0.55 | 0.5 |
| **fixed part** | **~10.8** | **~4.4** |

At LNC=1 the fixed part is about 3.5 µs (#41: about 8 µs).

#### µs per plain CG iteration ((t(576) − t(64)) / 512)

| n | blocks | #41 LNC=2 | new LNC=2 | C–G LNC=2 | #41 LNC=2 no split | new LNC=2 no split | #41 LNC=1 | new LNC=1 | C–G LNC=1 | new vs #41 (LNC=2 / LNC=1) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2048 | 256 | 19.5 | 14.6 | 13.7 | 23.1 | 20.5 | 21.9 | 20.1 | 18.5 | 1.33x / 1.09x |
| 4096 | 256 | 21.9 | 15.3 | 14.8 | 26.0 | 21.1 | 25.5 | 20.2 | 19.0 | 1.43x / 1.26x |
| 4096 | 512 | 31.7 | 24.3 | 23.3 | 71.9 | 67.9 | 68.3 | 62.4 | 61.7 | 1.30x / 1.09x |
| 8192 | 512 | 31.0 | 24.1 | 22.9 | 63.2 | 63.2 | 58.7 | 57.6 | 56.6 | 1.29x / 1.02x |
| 8192 | 1024 | 87.0 | 67.9 | 67.8 | 186.0 | 176.6 | 168.2 | 155.4 | 154.1 | 1.28x / 1.08x |

#### µs per PCG iteration ((t(576) − t(64)) / 512)

| n | blocks | #41 LNC=2 | new LNC=2 | #41 LNC=1 | new LNC=1 | new vs #41 (LNC=2 / LNC=1) |
|---:|---:|---:|---:|---:|---:|---|
| 2048 | 256 | 22.9 | 18.4 | 26.2 | 23.2 | 1.24x / 1.13x |
| 4096 | 256 | 27.2 | 20.3 | 31.4 | 23.8 | 1.34x / 1.32x |
| 4096 | 512 | 36.7 | 29.8 | 72.1 | 65.1 | 1.23x / 1.11x |
| 8192 | 512 | 37.3 | 30.5 | 77.9 | 68.8 | 1.22x / 1.13x |
| 8192 | 1024 | 91.7 | 78.4 | 170.5 | 166.2 | 1.17x / 1.03x |

#### Plain CG, LNC=2 (ms to tol 1e-5, warm median of 9)

| n | blocks | κ | #41 | new | speedup | Chronopoulos–Gear | C–G speedup | iters (#41 / new / C–G) | true res (#41 / new / C–G) | err vs fp64 (#41 / new / C–G) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1024 | 16 | 1e+03 | 2.31 | 1.72 | 1.34x | 1.40 | 1.66x | 117 / 118 / 123 | 1.9e-05 / 1.9e-05 / 2.0e-05 | 3.6e-06 / 3.5e-06 / 1.5e-05 |
| 1024 | 32 | 1e+03 | 1.40 | 0.96 | 1.46x | 1.09 | 1.28x | 36 / 36 / 37 | 6.0e-06 / 9.0e-06 / 5.1e-06 | 3.8e-06 / 4.0e-06 / 5.9e-06 |
| 1024 | 64 | 1e+03 | 2.53 | 1.79 | 1.41x | 1.84 | 1.38x | 116 / 120 / 126 | 1.5e-05 / 1.9e-05 / 2.0e-05 | 3.7e-06 / 3.5e-06 / 1.7e-05 |
| 2048 | 32 | 1e+03 | 1.31 | 1.14 | 1.15x | 1.17 | 1.12x | 25 / 25 / 25 | 6.2e-06 / 5.4e-06 / 5.6e-06 | 1.1e-06 / 1.1e-06 / 2.6e-06 |
| 2048 | 64 | 1e+03 | 1.50 | 1.28 | 1.18x | 1.29 | 1.16x | 33 / 33 / 33 | 6.9e-06 / 6.6e-06 / 6.7e-06 | 3.1e-06 / 2.7e-06 / 5.0e-06 |
| 2048 | 128 | 1e+03 | 2.05 | 1.73 | 1.18x | 1.65 | 1.24x | 61 / 62 / 63 | 9.2e-06 / 8.4e-06 / 7.0e-06 | 6.1e-06 / 4.2e-06 / 4.6e-06 |
| 2048 | 256 | 1e+03 | 3.47 | 2.85 | 1.22x | 2.90 | 1.20x | 128 / 127 / 134 | 1.5e-05 / 1.3e-05 / 2.7e-05 | 3.5e-06 / 3.5e-06 / 1.9e-05 |
| 4096 | 64 | 1e+03 | 1.43 | 1.29 | 1.10x | 1.23 | 1.16x | 29 / 29 / 29 | 8.7e-06 / 8.8e-06 / 8.6e-06 | 5.1e-06 / 5.1e-06 / 1.2e-05 |
| 4096 | 128 | 1e+03 | 1.94 | 1.53 | 1.27x | 1.51 | 1.28x | 53 / 54 / 53 | 8.5e-06 / 7.3e-06 / 8.7e-06 | 6.1e-06 / 5.5e-06 / 6.1e-06 |
| 4096 | 256 | 1e+03 | 2.53 | 2.16 | 1.18x | 2.07 | 1.22x | 66 / 66 / 66 | 8.0e-06 / 8.0e-06 / 8.4e-06 | 5.1e-06 / 5.1e-06 / 5.8e-06 |
| 4096 | 512 | 1e+03 | 4.85 | 4.12 | 1.18x | 3.86 | 1.25x | 117 / 117 / 120 | 8.9e-06 / 9.2e-06 / 7.6e-06 | 4.4e-06 / 4.3e-06 / 4.7e-06 |
| 8192 | 122 | 1e+03 | 1.75 | 1.54 | 1.13x | 1.27 | 1.37x | 34 / 34 / 34 | 9.8e-06 / 9.1e-06 / 9.9e-06 | 4.2e-06 / 4.0e-06 / 1.1e-05 |
| 8192 | 256 | 1e+03 | 2.13 | 1.79 | 1.19x | 1.84 | 1.16x | 48 / 48 / 48 | 7.0e-06 / 6.8e-06 / 7.6e-06 | 8.5e-06 / 8.4e-06 / 8.7e-06 |
| 8192 | 512 | 1e+03 | 3.47 | 3.08 | 1.13x | 3.01 | 1.15x | 75 / 76 / 76 | 8.0e-06 / 7.7e-06 / 7.2e-06 | 1.0e-05 / 1.0e-05 / 9.7e-06 |
| 8192 | 1024 | 1e+03 | 11.6 | 9.43 | 1.23x | 9.38 | 1.24x | 116 / 115 / 116 | 9.4e-06 / 9.7e-06 / 9.2e-06 | 9.7e-06 / 1.0e-05 / 9.8e-06 |
| 2048 | 64 | 1e+05 | 1.57 | 1.40 | 1.13x | 1.34 | 1.18x | 43 / 43 / 41 | 5.0e-05 / 4.4e-05 / 1.1e-04 | 2.3e-04 / 1.6e-04 / 2.0e-04 |
| 4096 | 128 | 1e+05 | 1.98 | 1.79 | 1.11x | 1.78 | 1.11x | 67 / 67 / 66 | 1.4e-05 / 1.5e-05 / 3.9e-05 | 1.2e-04 / 1.6e-04 / 6.8e-04 |
| 8192 | 256 | 1e+05 | 6.99 | 2.15 | 3.25x | 2.09 | 3.35x | 60 / 60 / 59 | 1.3e-05 / 1.5e-05 / 2.8e-05 | 2.8e-04 / 3.5e-04 / 2.0e-05 |
| 8192 | 1024 | 1e+05 | 20.1 | 16.1 | 1.25x | 16.0 | 1.25x | 213 / 214 / 214 | 1.0e-05 / 1.2e-05 / 3.5e-05 | 1.5e-05 / 1.2e-05 / 4.1e-05 |

#### Block-Jacobi PCG, LNC=2 (ms to tol 1e-5, warm median of 9)

| n | blocks | κ | #41 | new | speedup | iters (#41 / new) | true res (#41 / new) | err vs fp64 (#41 / new) |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 1024 | 16 | 1e+03 | 2.27 | 1.93 | 1.18x | 98 / 98 | 2.0e-05 / 1.2e-05 | 6.4e-06 / 6.4e-06 |
| 1024 | 32 | 1e+03 | 1.33 | 0.97 | 1.37x | 26 / 26 | 9.5e-06 / 9.4e-06 | 5.3e-06 / 5.3e-06 |
| 1024 | 64 | 1e+03 | 2.59 | 2.30 | 1.13x | 112 / 114 | 1.2e-05 / 2.3e-05 | 6.4e-06 / 6.0e-06 |
| 2048 | 32 | 1e+03 | 1.37 | 1.20 | 1.14x | 21 / 21 | 7.2e-06 / 7.3e-06 | 1.8e-06 / 2.8e-06 |
| 2048 | 64 | 1e+03 | 1.46 | 1.48 | 0.99x | 27 / 27 | 3.6e-06 / 3.6e-06 | 2.0e-06 / 2.4e-06 |
| 2048 | 128 | 1e+03 | 2.13 | 1.83 | 1.16x | 50 / 50 | 9.1e-06 / 9.0e-06 | 6.5e-06 / 6.5e-06 |
| 2048 | 256 | 1e+03 | 4.60 | 4.02 | 1.14x | 157 / 154 | 1.2e-05 / 1.2e-05 | 5.3e-06 / 5.9e-06 |
| 4096 | 64 | 1e+03 | 1.47 | 1.36 | 1.08x | 24 / 24 | 6.1e-06 / 6.3e-06 | 2.7e-06 / 2.7e-06 |
| 4096 | 128 | 1e+03 | 1.98 | 1.78 | 1.11x | 43 / 44 | 1.0e-05 / 7.2e-06 | 5.6e-06 / 3.3e-06 |
| 4096 | 256 | 1e+03 | 2.53 | 2.23 | 1.13x | 49 / 49 | 9.8e-06 / 9.7e-06 | 5.0e-06 / 4.9e-06 |
| 4096 | 512 | 1e+03 | 4.79 | 3.85 | 1.24x | 85 / 82 | 7.3e-06 / 9.8e-06 | 4.1e-06 / 6.4e-06 |
| 8192 | 122 | 1e+03 | 1.82 | 1.67 | 1.09x | 30 / 30 | 6.0e-06 / 6.0e-06 | 3.7e-06 / 3.8e-06 |
| 8192 | 256 | 1e+03 | 2.42 | 2.26 | 1.07x | 41 / 41 | 9.4e-06 / 9.5e-06 | 2.0e-05 / 2.0e-05 |
| 8192 | 512 | 1e+03 | 3.47 | 3.09 | 1.12x | 61 / 61 | 7.8e-06 / 7.9e-06 | 8.7e-06 / 8.7e-06 |
| 8192 | 1024 | 1e+03 | 8.27 | 7.27 | 1.14x | 76 / 76 | 7.8e-06 / 8.1e-06 | 1.1e-05 / 1.1e-05 |
| 2048 | 64 | 1e+05 | 1.63 | 1.47 | 1.11x | 33 / 33 | 3.5e-05 / 3.3e-05 | 2.3e-04 / 1.3e-04 |
| 4096 | 128 | 1e+05 | 2.21 | 1.93 | 1.14x | 54 / 54 | 1.5e-05 / 1.8e-05 | 1.1e-04 / 1.6e-04 |
| 8192 | 256 | 1e+05 | 7.99 | 2.35 | 3.39x | 54 / 54 | 1.5e-05 / 1.3e-05 | 3.2e-04 / 3.6e-04 |
| 8192 | 1024 | 1e+05 | 13.4 | 11.8 | 1.13x | 132 / 131 | 1.3e-05 / 1.2e-05 | 1.8e-05 / 9.6e-06 |

#### Plain CG, LNC=1 (ms to tol 1e-5, warm median of 9)

| n | blocks | κ | #41 | new | speedup | Chronopoulos–Gear | C–G speedup | iters (#41 / new / C–G) | true res (#41 / new / C–G) | err vs fp64 (#41 / new / C–G) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1024 | 16 | 1e+03 | 1.80 | 1.63 | 1.11x | 1.56 | 1.15x | 117 / 121 / 123 | 1.8e-05 / 1.4e-05 / 1.6e-05 | 3.7e-06 / 3.1e-06 / 1.5e-05 |
| 1024 | 32 | 1e+03 | 1.24 | 1.25 | 0.99x | 1.20 | 1.03x | 36 / 35 / 37 | 5.9e-06 / 8.4e-06 / 4.8e-06 | 3.8e-06 / 4.4e-06 / 6.8e-06 |
| 1024 | 64 | 1e+03 | 2.16 | 1.97 | 1.10x | 1.85 | 1.17x | 120 / 120 / 126 | 1.9e-05 / 1.9e-05 / 2.0e-05 | 3.5e-06 / 3.5e-06 / 1.7e-05 |
| 2048 | 32 | 1e+03 | 1.21 | 1.18 | 1.03x | 1.15 | 1.05x | 25 / 25 / 25 | 5.4e-06 / 5.5e-06 / 5.4e-06 | 1.1e-06 / 1.1e-06 / 2.6e-06 |
| 2048 | 64 | 1e+03 | 1.38 | 1.30 | 1.06x | 1.27 | 1.08x | 33 / 33 / 33 | 6.6e-06 / 6.7e-06 / 6.7e-06 | 2.7e-06 / 2.7e-06 / 5.0e-06 |
| 2048 | 128 | 1e+03 | 1.73 | 1.77 | 0.98x | 1.64 | 1.06x | 62 / 61 / 63 | 8.0e-06 / 8.9e-06 / 7.2e-06 | 4.9e-06 / 5.6e-06 / 4.7e-06 |
| 2048 | 256 | 1e+03 | 3.71 | 3.55 | 1.04x | 3.60 | 1.03x | 127 / 127 / 134 | 1.3e-05 / 1.3e-05 / 2.7e-05 | 3.5e-06 / 3.5e-06 / 1.9e-05 |
| 4096 | 64 | 1e+03 | 1.28 | 1.20 | 1.06x | 1.11 | 1.16x | 29 / 29 / 29 | 9.6e-06 / 8.8e-06 / 9.9e-06 | 4.5e-06 / 5.1e-06 / 9.7e-06 |
| 4096 | 128 | 1e+03 | 1.91 | 1.75 | 1.09x | 1.59 | 1.20x | 54 / 54 / 53 | 7.0e-06 / 8.0e-06 / 8.9e-06 | 5.4e-06 / 5.2e-06 / 6.3e-06 |
| 4096 | 256 | 1e+03 | 2.77 | 2.43 | 1.14x | 2.45 | 1.13x | 66 / 66 / 66 | 8.4e-06 / 8.9e-06 / 8.4e-06 | 5.0e-06 / 5.0e-06 / 5.2e-06 |
| 4096 | 512 | 1e+03 | 9.40 | 8.54 | 1.10x | 8.55 | 1.10x | 120 / 120 / 120 | 6.1e-06 / 6.1e-06 / 9.9e-06 | 2.9e-06 / 2.9e-06 / 4.8e-06 |
| 8192 | 122 | 1e+03 | 1.64 | 1.63 | 1.00x | 1.50 | 1.09x | 34 / 34 / 34 | 7.8e-06 / 8.6e-06 / 9.5e-06 | 3.8e-06 / 4.1e-06 / 1.1e-05 |
| 8192 | 256 | 1e+03 | 2.28 | 2.03 | 1.12x | 2.09 | 1.09x | 48 / 48 / 48 | 7.2e-06 / 7.4e-06 / 7.6e-06 | 8.1e-06 / 8.5e-06 / 8.5e-06 |
| 8192 | 512 | 1e+03 | 5.74 | 5.73 | 1.00x | 5.70 | 1.01x | 75 / 75 / 76 | 8.3e-06 / 8.3e-06 / 7.6e-06 | 1.0e-05 / 1.0e-05 / 1.0e-05 |
| 8192 | 1024 | 1e+03 | 21.4 | 20.0 | 1.07x | 20.1 | 1.06x | 115 / 115 / 116 | 9.7e-06 / 9.7e-06 / 9.2e-06 | 1.0e-05 / 1.0e-05 / 9.8e-06 |
| 2048 | 64 | 1e+05 | 1.95 | 1.45 | 1.35x | 1.33 | 1.46x | 43 / 42 / 41 | 4.1e-05 / 3.8e-05 / 1.0e-04 | 1.5e-04 / 2.5e-04 / 2.0e-04 |
| 4096 | 128 | 1e+05 | 2.60 | 2.02 | 1.29x | 2.04 | 1.28x | 67 / 67 / 92 | 1.6e-05 / 1.7e-05 / 3.8e-05 | 2.1e-05 / 5.9e-05 / 2.1e-04 |
| 8192 | 256 | 1e+05 | 3.09 | 2.49 | 1.24x | 2.40 | 1.28x | 59 / 60 / 59 | 1.6e-05 / 1.5e-05 / 2.7e-05 | 4.3e-04 / 3.7e-04 / 2.6e-04 |
| 8192 | 1024 | 1e+05 | 37.7 | 35.0 | 1.08x | 34.9 | 1.08x | 214 / 214 / 214 | 1.2e-05 / 1.2e-05 / 3.5e-05 | 1.2e-05 / 1.2e-05 / 4.1e-05 |

#### Block-Jacobi PCG, LNC=1 (ms to tol 1e-5, warm median of 9)

| n | blocks | κ | #41 | new | speedup | iters (#41 / new) | true res (#41 / new) | err vs fp64 (#41 / new) |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 1024 | 16 | 1e+03 | 1.98 | 1.80 | 1.10x | 98 / 98 | 1.9e-05 / 1.1e-05 | 6.2e-06 / 6.3e-06 |
| 1024 | 32 | 1e+03 | 1.21 | 1.19 | 1.02x | 26 / 26 | 9.5e-06 / 9.5e-06 | 5.3e-06 / 5.3e-06 |
| 1024 | 64 | 1e+03 | 2.58 | 2.26 | 1.14x | 114 / 114 | 2.4e-05 / 2.4e-05 | 5.9e-06 / 5.9e-06 |
| 2048 | 32 | 1e+03 | 1.18 | 1.17 | 1.00x | 21 / 21 | 7.2e-06 / 7.2e-06 | 2.0e-06 / 2.0e-06 |
| 2048 | 64 | 1e+03 | 1.44 | 1.23 | 1.17x | 27 / 27 | 3.6e-06 / 3.6e-06 | 2.1e-06 / 2.0e-06 |
| 2048 | 128 | 1e+03 | 1.85 | 1.82 | 1.02x | 51 / 51 | 9.8e-06 / 8.7e-06 | 4.4e-06 / 3.6e-06 |
| 2048 | 256 | 1e+03 | 5.38 | 4.92 | 1.09x | 161 / 161 | 1.4e-05 / 1.4e-05 | 4.5e-06 / 4.5e-06 |
| 4096 | 64 | 1e+03 | 1.33 | 1.24 | 1.07x | 24 / 24 | 5.8e-06 / 5.5e-06 | 2.6e-06 / 2.7e-06 |
| 4096 | 128 | 1e+03 | 2.06 | 1.82 | 1.13x | 44 / 43 | 4.9e-06 / 9.8e-06 | 3.1e-06 / 5.6e-06 |
| 4096 | 256 | 1e+03 | 2.70 | 2.35 | 1.15x | 50 / 49 | 6.1e-06 / 8.1e-06 | 4.1e-06 / 4.6e-06 |
| 4096 | 512 | 1e+03 | 7.42 | 6.78 | 1.10x | 82 / 82 | 9.0e-06 / 9.0e-06 | 6.4e-06 / 6.4e-06 |
| 8192 | 122 | 1e+03 | 1.88 | 1.61 | 1.17x | 30 / 30 | 6.0e-06 / 6.0e-06 | 4.0e-06 / 4.1e-06 |
| 8192 | 256 | 1e+03 | 2.71 | 2.44 | 1.11x | 41 / 41 | 9.7e-06 / 9.7e-06 | 2.0e-05 / 2.0e-05 |
| 8192 | 512 | 1e+03 | 6.02 | 5.55 | 1.08x | 61 / 61 | 7.9e-06 / 7.9e-06 | 8.7e-06 / 8.7e-06 |
| 8192 | 1024 | 1e+03 | 14.5 | 14.1 | 1.03x | 76 / 76 | 8.1e-06 / 8.1e-06 | 1.1e-05 / 1.1e-05 |
| 2048 | 64 | 1e+05 | 2.00 | 1.40 | 1.43x | 33 / 33 | 3.5e-05 / 3.2e-05 | 2.6e-04 / 1.7e-04 |
| 4096 | 128 | 1e+05 | 2.55 | 2.00 | 1.28x | 54 / 54 | 1.5e-05 / 1.4e-05 | 1.3e-05 / 6.4e-05 |
| 8192 | 256 | 1e+05 | 3.47 | 2.68 | 1.30x | 53 / 53 | 1.3e-05 / 1.3e-05 | 2.6e-04 / 3.0e-04 |
| 8192 | 1024 | 1e+05 | 24.1 | 23.6 | 1.02x | 131 / 131 | 1.2e-05 / 1.2e-05 | 1.5e-06 / 1.5e-06 |

### Reading the fixed-cost tables

- **The fixed cost per plain CG iteration fell from about 11 µs to about
  4.4 µs at LNC=2, and from about 8 µs to about 3.5 µs at LNC=1.** The
  biggest items were the exchanges (one per plain CG step instead of
  two, through GpSimd's SBUF-to-SBUF DMA, with no copies thanks to the
  segment layout) and the matvec tail (one PSUM copy and one transpose
  per chunk instead of four, from the spread source vectors).
- **Per iteration** the new kernel is 1.28–1.43x faster at LNC=2 split
  for plain CG and 1.17–1.34x for PCG; at LNC=1, 1.02–1.26x (plain) and
  1.03–1.32x (PCG). The matvec itself, about 270 ns per block on each of
  four concurrent column groups, now dominates from about 128 resident
  blocks per core; the row order that balances the groups takes 1–2 µs
  off at 256–512 blocks.
- **Streamed blocks.** When A does not fit in SBUF (n = 8192 with 512
  blocks on one core, 1024 blocks) the HBM stream sets the pace. There the
  kernel keeps the original row order and chunks of at most 32 rows,
  which measured faster than one large chunk (a first version with the
  balanced order and one 64-row chunk was 5–10% slower than #41 on these
  cases at LNC=1; the shipped rule is 1.02–1.13x faster).
- **Time to tolerance** changes by 0.98–1.46x (the low end is within run-to-run noise). It includes the launch and the
  host-to-device and device-to-host copies (about 0.8–1 ms), which this
  change does not touch, so the gain is largest where iterations
  dominate. Two #41 entries at n = 8192, 256 blocks, κ = 1e5, LNC=2
  (7.0 and 8.0 ms) are host-side outliers that persisted over 9 runs;
  the 3.3x there is not a kernel effect.
- **Accuracy is unchanged** for the default: iteration counts are within
  ±3 and the true residuals and errors against fp64 match #41 within
  run-to-run variation (both are fp32 with a different summation order).
- **Chronopoulos–Gear** saves about 1 µs per iteration (0.1–1.6 µs in the
  per-iteration table) but is less accurate: the true residual is up to
  2–3x and the error up to 5x those of standard CG, and at n = 4096,
  128 blocks, κ = 1e5 on one core it needed 92 iterations against 67. It
  stays opt-in.
- **PSUM allocation.** The production kernels no longer fall back to
  neuronx-cc allocation and scheduling (#41 noted the fallback); the
  per-scope PSUM tiles were worth about 0.3 µs per iteration.
