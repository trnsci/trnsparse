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
  of `nc_matmul` calls are skipped. Current example still materializes the full
  score matrix (follow-up: [#25](https://github.com/trnsci/trnsparse/issues/25)).
- **[#22 — on-chip iterative solvers](https://github.com/trnsci/trnsparse/issues/22)**:
  CG/power-iteration with A SBUF-resident across all iterations (replaces #20).
  v0.3.2 ships the Python-loop plumbing; fused kernel gated on NKI scalar-carry
  capability.

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
