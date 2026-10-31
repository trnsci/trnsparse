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
