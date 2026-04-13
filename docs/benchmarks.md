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

### Where NKI is expected to win (follow-up issues)

Not in raw "one-shot SpMM at N=128 vs CPU." The architectural exploits are:

- **[#20 — on-chip iterative solvers](https://github.com/trnsci/trnsparse/issues/20)**:
  iterate `A @ x` thousands of times with A SBUF-resident. Kernel launch
  amortizes over all iterations.
- **[#19 — fused screen + matmul](https://github.com/trnsci/trnsparse/issues/19)**:
  one kernel that fuses what's currently three HBM round-trips.
- **[#21 — block-sparse attention](https://github.com/trnsci/trnsparse/issues/21)**:
  large-scale sequences where the matmul work dominates dispatch.

v0.3.0 ships **BSR correctness + differentiability**. v0.4.0+ will cash
in the architectural wins. See
[`docs/architecture.md`](architecture.md) for why BSR is the
Trainium-native format.

BSR results from commit
[`411a3a7`](https://github.com/trnsci/trnsparse/commit/411a3a7).
