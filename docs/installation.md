# Installation

## From PyPI (once published)

```bash
pip install trnsparse

# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trnsparse[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trnsparse[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

`[torchneuron]` pins `torch>=2.14`, `torch-neuronx>=2.14`, and `nki>=0.6`
(torch-neuronx 2.14 requires Python ≥ 3.12). `[neuron]` is the XLA lineage
(`nki>=0.3.0`, `neuronxcc>=2.24`, `torch-neuronx>=2.9`). `nki` and
`torch-neuronx` are not on public PyPI; they ship with the Deep Learning
AMI Neuron or install from `https://pip.repos.neuron.amazonaws.com`.

## Two Neuron PyTorch lineages

| Lineage | Instances | Stack | Kernel dispatch |
|---|---|---|---|
| TorchNeuron (native) | Trn2, Trn3 | Neuron SDK 2.33+, Python ≥ 3.12, `torch` 2.14, `torch-neuronx` 2.14.* | Each kernel is a PyTorch custom op registered with `torch_neuronx.nki_op` (`torch.ops.trnsparse.spmm_dense`, `bsr_spmm`, `screened_spmm`, `attn_stats`, `attn_out`, `attn_bwd_dq`, `attn_bwd_dkdv`), plus the composite `trnsparse::bsr_attn` / `bsr_attn_bwd`; tensors live on `device="neuron"`. Backwards are attached with `register_autograd`. |
| PyTorch/XLA | Trn1 (also Inf2) | `torch-neuronx` 2.9 lineage, `torch_xla` | Unchanged: kernels run on the `torch_xla` device; autograd via `torch.autograd.Function` wrappers. |

Selection is automatic, on the first kernel launch:

1. **native** — `torch_neuronx` exports `nki_op` and a `neuron` device is
   available (`torch_neuronx.device_count() > 0`). This path never imports
   `torch_xla`.
2. **xla** — `torch_xla` is installed.
3. otherwise there is no device path; kernel launches fall back to
   PyTorch (or raise under `TRNSPARSE_REQUIRE_NKI=1`), as in earlier
   releases. `TRNSPARSE_USE_SIMULATOR=1` still routes through
   `nki.simulate`.

The `trnsparse::` custom ops are registered only when the native path is
selected, so importing trnsparse on CPU or on the XLA lineage has no
TorchNeuron side effects.

To force a path, set `TRNSCI_NKI_DISPATCH` before the first launch. The
variable is shared by every trnsci library:

| Value | Effect |
|---|---|
| `auto` (default) | The order above. |
| `native` | Use TorchNeuron; raise `RuntimeError` if no `neuron` device is available. |
| `xla` | Use torch_xla even if TorchNeuron is present; raise `RuntimeError` if torch_xla is missing. |
| `cpu` | No device path (PyTorch fallback / simulator). |

`TRNSCI_XLA_DEPRECATION=1` makes the XLA path emit a one-time
`DeprecationWarning`. It is off by default.

## Placement: sparse structure on the host

The TorchNeuron `neuron` device has no sparse CSR tensors
([aws-neuron-sdk#1442](https://github.com/aws-neuron/aws-neuron-sdk/issues/1442)):
`torch.sparse_csr_tensor` cannot be built on it or densified there. On the
native path trnsparse therefore never puts sparse structure on the device:

| Object | Where it lives | What moves to `neuron` |
|---|---|---|
| `CSRMatrix` (`values`, `col_indices`, `row_ptrs`) | host | the densified (padded) A tile, built on the host without a torch sparse tensor |
| `BSRMatrix.block_row_ptrs`, `.block_col_indices` | host | nothing; only the small gather index grids derived from them |
| `BSRMatrix.blocks` | host or `neuron` | dense blocks, gathered into kernel tiles on the device |
| attention mask (`BSRMatrix`) structure | host | nothing; Q, K, V are gathered on the device |
| dense operands (`B`, `x`, Q/K/V, screened `A`) | host or `neuron` | themselves |

Results come back on the dense operand's device (B's for `spmm` /
`bsr_spmm` / `screened_spmm`, x's for `spmv`, Q's for attention).
Index tensors found on a device are copied to the host on each call, which
works but costs a device sync; build them on the host.

```python
A = trnsparse.from_dense(D)  # host CSR
B = torch.randn(n, 64, device="neuron")  # device operand
C = trnsparse.spmm(A, B)  # C on neuron; differentiable in A.values and B
C = torch.compile(lambda B: trnsparse.spmm(A, B), backend="neuron")(B)
```

`spmv` / `spmm` against a device operand densify A on the host each call.
For repeated products with the same matrix (iterative solvers), use
`BSRMatrix` with `blocks` on the device.

## Editable installs and the NKI kernel cache

TorchNeuron enables the NKI trace cache by default
(`NKI_ENABLE_TRACE_CACHE=1`), which persists compiled kernels across
processes. The cache only versions source that belongs to an installed
wheel (a `site-packages` tree with `.dist-info`); an editable install is
not tracked and can get a stale kernel after you edit `kernels.py`. When
developing with `pip install -e`, turn the cache off before starting
Python (the test suite's `conftest.py` does this):

```bash
export NKI_ENABLE_TRACE_CACHE=0
```

See the TorchNeuron
[environment variables](https://github.com/aws-neuron/torch-neuronx/blob/main/docs/environment_variables.md)
reference.

## From source

```bash
git clone git@github.com:trnsci/trnsparse.git
cd trnsparse
pip install -e ".[dev]"
pytest tests/ -v
```

## Hardware compatibility

NKI kernels target **Neuron SDK 2.24+** on the **Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)** AMI (XLA lineage), and Neuron SDK 2.33+ with TorchNeuron on Trn2/Trn3 (native lineage). Without Neuron hardware, trnsparse falls through to PyTorch ops on CPU/GPU — all APIs remain functional.
