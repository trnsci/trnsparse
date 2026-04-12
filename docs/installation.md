# Installation

## From PyPI (once published)

```bash
pip install trnsparse
pip install trnsparse[neuron]   # on Neuron hardware
```

## From source

```bash
git clone git@github.com:trnsci/trnsparse.git
cd trnsparse
pip install -e ".[dev]"
pytest tests/ -v
```

## Hardware compatibility

NKI kernels target **Neuron SDK 2.24+** on the **Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)** AMI. Without Neuron hardware, trnsparse falls through to PyTorch ops on CPU/GPU — all APIs remain functional.
