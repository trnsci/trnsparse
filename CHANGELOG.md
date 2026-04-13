# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
