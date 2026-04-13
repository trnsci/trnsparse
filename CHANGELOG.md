# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
