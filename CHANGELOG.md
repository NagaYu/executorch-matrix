# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-13

Initial release.

### Added
- **Export orchestration** (`export/variants.py`): drives ExecuTorch's own pipeline
  (`torch.export` → `torchao` quantization → `to_edge_transform_and_lower` → `to_executorch`)
  across a backend × quantization matrix. Captures, per variant, an honest outcome —
  `ok` / `failed` / `skipped` — with real export time and `.pte` size for successes. Backend
  registry with lazily-resolved partitioners (XNNPACK, Vulkan, Core ML, Qualcomm, MediaTek,
  Arm, portable); missing vendor SDKs degrade to a clean skip.
- **Quantization**: int8 via `torchao` PT2E (`XNNPACKQuantizer` + `prepare_pt2e`/`convert_pt2e`)
  and int4 via `torchao`'s 4-bit weight quantizer with an adaptive group size.
- **ETDump parsing** (`measure/etdump_parser.py`): thin adapter over `executorch.devtools.Inspector`
  producing structured latency (p50/avg/p90) and throughput. Tested against a real ETDump
  fixture built with ExecuTorch's own serializer.
- **Device measurement** (`measure/device_runner.py`): ADB deploy/run/pull around ExecuTorch's
  example runner, with device discovery, SoC-based hardware matching, and honest skips when no
  device/hardware/runner is available.
- **Local catalog + report** (`catalog/catalog.py`): plain-JSON catalog (no telemetry) and
  JSON/Markdown comparison reports that keep export-only and device-measured results
  structurally separate.
- **Reasoned recommendation** (`recommend.py`): priority-aware (`speed`/`size`/`balanced`)
  recommendation with explicit reasoning, honest about measured vs. export-only evidence.
- **CLI** (`compare`, `list-backends`, `--version`), the `examples/sample-model` quickstart
  with a committed real report, and a dated verified ExecuTorch API reference under `docs/`.
- Tooling: `uv`, `ruff`, `mypy`, `pytest`, GitHub Actions CI (lint, typecheck, export-only +
  parsing tests). Device-measurement tests are excluded from CI and documented for manual runs.

[0.1.0]: https://github.com/NagaYu/executorch-matrix/releases/tag/v0.1.0
