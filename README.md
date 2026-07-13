# executorch-matrix

> Which of ExecuTorch's 12+ hardware backends and which quantization level actually performs best for *your* model on *your* device? This CLI runs the comparison for you — and is scrupulously honest about which results needed real hardware and which didn't.

[![CI](https://github.com/NagaYu/executorch-matrix/actions/workflows/ci.yml/badge.svg)](https://github.com/NagaYu/executorch-matrix/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/)
[![Built on ExecuTorch](https://img.shields.io/badge/built%20on-ExecuTorch%201.3-ee4c2c.svg)](https://github.com/pytorch/executorch)

## The problem

[ExecuTorch](https://github.com/pytorch/executorch) — Meta's PyTorch-native on-device
inference framework — is mature and well-tooled on its own terms. It ships quantization
via [`torchao`](https://github.com/pytorch/ao), ahead-of-time memory planning, and real
profiling tools (`ETDump`, `ETRecord`). What it doesn't hand you is an easy answer to a
very practical question:

> Given my model and my actual target devices, **which** of ExecuTorch's many hardware
> backends (XNNPACK, Vulkan, Qualcomm/QNN, MediaTek, ARM, Core ML, …) and **which**
> quantization level actually performs best?

Today that means manually exporting several variants, deploying each to a device by hand,
and comparing notes — for every new model or target device.

## What executorch-matrix does

It **automates the comparison, not the underlying export/quantization/profiling** — those
are ExecuTorch's job and it already does them well. `executorch-matrix` orchestrates:

1. **Export** a model through several backend × quantization combinations using ExecuTorch's
   own APIs (`torch.export` → `torchao` quantization → backend delegation → `.pte`).
2. **Deploy and run** each variant on a connected target device using ExecuTorch's own
   example runners (via ADB for Android).
3. **Parse** the resulting `ETDump` profiling output into structured metrics.
4. **Report** a comparison with a reasoned recommendation.

```bash
executorch-matrix compare llama-3.2-1b --backends xnnpack,vulkan,qualcomm --quantize int8,int4
```

```
Variant                          Export OK   Size     Latency (p50)   Notes
xnnpack / int8                     yes       612 MB     340 ms         CPU baseline
vulkan / int8                       yes       598 MB     210 ms
qualcomm / int8                      yes       590 MB     140 ms         fastest on this device
qualcomm / int4                       yes       340 MB     125 ms         smallest + fastest

Recommendation: qualcomm / int4 for this device — best latency and smallest footprint.
Full report: executorch-matrix-report.json / .md
```

*(The latency column above requires a connected device. Without one you still get the
Export/Size columns — see below.)*

## What needs a connected device — and what doesn't

executorch-matrix never blends these two kinds of results:

| Result | Needs a device? |
|--------|-----------------|
| Export success / failure per backend × quantization | **No** |
| Output `.pte` file size | **No** |
| Export time | **No** |
| Latency / throughput (`ETDump`) | **Yes** — a real, connected target device |

If no device is connected — or a specific backend has no matching hardware, or its vendor
SDK isn't installed — that variant's measurement is **skipped and reported as such**, never
estimated or silently omitted. **No number in a report is ever fabricated:** every size and
time comes from a real export, and every latency from a real on-device run.

Here's a real **export-only** run (no device) against the bundled example, on a machine
without the Qualcomm SDK — successes are real, skips are honest, and the recommendation is
explicitly provisional:

```
| Variant         | Export OK | Size     | Export time | Latency (p50) | Notes                 |
| xnnpack / none  | yes       | 163.5 KB | 1.93s       | not measured  |                       |
| xnnpack / int8  | yes       | 50.6 KB  | 1.02s       | not measured  |                       |
| xnnpack / int4  | yes       | 92.9 KB  | 1.01s       | not measured  | int4 group size 128   |
| qualcomm / int8 | skip      | —        | —           | not measured  | QNN SDK not installed |

Recommendation: Provisional (export-only): xnnpack / int8. Connect a device to rank balanced.
```

See [`examples/sample-model/`](examples/sample-model/) for the full committed report.

## Install

```bash
# with uv (recommended)
uv add executorch-matrix

# or pip
pip install executorch-matrix
```

This installs the ExecuTorch toolchain (`executorch` 1.3.x, `torch`, `torchao`) as
dependencies, since orchestrating them is the whole point. Supported on **Python 3.10–3.13**,
on Linux (x86_64/ARM64), macOS (Apple silicon), and Windows — the same platforms ExecuTorch
itself supports.

## Quickstart

```bash
# 1. Export-only comparison — works anywhere, no device:
executorch-matrix compare examples/sample-model/config.json --backends xnnpack --quantize none,int8,int4

# 2. See the backends this build knows about:
executorch-matrix list-backends

# 3. Add real latency (needs an Android device + an ExecuTorch runner built with the tracer):
executorch-matrix compare examples/sample-model/config.json \
    --backends xnnpack --quantize none,int8 \
    --runner /path/to/executor_runner --num-executions 50
```

Every run writes `executorch-matrix-report.json` / `.md` and appends to a local, plain-JSON
catalog (`~/.executorch-matrix/catalog.json` by default — no telemetry, no hosted service).

## How it works

| Stage | Module | ExecuTorch API it drives |
|---|---|---|
| Export matrix | [`export/variants.py`](src/executorch_matrix/export/variants.py) | `torch.export` → `to_edge_transform_and_lower` → `to_executorch` |
| Quantization | same | `torchao` PT2E (int8) and 4-bit weight (int4) |
| Device run | [`measure/device_runner.py`](src/executorch_matrix/measure/device_runner.py) | ADB + ExecuTorch example runner + `ETDump` |
| Parse profiling | [`measure/etdump_parser.py`](src/executorch_matrix/measure/etdump_parser.py) | `executorch.devtools.Inspector` |
| Catalog + report | [`catalog/catalog.py`](src/executorch_matrix/catalog/catalog.py) | — (local JSON) |
| Recommendation | [`recommend.py`](src/executorch_matrix/recommend.py) | — (priority-aware reasoning) |

The exact, dated ExecuTorch API this was built against is documented in
[`docs/executorch-api-reference.md`](docs/executorch-api-reference.md).

## Relationship to ExecuTorch (credit where due)

This project is a thin **orchestrator on top of ExecuTorch**. It deliberately does **not**
duplicate ExecuTorch's own `torchao`-based quantization or its `ETDump`/`ETRecord` profiling
tooling — it calls them. All the hard on-device work is ExecuTorch's; the value here is
running the backend × quantization matrix and comparing the results honestly.

## Relationship to `silicon-forge`

[`silicon-forge`](https://github.com/NagaYu/silicon-forge) (by the same author) does a
related job for Apple's MLX / Core AI stack, but builds its **own** conversion pipeline
because that ecosystem's tooling is thinner. `executorch-matrix` takes a deliberately
different shape: because ExecuTorch's own tooling is already mature, this project
orchestrates it rather than reimplementing it. Same author, different design — for a
principled reason.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — including how to add a backend as ExecuTorch adds
them, and the manual checklist for the device-measurement path that can't run in CI.

## License

MIT — see [LICENSE](LICENSE).
