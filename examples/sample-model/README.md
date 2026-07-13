# examples/sample-model

A tiny, download-free model config to run the executorch-matrix quickstart against.

[`config.json`](config.json) resolves to the built-in **`tiny`** model — a small
convolutional classifier that exports in about a second and exercises real
convolution + linear ops (so delegation and int8/int4 quantization actually do
something). No large LLM download is required.

## Run it

```bash
# export-only (no device needed) — compares export status and .pte size
executorch-matrix compare examples/sample-model/config.json \
    --backends xnnpack,qualcomm --quantize none,int8,int4
```

This produces a terminal table, a recommendation, and
`executorch-matrix-report.json` / `.md`.

## What you'll see

A real run of the above is committed here as
[`example-report.md`](example-report.md) / [`example-report.json`](example-report.json).
On a machine without the Qualcomm QNN SDK it looks like this — note that the
xnnpack variants export with **real** sizes while the qualcomm variants are
honestly **skipped** (not estimated), and the recommendation is explicitly
labeled *provisional / export-only* because no device latency was measured:

| Variant | Export OK | Size | Export time | Latency (p50) | Notes |
|---|---|---|---|---|---|
| xnnpack / none | yes | 163.5 KB | 1.93s | not measured | |
| xnnpack / int8 | yes | 50.6 KB | 1.02s | not measured | |
| xnnpack / int4 | yes | 92.9 KB | 1.01s | not measured | int4 group size 128 |
| qualcomm / none | skip | — | — | not measured | QNN SDK not installed |

> int4 is *larger* than int8 here — that's real: on a model this small the 4-bit
> packing overhead outweighs the savings. executorch-matrix reports what actually
> happened rather than what "should" happen.

## Add real latency

To measure latency you need a connected Android device and an ExecuTorch example
runner built with the event tracer — see the repo's `CONTRIBUTING.md` for the
manual device-measurement checklist. Then:

```bash
executorch-matrix compare examples/sample-model/config.json \
    --backends xnnpack --quantize none,int8 \
    --runner /path/to/executor_runner --num-executions 50
```
