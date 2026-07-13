# executorch-matrix report — tiny

- Generated: 2026-07-13T00:28:14+00:00
- Device: none connected
- **Export-only run — no device latency was measured. Sizes and export status are real; latency columns are intentionally blank.**

| Variant | Export OK | Size | Export time | Latency (p50) | Throughput | Notes |
|---|---|---|---|---|---|---|
| xnnpack / none | yes | 163.5 KB | 1.93s | not measured | — |  |
| xnnpack / int8 | yes | 50.6 KB | 1.02s | not measured | — |  |
| xnnpack / int4 | yes | 92.9 KB | 1.01s | not measured | — | int4 group size 128 |
| qualcomm / none | skip | — | — | not measured | — | qualcomm toolchain not available (ImportError: Please instal |
| qualcomm / int8 | skip | — | — | not measured | — | qualcomm toolchain not available (ImportError: Please instal |
| qualcomm / int4 | skip | — | — | not measured | — | qualcomm toolchain not available (ImportError: Please instal |

> Latency and throughput are blank because no device measurement was run. These export-only results are **not** a performance ranking.

## Recommendation

**Provisional (export-only): xnnpack / int8. Connect a device to rank balanced.**

- This was an export-only run — no latency was measured, so balanced cannot be ranked. Latency requires a real device run.
- As an export-side signal only, xnnpack / int8 is the smallest at 50.6 KB; this is a footprint hint, not a speed ranking.
- Re-run with a connected device to get a measured recommendation.
