# Contributing to executorch-matrix

Thanks for your interest! This covers local setup, the two classes of tests
(CI-automatable vs. real-hardware), and how to add a new ExecuTorch backend.

## Local setup

```bash
uv sync                       # core + dev tools (installs the ExecuTorch toolchain)
uv run pytest -m "not device" # export-only + parsing tests (no device)
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

`uv sync` pulls `executorch`, `torch`, and `torchao` — a large download the first time.

## Two classes of tests

executorch-matrix is honest about what can and cannot be verified without hardware, and the
test suite reflects that:

1. **CI-automatable (no device):** export-orchestration success/failure and metric capture
   (real exports of a tiny model), `ETDump` parsing against a real fixture binary, the
   device-runner's pure logic (device parsing, hardware matching, command building, a
   mocked-ADB happy path), catalog persistence, and recommendation logic. These run in
   ordinary CI via `pytest -m "not device"`.
2. **Real-hardware (not CI-automatable):** actual on-device deployment and latency
   measurement require a real connected target device (Android via ADB) *and* an ExecuTorch
   example runner built with the event tracer. These are marked `@pytest.mark.device` and are
   **excluded from CI**. Verify them manually with the checklist below.

### Manual verification checklist (device measurement)

Prerequisites: an Android device with USB debugging on, `adb` on your PATH, and an
`executor_runner` built for the device's ABI **with the developer tools + tracer**:

```bash
# in an ExecuTorch checkout, cross-compiled for Android:
cmake ... -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON
```

Then:

- [ ] `adb devices` shows the device as `device` (not `unauthorized`/`offline`).
- [ ] `executorch-matrix compare examples/sample-model/config.json --backends xnnpack --quantize none,int8 --runner /path/to/executor_runner` runs without ADB errors.
- [ ] Each measurable variant reports `measured` with a non-null `latency_p50_ms` and `throughput_ips`.
- [ ] A `*.etdump` file was pulled back into the `--out-dir` for each measured variant.
- [ ] The reported p50 latency is consistent with the raw ETDump (open it with
      `python -m executorch.devtools.inspector.inspector_cli --etdump_path <file>` and spot-check).
- [ ] Variants whose backend has no matching hardware (e.g. `qualcomm` on a non-Qualcomm SoC)
      are reported as **skipped with a reason**, not measured and not omitted.
- [ ] With the device unplugged, the same command falls back to an **export-only** report and
      says so — no latency is invented.

## Adding a new backend

As ExecuTorch adds hardware backends, adding support here is a one-entry change in the
registry in [`src/executorch_matrix/export/variants.py`](src/executorch_matrix/export/variants.py):

```python
BACKENDS["my_backend"] = BackendSpec(
    name="my_backend",
    partitioner_module="executorch.backends.my_backend.partition.my_partitioner",
    partitioner_class="MyPartitioner",
    target_hardware="…",
    needs_vendor_sdk=True,     # True if export needs a vendor SDK
    verified=False,            # set True once you confirm the import path against a real install
    notes="…",
)
```

The partitioner is resolved lazily, so if the import path is wrong or the SDK isn't present
the variant degrades to a clean `skipped` result rather than crashing. For measurement, add a
SoC-match rule to `_BACKEND_SOC_KEYS` (and `ANDROID_BACKENDS`) in
[`measure/device_runner.py`](src/executorch_matrix/measure/device_runner.py) if the backend
targets a specific SoC. Please mark `verified=True` only after confirming the import path
against a real ExecuTorch install, and update `docs/executorch-api-reference.md`.

## Non-negotiables

- **Never fabricate a benchmark number.** Every reported size/time traces to a real export;
  every latency to a real device run. Missing data is reported as skipped, never estimated.
- **Never blend export-only results with device-measured results** in output. The data model
  keeps them separate; keep it that way.

## Code style

`ruff` for lint + format, `mypy` for types. Run all three before opening a PR.
