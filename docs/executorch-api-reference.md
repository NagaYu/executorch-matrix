# ExecuTorch API Reference — verified for executorch-matrix (Phase 1)

**Verified:** 2026-07-12 against official ExecuTorch docs (stable = **1.3**, latest patch **1.3.1**)
and the torchao docs. This is the authoritative reference the rest of the codebase builds
against. Confidence is marked per item. Where a claim is `LOW`, the tool must degrade
gracefully (catch `ImportError` / report the real error) rather than assume the API.

Source of every claim is linked inline.

---

## 1. Version, install, platform support — `HIGH`

- Current stable release: **ExecuTorch 1.3** (`release/1.3`); latest patch **1.3.1**.
- Install: `pip install executorch` (pulls `torch`; `torchao` and `coremltools` come with it).
  Verified locally: `executorch==1.3.1` resolves with wheels on macOS **arm64 / Python 3.13**.
- **Python 3.10 – 3.13** supported.
- Host OS: **Linux** (x86_64 / ARM64), **macOS (ARM64)** — Intel macOS must build PyTorch from
  source; **Windows** (x86_64, must run inside a Visual Studio Developer PowerShell).
- Sources: [getting-started](https://docs.pytorch.org/executorch/stable/getting-started.html),
  [releases](https://github.com/pytorch/executorch/releases).

**Implication:** `executorch`, `torch`, `torchao` are hard runtime dependencies (matches the
brief). The CLI still lazy-imports them so `catalog`/`recommend`/report reading don't pay torch
import cost or require a successful export toolchain.

---

## 2. Canonical export recipe (nn.Module → .pte) — `HIGH`

```python
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

exported_program = export(model, sample_inputs)          # sample_inputs is a tuple
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],                   # [] = no delegation (portable CPU)
).to_executorch()

with open("model.pte", "wb") as f:
    f.write(executorch_program.buffer)                    # bytes of the serialized program
```

- `.to_executorch(config=ExecutorchBackendConfig(...))` accepts an optional backend config.
- Program-data separation (`.pte` + `.ptd` external weights):
  `executorch_program.write_tensor_data_to_file(output_dir)`.
- Source: [using-executorch-export](https://docs.pytorch.org/executorch/stable/using-executorch-export.html).

**Implication for `export/variants.py`:** the per-variant unit of work is
`export → (optional quantize) → to_edge_transform_and_lower(partitioner=[P]) → to_executorch →
write buffer`. Capture: success/failure + error string, export wall-time, and `.pte` size in
bytes. All three are available **without any device**.

---

## 3. Backends — `HIGH` for the list, `MEDIUM/LOW` for exact partitioner import paths

ExecuTorch ships **14** backends (the brief's "12+"). The backend is selected by the
`partitioner=[...]` argument to `to_edge_transform_and_lower`.

| Backend | Partitioner class | Import path | Target | Vendor SDK to export | Export w/o target HW? |
|---|---|---|---|---|---|
| XNNPACK | `XnnpackPartitioner` | `executorch.backends.xnnpack.partition.xnnpack_partitioner` | CPU (Arm/x86) | no | **yes** ✅ verified |
| Vulkan | `VulkanPartitioner` | `executorch.backends.vulkan.partitioner.vulkan_partitioner` | Android/desktop GPU | no | **yes** ✅ verified |
| Core ML | `CoreMLPartitioner` | `executorch.backends.apple.coreml.partition` | Apple NPU/GPU/CPU | macOS + coremltools | macOS only ✅ verified |
| MPS | `MPSPartitioner` | `executorch.backends.apple.mps.partition` | Apple GPU | macOS | macOS (deprecated) ⚠️ |
| Qualcomm (QNN) | `QnnPartitioner` | `executorch.backends.qualcomm.partition.qnn_partitioner` | Qualcomm NPU | **QNN SDK** | needs QNN SDK ⚠️ inferred |
| MediaTek | `NeuropilotPartitioner` | `executorch.backends.mediatek.partition...` | MediaTek NPU | **NeuroPilot SDK** | needs SDK ⚠️ inferred |
| Arm Ethos-U | `EthosUPartitioner` | `executorch.backends.arm...` | Arm MCU NPU | Arm toolchain | ⚠️ inferred |
| Arm Cortex-M | (Cortex-M) | `executorch.backends.arm...` | Arm MCU CPU | Arm toolchain | ⚠️ inferred |
| Arm VGF | (VGF) | `executorch.backends.arm...` | Arm GPU | Arm toolchain | ⚠️ inferred |
| OpenVINO | (OpenVINO) | `executorch.backends.openvino...` | Intel CPU/GPU/NPU | OpenVINO | ⚠️ inferred |
| NXP | (NXP) | `executorch.backends.nxp...` | NXP NPU | NXP SDK | ⚠️ inferred |
| Cadence | (Cadence) | `executorch.backends.cadence...` | DSP | Cadence tools | ⚠️ inferred |
| Samsung Exynos | (Exynos) | `executorch.backends.samsung...` | Exynos NPU/GPU | Exynos SDK | ⚠️ inferred |
| CUDA (experimental) | — | — | NVIDIA GPU | CUDA | ⚠️ experimental |

Verified import paths: XNNPACK, Vulkan, Core ML. The rest follow the pattern
`executorch.backends.<name>.partition.<name>_partitioner.<Name>Partitioner` but the **exact
class/module names are not yet individually confirmed** and are marked LOW.
Sources: [backends-overview](https://github.com/pytorch/executorch/blob/main/docs/source/backends-overview.md),
[coreml](https://docs.pytorch.org/executorch/stable/backends-coreml.html),
[vulkan](https://docs.pytorch.org/executorch/stable/android-vulkan.html).

**Implication for `export/variants.py`:** use a **backend registry** mapping a backend name →
`(module_path, partitioner_class_name)`, resolved **lazily** via `importlib`. If the import
fails (SDK/toolchain not installed), report the variant as `export skipped: <backend> toolchain
not installed` — distinct from an actual export failure. This keeps the tool honest and means
unverified import paths degrade to a clear message instead of a crash. Only the paths verified
here (XNNPACK/Vulkan/CoreML) are trusted; the others are confirmed against a real install in
Phase 2.

---

## 4. Quantization (torchao / PT2E) — `HIGH` for int8 flow, `MEDIUM` for int4

int8 goes through the **PT2E** flow with a backend quantizer; int4 goes through **torchao**
weight-only configs. Quantization happens **before** `to_edge_transform_and_lower`.

### int8 (PT2E, e.g. XNNPACK)
```python
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

m = export(model, sample_inputs).module()                 # capture a graph module
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
m = prepare_pt2e(m, quantizer)
m(*sample_inputs)                                          # calibrate on representative inputs
m = convert_pt2e(m)

exported = export(m, sample_inputs)                        # re-export the quantized module
et = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
```
- **Note:** `prepare_pt2e` / `convert_pt2e` now come from
  `torchao.quantization.pt2e.quantize_pt2e` (not `torch.ao...`) in current ExecuTorch.
- Backend quantizers: `XNNPACKQuantizer`, `CoreMLQuantizer` (per-backend modules).

### int4 (torchao weight-only)
```python
from torchao.quantization.quant_api import quantize_, Int4WeightOnlyConfig
quantize_(model, Int4WeightOnlyConfig())                  # source transform, before export
# ...then export + to_edge_transform_and_lower as usual.
```
- Also available: `Int8DynActInt4WeightQuantizer` (source-transform 4-bit) from
  `torchao.quantization.quant_api`.
- Sources: [quantization-overview](https://docs.pytorch.org/executorch/stable/quantization-overview.html),
  [xnnpack-quantization](https://docs.pytorch.org/executorch/1.1/backends/xnnpack/xnnpack-quantization.html),
  [torchao.quantize_](https://docs.pytorch.org/ao/stable/generated/torchao.quantization.quantize_.html).

**Implication:** a small **quantization registry** keyed by `("int8"|"int4", backend)` selecting
the right quantizer/config, applied in the pipeline between `export(...).module()` and the final
export. Exact int4 config names confirmed against a real install in Phase 2.

---

## 5. ETRecord + ETDump + Inspector (profiling) — `HIGH`

### Generate ETRecord at export time
```python
import copy
from executorch.devtools import generate_etrecord

edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
edge_copy = copy.deepcopy(edge)          # deepcopy BEFORE to_executorch (it mutates in place)
et = edge.to_executorch()
generate_etrecord("etrecord.bin", edge_copy, et)   # (path, EdgeProgramManager, ExecutorchProgramManager)
```

### Generate ETDump at runtime (on device or host runner)
- Build the runner with devtools + tracer:
  `-DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON`.
- Runner instantiates `executorch::etdump::ETDumpGen` and passes it as the `event_tracer` to
  `load_method`; after execution it writes `get_etdump_data()` to the path in `--etdump_path`.
- Inputs are typically supplied via a **BundledProgram**
  (`executorch.devtools.BundledProgram`, `MethodTestCase`/`MethodTestSuite`).

### Parse ETDump in Python (drives `etdump_parser.py`)
```python
from executorch.devtools import Inspector
from executorch.devtools.inspector import TimeScale   # source/target time scale enums

insp = Inspector(
    etdump_path="model.etdump",
    etrecord="etrecord.bin",              # optional; adds source-code linkage
    source_time_scale=TimeScale.NS,
    target_time_scale=TimeScale.MS,       # durations reported in ms by default
)

for block in insp.event_blocks:           # list[EventBlock]; block.name, block.events
    df = block.to_dataframe()             # pandas DataFrame of events
    for ev in block.events:               # Event: .name, .op_types, .is_delegated_op, .perf_data
        stats = ev.perf_data              # .p10 .p50 .p90 .avg .min .max  (None for non-profiled)
insp.print_data_tabular()
total_ms = insp.find_total_for_module(module_name)   # aggregate compute time for a module
```

- `Inspector.__init__(self, etdump_path=None, etdump_data=None, etrecord=None,
  source_time_scale=TimeScale.NS, target_time_scale=TimeScale.MS, debug_buffer_path=None,
  delegate_metadata_parser=None, delegate_time_scale_converter=None,
  enable_module_hierarchy=False, reference_graph_name='edge_dialect_graph_module')`.
- **Units:** standard profiling events use `target_time_scale` (default **ms**); delegate
  profiling events are in **cycles** (need a `delegate_time_scale_converter` for real time).
- CLI equivalent: `python3 -m executorch.devtools.inspector.inspector_cli
  --etdump_path <p> --etrecord_path <p>`.
- Sources: [model-inspector](https://docs.pytorch.org/executorch/stable/model-inspector.html),
  [etdump](https://docs.pytorch.org/executorch/stable/etdump.html),
  [devtools-integration-tutorial](https://docs.pytorch.org/executorch/stable/tutorials/devtools-integration-tutorial.html).

**Implication for `etdump_parser.py`:** wrap `Inspector`, iterate `event_blocks` → `events`,
pull `perf_data` (p50/avg/p90 etc.), identify the top-level execute/forward event for total
latency, and derive throughput = 1000 / latency_ms (inferences/s). Keep the mapping explicit and
carry the **units** and **profiled-event count** through so the report can't imply false
precision. Fixtures = a **real** ETDump captured on host (see §7), never hand-written numbers.

---

## 6. Device run via ADB (Android) — `MEDIUM` (validated fully only on real hardware, Phase 4)

```bash
adb push executor_runner /data/local/tmp/
adb push model.pte       /data/local/tmp/
adb shell /data/local/tmp/executor_runner \
    --model_path=/data/local/tmp/model.pte \
    --num_executions=50 \
    --etdump_path=/data/local/tmp/model.etdump
adb pull /data/local/tmp/model.etdump ./
# then: Inspector(etdump_path="model.etdump", etrecord="etrecord.bin")
```
- `--num_executions` gives multiple runs so `perf_data` percentiles (p50/p90) are meaningful.
- Sources: [using-executorch-android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html),
  [vulkan profiling tutorial](https://docs.pytorch.org/executorch/1.0/backends/vulkan/tutorials/etvk-profiling-tutorial.html).

**Implication for `device_runner.py`:** shell out to `adb` (devices / push / shell / pull),
detect connected devices, map a backend to whether matching hardware is present, and **skip +
report** when it isn't. The runner binary must be built with devtools — the tool checks for it
and instructs the user rather than fabricating a measurement. This is the part that genuinely
needs real hardware; verified end-to-end only in Phase 4 on a device.

---

## 7. Models — `HIGH`

- Official getting-started model: **MobileNetV2** from torchvision,
  `models.mobilenet_v2(weights=...).eval()`, input `torch.randn(1, 3, 224, 224)`.
- **CI test model (chosen):** a **tiny hand-built `nn.Module`** (a couple of conv/linear + add/mul
  ops) with a small fixed input. Rationale: no torchvision download, exports in well under a
  second, exercises real delegation/quantization paths. MobileNetV2 is offered as a named
  example for a more realistic (still CPU-only) run.
- Llama 3.2 1B: exported via ExecuTorch's own `examples/models/llama` (`export_llama`) flow;
  executorch-matrix treats large models as a named model spec, not a reimplementation.

---

## 8. Open questions to confirm against a real install in Phase 2

- Exact partitioner class/module names for Qualcomm, MediaTek, Arm (Ethos-U/Cortex-M/VGF),
  OpenVINO, NXP, Cadence, Samsung (only XNNPACK/Vulkan/CoreML verified from docs).
- Exact current int4 config class name (`Int4WeightOnlyConfig` vs `Int8DynActInt4WeightQuantizer`)
  and which is the recommended executorch path.
- Whether a **host** `executor_runner`/pybindings path can emit an ETDump without a device
  (to generate the parser fixture on CI hardware); fallback is to commit a real captured ETDump
  + a regeneration script.
- `TimeScale` import location (`executorch.devtools.inspector` vs `...inspector._inspector`).

All of these are resolved by installing the toolchain (confirmed resolvable) and running a real
export at the start of Phase 2 — none block the design, and every one degrades to an honest
error rather than a fabricated result.
