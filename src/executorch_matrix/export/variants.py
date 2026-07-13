"""Export orchestration across backend x quantization combinations.

For each requested (backend, quantization) pair this drives ExecuTorch's *own*
export pipeline — ``torch.export`` -> (optional ``torchao`` quantization) ->
``to_edge_transform_and_lower`` with the backend's partitioner -> ``to_executorch``
-> ``.pte`` — and records, honestly, one of three outcomes:

* ``ok``      — exported; ``export_seconds`` and ``pte_bytes`` are real measurements.
* ``failed``  — the export was attempted but raised; ``error`` holds the reason.
* ``skipped`` — the export could not be attempted at all (the backend's toolchain
  is not installed, or the quantization does not apply to this model). ``skipped_reason``
  explains why. A skip is never presented as a success.

Nothing here fabricates a number: ``export_seconds`` is wall-clock around the real
export call and ``pte_bytes`` is the length of the real serialized buffer. Latency
is *not* set here — that requires a device (see ``measure/``).

Heavy imports (``torch``/``executorch``/``torchao``) are done lazily inside the
functions so importing this module — e.g. to read the backend registry — is cheap.
"""

from __future__ import annotations

import copy
import importlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from executorch_matrix.export.models import ModelSpec

# Quantization levels understood by the orchestrator.
QUANTIZATIONS: tuple[str, ...] = ("none", "int8", "int4")


@dataclass(frozen=True)
class BackendSpec:
    """A hardware backend and how to reach its ExecuTorch partitioner.

    ``partitioner_class is None`` means "no delegation" (the portable CPU path).
    ``verified`` records whether the import path was confirmed against a real
    ExecuTorch install (2026-07, executorch 1.3.1); unverified paths still work
    if correct and otherwise degrade to a clean ``skipped`` result.
    """

    name: str
    partitioner_module: str | None
    partitioner_class: str | None
    target_hardware: str
    needs_vendor_sdk: bool
    verified: bool
    notes: str = ""


# Registry of backends. XNNPACK/Vulkan/Core ML import paths are verified against
# executorch 1.3.1; Qualcomm/MediaTek modules exist but require vendor SDKs (they
# raise on import without them, which we report as "skipped"); the remainder are
# best-effort paths marked unverified and are trivially correctable here as
# ExecuTorch stabilizes them. This is the single place to add a new backend.
BACKENDS: dict[str, BackendSpec] = {
    "portable": BackendSpec(
        "portable",
        None,
        None,
        "CPU (portable reference kernels)",
        False,
        True,
        "No delegation; the baseline every device can run.",
    ),
    "xnnpack": BackendSpec(
        "xnnpack",
        "executorch.backends.xnnpack.partition.xnnpack_partitioner",
        "XnnpackPartitioner",
        "CPU (Arm & x86, SIMD/threads)",
        False,
        True,
        "The general-purpose CPU baseline.",
    ),
    "vulkan": BackendSpec(
        "vulkan",
        "executorch.backends.vulkan.partitioner.vulkan_partitioner",
        "VulkanPartitioner",
        "GPU (Android; desktop experimental)",
        False,
        True,
    ),
    "coreml": BackendSpec(
        "coreml",
        "executorch.backends.apple.coreml.partition",
        "CoreMLPartitioner",
        "Apple NPU/GPU/CPU (iOS/macOS)",
        False,
        True,
        "Export requires macOS + coremltools.",
    ),
    "qualcomm": BackendSpec(
        "qualcomm",
        "executorch.backends.qualcomm.partition.qnn_partitioner",
        "QnnPartitioner",
        "Qualcomm NPU (Hexagon)",
        True,
        False,
        "Export needs the Qualcomm QNN SDK; otherwise skipped.",
    ),
    "mediatek": BackendSpec(
        "mediatek",
        "executorch.backends.mediatek.partition.mediatek_partitioner",
        "NeuropilotPartitioner",
        "MediaTek NPU (APU)",
        True,
        False,
        "Export needs the MediaTek NeuroPilot SDK; otherwise skipped.",
    ),
    "arm": BackendSpec(
        "arm",
        "executorch.backends.arm.ethosu_partitioner",
        "EthosUPartitioner",
        "Arm Ethos-U NPU (embedded)",
        True,
        False,
        "Path unverified; requires the Arm toolchain.",
    ),
}


@dataclass
class ExportResult:
    """The outcome of one (backend, quantization) export attempt.

    Only ``ok`` results carry ``export_seconds``/``pte_bytes``; those are the real
    export-side metrics that need no device. Latency lives in the measurement
    layer and is intentionally absent here.
    """

    backend: str
    quantization: str
    status: str  # "ok" | "failed" | "skipped"
    error: str | None = None
    skipped_reason: str | None = None
    export_seconds: float | None = None
    pte_bytes: int | None = None
    pte_path: str | None = None
    etrecord_path: str | None = None
    int4_group_size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def variant(self) -> str:
        return f"{self.backend} / {self.quantization}"

    @property
    def export_ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _short(exc: Exception, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _resolve_partitioner(spec: BackendSpec) -> Any:
    module = importlib.import_module(spec.partitioner_module)  # type: ignore[arg-type]
    return getattr(module, spec.partitioner_class)  # type: ignore[arg-type]


def _pick_int4_group_size(model: Any, candidates: Sequence[int] = (256, 128, 64, 32)) -> int | None:
    """Largest group size that divides every Linear's in_features, or None.

    torchao's 4-bit weight quantizer asserts ``in_features % group_size == 0`` for
    each Linear, so a model with no common divisor simply has no applicable int4
    variant — which we report as ``skipped`` rather than forcing.
    """
    import torch.nn as nn

    in_features = [m.in_features for m in model.modules() if isinstance(m, nn.Linear)]
    if not in_features:
        return None
    for gs in candidates:
        if all(f % gs == 0 for f in in_features):
            return gs
    return None


def _quantize_int8(model: Any, example_inputs: tuple[Any, ...]) -> Any:
    """PT2E symmetric int8 (the generic CPU quantizer; a per-backend extension point)."""
    import torch
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    captured = torch.export.export(model, example_inputs).module()
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True)
    )
    prepared = prepare_pt2e(captured, quantizer)  # type: ignore[arg-type]
    prepared(*example_inputs)  # calibrate on the example input
    return convert_pt2e(prepared)


def _quantize_int4(model: Any) -> tuple[Any | None, int | None]:
    """torchao 4-bit weight / 8-bit dynamic activation source transform.

    Returns ``(quantized_model, group_size)`` or ``(None, None)`` if int4 does not
    apply to this model.
    """
    from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

    group_size = _pick_int4_group_size(model)
    if group_size is None:
        return None, None
    quantized = Int8DynActInt4WeightQuantizer(groupsize=group_size).quantize(copy.deepcopy(model))
    return quantized, group_size


def export_variant(
    model_spec: ModelSpec,
    backend: str,
    quantization: str,
    out_dir: str | Path,
    *,
    with_etrecord: bool = True,
) -> ExportResult:
    """Export one (backend, quantization) variant and record the outcome."""
    import torch
    from executorch.exir import to_edge_transform_and_lower

    result = ExportResult(backend=backend, quantization=quantization, status="skipped")

    spec = BACKENDS.get(backend)
    if spec is None:
        result.skipped_reason = f"unknown backend {backend!r}; known: {', '.join(sorted(BACKENDS))}"
        return result
    if quantization not in QUANTIZATIONS:
        result.skipped_reason = (
            f"unknown quantization {quantization!r}; known: {', '.join(QUANTIZATIONS)}"
        )
        return result

    # Resolve the partitioner (a missing vendor SDK surfaces here as a clean skip).
    partitioners: list[Any] = []
    if spec.partitioner_class is not None:
        try:
            partitioner_cls = _resolve_partitioner(spec)
            partitioners = [partitioner_cls()]
        except Exception as exc:  # noqa: BLE001 - any import/instantiation failure => skip
            result.skipped_reason = f"{backend} toolchain not available ({_short(exc, 160)})"
            return result

    model, example_inputs = model_spec.build()

    # Quantize (each variant gets a fresh model; quantization mutates in place).
    try:
        if quantization == "none":
            prepared_model: Any = model
        elif quantization == "int8":
            prepared_model = _quantize_int8(model, example_inputs)
        else:  # int4
            prepared_model, group_size = _quantize_int4(model)
            if prepared_model is None:
                result.skipped_reason = (
                    "int4 not applicable: no Linear in_features divisible by a "
                    "supported group size (256/128/64/32)"
                )
                return result
            result.int4_group_size = group_size
    except Exception as exc:  # noqa: BLE001 - quantization failure is a real, reportable outcome
        result.status = "failed"
        result.error = f"quantization({quantization}): {_short(exc)}"
        return result

    # Export + lower + serialize; time only the real export work.
    start = time.perf_counter()
    try:
        exported = torch.export.export(prepared_model, example_inputs)
        edge = to_edge_transform_and_lower(exported, partitioner=partitioners)
        edge_for_etrecord = copy.deepcopy(edge) if with_etrecord else None
        executorch_program = edge.to_executorch()
        buffer = bytes(executorch_program.buffer)
    except Exception as exc:  # noqa: BLE001 - export failure is a real, reportable outcome
        result.status = "failed"
        result.export_seconds = round(time.perf_counter() - start, 4)
        result.error = f"export/lower: {_short(exc)}"
        return result
    result.export_seconds = round(time.perf_counter() - start, 4)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pte_path = out_path / f"{backend}_{quantization}.pte"
    pte_path.write_bytes(buffer)
    result.pte_bytes = len(buffer)
    result.pte_path = str(pte_path)

    # ETRecord links on-device profiling data back to the source graph; best-effort.
    if with_etrecord and edge_for_etrecord is not None:
        try:
            from executorch.devtools import generate_etrecord

            etrecord_path = out_path / f"{backend}_{quantization}.etrecord"
            generate_etrecord(str(etrecord_path), edge_for_etrecord, executorch_program)
            result.etrecord_path = str(etrecord_path)
        except Exception:  # noqa: BLE001 - ETRecord is optional; its absence is not a failure
            result.etrecord_path = None

    result.status = "ok"
    return result


def export_matrix(
    model_spec: ModelSpec,
    backends: Iterable[str],
    quantizations: Iterable[str],
    out_dir: str | Path,
    *,
    with_etrecord: bool = True,
) -> list[ExportResult]:
    """Export every (backend x quantization) combination and return all results."""
    results: list[ExportResult] = []
    for backend in backends:
        for quantization in quantizations:
            results.append(
                export_variant(
                    model_spec,
                    backend,
                    quantization,
                    out_dir,
                    with_etrecord=with_etrecord,
                )
            )
    return results
