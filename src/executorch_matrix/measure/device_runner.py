"""Deploy and run ``.pte`` variants on a connected device via ADB.

This wraps ExecuTorch's *own* example runner (e.g. ``executor_runner`` built with
the developer tools) rather than re-implementing on-device execution. The flow for
one variant is the documented ExecuTorch device workflow:

    adb push <runner> <remote>/ ; adb push <model.pte> <remote>/
    adb shell <remote>/<runner> --model_path=… --num_executions=N --etdump_path=…
    adb pull <remote>/<model.etdump>            # -> parsed by etdump_parser

The overriding rule here is honesty about measurability. A variant is only
reported as ``measured`` when a real ``ETDump`` came back from a real run. If there
is no device, no matching hardware for the backend, or no runner binary, the
variant is ``skipped`` with a plain reason — never estimated. Latency is *only*
ever produced by this path.

``adb`` is invoked as a subprocess; no device calls happen at import time, so this
module's pure logic (device parsing, hardware matching, command building) is unit
testable without hardware.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from executorch_matrix.export.variants import ExportResult
from executorch_matrix.measure.etdump_parser import LatencyMetrics, parse_etdump

# Backends that can actually run on an Android/ADB target.
ANDROID_BACKENDS: frozenset[str] = frozenset(
    {"portable", "xnnpack", "vulkan", "qualcomm", "mediatek"}
)

# Backends that run on any Android device regardless of SoC vendor (CPU/GPU).
_SOC_AGNOSTIC: frozenset[str] = frozenset({"portable", "xnnpack", "vulkan"})

# SoC-vendor substrings that indicate a backend's matching NPU is present.
_BACKEND_SOC_KEYS: dict[str, tuple[str, ...]] = {
    "qualcomm": ("qualcomm", "qcom", "snapdragon"),
    "mediatek": ("mediatek", "mtk", "dimensity"),
}

DEFAULT_REMOTE_DIR = "/data/local/tmp/executorch-matrix"
DEFAULT_NUM_EXECUTIONS = 50


@dataclass
class DeviceInfo:
    """Identity of a connected device, read from ``adb`` getprop."""

    serial: str
    model: str | None = None
    soc_manufacturer: str | None = None
    soc_model: str | None = None
    android_release: str | None = None

    @property
    def identifier(self) -> str:
        chip = self.soc_model or self.soc_manufacturer or "unknown-soc"
        return f"{self.model or self.serial} ({chip})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MeasureResult:
    """Outcome of trying to measure one variant on a device."""

    backend: str
    quantization: str
    status: str  # "measured" | "skipped" | "failed"
    reason: str | None = None
    device: str | None = None
    etdump_path: str | None = None
    latency: dict[str, Any] | None = None

    @property
    def variant(self) -> str:
        return f"{self.backend} / {self.quantization}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdbError(RuntimeError):
    """An ``adb`` invocation failed."""


def adb_available() -> bool:
    """True if the ``adb`` binary is on PATH."""
    return shutil.which("adb") is not None


def _adb(args: list[str], serial: str | None = None, timeout: float = 120.0) -> str:
    if not adb_available():
        raise AdbError("adb not found on PATH; install Android platform-tools")
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise AdbError(f"{' '.join(cmd)} -> exit {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def _parse_devices(adb_devices_output: str) -> list[str]:
    """Extract online device serials from ``adb devices`` output."""
    serials: list[str] = []
    for line in adb_devices_output.splitlines()[1:]:  # skip "List of devices attached"
        line = line.strip()
        if not line or "\t" not in line:
            continue
        serial, state = line.split("\t", 1)
        if state.strip() == "device":  # exclude offline/unauthorized
            serials.append(serial.strip())
    return serials


def describe_device(serial: str) -> DeviceInfo:
    """Read model/SoC/OS identity for a device via getprop."""

    def prop(name: str) -> str | None:
        try:
            value = _adb(["shell", "getprop", name], serial=serial, timeout=20).strip()
        except AdbError:
            return None
        return value or None

    return DeviceInfo(
        serial=serial,
        model=prop("ro.product.model"),
        soc_manufacturer=prop("ro.soc.manufacturer"),
        soc_model=prop("ro.soc.model"),
        android_release=prop("ro.build.version.release"),
    )


def list_devices() -> list[DeviceInfo]:
    """Return identity for every online device, or [] if none/no adb."""
    if not adb_available():
        return []
    try:
        output = _adb(["devices"])
    except AdbError:
        return []
    return [describe_device(serial) for serial in _parse_devices(output)]


def backend_hardware_available(backend: str, device: DeviceInfo) -> tuple[bool, str]:
    """Whether ``backend`` has matching hardware on ``device``.

    Returns ``(ok, reason)``; ``reason`` is empty when ok, else explains the skip.
    """
    if backend not in ANDROID_BACKENDS:
        return False, f"{backend} is not an Android/ADB-measurable backend"
    if backend in _SOC_AGNOSTIC:
        return True, ""
    keys = _BACKEND_SOC_KEYS.get(backend)
    if keys is None:
        return False, f"no hardware-match rule for {backend}; cannot confirm matching device"
    soc = " ".join(filter(None, [device.soc_manufacturer, device.soc_model])).lower()
    if soc and any(key in soc for key in keys):
        return True, ""
    return (
        False,
        f"{backend} needs matching hardware; device SoC is "
        f"{device.soc_manufacturer or 'unknown'} {device.soc_model or ''}".strip(),
    )


def build_run_command(
    remote_runner: str,
    remote_model: str,
    remote_etdump: str,
    num_executions: int = DEFAULT_NUM_EXECUTIONS,
) -> list[str]:
    """The ExecuTorch example-runner argv executed on-device."""
    return [
        remote_runner,
        f"--model_path={remote_model}",
        f"--num_executions={num_executions}",
        f"--etdump_path={remote_etdump}",
    ]


def measure_variant(
    export_result: ExportResult,
    device: DeviceInfo,
    runner_path: str | Path | None,
    *,
    local_out_dir: str | Path,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    num_executions: int = DEFAULT_NUM_EXECUTIONS,
) -> MeasureResult:
    """Deploy, run, and parse one variant on ``device``.

    Skips (never estimates) when the variant did not export, the backend has no
    matching hardware, or no runner binary is available.
    """
    result = MeasureResult(
        backend=export_result.backend,
        quantization=export_result.quantization,
        status="skipped",
        device=device.identifier,
    )

    if not export_result.export_ok or not export_result.pte_path:
        result.reason = "no successful export to measure"
        return result

    ok, reason = backend_hardware_available(export_result.backend, device)
    if not ok:
        result.reason = reason
        return result

    if runner_path is None or not Path(runner_path).exists():
        result.reason = (
            "no ExecuTorch runner binary available (build executor_runner with "
            "-DEXECUTORCH_ENABLE_EVENT_TRACER=ON and pass its path)"
        )
        return result

    pte = Path(export_result.pte_path)
    remote_runner = f"{remote_dir}/{Path(runner_path).name}"
    remote_model = f"{remote_dir}/{pte.name}"
    remote_etdump = f"{remote_dir}/{pte.stem}.etdump"
    local_etdump = Path(local_out_dir) / f"{pte.stem}.etdump"

    try:
        _adb(["shell", "mkdir", "-p", remote_dir], serial=device.serial, timeout=20)
        _adb(["push", str(runner_path), remote_runner], serial=device.serial)
        _adb(["push", str(pte), remote_model], serial=device.serial)
        _adb(["shell", "chmod", "755", remote_runner], serial=device.serial, timeout=20)
        _adb(
            [
                "shell",
                *build_run_command(remote_runner, remote_model, remote_etdump, num_executions),
            ],
            serial=device.serial,
            timeout=600,
        )
        Path(local_out_dir).mkdir(parents=True, exist_ok=True)
        _adb(["pull", remote_etdump, str(local_etdump)], serial=device.serial)
    except AdbError as exc:
        result.status = "failed"
        result.reason = str(exc)
        return result

    if not local_etdump.exists():
        result.status = "failed"
        result.reason = "run completed but no ETDump was retrieved from the device"
        return result

    try:
        latency: LatencyMetrics = parse_etdump(local_etdump, export_result.etrecord_path)
    except Exception as exc:  # noqa: BLE001 - surface parse failures honestly
        result.status = "failed"
        result.reason = f"ETDump parse failed: {type(exc).__name__}: {exc}"
        result.etdump_path = str(local_etdump)
        return result

    result.status = "measured"
    result.reason = None
    result.etdump_path = str(local_etdump)
    result.latency = latency.to_dict()
    return result


def measure_matrix(
    export_results: list[ExportResult],
    device: DeviceInfo,
    runner_path: str | Path | None,
    *,
    local_out_dir: str | Path,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    num_executions: int = DEFAULT_NUM_EXECUTIONS,
) -> list[MeasureResult]:
    """Measure every export result on ``device``."""
    return [
        measure_variant(
            er,
            device,
            runner_path,
            local_out_dir=local_out_dir,
            remote_dir=remote_dir,
            num_executions=num_executions,
        )
        for er in export_results
    ]
