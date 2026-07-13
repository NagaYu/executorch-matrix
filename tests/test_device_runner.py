"""Device-runner tests: parsing, hardware matching, command building, and a
mocked end-to-end measure (subprocess/parse are stubbed — no real hardware)."""

from __future__ import annotations

from pathlib import Path

from executorch_matrix.export.variants import ExportResult
from executorch_matrix.measure import device_runner as dr
from executorch_matrix.measure.device_runner import (
    DeviceInfo,
    backend_hardware_available,
    build_run_command,
    measure_variant,
)
from executorch_matrix.measure.etdump_parser import LatencyMetrics


def test_parse_devices_keeps_only_online():
    output = (
        "List of devices attached\n"
        "ABC123\tdevice\n"
        "OFFLINE1\toffline\n"
        "UNAUTH1\tunauthorized\n"
        "DEF456\tdevice\n"
    )
    assert dr._parse_devices(output) == ["ABC123", "DEF456"]


def test_soc_agnostic_backends_run_anywhere():
    device = DeviceInfo(serial="x", soc_manufacturer="Google", soc_model="Tensor")
    for backend in ("portable", "xnnpack", "vulkan"):
        ok, reason = backend_hardware_available(backend, device)
        assert ok and reason == ""


def test_qualcomm_requires_qualcomm_soc():
    qc = DeviceInfo(serial="x", soc_manufacturer="Qualcomm", soc_model="SM8650")
    mtk = DeviceInfo(serial="y", soc_manufacturer="MediaTek", soc_model="Dimensity 9300")
    assert backend_hardware_available("qualcomm", qc)[0] is True
    ok, reason = backend_hardware_available("qualcomm", mtk)
    assert ok is False
    assert "matching hardware" in reason


def test_non_android_backend_is_rejected():
    device = DeviceInfo(serial="x", soc_manufacturer="Apple")
    ok, reason = backend_hardware_available("coreml", device)
    assert ok is False
    assert "not an Android" in reason


def test_build_run_command_shape():
    cmd = build_run_command("/d/runner", "/d/m.pte", "/d/m.etdump", num_executions=25)
    assert cmd[0] == "/d/runner"
    assert "--model_path=/d/m.pte" in cmd
    assert "--num_executions=25" in cmd
    assert "--etdump_path=/d/m.etdump" in cmd


def test_measure_skips_when_export_failed(tmp_path):
    export = ExportResult("xnnpack", "int8", "failed", error="boom")
    device = DeviceInfo(serial="x", soc_manufacturer="Google")
    res = measure_variant(export, device, None, local_out_dir=tmp_path)
    assert res.status == "skipped"
    assert "no successful export" in (res.reason or "")


def test_measure_skips_when_no_runner(tmp_path):
    pte = tmp_path / "m.pte"
    pte.write_bytes(b"\x00")
    export = ExportResult("xnnpack", "none", "ok", pte_bytes=1, pte_path=str(pte))
    device = DeviceInfo(serial="x", soc_manufacturer="Google")
    res = measure_variant(export, device, None, local_out_dir=tmp_path)
    assert res.status == "skipped"
    assert "runner" in (res.reason or "").lower()


def test_measure_skips_when_hardware_missing(tmp_path):
    pte = tmp_path / "m.pte"
    pte.write_bytes(b"\x00")
    runner = tmp_path / "executor_runner"
    runner.write_bytes(b"\x00")
    export = ExportResult("qualcomm", "int8", "ok", pte_bytes=1, pte_path=str(pte))
    device = DeviceInfo(serial="x", soc_manufacturer="MediaTek", soc_model="Dimensity")
    res = measure_variant(export, device, runner, local_out_dir=tmp_path)
    assert res.status == "skipped"
    assert "matching hardware" in (res.reason or "")


def test_measure_happy_path_with_mocked_adb(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_adb(args, serial=None, timeout=120.0):
        calls.append(args)
        if args and args[0] == "pull":
            Path(args[2]).write_bytes(b"fake-etdump-bytes")
        return ""

    fake_metrics = LatencyMetrics(
        total_event_name="Method::execute",
        total_p50_ms=12.0,
        total_avg_ms=12.0,
        total_p90_ms=13.0,
        throughput_ips=83.3,
        runs=50,
        num_events=3,
        events=[],
    )

    monkeypatch.setattr(dr, "adb_available", lambda: True)
    monkeypatch.setattr(dr, "_adb", fake_adb)
    monkeypatch.setattr(dr, "parse_etdump", lambda p, etr=None: fake_metrics)

    pte = tmp_path / "xnnpack_none.pte"
    pte.write_bytes(b"\x00")
    runner = tmp_path / "executor_runner"
    runner.write_bytes(b"\x00")
    export = ExportResult("xnnpack", "none", "ok", pte_bytes=1, pte_path=str(pte))
    device = DeviceInfo(serial="ABC123", model="Pixel 8", soc_manufacturer="Google")

    res = measure_variant(export, device, runner, local_out_dir=tmp_path)
    assert res.status == "measured"
    assert res.latency is not None and res.latency["total_p50_ms"] == 12.0
    # the on-device run command was actually issued
    assert any(a[0] == "shell" and any("--etdump_path" in str(x) for x in a) for a in calls)
