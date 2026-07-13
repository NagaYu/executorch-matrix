"""Export-orchestration tests. Real exports through ExecuTorch; no device needed."""

from __future__ import annotations

import pytest

pytest.importorskip("executorch", reason="requires the ExecuTorch toolchain")

from executorch_matrix.export.models import resolve_model  # noqa: E402
from executorch_matrix.export.variants import (  # noqa: E402
    BACKENDS,
    export_matrix,
    export_variant,
)


@pytest.fixture(scope="module")
def spec():
    return resolve_model("tiny")


def test_export_ok_reports_real_metrics(spec, tmp_path):
    result = export_variant(spec, "xnnpack", "none", tmp_path, with_etrecord=False)
    assert result.status == "ok"
    assert result.export_ok
    assert result.pte_bytes and result.pte_bytes > 0
    assert result.export_seconds is not None and result.export_seconds >= 0
    assert result.pte_path is not None
    from pathlib import Path

    written = Path(result.pte_path)
    assert written.exists()
    assert written.stat().st_size == result.pte_bytes  # size traces to the real artifact


def test_int8_quantization_shrinks_pte(spec, tmp_path):
    fp32 = export_variant(spec, "xnnpack", "none", tmp_path, with_etrecord=False)
    int8 = export_variant(spec, "xnnpack", "int8", tmp_path, with_etrecord=False)
    assert fp32.status == "ok" and int8.status == "ok"
    assert int8.pte_bytes < fp32.pte_bytes  # int8 really is smaller on this model


def test_int4_records_group_size(spec, tmp_path):
    result = export_variant(spec, "xnnpack", "int4", tmp_path, with_etrecord=False)
    assert result.status == "ok"
    assert result.int4_group_size in (256, 128, 64, 32)


def test_unknown_backend_is_skipped_not_failed(spec, tmp_path):
    result = export_variant(spec, "does-not-exist", "none", tmp_path)
    assert result.status == "skipped"
    assert result.skipped_reason and "unknown backend" in result.skipped_reason
    assert result.pte_bytes is None  # a skip never carries fabricated metrics


def test_missing_vendor_sdk_is_skipped(spec, tmp_path):
    # Qualcomm needs the QNN SDK, which is not installed in CI -> honest skip.
    result = export_variant(spec, "qualcomm", "none", tmp_path)
    assert result.status == "skipped"
    assert result.skipped_reason is not None
    assert result.export_ok is False


def test_unknown_quantization_is_skipped(spec, tmp_path):
    result = export_variant(spec, "xnnpack", "int7", tmp_path)
    assert result.status == "skipped"
    assert "unknown quantization" in (result.skipped_reason or "")


def test_export_matrix_covers_every_combination(spec, tmp_path):
    backends = ["xnnpack", "qualcomm"]
    quants = ["none", "int8"]
    results = export_matrix(spec, backends, quants, tmp_path, with_etrecord=False)
    assert len(results) == len(backends) * len(quants)
    variants = {(r.backend, r.quantization) for r in results}
    assert variants == {(b, q) for b in backends for q in quants}


def test_portable_baseline_exports(spec, tmp_path):
    result = export_variant(spec, "portable", "none", tmp_path, with_etrecord=False)
    assert result.status == "ok"
    assert result.pte_bytes and result.pte_bytes > 0


def test_registry_has_expected_backends():
    for expected in ("portable", "xnnpack", "vulkan", "qualcomm"):
        assert expected in BACKENDS
    assert BACKENDS["xnnpack"].verified is True
