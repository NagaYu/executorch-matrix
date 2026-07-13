"""Catalog persistence, result merging, and report rendering (no toolchain needed)."""

from __future__ import annotations

import json

from executorch_matrix.catalog.catalog import (
    Catalog,
    ComparisonReport,
    merge_results,
    render_json,
    render_markdown,
)
from executorch_matrix.export.variants import ExportResult
from executorch_matrix.recommend import recommend


def _export(backend, quant, status="ok", pte_bytes=1000, secs=0.5, error=None, skipped=None):
    return ExportResult(
        backend=backend,
        quantization=quant,
        status=status,
        error=error,
        skipped_reason=skipped,
        export_seconds=secs if status == "ok" else None,
        pte_bytes=pte_bytes if status == "ok" else None,
        pte_path="/tmp/x.pte" if status == "ok" else None,
    )


def test_merge_export_only_marks_not_attempted():
    results = [
        _export("xnnpack", "none"),
        _export("qualcomm", "int8", status="skipped", skipped="no SDK"),
    ]
    entries = merge_results("tiny", results)
    assert len(entries) == 2
    for e in entries:
        assert e.measure_status == "not-attempted"
        assert e.measured is False
        assert e.latency_p50_ms is None


def test_merge_attaches_measurements():
    class FakeMeasure:
        def __init__(self):
            self.backend = "xnnpack"
            self.quantization = "none"
            self.status = "measured"
            self.reason = None
            self.latency = {
                "total_p50_ms": 12.0,
                "total_avg_ms": 12.5,
                "total_p90_ms": 14.0,
                "throughput_ips": 83.3,
                "runs": 50,
            }

    entries = merge_results("tiny", [_export("xnnpack", "none")], [FakeMeasure()])
    e = entries[0]
    assert e.measured is True
    assert e.latency_p50_ms == 12.0
    assert e.throughput_ips == 83.3
    assert e.runs == 50


def test_catalog_roundtrip(tmp_path):
    path = tmp_path / "catalog.json"
    cat = Catalog(path)
    cat.add(merge_results("tiny", [_export("xnnpack", "none")]))
    saved = cat.save()
    assert saved.exists()

    reloaded = Catalog(path).load()
    assert len(reloaded.entries) == 1
    assert reloaded.entries[0].backend == "xnnpack"
    # file is plain, human-readable JSON with a schema tag
    data = json.loads(path.read_text())
    assert data["schema"].startswith("executorch-matrix/catalog")


def test_markdown_flags_export_only_runs():
    entries = merge_results("tiny", [_export("xnnpack", "none"), _export("xnnpack", "int8")])
    rec = recommend(entries, "size")
    report = ComparisonReport(model="tiny", device=None, entries=entries, recommendation=rec)
    md = render_markdown(report)
    assert "Export-only" in md or "export-only" in md
    assert "not measured" in md.lower()
    assert rec.headline in md
    assert "| Variant |" in md  # a real table


def test_json_report_separates_measured_flag():
    entries = merge_results("tiny", [_export("xnnpack", "none")])
    report = ComparisonReport(model="tiny", device=None, entries=entries)
    payload = json.loads(render_json(report))
    assert payload["measured"] is False
    assert payload["model"] == "tiny"
    assert isinstance(payload["results"], list)
    assert payload["results"][0]["measure_status"] == "not-attempted"
