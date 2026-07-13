"""CLI tests. --version/list-backends are light; compare runs a real export."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from executorch_matrix.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "executorch-matrix" in result.stdout


def test_list_backends_lists_xnnpack():
    result = runner.invoke(app, ["list-backends"])
    assert result.exit_code == 0
    assert "xnnpack" in result.stdout
    assert "qualcomm" in result.stdout


def test_compare_unknown_model_errors():
    result = runner.invoke(app, ["compare", "no-such-model"])
    assert result.exit_code == 2


def test_compare_export_only_writes_reports(tmp_path):
    pytest.importorskip("executorch", reason="requires the ExecuTorch toolchain")
    catalog = tmp_path / "catalog.json"
    report_prefix = tmp_path / "report"
    result = runner.invoke(
        app,
        [
            "compare",
            "tiny",
            "--backends",
            "xnnpack",
            "--quantize",
            "none",
            "--no-etrecord",
            "--out-dir",
            str(tmp_path / "variants"),
            "--report",
            str(report_prefix),
            "--catalog",
            str(catalog),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    assert catalog.exists()
    # export-only run: the report must say so, not imply latency
    assert "not measured" in (tmp_path / "report.md").read_text().lower()
