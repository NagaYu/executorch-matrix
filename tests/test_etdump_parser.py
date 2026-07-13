"""ETDump parsing tests against a real (fixture) ETDump binary."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("executorch", reason="requires the ExecuTorch devtools Inspector")

from executorch_matrix.measure.etdump_parser import parse_etdump  # noqa: E402
from tests.fixtures.etdump_fixture import (  # noqa: E402
    EXPECTED,
    SAMPLE_PATH,
    write_sample_etdump,
)


@pytest.fixture(scope="module")
def etdump_path(tmp_path_factory):
    # Prefer the committed fixture; regenerate into a tmp dir if it's absent.
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH
    return write_sample_etdump(tmp_path_factory.mktemp("etdump") / "sample.etdump")


def test_parses_total_latency(etdump_path):
    metrics = parse_etdump(etdump_path)
    assert metrics.measured
    assert metrics.source == "etdump"
    assert metrics.units == "ms"
    assert metrics.total_event_name == EXPECTED["total_event_name"]
    assert metrics.total_p50_ms == pytest.approx(EXPECTED["total_p50_ms"])
    assert metrics.total_avg_ms == pytest.approx(EXPECTED["total_avg_ms"])


def test_throughput_is_derived_from_latency(etdump_path):
    metrics = parse_etdump(etdump_path)
    assert metrics.throughput_ips == pytest.approx(EXPECTED["throughput_ips"], rel=1e-3)
    # throughput must be consistent with the p50 it was derived from (parser rounds to 3dp)
    assert metrics.throughput_ips == pytest.approx(1000.0 / metrics.total_p50_ms, abs=1e-2)


def test_event_and_run_counts(etdump_path):
    metrics = parse_etdump(etdump_path)
    assert metrics.num_events == EXPECTED["num_events"]
    assert metrics.runs == EXPECTED["runs"]
    names = {e.name for e in metrics.events}
    assert names == EXPECTED["event_names"]


def test_percentiles_are_ordered(etdump_path):
    metrics = parse_etdump(etdump_path)
    total = next(e for e in metrics.events if e.name == EXPECTED["total_event_name"])
    assert total.min_ms is not None and total.max_ms is not None
    assert total.min_ms <= total.p50_ms <= total.max_ms
    assert total.min_ms == pytest.approx(EXPECTED["total_min_ms"])
    assert total.max_ms == pytest.approx(EXPECTED["total_max_ms"])
    assert not math.isnan(total.p50_ms)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_etdump(tmp_path / "nope.etdump")
