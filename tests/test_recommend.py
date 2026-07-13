"""Recommendation logic tests on synthetic catalog entries (no toolchain needed)."""

from __future__ import annotations

from executorch_matrix.catalog.catalog import CatalogEntry
from executorch_matrix.recommend import recommend


def entry(
    backend: str,
    quant: str,
    *,
    pte_bytes: int,
    latency: float | None = None,
    export_status: str = "ok",
) -> CatalogEntry:
    e = CatalogEntry(
        model="tiny",
        backend=backend,
        quantization=quant,
        export_status=export_status,
        pte_bytes=pte_bytes,
    )
    if latency is not None:
        e.measure_status = "measured"
        e.latency_p50_ms = latency
        e.latency_avg_ms = latency
        e.throughput_ips = round(1000.0 / latency, 3)
        e.runs = 50
    return e


def measured_set() -> list[CatalogEntry]:
    return [
        entry("xnnpack", "int8", pte_bytes=612_000_000, latency=340.0),
        entry("vulkan", "int8", pte_bytes=598_000_000, latency=210.0),
        entry("qualcomm", "int8", pte_bytes=590_000_000, latency=140.0),
        entry("qualcomm", "int4", pte_bytes=340_000_000, latency=125.0),
    ]


def test_speed_picks_lowest_latency():
    rec = recommend(measured_set(), "speed")
    assert rec.basis == "measured"
    assert rec.winner == "qualcomm / int4"  # 125 ms, the fastest
    assert any("125" in r for r in rec.reasoning)


def test_size_picks_smallest_pte():
    rec = recommend(measured_set(), "size")
    assert rec.winner == "qualcomm / int4"  # 340 MB, the smallest
    assert "smallest" in rec.headline.lower()


def test_balanced_considers_both():
    rec = recommend(measured_set(), "balanced")
    assert rec.basis == "measured"
    # qualcomm/int4 is both fastest and smallest here -> unambiguous winner
    assert rec.winner == "qualcomm / int4"
    assert len(rec.reasoning) >= 2


def test_reasoning_is_more_than_a_label():
    rec = recommend(measured_set(), "speed")
    assert rec.headline
    assert len(rec.reasoning) >= 2
    # mentions a comparison, not just the winner
    assert any("slower" in r or "smaller" in r or "size" in r for r in rec.reasoning)


def test_export_only_cannot_rank_speed():
    export_only = [
        entry("xnnpack", "none", pte_bytes=163_000),
        entry("xnnpack", "int8", pte_bytes=50_000),
    ]
    rec = recommend(export_only, "speed")
    assert rec.basis == "export-only"
    assert "export-only" in rec.headline.lower() or "provisional" in rec.headline.lower()
    assert any("device" in r.lower() for r in rec.reasoning)
    # falls back to the smallest as an honest footprint hint
    assert rec.winner == "xnnpack / int8"


def test_export_only_size_is_grounded():
    export_only = [
        entry("xnnpack", "none", pte_bytes=163_000),
        entry("xnnpack", "int8", pte_bytes=50_000),
    ]
    rec = recommend(export_only, "size")
    assert rec.winner == "xnnpack / int8"
    assert rec.basis == "export-only"


def test_no_successful_export_yields_no_winner():
    failed = [
        entry("qualcomm", "int8", pte_bytes=0, export_status="skipped"),
        entry("portable", "int8", pte_bytes=0, export_status="failed"),
    ]
    # pte_bytes present but status not ok -> filtered out
    for e in failed:
        e.pte_bytes = None
    rec = recommend(failed, "balanced")
    assert rec.winner is None
    assert rec.basis == "none"
    assert "No recommendation" in rec.headline
