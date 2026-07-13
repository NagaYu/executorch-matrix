"""Local, plain-JSON catalog of results, plus comparison-report rendering.

No telemetry, no hosted service — results live in a JSON file the user owns (the
same local-catalog pattern as ``silicon-forge``). A ``CatalogEntry`` records one
(device, model, backend, quantization) with its **export metrics** (always
present, no device needed) kept structurally separate from its **measured
metrics** (present only when a real device run produced an ETDump). The report
renderers preserve that separation so a reader can never mistake an export-only
row for a measured one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from executorch_matrix.export.variants import ExportResult

if TYPE_CHECKING:
    from executorch_matrix.measure.device_runner import DeviceInfo, MeasureResult

DEFAULT_CATALOG_PATH = Path.home() / ".executorch-matrix" / "catalog.json"


@dataclass
class CatalogEntry:
    """One backend x quantization result for a model on (optionally) a device.

    ``export_*`` fields are export-side facts that need no hardware. ``measured``
    is True only when ``latency_p50_ms`` came from a real ETDump; the two groups
    are never merged.
    """

    model: str
    backend: str
    quantization: str

    # --- export side (always populated) ---
    export_status: str = "unknown"  # ok | failed | skipped
    export_error: str | None = None
    export_skipped_reason: str | None = None
    export_seconds: float | None = None
    pte_bytes: int | None = None
    int4_group_size: int | None = None

    # --- device context (populated when measurement was attempted) ---
    device: str | None = None

    # --- measured side (populated only on a real device run) ---
    measure_status: str = "not-attempted"  # measured | skipped | failed | not-attempted
    measure_reason: str | None = None
    latency_p50_ms: float | None = None
    latency_avg_ms: float | None = None
    latency_p90_ms: float | None = None
    throughput_ips: float | None = None
    runs: int | None = None

    @property
    def variant(self) -> str:
        return f"{self.backend} / {self.quantization}"

    @property
    def measured(self) -> bool:
        return self.measure_status == "measured" and self.latency_p50_ms is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogEntry:
        fields = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in data.items() if k in fields})


def merge_results(
    model: str,
    export_results: list[ExportResult],
    measure_results: list[MeasureResult] | None = None,
    device: DeviceInfo | None = None,
) -> list[CatalogEntry]:
    """Combine export results (and optional measurements) into catalog entries."""
    measured_by_variant: dict[tuple[str, str], MeasureResult] = {}
    if measure_results:
        measured_by_variant = {(m.backend, m.quantization): m for m in measure_results}

    entries: list[CatalogEntry] = []
    for export in export_results:
        entry = CatalogEntry(
            model=model,
            backend=export.backend,
            quantization=export.quantization,
            export_status=export.status,
            export_error=export.error,
            export_skipped_reason=export.skipped_reason,
            export_seconds=export.export_seconds,
            pte_bytes=export.pte_bytes,
            int4_group_size=export.int4_group_size,
            device=device.identifier if device else None,
        )
        measurement = measured_by_variant.get((export.backend, export.quantization))
        if measurement is not None:
            entry.measure_status = measurement.status
            entry.measure_reason = measurement.reason
            latency = measurement.latency or {}
            entry.latency_p50_ms = latency.get("total_p50_ms")
            entry.latency_avg_ms = latency.get("total_avg_ms")
            entry.latency_p90_ms = latency.get("total_p90_ms")
            entry.throughput_ips = latency.get("throughput_ips")
            entry.runs = latency.get("runs")
        entries.append(entry)
    return entries


class Catalog:
    """A persistent JSON list of ``CatalogEntry`` records the user owns."""

    def __init__(self, path: str | Path = DEFAULT_CATALOG_PATH) -> None:
        self.path = Path(path)
        self.entries: list[CatalogEntry] = []

    def load(self) -> Catalog:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.entries = [CatalogEntry.from_dict(d) for d in data.get("entries", [])]
        return self

    def add(self, entries: list[CatalogEntry]) -> None:
        self.entries.extend(entries)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "executorch-matrix/catalog/v1",
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "entries": [e.to_dict() for e in self.entries],
        }
        self.path.write_text(json.dumps(payload, indent=2))
        return self.path


@dataclass
class ComparisonReport:
    """A single comparison run, ready to render to JSON or Markdown."""

    model: str
    device: str | None
    entries: list[CatalogEntry]
    recommendation: Any = None  # recommend.Recommendation (duck-typed to avoid a cycle)
    generated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def measured_any(self) -> bool:
        return any(e.measured for e in self.entries)

    @property
    def measurement_note(self) -> str:
        if self.measured_any:
            return f"Latency measured on a real device: {self.device}."
        return (
            "Export-only run — no device latency was measured. Sizes and export "
            "status are real; latency columns are intentionally blank."
        )

    def to_json_dict(self) -> dict[str, Any]:
        rec = self.recommendation
        return {
            "schema": "executorch-matrix/report/v1",
            "generated": self.generated,
            "model": self.model,
            "device": self.device,
            "measured": self.measured_any,
            "measurement_note": self.measurement_note,
            "recommendation": rec.to_dict() if rec is not None else None,
            "results": [e.to_dict() for e in self.entries],
        }


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    mb = n / 1_000_000
    return f"{mb:.1f} MB" if mb >= 1 else f"{n / 1000:.1f} KB"


def _fmt_ms(x: float | None) -> str:
    return f"{x:.1f} ms" if x is not None else "—"


def _export_cell(entry: CatalogEntry) -> str:
    return {"ok": "yes", "failed": "no", "skipped": "skip"}.get(entry.export_status, "?")


def _latency_cell(entry: CatalogEntry) -> str:
    if entry.measured:
        return _fmt_ms(entry.latency_p50_ms)
    if entry.measure_status == "not-attempted":
        return "not measured"
    return f"n/a ({entry.measure_status})"


def _note_cell(entry: CatalogEntry) -> str:
    if entry.export_status == "failed":
        return (entry.export_error or "export failed")[:60]
    if entry.export_status == "skipped":
        return (entry.export_skipped_reason or "skipped")[:60]
    if not entry.measured and entry.measure_reason:
        return entry.measure_reason[:60]
    if entry.int4_group_size:
        return f"int4 group size {entry.int4_group_size}"
    return ""


def render_markdown(report: ComparisonReport) -> str:
    """Render the comparison as a Markdown report."""
    lines: list[str] = []
    lines.append(f"# executorch-matrix report — {report.model}")
    lines.append("")
    lines.append(f"- Generated: {report.generated}")
    lines.append(f"- Device: {report.device or 'none connected'}")
    lines.append(f"- **{report.measurement_note}**")
    lines.append("")
    lines.append(
        "| Variant | Export OK | Size | Export time | Latency (p50) | Throughput | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for e in report.entries:
        export_s = f"{e.export_seconds:.2f}s" if e.export_seconds is not None else "—"
        tput = f"{e.throughput_ips:.1f}/s" if (e.measured and e.throughput_ips) else "—"
        lines.append(
            f"| {e.variant} | {_export_cell(e)} | {_fmt_bytes(e.pte_bytes)} | {export_s} "
            f"| {_latency_cell(e)} | {tput} | {_note_cell(e)} |"
        )
    lines.append("")
    if not report.measured_any:
        lines.append(
            "> Latency and throughput are blank because no device measurement was run. "
            "These export-only results are **not** a performance ranking."
        )
        lines.append("")
    rec = report.recommendation
    if rec is not None:
        lines.append("## Recommendation")
        lines.append("")
        lines.append(f"**{rec.headline}**")
        lines.append("")
        for reason in rec.reasoning:
            lines.append(f"- {reason}")
        lines.append("")
    return "\n".join(lines)


def render_json(report: ComparisonReport) -> str:
    return json.dumps(report.to_json_dict(), indent=2)
