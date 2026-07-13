"""Parse ExecuTorch ``ETDump`` profiler output into structured latency metrics.

This is a thin adapter over ExecuTorch's *own* ``executorch.devtools.Inspector``
(we do not re-implement profiling). The Inspector deserializes an ``ETDump`` — the
binary the runtime emits during an on-device run — and exposes per-event timing
(``perf_data`` with p10/p50/p90/avg/min/max) already converted to milliseconds.
We pull those out, pick the top-level execution event as the model's total
latency, and derive throughput. Every value traces to the ``ETDump``; nothing is
invented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Event names (case-insensitive prefix match) that denote a full model execution,
# in priority order. The first match becomes the "total latency" event.
_TOTAL_EVENT_HINTS: tuple[str, ...] = (
    "method::execute",
    "program::execute",
    "execute",
    "forward",
    "inference",
)


@dataclass
class EventTiming:
    """Timing for a single profiled event, in milliseconds."""

    name: str
    p50_ms: float | None
    avg_ms: float | None
    p90_ms: float | None
    min_ms: float | None
    max_ms: float | None
    runs: int | None
    is_delegated: bool
    backend: str | None


@dataclass
class LatencyMetrics:
    """Structured latency for one measured variant.

    ``source`` is always ``"etdump"`` and ``units`` always ``"ms"`` so a report can
    label these as real device measurements, never conflating them with the
    export-only metrics.
    """

    total_event_name: str | None
    total_p50_ms: float | None
    total_avg_ms: float | None
    total_p90_ms: float | None
    throughput_ips: float | None
    runs: int | None
    num_events: int
    events: list[EventTiming] = field(default_factory=list)
    units: str = "ms"
    source: str = "etdump"

    @property
    def measured(self) -> bool:
        """True if at least a total latency was recovered from the ETDump."""
        return self.total_p50_ms is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # Guard against NaN leaking into a report.
    return result if result == result else None


def _pick_total_event(events: list[EventTiming]) -> EventTiming:
    """Choose the event that represents whole-model execution.

    Prefer a known top-level execution name; otherwise fall back to the
    longest-running event (the outermost span).
    """
    for hint in _TOTAL_EVENT_HINTS:
        for event in events:
            if event.name.lower().startswith(hint):
                return event
    return max(events, key=lambda e: e.p50_ms or 0.0)


def _make_inspector(etdump_path: str | Path, etrecord_path: str | Path | None) -> Any:
    from executorch.devtools import Inspector
    from executorch.devtools.inspector import TimeScale

    return Inspector(
        etdump_path=str(etdump_path),
        etrecord=str(etrecord_path) if etrecord_path else None,
        target_time_scale=TimeScale.MS,
    )


def parse_inspector(inspector: Any) -> LatencyMetrics:
    """Turn a constructed ExecuTorch ``Inspector`` into ``LatencyMetrics``.

    Separated from IO so it can be unit-tested against an Inspector built from a
    fixture ETDump.
    """
    events: list[EventTiming] = []
    for block in inspector.event_blocks:
        for event in block.events:
            perf = getattr(event, "perf_data", None)
            if perf is None:
                continue
            p50 = _to_float(getattr(perf, "p50", None))
            if p50 is None:
                continue  # not a profiled timing event (e.g. debug/allocation)
            raw = getattr(perf, "raw", None)
            events.append(
                EventTiming(
                    name=str(event.name),
                    p50_ms=p50,
                    avg_ms=_to_float(getattr(perf, "avg", None)),
                    p90_ms=_to_float(getattr(perf, "p90", None)),
                    min_ms=_to_float(getattr(perf, "min", None)),
                    max_ms=_to_float(getattr(perf, "max", None)),
                    runs=len(raw) if raw is not None else None,
                    is_delegated=bool(getattr(event, "is_delegated_op", False)),
                    backend=getattr(event, "delegate_backend_name", None),
                )
            )

    if not events:
        return LatencyMetrics(
            total_event_name=None,
            total_p50_ms=None,
            total_avg_ms=None,
            total_p90_ms=None,
            throughput_ips=None,
            runs=None,
            num_events=0,
            events=[],
        )

    total = _pick_total_event(events)
    throughput = 1000.0 / total.p50_ms if total.p50_ms else None
    return LatencyMetrics(
        total_event_name=total.name,
        total_p50_ms=total.p50_ms,
        total_avg_ms=total.avg_ms,
        total_p90_ms=total.p90_ms,
        throughput_ips=round(throughput, 3) if throughput is not None else None,
        runs=total.runs,
        num_events=len(events),
        events=events,
    )


def parse_etdump(
    etdump_path: str | Path,
    etrecord_path: str | Path | None = None,
) -> LatencyMetrics:
    """Parse an ``ETDump`` file (optionally joined with its ``ETRecord``)."""
    etdump_path = Path(etdump_path)
    if not etdump_path.exists():
        raise FileNotFoundError(f"ETDump not found: {etdump_path}")
    inspector = _make_inspector(etdump_path, etrecord_path)
    return parse_inspector(inspector)
