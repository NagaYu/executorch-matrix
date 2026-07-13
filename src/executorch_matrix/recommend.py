"""Priority-aware, reasoned recommendation over catalog results.

``recommend()`` returns a ``Recommendation`` with a plain-language headline *and*
the reasoning behind it — not a bare "winner" label. Crucially, it is honest about
its evidence: latency can only be ranked from real device measurements, so when a
run is export-only it says so and falls back to size (never pretending to rank
speed it did not measure).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from executorch_matrix.catalog.catalog import CatalogEntry

PRIORITIES = ("speed", "size", "balanced")


@dataclass
class Recommendation:
    winner: str | None
    priority: str
    basis: str  # "measured" | "export-only" | "none"
    headline: str
    reasoning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fmt_mb(n: int | None) -> str:
    if n is None:
        return "unknown size"
    mb = n / 1_000_000
    return f"{mb:.1f} MB" if mb >= 1 else f"{n / 1000:.1f} KB"


def _smallest(entries: list[CatalogEntry]) -> CatalogEntry:
    return min(entries, key=lambda e: e.pte_bytes or float("inf"))


def _fastest(entries: list[CatalogEntry]) -> CatalogEntry:
    return min(entries, key=lambda e: e.latency_p50_ms or float("inf"))


def _balanced_measured(entries: list[CatalogEntry]) -> CatalogEntry:
    """Min-max normalize latency and size, score = mean of the two (lower better)."""
    lat = [e.latency_p50_ms for e in entries if e.latency_p50_ms is not None]
    size = [e.pte_bytes for e in entries if e.pte_bytes is not None]
    lat_lo, lat_hi = min(lat), max(lat)
    size_lo, size_hi = min(size), max(size)

    def norm(value: float, lo: float, hi: float) -> float:
        return 0.0 if hi == lo else (value - lo) / (hi - lo)

    def score(e: CatalogEntry) -> float:
        nl = norm(e.latency_p50_ms, lat_lo, lat_hi) if e.latency_p50_ms is not None else 1.0
        ns = norm(e.pte_bytes, size_lo, size_hi) if e.pte_bytes is not None else 1.0
        return 0.5 * nl + 0.5 * ns

    return min(entries, key=score)


def recommend(entries: list[CatalogEntry], priority: str = "balanced") -> Recommendation:
    """Recommend a variant for the given ``priority`` (speed | size | balanced)."""
    priority = priority.lower()
    if priority not in PRIORITIES:
        priority = "balanced"

    ok = [e for e in entries if e.export_status == "ok" and e.pte_bytes is not None]
    if not ok:
        return Recommendation(
            winner=None,
            priority=priority,
            basis="none",
            headline="No recommendation: no variant exported successfully.",
            reasoning=[
                "Every requested backend x quantization combination failed to export "
                "or was skipped (e.g. a vendor SDK was not installed).",
                "See each variant's note for the specific reason.",
            ],
        )

    measured = [e for e in ok if e.measured]

    # ---- measured basis: real latency available ----
    if measured and priority in ("speed", "balanced"):
        winner = _fastest(measured) if priority == "speed" else _balanced_measured(measured)
        reasoning = _reason_measured(winner, measured, ok, priority)
        return Recommendation(
            winner=winner.variant,
            priority=priority,
            basis="measured",
            headline=_headline(winner, priority, measured=True),
            reasoning=reasoning,
        )

    if priority == "size":
        winner = _smallest(ok)
        reasoning = [
            f"{winner.variant} produces the smallest .pte at {_fmt_mb(winner.pte_bytes)} "
            f"of {len(ok)} successfully-exported variants.",
        ]
        others = sorted((e for e in ok if e is not winner), key=lambda e: e.pte_bytes or 0)
        if others:
            biggest = others[-1]
            reasoning.append(f"For contrast, {biggest.variant} is {_fmt_mb(biggest.pte_bytes)}.")
        if not measured:
            reasoning.append(
                "Size needs no device, so this ranking is fully grounded even without "
                "on-device measurement."
            )
        return Recommendation(
            winner=winner.variant,
            priority=priority,
            basis="measured" if measured else "export-only",
            headline=_headline(winner, priority, measured=bool(measured)),
            reasoning=reasoning,
        )

    # ---- export-only basis, speed/balanced requested: cannot rank speed ----
    winner = _smallest(ok)
    return Recommendation(
        winner=winner.variant,
        priority=priority,
        basis="export-only",
        headline=(
            f"Provisional (export-only): {winner.variant}. Connect a device to rank {priority}."
        ),
        reasoning=[
            f"This was an export-only run — no latency was measured, so {priority} "
            "cannot be ranked. Latency requires a real device run.",
            f"As an export-side signal only, {winner.variant} is the smallest at "
            f"{_fmt_mb(winner.pte_bytes)}; this is a footprint hint, not a speed ranking.",
            "Re-run with a connected device to get a measured recommendation.",
        ],
    )


def _headline(winner: CatalogEntry, priority: str, *, measured: bool) -> str:
    if priority == "size":
        return f"{winner.variant} — smallest footprint ({_fmt_mb(winner.pte_bytes)})."
    if not measured:
        return f"{winner.variant} (provisional, export-only)."
    if priority == "speed":
        return f"{winner.variant} — fastest measured (p50 {winner.latency_p50_ms:.1f} ms)."
    return (
        f"{winner.variant} — best measured balance of latency "
        f"({winner.latency_p50_ms:.1f} ms) and size ({_fmt_mb(winner.pte_bytes)})."
    )


def _reason_measured(
    winner: CatalogEntry,
    measured: list[CatalogEntry],
    ok: list[CatalogEntry],
    priority: str,
) -> list[str]:
    reasoning: list[str] = []
    if priority == "speed":
        reasoning.append(
            f"{winner.variant} has the lowest measured p50 latency "
            f"({winner.latency_p50_ms:.1f} ms) of {len(measured)} measured variants."
        )
    else:
        reasoning.append(
            f"{winner.variant} gives the best balance of measured latency "
            f"({winner.latency_p50_ms:.1f} ms) and size ({_fmt_mb(winner.pte_bytes)})."
        )
    slower = sorted(
        (e for e in measured if e is not winner and e.latency_p50_ms),
        key=lambda e: e.latency_p50_ms or 0,
    )
    if slower and winner.latency_p50_ms:
        other = slower[-1]
        ratio = (other.latency_p50_ms or 0) / winner.latency_p50_ms
        reasoning.append(
            f"The slowest measured variant, {other.variant}, was "
            f"{other.latency_p50_ms:.1f} ms ({ratio:.1f}x slower)."
        )
    smallest = _smallest(ok)
    if smallest is winner:
        reasoning.append("It is also the smallest export, so there is no size trade-off.")
    else:
        reasoning.append(
            f"If footprint matters more, {smallest.variant} is smaller "
            f"({_fmt_mb(smallest.pte_bytes)} vs {_fmt_mb(winner.pte_bytes)})."
        )
    reasoning.append("Ranking is based on real ETDump measurements from the connected device.")
    return reasoning
