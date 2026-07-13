"""Builds a real ExecuTorch ``ETDump`` binary used by the parser tests.

Two things are true and both matter:

* The **binary format is real.** It is produced by ExecuTorch's own
  ``serialize_to_etdump_flatcc`` and wrapped in the size-prefixed framing the
  runtime writes, so ``Inspector`` deserializes it through the exact same code
  path a real on-device ETDump goes through.
* The **timing values are fixed test inputs, not measurements.** They are chosen
  to exercise the parser (percentile extraction, ns->ms conversion, total-event
  selection). They are NOT benchmark numbers and never appear in any report — the
  "no fabricated numbers" rule governs reports, and this is test data for a parser.

Regenerate the committed ``sample.etdump`` with:  ``python tests/fixtures/etdump_fixture.py``
"""

from __future__ import annotations

import struct
from pathlib import Path

NS_PER_MS = 1_000_000

BLOCK_NAME = "Execute"
TOTAL_EVENT_NAME = "Method::execute"
# Synthetic per-run totals (ms). Split 60/40 across a conv and a linear op so the
# total equals the sum of its parts — lets the test assert total-event selection.
SAMPLE_TOTALS_MS: list[float] = [12.0, 10.0, 11.0, 13.0, 14.0]

# The values the parser is expected to recover from the fixture above.
EXPECTED = {
    "block_name": BLOCK_NAME,
    "num_events": 3,
    "runs": len(SAMPLE_TOTALS_MS),
    "total_event_name": TOTAL_EVENT_NAME,
    "total_p50_ms": 12.0,  # median of SAMPLE_TOTALS_MS
    "total_avg_ms": 12.0,  # mean of SAMPLE_TOTALS_MS
    "total_min_ms": 10.0,
    "total_max_ms": 14.0,
    "throughput_ips": round(1000.0 / 12.0, 3),
    "event_names": {"Method::execute", "aten::convolution", "aten::linear"},
}

SAMPLE_PATH = Path(__file__).parent / "sample.etdump"


def build_sample_etdump_bytes() -> bytes:
    """Serialize the sample ETDump to the runtime's size-prefixed flatcc bytes."""
    import executorch.devtools.etdump.schema_flatcc as flatcc
    from executorch.devtools.etdump.serialize import serialize_to_etdump_flatcc

    def profile_event(name: str, start_ns: int, dur_ns: int) -> flatcc.Event:
        return flatcc.Event(
            profile_event=flatcc.ProfileEvent(
                name=name,
                chain_index=0,
                instruction_id=0,
                delegate_debug_id_str="",
                delegate_debug_id_int=-1,
                delegate_debug_metadata=b"",
                start_time=start_ns,
                end_time=start_ns + dur_ns,
            ),
            allocation_event=None,
            debug_event=None,
        )

    runs = []
    for total_ms in SAMPLE_TOTALS_MS:
        conv_ns = int(total_ms * 0.6 * NS_PER_MS)
        lin_ns = int(total_ms * 0.4 * NS_PER_MS)
        runs.append(
            flatcc.RunData(
                name=BLOCK_NAME,
                bundled_input_index=-1,
                allocators=[],
                events=[
                    profile_event(TOTAL_EVENT_NAME, 0, int(total_ms * NS_PER_MS)),
                    profile_event("aten::convolution", 0, conv_ns),
                    profile_event("aten::linear", conv_ns, lin_ns),
                ],
            )
        )

    program = flatcc.ETDumpFlatCC(version=0, run_data=runs)
    payload = bytes(serialize_to_etdump_flatcc(program))
    # The runtime writes a 4-byte little-endian size prefix ahead of the buffer.
    return struct.pack("<I", len(payload)) + payload


def write_sample_etdump(path: str | Path = SAMPLE_PATH) -> Path:
    path = Path(path)
    path.write_bytes(build_sample_etdump_bytes())
    return path


if __name__ == "__main__":
    written = write_sample_etdump()
    print(f"wrote {written} ({written.stat().st_size} bytes)")
