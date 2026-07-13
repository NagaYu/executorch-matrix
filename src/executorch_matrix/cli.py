"""Command-line interface for executorch-matrix.

``compare`` is the one command that matters: it exports a model across a matrix of
backends x quantization levels, optionally measures each on a connected device,
and prints a comparison plus a reasoned recommendation. Heavy imports live inside
the command body so ``--help`` and ``list-backends`` stay instant.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from executorch_matrix import __version__

app = typer.Typer(
    name="executorch-matrix",
    help=(
        "Compare ExecuTorch hardware backends and quantization levels for your "
        "model on your actual device, and get a reasoned recommendation."
    ),
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"executorch-matrix {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """executorch-matrix top-level entry point."""


def _split(csv: str) -> list[str]:
    return [item.strip() for item in csv.split(",") if item.strip()]


@app.command("list-backends")
def list_backends() -> None:
    """List the hardware backends known to executorch-matrix."""
    from executorch_matrix.export.variants import BACKENDS

    table = Table(title="ExecuTorch backends")
    table.add_column("Backend")
    table.add_column("Target hardware")
    table.add_column("Vendor SDK")
    table.add_column("Verified path")
    table.add_column("Notes")
    for spec in BACKENDS.values():
        table.add_row(
            spec.name,
            spec.target_hardware,
            "yes" if spec.needs_vendor_sdk else "no",
            "yes" if spec.verified else "unverified",
            spec.notes,
        )
    console.print(table)


def _render_terminal(report: object, entries: list) -> None:
    from executorch_matrix.catalog.catalog import (
        _export_cell,
        _fmt_bytes,
        _latency_cell,
        _note_cell,
    )

    table = Table(title=f"executorch-matrix — {report.model}")  # type: ignore[attr-defined]
    table.add_column("Variant")
    table.add_column("Export OK")
    table.add_column("Size", justify="right")
    table.add_column("Export time", justify="right")
    table.add_column("Latency (p50)", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Notes")
    for e in entries:
        export_s = f"{e.export_seconds:.2f}s" if e.export_seconds is not None else "—"
        tput = f"{e.throughput_ips:.1f}/s" if (e.measured and e.throughput_ips) else "—"
        table.add_row(
            e.variant,
            _export_cell(e),
            _fmt_bytes(e.pte_bytes),
            export_s,
            _latency_cell(e),
            tput,
            _note_cell(e),
        )
    console.print(table)


@app.command()
def compare(
    model: str = typer.Argument(
        ..., help="Model name (e.g. 'tiny') or path to an examples/sample-model config.json."
    ),
    backends: str = typer.Option(
        "xnnpack", "--backends", "-b", help="Comma-separated backends (see 'list-backends')."
    ),
    quantize: str = typer.Option(
        "none,int8,int4", "--quantize", "-q", help="Comma-separated quant levels: none,int8,int4."
    ),
    priority: str = typer.Option(
        "balanced", "--priority", "-p", help="Recommendation priority: speed | size | balanced."
    ),
    out_dir: Path = typer.Option(
        Path(".executorch-matrix/variants"), "--out-dir", help="Where to write .pte artifacts."
    ),
    report_prefix: str = typer.Option(
        "executorch-matrix-report", "--report", help="Prefix for the .json/.md report files."
    ),
    runner: Path | None = typer.Option(
        None,
        "--runner",
        help="Path to an ExecuTorch example runner built with the event tracer "
        "(enables on-device latency measurement).",
    ),
    device: str | None = typer.Option(
        None, "--device", help="ADB serial of the target device (defaults to the first connected)."
    ),
    num_executions: int = typer.Option(
        50, "--num-executions", help="Runs per variant on device (for latency percentiles)."
    ),
    etrecord: bool = typer.Option(True, "--etrecord/--no-etrecord", help="Generate ETRecord."),
    catalog_path: Path | None = typer.Option(
        None, "--catalog", help="Path to the local JSON catalog (default: ~/.executorch-matrix)."
    ),
) -> None:
    """Export a model across backends x quantization, measure if a device is present, and rank."""
    from executorch_matrix.catalog.catalog import (
        DEFAULT_CATALOG_PATH,
        Catalog,
        ComparisonReport,
        merge_results,
        render_json,
        render_markdown,
    )
    from executorch_matrix.export.models import resolve_model
    from executorch_matrix.export.variants import export_matrix
    from executorch_matrix.recommend import recommend

    try:
        model_spec = resolve_model(model)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    backend_list = _split(backends)
    quant_list = _split(quantize)
    console.print(
        f"[bold]Exporting[/bold] '{model_spec.name}' across "
        f"{len(backend_list)}x{len(quant_list)} = {len(backend_list) * len(quant_list)} variants…"
    )
    export_results = export_matrix(
        model_spec, backend_list, quant_list, out_dir, with_etrecord=etrecord
    )

    # Measurement only when the user opted in (via --runner or --device) AND a device is present.
    measure_results = None
    device_info = None
    if runner is not None or device is not None:
        from executorch_matrix.measure.device_runner import (
            list_devices,
            measure_matrix,
        )

        devices = list_devices()
        if device is not None:
            device_info = next((d for d in devices if d.serial == device), None)
        elif devices:
            device_info = devices[0]

        if device_info is None:
            console.print(
                "[yellow]Measurement requested but no connected device found — "
                "reporting export-only results.[/yellow]"
            )
        else:
            console.print(f"[bold]Measuring[/bold] on {device_info.identifier}…")
            measure_results = measure_matrix(
                export_results,
                device_info,
                runner,
                local_out_dir=out_dir,
                num_executions=num_executions,
            )
    else:
        console.print(
            "[dim]Export-only run (no --runner/--device). Latency needs a connected device.[/dim]"
        )

    entries = merge_results(model_spec.name, export_results, measure_results, device_info)
    recommendation = recommend(entries, priority)
    report = ComparisonReport(
        model=model_spec.name,
        device=device_info.identifier if device_info else None,
        entries=entries,
        recommendation=recommendation,
    )

    _render_terminal(report, entries)
    console.print()
    console.print(f"[bold green]Recommendation:[/bold green] {recommendation.headline}")
    for reason in recommendation.reasoning:
        console.print(f"  • {reason}")

    json_path = Path(f"{report_prefix}.json")
    md_path = Path(f"{report_prefix}.md")
    json_path.write_text(render_json(report))
    md_path.write_text(render_markdown(report))
    console.print()
    console.print(f"[dim]Full report: {json_path} / {md_path}[/dim]")

    catalog = Catalog(catalog_path or DEFAULT_CATALOG_PATH).load()
    catalog.add(entries)
    saved = catalog.save()
    console.print(f"[dim]Catalog updated: {saved}[/dim]")


if __name__ == "__main__":
    app()
