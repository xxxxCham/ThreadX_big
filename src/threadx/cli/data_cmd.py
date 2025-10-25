"""
ThreadX CLI - Data Commands
============================

Commands for dataset validation and management.

Usage:
    python -m threadx.cli data validate <path>

Author: ThreadX Framework
Version: Prompt 9 - Data Commands
"""

import logging
from pathlib import Path
from typing import Optional

import typer

from threadx.cli.utils import (
    async_runner,
    format_duration,
    handle_bridge_error,
    print_json,
    print_summary,
)

app = typer.Typer(help="Dataset validation and management commands")
logger = logging.getLogger("threadx.cli.data")


@app.command()
def validate(
    path: str = typer.Argument(..., help="Path to dataset file (CSV/Parquet)"),
    symbol: Optional[str] = typer.Option(None, help="Symbol override"),
    timeframe: Optional[str] = typer.Option(None, "--tf", help="Timeframe override"),
) -> None:
    """
    Validate dataset and register in data registry.

    Loads dataset, checks schema, validates OHLCV columns,
    and registers in ThreadX data registry for use in backtests.

    Args:
        path: Path to dataset file (CSV or Parquet).
        symbol: Optional symbol override (default: infer from filename).
        timeframe: Optional timeframe override (default: infer from filename).
    """
    try:
        from threadx.bridge import ThreadXBridge
    except ImportError as e:
        logger.error(f"Failed to import ThreadXBridge: {e}")
        typer.echo("❌ Bridge not available. Check installation.")
        raise typer.Exit(1)

    # Get context for JSON mode
    ctx = typer.Context.get_current()
    json_mode = ctx.obj.get("json", False) if ctx.obj else False

    # Validate path exists
    data_path = Path(path)
    if not data_path.exists():
        error_msg = f"File not found: {path}"
        if json_mode:
            print_json({"status": "error", "message": error_msg})
        else:
            typer.echo(f"❌ {error_msg}")
        raise typer.Exit(1)

    logger.info(f"Validating dataset: {data_path}")

    try:
        # Initialize Bridge
        bridge = ThreadXBridge()

        # Prepare validation request
        request = {
            "path": str(data_path.absolute()),
            "symbol": symbol,
            "timeframe": timeframe,
        }

        # Submit async validation
        task_id = bridge.validate_data_async(request)
        logger.debug(f"Validation task submitted: {task_id}")

        if not json_mode:
            typer.echo(f"⏳ Validating {data_path.name}...")

        # Poll for results
        event = async_runner(bridge.get_event, task_id, timeout=30.0)

        if event is None:
            error_msg = "Validation timed out"
            if json_mode:
                print_json({"status": "timeout", "task_id": task_id})
            else:
                typer.echo(f"⚠️  {error_msg}")
            raise typer.Exit(1)

        # Check status
        if event.get("status") == "error":
            handle_bridge_error(
                Exception(event.get("error", "Unknown error")), json_mode
            )

        # Extract result
        result = event.get("result", {})
        duration = event.get("duration", 0)

        # Prepare summary
        summary = {
            "file": str(data_path.name),
            "symbol": result.get("symbol", "N/A"),
            "timeframe": result.get("timeframe", "N/A"),
            "rows": result.get("rows", 0),
            "columns": result.get("columns", 0),
            "date_range": f"{result.get('start_date', 'N/A')} → {result.get('end_date', 'N/A')}",
            "quality_score": result.get("quality", 0.0),
            "validation_time": format_duration(duration),
            "status": "✅ Valid" if result.get("valid", False) else "❌ Invalid",
        }

        # Output
        if json_mode:
            print_json({"status": "success", "data": summary, "event": event})
        else:
            print_summary("Dataset Validation", summary)

            if result.get("warnings"):
                typer.echo("⚠️  Warnings:")
                for warning in result["warnings"]:
                    typer.echo(f"  - {warning}")

    except Exception as e:
        handle_bridge_error(e, json_mode)


@app.command()
def list() -> None:
    """
    List all registered datasets in data registry.

    Shows symbol, timeframe, rows, date range, and validation status
    for all datasets available for backtesting.
    """
    try:
        from threadx.bridge import ThreadXBridge
    except ImportError as e:
        logger.error(f"Failed to import ThreadXBridge: {e}")
        typer.echo("❌ Bridge not available. Check installation.")
        raise typer.Exit(1)

    ctx = typer.Context.get_current()
    json_mode = ctx.obj.get("json", False) if ctx.obj else False

    try:
        bridge = ThreadXBridge()

        # Get registry (synchronous call)
        registry = bridge.get_data_registry()

        if json_mode:
            print_json({"status": "success", "datasets": registry})
        else:
            if not registry:
                typer.echo("📁 No datasets registered yet.")
                typer.echo("   Use 'threadx data validate <path>' to add datasets.")
                return

            typer.echo(f"\n📁 Registered Datasets ({len(registry)} total)\n")
            typer.echo(
                f"{'Symbol':<12} {'Timeframe':<10} {'Rows':<10} {'Date Range':<30} {'Status':<10}"
            )
            typer.echo("-" * 80)

            for dataset in registry:
                symbol = dataset.get("symbol", "N/A")
                tf = dataset.get("timeframe", "N/A")
                rows = dataset.get("rows", 0)
                start = dataset.get("start_date", "N/A")
                end = dataset.get("end_date", "N/A")
                valid = "✅ Valid" if dataset.get("valid", False) else "❌ Invalid"

                date_range = f"{start} → {end}"
                typer.echo(
                    f"{symbol:<12} {tf:<10} {rows:<10} {date_range:<30} {valid:<10}"
                )

            typer.echo()

    except Exception as e:
        handle_bridge_error(e, json_mode)


if __name__ == "__main__":
    app()
