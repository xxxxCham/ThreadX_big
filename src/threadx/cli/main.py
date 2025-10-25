"""
ThreadX CLI - Main Entry Point
===============================

Main CLI application using Typer framework.
Provides commands for data, indicators, backtest, and optimization.

Usage:
    python -m threadx.cli --help
    python -m threadx.cli data validate <path>
    python -m threadx.cli backtest run --strategy <name>

Global Options:
    --json: Output results as JSON
    --debug: Enable debug logging
    --async: Enable async execution (experimental)

Author: ThreadX Framework
Version: Prompt 9 - CLI Main Application
"""

import logging
import sys

import typer

from .commands import backtest_cmd, data_cmd, indicators_cmd, optimize_cmd
from .utils import setup_logger

# Create main Typer app
app = typer.Typer(
    name="threadx",
    help="ThreadX - GPU-Accelerated Backtesting Framework CLI",
    add_completion=False,
)

# Add subcommands
app.add_typer(data_cmd.app, name="data")
app.add_typer(indicators_cmd.app, name="indicators")
app.add_typer(backtest_cmd.app, name="backtest")
app.add_typer(optimize_cmd.app, name="optimize")

# Global logger
logger = logging.getLogger("threadx.cli")


@app.callback()
def main(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON instead of human-readable text",
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug logging (verbose output)"
    ),
    async_mode: bool = typer.Option(
        False,
        "--async",
        help="Enable async execution mode (experimental)",
    ),
) -> None:
    """
    ThreadX CLI - GPU-Accelerated Backtesting Framework.

    Command-line interface for running backtests, building indicators,
    validating datasets, and optimizing strategy parameters.

    All commands use ThreadXBridge for async execution and are
    compatible with the Dash UI (same backend).

    Examples:
        # Validate dataset
        threadx data validate ./data/btc_1d.csv

        # Build indicators
        threadx indicators build --symbol BTCUSDT --tf 1h

        # Run backtest
        threadx backtest run --strategy ema_crossover --symbol ETHUSDT

        # Optimize parameters
        threadx optimize sweep --strategy bollinger --param period \\
            --min 10 --max 40 --step 5
    """
    # Setup logging level
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logger(log_level)

    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output
    ctx.obj["debug"] = debug
    ctx.obj["async"] = async_mode

    logger.debug(
        f"CLI initialized: json={json_output}, debug={debug}, " f"async={async_mode}"
    )

    if async_mode:
        logger.warning("Async mode is experimental and may not work with all commands")


@app.command()
def version(
    ctx: typer.Context,
) -> None:
    """Display ThreadX CLI version information."""
    py_ver = sys.version_info
    version_info = {
        "threadx_cli": "1.0.0",
        "prompt": "P9 - CLI Bridge Interface",
        "python_version": f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}",
    }

    json_mode = ctx.obj.get("json", False) if ctx.obj else False

    if json_mode:
        import json

        print(json.dumps(version_info, indent=2))
    else:
        typer.echo(f"ThreadX CLI v{version_info['threadx_cli']}")
        typer.echo(f"Prompt: {version_info['prompt']}")
        typer.echo(f"Python: {version_info['python_version']}")


def cli_entry() -> None:
    """
    Entry point for CLI when installed via pip/setuptools.

    Allows usage via `threadx` command instead of `python -m threadx.cli`.
    """
    app()


if __name__ == "__main__":
    app()
