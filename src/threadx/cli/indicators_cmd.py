"""
ThreadX CLI - Indicators Commands
==================================

Commands for building and caching technical indicators.

Usage:
    python -m threadx.cli indicators build --symbol BTCUSDT --tf 1h

Author: ThreadX Framework
Version: Prompt 9 - Indicators Commands
"""

import logging
from typing import Optional

import typer

from threadx.cli.utils import (
    async_runner,
    format_duration,
    handle_bridge_error,
    print_json,
    print_summary,
)

app = typer.Typer(help="Technical indicators building and caching")
logger = logging.getLogger("threadx.cli.indicators")


@app.command()
def build(
    symbol: str = typer.Option(..., help="Symbol to build indicators for"),
    timeframe: str = typer.Option("1h", "--tf", help="Timeframe (1m, 5m, 1h, 1d)"),
    ema_period: Optional[int] = typer.Option(None, help="EMA period (default: 20)"),
    rsi_period: Optional[int] = typer.Option(None, help="RSI period (default: 14)"),
    bollinger_period: Optional[int] = typer.Option(
        None, "--bb-period", help="Bollinger period (default: 20)"
    ),
    bollinger_std: Optional[float] = typer.Option(
        None, "--bb-std", help="Bollinger std dev (default: 2.0)"
    ),
    force: bool = typer.Option(False, "--force", help="Force rebuild if cache exists"),
) -> None:
    """
    Build and cache technical indicators for symbol/timeframe.

    Computes EMA, RSI, Bollinger Bands, ATR, and other indicators,
    then caches results for fast backtest execution.

    Args:
        symbol: Symbol to compute indicators for.
        timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d).
        ema_period: EMA period (default: 20).
        rsi_period: RSI period (default: 14).
        bollinger_period: Bollinger Bands period (default: 20).
        bollinger_std: Bollinger Bands std deviation (default: 2.0).
        force: Force rebuild even if cache exists.
    """
    try:
        from threadx.bridge import ThreadXBridge
    except ImportError as e:
        logger.error(f"Failed to import ThreadXBridge: {e}")
        typer.echo("❌ Bridge not available. Check installation.")
        raise typer.Exit(1)

    ctx = typer.Context.get_current()
    json_mode = ctx.obj.get("json", False) if ctx.obj else False

    logger.info(f"Building indicators: {symbol} @ {timeframe}")

    try:
        # Initialize Bridge
        bridge = ThreadXBridge()

        # Prepare request with optional params
        request = {
            "symbol": symbol,
            "timeframe": timeframe,
            "force": force,
            "params": {},
        }

        # Add indicator params if provided
        if ema_period is not None:
            request["params"]["ema_period"] = ema_period
        if rsi_period is not None:
            request["params"]["rsi_period"] = rsi_period
        if bollinger_period is not None:
            request["params"]["bollinger_period"] = bollinger_period
        if bollinger_std is not None:
            request["params"]["bollinger_std"] = bollinger_std

        # Submit async build
        task_id = bridge.build_indicators_async(request)
        logger.debug(f"Indicators build task submitted: {task_id}")

        if not json_mode:
            typer.echo(f"⏳ Building indicators for {symbol} ({timeframe})...")

        # Poll for results
        event = async_runner(bridge.get_event, task_id, timeout=120.0)

        if event is None:
            error_msg = "Indicators build timed out"
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
            "symbol": symbol,
            "timeframe": timeframe,
            "indicators_built": result.get("indicators", []),
            "cache_size_mb": result.get("cache_size_mb", 0.0),
            "rows_processed": result.get("rows", 0),
            "build_time": format_duration(duration),
            "cache_path": result.get("cache_path", "N/A"),
            "status": "✅ Cached" if result.get("cached", False) else "❌ Failed",
        }

        # Output
        if json_mode:
            print_json({"status": "success", "data": summary, "event": event})
        else:
            print_summary("Indicators Build", summary)

            # Show indicator details
            if result.get("details"):
                typer.echo("📊 Indicator Details:")
                for ind_name, ind_data in result["details"].items():
                    params_str = ", ".join(
                        f"{k}={v}" for k, v in ind_data.get("params", {}).items()
                    )
                    typer.echo(f"  • {ind_name:<15} ({params_str})")
                typer.echo()

    except Exception as e:
        handle_bridge_error(e, json_mode)


@app.command()
def cache() -> None:
    """
    List cached indicators and cache statistics.

    Shows all cached indicator sets with symbol, timeframe,
    cache size, and last update time.
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

        # Get cache info (synchronous)
        cache_info = bridge.get_indicators_cache()

        if json_mode:
            print_json({"status": "success", "cache": cache_info})
        else:
            if not cache_info.get("cached_sets"):
                typer.echo("💾 No indicators cached yet.")
                typer.echo("   Use 'threadx indicators build' to cache indicators.")
                return

            total_size = cache_info.get("total_size_mb", 0.0)
            count = len(cache_info.get("cached_sets", []))

            typer.echo(f"\n💾 Indicators Cache ({count} sets, {total_size:.2f} MB)\n")
            typer.echo(
                f"{'Symbol':<12} {'Timeframe':<10} {'Indicators':<30} {'Size (MB)':<12} {'Updated':<20}"
            )
            typer.echo("-" * 90)

            for cached_set in cache_info["cached_sets"]:
                symbol = cached_set.get("symbol", "N/A")
                tf = cached_set.get("timeframe", "N/A")
                indicators = ", ".join(cached_set.get("indicators", []))[:28]
                size = cached_set.get("size_mb", 0.0)
                updated = cached_set.get("updated", "N/A")

                typer.echo(
                    f"{symbol:<12} {tf:<10} {indicators:<30} {size:<12.2f} {updated:<20}"
                )

            typer.echo()

    except Exception as e:
        handle_bridge_error(e, json_mode)


if __name__ == "__main__":
    app()
