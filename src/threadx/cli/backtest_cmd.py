"""
ThreadX CLI - Backtest Commands
================================

Commands for running backtests and analyzing results.

Usage:
    python -m threadx.cli backtest run --strategy ema_crossover --symbol BTCUSDT

Author: ThreadX Framework
Version: Prompt 9 - Backtest Commands
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

app = typer.Typer(help="Backtest execution and analysis commands")
logger = logging.getLogger("threadx.cli.backtest")


@app.command()
def run(
    strategy: str = typer.Option(..., help="Strategy name"),
    symbol: str = typer.Option(..., help="Symbol to backtest"),
    timeframe: str = typer.Option("1h", "--tf", help="Timeframe"),
    period: Optional[int] = typer.Option(None, help="Strategy period parameter"),
    std: Optional[float] = typer.Option(None, help="Strategy std deviation parameter"),
    start_date: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD)"),
    initial_capital: float = typer.Option(10000.0, help="Initial capital"),
) -> None:
    """
    Run backtest for strategy on symbol/timeframe.

    Executes backtest via Bridge, computes performance metrics
    (equity curve, drawdown, Sharpe ratio, win rate),
    and displays results.

    Args:
        strategy: Strategy name (e.g., ema_crossover, bollinger_reversion).
        symbol: Symbol to backtest (e.g., BTCUSDT).
        timeframe: Timeframe (1m, 5m, 1h, 1d).
        period: Strategy period parameter (if applicable).
        std: Strategy std deviation parameter (if applicable).
        start_date: Start date for backtest period.
        end_date: End date for backtest period.
        initial_capital: Initial capital in USD.
    """
    try:
        from threadx.bridge import ThreadXBridge
    except ImportError as e:
        logger.error(f"Failed to import ThreadXBridge: {e}")
        typer.echo("❌ Bridge not available. Check installation.")
        raise typer.Exit(1)

    ctx = typer.Context.get_current()
    json_mode = ctx.obj.get("json", False) if ctx.obj else False

    logger.info(f"Running backtest: {strategy} on {symbol} @ {timeframe}")

    try:
        # Initialize Bridge
        bridge = ThreadXBridge()

        # Prepare backtest request
        request = {
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "initial_capital": initial_capital,
        }

        # Add optional params
        if period is not None:
            request["period"] = period
        if std is not None:
            request["std"] = std
        if start_date:
            request["start_date"] = start_date
        if end_date:
            request["end_date"] = end_date

        # Submit async backtest
        task_id = bridge.run_backtest_async(request)
        logger.debug(f"Backtest task submitted: {task_id}")

        if not json_mode:
            typer.echo(
                f"⏳ Running backtest: {strategy} on " f"{symbol} ({timeframe})..."
            )

        # Poll for results
        event = async_runner(bridge.get_event, task_id, timeout=300.0)

        if event is None:
            error_msg = "Backtest timed out"
            if json_mode:
                print_json({"status": "timeout", "task_id": task_id})
            else:
                typer.echo(f"⚠️  {error_msg}")
            raise typer.Exit(1)

        # Check status
        if event.get("status") == "error":
            error = Exception(event.get("error", "Unknown error"))
            handle_bridge_error(error, json_mode)

        # Extract result
        result = event.get("result", {})
        duration = event.get("duration", 0)

        # Extract metrics
        metrics = result.get("metrics", {})

        # Prepare summary
        summary = {
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "total_trades": metrics.get("total_trades", 0),
            "win_rate": f"{metrics.get('win_rate', 0) * 100:.2f}%",
            "total_return": f"{metrics.get('total_return', 0) * 100:.2f}%",
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": f"{metrics.get('max_drawdown', 0) * 100:.2f}%",
            "profit_factor": metrics.get("profit_factor", 0),
            "final_equity": metrics.get("final_equity", 0),
            "execution_time": format_duration(duration),
        }

        # Output
        if json_mode:
            output_data = {
                "status": "success",
                "summary": summary,
                "metrics": metrics,
                "trades": result.get("trades", []),
                "equity_curve": result.get("equity_curve", []),
            }
            print_json(output_data)
        else:
            print_summary("Backtest Results", summary)

            # Show best/worst trades
            trades = result.get("trades", [])
            if trades:
                sorted_trades = sorted(
                    trades, key=lambda t: t.get("pnl", 0), reverse=True
                )

                typer.echo("📊 Top 3 Best Trades:")
                for i, trade in enumerate(sorted_trades[:3], 1):
                    pnl = trade.get("pnl", 0)
                    pnl_pct = trade.get("pnl_pct", 0) * 100
                    entry = trade.get("entry_date", "N/A")
                    typer.echo(f"  {i}. ${pnl:,.2f} ({pnl_pct:+.2f}%) - " f"{entry}")

                typer.echo("\n📉 Top 3 Worst Trades:")
                for i, trade in enumerate(sorted_trades[-3:][::-1], 1):
                    pnl = trade.get("pnl", 0)
                    pnl_pct = trade.get("pnl_pct", 0) * 100
                    entry = trade.get("entry_date", "N/A")
                    typer.echo(f"  {i}. ${pnl:,.2f} ({pnl_pct:+.2f}%) - " f"{entry}")
                typer.echo()

    except Exception as e:
        handle_bridge_error(e, json_mode)


if __name__ == "__main__":
    app()
