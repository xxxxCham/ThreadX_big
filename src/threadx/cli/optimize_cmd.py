"""
ThreadX CLI - Optimization Commands
====================================

Commands for parameter optimization (sweeps) and analysis.

Usage:
    python -m threadx.cli optimize sweep --strategy bollinger --param period

Author: ThreadX Framework
Version: Prompt 9 - Optimization Commands
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

app = typer.Typer(help="Parameter optimization and sweep commands")
logger = logging.getLogger("threadx.cli.optimize")


@app.command()
def sweep(
    strategy: str = typer.Option(..., help="Strategy name"),
    symbol: str = typer.Option(..., help="Symbol to optimize"),
    timeframe: str = typer.Option("1h", "--tf", help="Timeframe"),
    param: str = typer.Option(..., help="Parameter to sweep (period, std, etc.)"),
    min_value: float = typer.Option(..., "--min", help="Min parameter value"),
    max_value: float = typer.Option(..., "--max", help="Max parameter value"),
    step: float = typer.Option(1.0, help="Step size"),
    metric: str = typer.Option("sharpe_ratio", help="Optimization metric"),
    start_date: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD)"),
    top_n: int = typer.Option(10, help="Number of top results to show"),
) -> None:
    """
    Run parameter sweep optimization for strategy.

    Tests multiple parameter values, ranks by metric
    (Sharpe ratio, total return, profit factor),
    and displays optimal parameters.

    Args:
        strategy: Strategy name to optimize.
        symbol: Symbol to test on.
        timeframe: Timeframe (1m, 5m, 1h, 1d).
        param: Parameter to sweep (e.g., period, std).
        min_value: Minimum parameter value.
        max_value: Maximum parameter value.
        step: Step size for sweep.
        metric: Metric to optimize (sharpe_ratio, total_return, etc.).
        start_date: Start date for backtest period.
        end_date: End date for backtest period.
        top_n: Number of top results to display.
    """
    try:
        from threadx.bridge import ThreadXBridge
    except ImportError as e:
        logger.error(f"Failed to import ThreadXBridge: {e}")
        typer.echo("❌ Bridge not available. Check installation.")
        raise typer.Exit(1)

    ctx = typer.Context.get_current()
    json_mode = ctx.obj.get("json", False) if ctx.obj else False

    logger.info(
        f"Running sweep: {strategy} on {symbol}, " f"{param}=[{min_value}, {max_value}]"
    )

    try:
        # Initialize Bridge
        bridge = ThreadXBridge()

        # Prepare sweep request
        request = {
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "param_name": param,
            "param_range": {
                "min": min_value,
                "max": max_value,
                "step": step,
            },
            "metric": metric,
        }

        # Add optional dates
        if start_date:
            request["start_date"] = start_date
        if end_date:
            request["end_date"] = end_date

        # Submit async sweep
        task_id = bridge.run_sweep_async(request)
        logger.debug(f"Sweep task submitted: {task_id}")

        if not json_mode:
            num_tests = int((max_value - min_value) / step) + 1
            typer.echo(
                f"⏳ Running optimization sweep: {strategy} on "
                f"{symbol} ({timeframe})"
            )
            typer.echo(
                f"   Testing {num_tests} values of '{param}' "
                f"from {min_value} to {max_value}..."
            )

        # Poll for results
        event = async_runner(bridge.get_event, task_id, timeout=600.0)

        if event is None:
            error_msg = "Optimization sweep timed out"
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

        # Extract top results
        top_results = result.get("top_results", [])[:top_n]

        # Prepare summary
        summary = {
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "parameter": param,
            "range": f"[{min_value}, {max_value}] (step={step})",
            "tests_run": result.get("tests_run", 0),
            "optimization_metric": metric,
            "execution_time": format_duration(duration),
        }

        # Add best result
        if top_results:
            best = top_results[0]
            summary["best_param_value"] = best.get("param_value")
            summary[f"best_{metric}"] = best.get(metric)

        # Output
        if json_mode:
            output_data = {
                "status": "success",
                "summary": summary,
                "top_results": top_results,
                "heatmap_data": result.get("heatmap_data", []),
            }
            print_json(output_data)
        else:
            print_summary("Optimization Sweep Results", summary)

            # Show top N results
            if top_results:
                typer.echo(
                    f"🏆 Top {len(top_results)} Results " f"(ranked by {metric}):\n"
                )
                typer.echo(
                    f"{'Rank':<6} {param.title():<15} "
                    f"{metric.replace('_', ' ').title():<20} "
                    f"{'Total Return':<15} {'Win Rate':<10}"
                )
                typer.echo("-" * 70)

                for i, res in enumerate(top_results, 1):
                    param_val = res.get("param_value", 0)
                    metric_val = res.get(metric, 0)
                    total_ret = res.get("total_return", 0) * 100
                    win_rate = res.get("win_rate", 0) * 100

                    metric_str = (
                        f"{metric_val:.4f}"
                        if isinstance(metric_val, float)
                        else str(metric_val)
                    )

                    typer.echo(
                        f"{i:<6} {param_val:<15} "
                        f"{metric_str:<20} "
                        f"{total_ret:>6.2f}%{'':<8} "
                        f"{win_rate:>5.1f}%"
                    )

                typer.echo()

    except Exception as e:
        handle_bridge_error(e, json_mode)


if __name__ == "__main__":
    app()
