"""
Exemple d'intégration ThreadXBridge avec CLI
=============================================

Démontre pattern synchrone (Future.result()) pour CLI.

Usage:
    python examples/async_bridge_cli_example.py BTCUSDT 1h bb

Architecture:
    CLI Main Thread
         ↓
    ThreadXBridge.run_backtest_async()
         ↓
    future.result(timeout=300)  # BLOQUE jusqu'à résultat
         ↓
    Print résultats
"""

import argparse
import sys

from threadx.bridge import (
    BacktestRequest,
    BacktestResult,
    Configuration,
    ThreadXBridge,
)


def print_progress(current: int, total: int, prefix: str = "Progress"):
    """Affiche barre progression ASCII."""
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    percent = 100.0 * current / total
    print(
        f"\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})",
        end="",
        flush=True,
    )


def main():
    """CLI principal avec ThreadXBridge."""
    # Arguments CLI
    parser = argparse.ArgumentParser(description="ThreadX Bridge CLI - Async Backtest")
    parser.add_argument("symbol", help="Symbol (ex: BTCUSDT)")
    parser.add_argument("timeframe", help="Timeframe (ex: 1h, 4h, 1d)")
    parser.add_argument("strategy", help="Strategy (ex: bb, ema, rsi)")
    parser.add_argument(
        "--workers", type=int, default=4, help="Max workers (default: 4)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout seconds (default: 300)",
    )

    args = parser.parse_args()

    # Initialize Bridge
    print(f"Initializing ThreadXBridge (workers={args.workers})...")
    bridge = ThreadXBridge(
        max_workers=args.workers,
        config=Configuration(max_workers=args.workers),
    )

    # Créer requête
    print(f"\nSubmitting backtest: {args.symbol} {args.timeframe} " f"{args.strategy}")
    req = BacktestRequest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy=args.strategy,
        params=(
            {"period": 20, "std": 2.0}
            if args.strategy == "bb"
            else {"fast": 12, "slow": 26}
        ),
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    # Callback optionnel (pour monitoring)
    def on_complete(result: BacktestResult | None, error: Exception | None):
        """Callback appelé au résultat (optionnel)."""
        if error:
            print(f"\n❌ Callback: Error - {error}")
        else:
            print("\n✅ Callback: Backtest complete!")

    # Soumettre tâche async
    future = bridge.run_backtest_async(req, callback=on_complete)

    print("Task submitted (task_id available in bridge.active_tasks)")
    print(f"Waiting for result (timeout={args.timeout}s)...\n")

    # BLOQUER jusqu'à résultat (pattern CLI)
    try:
        result = future.result(timeout=args.timeout)

        # Afficher résultats
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Symbol:          {args.symbol}")
        print(f"Timeframe:       {args.timeframe}")
        print(f"Strategy:        {args.strategy}")
        print("-" * 60)
        print(f"Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        print(f"Total Return:    {result.total_return:.2%}")
        print(f"Max Drawdown:    {result.max_drawdown:.2%}")
        print(f"Win Rate:        {result.win_rate:.2%}")
        print(f"Total Trades:    {result.total_trades}")
        print("-" * 60)
        print(
            f"Equity Curve:    {len(result.equity_curve)} points "
            f"(${result.equity_curve[0]:.2f} → "
            f"${result.equity_curve[-1]:.2f})"
        )
        print("=" * 60)

        # State final
        state = bridge.get_state()
        print(
            f"\nBridge State: {state['total_completed']} completed, "
            f"{state['total_failed']} failed"
        )

    except TimeoutError:
        print(f"\n❌ Timeout after {args.timeout}s")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    finally:
        # Cleanup
        print("\nShutting down ThreadXBridge...")
        bridge.shutdown(wait=True, timeout=10)
        print("Goodbye!")


if __name__ == "__main__":
    main()
