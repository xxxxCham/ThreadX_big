#!/usr/bin/env python3
"""Test minimal pour diagnostiquer BBAtrStrategy"""

import sys

sys.path.insert(0, "src")

import pandas as pd
import numpy as np
from threadx.strategy.bb_atr import BBAtrStrategy, BBAtrParams

# Données simples
np.random.seed(42)
n = 100
dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
base = 100.0
prices = base + np.cumsum(np.random.randn(n) * 2.0)

df = pd.DataFrame(
    {
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": np.ones(n) * 1000,
    },
    index=dates,
)

print(
    f"Données: {len(df)} barres, prix {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}"
)

# Stratégie
strategy = BBAtrStrategy(symbol="TEST", timeframe="15m")

# Paramètres
params = {
    "bb_period": 20,
    "bb_std": 2.0,
    "entry_z": 1.5,
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "min_pnl_pct": 0.0,  # ← CRITIQUE
    "risk_per_trade": 0.02,
}

print(f"\nParamètres:")
print(f"  min_pnl_pct = {params['min_pnl_pct']}")

# Backtest
print("\nExécution backtest...")
equity_curve, run_stats = strategy.backtest(
    df=df,
    params=params,
    initial_capital=10000.0,
    fee_bps=4.5,
)

print(f"\nRÉSULTATS:")
print(f"  Trades: {run_stats.total_trades}")
print(f"  PnL: {run_stats.total_pnl:.2f} ({run_stats.total_pnl_pct:.2f}%)")
print(f"  Capital final: {run_stats.final_capital:.2f}")

if run_stats.total_trades == 0:
    print("\n❌ ÉCHEC: 0 trades!")
    print("\nVérifions les signaux...")

    signals_df = strategy.generate_signals(df, params)
    enter_long = (signals_df["signal"] == "ENTER_LONG").sum()
    enter_short = (signals_df["signal"] == "ENTER_SHORT").sum()

    print(f"  ENTER_LONG: {enter_long}")
    print(f"  ENTER_SHORT: {enter_short}")

    if enter_long + enter_short > 0:
        print("\n⚠️  Signaux MAIS pas de trades → problème exécution!")
else:
    print(f"\n✅ OK: {run_stats.total_trades} trades générés")
