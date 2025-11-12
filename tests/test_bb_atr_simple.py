#!/usr/bin/env python3
"""
Test simple pour diagnostiquer pourquoi BBAtrStrategy ne génère aucun trade.
"""

import logging
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from threadx.strategy.bb_atr import BBAtrStrategy

try:
    from threadx.strategy.bb_atr import BBAtrParams  # legacy
except Exception:
    BBAtrParams = None  # type: ignore

logging.basicConfig(level=logging.INFO)

# Créer des données de test simples
np.random.seed(42)
n = 100
dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")

# Prix avec forte variation pour garantir des signaux
base_price = 100.0
prices = base_price + np.cumsum(np.random.randn(n) * 2.0)  # Forte volatilité
high = prices + np.abs(np.random.randn(n) * 0.5)
low = prices - np.abs(np.random.randn(n) * 0.5)
volume = np.random.uniform(1000, 10000, n)

df = pd.DataFrame(
    {"open": prices, "high": high, "low": low, "close": prices, "volume": volume},
    index=dates,
)

print(f"Données créées: {len(df)} barres")
print(f"Prix: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")
print(f"Volatilité: {df['close'].std():.2f}")

# Créer la stratégie
strategy = BBAtrStrategy(symbol="TESTUSDC", timeframe="15m")

# Paramètres avec min_pnl_pct = 0.0
params_dict = {
    "bb_period": 20,
    "bb_std": 2.0,
    "entry_z": 1.5,
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "min_pnl_pct": 0.0,  # ← Le plus important
    "risk_per_trade": 0.02,
    "leverage": 1.0,
    "max_hold_bars": 50,
    "spacing_bars": 2,
    "take_profit_bb_middle": True,
    "trailing_stop": False,
    "use_sl_hard": False,
}

print("\n" + "=" * 60)
print("Paramètres utilisés:")
print("=" * 60)
if BBAtrParams is not None:
    params = BBAtrParams.from_dict(params_dict)
    for key, value in params_dict.items():
        actual_value = getattr(params, key, value)
        status = "✅" if actual_value == value else f"❌ ({actual_value})"
        print(f"{key:25s} = {value:10} {status}")
else:
    for key, value in params_dict.items():
        print(f"{key:25s} = {value}")

print("\n" + "=" * 60)
print("Exécution backtest...")
print("=" * 60)

# Exécuter le backtest
equity_curve, run_stats = strategy.backtest(
    df=df, params=params_dict, initial_capital=10000.0, fee_bps=4.5, slippage_bps=0.0
)

print("\n" + "=" * 60)
print("RÉSULTATS")
print("=" * 60)
print(f"Trades: {run_stats.total_trades}")
print(f"PnL: {run_stats.total_pnl:.2f} ({run_stats.total_pnl_pct:.2f}%)")
print("Capital initial: 10000.00")
print(f"Capital final: {run_stats.final_capital:.2f}")
print(f"Equity min: {equity_curve.min():.2f}")
print(f"Equity max: {equity_curve.max():.2f}")

if run_stats.total_trades == 0:
    print("\n❌ ÉCHEC: Aucun trade généré!")
    print("\nVérifions les signaux générés...")

    # Générer les signaux pour voir
    signals_df = strategy.generate_signals(df, params_dict)

    enter_long = (signals_df["signal"] == "ENTER_LONG").sum()
    enter_short = (signals_df["signal"] == "ENTER_SHORT").sum()
    exit_signals = (signals_df["signal"].str.contains("EXIT")).sum()

    print(f"  ENTER_LONG: {enter_long}")
    print(f"  ENTER_SHORT: {enter_short}")
    print(f"  EXIT signals: {exit_signals}")
    print(f"  Total signaux: {enter_long + enter_short + exit_signals}")

    if enter_long + enter_short > 0:
        print("\n⚠️  Signaux générés mais aucun trade exécuté!")
        print("Cause probable: problème de cash, position sizing ou filtrage")
else:
    print(f"\n✅ SUCCESS: {run_stats.total_trades} trades générés")
