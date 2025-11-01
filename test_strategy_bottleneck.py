#!/usr/bin/env python3
"""
Test de diagnostic du goulot d'étranglement dans l'évaluation des stratégies.
Analyse précise de où le temps est perdu dans strategy.backtest()
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("🔍 DIAGNOSTIC GOULOT D'ÉTRANGLEMENT - ÉVALUATION STRATÉGIE")
print("=" * 80)
print()

# ==================== 1. PRÉPARATION DONNÉES ====================
print("📊 Préparation données de test...")
dates = pd.date_range(start="2024-01-01", periods=5000, freq="1h")
test_data = pd.DataFrame(
    {
        "timestamp": dates,
        "open": 50000 + np.random.randn(5000) * 100,
        "high": 50100 + np.random.randn(5000) * 100,
        "low": 49900 + np.random.randn(5000) * 100,
        "close": 50000 + np.random.randn(5000) * 100,
        "volume": np.random.randint(1000, 10000, 5000),
    }
)
test_data.set_index("timestamp", inplace=True)
print(f"✅ Données: {len(test_data)} barres")
print()

# ==================== 2. IMPORT STRATÉGIE ====================
print("📦 Import stratégie BBAtrStrategy...")
try:
    from threadx.strategy import BBAtrStrategy
    from threadx.indicators.bank import IndicatorBank, IndicatorSettings

    print("✅ Imports réussis")
except Exception as e:
    print(f"❌ Erreur import: {e}")
    exit(1)
print()

# ==================== 3. PRÉPARATION INDICATEURS PRÉ-CALCULÉS ====================
print("🔧 Calcul indicateurs pré-calculés...")
bank = IndicatorBank(IndicatorSettings(max_workers=8, use_gpu=False))

# Calcul Bollinger
t_bb_start = time.perf_counter()
bb_result = bank.ensure(
    "bollinger",
    {"period": 20, "std": 2.0},
    test_data["close"],
    symbol="TEST",
    timeframe="1h",
)
t_bb_end = time.perf_counter()
print(f"   ⏱️  Bollinger: {(t_bb_end - t_bb_start)*1000:.2f}ms")

# Calcul ATR
t_atr_start = time.perf_counter()
atr_result = bank.ensure(
    "atr",
    {"period": 14, "method": "ema"},
    test_data[["high", "low", "close"]],
    symbol="TEST",
    timeframe="1h",
)
t_atr_end = time.perf_counter()
print(f"   ⏱️  ATR: {(t_atr_end - t_atr_start)*1000:.2f}ms")

# Préparation du dict precomputed
precomputed = {
    "bollinger": {'{"period":20,"std":2.0}': bb_result},
    "atr": {'{"method":"ema","period":14}': atr_result},
}
print(f"✅ Indicateurs pré-calculés prêts")
print()

# ==================== 4. TEST BACKTEST AVEC CHRONO DÉTAILLÉ ====================
print("=" * 80)
print("🔬 TEST BACKTEST - CHRONOMÉTRAGE DÉTAILLÉ")
print("=" * 80)
print()

strategy = BBAtrStrategy(symbol="TEST", timeframe="1h")

params = {
    "bb_period": 20,
    "bb_std": 2.0,
    "entry_z": 1.0,
    "atr_period": 14,
    "atr_multiplier": 1.5,
}

print(f"📋 Paramètres: {params}")
print()

# ==================== Test 1: AVEC indicateurs pré-calculés ====================
print("🟢 TEST 1: AVEC indicateurs pré-calculés")
print("-" * 80)

t_total_start = time.perf_counter()

try:
    equity_curve, run_stats = strategy.backtest(
        df=test_data,
        params=params,
        initial_capital=10000.0,
        fee_bps=4.5,
        slippage_bps=0.0,
        precomputed_indicators=precomputed,
    )

    t_total_end = time.perf_counter()
    total_time = (t_total_end - t_total_start) * 1000

    print(f"✅ Backtest réussi")
    print(f"   ⏱️  Temps TOTAL: {total_time:.2f}ms")
    print(f"   📊 Trades: {run_stats.get('num_trades', 0)}")
    print(f"   💰 PnL: {run_stats.get('total_pnl_pct', 0):.2f}%")
    print(f"   🎯 Win rate: {run_stats.get('win_rate', 0):.1f}%")

except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback

    traceback.print_exc()

print()

# ==================== Test 2: SANS indicateurs pré-calculés ====================
print("🔴 TEST 2: SANS indicateurs pré-calculés (re-calcul)")
print("-" * 80)

strategy2 = BBAtrStrategy(symbol="TEST", timeframe="1h")

t_total_start = time.perf_counter()

try:
    equity_curve, run_stats = strategy2.backtest(
        df=test_data,
        params=params,
        initial_capital=10000.0,
        fee_bps=4.5,
        slippage_bps=0.0,
        precomputed_indicators=None,  # Force re-calcul
    )

    t_total_end = time.perf_counter()
    total_time = (t_total_end - t_total_start) * 1000

    print(f"✅ Backtest réussi")
    print(f"   ⏱️  Temps TOTAL: {total_time:.2f}ms")
    print(f"   📊 Trades: {run_stats.get('num_trades', 0)}")
    print(f"   💰 PnL: {run_stats.get('total_pnl_pct', 0):.2f}%")
    print(f"   🎯 Win rate: {run_stats.get('win_rate', 0):.1f}%")

except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback

    traceback.print_exc()

print()

# ==================== Test 3: BATCH de 10 backtests ====================
print("=" * 80)
print("🔵 TEST 3: BATCH de 10 backtests avec indicateurs pré-calculés")
print("=" * 80)
print()

n_tests = 10
params_list = [
    {
        "bb_period": 20,
        "bb_std": std,
        "entry_z": 1.0,
        "atr_period": 14,
        "atr_multiplier": mult,
    }
    for std in [1.5, 2.0]
    for mult in [1.0, 1.5, 2.0, 2.5, 3.0]
]

print(f"📋 {len(params_list)} combinaisons de paramètres")
print()

t_batch_start = time.perf_counter()

results = []
for i, p in enumerate(params_list):
    try:
        strategy_batch = BBAtrStrategy(symbol="TEST", timeframe="1h")
        equity_curve, run_stats = strategy_batch.backtest(
            df=test_data,
            params=p,
            initial_capital=10000.0,
            fee_bps=4.5,
            slippage_bps=0.0,
            precomputed_indicators=precomputed,
        )
        results.append(run_stats)

        if (i + 1) % 5 == 0:
            elapsed = (time.perf_counter() - t_batch_start) * 1000
            rate = (i + 1) / (elapsed / 1000)
            print(f"   ✅ {i+1}/{len(params_list)} complétés - {rate:.1f} tests/sec")

    except Exception as e:
        print(f"   ❌ Erreur combo {i+1}: {e}")

t_batch_end = time.perf_counter()
batch_time = (t_batch_end - t_batch_start) * 1000

print()
print(f"✅ Batch terminé: {len(results)}/{len(params_list)} réussis")
print(f"   ⏱️  Temps total: {batch_time:.2f}ms")
print(f"   ⏱️  Temps moyen par test: {batch_time/len(params_list):.2f}ms")
print(f"   🚀 Vitesse: {len(params_list) / (batch_time/1000):.1f} tests/sec")

print()
print("=" * 80)
print("✅ DIAGNOSTIC TERMINÉ")
print("=" * 80)
