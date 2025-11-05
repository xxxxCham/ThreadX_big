#!/usr/bin/env python3
"""
Test l'impact du cache sur la vitesse de backtest.
Compare AVEC cache (activé) vs SANS cache (désactivé).
"""

import os
import sys
import time
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.strategy.bb_atr import BBATRStrategy
from threadx.data_access import load_ohlcv
from concurrent.futures import ProcessPoolExecutor


def run_single_backtest(args):
    """Execute un backtest avec paramètres donnés."""
    symbol, timeframe, params, use_cache = args

    # Désactiver cache si demandé
    if not use_cache:
        os.environ["THREADX_DISABLE_CACHE"] = "1"

    # Charger données
    df = load_ohlcv(symbol, timeframe, days=90)

    # Créer stratégie et exécuter
    strategy = BBATRStrategy(**params)
    result = strategy.backtest(df, initial_cash=10000.0)

    return result


def benchmark_cache_impact(num_tests=200, max_workers=30):
    """
    Compare vitesse AVEC vs SANS cache.

    Args:
        num_tests: Nombre de backtests à exécuter
        max_workers: Nombre de workers parallèles
    """
    symbol = "BTCUSDC"
    timeframe = "3m"

    # Paramètres de test (variation simple)
    params_list = []
    for i in range(num_tests):
        params_list.append(
            {
                "bb_period": 20 + (i % 10),
                "bb_std": 2.0 + (i % 5) * 0.1,
                "atr_period": 14 + (i % 5),
                "atr_mult_sl": 1.5 + (i % 10) * 0.1,
                "atr_mult_tp": 2.0 + (i % 10) * 0.1,
            }
        )

    print(f"🔬 Test impact cache: {num_tests} backtests avec {max_workers} workers\n")

    # ============================================
    # TEST 1: AVEC CACHE (situation actuelle)
    # ============================================
    print("=" * 60)
    print("📦 TEST 1: AVEC CACHE (situation actuelle)")
    print("=" * 60)

    # S'assurer que cache est activé
    if "THREADX_DISABLE_CACHE" in os.environ:
        del os.environ["THREADX_DISABLE_CACHE"]

    args_with_cache = [(symbol, timeframe, p, True) for p in params_list]

    start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_backtest, args_with_cache))
    duration_with_cache = time.time() - start

    speed_with_cache = num_tests / duration_with_cache

    print(f"✅ Terminé en {duration_with_cache:.1f}s")
    print(f"📊 Vitesse: {speed_with_cache:.1f} tests/sec\n")

    # ============================================
    # TEST 2: SANS CACHE (test diagnostic)
    # ============================================
    print("=" * 60)
    print("🚫 TEST 2: SANS CACHE (test diagnostic)")
    print("=" * 60)

    args_without_cache = [(symbol, timeframe, p, False) for p in params_list]

    start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_backtest, args_without_cache))
    duration_without_cache = time.time() - start

    speed_without_cache = num_tests / duration_without_cache

    print(f"✅ Terminé en {duration_without_cache:.1f}s")
    print(f"📊 Vitesse: {speed_without_cache:.1f} tests/sec\n")

    # ============================================
    # COMPARAISON
    # ============================================
    print("=" * 60)
    print("📈 RÉSULTATS COMPARATIFS")
    print("=" * 60)
    print(
        f"Avec cache:  {speed_with_cache:>8.1f} tests/sec  ({duration_with_cache:>6.1f}s)"
    )
    print(
        f"Sans cache:  {speed_without_cache:>8.1f} tests/sec  ({duration_without_cache:>6.1f}s)"
    )
    print()

    delta = speed_without_cache - speed_with_cache
    pct = (delta / speed_with_cache) * 100

    if delta > 0:
        print(
            f"🚀 SANS cache est {pct:+.1f}% PLUS RAPIDE (gain: {delta:.1f} tests/sec)"
        )
        print()
        print("⚠️  CONCLUSION: Le cache RALENTIT à cause des race conditions!")
        print("    → Recommandation: DÉSACTIVER le cache ou implémenter file locking")
    else:
        print(
            f"📦 AVEC cache est {-pct:+.1f}% PLUS RAPIDE (gain: {-delta:.1f} tests/sec)"
        )
        print()
        print("✅ CONCLUSION: Le cache fonctionne correctement")
        print("    → Le problème de vitesse est ailleurs (backtest loop, GPU, etc.)")

    print()
    print("=" * 60)


if __name__ == "__main__":
    benchmark_cache_impact(num_tests=200, max_workers=30)
