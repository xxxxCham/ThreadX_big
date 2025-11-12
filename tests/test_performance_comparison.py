#!/usr/bin/env python3
"""
Comparaison de performance avant/après optimisation O(1)
"""
import sys

sys.path.insert(0, "src")

import time
import numpy as np
import pandas as pd
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec
from threadx.indicators.bank import IndicatorBank


def create_test_data(n_bars: int) -> pd.DataFrame:
    """Données de test avec timezone UTC (comme test_parallel_engine.py)."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="15min", tz="UTC")

    trend = np.linspace(95000, 98000, n_bars)
    noise = np.random.randn(n_bars) * 2000
    cycles = 3000 * np.sin(np.linspace(0, 4 * np.pi, n_bars))
    close = trend + noise + cycles

    high = close + np.abs(np.random.randn(n_bars) * 500)
    low = close - np.abs(np.random.randn(n_bars) * 500)
    open_price = close + np.random.randn(n_bars) * 300
    volume = np.random.uniform(100, 500, n_bars)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def benchmark(n_bars: int, workers: int):
    """Test de performance avec n_bars."""
    df = create_test_data(n_bars)

    scenario = ScenarioSpec(
        type="grid",
        params={
            "bb_period": {"values": [15, 20, 25, 30, 35]},
            "bb_std": {"values": [1.5, 2.0, 2.5, 3.0, 3.5]},
            "entry_z": {"values": [1.0, 1.5, 2.0]},
            "atr_period": {"value": 14},
            "min_pnl_pct": {"value": 0.0},
            "risk_per_trade": {"value": 0.02},
        },
    )

    runner = SweepRunner(max_workers=workers, use_multigpu=True)

    start = time.time()
    results = runner.run_grid(df, scenario)
    duration = time.time() - start

    n_results = len(results)
    speed = n_results / duration if duration > 0 else 0

    return {
        "n_bars": n_bars,
        "n_results": n_results,
        "duration": duration,
        "speed": speed,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("TEST DE PERFORMANCE - Optimisation O(n²) → O(1)")
    print("=" * 80)

    # Test 1: Petites données (comme avant)
    print("\n🔬 Test 1: 500 barres (baseline)")
    result1 = benchmark(500, workers=8)
    print(f"   Résultats: {result1['n_results']} scénarios")
    print(f"   Durée: {result1['duration']:.2f}s")
    print(f"   Vitesse: {result1['speed']:.1f} tests/sec")

    # Test 2: Données moyennes
    print("\n🔬 Test 2: 2000 barres (4x plus)")
    result2 = benchmark(2000, workers=8)
    print(f"   Résultats: {result2['n_results']} scénarios")
    print(f"   Durée: {result2['duration']:.2f}s")
    print(f"   Vitesse: {result2['speed']:.1f} tests/sec")
    print(f"   Ratio vs 500 bars: {result2['speed'] / result1['speed']:.2f}x")

    # Test 3: Grandes données (vraies conditions)
    print("\n🔬 Test 3: 5760 barres (11.5x plus)")
    result3 = benchmark(5760, workers=8)
    print(f"   Résultats: {result3['n_results']} scénarios")
    print(f"   Durée: {result3['duration']:.2f}s")
    print(f"   Vitesse: {result3['speed']:.1f} tests/sec")
    print(f"   Ratio vs 500 bars: {result3['speed'] / result1['speed']:.2f}x")

    print("\n" + "=" * 80)
    print("📊 ANALYSE DE COMPLEXITÉ")
    print("=" * 80)
    print(f"Si complexité O(n): vitesse devrait rester constante")
    print(f"Si complexité O(n²): vitesse devrait chuter proportionnellement")
    print()
    print(f"Vitesse 500 bars:  {result1['speed']:.1f} tests/sec (baseline)")
    print(
        f"Vitesse 2000 bars: {result1['speed']:.1f} tests/sec ({result2['speed'] / result1['speed']:.0%} du baseline)"
    )
    print(
        f"Vitesse 5760 bars: {result3['speed']:.1f} tests/sec ({result3['speed'] / result1['speed']:.0%} du baseline)"
    )
    print()

    # Évaluation
    ratio_5760 = result3["speed"] / result1["speed"]
    if ratio_5760 > 0.8:
        print("✅ EXCELLENT: Complexité proche de O(n) - Optimisation réussie!")
    elif ratio_5760 > 0.5:
        print("⚠️  ACCEPTABLE: Légère dégradation, mais mieux que O(n²)")
    else:
        print("❌ PROBLÈME: Dégradation importante, investigation nécessaire")
