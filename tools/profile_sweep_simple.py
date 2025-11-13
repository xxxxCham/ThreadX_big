"""
ThreadX - Profilage Simplifié d'un Sweep
=========================================

Script qui mesure précisément le temps de chaque composant du sweep.
"""

import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from threadx.data_access import load_ohlcv
from threadx.strategy.bb_atr import BBAtrStrategy, BBAtrParams


def profile_backtest_direct():
    """Profile un backtest direct sans passer par SweepRunner."""

    print("\n" + "="*80)
    print("PROFILAGE DIRECT D'UN BACKTEST")
    print("="*80)

    # 1. Chargement données
    print("\n[1/4] Chargement données...")
    t_start = time.perf_counter()
    data = load_ohlcv("BTCUSDC", "15m", date(2024, 12, 1), date(2024, 12, 31))
    t_load = time.perf_counter() - t_start
    print(f"  ✅ {len(data)} barres - {t_load*1000:.2f} ms")

    # 2. Création stratégie
    print("\n[2/4] Création stratégie...")
    t_start = time.perf_counter()
    strategy = BBAtrStrategy(symbol="BTCUSDC", timeframe="15m")
    t_strategy = time.perf_counter() - t_start
    print(f"  ✅ Stratégie créée - {t_strategy*1000:.2f} ms")

    # 3. Calcul indicateurs (première fois)
    print("\n[3/4] Calcul indicateurs (cold cache)...")
    params = BBAtrParams(
        bb_period=20,
        bb_std=2.0,
        atr_period=14,
        atr_multiplier=1.5
    )

    t_start = time.perf_counter()
    _ = strategy._ensure_indicators(data, params, precomputed_indicators=None)
    t_indicators_cold = time.perf_counter() - t_start
    print(f"  ✅ Indicateurs calculés - {t_indicators_cold*1000:.2f} ms")

    # 4. Backtest (avec indicateurs pré-calculés en cache)
    print("\n[4/4] Backtest (warm cache)...")
    t_start = time.perf_counter()
    # Convertir BBAtrParams en dict pour l'API backtest
    params_dict = {
        "bb_period": params.bb_period,
        "bb_std": params.bb_std,
        "atr_period": params.atr_period,
        "atr_multiplier": params.atr_multiplier,
        "risk_per_trade": params.risk_per_trade,
        "leverage": params.leverage,
        "max_hold_bars": params.max_hold_bars,
        "spacing_bars": params.spacing_bars,
        "min_pnl_pct": params.min_pnl_pct,
        "entry_z": params.entry_z,
        "trailing_stop": params.trailing_stop,
    }
    equity, stats = strategy.backtest(data, params_dict, initial_capital=10000, fee_bps=4)
    t_backtest = time.perf_counter() - t_start
    print(f"  ✅ Backtest terminé - {t_backtest*1000:.2f} ms")
    print(f"     Trades: {stats.total_trades}, PnL: {stats.total_pnl:.2f}")

    # Rapport
    print("\n" + "="*80)
    print("RÉSUMÉ TEMPS PAR COMPOSANT")
    print("="*80)

    total = t_load + t_strategy + t_indicators_cold + t_backtest

    print(f"\n{'Composant':<30} {'Temps (ms)':<15} {'% Total':<10}")
    print("-" * 80)

    components = [
        ("Chargement données", t_load),
        ("Création stratégie", t_strategy),
        ("Calcul indicateurs (cold)", t_indicators_cold),
        ("Backtest (warm)", t_backtest),
    ]

    for name, duration in components:
        pct = (duration / total * 100) if total > 0 else 0
        print(f"{name:<30} {duration*1000:>10.2f} ms   {pct:>6.2f}%")

    print("-" * 80)
    print(f"{'TOTAL':<30} {total*1000:>10.2f} ms   100.00%")

    # Estimation pour sweep massif
    print("\n" + "="*80)
    print("EXTRAPOLATION SWEEP MASSIF")
    print("="*80)

    # Hypothèse: indicateurs batch (calcul 1x pour N combos)
    # Temps par combo = temps_backtest_seul (indicateurs déjà en cache)

    time_per_combo_ms = t_backtest * 1000

    combos_to_test = 2903040
    estimated_time_sec = (time_per_combo_ms / 1000) * combos_to_test
    estimated_time_min = estimated_time_sec / 60
    estimated_time_hours = estimated_time_min / 60

    print(f"\n  Temps par combo (backtest seul): {time_per_combo_ms:.2f} ms")
    print(f"\n  Pour {combos_to_test:,} combinaisons:")
    print(f"    - Temps estimé (séquentiel): {estimated_time_sec:,.0f} sec")
    print(f"    - Soit: {estimated_time_min:,.2f} minutes")
    print(f"    - Soit: {estimated_time_hours:,.2f} heures")

    # Avec workers parallèles
    print("\n  Avec parallélisme (4 workers):")
    parallel_4 = estimated_time_hours / 4
    print(f"    - Temps estimé: {parallel_4:.2f} heures")

    print("\n  Avec parallélisme (30 workers):")
    parallel_30 = estimated_time_hours / 30
    print(f"    - Temps estimé: {parallel_30:.2f} heures")

    print("\n  ⚠️ Note: Vitesse actuelle observée = 10.2 tests/sec")
    observed_hours = combos_to_test / 10.2 / 3600
    print(f"    Cela donne: {observed_hours:.2f} heures (79 heures)")
    print(f"    Soit ~{observed_hours / parallel_30 * 30:.1f}x plus lent que l'estimation optimiste")


def measure_indicator_batching():
    """Mesure l'efficacité du batch processing des indicateurs."""

    print("\n" + "="*80)
    print("MESURE EFFICIENCY BATCH INDICATORS")
    print("="*80)

    from threadx.indicators.bank import IndicatorBank, IndicatorSettings

    data = load_ohlcv("BTCUSDC", "15m", date(2024, 12, 1), date(2024, 12, 31))

    settings = IndicatorSettings(use_gpu=True, cache_dir="indicators_cache")
    bank = IndicatorBank(settings=settings)

    # Test 1: Calcul séquentiel de 10 indicateurs différents
    print("\n[Test 1] Calcul séquentiel de 10 Bollinger Bands...")

    params_list = [
        {"period": 10 + i*5, "std": 2.0}
        for i in range(10)
    ]

    t_start = time.perf_counter()
    for params in params_list:
        _ = bank.ensure("bollinger", params, data["close"].values)
    t_sequential = time.perf_counter() - t_start

    print(f"  ✅ Temps séquentiel: {t_sequential*1000:.2f} ms ({t_sequential*1000/10:.2f} ms/indicateur)")

    # Test 2: Calcul batch des mêmes 10 indicateurs
    print("\n[Test 2] Calcul batch de 10 Bollinger Bands...")

    # Nettoyer cache pour test équitable
    bank._cache = {}

    t_start = time.perf_counter()
    _ = bank.batch_ensure("bollinger", params_list, data["close"].values)
    t_batch = time.perf_counter() - t_start

    print(f"  ✅ Temps batch: {t_batch*1000:.2f} ms ({t_batch*1000/10:.2f} ms/indicateur)")

    speedup = t_sequential / t_batch
    print(f"\n  Speedup batch vs séquentiel: {speedup:.2f}x")


def main():
    """Point d'entrée."""

    print("\n" + "="*80)
    print("ThreadX - Profilage Simplifié d'un Sweep")
    print("="*80)

    profile_backtest_direct()
    measure_indicator_batching()

    print("\n✅ Profilage terminé!\n")


if __name__ == "__main__":
    main()
