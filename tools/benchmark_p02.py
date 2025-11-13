"""
ThreadX - Benchmark P0.2 (Sweep Moyen)
========================================

Benchmark avec ~150 combinaisons pour mesurer performance réelle.
"""

import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from threadx.data_access import load_ohlcv
from threadx.indicators.bank import IndicatorBank, IndicatorSettings
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec


def benchmark_p02():
    """Benchmark P0.2 avec sweep moyen (~150 combos)."""

    print("\n" + "="*80)
    print("BENCHMARK P0.2 - Sweep Moyen (~150 combinaisons)")
    print("="*80)

    # 1. Setup
    print("\n[1/3] Initialisation...")
    settings = IndicatorSettings(use_gpu=True)
    bank = IndicatorBank(settings)

    runner = SweepRunner(
        indicator_bank=bank,
        max_workers=None,  # Auto
        use_multigpu=True
    )

    data = load_ohlcv("BTCUSDC", "15m", date(2024, 12, 1), date(2024, 12, 10))
    print(f"  ✅ Données: {len(data)} barres")
    print(f"  ✅ Workers: {runner.max_workers}")

    # 2. Grille moyene: 5×3×3×3 = 135 combinaisons
    print("\n[2/3] Génération grille (5×3×3×3 = 135 combos)...")
    spec = ScenarioSpec(
        type="grid",
        params={
            "bb_period": {"values": [15, 20, 25, 30, 35]},       # 5 valeurs
            "bb_std": {"values": [1.5, 2.0, 2.5]},               # 3 valeurs
            "atr_period": {"values": [10, 14, 21]},              # 3 valeurs
            "atr_multiplier": {"values": [1.5, 2.0, 2.5]},       # 3 valeurs
        }
    )

    # 3. Sweep
    print("\n[3/3] Exécution sweep...")
    t_start = time.perf_counter()

    try:
        results = runner.run_grid(
            grid_spec=spec,
            real_data=data,
            symbol="BTCUSDC",
            timeframe="15m",
            strategy_name="Bollinger_Breakout"
        )

        t_elapsed = time.perf_counter() - t_start

        print(f"\n{'='*80}")
        print("RÉSULTATS")
        print("="*80)
        print(f"\n  Combinaisons: {len(results)}")
        print(f"  Temps: {t_elapsed:.2f} sec")
        print(f"  Vitesse: {len(results) / t_elapsed:.2f} tests/sec")

        # Extrapolation
        if len(results) > 0:
            speed = len(results) / t_elapsed
            eta_hours = 2903040 / speed / 3600

            print(f"\n  📊 Extrapolation 2,903,040 combinaisons:")
            print(f"     ETA: {eta_hours:.2f} heures")

            if eta_hours < 20:
                print(f"     ✅ EXCELLENTE (<20h)")
            elif eta_hours < 40:
                print(f"     ✅ BONNE (20-40h)")
            elif eta_hours < 60:
                print(f"     ⚠️ MOYENNE (40-60h)")
            else:
                print(f"     ❌ FAIBLE (>60h)")

            # Comparaison baseline
            baseline_speed = 10.2
            improvement = speed / baseline_speed
            print(f"\n  📈 Amélioration vs baseline:")
            print(f"     Baseline: {baseline_speed} tests/sec (79h)")
            print(f"     Actuel: {speed:.2f} tests/sec ({eta_hours:.2f}h)")
            print(f"     Gain: {improvement:.2f}x speedup")

    except Exception as e:
        print(f"\n  ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)


if __name__ == "__main__":
    benchmark_p02()
