"""
ThreadX - Benchmark Référence (Standard)
=========================================

Benchmark STANDARDISÉ pour comparer validement les performances entre modifications.

Configuration fixe:
- Données: BTCUSDC 15m, 2024-12-01 → 2024-12-10 (960 barres)
- Grille: 4 bb_period × 2 bb_std × 3 atr_period = 24 combinaisons
- Indicateurs uniques: 4 BB × 3 ATR = 12 indicateurs
- Ratio: 24 combos / 12 indicateurs = 2.0

Ce ratio 2:1 représente un cas réaliste d'optimisation.
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


def run_reference_benchmark():
    """Exécute le benchmark de référence standardisé."""

    print("\n" + "="*80)
    print("BENCHMARK RÉFÉRENCE STANDARDISÉ")
    print("="*80)
    print("\nConfiguration:")
    print("  - Données: BTCUSDC 15m, 960 barres")
    print("  - Grille: 4 BB × 2 BB_std × 3 ATR = 24 combinaisons")
    print("  - Indicateurs: 4 BB × 3 ATR = 12 indicateurs uniques")
    print("  - Ratio combos/indicateurs: 2.0 (réaliste)")

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

    # 2. Grille standardisée
    print("\n[2/3] Grille standardisée (4×2×3 = 24 combos)...")
    spec = ScenarioSpec(
        type="grid",
        params={
            "bb_period": {"values": [15, 20, 25, 30]},      # 4 valeurs
            "bb_std": {"values": [2.0, 2.5]},               # 2 valeurs
            "atr_period": {"values": [10, 14, 21]},         # 3 valeurs
        }
    )

    # 3. Sweep (répété 3x pour moyenne)
    print("\n[3/3] Exécution sweep (3 runs pour moyenne)...")

    speeds = []
    for run in range(3):
        print(f"\n  Run {run+1}/3...")

        # Vider cache pour test équitable
        if hasattr(bank, '_cache'):
            bank._cache.clear()

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
            speed = len(results) / t_elapsed
            speeds.append(speed)

            print(f"    Temps: {t_elapsed:.2f}s, Vitesse: {speed:.2f} tests/sec")

        except Exception as e:
            print(f"    ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return

    # Résultats
    print(f"\n{'='*80}")
    print("RÉSULTATS")
    print("="*80)

    avg_speed = sum(speeds) / len(speeds)
    min_speed = min(speeds)
    max_speed = max(speeds)

    print(f"\n  Vitesses: {[f'{s:.2f}' for s in speeds]} tests/sec")
    print(f"  Moyenne: {avg_speed:.2f} tests/sec")
    print(f"  Min/Max: {min_speed:.2f} / {max_speed:.2f} tests/sec")

    # Extrapolation pour 2.9M combinaisons
    eta_hours = 2903040 / avg_speed / 3600

    print(f"\n  📊 Extrapolation 2,903,040 combinaisons:")
    print(f"     ETA: {eta_hours:.2f} heures")

    if eta_hours < 20:
        status = "EXCELLENTE (<20h)"
    elif eta_hours < 40:
        status = "BONNE (20-40h)"
    elif eta_hours < 60:
        status = "MOYENNE (40-60h)"
    else:
        status = "FAIBLE (>60h)"

    print(f"     Statut: {status}")

    # Comparaison baseline (10.2 tests/sec = 79h)
    baseline_speed = 10.2
    baseline_eta = 79.0
    improvement = avg_speed / baseline_speed
    time_saved = baseline_eta - eta_hours

    print(f"\n  📈 Amélioration vs baseline (avant optimisations):")
    print(f"     Baseline: {baseline_speed} tests/sec, ETA: {baseline_eta:.1f}h")
    print(f"     Actuel: {avg_speed:.2f} tests/sec, ETA: {eta_hours:.2f}h")
    print(f"     Gain: {improvement:.2f}x speedup")
    print(f"     Temps économisé: {time_saved:.2f} heures ({time_saved/24:.1f} jours)")

    print("\n" + "="*80)


if __name__ == "__main__":
    run_reference_benchmark()
