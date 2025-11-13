"""
ThreadX - Test Multi-Sweep Parallèle (P0.4)
============================================

Teste l'exécution de 4 sweeps simultanés pour saturer les ressources.
"""

import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from threadx.data_access import load_ohlcv
from threadx.optimization.multi_sweep import MultiSweepConfig, MultiSweepRunner
from threadx.optimization.scenarios import ScenarioSpec


def test_multi_sweep_parallel():
    """Teste 4 sweeps simultanés (saturation ressources)."""

    print("\n" + "="*80)
    print("TEST P0.4 - MULTI-SWEEP PARALLÈLE")
    print("="*80)

    # 1. Chargement données
    print("\n[1/3] Chargement données...")
    data = load_ohlcv("BTCUSDC", "15m", date(2024, 12, 1), date(2024, 12, 10))
    print(f"  ✅ Données: {len(data)} barres")

    # 2. Définir 4 grilles (chacune avec 24 combinaisons)
    print("\n[2/3] Génération 4 grilles (4×24 = 96 combos total)...")

    # Grille 1: BB period 15-30
    grid_1 = ScenarioSpec(
        type="grid",
        params={
            "bb_period": {"values": [15, 20, 25, 30]},
            "bb_std": {"values": [2.0, 2.5]},
            "atr_period": {"values": [10, 14, 21]},
        }
    )

    # Grille 2: BB period 35-60
    grid_2 = ScenarioSpec(
        type="grid",
        params={
            "bb_period": {"values": [35, 40, 45, 50]},
            "bb_std": {"values": [2.0, 2.5]},
            "atr_period": {"values": [10, 14, 21]},
        }
    )

    # Grille 3: BB std variations
    grid_3 = ScenarioSpec(
        type="grid",
        params={
            "bb_period": {"values": [20, 25, 30, 35]},
            "bb_std": {"values": [1.5, 3.0]},
            "atr_period": {"values": [10, 14, 21]},
        }
    )

    # Grille 4: ATR period variations
    grid_4 = ScenarioSpec(
        type="grid",
        params={
            "bb_period": {"values": [20, 25, 30, 35]},
            "bb_std": {"values": [2.0, 2.5]},
            "atr_period": {"values": [7, 14, 28]},
        }
    )

    grids = [grid_1, grid_2, grid_3, grid_4]
    print(f"  ✅ 4 grilles créées (24 combos chacune)")

    # 3. Exécution multi-sweep
    print("\n[3/3] Exécution multi-sweep (4 sweeps × 24 combos = 96 total)...")

    config = MultiSweepConfig(
        n_parallel_sweeps=4,
        workers_per_sweep=None,  # Auto
        use_multigpu=True
    )

    runner = MultiSweepRunner(config)

    t_start = time.perf_counter()

    results_by_sweep = runner.run_parallel_sweeps(
        grid_specs=grids,
        real_data=data,
        symbol="BTCUSDC",
        timeframe="15m",
        strategy_name="Bollinger_Breakout"
    )

    t_elapsed = time.perf_counter() - t_start

    # Résultats
    print("\n" + "="*80)
    print("RÉSULTATS")
    print("="*80)

    total_combos = sum(len(r) for r in results_by_sweep.values())

    print(f"\n  Sweeps exécutés: {len(results_by_sweep)}")
    print(f"  Résultats par sweep:")
    for sweep_id, results in sorted(results_by_sweep.items()):
        print(f"    - Sweep {sweep_id}: {len(results)} combinaisons")

    print(f"\n  Total combinaisons: {total_combos}")
    print(f"  Temps total: {t_elapsed:.2f} sec")
    print(f"  Vitesse: {total_combos / t_elapsed:.2f} tests/sec")

    # Extrapolation
    if total_combos > 0:
        speed = total_combos / t_elapsed
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

        # Comparaison P0.2 seul
        baseline_p02 = 100.74
        improvement = speed / baseline_p02

        print(f"\n  📈 Amélioration vs P0.2 seul:")
        print(f"     P0.2: {baseline_p02} tests/sec (ETA: 8h)")
        print(f"     P0.2+P0.4: {speed:.2f} tests/sec (ETA: {eta_hours:.2f}h)")
        print(f"     Gain: {improvement:.2f}x speedup")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_multi_sweep_parallel()
