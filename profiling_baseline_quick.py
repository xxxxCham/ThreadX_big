"""
ThreadX Profiling Baseline - Version Simplifiée
================================================

Profiling rapide pour mesurer la performance baseline actuelle.

Usage:
    python profiling_baseline_quick.py

Outputs:
    - Console avec mesures de performance
    - profiling_baseline_quick.txt
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.threadx.optimization.engine import UnifiedOptimizationEngine
from src.threadx.data_access import load_ohlcv


def main():
    """Point d'entrée principal."""

    print("=" * 100)
    print("🔬 PROFILING BASELINE QUICK - ThreadX")
    print("=" * 100)

    # Chargement données
    print("\n📊 Chargement données...")
    try:
        df = load_ohlcv("BTCUSDT", "1h")
        print(f"   ✅ {len(df)} barres chargées (BTCUSDT 1h)")
    except Exception as e:
        print(f"   ⚠️  Erreur: {e}")
        print("   📝 Utilisation de données synthétiques...")
        dates = pd.date_range("2024-01-01", periods=1000, freq="1h")
        df = pd.DataFrame(
            {
                "open": np.random.randn(1000).cumsum() + 100,
                "high": np.random.randn(1000).cumsum() + 102,
                "low": np.random.randn(1000).cumsum() + 98,
                "close": np.random.randn(1000).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 1000),
            },
            index=dates,
        )

    # Configuration sweep
    config = {
        "grid": {
            "bb_period": [20, 30, 40],  # 3
            "bb_std": [1.5, 2.0, 2.5],  # 3
            "atr_multiplier": [1.5, 2.0],  # 2
            "entry_z": [1.0, 1.5],  # 2
            "risk_per_trade": [0.01, 0.02],  # 2
        }
        # Total: 3 * 3 * 2 * 2 * 2 = 72 combinaisons
    }

    # Tests avec différents workers
    worker_configs = [4, 8, 16, 32]
    results = []

    for n_workers in worker_configs:
        print(f"\n{'=' * 100}")
        print(f"🧪 TEST: max_workers={n_workers}")
        print(f"{'=' * 100}")

        # Création engine
        print(f"   🚀 Création UnifiedOptimizationEngine (workers={n_workers})...")
        engine = UnifiedOptimizationEngine(max_workers=n_workers)

        # Benchmark
        print(f"   ⏱️  Exécution sweep...")

        start = time.perf_counter()
        results_df = engine.run_parameter_sweep(config=config, data=df)
        elapsed = time.perf_counter() - start

        n_combos = len(results_df)
        tests_per_sec = n_combos / elapsed if elapsed > 0 else 0

        print(f"   ✅ RÉSULTATS:")
        print(f"      Combinaisons: {n_combos}")
        print(f"      Temps total:  {elapsed:.2f}s")
        print(f"      ⚡ VITESSE:    {tests_per_sec:.1f} tests/sec")
        print(f"      Temps/combo:  {(elapsed/n_combos)*1000:.2f}ms")

        results.append(
            {
                "workers": n_workers,
                "combos": n_combos,
                "time": elapsed,
                "tests_per_sec": tests_per_sec,
            }
        )

    # Résumé comparatif
    print("\n\n" + "=" * 100)
    print("📊 RÉSUMÉ COMPARATIF")
    print("=" * 100)

    print(f"\n{'Workers':<12} {'Temps (s)':<15} {'Tests/sec':<15} {'Speedup':<15}")
    print("-" * 100)

    baseline = results[0]["tests_per_sec"] if results else 1.0

    for res in results:
        speedup = res["tests_per_sec"] / baseline
        speedup_pct = (speedup - 1.0) * 100

        print(
            f"{res['workers']:<12} {res['time']:>13.2f}s  {res['tests_per_sec']:>13.1f}  "
            f"{speedup:>5.2f}x ({speedup_pct:+.1f}%)"
        )

    # Meilleur config
    best = max(results, key=lambda x: x["tests_per_sec"])
    print("\n" + "=" * 100)
    print(f"🏆 MEILLEURE CONFIGURATION: max_workers={best['workers']}")
    print(f"   Performance: {best['tests_per_sec']:.1f} tests/sec")
    print(f"   Gain vs workers=4: {(best['tests_per_sec']/baseline - 1)*100:+.1f}%")

    # Projection sweep 8448 combos
    print("\n📈 PROJECTION SWEEP COMPLET (8448 combos)")
    print("-" * 100)

    for res in results:
        estimated_time = 8448 / res["tests_per_sec"]
        estimated_min = estimated_time / 60

        print(
            f"  workers={res['workers']:<4}  Temps estimé: {estimated_time:>7.1f}s ({estimated_min:>5.1f} min)"
        )

    print("=" * 100)

    # Sauvegarde
    output_file = Path("profiling_baseline_quick.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("THREADX PROFILING BASELINE QUICK REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write(
            f"{'Workers':<12} {'Temps (s)':<15} {'Tests/sec':<15} {'Speedup':<15}\n"
        )
        f.write("-" * 100 + "\n")

        for res in results:
            speedup = res["tests_per_sec"] / baseline
            speedup_pct = (speedup - 1.0) * 100
            f.write(
                f"{res['workers']:<12} {res['time']:>13.2f}s  {res['tests_per_sec']:>13.1f}  "
                f"{speedup:>5.2f}x ({speedup_pct:+.1f}%)\n"
            )

        f.write("\n" + "=" * 100 + "\n")
        f.write(f"MEILLEURE CONFIGURATION: max_workers={best['workers']}\n")
        f.write(f"Performance: {best['tests_per_sec']:.1f} tests/sec\n\n")

        f.write("PROJECTION SWEEP COMPLET (8448 combos)\n")
        f.write("-" * 100 + "\n")

        for res in results:
            estimated_time = 8448 / res["tests_per_sec"]
            estimated_min = estimated_time / 60
            f.write(
                f"workers={res['workers']:<4}  Estimated time: {estimated_time:>7.1f}s ({estimated_min:>5.1f} min)\n"
            )

    print(f"\n💾 Rapport sauvé: {output_file}\n")


if __name__ == "__main__":
    main()
