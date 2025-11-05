"""
ThreadX Profiling Baseline - Version Production
================================================

Profiling utilisant SweepRunner (API production).

Usage:
    python profiling_baseline_production.py

Outputs:
    - Console avec mesures de performance
    - profiling_baseline_production.txt
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.threadx.optimization.engine import SweepRunner
from src.threadx.optimization.scenarios import ScenarioSpec


def create_test_data(n_bars=1000):
    """Crée des données OHLCV synthétiques."""
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")
    return pd.DataFrame(
        {
            "open": 50000 + np.random.randn(n_bars) * 100,
            "high": 50100 + np.random.randn(n_bars) * 100,
            "low": 49900 + np.random.randn(n_bars) * 100,
            "close": 50000 + np.random.randn(n_bars) * 100,
            "volume": np.random.randint(1000, 10000, n_bars),
        },
        index=dates,
    )


def benchmark_sweep(max_workers, n_combos=72):
    """
    Benchmark du sweep avec configuration spécifique.

    Args:
        max_workers: Nombre de workers
        n_combos: Nombre de combinaisons cible

    Returns:
        Dict avec résultats
    """

    print(f"\n🧪 TEST: max_workers={max_workers}, target_combos={n_combos}")
    print("-" * 80)

    # Données test
    test_data = create_test_data(n_bars=1000)

    # Grid spec (ajuster pour obtenir n_combos cible)
    if n_combos <= 81:
        grid_spec = ScenarioSpec(
            type="grid",
            params={
                "bb_window": [10, 20, 30],  # 3
                "bb_num_std": [1.5, 2.0, 2.5],  # 3
                "atr_window": [10, 14, 20],  # 3
                "atr_multiplier": [1.5, 2.0, 2.5],  # 3
            },  # 3*3*3*3 = 81 combos
            sampler="grid",
        )
        expected_combos = 81
    else:
        grid_spec = ScenarioSpec(
            type="grid",
            params={
                "bb_window": [10, 15, 20, 25, 30],  # 5
                "bb_num_std": [1.5, 2.0, 2.5, 3.0],  # 4
                "atr_window": [10, 14, 20],  # 3
                "atr_multiplier": [1.5, 2.0, 2.5],  # 3
            },  # 5*4*3*3 = 180 combos
            sampler="grid",
        )
        expected_combos = 180

    # Runner
    runner = SweepRunner(max_workers=max_workers)

    # Benchmark avec 3 runs
    times = []
    results_counts = []

    for run in range(3):
        print(f"   Run {run+1}/3...", end=" ", flush=True)

        start = time.perf_counter()
        results = runner.run_grid(
            grid_spec=grid_spec, real_data=test_data, symbol="TEST", timeframe="1h"
        )
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        results_counts.append(len(results))

        tests_per_sec = len(results) / elapsed if elapsed > 0 else 0
        print(
            f"{elapsed:.2f}s ({tests_per_sec:.1f} tests/sec, {len(results)} résultats)"
        )

    # Stats
    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_count = int(np.mean(results_counts))
    mean_tps = mean_count / mean_time if mean_time > 0 else 0

    print(
        f"   ✅ Moyenne: {mean_time:.2f}s ± {std_time:.2f}s ({mean_tps:.1f} tests/sec)"
    )

    return {
        "workers": max_workers,
        "combos": mean_count,
        "mean_time": mean_time,
        "std_time": std_time,
        "mean_tps": mean_tps,
        "times": times,
    }


def main():
    """Point d'entrée principal."""

    print("=" * 100)
    print("🔬 PROFILING BASELINE PRODUCTION - ThreadX SweepRunner")
    print("=" * 100)

    # Tests avec différents workers
    worker_configs = [4, 8, 16, 32]
    results = []

    for n_workers in worker_configs:
        results.append(benchmark_sweep(max_workers=n_workers, n_combos=81))

    # Résumé comparatif
    print("\n\n" + "=" * 100)
    print("📊 RÉSUMÉ COMPARATIF")
    print("=" * 100)

    print(
        f"\n{'Workers':<12} {'Combos':<10} {'Temps (s)':<15} {'Tests/sec':<15} {'Speedup':<15}"
    )
    print("-" * 100)

    baseline = results[0]["mean_tps"] if results else 1.0

    for res in results:
        speedup = res["mean_tps"] / baseline if baseline > 0 else 1.0
        speedup_pct = (speedup - 1.0) * 100

        print(
            f"{res['workers']:<12} {res['combos']:<10} {res['mean_time']:>13.2f}s  "
            f"{res['mean_tps']:>13.1f}  {speedup:>5.2f}x ({speedup_pct:+.1f}%)"
        )

    # Meilleure config
    best = max(results, key=lambda x: x["mean_tps"])
    print("\n" + "=" * 100)
    print(f"🏆 MEILLEURE CONFIGURATION: max_workers={best['workers']}")
    print(f"   Performance: {best['mean_tps']:.1f} tests/sec")
    print(f"   Gain vs workers=4: {(best['mean_tps']/baseline - 1)*100:+.1f}%")

    # Projection sweep 8448 combos
    print("\n📈 PROJECTION SWEEP COMPLET (8448 combos)")
    print("-" * 100)

    for res in results:
        if res["mean_tps"] > 0:
            estimated_time = 8448 / res["mean_tps"]
            estimated_min = estimated_time / 60

            print(
                f"  workers={res['workers']:<4}  Temps estimé: {estimated_time:>7.1f}s ({estimated_min:>5.1f} min)"
            )

    print("=" * 100)

    # Identification des bottlenecks
    print("\n\n📊 ANALYSE DES BOTTLENECKS")
    print("=" * 100)

    if len(results) >= 2:
        # Comparer workers=4 vs workers=32
        low = results[0]
        high = results[-1]

        scaling_efficiency = (high["mean_tps"] / low["mean_tps"]) / (
            high["workers"] / low["workers"]
        )

        print(
            f"\nScaling efficiency (workers {low['workers']}→{high['workers']}): {scaling_efficiency:.1%}"
        )

        if scaling_efficiency < 0.5:
            print("⚠️  FAIBLE SCALING: Bottleneck probable dans:")
            print("    - GIL Python (trop de temps CPU pur Python)")
            print("    - I/O contention (checkpoints Parquet)")
            print("    - Mémoire partagée (indicateurs recalculés)")
        elif scaling_efficiency < 0.7:
            print("⚡ SCALING MOYEN: Optimisations possibles dans:")
            print("    - Réduction overhead multiprocessing")
            print("    - Batching des combinaisons")
            print("    - Cache partagé pour indicateurs")
        else:
            print("✅ BON SCALING: Peu de contention, CPU bien utilisé")
            print("   → Optimisations doivent cibler la vitesse individuelle par combo")

    print("=" * 100)

    # Sauvegarde
    output_file = Path("profiling_baseline_production.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("THREADX PROFILING BASELINE PRODUCTION REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write("CONFIGURATION SYSTÈME:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Processeur: Ryzen 9 9950X (32 threads logiques)\n")
        f.write(f"RAM: 64 Go\n")
        f.write(f"GPU: RTX 5080 (CUDA indisponible actuellement)\n\n")

        f.write("RÉSULTATS BENCHMARK:\n")
        f.write("-" * 100 + "\n")
        f.write(
            f"{'Workers':<12} {'Combos':<10} {'Temps (s)':<15} {'Tests/sec':<15} {'Speedup':<15}\n"
        )
        f.write("-" * 100 + "\n")

        for res in results:
            speedup = res["mean_tps"] / baseline if baseline > 0 else 1.0
            speedup_pct = (speedup - 1.0) * 100
            f.write(
                f"{res['workers']:<12} {res['combos']:<10} {res['mean_time']:>13.2f}s  "
                f"{res['mean_tps']:>13.1f}  {speedup:>5.2f}x ({speedup_pct:+.1f}%)\n"
            )

        f.write("\n" + "=" * 100 + "\n")
        f.write(f"MEILLEURE CONFIGURATION: max_workers={best['workers']}\n")
        f.write(f"Performance: {best['mean_tps']:.1f} tests/sec\n\n")

        f.write("PROJECTION SWEEP COMPLET (8448 combos):\n")
        f.write("-" * 100 + "\n")

        for res in results:
            if res["mean_tps"] > 0:
                estimated_time = 8448 / res["mean_tps"]
                estimated_min = estimated_time / 60
                f.write(
                    f"workers={res['workers']:<4}  Estimated time: {estimated_time:>7.1f}s ({estimated_min:>5.1f} min)\n"
                )

        # Scaling analysis
        if len(results) >= 2:
            low = results[0]
            high = results[-1]
            scaling_efficiency = (high["mean_tps"] / low["mean_tps"]) / (
                high["workers"] / low["workers"]
            )

            f.write("\n\nSCALING ANALYSIS:\n")
            f.write("-" * 100 + "\n")
            f.write(
                f"Scaling efficiency ({low['workers']}→{high['workers']} workers): {scaling_efficiency:.1%}\n"
            )

            if scaling_efficiency < 0.5:
                f.write("\nBOTTLENECKS IDENTIFIÉS:\n")
                f.write("  - GIL Python (trop de code Python pur)\n")
                f.write("  - I/O contention (Parquet writes)\n")
                f.write("  - Mémoire non partagée (indicateurs recalculés)\n")

    print(f"\n💾 Rapport sauvé: {output_file}\n")


if __name__ == "__main__":
    main()
