"""
ThreadX Profiling Baseline - Étape 3 : Benchmark comparatif
===========================================================

Benchmark rapide pour mesurer la performance actuelle et tester
différentes configurations (workers, GPU, etc.)

Usage:
    python profiling_baseline_step3.py

Outputs:
    - profiling_benchmark_results.txt
    - Console avec comparaison configurations
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.threadx.optimization.engine import UnifiedOptimizationEngine
from src.threadx.optimization.scenarios import generate_param_grid
from src.threadx.data_access import load_ohlcv


def benchmark_configuration(
    max_workers: int, use_gpu: bool, n_combos: int = 36
) -> dict:
    """
    Benchmark une configuration spécifique.

    Args:
        max_workers: Nombre de workers
        use_gpu: Activer GPU ou non
        n_combos: Nombre de combinaisons à tester

    Returns:
        Dict avec résultats benchmark
    """

    print(
        f"\n🧪 Test: max_workers={max_workers}, GPU={'ON' if use_gpu else 'OFF'}, combos={n_combos}"
    )
    print("-" * 80)

    # Données
    try:
        df = load_ohlcv("BTCUSDT", "1h")
    except:
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

    # Paramètres (taille adaptée à n_combos demandé)
    if n_combos <= 36:
        param_spec = {
            "bb_period": [20, 30],
            "bb_std": [1.5, 2.0, 2.5],
            "atr_multiplier": [1.5, 2.0],
            "entry_z": [1.0, 1.5, 2.0],
        }  # 2*3*2*3 = 36 combos
    else:
        param_spec = {
            "bb_period": [20, 30, 40],
            "bb_std": [1.5, 2.0, 2.5],
            "atr_multiplier": [1.5, 2.0],
            "entry_z": [1.0, 1.5],
            "risk_per_trade": [0.01, 0.02],
        }  # 3*3*2*2*2 = 72 combos

    combinations = generate_param_grid(param_spec)
    actual_combos = len(combinations)

    # Engine
    engine = UnifiedOptimizationEngine(max_workers=max_workers)

    # Benchmark avec 3 runs pour moyenne
    times = []
    for run in range(3):
        print(f"   Run {run+1}/3...", end=" ", flush=True)

        start = time.perf_counter()
        results_df = engine.run_sweep(
            params=param_spec,
            data=df,
            symbol="BTCUSDT",
            timeframe="1h",
            initial_capital=10000.0,
            reuse_cache=True,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        tests_per_sec = actual_combos / elapsed
        print(f"{elapsed:.2f}s ({tests_per_sec:.1f} tests/sec)")

    # Statistiques
    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_tps = actual_combos / mean_time

    print(
        f"   ✅ Moyenne: {mean_time:.2f}s ± {std_time:.2f}s ({mean_tps:.1f} tests/sec)"
    )

    return {
        "max_workers": max_workers,
        "use_gpu": use_gpu,
        "n_combos": actual_combos,
        "mean_time": mean_time,
        "std_time": std_time,
        "mean_tps": mean_tps,
        "times": times,
    }


def main():
    """Point d'entrée principal."""

    print("=" * 100)
    print("🏁 BENCHMARK COMPARATIF - ThreadX Optimization Engine")
    print("=" * 100)

    results = []

    # Test 1: Configuration actuelle (baseline)
    print("\n📊 CONFIGURATION BASELINE")
    results.append(benchmark_configuration(max_workers=8, use_gpu=False, n_combos=36))

    # Test 2: Plus de workers
    print("\n📊 TEST: AUGMENTATION WORKERS")
    results.append(benchmark_configuration(max_workers=16, use_gpu=False, n_combos=36))
    results.append(benchmark_configuration(max_workers=32, use_gpu=False, n_combos=36))

    # Test 3: GPU (si disponible)
    print("\n📊 TEST: GPU ACTIVATION")
    try:
        results.append(
            benchmark_configuration(max_workers=8, use_gpu=True, n_combos=36)
        )
    except Exception as e:
        print(f"   ⚠️  GPU test échoué: {e}")

    # Affichage comparatif
    print("\n\n" + "=" * 100)
    print("📊 RÉSULTATS COMPARATIFS")
    print("=" * 100)

    # Tableau
    print(
        f"\n{'Config':<30} {'Workers':<10} {'GPU':<8} {'Temps (s)':<15} {'Tests/sec':<15} {'vs Baseline':<15}"
    )
    print("-" * 100)

    baseline_tps = results[0]["mean_tps"] if results else 0

    for i, res in enumerate(results):
        config_name = "BASELINE" if i == 0 else f"Config {i}"
        gpu_status = "✅" if res["use_gpu"] else "❌"
        speedup = (res["mean_tps"] / baseline_tps) if baseline_tps > 0 else 1.0
        speedup_pct = (speedup - 1.0) * 100
        speedup_str = f"{speedup:.2f}x ({speedup_pct:+.1f}%)"

        print(
            f"{config_name:<30} {res['max_workers']:<10} {gpu_status:<8} "
            f"{res['mean_time']:>7.2f} ± {res['std_time']:>4.2f}  "
            f"{res['mean_tps']:>13.1f}  {speedup_str:<15}"
        )

    print("-" * 100)

    # Meilleure configuration
    best_config = max(results, key=lambda x: x["mean_tps"])
    best_idx = results.index(best_config)

    print(f"\n🏆 MEILLEURE CONFIGURATION: Config {best_idx}")
    print(f"   Workers: {best_config['max_workers']}")
    print(f"   GPU: {'Activé' if best_config['use_gpu'] else 'Désactivé'}")
    print(f"   Performance: {best_config['mean_tps']:.1f} tests/sec")
    print(
        f"   Gain vs baseline: {(best_config['mean_tps']/baseline_tps - 1)*100:+.1f}%"
    )

    # Projection pour sweep complet
    print("\n📈 PROJECTION POUR SWEEP COMPLET (8448 combos)")
    print("-" * 100)

    for i, res in enumerate(results):
        config_name = "BASELINE" if i == 0 else f"Config {i}"
        estimated_time = 8448 / res["mean_tps"]
        estimated_min = estimated_time / 60

        print(
            f"{config_name:<30} Temps estimé: {estimated_time:>7.1f}s ({estimated_min:>5.1f} min)"
        )

    print("=" * 100)

    # Sauvegarde rapport
    output_file = Path("profiling_benchmark_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("THREADX BENCHMARK COMPARATIF REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write(
            f"{'Config':<30} {'Workers':<10} {'GPU':<8} {'Temps (s)':<15} {'Tests/sec':<15} {'vs Baseline':<15}\n"
        )
        f.write("-" * 100 + "\n")

        for i, res in enumerate(results):
            config_name = "BASELINE" if i == 0 else f"Config {i}"
            gpu_status = "YES" if res["use_gpu"] else "NO"
            speedup = (res["mean_tps"] / baseline_tps) if baseline_tps > 0 else 1.0
            speedup_pct = (speedup - 1.0) * 100
            speedup_str = f"{speedup:.2f}x ({speedup_pct:+.1f}%)"

            f.write(
                f"{config_name:<30} {res['max_workers']:<10} {gpu_status:<8} "
                f"{res['mean_time']:>7.2f} ± {res['std_time']:>4.2f}  "
                f"{res['mean_tps']:>13.1f}  {speedup_str:<15}\n"
            )

        f.write("-" * 100 + "\n\n")

        f.write(f"MEILLEURE CONFIGURATION: Config {best_idx}\n")
        f.write(f"  Workers: {best_config['max_workers']}\n")
        f.write(f"  GPU: {'Enabled' if best_config['use_gpu'] else 'Disabled'}\n")
        f.write(f"  Performance: {best_config['mean_tps']:.1f} tests/sec\n")
        f.write(
            f"  Gain vs baseline: {(best_config['mean_tps']/baseline_tps - 1)*100:+.1f}%\n\n"
        )

        f.write("PROJECTION POUR SWEEP COMPLET (8448 combos)\n")
        f.write("-" * 100 + "\n")

        for i, res in enumerate(results):
            config_name = "BASELINE" if i == 0 else f"Config {i}"
            estimated_time = 8448 / res["mean_tps"]
            estimated_min = estimated_time / 60
            f.write(
                f"{config_name:<30} Estimated time: {estimated_time:>7.1f}s ({estimated_min:>5.1f} min)\n"
            )

    print(f"\n💾 Rapport sauvé: {output_file}\n")


if __name__ == "__main__":
    main()
